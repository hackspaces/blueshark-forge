"""P5.6 — self-harvested few-shot exemplar store.

On every SUCCESSFULLY executed action the harness appends the model's OWN raw
action JSON to ``~/.forge/exemplars/<model-slug>.jsonl``, keyed by action kind
(capped ``PER_KIND`` most-recent-wins, deduped, write_file bodies redacted).
Those harvested exemplars are then fed back where FORMAT anchoring matters:

  * the malformed-JSON retry nudge embeds one of the model's own past valid
    actions of the *guessed* kind ("here is a valid action you emitted before —
    reply in exactly that shape"), a far stronger signal than the bare "that was
    not valid action JSON" text, which gives zero signal about what valid looks
    like in the model's own voice; and
  * a cold-start session for a model with recorded malformed history gets one
    ``user("example task")`` / ``assistant(exemplar)`` pair pinned into its head
    so the very first generation already sees its own valid format.

Mirrors ``fleet._learn_path``: the same slug + one-jsonl-per-key shape,
stdlib-only, and best-effort — every entry point swallows its own I/O errors so a
broken or read-only store can never raise into the agent loop.

Honest scope (see ROADMAP P5.6): on grammar-forcing engines (Ollama
``format:schema``, OpenAI ``json_schema``) malformed output is dominated by
NUM_PREDICT truncation, which an exemplar cannot fix; the exemplar mainly helps
endpoints that ignore ``response_format``. Cheap, local, additive — the harness
gets exemplars for free as a byproduct of success, per model family.
"""
import json
import os
import re

from .util import slurp

# Resolved through this module global at CALL time, so tests can redirect the
# whole store to a tempdir (monkeypatch ``exemplars.EXEMPLAR_DIR``) without ever
# touching the real ~/.forge. Created lazily on first write.
EXEMPLAR_DIR = os.path.join(os.path.expanduser("~/.forge"), "exemplars")

PER_KIND = 5      # most-recent-wins cap, per action kind, per model
MAX_LEN = 600     # a stored exemplar is never longer than this many chars

# The action kinds we harvest / anchor. A local copy of tools.ALL_ACTIONS (kept
# local to avoid an import cycle at Agent construction time; a foreign kind simply
# never matches and is stored/served verbatim).
_KINDS = ("bash", "read_file", "write_file", "edit_file", "list_files",
          "grep", "glob", "fleet_send", "say")


def _slug(model):
    return re.sub(r"[^A-Za-z0-9]", "-", model or "unknown")


def _path(model):
    return os.path.join(EXEMPLAR_DIR, _slug(model) + ".jsonl")


def _counts_path():
    return os.path.join(EXEMPLAR_DIR, "_malformed.json")


def _load(model):
    """Every stored exemplar for ``model``, oldest→newest, as {kind, raw} dicts."""
    p = _path(model)
    if not os.path.exists(p):
        return []
    out = []
    for line in slurp(p).splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("kind") and d.get("raw"):
            out.append(d)
    return out


def _redact(kind, raw):
    """Shrink one raw action into a compact, storable exemplar. A write_file echo
    carries the ENTIRE file verbatim in "content" — replace it with a short
    placeholder so the exemplar teaches SHAPE, not payload. Always capped at
    MAX_LEN chars (so a non-write action too large to be a useful exemplar is
    simply truncated)."""
    if not isinstance(raw, str):
        return ""
    if kind == "write_file":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("content"), str):
            obj["content"] = "…"
            raw = json.dumps(obj, ensure_ascii=False)
    return raw[:MAX_LEN]


def record(model, kind, raw):
    """Harvest one SUCCESSFUL action as a few-shot exemplar for (model, kind).

    Keeps at most PER_KIND per kind (most-recent-wins), dedupes an identical body,
    and redacts write_file content. ``raw`` is the full raw model output — record
    redacts-then-caps here so a big write_file is elided on COMPLETE json rather
    than truncated mid-string. Best-effort; never raises into the caller."""
    if not model or not kind or not raw:
        return
    body = _redact(kind, raw)
    if not body:
        return
    try:
        recs = [d for d in _load(model)
                if not (d["kind"] == kind and d["raw"] == body)]   # drop an identical prior
        recs.append({"kind": kind, "raw": body})
        # Cap THIS kind to its PER_KIND most-recent; leave every other kind intact.
        kept, seen = [], 0
        for d in reversed(recs):
            if d["kind"] == kind:
                if seen >= PER_KIND:
                    continue
                seen += 1
            kept.append(d)
        kept.reverse()
        os.makedirs(EXEMPLAR_DIR, exist_ok=True)
        with open(_path(model), "w") as f:
            for d in kept:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except OSError:
        pass


def fetch(model, kind):
    """The model's most-recent stored exemplar body for ``kind``, or None."""
    if not model or not kind:
        return None
    for d in reversed(_load(model)):
        if d["kind"] == kind:
            return d["raw"]
    return None


def fetch_any(model):
    """The model's single most-recent exemplar of ANY kind, or None — used to pin
    ONE demonstration into a cold-start head when the intended kind is not yet
    known."""
    recs = _load(model)
    return recs[-1]["raw"] if recs else None


def _load_counts():
    p = _counts_path()
    if not os.path.exists(p):
        return {}
    try:
        d = json.loads(slurp(p))
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) else {}


def record_malformed(model):
    """Tally one malformed-JSON strike for ``model``. The count both keys the
    cold-start head-pin and, being nonzero, is itself evidence that this engine
    is NOT reliably grammar-forcing its output. Best-effort; never raises."""
    if not model:
        return
    try:
        counts = _load_counts()
        counts[_slug(model)] = int(counts.get(_slug(model), 0)) + 1
        os.makedirs(EXEMPLAR_DIR, exist_ok=True)
        with open(_counts_path(), "w") as f:
            f.write(json.dumps(counts))
    except OSError:
        pass


def malformed_count(model):
    """How many malformed-JSON strikes have been recorded for ``model``."""
    if not model:
        return 0
    try:
        return int(_load_counts().get(_slug(model), 0))
    except (TypeError, ValueError):
        return 0


def guess_kind(raw):
    """Best-effort guess of the action kind a malformed / truncated output was
    ATTEMPTING, so its retry nudge can quote an exemplar of that same kind. Reads
    an explicit ``"action":"…"`` first; falls back to the distinctive field a
    truncation left behind (a cut-off value never reaches the action field).
    Returns a known kind or None."""
    raw = raw or ""
    m = re.search(r'"action"\s*:\s*"([a-z_]+)"', raw)
    if m and m.group(1) in _KINDS:
        return m.group(1)
    if re.search(r'"command"\s*:', raw):
        return "bash"
    if re.search(r'"pattern"\s*:', raw):
        return "grep"
    if re.search(r'"(?:old|new|start_line|end_line|anchor)"\s*:', raw):
        return "edit_file"
    if re.search(r'"content"\s*:', raw):
        return "write_file"
    if re.search(r'"(?:offset|limit|outline)"\s*:', raw):
        return "read_file"
    if re.search(r'"path"\s*:', raw):
        return "read_file"
    return None
