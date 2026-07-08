"""P5.8 — model passports: the harness measures each model and auto-tunes its levers.

Every knob in forge is one-size-fits-all (loop threshold 3, NUM_PREDICT 2048, retry
heat +0.4) even though the failure modes are demonstrably model-specific — the test
suite is a museum of per-model postmortems. A passport is a small learned capability
profile the harness keeps PER MODEL and reads at Agent construction (and at every
ladder swap) to reshape itself around that model's weaknesses.

Two sources feed a passport, both stdlib, both harness-side:

  * PASSIVE telemetry — the live loop calls ``record(model, event)`` at the existing
    stuck sites (malformed strike, loop trip, escalation, alias repair, fuzzy-vs-exact
    edit) plus one ``session`` tick per real Agent. Counts accumulate in
    ``~/.forge/profile/<model-slug>.json``. Rates are normalized PER SESSION (a robust,
    denominator that every counter already carries), NOT per step — so nothing is
    recorded on the hot per-step path.

  * ACTIVE probe — ``setup.py`` runs ~10 canned micro-prompts through the real backend
    once at install (``score_probe`` below scores format-holding / field completeness /
    exact-text reproduction) and ``write_passport`` persists the result, so a FRESH
    install is tuned before its first task. The probe itself calls a model; only its
    offline scoring lives here and is unit-tested.

``knobs(model, defaults)`` fuses both into the three knobs the loop actually consumes
(``loop_threshold``, ``num_predict``, ``heat_bump``). With an EMPTY store it returns the
defaults verbatim, so an un-profiled model — and the whole offline test suite — runs
byte-for-byte as before. (``stuck_at`` is intentionally left to config/env, already made
tunable in P5.7, to avoid overriding an explicit user choice.)

Mirrors ``exemplars.py``: same one-file-per-model-slug store, resolved through a module
global so tests redirect the whole thing to a tempdir, and every writer swallows its own
I/O errors so a broken or read-only store can never raise into the agent loop.
"""
import json
import os
import re

from .util import slurp

# Resolved through this module global at CALL time so tests can redirect the whole
# store to a tempdir (monkeypatch ``profile.PROFILE_DIR``) without touching real
# ~/.forge. Created lazily on the first write.
PROFILE_DIR = os.path.join(os.path.expanduser("~/.forge"), "profile")

# The passive events the loop records. `session` is the per-session denominator; the
# rest are the stuck signals. All are rare (none fire on the hot per-step path).
EVENTS = ("session", "malformed", "trunc_write", "loop", "escalate",
          "alias_repair", "fuzzy_edit", "exact_edit")

# --- tuning thresholds (per-session rates) ---------------------------------------
MIN_SESSIONS = 3        # don't tune from passive signal until this many sessions of data
LOOP_PRONE = 0.5        # ≥ this many loop trips per session → tighten loop_threshold to 2
TRUNC_PRONE = 0.5       # ≥ this many write truncations per session → raise num_predict
MALFORMED_PRONE = 1.0   # ≥ this many malformed strikes per session → hotter retry heat

TIGHT_LOOP_THRESHOLD = 2      # for loop-prone models
TRUNC_NUM_PREDICT = 4096      # for write truncators (doubled from the 2048 default)
HOT_HEAT_BUMP = 0.5           # per-nudge heat step for malformed-prone models (default 0.4)

# active-probe score floors that also flag a malformed-prone model (fresh-install tuning,
# before any passive session has run)
PROBE_FORMAT_FLOOR = 0.8


def _slug(model):
    return re.sub(r"[^A-Za-z0-9]", "-", model or "unknown")


def _path(model):
    return os.path.join(PROFILE_DIR, _slug(model) + ".json")


def _blank():
    return {"counts": {}, "probe": {}}


def load(model):
    """The full passport for ``model``: {"counts": {...}, "probe": {...}}. Missing or
    corrupt store → a blank passport (never raises)."""
    if not model:
        return _blank()
    p = _path(model)
    if not os.path.exists(p):
        return _blank()
    try:
        d = json.loads(slurp(p))
    except (json.JSONDecodeError, OSError):
        return _blank()
    if not isinstance(d, dict):
        return _blank()
    d.setdefault("counts", {})
    d.setdefault("probe", {})
    if not isinstance(d["counts"], dict):
        d["counts"] = {}
    if not isinstance(d["probe"], dict):
        d["probe"] = {}
    return d


def _save(model, data):
    try:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        tmp = _path(model) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, _path(model))
    except OSError:
        pass


def record(model, event, n=1):
    """Tally ``n`` occurrences of a passive ``event`` for ``model`` (write-through,
    best-effort). Unknown events are stored verbatim — a forward-compatible superset —
    but only the EVENTS above drive knob resolution. Never raises into the caller."""
    if not model or not event or n <= 0:
        return
    data = load(model)
    c = data["counts"]
    c[event] = int(c.get(event, 0)) + n
    _save(model, data)


def write_passport(model, probe_scores):
    """Persist an active-probe result (from ``score_probe``) into ``model``'s passport,
    leaving passive counts intact. Best-effort; never raises."""
    if not model or not isinstance(probe_scores, dict):
        return
    data = load(model)
    data["probe"] = dict(probe_scores)
    _save(model, data)


def rates(model):
    """Derived per-session rates + the self-normalizing fuzzy-edit fraction. Used by
    ``knobs`` and by ``forge passport`` to explain a tuning decision."""
    data = load(model)
    c = data["counts"]
    sessions = max(int(c.get("session", 0)), 0)
    denom = max(sessions, 1)
    edits = int(c.get("fuzzy_edit", 0)) + int(c.get("exact_edit", 0))
    return {
        "sessions": sessions,
        "malformed_per_session": c.get("malformed", 0) / denom,
        "loop_per_session": c.get("loop", 0) / denom,
        "trunc_per_session": c.get("trunc_write", 0) / denom,
        "escalate_per_session": c.get("escalate", 0) / denom,
        "fuzzy_edit_frac": (c.get("fuzzy_edit", 0) / edits) if edits else 0.0,
    }


def knobs(model, defaults):
    """Fuse passive rates + the active probe into the loop's consumed knobs.

    ``defaults`` carries {loop_threshold, num_predict, heat_bump}; a copy is returned
    with any model-specific overrides applied. An EMPTY passport returns the defaults
    unchanged (the load-bearing invariant: un-profiled models run exactly as before).
    Passive tuning waits for ``MIN_SESSIONS`` of evidence; the active probe can tune a
    fresh install with zero sessions."""
    d = dict(defaults)
    data = load(model)
    c = data["counts"]
    probe = data["probe"]
    sessions = max(int(c.get("session", 0)), 0)

    if sessions >= MIN_SESSIONS:
        if c.get("loop", 0) / sessions >= LOOP_PRONE:
            d["loop_threshold"] = TIGHT_LOOP_THRESHOLD
        if c.get("trunc_write", 0) / sessions >= TRUNC_PRONE:
            d["num_predict"] = max(int(d.get("num_predict", 0)), TRUNC_NUM_PREDICT)
        if c.get("malformed", 0) / sessions >= MALFORMED_PRONE:
            d["heat_bump"] = max(float(d.get("heat_bump", 0.0)), HOT_HEAT_BUMP)

    # Active probe: a model that couldn't hold the action format under a temp-0 probe is
    # malformed-prone from its very first task — start its retries hotter without waiting
    # for MIN_SESSIONS of live evidence.
    if probe.get("n") and probe.get("format_hold", 1.0) < PROBE_FORMAT_FLOOR:
        d["heat_bump"] = max(float(d.get("heat_bump", 0.0)), HOT_HEAT_BUMP)

    return d


# --- active probe: canned micro-prompts + offline scoring -------------------------
# The probe drives a REAL backend (in setup.py); only the prompt set and the scoring
# live here so the scoring is unit-testable with no model. Each spec: the user prompt,
# the action it should elicit, that action's required fields, and (for the exact-repro
# specs) the verbatim text a field must reproduce.

def probe_specs():
    """The ~10 canned probe prompts + their expectations. Three dimensions:
    format-holding (does it emit a valid action at all), field completeness (are the
    required fields present), and exact-text reproduction (does `old` come back
    character-for-character)."""
    from .tools import required_fields
    raw = [
        # format-holding — one required-field-light action each
        {"prompt": "List the files in the current directory.", "action": "list_files"},
        {"prompt": "Show me the contents of README.md.", "action": "read_file"},
        {"prompt": "Run the project's test suite.", "action": "bash"},
        {"prompt": "Search the code for the word TODO.", "action": "grep"},
        # field completeness — multi-required-field actions
        {"prompt": "Create a file called notes.txt containing exactly: hello world",
         "action": "write_file"},
        {"prompt": "In config.py, replace the text `debug = False` with `debug = True`.",
         "action": "edit_file"},
        {"prompt": "Print the current working directory.", "action": "bash"},
        # exact reproduction — the `old` field must come back verbatim
        {"prompt": "In server.py, change the line `PORT = 8080` to `PORT = 9090`. "
                   "Reproduce the existing line character-for-character.",
         "action": "edit_file", "exact_field": "old", "exact_text": "PORT = 8080"},
        {"prompt": "In app.py, replace `TIMEOUT = 30` with `TIMEOUT = 60`. "
                   "Copy the old text exactly as written.",
         "action": "edit_file", "exact_field": "old", "exact_text": "TIMEOUT = 30"},
        {"prompt": "In main.py, change `version = \"1.0\"` to `version = \"2.0\"`. "
                   "Reproduce the old text exactly.",
         "action": "edit_file", "exact_field": "old", "exact_text": 'version = "1.0"'},
    ]
    for spec in raw:
        spec["required"] = required_fields(spec["action"])
    return raw


def _parse_action(raw):
    """Parse one probe output into an action dict, tolerating the same wrappings the
    live salvage pass recovers (fences / prose prefix / trailing comma). Returns the
    dict or None. Kept independent of agent._salvage to avoid an import cycle."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # strip a leading ```lang fence and trailing ```; drop a trailing comma
    stripped = re.sub(r"\A\s*```[a-zA-Z0-9_+.-]*[ \t]*\n?|\n?[ \t]*```\s*\Z", "", raw)
    stripped = re.sub(r",(\s*[}\]])", r"\1", stripped)
    # brace-scan the first balanced top-level object
    start = stripped.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(stripped[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def score_probe(raws, specs=None):
    """Score a list of probe outputs (aligned with ``probe_specs()``) into a passport
    ``probe`` dict: {format_hold, field_complete, exact_repro, n}. Pure and offline —
    the live probe calls the model, this scores the strings it returned.

    * format_hold   — fraction that parse to a dict carrying an ``action`` field
    * field_complete— fraction (of format-holders) whose required fields are all present
    * exact_repro   — fraction (of the exact-repro specs) reproducing the target verbatim
    """
    specs = specs if specs is not None else probe_specs()
    n = len(specs)
    if not n:
        return {"format_hold": 0.0, "field_complete": 0.0, "exact_repro": 0.0, "n": 0}
    held = complete = 0
    exact_total = exact_ok = 0
    for i, spec in enumerate(specs):
        raw = raws[i] if i < len(raws) else ""
        act = _parse_action(raw)
        formatted = isinstance(act, dict) and isinstance(act.get("action"), str)
        if formatted:
            held += 1
            if all(act.get(f) not in (None, "") for f in spec.get("required", [])):
                complete += 1
        if spec.get("exact_field"):
            exact_total += 1
            if formatted and act.get(spec["exact_field"]) == spec.get("exact_text"):
                exact_ok += 1
    return {
        "format_hold": held / n,
        "field_complete": (complete / held) if held else 0.0,
        "exact_repro": (exact_ok / exact_total) if exact_total else 0.0,
        "n": n,
    }


def describe(model, defaults):
    """A short human summary of a model's learned passport + the knobs it resolves to —
    powers ``forge passport``. Returns a list of display lines."""
    data = load(model)
    r = rates(model)
    probe = data["probe"]
    resolved = knobs(model, defaults)
    lines = [f"{model}"]
    if not data["counts"] and not probe:
        lines.append("  (no passport yet — runs on defaults; `forge setup` probes a fresh install)")
        return lines
    lines.append(f"  sessions observed: {r['sessions']}")
    if data["counts"]:
        lines.append("  per-session:  "
                     f"malformed {r['malformed_per_session']:.2f}  "
                     f"loop {r['loop_per_session']:.2f}  "
                     f"trunc {r['trunc_per_session']:.2f}  "
                     f"escalate {r['escalate_per_session']:.2f}")
        lines.append(f"  fuzzy-edit fraction: {r['fuzzy_edit_frac']:.2f}")
    if probe.get("n"):
        lines.append(f"  probe (n={probe['n']}):  "
                     f"format {probe.get('format_hold', 0):.2f}  "
                     f"fields {probe.get('field_complete', 0):.2f}  "
                     f"exact-repro {probe.get('exact_repro', 0):.2f}")
    changed = {k: v for k, v in resolved.items() if v != defaults.get(k)}
    lines.append(f"  tuned knobs: {changed}" if changed else "  tuned knobs: (defaults)")
    return lines
