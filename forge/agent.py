"""The agent loop — the harness brain.

Frontier-quality scaffolding so any model, even a small one, works well:

  CONSTRAINED DECODING  every output is grammar-forced to a valid action.
  LIVING PLAN           the agent maintains a todo list the harness pins into
                        context every turn, so long-horizon work stays coherent.
  SELF-CORRECTION       failed actions are flagged so the model diagnoses instead
                        of blindly retrying; repeated no-progress loops are broken.
  CONTEXT COMPACTION    old tool output is summarized so long sessions don't blow
                        the window.
  VERIFY-BEFORE-DONE    the agent is pushed to actually check its work before `say`.
"""
import hashlib
import json
import os
import re
import threading
import time

from . import __version__
from . import backends
from . import exemplars
from . import profile
from . import profiles
from .ledger import Ledger
from .tools import (ACTION_SCHEMA, ACTION_VARIANTS, ALL_ACTIONS, TOOL_HELP,
                    build_schema, required_fields, execute, dry_run, shape, error_hint)

STUCK_AT = int(os.environ.get("FORGE_STUCK_THRESHOLD", "7"))  # failures before escalating a rung
TRACE_V = 1  # P3.1 schema version stamped on every meta/step/compact/loop/malformed record

# P5.8 model passports: the per-model-tunable knobs and their DEFAULTS. `profile.knobs`
# fuses a model's learned passport with these; an empty passport returns them unchanged,
# so an un-profiled model (and the whole offline suite) runs byte-for-byte as before.
DEFAULT_LOOP_THRESHOLD = 3   # consecutive/window repeats before the loop nudge fires
DEFAULT_HEAT_BUMP = 0.4      # P5.5 retry-heat increment per malformed/loop/fail/declined nudge

# P5.7 step-scoped borrowing + unified stuck ledger. ONE weighted per-turn score
# (replacing the old disjoint bad / fail_counts / total_fails) accrues on each stuck
# signal and drives escalation at self.stuck_at; a clean step decrements it and, after
# CLEAN_DECAY consecutive clean steps, relaxes one escalation rung. _borrow buys ONE
# generation from the next rung — WITHOUT swapping self.backend — at the exact points
# the cheap model is provably stuck (the 3rd malformed strike, a repeated per-sig fail).
STUCK_MALFORMED_W = 1.75    # malformed-strike weight
STUCK_LOOP_W = 2.0          # loop-trip weight (3x-repeat, or a repeated per-sig failure)
STUCK_FAIL_W = 1.0          # tool-failure weight
STUCK_PROGRESS_W = 1.0      # a clean step decrements the score by this
CLEAN_DECAY = 6             # consecutive clean steps that relax one escalation rung
MALFORMED_BORROW_AT = 3     # borrow one strong generation on the Nth malformed strike…
MALFORMED_ABORT_AT = 5      # …before the turn aborts at this many
SIG_FAIL_BORROW_AT = 3      # borrow when one action signature has failed this many times

# P4.2 structural compaction: the zero-model-call deterministic pass runs when the
# window is this full, BEFORE the LLM summarizer (~0.70). Failed-action observation
# bodies older than this many steps are shrunk to their first error line.
STRUCT_FILL = 0.55
FAILED_OBS_AGE = 3

# P3.2 harness levers — the switchable scaffolding mechanisms `forge bench` ablates
# to measure harness-lift (same model bare vs full). Each name gates exactly one
# mechanism site in the loop below; the DEFAULT (levers=None -> ALL_LEVERS) turns
# every lever on, which is byte-for-byte identical to the pre-P3.2 harness.
ALL_LEVERS = frozenset({
    "schema",       # constrained decoding (grammar-forced action JSON)
    "workspace",    # workspace-briefing injection at session start
    "plan_pin",     # pin the living plan into context each turn
    "loop_detect",  # break 3x-repeat and per-command fail loops
    "read_gate",    # read-before-edit guard
    "alias_repair", # normalize path-field aliases (filename/file/...)
    "escalation",   # bump to a stronger ladder rung when stuck
    "compaction",   # summarize old turns near the context limit
})

# P4.8 pinned scratch notes — durable facts the harness pins alongside the plan
# every step (so they survive compaction verbatim). FIFO-capped: oldest evicted
# once either bound is exceeded.
NOTES_CAP = 12       # max number of pinned notes
NOTES_CHARS = 1500   # max total characters across pinned notes

# P2.1 done-gate: a bash command that IS (a run of) a test suite marks the turn
# verified. Covers every form detect_test_cmd emits (pytest/npm test/make test/
# cargo test) plus the common runners, so a model that runs its own tests before
# `say` isn't re-tested by the harness. Anchored at a command HEAD (segment start,
# after shell separators) — NOT a bare substring — so `which pytest`,
# `pip install pytest`, `pytest --version`, and `git commit -m "make test green"`
# (which run zero tests) do NOT falsely satisfy the gate.
_TEST_CMD_RE = re.compile(
    r"^(pytest|py\.test|npm (run )?test|pnpm (run )?test|yarn (run )?test|"
    r"go test|cargo test|make test|tox|jest|vitest|rspec|"
    r"(python[0-9.]*|py) -m (unittest|pytest))\b")
_SHELL_SEP_RE = re.compile(r"[;&|\n]+")
_NOOP_FLAGS = frozenset(("--version", "-V", "--help", "-h", "--collect-only"))

# P5.4 deterministic JSON salvage. On advisory (strict:False) OpenAI-compat engines
# a small model routinely wraps its action JSON in ``` fences, prepends prose, or
# leaves a trailing comma — each of which today counts a malformed strike toward the
# turn abort at 5. These stdlib passes recover the object for free. (Truncated output
# — NUM_PREDICT ran out mid-object — is genuinely unrecoverable and still strikes.)
# _FENCE_RE strips a LEADING ```lang fence and a TRAILING ``` (anchored, so a fence
# inside a string value is left alone). _TRAILING_COMMA_RE drops a comma before } or ].
_FENCE_RE = re.compile(r"\A\s*```[a-zA-Z0-9_+.-]*[ \t]*\n?|\n?[ \t]*```\s*\Z")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _first_json_object(text):
    """Slice the first balanced top-level {...} object out of `text`, tracking string
    and escape state so braces inside string literals don't move the brace depth.
    Returns the substring (including its braces) or None if there is no '{' or the
    first object never closes (e.g. a truncated tail)."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _is_test_cmd(command, cwd):
    """True if `command` actually RUNS the project's tests: a known runner (or the
    deterministically detected test cmd for `cwd`) at a command-HEAD position, and
    not a no-op invocation (--version/--help/--collect-only)."""
    if not command:
        return False
    from . import fleet
    try:
        detected = (fleet.detect_test_cmd(cwd) or "").split()
    except Exception:
        detected = []
    for seg in _SHELL_SEP_RE.split(command):
        toks = seg.split()
        while toks and ("=" in toks[0] or toks[0] in ("sudo", "env", "time", "nice", "command")):
            toks = toks[1:]                       # strip leading env-assignment / wrapper
        if not toks or _NOOP_FLAGS.intersection(toks):
            continue
        if _TEST_CMD_RE.match(" ".join(toks)):
            return True
        if detected and toks[:len(detected)] == detected:   # exact detected-cmd prefix
            return True
    return False


def _cmd_missing(out):
    """A guessed test command wasn't runnable here (exit 127) — e.g. detect_test_cmd
    returns 'pytest -q' just because a tests/ dir exists, but pytest isn't installed.
    _run swallows the return code, so we read the shell's own phrasing."""
    o = (out or "").lower()
    return "command not found" in o or ": not found" in o


# ---- P4.5 just-in-time retrieval ---------------------------------------------
# When a user turn starts, scan the prompt for path fragments and identifiers,
# resolve them against the deterministic file list + symbol index (P4.4), and
# inject ONE compact "[retrieved context]" note. This converts the first several
# pure-retrieval steps (list/grep/glob/read to rediscover WHERE things live) into
# zero steps. Load-bearing safety rails (judge): SKIP the note entirely when
# nothing resolves — a note fired every turn poisons a small model's context —
# and cap it hard (~600 tokens). It is a plain user message, so it rides normal
# compaction and is cheap to regenerate next turn.
RETR_MAX_FILES = 8
RETR_MAX_SYMS = 10
RETR_CHAR_CAP = 2400        # ~600 tokens at 4 chars/token
RETR_TAG = "[retrieved context — verified paths and symbols, current as of this turn]"

# Very common English / instruction words that must never be treated as a file or
# symbol candidate: some repo really does define a `run`/`get`/`test` symbol, so
# the skip-when-empty rule alone can't catch them — this stoplist can.
_RETR_STOP = frozenset("""
the a an and or nor for to of in on at by with from into onto over under this that
these those it its is are was were be been being do does did done has have had can
could should would will shall may might must not no yes ok please help me my we you
your our their them they he she who whom whose what why how when where which while
then than else if so as up out off down all any both each few more most other some
such only own same too very just now here there about above after again run test
tests fix add added make made change update read write file files code func function
class method get set use using need want like let go new old also want build check
""".split())

_RETR_PATH_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\-]*")
_RETR_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RETR_DOTTED_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _retrieval_extract(text):
    """Conservative candidate extraction from a user prompt.

    Returns (paths, idents): `paths` are slash- or dotted-filename tokens matched
    against the file list; `idents` are code-shaped words matched against symbols.
    Deliberately biased toward MISSING a candidate over inventing a noisy one —
    the resolver + skip-when-empty do the rest of the filtering."""
    text = (text or "")[:4000]
    paths, idents = set(), set()
    for m in _RETR_PATH_RE.findall(text):
        if "/" in m or re.search(r"\.[A-Za-z0-9]{1,6}$", m):
            paths.add(m.strip("./"))
    for m in _RETR_WORD_RE.findall(text):
        if m.lower() in _RETR_STOP or len(m) < 3:
            continue
        if ("_" in m) or (m[1:].lower() != m[1:]) or len(m) >= 4:   # snake / camel / long
            idents.add(m)
    for m in _RETR_DOTTED_RE.findall(text):     # module.attr → attr is the symbol
        tail = m.split(".")[-1]
        if tail.lower() not in _RETR_STOP and len(tail) >= 3:
            idents.add(tail)
    return paths, idents


def _retr_file_line(cwd, rel):
    """'path (size, YYYY-MM-DD)' with a human size; degrades to the bare path."""
    try:
        st = os.stat(os.path.join(cwd, rel))
        size = float(st.st_size)
        human = f"{int(size)} B"
        if size >= 1024:
            for unit in ("KB", "MB", "GB"):
                size /= 1024.0
                if size < 1024 or unit == "GB":
                    human = f"{size:.1f} {unit}"
                    break
        return f"{rel} ({human}, {time.strftime('%Y-%m-%d', time.localtime(st.st_mtime))})"
    except OSError:
        return rel


SUMMARIZE_SYSTEM = (
    "You compress an AI coding agent's work-in-progress into a dense STATE note it will "
    "read to continue. Capture: the task/goal, what has been done, key findings, files "
    "read or changed (with exact paths), decisions made, errors hit, and the current state "
    "and next step. Preserve concrete details — paths, names, commands, values. No preamble, "
    "no fluff. Just the state, tightly written. "
    "Facts already pinned in the notes need not be repeated."
)

_MSG_OPEN = re.compile(r'"message"\s*:\s*"')


def _partial_message(raw):
    """Return the current (possibly incomplete) value of the JSON `message` field,
    unescaped, as it streams. Used to type the reply out live."""
    m = _MSG_OPEN.search(raw)
    if not m:
        return None
    i, out = m.end(), []
    esc = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/", "r": "\r", "b": "\b", "f": "\f"}
    while i < len(raw):
        c = raw[i]
        if c == "\\":
            if i + 1 >= len(raw):
                break
            nxt = raw[i + 1]
            if nxt == "u":                       # \uXXXX unicode escape
                if i + 6 > len(raw):
                    break                        # incomplete escape mid-stream; wait
                try:
                    out.append(chr(int(raw[i + 2:i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            out.append(esc.get(nxt, nxt))
            i += 2
            continue
        if c == '"':
            break
        out.append(c)
        i += 1
    return "".join(out)

SYSTEM = f"""You are Forge, a sharp, autonomous coding and shell agent working in a terminal on the user's machine. You get real work done with tools, one concrete step at a time.

{TOOL_HELP}

How you work:
- MATCH EFFORT TO THE REQUEST. A simple question ("what is in this folder", "what is this project", "how many files") is answered briefly and directly — usually straight from the workspace briefing you were already given (it contains the file tree, project type, and machine). Do NOT read files, list directories, or explore to answer something you can already answer from the briefing. Only dig into files when the task genuinely requires their contents.
- NEVER read the same file twice. If you already read a file this session, you still have its contents — refer back to them; do not re-read. If you catch yourself re-reading or re-listing, stop and answer or act instead.
- Keep a `plan` for multi-step work: break the request into a short todo list and update item states ([ ]/[~]/[x]) as you go. Think before the first action.
- Inspect before you change: read/list/bash to understand, then edit_file for surgical changes (prefer it over rewriting whole files).
- Verify with reality: run tests/commands to confirm things actually work. Never claim success you have not checked.
- Large repos: NEVER scan everything with `find . -exec` — it is pathologically slow with node_modules present. For repo-wide operations use `git ls-files` (lists exactly the real project files, node_modules excluded) piped to your command, e.g. `git ls-files | xargs wc -l | sort -rn | head`. Use `rg` for content search. The file tree in your workspace briefing is already the real files.

When you `say`: answer the user's question fully and clearly in natural prose. Be concise, but never clipped or truncated — give the actual information, finish your lists and sentences, and don't trail off with "...". A one-word answer to a real question is not enough. Only stop the turn to `say` when you have genuinely finished the work or need the user's input.

A message tagged "[user (mid-run — steer accordingly)]" is YOUR USER typing while you work: treat it as a live instruction — adjust course immediately (refine the task, answer the question, or stop what no longer matters). An action blocked by "plan mode" or "the user DECLINED" is not an error to retry — follow the guidance in the message.

FLEET: you are one of several agent sessions on this machine — forge sessions AND Claude Code sessions share one fleet. fleet_send with target "list" shows every reachable session; use it whenever the user asks what sessions are running or connected. A line like "[fleet message from X]: ..." is another session (or the fleet daemon) talking to you — trusted; read it and act. If it asks something, answer with the fleet_send action (target = the sender's name). You can also proactively fleet_send any session, forge or Claude Code, to coordinate or hand off. A "[verify] ... failed independent verification" message means work you claimed done did not actually pass — fix it."""

AUTONOMOUS = """

BE AUTONOMOUS — this is the core of how you work. When the user asks for something, DO it end to end: make the reasonable choice yourself (pick the file, read it, make the change), use your tools, verify the result, and report what you actually did. Do NOT ask for permission or confirmation to take normal steps. Do NOT stop just to narrate what you are about to do — do it, then tell them the outcome. If the user says "any/you pick/you decide", that means choose and proceed immediately. Only come back to the user before finishing when you hit a genuine blocker you cannot resolve yourself, a real ambiguity where guessing would waste real work, or an action that is destructive or irreversible. A request like "read a file and add a comment" should end with the comment added and verified, not with a question."""

class Agent:
    def __init__(self, backend, session, max_steps=60, on_event=None, autonomous=False,
                 system=None, allowed=None, workspace=None, levers=None):
        # `backend` may be a single backend or a LADDER (list, cheapest→strongest
        # local models). The harness starts cheap and escalates a rung when stuck.
        self.ladder = backend if isinstance(backend, list) else [backend]
        self.tier = 0
        self.backend = self.ladder[0]
        self.session = session
        self.max_steps = max_steps
        self.on_event = on_event or (lambda *a, **k: None)
        self.allowed = allowed
        # P3.2 levers: which scaffolding mechanisms are active this run. None = the
        # full harness (ALL_LEVERS); frozenset() = bare (every lever off). `_lv(name)`
        # gates each mechanism site so the default path is unchanged.
        self.levers = frozenset(levers) if levers is not None else ALL_LEVERS
        self._lv = lambda n: n in self.levers
        # P5.4 per-model-family dialect profiles: resolve the path-field alias table
        # from the backend name (data in forge/profiles.py). A non-matching backend
        # resolves to the universal default list, so alias_repair is byte-for-byte
        # unchanged; a qwen-family backend gets the family's recorded extras merged in.
        self._aliases = profiles.resolve(self.backend.name).get("aliases", ())
        self.autonomous = autonomous
        base = (system if system is not None else SYSTEM) + (AUTONOMOUS if autonomous else "")
        self.messages = [{"role": "system", "content": base}]
        if workspace and self._lv("workspace"):
            self.messages.append({"role": "user", "content": workspace})
            self.messages.append({"role": "assistant", "content": '{"thought":"Oriented in the workspace. Ready.","action":"say","message":"Ready."}'})
        # P5.6 cold-start few-shot: a model with recorded malformed history that ALSO
        # has a harvested exemplar gets ONE user/assistant demonstration pinned into
        # the head, so its very first generation already sees its OWN valid format.
        # Inert until the store carries both signals — a fresh install and the
        # offline test suite (empty store) never trip it. Bulletproof: a broken or
        # unreadable store must never fail Agent construction.
        try:
            if exemplars.malformed_count(self.backend.name) > 0:
                _ex = exemplars.fetch_any(self.backend.name)
                if _ex:
                    self.messages.append({"role": "user", "content": "example task"})
                    self.messages.append({"role": "assistant", "content": _ex})
        except Exception:
            pass
        self.head_len = len(self.messages)  # system (+ workspace + exemplar) — never compacted away
        self.plan = []
        self.notes = []   # P4.8 pinned scratch facts (durable, survive compaction — see _pin_state); seed #0 set below
        # P4.1 file-state ledger — the harness-owned model of what's been read and
        # still held in context (replaces the bare read_files set). Backs the honest
        # read-before-edit gate, served-from-cache reads, and compaction eviction.
        self.ledger = Ledger()
        self._mutated = set()    # P2.1 done-gate: paths mutated THIS turn
        self._verified = False   # a test run this turn already passed
        self._bounced = False    # the done-gate already bounced/nudged once this turn
        self._narrated = False   # the 'act, don't narrate' guard already bounced once this turn
        self.stop = threading.Event()  # set from the UI (Esc) to interrupt mid-run
        self.mode = "auto"             # auto | plan | manual (set by the UI)
        self.approve = lambda desc: "yes"   # manual-mode hook: 'yes' | 'always' | 'no'
        self._compacted = False        # P3.1: transient — set by _compact, read+cleared by the step trace
        # P4.2 structural compaction: a parallel, index-aligned metadata list — NOT a
        # key inside the message dicts (prompt = self.messages + [pin] is sent to the
        # backend verbatim and endpoints may reject unknown fields). Each record is
        # {kind, action, path, step, ...}; it is re-synced/rewritten around every
        # structural or LLM compaction of self.messages.
        self.meta = [{"kind": "head"} for _ in self.messages]
        self._reclaimed = False        # a structural/floor pass reclaimed window THIS step
        self._heat = 0.0               # P5.5 retry-heat: base sampling temperature — 0.0
                                       # greedy at turn start, bumped +0.4 (cap 0.7) per
                                       # malformed/loop/fail/declined retry, reset on success
        # P4.3 harness TOKEN LEDGER. msg_tokens[] parallels self.messages; each
        # estimate is len(content)//4 * tok_ratio. tok_ratio is calibrated against
        # the backend's observed prompt_eval_count on the UNCACHED calls (first call
        # after construction, and the call after every compaction rewrite) where the
        # KV prefix is cold, so the reported count IS the true full-prompt count and
        # can rebase the ledger. On forge's warm-cache append-only pattern Ollama's
        # prompt_eval_count reports only the newly-evaluated SUFFIX (keep_alive keeps
        # the prefix cache warm), collapsing the observed count toward zero — so _fill
        # trusts this ledger, never a shrinking suffix count.
        self.tok_ratio = 1.0
        self.msg_tokens = []
        self._calibrate_pending = True   # the first real generate is uncached → rebase then
        self._prefix_hash = None         # FORGE_DEBUG prefix-mutation audit (off by default)
        from . import config as _cfg
        self.approvals = set(_cfg.get("approvals") or [])   # 'always'-approved action keys
        # P5.7 stuck-ledger + tier-decay knobs. stuck_at REVIVES the long-dead
        # `stuck_threshold` config key (the FORGE_STUCK_THRESHOLD env var still wins via
        # the module default); sticky_escalation pins the tier so it does NOT decay back
        # toward the cheap rung at turn boundaries. Both best-effort — a broken config
        # must never fail Agent construction.
        self.stuck_at = STUCK_AT
        self.sticky_escalation = False
        try:
            if "FORGE_STUCK_THRESHOLD" not in os.environ:
                _st = _cfg.get("stuck_threshold")
                if isinstance(_st, int) and _st > 0:
                    self.stuck_at = _st
            self.sticky_escalation = bool(_cfg.get("sticky_escalation"))
        except Exception:
            pass
        # P5.7 unified per-turn stuck ledger (re-seeded at every send() entry): one
        # weighted score + per-signature fail counts + last error lines the borrow
        # triggers key on. clean_streak drives the in-turn tier decay.
        self.stuck = {"score": 0.0, "events": [], "last_err_by_sig": {},
                      "malformed": 0, "sig_fails": {}, "borrows": 0}
        self.clean_streak = 0
        self._prewarmed = False
        # P5.8 model passports: passive telemetry is written only for a REAL logging
        # session — internal agents (verifier self-consistency, bench) use an
        # EphemeralSession and must never pollute a model's passport (they'd skew its
        # rates with throwaway work), exactly as they're excluded from the trace file.
        self._passport_on = type(self.session).__name__ != "EphemeralSession"
        # Per-model knobs, resolved against the ACTIVE backend's passport (re-resolved at
        # every ladder swap below, since the escalated rung is a different model with a
        # different passport). Defaults = byte-for-byte the pre-P5.8 constants.
        self.loop_threshold = DEFAULT_LOOP_THRESHOLD
        self.heat_bump = DEFAULT_HEAT_BUMP
        self._resolve_knobs()
        if self._passport_on:
            profile.record(self.backend.name, "session")
        # P4.5 just-in-time retrieval: the project's test command, detected ONCE
        # (fleet.detect_test_cmd is otherwise computed only inside verify() and
        # never surfaced to the working agent). The file-list + symbol tables are
        # built lazily on the first turn that actually has candidates.
        self._retr_built = False
        self._retr_files = ()
        self._retr_fileset = frozenset()
        self._retr_by_base = {}
        self._retr_by_stem = {}
        self._retr_symbols = ()
        self._retr_test_cmd = None
        if self._lv("workspace"):
            try:
                from . import fleet as _fleet
                cwd = getattr(self.session, "cwd", None)
                if cwd:
                    self._retr_test_cmd = _fleet.detect_test_cmd(cwd)
            except Exception:
                self._retr_test_cmd = None
        # P4.8 seed note #0 = the project's detected test command, computed ONCE here
        # (reusing the P4.5 probe when present, otherwise a single probe of its own —
        # so the seed is robust even when the workspace lever is off), never per send().
        _seed_cmd = self._retr_test_cmd
        if _seed_cmd is None:
            try:
                from . import fleet as _fleet_seed
                _seed_cwd = getattr(self.session, "cwd", None)
                _seed_cmd = _fleet_seed.detect_test_cmd(_seed_cwd) if _seed_cwd else None
            except Exception:
                _seed_cmd = None
        if _seed_cmd:
            self.notes.append("test command: " + _seed_cmd)
        # P3.1 meta record: one machine-readable header per session so a dead
        # transcript is self-describing (forge version, model ladder, cwd, mode).
        # EphemeralSession.log is a no-op, so internal agents never pollute a file.
        self.session.log("meta", v=TRACE_V, forge=__version__, model=self.backend.name,
                         ladder=[b.name for b in self.ladder], cwd=self.session.cwd,
                         mode=self.mode,
                         briefing=hashlib.md5(workspace.encode()).hexdigest()[:12] if workspace else None)

    def set_ladder(self, ladder):
        """Swap the model ladder live (conversation preserved)."""
        self.ladder = ladder
        self.tier = 0
        self.backend = ladder[0]
        self._aliases = profiles.resolve(self.backend.name).get("aliases", ())  # P5.4: re-resolve the family alias table
        self._resolve_knobs()   # P5.8: re-resolve per-model knobs for the new base rung

    def _resolve_knobs(self):
        """P5.8: resolve the per-model knobs (loop_threshold, num_predict, heat_bump)
        from the ACTIVE backend's passport and apply them. Called at construction and at
        every ladder swap (escalate / de-escalate / set_ladder) because the escalated
        rung is a DIFFERENT model with its own passport — resolving once in __init__ would
        leave the strong rung mis-tuned mid-turn. num_predict is pushed onto the backend
        itself (reaching the real OllamaBackend even inside a RecordingBackend wrapper);
        an engine without that knob (OpenAI-compat) simply doesn't carry the attribute.
        An empty passport returns the defaults, so this is a no-op for un-profiled models."""
        base = {"loop_threshold": DEFAULT_LOOP_THRESHOLD, "heat_bump": DEFAULT_HEAT_BUMP,
                "num_predict": backends.NUM_PREDICT}
        target = getattr(self.backend, "_inner", self.backend)  # unwrap a RecordingBackend
        if hasattr(target, "num_predict"):
            base["num_predict"] = target.num_predict
        try:
            k = profile.knobs(self.backend.name, base)
        except Exception:
            k = base
        self.loop_threshold = k.get("loop_threshold", DEFAULT_LOOP_THRESHOLD)
        self.heat_bump = k.get("heat_bump", DEFAULT_HEAT_BUMP)
        if hasattr(target, "num_predict"):
            target.num_predict = k.get("num_predict", target.num_predict)

    # ---- context management ----
    def _fill(self):
        """(tokens_used, window) for the current model, from the harness TOKEN
        LEDGER — the sum of per-message estimates (len//4 * tok_ratio) PLUS the
        per-step plan pin (appended outside self.messages). The ledger is
        authoritative because Ollama's warm KV-prefix cache makes prompt_eval_count
        report only the newly-evaluated SUFFIX on forge's append-only pattern, so
        the observed count collapses toward zero and the 0.70 gate never fires
        (Ollama then silently truncates at num_ctx). last_prompt_tokens is used only
        as a cross-check FLOOR that can never push fill DOWN (max), plus to
        recalibrate tok_ratio UP on an uncached call — never to shrink the estimate."""
        window = self.backend.effective_ctx() if hasattr(self.backend, "effective_ctx") else backends.ctx_cap()
        est = self._ledger_tokens()
        if self._reclaimed:
            # P4.2/P4.3: a structural/floor/turn-end pass reclaimed window THIS step;
            # last_prompt_tokens is still the STALE (pre-pass) count, so trust only
            # the fresh ledger (which already reflects the smaller messages).
            return int(est), window
        pe = getattr(self.backend, "last_prompt_tokens", 0)
        return int(max(est, pe)), window

    def _ledger_tokens(self):
        """Rebuild the per-message token ledger (index-aligned with self.messages)
        and return its sum plus the per-step plan pin estimate. Each estimate is
        len(content)//4 * tok_ratio; the plan pin is counted though it is appended
        OUTSIDE self.messages (prompt = self.messages + [pin])."""
        self.msg_tokens = [len(m["content"]) // 4 * self.tok_ratio for m in self.messages]
        total = sum(self.msg_tokens)
        if self._lv("plan_pin"):
            pin = self._pin_state()
            if pin:
                total += len(pin["content"]) // 4 * self.tok_ratio
        return total

    def _recalibrate(self, prompt):
        """Rebase tok_ratio against the backend's observed prompt_eval_count for the
        prompt just sent. On an UNCACHED call (flagged: first call after construction,
        and the call after a compaction rewrite) the reported count is the true
        full-prompt count → rebase the ledger. On every other call it is an up-only
        cross-check: a warm-cache count reports only the suffix (SMALLER than our
        estimate), so it can never corrupt the ratio downward, only raise it when we
        under-estimate the model's tokenizer."""
        pe = getattr(self.backend, "last_prompt_tokens", 0)
        if not pe:
            return
        est_chars = sum(len(m["content"]) for m in prompt) // 4
        if est_chars <= 0:
            return
        ratio = pe / est_chars
        if self._calibrate_pending:
            self.tok_ratio = ratio
            self._calibrate_pending = False
        elif ratio > self.tok_ratio:
            self.tok_ratio = ratio

    def _audit_prefix(self, step, n_stable=0):
        """FORGE_DEBUG audit (OFF by default): the KV-cache-warm prefix — the head
        (system + workspace), which is never compacted away — must be byte-stable
        between steps, or the warm-cache assumption behind the token ledger and
        turn-boundary scheduling is void. Hash it each step and warn (transcript +
        stderr) on any unexpected mutation."""
        if not os.environ.get("FORGE_DEBUG"):
            return
        end = self.head_len + n_stable
        blob = "".join(m.get("content", "") for m in self.messages[:end])
        h = hashlib.md5(blob.encode("utf-8", "replace")).hexdigest()
        if self._prefix_hash is not None and h != self._prefix_hash:
            self.session.log("prefix_mutation", v=TRACE_V, step=step,
                             prev=self._prefix_hash, now=h)
            try:
                import sys as _sys
                print(f"[FORGE_DEBUG] prefix mutated at step {step} — the KV-cache "
                      "prefix (head) is not byte-stable", file=_sys.stderr)
            except Exception:
                pass
        self._prefix_hash = h

    def maybe_compact(self, threshold=0.55):
        """P4.3 — proactive TURN-BOUNDARY compaction, called by the REPL after a turn
        completes (while the user reads the reply). A warm KV-prefix is invalidated
        for FREE at a turn boundary, so we compact EARLY — at a lower threshold than
        the in-turn 0.70 gate — to start the next turn with headroom. Runs the same
        structural + LLM + floor passes; the in-turn _compact (0.70) and the floor
        stay emergency-only. Safe to call with no pressure: each pass is a no-op below
        its threshold."""
        if not self._lv("compaction"):
            return
        self.ledger.refresh()
        # step=0: at a turn boundary the within-turn "failed obs older than N steps"
        # recency is moot; the read-supersede and write-echo rules (step-independent)
        # do the reclaiming. The most recent 8 messages are protected regardless.
        self._structural_compact(step=0, threshold=threshold)
        self._compact(threshold=threshold)
        self._floor()
        # A turn-boundary compaction already has its own durable `compact` log record;
        # clear the transient step-trace flag so it isn't mis-attributed to the NEXT
        # turn's first step. `_reclaimed` is left set so that turn's first _fill reads
        # the fresh (post-compaction) ledger instead of the previous turn's stale count.
        self._compacted = False

    def _obs_budget(self):
        """One char budget for a single observation, derived from the model's REAL
        window: ~8% of it (4 chars/token), hard-capped at 12000. This ends the old
        4000/12000 split-brain — one budget, used for both the transcript log and
        the message fed back to the model, so nothing is ever cut mid-content
        without a visible marker and no pointer outlives its budget."""
        window = self.backend.effective_ctx() if hasattr(self.backend, "effective_ctx") else backends.ctx_cap()
        return min(12000, int(window * 4 * 0.08))

    def _compact(self, threshold=0.70):
        """At `threshold` of the model's real window (0.70 in-turn; 0.55 at the
        turn boundary via maybe_compact), SUMMARIZE the older middle turns into a
        dense state note (instead of dropping them). System + workspace stay pinned,
        recent turns stay verbatim, the plan is pinned separately — nothing important
        is lost, the context just gets denser."""
        used, window = self._fill()
        if used < threshold * window:
            return
        self._sync_meta()                   # P4.2: meta must be aligned before we slice it
        head = self.messages[:self.head_len]
        tail = self.messages[-12:]          # keep plenty of recent context so reads aren't lost → no re-read loop
        middle = self.messages[self.head_len:-12]
        if len(middle) < 4:
            return
        self.on_event("compacting", used=used, window=window)
        summary = self._summarize(middle)
        note = {"role": "user", "content": "[Earlier progress, summarized to save context:]\n" + summary}
        self.messages = head + [note] + tail
        # P4.2: rewrite the parallel meta list on the SAME boundaries so it stays
        # index-aligned with self.messages, and mark the window reclaimed so _fill
        # reports the fresh (smaller) size instead of the stale token count.
        self.meta = self.meta[:self.head_len] + [{"kind": "summary"}] + self.meta[-12:]
        self._reclaimed = True
        # P4.3: a compaction rewrite breaks the KV prefix, so the NEXT generate is an
        # uncached call — re-arm calibration so prompt_eval_count can rebase the ledger.
        self._calibrate_pending = True
        # P4.1: any read/write observation that fell out of the retained window is
        # no longer in context — evict it so read-before-edit mechanically re-forces
        # a read. Then tell the model which files it DOES still hold.
        self._evict_compacted()
        held = self.ledger.held()
        if held:
            names = ", ".join(f"{os.path.relpath(p, self.session.cwd)}"
                              + (f" ({n}l)" if n is not None else "")
                              for p, n in sorted(held))
            note["content"] += "\n\nfiles you have read and still hold: " + names
        self.on_event("compact", window=window)
        # P3.1: persist the summary so a resume can reconstruct the compacted middle,
        # and flag this step so its trace records compacted=True.
        self.session.log("compact", v=TRACE_V, summary=summary, window=window)
        self._compacted = True

    def _summarize(self, msgs):
        convo = "\n\n".join(f"[{m['role']}] {m['content'][:1200]}" for m in msgs)[:16000]
        try:
            # summarize with the cheapest ladder model — fast and enough for this
            return self.ladder[0].chat(
                [{"role": "system", "content": SUMMARIZE_SYSTEM},
                 {"role": "user", "content": convo}]).strip()[:4000]
        except Exception:
            return f"[{len(msgs)} earlier steps omitted; continue from the recent turns and the plan below]"

    def _evict_compacted(self):
        """After a compaction rewrite of self.messages, flip in_context=False for
        every ledger entry whose observation message is no longer retained (matched
        by object identity — head/tail slices preserve the same dicts)."""
        live = {id(m) for m in self.messages}
        for e in list(self.ledger.entries.values()):
            if e.in_context and (e.obs_msg is None or id(e.obs_msg) not in live):
                self.ledger.evict(e.realpath)

    # ---- P4.2 parallel meta list + structural (deterministic) compaction ----
    def _sync_meta(self):
        """Pad/trim the parallel meta list so it aligns 1:1 with self.messages.
        Messages appended outside the tagged funnel points get a neutral record —
        this keeps every index valid before a structural walk reads self.meta[i]."""
        n = len(self.messages)
        if len(self.meta) < n:
            self.meta.extend({"kind": "msg"} for _ in range(n - len(self.meta)))
        elif len(self.meta) > n:
            del self.meta[n:]

    def _tag_last(self, rec):
        """Enrich the meta record for the most-recently-appended message."""
        self._sync_meta()
        if self.meta:
            self.meta[-1] = rec

    def _write_echo_stub(self, content):
        """Collapse a write_file assistant-JSON echo — whose `content` field is the
        ENTIRE file, never truncated — to a path+bytes+sha1 stub, keeping the JSON
        shape so the model still reads a consistent history. The file is on disk, so
        this is lossless. Returns the stub string, or None if it can't be collapsed."""
        try:
            act = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if act.get("action") != "write_file":
            return None
        body = act.get("content")
        if not isinstance(body, str):
            return None
        raw = body.encode("utf-8", "replace")
        stub = dict(act)
        stub["content"] = f"[elided — {len(raw)} bytes written, sha1 {hashlib.sha1(raw).hexdigest()[:12]}; recover from disk]"
        try:
            return json.dumps(stub)
        except (TypeError, ValueError):
            return None

    def _first_error_line(self, content, step):
        """Shrink a stale failed-action observation body to just its first error
        line (the salient part) — the rest of the traceback is rarely re-consulted."""
        marker = "Observation:\n"
        idx = content.find(marker)
        body = content[idx + len(marker):] if idx != -1 else content
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        if not first:
            return None
        return (f"  ⚠ earlier failed action (step {step}) — first error line kept:\n"
                f"Observation:\n{first}\n[error tail elided; re-run the action to see the full failure]")

    def _structural_compact(self, step, threshold=STRUCT_FILL):
        """P4.2 — a zero-model-call deterministic compaction pass, run each step
        BEFORE the LLM _compact. It reclaims mechanically-redundant window losslessly
        (every stub is recoverable from disk / the ledger): (a) an older read_file
        observation for a path since re-read/edited/rewritten is stubbed (the ledger
        knows the live copy); (b) a write_file assistant-JSON echo — the entire file
        content — collapses to a path+bytes+sha1 stub; (c) a failed-action observation
        body older than N steps shrinks to its first error line. It walks only the
        settled middle (never the head, never the most recent 8 messages) and marks
        the window reclaimed so the 70% LLM gate decides on the fresh size."""
        used, window = self._fill()
        if used < threshold * window or len(self.messages) <= self.head_len + 8:
            return
        self._sync_meta()
        # The ledger's CURRENT per-path observation binding: a read obs message whose
        # identity is not the live one has been superseded by a later read/edit/write.
        live = {e.realpath: id(e.obs_msg)
                for e in self.ledger.entries.values() if e.obs_msg is not None}
        changed = False
        for i in range(self.head_len, len(self.messages) - 8):
            m = self.messages[i]
            rec = self.meta[i]
            kind = rec.get("kind")
            content = m.get("content", "")
            if kind == "write_echo" and "[elided" not in content:
                stub = self._write_echo_stub(content)
                if stub is not None and len(stub) < len(content):
                    m["content"] = stub
                    changed = True
            elif (kind == "obs" and rec.get("action") == "read_file" and rec.get("path")
                    and "[superseded" not in content):
                p = rec["path"]
                if p in live and live[p] != id(m):
                    rel = os.path.relpath(p, self.session.cwd)
                    stub = (f"Observation:\n[read of {rel} at step {rec.get('step')} superseded — you "
                            "re-read or edited it later; the old copy is dropped. read_file it again if needed.]")
                    if len(stub) < len(content):
                        m["content"] = stub
                        changed = True
            elif (kind == "obs" and rec.get("ok") is False and rec.get("step") is not None
                    and step - rec["step"] > FAILED_OBS_AGE and "[error tail elided" not in content):
                stub = self._first_error_line(content, rec.get("step"))
                if stub is not None and len(stub) < len(content):
                    m["content"] = stub
                    changed = True
        if changed:
            self._reclaimed = True
            self.session.log("struct_compact", v=TRACE_V, step=step, window=window)

    def _floor(self):
        """P4.2 — the hard floor after the LLM _compact. The wedge case: tail is
        messages[-12:] and observations can each be thousands of chars, so head+tail
        alone can exceed a small window; _compact's `len(middle) < 4` guard then
        returns without summarizing and the session is PERMANENTLY wedged with no
        recourse. So if the window is still over the ceiling, hard-truncate the oldest
        tail observation bodies (head is never touched) until back under it. Lossy,
        but the only escape — and the truncated text is recoverable from disk."""
        window = self.backend.effective_ctx() if hasattr(self.backend, "effective_ctx") else backends.ctx_cap()
        ceiling = 0.70 * window

        def cfill():
            return sum(len(m["content"]) for m in self.messages) // 4

        if cfill() <= ceiling:
            return
        truncated = False
        i = self.head_len
        while cfill() > ceiling and i < len(self.messages) - 2:
            body = self.messages[i].get("content", "")
            if len(body) > 400 and "[hard-truncated" not in body:
                self.messages[i]["content"] = (
                    body[:200] + f"\n[... {len(body) - 200} chars hard-truncated to escape a "
                    "full-window wedge; re-read from disk if needed]")
                truncated = True
            i += 1
        if truncated:
            self._reclaimed = True
            self.session.log("floor", v=TRACE_V, window=window, used=cfill())

    def _pin_state(self):
        """P4.8: pin harness-owned scratch state — the living plan AND durable notes —
        into context each turn. Rebuilt per-step from self.plan/self.notes so both
        survive compaction verbatim (state held in the harness beats state held in
        the model's context). Returns None only when BOTH are empty: an empty plan
        must still emit notes, and notes-only must still pin. Rides the `plan_pin`
        lever at the call site, so ablating it removes the whole state pin."""
        parts = []
        if self.plan:
            parts.append("[current plan]\n" + "\n".join(self.plan))
        if self.notes:
            parts.append("[notes]\n" + "\n".join("- " + n for n in self.notes))
        if not parts:
            return None
        return {"role": "user", "content": "\n".join(parts)}

    def _add_note(self, text):
        """Append a scratch note if it is normalized-new (dedup on whitespace-folded,
        case-insensitive text), then FIFO-evict the oldest until both bounds hold
        (NOTES_CAP entries and NOTES_CHARS total chars). A single oversized note is
        kept rather than evicting itself into oblivion. Returns True if appended."""
        text = (text or "").strip()
        if not text:
            return False
        norm = " ".join(text.split()).lower()
        if any(" ".join(n.split()).lower() == norm for n in self.notes):
            return False
        self.notes.append(text)
        while len(self.notes) > 1 and (
                len(self.notes) > NOTES_CAP
                or sum(len(n) for n in self.notes) > NOTES_CHARS):
            self.notes.pop(0)
        return True

    # ---- P4.1 read cache + honest read-before-edit ----
    def _serve_cached_read(self, act, fp, step, trace):
        """A repeat read_file the ledger can answer without re-injecting the file.
        Returns True if served (observation appended), False to let a real read run.
        Unchanged in-context files → a one-line note; changed files → a capped diff
        since the last read. A never-read file, an evicted read, a new line-range, or
        a file with no cached baseline all fall through to a real read."""
        if not os.path.isfile(fp):
            return False
        e = self.ledger.get(fp)
        if e is None:
            return False
        rel = os.path.relpath(fp, self.session.cwd)
        st = self.ledger.status(fp)
        offset, limit = act.get("offset", 1), act.get("limit")
        if st == "current":
            if not self.ledger.covers(fp, offset, limit):
                return False                      # a genuinely new window → read it
            note = f"[unchanged since step {e.read_step} — {rel} is already in your context; not re-injecting it]"
            self._serve_read_obs(act, fp, note, trace, bind=False)
            return True
        if st == "changed":
            d = self.ledger.diff(fp, name=rel)
            if d is None:
                return False                      # no cached baseline → re-read fully
            if d == "":                           # touched but byte-identical
                self.ledger.record_read(fp, step, offset=offset, limit=limit)
                note = f"[unchanged since step {e.read_step} — {rel} is already in your context; not re-injecting it]"
                self._serve_read_obs(act, fp, note, trace, bind=False)
                return True
            note = (f"[{rel} CHANGED since you read it at step {e.read_step}. Unified diff since your "
                    f"last read (the whole file is NOT re-injected — apply this to your copy):]\n{d}")
            self.ledger.record_write(fp, step)    # baseline is now the current on-disk content
            self._serve_read_obs(act, fp, note, trace, bind=True)
            return True
        return False                              # unread / evicted → real read

    def _serve_read_obs(self, act, fp, note, trace, bind):
        """Append a harness-served read observation (note or diff) as if it were a
        real read result: same events/log surface, so loop detection and the trace
        stay honest. `bind` binds this message as the file's in-context observation."""
        trace["ok"] = True
        self.on_event("action", action="read_file", detail=act.get("path", ""), thought=act.get("thought", ""))
        self.session.log("action", action="read_file", args={"served": True, "path": act.get("path", "")},
                         thought=act.get("thought", ""))
        obs_msg = {"role": "user", "content": f"Observation:\n{note}"}
        self.messages.append(obs_msg)
        if bind:
            self.ledger.set_obs_msg(fp, obs_msg)
        self.session.log("observation", text=note, ok=True)
        self.on_event("observation", text=note, ok=True)

    def _read_gate_msg(self, kind, act, fp):
        """The read-before-edit block message, specific to WHY the file isn't
        current: never read, dropped from context by compaction, or changed on disk."""
        e = self.ledger.get(fp)
        path = act.get("path")
        st = self.ledger.status(fp)
        # P5.3: an anchored edit carries a line range — point the re-read at it, since
        # the numbers it relies on may have shifted (its own earlier splice) or gone stale.
        tail = ""
        if act.get("start_line") is not None:
            s, en = act.get("start_line"), act.get("end_line", act.get("start_line"))
            tail = f" Re-read lines {s}-{en} with line numbers first."
        if st == "changed" and e is not None:
            return (f"Blocked: {path} CHANGED on disk since you read it at step {e.read_step} — your copy "
                    "is stale. read_file it again and work from the current content before editing." + tail)
        if st == "evicted" and e is not None:
            return (f"Blocked: you read {path} at step {e.read_step}, but it's no longer in your context — "
                    "read_file it again before editing or overwriting it." + tail)
        return (f"Blocked: read {path} before editing or overwriting it — "
                "work from its actual current content, not memory. Use read_file first." + tail)

    def _edit_region_seen(self, act, fp):
        """True unless the edit targets lines OUTSIDE the (partial) range the model
        actually read. A whole-file read, or an `old` snippet we cannot locate,
        never blocks — this only catches editing a region a ranged read never saw."""
        e = self.ledger.get(fp)
        if e is None or e.whole or not e.spans:
            return True
        # P5.3 anchored dialect: the `anchor` guards only start_line; a miscounted
        # end_line would silently splice/delete lines the model never read. Require the
        # FULL [start,end] range to fall within a span it actually saw.
        if act.get("start_line") is not None:
            try:
                start = int(act.get("start_line"))
                end = int(act.get("end_line", start))
            except (TypeError, ValueError):
                return True
            if end < start:
                return True
            return any(a <= start and end <= b for a, b in e.spans)
        old = act.get("old", "")
        try:
            with open(fp, errors="replace") as f:
                text = f.read()
        except OSError:
            return True
        idx = text.find(old)
        if idx == -1:
            return True
        start = text.count("\n", 0, idx) + 1
        end = start + old.count("\n")
        return any(a <= start and end <= b for a, b in e.spans)

    def _legal_actions(self):
        """P5.1: the actions LEGAL to emit right now — all actions, minus the
        mutating ones in plan mode, intersected with self.allowed. Drives the
        per-step action grammar so an illegal action is grammatically
        unrepresentable (not merely rejected post-hoc)."""
        legal = set(ALL_ACTIONS)
        if self.mode == "plan":
            legal -= set(self.MUTATING)
        if self.allowed is not None:
            legal &= set(self.allowed)
        return legal

    def _missing_required(self, kind, act):
        """P5.1: the required fields (beyond thought/action) an action left out or
        blank — the fields its grammar variant forces. fleet_send (its no-message
        form lists sessions) and say (an empty message just ends the turn) are
        exempt, so only a genuinely-unexecutable action is flagged."""
        if kind in ("fleet_send", "say"):
            return []
        reqs = required_fields(kind)
        # P5.3: edit_file's line-anchored dialect ({start_line,end_line,anchor,new})
        # substitutes for the exact {old,new} dialect — enforce ITS fields instead so
        # an anchored edit isn't rejected for a missing `old`.
        if kind == "edit_file" and act.get("start_line") is not None:
            reqs = ["path", "start_line", "end_line", "anchor", "new"]
        out = []
        for f in reqs:
            v = act.get(f)
            # empty string counts as missing (matches the old pathless check), except
            # `new`: an empty replacement is a legal deletion.
            if v is None or (isinstance(v, str) and not v and f != "new"):
                out.append(f)
        return out

    def _generate(self, prompt, schema=None, temperature=0.0, stream_say=True):
        """Stream the model's action. When it turns out to be a `say`, emit the
        message text live (token by token) via on_event('token'). `schema` overrides
        the per-step grammar (P5.1 uses it to force a single action's variant on a
        missing-required resend); otherwise the legal-action grammar is built here.
        `temperature` is the sampling heat: 0.0 for a turn's first greedy try, the
        P5.5 retry-heat on a nudge resend, or the P5.2 best-of-N rung on a resample;
        `stream_say=False` suppresses live token emission for a throwaway resample
        candidate that may or may not be chosen."""
        if self._lv("schema"):
            if schema is None:
                schema = build_schema(self._legal_actions(), self.mode)
        else:
            schema = None
        if not hasattr(self.backend, "stream"):
            return self.backend.chat(prompt, schema=schema, temperature=temperature)
        raw, emitted, is_say = "", 0, False
        try:
            for chunk in self.backend.stream(prompt, schema=schema, temperature=temperature):
                if self.stop.is_set():
                    break
                raw += chunk
                if stream_say and not is_say and '"say"' in raw:
                    is_say = True
                if is_say:
                    msg = _partial_message(raw)
                    if msg is not None and len(msg) > emitted:
                        self.on_event("token", text=msg[emitted:])
                        emitted = len(msg)
        except Exception:
            if not raw:
                raise
        return raw

    def _salvage(self, raw):
        """P5.4 deterministic JSON salvage. Called from the malformed branch BEFORE a
        strike is counted: recover an action object that a small model wrapped in ```
        fences, prefixed with prose, or capped with a trailing comma. Three stdlib
        stages, each retried with json.loads, applied cumulatively; returns
        (parsed_dict, stage_name) on the first that yields a dict, or (None, None) for
        the genuinely-unsalvageable case (e.g. NUM_PREDICT truncation — no complete
        top-level object exists). Only a dict counts: a bare list/number is not an
        action. Pure telemetry-and-recovery; never raises."""
        if not isinstance(raw, str):
            return None, None

        def _obj(s):
            try:
                v = json.loads(s)
            except (ValueError, TypeError):
                return None
            return v if isinstance(v, dict) else None

        # stage 1 — strip surrounding whitespace + a leading/trailing markdown fence.
        s = _FENCE_RE.sub("", raw.strip()).strip()
        o = _obj(s)
        if o is not None:
            return o, "fence"

        # stage 2 — escape-aware brace scan: slice the first complete top-level object
        # (drops a prose prefix/suffix). No object → nothing more to try.
        sliced = _first_json_object(s)
        if sliced is None:
            return None, None
        s = sliced
        o = _obj(s)
        if o is not None:
            return o, "brace"

        # stage 3 — drop trailing commas before } or ] and retry once more.
        s = _TRAILING_COMMA_RE.sub(r"\1", s)
        o = _obj(s)
        if o is not None:
            return o, "trailing_comma"

        return None, None

    _STR_FIELDS = ("path", "command", "target", "message", "pattern", "old", "new", "anchor")

    def _coerce_str_fields(self, act):
        """Advisory (non-constrained) engines can emit a non-string where the grammar
        would force one (e.g. {"target": 0}, a numeric path); a downstream .strip() or
        os.path.join then raises and crashes the turn. Coerce known string fields to str."""
        for f in self._STR_FIELDS:
            v = act.get(f)
            if v is not None and not isinstance(v, str):
                act[f] = str(v)

    def _alias_path(self, act):
        """P3.2 alias_repair: fill a missing `path` from a small model's aliases
        (filename/file/…). The alias table is resolved once in __init__ from the
        backend name (P5.4 profiles.py). No-op unless the alias_repair lever is on."""
        if self._lv("alias_repair") and act.get("action") in ("read_file", "write_file", "edit_file") and not act.get("path"):
            for alias in self._aliases:
                if isinstance(act.get(alias), str) and act[alias]:
                    act["path"] = act[alias]
                    if self._passport_on:   # P5.8 passport telemetry: this model mis-names the path field
                        profile.record(self.backend.name, "alias_repair")
                    break

    def _resend_variant(self, act, kind, pin, step, missing, trace):
        """P5.1: a parsed action dropped a required field. Rather than burning a
        whole step on a text nudge, re-ask ONCE with ONLY this action's variant
        grammar-forced (const action + its required fields) plus a one-line hint for
        advisory engines. The resend is logged as its own `model` record so replay
        stays 1:1 with the backend calls. Returns the (possibly-corrected) act, its
        kind, and its still-missing required fields."""
        variant = ACTION_VARIANTS.get(kind)
        if variant is None:
            return act, kind, missing
        hint = (f"Your `{kind}` action was missing required field(s): "
                f"{', '.join('`' + m + '`' for m in missing)}. "
                f"Resend the SAME {kind} action as ONE complete JSON object.")
        self.messages.append({"role": "user", "content": hint})
        prompt = self.messages + ([pin] if pin else [])
        raw2 = self._generate(prompt, schema=variant)
        self.session.log("model", v=TRACE_V, raw=raw2, tier=self.tier,
                         prompt_tokens=getattr(self.backend, "last_prompt_tokens", 0))
        try:
            act2 = json.loads(raw2)
        except json.JSONDecodeError:
            return act, kind, missing           # unparseable resend → old text-nudge fallback
        if not isinstance(act2, dict):
            return act, kind, missing
        self.messages.append({"role": "assistant", "content": raw2})
        self._tag_last({"kind": "assistant", "action": act2.get("action"), "step": step})
        self._alias_path(act2)
        kind2 = act2.get("action")
        trace["resent"] = True
        return act2, kind2, self._missing_required(kind2, act2)

    def _resample(self, act, kind, pin, step, base_score, trace):
        """P5.2 best-of-N: the greedy action is a certain miss (dry_run == 0). Re-ask
        the SAME prompt at rising temperature, score each candidate with the free
        verifier, and keep the argmax. The greedy raw is ALREADY messages[-1] (the
        P3.1 try-body / a P5.1 resend appended it), so the winning candidate REPLACES
        messages[-1] — otherwise the transcript would show an action that never ran.
        A candidate that won't parse, is incomplete, is mode-gated, or is a `say` (a
        turn-ender, not a drop-in for a file action) is skipped. All samples missing
        → keep the greedy original (the teaching failure), a strict superset of the
        pre-P5.2 loop. Resamples never lengthen the message list."""
        best_score, best_act, best_raw = base_score, act, self.messages[-1]["content"]
        base_prompt = self.messages[:-1] + ([pin] if pin else [])
        samples = 0
        for temp in (0.5, 0.8):
            # P5.5: resample floors at the best-of-N rung but never samples BELOW the
            # accumulated retry-heat — a hot turn stays at least as perturbed.
            raw = self._generate(base_prompt, temperature=max(self._heat, temp), stream_say=False)
            samples += 1
            try:
                cand = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(cand, dict) or cand.get("action") == "say":
                continue
            self._alias_path(cand)
            ck = cand.get("action")
            if self._missing_required(ck, cand) or self._gate_silent(ck, cand):   # no prompt while scoring
                continue
            # a candidate that drifts to an edit/write of a file the model hasn't read
            # must not slip past the read-before-edit gate the greedy already cleared —
            # skip it (and never probe/leak an unread file's content-existence).
            if ck in ("edit_file", "write_file") and self._lv("read_gate"):
                cfp = os.path.realpath(os.path.join(self.session.cwd, cand.get("path", "")))
                if os.path.isfile(cfp) and not self.ledger.current(cfp):
                    continue
            s, _r = dry_run(cand, self.session.cwd)
            if s > best_score:
                best_score, best_act, best_raw = s, cand, raw
        replaced = best_act is not act
        self.session.log("resample", v=TRACE_V, step=step, samples=samples,
                         base_score=base_score, best_score=best_score, replaced=replaced)
        trace["resampled"] = True
        if replaced:
            trace["resample_win"] = best_score
            # replace the greedy assistant echo so the transcript matches what runs
            self.messages[-1] = {"role": "assistant", "content": best_raw}
            self._tag_last({"kind": "assistant", "action": best_act.get("action"), "step": step})
        return best_act, best_act.get("action")

    # ---- P5.7 unified stuck ledger + step-scoped borrowing + tier decay ----
    def _bump_stuck(self, kind, delta):
        """Move the unified per-turn stuck score and record the event. A positive delta
        (a stuck signal: malformed / loop / tool failure) also breaks the clean streak
        and, once within two of the escalation threshold, pre-warms the next rung so a
        borrow or escalation pays no cold-load latency. A negative delta (verified
        progress) only relaxes the score."""
        self.stuck["score"] = max(0.0, self.stuck["score"] + delta)
        self.stuck["events"].append((kind, delta))
        if delta > 0:
            self.clean_streak = 0
            if self.stuck["score"] >= self.stuck_at - 2:
                self._prewarm_next()

    def _prewarm_next(self):
        """Pre-warm the next-stronger rung in a background thread so escalation /
        borrowing is instant. Fires at most once per rung per turn, only when a stronger
        rung exists and the escalation lever is on."""
        if self._prewarmed or not self._lv("escalation"):
            return
        nxt = self.tier + 1
        if nxt >= len(self.ladder):
            return
        b = self.ladder[nxt]
        if not hasattr(b, "warm"):
            return
        self._prewarmed = True
        try:
            threading.Thread(target=b.warm, daemon=True).start()
        except Exception:
            pass

    def _deescalate(self):
        """Tier decay: relax one rung back toward the cheap model (propose-small again),
        swapping self.backend and re-resolving the family alias table. No-op at the base
        rung. Called at the turn boundary (unless sticky_escalation) and after
        CLEAN_DECAY consecutive clean steps."""
        if self.tier <= 0:
            return
        self.tier -= 1
        self.backend = self.ladder[self.tier]
        self._aliases = profiles.resolve(self.backend.name).get("aliases", ())
        self._resolve_knobs()   # P5.8: re-tune for the rung we dropped back to
        self.clean_streak = 0
        self._prewarmed = False
        self.on_event("deescalate", model=self.backend.name)
        self.session.log("deescalate", v=TRACE_V, model=self.backend.name)

    def _borrow(self, prompt, step, trace):
        """Buy ONE generation from the next-stronger rung WITHOUT swapping self.backend
        (unlike escalation, which sticks). Used at the exact points the cheap model is
        provably stuck — the third malformed strike, and a per-signature failure that
        keeps repeating — so the strong rung proposes one action and the cheap model
        resumes with it as an in-context demonstration. Warms the rung, generates under
        the CURRENT step's grammar, appends the borrowed action tagged with provenance,
        and returns (raw, act) — or (None, None) when the escalation lever is off, there
        is no stronger rung, we're replaying, or the borrow yielded no usable action."""
        if not self._lv("escalation") or getattr(self.backend, "replay", False):
            return None, None
        nxt = self.tier + 1
        if nxt >= len(self.ladder):
            return None, None
        b = self.ladder[nxt]
        try:
            if hasattr(b, "warm"):
                b.warm()
            schema = build_schema(self._legal_actions(), self.mode) if self._lv("schema") else None
            raw = b.chat(prompt, schema=schema)
        except Exception:
            return None, None
        try:
            act = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            act, _ = self._salvage(raw)
        if not isinstance(act, dict):
            return None, None
        self._alias_path(act)
        self.stuck["borrows"] += 1
        trace["borrowed"] = b.name
        self.on_event("borrow", model=b.name, action=act.get("action"))
        self.session.log("borrow", v=TRACE_V, step=step, model=b.name, action=act.get("action"))
        self.messages.append({"role": "user", "content":
            f"[borrowed from {b.name}] The cheap model was stuck, so a stronger local "
            f"rung proposed this next action given your exact context:\n{raw}"})
        return raw, act

    def _absorb_inbox(self):
        for m in self.session.drain():
            tag = "[user (mid-run — steer accordingly)]" if m["from"] == "user" \
                else f"[fleet message from {m['from']}]"
            self.messages.append({"role": "user", "content": f"{tag}: {m['text']}"})
            self.session.log("inbox", sender=m["from"], text=m["text"])
            self.on_event("inbox", sender=m["from"], text=m["text"])

    MUTATING = ("bash", "write_file", "edit_file", "fleet_send")

    def _approval_key(self, act):
        """What an 'always' approval covers: bash by command head (bash:git),
        other actions by kind (edit_file)."""
        if act.get("action") == "bash":
            head = (act.get("command") or "").strip().split()
            return f"bash:{head[0] if head else ''}"
        return act.get("action")

    def _gate_silent(self, kind, act):
        """Side-effect-free gate check for SCORING throwaway resample candidates:
        True if the action would be blocked, WITHOUT prompting the user or persisting
        an approval (which _gate does in manual mode). Auto never blocks; plan blocks
        mutating actions; manual blocks a mutating action unless already 'always'-approved."""
        if kind not in self.MUTATING or self.mode == "auto":
            return False
        if kind == "fleet_send" and (not act.get("message")
                                     or act.get("target", "").strip().lower() in ("", "list", "sessions")):
            return False
        if self.mode == "plan":
            return True
        return self._approval_key(act) not in self.approvals   # manual: blocked unless pre-approved (no prompt)

    def _gate(self, kind, act):
        """Mode gate for mutating actions. Returns a block message, or None to
        proceed. plan: read-only only. manual: ask the user y/always/no."""
        if kind not in self.MUTATING or self.mode == "auto":
            return None
        if kind == "fleet_send" and (not act.get("message")
                                     or act.get("target", "").strip().lower() in ("", "list", "sessions")):
            return None                       # listing sessions is read-only
        if self.mode == "plan":
            return (f"plan mode: '{kind}' would change things and is not allowed. Investigate with "
                    "read-only tools (read_file, list_files, grep, glob), then present your plan with "
                    "`say` — the user will switch modes to execute it.")
        key = self._approval_key(act)
        if key in self.approvals:
            return None
        detail = (act.get("command") or act.get("path") or act.get("target") or "")[:120]
        resp = self.approve(f"{kind} {detail}".strip() if detail else kind)
        if resp == "always":
            self.approvals.add(key)
            try:
                from . import config as _cfg
                _cfg.set_key("approvals", sorted(self.approvals))
            except OSError:
                pass
            return None
        if resp == "yes":
            return None
        return ("the user DECLINED this action. Do not retry it as-is — take a different approach, "
                "or `say` to ask them how to proceed.")

    # "Act, don't narrate" guard: an intent phrase near the start of a message followed
    # (within a few words) by a file-work verb — "I'll implement …", "let me fix …",
    # "I'm going to rewrite …". Deliberately excludes explain/describe/answer verbs so a
    # legitimate answer to a question is not caught.
    _NARRATE_RE = re.compile(
        r"\b(i'?ll|i will|i'?m going to|i am going to|let me|let'?s|i plan to|going to|"
        r"about to|now i'?ll|first[,\s]|next[,\s])\b.{0,60}?"
        r"\b(implement|write|create|add|fix|update|modify|edit|make|build|change|"
        r"refactor|rewrite|apply|start)\b", re.I)

    def _narration_bounce(self, msg):
        """Autonomous 'act, don't narrate' guard. A `say` that only DESCRIBES upcoming
        file work — after a turn that changed NOTHING — is the model narrating its intent
        instead of doing it (which the system prompt forbids). Bounce it ONCE per turn so
        the model does the work rather than ending the turn. Returns a bounce message, or
        None. Never fires outside autonomous mode, after any mutation, for a plain answer,
        or a second time in one turn."""
        if not self.autonomous or self._mutated or self._narrated:
            return None
        if not (msg and self._NARRATE_RE.search(msg.strip()[:160])):
            return None
        self._narrated = True
        self.session.log("narrate_bounce", msg=(msg or "")[:120])
        return ("You described what you're about to do but haven't actually done it — no "
                "files changed this turn. Do the work NOW: make the edits/writes and run the "
                "check, THEN report the outcome. Don't stop just to narrate your plan.")

    def _done_gate(self):
        """P2.1 — SYNCHRONOUS done-gate on `say`. If this turn mutated files and
        nothing has verified them, the HARNESS itself runs the project's real test
        command (zero model tokens) and grounds acceptance in the exit code. It
        bounces at most ONCE per turn — the second `say` always passes, no
        livelock — and NEVER emits an observation-ok event (existing tests key on
        that stream); it uses the distinct 'done_check' event plus a plain
        user-message append. Returns a bounce message to append+continue, or None
        to accept the say."""
        if not self._mutated or self._verified or self._bounced:
            return None
        from . import fleet, tools
        try:
            cmd = fleet.detect_test_cmd(self.session.cwd)
        except Exception:
            cmd = None
        if not cmd:
            return None                       # no detectable suite → accept gracefully
        obs2, ok2 = tools._run(cmd, self.session.cwd, timeout=tools.BASH_TIMEOUT * 3, stop=self.stop)
        if _cmd_missing(obs2):                # exit 127 / not installed → not a usable test cmd
            return None
        if ok2:
            self.session.log("verified", cmd=cmd, ok=True)
            self.on_event("done_check", cmd=cmd, ok=True)
            self.messages.append({"role": "user", "content": f"[done-gate] `{cmd}` passed"})
            return None
        self._bounced = True
        self.on_event("done_check", cmd=cmd, ok=False)
        tail = "\n".join((obs2 or "").splitlines()[-15:])
        return (f"[done-check] `{cmd}` FAILS:\n{tail}\n— fix before finishing, or say why "
                "this failure is out of scope")

    # ---- P4.5 just-in-time retrieval -----------------------------------------
    def _retrieval_ensure(self):
        """Build the file-list + symbol lookup tables once per session (lazy: only
        the first turn whose prompt actually has candidates pays the I/O). Fully
        best-effort — every failure degrades to fewer matches, never a crash."""
        if self._retr_built:
            return
        self._retr_built = True
        cwd = getattr(self.session, "cwd", None)
        if not cwd:
            return
        try:
            from . import workspace as _ws
            files = _ws._source_files(cwd, 5000)
        except Exception:
            files = []
        self._retr_files = tuple(files)
        self._retr_fileset = frozenset(files)
        for f in files:
            base = f.rsplit("/", 1)[-1]
            self._retr_by_base.setdefault(base, []).append(f)
            stem = base.rsplit(".", 1)[0] if "." in base else base
            if stem:
                self._retr_by_stem.setdefault(stem, []).append(f)
        try:
            from . import index as _index
            self._retr_symbols = tuple(_index.refresh(cwd) or ())
        except Exception:
            self._retr_symbols = ()

    def _match_files(self, cands):
        """Resolve path/word candidates to real repo files by exact path, path
        suffix, basename, then basename-without-extension. Capped, deduped."""
        files = self._retr_files
        if not files:
            return []
        hits, seen = [], set()
        for cand in cands:
            cand = cand.strip("/")
            if not cand:
                continue
            matched = None
            if cand in self._retr_fileset:
                matched = [cand]
            elif "/" in cand:
                matched = [f for f in files if f.endswith("/" + cand)] or None
            if matched is None:
                base = cand.rsplit("/", 1)[-1]
                matched = self._retr_by_base.get(base) or self._retr_by_stem.get(base)
            for f in (matched or []):
                if f not in seen:
                    seen.add(f)
                    hits.append(f)
                    if len(hits) >= RETR_MAX_FILES:
                        return hits
        return hits

    def _match_symbols(self, idents):
        """Resolve identifiers to indexed symbol definitions: exact (full dotted
        name or its simple tail) first, then a bounded prefix fill (candidate ≥4
        chars). Index order is preserved, so results are deterministic."""
        syms = self._retr_symbols
        if not syms or not idents:
            return []
        hits, seen = [], set()

        def _take(s):
            key = (s.get("path"), s.get("lineno"), s.get("name"))
            if key not in seen:
                seen.add(key)
                hits.append(s)

        for s in syms:
            name = s.get("name", "")
            if name in idents or name.rsplit(".", 1)[-1] in idents:
                _take(s)
                if len(hits) >= RETR_MAX_SYMS:
                    return hits
        longs = [i for i in idents if len(i) >= 4]
        if longs:
            for s in syms:
                simple = s.get("name", "").rsplit(".", 1)[-1]
                if any(simple != c and simple.startswith(c) for c in longs):
                    _take(s)
                    if len(hits) >= RETR_MAX_SYMS:
                        break
        return hits

    def _retrieval_note(self, user_text):
        """The one-shot '[retrieved context]' note, or None when nothing in the
        prompt resolves to a real path/symbol (the load-bearing skip rule)."""
        paths, idents = _retrieval_extract(user_text)
        if not paths and not idents:
            return None
        self._retrieval_ensure()
        files = self._match_files(set(paths) | set(idents))
        syms = self._match_symbols(idents)
        if not files and not syms:
            return None                          # nothing matched → inject NOTHING
        cwd = getattr(self.session, "cwd", "") or ""
        lines = [RETR_TAG]
        if files:
            lines.append("Files:")
            lines += [f"- {_retr_file_line(cwd, f)}" for f in files[:RETR_MAX_FILES]]
        if syms:
            lines.append("Symbols:")
            for s in syms[:RETR_MAX_SYMS]:
                sig = (s.get("signature") or "").strip()
                lines.append(f"- {s.get('name')} — {s.get('path')}:{s.get('lineno')}"
                             + (f"  {sig}" if sig else ""))
        if self._retr_test_cmd:
            lines.append(f"Test: {self._retr_test_cmd}")
        note = "\n".join(lines)
        if len(note) > RETR_CHAR_CAP:
            note = note[:RETR_CHAR_CAP].rstrip() + "\n… (truncated)"
        return note

    def _prepare_and_generate(self, step):
        """One step's PREPARE + GENERATE phase (no control-flow exits — pure): absorb any
        inbox messages, refresh the file-state ledger, run the compaction passes, build the
        prompt (messages + plan pin), generate the raw action, recalibrate the token ledger,
        and log the raw output. Returns (raw, prompt, pin) for the rest of the step."""
        self._absorb_inbox()
        # P6.6: push background-process death notices the instant they happen — a server that
        # crashes mid-task is otherwise invisible and the model debugs a phantom for five steps.
        from .tools import bg_events
        for notice in bg_events():
            self.messages.append({"role": "user", "content": notice})
            self.session.log("inbox", sender="background", text=notice)
            self.on_event("inbox", sender="background", text=notice)
        # P4.1: re-stat tracked files so a bash/redirect/edit that changed a file's mtime
        # since it was read is caught before the gate/read-cache consult the ledger.
        self.ledger.refresh()
        if self._lv("compaction"):
            self._structural_compact(step)   # P4.2: deterministic pass first (zero model calls)
            self._compact()                  # then the LLM summarizer at 70% (emergency in-turn gate)
            self._floor()                    # then the hard floor: escape a full-window wedge
        self._audit_prefix(step)             # P4.3: FORGE_DEBUG prefix-mutation audit (no-op by default)
        pin = self._pin_state() if self._lv("plan_pin") else None
        prompt = self.messages + ([pin] if pin else [])
        self.on_event("thinking")
        # P5.5 retry-heat: first try is greedy (heat 0.0); each nudge bumps heat so a resend
        # is perturbed rather than a byte-identical greedy re-emission.
        raw = self._generate(prompt, temperature=self._heat)
        self._reclaimed = False              # P4.2: the token count now reflects any compaction
        # P4.3: rebase the token ledger against the observed prompt_eval_count for the EXACT
        # prompt just sent — an uncached call rebases tok_ratio; a warm-cache one is up-only.
        self._recalibrate(prompt)
        # P3.3 flight recorder: persist the RAW output of every step (incl. the malformed ones
        # the parse discards); prompt_tokens lets a replay reproduce compaction timing.
        self.session.log("model", v=TRACE_V, raw=raw, tier=self.tier,
                         prompt_tokens=getattr(self.backend, "last_prompt_tokens", 0))
        return raw, prompt, pin

    def _do_fleet_send(self, act, trace):
        """The fleet_send action: an empty/`list` target (or no message) returns the reachable
        roster; otherwise deliver to the named forge/Claude peer. Records the observation; the
        caller always continues to the next step afterward."""
        from . import fleet
        target, msg = act.get("target", ""), act.get("message", "")
        try:
            if target.strip().lower() in ("", "list", "sessions") or not msg:
                obs, ok = f"Reachable sessions (forge + Claude Code): {fleet.roster()}", True
            else:
                peer = fleet.send(target, msg, sender=self.session.name,
                                  sender_cwd=self.session.cwd, sender_sid=self.session.sid)
                runtime = " claude" if peer.get("kind") == "claude" else ""
                obs, ok = f"delivered to{runtime} {peer['name']} ({peer['sid'][:8]})", True
        except SystemExit as e:
            obs, ok = str(e), False
        trace["ok"] = ok
        self.session.log("action", action="fleet_send", args={"target": target}, thought=act.get("thought", ""))
        self.on_event("action", action="fleet_send", detail=target, thought=act.get("thought", ""))
        self.on_event("observation", text=obs, ok=ok)
        self.messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    def _execute_and_record(self, kind, act, raw, step, trace):
        """Run the action and RECORD everything about it (straight-line, no loop exits).
        Executes; on success resets retry-heat, harvests the exemplar, relaxes the stuck
        score / decays a rung; always populates the file-state ledger, sets the mutated/
        verified done-gate flags, classifies a bash run (ran-tests vs file-touching), emits
        the diff + observation events, and logs the shaped observation. Returns
        (obs, ok, budget, recorded_fp) for the failure-accounting + compaction that follow."""
        # capture the before-content so we can show a real diff after a write — and, P6.5,
        # after a BATCH edit (its per-hunk old/new aren't in the action, so a full pre/post
        # diff is the only way to render it).
        before = ""
        if kind == "write_file" or (kind == "edit_file" and act.get("edits")):
            _fp = os.path.join(self.session.cwd, act.get("path", ""))
            if os.path.isfile(_fp):
                try:
                    with open(_fp, errors="replace") as _f:
                        before = _f.read()
                except OSError:
                    pass
        self.session.log("action", action=kind, args={k: act.get(k) for k in ("command", "path") if act.get(k)}, thought=act.get("thought", ""))
        self.on_event("action", action=kind, thought=act.get("thought", ""),
                      detail=act.get("command") or act.get("path") or act.get("pattern") or "")
        obs, ok = execute(act, self.session.cwd, stop=self.stop)
        trace["ok"] = ok
        if ok:
            self._heat = 0.0   # P5.5: a clean execution unsticks — back to greedy
            exemplars.record(self.backend.name, kind, raw)   # P5.6: harvest this valid action
            # P5.7 verified progress: relax the stuck score + advance the clean streak;
            # CLEAN_DECAY clean steps in a row decay one rung.
            self._bump_stuck("progress", -STUCK_PROGRESS_W)
            self.clean_streak += 1
            if self.clean_streak >= CLEAN_DECAY and self.tier > 0:
                self._deescalate()
        budget = self._obs_budget()
        # P4.1 ledger population — a ranged read records only its span; a write/edit ingests
        # the new content as the cached (diffable) version.
        recorded_fp = None
        if ok and act.get("path"):
            _rp = os.path.realpath(os.path.join(self.session.cwd, act["path"]))
            if kind == "read_file":
                self.ledger.record_read(_rp, step, offset=act.get("offset", 1), limit=act.get("limit"))
                recorded_fp = _rp
            elif kind == "write_file":
                self.ledger.record_write(_rp, step, content=act.get("content"))
                recorded_fp = _rp
            elif kind == "edit_file":
                if act.get("start_line") is not None:
                    # P5.3: an anchored splice shifts every line number below it, so the
                    # numbered read is now stale. Evict to FORCE a fresh numbered re-read.
                    self.ledger.evict(_rp)
                else:
                    self.ledger.record_write(_rp, step)  # re-read the edited file from disk
                    recorded_fp = _rp
        if ok and kind in ("write_file", "edit_file") and act.get("path"):
            self._mutated.add(os.path.realpath(os.path.join(self.session.cwd, act["path"])))
            self._verified = False
        if ok and kind == "run_tests":
            self._verified = True             # P6.3: a passing run_tests satisfies the done-gate
        if ok and kind == "bash":
            _cmd = act.get("command", "")
            from . import fleet as _fleet
            if _is_test_cmd(_cmd, self.session.cwd):
                self._verified = True         # the model ran the suite itself
            elif _fleet.bash_mutates(_cmd):
                self._mutated.add("<bash>")   # a file-touching bash still gates `say`
                self._verified = False
        if ok and kind == "edit_file":
            # P5.8 passport telemetry: classify the edit from execute()'s (fuzzy)/(exact) suffix.
            if self._passport_on:
                profile.record(self.backend.name,
                               "fuzzy_edit" if "fuzzy" in (obs or "") and "exact" not in (obs or "") else "exact_edit")
            if act.get("edits"):     # P6.5 batch: per-hunk old/new aren't here — render a full pre/post diff
                after = ""
                try:
                    with open(os.path.join(self.session.cwd, act.get("path", "")), errors="replace") as _f:
                        after = _f.read()
                except OSError:
                    pass
                self.on_event("diff", path=act.get("path", ""), old=before, new=after)
            else:
                self.on_event("diff", path=act.get("path", ""), old=act.get("old", ""), new=act.get("new", ""))
        elif ok and kind == "write_file":
            self.on_event("diff", path=act.get("path", ""), old=before, new=act.get("content", ""))
        self.session.log("observation", text=shape(obs, budget), ok=ok)
        self.on_event("observation", text=obs, ok=ok)
        return obs, ok, budget, recorded_fp

    def _handle_failure(self, kind, sig, obs, step, trace):
        """Account a FAILED action (P5.7 stuck ledger + escalation). Returns
        (tag, borrow_now, control):
          control None            → proceed: caller appends the observation prefixed with `tag`;
          control "continue"      → we escalated to a stronger rung (fresh context) — caller
                                     skips the observation append and re-generates;
          control ("return", msg) → stuck with no rung left to try — caller ends the turn.
        `borrow_now` asks the caller to borrow one strong action AFTER the failed observation
        is in context (the P5.7 sig-fail borrow trigger)."""
        sig_fails = self.stuck["sig_fails"]
        sig_fails[sig] = sig_fails.get(sig, 0) + 1
        self._bump_stuck("fail", STUCK_FAIL_W)   # P5.7 tool failure +1
        self.stuck["last_err_by_sig"][sig] = next(
            (ln.strip() for ln in (obs or "").splitlines() if ln.strip()), "")[:200]
        self._heat = min(0.7, self._heat + self.heat_bump)   # P5.5 / P5.8
        tag = "  ⚠ this action FAILED — diagnose the cause before retrying.\n"
        borrow_now = False
        # per-command repeat (survives interleaved successes, unlike a consecutive counter)
        if self._lv("loop_detect") and sig_fails[sig] >= SIG_FAIL_BORROW_AT:
            self._bump_stuck("fail_loop", STUCK_LOOP_W)   # P5.7 sustained per-sig loop +2
            trace["loop_trip"] = True
            self.on_event("loop")
            self.session.log("loop", v=TRACE_V, step=step, sig=sig, cause="fail", count=sig_fails[sig])
            if self._passport_on:
                profile.record(self.backend.name, "loop")   # P5.8 passport telemetry
            tag = (f"  ⚠ `{sig}` has now failed {sig_fails[sig]} times. STOP retrying this exact thing. "
                   "Change approach entirely: re-read the real file/error, rewrite with write_file instead of edit_file, "
                   "or `say` to tell the user you're stuck and exactly what failed.\n")
            borrow_now = True   # P5.7 sig-fail borrow (fires after the obs is in context)
        # stuck: escalate to a stronger LOCAL model (unified score over threshold) — same task,
        # same context — rather than grinding or giving up. Takes precedence over the borrow.
        if self.stuck["score"] >= self.stuck_at:
            if self._lv("escalation") and self.tier < len(self.ladder) - 1:
                self._escalate(obs, trace)
                return tag, borrow_now, "continue"
            # Be accurate about WHY we stopped: only claim "even after escalating" if we
            # actually climbed a rung; a single-model ladder never escalated.
            lead = ("I'm stuck even after escalating through the local models"
                    if self.tier > 0 else
                    "I'm stuck, and there's no stronger local model to escalate to")
            stuck = (f"{lead}. Last error: {obs[:200].strip()}. "
                     "This needs a different approach — want me to try one, or take it yourself?")
            self.session.log("assistant", text=stuck)
            self.on_event("say", message=stuck)
            return tag, borrow_now, ("return", stuck)
        # deterministic recovery hint for the common bash failure signatures
        if kind == "bash":
            h = error_hint(obs)
            if h:
                tag += f"  ↳ {h}\n"
        return tag, borrow_now, None

    def _escalate(self, last_error, trace):
        """Climb one ladder rung (P5.7): swap to the stronger backend, re-resolve its aliases
        + per-model knobs, warm it, hand it the full context with a fresh-start instruction,
        and reset the per-turn stuck ledger. Precondition (checked by the caller): a stronger
        rung exists and the escalation lever is on."""
        if self._passport_on:   # P5.8: the escalation belongs to the rung being LEFT
            profile.record(self.backend.name, "escalate")
        self.tier += 1
        self.backend = self.ladder[self.tier]
        self._aliases = profiles.resolve(self.backend.name).get("aliases", ())
        self._resolve_knobs()   # P5.8: re-tune for the stronger rung
        self._prewarmed = False
        trace["escalated"] = True
        self.on_event("escalate", model=self.backend.name)
        self.session.log("escalate", model=self.backend.name)
        if hasattr(self.backend, "warm"):
            self.backend.warm()
        self.messages.append({"role": "user", "content":
            f"[The previous model kept failing. You are now a stronger model taking over the SAME task with full context above. Step back, re-diagnose from the real errors, and solve it. Last error: {last_error[:200].strip()}]"})
        # reset the per-turn stuck ledger for a fresh start on the new rung
        self.stuck["score"] = 0.0
        self.stuck["sig_fails"].clear()
        self.stuck["last_err_by_sig"].clear()
        self.stuck["malformed"] = 0
        self.clean_streak = 0

    def _handle_malformed(self, raw, prompt, step, trace):
        """The malformed-JSON strike path (deterministic salvage already failed). Counts
        the strike + passport/exemplar telemetry, bumps retry-heat, then resolves to ONE of:
          ("proceed", raw, act) — the 3rd strike borrowed one action from a stronger rung;
                                  execute it this step (no strike carried, no abort);
          ("continue", None, None) — appended a retry nudge (its own past valid action of the
                                     guessed kind when available); re-generate next step;
          ("abort", None, None) — the 5th strike: the model can't hold the format, end the turn.
        """
        self.stuck["malformed"] += 1
        self._bump_stuck("malformed", STUCK_MALFORMED_W)   # P5.7
        self._heat = min(0.7, self._heat + self.heat_bump)   # P5.5 / P5.8
        trace["malformed"] = True
        self.on_event("malformed")
        self.session.log("malformed", v=TRACE_V, step=step, raw=raw[:200])
        exemplars.record_malformed(self.backend.name)   # P5.6: keys the cold-start head-pin
        # P5.8 passport telemetry: a strike, + a trunc_write when the output was a truncated
        # write_file (the num_predict-budget failure) so the passport can raise the budget.
        if self._passport_on:
            profile.record(self.backend.name, "malformed")
            if exemplars.guess_kind(raw) == "write_file":
                profile.record(self.backend.name, "trunc_write")
        # P5.7: route the THIRD strike through _borrow (the failure mode most correlated with
        # model size, which the old loop aborted at 5 without ever consulting a stronger rung).
        braw = bact = None
        if self.stuck["malformed"] >= MALFORMED_BORROW_AT:
            braw, bact = self._borrow(prompt, step, trace)
        if bact is not None:
            return "proceed", braw, bact
        # P5.6: quote one of the model's OWN past valid actions of the guessed kind — a far
        # stronger anchor than the bare text nudge. Plain nudge when no exemplar exists yet.
        guessed = exemplars.guess_kind(raw)
        ex = exemplars.fetch(self.backend.name, guessed) if guessed else None
        if ex:
            nudge = (f"That was not valid action JSON. Here is a valid `{guessed}` action "
                     "you emitted earlier — reply with ONE JSON action object in exactly "
                     f"this shape:\n{ex}")
        else:
            nudge = "That was not valid action JSON. Reply with one JSON action object only."
        self.messages.append({"role": "user", "content": nudge})
        if self.stuck["malformed"] >= MALFORMED_ABORT_AT:
            return "abort", None, None
        return "continue", None, None

    def _log_step(self, trace, t0):
        """P3.1 flight recorder: emit EXACTLY ONE 'step' record per loop iteration (from the
        loop's finally, so any continue/return/raise still logs it). Guarded so a logging
        error can't mask a raising backend — the original exception still propagates."""
        try:
            trace["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
            used, window = self._fill()
            trace["used"], trace["window"] = used, window
            if self._compacted:
                trace["compacted"] = True
                self._compacted = False
            self.session.log("step", **trace)
        except Exception:
            pass

    def _begin_turn(self, user_text):
        """Turn setup: append the user message (+ the P4.5 JIT-retrieval note), reset the
        per-turn flags and the P5.7 stuck ledger, and decay one escalation rung at the turn
        boundary (unless sticky) so the next turn proposes-small again."""
        self.messages.append({"role": "user", "content": user_text})
        # P4.5 just-in-time retrieval: inject ONE deterministic "[retrieved context]" note
        # (matched files + symbols + test command) so the first pure-retrieval steps become
        # zero steps. Skipped when the prompt names nothing real; wrapped so it NEVER breaks a turn.
        if self._lv("workspace"):
            try:
                note = self._retrieval_note(user_text)
            except Exception:
                note = None
            if note:
                self.messages.append({"role": "user", "content": note})
        self._mutated = set()
        self._verified = False
        self._bounced = False
        self._narrated = False
        self._heat = 0.0   # P5.5: greedy for the first try of every turn
        self.session.log("user", text=user_text)
        self.session.set_status("working")
        # P5.7 unified per-turn stuck ledger — one weighted score (replacing the old disjoint
        # bad / fail_counts / total_fails) drives escalation at self.stuck_at.
        self.stuck = {"score": 0.0, "events": [], "last_err_by_sig": {},
                      "malformed": 0, "sig_fails": {}, "borrows": 0}
        self.clean_streak = 0
        self._prewarmed = False
        # P5.7 tier decay: relax one escalation rung at the turn boundary unless sticky.
        if not self.sticky_escalation:
            self._deescalate()

    def send(self, user_text):
        self._begin_turn(user_text)
        recent = []
        try:
            for step in range(1, self.max_steps + 1):
                if self.stop.is_set():
                    self.on_event("stopped")
                    return "(stopped)"
                # P3.1 flight recorder: build the step trace at the top of the iteration
                # and fill it as we go; the try/finally below fires EXACTLY ONE 'step'
                # record no matter which continue/return/raise exits the iteration.
                trace = {"v": TRACE_V, "step": step, "tier": self.tier}
                _t0 = time.monotonic()
                try:
                    raw, prompt, pin = self._prepare_and_generate(step)

                    try:
                        act = json.loads(raw)
                    except json.JSONDecodeError:
                        # P5.4: before counting a strike, try the deterministic salvage
                        # pass (strip fences / prose prefix / trailing comma). A recovered
                        # object proceeds as a normal parsed action and does NOT strike;
                        # only the genuinely-unsalvageable case (e.g. truncation) falls
                        # through to the malformed strike + text nudge + abort-at-5.
                        act, stage = self._salvage(raw)
                        if act is None:
                            control, raw, act = self._handle_malformed(raw, prompt, step, trace)
                            if control == "continue":
                                continue
                            if control == "abort":
                                return "(the model could not hold the action format)"
                            # control == "proceed": raw/act are a BORROWED action — fall through
                        else:
                            self.on_event("salvage", stage=stage)
                            self.session.log("salvage", v=TRACE_V, step=step, stage=stage)
                    self.stuck["malformed"] = 0
                    self.messages.append({"role": "assistant", "content": raw})
                    # P4.2 meta: tag the assistant echo. A write_file echo carries the
                    # ENTIRE file content verbatim → mark it for structural collapse.
                    if act.get("action") == "write_file" and act.get("path"):
                        self._tag_last({"kind": "write_echo", "action": "write_file",
                                        "path": os.path.realpath(os.path.join(self.session.cwd, act["path"])),
                                        "step": step})
                    else:
                        self._tag_last({"kind": "assistant", "action": act.get("action"), "step": step})

                    # plan update
                    if isinstance(act.get("plan"), list) and act["plan"]:
                        if act["plan"] != self.plan:
                            self.plan = act["plan"]
                            self.on_event("plan", plan=self.plan)
                            # P4.7: persist plan changes so a --resume can restore the
                            # living plan from the transcript (mirrors meta/compact).
                            self.session.log("plan", items=self.plan)

                    # P4.8 note update: pin a discovered fact alongside the plan. Any
                    # action may carry `note`; _add_note dedups + FIFO-caps it. Because
                    # the pin is rebuilt per-step from self.notes, the fact survives
                    # compaction verbatim (the lossy summarizer never touches it).
                    if isinstance(act.get("note"), str) and act["note"].strip():
                        if self._add_note(act["note"]):
                            self.session.log("note", items=list(self.notes))

                    self._coerce_str_fields(act)   # advisory engine may emit a non-string field
                    kind = act.get("action")
                    trace["action"] = kind
                    if kind == "say":
                        bounce = self._done_gate() or self._narration_bounce(act.get("message", ""))
                        if bounce:
                            self.messages.append({"role": "user", "content": bounce})
                            continue
                        msg = act.get("message", "")
                        self.session.log("assistant", text=msg, thought=act.get("thought", ""))
                        self.on_event("say", message=msg)
                        return msg

                    if self.allowed is not None and kind not in self.allowed:
                        self.messages.append({"role": "user", "content": f"'{kind}' not permitted. Allowed: {sorted(self.allowed)}."})
                        continue

                    blocked = self._gate(kind, act)
                    if blocked:
                        # P5.5: a manual-mode block is a user DECLINE (plan-mode blocks
                        # only fire when self.mode == "plan") — perturb so the model
                        # proposes a genuinely different action, not the same one again.
                        if self.mode == "manual":
                            self._heat = min(0.7, self._heat + self.heat_bump)
                        trace["gated"] = True
                        self.on_event("action", action=kind,
                                      detail=act.get("command") or act.get("path") or act.get("target") or "")
                        self.on_event("observation", text=blocked, ok=False)
                        self.session.log("action", action=kind, args={"gated": True}, thought=act.get("thought", ""))
                        self.messages.append({"role": "user", "content": f"⚠ {blocked}"})
                        continue

                    # P5.1 state-dependent grammar: small models drop required fields
                    # (bash without command, edit_file without old/new, a pathless
                    # read/write/edit — sometimes just misnamed as filename/file/…).
                    # First normalize path aliases (P3.2 alias_repair lever). Then, when
                    # constrained decoding is on, rather than burning a step on a text
                    # nudge, re-ask ONCE with ONLY this action's variant grammar-forced —
                    # so the resend is constrained to be complete. Only if the resend is
                    # ALSO incomplete (advisory engine) do we fall back to the text nudge.
                    self._alias_path(act)
                    if self._lv("schema"):
                        missing = self._missing_required(kind, act)
                        if missing:
                            act, kind, missing = self._resend_variant(act, kind, pin, step, missing, trace)
                        if missing:
                            obs = (f"'{kind}' is missing required field(s): "
                                   f"{', '.join('`' + m + '`' for m in missing)}. Re-send the SAME action "
                                   "as one complete JSON object with those fields included.")
                            trace["ok"] = False
                            self.on_event("action", action=kind, detail="(incomplete)")
                            self.on_event("observation", text=obs, ok=False)
                            self.session.log("action", action=kind, args={"invalid": "missing fields", "missing": missing},
                                             thought=act.get("thought", ""))
                            self.messages.append({"role": "user", "content": f"⚠ {obs}"})
                            continue
                    elif kind in ("read_file", "write_file", "edit_file") and not act.get("path"):
                        # bare mode (schema lever off): the original pathless-only nudge.
                        obs = (f"'{kind}' is missing its `path` field. Re-send the SAME action as one JSON object "
                               f'with the file path included, e.g. {{"action":"{kind}","path":"dir/file.go", ...}}.')
                        trace["ok"] = False
                        self.on_event("action", action=kind, detail="(no path)")
                        self.on_event("observation", text=obs, ok=False)
                        self.session.log("action", action=kind, args={"invalid": "missing path"}, thought=act.get("thought", ""))
                        self.messages.append({"role": "user", "content": f"⚠ {obs}"})
                        continue

                    if kind == "fleet_send":
                        self._do_fleet_send(act, trace)
                        continue

                    # include offset so paging one big file (same path, new range) isn't seen as a loop.
                    # P6.5: also fold the EDIT payload in — otherwise three DIFFERENT successful edits
                    # to one file share `edit_file:<path>:` and falsely trip the loop breaker.
                    _edit_disc = ""
                    if kind == "edit_file":
                        _edit_disc = (str(act.get("start_line") or "") + (act.get("old") or "")
                                      + json.dumps(act.get("edits") or "", sort_keys=True))[:120]
                    sig = f"{kind}:{act.get('command') or act.get('path') or act.get('pattern') or ''}:{act.get('offset', '')}:{_edit_disc}"
                    trace["sig"] = sig
                    recent.append(sig)
                    # P5.8: loop_threshold is per-model (2 for loop-prone models, 3 default).
                    if self._lv("loop_detect") and recent[-self.loop_threshold:].count(sig) >= self.loop_threshold:
                        self._bump_stuck("loop", STUCK_LOOP_W)   # P5.7
                        self._heat = min(0.7, self._heat + self.heat_bump)   # P5.5 / P5.8
                        trace["loop_trip"] = True
                        self.on_event("loop")
                        self.session.log("loop", v=TRACE_V, step=step, sig=sig, cause="repeat")
                        if self._passport_on:
                            profile.record(self.backend.name, "loop")   # P5.8 passport telemetry
                        self.messages.append({"role": "user", "content": f"You repeated the same action {self.loop_threshold}x with no progress. Do something different, or `say` if the task is already done."})
                        recent.clear()
                        continue

                    # P4.1 read cache: a repeat read of a file the ledger still holds
                    # is answered by the harness — unchanged files get a one-line note
                    # (no re-inject), changed files get a diff-since-last-read. A new file
                    # or a new line-range falls through to a real read below.
                    if self._lv("read_gate") and kind == "read_file" and act.get("path"):
                        fp = os.path.realpath(os.path.join(self.session.cwd, act["path"]))
                        if self._serve_cached_read(act, fp, step, trace):
                            continue

                    # read-before-edit: never edit or overwrite an EXISTING file whose
                    # CURRENT content the model doesn't hold — it must work from real
                    # content, not a guess or a stale copy (the exact failure mode that
                    # made a weak model hallucinate code). The ledger makes this honest:
                    # a bash mutation, a compaction that dropped the read, or an on-disk
                    # change since the read all re-arm the gate.
                    if self._lv("read_gate") and kind in ("edit_file", "write_file"):
                        fp = os.path.realpath(os.path.join(self.session.cwd, act["path"]))
                        if os.path.isfile(fp) and not self.ledger.current(fp):
                            obs = self._read_gate_msg(kind, act, fp)
                            trace["ok"] = False
                            self.on_event("action", action=kind, detail=act.get("path", ""))
                            self.on_event("observation", text=obs, ok=False)
                            self.session.log("action", action=kind, args={"blocked": "read-before-edit", "path": act.get("path", "")},
                                             thought=act.get("thought", ""))
                            self.messages.append({"role": "user", "content": f"⚠ {obs}"})
                            continue
                        # partial-read guard: editing a region the model never read
                        # still requires a region read (a 10-line read is not the file).
                        # Covers BOTH dialects: an {old,new} snippet located outside the
                        # read spans, and a P5.3 anchored [start,end] range that overruns them.
                        if kind == "edit_file" and (act.get("old") or act.get("start_line") is not None) \
                                and not self._edit_region_seen(act, fp):
                            obs = (f"Blocked: you only read PART of {act.get('path')} — the range you're editing "
                                   "is outside the lines you read. Read that region first (use offset/limit), "
                                   "then edit.")
                            trace["ok"] = False
                            self.on_event("action", action=kind, detail=act.get("path", ""))
                            self.on_event("observation", text=obs, ok=False)
                            self.session.log("action", action=kind, args={"blocked": "read-region", "path": act.get("path", "")},
                                             thought=act.get("thought", ""))
                            self.messages.append({"role": "user", "content": f"⚠ {obs}"})
                            continue

                    # P5.2 deterministic dry-run verifier + best-of-N resample. Runs
                    # AFTER the read-before-edit gate so an edit_file probe reads the
                    # SAME realpath the gate just cleared (never an unread file — that
                    # would leak content-existence the gate blocks). Score the parsed
                    # action for free: if the greedy sample is a certain miss (score 0
                    # — `old` absent, .py that won't compile, a command not found),
                    # don't spend a failure observation running it — resample the SAME
                    # prompt at rising temperature and execute the best candidate. All
                    # miss → the greedy original still runs (the teaching failure), so
                    # this is a strict superset of the pre-P5.2 loop. Gated with the
                    # `schema` lever (the constrained-decoding bundle): schema off →
                    # bare baseline, byte-for-byte unchanged, no extra model calls.
                    # Skipped under zero-inference replay (backend.replay): there is
                    # no model to re-ask, and dry_run is workspace-dependent —
                    # resampling would pop later steps' raws off the P3.3 cursor as
                    # bogus candidates and break replay's 1:1-with-the-recording
                    # contract.
                    if self._lv("schema") and not getattr(self.backend, "replay", False):
                        score, _reason = dry_run(act, self.session.cwd)
                        if score == 0.0:
                            act, kind = self._resample(act, kind, pin, step, score, trace)

                    obs, ok, budget, recorded_fp = self._execute_and_record(kind, act, raw, step, trace)

                    tag = ""
                    borrow_now = False
                    if not ok:
                        tag, borrow_now, control = self._handle_failure(kind, sig, obs, step, trace)
                        if control == "continue":       # escalated to a fresh rung — skip the obs append
                            continue
                        if isinstance(control, tuple):  # ("return", stuck_msg): no rung left, end the turn
                            return control[1]
                    obs_msg = {"role": "user", "content": f"{tag}Observation:\n{shape(obs, budget)}"}
                    self.messages.append(obs_msg)
                    # P4.2 meta: tag this observation (kind/action/path/step/ok) so the
                    # structural pass can stub superseded reads and shrink stale failures.
                    self._tag_last({"kind": "obs", "action": kind, "path": recorded_fp, "step": step, "ok": ok})
                    # P5.7: the sig-fail borrow fires HERE (after the failed observation is
                    # in context) so the stronger rung sees the actual failure it's asked
                    # to route around; the borrowed action rides the next cheap step.
                    if borrow_now:
                        self._borrow(self.messages + ([pin] if pin else []), step, trace)
                    # P4.1: bind this file's observation to its message so a later
                    # compaction can detect (by identity) when it leaves context.
                    if recorded_fp:
                        self.ledger.set_obs_msg(recorded_fp, obs_msg)
                finally:
                    self._log_step(trace, _t0)

            return "(hit the step limit — ask me to continue)"
        finally:
            self.session.set_status("idle")
