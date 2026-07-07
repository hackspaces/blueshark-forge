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
from .ledger import Ledger
from .tools import ACTION_SCHEMA, TOOL_HELP, execute, shape, error_hint

STUCK_AT = int(os.environ.get("FORGE_STUCK_THRESHOLD", "7"))  # failures before escalating a rung
TRACE_V = 1  # P3.1 schema version stamped on every meta/step/compact/loop/malformed record

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
        base = (system if system is not None else SYSTEM) + (AUTONOMOUS if autonomous else "")
        self.messages = [{"role": "system", "content": base}]
        if workspace and self._lv("workspace"):
            self.messages.append({"role": "user", "content": workspace})
            self.messages.append({"role": "assistant", "content": '{"thought":"Oriented in the workspace. Ready.","action":"say","message":"Ready."}'})
        self.head_len = len(self.messages)  # system (+ workspace) — never compacted away
        self.plan = []
        self.notes = []   # P4.8 pinned scratch facts (durable, survive compaction — see _pin_state); seed #0 set below
        # P4.1 file-state ledger — the harness-owned model of what's been read and
        # still held in context (replaces the bare read_files set). Backs the honest
        # read-before-edit gate, served-from-cache reads, and compaction eviction.
        self.ledger = Ledger()
        self._mutated = set()    # P2.1 done-gate: paths mutated THIS turn
        self._verified = False   # a test run this turn already passed
        self._bounced = False    # the done-gate already bounced/nudged once this turn
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
        if st == "changed" and e is not None:
            return (f"Blocked: {path} CHANGED on disk since you read it at step {e.read_step} — your copy "
                    "is stale. read_file it again and work from the current content before editing.")
        if st == "evicted" and e is not None:
            return (f"Blocked: you read {path} at step {e.read_step}, but it's no longer in your context — "
                    "read_file it again before editing or overwriting it.")
        return (f"Blocked: read {path} before editing or overwriting it — "
                "work from its actual current content, not memory. Use read_file first.")

    def _edit_region_seen(self, act, fp):
        """True unless the edit targets lines OUTSIDE the (partial) range the model
        actually read. A whole-file read, or an `old` snippet we cannot locate,
        never blocks — this only catches editing a region a ranged read never saw."""
        e = self.ledger.get(fp)
        if e is None or e.whole or not e.spans:
            return True
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

    def _generate(self, prompt):
        """Stream the model's action. When it turns out to be a `say`, emit the
        message text live (token by token) via on_event('token')."""
        schema = ACTION_SCHEMA if self._lv("schema") else None
        if not hasattr(self.backend, "stream"):
            return self.backend.chat(prompt, schema=schema)
        raw, emitted, is_say = "", 0, False
        try:
            for chunk in self.backend.stream(prompt, schema=schema):
                if self.stop.is_set():
                    break
                raw += chunk
                if not is_say and '"say"' in raw:
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

    def send(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        # P4.5 just-in-time retrieval: turn-start, inject ONE deterministic
        # "[retrieved context]" note (matched files + symbols + the test command)
        # so the first pure-retrieval steps become zero steps. Skipped when the
        # prompt names nothing real; wrapped so retrieval NEVER breaks a turn.
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
        self.session.log("user", text=user_text)
        self.session.set_status("working")
        bad = 0
        recent = []
        fail_counts = {}
        total_fails = 0
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
                    self._absorb_inbox()
                    # P4.1: re-stat tracked files (bounded) so any bash/redirect/edit
                    # that changed a file's mtime since it was read is caught before
                    # the gate and the read cache consult the ledger this step.
                    self.ledger.refresh()
                    if self._lv("compaction"):
                        self._structural_compact(step)   # P4.2: deterministic pass first (zero model calls)
                        self._compact()                  # then the LLM summarizer at 70% (emergency in-turn gate)
                        self._floor()                    # then the hard floor: escape a full-window wedge
                    self._audit_prefix(step)             # P4.3: FORGE_DEBUG prefix-mutation audit (no-op by default)
                    pin = self._pin_state() if self._lv("plan_pin") else None
                    prompt = self.messages + ([pin] if pin else [])
                    self.on_event("thinking")
                    raw = self._generate(prompt)
                    # P4.2: the real prompt-token count now reflects any compaction —
                    # stop overriding _fill with the char estimate.
                    self._reclaimed = False
                    # P4.3: rebase the token ledger against the observed prompt_eval_count
                    # for the EXACT prompt just sent (messages + pin) — an uncached call
                    # rebases tok_ratio; a warm-cache one is an up-only cross-check.
                    self._recalibrate(prompt)
                    # P3.3 flight recorder: persist the RAW model output of every
                    # step — including the malformed ones the parse below discards
                    # (the valuable ones). prompt_tokens is the exact count for THIS
                    # generation, so a replay can reproduce compaction timing.
                    self.session.log("model", v=TRACE_V, raw=raw, tier=self.tier,
                                     prompt_tokens=getattr(self.backend, "last_prompt_tokens", 0))

                    try:
                        act = json.loads(raw)
                    except json.JSONDecodeError:
                        bad += 1
                        trace["malformed"] = True
                        self.on_event("malformed")
                        self.session.log("malformed", v=TRACE_V, step=step, raw=raw[:200])
                        self.messages.append({"role": "user", "content": "That was not valid action JSON. Reply with one JSON action object only."})
                        if bad >= 5:
                            return "(the model could not hold the action format)"
                        continue
                    bad = 0
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

                    kind = act.get("action")
                    trace["action"] = kind
                    if kind == "say":
                        bounce = self._done_gate()
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
                        trace["gated"] = True
                        self.on_event("action", action=kind,
                                      detail=act.get("command") or act.get("path") or act.get("target") or "")
                        self.on_event("observation", text=blocked, ok=False)
                        self.session.log("action", action=kind, args={"gated": True}, thought=act.get("thought", ""))
                        self.messages.append({"role": "user", "content": f"⚠ {blocked}"})
                        continue

                    # small models sometimes name the path field differently, or drop it —
                    # normalize aliases, and reject pathless file actions with an exact fix
                    # (an empty path must never fall through: it used to resolve to the cwd
                    # and hit the read-before-edit guard with a nonsense message).
                    if self._lv("alias_repair") and kind in ("read_file", "write_file", "edit_file") and not act.get("path"):
                        for alias in ("filename", "file", "filepath", "file_path", "name"):
                            if isinstance(act.get(alias), str) and act[alias]:
                                act["path"] = act[alias]
                                break
                    if kind in ("read_file", "write_file", "edit_file") and not act.get("path"):
                        obs = (f"'{kind}' is missing its `path` field. Re-send the SAME action as one JSON object "
                               f'with the file path included, e.g. {{"action":"{kind}","path":"dir/file.go", ...}}.')
                        trace["ok"] = False
                        self.on_event("action", action=kind, detail="(no path)")
                        self.on_event("observation", text=obs, ok=False)
                        self.session.log("action", action=kind, args={"invalid": "missing path"}, thought=act.get("thought", ""))
                        self.messages.append({"role": "user", "content": f"⚠ {obs}"})
                        continue

                    if kind == "fleet_send":
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
                        continue

                    # include offset so paging one big file (same path, new range) isn't seen as a loop
                    sig = f"{kind}:{act.get('command') or act.get('path') or act.get('pattern') or ''}:{act.get('offset', '')}"
                    trace["sig"] = sig
                    recent.append(sig)
                    if self._lv("loop_detect") and recent[-3:].count(sig) >= 3:
                        trace["loop_trip"] = True
                        self.on_event("loop")
                        self.session.log("loop", v=TRACE_V, step=step, sig=sig, cause="repeat")
                        self.messages.append({"role": "user", "content": "You repeated the same action 3x with no progress. Do something different, or `say` if the task is already done."})
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
                        if kind == "edit_file" and act.get("old") and not self._edit_region_seen(act, fp):
                            obs = (f"Blocked: you only read PART of {act.get('path')} — the text you're editing "
                                   "is outside the lines you read. Read that region first (use offset/limit), "
                                   "then edit.")
                            trace["ok"] = False
                            self.on_event("action", action=kind, detail=act.get("path", ""))
                            self.on_event("observation", text=obs, ok=False)
                            self.session.log("action", action=kind, args={"blocked": "read-region", "path": act.get("path", "")},
                                             thought=act.get("thought", ""))
                            self.messages.append({"role": "user", "content": f"⚠ {obs}"})
                            continue

                    # capture the before-content so we can show a real diff after a write
                    before = ""
                    if kind == "write_file":
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
                    budget = self._obs_budget()
                    # P4.1 ledger population — a ranged read records only its span; a
                    # write/edit ingests the new content as the cached (diffable) version.
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
                            self.ledger.record_write(_rp, step)  # re-read the edited file from disk
                            recorded_fp = _rp
                    if ok and kind in ("write_file", "edit_file") and act.get("path"):
                        self._mutated.add(os.path.realpath(os.path.join(self.session.cwd, act["path"])))
                        self._verified = False
                    if ok and kind == "bash":
                        _cmd = act.get("command", "")
                        from . import fleet as _fleet
                        if _is_test_cmd(_cmd, self.session.cwd):
                            self._verified = True         # the model ran the suite itself
                        elif _fleet.bash_mutates(_cmd):
                            self._mutated.add("<bash>")   # a file-touching bash still gates `say`
                            self._verified = False
                    if ok and kind == "edit_file":
                        self.on_event("diff", path=act.get("path", ""), old=act.get("old", ""), new=act.get("new", ""))
                    elif ok and kind == "write_file":
                        self.on_event("diff", path=act.get("path", ""), old=before, new=act.get("content", ""))
                    self.session.log("observation", text=shape(obs, budget), ok=ok)
                    self.on_event("observation", text=obs, ok=ok)

                    tag = ""
                    if not ok:
                        fail_counts[sig] = fail_counts.get(sig, 0) + 1
                        total_fails += 1
                        tag = "  ⚠ this action FAILED — diagnose the cause before retrying.\n"
                        # per-command repeat (survives interleaved successes, unlike a consecutive counter)
                        if self._lv("loop_detect") and fail_counts[sig] >= 3:
                            trace["loop_trip"] = True
                            self.on_event("loop")
                            self.session.log("loop", v=TRACE_V, step=step, sig=sig, cause="fail", count=fail_counts[sig])
                            tag = (f"  ⚠ `{sig}` has now failed {fail_counts[sig]} times. STOP retrying this exact thing. "
                                   "Change approach entirely: re-read the real file/error, rewrite with write_file instead of edit_file, "
                                   "or `say` to tell the user you're stuck and exactly what failed.\n")
                        # stuck: escalate to a stronger LOCAL model (same task, same
                        # context) rather than grinding or giving up. All still local.
                        if total_fails >= STUCK_AT:
                            if self._lv("escalation") and self.tier < len(self.ladder) - 1:
                                self.tier += 1
                                self.backend = self.ladder[self.tier]
                                trace["escalated"] = True
                                self.on_event("escalate", model=self.backend.name)
                                self.session.log("escalate", model=self.backend.name)
                                if hasattr(self.backend, "warm"):
                                    self.backend.warm()
                                self.messages.append({"role": "user", "content":
                                    f"[The previous model kept failing. You are now a stronger model taking over the SAME task with full context above. Step back, re-diagnose from the real errors, and solve it. Last error: {obs[:200].strip()}]"})
                                fail_counts.clear()
                                total_fails = 0
                                continue
                            stuck = (f"I'm stuck even after escalating through the local models. Last error: {obs[:200].strip()}. "
                                     "This needs a different approach — want me to try one, or take it yourself?")
                            self.session.log("assistant", text=stuck)
                            self.on_event("say", message=stuck)
                            return stuck
                        # deterministic recovery hint for the common bash failure signatures
                        if kind == "bash":
                            h = error_hint(obs)
                            if h:
                                tag += f"  ↳ {h}\n"
                    obs_msg = {"role": "user", "content": f"{tag}Observation:\n{shape(obs, budget)}"}
                    self.messages.append(obs_msg)
                    # P4.2 meta: tag this observation (kind/action/path/step/ok) so the
                    # structural pass can stub superseded reads and shrink stale failures.
                    self._tag_last({"kind": "obs", "action": kind, "path": recorded_fp, "step": step, "ok": ok})
                    # P4.1: bind this file's observation to its message so a later
                    # compaction can detect (by identity) when it leaves context.
                    if recorded_fp:
                        self.ledger.set_obs_msg(recorded_fp, obs_msg)
                finally:
                    # ONE step record per iteration. Guarded so a raising backend is
                    # surfaced (the original exception propagates) — not masked by a
                    # logging error — while the trace still lands on the happy path.
                    try:
                        trace["elapsed_ms"] = int((time.monotonic() - _t0) * 1000)
                        used, window = self._fill()
                        trace["used"], trace["window"] = used, window
                        if self._compacted:
                            trace["compacted"] = True
                            self._compacted = False
                        self.session.log("step", **trace)
                    except Exception:
                        pass

            return "(hit the step limit — ask me to continue)"
        finally:
            self.session.set_status("idle")
