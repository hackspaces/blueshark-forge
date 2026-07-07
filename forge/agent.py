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
from .tools import ACTION_SCHEMA, TOOL_HELP, execute, shape, error_hint

STUCK_AT = int(os.environ.get("FORGE_STUCK_THRESHOLD", "7"))  # failures before escalating a rung
TRACE_V = 1  # P3.1 schema version stamped on every meta/step/compact/loop/malformed record

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

SUMMARIZE_SYSTEM = (
    "You compress an AI coding agent's work-in-progress into a dense STATE note it will "
    "read to continue. Capture: the task/goal, what has been done, key findings, files "
    "read or changed (with exact paths), decisions made, errors hit, and the current state "
    "and next step. Preserve concrete details — paths, names, commands, values. No preamble, "
    "no fluff. Just the state, tightly written."
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
        self.read_files = set()  # abs paths read this session — enforces read-before-edit
        self._mutated = set()    # P2.1 done-gate: paths mutated THIS turn
        self._verified = False   # a test run this turn already passed
        self._bounced = False    # the done-gate already bounced/nudged once this turn
        self.stop = threading.Event()  # set from the UI (Esc) to interrupt mid-run
        self.mode = "auto"             # auto | plan | manual (set by the UI)
        self.approve = lambda desc: "yes"   # manual-mode hook: 'yes' | 'always' | 'no'
        self._compacted = False        # P3.1: transient — set by _compact, read+cleared by the step trace
        from . import config as _cfg
        self.approvals = set(_cfg.get("approvals") or [])   # 'always'-approved action keys
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
        """(tokens_used, window) for the current model. Uses the EXACT prompt-token
        count from the last response and the model's REAL context window when the
        backend reports them; falls back to a char estimate before the first call."""
        window = self.backend.effective_ctx() if hasattr(self.backend, "effective_ctx") else backends.ctx_cap()
        used = getattr(self.backend, "last_prompt_tokens", 0)
        if not used:  # no real count yet (first turn) — estimate from chars
            used = sum(len(m["content"]) for m in self.messages) // 4
        return used, window

    def _obs_budget(self):
        """One char budget for a single observation, derived from the model's REAL
        window: ~8% of it (4 chars/token), hard-capped at 12000. This ends the old
        4000/12000 split-brain — one budget, used for both the transcript log and
        the message fed back to the model, so nothing is ever cut mid-content
        without a visible marker and no pointer outlives its budget."""
        window = self.backend.effective_ctx() if hasattr(self.backend, "effective_ctx") else backends.ctx_cap()
        return min(12000, int(window * 4 * 0.08))

    def _compact(self):
        """At ~70% of the model's real window, SUMMARIZE the older middle turns
        into a dense state note (instead of dropping them). System + workspace
        stay pinned, recent turns stay verbatim, the plan is pinned separately —
        nothing important is lost, the context just gets denser."""
        used, window = self._fill()
        if used < 0.70 * window:
            return
        head = self.messages[:self.head_len]
        tail = self.messages[-12:]          # keep plenty of recent context so reads aren't lost → no re-read loop
        middle = self.messages[self.head_len:-12]
        if len(middle) < 4:
            return
        self.on_event("compacting", used=used, window=window)
        summary = self._summarize(middle)
        note = {"role": "user", "content": "[Earlier progress, summarized to save context:]\n" + summary}
        self.messages = head + [note] + tail
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

    def _pin_plan(self):
        if self.plan:
            return {"role": "user", "content": "[current plan]\n" + "\n".join(self.plan)}
        return None

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

    def send(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
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
                    if self._lv("compaction"):
                        self._compact()
                    pin = self._pin_plan() if self._lv("plan_pin") else None
                    prompt = self.messages + ([pin] if pin else [])
                    self.on_event("thinking")
                    raw = self._generate(prompt)
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

                    # plan update
                    if isinstance(act.get("plan"), list) and act["plan"]:
                        if act["plan"] != self.plan:
                            self.plan = act["plan"]
                            self.on_event("plan", plan=self.plan)

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

                    # read-before-edit: never edit or overwrite an EXISTING file the model
                    # hasn't actually read — this forces it to work from real content, not a
                    # guess (the exact failure mode that made a weak model hallucinate code).
                    if self._lv("read_gate") and kind in ("edit_file", "write_file"):
                        fp = os.path.realpath(os.path.join(self.session.cwd, act["path"]))
                        if os.path.isfile(fp) and fp not in self.read_files:
                            obs = (f"Blocked: read {act.get('path')} before editing or overwriting it — "
                                   "work from its actual current content, not memory. Use read_file first.")
                            trace["ok"] = False
                            self.on_event("action", action=kind, detail=act.get("path", ""))
                            self.on_event("observation", text=obs, ok=False)
                            self.session.log("action", action=kind, args={"blocked": "read-before-edit", "path": act.get("path", "")},
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
                    if ok and kind in ("read_file", "write_file") and act.get("path"):
                        self.read_files.add(os.path.realpath(os.path.join(self.session.cwd, act["path"])))
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
                    self.messages.append({"role": "user", "content": f"{tag}Observation:\n{shape(obs, budget)}"})
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
