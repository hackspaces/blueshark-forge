"""The interactive terminal — made to feel alive.

A live plan panel, clean tool-step rendering with pass/fail and timing, a spinner
while the model thinks, and the agent's reply set apart. Designed so watching a
small local model work is legible and fast to read."""
import itertools
import os
import sys
import threading
import time

import subprocess

from .agent import Agent
from .backends import make_backend, ForgeError
from . import config as cfgmod
from .util import slurp

DIM = "\033[2m"; B = "\033[1m"; CY = "\033[36m"; GR = "\033[32m"; YE = "\033[33m"; RD = "\033[31m"; MG = "\033[35m"; RST = "\033[0m"

ICON = {"bash": "⚡", "read_file": "▸", "write_file": "✎", "edit_file": "✎", "list_files": "▸",
        "grep": "⌕", "glob": "⌕", "fleet_send": "✉", "say": "▪"}


class Spinner:
    """Animated spinner that shows elapsed time (and optional live suffix)."""
    def __init__(self, label="thinking"):
        self.label = label; self._stop = False; self._t = None; self._suffix = ""
        self._start = None
    def suffix(self, s):
        self._suffix = s
    def __enter__(self):
        self._start = time.monotonic()
        def spin():
            for c in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if self._stop: break
                el = time.monotonic() - self._start
                t = f"{el:.0f}s" if el < 60 else f"{int(el//60)}m {int(el%60)}s"
                extra = f" · {self._suffix}" if self._suffix else ""
                sys.stdout.write(f"\r{MG}{c}{RST} {DIM}{self.label}… ({t}{extra}){RST}\033[K")
                sys.stdout.flush(); time.sleep(0.09)
            sys.stdout.write("\r\033[K"); sys.stdout.flush()
        self._t = threading.Thread(target=spin, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop = True
        if self._t: self._t.join()






class UI:
    """Renders agent events into the scrolling transcript via `emit`. Reply text
    is word-wrapped with a 2-space margin — never split mid-word — whether it
    streams token by token or arrives whole. The spinner lives in the pinned
    footer (owned by run()), not here."""
    INDENT = "  "

    def __init__(self, emit, verbose=False, width=lambda: 100):
        self.emit = emit; self.verbose = verbose; self.width = width; self.t0 = None
        self.streamed = False; self.said = False
        self._col = 0; self._pend = ""
    def new_turn(self):
        self.streamed = False; self.said = False
        self._col = 0; self._pend = ""
    def _line(self, s=""):
        self.emit(s + "\n")
    def _wrap_width(self):
        return max(24, min(self.width() - 2, 100))

    def wrap_block(self, text):
        """Wrap a whole reply: word boundaries, 2-space margin, paragraphs kept."""
        import textwrap
        out = []
        for para in text.split("\n"):
            if not para.strip():
                out.append("")
                continue
            out.append(textwrap.fill(para, width=self._wrap_width(),
                                     initial_indent=self.INDENT, subsequent_indent=self.INDENT,
                                     break_long_words=False, break_on_hyphens=False))
        return "\n".join(out)

    def _stream(self, chunk):
        """Emit streamed text with live word-wrapping: hold back a partial word
        until its end arrives, break lines before a word that would overflow."""
        import re as _re
        W = self._wrap_width()
        for piece in _re.split(r"(\s+)", chunk):
            if not piece:
                continue
            if not piece.isspace():
                self._pend += piece
                continue
            self._flush_word()
            if "\n" in piece:
                self.emit("\n" * min(piece.count("\n"), 2) + self.INDENT)
                self._col = len(self.INDENT)
            elif self._col >= W:
                self.emit("\n" + self.INDENT); self._col = len(self.INDENT)
            else:
                self.emit(" "); self._col += 1

    def _flush_word(self):
        if not self._pend:
            return
        if self._col + len(self._pend) > self._wrap_width() and self._col > len(self.INDENT):
            self.emit("\n" + self.INDENT); self._col = len(self.INDENT)
        self.emit(self._pend); self._col += len(self._pend)
        self._pend = ""

    def __call__(self, kind, **k):
        if kind == "token":
            if not self.streamed:
                self.emit("\n" + self.INDENT); self._col = len(self.INDENT); self.streamed = True
            self._stream(k["text"])
        elif kind == "say":
            self.said = True
            self._flush_word()
            self.emit("\n" if self.streamed else f"\n{self.wrap_block(k.get('message',''))}\n")
        elif kind == "plan":
            self._render_plan(k["plan"])
        elif kind == "action":
            ic = ICON.get(k["action"], "·"); detail = (k.get("detail") or "").replace("\n", " ")[:74]
            self.emit(f"  {CY}{ic}{RST} {k['action']} {DIM}{detail}{RST}")
            self.t0 = time.time()
        elif kind == "observation":
            dt = f"{time.time()-self.t0:.1f}s" if self.t0 else ""
            mark = f"{GR}ok{RST}" if k.get("ok") else f"{RD}fail{RST}"
            self._line(f"  {DIM}{dt}{RST} {mark}")
            if self.verbose:
                first = (k.get("text") or "").strip().splitlines()[:1]
                if first: self._line(f"    {DIM}→ {first[0][:80]}{RST}")
            self.t0 = None
        elif kind == "diff":
            self._render_diff(k.get("path", ""), k.get("old", ""), k.get("new", ""))
        elif kind == "escalate":
            self._line(f"  {MG}↑ stuck — escalating to a stronger local model: {k['model']}{RST}")
        elif kind == "borrow":
            self._line(f"  {MG}⇡ borrowing one action from {k['model']}{RST}")
        elif kind == "deescalate":
            self._line(f"  {DIM}↓ recovered — back to {k['model']}{RST}")
        elif kind == "inbox":
            self._line(f"  {YE}✉ {k['sender']}: {k['text'][:76]}{RST}")
        elif kind == "compact":
            self._line(f"  {DIM}⟲ context compacted{RST}")
        elif kind == "done_check":
            if k.get("ok"):
                self._line(f"  {GR}✓ done-gate: {k.get('cmd','')} passed{RST}")
            else:
                self._line(f"  {YE}▸ done-gate: {k.get('cmd','')} failed — sent back to fix{RST}")
        elif kind in ("malformed", "loop"):
            self._line(f"  {DIM}· ({kind}, recovering){RST}")

    def _render_plan(self, plan):
        self._line(f"{DIM}  ┌─ plan{RST}")
        for item in plan:
            s = item.strip()
            if s.startswith("[x]"):
                self._line(f"{DIM}  │{RST} {GR}✓{RST} {DIM}{s[3:].strip()}{RST}")
            elif s.startswith("[~]"):
                self._line(f"{DIM}  │{RST} {YE}▸{RST} {B}{s[3:].strip()}{RST}")
            else:
                self._line(f"{DIM}  │  {s.lstrip('[ ]').strip()}{RST}")
        self._line(f"{DIM}  └─{RST}")

    def _render_diff(self, path, old, new, max_lines=40):
        import difflib
        old_l, new_l = old.splitlines(), new.splitlines()
        sm = difflib.SequenceMatcher(a=old_l, b=new_l)
        rows = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            rows += [(RD, "-", ln) for ln in old_l[i1:i2]]
            rows += [(GR, "+", ln) for ln in new_l[j1:j2]]
        add = sum(1 for c, s, _ in rows if s == "+"); rem = len(rows) - add
        self._line(f"{DIM}  ┌ {path}{RST}  {GR}+{add}{RST} {RD}-{rem}{RST}")
        for color, sign, line in rows[:max_lines]:
            self._line(f"{DIM}  │{RST} {color}{sign} {line[:160]}{RST}")
        if len(rows) > max_lines:
            self._line(f"{DIM}  │ … {len(rows) - max_lines} more lines{RST}")
        self._line(f"{DIM}  └{RST}")


def _banner(models, ctx, cwd, ptype):
    """A welcome banner with a small forge emblem, shown at the top on start."""
    from . import __version__
    logo = [f"{MG}▟██████▙{RST}", f"{MG} ▜████▛ {RST}", f"{MG}  ▀██▀  {RST}"]
    info = [
        f"{B}{MG}forge{RST} {DIM}v{__version__}{RST}",
        f"{models}{DIM}{ctx}{RST}",
        f"{DIM}{cwd}{ptype}{RST}",
    ]
    lines = ["", *(f"  {logo[i]}   {info[i]}" for i in range(3)), ""]
    return "\n".join(lines) + "\n"


def _ollama_models():
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, timeout=10)
        return [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]
    except (subprocess.SubprocessError, OSError):
        return []


def _menu_model(agent, screen, history):
    """/model — pick a new ladder from installed models. Persists to config."""
    models = _ollama_models()
    cur = " → ".join(b.name.split(":", 1)[-1] for b in agent.ladder)
    screen.emit(f"{DIM}  current ladder: {cur}{RST}\n")
    if not models:
        screen.emit(f"{DIM}  (no ollama models found){RST}\n"); return
    for i, m in enumerate(models, 1):
        screen.emit(f"    {CY}{i}{RST} {m}\n")
    screen.emit(f"{DIM}  type numbers cheap→strong (e.g. '1 3'), a name, or blank to cancel{RST}\n")
    pick = screen.prompt(f"{GR}model›{RST} ", history, "")
    if not pick or not pick.strip():
        screen.emit(f"{DIM}  cancelled{RST}\n"); return
    chosen = [models[int(t) - 1] if t.isdigit() and 1 <= int(t) <= len(models) else t for t in pick.split()]
    if not chosen:
        return
    cfg = cfgmod.load()          # route through the CONFIGURED engine, not always ollama
    agent.set_ladder([make_backend(m, engine=cfg.get("engine", "ollama"),
                                   base_url=cfg.get("base_url") or None,
                                   api_key=cfg.get("api_key") or None) for m in chosen])
    cfgmod.set_key("ladder", chosen)
    screen.emit(f"{GR}  ✓ ladder → {' → '.join(chosen)}{RST}  {DIM}(saved){RST}\n")


def _menu_config(agent, screen):
    """/config — show current settings."""
    cfg = cfgmod.load()
    screen.emit(f"{DIM}  config ({cfgmod.PATH}):{RST}\n")
    for k in ("engine", "ladder", "num_ctx", "keep_alive", "num_predict", "stuck_threshold"):
        v = cfg.get(k)
        screen.emit(f"    {CY}{k}{RST}: {v if not isinstance(v, list) else ' → '.join(v)}\n")
    m = cfg.get("machine", {})
    if m:
        screen.emit(f"    {DIM}machine: {m.get('chip','')} · {m.get('ram_gb','?')}GB · {m.get('cores','?')} cores{RST}\n")


def _expand_ats(text, cwd):
    """Expand @path tokens into inline file contents, so `read this @foo.js` just works."""
    import os
    import re
    out = text
    for tok in re.findall(r"@([\w./\-]+)", text):
        p = os.path.join(cwd, tok)
        if os.path.isfile(p):
            try:
                body = slurp(p)[:8000]
                out += f"\n\n[contents of {tok}]\n{body}"
            except OSError:
                pass
    return out


# ---- persistent prompt history (P4.7) ----------------------------------------
HISTORY_PATH = os.path.expanduser("~/.forge/history")
HISTORY_CAP = 1000          # keep the last N prompts across sessions


def _load_history():
    """The persisted prompt history (last HISTORY_CAP lines), newest last. Rewrites
    the file trimmed when it has grown past the cap. Best-effort — any error yields
    an empty history rather than blocking startup."""
    try:
        with open(HISTORY_PATH, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    except OSError:
        return []
    if len(lines) > HISTORY_CAP:
        lines = lines[-HISTORY_CAP:]
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass
    return lines


def _append_history(line):
    """Append one prompt to the on-disk history. Newlines are flattened so one
    prompt stays one line. Best-effort and never raises."""
    if not line or not line.strip():
        return
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(line.replace("\n", " ") + "\n")
    except OSError:
        pass


MODES = ("auto", "plan", "manual")
MODE_HINT = {
    "auto":   "acts freely, no questions",
    "plan":   "read-only — investigates and presents a plan",
    "manual": "asks before every mutating action (y / a=always / n)",
}

# ---- wake-on-inbox (P2.3) ----------------------------------------------------
import re as _re

AUTO_ACT_SENDERS = {"verifier", "guard", "learn"}   # the daemon's system senders
_WAKE_ACT_RE = _re.compile(r"^\[(verify|task |ask )")
WAKE_BUDGET = 2   # max autonomous turns per human-free message chain (anti-livelock)


def wake_should_act(mode, wake_cfg, sender, text, budget_left):
    """PURE policy: should an idle-wake fleet message auto-start an agent turn?

    Only in auto mode, only when config wake=='act', only for the daemon's system
    senders or messages tagged [verify]/[task ]/[ask ], and only while the
    per-chain autonomous-turn budget is not spent. manual/plan modes never
    auto-act (they only render), so a red verify loop cannot spin a session."""
    if mode != "auto" or wake_cfg != "act" or budget_left <= 0:
        return False
    return sender in AUTO_ACT_SENDERS or bool(_WAKE_ACT_RE.match(text or ""))


def _on_wake(ui, agent, session, budget_left):
    """A wake byte fired while idle: drain the inbox and render every message.
    Returns a synthetic user_text to auto-start ONE turn (the actionable messages,
    tagged as fleet messages), or None to render-only — in which case the messages
    are folded into the agent's context so the next turn still sees them."""
    from . import config as _cfg
    wake_cfg = _cfg.get("wake", "off")
    act = []
    for m in session.drain():
        sender, text = m["from"], m["text"]
        ui("inbox", sender=sender, text=text)               # render the moment it lands
        if wake_should_act(agent.mode, wake_cfg, sender, text, budget_left):
            act.append(m)
        else:                                               # keep it in context (as _absorb_inbox would)
            tag = "[user (mid-run — steer accordingly)]" if sender == "user" \
                else f"[fleet message from {sender}]"
            agent.messages.append({"role": "user", "content": f"{tag}: {text}"})
            session.log("inbox", sender=sender, text=text)
    if not act:
        return None
    return "\n".join(f"[fleet message from {m['from']}]: {m['text']}" for m in act)


def run(backend, session, verbose=False, workspace=None, resume=None):
    from .tui import Screen, FooterSpinner, ApprovalGate
    screen = Screen()           # activity (working…) · boxed 2-line input · status
    ui = UI(screen.emit, verbose, width=lambda: screen.w)
    ladder = backend if isinstance(backend, list) else [backend]
    agent = Agent(ladder, session, on_event=ui, workspace=workspace, autonomous=True)
    if resume:                  # P4.7: splice reconstructed memory onto the fresh Agent
        from . import resume as resumemod
        try:
            resumemod.apply(agent, resume)
        except Exception:
            pass
    ptype = ""
    if workspace:
        for line in workspace.splitlines():
            if line.startswith("Project type:"):
                ptype = " · " + line.split(":", 1)[1].strip()[:30]
    models = " → ".join(b.name.split(":", 1)[-1] for b in ladder)
    try:
        w = ladder[0].context_window()
        ctx = f" · {w//1024}K ctx" if w >= 1024 else ""
    except Exception:
        ctx = ""
    if hasattr(ladder[0], "warm"):                 # warm before entering screen mode
        with Spinner("loading model"):
            ladder[0].warm()

    gate = ApprovalGate()
    agent.approve = lambda desc: gate.request(desc, agent.stop)

    def status_line():
        try:
            used, window = agent._fill()
            pct = min(99, int(100 * used / window)) if window else 0
        except Exception:
            pct = 0
        m = agent.backend.name.split(":", 1)[-1]
        return f"  {agent.mode} mode (shift+tab cycles) · {m} · {pct}% context · /help"

    screen.enter()
    try:
        _run_loop(screen, ui, agent, session, status_line, gate, models, ctx, ptype)
    except KeyboardInterrupt:
        pass                                   # clean exit, never a traceback
    finally:
        screen.exit()


def _set_mode(agent, screen, mode):
    agent.mode = mode
    screen.emit(f"{MG}  ⏵ {mode} mode{RST} {DIM}— {MODE_HINT[mode]}{RST}\n")


def _run_loop(screen, ui, agent, session, status_line, gate, models, ctx, ptype):
    from .tui import FooterSpinner, WAKE
    screen.emit(_banner(models, ctx, session.cwd, ptype))
    screen.emit(f"{DIM}  Esc clears/stops · shift+tab: auto/plan/manual · /files explorer · type while it works to queue · /help{RST}\n\n")
    resumed = getattr(agent, "_resume_info", None)   # P4.7: set by resume.apply()
    if resumed:
        screen.emit(f"{MG}  ⟲ {resumed}{RST}\n\n")
    history = _load_history()                        # P4.7: prompt history persists across sessions
    pending = []                                        # messages queued for the NEXT turn
    prefill = ""                                        # e.g. "@file " picked in the explorer
    # Wake-on-inbox (P2.3): off by default, so behaviour is unchanged unless opted in.
    wake_cfg = cfgmod.get("wake", "off")
    wake_fd = getattr(session, "wake_fd", None) if wake_cfg in ("render", "act") else None
    wake_budget = WAKE_BUDGET                           # autonomous turns left in this human-free chain

    def cycle_mode():
        _set_mode(agent, screen, MODES[(MODES.index(agent.mode) + 1) % len(MODES)])

    while True:
        if pending:
            user = pending.pop(0)
            wake_budget = WAKE_BUDGET                    # a human message resets the autonomous chain
            screen.emit(f"\n{GR}❯{RST} {user} {DIM}(queued){RST}\n")
        else:
            got = screen.prompt(f"{GR}❯{RST} ", history, status_line, on_mode=cycle_mode,
                                initial=prefill, wake_fd=wake_fd)
            prefill = ""
            if got is None:
                break
            if got is WAKE:                             # a fleet message arrived while idle
                act_text = _on_wake(ui, agent, session, wake_budget)
                if act_text is None:
                    continue                            # render-only — back to the prompt
                wake_budget -= 1                        # spend one autonomous turn from the chain
                user = act_text                         # fall through to a turn, same path as a user turn
            else:
                user = got.strip()
                if not user:
                    continue
                wake_budget = WAKE_BUDGET               # a human turn resets the autonomous chain
                history.append(user)
                _append_history(user)                  # P4.7: persist to ~/.forge/history
                if user in ("/exit", "/quit"):
                    break
                screen.emit(f"\n{GR}❯{RST} {user}\n")   # echo into the transcript
                if user == "/help":
                    screen.emit(f"{DIM}  Esc: clear/stop · ↑↓ history · Ctrl-A/E home/end · Ctrl-U/K/W kill · shift+tab: mode{RST}\n")
                    screen.emit(f"{DIM}  modes — auto: {MODE_HINT['auto']} · plan: {MODE_HINT['plan']} · manual: {MODE_HINT['manual']}{RST}\n")
                    screen.emit(f"{DIM}  while working: type + Enter queues a message the agent absorbs between steps{RST}\n")
                    screen.emit(f"{DIM}  /files: 3-pane folder explorer (Enter on a file attaches it as @file){RST}\n")
                    screen.emit(f"{DIM}  /mode [auto|plan|manual] · /files · /model · /config · /verbose · /plan · /cwd · /exit{RST}\n"); continue
                if user.startswith("/mode"):
                    arg = user.split(None, 1)[1].strip().lower() if " " in user else ""
                    if arg in MODES:
                        _set_mode(agent, screen, arg)
                    else:
                        screen.emit(f"{DIM}  {agent.mode} mode — /mode auto|plan|manual (or shift+tab){RST}\n")
                    continue
                if user in ("/files", "/f", "/explore"):
                    from .tui import Explorer
                    pick = Explorer(screen, session.cwd).run()
                    if pick:
                        prefill = f"@{pick} "
                        screen.emit(f"{DIM}  ⌘ attached @{pick} — finish your message and send{RST}\n")
                    continue
                if user in ("/model", "/models"):
                    _menu_model(agent, screen, history); continue
                if user == "/config":
                    _menu_config(agent, screen); continue
                if user == "/verbose":
                    ui.verbose = not ui.verbose; screen.emit(f"{DIM}  verbose {'on' if ui.verbose else 'off'}{RST}\n"); continue
                if user == "/plan":
                    (ui._render_plan(agent.plan) if agent.plan else screen.emit(f"{DIM}  (no plan yet){RST}\n")); continue
                if user == "/cwd":
                    screen.emit(f"{DIM}  {session.cwd}{RST}\n"); continue

        ui.new_turn()
        agent.stop.clear()

        def queue_msg(txt):
            history.append(txt)
            _append_history(txt)                        # P4.7: persist to ~/.forge/history
            session.push("user", txt)                   # absorbed between agent steps
            screen.emit(f"{DIM}  ⧉ queued: {txt[:90]}{RST}\n")

        spin = FooterSpinner(screen, "working", gate=gate).start()
        reply = None
        try:
            reply = screen.attend(lambda u=user: agent.send(_expand_ats(u, session.cwd)),
                                  agent.stop, f"{GR}❯{RST} ", history,
                                  status=status_line, on_queue=queue_msg, gate=gate)
        except ForgeError as e:
            screen.emit(f"\n  {RD}✗ {e}{RST}\n")
        finally:
            spin.stop()
        for m in session.drain():                       # queued but not absorbed → next turn
            if m["from"] == "user":
                pending.append(m["text"])
            else:                                       # fleet msg: leave for the next turn
                session.push(m["from"], m["text"])
        if reply == "(stopped)":
            screen.emit(f"{DIM}  ⊘ stopped. what next?{RST}\n")
        elif reply is not None and not ui.said:        # fallback (step limit / malformed)
            screen.emit(f"\n{ui.wrap_block(reply)}\n")
        # P4.3: proactive turn-boundary compaction. The turn is done and the user is
        # reading the reply, so invalidating the warm KV prefix costs nothing here —
        # compact EARLY (0.55) so the next turn starts with headroom, leaving the
        # in-turn 0.70 gate + floor as emergency-only.
        try:
            agent.maybe_compact(0.55)
        except Exception:
            pass
