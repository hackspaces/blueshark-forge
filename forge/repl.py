"""The interactive terminal — made to feel alive.

A live plan panel, clean tool-step rendering with pass/fail and timing, a spinner
while the model thinks, and the agent's reply set apart. Designed so watching a
small local model work is legible and fast to read."""
import itertools
import sys
import threading
import time

import subprocess

from .agent import Agent
from .backends import make_backend, ForgeError
from . import config as cfgmod
from .util import slurp
from .tui import run_interruptible

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
    """Renders agent events into the scrolling transcript via `emit`. The spinner
    lives in the pinned footer (owned by run()), not here."""
    def __init__(self, emit, verbose=False):
        self.emit = emit; self.verbose = verbose; self.t0 = None
        self.streamed = False; self.said = False
    def new_turn(self):
        self.streamed = False; self.said = False
    def _line(self, s=""):
        self.emit(s + "\n")
    def __call__(self, kind, **k):
        if kind == "token":
            if not self.streamed:
                self.emit("\n"); self.streamed = True
            self.emit(k["text"])
        elif kind == "say":
            self.said = True
            self.emit("\n" if self.streamed else f"\n{k.get('message','')}\n")
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
        elif kind == "inbox":
            self._line(f"  {YE}✉ {k['sender']}: {k['text'][:76]}{RST}")
        elif kind == "compact":
            self._line(f"  {DIM}⟲ context compacted{RST}")
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
    agent.set_ladder([make_backend(m) for m in chosen])
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


def run(backend, session, verbose=False, workspace=None):
    from .tui import Screen, FooterSpinner
    screen = Screen(footer=3)
    ui = UI(screen.emit, verbose)
    ladder = backend if isinstance(backend, list) else [backend]
    agent = Agent(ladder, session, on_event=ui, workspace=workspace, autonomous=True)
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

    def status_line():
        try:
            used, window = agent._fill()
            pct = min(99, int(100 * used / window)) if window else 0
        except Exception:
            pct = 0
        m = agent.backend.name.split(":", 1)[-1]
        return f"  {m} · {pct}% context · Esc stops · /model /config /help"

    screen.enter()
    try:
        screen.emit(f"{B}{MG}forge{RST} · {models}{ctx} · {DIM}{session.cwd}{ptype}{RST}\n")
        screen.emit(f"{DIM}Esc clears the line (or stops the agent mid-run) · @file to include a file · /help{RST}\n\n")
        history = []
        while True:
            user = screen.prompt(f"{GR}❯{RST} ", history, status_line())
            if user is None:
                break
            user = user.strip()
            if not user:
                continue
            history.append(user)
            if user in ("/exit", "/quit"):
                break
            screen.emit(f"\n{GR}❯{RST} {user}\n")           # echo into the transcript
            if user == "/help":
                screen.emit(f"{DIM}  Esc: clear/stop · /model · /config · /verbose · /plan · /cwd · /exit{RST}\n"); continue
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
            screen.show_submitted(f"{GR}❯{RST} ", user)
            spin = FooterSpinner(screen, "working").start()
            reply = None
            try:
                reply = run_interruptible(lambda: agent.send(_expand_ats(user, session.cwd)), agent.stop)
            except ForgeError as e:
                screen.emit(f"\n  {RD}✗ {e}{RST}\n")
            finally:
                spin.stop()
            if reply == "(stopped)":
                screen.emit(f"{DIM}  ⊘ stopped. what next?{RST}\n")
            elif reply is not None and not ui.said:        # fallback (step limit / malformed)
                screen.emit(f"\n{reply}\n")
    finally:
        screen.exit()
