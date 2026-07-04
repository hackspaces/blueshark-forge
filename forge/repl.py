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
from .backends import make_backend
from . import config as cfgmod
from .tui import read_line, run_interruptible

DIM = "\033[2m"; B = "\033[1m"; CY = "\033[36m"; GR = "\033[32m"; YE = "\033[33m"; RD = "\033[31m"; MG = "\033[35m"; RST = "\033[0m"

ICON = {"bash": "⚡", "read_file": "▸", "write_file": "✎", "edit_file": "✎", "list_files": "▸", "say": "▪"}


class Spinner:
    def __init__(self, label="thinking"):
        self.label = label; self._stop = False; self._t = None
    def __enter__(self):
        def spin():
            for c in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if self._stop: break
                sys.stdout.write(f"\r{DIM}{c} {self.label}…{RST}\033[K"); sys.stdout.flush(); time.sleep(0.08)
            sys.stdout.write("\r\033[K"); sys.stdout.flush()
        self._t = threading.Thread(target=spin, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop = True
        if self._t: self._t.join()


def _render_plan(plan):
    print(f"{DIM}  ┌─ plan{RST}")
    for item in plan:
        s = item.strip()
        if s.startswith("[x]"):
            print(f"{DIM}  │{RST} {GR}✓{RST} {DIM}{s[3:].strip()}{RST}")
        elif s.startswith("[~]"):
            print(f"{DIM}  │{RST} {YE}▸{RST} {B}{s[3:].strip()}{RST}")
        else:
            print(f"{DIM}  │  {s.lstrip('[ ]').strip()}{RST}")
    print(f"{DIM}  └─{RST}")


class UI:
    def __init__(self, verbose=False):
        self.verbose = verbose; self.spin = None; self.t0 = None
        self.streamed = False; self.said = False
    def new_turn(self):
        self.streamed = False; self.said = False
    def __call__(self, kind, **k):
        if kind == "thinking":
            self.start_spin()
        elif kind == "token":
            self._end_spin()
            if not self.streamed:
                print()  # blank line before the reply begins
                self.streamed = True
            print(k["text"], end="", flush=True)
        elif kind == "say":
            self.said = True
            if self.streamed:
                print("\n")
            else:
                self._end_spin(); print(f"\n{k.get('message','')}\n")
        elif kind == "plan":
            self._end_spin(); _render_plan(k["plan"])
        elif kind == "action":
            self._end_spin()
            ic = ICON.get(k["action"], "·"); detail = (k.get("detail") or "").replace("\n", " ")[:74]
            print(f"  {CY}{ic}{RST} {k['action']} {DIM}{detail}{RST}", end="", flush=True)
            self.t0 = time.time()
        elif kind == "observation":
            dt = f"{time.time()-self.t0:.1f}s" if self.t0 else ""
            mark = f"{GR}ok{RST}" if k.get("ok") else f"{RD}fail{RST}"
            print(f"  {DIM}{dt}{RST} {mark}")
            if self.verbose:
                first = (k.get("text") or "").strip().splitlines()[:1]
                if first: print(f"    {DIM}→ {first[0][:80]}{RST}")
            self.t0 = None
        elif kind == "escalate":
            self._end_spin(); print(f"  {MG}↑ stuck — escalating to a stronger local model: {k['model']}{RST}")
        elif kind == "inbox":
            self._end_spin(); print(f"  {YE}✉ {k['sender']}: {k['text'][:76]}{RST}")
        elif kind == "compacting":
            self.start_spin("compacting context")
        elif kind == "compact":
            self._end_spin(); print(f"  {DIM}⟲ context compacted (now ~{k.get('tokens','?')} tokens){RST}")
        elif kind in ("malformed", "loop"):
            self._end_spin(); print(f"  {DIM}· ({kind}, recovering){RST}")
    def start_spin(self, label="thinking"):
        self.spin = Spinner(label).__enter__()
    def _end_spin(self):
        if self.spin: self.spin.__exit__(); self.spin = None


def _ollama_models():
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, timeout=10)
        return [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]
    except (subprocess.SubprocessError, OSError):
        return []


def _menu_model(agent, history):
    """/model — pick a new ladder from installed models. Persists to config."""
    models = _ollama_models()
    cur = " → ".join(b.name.split(":", 1)[-1] for b in agent.ladder)
    print(f"{DIM}  current ladder: {cur}{RST}")
    if not models:
        print(f"{DIM}  (no ollama models found){RST}"); return
    for i, m in enumerate(models, 1):
        print(f"    {CY}{i}{RST} {m}")
    print(f"{DIM}  type numbers cheap→strong (e.g. '1 3'), or a name, or blank to cancel{RST}")
    pick = read_line(f"{GR}model›{RST} ", history)
    if not pick or not pick.strip():
        print(f"{DIM}  cancelled{RST}"); return
    chosen = []
    for tok in pick.split():
        if tok.isdigit() and 1 <= int(tok) <= len(models):
            chosen.append(models[int(tok) - 1])
        else:
            chosen.append(tok)
    if not chosen:
        return
    agent.set_ladder([make_backend(m) for m in chosen])
    cfgmod.set_key("ladder", chosen)
    print(f"{GR}  ✓ ladder → {' → '.join(chosen)}{RST}  {DIM}(saved to config){RST}")


def _menu_config(agent):
    """/config — show current settings."""
    cfg = cfgmod.load()
    print(f"{DIM}  config ({cfgmod.PATH}):{RST}")
    for k in ("ladder", "num_ctx", "keep_alive", "num_predict", "stuck_threshold"):
        v = cfg.get(k)
        print(f"    {CY}{k}{RST}: {v if not isinstance(v, list) else ' → '.join(v)}")
    m = cfg.get("machine", {})
    if m:
        print(f"    {DIM}machine: {m.get('chip','')} · {m.get('ram_gb','?')}GB · {m.get('cores','?')} cores{RST}")


def _expand_ats(text, cwd):
    """Expand @path tokens into inline file contents, so `read this @foo.js` just works."""
    import os
    import re
    out = text
    for tok in re.findall(r"@([\w./\-]+)", text):
        p = os.path.join(cwd, tok)
        if os.path.isfile(p):
            try:
                body = open(p).read()[:8000]
                out += f"\n\n[contents of {tok}]\n{body}"
            except OSError:
                pass
    return out


def run(backend, session, verbose=False, workspace=None):
    ui = UI(verbose)
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
    print(f"{B}{MG}forge{RST} · {models}{ctx} · {DIM}{session.cwd}{ptype}{RST}")
    if hasattr(ladder[0], "warm"):
        with Spinner("loading model"):
            ladder[0].warm()
    print(f"{DIM}Esc clears the line · Esc stops the agent mid-run · @file to include a file · /help · Ctrl-D to quit{RST}\n")
    history = []
    while True:
        user = read_line(f"{GR}❯{RST} ", history)
        if user is None:
            print(); break
        user = user.strip()
        if not user:
            continue
        history.append(user)
        if user in ("/exit", "/quit"):
            break
        if user == "/help":
            print(f"{DIM}  Esc: clear line / stop agent · /model switch models · /config settings · /verbose · /plan · /cwd · /exit{RST}"); continue
        if user in ("/model", "/models"):
            _menu_model(agent, history); continue
        if user == "/config":
            _menu_config(agent); continue
        if user == "/verbose":
            ui.verbose = not ui.verbose; print(f"{DIM}  verbose {'on' if ui.verbose else 'off'}{RST}"); continue
        if user == "/plan":
            _render_plan(agent.plan) if agent.plan else print(f"{DIM}  (no plan yet){RST}"); continue
        if user == "/cwd":
            print(f"{DIM}  {session.cwd}{RST}"); continue
        ui.new_turn()
        agent.stop.clear()
        def hint():
            ui._end_spin(); print(f"\n{DIM}  stopping…{RST}")
        reply = run_interruptible(lambda: agent.send(_expand_ats(user, session.cwd)), agent.stop, on_hint=hint)
        ui._end_spin()
        if reply == "(stopped)":
            print(f"{DIM}  ⊘ stopped. what next?{RST}\n")
        elif not ui.said:        # fallback (step limit / malformed): print the raw reply
            print(f"\n{reply}\n")
