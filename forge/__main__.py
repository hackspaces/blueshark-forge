"""forge CLI entry.

  forge                       chat with an agent (default model) in the cwd
  forge --model gemma2:9b     pick any model (Ollama spec, or openai:...)
  forge run "<task>"          one-shot: run a task to completion, non-interactive
  forge status                list live forge sessions
"""
import argparse
import json
import os
import sys
import time
import uuid

from .backends import make_backend
from . import session as sessmod
from . import config as cfgmod


def _default_model():
    return os.environ.get("FORGE_MODEL") or ",".join(cfgmod.get("ladder", ["gemma2:9b"]))


def _new_session(model, cwd, name=None):
    sid = uuid.uuid4().hex[:12]
    s = sessmod.Session(sid, cwd, model, name=name)
    s.start_inbox()
    s.register()
    return s


def _workspace_ctx(cwd):
    from . import workspace, fleet
    try:
        return workspace.context(cwd, learnings=fleet.learnings(cwd))
    except Exception:
        return None


def _make_ladder(spec):
    """`--model a,b,c` = a local model ladder (cheapest→strongest); forge starts
    on `a` and escalates a rung whenever it gets stuck. A single model = no ladder."""
    return [make_backend(s.strip()) for s in spec.split(",") if s.strip()]


def cmd_chat(args):
    from .repl import run
    ladder = _make_ladder(args.model)
    cwd = os.path.abspath(args.dir)
    s = _new_session(ladder[0].name, cwd, name=args.name)
    try:
        run(ladder, s, verbose=args.verbose, workspace=_workspace_ctx(cwd))
    finally:
        s.deregister()


def cmd_run(args):
    from .agent import Agent
    ladder = _make_ladder(args.model)
    s = _new_session(ladder[0].name, os.path.abspath(args.dir))
    state = {"streamed": False, "said": False}
    def on_event(kind, **k):
        if kind == "plan":
            print("  plan:")
            for it in k["plan"]:
                print(f"    {it}")
        elif kind == "action":
            print(f"  · {k['action']}: {(k.get('detail') or '')[:80]}", end="", flush=True)
        elif kind == "observation":
            print(f"  [{'ok' if k.get('ok') else 'fail'}]")
        elif kind == "token":
            if not state["streamed"]:
                print(); state["streamed"] = True
            print(k["text"], end="", flush=True)
        elif kind == "say":
            state["said"] = True
            if state["streamed"]:
                print()
            else:
                print(f"\n{k.get('message','')}")
        elif kind == "escalate":
            print(f"  ↑ stuck — escalating to {k['model']}")
        elif kind == "inbox":
            print(f"  ✉ {k['sender']}: {k['text'][:70]}")
    agent = Agent(_make_ladder(args.model), s, on_event=on_event, max_steps=args.max_steps, autonomous=True,
                  workspace=_workspace_ctx(os.path.abspath(args.dir)))
    try:
        reply = agent.send(args.task)
        if not state["said"]:
            print(f"\n{reply}")
    finally:
        s.deregister()


def _daemon_running():
    pidf = os.path.expanduser("~/.forge/state/forged.pid")
    if not os.path.exists(pidf):
        return None
    try:
        pid = int(open(pidf).read())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def cmd_status(args):
    pid = _daemon_running()
    print(f"● autopilot UP (pid {pid})" if pid else "○ autopilot down  (start: forge up)")
    print()
    live = sessmod.registry()
    if not live:
        print("no live forge sessions."); return
    for e in live:
        print(f"● {e['name']:16} {e.get('model',''):22} {e['status']:8} ({e['sid'][:8]}) {e['cwd']}")


def cmd_send(args):
    from . import fleet
    e = fleet.send(args.target, " ".join(args.message), sender="user")
    print(f"delivered to {e['name']} ({e['sid'][:8]})")


def cmd_up(args):
    import subprocess
    if _daemon_running():
        print(f"autopilot already up (pid {_daemon_running()})"); return
    os.makedirs(os.path.expanduser("~/.forge/state"), exist_ok=True)
    log = open(os.path.expanduser("~/.forge/forged.log"), "a")
    p = subprocess.Popen([sys.executable, "-m", "forge.daemon", args.model, str(args.interval)],
                         stdout=log, stderr=log, start_new_session=True,
                         env={**os.environ, "PYTHONPATH": os.path.expanduser("~/forge")})
    open(os.path.expanduser("~/.forge/state/forged.pid"), "w").write(str(p.pid))
    time.sleep(1)
    print(f"forge autopilot up (pid {p.pid}) — TRUST + COORDINATE + LEARN, checker model {args.model}")


def cmd_down(args):
    pid = _daemon_running()
    pidf = os.path.expanduser("~/.forge/state/forged.pid")
    if pid:
        os.kill(pid, 15); os.remove(pidf); print("autopilot stopped")
    else:
        print("not running")


def cmd_receipts(args):
    f = os.path.expanduser("~/.forge/verdicts.jsonl")
    if not os.path.exists(f):
        print("no verdicts yet."); return
    for line in open(f).read().splitlines()[-args.n:]:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        mark = "✓" if d.get("verdict") == "CONFIRMED" else "✗"
        proj = d.get("cwd", "").rstrip("/").split("/")[-1]
        print(f"{mark} {time.strftime('%m-%d %H:%M', time.localtime(d.get('ts', 0)))}  {proj[:16]:16} {d.get('verdict')}")
        print(f"    {(d.get('evidence','') or '')[:150]}")


def cmd_learnings(args):
    from . import fleet
    facts = fleet.learnings(os.path.abspath(args.dir))
    if not facts:
        print("(no learnings yet for this repo)"); return
    for f in facts:
        print(f"• {f}")


def main():
    ap = argparse.ArgumentParser(prog="forge")
    # default LOCAL LADDER comes from ~/.forge/config.json (written by `forge setup`)
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--dir", default=os.getcwd())
    ap.add_argument("--name", default=None)
    ap.add_argument("--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="one-shot task")
    p_run.add_argument("task")
    p_run.add_argument("--max-steps", type=int, default=40)

    sub.add_parser("status", help="autopilot state + live sessions")

    p_send = sub.add_parser("send", help="message another session")
    p_send.add_argument("target")
    p_send.add_argument("message", nargs="+")

    p_up = sub.add_parser("up", help="start the fleet autopilot")
    p_up.add_argument("--interval", type=int, default=20)
    sub.add_parser("down", help="stop the fleet autopilot")

    p_rc = sub.add_parser("receipts", help="trust audit trail")
    p_rc.add_argument("n", nargs="?", type=int, default=15)

    p_ln = sub.add_parser("learnings", help="facts learned in a repo")
    p_ln.add_argument("dir", nargs="?", default=os.getcwd())

    p_setup = sub.add_parser("setup", help="detect hardware, pull the right models, write config")
    p_setup.add_argument("--auto", action="store_true", help="no prompts")

    args = ap.parse_args()
    if args.cmd == "setup":
        from . import setup as setupmod
        sys.exit(setupmod.run(auto=args.auto))
    dispatch = {"run": cmd_run, "status": cmd_status, "send": cmd_send, "up": cmd_up,
                "down": cmd_down, "receipts": cmd_receipts, "learnings": cmd_learnings}
    (dispatch.get(args.cmd) or cmd_chat)(args)


if __name__ == "__main__":
    main()
