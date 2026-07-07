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
import signal

from . import __version__

from .backends import make_backend, ForgeError
from . import session as sessmod
from .util import slurp, dump
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
    """`--model a,b,c` = a model ladder (cheapest→strongest); forge starts on `a`
    and escalates a rung when stuck. Bare model names route to the configured
    engine (ollama by default; vLLM/llama.cpp/etc. after `forge setup`)."""
    cfg = cfgmod.load()
    eng = cfg.get("engine", "ollama")
    url = cfg.get("base_url") or None
    key = cfg.get("api_key") or None
    return [make_backend(s.strip(), engine=eng, base_url=url, api_key=key)
            for s in spec.split(",") if s.strip()]


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
    except ForgeError as e:
        print(f"\n✗ {e}", file=sys.stderr); s.deregister(); sys.exit(1)
    finally:
        s.deregister()


def _daemon_running():
    pidf = os.path.expanduser("~/.forge/state/forged.pid")
    if not os.path.exists(pidf):
        return None
    try:
        pid = int(slurp(pidf))
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def cmd_status(args):
    from . import fleet, bridge
    pid = _daemon_running()
    print(f"● autopilot UP (pid {pid})" if pid else "○ autopilot down  (start: forge up)")
    print()
    live = sessmod.registry()
    claude = bridge.claude_peers()
    if not live and not claude:
        print("no live forge or Claude Code sessions."); return
    icon = {"working": "◐", "idle": "●"}
    for e in live:
        st = e.get("status", "idle")
        print(f"{icon.get(st,'●')} {e['name']}  [forge/{st}]  {e.get('model','')}  ({e['sid'][:8]})")
        print(f"   {e['cwd']}")
        # what is it doing? last user request + last assistant reply from the transcript
        recs = fleet._records(e["sid"])
        last_user = next((r["text"] for r in reversed(recs) if r.get("type") == "user" and r.get("text")), None)
        last_say = fleet.last_say(e["sid"])
        if last_user:
            print(f"   you:    {last_user[:120]}")
        if last_say:
            print(f"   forge:  {' '.join(last_say.split())[:150]}")
        print()
    for e in claude:
        print(f"◇ {e['name']}  [claude]  ({e['sid'][:8]})")
        print(f"   {e['cwd']}")
        info = bridge.summarize(e)
        if info["title"]:
            print(f"   task:   {info['title'][:120]}")
        if info["prompt"]:
            print(f"   you:    {info['prompt'][:120]}")
        if info["claude"]:
            print(f"   claude: {info['claude'][:150]}")
        print()


def cmd_send(args):
    from . import fleet
    e = fleet.send(args.target, " ".join(args.message), sender="user")
    print(f"delivered to {e['name']} ({e['sid'][:8]})")


def cmd_up(args):
    import subprocess
    if _daemon_running():
        print(f"autopilot already up (pid {_daemon_running()})"); return
    os.makedirs(os.path.expanduser("~/.forge/state"), exist_ok=True)
    with open(os.path.expanduser("~/.forge/forged.log"), "a") as log:
        p = subprocess.Popen([sys.executable, "-m", "forge.daemon", args.model, str(args.interval)],
                             stdout=log, stderr=log, start_new_session=True)  # own process group
    dump(os.path.expanduser("~/.forge/state/forged.pid"), str(p.pid))
    time.sleep(1)
    print(f"forge autopilot up (pid {p.pid}) — TRUST + COORDINATE + LEARN, checker model {args.model}")


def cmd_down(args):
    pid = _daemon_running()
    pidf = os.path.expanduser("~/.forge/state/forged.pid")
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)   # kill the daemon AND its verify subprocesses
        except OSError:
            os.kill(pid, signal.SIGTERM)
        os.remove(pidf); print("autopilot stopped")
    else:
        print("not running")


def cmd_receipts(args):
    f = os.path.expanduser("~/.forge/verdicts.jsonl")
    if not os.path.exists(f):
        print("no verdicts yet."); return
    for line in slurp(f).splitlines()[-args.n:]:
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


def cmd_trace(args):
    """Pretty-print a session's structured step trace (P3.1): the meta header plus
    one row per loop iteration. `sid` defaults to 'last' — the most recently
    modified transcript in ~/.forge/sessions."""
    import glob
    from . import fleet
    sid = args.sid
    if sid == "last":
        files = glob.glob(os.path.join(sessmod.SESSIONS, "*.jsonl"))
        if not files:
            print("no sessions found."); return
        sid = os.path.basename(max(files, key=os.path.getmtime))[:-len(".jsonl")]
    recs = fleet._records(sid, tail_bytes=10 ** 9)   # whole file — the meta record is at the top
    if not recs:
        print(f"no records for session {sid}."); return
    meta = next((r for r in recs if r.get("type") == "meta"), None)
    if meta:
        print(f"forge {meta.get('forge', '?')}  ·  model {meta.get('model', '?')}  ·  mode {meta.get('mode', '?')}")
        ladder = meta.get("ladder") or []
        if ladder:
            print(f"ladder: {', '.join(ladder)}")
        print(f"cwd:    {meta.get('cwd', '?')}")
        print()
    steps = [r for r in recs if r.get("type") == "step"]
    if not steps:
        print("(no step records)"); return
    FLAGS = ("malformed", "loop_trip", "gated", "escalated", "compacted")
    print(f"{'step':>4}  {'tier':>4}  {'action':<11}  {'fill':>5}  {'ok':>4}  {'flags':<30}  {'ms':>6}")
    print("-" * 74)
    for r in steps:
        used, window = r.get("used"), r.get("window")
        fill = f"{100 * used / window:.0f}%" if used and window else "-"
        ok = r.get("ok")
        oks = "-" if ok is None else ("ok" if ok else "FAIL")
        flags = ",".join(f for f in FLAGS if r.get(f)) or "-"
        ms = r.get("elapsed_ms")
        print(f"{r.get('step', '?'):>4}  {r.get('tier', 0):>4}  {(r.get('action') or '-'):<11}  "
              f"{fill:>5}  {oks:>4}  {flags:<30}  {('?' if ms is None else ms):>6}")


def main():
    ap = argparse.ArgumentParser(prog="forge")
    # default LOCAL LADDER comes from ~/.forge/config.json (written by `forge setup`)
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--dir", default=os.getcwd())
    ap.add_argument("--name", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"forge {__version__}")
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

    p_tr = sub.add_parser("trace", help="pretty-print a session's step trace")
    p_tr.add_argument("sid", nargs="?", default="last", help="session id (or 'last', the default)")

    p_setup = sub.add_parser("setup", help="detect hardware, choose an engine, pull/point at models, write config")
    p_setup.add_argument("--auto", action="store_true", help="no prompts (Ollama, RAM-sized ladder)")
    p_setup.add_argument("--engine", help="ollama | vllm | llamacpp | mlx | lmstudio | tgi | sglang | openai")
    p_setup.add_argument("--url", help="base URL for an OpenAI-compatible engine")
    p_setup.add_argument("--api-key", dest="api_key", help="API key for the engine (if needed)")
    p_setup.add_argument("--models", help="comma-separated model names, cheap→strong (for non-ollama engines)")

    args = ap.parse_args()
    if args.cmd == "setup":
        from . import setup as setupmod
        models = [m.strip() for m in args.models.split(",")] if args.models else None
        sys.exit(setupmod.run(auto=args.auto, engine=args.engine, url=args.url,
                              api_key=args.api_key, models=models))
    dispatch = {"run": cmd_run, "status": cmd_status, "send": cmd_send, "up": cmd_up,
                "down": cmd_down, "receipts": cmd_receipts, "learnings": cmd_learnings,
                "trace": cmd_trace}
    (dispatch.get(args.cmd) or cmd_chat)(args)


if __name__ == "__main__":
    main()
