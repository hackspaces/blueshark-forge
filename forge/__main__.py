"""forge CLI entry.

  forge                       chat with an agent (default model) in the cwd
  forge --model gemma2:9b     pick any model (Ollama spec, or openai:...)
  forge --resume <sid|last>   resume a prior session from its transcript
  forge run "<task>"          one-shot: run a task to completion, non-interactive
  forge status                list live forge sessions
"""
import argparse
import json
import os
import shutil
import sys
import time
import uuid
import signal

from . import __version__

from .backends import make_backend, ForgeError, RecordingBackend
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


def _ctx_budget(backend):
    """The starting rung's effective context window, for sizing the briefing.
    Best-effort — a probe failure just yields None (→ full briefing)."""
    try:
        return backend.effective_ctx()
    except Exception:
        return None


def _workspace_ctx(cwd, budget=None):
    from . import workspace, fleet
    try:
        return workspace.context(cwd, learnings=fleet.learnings(cwd), budget=budget)
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
    ladder = [make_backend(s.strip(), engine=eng, base_url=url, api_key=key)
              for s in (spec or "").split(",") if s.strip()]
    if not ladder:
        raise ForgeError("no model configured — pass --model or run `forge setup`.")
    # P3.3 flight recorder: FORGE_RECORD=<path> wraps every rung so each model call
    # appends a {digest, raw, prompt_tokens} cassette row. Unset → zero wrapping,
    # zero behavior change.
    rec = os.environ.get("FORGE_RECORD")
    if rec:
        ladder = [RecordingBackend(b, rec) for b in ladder]
    return ladder


def cmd_chat(args):
    from .repl import run
    ladder = _make_ladder(args.model)
    cwd = os.path.abspath(args.dir)
    resume_data = None
    if getattr(args, "resume", None):
        from . import resume as resumemod
        sid = resumemod.resolve_sid(args.resume, cwd)
        if not sid:
            print(f"✗ no resumable session found for {args.resume!r} in {cwd}", file=sys.stderr)
            sys.exit(1)
        if resumemod.is_live(sid):
            print(f"✗ session {sid} is still running — refusing to resume a live session", file=sys.stderr)
            sys.exit(1)
        resume_data = resumemod.load(sid)
        if not resume_data:
            print(f"✗ session {sid} has no transcript to resume", file=sys.stderr)
            sys.exit(1)
    s = _new_session(ladder[0].name, cwd, name=args.name)
    try:
        run(ladder, s, verbose=args.verbose,
            workspace=_workspace_ctx(cwd, _ctx_budget(ladder[0])), resume=resume_data)
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
        elif kind == "borrow":
            print(f"  ⇡ borrowing one action from {k['model']}")
        elif kind == "deescalate":
            print(f"  ↓ recovered — back to {k['model']}")
        elif kind == "inbox":
            print(f"  ✉ {k['sender']}: {k['text'][:70]}")
    agent = Agent(_make_ladder(args.model), s, on_event=on_event, max_steps=args.max_steps, autonomous=True,
                  workspace=_workspace_ctx(os.path.abspath(args.dir), _ctx_budget(ladder[0])))
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


# --- status rendering helpers -------------------------------------------------

_ANSI = {"reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
         "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
         "blue": "\033[34m", "red": "\033[31m"}


def _color_on():
    return (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") != "dumb")


def _paint(text, *styles):
    if not _color_on():
        return text
    return "".join(_ANSI[s] for s in styles) + text + _ANSI["reset"]


def _tilde(path):
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path and path.startswith(home) else (path or "")


def _fit(text, width):
    """Collapse whitespace and hard-truncate to `width` columns with an ellipsis."""
    text = " ".join((text or "").split())
    if width < 1:
        return ""
    return text if len(text) <= width else text[:width - 1].rstrip() + "…"


def cmd_status(args):
    from . import fleet, bridge
    pid = _daemon_running()
    head = (_paint("●", "green") + " autopilot " + _paint("UP", "green", "bold") + f" · pid {pid}"
            if pid else _paint("○", "dim") + " autopilot " + _paint("down", "dim")
            + _paint("  · start: forge up", "dim"))

    live = sessmod.registry()
    claude = bridge.claude_peers()
    if not live and not claude:
        print(head + "\n\n" + _paint("no live forge or Claude Code sessions.", "dim"))
        return

    # unify both runtimes into one shape so they render identically
    rows = []
    for e in live:
        st = e.get("status", "idle")
        recs = fleet._records(e["sid"])
        you = next((r["text"] for r in reversed(recs)
                    if r.get("type") == "user" and r.get("text")), None)
        rows.append({"runtime": "forge", "status": st, "name": e["name"],
                     "model": e.get("model", ""), "sid": e["sid"], "cwd": e["cwd"],
                     "task": None, "you": you, "reply": fleet.last_say(e["sid"])})
    for e in claude:
        info = bridge.summarize(e)
        rows.append({"runtime": "claude", "status": None, "name": e["name"],
                     "model": "", "sid": e["sid"], "cwd": e["cwd"],
                     "task": info["title"], "you": info["prompt"], "reply": info["claude"]})

    W = shutil.get_terminal_size((100, 24)).columns
    name_w = min(20, max(len(r["name"]) for r in rows))
    n_forge = sum(1 for r in rows if r["runtime"] == "forge")
    counts = _paint(f"{n_forge} forge · {len(rows) - n_forge} claude", "dim")
    pad = max(1, W - len(_strip_ansi(head)) - len(_strip_ansi(counts)) - 2)
    print(head + " " * pad + counts + "\n")

    glyphs = {("forge", "idle"): ("●", "green"), ("forge", "working"): ("◐", "yellow"),
              ("forge", "stuck"): ("◍", "red"), ("claude", None): ("◇", "cyan")}
    for r in rows:
        glyph, gcolor = glyphs.get((r["runtime"], r["status"]), ("●", "green"))
        tag = f"forge/{r['status']}" if r["runtime"] == "forge" else "claude"
        header = (f" {_paint(glyph, gcolor)}  {_paint(r['name'].ljust(name_w), 'cyan', 'bold')}"
                  f"  {_paint(tag.ljust(13), 'dim')}  {_paint(r['sid'][:8], 'dim')}")
        if r["model"]:
            header += _paint(f"  {r['model']}", "dim")
        print(header)
        print(f"     {_paint(_fit(_tilde(r['cwd']), W - 5), 'dim')}")
        # activity: task (claude) + last ask + last reply, each on its own fitted line
        label_w = 6
        avail = W - 5 - label_w - 1
        if r["task"]:
            print(f"     {_paint('task'.ljust(label_w), 'dim')} {_fit(r['task'], avail)}")
        if r["you"]:
            print(f"     {_paint('you'.ljust(label_w), 'dim')} {_paint(_fit(r['you'], avail), 'dim')}")
        if r["reply"]:
            rlabel = r["runtime"]
            print(f"     {_paint(rlabel.ljust(label_w), gcolor)} {_fit(r['reply'], avail)}")
        print()


def _strip_ansi(s):
    import re as _re
    return _re.sub(r"\033\[[0-9;]*m", "", s)


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
    for r in facts:
        mark = "✓" if r.get("verified") else " "
        print(f"{mark} {r.get('fact', '')}")


def cmd_forget(args):
    from . import fleet
    n = fleet.forget(os.path.abspath(args.dir), args.pattern)
    tail = f" matching '{args.pattern}'" if args.pattern else ""
    print(f"forgot {n} fact(s){tail}")


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
    FLAGS = ("malformed", "loop_trip", "gated", "escalated", "borrowed", "compacted")
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


def cmd_corpus(args):
    """Flywheel turn one: turn forge session transcripts into harness-native training data —
    SFT examples (context → the action that worked) + preference pairs (the moments the
    harness corrected the model: malformed→valid JSON, narrate→act). Writes JSONL."""
    import glob
    from . import corpus, fleet
    from .agent import SYSTEM
    if args.all:
        files = sorted(glob.glob(os.path.join(sessmod.SESSIONS, "*.jsonl")), key=os.path.getmtime)
        sids = [os.path.basename(f)[:-len(".jsonl")] for f in files]
    else:
        sid = args.sid
        if sid == "last":
            files = glob.glob(os.path.join(sessmod.SESSIONS, "*.jsonl"))
            if not files:
                print("no sessions found."); return
            sid = os.path.basename(max(files, key=os.path.getmtime))[:-len(".jsonl")]
        sids = [sid]
    if not sids:
        print("no sessions found."); return
    system = None if args.no_system else SYSTEM
    rows = []
    for sid in sids:
        recs = fleet._records(sid, tail_bytes=10 ** 9)
        rows.extend(corpus.build_jsonl(recs, sid=sid, system=system))
    nsft = sum(1 for r in rows if r["split"] == "sft")
    npref = sum(1 for r in rows if r["split"] == "pref")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows ({nsft} SFT, {npref} preference) from {len(sids)} "
              f"session(s) → {args.out}")
    else:
        print(f"{nsft} SFT examples + {npref} preference pairs from {len(sids)} session(s). "
              f"Add --out corpus.jsonl to write them.")


def cmd_replay(args):
    """P3.3 — re-drive a recorded session through the harness with NO model.
    `sid` defaults to 'last'. With --to-fixture, snapshot the session's raws into
    tests/fixtures/<NAME>.jsonl instead of replaying."""
    import glob
    from . import replay as replaymod
    sid = args.sid
    if sid == "last":
        files = glob.glob(os.path.join(sessmod.SESSIONS, "*.jsonl"))
        if not files:
            print("no sessions found."); return
        sid = os.path.basename(max(files, key=os.path.getmtime))[:-len(".jsonl")]
    if args.to_fixture:
        try:
            path = replaymod.write_fixture(sid, args.to_fixture)
        except ValueError as e:
            print(f"✗ {e}", file=sys.stderr); sys.exit(1)
        print(f"wrote fixture {path}")
        return
    if args.fault:
        print(replaymod.replay_faults(sid, args.fault, strict=args.strict))
    else:
        print(replaymod.replay(sid, strict=args.strict))


def cmd_bench(args):
    """P3.2 — harness-lift eval. Run each task through the real Agent.send loop for
    every selected lever-config (bare vs full, plus any ablation), append rows to
    ~/.forge/bench/results.jsonl, and (with --report) print the lift + ablation
    tables. 'bare' = every lever off; the loop still demands JSON and bails after 5
    malformed replies, so a bare pass-rate substantially measures format compliance
    — that is the honest harness-lift story."""
    from . import bench
    tasks = bench.list_tasks()
    if args.tasks:
        want = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t in want]
    if not tasks:
        print("no bench tasks found (looked in bench/)."); return
    configs = bench.configs_for(args)
    rows = []
    for task in tasks:
        task_dir = os.path.join(bench.bench_dir(), task)
        for label, levers in configs:
            ladder = _make_ladder(args.model)
            print(f"· {task}  [{label}]  {args.model} …", flush=True)
            try:
                row = bench.run_task(task_dir, ladder, levers,
                                     max_steps=args.max_steps, model=args.model)
            except ForgeError as e:
                print(f"  ✗ {e}", file=sys.stderr); continue
            verdict = {True: "PASS", False: "FAIL", None: "no-verdict"}[row["pass"]]
            print(f"  {verdict}  steps={row['steps']} {row['seconds']}s "
                  f"malformed={row['malformed']} loops={row['loops']} esc={row['escalations']}")
            rows.append(row)
    bench.append_results(rows)
    print(f"\n{len(rows)} run(s) appended to {bench.RESULTS}")
    if args.report:
        print()
        print(bench.report(rows))


def cmd_passport(args):
    """P5.8 — show each model's learned capability passport (passive per-session rates +
    the active-probe scores) and the per-model knobs it resolves to. With --probe, run
    the ~90s active probe against each model first and (re)write its passport."""
    from . import profile
    from . import backends
    from .backends import make_backend
    from .agent import DEFAULT_LOOP_THRESHOLD, DEFAULT_HEAT_BUMP
    defaults = {"loop_threshold": DEFAULT_LOOP_THRESHOLD, "heat_bump": DEFAULT_HEAT_BUMP,
                "num_predict": backends.NUM_PREDICT}
    cfg = cfgmod.load()
    models = [args.model] if args.model else list(cfg.get("ladder") or [])
    if not models:
        print("no models configured — run `forge setup` first."); return
    eng = cfg.get("engine", "ollama")
    url = cfg.get("base_url") or None
    key = cfg.get("api_key") or None
    # canonical backend name (e.g. "ollama:gemma2:9b") is the passport store key.
    backs = [make_backend(m, engine=eng, base_url=url, api_key=key) for m in models]
    if args.probe:
        from . import setup as setupmod
        print("probing (real model calls — this can take ~90s per model):")
        for b in backs:
            try:
                setupmod.passport(b)
            except Exception as e:
                print(f"    · {b.name}: probe failed ({e})")
        print()
    for b in backs:
        for line in profile.describe(b.name, defaults):
            print(line)
        print()


def main():
    ap = argparse.ArgumentParser(prog="forge")
    # default LOCAL LADDER comes from ~/.forge/config.json (written by `forge setup`)
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--dir", default=os.getcwd())
    ap.add_argument("--name", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--resume", metavar="SID|last", default=None,
                    help="resume a prior session from its transcript ('last' = newest for this dir)")
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

    p_fg = sub.add_parser("forget", help="prune learned facts (optionally matching a pattern)")
    p_fg.add_argument("pattern", nargs="?", default=None, help="substring; omit to clear all facts for --dir")

    p_tr = sub.add_parser("trace", help="pretty-print a session's step trace")
    p_tr.add_argument("sid", nargs="?", default="last", help="session id (or 'last', the default)")

    p_co = sub.add_parser("corpus", help="turn session transcripts into harness-native training data (SFT + preference pairs)")
    p_co.add_argument("sid", nargs="?", default="last", help="session id (or 'last', the default)")
    p_co.add_argument("--all", action="store_true", help="every recorded session, not just one")
    p_co.add_argument("--out", metavar="FILE", help="write JSONL to FILE (default: just print the counts)")
    p_co.add_argument("--no-system", dest="no_system", action="store_true", help="omit the forge system prompt from each example's context")

    p_rp = sub.add_parser("replay", help="re-drive a recorded session through the harness with NO model")
    p_rp.add_argument("sid", nargs="?", default="last", help="session id (or 'last', the default)")
    p_rp.add_argument("--strict", action="store_true", help="assert each recorded prompt digest matches (trips on any prompt change)")
    p_rp.add_argument("--to-fixture", dest="to_fixture", metavar="NAME", help="snapshot this session's raws into tests/fixtures/<NAME>.jsonl")
    from .faults import FAULTS as _FAULT_NAMES
    p_rp.add_argument("--fault", action="append", choices=_FAULT_NAMES,
                      help="inject a deterministic fault before zero-inference replay; repeat for several")

    p_bench = sub.add_parser("bench", help="harness-lift eval: same model bare vs full harness + per-lever ablation")
    p_bench.add_argument("--tasks", help="comma-separated subset of bench task names (default: all)")
    p_bench.add_argument("--max-steps", type=int, default=40)
    p_bench.add_argument("--bare", action="store_true", help="also run with NO harness (every lever off)")
    p_bench.add_argument("--no-compact", dest="no_compact", action="store_true", help="ablate: full harness minus compaction")
    p_bench.add_argument("--no-loop-detect", dest="no_loop_detect", action="store_true", help="ablate: full harness minus loop detection")
    p_bench.add_argument("--no-read-gate", dest="no_read_gate", action="store_true", help="ablate: full harness minus read-before-edit")
    p_bench.add_argument("--single-rung", dest="single_rung", action="store_true", help="ablate: full harness minus escalation")
    p_bench.add_argument("--report", action="store_true", help="print the lift + ablation tables after running")

    p_pp = sub.add_parser("passport", help="show each model's learned capability passport + the knobs it tunes")
    p_pp.add_argument("model", nargs="?", help="a single model name (default: every model in the configured ladder)")
    p_pp.add_argument("--probe", action="store_true", help="run the active probe now and (re)write the passport(s)")

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
                "forget": cmd_forget, "trace": cmd_trace, "corpus": cmd_corpus, "bench": cmd_bench, "replay": cmd_replay,
                "passport": cmd_passport}
    try:
        (dispatch.get(args.cmd) or cmd_chat)(args)
    except ForgeError as e:
        print(f"✗ {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
