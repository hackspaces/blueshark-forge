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


def _resolve_model(args):
    """The model spec to use: an explicit --model / FORGE_MODEL, else the configured ladder."""
    return getattr(args, "model", None) or _default_model()


def _first_run(args):
    """A fresh install — no config written and no model explicitly chosen. Guide the user
    instead of spinning up a chat on the placeholder default (which they don't have)."""
    return (getattr(args, "model", None) is None
            and not os.environ.get("FORGE_MODEL") and not cfgmod.exists())


def _first_run_welcome():
    """Shown when a fresh install runs bare `forge`: what this machine can run, and the two
    commands to get going — not a chat pointed at a model that isn't installed."""
    from . import setup as setupmod
    from . import registry
    from .models_cmd import _accelerated, _pnum, _hw_desc
    from .render import paint as P
    hw = setupmod.detect_machine()
    accel = _accelerated(hw)
    cap_p, cap_name = registry.ceiling(hw.get("ram_gb") or 0, accel, hw.get("vram_gb", 0))

    def line(cmd, desc):
        print("  " + P(f"{cmd:<24}", "cyan") + desc)

    print()
    print("  " + P("✦ forge", "magenta", "bold") + P("  ·  run any model your machine can handle", "dim"))
    print()
    print("  Nothing set up yet — but " + P(_hw_desc(hw, accel), "bold"))
    if cap_name:
        print("  can run models " + P(f"up to ~{_pnum(cap_p)}B", "green") + " right here.")
    print()
    line("forge models", "see everything it can run")
    line("forge models use phi-2", "a quick starter — pulled + ready in ~2 min")
    line('forge run "…"', "then put it to work")
    print()
    print(P("  (or  forge setup  to pick a model ladder yourself)", "dim"))
    print()


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
    if _first_run(args):
        _first_run_welcome()
        return
    from .repl import run
    ladder = _make_ladder(_resolve_model(args))
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
    if _first_run(args):
        print("✗ no model set up yet.\n  See what this machine can run:  forge models\n"
              "  then pick one:                  forge models use <name>   (or: forge setup)",
              file=sys.stderr)
        sys.exit(1)
    from .agent import Agent
    ladder = _make_ladder(_resolve_model(args))
    s = _new_session(ladder[0].name, os.path.abspath(args.dir))
    state = {"streamed": False, "said": False}
    def on_event(kind, **k):
        W = term_width()
        if kind == "plan":
            print(_paint("  plan:", "cyan"))
            for it in k["plan"]:
                print(f"    {_fit(str(it), W - 4)}")
        elif kind == "action":
            detail = _fit(k.get("detail") or "", W - len(k["action"]) - 12)
            print(f"  {_paint('·', 'cyan')} {k['action']}: {_paint(detail, 'dim')}", end="", flush=True)
        elif kind == "observation":
            print("  " + (_paint("[ok]", "green") if k.get("ok") else _paint("[fail]", "red")))
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
            print(_paint(f"  ↑ stuck — escalating to {k['model']}", "yellow"))
        elif kind == "borrow":
            print(_paint(f"  ⇡ borrowing one action from {k['model']}", "yellow"))
        elif kind == "deescalate":
            print(_paint(f"  ↓ recovered — back to {k['model']}", "green"))
        elif kind == "inbox":
            print(_paint(f"  ✉ {k['sender']}: {_fit(k['text'], W - len(k['sender']) - 6)}", "magenta"))
    from . import mcp as _mcp
    _servers = _mcp.connect(cfgmod.load(), warn=lambda n, e: print(_paint(f"  ⚠ MCP server {n!r} unavailable: {e}", "yellow")))
    agent = Agent(_make_ladder(_resolve_model(args)), s, on_event=on_event, max_steps=args.max_steps, autonomous=True,
                  goal=args.task, mcp_servers=_servers,
                  workspace=_workspace_ctx(os.path.abspath(args.dir), _ctx_budget(ladder[0])))
    try:
        reply = agent.send(args.task)
        if not state["said"]:
            print(f"\n{reply}")
        # H05: a non-interactive run that ended with the harness REJECTING the completion
        # (e.g. strict mode with proof missing, or a failed verification never approved)
        # exits non-zero — CI must not read an unverified stop as success.
        decision = getattr(agent, "_completion_decision", None)
        if decision is not None and not decision.allowed:
            print(f"\n✗ completion not accepted: {decision.reason}", file=sys.stderr)
            s.deregister(); sys.exit(1)
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


from .render import paint as _paint, fit as _fit, tilde as _tilde, term_width, strip_ansi as _strip_ansi


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

    W = term_width()
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


def cmd_send(args):
    from . import fleet
    e = fleet.send(args.target, " ".join(args.message), sender="user")
    print(f"delivered to {e['name']} ({e['sid'][:8]})")


def _daemon_launch_cmd(model, interval):
    """Command to spawn the autopilot daemon as a detached process. A frozen single-file
    binary (PyInstaller/Nuitka) has no `python -m forge.daemon` — sys.executable IS the
    forge binary — so it re-invokes forge's own hidden `daemon` subcommand instead. From
    source, sys.frozen is unset and the `-m` form runs unchanged."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "daemon", model, str(interval)]
    return [sys.executable, "-m", "forge.daemon", model, str(interval)]


def cmd_dex(args):
    """The gamified model-card profile: `forge dex` shows your collection, `forge dex <model>`
    shows one model's full card. Deterministic yet globally unique per install; grows with
    your real work."""
    from . import cards, registry
    if getattr(args, "model", None):
        m = registry.get(args.model)
        if not m:
            print(_paint(f"✗ no model named {args.model!r} in the catalog", "red"), file=sys.stderr)
            sys.exit(1)
        print(cards.render_card(cards.card(m, telemetry=cards._telemetry(m["name"]))))
        return
    tid, owned, total = cards.collection(installed_only=True)
    if not owned:
        print(_paint("No models caught yet — pull one to add its card:  forge models use <name>", "dim"))
        print(_paint("  (or see any model's card:  forge dex <name>)", "dim"))
        return
    print(cards.render_dex(tid, owned, total))
    print(_paint("\n  one model's full card:  forge dex <name>", "dim"))


def cmd_team(args):
    """P9.2 first swarm slice: plan a goal into a task DAG, run each task as a worker in
    its own git worktree, and merge only the branches that pass the verify gate."""
    if _first_run(args):
        print("✗ no model set up yet.\n  See what this machine can run:  forge models",
              file=sys.stderr)
        sys.exit(1)
    from . import team as teammod
    ladder = _make_ladder(_resolve_model(args))
    _colors = {"start": "cyan", "merged": "green", "refuted": "red", "failed": "red",
               "skipped": "yellow", "conflict": "yellow"}
    def on_event(kind, **k):
        if kind == "team_planned":
            print(_paint(f"  planned {len(k['subtasks'])} tasks: {', '.join(k['subtasks'])}", "cyan"))
        elif kind == "team_task":
            st = k.get("status", "")
            tail = f" — {k['detail']}" if k.get("detail") else (f"  {k['title']}" if k.get("title") else "")
            print(_paint(f"  · {k['id']}: {st}{tail}", _colors.get(st, "dim")))
    try:
        res = teammod.run_team(args.goal, os.path.abspath(args.dir), ladder,
                               on_event=on_event, max_steps=args.max_steps)
    except teammod.TeamError as e:
        print(_paint(f"✗ {e}", "red"), file=sys.stderr)
        sys.exit(1)
    merged, total = len(res["merged"]), len(res["results"])
    fin = res["final"]
    print(f"\n{_paint('team done', 'green')}: {merged}/{total} tasks merged into {res['integration_branch']}")
    print(f"  final suite: {'✓' if fin['ok'] else '✗'} {fin['detail']}")
    print(_paint(f"  review:  git diff {res['base']}..{res['integration_branch']}", "dim"))
    print(_paint(f"  keep it: git checkout {res['base']} && git merge {res['integration_branch']}", "dim"))


def cmd_up(args):
    import subprocess
    if _daemon_running():
        print(f"autopilot already up (pid {_daemon_running()})"); return
    model = _resolve_model(args)
    os.makedirs(os.path.expanduser("~/.forge/state"), exist_ok=True)
    with open(os.path.expanduser("~/.forge/forged.log"), "a") as log:
        p = subprocess.Popen(_daemon_launch_cmd(model, args.interval),
                             stdout=log, stderr=log, start_new_session=True)  # own process group
    dump(os.path.expanduser("~/.forge/state/forged.pid"), str(p.pid))
    time.sleep(1)
    print(f"forge autopilot up (pid {p.pid}) — TRUST + COORDINATE + LEARN, checker model {model}")


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
    W = term_width()
    for line in slurp(f).splitlines()[-args.n:]:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ok = d.get("verdict") == "CONFIRMED"
        mark = _paint("✓", "green") if ok else _paint("✗", "red")
        proj = d.get("cwd", "").rstrip("/").split("/")[-1]
        ts = time.strftime("%m-%d %H:%M", time.localtime(d.get("ts", 0)))
        verdict = _paint(d.get("verdict", "") or "", "green" if ok else "red")
        print(f"{mark} {_paint(ts, 'dim')}  {_paint(proj[:16].ljust(16), 'cyan')} {verdict}")
        ev = d.get("evidence", "") or ""
        if ev:
            print(f"    {_paint(_fit(ev, W - 4), 'dim')}")


def cmd_learnings(args):
    from . import fleet
    facts = fleet.learnings(os.path.abspath(args.dir))
    if not facts:
        print(_paint("(no learnings yet for this repo)", "dim")); return
    W = term_width()
    for r in facts:
        mark = _paint("✓", "green") if r.get("verified") else _paint("·", "dim")
        print(f"{mark} {_fit(r.get('fact', ''), W - 2)}")


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
        print(_paint(f"forge {meta.get('forge', '?')}  ·  model {meta.get('model', '?')}  "
                     f"·  mode {meta.get('mode', '?')}", "cyan", "bold"))
        ladder = meta.get("ladder") or []
        if ladder:
            print(_paint(f"ladder: {' → '.join(ladder)}", "dim"))
        print(_paint(f"cwd:    {_tilde(meta.get('cwd', '?'))}", "dim"))
        # H01: recover and show the run's task contract (works for legacy metas too)
        from . import contract as contractmod
        c = contractmod.TaskContract.from_dict(meta.get("contract"), fallback_mode=meta.get("mode", "auto"))
        print(_paint(f"contract: {c.authority} authority · {c.completion_policy} policy · "
                     f"{'verify-required' if c.requires_verification else 'audit-only'} · "
                     f"{(c.max_steps or '∞')} steps", "dim"))
        if c.goal:
            print(_paint(f"goal:   {_fit(c.goal, term_width() - 8)}", "dim"))
        print()
    steps = [r for r in recs if r.get("type") == "step"]
    if not steps:
        print(_paint("(no step records)", "dim")); return
    FLAGS = ("malformed", "loop_trip", "gated", "escalated", "borrowed", "compacted")
    header = f"{'step':>4}  {'tier':>4}  {'action':<11}  {'fill':>5}  {'ok':>4}  {'flags':<30}  {'ms':>6}"
    print(_paint(header, "dim"))
    print(_paint("─" * len(header), "dim"))
    for r in steps:
        used, window = r.get("used"), r.get("window")
        fill = f"{100 * used / window:.0f}%" if used and window else "-"
        ok = r.get("ok")
        oks = ("-" if ok is None else ("ok" if ok else "FAIL"))
        oks_c = _paint(f"{oks:>4}", "red") if ok is False else _paint(f"{oks:>4}", "dim" if ok is None else "green")
        active = [f for f in FLAGS if r.get(f)]
        flags = ",".join(active) or "-"
        flags_c = _paint(f"{flags:<30}", "yellow") if active else _paint(f"{flags:<30}", "dim")
        ms = r.get("elapsed_ms")
        print(f"{r.get('step', '?'):>4}  {r.get('tier', 0):>4}  {_paint((r.get('action') or '-').ljust(11), 'cyan')}  "
              f"{fill:>5}  {oks_c}  {flags_c}  {('?' if ms is None else ms):>6}")


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


def cmd_export(args):
    """H13 — export a session's event log as process-mining data (CSV / JSON / OCEL 2.0),
    with secrets deterministically redacted. `sid` defaults to 'last'."""
    import glob
    from . import export as exportmod, fleet
    sid = args.sid
    if sid == "last":
        files = glob.glob(os.path.join(sessmod.SESSIONS, "*.jsonl"))
        if not files:
            print("no sessions found."); return
        sid = os.path.basename(max(files, key=os.path.getmtime))[:-len(".jsonl")]
    recs = fleet._records(sid, tail_bytes=10 ** 9)
    if not recs:
        print(f"no records for session {sid}."); return
    out = exportmod.export(recs, args.format)
    n = len(exportmod.to_events(recs))
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"exported {n} event(s) as {args.format} → {args.out}")
    else:
        print(out)


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
    model = _resolve_model(args)
    rows = []
    for task in tasks:
        task_dir = os.path.join(bench.bench_dir(), task)
        for label, levers in configs:
            ladder = _make_ladder(model)
            print(f"· {task}  [{label}]  {model} …", flush=True)
            try:
                row = bench.run_task(task_dir, ladder, levers,
                                     max_steps=args.max_steps, model=model)
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


# `forge --help`, grouped by what you're trying to DO. argparse's default is one
# flat alphabetical list plus the 16-name brace blob printed twice — which tells a
# newcomer nothing about where to start, and never mentions that bare `forge` is
# the most common invocation of all. Order here is deliberate: first run first.
_COMMAND_GROUPS = [
    ("Start here", [
        ("models", "what this machine can run — and provision one"),
        ("setup", "detect hardware, pick an engine, write the config"),
    ]),
    ("Work", [
        ("run", 'one task, start to finish  ·  forge run "fix the failing test"'),
        ("team", 'a goal → a task DAG → isolated worktree workers → merge only what verifies'),
        ("bench", "harness-lift eval: the same model bare vs full harness"),
        ("dex", "your model-card collection — each model you run becomes a collectible card"),
    ]),
    ("The fleet", [
        ("up, down", "start / stop the autopilot"),
        ("status", "autopilot state + live sessions"),
        ("send", "message another session"),
    ]),
    ("What happened", [
        ("trace", "pretty-print a session's step trace"),
        ("receipts", "the trust audit trail"),
        ("learnings, forget", "facts learned in a repo, and pruning them"),
        ("passport", "each model's learned capability passport"),
    ]),
    ("Data out", [
        ("export", "a session's event log as process-mining data (CSV/JSON/OCEL)"),
        ("corpus", "transcripts → training data (SFT + preference pairs)"),
        ("replay", "re-drive a recorded session with NO model"),
    ]),
]

# The three invocations people actually type. Kept as data, not inline literals:
# `forge run "<task>"` embeds quotes, and a quoted literal inside an f-string
# expression is a SyntaxError on Python ≤3.11 (PEP 701 only lifted that in 3.12),
# which forge still supports.
_USAGE_EXAMPLES = [
    ("forge", "chat, oriented in the current directory"),
    ('forge run "<task>"', "one task, start to finish"),
    ("forge models", "what this machine can run"),
]

_OPTIONS = [
    ("--model M[,M]", "a model, or a ladder cheap→strong; overrides the config"),
    ("--dir PATH", "where to work (default: this directory)"),
    ("--name NAME", "name this session — the fleet sees it"),
    ("--resume SID|last", "resume a prior session from its transcript"),
    ("--verbose", "show the full step trace"),
    ("--version", "print the version"),
    ("-h, --help", "this help"),
]


def _help_text():
    """The grouped help. Colour is TTY-gated by render.paint, so piping
    `forge --help` into a file or a pager stays plain text."""
    from .render import paint as P
    W = 20                                   # command column
    out = []
    add = out.append

    add("")
    add(P("forge", "bold") + " — a model-agnostic agentic runtime. Any model, frontier or a small")
    add("local one, becomes a capable agent: the intelligence lives in the harness,")
    add("not the weights.")
    add("")
    # The test suite pins "usage: forge" — and it's the line people actually scan for.
    add(P("usage: forge", "bold") + " [options] <command> [args]")
    add("")
    for cmd, desc in _USAGE_EXAMPLES:
        add("  " + P(f"{cmd:<{W}}", "cyan") + desc)

    for title, rows in _COMMAND_GROUPS:
        add("")
        add(P(title, "bold"))
        for name, desc in rows:
            add("  " + P(f"{name:<{W}}", "cyan") + desc)

    add("")
    add(P("Options", "bold"))
    for flag, desc in _OPTIONS:
        add("  " + P(f"{flag:<{W}}", "dim") + desc)

    add("")
    add(P("forge <command> --help", "dim") + P("  for any command in detail.", "dim"))
    add("")
    return "\n".join(out) + "\n"


def main():
    # Frozen-binary re-invocation hook (see _daemon_launch_cmd): `forge daemon <model>
    # [interval]` runs the autopilot loop that `forge up` spawns. Intercepted BEFORE
    # argparse so it never becomes a user-facing subcommand — the exact counterpart of
    # `python -m forge.daemon <model> <interval>` from source (forge/daemon.py __main__).
    if len(sys.argv) >= 3 and sys.argv[1] == "daemon":
        from .daemon import Forged
        Forged(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20).run()
        return
    from . import render as _render
    _render.enable_vt()            # ANSI/VT on the Windows console; no-op on POSIX
    ap = argparse.ArgumentParser(prog="forge")
    ap.format_help = _help_text          # -h/--help routes through print_help → format_help
    # default LOCAL LADDER comes from ~/.forge/config.json (written by `forge setup`)
    ap.add_argument("--model", default=None,
                    help="a model, or a comma-separated ladder cheap→strong; overrides the config")
    ap.add_argument("--dir", default=os.getcwd(), help="where to work (default: this directory)")
    ap.add_argument("--name", default=None, help="name this session — the fleet sees it")
    ap.add_argument("--verbose", action="store_true", help="show the full step trace")
    ap.add_argument("--resume", metavar="SID|last", default=None,
                    help="resume a prior session from its transcript ('last' = newest for this dir)")
    ap.add_argument("--version", action="version", version=f"forge {__version__}")
    # metavar, or argparse spells all 16 command names into every usage line —
    # including the one printed on an error, where it buries the actual message.
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")

    p_run = sub.add_parser("run", help="one-shot task")
    p_run.add_argument("task")
    p_run.add_argument("--max-steps", type=int, default=40)

    p_team = sub.add_parser("team", help="swarm: plan a goal into a task DAG, run each task in an isolated worktree, merge only what verifies")
    p_team.add_argument("goal")
    p_team.add_argument("--max-steps", type=int, default=40)

    p_dex = sub.add_parser("dex", help="your model-card collection (a model becomes a collectible card as you use it); `forge dex <model>` shows one card")
    p_dex.add_argument("model", nargs="?", default=None)

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

    p_ex = sub.add_parser("export", help="export a session's event log as process-mining data (CSV/JSON/OCEL), secrets redacted")
    p_ex.add_argument("sid", nargs="?", default="last", help="session id (or 'last', the default)")
    p_ex.add_argument("--format", choices=["csv", "json", "ocel"], default="json", help="output format (default: json)")
    p_ex.add_argument("--out", metavar="FILE", help="write to FILE (default: print to stdout)")

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

    p_md = sub.add_parser("models", help="curated model catalog: recipes forge has actually run, checked against this machine")
    p_md.add_argument("action", nargs="?", default="list", choices=["list", "show", "use", "stop"])
    p_md.add_argument("name", nargs="?", help="entry name (for `show` / `use` / `stop`)")
    p_md.add_argument("--all", action="store_true", help="scan the FULL downloadable catalog (Ollama + HF GGUF + MLX), not just the curated spread")
    p_md.add_argument("--refresh", action="store_true", help="re-fetch the catalog cache (with --all)")

    p_setup = sub.add_parser("setup", help="detect hardware, choose an engine, pull/point at models, write config")
    p_setup.add_argument("--auto", action="store_true", help="no prompts (Ollama, RAM-sized ladder)")
    p_setup.add_argument("--engine", help="ollama | vllm | llamacpp | mlx | lmstudio | tgi | sglang | openai | anthropic")
    p_setup.add_argument("--url", help="base URL for an OpenAI-compatible engine")
    p_setup.add_argument("--api-key", dest="api_key", help="API key for the engine (if needed)")
    p_setup.add_argument("--models", help="comma-separated model names, cheap→strong (for non-ollama engines)")

    args = ap.parse_args()
    from .models_cmd import cmd_models
    dispatch = {"run": cmd_run, "team": cmd_team, "dex": cmd_dex, "status": cmd_status, "send": cmd_send, "up": cmd_up,
                "down": cmd_down, "receipts": cmd_receipts, "learnings": cmd_learnings,
                "forget": cmd_forget, "trace": cmd_trace, "corpus": cmd_corpus, "export": cmd_export, "bench": cmd_bench, "replay": cmd_replay,
                "passport": cmd_passport, "models": cmd_models}
    try:
        if args.cmd == "setup":
            from . import setup as setupmod
            models = [m.strip() for m in args.models.split(",")] if args.models else None
            sys.exit(setupmod.run(auto=args.auto, engine=args.engine, url=args.url,
                                  api_key=args.api_key, models=models))
        (dispatch.get(args.cmd) or cmd_chat)(args)
    except ForgeError as e:
        print(f"✗ {e}", file=sys.stderr); sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        # Ctrl-C / Ctrl-D anywhere — during a prompt, a pull, a long run — exits clean,
        # never a traceback. 130 is the shell convention for SIGINT. (The chat REPL
        # swallows its own Ctrl-C to cancel a line, so it never reaches here.)
        print()
        sys.exit(130)


if __name__ == "__main__":
    main()
