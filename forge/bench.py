"""forge bench — harness-lift eval with per-lever ablation (P3.2).

Measures HARNESS-LIFT: the SAME model run *bare* (every scaffolding lever off) vs
with the FULL forge harness, plus per-lever ablation (full-minus-one) to show
which lever earned its complexity. Every task runs through the real Agent.send()
loop in a throwaway git workspace; the verdict is an exit code (verify.sh, or the
detected test command) — the same exit-code-is-truth philosophy as fleet.verify.

HONEST FRAMING (read this before quoting a number).
    "bare" turns the `schema` lever OFF, but the agent loop STILL demands one JSON
    action object per step and bails after 5 malformed replies. So a bare
    pass-rate substantially measures FORMAT COMPLIANCE — whether the raw weights
    can hold the action-JSON contract without constrained decoding — not pure task
    skill. That IS the headline harness-lift story ("qwen3:4b — 14% bare, 61% with
    forge"): the scaffolding is what makes a small model usable at all. Publish it
    that way, honestly, rather than dressing it up as a clean capability delta.

Task fixtures live under bench/ (repo root). Each task is a directory:
    bench/<task>/prompt.txt    the task text handed to the agent   (required)
    bench/<task>/setup.sh      optional — run in the workspace before the agent
    bench/<task>/verify.sh     optional — exit 0 = pass; the verdict

run_task copies a fixture into a fresh tempdir, git-inits it, and runs the real
loop against an INJECTED backend/ladder — so tests never touch a real model. It
returns a metrics row; the CLI appends rows to ~/.forge/bench/results.jsonl and
`--report` prints the lift + ablation tables.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time

from .agent import Agent, ALL_LEVERS, _cmd_missing
from .session import EphemeralSession

RESULTS = os.path.expanduser("~/.forge/bench/results.jsonl")

# Ablation flags -> the single lever each removes from the full harness.
ABLATIONS = {
    "no_compact": "compaction",
    "no_loop_detect": "loop_detect",
    "no_read_gate": "read_gate",
    "single_rung": "escalation",
}


def bench_dir():
    """The repo-root bench/ fixtures directory (…/forge/bench)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")


def list_tasks(root=None):
    """Names of the task fixtures under bench/ (each a dir with prompt.txt)."""
    root = root or bench_dir()
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "prompt.txt")))


def _backend_name(backend):
    b = backend[0] if isinstance(backend, list) else backend
    return getattr(b, "name", "model")


def _sh(script, cwd, timeout=120):
    """Run a fixture bash script; return (exit_code, combined_output)."""
    try:
        p = subprocess.run(["bash", script], cwd=cwd, timeout=timeout,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode, (p.stdout or b"").decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "(timed out)"
    except OSError as e:
        return 127, str(e)


def _verdict(work):
    """Grade the finished workspace. Returns True (pass), False (fail), or None
    (no verdict — a missing/uninstalled test command is NOT a failure)."""
    verify = os.path.join(work, "verify.sh")
    if os.path.isfile(verify):
        rc, _ = _sh(verify, work)
        return rc == 0
    # no verify.sh — fall back to the project's detected test command
    from . import fleet, tools
    try:
        cmd = fleet.detect_test_cmd(work)
    except Exception:
        cmd = None
    if not cmd:
        return None
    out, ok = tools._run(cmd, work, timeout=tools.BASH_TIMEOUT * 3)
    if _cmd_missing(out):
        return None                      # command not installed here → no verdict
    return ok


def run_task(task_dir, backend, levers, max_steps=40, model=None):
    """Run ONE fixture through the real Agent.send loop against an injected
    `backend` (single backend or ladder list) with `levers` active. Returns a
    result row: {model, levers, task, pass, steps, seconds, escalations,
    malformed, loops, obs_fail}. `pass` is True/False, or None when there is no
    verdict. NO real model is required — inject a fake/ScriptBackend in tests."""
    task = os.path.basename(os.path.normpath(task_dir))
    model = model or _backend_name(backend)
    prompt_path = os.path.join(task_dir, "prompt.txt")
    with open(prompt_path, errors="replace") as f:
        prompt = f.read().strip()

    work = tempfile.mkdtemp(prefix="forge-bench-")
    counts = {"steps": 0, "malformed": 0, "loops": 0, "escalations": 0, "obs_fail": 0}

    def on_event(kind, **k):
        if kind == "action":
            counts["steps"] += 1
        elif kind == "malformed":
            counts["malformed"] += 1
        elif kind == "loop":
            counts["loops"] += 1
        elif kind == "escalate":
            counts["escalations"] += 1
        elif kind == "observation" and k.get("ok") is False:
            counts["obs_fail"] += 1

    try:
        shutil.copytree(task_dir, work, dirs_exist_ok=True)
        try:
            subprocess.run(["git", "init", "-q"], cwd=work,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass          # git absent → the task still runs; verdict is its verify.sh exit code

        setup = os.path.join(work, "setup.sh")
        if os.path.isfile(setup):
            _sh(setup, work)

        session = EphemeralSession(work, model)
        agent = Agent(backend, session, max_steps=max_steps, autonomous=True,
                      on_event=on_event, levers=levers)
        t0 = time.monotonic()
        try:
            agent.send(prompt)
        except Exception:
            pass                          # a crashed run is graded by the verdict below
        seconds = round(time.monotonic() - t0, 3)

        passed = _verdict(work)
        return {
            "model": model,
            "levers": sorted(levers),
            "task": task,
            "pass": passed,
            "steps": counts["steps"],
            "seconds": seconds,
            "escalations": counts["escalations"],
            "malformed": counts["malformed"],
            "loops": counts["loops"],
            "obs_fail": counts["obs_fail"],
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def append_results(rows, path=RESULTS):
    """Append result rows to the JSONL log (creates ~/.forge/bench/ if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---- reporting ----------------------------------------------------------------

def config_label(levers):
    """A short name for a lever configuration: 'bare', 'full', or 'full-<lever>'
    for a single-lever ablation; otherwise the sorted lever list joined by '+'."""
    s = frozenset(levers)
    if not s:
        return "bare"
    if s == ALL_LEVERS:
        return "full"
    missing = ALL_LEVERS - s
    if len(missing) == 1 and not (s - ALL_LEVERS):
        return "full-" + next(iter(missing))
    return "+".join(sorted(s))


def _rate(rows):
    """(passed, graded, fraction) over rows with a real verdict; None if none."""
    graded = [r for r in rows if r.get("pass") is not None]
    if not graded:
        return None
    passed = sum(1 for r in graded if r["pass"])
    return passed, len(graded), passed / len(graded)


def _cell(rate):
    if rate is None:
        return "-"
    passed, total, frac = rate
    return f"{passed}/{total} {100 * frac:.0f}%"


def report(rows):
    """Format the harness-lift table (per model: bare vs full pass-rate) plus a
    per-lever ablation table (full minus one lever), as a printable string."""
    out = []
    out.append("HARNESS-LIFT  (bare = schema lever off; the loop still demands JSON, "
               "bails after 5 malformed — so bare largely measures format compliance)")
    out.append("")
    out.append(f"  {'model':<24}  {'bare':>12}  {'full':>12}  {'lift':>8}")
    out.append("  " + "-" * 62)
    models = sorted({r["model"] for r in rows})
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        bare = _rate([r for r in mrows if config_label(r["levers"]) == "bare"])
        full = _rate([r for r in mrows if config_label(r["levers"]) == "full"])
        lift = "-"
        if bare is not None and full is not None:
            lift = f"{100 * (full[2] - bare[2]):+.0f}pts"
        out.append(f"  {m:<24}  {_cell(bare):>12}  {_cell(full):>12}  {lift:>8}")

    # ablation table: any config that is full-minus-exactly-one-lever
    abl = [r for r in rows if config_label(r["levers"]).startswith("full-")]
    if abl:
        out.append("")
        out.append("PER-LEVER ABLATION  (full harness minus one lever)")
        out.append("")
        out.append(f"  {'model':<24}  {'lever removed':<16}  {'pass':>12}  {'vs full':>8}")
        out.append("  " + "-" * 66)
        for m in models:
            full = _rate([r for r in rows if r["model"] == m
                          and config_label(r["levers"]) == "full"])
            labels = sorted({config_label(r["levers"]) for r in abl if r["model"] == m})
            for lab in labels:
                lever = lab[len("full-"):]
                rate = _rate([r for r in rows if r["model"] == m
                              and config_label(r["levers"]) == lab])
                delta = "-"
                if rate is not None and full is not None:
                    delta = f"{100 * (rate[2] - full[2]):+.0f}pts"
                out.append(f"  {m:<24}  {lever:<16}  {_cell(rate):>12}  {delta:>8}")
    return "\n".join(out)


def configs_for(args):
    """The list of (label, levers-frozenset) configurations to run, from the CLI
    flags. Default (no config flag) = the headline bare-vs-full measurement."""
    configs = [("full", ALL_LEVERS)]
    picked = False
    if getattr(args, "bare", False):
        configs.insert(0, ("bare", frozenset()))
        picked = True
    for flag, lever in ABLATIONS.items():
        if getattr(args, flag, False):
            configs.append(("full-" + lever, ALL_LEVERS - {lever}))
            picked = True
    if not picked:                       # no selector → measure bare vs full
        configs.insert(0, ("bare", frozenset()))
    # dedup by lever-set, preserve order
    seen, uniq = set(), []
    for label, levers in configs:
        key = frozenset(levers)
        if key not in seen:
            seen.add(key)
            uniq.append((label, levers))
    return uniq
