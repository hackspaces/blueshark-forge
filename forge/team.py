"""P9.2 forge team — the first swarm slice.

A planner (the strongest ladder rung) decomposes a goal into a task DAG; each task
runs as a worker agent in its OWN git worktree branch; a task's changes merge into an
integration branch ONLY after the deterministic verify gate passes in that worktree;
tasks run in dependency order. The parallel scheduler + live board is phase 2.

Design (grounded in Anthropic's multi-agent engineering — see the swarm-design notes):
the PLAN lives in code, not the model's head — the harness owns decomposition validation,
worktree isolation, the verify-before-merge gate, and merge ordering, all of which a small
model cannot do. Each worker only ever faces a narrow, single-file-scope task — the regime
where 7B models are already reliable. Worktrees keep `.git`, so git-dependent test suites
verify correctly (the rsync-copy path drops .git and false-REFUTEs).
"""
import json
import os
import shutil
import subprocess

from .agent import Agent
from .fleet import detect_test_cmd, runner_missing
from . import session as sessionmod

# Grammar-forced planner schema: a task DAG. Kept flat and free of nested constraints so
# Ollama's `format` grammar can enforce it on a 7B (the same lesson as the MCP `args`).
TEAM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"subtasks": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "prompt": {"type": "string", "description": "the full instruction for the worker"},
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "exact files this task touches — must not overlap another task"},
            "depends_on": {"type": "array", "items": {"type": "string"},
                           "description": "task ids that must finish first; empty = runs in parallel"},
        },
        "required": ["id", "title", "prompt", "files", "depends_on"]}}},
    "required": ["subtasks"],
}

PLANNER_SYSTEM = (
    "You are a PLANNER. Decompose the goal into the SMALLEST set of INDEPENDENT subtasks. "
    "RULES: (1) each subtask lists the exact files it will touch in `files`; (2) two subtasks "
    "must NEVER list the same file — non-overlapping scopes only; (3) put an id in `depends_on` "
    "ONLY when a subtask genuinely needs another's output first; (4) independent work has an "
    "empty depends_on so it can run in parallel. Emit the task DAG as JSON.")


def validate_dag(dag):
    """Return a list of structural problems (empty = a usable DAG): overlapping file
    scopes, dependencies on unknown ids, or a cycle. The grammar forces the SHAPE; the
    harness must validate the SEMANTICS — the planner-quality risk the roadmap flags."""
    subs = dag.get("subtasks", []) if isinstance(dag, dict) else []
    issues = []
    if not subs:
        return ["planner returned no subtasks"]
    ids = [s.get("id") for s in subs]
    if len(set(ids)) != len(ids):
        issues.append("duplicate task ids")
    owner = {}
    for s in subs:
        for f in s.get("files", []):
            if f in owner:
                issues.append(f"file {f!r} claimed by both {owner[f]!r} and {s.get('id')!r}")
            owner[f] = s.get("id")
    idset = set(ids)
    for s in subs:
        for d in s.get("depends_on", []):
            if d not in idset:
                issues.append(f"task {s.get('id')!r} depends on unknown {d!r}")
    if not _topo_order(subs):
        issues.append("dependency cycle")
    return issues


def _topo_order(subs):
    """Kahn's algorithm. Returns the subtasks in an order where every dependency precedes
    its dependents, or None if there is a cycle."""
    by_id = {s["id"]: s for s in subs if s.get("id")}
    indeg = {i: 0 for i in by_id}
    adj = {i: [] for i in by_id}
    for s in subs:
        for d in s.get("depends_on", []):
            if d in by_id and s["id"] in indeg:
                indeg[s["id"]] += 1
                adj[d].append(s["id"])
    ready = [i for i in by_id if indeg[i] == 0]
    order = []
    while ready:
        n = ready.pop(0)
        order.append(by_id[n])
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    return order if len(order) == len(by_id) else None


def plan(goal, files, planner, repair=True):
    """Ask the planner backend for a validated task DAG. One repair retry that shows the
    planner its own structural errors — the grammar can't express 'non-overlapping', so the
    harness teaches it by feedback (the same shape as the loop's other corrections)."""
    user = f"GOAL:\n{goal}\n\nPROJECT FILES:\n" + "\n".join(files)
    messages = [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user}]
    raw = planner.chat(messages, schema=TEAM_SCHEMA)
    dag = _parse(raw)
    issues = validate_dag(dag)
    if issues and repair:
        messages.append({"role": "assistant", "content": raw if isinstance(raw, str) else json.dumps(dag)})
        messages.append({"role": "user", "content":
                         "That plan is invalid: " + "; ".join(issues) +
                         ". Fix it — non-overlapping file scopes, real dependency ids, no cycles."})
        dag = _parse(planner.chat(messages, schema=TEAM_SCHEMA))
        issues = validate_dag(dag)
    if issues:
        raise TeamError("planner could not produce a usable DAG: " + "; ".join(issues))
    return dag


def _parse(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


class TeamError(Exception):
    pass


def _git(cwd, *args, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise TeamError(f"git {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r


def _repo_files(cwd):
    r = _git(cwd, "ls-files", check=False)
    return [f for f in r.stdout.splitlines() if f] if r.returncode == 0 else []


def _verify(cwd, files=None):
    """The deterministic merge gate, run IN the worktree (which already holds the worker's
    changes). Returns (ok, detail). `files` (abs paths this task touched) SCOPES a pytest
    run to the nearest test target, so an independent task isn't refuted by another task's
    not-yet-merged red test. No suite found OR the runner is absent → unverified, which is
    accepted — a task with no runnable check is not a failure (forge's run_tests policy)."""
    cmd = detect_test_cmd(cwd, files=files)
    if not cmd:
        return True, "no suite (unverified)"
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if runner_missing(out):
        return True, f"runner unavailable, unverified ({cmd})"
    tail = out.strip().splitlines()[-1:] or [""]
    return r.returncode == 0, f"$ {cmd} → {'pass' if r.returncode == 0 else 'FAIL'} · {tail[0][:80]}"


def run_team(goal, cwd, ladder, planner=None, on_event=None, max_steps=40,
             agent_factory=None, retries=1):
    """Run the team slice end to end and return a report dict. `planner` defaults to the
    strongest ladder rung. `agent_factory(ladder, session, worktree)` is injectable so tests
    drive scripted workers without a real model; the default builds a real Agent."""
    emit = on_event or (lambda *a, **k: None)
    ladder = ladder if isinstance(ladder, list) else [ladder]
    planner = planner or ladder[-1]
    agent_factory = agent_factory or _default_agent

    _git(cwd, "rev-parse", "--is-inside-work-tree")            # fail early if not a git repo
    files = _repo_files(cwd)
    emit("team_plan_start", goal=goal, files=len(files))
    dag = plan(goal, files, planner)
    order = _topo_order(dag["subtasks"])
    emit("team_planned", subtasks=[s["id"] for s in order])

    integration = "forge-team/integration"
    base = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git(cwd, "branch", "-f", integration, "HEAD")            # integration starts at HEAD
    wt_root = os.path.join(cwd, ".forge-team")
    # A dedicated worktree checked out ON the integration branch: every verified task
    # branch is merged INTO it here, so the user's own checkout (cwd/HEAD) is never
    # touched and the integration branch actually advances between tasks (a dependent
    # task branches from the already-merged result).
    integ_wt = os.path.join(wt_root, "_integration")
    _git(cwd, "worktree", "add", "-f", integ_wt, integration)
    results = {}
    try:
        for task in order:
            tid = task["id"]
            if any(results.get(d, {}).get("status") != "merged" for d in task.get("depends_on", [])):
                results[tid] = {"status": "skipped", "detail": "a dependency did not merge"}
                emit("team_task", id=tid, status="skipped")
                continue
            results[tid] = _run_one(task, cwd, wt_root, integration, integ_wt, ladder,
                                    max_steps, agent_factory, emit, retries)
        final_ok, final_detail = _verify(integ_wt)             # whole suite on the merged result
    finally:
        _cleanup_worktrees(cwd, wt_root)

    merged = [t for t, r in results.items() if r["status"] == "merged"]
    emit("team_done", merged=len(merged), total=len(order))
    return {"goal": goal, "integration_branch": integration, "base": base,
            "results": results, "merged": merged,
            "final": {"ok": final_ok, "detail": final_detail}}


def _run_one(task, cwd, wt_root, integration, integ_wt, ladder, max_steps, agent_factory, emit, retries):
    tid = task["id"]
    branch = f"forge-team/{tid}"
    wt = os.path.join(wt_root, tid)
    emit("team_task", id=tid, status="start", title=task.get("title", ""))
    _git(cwd, "worktree", "add", "-f", "-B", branch, wt, integration)   # branch from integration's CURRENT tip
    try:
        prompt = _worker_prompt(task)
        attempt, verdict = 0, (False, "not run")
        while attempt <= retries:
            agent = agent_factory(ladder, wt, task)
            agent.send(prompt if attempt == 0 else
                       f"{prompt}\n\n[verify] the previous attempt FAILED: {verdict[1]}. Fix it.")
            if not _git(wt, "status", "--porcelain", check=False).stdout.strip():
                verdict = (False, "worker made no changes")
                attempt += 1
                continue
            _git(wt, "add", "-A")
            _git(wt, "commit", "-m", f"forge-team {tid}: {task.get('title','')[:60]}", check=False)
            verdict = _verify(wt, files=[os.path.join(wt, f) for f in task.get("files", [])])
            if verdict[0]:
                break
            attempt += 1
        if not verdict[0]:
            emit("team_task", id=tid, status="refuted", detail=verdict[1])
            return {"status": "failed", "detail": verdict[1]}
        # CONFIRMED → merge the branch INTO the integration worktree (no-ff keeps the task
        # boundary), so the integration branch advances and cwd/HEAD stay untouched.
        m = _git(integ_wt, "merge", "--no-ff", "-m", f"forge-team merge {tid}", branch, check=False)
        if m.returncode != 0:
            _git(integ_wt, "merge", "--abort", check=False)
            emit("team_task", id=tid, status="conflict")
            return {"status": "conflict", "detail": m.stdout.strip()[:200]}
        emit("team_task", id=tid, status="merged", detail=verdict[1])
        return {"status": "merged", "detail": verdict[1]}
    finally:
        _git(cwd, "worktree", "remove", "--force", wt, check=False)


def _cleanup_worktrees(cwd, wt_root):
    if os.path.isdir(wt_root):
        shutil.rmtree(wt_root, ignore_errors=True)
    _git(cwd, "worktree", "prune", check=False)


def _worker_prompt(task):
    files = ", ".join(task.get("files", [])) or "(the files this task needs)"
    return (f"{task.get('prompt', task.get('title',''))}\n\n"
            f"Scope: work ONLY within these files: {files}. When done and the change is "
            "correct, run the tests to verify, then say done.")


def _default_agent(ladder, worktree, task):
    from .__main__ import _workspace_ctx, _ctx_budget
    sess = sessionmod.EphemeralSession(worktree, ladder[0].name if ladder else "team")
    return Agent(ladder, sess, max_steps=40, autonomous=True,
                 workspace=_workspace_ctx(worktree, _ctx_budget(ladder[0])),
                 allowed=["bash", "read_file", "write_file", "edit_file",
                          "list_files", "grep", "glob", "run_tests", "say"])
