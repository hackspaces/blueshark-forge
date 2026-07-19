"""forge company — run a goal through a small organization of forge sessions.

Slice 1, the company core. A MANAGER (strong rung) decomposes a goal into work items,
each with its own verify condition (MBO cascade); WORKERS (cheap rung) execute them as
real forge sessions in isolated worktrees; the existing TRUST verifier audits every
done-claim, structurally separate from the sessions it judges (principal-agent monitoring);
the manager rolls up a STATUS summary (compression upward — the CEO sees the roll-up, never
raw worker output); a failed item is re-dispatched once, then escalated to the human.

The thesis: management science is a century of R&D on coordinating bounded-rationality
processors (Simon), and an LLM session with a finite context window is one — literally. forge
already grew four of Beer's five viable-system functions (S1 workers, S2 COORDINATE, S3 TRUST,
S4 LEARN); the charter + the human as CEO complete it with S5 policy. What no prior art has:
agents doing REAL verified work on REAL repos, every done-claim audited by an independent
verifier, every decision a receipt — fully local, which is what makes a hierarchy's many
inference hops economical.

The board, receipts, and STATUS.md are the only durable state (Weber: files as memory); this
module writes them and `company status` / the future TUI are pure renderers over that truth.
"""
import json
import os
import time

from . import team
from .authority import AuthorityLevel

COMPANY_DIR = os.path.join(os.path.expanduser("~/.forge"), "company")

# Grammar-forced manager decomposition: a flat list of work items, each carrying its OWN
# verify condition (MBO made hierarchical) and its assignee. Kept flat + free of nested
# constraints so a small manager model can satisfy it (same lesson as team/MCP schemas).
MANAGER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"work_items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "brief": {"type": "string", "description": "the full instruction for the worker"},
            "assignee": {"type": "string", "description": "a worker role id from the charter"},
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "files this item touches — must not overlap another item"},
            "verify": {"type": "string",
                       "description": "the concrete check that proves this item done (a shell command)"},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "title", "brief", "assignee", "files", "verify", "depends_on"]}}},
    "required": ["work_items"],
}


class CompanyError(Exception):
    pass


# ---- charter (Weber: offices independent of occupants) ----------------------
def _company_path(name):
    return os.path.join(COMPANY_DIR, name)


def create_charter(name, manager_rung, worker_rungs, verifier_model=None,
                   manager_prompt="", worker_prompt=""):
    """Write a company charter: one manager (strong rung), N workers (cheap rung), and the
    verifier (the existing forged TRUST, read-only). A role is an OFFICE — its model rung can
    be swapped later without changing the job. Returns the charter dict."""
    roles = {"manager": {
        "title": "Manager", "rung": manager_rung, "authority": "operator",
        "prompt": manager_prompt or _DEFAULT_MANAGER_PROMPT, "reports_to": None}}
    for i, rung in enumerate(worker_rungs, 1):
        roles[f"worker-{i}"] = {
            "title": f"Worker {i}", "rung": rung, "authority": "contribute",
            "prompt": worker_prompt or _DEFAULT_WORKER_PROMPT, "reports_to": "manager"}
    charter = {"name": name, "roles": roles,
               "verifier": {"model": verifier_model or worker_rungs[-1] if worker_rungs else manager_rung,
                            "authority": "observe"},
               "version": 1}
    d = _company_path(name)
    os.makedirs(os.path.join(d, "board"), exist_ok=True)
    with open(os.path.join(d, "company.json"), "w") as f:
        json.dump(charter, f, indent=2)
    return charter


def load_charter(name):
    try:
        with open(os.path.join(_company_path(name), "company.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        raise CompanyError(f"no company named {name!r} — create one with `forge company new {name}`")


def workers(charter):
    return [r for r, spec in charter["roles"].items() if spec.get("reports_to") == "manager"]


# ---- routing law (Fayol: unity of command) ----------------------------------
def can_message(charter, sender, recipient):
    """The router law, enforced in code not prompt: a message is allowed only along real org
    lines. A worker's inbox accepts only its manager and its team peers (same reports_to); a
    worker may report UP to its manager; skip-level (a worker → the CEO, or the CEO → a worker)
    is refused — escalation goes through the chain. The manager is the single point of contact
    with the human (compression upward). Prevents conflicting orders from two bosses."""
    roles = charter["roles"]
    if recipient not in roles and recipient not in ("ceo", "manager"):
        return False
    if sender == "ceo":
        return recipient == "manager"            # the CEO steers only the manager
    if recipient == "ceo":
        return sender == "manager"               # only the manager reports to the CEO
    s_mgr = roles.get(sender, {}).get("reports_to")
    r_mgr = roles.get(recipient, {}).get("reports_to")
    if recipient == "manager":
        return s_mgr == "manager"                # a worker reports up to its manager
    if sender == "manager":
        return r_mgr == "manager"                # the manager directs its workers
    return s_mgr is not None and s_mgr == r_mgr  # peers on the same team


# ---- the board (durable work-item state, files as memory) -------------------
def _board_dir(name):
    return os.path.join(_company_path(name), "board")


def write_item(name, item):
    with open(os.path.join(_board_dir(name), f"{item['id']}.json"), "w") as f:
        json.dump(item, f, indent=2)


def read_board(name):
    d = _board_dir(name)
    items = []
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.endswith(".json"):
            try:
                with open(os.path.join(d, fn)) as f:
                    items.append(json.load(f))
            except (OSError, ValueError):
                pass
    return items


# ---- decomposition (the manager plans the board) ----------------------------
def decompose(goal, charter, planner, repair=True):
    """Ask the manager to break the goal into assigned, verify-carrying work items. Validated
    like a team DAG PLUS company invariants: every assignee is a real worker role, and each
    item carries a concrete verify. One repair retry shows the manager its own errors."""
    ids = workers(charter)
    sys_prompt = (charter["roles"]["manager"]["prompt"]
                  + f"\n\nWorkers you can assign to: {', '.join(ids)}. Every work item MUST name "
                  "one of them as `assignee`, MUST carry a concrete `verify` shell command that "
                  "proves it done, and two items must not touch the same file.")
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"GOAL:\n{goal}"}]
    dag = _plan_once(planner, messages)
    issues = _validate(dag, ids)
    if issues and repair:
        messages.append({"role": "assistant", "content": json.dumps(dag)})
        messages.append({"role": "user", "content": "That plan is invalid: " + "; ".join(issues)
                         + ". Fix it — real assignees, a concrete verify per item, no file overlap."})
        dag = _plan_once(planner, messages)
        issues = _validate(dag, ids)
    if issues:
        raise CompanyError("manager could not produce a usable plan: " + "; ".join(issues))
    return dag["work_items"]


def _plan_once(planner, messages):
    raw = planner.chat(messages, schema=MANAGER_SCHEMA)
    try:
        return raw if isinstance(raw, dict) else json.loads(raw)
    except (TypeError, ValueError):
        return {"work_items": []}


def _validate(dag, worker_ids):
    items = dag.get("work_items", []) if isinstance(dag, dict) else []
    issues = []
    if not items:
        return ["manager returned no work items"]
    ids = [it.get("id") for it in items]
    if len(set(ids)) != len(ids):
        issues.append("duplicate item ids")
    owner = {}
    for it in items:
        if it.get("assignee") not in worker_ids:
            issues.append(f"item {it.get('id')!r} assigned to unknown worker {it.get('assignee')!r}")
        if not (it.get("verify") or "").strip():
            issues.append(f"item {it.get('id')!r} has no verify condition")
        for fpath in it.get("files", []):
            if fpath in owner:
                issues.append(f"file {fpath!r} claimed by {owner[fpath]!r} and {it.get('id')!r}")
            owner[fpath] = it.get("id")
    # reuse team's DAG shape checks (unknown deps / cycles)
    issues += [i for i in team.validate_dag({"subtasks": [
        {"id": it.get("id"), "files": it.get("files", []),
         "depends_on": it.get("depends_on", [])} for it in items]})
        if "no subtasks" not in i]
    return issues


# ---- the run (decompose → workers execute → TRUST audits → roll up) ---------
def run(name, goal, cwd, ladder_for, on_event=None, max_steps=40,
        agent_factory=None, verifier=None, planner_override=None):
    """Run a goal through the company. `ladder_for(rung)` resolves a role's model rung to a
    backend ladder; `agent_factory(ladder, worktree, item)` builds a worker (injectable for
    tests); `verifier(cwd, files)->(ok,detail)` is the independent TRUST audit (defaults to
    the deterministic verify gate). Returns a report; writes the board + STATUS.md."""
    emit = on_event or (lambda *a, **k: None)
    charter = load_charter(name)
    agent_factory = agent_factory or team._default_agent
    verifier = verifier or (lambda wt, files: team._verify(wt, files=files))
    if planner_override is not None:
        planner = planner_override
    else:
        manager_ladder = ladder_for(charter["roles"]["manager"]["rung"])
        planner = manager_ladder[-1] if isinstance(manager_ladder, list) else manager_ladder

    team._git(cwd, "rev-parse", "--is-inside-work-tree")
    emit("company_plan", goal=goal)
    items = decompose(goal, charter, planner)
    order = team._topo_order([{"id": it["id"], "depends_on": it.get("depends_on", [])} for it in items])
    by_id = {it["id"]: it for it in items}
    for it in items:
        it.update(state="queued", attempts=0, result="")
        write_item(name, it)
    emit("company_board", items=[it["id"] for it in items])

    integration = f"forge-company/{name}/integration"
    base = team._git(cwd, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    team._git(cwd, "branch", "-f", integration, "HEAD")
    wt_root = os.path.join(cwd, ".forge-company")
    integ_wt = os.path.join(wt_root, "_integration")
    team._git(cwd, "worktree", "add", "-f", integ_wt, integration)
    try:
        for stub in order:
            it = by_id[stub["id"]]
            if any(by_id[d].get("state") != "verified" for d in it.get("depends_on", []) if d in by_id):
                it.update(state="blocked", result="a dependency did not verify")
                write_item(name, it); emit("company_item", id=it["id"], state="blocked")
                continue
            _run_item(name, it, charter, cwd, wt_root, integration, integ_wt,
                      ladder_for, agent_factory, verifier, emit, max_steps)
        final_ok, final_detail = verifier(integ_wt, None)
    finally:
        team._cleanup_worktrees(cwd, wt_root)

    status = roll_up(name, goal, base, integration, final_ok, final_detail)
    verified = [it["id"] for it in read_board(name) if it.get("state") == "verified"]
    escalated = [it["id"] for it in read_board(name) if it.get("state") == "escalated"]
    emit("company_done", verified=len(verified), escalated=len(escalated))
    return {"company": name, "goal": goal, "integration": integration, "base": base,
            "verified": verified, "escalated": escalated,
            "final": {"ok": final_ok, "detail": final_detail}, "status_path": _status_path(name)}


def _run_item(name, it, charter, cwd, wt_root, integration, integ_wt,
              ladder_for, agent_factory, verifier, emit, max_steps, retries=1):
    """One work item: a worker executes it in an isolated worktree; the item's OWN verify
    condition + the independent verifier audit the result; a rejection is re-dispatched once,
    then the item is ESCALATED to the human (never silently dropped). A verified item's branch
    merges into the integration branch."""
    tid = it["id"]
    branch = f"forge-company/{name}/item/{tid}"      # sibling namespace to .../integration (no ref collision)
    wt = os.path.join(wt_root, tid)
    ladder = ladder_for(charter["roles"][it["assignee"]]["rung"])
    it.update(state="running"); write_item(name, it)
    emit("company_item", id=tid, state="running", assignee=it["assignee"], title=it.get("title", ""))
    team._git(cwd, "worktree", "add", "-f", "-B", branch, wt, integration)
    try:
        attempt, verdict = 0, (False, "not run")
        while attempt <= retries:
            worker = agent_factory(ladder, wt, it)
            brief = _worker_brief(it, attempt, verdict)
            worker.send(brief)
            if not team._git(wt, "status", "--porcelain", check=False).stdout.strip():
                verdict = (False, "worker made no changes"); attempt += 1; continue
            team._git(wt, "add", "-A")
            team._git(wt, "commit", "-m", f"company {tid}: {it.get('title','')[:60]}", check=False)
            # the item's OWN verify condition (MBO), then the independent TRUST audit
            item_ok, item_detail = _run_verify(it.get("verify", ""), wt)
            audit_ok, audit_detail = verifier(wt, [os.path.join(wt, f) for f in it.get("files", [])])
            if item_ok and audit_ok:
                verdict = (True, f"{item_detail}; audit {audit_detail}"); break
            verdict = (False, item_detail if not item_ok else f"TRUST rejected: {audit_detail}")
            attempt += 1
        it["attempts"] = attempt + 1
        if not verdict[0]:
            it.update(state="escalated", result=verdict[1]); write_item(name, it)
            emit("company_item", id=tid, state="escalated", detail=verdict[1])
            return
        m = team._git(integ_wt, "merge", "--no-ff", "-m", f"company merge {tid}", branch, check=False)
        if m.returncode != 0:
            team._git(integ_wt, "merge", "--abort", check=False)
            it.update(state="escalated", result="merge conflict"); write_item(name, it)
            emit("company_item", id=tid, state="escalated", detail="merge conflict"); return
        it.update(state="verified", result=verdict[1]); write_item(name, it)
        emit("company_item", id=tid, state="verified", detail=verdict[1])
    finally:
        team._git(cwd, "worktree", "remove", "--force", wt, check=False)


def _run_verify(cmd, cwd):
    """Run an item's own verify condition (a shell command). No command → unverified-accept,
    forge's own run_tests policy (a missing check is not a failure)."""
    cmd = (cmd or "").strip()
    if not cmd:
        return True, "no verify (unverified)"
    import subprocess
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    return r.returncode == 0, f"$ {cmd} → {'ok' if r.returncode == 0 else 'FAIL'} · {tail[0][:70]}"


def _worker_brief(it, attempt, verdict):
    scope = ", ".join(it.get("files", [])) or "the files this item needs"
    base = (f"{it.get('brief', it.get('title',''))}\n\nWork ONLY within: {scope}. Done means this "
            f"passes:  {it.get('verify','(run the tests)')}\nWhen the change is correct, verify it, then say done.")
    if attempt > 0:
        base += f"\n\n[verify] your previous attempt was REJECTED: {verdict[1]}. Fix it."
    return base


# ---- compression upward: the roll-up the CEO sees ---------------------------
def _status_path(name):
    return os.path.join(_company_path(name), "STATUS.md")


def roll_up(name, goal, base, integration, final_ok, final_detail):
    """Write STATUS.md — the manager's roll-up. Compression upward: the CEO sees per-item
    STATE + a ONE-LINE result, never raw worker transcripts. Realism and context-budget
    engineering are the same move: the summary is what fits, and what's decision-relevant."""
    board = read_board(name)
    glyph = {"verified": "✓", "escalated": "⚠", "blocked": "◌", "running": "●", "queued": "◔"}
    lines = [f"# {name} — status", "", f"**Goal:** {goal}", "",
             f"Final check: {'✓ ' if final_ok else '✗ '}{final_detail}",
             f"Review:  `git diff {base}..{integration}`", "", "## Work items", ""]
    for it in board:
        g = glyph.get(it.get("state"), "?")
        one_line = " ".join((it.get("result", "") or "").split())[:100]
        lines.append(f"- {g} **{it.get('title', it['id'])}** ({it['assignee']}) — {it.get('state')}"
                     + (f": {one_line}" if one_line else ""))
    esc = [it for it in board if it.get("state") == "escalated"]
    if esc:
        lines += ["", "## Escalated to you", ""]
        lines += [f"- **{it.get('title', it['id'])}**: {it.get('result','')}" for it in esc]
    text = "\n".join(lines) + "\n"
    with open(_status_path(name), "w") as f:
        f.write(text)
    return text


def status(name):
    """A plain, scriptable view of the company: STATUS.md if the run wrote one, else the live
    board. `company status` and the future TUI are two renderers of one truth (the files)."""
    sp = _status_path(name)
    if os.path.isfile(sp):
        with open(sp) as f:
            return f.read()
    board = read_board(name)
    if not board:
        return f"{name}: chartered, no run yet."
    return "\n".join(f"{it.get('state','?'):9} {it['id']:6} {it.get('title','')}" for it in board)


_DEFAULT_MANAGER_PROMPT = (
    "You are the MANAGER of a small engineering team. Decompose the CEO's goal into the "
    "SMALLEST set of independent work items, each assigned to one worker, each with a concrete "
    "verify command that proves it done. Keep file scopes non-overlapping so workers never "
    "collide. You report a roll-up to the CEO; you never forward raw worker output.")
_DEFAULT_WORKER_PROMPT = (
    "You are a WORKER. Do exactly the item your manager assigned, within its file scope, and "
    "make its verify condition pass. Report done only when it truly passes.")
