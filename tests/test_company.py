"""forge company — Slice 1 core: charter, the routing law (unity of command), decomposition
with per-item verify (MBO cascade), the independent TRUST audit, compression-upward roll-up,
and escalation on failure.

Offline/stdlib: a REAL git repo in a tempdir (worktrees need one) with a MOCKED manager
(scripted plan) and scripted workers. Never a real model, never the network (CLAUDE.md)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import company   # noqa: E402


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)


def _repo(files):
    d = tempfile.mkdtemp()
    _git(d, "init", "-q"); _git(d, "config", "user.email", "t@t"); _git(d, "config", "user.name", "t")
    _git(d, "checkout", "-q", "-b", "main")
    for n, b in files.items():
        os.makedirs(os.path.dirname(os.path.join(d, n)), exist_ok=True) if os.path.dirname(n) else None
        with open(os.path.join(d, n), "w") as f:
            f.write(b)
    _git(d, "add", "-A"); _git(d, "commit", "-qm", "init")
    return d


class _Manager:
    """A manager backend stub returning a scripted plan (work_items JSON)."""
    def __init__(self, *plans):
        self._raws = [json.dumps(p) for p in plans]; self.i = 0
    def chat(self, messages, schema=None, temperature=0.0):
        raw = self._raws[min(self.i, len(self._raws) - 1)]; self.i += 1; return raw


class _Worker:
    def __init__(self, wt, item, muts):
        self.wt, self.item, self.muts = wt, item, muts
    def send(self, brief):
        fn = self.muts.get(self.item["id"])
        if fn:
            fn(self.wt)
        return "done"


def _write(wt, name, body):
    with open(os.path.join(wt, name), "w") as f:
        f.write(body)


class TestRoutingLaw(unittest.TestCase):
    """Fayol's unity of command, enforced in the router not the prompt."""

    def setUp(self):
        self.charter = {"roles": {
            "manager": {"reports_to": None},
            "worker-1": {"reports_to": "manager"},
            "worker-2": {"reports_to": "manager"}}}

    def test_ceo_can_only_reach_the_manager(self):
        self.assertTrue(company.can_message(self.charter, "ceo", "manager"))
        self.assertFalse(company.can_message(self.charter, "ceo", "worker-1"))   # no skip-level

    def test_only_the_manager_reports_to_the_ceo(self):
        self.assertTrue(company.can_message(self.charter, "manager", "ceo"))
        self.assertFalse(company.can_message(self.charter, "worker-1", "ceo"))

    def test_worker_reports_up_to_its_manager(self):
        self.assertTrue(company.can_message(self.charter, "worker-1", "manager"))

    def test_peers_on_the_same_team_can_talk(self):
        self.assertTrue(company.can_message(self.charter, "worker-1", "worker-2"))

    def test_manager_directs_its_workers(self):
        self.assertTrue(company.can_message(self.charter, "manager", "worker-2"))


class TestCharter(unittest.TestCase):
    def test_create_and_load(self):
        old = company.COMPANY_DIR
        company.COMPANY_DIR = tempfile.mkdtemp()
        try:
            c = company.create_charter("acme", "qwen3-coder:30b", ["qwen2.5-coder:7b", "gemma2:9b"])
            self.assertEqual(set(company.workers(c)), {"worker-1", "worker-2"})
            self.assertEqual(c["roles"]["manager"]["authority"], "operator")
            self.assertEqual(c["roles"]["worker-1"]["authority"], "contribute")
            self.assertEqual(company.load_charter("acme")["name"], "acme")
        finally:
            company.COMPANY_DIR = old


class TestDecomposition(unittest.TestCase):
    CHARTER = {"roles": {"manager": {"prompt": "m", "reports_to": None},
                         "worker-1": {"reports_to": "manager"}, "worker-2": {"reports_to": "manager"}}}

    def test_valid_plan_passes(self):
        plan = {"work_items": [
            {"id": "a", "title": "A", "brief": "", "assignee": "worker-1", "files": ["a.py"],
             "verify": "true", "depends_on": []},
            {"id": "b", "title": "B", "brief": "", "assignee": "worker-2", "files": ["b.py"],
             "verify": "true", "depends_on": []}]}
        items = company.decompose("goal", self.CHARTER, _Manager(plan))
        self.assertEqual({i["id"] for i in items}, {"a", "b"})

    def test_unknown_assignee_rejected(self):
        bad = {"work_items": [{"id": "a", "title": "", "brief": "", "assignee": "worker-9",
                               "files": ["a.py"], "verify": "true", "depends_on": []}]}
        with self.assertRaises(company.CompanyError):
            company.decompose("g", self.CHARTER, _Manager(bad, bad))

    def test_missing_verify_rejected(self):
        bad = {"work_items": [{"id": "a", "title": "", "brief": "", "assignee": "worker-1",
                               "files": ["a.py"], "verify": "", "depends_on": []}]}
        with self.assertRaises(company.CompanyError):
            company.decompose("g", self.CHARTER, _Manager(bad, bad))

    def test_repair_fixes_a_bad_first_plan(self):
        bad = {"work_items": [{"id": "a", "title": "", "brief": "", "assignee": "nobody",
                               "files": ["a.py"], "verify": "true", "depends_on": []}]}
        good = {"work_items": [{"id": "a", "title": "", "brief": "", "assignee": "worker-1",
                                "files": ["a.py"], "verify": "true", "depends_on": []}]}
        items = company.decompose("g", self.CHARTER, _Manager(bad, good))
        self.assertEqual(items[0]["assignee"], "worker-1")


class TestRun(unittest.TestCase):
    def setUp(self):
        self.old = company.COMPANY_DIR
        company.COMPANY_DIR = tempfile.mkdtemp()

    def tearDown(self):
        company.COMPANY_DIR = self.old

    def _company(self, worker_rungs=("w",)):
        return company.create_charter("co", "m", list(worker_rungs))

    def _run(self, repo, plan, muts, verifier=None):
        self._company()
        planner = _Manager(plan)
        ladder_for = lambda rung: [object()]        # roles resolve to a dummy ladder
        agent_factory = lambda ladder, wt, item: _Worker(wt, item, muts)
        return company.run("co", "goal", repo, ladder_for,
                           planner_override=planner, agent_factory=agent_factory, verifier=verifier)

    def test_verified_work_merges_and_rolls_up(self):
        repo = _repo({"README": "x\n"})
        plan = {"work_items": [
            {"id": "w1", "title": "add a", "brief": "", "assignee": "worker-1",
             "files": ["a.py"], "verify": "test -f a.py", "depends_on": []}]}
        res = self._run(repo, plan, {"w1": lambda wt: _write(wt, "a.py", "A=1\n")})
        self.assertEqual(res["verified"], ["w1"])
        self.assertIn("A=1", _git(repo, "show", "forge-company/co/integration:a.py").stdout)
        # STATUS.md is the ROLL-UP — a one-line result, NOT the worker's raw output
        status = company.status("co")
        self.assertIn("✓", status)
        self.assertIn("add a", status)

    def test_failed_verify_escalates_not_merges(self):
        repo = _repo({"README": "x\n"})
        plan = {"work_items": [
            {"id": "w1", "title": "make b", "brief": "", "assignee": "worker-1",
             "files": ["b.py"], "verify": "test -f b.py", "depends_on": []}]}
        # worker writes the WRONG file → the item's verify fails → escalated after re-dispatch
        res = self._run(repo, plan, {"w1": lambda wt: _write(wt, "other.py", "x\n")})
        self.assertEqual(res["escalated"], ["w1"])
        self.assertEqual(res["verified"], [])
        self.assertIn("Escalated to you", company.status("co"))

    def test_trust_audit_can_reject_a_passing_item(self):
        # the item's own verify passes, but the INDEPENDENT verifier rejects → not merged.
        # This is the principal-agent monitoring win: a self-check isn't enough.
        repo = _repo({"README": "x\n"})
        plan = {"work_items": [
            {"id": "w1", "title": "sneaky", "brief": "", "assignee": "worker-1",
             "files": ["c.py"], "verify": "true", "depends_on": []}]}
        res = self._run(repo, plan, {"w1": lambda wt: _write(wt, "c.py", "x\n")},
                        verifier=lambda wt, files: (False, "TRUST: independent check failed"))
        self.assertEqual(res["escalated"], ["w1"])

    def test_verify_with_a_missing_runner_is_unverified_not_failed(self):
        # found by running: the manager wrote `python ...` where only python3 exists →
        # command-not-found. That must be UNVERIFIED (the item's self-check couldn't run),
        # not a hard failure — the independent TRUST audit is the real gate.
        ok, detail = company._run_verify("definitely-not-a-real-binary-xyz --check", tempfile.mkdtemp())
        self.assertTrue(ok)
        self.assertIn("unverified", detail)

    def test_a_verify_that_runs_and_fails_still_fails(self):
        ok, _ = company._run_verify("false", tempfile.mkdtemp())   # runs, exits 1
        self.assertFalse(ok)

    def test_compression_upward_status_has_no_raw_worker_output(self):
        repo = _repo({"README": "x\n"})
        plan = {"work_items": [
            {"id": "w1", "title": "task", "brief": "", "assignee": "worker-1",
             "files": ["d.py"], "verify": "true", "depends_on": []}]}
        self._run(repo, plan, {"w1": lambda wt: _write(wt, "d.py", "SECRET_WORKER_INTERNALS\n")})
        # the CEO's roll-up must NOT contain the worker's file contents — only the summary
        self.assertNotIn("SECRET_WORKER_INTERNALS", company.status("co"))


class TestHarnessManagedGit(unittest.TestCase):
    """Git is the harness's job: a company runs in a plain directory (auto-init), and
    verified work is applied back to the user's files (reversibly), not left as a chore."""

    def setUp(self):
        self.old = company.COMPANY_DIR
        company.COMPANY_DIR = tempfile.mkdtemp()

    def tearDown(self):
        company.COMPANY_DIR = self.old

    def test_auto_inits_a_non_git_directory(self):
        from forge import team
        d = tempfile.mkdtemp()                              # NOT a git repo
        with open(os.path.join(d, "x.txt"), "w") as f:
            f.write("hi\n")
        created = team.ensure_repo(d)
        self.assertTrue(created)
        self.assertEqual(_git(d, "rev-parse", "--is-inside-work-tree").stdout.strip(), "true")
        self.assertTrue(team.working_tree_clean(d))         # the snapshot committed the files

    def test_existing_repo_is_not_reinitialised(self):
        from forge import team
        d = _repo({"a": "1\n"})
        self.assertFalse(team.ensure_repo(d))

    def test_verified_work_is_applied_to_a_plain_directory(self):
        # the whole point: run a company in a bare dir, get the files back — no git by hand
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "a.py"), "w") as f:
            f.write("x = 1\n")
        company.create_charter("co", "m", ["w"])
        plan = {"work_items": [{"id": "w1", "title": "t", "brief": "", "assignee": "worker-1",
                                "files": ["b.py"], "verify": "test -f b.py", "depends_on": []}]}
        res = company.run("co", "goal", d, lambda r: [object()], planner_override=_Manager(plan),
                          agent_factory=lambda ladder, wt, item: _Worker(wt, item,
                              {"w1": lambda wt: _write(wt, "b.py", "B=1\n")}))
        self.assertTrue(res["applied"]["applied"])
        self.assertTrue(os.path.exists(os.path.join(d, "b.py")))   # applied to the user's files
        self.assertTrue(res["applied"]["undo_to"])                 # and reversible

    def test_dirty_working_tree_is_not_auto_applied(self):
        from forge import team
        d = _repo({"a.py": "x = 1\n"})
        with open(os.path.join(d, "a.py"), "w") as f:
            f.write("x = 999  # uncommitted\n")               # make the tree DIRTY
        company.create_charter("co", "m", ["w"])
        plan = {"work_items": [{"id": "w1", "title": "t", "brief": "", "assignee": "worker-1",
                                "files": ["b.py"], "verify": "test -f b.py", "depends_on": []}]}
        res = company.run("co", "goal", d, lambda r: [object()], planner_override=_Manager(plan),
                          agent_factory=lambda ladder, wt, item: _Worker(wt, item,
                              {"w1": lambda wt: _write(wt, "b.py", "B=1\n")}))
        self.assertFalse(res["applied"]["applied"])          # not applied — the user's edits are safe
        self.assertEqual(_read(os.path.join(d, "a.py")), "x = 999  # uncommitted\n")


def _read(p):
    with open(p) as f:
        return f.read()


class TestSetupOffersCompany(unittest.TestCase):
    """forge setup ends by chartering a 'starter' company, so a first-time user comes out
    with an org ready, not just a lone agent."""

    def setUp(self):
        self.old = company.COMPANY_DIR
        company.COMPANY_DIR = tempfile.mkdtemp()

    def tearDown(self):
        company.COMPANY_DIR = self.old

    def test_auto_charters_a_starter_from_the_ladder(self):
        import unittest.mock as mock
        from forge import setup
        with mock.patch("forge.config.load", return_value={"ladder": ["cheap:1b", "strong:30b"]}):
            setup._offer_company(auto=True)                  # auto → no prompt, just charter
        self.assertTrue(company.charter_exists("starter"))
        c = company.load_charter("starter")
        self.assertEqual(c["roles"]["manager"]["rung"], "strong:30b")   # strongest rung manages
        self.assertEqual(set(company.workers(c)), {"worker-1", "worker-2"})

    def test_does_not_duplicate_an_existing_starter(self):
        import unittest.mock as mock
        from forge import setup
        company.create_charter("starter", "m", ["w"])
        with mock.patch("forge.config.load", return_value={"ladder": ["x:1b", "y:30b"]}):
            setup._offer_company(auto=True)                  # already exists → no-op
        self.assertEqual(company.load_charter("starter")["roles"]["manager"]["rung"], "m")

    def test_no_ladder_is_a_clean_noop(self):
        import unittest.mock as mock
        from forge import setup
        with mock.patch("forge.config.load", return_value={"ladder": []}):
            setup._offer_company(auto=True)                  # nothing to build from
        self.assertFalse(company.charter_exists("starter"))


if __name__ == "__main__":
    unittest.main()
