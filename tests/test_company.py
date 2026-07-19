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

    def test_compression_upward_status_has_no_raw_worker_output(self):
        repo = _repo({"README": "x\n"})
        plan = {"work_items": [
            {"id": "w1", "title": "task", "brief": "", "assignee": "worker-1",
             "files": ["d.py"], "verify": "true", "depends_on": []}]}
        self._run(repo, plan, {"w1": lambda wt: _write(wt, "d.py", "SECRET_WORKER_INTERNALS\n")})
        # the CEO's roll-up must NOT contain the worker's file contents — only the summary
        self.assertNotIn("SECRET_WORKER_INTERNALS", company.status("co"))


if __name__ == "__main__":
    unittest.main()
