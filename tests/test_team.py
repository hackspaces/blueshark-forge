"""P9.2 forge team — the deterministic orchestrator: DAG validation, dependency order,
planner repair, worktree isolation, the verify-before-merge gate, and dependency-skip.

Offline and stdlib-only: a REAL git repo in a tempdir (worktrees need one) but MOCKED
models — a planner backend returning a scripted DAG and a scripted worker factory that
mutates files. Never a real model, never the network (CLAUDE.md)."""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import team   # noqa: E402


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _repo(files):
    d = tempfile.mkdtemp()
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@t")
    _git(d, "config", "user.name", "t")
    _git(d, "checkout", "-q", "-b", "main")
    for name, body in files.items():
        p = os.path.join(d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(name) else None
        with open(p, "w") as f:
            f.write(body)
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "init")
    return d


class _Planner:
    """A backend stub whose chat() returns scripted DAG JSON(s), one per call."""
    def __init__(self, *dags):
        import json
        self._raws = [json.dumps(d) for d in dags]
        self.i = 0
    def chat(self, messages, schema=None, temperature=0.0):
        raw = self._raws[min(self.i, len(self._raws) - 1)]
        self.i += 1
        return raw


class _Worker:
    """A scripted worker: applies a per-task mutation to its worktree instead of running a
    real Agent loop. `mutations[task_id](worktree)` writes files."""
    def __init__(self, worktree, task, mutations):
        self.worktree, self.task, self.mutations = worktree, task, mutations
    def send(self, prompt):
        fn = self.mutations.get(self.task["id"])
        if fn:
            fn(self.worktree)
        return "done"


def _factory(mutations):
    return lambda ladder, worktree, task: _Worker(worktree, task, mutations)


def _write(wt, name, body):
    with open(os.path.join(wt, name), "w") as f:
        f.write(body)


class TestDagValidation(unittest.TestCase):
    def test_overlapping_scopes_rejected(self):
        dag = {"subtasks": [
            {"id": "a", "title": "", "prompt": "", "files": ["x.py"], "depends_on": []},
            {"id": "b", "title": "", "prompt": "", "files": ["x.py"], "depends_on": []}]}
        self.assertTrue(any("x.py" in i for i in team.validate_dag(dag)))

    def test_unknown_dependency_rejected(self):
        dag = {"subtasks": [{"id": "a", "title": "", "prompt": "", "files": ["x"], "depends_on": ["ghost"]}]}
        self.assertTrue(any("ghost" in i for i in team.validate_dag(dag)))

    def test_cycle_rejected(self):
        dag = {"subtasks": [
            {"id": "a", "title": "", "prompt": "", "files": ["x"], "depends_on": ["b"]},
            {"id": "b", "title": "", "prompt": "", "files": ["y"], "depends_on": ["a"]}]}
        self.assertIn("dependency cycle", team.validate_dag(dag))

    def test_clean_dag_has_no_issues(self):
        dag = {"subtasks": [
            {"id": "a", "title": "", "prompt": "", "files": ["x"], "depends_on": []},
            {"id": "b", "title": "", "prompt": "", "files": ["y"], "depends_on": ["a"]}]}
        self.assertEqual(team.validate_dag(dag), [])

    def test_topo_order_respects_deps(self):
        subs = [{"id": "b", "depends_on": ["a"]}, {"id": "a", "depends_on": []}]
        order = [s["id"] for s in team._topo_order(subs)]
        self.assertLess(order.index("a"), order.index("b"))


class TestPlannerRepair(unittest.TestCase):
    def test_repairs_an_invalid_first_plan(self):
        bad = {"subtasks": [
            {"id": "a", "title": "", "prompt": "", "files": ["x"], "depends_on": []},
            {"id": "b", "title": "", "prompt": "", "files": ["x"], "depends_on": []}]}  # overlap
        good = {"subtasks": [
            {"id": "a", "title": "", "prompt": "", "files": ["x"], "depends_on": []},
            {"id": "b", "title": "", "prompt": "", "files": ["y"], "depends_on": []}]}
        dag = team.plan("g", ["x", "y"], _Planner(bad, good))
        self.assertEqual(team.validate_dag(dag), [])

    def test_gives_up_when_never_valid(self):
        bad = {"subtasks": [
            {"id": "a", "title": "", "prompt": "", "files": ["x"], "depends_on": []},
            {"id": "b", "title": "", "prompt": "", "files": ["x"], "depends_on": []}]}
        with self.assertRaises(team.TeamError):
            team.plan("g", ["x"], _Planner(bad, bad))


class TestRunTeam(unittest.TestCase):
    def _run(self, repo, dag, mutations, **kw):
        return team.run_team("goal", repo, ladder=[object()], planner=_Planner(dag),
                             agent_factory=_factory(mutations), **kw)

    def test_two_independent_tasks_both_merge(self):
        repo = _repo({"README": "x\n"})   # no test suite -> unverified -> accepted
        dag = {"subtasks": [
            {"id": "a", "title": "a", "prompt": "", "files": ["a.py"], "depends_on": []},
            {"id": "b", "title": "b", "prompt": "", "files": ["b.py"], "depends_on": []}]}
        muts = {"a": lambda wt: _write(wt, "a.py", "A=1\n"),
                "b": lambda wt: _write(wt, "b.py", "B=1\n")}
        res = self._run(repo, dag, muts)
        self.assertEqual(sorted(res["merged"]), ["a", "b"])
        # both files exist on the integration branch
        show = _git(repo, "show", "forge-team/integration:a.py")
        self.assertIn("A=1", show.stdout)
        self.assertIn("B=1", _git(repo, "show", "forge-team/integration:b.py").stdout)
        # the user's working tree/branch is untouched (still just README)
        self.assertFalse(os.path.exists(os.path.join(repo, "a.py")))

    def test_verify_gate_blocks_a_failing_task(self):
        # a repo WITH a suite: the task must make it pass. A worker that writes a wrong
        # value fails verification and must NOT merge.
        repo = _repo({
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": "import unittest\nfrom calc import add\n"
                            "class T(unittest.TestCase):\n    def test(self): self.assertEqual(add(2,3),5)\n"})
        dag = {"subtasks": [{"id": "fix", "title": "fix add", "prompt": "",
                             "files": ["calc.py"], "depends_on": []}]}
        # worker "fixes" it wrong -> suite still fails -> refuted
        muts = {"fix": lambda wt: _write(wt, "calc.py", "def add(a, b):\n    return a * b\n")}
        res = self._run(repo, dag, muts, retries=0)
        self.assertEqual(res["results"]["fix"]["status"], "failed")
        self.assertEqual(res["merged"], [])

    def test_verify_gate_merges_a_passing_task(self):
        repo = _repo({
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": "import unittest\nfrom calc import add\n"
                            "class T(unittest.TestCase):\n    def test(self): self.assertEqual(add(2,3),5)\n"})
        dag = {"subtasks": [{"id": "fix", "title": "fix add", "prompt": "",
                             "files": ["calc.py"], "depends_on": []}]}
        muts = {"fix": lambda wt: _write(wt, "calc.py", "def add(a, b):\n    return a + b\n")}  # correct
        res = self._run(repo, dag, muts)
        self.assertEqual(res["results"]["fix"]["status"], "merged")
        self.assertIn("def add(a, b):\n    return a + b", _git(repo, "show", "forge-team/integration:calc.py").stdout)

    def test_dependent_task_skipped_when_dependency_fails(self):
        repo = _repo({
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": "import unittest\nfrom calc import add\n"
                            "class T(unittest.TestCase):\n    def test(self): self.assertEqual(add(2,3),5)\n"})
        dag = {"subtasks": [
            {"id": "fix", "title": "fix", "prompt": "", "files": ["calc.py"], "depends_on": []},
            {"id": "doc", "title": "doc", "prompt": "", "files": ["DOC.md"], "depends_on": ["fix"]}]}
        muts = {"fix": lambda wt: _write(wt, "calc.py", "def add(a, b):\n    return a - b\n"),  # still wrong
                "doc": lambda wt: _write(wt, "DOC.md", "docs\n")}
        res = self._run(repo, dag, muts, retries=0)
        self.assertEqual(res["results"]["fix"]["status"], "failed")
        self.assertEqual(res["results"]["doc"]["status"], "skipped")

    def test_worker_that_changes_nothing_is_not_merged(self):
        repo = _repo({"README": "x\n"})
        dag = {"subtasks": [{"id": "noop", "title": "noop", "prompt": "", "files": ["z.py"], "depends_on": []}]}
        res = self._run(repo, dag, {"noop": lambda wt: None}, retries=0)   # worker writes nothing
        self.assertEqual(res["results"]["noop"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
