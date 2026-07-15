"""P6.3 — structured test-run digests. Offline, stdlib-only.

Covers the parser (pytest / unittest / go / cargo / jest failure extraction, workspace
frame filtering, passing-run summary, non-test output → None, runner detection) and the
`run_tests` action end to end via execute()."""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import testparse                       # noqa: E402
from forge.tools import execute                    # noqa: E402


def _w(d, name, s):
    with open(os.path.join(d, name), "w") as f:
        f.write(s)


class TestRunnerDetect(unittest.TestCase):
    def test_is_test_runner(self):
        for c in ("pytest -q", "python3 -m unittest", "py.test", "go test ./...",
                  "cargo test", "npm test", "npm run test", "jest", "vitest"):
            self.assertTrue(testparse.is_test_runner(c), c)
        for c in ("ls -la", "python app.py", "git status", "grep -r x .", "", "echo pytest"):
            self.assertFalse(testparse.is_test_runner(c), c)


class TestDigest(unittest.TestCase):
    def test_real_unittest_failures(self):
        d = tempfile.mkdtemp()
        _w(d, "mathx.py", "def add(a, b):\n    return a - b\n")   # BUG: subtracts
        _w(d, "test_mathx.py",
           "import unittest\nfrom mathx import add\n"
           "class T(unittest.TestCase):\n"
           "    def test_a(self):\n        self.assertEqual(add(2, 3), 5)\n"
           "    def test_ok(self):\n        self.assertEqual(add(5, 0), 5)\n")
        r = subprocess.run(["python3", "-m", "unittest", "test_mathx"], cwd=d,
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        dg = testparse.digest(out, d)
        self.assertIn("failures=1", dg)                        # summary line
        self.assertIn("✗", dg)                                 # a failing entry
        self.assertIn("AssertionError", dg)                    # the reason
        self.assertIn("test_mathx.py:5", dg)                   # workspace frame
        self.assertNotIn("test_ok", dg)                        # a PASSING test is not listed

    def test_synthetic_pytest(self):
        out = (
            "=================================== FAILURES ===================================\n"
            "    def test_one():\n>       assert 6 == 5\nE       assert 6 == 5\n"
            "tests/test_a.py:4: AssertionError\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_a.py::test_one - assert 6 == 5\n"
            "========================= 1 failed, 3 passed in 0.03s =========================\n")
        dg = testparse.digest(out, "/nowhere")
        self.assertIn("1 failed, 3 passed", dg)
        self.assertIn("tests/test_a.py::test_one", dg)
        self.assertIn("assert 6 == 5", dg)

    def test_unittest_summary_not_mistaken_for_a_pytest_failure(self):
        # `FAILED (failures=1)` must NOT be parsed as a pytest `FAILED <nodeid>` line
        out = ("FAIL: test_x (m.T.test_x)\n----\nAssertionError: nope\n"
               "----------------------------------------------------------------------\n"
               "Ran 1 test in 0.0s\n\nFAILED (failures=1)\n")
        dg = testparse.digest(out, "")
        self.assertNotIn("(failures=1) —", dg)                 # not a bogus failure entry
        self.assertIn("test_x", dg)

    def test_passing_run_is_summary_only(self):
        out = "....\n----------------------------------------------------------------------\nRan 4 tests in 0.01s\n\nOK\n"
        dg = testparse.digest(out, "")
        self.assertIsNotNone(dg)
        self.assertIn("OK", dg)
        self.assertNotIn("✗", dg)                              # nothing failing

    def test_non_test_output_returns_none(self):
        self.assertIsNone(testparse.digest("total 8\ndrwxr-xr-x  a.py\n", ""))
        self.assertIsNone(testparse.digest("", ""))
        self.assertIsNone(testparse.digest("hello world", ""))

    def test_workspace_frames_only(self):
        out = ('FAIL: t (m.T.t)\n'
               '  File "/usr/lib/python3/site-packages/pytest/x.py", line 99, in run\n'
               '  File "%s", line 7, in t\n'
               'AssertionError: x\n'
               '----------------------------------------------------------------------\nRan 1 test\n\nFAILED (failures=1)\n')
        d = tempfile.mkdtemp()
        _w(d, "code.py", "x = 1\n")
        dg = testparse.digest(out % os.path.join(d, "code.py"), d)
        self.assertIn("code.py:7", dg)                         # workspace frame kept
        self.assertNotIn("site-packages", dg)                  # stdlib/site-packages frame dropped


class TestZeroCollected(unittest.TestCase):
    def test_zero_collected_shapes(self):
        for out in ("Ran 0 tests in 0.000s\n\nOK",              # unittest, exit 0 before 3.12
                    "Ran 0 tests in 0.000s\n\nNO TESTS RAN",    # unittest 3.12+
                    "===== no tests ran in 0.01s =====",        # pytest banner
                    "no tests ran in 0.01s"):                   # pytest -q
            self.assertTrue(testparse.zero_collected(out), out)

    def test_real_runs_are_not_zero(self):
        for out in ("Ran 3 tests in 0.1s\n\nOK",
                    "Ran 725 tests in 15.7s\n\nFAILED (failures=1)",
                    "collected 12 items\n... 12 passed in 0.2s",
                    "5 passed in 0.2s", "1 failed, 4 passed in 0.3s", "", None):
            self.assertFalse(testparse.zero_collected(out), repr(out))

    def test_leaked_zero_lines_inside_a_passing_run_do_not_flip_it(self):
        # a passing suite whose tests spawn a NESTED runner (meta-tests of a CLI
        # wrapper) leaks 'Ran 0 tests' / 'no tests ran' into the outer output —
        # only the FINAL summary decides, so the outer pass must stand.
        for out in ("test_meta ... ok\nRan 0 tests in 0.000s\n\nOK\nRan 5 tests in 0.1s\n\nOK",
                    "no tests ran in 0.01s\n...\n7 passed in 0.3s"):
            self.assertFalse(testparse.zero_collected(out), out)

    def test_collection_errors_are_breakage_not_zero(self):
        # pytest's 'collected 0 items / 1 error' (rc 2) is positive evidence the
        # change BROKE the suite — treating it as a zero run would let the claim
        # through the verifier-unavailable escape.
        for out in ("collected 0 items / 1 error\nE ImportError: boom\n"
                    "=========== 1 error in 0.12s ===========",
                    "!!!!! Interrupted: 1 error during collection !!!!!\n1 error in 0.05s"):
            self.assertFalse(testparse.zero_collected(out), out)


class TestRunTestsAction(unittest.TestCase):
    def test_run_tests_digests_a_failing_suite(self):
        d = tempfile.mkdtemp()
        _w(d, "mathx.py", "def add(a, b):\n    return a - b\n")
        _w(d, "test_mathx.py",
           "import unittest\nfrom mathx import add\n"
           "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(add(2, 3), 5)\n")
        obs, ok = execute({"action": "run_tests"}, d)
        self.assertFalse(ok)                                   # a failing suite → not ok
        self.assertIn("python3 -m unittest", obs)              # auto-detected command shown
        self.assertIn("✗", obs)
        self.assertIn("AssertionError", obs)

    def test_run_tests_on_passing_suite(self):
        d = tempfile.mkdtemp()
        _w(d, "mathx.py", "def add(a, b):\n    return a + b\n")
        _w(d, "test_mathx.py",
           "import unittest\nfrom mathx import add\n"
           "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(add(2, 3), 5)\n")
        obs, ok = execute({"action": "run_tests"}, d)
        self.assertTrue(ok)

    def test_run_tests_with_no_suite(self):
        d = tempfile.mkdtemp()
        _w(d, "readme.txt", "hi\n")
        obs, ok = execute({"action": "run_tests"}, d)
        self.assertFalse(ok)
        self.assertIn("no test suite detected", obs)

    def test_run_tests_zero_collected_is_never_a_pass(self):
        # a unittest-style file with no TestCase in it: discovery runs, collects 0
        # tests, and exits 0 before Python 3.12 — that must NOT read as a pass, and
        # the observation must say WHY instead of a bare "Ran 0 tests".
        d = tempfile.mkdtemp()
        _w(d, "test_empty.py", "import unittest\n")
        obs, ok = execute({"action": "run_tests"}, d)
        self.assertFalse(ok)
        self.assertIn("0 tests were collected", obs)

    def test_run_tests_routes_pytest_style_files_to_pytest(self):
        # module-level test functions never reach stdlib unittest (which would
        # collect 0 of them) — the detected command is a pytest invocation.
        d = tempfile.mkdtemp()
        _w(d, "main.py", "def add(a, b):\n    return a + b\n")
        _w(d, "test_main.py", "from main import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
        obs, ok = execute({"action": "run_tests"}, d)
        self.assertIn("pytest", obs.splitlines()[0])           # `$ pytest -q` or `$ python3 -m pytest -q`
        self.assertNotIn("no test suite detected", obs)


class _Backend:
    name = "b"
    def __init__(self, acts):
        self.a, self.i = list(acts), 0
    def stream(self, m, schema=None, temperature=0.0):
        act = self.a[min(self.i, len(self.a) - 1)]; self.i += 1; yield act
    def chat(self, m, schema=None, temperature=0.0):
        return '{"thought":"x","action":"say","message":"done"}'


class TestRunTestsSatisfiesDoneGate(unittest.TestCase):
    def test_passing_run_tests_lets_a_mutating_turn_finish(self):
        from forge.agent import Agent
        from forge import session as sm
        d = tempfile.mkdtemp()
        _w(d, "mathx.py", "def add(a, b):\n    return a - b\n")   # bug
        _w(d, "test_mathx.py",
           "import unittest\nfrom mathx import add\n"
           "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(add(2, 3), 5)\n")
        acts = [
            '{"thought":"read","action":"read_file","path":"mathx.py"}',
            '{"thought":"fix","action":"edit_file","path":"mathx.py","old":"return a - b","new":"return a + b"}',
            '{"thought":"verify","action":"run_tests"}',
            '{"thought":"done","action":"say","message":"Fixed the bug and the tests pass."}',
        ]
        a = Agent(_Backend(acts), sm.EphemeralSession(d, "b"), max_steps=8, autonomous=True)
        reply = a.send("fix the bug in mathx.py")
        self.assertIn("Fixed", reply)                          # say accepted — done-gate saw run_tests pass
        self.assertEqual(open(os.path.join(d, "mathx.py")).read(),
                         "def add(a, b):\n    return a + b\n")


if __name__ == "__main__":
    unittest.main()
