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
