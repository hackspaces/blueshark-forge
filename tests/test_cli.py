"""CLI top-level behavior — Ctrl-C / Ctrl-D exit clean from any command.

A KeyboardInterrupt (real Ctrl-C) or EOFError (Ctrl-D) raised anywhere under a
forge subcommand — a setup prompt, a model pull, a long run — must exit with the
shell SIGINT convention (130) and NEVER dump a traceback. Stdlib, offline.
"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import __main__ as M          # noqa: E402


class _Exit(Exception):
    pass


def _run(argv, stub_exc):
    """Run main() with a subcommand stubbed to raise `stub_exc`; return
    (exit_code, stderr_text)."""
    orig_status, orig_argv, orig_stderr = M.cmd_status, sys.argv, sys.stderr

    def _raise(_args):
        raise stub_exc

    M.cmd_status = _raise
    sys.argv = argv
    err = io.StringIO()
    sys.stderr = err
    try:
        M.main()
        code = 0
    except SystemExit as e:
        code = e.code
    finally:
        M.cmd_status, sys.argv, sys.stderr = orig_status, orig_argv, orig_stderr
    return code, err.getvalue()


class TestCleanInterrupt(unittest.TestCase):
    def test_ctrl_c_exits_130_without_traceback(self):
        code, err = _run(["forge", "status"], KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertNotIn("Traceback", err)

    def test_ctrl_d_exits_130_without_traceback(self):
        code, err = _run(["forge", "status"], EOFError())
        self.assertEqual(code, 130)
        self.assertNotIn("Traceback", err)

    def test_setup_normal_exit_passes_through(self):
        # setup's own sys.exit(code) must not be swallowed by the interrupt handler.
        from forge import setup as setupmod
        orig, orig_argv = setupmod.run, sys.argv
        setupmod.run = lambda **k: 0
        sys.argv = ["forge", "setup"]
        try:
            with self.assertRaises(SystemExit) as cm:
                M.main()
            self.assertEqual(cm.exception.code, 0)
        finally:
            setupmod.run, sys.argv = orig, orig_argv

    def test_ctrl_c_during_setup_prompt_is_clean(self):
        from forge import setup as setupmod
        orig, orig_argv, orig_stderr = setupmod.run, sys.argv, sys.stderr

        def _raise(**k):
            raise KeyboardInterrupt

        setupmod.run = _raise
        sys.argv = ["forge", "setup"]
        err = io.StringIO()
        sys.stderr = err
        try:
            with self.assertRaises(SystemExit) as cm:
                M.main()
            self.assertEqual(cm.exception.code, 130)
            self.assertNotIn("Traceback", err.getvalue())
        finally:
            setupmod.run, sys.argv, sys.stderr = orig, orig_argv, orig_stderr


class TestEntrypoint(unittest.TestCase):
    """The console-script surface the release pipeline smoke-tests (forge --version /
    --help) and the version coherence its tag-gate depends on."""

    def _run_flag(self, flag):
        orig_argv, orig_out = sys.argv, sys.stdout
        sys.argv = ["forge", flag]
        out = io.StringIO()
        sys.stdout = out
        try:
            with self.assertRaises(SystemExit) as cm:
                M.main()
            return cm.exception.code, out.getvalue()
        finally:
            sys.argv, sys.stdout = orig_argv, orig_out

    def test_version_flag_prints_version_and_exits_zero(self):
        import forge
        code, text = self._run_flag("--version")
        self.assertEqual(code, 0)
        self.assertIn(forge.__version__, text)

    def test_help_flag_exits_zero(self):
        code, text = self._run_flag("--help")
        self.assertEqual(code, 0)
        self.assertIn("usage: forge", text)

    def test_version_is_semver(self):
        # the publish tag-gate compares this to the release tag — it must be a clean
        # X.Y.Z so a malformed version can never ship.
        import forge
        self.assertRegex(forge.__version__, r"^\d+\.\d+\.\d+$")


class TestFirstRun(unittest.TestCase):
    """A fresh install (no config, no model chosen) must guide the user to
    `forge models` instead of spinning up a chat/run against a model they never
    installed. Stdlib, offline — no ladder is ever constructed on this path."""

    def setUp(self):
        # Neutralize the real ~/.forge and FORGE_MODEL for a clean fresh-install state.
        self._exists, self._env = M.cfgmod.exists, os.environ.pop("FORGE_MODEL", None)
        M.cfgmod.exists = lambda: False
        # A tripwire: the first-run path must NOT build a ladder (that would hit a model).
        self._ladder = M._make_ladder
        M._make_ladder = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("first-run must not construct a ladder"))

    def tearDown(self):
        M.cfgmod.exists, M._make_ladder = self._exists, self._ladder
        if self._env is not None:
            os.environ["FORGE_MODEL"] = self._env
        else:
            os.environ.pop("FORGE_MODEL", None)

    def _ns(self, **kw):
        import types
        kw.setdefault("model", None)
        return types.SimpleNamespace(**kw)

    def test_truth_table(self):
        self.assertTrue(M._first_run(self._ns()))                    # fresh install
        self.assertFalse(M._first_run(self._ns(model="phi")))        # explicit --model
        M.cfgmod.exists = lambda: True
        self.assertFalse(M._first_run(self._ns()))                   # config written
        M.cfgmod.exists = lambda: False
        os.environ["FORGE_MODEL"] = "phi"
        self.assertFalse(M._first_run(self._ns()))                   # env override

    def test_run_exits_1_with_guidance(self):
        err = io.StringIO()
        orig = sys.stderr
        sys.stderr = err
        try:
            with self.assertRaises(SystemExit) as cm:
                M.cmd_run(self._ns(dir=".", max_steps=1))
            self.assertEqual(cm.exception.code, 1)
        finally:
            sys.stderr = orig
        self.assertIn("forge models", err.getvalue())

    def test_bare_chat_shows_welcome_and_returns(self):
        out = io.StringIO()
        orig = sys.stdout
        sys.stdout = out
        try:
            # returns cleanly (no ladder built, no REPL) — the tripwire would fire otherwise.
            self.assertIsNone(M.cmd_chat(self._ns(name=None)))
        finally:
            sys.stdout = orig
        self.assertIn("forge models", out.getvalue())


if __name__ == "__main__":
    unittest.main()
