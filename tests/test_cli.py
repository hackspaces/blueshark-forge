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


if __name__ == "__main__":
    unittest.main()
