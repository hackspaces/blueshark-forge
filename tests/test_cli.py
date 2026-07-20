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

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

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


class TestGroupedHelp(unittest.TestCase):
    """`forge --help` is hand-grouped by what you're trying to do, which buys
    readability at the cost of argparse's automatic command list. So the thing
    worth pinning is that the hand-written groups can never drift from the
    parser: every real subcommand documented, and nothing documented that isn't
    real."""

    def _help(self):
        orig_argv, orig_out = sys.argv, sys.stdout
        sys.argv = ["forge", "--help"]
        out = io.StringIO()
        sys.stdout = out
        try:
            with self.assertRaises(SystemExit) as cm:
                M.main()
            self.assertEqual(cm.exception.code, 0)
            return out.getvalue()
        finally:
            sys.argv, sys.stdout = orig_argv, orig_out

    def _real_subcommands(self):
        """The names argparse itself knows — read back out of its own
        invalid-choice error, so this can't drift from the parser."""
        import re
        orig_argv, orig_err = sys.argv, sys.stderr
        sys.argv = ["forge", "__definitely_not_a_command__"]
        err = io.StringIO()
        sys.stderr = err
        try:
            with self.assertRaises(SystemExit):
                M.main()
        finally:
            sys.argv, sys.stderr = orig_argv, orig_err
        m = re.search(r"choose from (.+?)\)\s*$", err.getvalue().strip(), re.S)
        self.assertIsNotNone(m, f"couldn't read choices from: {err.getvalue()!r}")
        return {c.strip().strip("'\"") for c in m.group(1).split(",")}

    def _documented(self):
        return {n.strip()
                for _, rows in M._COMMAND_GROUPS
                for name, _ in rows
                for n in name.split(",")}

    def test_groups_cover_exactly_the_real_commands(self):
        # Add a subcommand and forget to group it → this fails. Good.
        self.assertEqual(self._documented(), self._real_subcommands())

    def test_help_is_grouped_and_names_bare_forge(self):
        text = self._help()
        self.assertIn("usage: forge", text)            # the entrypoint smoke-test pins this too
        for title, _ in M._COMMAND_GROUPS:
            self.assertIn(title, text)
        # bare `forge` is the most common invocation and argparse never mentioned it
        self.assertIn("chat, oriented in the current directory", text)

    def test_every_option_is_described(self):
        # the old help listed --model/--dir/--name/--verbose with no help text at all
        text = self._help()
        for flag in ("--model", "--dir", "--name", "--resume", "--verbose", "--version"):
            self.assertIn(flag, text)
        self.assertNotIn("show this help message and exit", text)   # argparse's default phrasing


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


class TestFrozenDaemonReinvoke(unittest.TestCase):
    """A frozen single-file binary (PyInstaller/Nuitka) has no `python -m forge.daemon`
    — sys.executable IS the forge binary — so `forge up` must re-invoke forge's own
    `daemon` hook, and `forge daemon <model> [interval]` must run the autopilot loop
    (intercepted before argparse). From source, sys.frozen is unset and behavior is
    unchanged. Verified end-to-end in a real PyInstaller build; this pins the logic."""

    def test_source_launch_uses_dash_m(self):
        self.assertFalse(getattr(sys, "frozen", False))       # from source
        self.assertEqual(M._daemon_launch_cmd("qwen2.5:0.5b", 20),
                         [sys.executable, "-m", "forge.daemon", "qwen2.5:0.5b", "20"])

    def test_frozen_launch_uses_daemon_subcommand(self):
        import unittest.mock as mock
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertEqual(M._daemon_launch_cmd("qwen2.5:0.5b", 20),
                             [sys.executable, "daemon", "qwen2.5:0.5b", "20"])

    def test_daemon_argv_is_intercepted_before_argparse(self):
        import unittest.mock as mock
        seen = {}

        class FakeForged:
            def __init__(self, model, interval): seen["init"] = (model, interval)
            def run(self): seen["ran"] = True

        with mock.patch("forge.daemon.Forged", FakeForged), \
             mock.patch.object(sys, "argv", ["forge", "daemon", "m1", "5"]):
            M.main()                                          # must NOT reach argparse
        self.assertEqual(seen.get("init"), ("m1", 5))
        self.assertTrue(seen.get("ran"))


if __name__ == "__main__":
    unittest.main()
