"""Environment fluency — OS-userland awareness. Offline, stdlib-only, OS-agnostic
(the OS-gated behavior is exercised by patching, so this passes on Linux CI and macOS)."""
import os
import sys
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import tools                            # noqa: E402
from forge import workspace                         # noqa: E402


class TestOSUserlandHints(unittest.TestCase):
    def _fires(self, text):
        return any(rx.search(text) for rx, _ in tools._DARWIN_HINTS)

    def test_darwin_patterns_match_real_error_strings(self):
        # the exact message this session hit, plus the other BSD-vs-GNU gotchas
        self.assertTrue(self._fires("(eval):3: command not found: timeout"))
        self.assertTrue(self._fires("timeout: command not found"))
        self.assertTrue(self._fires("zsh: command not found: gtimeout"))
        self.assertTrue(self._fires('sed: 1: "f.txt": invalid command code f'))
        self.assertTrue(self._fires("readlink: illegal option -- f"))
        self.assertTrue(self._fires("date: illegal option -- d"))
        self.assertFalse(self._fires("ModuleNotFoundError: No module named x"))

    def test_error_hint_is_os_gated(self):
        with mock.patch.object(tools, "_IS_DARWIN", True):
            self.assertIn("gtimeout", tools.error_hint("command not found: timeout"))
        with mock.patch.object(tools, "_IS_DARWIN", False):
            h = tools.error_hint("command not found: timeout")     # linux HAS timeout → generic hint
            self.assertNotIn("gtimeout", h)
            self.assertIn("PATH", h)

    def test_generic_hints_unaffected(self):
        self.assertIn("module is missing", tools.error_hint("ModuleNotFoundError: No module named x"))
        self.assertIn("port is already taken", tools.error_hint("Error: address already in use"))


class TestEnvironmentBriefing(unittest.TestCase):
    def test_darwin_surfaces_bsd_userland_note(self):
        with mock.patch.object(workspace.platform, "system", return_value="Darwin"):
            env = workspace.environment(os.getcwd())
        self.assertIn("Userland: BSD", env)
        self.assertIn("gtimeout", env)                             # tells the model the right command
        self.assertIn("sed -i ''", env)

    def test_no_bsd_note_on_linux(self):
        with mock.patch.object(workspace.platform, "system", return_value="Linux"):
            env = workspace.environment(os.getcwd())
        self.assertNotIn("Userland: BSD", env)

    def test_briefing_still_lists_tools_and_os(self):
        env = workspace.environment(os.getcwd())
        self.assertIn("Tools available:", env)
        self.assertIn("OS:", env)


if __name__ == "__main__":
    unittest.main()
