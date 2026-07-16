"""Native-Windows regression: forge.tui must IMPORT and degrade to plain input()
on a machine with no POSIX terminal control (termios/tty absent).

The bug this pins: tui.py imported termios/tty at module top level, so on Windows
`forge` (bare -> repl.run -> `from .tui import ...`) died with ImportError before
tui's own non-TTY fallback could run — `forge setup` worked, then `forge` crashed.

The check runs in a SUBPROCESS with `sys.modules['termios'] = None` (which makes
`import termios` raise ImportError, exactly as on Windows), because this test
process itself is on a real POSIX box where termios imports fine."""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Simulate native Windows, then exercise the whole degrade path. Any failed
# assertion (or an unguarded termios.* call) aborts with a non-zero exit.
_SCRIPT = r"""
import sys, io, threading
sys.modules['termios'] = None      # `import termios` now raises ImportError, as on Windows
sys.modules['tty'] = None

import forge.tui as tui             # (1) must not crash on import — the showstopper
assert tui.termios is None and tui.tty is None, "termios/tty import is not guarded"

# (2) _supported() must be False even when both streams report isatty()==True,
#     because a Windows console IS a tty but has no termios to drive raw mode.
class FakeTTY:
    def isatty(self): return True
_in, _out = sys.stdin, sys.stdout
sys.stdin, sys.stdout = FakeTTY(), FakeTTY()
sup = tui._supported()
sys.stdin, sys.stdout = _in, _out
assert sup is False, "_supported() must be False when termios is absent"

# (3) Screen constructs but is not enabled, and prompt() falls back to input().
s = tui.Screen()
assert s.enabled is False, "Screen must not be enabled without termios"
_in = sys.stdin
sys.stdin = io.StringIO("hi from windows\n")
try:
    line = s.prompt("> ", history=[])
finally:
    sys.stdin = _in
assert line == "hi from windows", "prompt() did not fall back to input()"

# (4) the interrupt watcher runs fn() directly instead of entering raw mode.
assert tui.run_interruptible(lambda: 7, threading.Event()) == 7

# (5) the spinner is a no-op, not a crash.
sp = tui.FooterSpinner(s); sp.start(); sp.stop()

print("WINDOWS_TUI_OK")
"""


class TestTuiWindowsFallback(unittest.TestCase):
    def test_tui_imports_and_degrades_without_termios(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        r = subprocess.run([sys.executable, "-c", _SCRIPT],
                           capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"tui failed on simulated Windows:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}")
        self.assertIn("WINDOWS_TUI_OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
