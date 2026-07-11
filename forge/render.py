"""Shared terminal-rendering helpers — one readable theme for every human-facing
CLI surface (status, receipts, learnings, trace, run).

The rules everything obeys: collapse to one line and fit the ACTUAL terminal
width with an ellipsis (never wrap into an unreadable block), and emit colour
only to a real TTY — suppressed under NO_COLOR or TERM=dumb — so piped or
redirected output stays clean plaintext.
"""
import os
import re
import shutil
import sys

_CODES = {"reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
          "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
          "blue": "\033[34m", "red": "\033[31m", "magenta": "\033[35m"}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def color_on():
    """True when stdout is an interactive terminal that wants colour."""
    return (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") != "dumb")


def paint(text, *styles):
    """Wrap `text` in the named ANSI styles — a no-op when colour is off."""
    if not styles or not color_on():
        return text
    return "".join(_CODES[s] for s in styles) + text + _CODES["reset"]


def strip_ansi(s):
    """The visible text with any ANSI escapes removed (for width math)."""
    return _ANSI_RE.sub("", s)


def term_width(default=100):
    return shutil.get_terminal_size((default, 24)).columns


def tilde(path):
    """Collapse the home prefix to ~ for a shorter, readable path."""
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path and path.startswith(home) else (path or "")


def fit(text, width):
    """Collapse internal whitespace and hard-truncate to `width` columns, adding
    an ellipsis when the text is clipped. Returns '' for a non-positive width."""
    text = " ".join((text or "").split())
    if width < 1:
        return ""
    return text if len(text) <= width else text[:width - 1].rstrip() + "…"
