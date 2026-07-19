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
import unicodedata

_CODES = {"reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
          "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
          "blue": "\033[34m", "red": "\033[31m", "magenta": "\033[35m"}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


# ---- display width (the science of terminal spacing) -------------------------
# A terminal aligns by DISPLAY COLUMNS, not code points, so len() is wrong the moment
# text carries a CJK char (2 cols), an emoji (2), a combining accent (0), or a zero-width
# joiner. Every box border and column that used len() drifted on such input. This is a
# pragmatic stdlib wcwidth — no dependency, correct for the cases CLI output actually hits.
_ZERO = {0x200b, 0x200d, 0xfeff}                 # ZW space, ZW joiner, BOM
def _char_width(ch):
    o = ord(ch)
    if o == 0 or o < 32 or 0x7f <= o < 0xa0:      # NUL / C0 / C1 controls: no print width
        return 0
    if o in _ZERO or 0xfe00 <= o <= 0xfe0f:       # zero-width + variation selectors
        return 0
    if unicodedata.combining(ch):                 # combining marks (accents) add nothing
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):   # CJK wide / fullwidth
        return 2
    if 0x1f000 <= o <= 0x1faff or 0x2600 <= o <= 0x27bf:  # emoji blocks (EAW is inconsistent here)
        return 2
    return 1


def display_width(s):
    """Visible width of `s` in terminal COLUMNS: ANSI stripped, wide chars = 2, combining
    and zero-width = 0, and a ZWJ sequence (emoji family) collapses to one glyph's width."""
    total, joined = 0, False
    for ch in strip_ansi(s):
        if ord(ch) == 0x200d:                     # ZWJ: the next glyph merges into this cluster
            joined = True
            continue
        w = 0 if joined else _char_width(ch)
        joined = False
        total += w
    return total


def clip(s, cols):
    """Truncate `s` to `cols` DISPLAY columns, keeping ANSI codes intact and never
    splitting a wide char across the edge. Appends a reset if it cut inside styled text."""
    out, used, i, cut = [], 0, 0, False
    while i < len(s):
        m = _ANSI_RE.match(s, i)
        if m:
            out.append(m.group()); i = m.end(); continue
        w = _char_width(s[i])
        if used + w > cols:
            cut = True
            break
        out.append(s[i]); used += w; i += 1
    tail = _CODES["reset"] if (cut and "\033[" in "".join(out)) else ""
    return "".join(out) + tail


def color_on():
    """True when stdout is an interactive terminal that wants colour."""
    return (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") != "dumb")


def color_depth():
    """What the terminal can render: 'truecolor' (24-bit), '256', '16', or 'none'.
    Lets richer surfaces (the model-card foils) pick a palette that degrades gracefully
    instead of assuming 24-bit everywhere."""
    if not color_on():
        return "none"
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    term = os.environ.get("TERM", "")
    if "256" in term or "truecolor" in term:
        return "256"
    return "16"


_VT_DONE = False
def enable_vt():
    """Turn on ANSI/VT processing on the Windows console so escape codes render as colour
    instead of literal garbage (Win10+ needs ENABLE_VIRTUAL_TERMINAL_PROCESSING set
    explicitly). No-op on POSIX and idempotent, so it is safe to call at every startup."""
    global _VT_DONE
    if _VT_DONE or os.name != "nt":
        _VT_DONE = True
        return
    _VT_DONE = True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        for handle in (-11, -12):                     # STDOUT, STDERR
            h = k.GetStdHandle(handle)
            mode = ctypes.c_uint32()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)   # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass                                          # a console that refuses VT just keeps its own handling


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
    """Collapse internal whitespace and hard-truncate to `width` display COLUMNS, adding
    an ellipsis when clipped. Column-accurate: a CJK/emoji char counts as 2, so the result
    never overruns the width the way a code-point count did. '' for a non-positive width."""
    text = " ".join((text or "").split())
    if width < 1:
        return ""
    if display_width(text) <= width:
        return text
    return clip(text, width - 1).rstrip() + "…"      # reserve one column for the ellipsis
