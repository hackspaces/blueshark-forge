"""Terminal UX: a raw-mode line editor drawn in a bordered box (Esc clears the
line, arrows, history) and an interrupt watcher so Esc stops the agent mid-run.
Dependency-free (termios). Falls back to plain input() when not a TTY."""
import itertools
import os
import re
import select
import shutil
import sys
import termios
import threading
import tty

DIM = "\033[2m"; GR = "\033[32m"; MG = "\033[35m"; RST = "\033[0m"

ESC, CTRL_C, CTRL_D, CR, LF, BS1, BS2 = b"\x1b", b"\x03", b"\x04", b"\r", b"\n", b"\x7f", b"\x08"
_ANSI = re.compile(r"\033\[[0-9;]*m")


def _supported():
    return sys.stdin.isatty() and sys.stdout.isatty()


def _vis(s):
    return len(_ANSI.sub("", s))


class Screen:
    """A bottom-pinned terminal: the conversation scrolls in the top region while
    a fixed footer (rule · prompt · status) stays anchored at the bottom — the way
    Claude Code / htop-style TUIs do it, via a DECSTBM scroll region.

    All transcript output goes through `emit()`; the footer is painted with
    absolute positioning, wrapped in save/restore so it never disturbs the
    scroll cursor. Degrades to plain stdout when not a TTY."""

    def __init__(self, footer=3):
        self.footer = footer
        self.enabled = _supported()
        self._lock = threading.Lock()   # serialize stdout between the agent + the spinner
        self._resize()

    def _resize(self):
        size = shutil.get_terminal_size((80, 24))
        self.w, self.h = size.columns, size.lines

    def enter(self):
        if not self.enabled:
            return
        self._resize()
        sys.stdout.write("\033[2J")                           # clear the screen
        sys.stdout.write(f"\033[1;{self.h - self.footer}r")   # scroll region = everything above the footer
        sys.stdout.write("\033[1;1H")                         # park the cursor at the TOP (content fills down)
        sys.stdout.flush()

    def exit(self):
        if not self.enabled:
            return
        sys.stdout.write("\033[r")                            # release the scroll region
        sys.stdout.write(f"\033[{self.h};1H\n")
        sys.stdout.flush()

    def emit(self, text):
        with self._lock:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _paint(self, prompt, text, status, cursor_col=None):
        base = self.h - self.footer + 1
        rule = f"{DIM}{'─' * self.w}{RST}"
        rows = [rule, f"{prompt}{text}", f"{DIM}{status}{RST}"][:self.footer]
        for i in range(self.footer):
            sys.stdout.write(f"\033[{base + i};1H\033[K")
            if i < len(rows):
                sys.stdout.write(rows[i])
        if cursor_col is not None:
            sys.stdout.write(f"\033[{base + 1};{cursor_col}H")
        sys.stdout.flush()

    def set_status(self, status):
        """Update just the footer status row (used by the running spinner)."""
        if not self.enabled:
            return
        base = self.h - self.footer + 1
        with self._lock:
            sys.stdout.write("\0337" + f"\033[{base + 2};1H\033[K{DIM}{status[:self.w]}{RST}" + "\0338")
            sys.stdout.flush()

    def show_submitted(self, prompt, text):
        """Keep the submitted line + a placeholder status visible while the agent runs."""
        if not self.enabled:
            return
        sys.stdout.write("\0337")
        self._paint(prompt, f"{DIM}{text}{RST}", "")
        sys.stdout.write("\0338")
        sys.stdout.flush()

    def prompt(self, prompt, history, status=""):
        """Read one line in the pinned footer (raw-mode editor). Esc clears; ↑/↓
        history; ←/→ move; Enter submits; Ctrl-C clears then exits; Ctrl-D exits."""
        if not self.enabled:
            try:
                return input(_ANSI.sub("", prompt))
            except EOFError:
                return None
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        buf, cur, hidx, saved = [], 0, len(history), ""
        plen = _vis(prompt)
        sys.stdout.write("\0337")   # save the scroll-region cursor for the duration of editing

        def draw():
            self._resize()
            text = "".join(buf)
            avail = max(4, self.w - plen - 1)
            start = max(0, cur - avail) if cur > avail else 0
            self._paint(prompt, text[start:start + avail], status, cursor_col=1 + plen + (cur - start))

        try:
            tty.setraw(fd)
            draw()
            while True:
                ch = os.read(fd, 1)
                if ch in (CR, LF):
                    return "".join(buf)
                if ch == CTRL_D:
                    return None
                if ch == CTRL_C:
                    if buf:
                        buf, cur = [], 0; draw(); continue
                    return None
                if ch == ESC:
                    r, _, _ = select.select([fd], [], [], 0.02)
                    if not r:
                        buf, cur = [], 0; draw(); continue
                    seq = os.read(fd, 2)
                    if seq == b"[A" and history and hidx > 0:
                        if hidx == len(history): saved = "".join(buf)
                        hidx -= 1; buf = list(history[hidx]); cur = len(buf); draw()
                    elif seq == b"[B" and hidx < len(history):
                        hidx += 1
                        buf = list(history[hidx]) if hidx < len(history) else list(saved)
                        cur = len(buf); draw()
                    elif seq == b"[C" and cur < len(buf):
                        cur += 1; draw()
                    elif seq == b"[D" and cur > 0:
                        cur -= 1; draw()
                    continue
                if ch in (BS1, BS2):
                    if cur > 0:
                        del buf[cur - 1]; cur -= 1; draw()
                    continue
                try:
                    c = ch.decode("utf-8", "ignore")
                except Exception:
                    continue
                if c and c.isprintable():
                    buf.insert(cur, c); cur += 1; draw()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\0338")   # restore the scroll-region cursor so emit() flows above
            sys.stdout.flush()


class FooterSpinner:
    """Animate a spinner + elapsed time in the pinned footer's status row."""
    def __init__(self, screen, label="thinking"):
        self.screen = screen; self.label = label; self._stop = False; self._t = None
    def start(self):
        import time
        start = time.monotonic()
        def spin():
            for c in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if self._stop:
                    break
                el = time.monotonic() - start
                t = f"{el:.0f}s" if el < 60 else f"{int(el // 60)}m {int(el % 60)}s"
                self.screen.set_status(f"{c} {self.label}… ({t})   ·   Esc to stop")
                time.sleep(0.1)
        self._t = threading.Thread(target=spin, daemon=True); self._t.start(); return self
    def stop(self):
        self._stop = True
        if self._t:
            self._t.join()


def read_line(prompt, history, status=""):
    """Read one line, drawn in a rounded box with an optional dim status line
    above it. Esc clears (or, mid-run, stops); ↑/↓ history; ←/→ move; Enter
    submits; Ctrl-C clears then exits; Ctrl-D exits. Returns the string or None."""
    if not _supported():
        try:
            return input(_ANSI.sub("", prompt))
        except EOFError:
            return None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf, cur, hidx, saved = [], 0, len(history), ""
    drawn = [False]

    if status:
        print(f"{DIM}{status}{RST}")

    def render():
        W = max(24, shutil.get_terminal_size((80, 24)).columns)
        plen = _vis(prompt)
        avail = max(4, W - 4 - plen)
        text = "".join(buf)
        start = max(0, cur - avail) if cur > avail else 0
        vis = text[start:start + avail]
        vcur = cur - start
        top = "╭" + "─" * (W - 2) + "╮"
        content = f"│ {prompt}{vis}"
        mid = content + " " * max(0, W - 1 - _vis(content)) + "│"
        bot = "╰" + "─" * (W - 2) + "╯"
        if drawn[0]:
            sys.stdout.write("\r\033[2A")          # up to the top border
        sys.stdout.write(f"\r\033[K{DIM}{top}\r\n\033[K{RST}{mid}{DIM}\r\n\033[K{bot}{RST}")
        drawn[0] = True
        sys.stdout.write("\033[1A\r")              # back up to the input line
        col = 2 + plen + vcur
        if col:
            sys.stdout.write(f"\033[{col}C")
        sys.stdout.flush()

    def finish():
        sys.stdout.write("\033[1B\r\n"); sys.stdout.flush()   # move below the box

    try:
        tty.setraw(fd)
        render()
        while True:
            ch = os.read(fd, 1)
            if ch in (CR, LF):
                finish(); return "".join(buf)
            if ch == CTRL_D:
                finish(); return None
            if ch == CTRL_C:
                if buf:
                    buf, cur = [], 0; render(); continue
                finish(); return None
            if ch == ESC:
                r, _, _ = select.select([fd], [], [], 0.02)
                if not r:
                    buf, cur = [], 0; render(); continue      # bare Esc → clear
                seq = os.read(fd, 2)
                if seq == b"[A":
                    if history and hidx > 0:
                        if hidx == len(history): saved = "".join(buf)
                        hidx -= 1; buf = list(history[hidx]); cur = len(buf); render()
                elif seq == b"[B":
                    if hidx < len(history):
                        hidx += 1
                        buf = list(history[hidx]) if hidx < len(history) else list(saved)
                        cur = len(buf); render()
                elif seq == b"[C":
                    if cur < len(buf): cur += 1; render()
                elif seq == b"[D":
                    if cur > 0: cur -= 1; render()
                continue
            if ch in (BS1, BS2):
                if cur > 0:
                    del buf[cur - 1]; cur -= 1; render()
                continue
            try:
                c = ch.decode("utf-8", "ignore")
            except Exception:
                continue
            if c and c.isprintable():
                buf.insert(cur, c); cur += 1; render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_interruptible(fn, stop_event, on_hint=None):
    """Run fn() in a thread; while it runs, watch the keyboard and set stop_event
    on Esc or Ctrl-C so the agent bails gracefully. Returns fn()'s result."""
    if not _supported():
        return fn()
    result = [None]
    err = [None]
    done = threading.Event()

    def work():
        try:
            result[0] = fn()
        except BaseException as e:  # capture, re-raise on the caller's thread
            err[0] = e
        finally:
            done.set()

    threading.Thread(target=work, daemon=True).start()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    hinted = False
    try:
        tty.setcbreak(fd)
        while not done.is_set():
            r, _, _ = select.select([fd], [], [], 0.1)
            if not r:
                continue
            ch = os.read(fd, 1)
            if ch in (ESC, CTRL_C):
                if not hinted and on_hint:
                    on_hint()
                    hinted = True
                stop_event.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        done.wait(timeout=5)
    if err[0] is not None:
        raise err[0]
    return result[0]
