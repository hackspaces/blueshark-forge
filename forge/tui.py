"""Terminal UX: a raw-mode line editor drawn in a bordered box (Esc clears the
line, arrows, history) and an interrupt watcher so Esc stops the agent mid-run.
Dependency-free (termios). Falls back to plain input() when not a TTY."""
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
