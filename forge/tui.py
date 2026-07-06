"""Terminal UX: a raw-mode line editor (Esc clears the line, arrows, history) and
an interrupt watcher so Esc stops the agent mid-run. Dependency-free (termios)."""
import os
import select
import sys
import termios
import threading
import tty

DIM = "\033[2m"; GR = "\033[32m"; RST = "\033[0m"

ESC, CTRL_C, CTRL_D, CR, LF, BS1, BS2 = b"\x1b", b"\x03", b"\x04", b"\r", b"\n", b"\x7f", b"\x08"


def _supported():
    return sys.stdin.isatty() and sys.stdout.isatty()


def read_line(prompt, history):
    """Read one line in raw mode. Esc clears the current line; ↑/↓ walk history;
    ←/→ move; Enter submits; Ctrl-C clears (or exits if empty); Ctrl-D exits.
    Returns the string, or None to quit. Falls back to input() if not a TTY."""
    if not _supported():
        try:
            return input(prompt)
        except EOFError:
            return None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf, cur, hidx = [], 0, len(history)
    saved = ""

    def redraw():
        sys.stdout.write("\r\033[K" + prompt + "".join(buf))
        # move cursor back to position
        back = len(buf) - cur
        if back:
            sys.stdout.write(f"\033[{back}D")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        redraw()
        while True:
            ch = os.read(fd, 1)
            if ch in (CR, LF):
                sys.stdout.write("\r\n"); sys.stdout.flush()
                return "".join(buf)
            if ch == CTRL_D:
                sys.stdout.write("\r\n"); sys.stdout.flush()
                return None
            if ch == CTRL_C:
                if buf:
                    buf, cur = [], 0; redraw(); continue
                sys.stdout.write("\r\n"); sys.stdout.flush()
                return None
            if ch == ESC:
                # could be a bare Esc (clear) or an arrow sequence
                r, _, _ = select.select([fd], [], [], 0.02)
                if not r:
                    buf, cur = [], 0; redraw(); continue     # bare Esc → clear line
                seq = os.read(fd, 2)
                if seq == b"[A":                              # up → older history
                    if history and hidx > 0:
                        if hidx == len(history): saved = "".join(buf)
                        hidx -= 1; buf = list(history[hidx]); cur = len(buf); redraw()
                elif seq == b"[B":                            # down → newer
                    if hidx < len(history):
                        hidx += 1
                        buf = list(history[hidx]) if hidx < len(history) else list(saved)
                        cur = len(buf); redraw()
                elif seq == b"[C":                            # right
                    if cur < len(buf): cur += 1; redraw()
                elif seq == b"[D":                            # left
                    if cur > 0: cur -= 1; redraw()
                continue
            if ch in (BS1, BS2):
                if cur > 0:
                    del buf[cur - 1]; cur -= 1; redraw()
                continue
            try:
                c = ch.decode("utf-8", "ignore")
            except Exception:
                continue
            if c and c.isprintable():
                buf.insert(cur, c); cur += 1; redraw()
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
