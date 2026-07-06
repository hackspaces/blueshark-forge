"""Terminal UX: a bottom-pinned screen (DECSTBM scroll region) with a raw-mode
line editor in the footer (Esc clears, arrows, Home/End, history, readline-style
Ctrl keys, UTF-8 input) and an interrupt watcher so Esc stops the agent mid-run.
Dependency-free (termios). Falls back to plain input() when not a TTY."""
import itertools
import os
import re
import select
import shutil
import signal
import sys
import termios
import threading
import tty

DIM = "\033[2m"; GR = "\033[32m"; MG = "\033[35m"; RST = "\033[0m"

ESC, CTRL_C, CTRL_D, CR, LF, BS1, BS2 = b"\x1b", b"\x03", b"\x04", b"\r", b"\n", b"\x7f", b"\x08"
CTRL_A, CTRL_E, CTRL_K, CTRL_U, CTRL_W = b"\x01", b"\x05", b"\x0b", b"\x15", b"\x17"
_ANSI = re.compile(r"\033\[[0-9;]*m")


def _supported():
    return sys.stdin.isatty() and sys.stdout.isatty()


def _vis(s):
    return len(_ANSI.sub("", s))


def _clip(s, width):
    """Truncate to `width` visible columns, keeping ANSI codes intact."""
    out, n, i = [], 0, 0
    while i < len(s):
        m = _ANSI.match(s, i)
        if m:
            out.append(m.group()); i = m.end(); continue
        if n >= width:
            return "".join(out) + RST
        out.append(s[i]); n += 1; i += 1
    return "".join(out)


def _read_key(fd):
    """Read one keypress: bytes for a control key or a whole escape sequence,
    str for a decoded (possibly multi-byte) printable character, b"" on EOF."""
    ch = os.read(fd, 1)
    if not ch:
        return b""
    if ch == ESC:
        r, _, _ = select.select([fd], [], [], 0.02)
        if not r:
            return ESC                                    # bare Esc
        seq = os.read(fd, 1)
        if seq != b"[":
            return ESC + seq                              # Alt-<key> — callers ignore
        while len(seq) < 8:                               # CSI: params, then a final byte
            c = os.read(fd, 1)
            if not c:
                break
            seq += c
            if c.isalpha() or c == b"~":
                break
        return ESC + seq
    b0 = ch[0]
    if b0 < 0x20 or b0 == 0x7F:
        return ch                                         # control byte
    n = 2 if 0xC0 <= b0 < 0xE0 else 3 if 0xE0 <= b0 < 0xF0 else 4 if 0xF0 <= b0 < 0xF8 else 1
    if n > 1:
        ch += os.read(fd, n - 1)                          # UTF-8 continuation bytes
    return ch.decode("utf-8", "ignore")


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
        self._redraw = None             # repaint hook, set while the editor is live
        self._entered = False
        self._resize()
        if self.enabled:
            try:
                signal.signal(signal.SIGWINCH, self._winch)
            except (ValueError, OSError):                 # not the main thread
                pass

    def _resize(self):
        size = shutil.get_terminal_size((80, 24))
        self.w, self.h = size.columns, size.lines

    def _winch(self, *_):
        """Terminal resized: re-pin the scroll region to the new height and
        repaint the footer (otherwise it drifts into the transcript)."""
        self._resize()
        if not self._entered:
            return
        sys.stdout.write("\0337" + f"\033[1;{self.h - self.footer}r" + "\0338")
        sys.stdout.flush()
        if self._redraw:
            try:
                self._redraw()
            except Exception:
                pass

    def enter(self):
        if not self.enabled:
            return
        self._resize()
        sys.stdout.write("\033[2J")                           # clear the screen
        sys.stdout.write(f"\033[1;{self.h - self.footer}r")   # scroll region = everything above the footer
        sys.stdout.write("\033[1;1H")                         # park the cursor at the TOP (content fills down)
        sys.stdout.flush()
        self._entered = True

    def exit(self):
        if not self.enabled:
            return
        self._entered = False
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
        rows = [rule, _clip(f"{prompt}{text}", self.w), _clip(f"{DIM}{status}{RST}", self.w)][:self.footer]
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
        history; ←/→/Home/End move; Delete deletes forward; Ctrl-A/E home/end;
        Ctrl-U/K/W kill line-start/line-end/word; Enter submits; Ctrl-C clears
        then exits; Ctrl-D exits."""
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
            text = "".join(buf)
            avail = max(4, self.w - plen - 1)
            start = max(0, cur - avail) if cur > avail else 0
            self._paint(prompt, text[start:start + avail], status, cursor_col=1 + plen + (cur - start))

        self._redraw = draw
        try:
            tty.setraw(fd)
            draw()
            while True:
                key = _read_key(fd)
                if isinstance(key, str):                     # printable (incl. multi-byte)
                    if key and key.isprintable():
                        buf.insert(cur, key); cur += 1; draw()
                    continue
                if not key or key == CTRL_D:                 # EOF / Ctrl-D
                    return None
                if key in (CR, LF):
                    return "".join(buf)
                if key == CTRL_C:
                    if buf:
                        buf, cur = [], 0; draw(); continue
                    return None
                if key == ESC:                               # bare Esc → clear
                    buf, cur = [], 0; draw(); continue
                if key in (BS1, BS2):
                    if cur > 0:
                        del buf[cur - 1]; cur -= 1; draw()
                    continue
                if key == CTRL_A:
                    cur = 0; draw(); continue
                if key == CTRL_E:
                    cur = len(buf); draw(); continue
                if key == CTRL_U:
                    del buf[:cur]; cur = 0; draw(); continue
                if key == CTRL_K:
                    del buf[cur:]; draw(); continue
                if key == CTRL_W:
                    j = cur
                    while j > 0 and buf[j - 1] == " ": j -= 1
                    while j > 0 and buf[j - 1] != " ": j -= 1
                    del buf[j:cur]; cur = j; draw(); continue
                seq = key[1:]                                # CSI sequence, ESC stripped
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
                elif seq in (b"[H", b"[1~"):
                    cur = 0; draw()
                elif seq in (b"[F", b"[4~"):
                    cur = len(buf); draw()
                elif seq == b"[3~" and cur < len(buf):
                    del buf[cur]; draw()
        finally:
            self._redraw = None
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
    forced = False
    try:
        # cbreak, but with ISIG off: Ctrl-C arrives as the \x03 byte we handle
        # (a graceful stop) instead of raising KeyboardInterrupt mid-run.
        # Output processing stays on so the agent's transcript renders normally.
        tty.setcbreak(fd)
        attrs = termios.tcgetattr(fd)
        attrs[3] &= ~termios.ISIG
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        while not done.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.1)
            except KeyboardInterrupt:       # SIGINT racing the ISIG switch-off
                stop_event.set(); continue
            if not r:
                continue
            ch = os.read(fd, 1)
            if ch == ESC:
                r, _, _ = select.select([fd], [], [], 0.02)
                if r:                       # escape *sequence* (arrow key etc) — drain, ignore
                    while select.select([fd], [], [], 0)[0]:
                        os.read(fd, 32)
                    continue
            if ch in (ESC, CTRL_C):
                if stop_event.is_set():     # second press — give the prompt back now
                    forced = True
                    break
                if not hinted and on_hint:
                    on_hint()
                    hinted = True
                stop_event.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        done.wait(timeout=0.5 if forced else 5)
    if forced and not done.is_set():
        return "(stopped)"                  # abandon the stuck step; the daemon thread dies with it
    if err[0] is not None:
        raise err[0]
    return result[0]
