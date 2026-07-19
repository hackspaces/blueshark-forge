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
import threading

try:
    import termios
    import tty
except ImportError:                 # native Windows has no POSIX terminal control —
    termios = tty = None            # _supported() goes False, so every raw-mode path
                                    # below falls back to plain input() and none of the
                                    # termios.*/tty.* calls (all gated on it) are reached.

DIM = "\033[2m"; GR = "\033[32m"; CY = "\033[36m"; MG = "\033[35m"; RST = "\033[0m"

ESC, CTRL_C, CTRL_D, CR, LF, BS1, BS2 = b"\x1b", b"\x03", b"\x04", b"\r", b"\n", b"\x7f", b"\x08"
CTRL_A, CTRL_E, CTRL_K, CTRL_U, CTRL_W = b"\x01", b"\x05", b"\x0b", b"\x15", b"\x17"
_ANSI = re.compile(r"\033\[[0-9;]*m")

WAKE = object()   # prompt() returns this when the wake fd fires (idle inbox arrival)


def _supported():
    # termios is None on native Windows; a Windows console still reports isatty()==True,
    # so without this check the raw-mode editor would run and hit termios.tcgetattr(None).
    return termios is not None and sys.stdin.isatty() and sys.stdout.isatty()


# Display width + clip come from the shared render engine (wcwidth-correct: CJK/emoji = 2
# columns, combining/zero-width = 0), so the footer box and every column align on real
# terminal columns, not code points — the bug that drifted the border on any non-ASCII input.
from . import render as _render


def _vis(s):
    return _render.display_width(s)


def _clip(s, width):
    """Truncate to `width` display columns, keeping ANSI codes intact (shared engine)."""
    return _render.clip(s, width)


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
    decoded = ch.decode("utf-8", "ignore")
    # An undecodable byte (bad paste, mis-set locale) decodes to "" — which callers
    # would conflate with the b"" EOF sentinel and QUIT the session. Skip it and read
    # the next key instead of signalling EOF.
    return decoded if decoded else _read_key(fd)


def _read_key_or_wake(fd, wake_fd):
    """Like _read_key, but also watches a wake pipe: blocks in select([fd, wake_fd])
    and returns the WAKE sentinel when only the wake fd is readable. Stdin is
    serviced first, so a keypress racing a wake is never dropped. With wake_fd None
    this is a plain blocking _read_key (zero behaviour change)."""
    if wake_fd is None:
        return _read_key(fd)
    while True:
        try:
            r, _, _ = select.select([fd, wake_fd], [], [])
        except (InterruptedError, OSError):
            return _read_key(fd)
        if fd in r:
            return _read_key(fd)
        if wake_fd in r:
            return WAKE


class LineEditor:
    """Editing state + key handling for one logical input line. Callers deal
    with the control keys that end editing (Enter, Ctrl-C/D, bare Esc); this
    handles movement, deletion, kills, and ↑/↓ history."""

    def __init__(self, history=(), initial=""):
        self.history = list(history)
        self.buf, self.cur = list(initial), len(initial)
        self.hidx, self.saved = len(self.history), ""

    def text(self):
        return "".join(self.buf)

    def clear(self):
        self.buf, self.cur = [], 0

    def handle(self, key):
        """Apply one key from _read_key. Returns True if state changed."""
        if isinstance(key, str):
            if key and key.isprintable():
                self.buf.insert(self.cur, key); self.cur += 1
                return True
            return False
        if key in (BS1, BS2):
            if self.cur > 0:
                del self.buf[self.cur - 1]; self.cur -= 1
                return True
            return False
        if key == CTRL_A:
            self.cur = 0; return True
        if key == CTRL_E:
            self.cur = len(self.buf); return True
        if key == CTRL_U:
            del self.buf[:self.cur]; self.cur = 0; return True
        if key == CTRL_K:
            del self.buf[self.cur:]; return True
        if key == CTRL_W:
            j = self.cur
            while j > 0 and self.buf[j - 1] == " ": j -= 1
            while j > 0 and self.buf[j - 1] != " ": j -= 1
            del self.buf[j:self.cur]; self.cur = j; return True
        if key.startswith(ESC):
            seq = key[1:]
            if seq == b"[A" and self.history and self.hidx > 0:
                if self.hidx == len(self.history): self.saved = self.text()
                self.hidx -= 1; self.buf = list(self.history[self.hidx]); self.cur = len(self.buf)
                return True
            if seq == b"[B" and self.hidx < len(self.history):
                self.hidx += 1
                self.buf = list(self.history[self.hidx]) if self.hidx < len(self.history) else list(self.saved)
                self.cur = len(self.buf); return True
            if seq == b"[C" and self.cur < len(self.buf):
                self.cur += 1; return True
            if seq == b"[D" and self.cur > 0:
                self.cur -= 1; return True
            if seq in (b"[H", b"[1~"):
                self.cur = 0; return True
            if seq in (b"[F", b"[4~"):
                self.cur = len(self.buf); return True
            if seq == b"[3~" and self.cur < len(self.buf):
                del self.buf[self.cur]; return True
        return False


class Explorer:
    """A Miller-column folder explorer inside the TUI — three panes like
    ranger/Finder: parent · current · preview. Dependency-free.

        ↑/↓ move · ← parent · →/Enter enter dir · Enter on a file PICKS it
        (returned as a path) · . toggles hidden files · q/Esc closes

    Runs on the terminal's alternate screen, so the conversation underneath
    is untouched and restored pixel-perfect on close."""

    CAP = 400            # entries listed per directory
    PREVIEW_BYTES = 4096

    def __init__(self, screen, root):
        self.s = screen
        self.root = os.path.abspath(root)
        self.cwd = self.root
        self.cursor = 0
        self.show_hidden = False

    def _entries(self, path):
        """[(name, is_dir)] — directories first, each group sorted."""
        try:
            names = sorted(os.listdir(path), key=str.lower)
        except OSError:
            return []
        if not self.show_hidden:
            names = [n for n in names if not n.startswith(".")]
        dirs, files = [], []
        for n in names:
            (dirs if os.path.isdir(os.path.join(path, n)) else files).append(n)
        return ([(n, True) for n in dirs] + [(n, False) for n in files])[:self.CAP]

    def _preview(self, path, is_dir, width, rows):
        if is_dir:
            return [f"{CY}{n}/{RST}" if d else n for n, d in self._entries(path)[:rows]]
        try:
            with open(path, "rb") as f:
                blob = f.read(self.PREVIEW_BYTES)
            if b"\0" in blob:
                size = os.path.getsize(path)
                return [f"{DIM}(binary · {size:,} bytes){RST}"]
            text = blob.decode("utf-8", "replace").expandtabs(4)
            return [ln[:width] for ln in text.splitlines()[:rows]]
        except OSError as e:
            return [f"{DIM}({e}){RST}"]

    def _col(self, lines, entries, width, rows, hi_index, scroll=0):
        """Render one pane's rows into `lines` (a list of per-row strings)."""
        for i in range(rows):
            j = i + scroll
            if j >= len(entries):
                cell = ""
            elif isinstance(entries[j], tuple):
                n, d = entries[j]
                label = _clip((n + "/") if d else n, width - 1)
                if j == hi_index:
                    cell = f"\033[7m{label}{RST}"
                else:
                    cell = f"{CY}{label}{RST}" if d else label
            else:
                cell = _clip(entries[j], width)
            lines[i].append(cell + " " * max(0, width - _vis(cell)))

    def _draw(self):
        w, h = self.s.w, self.s.h
        rows = max(3, h - 3)
        c1 = max(14, w * 22 // 100)
        c2 = max(18, w * 34 // 100)
        c3 = max(10, w - c1 - c2 - 4)
        ents = self._entries(self.cwd)
        self.cursor = min(self.cursor, max(0, len(ents) - 1))
        scroll = max(0, self.cursor - rows + 2)
        parent = os.path.dirname(self.cwd)
        p_ents = self._entries(parent) if parent != self.cwd else []
        p_hi = next((i for i, (n, d) in enumerate(p_ents) if n == os.path.basename(self.cwd)), -1)
        p_scroll = max(0, p_hi - rows + 2) if p_hi >= 0 else 0
        if ents:
            name, is_dir = ents[self.cursor]
            pv = self._preview(os.path.join(self.cwd, name), is_dir, c3, rows)
        else:
            pv = [f"{DIM}(empty){RST}"]

        lines = [[] for _ in range(rows)]
        self._col(lines, p_ents, c1, rows, p_hi, p_scroll)
        for ln in lines:
            ln.append(f"{DIM}│{RST} ")
        self._col(lines, ents, c2, rows, self.cursor, scroll)
        for ln in lines:
            ln.append(f"{DIM}│{RST} ")
        self._col(lines, pv, c3, rows, -1)

        out = [f"\033[1;1H\033[K{MG}{_clip(self.cwd, w - 20)}{RST}"
               + (f"  {DIM}(hidden shown){RST}" if self.show_hidden else "")]
        for i, ln in enumerate(lines):
            out.append(f"\033[{i + 2};1H\033[K" + "".join(ln)[:2048])
        out.append(f"\033[{h};1H\033[K{DIM}↑↓ move · ←→ navigate · Enter: pick file (@attach) · . hidden · q/Esc close{RST}")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def run(self):
        """Open the explorer; returns the picked file path (relative to the
        root when inside it) or None."""
        if not self.s.enabled:
            return None
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        sys.stdout.write("\033[?1049h\033[r\033[2J")     # alternate screen, full region
        self.s._redraw = self._draw
        try:
            tty.setraw(fd)
            self._draw()
            while True:
                key = _read_key(fd)
                if not key or key in ("q", "Q") or key in (ESC, CTRL_C, CTRL_D):
                    return None
                ents = self._entries(self.cwd)
                sel = ents[self.cursor] if ents and self.cursor < len(ents) else None
                if key == ".":
                    self.show_hidden = not self.show_hidden; self.cursor = 0
                elif isinstance(key, bytes) and key.endswith(b"[A"):
                    self.cursor = max(0, self.cursor - 1)
                elif isinstance(key, bytes) and key.endswith(b"[B"):
                    self.cursor = min(max(0, len(ents) - 1), self.cursor + 1)
                elif isinstance(key, bytes) and key.endswith(b"[D"):
                    parent = os.path.dirname(self.cwd)
                    if parent != self.cwd:
                        was = os.path.basename(self.cwd)
                        self.cwd, self.cursor = parent, 0
                        self.cursor = next((i for i, (n, d) in enumerate(self._entries(parent)) if n == was), 0)
                elif sel and (key in (CR, LF) or (isinstance(key, bytes) and key.endswith(b"[C"))):
                    name, is_dir = sel
                    full = os.path.join(self.cwd, name)
                    if is_dir:
                        self.cwd, self.cursor = full, 0
                    elif key in (CR, LF):                 # Enter on a file → pick it
                        try:
                            rel = os.path.relpath(full, self.root)
                            return rel if not rel.startswith("..") else full
                        except ValueError:
                            return full
                self._draw()
        finally:
            self.s._redraw = None
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\033[?1049l")               # back to the conversation
            if self.s._entered:
                sys.stdout.write(f"\0337\033[1;{self.s.h - self.s.footer}r\0338")
            sys.stdout.flush()


class ApprovalGate:
    """Cross-thread approval channel for manual mode: the agent thread asks
    (blocking), the UI thread answers with 'yes' / 'always' / 'no'."""

    def __init__(self):
        self._lock = threading.Lock()
        self._desc = None
        self._resp = None
        self._evt = threading.Event()

    def request(self, desc, stop_event=None):
        """Block until the user answers (or the agent is stopped → 'no')."""
        with self._lock:
            self._desc, self._resp = desc, None
            self._evt.clear()
        while not self._evt.wait(0.2):
            if stop_event is not None and stop_event.is_set():
                with self._lock:
                    self._desc = None
                return "no"
        with self._lock:
            self._desc = None
            return self._resp or "no"

    def pending(self):
        """The pending request's description, or None."""
        return self._desc

    def answer(self, resp):
        with self._lock:
            if self._desc is None:
                return False
            self._resp = resp
        self._evt.set()
        return True


class Screen:
    """A bottom-pinned terminal: the conversation scrolls in the top region while
    a fixed footer stays anchored at the bottom — the way Claude Code /
    htop-style TUIs do it, via a DECSTBM scroll region. Footer layout:

        ⠋ working… (12s)                      <- activity, in the output flow
        ╭──────────────────────────────╮
        │ ❯ your input, wrapping onto  │      <- a rounded box with two
        │ a second line as you type    │         writable rows
        ╰──────────────────────────────╯
        model · 9% context · /help            <- status

    All transcript output goes through `emit()`; the footer is painted with
    absolute positioning, wrapped in save/restore so it never disturbs the
    scroll cursor. Degrades to plain stdout when not a TTY."""

    def __init__(self):
        self.enabled = _supported()
        self._lock = threading.Lock()   # serialize stdout between the agent + the spinner
        self._redraw = None             # repaint hook, set while the editor is live
        self._entered = False
        self._activity = ""             # the 'working…' line above the box
        self._resize()
        if self.enabled:
            try:
                signal.signal(signal.SIGWINCH, self._winch)
            except (ValueError, OSError):                 # not the main thread
                pass

    def _resize(self):
        size = shutil.get_terminal_size((80, 24))
        self.w, self.h = size.columns, size.lines
        self.rows = 2 if self.h >= 14 else 1          # writable lines in the box
        self.footer = self.rows + 4                   # activity · top · rows · bottom · status

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

    def _layout(self, plen, text, cur):
        """Wrap the logical input across the box's writable rows, keeping the
        cursor visible. Returns (segments, cursor_row, cursor_col) where col is
        an offset into the row's content area."""
        inner = max(8, self.w - 4)                    # inside '│ ' … ' │'
        caps = [inner - plen] + [inner] * (self.rows - 1)
        total = sum(caps)
        start = max(0, cur - (total - 1))             # scroll left if overlong
        vis = text[start:start + total]
        segs, i = [], 0
        for c in caps:
            segs.append(vis[i:i + c]); i += c
        rel = cur - start
        if rel < caps[0] or self.rows == 1:
            return segs, 0, plen + min(rel, caps[0])
        return segs, 1 + (rel - caps[0]) // inner, (rel - caps[0]) % inner

    def _paint(self, prompt, text, status, cur=None, dim=False):
        """Repaint the footer. `text` is the plain logical input line; it wraps
        across the box's writable rows."""
        base = self.h - self.footer + 1
        plen = _vis(prompt)
        segs, crow, ccol = self._layout(plen, text, len(text) if cur is None else cur)
        style = DIM if dim else ""
        lines = [_clip(f"{MG}{self._activity}{RST}" if self._activity else "", self.w),
                 f"{DIM}╭{'─' * (self.w - 2)}╮{RST}"]
        pad0 = max(0, (self.w - 4) - plen - len(segs[0]))
        lines.append(f"{DIM}│{RST} {prompt}{style}{segs[0]}{RST}{' ' * pad0} {DIM}│{RST}")
        for s in segs[1:]:
            lines.append(f"{DIM}│{RST} {style}{s}{RST}{' ' * max(0, (self.w - 4) - len(s))} {DIM}│{RST}")
        lines.append(f"{DIM}╰{'─' * (self.w - 2)}╯{RST}")
        lines.append(_clip(f"{DIM}{status}{RST}", self.w))
        for i in range(self.footer):
            sys.stdout.write(f"\033[{base + i};1H\033[K")
            if i < len(lines):
                sys.stdout.write(lines[i])
        if cur is not None:
            sys.stdout.write(f"\033[{base + 2 + crow};{3 + ccol}H")
        sys.stdout.flush()

    def set_status(self, status):
        """Update just the footer status row (the last footer line)."""
        if not self.enabled:
            return
        base = self.h - self.footer + 1
        with self._lock:
            sys.stdout.write("\0337" + f"\033[{base + self.footer - 1};1H\033[K{DIM}{status[:self.w]}{RST}" + "\0338")
            sys.stdout.flush()

    def set_activity(self, text):
        """The 'working…' line — pinned just above the input rule, adjacent to
        where output streams (not below the input)."""
        self._activity = text
        if not self.enabled or self.footer < 4:
            return
        base = self.h - self.footer + 1
        with self._lock:
            sys.stdout.write("\0337" + f"\033[{base};1H\033[K" + _clip(f"{MG}{text}{RST}" if text else "", self.w) + "\0338")
            sys.stdout.flush()

    def prompt(self, prompt, history, status="", on_mode=None, initial="", wake_fd=None):
        """Read one line in the pinned footer (raw-mode editor). Esc clears; ↑/↓
        history; ←/→/Home/End move; Delete deletes forward; Ctrl-A/E home/end;
        Ctrl-U/K/W kill line-start/line-end/word; Shift-Tab cycles the mode
        (via `on_mode`); Enter submits; Ctrl-C clears then exits; Ctrl-D exits.
        `status` may be a string or a zero-arg callable (repainted live);
        `initial` pre-fills the line (e.g. an @file picked in the explorer).
        When `wake_fd` is given, a byte on it returns the WAKE sentinel so an idle
        REPL can render/act on a fleet message the instant it arrives."""
        if not self.enabled:
            try:
                return input(_ANSI.sub("", prompt))
            except EOFError:
                return None
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        ed = LineEditor(history, initial)
        sys.stdout.write("\0337")   # save the scroll-region cursor for the duration of editing

        def draw():
            self._paint(prompt, ed.text(), status() if callable(status) else status, cur=ed.cur)

        self._redraw = draw
        try:
            tty.setraw(fd)
            draw()
            while True:
                key = _read_key_or_wake(fd, wake_fd)
                if key is WAKE:                              # inbox arrived while idle
                    return WAKE
                if not key or key == CTRL_D:                 # EOF / Ctrl-D
                    return None
                if key in (CR, LF):
                    return ed.text()
                if key == CTRL_C:
                    if ed.text():
                        ed.clear(); draw(); continue
                    return None
                if key == ESC:                               # bare Esc → clear
                    ed.clear(); draw(); continue
                if isinstance(key, bytes) and key.endswith(b"[Z") and on_mode:   # shift-tab
                    on_mode(); draw(); continue
                if ed.handle(key):
                    draw()
        finally:
            self._redraw = None
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\0338")   # restore the scroll-region cursor so emit() flows above
            sys.stdout.flush()

    def attend(self, fn, stop_event, prompt, history, status="", on_queue=None, gate=None, on_hint=None):
        """Run fn() in a worker thread while the footer stays ALIVE:

          - typing goes into the input box; Enter hands the text to `on_queue`
            (a message queued for the running agent)
          - when `gate` has a pending approval, y / a / n answer it
            (yes / always—don't ask again / no); Esc answers no
          - Esc clears typed text; on an empty box it stops the agent
            gracefully; pressed again it force-returns the prompt
          - Ctrl-C behaves like Esc, and never raises a bare traceback

        Returns fn()'s result ('(stopped)' when force-returned)."""
        if not self.enabled:
            return fn()
        result, err, done = [None], [None], threading.Event()

        def work():
            try:
                result[0] = fn()
            except BaseException as e:   # re-raised on the caller's thread
                err[0] = e
            finally:
                done.set()

        threading.Thread(target=work, daemon=True).start()
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        ed = LineEditor(history)
        hinted = forced = False

        def draw():
            with self._lock:
                sys.stdout.write("\0337")
                self._paint(prompt, ed.text(), status() if callable(status) else status, cur=ed.cur)
                sys.stdout.write("\0338")
                sys.stdout.flush()

        try:
            # cbreak with ISIG off: Ctrl-C arrives as the \x03 byte we handle
            # (graceful stop), output processing stays on for the transcript.
            tty.setcbreak(fd)
            attrs = termios.tcgetattr(fd)
            attrs[3] &= ~termios.ISIG
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            draw()
            while not done.is_set():
                try:
                    r, _, _ = select.select([fd], [], [], 0.1)
                except KeyboardInterrupt:       # SIGINT racing the ISIG switch-off
                    stop_event.set(); continue
                if not r:
                    continue
                key = _read_key(fd)
                if not key:
                    continue
                if gate and gate.pending():     # a pending approval owns y/a/n + Esc
                    if key in ("y", "Y", CR, LF):
                        gate.answer("yes")
                    elif key in ("a", "A"):
                        gate.answer("always")
                    elif key in ("n", "N") or key in (ESC, CTRL_C):
                        gate.answer("no")
                    continue
                if key in (ESC, CTRL_C):
                    if ed.text():
                        ed.clear(); draw(); continue
                    if stop_event.is_set():     # second press — give the prompt back now
                        forced = True
                        break
                    if not hinted and on_hint:
                        on_hint(); hinted = True
                    stop_event.set(); continue
                if key in (CR, LF):
                    txt = ed.text().strip()
                    ed = LineEditor(history)
                    if txt and on_queue:
                        on_queue(txt)
                    draw(); continue
                if key == CTRL_D:
                    continue                    # no exit mid-run
                if ed.handle(key):
                    draw()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            done.wait(timeout=0.5 if forced else 5)
        if forced and not done.is_set():
            return "(stopped)"                  # abandon the stuck step; the daemon thread dies with it
        if err[0] is not None:
            raise err[0]
        return result[0]


class FooterSpinner:
    """Animate the 'working…' spinner in the activity row — just above the
    input box, adjacent to the streaming output. While an approval is pending
    (manual mode), the row shows the y/a/n question instead."""
    def __init__(self, screen, label="thinking", gate=None):
        self.screen = screen; self.label = label; self.gate = gate
        self._stop = False; self._t = None
    def start(self):
        import time
        start = time.monotonic()
        def spin():
            for c in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if self._stop:
                    break
                pend = self.gate.pending() if self.gate else None
                if pend:
                    self.screen.set_activity(f"● allow {pend}?   [y]es · [a]lways (don't ask again) · [n]o")
                else:
                    el = time.monotonic() - start
                    t = f"{el:.0f}s" if el < 60 else f"{int(el // 60)}m {int(el % 60)}s"
                    self.screen.set_activity(f"{c} {self.label}… ({t})   ·   Esc to stop · type to queue a message")
                time.sleep(0.1)
        self._t = threading.Thread(target=spin, daemon=True); self._t.start(); return self
    def stop(self):
        self._stop = True
        if self._t:
            self._t.join()
        self.screen.set_activity("")


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
