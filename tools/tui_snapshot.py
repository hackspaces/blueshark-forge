#!/usr/bin/env python3
"""Dev tool (NOT shipped, NOT a CI test) — render forge's TUI footer to a text grid so a
developer, or an agent, can SEE the terminal UI and verify it, not just its width math.

It runs a tiny driver on a fixed-size pseudo-terminal, feeds the emitted ANSI through a VT
emulator (pyte), and prints the resolved character grid plus a COLUMN-ACCURATE box-alignment
check (eyeballing a printed grid lies — a wide char occupies two cells but renders two
columns, so the trailing border only *looks* misplaced; this checks the actual cells).

Requires pyte, which is dev-only:  pip install pyte
Usage:  python tools/tui_snapshot.py [cols]      (default 92)
"""
import os, pty, sys, struct, fcntl, termios, subprocess, select, time

DRIVER = r'''
import sys, time
sys.path.insert(0, %r)
from forge.tui import Screen
s = Screen()
s.enter()
s.emit("  forge - session\n")
s.emit("  * read_file: cafe.py       [ok]\n")
s.emit("  * edit_file: 日本語.py  \U0001f680   [ok]\n")
s.set_activity("working...")
s._paint("❯ ", "fix the cafe 日本語 bug \U0001f680 now",
         "  auto - qwen2.5-coder:7b - 12%% ctx - /help", cur=None)
sys.stdout.flush(); time.sleep(0.4); s.exit()
'''


def snapshot(cols=92, rows=22, driver_src=None):
    try:
        import pyte
    except ImportError:
        sys.exit("this dev tool needs pyte:  pip install pyte")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = driver_src or (DRIVER % repo)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    p = subprocess.Popen([sys.executable, "-c", src], stdin=slave, stdout=slave,
                         stderr=slave, start_new_session=True)
    os.close(slave)
    buf, deadline = b"", time.time() + 6
    while time.time() < deadline:
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try: d = os.read(master, 65536)
            except OSError: break
            if not d: break
            buf += d
        elif p.poll() is not None:
            break
    scr = pyte.Screen(cols, rows); st = pyte.Stream(scr); st.feed(buf.decode("utf-8", "replace"))
    print("+" + "-" * cols + "+")
    for line in scr.display:
        print("|" + line + "|")
    print("+" + "-" * cols + "+")
    box = set("╭╮╰╯│─")
    rights = {min((x for x in range(cols) if scr.buffer[y][x].data in box), default=None):
              None for y in range(rows)}
    rcols = sorted({max((x for x in range(cols) if scr.buffer[y][x].data in box), default=-1)
                    for y in range(rows)} - {-1})
    print("box right-border columns:", rcols,
          "->", "ALIGNED" if len(rcols) == 1 else "MISALIGNED")


if __name__ == "__main__":
    snapshot(int(sys.argv[1]) if len(sys.argv) > 1 else 92)
