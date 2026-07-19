"""The office — a spatial renderer for a running company (Slices 3-4).

The world is a GRAPH, not a grid: rooms (the manager's office, worker desks, the notice
board, the verifier's booth) are nodes; the rendered floorplan is a view, the graph is the
truth. This module is renderer #2 over the same company state `company status` reads — it
owns zero state and only draws what the board/receipts say, so it can crash without the
company noticing and `forge replay` of a company becomes a movie of dots for free.

Rendering is a dependency-free braille canvas: each terminal cell is a 2×4 dot grid
(U+2800–U+28FF), so an 80×24 terminal is a 160×96 dot bitmap. Rooms are filled with ordered
(Bayer) dithered textures so they read as distinct spaces in monochrome; agents are coloured
dots that move along edges between rooms. The aesthetic is a simulation you READ (Obra Dinn /
Playdate), not a game you play — space is drawn only where it does mechanical observability
work: a dot walking to the manager's office announces an escalation before any log line.
"""
import math

from . import render as _r

# Braille cell = 2 wide × 4 tall dots. Bit order per the Unicode braille pattern:
#   (0,0)=0x01 (1,0)=0x08   (0,1)=0x02 (1,1)=0x10   (0,2)=0x04 (1,2)=0x20   (0,3)=0x40 (1,3)=0x80
_BRAILLE_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

# 4×4 Bayer ordered-dither matrix (0..15). A texture level 0..16 fills a room with a stable
# dot pattern, so carpet / corridor / booth read as different greys without any colour.
_BAYER4 = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))

ROLE_COLOR = {"manager": "yellow", "worker": "cyan", "verifier": "green", "ceo": "magenta"}


class Canvas:
    """A braille bitmap: set_dot in dot-space, then rows() packs each 2×4 block to a glyph.
    Colour is tracked per-cell (the dot that most recently claimed the cell wins) so agents
    render in their role colour over the dithered room texture."""

    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows                 # terminal cells
        self.w, self.h = cols * 2, rows * 4               # dot resolution
        self._cells = [[0] * cols for _ in range(rows)]
        self._color = [[None] * cols for _ in range(rows)]

    def set_dot(self, x, y, color=None):
        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        cx, cy = x // 2, y // 4
        self._cells[cy][cx] |= _BRAILLE_BITS[y % 4][x % 2]
        if color:
            self._color[cy][cx] = color

    def fill_rect(self, x0, y0, x1, y1, level, color=None):
        """Fill a dot-space rectangle with a Bayer-dithered texture at `level` (0..16)."""
        for y in range(max(0, y0), min(self.h, y1)):
            for x in range(max(0, x0), min(self.w, x1)):
                if _BAYER4[y % 4][x % 4] < level:
                    self.set_dot(x, y, color)

    def frame_rect(self, x0, y0, x1, y1, color=None):
        for x in range(x0, x1):
            self.set_dot(x, y0, color); self.set_dot(x, y1 - 1, color)
        for y in range(y0, y1):
            self.set_dot(x0, y, color); self.set_dot(x1 - 1, y, color)

    def rows_out(self):
        """The rendered lines: each cell → its braille glyph, wrapped in its colour."""
        out = []
        for cy in range(self.rows):
            line = []
            for cx in range(self.cols):
                bits = self._cells[cy][cx]
                ch = chr(0x2800 + bits) if bits else " "
                col = self._color[cy][cx]
                line.append(_r.paint(ch, col) if (col and bits) else ch)
            out.append("".join(line))
        return out


# ---- the office graph -------------------------------------------------------
def office_graph(company_name, roles):
    """Lay out rooms for a company: the manager's office centre-left, the notice board top,
    the verifier's booth right, worker desks along the bottom. Returns {node: {x,y,label,
    kind}} in dot-space fractions (0..1), resolution-independent (the floorplan is a view)."""
    nodes = {
        "board": {"fx": 0.5, "fy": 0.12, "label": "NOTICE", "kind": "board"},
        "manager": {"fx": 0.22, "fy": 0.42, "label": "MANAGER", "kind": "office"},
        "verifier": {"fx": 0.80, "fy": 0.42, "label": "TRUST", "kind": "booth"},
    }
    workers = [r for r in roles if r.startswith("worker")]
    n = max(1, len(workers))
    for i, w in enumerate(workers):
        nodes[w] = {"fx": 0.12 + 0.76 * (i + 0.5) / n, "fy": 0.82,
                    "label": w.upper().replace("WORKER-", "DESK "), "kind": "desk"}
    return nodes


def _abs(node, W, H, rw, rh):
    return int(node["fx"] * W - rw / 2), int(node["fy"] * H - rh / 2)


_TEXTURE = {"office": 10, "booth": 8, "desk": 6, "board": 12}
_STATE_COLOR = {"verified": "green", "escalated": "red", "running": "cyan",
                "blocked": "yellow", "queued": "dim"}


def render_office(company_name, roles, item_states=None, agent_at=None, cols=72, rows=22):
    """Draw the office once. `item_states`: {worker_role: state} tints a busy desk; `agent_at`:
    {agent: node} places moving dots (default: everyone home). Returns the rendered lines."""
    item_states = item_states or {}
    agent_at = agent_at or {}
    cv = Canvas(cols, rows)
    nodes = office_graph(company_name, roles)
    W, H = cv.w, cv.h
    rw, rh = int(W * 0.20), int(H * 0.22)

    # edges (corridors) first, so rooms draw over them
    for a in ("manager",):
        for b in nodes:
            if b == a:
                continue
            ax, ay = int(nodes[a]["fx"] * W), int(nodes[a]["fy"] * H)
            bx, by = int(nodes[b]["fx"] * W), int(nodes[b]["fy"] * H)
            _line(cv, ax, ay, bx, by, level=2, color="dim")

    boxes = {}
    for name, node in nodes.items():
        x0, y0 = _abs(node, W, H, rw, rh)
        x1, y1 = x0 + rw, y0 + rh
        boxes[name] = (x0, y0, x1, y1)
        col = _STATE_COLOR.get(item_states.get(name)) if node["kind"] == "desk" else None
        cv.fill_rect(x0 + 1, y0 + 1, x1 - 1, y1 - 1, _TEXTURE.get(node["kind"], 6), col)
        cv.frame_rect(x0, y0, x1, y1, ROLE_COLOR.get(_role_kind(name), "dim"))

    # agents as bright dots at their current node (a small plus so they stand out on texture)
    for agent, at in agent_at.items():
        if at in nodes:
            cx, cy = int(nodes[at]["fx"] * W), int(nodes[at]["fy"] * H)
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                cv.set_dot(cx + dx, cy + dy, ROLE_COLOR.get(_role_kind(agent), "white"))

    lines = cv.rows_out()
    # overlay labels centred under each room (plain text row, like the dither spec)
    return _with_labels(lines, nodes, cols, rows, rw, rh, item_states)


def _role_kind(name):
    if name.startswith("worker"):
        return "worker"
    return name if name in ROLE_COLOR else "worker"


def _line(cv, x0, y0, x1, y1, level=2, color=None):
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(steps + 1):
        t = i / steps
        if int(t * steps) % 3 == 0:                        # dotted corridor
            cv.set_dot(int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t), color)


def _with_labels(lines, nodes, cols, rows, rw, rh, item_states):
    grid = [list(_r.strip_ansi(ln).ljust(cols)) if False else ln for ln in lines]
    # place a label row beneath each room centre
    labelled = list(lines)
    for name, node in nodes.items():
        cy = int(node["fy"] * rows * 4 / 4)                 # cell row of centre
        row = min(rows - 1, cy // 1)
        # find the label's terminal-cell row: centre cell of the room's bottom
        r = min(rows - 1, int(node["fy"] * rows) + int(rh / 8) + 1)
        c = max(0, int(node["fx"] * cols) - len(node["label"]) // 2)
        lab = node["label"]
        if node["kind"] == "desk" and name in item_states:
            g = {"verified": "✓", "escalated": "⚠", "running": "●", "queued": "◔",
                 "blocked": "◌"}.get(item_states[name], "")
            lab = f"{lab} {g}"
        labelled[r] = _overlay(labelled[r], c, _r.paint(lab, ROLE_COLOR.get(_role_kind(name), "dim")))
    return labelled


def _overlay(line, col, text):
    """Overlay `text` onto `line` starting at display column `col`, ANSI-aware."""
    plain = _r.strip_ansi(line)
    plain = plain.ljust(col + _r.display_width(_r.strip_ansi(text)))
    tlen = _r.display_width(_r.strip_ansi(text))
    new_plain = plain[:col] + _r.strip_ansi(text) + plain[col + tlen:]
    # re-apply just the overlaid text's colour; the rest returns as plain (labels sit on gaps)
    return new_plain[:col] + text + new_plain[col + tlen:]
