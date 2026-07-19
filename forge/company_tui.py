"""The company TUI (Slice 2) — the interface IS the org design.

One screen over a running company: the animated OFFICE (the spatial environment) beside the
live BOARD (work items + state) and the RECEIPTS ticker (the trust layer, live — a "done"
claim getting REJECTED on screen is the demo). A status bar names the roll-up.

Architecture rule (the whole point): this is a VIEW over files — board/*.json, receipts.jsonl,
STATUS.md — composed with the wcwidth display-width engine so every pane column aligns. It
owns ZERO state: it tails the files and re-renders, so it can crash without the company
noticing, `company status` and this are two renderers of one truth, and `forge replay` of a
company is a movie of this dashboard for free. Compression upward is structural here — the
board shows STATE, never raw worker output; you read anyone, the chat (future) talks only to
the manager.
"""
from . import company as co
from . import office
from . import render as _r

_STATE_GLYPH = {"verified": ("✓", "green"), "escalated": ("⚠", "red"),
                "running": ("●", "cyan"), "queued": ("◔", "dim"),
                "blocked": ("◌", "yellow")}
_VERDICT_STYLE = {"CONFIRMED": "green", "REJECTED": "red", "UNKNOWN": "yellow"}


def _pad(s, width):
    """Clip to `width` display columns and pad to fill — the aligned-cell primitive."""
    s = _r.clip(s, width)
    return s + " " * max(0, width - _r.display_width(s))


def _board_pane(name, height):
    """The live board: one row per work item, state glyph + assignee + title."""
    board = co.read_board(name)
    rows = [_r.paint("BOARD", "bold")]
    for it in board:
        g, style = _STATE_GLYPH.get(it.get("state"), ("·", "dim"))
        who = it.get("assignee", "").replace("worker-", "w")
        rows.append(f"{_r.paint(g, style)} {_r.paint(who, 'dim')} {it.get('title', it['id'])}")
    if not board:
        rows.append(_r.paint("(no work items yet — manager planning)", "dim"))
    return rows[:height]


def _receipts_pane(name, height):
    """The trust ticker: the most recent audit verdicts, newest last."""
    recs = co.read_receipts(name, limit=height)
    rows = [_r.paint("RECEIPTS", "bold")]
    for rec in recs:
        style = _VERDICT_STYLE.get(rec.get("verdict"), "dim")
        who = rec.get("assignee", "").replace("worker-", "w")
        rows.append(f"{_r.paint(rec.get('verdict', '?'), style)} {_r.paint(who, 'dim')} "
                    + _r.fit(rec.get("detail", ""), 40))
    if not recs:
        rows.append(_r.paint("(no verdicts yet)", "dim"))
    return rows[:height]


def render_dashboard(name, cols=96, rows=30):
    """Compose the whole dashboard into `rows` lines of exactly `cols` display columns:
    the animated office on the left, the board (top) and receipts (bottom) stacked on the
    right, a title and a status bar. Pure over the company's files."""
    charter = co.load_charter(name)
    roles = ["manager"] + co.workers(charter) + ["verifier"]
    board = co.read_board(name)
    states = {it["assignee"]: it.get("state", "queued") for it in board}
    agents = {r: r for r in roles}
    agents["manager"] = "board" if any(s == "running" for s in states.values()) else "manager"

    left_w = max(28, int(cols * 0.56))
    right_w = cols - left_w - 1                      # 1 col gutter
    body_h = rows - 2                                # title + status bar

    office_lines = office.render_office(name, roles, item_states=states, agent_at=agents,
                                        cols=left_w, rows=body_h)
    half = body_h // 2
    right = _board_pane(name, half) + [_r.paint("─" * right_w, "dim")] + _receipts_pane(name, body_h - half - 1)

    n_verified = sum(1 for it in board if it.get("state") == "verified")
    n_esc = sum(1 for it in board if it.get("state") == "escalated")
    title = _pad("  " + _r.paint(f"{name}", "bold") + _r.paint("  — company", "dim"), cols)
    status = _pad("  " + _r.paint(f"{n_verified} verified", "green") + " · "
                  + _r.paint(f"{n_esc} escalated", "red" if n_esc else "dim")
                  + _r.paint("   Tab panes · Enter drill-in · Esc stop   (view over files)", "dim"), cols)

    out = [title]
    for i in range(body_h):
        l = office_lines[i] if i < len(office_lines) else ""
        r = right[i] if i < len(right) else ""
        out.append(_pad(l, left_w) + " " + _pad(r, right_w))
    out.append(status)
    return out
