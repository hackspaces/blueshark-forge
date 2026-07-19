"""The company TUI dashboard — a composed view over company files (Slice 2).

Pure renderer: the three panes (office + board + receipts) compose into one aligned grid,
and item state / receipt verdicts surface. Testable without a running company by seeding the
board + receipts files directly (the TUI owns zero state — it only reads those files)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import company, company_tui, render   # noqa: E402


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.old = company.COMPANY_DIR
        company.COMPANY_DIR = tempfile.mkdtemp()
        company.create_charter("co", "m", ["w"])

    def tearDown(self):
        company.COMPANY_DIR = self.old

    def _item(self, **kw):
        company.write_item("co", kw)

    def test_composes_to_the_requested_size_and_aligns(self):
        lines = company_tui.render_dashboard("co", cols=90, rows=24)
        self.assertEqual(len(lines), 24)
        widths = {render.display_width(render.strip_ansi(l)) for l in lines}
        self.assertEqual(widths, {90})                 # every pane row aligned to the width

    def test_board_shows_item_state(self):
        self._item(id="w1", title="do a thing", assignee="worker-1", state="verified")
        plain = "\n".join(render.strip_ansi(l) for l in company_tui.render_dashboard("co", 90, 24))
        self.assertIn("BOARD", plain)
        self.assertIn("do a thing", plain)
        self.assertIn("✓", plain)                      # verified glyph

    def test_receipts_ticker_shows_verdicts(self):
        company.record_receipt("co", "w1", "worker-1", "CONFIRMED", "tests pass")
        company.record_receipt("co", "w2", "worker-1", "REJECTED", "TRUST rejected")
        plain = "\n".join(render.strip_ansi(l) for l in company_tui.render_dashboard("co", 90, 24))
        self.assertIn("RECEIPTS", plain)
        self.assertIn("CONFIRMED", plain)
        self.assertIn("REJECTED", plain)               # the trust layer, live

    def test_still_aligns_with_cjk_in_an_item_title(self):
        self._item(id="w1", title="日本語 のタスク 🚀", assignee="worker-1", state="running")
        widths = {render.display_width(render.strip_ansi(l))
                  for l in company_tui.render_dashboard("co", 90, 24)}
        self.assertEqual(widths, {90})                 # wide chars don't break the grid


if __name__ == "__main__":
    unittest.main()
