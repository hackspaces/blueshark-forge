"""The office renderer — a braille/dither spatial view over company state (Slices 3-4).

Pure renderer, so it's unit-testable without a running company: the braille packing is
correct, the canvas aligns to a fixed cell width, and item states surface as room glyphs."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import office, render   # noqa: E402


class TestCanvas(unittest.TestCase):
    def test_dot_packs_to_the_right_braille_glyph(self):
        cv = office.Canvas(1, 1)          # one cell = a 2×4 dot block
        cv.set_dot(0, 0)                  # top-left dot → 0x2800 | 0x01
        self.assertEqual(cv.rows_out()[0], chr(0x2801))

    def test_full_cell_is_solid_braille(self):
        cv = office.Canvas(1, 1)
        for x in range(2):
            for y in range(4):
                cv.set_dot(x, y)
        self.assertEqual(cv.rows_out()[0], chr(0x28FF))   # all 8 dots set

    def test_empty_cell_is_a_space(self):
        self.assertEqual(office.Canvas(1, 1).rows_out()[0], " ")

    def test_resolution_is_two_by_four_per_cell(self):
        cv = office.Canvas(10, 5)
        self.assertEqual((cv.w, cv.h), (20, 20))


class TestOfficeRender(unittest.TestCase):
    ROLES = ["manager", "worker-1", "worker-2", "verifier"]

    def test_renders_the_requested_cell_height(self):
        lines = office.render_office("co", self.ROLES, cols=60, rows=18)
        self.assertEqual(len(lines), 18)

    def test_every_line_fits_the_width(self):
        lines = office.render_office("co", self.ROLES, cols=60, rows=18)
        for ln in lines:
            self.assertLessEqual(render.display_width(ln), 60 + 2)   # labels may nudge; never wild

    def test_item_state_surfaces_as_a_room_glyph(self):
        lines = office.render_office("co", self.ROLES,
                                     item_states={"worker-1": "verified"}, cols=60, rows=18)
        joined = "\n".join(render.strip_ansi(l) for l in lines)
        self.assertIn("✓", joined)                         # verified desk shows its glyph

    def test_graph_has_a_node_per_role_plus_board(self):
        nodes = office.office_graph("co", self.ROLES)
        self.assertIn("manager", nodes)
        self.assertIn("verifier", nodes)
        self.assertIn("board", nodes)
        self.assertIn("worker-1", nodes)


if __name__ == "__main__":
    unittest.main()
