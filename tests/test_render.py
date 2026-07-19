"""Shared terminal-render helpers: width-fit, colour gating, path shortening."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import render                      # noqa: E402


class TestFit(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(render.fit("hello", 20), "hello")

    def test_long_text_is_clipped_with_ellipsis(self):
        out = render.fit("x" * 50, 10)
        self.assertEqual(len(out), 10)
        self.assertTrue(out.endswith("…"))

    def test_whitespace_is_collapsed_to_one_line(self):
        self.assertEqual(render.fit("a\n  b\t c", 40), "a b c")

    def test_nonpositive_width_is_empty(self):
        self.assertEqual(render.fit("anything", 0), "")


class TestDisplayWidth(unittest.TestCase):
    """The terminal aligns by display COLUMNS, not code points — the wcwidth engine every
    box border and column now measures with (the fix for emoji/CJK misalignment)."""

    def test_ascii_is_one_per_char(self):
        self.assertEqual(render.display_width("hello"), 5)

    def test_cjk_is_two_columns(self):
        self.assertEqual(render.display_width("日本語"), 6)

    def test_emoji_is_two(self):
        self.assertEqual(render.display_width("🚀"), 2)

    def test_combining_accent_adds_nothing(self):
        self.assertEqual(render.display_width("café"), 4)   # e + combining acute

    def test_zwj_family_collapses_to_one_glyph(self):
        self.assertEqual(render.display_width("👨‍👩‍👧"), 2)

    def test_ansi_codes_are_zero_width(self):
        self.assertEqual(render.display_width("\033[32mok\033[0m"), 2)

    def test_clip_never_splits_a_wide_char(self):
        # 3 CJK chars = 6 cols; clip to 5 must stop at 2 chars (4 cols), not split the third
        self.assertEqual(render.clip("日本語", 5), "日本")
        self.assertEqual(render.display_width(render.clip("日本語", 5)), 4)

    def test_fit_is_column_accurate(self):
        out = render.fit("日本語ab", 5)          # 8 cols → clip to 4 + ellipsis
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(render.display_width(out), 5)


class TestCapability(unittest.TestCase):
    def test_color_depth_none_when_off(self):
        orig = render.color_on
        render.color_on = lambda: False
        try:
            self.assertEqual(render.color_depth(), "none")
        finally:
            render.color_on = orig

    def test_color_depth_reads_colorterm(self):
        orig = render.color_on
        render.color_on = lambda: True
        os.environ["COLORTERM"] = "truecolor"
        try:
            self.assertEqual(render.color_depth(), "truecolor")
        finally:
            render.color_on = orig
            del os.environ["COLORTERM"]

    def test_enable_vt_is_noop_and_idempotent_on_posix(self):
        render.enable_vt(); render.enable_vt()      # must not raise on POSIX


class TestColour(unittest.TestCase):
    def test_paint_is_noop_when_colour_off(self):
        orig = render.color_on
        render.color_on = lambda: False
        try:
            self.assertEqual(render.paint("hi", "green", "bold"), "hi")
        finally:
            render.color_on = orig

    def test_paint_wraps_when_colour_on(self):
        orig = render.color_on
        render.color_on = lambda: True
        try:
            out = render.paint("hi", "green")
            self.assertTrue(out.startswith("\033["))
            self.assertTrue(out.endswith("\033[0m"))
            self.assertEqual(render.strip_ansi(out), "hi")
        finally:
            render.color_on = orig

    def test_no_styles_is_plain(self):
        self.assertEqual(render.paint("hi"), "hi")


class TestSpinner(unittest.TestCase):
    """The 'thinking'/'loading model' spinner must not leak escape codes into a
    pipe or file. On a non-TTY stdout it stays fully silent — no `\\r`, no ANSI,
    no animation thread — so `forge run … > log` / `| tee` captures clean output."""

    def test_silent_and_no_thread_on_non_tty(self):
        import io
        from forge.repl import Spinner
        orig = sys.stdout
        buf = io.StringIO()          # StringIO.isatty() is False
        sys.stdout = buf
        try:
            with Spinner("loading model") as sp:
                self.assertIsNone(sp._t)          # no animation thread spawned
            self.assertFalse(sp._tty)
        finally:
            sys.stdout = orig
        self.assertEqual(buf.getvalue(), "")      # nothing written to the pipe


class TestTilde(unittest.TestCase):
    def test_home_prefix_collapses(self):
        home = os.path.expanduser("~")
        self.assertEqual(render.tilde(os.path.join(home, "forge")), "~/forge")

    def test_non_home_path_is_unchanged(self):
        self.assertEqual(render.tilde("/etc/hosts"), "/etc/hosts")

    def test_empty_path(self):
        self.assertEqual(render.tilde(""), "")


if __name__ == "__main__":
    unittest.main()
