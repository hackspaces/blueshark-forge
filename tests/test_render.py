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
