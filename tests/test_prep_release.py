"""Release-prep logic: version math, bump inference, doc sync, notes rendering."""
import importlib.util
import os
import unittest

# prep_release.py lives under .github/scripts (outside the package) — load it directly.
_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     ".github", "scripts", "prep_release.py")
_spec = importlib.util.spec_from_file_location("prep_release", _PATH)
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)


class TestVersion(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(pr.parse_version('__version__ = "0.9.0"\n'), (0, 9, 0))

    def test_bump(self):
        self.assertEqual(pr.bump_version((0, 9, 0), "patch"), "0.9.1")
        self.assertEqual(pr.bump_version((0, 9, 3), "minor"), "0.10.0")
        self.assertEqual(pr.bump_version((0, 9, 3), "major"), "1.0.0")
        with self.assertRaises(ValueError):
            pr.bump_version((0, 9, 0), "nope")

    def test_set_version_preserves_the_rest(self):
        out = pr.set_version('# c\n__version__ = "0.9.0"\nX = 1\n', "0.10.0")
        self.assertIn('__version__ = "0.10.0"', out)
        self.assertIn("X = 1", out)          # only the version line changed


class TestDetectBump(unittest.TestCase):
    def test_feat_is_minor(self):
        self.assertEqual(pr.detect_bump(["fix: a", "feat: b", "docs: c"]), "minor")

    def test_only_fixes_is_patch(self):
        self.assertEqual(pr.detect_bump(["fix: a", "docs: b"]), "patch")

    def test_breaking_is_major(self):
        self.assertEqual(pr.detect_bump(["feat!: drop x"]), "major")
        self.assertEqual(pr.detect_bump(["refactor: y", "BREAKING CHANGE: z"]), "major")

    def test_empty_defaults_to_patch(self):
        self.assertEqual(pr.detect_bump([]), "patch")


class TestSecuritySync(unittest.TestCase):
    SEC = ("## Supported versions\n\n"
           "| Version | Supported          |\n"
           "| ------- | ------------------ |\n"
           "| 0.8.x   | :white_check_mark: |\n"
           "| < 0.8   | :x:                |\n\n## Reporting\n")

    def test_table_advances_to_new_minor(self):
        out = pr.update_security(self.SEC, "0.10.0")
        self.assertIn("| 0.10.x   | :white_check_mark: |", out)
        self.assertIn("| < 0.10   | :x:", out)
        self.assertNotIn("0.8.x", out)
        self.assertIn("## Reporting", out)      # rest of the doc untouched

    def test_idempotent(self):
        once = pr.update_security(self.SEC, "0.9.0")
        twice = pr.update_security(once, "0.9.0")
        self.assertEqual(once, twice)


class TestNotes(unittest.TestCase):
    def test_notes_group_and_head(self):
        subs = ["feat: the catalog", "fix: a tier bug", "docs: a report", "chore: ignore me"]
        out = pr.render_notes("0.9.0", subs, summary="the catalog")
        self.assertTrue(out.startswith("release: v0.9.0 — the catalog\n"))
        self.assertIn("New:\n  · the catalog", out)
        self.assertIn("Fixed:\n  · a tier bug", out)
        self.assertIn("Docs:\n  · a report", out)
        self.assertNotIn("ignore me", out)        # non-conventional commits are dropped

    def test_notes_without_summary_or_changes(self):
        out = pr.render_notes("1.0.0", [])
        self.assertTrue(out.startswith("release: v1.0.0\n"))
        self.assertIn("Release v1.0.0.", out)     # graceful fallback


if __name__ == "__main__":
    unittest.main()
