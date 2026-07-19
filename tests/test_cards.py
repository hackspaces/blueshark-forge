"""Model cards — the Pokémon-authentic, deterministic-yet-unique profile engine.

Rigor: the stat MATH is checked against documented in-game values (a wrong formula would
miss them), the specimen is proven deterministic AND unique per trainer, and the shiny rate
is checked statistically against 1/4096. Stdlib/offline."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import cards   # noqa: E402


class TestStatFormula(unittest.TestCase):
    """The Gen-3 formula, checked against real documented maxima."""

    def test_blissey_max_hp_is_714(self):
        # Blissey base HP 255, L100, 31 IV, 252 EV → the famous 714
        self.assertEqual(cards._stat("hp", 255, 31, 252, 100, None, None), 714)

    def test_garchomp_neutral_attack_is_359(self):
        self.assertEqual(cards._stat("atk", 130, 31, 252, 100, None, None), 359)

    def test_adamant_raises_attack_10pct(self):
        # Adamant (+Atk) on Garchomp's 359 → 394, the documented Adamant max
        self.assertEqual(cards._stat("atk", 130, 31, 252, 100, "atk", "spa"), 394)

    def test_hindering_nature_lowers_10pct(self):
        self.assertEqual(cards._stat("atk", 130, 31, 252, 100, "spa", "atk"), 323)  # 359*0.9

    def test_hp_ignores_nature(self):
        self.assertEqual(cards._stat("hp", 108, 31, 252, 100, "hp", "atk"),
                         cards._stat("hp", 108, 31, 252, 100, None, None))


class TestNatures(unittest.TestCase):
    def test_adamant_is_attack_up_spatk_down(self):
        # nature index 3 = Adamant
        name, (up, down) = cards.nature(3)
        self.assertEqual((name, up, down), ("Adamant", "atk", "spa"))

    def test_neutral_natures_change_nothing(self):
        for i in (0, 6, 12, 18, 24):                        # Hardy/Docile/Serious/Bashful/Quirky
            _, (up, down) = cards.nature(i)
            self.assertEqual((up, down), (None, None))

    def test_twenty_five_natures_five_neutral(self):
        neutral = sum(1 for i in range(25) if cards.nature(i)[1] == (None, None))
        self.assertEqual(neutral, 5)


class TestSpecies(unittest.TestCase):
    def test_speed_falls_with_size(self):
        tiny = cards.species({"name": "t", "params_b": 0.5})
        big = cards.species({"name": "b", "params_b": 70})
        self.assertGreater(tiny["base"]["spe"], big["base"]["spe"])   # small is fast

    def test_bst_rises_with_params(self):
        self.assertLess(cards.species({"name": "a", "params_b": 1})["bst"],
                        cards.species({"name": "b", "params_b": 32})["bst"])

    def test_verified_and_lift_harden_defenses(self):
        plain = cards.species({"name": "x", "params_b": 7})
        proven = cards.species({"name": "x", "params_b": 7, "status": "verified", "lift_pts": 10})
        self.assertGreater(proven["base"]["def"], plain["base"]["def"])


class TestDeterminismAndUniqueness(unittest.TestCase):
    MODEL = {"name": "qwen2.5-coder:7b", "params_b": 7, "notes": "coding"}

    def test_same_trainer_and_model_is_identical(self):
        a = cards.card(self.MODEL, tid="a" * 32)
        b = cards.card(self.MODEL, tid="a" * 32)
        self.assertEqual(a, b)                              # pure function → regenerable anywhere

    def test_different_trainers_get_different_specimens(self):
        # across many trainer ids, IVs/nature vary — no two people share a specimen
        specimens = {(tuple(cards.card(self.MODEL, tid=os.urandom(16).hex())["ivs"].values()))
                     for _ in range(50)}
        self.assertGreater(len(specimens), 45)              # essentially all distinct

    def test_work_levels_the_card(self):
        fresh = cards.card(self.MODEL, tid="c" * 32, telemetry={})
        worked = cards.card(self.MODEL, tid="c" * 32, telemetry={"verified": 20, "sessions": 10})
        self.assertEqual(fresh["ivs"], worked["ivs"])       # same specimen (IVs fixed)...
        self.assertGreater(worked["level"], fresh["level"])  # ...but leveled by real work
        self.assertGreater(sum(worked["stats"].values()), sum(fresh["stats"].values()))


class TestShinyOdds(unittest.TestCase):
    def test_shiny_rate_is_about_one_in_4096(self):
        model = "some-model"
        n, shiny = 60000, 0
        for _ in range(n):
            tid = os.urandom(16).hex()
            if cards.is_shiny(tid, cards.pid(tid, model)):
                shiny += 1
        rate = shiny / n
        # expected ~1/4096 ≈ 0.000244; allow a wide statistical band
        self.assertTrue(0.00010 < rate < 0.00045, f"shiny rate {rate:.5f} (~{n//max(shiny,1)} : 1)")


class TestRendering(unittest.TestCase):
    """The card box aligns on DISPLAY columns (the render foundation) even with wide-char
    model names — the reason the rendering foundation came first."""

    def _lines(self, model, **kw):
        from forge import render
        card = cards.render_card(cards.card(model, **kw))
        return card.splitlines(), render

    def test_every_line_is_the_same_display_width(self):
        lines, render = self._lines({"name": "qwen2.5-coder:7b", "params_b": 7, "notes": "coding"},
                                    tid="a" * 32)
        widths = {render.display_width(ln) for ln in lines}
        self.assertEqual(len(widths), 1, f"card box misaligned: {widths}")

    def test_box_holds_with_cjk_emoji_model_name(self):
        lines, render = self._lines({"name": "日本語-coder🚀:32b", "params_b": 32,
                                     "weights": {"size_gb": 20}}, tid="f" * 32)
        widths = {render.display_width(ln) for ln in lines}
        self.assertEqual(len(widths), 1, f"wide-char name broke the box: {widths}")

    def test_shiny_specimen_renders_a_sparkle(self):
        # find a tid that makes this model shiny, then assert the ✨ shows
        model = {"name": "m", "params_b": 7}
        tid = next(t for t in (os.urandom(16).hex() for _ in range(100000))
                   if cards.is_shiny(t, cards.pid(t, "m")))
        self.assertTrue(cards.card(model, tid=tid)["shiny"])
        self.assertIn("✨", cards.render_card(cards.card(model, tid=tid)))


if __name__ == "__main__":
    unittest.main()
