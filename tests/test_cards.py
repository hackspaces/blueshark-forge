"""Agent cards — the deterministic-yet-unique, agent-native profile engine.

Rigor: the scoring MATH is checked against documented reference values (a wrong formula
would miss them), the specimen is proven deterministic AND unique per install, and the foil
rate is checked statistically. All terms are forge's own (agentic attributes/classes), not
borrowed. Stdlib/offline."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import cards   # noqa: E402


class TestAttributeFormula(unittest.TestCase):
    """The deterministic stat formula, checked against known reference maxima (255-base
    endurance → 714; 130-base tunable → 359 neutral / 394 boosted / 323 hindered)."""

    def test_endurance_max_is_714(self):
        self.assertEqual(cards._attr("stm", 255, 31, 252, 100, None, None), 714)

    def test_tunable_neutral_is_359(self):
        self.assertEqual(cards._attr("pre", 130, 31, 252, 100, None, None), 359)

    def test_temperament_boost_is_plus_10pct(self):
        self.assertEqual(cards._attr("pre", 130, 31, 252, 100, "pre", "rea"), 394)

    def test_temperament_hinder_is_minus_10pct(self):
        self.assertEqual(cards._attr("pre", 130, 31, 252, 100, "rea", "pre"), 323)

    def test_stamina_ignores_temperament(self):
        self.assertEqual(cards._attr("stm", 108, 31, 252, 100, "stm", "pre"),
                         cards._attr("stm", 108, 31, 252, 100, None, None))


class TestTemperaments(unittest.TestCase):
    def test_twenty_five_with_five_neutral(self):
        neutral = sum(1 for i in range(25) if cards.temperament(i)[1] == (None, None))
        self.assertEqual(neutral, 5)

    def test_never_tilts_stamina(self):
        for i in range(25):
            _, (up, down) = cards.temperament(i)
            self.assertNotEqual(up, "stm")
            self.assertNotEqual(down, "stm")

    def test_names_are_forge_native(self):
        # sanity: our disposition words, not borrowed ones
        self.assertIn("Methodical", cards._TEMPERAMENTS)
        self.assertNotIn("Adamant", cards._TEMPERAMENTS)


class TestForgedForm(unittest.TestCase):
    def test_pace_falls_with_size(self):
        tiny = cards.forged_form({"name": "t", "params_b": 0.5})
        big = cards.forged_form({"name": "b", "params_b": 70})
        self.assertGreater(tiny["base"]["pac"], big["base"]["pac"])   # small is fast

    def test_rating_rises_with_params(self):
        self.assertLess(cards.forged_form({"name": "a", "params_b": 1})["rating"],
                        cards.forged_form({"name": "b", "params_b": 32})["rating"])

    def test_verified_hardens_resilience_and_reliability(self):
        plain = cards.forged_form({"name": "x", "params_b": 7})
        proven = cards.forged_form({"name": "x", "params_b": 7, "status": "verified", "lift_pts": 10})
        self.assertGreater(proven["base"]["res"], plain["base"]["res"])
        self.assertGreater(proven["base"]["rel"], plain["base"]["rel"])

    def test_coder_class_for_a_coding_model(self):
        self.assertIn("Coder", cards.forged_form({"name": "q-coder:7b", "params_b": 7})["classes"])


class TestDeterminismAndUniqueness(unittest.TestCase):
    MODEL = {"name": "qwen2.5-coder:7b", "params_b": 7, "notes": "coding"}

    def test_same_forge_and_model_is_identical(self):
        a = cards.card(self.MODEL, fid="a" * 32)
        b = cards.card(self.MODEL, fid="a" * 32)
        self.assertEqual(a, b)                              # pure function → regenerable anywhere

    def test_different_installs_get_different_specimens(self):
        specimens = {tuple(cards.card(self.MODEL, fid=os.urandom(16).hex())["grain"].values())
                     for _ in range(50)}
        self.assertGreater(len(specimens), 45)              # essentially all distinct

    def test_work_grows_the_card(self):
        fresh = cards.card(self.MODEL, fid="c" * 32, telemetry={})
        worked = cards.card(self.MODEL, fid="c" * 32, telemetry={"edits": 20, "sessions": 10})
        self.assertEqual(fresh["grain"], worked["grain"])   # same specimen (Grain fixed)...
        self.assertGreater(worked["mastery"], fresh["mastery"])   # ...Mastery grows with work
        self.assertGreater(sum(worked["attrs"].values()), sum(fresh["attrs"].values()))


class TestFoilOdds(unittest.TestCase):
    def test_foil_rate_is_about_one_in_4096(self):
        n, foil = 60000, 0
        for _ in range(n):
            fid = os.urandom(16).hex()
            if cards.is_foil(fid, cards.pid(fid, "m")):
                foil += 1
        rate = foil / n
        self.assertTrue(0.00010 < rate < 0.00045, f"foil rate {rate:.5f}")


class TestRendering(unittest.TestCase):
    def _widths(self, model, **kw):
        from forge import render
        card = cards.render_card(cards.card(model, **kw))
        return {render.display_width(ln) for ln in card.splitlines()}

    def test_box_aligns_to_one_display_width(self):
        self.assertEqual(len(self._widths({"name": "qwen2.5-coder:7b", "params_b": 7,
                                           "notes": "coding"}, fid="a" * 32)), 1)

    def test_box_holds_with_cjk_emoji_name(self):
        self.assertEqual(len(self._widths({"name": "日本語-coder🚀:32b", "params_b": 32,
                                           "weights": {"size_gb": 20}}, fid="f" * 32)), 1)

    def test_foil_renders_a_mark(self):
        fid = next(f for f in (os.urandom(16).hex() for _ in range(100000))
                   if cards.is_foil(f, cards.pid(f, "m")))
        self.assertIn("✦", cards.render_card(cards.card({"name": "m", "params_b": 7}, fid=fid)))


if __name__ == "__main__":
    unittest.main()
