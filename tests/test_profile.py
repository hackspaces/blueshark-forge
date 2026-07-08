"""P5.8 — model passports.

Offline, stdlib-only. Two halves:

  * the store + scoring primitives — record/load, per-session rates, the active-probe
    scorer (score_probe) and prompt set (probe_specs), knob fusion (knobs), and the
    human describe() view; and
  * the agent-loop wiring that CONSUMES a passport — per-model loop_threshold /
    heat_bump / num_predict resolved at construction and re-resolved on a ladder swap,
    and the passive telemetry the live loop writes at its stuck sites (malformed, loop,
    escalate, alias-repair, fuzzy/exact edit, session) — including the invariant that an
    empty passport changes NOTHING and that an internal EphemeralSession never records.

The store is redirected to a throwaway tempdir the moment this module is imported
(process-wide, before any test runs), so the whole offline suite stays hermetic — no
test touches the real ~/.forge/profile.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import profile                       # noqa: E402
from forge import session as sm                 # noqa: E402
from forge import agent as agent_mod             # noqa: E402
from forge.agent import Agent, DEFAULT_LOOP_THRESHOLD, DEFAULT_HEAT_BUMP  # noqa: E402
from forge import backends                       # noqa: E402

profile.PROFILE_DIR = tempfile.mkdtemp(prefix="forge-profile-suite-")

DEFAULTS = {"loop_threshold": DEFAULT_LOOP_THRESHOLD, "heat_bump": DEFAULT_HEAT_BUMP,
            "num_predict": backends.NUM_PREDICT}


class _Store(unittest.TestCase):
    """Each test gets its own fresh store dir; the suite dir is restored after."""
    def setUp(self):
        self._prev = profile.PROFILE_DIR
        self.dir = tempfile.mkdtemp()
        profile.PROFILE_DIR = self.dir

    def tearDown(self):
        profile.PROFILE_DIR = self._prev

    def _sessions(self, model, n):
        for _ in range(n):
            profile.record(model, "session")


# ---------------------------------------------------------------- store primitives

class TestStore(_Store):
    def test_record_and_load_roundtrip(self):
        profile.record("m", "malformed")
        profile.record("m", "malformed", n=2)
        profile.record("m", "loop")
        d = profile.load("m")
        self.assertEqual(d["counts"], {"malformed": 3, "loop": 1})
        self.assertEqual(d["probe"], {})

    def test_missing_and_corrupt_are_blank(self):
        self.assertEqual(profile.load("never")["counts"], {})
        p = profile._path("bad")
        with open(p, "w") as f:
            f.write("{not json")
        self.assertEqual(profile.load("bad"), {"counts": {}, "probe": {}})

    def test_record_ignores_empty_and_nonpositive(self):
        profile.record("", "malformed")
        profile.record("m", "")
        profile.record("m", "loop", n=0)
        self.assertEqual(profile.load("m")["counts"], {})

    def test_rates_normalize_per_session(self):
        self._sessions("m", 4)
        for _ in range(2):
            profile.record("m", "loop")
        profile.record("m", "malformed")
        profile.record("m", "fuzzy_edit")
        for _ in range(3):
            profile.record("m", "exact_edit")
        r = profile.rates("m")
        self.assertEqual(r["sessions"], 4)
        self.assertAlmostEqual(r["loop_per_session"], 0.5)
        self.assertAlmostEqual(r["malformed_per_session"], 0.25)
        self.assertAlmostEqual(r["fuzzy_edit_frac"], 0.25)   # 1 fuzzy of 4 edits

    def test_write_passport_keeps_counts(self):
        profile.record("m", "malformed")
        profile.write_passport("m", {"format_hold": 0.9, "n": 10})
        d = profile.load("m")
        self.assertEqual(d["counts"], {"malformed": 1})
        self.assertEqual(d["probe"], {"format_hold": 0.9, "n": 10})


# ------------------------------------------------------------------ knob fusion

class TestKnobs(_Store):
    def test_empty_passport_is_identity(self):
        # The load-bearing invariant: an un-profiled model runs on the exact defaults.
        self.assertEqual(profile.knobs("fresh", DEFAULTS), DEFAULTS)

    def test_no_tuning_below_min_sessions(self):
        self._sessions("m", profile.MIN_SESSIONS - 1)
        for _ in range(10):
            profile.record("m", "loop")           # wildly loop-prone…
        self.assertEqual(profile.knobs("m", DEFAULTS)["loop_threshold"],
                         DEFAULT_LOOP_THRESHOLD)   # …but not enough sessions yet

    def test_loop_prone_tightens_threshold(self):
        self._sessions("m", 4)
        for _ in range(3):
            profile.record("m", "loop")           # 0.75/session ≥ LOOP_PRONE
        self.assertEqual(profile.knobs("m", DEFAULTS)["loop_threshold"], 2)

    def test_truncator_raises_num_predict(self):
        self._sessions("m", 4)
        for _ in range(3):
            profile.record("m", "trunc_write")
        k = profile.knobs("m", DEFAULTS)
        self.assertEqual(k["num_predict"], profile.TRUNC_NUM_PREDICT)
        self.assertGreater(k["num_predict"], DEFAULTS["num_predict"])

    def test_malformed_prone_hotter_heat(self):
        self._sessions("m", 4)
        for _ in range(5):
            profile.record("m", "malformed")      # ≥ 1.0/session
        self.assertEqual(profile.knobs("m", DEFAULTS)["heat_bump"], profile.HOT_HEAT_BUMP)

    def test_probe_format_floor_tunes_a_fresh_install(self):
        # Zero live sessions, but a poor active probe flags a malformed-prone model.
        profile.write_passport("m", {"format_hold": 0.5, "field_complete": 1.0,
                                     "exact_repro": 1.0, "n": 10})
        self.assertEqual(profile.knobs("m", DEFAULTS)["heat_bump"], profile.HOT_HEAT_BUMP)

    def test_knobs_never_mutates_the_defaults_dict(self):
        self._sessions("m", 4)
        for _ in range(3):
            profile.record("m", "loop")
        profile.knobs("m", DEFAULTS)
        self.assertEqual(DEFAULTS["loop_threshold"], DEFAULT_LOOP_THRESHOLD)


# --------------------------------------------------------------- active probe scoring

class TestProbeScoring(_Store):
    def test_probe_specs_carry_required_fields(self):
        specs = profile.probe_specs()
        self.assertGreaterEqual(len(specs), 8)
        wf = next(s for s in specs if s["action"] == "write_file")
        self.assertIn("path", wf["required"])
        self.assertIn("content", wf["required"])

    def test_score_perfect_run(self):
        specs = profile.probe_specs()
        raws = []
        for s in specs:
            act = {"thought": "t", "action": s["action"]}
            for f in s["required"]:
                act[f] = s.get("exact_text", "x") if f == s.get("exact_field") else "x"
            # every non-exact spec still needs its exact_field populated verbatim
            if s.get("exact_field"):
                act[s["exact_field"]] = s["exact_text"]
            import json
            raws.append(json.dumps(act))
        sc = profile.score_probe(raws, specs)
        self.assertEqual(sc["format_hold"], 1.0)
        self.assertEqual(sc["field_complete"], 1.0)
        self.assertEqual(sc["exact_repro"], 1.0)
        self.assertEqual(sc["n"], len(specs))

    def test_score_all_malformed(self):
        specs = profile.probe_specs()
        sc = profile.score_probe(["not json at all"] * len(specs), specs)
        self.assertEqual(sc["format_hold"], 0.0)
        self.assertEqual(sc["field_complete"], 0.0)

    def test_score_salvages_fenced_json(self):
        specs = profile.probe_specs()[:1]   # list_files, no required fields
        raw = '```json\n{"thought":"t","action":"list_files"}\n```'
        sc = profile.score_probe([raw], specs)
        self.assertEqual(sc["format_hold"], 1.0)

    def test_exact_repro_penalizes_paraphrase(self):
        specs = [s for s in profile.probe_specs() if s.get("exact_field")]
        # reproduce a DIFFERENT string than asked → exact_repro 0
        import json
        raws = [json.dumps({"thought": "t", "action": "edit_file", "path": "f",
                            "old": "WRONG", "new": "y"}) for _ in specs]
        sc = profile.score_probe(raws, specs)
        self.assertEqual(sc["exact_repro"], 0.0)


# --------------------------------------------------------- agent consumption + telemetry

class _NPBackend:
    """A fake Ollama-shaped backend: carries a per-instance num_predict (so the passport
    can push a bigger output budget onto it) and yields scripted actions."""
    def __init__(self, actions, name="np-model"):
        self.actions = list(actions)
        self.i = 0
        self.name = name
        self.num_predict = backends.NUM_PREDICT

    def stream(self, messages, schema=None, temperature=0.0):
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        yield act

    def chat(self, messages, schema=None, temperature=0.0):
        return '{"thought":"x","action":"say","message":"done"}'


class _RecSession:
    """Minimal real (non-ephemeral) session: drives Agent.send and records telemetry."""
    def __init__(self, cwd, sid="pp"):
        self.cwd, self.sid, self.name = cwd, sid, "pp"
        self.status = "idle"
    def log(self, *a, **k): pass
    def drain(self): return []
    def set_status(self, s): self.status = s
    def push(self, sender, text): pass


_SAY = '{"thought":"d","action":"say","message":"done"}'


class TestAgentConsumption(_Store):
    def test_default_knobs_when_unprofiled(self):
        a = Agent(_NPBackend([_SAY], name="fresh-x"), sm.EphemeralSession(self.dir, "fresh-x"))
        self.assertEqual(a.loop_threshold, DEFAULT_LOOP_THRESHOLD)
        self.assertEqual(a.heat_bump, DEFAULT_HEAT_BUMP)

    def test_loop_prone_model_gets_tight_threshold_at_construction(self):
        self._sessions("np:lp", 4)
        for _ in range(3):
            profile.record("np:lp", "loop")
        a = Agent(_NPBackend([_SAY], name="np:lp"), sm.EphemeralSession(self.dir, "np:lp"))
        self.assertEqual(a.loop_threshold, 2)

    def test_truncator_num_predict_reaches_the_backend(self):
        self._sessions("np:tr", 4)
        for _ in range(3):
            profile.record("np:tr", "trunc_write")
        b = _NPBackend([_SAY], name="np:tr")
        Agent(b, sm.EphemeralSession(self.dir, "np:tr"))
        self.assertEqual(b.num_predict, profile.TRUNC_NUM_PREDICT)

    def test_knobs_reresolve_on_ladder_swap(self):
        # base rung un-profiled (defaults); a stronger rung is loop-prone. set_ladder
        # swaps to a new base — resolve for whoever is active.
        self._sessions("np:strong", 4)
        for _ in range(3):
            profile.record("np:strong", "loop")
        a = Agent(_NPBackend([_SAY], name="np:plain"), sm.EphemeralSession(self.dir, "np:plain"))
        self.assertEqual(a.loop_threshold, DEFAULT_LOOP_THRESHOLD)
        a.set_ladder([_NPBackend([_SAY], name="np:strong")])
        self.assertEqual(a.loop_threshold, 2)

    def test_live_loop_records_session_and_malformed(self):
        d = tempfile.mkdtemp()
        actions = ["not valid json", _SAY]     # one malformed strike, then a say
        a = Agent(_NPBackend(actions, name="live:m"), _RecSession(d), max_steps=6)
        a.send("go")
        c = profile.load("live:m")["counts"]
        self.assertEqual(c.get("session"), 1)
        self.assertGreaterEqual(c.get("malformed", 0), 1)

    def test_ephemeral_session_never_records(self):
        d = tempfile.mkdtemp()
        a = Agent(_NPBackend(["not valid json", _SAY], name="eph:m"),
                  sm.EphemeralSession(d, "eph:m"), max_steps=6)
        a.send("go")
        self.assertEqual(profile.load("eph:m")["counts"], {})   # gated off for internal agents

    def test_live_loop_records_exact_edit(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "r.py"), "w") as f:
            f.write("x = 1\n")
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"edit","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',
            _SAY,
        ]
        a = Agent(_NPBackend(actions, name="live:e"), _RecSession(d), max_steps=8)
        a.send("go")
        c = profile.load("live:e")["counts"]
        self.assertEqual(c.get("exact_edit"), 1)
        self.assertIsNone(c.get("fuzzy_edit"))


if __name__ == "__main__":
    unittest.main()
