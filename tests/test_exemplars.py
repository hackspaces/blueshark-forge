"""P5.6 — self-harvested few-shot exemplar store.

Offline, stdlib-only. Covers the store primitives (cap / dedupe / redact / fetch /
malformed tally / kind-guess) and the two agent-loop wirings that consume them:
the malformed-JSON retry nudge embedding one of the model's OWN past valid actions
of the guessed kind, and the cold-start head-pin that fires ONLY when a model has
both recorded malformed history and an available exemplar.

The store is redirected to a throwaway tempdir the moment this module is imported
(process-wide, before any test in ANY module runs), so the whole offline suite is
hermetic — no test ever reads or writes the real ~/.forge/exemplars, and no
cross-run state can leak into another module's head-layout assertions.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import exemplars                      # noqa: E402
from forge import session as sm                  # noqa: E402
from forge.agent import Agent                     # noqa: E402

# Process-wide hermetic redirect: unittest imports every test module during
# discovery BEFORE running any test, so assigning here guarantees that any Agent
# constructed by ANY test module records/reads exemplars under a fresh tempdir and
# never the real home. Fresh per process → no cross-run accumulation.
exemplars.EXEMPLAR_DIR = tempfile.mkdtemp(prefix="forge-exemplars-suite-")


class _Backend:
    """Yields a scripted sequence of raw model outputs (clamps to the last when
    exhausted), with a controllable ``name`` so a test can key the store."""

    def __init__(self, actions, name="mymodel"):
        self.actions = list(actions)
        self.i = 0
        self.name = name

    def stream(self, messages, schema=None, temperature=0.0):
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        yield act

    def chat(self, messages, schema=None, temperature=0.0):
        return '{"thought":"x","action":"say","message":"done"}'


_SAY = '{"thought":"d","action":"say","message":"done"}'
_BASH = '{"thought":"t","action":"bash","command":"ls"}'


class _StoreBase(unittest.TestCase):
    """Isolate each test in its own store dir; restore the suite dir after (so
    other modules keep their hermetic redirect)."""

    def setUp(self):
        self._prev = exemplars.EXEMPLAR_DIR
        self.dir = tempfile.mkdtemp()
        exemplars.EXEMPLAR_DIR = self.dir

    def tearDown(self):
        exemplars.EXEMPLAR_DIR = self._prev


class TestStore(_StoreBase):
    def test_record_and_fetch_roundtrip(self):
        exemplars.record("m", "bash", _BASH)
        self.assertEqual(exemplars.fetch("m", "bash"), _BASH)
        self.assertIsNone(exemplars.fetch("m", "grep"))         # a kind never recorded
        self.assertIsNone(exemplars.fetch("other", "bash"))     # a model never recorded

    def test_unreadable_present_store_never_raises_into_the_loop(self):
        # a present-but-unreadable store (here: a DIRECTORY at the file path → slurp
        # raises OSError) must degrade to empty, not raise into the malformed-nudge path.
        os.makedirs(exemplars._path("dm"))                      # dir where a .jsonl is expected
        os.makedirs(exemplars._counts_path())                   # dir where _malformed.json is expected
        self.assertEqual(exemplars._load("dm"), [])
        self.assertIsNone(exemplars.fetch("dm", "bash"))        # was: OSError escaped here
        self.assertIsNone(exemplars.fetch_any("dm"))
        self.assertEqual(exemplars.malformed_count("dm"), 0)

    def test_caps_at_five_per_kind_most_recent_wins(self):
        for n in range(7):
            exemplars.record("m", "bash", '{"action":"bash","command":"c%d"}' % n)
        recs = exemplars._load("m")
        self.assertEqual(len(recs), 5)                          # capped at PER_KIND
        cmds = [r["raw"] for r in recs]
        self.assertNotIn('{"action":"bash","command":"c0"}', cmds)   # oldest two evicted
        self.assertNotIn('{"action":"bash","command":"c1"}', cmds)
        self.assertEqual(exemplars.fetch("m", "bash"),
                         '{"action":"bash","command":"c6"}')    # newest served

    def test_cap_is_per_kind_not_global(self):
        for n in range(6):
            exemplars.record("m", "bash", '{"action":"bash","command":"c%d"}' % n)
        exemplars.record("m", "grep", '{"action":"grep","pattern":"x"}')
        exemplars.record("m", "grep", '{"action":"grep","pattern":"y"}')
        recs = exemplars._load("m")
        self.assertEqual(sum(r["kind"] == "bash" for r in recs), 5)   # bash capped
        self.assertEqual(sum(r["kind"] == "grep" for r in recs), 2)   # grep untouched

    def test_dedupe_identical_body(self):
        exemplars.record("m", "bash", _BASH)
        exemplars.record("m", "bash", _BASH)
        exemplars.record("m", "bash", _BASH)
        self.assertEqual(len(exemplars._load("m")), 1)          # stored once

    def test_write_file_content_is_redacted(self):
        raw = '{"thought":"t","action":"write_file","path":"a.py","content":"%s"}' % ("X" * 5000)
        exemplars.record("m", "write_file", raw)
        body = exemplars.fetch("m", "write_file")
        self.assertLessEqual(len(body), exemplars.MAX_LEN)      # small
        self.assertNotIn("XXXX", body)                          # the payload is gone
        import json
        obj = json.loads(body)                                  # still valid JSON…
        self.assertEqual(obj["action"], "write_file")           # …that teaches the shape
        self.assertEqual(obj["path"], "a.py")
        self.assertNotIn("X", obj["content"])                   # content elided to a placeholder

    def test_fetch_any_returns_most_recent_of_any_kind(self):
        self.assertIsNone(exemplars.fetch_any("m"))
        exemplars.record("m", "bash", _BASH)
        exemplars.record("m", "grep", '{"action":"grep","pattern":"z"}')
        self.assertEqual(exemplars.fetch_any("m"), '{"action":"grep","pattern":"z"}')

    def test_malformed_count_tally(self):
        self.assertEqual(exemplars.malformed_count("m"), 0)
        exemplars.record_malformed("m")
        exemplars.record_malformed("m")
        self.assertEqual(exemplars.malformed_count("m"), 2)
        self.assertEqual(exemplars.malformed_count("other"), 0)  # keyed per model

    def test_guess_kind(self):
        self.assertEqual(exemplars.guess_kind('{"action":"bash","command":"ls"}'), "bash")
        # truncated before the action field → fall back to a distinctive field name
        self.assertEqual(exemplars.guess_kind('{"thought":"t","command":"echo hi'), "bash")
        self.assertEqual(exemplars.guess_kind('{"pattern":"foo'), "grep")
        self.assertEqual(exemplars.guess_kind('{"old":"a","new'), "edit_file")
        self.assertEqual(exemplars.guess_kind('{"content":"def f('), "write_file")
        self.assertEqual(exemplars.guess_kind('{"path":"x.py"'), "read_file")
        self.assertIsNone(exemplars.guess_kind("this is not json at all"))


class TestMalformedEmbed(_StoreBase):
    """The malformed-retry nudge quotes a prior exemplar of the guessed kind."""

    def test_retry_embeds_prior_exemplar_of_guessed_kind(self):
        exemplars.record("mymodel", "bash", _BASH)              # a harvested valid bash
        d = tempfile.mkdtemp()
        # step0: a truncated bash (guess_kind → "bash") strikes; step1: say ends it.
        be = _Backend(['{"thought":"go","action":"bash","command":"echo hi', _SAY],
                      name="mymodel")
        a = Agent(be, sm.EphemeralSession(d, "mymodel"), max_steps=6)
        self.assertEqual(a.head_len, 1)                         # no cold-start (0 malformed history)
        a.send("go")
        nudges = [m["content"] for m in a.messages if m["role"] == "user"]
        self.assertTrue(any(_BASH in n and "valid `bash` action" in n for n in nudges),
                        "the retry nudge must embed the model's own bash exemplar")

    def test_no_exemplar_falls_back_to_plain_nudge(self):
        d = tempfile.mkdtemp()
        be = _Backend(['{"thought":"go","action":"bash","command":"echo hi', _SAY],
                      name="freshmodel")                        # store empty for this model
        a = Agent(be, sm.EphemeralSession(d, "freshmodel"), max_steps=6)
        a.send("go")
        nudges = [m["content"] for m in a.messages if m["role"] == "user"]
        self.assertIn("That was not valid action JSON. Reply with one JSON action object only.",
                      nudges)
        self.assertFalse(any("valid `bash` action" in n for n in nudges))  # nothing embedded


class TestColdStartPin(_StoreBase):
    """The cold-start head-pin fires ONLY with prior malformed history AND an
    available exemplar; it pins ONE user/assistant demonstration into the head."""

    def _agent(self, name):
        d = tempfile.mkdtemp()
        return Agent(_Backend([_SAY], name=name), sm.EphemeralSession(d, name), max_steps=3)

    def test_fires_with_history_and_exemplar(self):
        exemplars.record("coldmodel", "bash", _BASH)
        exemplars.record_malformed("coldmodel")
        a = self._agent("coldmodel")
        self.assertEqual(a.head_len, 3)                         # system + pinned user/assistant pair
        self.assertEqual(a.messages[1]["content"], "example task")
        self.assertEqual(a.messages[2]["content"], _BASH)       # the exemplar, pinned into the head

    def test_no_injection_without_malformed_history(self):
        exemplars.record("hascold", "bash", _BASH)              # exemplar but NO malformed history
        a = self._agent("hascold")
        self.assertEqual(a.head_len, 1)
        self.assertFalse(any(m["content"] == "example task" for m in a.messages))

    def test_no_injection_without_an_exemplar(self):
        exemplars.record_malformed("onlymalformed")             # malformed history but NO exemplar
        a = self._agent("onlymalformed")
        self.assertEqual(a.head_len, 1)
        self.assertFalse(any(m["content"] == "example task" for m in a.messages))


if __name__ == "__main__":
    unittest.main()
