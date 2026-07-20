"""Flywheel turn one — forge transcripts → harness-native training data. Offline, stdlib."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import corpus                            # noqa: E402


def _clean_turn():
    return [
        {"type": "meta", "model": "m"},
        {"type": "user", "text": "fix the bug in x.py"},
        {"type": "model", "raw": '{"action":"read_file","path":"x.py"}'},
        {"type": "action", "action": "read_file"},
        {"type": "observation", "text": "1\tx = 1", "ok": True},
        {"type": "model", "raw": '{"action":"edit_file","path":"x.py","old":"x = 1","new":"x = 2"}'},
        {"type": "action", "action": "edit_file"},
        {"type": "observation", "text": "edited x.py", "ok": True},
        {"type": "model", "raw": '{"action":"say","message":"Fixed."}'},
        {"type": "assistant", "text": "Fixed."},
    ]


class TestSFT(unittest.TestCase):
    def test_successful_actions_and_say_become_sft(self):
        b = corpus.build(_clean_turn(), sid="s")
        self.assertEqual(len(b["sft"]), 3)                     # read, edit, say
        kinds = [e["meta"]["kind"] for e in b["sft"]]
        self.assertEqual(kinds, ["action", "action", "say"])
        # the completion is the model's raw action; context ends with what it had just seen
        edit = b["sft"][1]
        self.assertIn('"edit_file"', edit["completion"])
        self.assertEqual(edit["messages"][-1]["role"], "user")   # the observation of the read
        self.assertIn("x = 1", edit["messages"][-1]["content"])

    def test_failed_action_is_not_a_positive_example(self):
        recs = [
            {"type": "user", "text": "go"},
            {"type": "model", "raw": '{"action":"edit_file","path":"x.py","old":"NOPE","new":"y"}'},
            {"type": "action", "action": "edit_file"},
            {"type": "observation", "text": "edit failed: old not found", "ok": False},
            {"type": "model", "raw": '{"action":"say","message":"done"}'},
            {"type": "assistant", "text": "done"},
        ]
        b = corpus.build(recs, sid="s")
        completions = [e["completion"] for e in b["sft"]]
        self.assertFalse(any("NOPE" in c for c in completions))   # the failing edit is excluded
        self.assertTrue(any('"say"' in c for c in completions))   # the say still counts

    def test_system_prompt_is_prepended(self):
        b = corpus.build(_clean_turn(), sid="s", system="SYS")
        self.assertEqual(b["sft"][0]["messages"][0], {"role": "system", "content": "SYS"})


class TestPreferencePairs(unittest.TestCase):
    def test_grammar_correction_pair(self):
        recs = [
            {"type": "user", "text": "go"},
            {"type": "model", "raw": "sure, here you go (not json)"},
            {"type": "malformed", "step": 1, "raw": "sure, here you go (not json)"},
            {"type": "model", "raw": '{"action":"read_file","path":"x.py"}'},
            {"type": "action", "action": "read_file"},
            {"type": "observation", "text": "ok", "ok": True},
        ]
        b = corpus.build(recs, sid="s")
        self.assertEqual(len(b["pref"]), 1)
        p = b["pref"][0]
        self.assertEqual(p["kind"], "grammar")
        self.assertIn("not json", p["rejected"])
        self.assertIn('"read_file"', p["chosen"])

    def test_narrate_correction_pair(self):
        recs = [
            {"type": "user", "text": "implement it"},
            {"type": "model", "raw": '{"action":"say","message":"I will now implement it. Let me start."}'},
            {"type": "narrate_bounce", "msg": "I will now implement it. Let me start."},
            {"type": "model", "raw": '{"action":"write_file","path":"x.py","content":"x=2\\n"}'},
            {"type": "action", "action": "write_file"},
            {"type": "observation", "text": "wrote x.py", "ok": True},
        ]
        b = corpus.build(recs, sid="s")
        self.assertEqual(len(b["pref"]), 1)
        p = b["pref"][0]
        self.assertEqual(p["kind"], "narrate")
        self.assertIn("I will now implement", p["rejected"])     # the preamble = rejected
        self.assertIn('"write_file"', p["chosen"])               # the real work = chosen

    def test_build_jsonl_tags_splits(self):
        recs = _clean_turn() + [
            {"type": "user", "text": "again"},
            {"type": "model", "raw": "not json"},
            {"type": "malformed", "step": 1, "raw": "not json"},
            {"type": "model", "raw": '{"action":"say","message":"ok"}'},
            {"type": "assistant", "text": "ok"},
        ]
        rows = corpus.build_jsonl(recs, sid="s")
        self.assertTrue(all(r["split"] in ("sft", "pref") for r in rows))
        self.assertTrue(any(r["split"] == "sft" for r in rows))


if __name__ == "__main__":
    unittest.main()
