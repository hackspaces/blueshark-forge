"""Deterministic zero-inference fault-injection tests."""
import copy
import unittest

from forge import faults
from forge import replay as replaymod


SAY = '{"thought":"done","action":"say","message":"all set"}'
BASH = '{"thought":"look","action":"bash","command":"echo hi"}'
EDIT = ('{"thought":"edit","action":"edit_file","path":"x.py",'
        '"start_line":1,"end_line":1,"anchor":"x = 1","new":"x = 2"}')


def _turns(*raws):
    return [{"user": "do it", "model": [
        {"raw": raw, "tier": 0, "prompt_tokens": 10} for raw in raws]}]


class TestFaultInjection(unittest.TestCase):
    def test_injection_never_mutates_source_turns(self):
        source = _turns(BASH, SAY)
        original = copy.deepcopy(source)
        changed, rows = faults.inject(source, ["truncate_output"])
        self.assertEqual(source, original)
        self.assertNotEqual(changed, source)
        self.assertTrue(rows[0].injected)

    def test_truncate_output_cuts_first_raw(self):
        changed, rows = faults.inject(_turns(BASH, SAY), ["truncate_output"])
        self.assertLess(len(changed[0]["model"][0]["raw"]), len(BASH))
        self.assertEqual(rows[0].step, 1)

    def test_malformed_burst_inserts_three_rows(self):
        changed, rows = faults.inject(_turns(SAY), ["malformed_burst"])
        self.assertEqual(len(changed[0]["model"]), 4)
        self.assertTrue(all(r["raw"] == "{truncated"
                            for r in changed[0]["model"][:3]))
        self.assertTrue(rows[0].injected)

    def test_wrong_anchor_only_changes_line_anchored_edit(self):
        changed, rows = faults.inject(_turns(EDIT, SAY), ["wrong_edit_anchor"])
        self.assertIn("__FORGE_FAULT_INCORRECT_ANCHOR__",
                      changed[0]["model"][0]["raw"])
        self.assertTrue(rows[0].injected)
        _unchanged, skipped = faults.inject(_turns(BASH, SAY), ["wrong_edit_anchor"])
        self.assertFalse(skipped[0].injected)

    def test_force_compaction_changes_recorded_token_pressure(self):
        changed, rows = faults.inject(_turns(BASH), ["force_compaction"])
        self.assertEqual(changed[0]["model"][0]["prompt_tokens"], 10 ** 9)
        self.assertTrue(rows[0].injected)

    def test_authority_fault_becomes_admin_only_action(self):
        changed, rows = faults.inject(_turns(BASH, SAY), ["authority_violation"])
        self.assertIn("sudo true", changed[0]["model"][0]["raw"])
        self.assertTrue(rows[0].injected)

    def test_unknown_fault_is_reported_not_raised(self):
        changed, rows = faults.inject(_turns(SAY), ["does-not-exist"])
        self.assertEqual(changed, _turns(SAY))
        self.assertFalse(rows[0].injected)
        self.assertIn("unknown", rows[0].detail)


class _Session:
    def __init__(self, records):
        self.records = records


class TestFaultMetrics(unittest.TestCase):
    def test_score_reports_recovery_and_efficiency(self):
        result = {
            "terminals": ["done"],
            "session": _Session([
                {"type": "model", "prompt_tokens": 12},
                {"type": "action", "action": "bash"},
                {"type": "observation", "ok": False},
                {"type": "authority_denied"},
                {"type": "assistant", "verified": True},
            ]),
        }
        injection = [faults.Injection("authority_violation", True, 1, "x")]
        row = faults.score(result, injection)
        self.assertTrue(row["recovered"])
        self.assertEqual(row["authority_denials"], 1)
        self.assertEqual(row["action_count"], 1)
        self.assertEqual(row["context_tokens"], 12)
        self.assertEqual(row["tool_call_efficiency"], 1.0)

    def test_unverified_acceptance_is_false_completion(self):
        result = {
            "terminals": ["claimed done"],
            "session": _Session([
                {"type": "assistant", "verified": False},
            ]),
        }
        row = faults.score(
            result, [faults.Injection("truncate_output", True, 1, "x")])
        self.assertTrue(row["false_completion"])
        self.assertFalse(row["recovered"])

    def test_report_is_compact_and_explicit(self):
        injection = [faults.Injection("truncate_output", True, 1, "cut")]
        metrics = faults.score(
            {"terminals": ["done"], "session": _Session([])}, injection)
        text = faults.report(injection, metrics)
        self.assertIn("zero inference", text)
        self.assertIn("truncate_output", text)
        self.assertIn("recovered:", text)
        self.assertIn("false-completion:", text)


class TestReplayFaultIntegration(unittest.TestCase):
    def test_recorded_session_recovers_from_one_truncated_output(self):
        transcript = [
            {"type": "meta", "model": "m", "mode": "auto", "window": 8192},
            {"type": "user", "text": "do it"},
            {"type": "model", "raw": BASH, "tier": 0, "prompt_tokens": 2},
            {"type": "model", "raw": SAY, "tier": 0, "prompt_tokens": 3},
        ]
        original = replaymod._records_for
        replaymod._records_for = lambda sid: transcript
        try:
            report = replaymod.replay_faults("sid", ["truncate_output"])
        finally:
            replaymod._records_for = original
        self.assertIn("truncate_output", report)
        self.assertIn("recovered: YES", report)
        self.assertIn("terminal: all set", report)

    def test_authority_violation_is_denied_during_replay(self):
        transcript = [
            {"type": "meta", "model": "m", "mode": "auto", "window": 8192},
            {"type": "user", "text": "do it"},
            {"type": "model", "raw": BASH, "tier": 0, "prompt_tokens": 2},
            {"type": "model", "raw": SAY, "tier": 0, "prompt_tokens": 3},
        ]
        original = replaymod._records_for
        replaymod._records_for = lambda sid: transcript
        try:
            report = replaymod.replay_faults("sid", ["authority_violation"])
        finally:
            replaymod._records_for = original
        self.assertIn("authority-denials: 1", report)
        self.assertIn("recovered: YES", report)


if __name__ == "__main__":
    unittest.main()
