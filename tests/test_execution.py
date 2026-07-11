"""Execution-state protocol and evidence-contract tests."""
import json
import os
import tempfile
import unittest

from forge.execution import (EvidenceContract, ExecutionState, ExecutionTracker,
                             VerificationEvidence)
from forge.session import Session


class TestEvidenceContract(unittest.TestCase):
    def test_verified_contract_round_trips(self):
        contract = EvidenceContract(
            claim="implemented dry-run support",
            changed_files=["src/cli.py", "tests/test_cli.py"],
            verification=[VerificationEvidence("pytest tests/test_cli.py", 0, "abc123")],
        )
        rebuilt = EvidenceContract.from_dict(contract.to_dict())
        self.assertEqual(rebuilt, contract)
        self.assertTrue(rebuilt.verified)

    def test_changed_files_require_evidence(self):
        contract = EvidenceContract("changed it", ["src/cli.py"])
        self.assertIn("changed files require verification evidence", contract.validate())
        self.assertFalse(contract.verified)

    def test_failed_check_is_not_verified(self):
        contract = EvidenceContract("changed it", ["x.py"],
                                    [VerificationEvidence("python -m unittest", 1)])
        self.assertFalse(contract.verified)


class TestExecutionTracker(unittest.TestCase):
    def test_mutation_creates_verification_obligation(self):
        tracker = ExecutionTracker()
        start = tracker.observe("action", {"action": "edit_file"})
        changed = tracker.observe("observation", {"ok": True})
        self.assertEqual(start["state_to"], "MUTATE")
        self.assertEqual(changed["event"], "WorkspaceChanged")
        self.assertEqual(changed["state_to"], "MUTATE")
        self.assertIn("verify", changed["verification_obligation"])

    def test_failed_verification_enters_diagnose(self):
        tracker = ExecutionTracker()
        tracker.observe("action", {"action": "run_tests"})
        event = tracker.observe("observation", {"ok": False})
        self.assertEqual(event["event"], "VerificationFailed")
        self.assertEqual(event["state_to"], "DIAGNOSE")
        self.assertEqual(event["recovery_transition"], "DIAGNOSE")

    def test_completion_rejection_has_recovery(self):
        tracker = ExecutionTracker(ExecutionState.MUTATE)
        event = tracker.observe("narrate_bounce", {})
        self.assertEqual(event["event"], "CompletionRejected")
        self.assertEqual(event["recovery_transition"], "VERIFY")

    def test_background_crash_is_a_first_class_event(self):
        tracker = ExecutionTracker()
        event = tracker.observe("inbox", {"sender": "background", "text": "pid 7 EXITED code 1"})
        self.assertEqual(event["event"], "ProcessExited")
        self.assertEqual(event["state_to"], "DIAGNOSE")

    def test_completion_claim_projects_to_complete(self):
        tracker = ExecutionTracker()
        event = tracker.observe("assistant", {"text": "all done"})
        self.assertEqual(event["event"], "CompletionClaimed")
        self.assertEqual(event["state_to"], "COMPLETE")

    def test_stuck_handoff_is_not_a_completion(self):
        # A stuck message is an `assistant` record but the agent is giving up, not
        # finishing — it must not be projected as a completion into COMPLETE.
        tracker = ExecutionTracker(ExecutionState.DIAGNOSE)
        event = tracker.observe("assistant", {"text": "I'm stuck…", "stuck": True})
        self.assertIsNone(event)
        self.assertEqual(tracker.state, ExecutionState.DIAGNOSE)


class TestSessionProjection(unittest.TestCase):
    def test_log_keeps_legacy_shape_and_adds_runtime_envelope(self):
        session = object.__new__(Session)
        session.path = os.path.join(tempfile.mkdtemp(), "session.jsonl")
        session.execution = ExecutionTracker()
        session.log("action", action="edit_file", args={"path": "x.py"})
        session.log("observation", text="edited x.py", ok=True)
        with open(session.path) as stream:
            rows = [json.loads(line) for line in stream]
        self.assertEqual(rows[0]["type"], "action")
        self.assertEqual(rows[0]["action"], "edit_file")
        self.assertEqual(rows[0]["runtime"]["event"], "ActionStarted")
        self.assertEqual(rows[1]["runtime"]["event"], "WorkspaceChanged")
        self.assertEqual(rows[1]["runtime"]["verification_obligation"],
                         "verify the changed workspace before completion")


if __name__ == "__main__":
    unittest.main()
