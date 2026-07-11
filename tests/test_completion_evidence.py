"""Evidence-aware completion gate tests."""
import os
import tempfile
import unittest
import unittest.mock as mock

from forge.agent import Agent
from forge.execution import CompletionPolicy, EvidenceContract, VerificationEvidence


class _Backend:
    name = "evidence-test"

    def __init__(self, raw='{"action":"say","message":"done"}'):
        self.raw = raw
        self.last_prompt_tokens = 0

    def stream(self, messages, schema=None, temperature=0.0):
        yield self.raw

    def effective_ctx(self):
        return 8192


class _Session:
    def __init__(self, cwd):
        self.cwd = cwd
        self.sid = "evidence"
        self.name = "evidence"
        self.status = "idle"
        self.logs = []

    def log(self, kind, **fields):
        self.logs.append((kind, fields))

    def drain(self):
        return []

    def set_status(self, status):
        self.status = status


class TestEvidenceAwareDoneGate(unittest.TestCase):
    def _agent_with_change(self):
        cwd = tempfile.mkdtemp()
        path = os.path.join(cwd, "changed.py")
        with open(path, "w") as stream:
            stream.write("x = 1\n")
        session = _Session(cwd)
        agent = Agent(_Backend(), session, max_steps=3)
        agent._mutated = {path}
        agent.evidence.record_change(path)
        return agent, session

    def test_passing_done_check_builds_verified_contract(self):
        agent, _session = self._agent_with_change()
        with mock.patch("forge.fleet.detect_test_cmd", return_value="python -m unittest"), \
                mock.patch("forge.tools._run", return_value=("Ran 12 tests\nOK", True)):
            self.assertIsNone(agent._done_gate("implemented the change"))
        contract = agent.evidence.contract("implemented the change")
        self.assertTrue(contract.verified)
        self.assertEqual(contract.changed_files, ["changed.py"])
        self.assertEqual(contract.verification[0].command, "python -m unittest")
        self.assertEqual(len(contract.verification[0].artifact_digest), 16)

    def test_failed_done_check_logs_rejected_contract(self):
        agent, session = self._agent_with_change()
        with mock.patch("forge.fleet.detect_test_cmd", return_value="python -m unittest"), \
                mock.patch("forge.tools._run", return_value=("FAILED (failures=1)", False)):
            bounce = agent._done_gate("implemented the change")
        self.assertIn("verification_failed", bounce)
        rejected = [fields for kind, fields in session.logs
                    if kind == "completion_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["policy"]["code"], "verification_failed")
        self.assertEqual(rejected[0]["evidence"]["verification"][0]["exit_code"], 1)

    def test_missing_runner_is_an_explicit_assumption(self):
        agent, _session = self._agent_with_change()
        with mock.patch("forge.fleet.detect_test_cmd", return_value=None):
            self.assertIsNone(agent._done_gate("implemented the change"))
        contract = agent.evidence.contract("implemented the change")
        self.assertIn("no test command detected", contract.unverified_assumptions)
        self.assertFalse(contract.verified)

    def test_accepted_completion_record_carries_evidence(self):
        cwd = tempfile.mkdtemp()
        session = _Session(cwd)
        result = Agent(_Backend(), session, max_steps=2).send("answer")
        self.assertEqual(result, "done")
        accepted = [fields for kind, fields in session.logs if kind == "assistant"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["evidence"]["claim"], "done")
        self.assertTrue(accepted[0]["verified"])


class TestCompletionPolicy(unittest.TestCase):
    def _contract(self, checks=(), assumptions=()):
        return EvidenceContract("done", ["changed.py"], list(checks), list(assumptions))

    def test_balanced_rejects_failed_first_claim_then_audits_override(self):
        policy = CompletionPolicy("balanced")
        contract = self._contract([VerificationEvidence("pytest", 1, "bad")])
        first = policy.evaluate(contract, attempt=1)
        second = policy.evaluate(contract, attempt=2)
        self.assertFalse(first.allowed)
        self.assertEqual(first.code, "verification_failed")
        self.assertTrue(second.allowed)
        self.assertEqual(second.outcome.value, "accept_unverified")
        self.assertTrue(second.override_used)

    def test_strict_never_accepts_changed_files_without_passing_evidence(self):
        policy = CompletionPolicy("strict")
        missing = self._contract(assumptions=["no test command detected"])
        failed = self._contract([VerificationEvidence("pytest", 1)])
        self.assertFalse(policy.evaluate(missing, attempt=9).allowed)
        self.assertFalse(policy.evaluate(failed, attempt=9).allowed)

    def test_audit_mode_labels_but_never_blocks(self):
        decision = CompletionPolicy("audit").evaluate(
            self._contract([VerificationEvidence("pytest", 1)]))
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.code, "audit_only")

    def test_verified_and_no_change_contracts_are_accepted(self):
        policy = CompletionPolicy("strict")
        verified = self._contract([VerificationEvidence("pytest", 0, "ok")])
        no_change = EvidenceContract("answered a question")
        self.assertEqual(policy.evaluate(verified).code, "verified")
        self.assertEqual(policy.evaluate(no_change).code, "no_changes")


class TestAgentPolicyIntegration(unittest.TestCase):
    def test_balanced_second_claim_is_an_explicit_override(self):
        agent, session = TestEvidenceAwareDoneGate()._agent_with_change()
        agent.completion_policy = CompletionPolicy("balanced")
        with mock.patch("forge.fleet.detect_test_cmd", return_value="python -m unittest"), \
                mock.patch("forge.tools._run", return_value=("FAILED", False)):
            self.assertIsNotNone(agent._done_gate("done"))
            self.assertIsNone(agent._done_gate("done anyway"))
        decisions = [fields["decision"] for kind, fields in session.logs
                     if kind == "completion_policy"]
        self.assertEqual(decisions[-1]["code"], "single_bounce_override")
        self.assertTrue(decisions[-1]["override_used"])

    def test_strict_policy_rejects_repeated_unverified_claims(self):
        agent, _session = TestEvidenceAwareDoneGate()._agent_with_change()
        agent.completion_policy = CompletionPolicy("strict")
        with mock.patch("forge.fleet.detect_test_cmd", return_value=None):
            self.assertIsNotNone(agent._done_gate("done"))
            self.assertIsNotNone(agent._done_gate("still done"))


if __name__ == "__main__":
    unittest.main()
