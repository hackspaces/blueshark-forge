"""Evidence-aware completion gate tests."""
import os
import tempfile
import unittest
import unittest.mock as mock

from forge.agent import Agent


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
        self.assertIn("FAILS", bounce)
        rejected = [fields for kind, fields in session.logs
                    if kind == "completion_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "verification failed")
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


if __name__ == "__main__":
    unittest.main()
