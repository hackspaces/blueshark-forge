"""Evidence-aware completion gate tests."""
import io
import os
import sys
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

    def test_balanced_failed_verification_is_not_overridable_by_repetition(self):
        # H05: a repeated claim must NOT convert failed verification into success.
        policy = CompletionPolicy("balanced")
        contract = self._contract([VerificationEvidence("pytest", 1, "bad")])
        self.assertFalse(policy.evaluate(contract, attempt=1).allowed)
        repeat = policy.evaluate(contract, attempt=2)          # the model just says done again
        self.assertFalse(repeat.allowed)                       # still rejected
        self.assertEqual(repeat.code, "verification_failed")
        approved = policy.evaluate(contract, approved=True)    # only a RECORDED approval overrides
        self.assertTrue(approved.allowed)
        self.assertEqual(approved.outcome.value, "accept_unverified")   # never "verified"
        self.assertEqual(approved.code, "approved_override")
        self.assertTrue(approved.override_used)

    def test_strict_is_never_overridable_even_with_approval(self):
        policy = CompletionPolicy("strict")
        failed = self._contract([VerificationEvidence("pytest", 1)])
        self.assertFalse(policy.evaluate(failed, approved=True).allowed)   # approval can't rescue strict

    def test_a_passing_check_supersedes_a_prior_failure(self):
        # H05 fix-then-pass: edit → tests FAIL → fix → tests PASS → the contract is
        # verified. A stale failure must not poison the turn (the common dev loop).
        from forge.execution import EvidenceCollector
        ev = EvidenceCollector(tempfile.mkdtemp())
        ev.record_change("app.py")
        ev.record_verification("pytest", False, "1 failed")   # first run fails
        ev.record_verification("pytest", True, "5 passed")    # after the fix, it passes
        contract = ev.contract("done")
        self.assertTrue(contract.verified)                    # not poisoned by the stale failure
        self.assertEqual(CompletionPolicy("balanced").evaluate(contract).code, "verified")
        self.assertEqual(CompletionPolicy("strict").evaluate(contract).code, "verified")
        # a NEW failure after a pass (a regression) still rejects
        ev.record_verification("pytest", False, "broke it")
        self.assertFalse(ev.contract("done").verified)

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
    def test_balanced_failed_completion_needs_approval_not_repetition(self):
        agent, session = TestEvidenceAwareDoneGate()._agent_with_change()
        agent.completion_policy = CompletionPolicy("balanced")
        with mock.patch("forge.fleet.detect_test_cmd", return_value="python -m unittest"), \
                mock.patch("forge.tools._run", return_value=("FAILED", False)):
            self.assertIsNotNone(agent._done_gate("done"))          # 1st claim: rejected
            self.assertIsNotNone(agent._done_gate("done anyway"))   # H05: repeating does NOT accept
            agent.approve_unverified = lambda reason: True          # a recorded approval arrives
            self.assertIsNone(agent._done_gate("accept it"))        # now accepted (as unverified)
        decisions = [fields["decision"] for kind, fields in session.logs
                     if kind == "completion_policy"]
        self.assertEqual(decisions[-1]["code"], "approved_override")
        self.assertEqual(decisions[-1]["outcome"], "accept_unverified")   # never "verified"
        self.assertTrue(decisions[-1]["override_used"])
        self.assertTrue(any(kind == "completion_approval" for kind, _ in session.logs))  # recorded

    def test_strict_policy_rejects_repeated_unverified_claims(self):
        agent, _session = TestEvidenceAwareDoneGate()._agent_with_change()
        agent.completion_policy = CompletionPolicy("strict")
        with mock.patch("forge.fleet.detect_test_cmd", return_value=None):
            self.assertIsNotNone(agent._done_gate("done"))
            self.assertIsNotNone(agent._done_gate("still done"))

    def test_strict_failed_verification_is_not_overridable_by_approval_at_loop(self):
        # H05 loop-level twin of test_strict_is_never_overridable_even_with_approval:
        # even with a granted approval hook, strict mode must still BOUNCE a failed
        # verification at the done-gate (the approval branch is gated to balanced only).
        agent, session = TestEvidenceAwareDoneGate()._agent_with_change()
        agent.completion_policy = CompletionPolicy("strict")
        agent.approve_unverified = lambda reason: True          # a standing approval
        with mock.patch("forge.fleet.detect_test_cmd", return_value="python -m unittest"), \
                mock.patch("forge.tools._run", return_value=("FAILED", False)):
            self.assertIsNotNone(agent._done_gate("done"))       # approval cannot rescue strict
            self.assertIsNotNone(agent._done_gate("done anyway"))
        self.assertFalse(agent._completion_decision.allowed)
        self.assertFalse(agent._completion_decision.override_used)
        self.assertEqual(agent._completion_decision.code, "verification_failed")
        self.assertFalse(agent._approved_unverified)             # never flipped under strict
        self.assertFalse(any(kind == "completion_approval" for kind, _ in session.logs))

    def test_approved_override_emits_receipt_marked_approved_but_unverified(self):
        # H05 criterion "approval appears in the receipt": once a recorded approval
        # accepts a failed verification, the emitted v2 receipt carries approved=True
        # while remaining NOT verified (a failed check is still a failed check).
        agent, session = TestEvidenceAwareDoneGate()._agent_with_change()
        agent.completion_policy = CompletionPolicy("balanced")
        with mock.patch("forge.fleet.detect_test_cmd", return_value="python -m unittest"), \
                mock.patch("forge.tools._run", return_value=("FAILED", False)):
            self.assertIsNotNone(agent._done_gate("done"))       # rejected first
            before = [f for k, f in session.logs if k == "evidence_receipt"][-1]
            self.assertFalse(before["approved"])                 # no approval yet
            agent.approve_unverified = lambda reason: True        # a recorded approval arrives
            self.assertIsNone(agent._done_gate("accept it"))     # accepted as unverified
        after = [f for k, f in session.logs if k == "evidence_receipt"][-1]
        self.assertTrue(after["approved"])                       # H05: recorded in the receipt
        self.assertFalse(agent.evidence.contract("accept it").verified)   # still not verified


class TestNonInteractiveExit(unittest.TestCase):
    """H05 criterion 3: a non-interactive `forge run` whose completion the harness
    REJECTED must exit non-zero, so CI never reads an unverified stop as success."""

    def _run_cmd_run(self, decision):
        import forge.agent as agentmod
        from forge.execution import CompletionDecision
        from forge import __main__ as M

        class _FakeBackend:
            name = "fake"
            def effective_ctx(self):
                return 8192

        class _FakeSession:
            name = "fake"
            def deregister(self):
                pass

        class _FakeAgent:
            def __init__(self, *a, **k):
                self._completion_decision = None
            def send(self, task):
                self._completion_decision = decision
                return "stopped"

        args = type("A", (), {"model": "m", "dir": ".", "task": "do it", "max_steps": 3})()
        orig = (M._make_ladder, M._new_session, M._workspace_ctx, agentmod.Agent,
                sys.stderr, sys.stdout)
        M._make_ladder = lambda spec: [_FakeBackend()]
        M._new_session = lambda name, cwd: _FakeSession()
        M._workspace_ctx = lambda cwd, budget=None: ""
        agentmod.Agent = _FakeAgent
        sys.stderr = io.StringIO()
        sys.stdout = io.StringIO()
        try:
            code = None
            try:
                M.cmd_run(args)
            except SystemExit as e:
                code = e.code
            return code
        finally:
            (M._make_ladder, M._new_session, M._workspace_ctx,
             agentmod.Agent, sys.stderr, sys.stdout) = orig

    def test_rejected_completion_exits_nonzero(self):
        from forge.execution import CompletionDecision, PolicyOutcome, ExecutionState
        rejected = CompletionDecision(PolicyOutcome.REJECT, "verification_missing",
                                      "changed files require passing verification evidence",
                                      ExecutionState.VERIFY)
        self.assertEqual(self._run_cmd_run(rejected), 1)

    def test_accepted_completion_exits_zero(self):
        from forge.execution import CompletionDecision, PolicyOutcome
        accepted = CompletionDecision(PolicyOutcome.ACCEPT_UNVERIFIED, "verifier_unavailable",
                                      "no runnable verifier was available")
        self.assertIsNone(self._run_cmd_run(accepted))   # no SystemExit → clean exit


if __name__ == "__main__":
    unittest.main()
