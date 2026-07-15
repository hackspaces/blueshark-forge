"""H02 — the authoritative execution reducer. Pure, deterministic, offline."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge.reducer import reduce, Transition, REDUCER_VERSION       # noqa: E402
from forge.execution import ExecutionState as S, RuntimeEvent as E   # noqa: E402
from forge.contract import TaskContract                              # noqa: E402

VERIFY_REQUIRED = TaskContract(requires_verification=True)
AUDIT = TaskContract(requires_verification=False)


class TestCompletionGate(unittest.TestCase):
    def test_mutation_cannot_jump_to_verified_completion(self):
        # THE defining invariant.
        t = reduce(S.MUTATE, E.COMPLETION_CLAIMED, VERIFY_REQUIRED)
        self.assertFalse(t.allowed)
        self.assertEqual(t.code, "unverified_completion")
        self.assertEqual(t.recovery_state, S.VERIFY)
        self.assertNotEqual(t.state_to, S.COMPLETE)

    def test_diagnose_cannot_complete_either(self):
        t = reduce(S.DIAGNOSE, E.COMPLETION_CLAIMED, VERIFY_REQUIRED)
        self.assertFalse(t.allowed)
        self.assertEqual(t.recovery_state, S.VERIFY)

    def test_completion_is_legal_from_verified_or_unmutated_states(self):
        for state in (S.VERIFY, S.ORIENT, S.INVESTIGATE, S.PLAN, S.COMPLETE):
            t = reduce(state, E.COMPLETION_CLAIMED, VERIFY_REQUIRED)
            self.assertTrue(t.allowed, state)
            self.assertEqual(t.state_to, S.COMPLETE, state)

    def test_audit_contract_permits_completion_from_mutate(self):
        # when the contract carries no verification obligation, the gate opens.
        t = reduce(S.MUTATE, E.COMPLETION_CLAIMED, AUDIT)
        self.assertTrue(t.allowed)
        self.assertEqual(t.state_to, S.COMPLETE)


class TestRecoveryPaths(unittest.TestCase):
    def test_authority_denied_is_rejected_and_recovers_to_plan(self):
        t = reduce(S.MUTATE, E.AUTHORITY_DENIED, VERIFY_REQUIRED)
        self.assertFalse(t.allowed)
        self.assertEqual(t.state_to, S.DIAGNOSE)
        self.assertEqual(t.recovery_state, S.PLAN)

    def test_failed_verification_enters_and_recovers_via_diagnose(self):
        t = reduce(S.VERIFY, E.VERIFICATION_FAILED, VERIFY_REQUIRED)
        self.assertEqual(t.state_to, S.DIAGNOSE)
        self.assertEqual(t.recovery_state, S.DIAGNOSE)

    def test_process_exit_and_loop_route_to_diagnose(self):
        for ev in (E.PROCESS_EXITED, E.LOOP_SUSPECTED):
            self.assertEqual(reduce(S.MUTATE, ev, VERIFY_REQUIRED).state_to, S.DIAGNOSE)


class TestActionTransitions(unittest.TestCase):
    def test_action_started_moves_to_the_right_state(self):
        cases = {"read_file": S.INVESTIGATE, "grep": S.INVESTIGATE, "plan": S.PLAN,
                 "edit_file": S.MUTATE, "write_file": S.MUTATE, "run_tests": S.VERIFY}
        for action, expect in cases.items():
            t = reduce(S.ORIENT, E.ACTION_STARTED, VERIFY_REQUIRED, action=action)
            self.assertTrue(t.allowed)
            self.assertEqual(t.state_to, expect, action)

    def test_workspace_changed_enters_mutate_with_a_verify_obligation(self):
        t = reduce(S.MUTATE, E.WORKSPACE_CHANGED, VERIFY_REQUIRED)
        self.assertEqual(t.state_to, S.MUTATE)
        self.assertEqual(t.recovery_state, S.DIAGNOSE)


class TestDeterminismAndSerialization(unittest.TestCase):
    def _drive(self, events):
        """Thread a sequence of (event, action) through the reducer, collecting the
        state after each step — exactly what a replay would reconstruct."""
        state = S.ORIENT
        seq = []
        for event, action in events:
            t = reduce(state, event, VERIFY_REQUIRED, action=action)
            state = t.state_to
            seq.append(state)
        return seq

    def test_same_events_reconstruct_the_same_state_sequence(self):
        events = [
            (E.ACTION_STARTED, "read_file"),     # INVESTIGATE
            (E.ACTION_STARTED, "edit_file"),     # MUTATE
            (E.WORKSPACE_CHANGED, ""),           # MUTATE
            (E.ACTION_STARTED, "run_tests"),     # VERIFY
            (E.VERIFICATION_PASSED, ""),         # VERIFY
            (E.COMPLETION_CLAIMED, ""),          # COMPLETE
        ]
        first = self._drive(events)
        second = self._drive(events)             # a "replay"
        self.assertEqual(first, second)          # deterministic
        self.assertEqual(first[-1], S.COMPLETE)
        self.assertEqual(first, [S.INVESTIGATE, S.MUTATE, S.MUTATE, S.VERIFY, S.VERIFY, S.COMPLETE])

    def test_transition_is_versioned_and_serializable(self):
        d = reduce(S.MUTATE, E.COMPLETION_CLAIMED, VERIFY_REQUIRED).to_dict()
        self.assertEqual(d["v"], REDUCER_VERSION)
        self.assertEqual(d["event"], "CompletionClaimed")
        self.assertEqual(d["state_from"], "MUTATE")
        self.assertEqual(d["recovery_state"], "VERIFY")
        import json
        json.dumps(d)   # must be JSON-serializable for the transcript


class _CaptureSession:
    """Captures log() records so a full Agent.send() run can be inspected."""
    def __init__(self, cwd):
        self.cwd, self.sid, self.name = cwd, "cap", "cap"
        self.model, self.status, self.port = "test", "idle", None
        self.records = []

    def log(self, kind, **f): self.records.append({"type": kind, **f})
    def drain(self): return []
    def set_status(self, s): self.status = s
    def register(self): pass
    def deregister(self): pass
    def push(self, *a): pass


class _ScriptBackend:
    name = "script"

    def __init__(self, actions):
        self.actions, self.i = actions, 0

    def stream(self, *a, **k):
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        yield act

    def chat(self, *a, **k): return "[summary]"
    def context_window(self): return 8192
    def effective_ctx(self): return 8192
    def warm(self): pass


class TestReducerControlsLoop(unittest.TestCase):
    def test_unverified_mutation_completion_is_flagged_by_the_reducer(self):
        # Drive a real loop: write a file (mutation, no verification), then claim done.
        # The reducer must record a MUTATE→(rejected, recover VERIFY) transition — the
        # state machine flagging that this is not a VERIFIED completion.
        import tempfile
        from forge.agent import Agent
        write = '{"thought":"w","action":"write_file","path":"a.py","content":"x = 1\\n"}'
        say = '{"thought":"d","action":"say","message":"done"}'
        sess = _CaptureSession(tempfile.mkdtemp())
        Agent(_ScriptBackend([write, say]), sess, max_steps=6, autonomous=True).send("write a.py")

        claims = [r for r in sess.records if r["type"] == "execution_transition"
                  and r.get("event") == "CompletionClaimed"]
        self.assertTrue(claims, "the loop must consult the reducer at the completion claim")
        vetoed = [t for t in claims if t["allowed"] is False]
        self.assertTrue(vetoed, "an unverified mutation completion must be flagged")
        self.assertEqual(vetoed[0]["state_from"], "MUTATE")
        self.assertEqual(vetoed[0]["recovery_state"], "VERIFY")
        # and the loop asked the reducer before executing the write, too
        started = [r for r in sess.records if r["type"] == "execution_transition"
                   and r.get("event") == "ActionStarted"]
        self.assertTrue(started, "the loop must consult the reducer before executing an action")

    def test_reducer_veto_blocks_a_stale_verified_completion(self):
        # The invariant hole the review found: a verification that PASSED, then a NEW
        # mutation, then a completion claim with no fresh test. The evidence keeps the
        # stale PASS so the policy would ACCEPT it as "verified" — but the reducer holds
        # MUTATE and its veto must now DOWNGRADE that accept to a rejection.
        import tempfile
        from forge.agent import Agent
        a = Agent(_ScriptBackend(['{"thought":"d","action":"say","message":"done"}']),
                  _CaptureSession(tempfile.mkdtemp()), max_steps=4, autonomous=True)
        a._begin_turn("do it")
        a.evidence.record_verification("pytest", True)   # a PASS lands in the evidence
        a._verified = True
        a._mutated.add("a.py")                            # then a NEW mutation invalidates it
        a.evidence.record_change("a.py")
        a._verified = False
        # sanity: the policy alone would accept this as verified (the stale pass)
        self.assertTrue(a.evidence.contract("done").verified)
        # but the done-gate must now REJECT it via the reducer veto (returns a bounce)
        bounce = a._done_gate("done")
        self.assertIsNotNone(bounce)
        self.assertEqual(a._completion_decision.code, "reducer_veto")
        self.assertFalse(a._completion_decision.allowed)


if __name__ == "__main__":
    unittest.main()
