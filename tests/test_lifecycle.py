"""H03 — complete action lifecycle: identity, single terminal, retry causality."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.lifecycle import (LifecycleTracker, Stage, TERMINAL, DuplicateTerminal,
                             outcome_for, LIFECYCLE_VERSION)     # noqa: E402


def _clock():
    t = {"n": 0}
    def tick():
        t["n"] += 1
        return float(t["n"])
    return tick


class TestIdentity(unittest.TestCase):
    def test_action_has_stable_composite_identity(self):
        tr = LifecycleTracker("run-1", clock=_clock())
        lc = tr.request("t1", "edit_file")
        self.assertEqual(lc.run_id, "run-1")
        self.assertEqual(lc.turn_id, "t1")
        self.assertEqual(lc.action_id, "a1")
        self.assertEqual(lc.attempt, 1)
        self.assertIsNone(lc.parent_action_id)
        self.assertEqual(lc.stage, Stage.REQUESTED)

    def test_ids_are_monotonic_and_deterministic(self):
        a = LifecycleTracker("r", clock=_clock())
        ids1 = [a.request("t", "read_file").action_id for _ in range(3)]
        b = LifecycleTracker("r", clock=_clock())
        ids2 = [b.request("t", "read_file").action_id for _ in range(3)]
        self.assertEqual(ids1, ["a1", "a2", "a3"])
        self.assertEqual(ids1, ids2)   # a replay reconstructs the same identities


class TestLifecycleReachesExactlyOneTerminal(unittest.TestCase):
    def _full(self, tr, kind):
        lc = tr.request("t1", kind)
        tr.authorize(lc)
        tr.start(lc)
        return lc

    def test_foreground_success(self):
        tr = LifecycleTracker("r", clock=_clock())
        lc = self._full(tr, "run_tests")
        tr.finish(lc, Stage.SUCCEEDED)
        self.assertEqual(lc.outcome, Stage.SUCCEEDED)
        self.assertTrue(lc.terminal)
        self.assertIn("terminal", lc.timestamps)

    def test_each_terminal_outcome_is_reachable(self):
        for term in (Stage.SUCCEEDED, Stage.FAILED, Stage.CANCELLED,
                     Stage.TIMED_OUT, Stage.INDETERMINATE):
            tr = LifecycleTracker("r", clock=_clock())
            lc = self._full(tr, "bash")
            tr.finish(lc, term)
            self.assertEqual(lc.outcome, term)

    def test_denial_terminates_before_start(self):
        tr = LifecycleTracker("r", clock=_clock())
        lc = tr.request("t1", "bash")
        tr.deny(lc, "requires admin authority")
        self.assertEqual(lc.outcome, Stage.DENIED)
        self.assertNotIn("started", lc.timestamps)   # never started

    def test_duplicate_terminal_is_invalid(self):
        tr = LifecycleTracker("r", clock=_clock())
        lc = self._full(tr, "bash")
        tr.finish(lc, Stage.SUCCEEDED)
        with self.assertRaises(DuplicateTerminal):
            tr.finish(lc, Stage.FAILED)
        with self.assertRaises(DuplicateTerminal):
            tr.start(lc)                              # can't progress a terminated action either

    def test_finish_rejects_a_non_terminal_stage(self):
        tr = LifecycleTracker("r", clock=_clock())
        lc = tr.request("t1", "bash")
        with self.assertRaises(ValueError):
            tr.finish(lc, Stage.STARTED)


class TestRetryCausality(unittest.TestCase):
    def test_retry_gets_new_id_and_attempt_but_keeps_causal_parent(self):
        tr = LifecycleTracker("r", clock=_clock())
        first = tr.request("t1", "edit_file")
        tr.authorize(first); tr.start(first); tr.finish(first, Stage.FAILED)
        retry = tr.retry_of(first)
        self.assertNotEqual(retry.action_id, first.action_id)
        self.assertEqual(retry.attempt, 2)
        self.assertEqual(retry.parent_action_id, first.action_id)   # causal chain intact
        self.assertEqual(retry.action_kind, first.action_kind)
        again = tr.retry_of(retry)
        self.assertEqual(again.attempt, 3)
        self.assertEqual(again.parent_action_id, retry.action_id)


class TestOutcomeMapping(unittest.TestCase):
    def test_outcome_for_precedence(self):
        self.assertEqual(outcome_for(True), Stage.SUCCEEDED)
        self.assertEqual(outcome_for(False), Stage.FAILED)
        self.assertEqual(outcome_for(True, cancelled=True), Stage.CANCELLED)   # cancel wins
        self.assertEqual(outcome_for(False, timed_out=True), Stage.TIMED_OUT)  # timeout wins over fail
        self.assertEqual(outcome_for(True, indeterminate=True), Stage.INDETERMINATE)


class TestSerialization(unittest.TestCase):
    def test_to_dict_is_json_and_versioned(self):
        tr = LifecycleTracker("r", clock=_clock())
        lc = tr.request("t1", "bash")
        tr.finish(lc, Stage.TIMED_OUT, "timed out after 30s")
        d = lc.to_dict()
        self.assertEqual(d["v"], LIFECYCLE_VERSION)
        self.assertEqual(d["outcome"], "timed_out")
        self.assertEqual(d["stage"], "timed_out")
        import json
        json.dumps(d)


class _CaptureSession:
    def __init__(self, cwd):
        self.cwd, self.sid, self.name = cwd, "run-cap", "cap"
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


class TestLifecycleInLoop(unittest.TestCase):
    def test_foreground_action_gets_a_complete_lifecycle(self):
        import tempfile
        from forge.agent import Agent
        write = '{"thought":"w","action":"write_file","path":"a.py","content":"x = 1\\n"}'
        say = '{"thought":"d","action":"say","message":"done"}'
        sess = _CaptureSession(tempfile.mkdtemp())
        Agent(_ScriptBackend([write, say]), sess, max_steps=6, autonomous=True).send("write a.py")
        lcs = [r for r in sess.records if r["type"] == "action_lifecycle"]
        self.assertTrue(lcs, "an executed action must log a lifecycle")
        wf = next(r for r in lcs if r["action_kind"] == "write_file")
        self.assertEqual(wf["outcome"], "succeeded")     # exactly one terminal, and it's a success
        self.assertEqual(wf["run_id"], "run-cap")
        self.assertTrue(wf["action_id"] and wf["turn_id"])
        self.assertIn("requested", wf["timestamps"])
        self.assertIn("terminal", wf["timestamps"])

    def _outcome_for_obs(self, action_json, obs, ok, background=False):
        """Drive one action through the loop with a canned execution result, and
        return the lifecycle outcome the loop recorded for it."""
        import tempfile
        from forge.agent import Agent
        say = '{"thought":"d","action":"say","message":"done"}'
        sess = _CaptureSession(tempfile.mkdtemp())
        a = Agent(_ScriptBackend([action_json, say]), sess, max_steps=4, autonomous=True,
                  levers=frozenset())     # bare loop — no dry-run resample to perturb the single action
        a._execute_and_record = lambda kind, act, raw, step, trace: (obs, ok, 0, None)
        a.send("go")
        lcs = [r for r in sess.records if r["type"] == "action_lifecycle" and r["action_kind"] != "say"]
        return lcs[-1]["outcome"] if lcs else None

    def test_success_output_mentioning_timeout_is_not_misclassified(self):
        # regression for the review finding: a SUCCESSFUL action whose output merely
        # contains "timed out after" must record succeeded, not timed_out.
        out = self._outcome_for_obs('{"thought":"r","action":"read_file","path":"x.py"}',
                                    "log: connection timed out after 30s", True)
        self.assertEqual(out, "succeeded")

    def test_real_timeout_maps_to_timed_out(self):
        out = self._outcome_for_obs('{"thought":"b","action":"bash","command":"sleep 99"}',
                                    "(timed out after 60s — the command was too slow.)", False)
        self.assertEqual(out, "timed_out")

    def test_cancellation_maps_to_cancelled(self):
        out = self._outcome_for_obs('{"thought":"b","action":"bash","command":"sleep 99"}',
                                    "(stopped by user)", False)
        self.assertEqual(out, "cancelled")

    def test_background_launch_is_indeterminate(self):
        out = self._outcome_for_obs(
            '{"thought":"s","action":"bash","command":"python server.py","background":true}',
            "✓ running in the background: pid 42", True, background=True)
        self.assertEqual(out, "indeterminate")

    def test_fleet_send_records_a_lifecycle(self):
        import tempfile
        from forge.agent import Agent
        roster = '{"thought":"f","action":"fleet_send","target":"list"}'   # read-only roster, no real delivery
        say = '{"thought":"d","action":"say","message":"done"}'
        sess = _CaptureSession(tempfile.mkdtemp())
        Agent(_ScriptBackend([roster, say]), sess, max_steps=4, autonomous=True).send("who's around")
        fs = [r for r in sess.records if r["type"] == "action_lifecycle" and r["action_kind"] == "fleet_send"]
        self.assertTrue(fs, "a successful fleet_send must record a lifecycle (symmetric with a denied one)")
        self.assertEqual(fs[-1]["outcome"], "succeeded")

    def test_denied_action_gets_a_denied_lifecycle(self):
        import tempfile
        from forge.agent import Agent
        from forge.authority import AuthorityPolicy
        a = Agent(_ScriptBackend(['{"thought":"d","action":"say","message":"x"}']),
                  _CaptureSession(tempfile.mkdtemp()), max_steps=2, autonomous=True)
        a.authority = AuthorityPolicy("observe")         # observer cannot edit
        a._gate("edit_file", {"action": "edit_file", "path": "x.py"})
        lcs = [r for r in a.session.records if r["type"] == "action_lifecycle"]
        self.assertTrue(lcs)
        self.assertEqual(lcs[-1]["outcome"], "denied")
        self.assertEqual(lcs[-1]["action_kind"], "edit_file")
        self.assertNotIn("started", lcs[-1]["timestamps"])   # denied before it ever started


if __name__ == "__main__":
    unittest.main()
