"""H09 — event-sourced checkpoints + crash recovery."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import checkpoint as C                                    # noqa: E402
from forge.execution import ExecutionState                          # noqa: E402


def _log(path, *recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


class TestCorruptTailQuarantine(unittest.TestCase):
    def test_clean_log_reads_all_records(self):
        p = os.path.join(tempfile.mkdtemp(), "s.jsonl")
        _log(p, {"type": "meta"}, {"type": "user", "text": "hi"}, {"type": "assistant", "text": "ok"})
        r = C.read_committed(p)
        self.assertEqual(len(r.records), 3)
        self.assertTrue(r.clean)
        self.assertFalse(r.corrupt_tail)
        self.assertEqual(r.last_valid_offset, os.path.getsize(p))

    def test_torn_final_line_is_quarantined_and_prefix_recovered(self):
        # a process killed mid-append: the last record is a partial (un-terminated) line.
        p = os.path.join(tempfile.mkdtemp(), "s.jsonl")
        _log(p, {"type": "meta"}, {"type": "action", "action": "edit_file"})
        good_size = os.path.getsize(p)
        with open(p, "a") as f:
            f.write('{"type": "observation", "ok": tr')   # torn: no newline, invalid JSON
        r = C.read_committed(p)
        self.assertEqual(len(r.records), 2)                # the two committed records survive
        self.assertTrue(r.corrupt_tail)
        self.assertEqual(len(r.quarantined), 1)
        self.assertEqual(r.last_valid_offset, good_size)   # exact truncation point reported

    def test_kill_at_every_byte_boundary_never_loses_a_committed_record(self):
        # simulate a kill at each byte offset of the last record: every committed
        # (newline-terminated) record before the cut must always be recovered.
        base = os.path.join(tempfile.mkdtemp(), "s.jsonl")
        _log(base, {"type": "meta"}, {"type": "user", "text": "x"}, {"type": "action", "action": "bash"})
        with open(base, "rb") as f:
            full = f.read()
        committed_prefix_len = full.rfind(b"\n", 0, len(full) - 1) + 1   # end of the 2nd record
        for cut in range(committed_prefix_len + 1, len(full)):
            p = os.path.join(tempfile.mkdtemp(), "s.jsonl")
            with open(p, "wb") as f:
                f.write(full[:cut])
            r = C.read_committed(p)
            self.assertGreaterEqual(len(r.records), 2, f"cut={cut}")   # the 2 committed survive
            self.assertEqual(r.last_valid_offset, committed_prefix_len, f"cut={cut}")

    def test_missing_file_is_empty_recovery(self):
        r = C.read_committed("/no/such/path.jsonl")
        self.assertEqual(r.records, [])
        self.assertTrue(r.clean)


class TestIndeterminateReconciliation(unittest.TestCase):
    def test_indeterminate_last_action_needs_reconciliation(self):
        records = [{"type": "action_lifecycle", "action_kind": "bash", "outcome": "succeeded"},
                   {"type": "action_lifecycle", "action_kind": "bash", "outcome": "indeterminate"}]
        self.assertTrue(C.needs_reconciliation(records))   # may have executed → don't blind-retry

    def test_completed_last_action_is_safe_to_resume(self):
        records = [{"type": "action_lifecycle", "action_kind": "edit_file", "outcome": "succeeded"}]
        self.assertFalse(C.needs_reconciliation(records))

    def test_no_lifecycle_is_safe(self):
        self.assertFalse(C.needs_reconciliation([{"type": "user", "text": "hi"}]))

    def test_dangling_executed_action_without_terminal_needs_reconciliation(self):
        # the HARD-KILL case: the committed log ends at a complete `action` record whose
        # lifecycle terminal was never written (crash between committing and finishing).
        records = [{"type": "meta"}, {"type": "user", "text": "deploy"},
                   {"type": "action", "action": "bash", "args": {"command": "git push"}}]
        self.assertTrue(C.needs_reconciliation(records))     # may have run → don't blind-retry
        self.assertEqual(C.reconciliation_action(records), "bash")

    def test_completed_action_with_its_terminal_is_safe(self):
        records = [{"type": "action", "action": "bash"},
                   {"type": "observation", "ok": True},
                   {"type": "action_lifecycle", "action_kind": "bash", "outcome": "succeeded"}]
        self.assertFalse(C.needs_reconciliation(records))    # terminal present → completed

    def test_blocked_action_is_not_a_dangling_execution(self):
        # a read-before-edit block logs an `action` with no lifecycle even in a clean run
        # — it never executed, so it must NOT trigger reconciliation.
        records = [{"type": "action", "action": "edit_file", "args": {"blocked": "read-before-edit"}},
                   {"type": "assistant", "text": "done"}]
        self.assertFalse(C.needs_reconciliation(records))


class TestRecoveryState(unittest.TestCase):
    def test_replays_committed_events_to_a_deterministic_state(self):
        records = [
            {"type": "meta", "mode": "auto"},
            {"type": "action", "action": "edit_file"},
            {"type": "observation", "ok": True},          # WorkspaceChanged → MUTATE
        ]
        self.assertEqual(C.recovery_state(records), ExecutionState.MUTATE)
        # a failed verification lands in DIAGNOSE
        records.append({"type": "action", "action": "run_tests"})
        records.append({"type": "observation", "ok": False})
        self.assertEqual(C.recovery_state(records), ExecutionState.DIAGNOSE)

    def test_recovery_state_is_deterministic(self):
        records = [{"type": "meta"}, {"type": "action", "action": "read_file"},
                   {"type": "observation", "ok": True}]
        self.assertEqual(C.recovery_state(records), C.recovery_state(records))


class TestResumeRecovery(unittest.TestCase):
    def test_resume_reads_committed_records_and_flags_recovery(self):
        from forge import resume, session as sessmod
        d = tempfile.mkdtemp()
        orig = sessmod.SESSIONS
        sessmod.SESSIONS = d
        try:
            sid = "recov1"
            p = os.path.join(d, sid + ".jsonl")
            _log(p, {"type": "meta", "mode": "auto"},
                    {"type": "user", "text": "do it"},
                    {"type": "action_lifecycle", "action_kind": "bash", "outcome": "indeterminate"})
            with open(p, "a") as f:
                f.write('{"type": "observation", "ok": tr')   # a torn tail (crash mid-append)
            data = resume.load(sid)
            self.assertIsNotNone(data)
            self.assertTrue(data["recovery"]["corrupt_tail"])          # torn tail quarantined
            self.assertEqual(data["recovery"]["reconcile_action"], "bash")  # last action may have run
        finally:
            sessmod.SESSIONS = orig


if __name__ == "__main__":
    unittest.main()
