"""H01 — the harness-owned task contract: build, serialize, recover, and its
presence in the session meta record. Stdlib, offline, no model calls."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import contract as C                # noqa: E402


class TestTaskContract(unittest.TestCase):
    def test_round_trips_through_dict(self):
        c = C.TaskContract(goal="fix x", mode="plan", authority="contribute",
                           allowed_actions=["read_file", "edit_file"],
                           completion_policy="strict", requires_verification=True,
                           max_steps=25, extensions={"domain": "test"})
        back = C.TaskContract.from_dict(c.to_dict())
        self.assertEqual(back, c)

    def test_from_dict_none_is_permissive_default_with_fallback_mode(self):
        c = C.TaskContract.from_dict(None, fallback_mode="manual")
        self.assertEqual(c.mode, "manual")            # legacy mode carried through
        self.assertEqual(c.authority, "operator")     # permissive default
        self.assertEqual(c.version, C.CONTRACT_VERSION)

    def test_from_dict_tolerates_partial_legacy_payload(self):
        c = C.TaskContract.from_dict({"mode": "auto", "authority": "observe"})
        self.assertEqual(c.authority, "observe")
        self.assertEqual(c.completion_policy, "balanced")   # defaulted
        self.assertEqual(c.max_steps, 0)
        self.assertEqual(c.allowed_actions, [])

    def test_permits(self):
        c = C.TaskContract(allowed_actions=["read_file", "grep"])
        self.assertTrue(c.permits("read_file"))
        self.assertFalse(c.permits("bash"))

    def test_from_runtime_sorts_actions_and_derives_verification(self):
        audit = C.from_runtime(goal="g", mode="auto", authority_level="operator",
                               allowed_actions={"bash", "read_file", "grep"},
                               completion_policy_mode="audit", max_steps=40)
        self.assertEqual(audit.allowed_actions, ["bash", "grep", "read_file"])  # sorted
        self.assertFalse(audit.requires_verification)   # audit never blocks → no obligation
        strict = C.from_runtime(goal="g", mode="auto", authority_level="admin",
                                allowed_actions=["bash"], completion_policy_mode="strict",
                                max_steps=40)
        self.assertTrue(strict.requires_verification)


class _CaptureSession:
    """Minimal session that records log() calls — enough to construct an Agent and
    inspect its meta record. Never registers, serves no inbox."""
    def __init__(self, cwd):
        self.sid, self.cwd, self.model, self.name = "cap", cwd, "test", "cap"
        self.status, self.port = "idle", None
        self.records = []

    def log(self, kind, **f): self.records.append({"type": kind, **f})
    def drain(self): return []
    def set_status(self, s): self.status = s
    def register(self): pass
    def deregister(self): pass
    def push(self, *a): pass


class _Backend:
    name = "test:model"

    def stream(self, *a, **k): yield ""
    def chat(self, *a, **k): return ""
    def context_window(self): return 8192
    def effective_ctx(self): return 8192
    def warm(self): pass


class TestContractInMeta(unittest.TestCase):
    def test_agent_records_a_contract_in_its_meta(self):
        import tempfile
        from forge.agent import Agent
        sess = _CaptureSession(tempfile.mkdtemp())
        a = Agent(_Backend(), sess, goal="fix the bug in x.py", max_steps=25)
        meta = next((r for r in sess.records if r["type"] == "meta"), None)
        self.assertIsNotNone(meta)
        self.assertIn("contract", meta)
        c = C.TaskContract.from_dict(meta["contract"])
        self.assertEqual(c.goal, "fix the bug in x.py")
        self.assertEqual(c.max_steps, 25)
        self.assertEqual(c.version, C.CONTRACT_VERSION)
        # the recorded contract must FAITHFULLY mirror the live runtime source, not
        # merely look plausible — pin it to authority itself so a narrowing regression
        # (e.g. sourcing a mode-filtered set, or hardcoding) fails here.
        self.assertEqual(c.authority, a.authority.level.name.lower())
        self.assertEqual(set(c.allowed_actions), set(a.authority.legal_actions()))
        self.assertEqual(c.allowed_actions, sorted(c.allowed_actions))

    def test_open_session_has_empty_goal_but_full_contract(self):
        # goal="" is the DEFAULT (interactive REPL, fleet-served, replay) — it must
        # stay empty (no fabricated placeholder) yet still carry a full contract.
        import tempfile
        from forge.agent import Agent
        sess = _CaptureSession(tempfile.mkdtemp())
        a = Agent(_Backend(), sess)                    # no goal
        c = C.TaskContract.from_dict(next(r for r in sess.records if r["type"] == "meta")["contract"])
        self.assertEqual(c.goal, "")
        self.assertEqual(set(c.allowed_actions), set(a.authority.legal_actions()))
        self.assertIn(c.authority, ("observe", "contribute", "operator", "admin"))

    def test_trace_shows_contract_and_omits_goal_when_open(self):
        # covers the render path + the `if c.goal:` suppression branch in cmd_trace.
        import io, glob
        from forge import __main__ as M, fleet, render
        orig_color = render.color_on
        render.color_on = lambda: False

        def run_trace(goal):
            c = C.from_runtime(goal=goal, mode="auto", authority_level="operator",
                               allowed_actions={"read_file", "say"},
                               completion_policy_mode="balanced", max_steps=10)
            recs = [{"type": "meta", "forge": "x", "model": "m", "mode": "auto",
                     "cwd": "/tmp", "contract": c.to_dict()},
                    {"type": "step", "step": 1, "action": "say", "ok": True, "elapsed_ms": 1}]
            orig_rec, orig_glob, orig_out = fleet._records, glob.glob, sys.stdout
            fleet._records = lambda sid, **k: recs
            glob.glob = lambda p: ["x"]
            out = io.StringIO()
            sys.stdout = out
            try:
                M.cmd_trace(type("A", (), {"sid": "s"})())
            finally:
                fleet._records, glob.glob, sys.stdout = orig_rec, orig_glob, orig_out
            return out.getvalue()

        try:
            with_goal = run_trace("do the thing")
            self.assertIn("contract:", with_goal)
            self.assertIn("goal:", with_goal)
            self.assertIn("do the thing", with_goal)
            open_sess = run_trace("")
            self.assertIn("contract:", open_sess)      # contract still shown
            self.assertNotIn("goal:", open_sess)        # but no goal line for an open session
        finally:
            render.color_on = orig_color


if __name__ == "__main__":
    unittest.main()
