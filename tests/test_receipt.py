"""H04 — evidence receipt v2: workspace-anchored, staleness-invalidating receipts."""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import receipt as R                                    # noqa: E402
from forge.receipt import EvidenceReceipt, Check, workspace_digest, RECEIPT_VERSION  # noqa: E402


def _write(d, name, content):
    with open(os.path.join(d, name), "w") as f:
        f.write(content)


def _passing_receipt(cwd, claim="done"):
    return EvidenceReceipt(
        claim=claim, contract_id="c1",
        verified_workspace=workspace_digest(cwd),
        checks=[Check("run_tests", "pytest", 0, "abc", 1.0)])


class TestWorkspaceDigest(unittest.TestCase):
    def test_identical_content_yields_identical_digest_git_or_not(self):
        # acceptance: identical content -> identical digest. ONE backend now, so a git
        # repo and a plain dir with the same files agree (git internals are skipped).
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        for d in (a, b):
            _write(d, "x.py", "print('hi')\n")
            _write(d, "y.txt", "same\n")
        subprocess.run(["git", "-C", b, "init", "-q"], check=True)   # only b is a git repo
        self.assertEqual(workspace_digest(a), workspace_digest(b))

    def test_a_content_change_flips_the_digest(self):
        for git in (False, True):
            d = tempfile.mkdtemp()
            _write(d, "x.py", "a\n")
            if git:
                subprocess.run(["git", "-C", d, "init", "-q"], check=True)
            before = workspace_digest(d)
            _write(d, "x.py", "b\n")
            self.assertNotEqual(workspace_digest(d), before, f"git={git}")

    def test_gitignored_deliverable_change_still_invalidates(self):
        # review finding: the old git backend skipped .gitignored files, so a change to
        # a gitignored DELIVERABLE (build output, .env, data) did NOT flip the digest.
        d = tempfile.mkdtemp()
        subprocess.run(["git", "-C", d, "init", "-q"], check=True)
        _write(d, ".gitignore", "build.log\n")
        _write(d, "build.log", "run 1\n")
        before = workspace_digest(d)
        _write(d, "build.log", "run 2\n")             # a gitignored deliverable changes
        self.assertNotEqual(workspace_digest(d), before)   # it MUST invalidate now

    def test_volatile_tooling_files_do_not_flip_the_digest(self):
        # review finding: OS/editor/test artifacts must not cause false invalidation.
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        before = workspace_digest(d)
        for junk in (".DS_Store", ".coverage", "app.pyc", "app.py.swp", "app.py~"):
            _write(d, junk, "noise\n")
        self.assertEqual(workspace_digest(d), before)   # unchanged — junk is skipped


class TestReceiptStaleness(unittest.TestCase):
    def test_changing_a_file_after_verification_invalidates_the_receipt(self):
        # THE headline acceptance.
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        rcpt = _passing_receipt(d)
        self.assertTrue(rcpt.verified(d))            # fresh: valid
        _write(d, "app.py", "x = 2\n")               # change a file AFTER verification
        self.assertFalse(rcpt.verified(d))           # receipt is now stale → invalid

    def test_a_receipt_with_no_passing_check_is_not_verified(self):
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        failed = EvidenceReceipt(claim="d", verified_workspace=workspace_digest(d),
                                 checks=[Check("run_tests", "pytest", 1, "z", 1.0)])
        self.assertFalse(failed.verified(d))
        no_ws = EvidenceReceipt(claim="d", checks=[Check("run_tests", "pytest", 0, "z", 1.0)])
        self.assertFalse(no_ws.verified(d))          # no workspace captured → cannot vouch


class TestOpaqueChanges(unittest.TestCase):
    def test_opaque_mutation_is_not_a_specific_verified_file_list(self):
        # acceptance: an unattributed (bash) mutation must not masquerade as measured files.
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        rcpt = EvidenceReceipt(
            claim="ran a script", verified_workspace=workspace_digest(d),
            opaque_changes=True, changed_paths=[],   # nothing MEASURED
            checks=[Check("run_tests", "pytest", 0, "ok", 1.0)])
        self.assertTrue(rcpt.opaque_changes)
        self.assertEqual(rcpt.changed_paths, [])      # no fabricated file list
        # and it still tracks the real workspace state for staleness
        _write(d, "app.py", "x = 2\n")
        self.assertFalse(rcpt.verified(d))


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


class TestStaleVerificationInLoop(unittest.TestCase):
    def test_a_change_after_verification_invalidates_it_at_the_done_gate(self):
        # H04 integration: a check passes and pins the workspace; a later change makes
        # the verification deterministically stale even though the _verified FLAG is still
        # set — the done-gate invalidates it via the content digest, not just the flags.
        from forge.agent import Agent
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        a = Agent(_ScriptBackend(['{"thought":"d","action":"say","message":"done"}']),
                  _CaptureSession(d), max_steps=2, autonomous=True)
        a._begin_turn("do it")
        a._mutated.add("app.py")
        a._mark_verified()                       # verified against the current workspace
        self.assertTrue(a._verified)
        _write(d, "app.py", "x = 2\n")           # workspace changes AFTER verification
        a._done_gate("done")
        self.assertFalse(a._verified)            # deterministically invalidated
        self.assertTrue(any(r["type"] == "verification_stale" for r in a.session.records))

    def test_unchanged_verification_survives_the_done_gate(self):
        # regression: a fresh, still-valid verification must NOT be spuriously invalidated.
        from forge.agent import Agent
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        a = Agent(_ScriptBackend(['{"thought":"d","action":"say","message":"done"}']),
                  _CaptureSession(d), max_steps=2, autonomous=True)
        a._begin_turn("do it")
        a._mutated.add("app.py")
        a._mark_verified()
        a._done_gate("done")                     # nothing changed since verification
        self.assertTrue(a._verified)             # survives
        self.assertFalse(any(r["type"] == "verification_stale" for r in a.session.records))

    def test_a_passing_test_bash_marks_verified_via_the_loop(self):
        # exercises the REAL _mark_verified call site through the action loop.
        import forge.agent as agentmod
        from forge.agent import Agent
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        orig = agentmod._is_test_cmd
        agentmod._is_test_cmd = lambda cmd, cwd: cmd == "true"
        try:
            a = Agent(_ScriptBackend(['{"thought":"t","action":"bash","command":"true"}',
                                      '{"thought":"d","action":"say","message":"done"}']),
                      _CaptureSession(d), max_steps=4, autonomous=True)
            a.send("check")
            self.assertTrue(a._verified)
            self.assertIsNotNone(a._verified_workspace)
        finally:
            agentmod._is_test_cmd = orig

    def test_emitted_receipt_has_checks_and_contract_id(self):
        from forge.agent import Agent
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        a = Agent(_ScriptBackend(['{"thought":"d","action":"say","message":"done"}']),
                  _CaptureSession(d), max_steps=2, autonomous=True)
        a._begin_turn("do it")
        a._mutated.add("app.py")
        a.evidence.record_change("app.py")
        a.evidence.record_verification("pytest", True, "5 passed")   # a passing check
        a._done_gate("done")
        r = [x for x in a.session.records if x["type"] == "evidence_receipt"][-1]
        self.assertTrue(r["contract_id"])
        self.assertTrue(r["checks"])
        self.assertEqual(r["checks"][0]["exit_code"], 0)
        self.assertEqual([p["path"] for p in r["changed_paths"]], ["app.py"])

    def test_done_gate_emits_a_v2_receipt_and_flags_opaque_changes(self):
        from forge.agent import Agent
        d = tempfile.mkdtemp()
        _write(d, "app.py", "x = 1\n")
        a = Agent(_ScriptBackend(['{"thought":"d","action":"say","message":"done"}']),
                  _CaptureSession(d), max_steps=2, autonomous=True)
        a._begin_turn("do it")
        a._mutated.add("<bash>")                 # an OPAQUE (unattributed) mutation
        a.evidence.record_change("<bash>")       # ...recorded in the harness evidence, as the loop does
        a._done_gate("done")
        receipts = [r for r in a.session.records if r["type"] == "evidence_receipt"]
        self.assertTrue(receipts)
        r = receipts[-1]
        self.assertEqual(r["version"], RECEIPT_VERSION)
        self.assertTrue(r["opaque_changes"])     # recorded AS opaque
        self.assertEqual(r["changed_paths"], []) # never a fabricated verified file list


class TestSerialization(unittest.TestCase):
    def test_to_dict_is_json_and_versioned(self):
        d = tempfile.mkdtemp()
        _write(d, "x.py", "1\n")
        out = _passing_receipt(d).to_dict()
        self.assertEqual(out["version"], RECEIPT_VERSION)
        self.assertEqual(out["checks"][0]["exit_code"], 0)
        import json
        json.dumps(out)


if __name__ == "__main__":
    unittest.main()
