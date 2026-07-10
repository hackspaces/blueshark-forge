"""Test suite for forge. Stdlib unittest (no deps), no model calls — covers the
harness invariants, tools, config, fleet, and workspace logic.

    python -m unittest discover -s tests      # or: python -m pytest tests
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import session as sm            # noqa: E402
from forge import ledger as ledger_mod     # noqa: E402
from forge import profile as _profile      # noqa: E402
from forge.agent import Agent              # noqa: E402

# P5.8 hermetic redirect: Agents built here record passport telemetry; keep it out of
# the real ~/.forge/profile (and out of the heat/loop assertions below) by pointing the
# store at a throwaway tempdir for this module's whole lifetime.
_profile.PROFILE_DIR = tempfile.mkdtemp(prefix="forge-profile-forge-suite-")
from forge.ledger import Ledger            # noqa: E402
from forge.backends import make_backend, OllamaBackend, OpenAICompatBackend  # noqa: E402
from forge.tools import (execute, dry_run, _fuzzy_replace, _syntax_error, _which,  # noqa: E402
                         shape, overflow_dir, _maybe_offload, MAX_OUTPUT,
                         error_hint, _closest_region, _group_grep)


def _write(p, s):
    with open(p, 'w') as f: f.write(s)


def _read(p):
    with open(p) as f: return f.read()


class ScriptBackend:
    """A fake backend that yields a scripted sequence of action JSONs."""
    name = "script"

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def stream(self, messages, schema=None, temperature=0.0):
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        yield act

    def chat(self, messages, schema=None, temperature=0.0):
        return '{"thought":"x","action":"say","message":"done"}'


class TestBackends(unittest.TestCase):
    def test_spec_parsing(self):
        self.assertIsInstance(make_backend("gemma2:9b"), OllamaBackend)
        self.assertIsInstance(make_backend("ollama:qwen"), OllamaBackend)
        b = make_backend("openai:gpt-x@https://h/v1")
        self.assertIsInstance(b, OpenAICompatBackend)
        self.assertEqual(b.url, "https://h/v1")
        self.assertEqual(b.model, "gpt-x")


class TestTools(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_write_read_roundtrip(self):
        out, ok = execute({"action": "write_file", "path": "a.txt", "content": "hello\nworld\n"}, self.d)
        self.assertTrue(ok)
        body, ok = execute({"action": "read_file", "path": "a.txt"}, self.d)
        self.assertTrue(ok)
        self.assertIn("hello", body)

    def test_read_offset_limit(self):
        execute({"action": "write_file", "path": "n.txt", "content": "".join(f"line{i}\n" for i in range(20))}, self.d)
        body, ok = execute({"action": "read_file", "path": "n.txt", "offset": 5, "limit": 3}, self.d)
        self.assertIn("line4", body)          # 1-based offset 5 -> index 4
        self.assertIn("showing lines 5-7 of 20", body)
        self.assertNotIn("line10", body)

    def test_edit_exact_and_uniqueness(self):
        execute({"action": "write_file", "path": "c.py", "content": "a = 1\nb = 1\n"}, self.d)
        # non-unique 'old' should be rejected
        _, ok = execute({"action": "edit_file", "path": "c.py", "old": "= 1", "new": "= 9"}, self.d)
        self.assertFalse(ok)
        # unique edit works
        _, ok = execute({"action": "edit_file", "path": "c.py", "old": "a = 1", "new": "a = 2"}, self.d)
        self.assertTrue(ok)
        self.assertIn("a = 2", _read(os.path.join(self.d, "c.py")))

    def test_fuzzy_replace(self):
        text = "def f():\n        return 1\n"
        new, ok, how = _fuzzy_replace(text, "def f():\n    return 1", "def f():\n    return 2")
        self.assertTrue(ok)
        self.assertIn("return 2", new)

    def test_grep_and_glob(self):
        execute({"action": "write_file", "path": "x.py", "content": "def foo(): pass\n"}, self.d)
        out, ok = execute({"action": "grep", "pattern": "def foo"}, self.d)
        self.assertTrue(ok)
        self.assertIn("foo", out)
        out, ok = execute({"action": "glob", "pattern": "*.py"}, self.d)
        self.assertTrue(ok)
        self.assertIn("x.py", out)


class TestSyntaxGate(unittest.TestCase):
    """P1.1 — check-before-write: a write/edit that would leave the file
    syntactically broken is refused in the SAME observation, file untouched."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_write_invalid_py_blocked_and_not_created(self):
        out, ok = execute({"action": "write_file", "path": "bad.py", "content": "def f(:\n    pass\n"}, self.d)
        self.assertFalse(ok)
        self.assertIn("invalid", out)
        self.assertIn("NOT written", out)
        self.assertFalse(os.path.exists(os.path.join(self.d, "bad.py")), "invalid file was created!")

    def test_write_valid_py_reports_syntax_ok(self):
        out, ok = execute({"action": "write_file", "path": "ok.py", "content": "def f():\n    return 1\n"}, self.d)
        self.assertTrue(ok)
        self.assertIn("syntax OK", out)
        self.assertEqual(_read(os.path.join(self.d, "ok.py")), "def f():\n    return 1\n")

    def test_edit_introducing_syntax_error_blocked_and_unchanged(self):
        good = "def f():\n    return 1\n"
        _write(os.path.join(self.d, "e.py"), good)
        out, ok = execute({"action": "edit_file", "path": "e.py", "old": "    return 1", "new": "    return 1)"}, self.d)
        self.assertFalse(ok)
        self.assertIn("NOT changed", out)
        self.assertEqual(_read(os.path.join(self.d, "e.py")), good, "file was mutated despite failed check!")

    def test_edit_valid_py_reports_syntax_ok(self):
        _write(os.path.join(self.d, "e.py"), "x = 1\n")
        out, ok = execute({"action": "edit_file", "path": "e.py", "old": "x = 1", "new": "x = 2"}, self.d)
        self.assertTrue(ok)
        self.assertIn("syntax OK", out)
        self.assertIn("x = 2", _read(os.path.join(self.d, "e.py")))

    def test_write_invalid_json_blocked(self):
        out, ok = execute({"action": "write_file", "path": "c.json", "content": '{"a": 1,}'}, self.d)
        self.assertFalse(ok)
        self.assertIn("invalid", out)
        self.assertFalse(os.path.exists(os.path.join(self.d, "c.json")))

    def test_write_valid_json_reports_syntax_ok(self):
        out, ok = execute({"action": "write_file", "path": "c.json", "content": '{"a": 1}'}, self.d)
        self.assertTrue(ok)
        self.assertIn("syntax OK", out)

    def test_unknown_extension_no_syntax_check(self):
        out, ok = execute({"action": "write_file", "path": "notes.txt", "content": "def f(:\n"}, self.d)
        self.assertTrue(ok)                 # no checker applies to .txt
        self.assertNotIn("syntax OK", out)

    def test_syntax_error_helper_dispatch(self):
        self.assertIsNone(_syntax_error("a.txt", "def f(:\n"))       # no checker
        self.assertEqual(_syntax_error("a.py", "x = 1\n"), "")       # ran + passed
        self.assertTrue(_syntax_error("a.py", "def f(:\n"))          # ran + failed → error string
        self.assertEqual(_syntax_error("a.json", '{"a":1}'), "")
        self.assertTrue(_syntax_error("a.json", "{,}"))

    @unittest.skipUnless(_which("bash"), "bash not available")
    def test_invalid_bash_blocked(self):
        out, ok = execute({"action": "write_file", "path": "s.sh", "content": "if then fi\n"}, self.d)
        self.assertFalse(ok)
        self.assertIn("invalid", out)
        self.assertFalse(os.path.exists(os.path.join(self.d, "s.sh")))

    @unittest.skipUnless(_which("bash"), "bash not available")
    def test_valid_bash_reports_syntax_ok(self):
        out, ok = execute({"action": "write_file", "path": "s.sh", "content": "echo hi\n"}, self.d)
        self.assertTrue(ok)
        self.assertIn("syntax OK", out)

    @unittest.skipUnless(_which("node"), "node not available")
    def test_invalid_js_blocked(self):
        out, ok = execute({"action": "write_file", "path": "a.js", "content": "function(\n"}, self.d)
        self.assertFalse(ok)
        self.assertIn("invalid", out)

    def test_toml_check_when_available(self):
        try:
            import tomllib  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("tomllib is 3.11+; py3.10 has no stdlib toml checker")
        out, ok = execute({"action": "write_file", "path": "p.toml", "content": "a = = 1\n"}, self.d)
        self.assertFalse(ok)
        self.assertIn("invalid", out)
        out, ok = execute({"action": "write_file", "path": "p.toml", "content": "a = 1\n"}, self.d)
        self.assertTrue(ok)
        self.assertIn("syntax OK", out)

    def test_edit_repairs_broken_file_one_hunk_at_a_time(self):
        # a pre-existing file with TWO syntax errors; fixing one must persist even
        # though the result is still broken (block only valid→invalid, not any-invalid)
        broken = "def f(:\n    return 1\n\ndef g(:\n    return 2\n"
        _write(os.path.join(self.d, "broken.py"), broken)
        out, ok = execute({"action": "edit_file", "path": "broken.py", "old": "def f(:", "new": "def f():"}, self.d)
        self.assertTrue(ok, "edit that REDUCES errors on an already-broken file must be allowed")
        self.assertIn("still has errors", out)
        saved = _read(os.path.join(self.d, "broken.py"))
        self.assertIn("def f():", saved)                     # progress persisted
        self.assertIn("def g(:", saved)                      # second error still there
        # now fix the second error → fully valid, reports syntax OK
        out, ok = execute({"action": "edit_file", "path": "broken.py", "old": "def g(:", "new": "def g():"}, self.d)
        self.assertTrue(ok)
        self.assertIn("syntax OK", out)

    def test_write_saves_partial_progress_on_broken_file(self):
        _write(os.path.join(self.d, "broken.py"), "def f(:\n    return 1\n\ndef g(:\n    return 2\n")
        out, ok = execute({"action": "write_file", "path": "broken.py",
                           "content": "def f():\n    return 1\n\ndef g(:\n    return 2\n"}, self.d)
        self.assertTrue(ok, "overwriting an already-broken file with a less-broken version must be allowed")
        self.assertIn("still has errors", out)
        self.assertIn("def f():", _read(os.path.join(self.d, "broken.py")))

    def test_new_invalid_file_still_blocked_after_gate_change(self):
        # regression guard: the valid→invalid gate must NOT open a hole for brand-new invalid files
        out, ok = execute({"action": "write_file", "path": "fresh.py", "content": "def f(:\n"}, self.d)
        self.assertFalse(ok)
        self.assertIn("NOT written", out)
        self.assertFalse(os.path.exists(os.path.join(self.d, "fresh.py")))

    def test_jsonc_config_with_comments_allowed(self):
        content = '{\n  // editor settings\n  "compilerOptions": {"strict": true,}\n}\n'
        out, ok = execute({"action": "write_file", "path": "tsconfig.json", "content": content}, self.d)
        self.assertTrue(ok, "tsconfig.json legitimately uses JSONC comments/trailing commas")
        self.assertEqual(_read(os.path.join(self.d, "tsconfig.json")), content)
        # a .json under .vscode/ is JSONC too
        out, ok = execute({"action": "write_file", "path": ".vscode/settings.json",
                           "content": '{\n  // ok\n  "a": 1,\n}\n'}, self.d)
        self.assertTrue(ok)

    def test_empty_json_placeholder_allowed(self):
        out, ok = execute({"action": "write_file", "path": "data.json", "content": ""}, self.d)
        self.assertTrue(ok, "creating an empty placeholder .json must be allowed")
        self.assertTrue(os.path.exists(os.path.join(self.d, "data.json")))

    def test_plain_json_still_strict(self):
        # the JSONC allowlist must not weaken the check for ordinary .json files
        out, ok = execute({"action": "write_file", "path": "data.json", "content": '{"a": 1,}'}, self.d)
        self.assertFalse(ok)
        self.assertIn("invalid", out)
        self.assertFalse(os.path.exists(os.path.join(self.d, "data.json")))


class TestReadBeforeEdit(unittest.TestCase):
    def test_blind_edit_blocked_until_read(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        actions = [
            '{"thought":"blind","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"edit","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        events = []
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(d, "script"), max_steps=6,
                  on_event=lambda k, **kw: events.append((k, kw.get("ok"))))
        a.send("change x")
        obs = [ok for k, ok in events if k == "observation"]
        self.assertEqual(obs[0], False)       # blind edit blocked
        self.assertTrue(obs[-1])              # edit after read allowed
        self.assertIn("x = 2", _read(os.path.join(d, "r.py")))


class _RecSession:
    """A minimal session that records log() calls — enough to drive Agent.send
    and inspect the transcript the done-gate writes."""
    def __init__(self, cwd, sid="rec"):
        self.cwd, self.sid, self.name = cwd, sid, "rec"
        self.status = "idle"
        self.logs = []
    def log(self, kind, **fields): self.logs.append((kind, fields))
    def drain(self): return []
    def set_status(self, s): self.status = s
    def push(self, sender, text): pass


class TestDoneGate(unittest.TestCase):
    """P2.1 — the harness runs the real test command before accepting `say`:
    a passing suite is grounded and logged 'verified', a failing one bounces the
    say exactly once (never a second time — no livelock), a no-mutation turn is
    never gated, and the gate uses the 'done_check' event, never observation-ok."""

    def setUp(self):
        from forge import fleet
        self.fleet = fleet
        self._orig = fleet.detect_test_cmd

    def tearDown(self):
        self.fleet.detect_test_cmd = self._orig

    def _run_turn(self, actions, test_cmd):
        d = tempfile.mkdtemp()
        self.fleet.detect_test_cmd = lambda cwd: test_cmd
        events = []
        sess = _RecSession(d)
        a = Agent(ScriptBackend(actions), sess, max_steps=8,
                  on_event=lambda k, **kw: events.append((k, kw)))
        result = a.send("do it")
        return result, events, sess, d

    _WRITE = '{"thought":"w","action":"write_file","path":"new.py","content":"x = 1\\n"}'
    _SAY = '{"thought":"d","action":"say","message":"all done"}'

    def test_passing_test_accepts_and_logs_verified(self):
        result, events, sess, _ = self._run_turn([self._WRITE, self._SAY], test_cmd="true")
        self.assertEqual(result, "all done")                     # say accepted
        checks = [kw for k, kw in events if k == "done_check"]
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0]["ok"])                         # harness ran it, it passed
        self.assertTrue(any(k == "verified" and f.get("ok") for k, f in sess.logs))
        # the gate must NOT masquerade as a failed observation
        self.assertFalse(any(k == "observation" and kw.get("ok") is False for k, kw in events))

    def test_failing_test_bounces_once_then_second_say_passes(self):
        result, events, sess, _ = self._run_turn([self._WRITE, self._SAY], test_cmd="false")
        self.assertEqual(result, "all done")                     # the SECOND say gets through
        checks = [kw for k, kw in events if k == "done_check"]
        self.assertEqual(len(checks), 1)                         # bounced exactly once, no livelock
        self.assertFalse(checks[0]["ok"])
        self.assertFalse(any(k == "verified" for k, f in sess.logs))  # never grounded
        # CRITICAL: the bounce is NOT an observation-ok=False event
        self.assertFalse(any(k == "observation" and kw.get("ok") is False for k, kw in events))
        # it left a plain user-visible done-check note in the transcript context
        self.assertTrue(any(k == "say" for k, kw in events))

    def test_no_mutation_turn_is_never_gated(self):
        _write(os.path.join(tempfile.mkdtemp(), "z"), "z")       # unrelated
        asked = []
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "seen.py"), "y = 2\n")
        self.fleet.detect_test_cmd = lambda cwd: asked.append(cwd) or "true"
        actions = [
            '{"thought":"r","action":"read_file","path":"seen.py"}',
            self._SAY,
        ]
        events = []
        a = Agent(ScriptBackend(actions), _RecSession(d), max_steps=6,
                  on_event=lambda k, **kw: events.append((k, kw)))
        asked.clear()   # P4.5 memoizes the test cmd ONCE at construction; the gate itself must not seek one
        result = a.send("look")
        self.assertEqual(result, "all done")
        self.assertEqual(asked, [])                              # gate never even sought a test cmd
        self.assertFalse(any(k == "done_check" for k, kw in events))

    def test_missing_command_is_not_a_failure(self):
        # detect_test_cmd guesses 'pytest -q' from a bare tests/ dir; if it isn't
        # installed the command exits 127 — treat as no usable suite, accept, no bounce.
        result, events, sess, _ = self._run_turn(
            [self._WRITE, self._SAY], test_cmd="forge-no-such-cmd-xyz")
        self.assertEqual(result, "all done")
        self.assertFalse(any(k == "done_check" for k, kw in events))
        self.assertFalse(any(k == "verified" for k, f in sess.logs))

    def test_is_test_cmd_recognizes_runners(self):
        from forge.agent import _is_test_cmd
        d = tempfile.mkdtemp()
        for c in ("pytest -q", "npm test --silent", "npm run test", "go test ./...",
                  "cargo test", "make test", "python3 -m unittest"):
            self.assertTrue(_is_test_cmd(c, d), c)
        for c in ("echo hi", "ls -la", "git status", ""):
            self.assertFalse(_is_test_cmd(c, d), c)

    def test_is_test_cmd_rejects_lookalikes(self):
        # a runner NAMED somewhere in the string but not actually running tests must
        # NOT satisfy the gate (was a soundness hole: substring match set _verified).
        from forge.agent import _is_test_cmd
        d = tempfile.mkdtemp()
        for c in ("pytest --version", "which pytest", "pip install pytest",
                  'git commit -m "make test now green"', "pytest --collect-only",
                  "cargo test --help", "echo 'run pytest later'"):
            self.assertFalse(_is_test_cmd(c, d), c)
        # a real runner reached after a shell separator still counts
        self.assertTrue(_is_test_cmd("cd sub && pytest -q", d))

    def test_bash_write_gates_say(self):
        # a file mutation made via BASH (echo > f, sed -i) must still gate `say` —
        # it used to bypass the gate entirely because only write_file/edit_file
        # were tracked. Failing suite -> bounce exactly once, second say passes.
        d = tempfile.mkdtemp()
        self.fleet.detect_test_cmd = lambda cwd: "false"
        actions = [
            '{"thought":"w","action":"bash","command":"echo broken > app.py"}',
            self._SAY,
        ]
        events = []
        a = Agent(ScriptBackend(actions), _RecSession(d), max_steps=8,
                  on_event=lambda k, **kw: events.append((k, kw)))
        result = a.send("do it")
        self.assertEqual(result, "all done")                     # second say passes
        checks = [kw for k, kw in events if k == "done_check"]
        self.assertEqual(len(checks), 1)                         # gate DID run, bounced once
        self.assertFalse(checks[0]["ok"])

    def test_readonly_bash_is_not_gated(self):
        # a turn whose only bash is read-only (ls) is not a mutation, so the gate
        # never runs the suite (no done_check) even though a test cmd is available.
        d = tempfile.mkdtemp()
        self.fleet.detect_test_cmd = lambda cwd: "false"       # would bounce IF the gate ran
        actions = ['{"thought":"l","action":"bash","command":"ls -la"}', self._SAY]
        events = []
        a = Agent(ScriptBackend(actions), _RecSession(d), max_steps=6,
                  on_event=lambda k, **kw: events.append((k, kw)))
        result = a.send("look")
        self.assertEqual(result, "all done")                    # accepted, never bounced
        self.assertFalse(any(k == "done_check" for k, kw in events))

    def test_bash_mutates_classifier(self):
        from forge import fleet
        for c in ("echo x > f", "cat x >> log", "sed -i s/a/b/ f", "perl -i -pe s/a/b/ f",
                  "rm f", "mv a b", "cp a b", "mkdir d", "chmod +x s", "git checkout f",
                  "git reset --hard", "/bin/rm -rf x", "FOO=1 tee out",
                  "python -c \"open('f','w').write('x')\"", "node -e fs.writeFileSync"):
            self.assertTrue(fleet.bash_mutates(c), c)
        for c in ("ls -la", "cat f", "grep -rn x .", "git status", "git diff", "git log",
                  "pytest -q", "make test", "python app.py 2>&1", "echo hi", "git add -A",
                  "python --version", "node script.js", ""):
            self.assertFalse(fleet.bash_mutates(c), c)

    def test_model_running_its_own_tests_satisfies_the_gate(self):
        # if the model itself runs the suite green (a bash matching a test runner
        # that exits 0), the harness does not re-run it: no done_check, say passes.
        import shutil
        if not shutil.which("make"):
            self.skipTest("make not available")
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "Makefile"), "test:\n\t@true\n")
        # gate's own cmd would FAIL if it ran — proves the model's green run short-circuits it
        self.fleet.detect_test_cmd = lambda cwd: "false"
        actions = [
            self._WRITE,
            '{"thought":"t","action":"bash","command":"make test"}',
            self._SAY,
        ]
        events = []
        a = Agent(ScriptBackend(actions), _RecSession(d), max_steps=8,
                  on_event=lambda k, **kw: events.append((k, kw)))
        result = a.send("build and test")
        self.assertEqual(result, "all done")
        self.assertFalse(any(k == "done_check" for k, kw in events))  # gate short-circuited

    def test_harness_verified_skip_predicate(self):
        # daemon.verify_pass skips a claim the harness already verified: a passing
        # 'verified' record no older than the last mutation.
        import json as _json
        from forge import session as sm2
        d = tempfile.mkdtemp()
        orig = sm2.SESSIONS
        sm2.SESSIONS = d
        try:
            sid = "vsid"
            path = os.path.join(d, sid + ".jsonl")
            with open(path, "w") as f:
                for r in [
                    {"ts": 1.0, "type": "action", "action": "write_file", "args": {"path": "a.py"}},
                    {"ts": 2.0, "type": "verified", "cmd": "pytest -q", "ok": True},
                    {"ts": 3.0, "type": "assistant", "text": "all tests pass"},
                ]:
                    f.write(_json.dumps(r) + "\n")
            self.assertTrue(self.fleet.harness_verified(sid))
            # a mutation AFTER the last verification invalidates it
            with open(path, "a") as f:
                f.write(_json.dumps({"ts": 4.0, "type": "action", "action": "edit_file",
                                     "args": {"path": "a.py"}}) + "\n")
            self.assertFalse(self.fleet.harness_verified(sid))
        finally:
            sm2.SESSIONS = orig

    def test_harness_verified_bash_edit_invalidates(self):
        # REGRESSION: a file-touching BASH edit after the verified record must make
        # the daemon re-verify — otherwise a `sed -i` breaking the tests slips past
        # both the gate (already fired) and the daemon's skip.
        import json as _json
        from forge import session as sm2
        d = tempfile.mkdtemp()
        orig = sm2.SESSIONS
        sm2.SESSIONS = d
        try:
            sid = "vsid2"
            path = os.path.join(d, sid + ".jsonl")
            base = [
                {"ts": 1.0, "type": "action", "action": "write_file", "args": {"path": "a.py"}},
                {"ts": 2.0, "type": "verified", "cmd": "pytest -q", "ok": True},
            ]
            with open(path, "w") as f:
                for r in base:
                    f.write(_json.dumps(r) + "\n")
            self.assertTrue(self.fleet.harness_verified(sid))
            # a read-only bash after the verification does NOT invalidate it
            with open(path, "a") as f:
                f.write(_json.dumps({"ts": 3.0, "type": "action", "action": "bash",
                                     "args": {"command": "cat a.py"}}) + "\n")
            self.assertTrue(self.fleet.harness_verified(sid))
            # but a mutating bash (sed -i) after it DOES
            with open(path, "a") as f:
                f.write(_json.dumps({"ts": 4.0, "type": "action", "action": "bash",
                                     "args": {"command": "sed -i s/True/False/ a.py"}}) + "\n")
            self.assertFalse(self.fleet.harness_verified(sid))
        finally:
            sm2.SESSIONS = orig


class TestNarrationGuard(unittest.TestCase):
    """The autonomous 'act, don't narrate' guard: a `say` that only DESCRIBES upcoming
    file work — after a turn that changed nothing — is bounced once so the model does the
    work instead of stopping. It must NOT fire outside autonomous mode, after a mutation,
    or for a plain answer to a question."""

    def _drive(self, actions, autonomous):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "x.py"), "y = 1\n")
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(d, "n"),
                  max_steps=8, autonomous=autonomous)
        return a.send("implement the thing in x.py"), a

    def test_preamble_say_bounced_in_autonomous_mode(self):
        actions = [
            '{"thought":"look","action":"read_file","path":"x.py"}',
            '{"thought":"narrate","action":"say","message":"I will implement the fix now. Let me start."}',
            '{"thought":"done","action":"say","message":"All done — the change is in place and verified."}',
        ]
        reply, a = self._drive(actions, autonomous=True)
        self.assertIn("in place", reply)                       # the SECOND say returned, not the preamble
        self.assertTrue(any("narrate_bounce" == k for k, _ in getattr(a.session, "logs", [])) or
                        any("haven't actually done it" in m.get("content", "") for m in a.messages))

    def test_preamble_say_accepted_when_not_autonomous(self):
        actions = [
            '{"thought":"look","action":"read_file","path":"x.py"}',
            '{"thought":"narrate","action":"say","message":"I will implement the fix now. Let me start."}',
        ]
        reply, _ = self._drive(actions, autonomous=False)
        self.assertIn("implement the fix now", reply)          # not autonomous → preamble ends the turn

    def test_plain_answer_not_bounced(self):
        # a real answer to a question (no intent-to-act phrase) must pass straight through
        actions = [
            '{"thought":"look","action":"read_file","path":"x.py"}',
            '{"thought":"answer","action":"say","message":"This module defines y = 1 and nothing else."}',
        ]
        reply, _ = self._drive(actions, autonomous=True)
        self.assertIn("defines y = 1", reply)

    def test_not_bounced_after_a_real_change(self):
        # once the turn has mutated a file, a say is the done-gate's business, not this guard
        actions = [
            '{"thought":"look","action":"read_file","path":"x.py"}',
            '{"thought":"edit","action":"edit_file","path":"x.py","old":"y = 1","new":"y = 2"}',
            '{"thought":"narrate","action":"say","message":"I will now update the docs too."}',
        ]
        reply, _ = self._drive(actions, autonomous=True)
        self.assertIn("update the docs", reply)                # mutation happened → guard silent


class TestConfig(unittest.TestCase):
    def test_load_defaults(self):
        from forge import config
        cfg = config.load()
        self.assertIn("ladder", cfg)
        self.assertIsInstance(cfg["ladder"], list)


class TestFleet(unittest.TestCase):
    def test_learnings_store_and_dedupe(self):
        from forge import fleet
        d = tempfile.mkdtemp()
        # monkeypatch the store path to a temp file
        orig = fleet._learn_path
        fleet._learn_path = lambda cwd: os.path.join(d, "l.jsonl")
        try:
            fresh = fleet._store_learnings("x", ["Tests run with npm test", "Migrate before seed"], "s1")
            self.assertEqual(len(fresh), 2)
            again = fleet._store_learnings("x", ["Tests run with npm test"], "s2")  # dupe
            self.assertEqual(len(again), 0)
            self.assertEqual(len(fleet.learnings("x")), 2)
        finally:
            fleet._learn_path = orig


class TestLearnV2(unittest.TestCase):
    """P4.6 — validated LEARN v2: classify → run/existence-check → supersede → forget.
    Executable facts run a REAL subprocess (sh) against a temp fixture — allowed, as
    it is not a model call — while the store is redirected off ~/.forge."""

    def setUp(self):
        from forge import fleet
        self.fleet = fleet
        store = tempfile.mkdtemp()
        self._orig_lp = fleet._learn_path
        fleet._learn_path = lambda cwd, _s=store: os.path.join(_s, "l.jsonl")

    def tearDown(self):
        self.fleet._learn_path = self._orig_lp

    def test_passing_command_verified(self):
        d = tempfile.mkdtemp()            # no markers → detect returns None → really runs
        self.fleet._store_learnings(d, ['Always run `sh -c "exit 0"` first'], "s1")
        recs = self.fleet.learnings(d)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["kind"], "executable")
        self.assertTrue(recs[0]["verified"])
        self.assertEqual(recs[0]["method"], "run")

    def test_failing_command_stored_unverified_not_dropped(self):
        d = tempfile.mkdtemp()
        fact = 'Run `sh -c "exit 3"` to check'
        self.fleet._store_learnings(d, [fact], "s1")
        recs = self.fleet.learnings(d)
        self.assertEqual(len(recs), 1)                       # NOT discarded
        self.assertFalse(recs[0]["verified"])                # stored UNVERIFIED
        self.assertEqual(recs[0]["fact"], fact)

    def test_path_fact_existence_checked(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "present.py"), "x=1")
        self.fleet._store_learnings(d, ["The entry point lives in present.py",
                                        "Config is in absent.toml"], "s1")
        recs = {r["fact"]: r for r in self.fleet.learnings(d)}
        self.assertEqual(recs["The entry point lives in present.py"]["kind"], "path")
        self.assertTrue(recs["The entry point lives in present.py"]["verified"])
        self.assertFalse(recs["Config is in absent.toml"]["verified"])

    def test_detect_crosscheck_short_circuits_without_running(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "pyproject.toml"), "[project]\nname='x'\n")
        # detect_test_cmd(d) → 'pytest -q'; the fact AGREES so it must never isolate/run.
        orig = self.fleet._isolate
        self.fleet._isolate = lambda cwd: (_ for _ in ()).throw(AssertionError("must not run"))
        try:
            self.fleet._store_learnings(d, ["The test command is `pytest -q`"], "s1")
        finally:
            self.fleet._isolate = orig
        recs = self.fleet.learnings(d)
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0]["verified"])
        self.assertEqual(recs[0]["method"], "detect")

    def test_contradiction_supersedes_by_key(self):
        d = tempfile.mkdtemp()
        self.fleet._store_learnings(d, ['The check command is `sh -c "exit 0"`'], "s1")
        self.fleet._store_learnings(d, ['The check command is `sh -c "exit 1"`'], "s2")
        recs = self.fleet.learnings(d)
        self.assertEqual(len(recs), 1)                       # superseded, not appended
        self.assertIn("exit 1", recs[0]["fact"])             # newest wins
        self.assertFalse(recs[0]["verified"])

    def test_learnings_verified_first(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "here.py"), "x=1")
        self.fleet._store_learnings(d, ["Prefer small commits over big ones",  # note → None
                                        "The loader is in here.py"], "s1")      # path → True
        recs = self.fleet.learnings(d)
        self.assertEqual(len(recs), 2)
        self.assertTrue(recs[0]["verified"])                 # verified fact sorts first
        self.assertIn("here.py", recs[0]["fact"])

    def test_forget_removes_matching_then_all(self):
        d = tempfile.mkdtemp()
        self.fleet._store_learnings(d, ["Prefer tabs over spaces",
                                        "Deploy with the red button"], "s1")
        self.assertEqual(len(self.fleet.learnings(d)), 2)
        self.assertEqual(self.fleet.forget(d, "tabs"), 1)
        self.assertEqual([r["fact"] for r in self.fleet.learnings(d)],
                         ["Deploy with the red button"])
        self.assertEqual(self.fleet.forget(d), 1)            # no pattern clears the rest
        self.assertEqual(self.fleet.learnings(d), [])

    def test_store_returns_fresh_strings_for_daemon(self):
        # daemon.learn_pass relies on _store_learnings returning fact STRINGS.
        d = tempfile.mkdtemp()
        fresh = self.fleet._store_learnings(d, ["Prefer small commits"], "s1")
        self.assertEqual(fresh, ["Prefer small commits"])
        self.assertTrue(all(isinstance(x, str) for x in fresh))


class TestWorkspace(unittest.TestCase):
    def test_detect_project(self):
        from forge import workspace
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "package.json"), "{}")
        label, markers = workspace.detect_project(d)
        self.assertIn("Node", label)

    def test_no_marker_means_no_claim(self):
        # stray source files must NOT get a directory labeled as a project
        # (a home dir with one .go file is not 'a Go project')
        from forge import workspace
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "a.py"), "x=1")
        _write(os.path.join(d, "b.go"), "package main")
        label, markers = workspace.detect_project(d)
        self.assertEqual(label, "")
        self.assertEqual(markers, [])
        self.assertNotIn("Project type", workspace.context(d))

    def test_instructions_file_pinned_above_learnings(self):
        # P4.6: the first-found FORGE.md/AGENTS.md/CLAUDE.md is pinned as
        # user-authored PROJECT INSTRUCTIONS, above the fleet's learnings.
        from forge import workspace
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "CLAUDE.md"), "claude fallback rules")
        _write(os.path.join(d, "AGENTS.md"), "agents rules body")
        ctx = workspace.context(d, learnings=[{"fact": "L1", "verified": True},
                                              {"fact": "L2", "verified": False}])
        self.assertIn("PROJECT INSTRUCTIONS", ctx)
        self.assertIn("agents rules body", ctx)              # AGENTS.md beats CLAUDE.md
        self.assertNotIn("claude fallback rules", ctx)
        self.assertLess(ctx.index("PROJECT INSTRUCTIONS"), ctx.index("already learned"))
        self.assertIn("✓ L1", ctx)                           # verified learning annotated
        self.assertIn("- L2", ctx)                           # unverified learning plain
        _write(os.path.join(d, "FORGE.md"), "forge top-priority rules")
        ctx2 = workspace.context(d)
        self.assertIn("forge top-priority rules", ctx2)      # FORGE.md wins over all
        self.assertNotIn("agents rules body", ctx2)

    def test_instructions_file_capped(self):
        from forge import workspace
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "FORGE.md"), "x" * 5000)
        name, text = workspace._instructions(d)
        self.assertEqual(name, "FORGE.md")
        self.assertLessEqual(len(text), workspace.INSTRUCTIONS_CAP + 40)
        self.assertIn("truncated", text)


class TestEdgeCases(unittest.TestCase):
    """Adversarial inputs — the harness must degrade gracefully, never crash or escape."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    # --- path safety ---
    def test_absolute_path_write_blocked(self):
        target = os.path.join(tempfile.gettempdir(), "forge_escape_unit.txt")
        if os.path.exists(target):
            os.remove(target)
        _, ok = execute({"action": "write_file", "path": target, "content": "x"}, self.d)
        self.assertFalse(ok)
        self.assertFalse(os.path.exists(target), "absolute path escaped the workspace!")

    def test_traversal_write_blocked(self):
        _, ok = execute({"action": "write_file", "path": "../escape.txt", "content": "x"}, self.d)
        self.assertFalse(ok)
        self.assertFalse(os.path.exists(os.path.join(self.d, "..", "escape.txt")))

    def test_nested_path_within_workspace_ok(self):
        _, ok = execute({"action": "write_file", "path": "a/b/c.txt", "content": "hi"}, self.d)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(os.path.join(self.d, "a", "b", "c.txt")))

    # --- read robustness ---
    def test_read_missing_file(self):
        _, ok = execute({"action": "read_file", "path": "ghost.txt"}, self.d)
        self.assertFalse(ok)

    def test_read_directory(self):
        os.makedirs(os.path.join(self.d, "sub"))
        out, ok = execute({"action": "read_file", "path": "sub"}, self.d)
        self.assertFalse(ok)
        self.assertIn("directory", out)

    def test_read_offset_past_end(self):
        _write(os.path.join(self.d, "s.txt"), "a\nb\n")
        out, ok = execute({"action": "read_file", "path": "s.txt", "offset": 99}, self.d)
        self.assertFalse(ok)
        self.assertIn("past the end", out)

    # --- edit robustness ---
    def test_edit_missing_file(self):
        _, ok = execute({"action": "edit_file", "path": "ghost.py", "old": "a", "new": "b"}, self.d)
        self.assertFalse(ok)

    def test_edit_pattern_not_found(self):
        _write(os.path.join(self.d, "f.py"), "x = 1\n")
        _, ok = execute({"action": "edit_file", "path": "f.py", "old": "not-there", "new": "y"}, self.d)
        self.assertFalse(ok)

    def test_edit_empty_old(self):
        _write(os.path.join(self.d, "f.py"), "x = 1\n")
        _, ok = execute({"action": "edit_file", "path": "f.py", "old": "", "new": "y"}, self.d)
        self.assertFalse(ok)

    # --- search robustness ---
    def test_grep_no_matches_is_ok(self):
        _write(os.path.join(self.d, "a.txt"), "hello\n")
        out, ok = execute({"action": "grep", "pattern": "zzzznotfound"}, self.d)
        self.assertTrue(ok)   # no match is a valid result, not a failure

    def test_glob_no_matches(self):
        out, ok = execute({"action": "glob", "pattern": "*.nonexistent"}, self.d)
        self.assertTrue(ok)

    # --- P1.5: grep v2 (rc inspection, literal fallback, per-file grouping) ---
    def test_grep_invalid_regex_searches_literally(self):
        # '[' is an unbalanced bracket — a parse error in BOTH rg and grep. It must
        # NOT be reported as raw parse-error text (a small model reads that as 'no
        # results'); it retries as a literal string and finds the real occurrence.
        _write(os.path.join(self.d, "c.py"), "arr = data[0]\n")
        out, ok = execute({"action": "grep", "pattern": "["}, self.d)
        self.assertTrue(ok)                              # a bad regex is not a hard failure
        low = out.lower()
        self.assertTrue("literally" in low or "invalid regex" in low,
                        f"invalid regex must be explained, got: {out!r}")
        self.assertIn("data[0]", out)                    # the literal hit is present, not swallowed
        self.assertNotIn("parse error", low)             # NOT the raw tool error masquerading as results

    def test_grep_no_match_message_is_deterministic(self):
        _write(os.path.join(self.d, "a.txt"), "hello\n")
        out, ok = execute({"action": "grep", "pattern": "zzzznotfound_xyz"}, self.d)
        self.assertTrue(ok)
        self.assertTrue(out.startswith("no matches for zzzznotfound_xyz"), out)

    def test_grep_groups_by_file_with_counts(self):
        _write(os.path.join(self.d, "m.py"), "needle 1\nx\nneedle 2\ny\nneedle 3\n")
        out, ok = execute({"action": "grep", "pattern": "needle"}, self.d)
        self.assertTrue(ok)
        self.assertIn("3 matches, showing", out)         # per-file count header present
        self.assertIn("m.py", out)                       # filename preserved
        self.assertIn("needle 1", out)                   # matched line text preserved

    def test_grep_context_field_controls_neighbors(self):
        _write(os.path.join(self.d, "ctx.py"), "line one\nTARGET here\nline three\n")
        tight, ok = execute({"action": "grep", "pattern": "TARGET", "context": 0}, self.d)
        self.assertTrue(ok)
        self.assertNotIn("line one", tight)              # context 0 → no surrounding lines
        wide, ok = execute({"action": "grep", "pattern": "TARGET", "context": 1}, self.d)
        self.assertTrue(ok)
        self.assertIn("line one", wide)                  # context 1 → neighbor shown
        self.assertIn("line three", wide)

    def test_grep_context_is_capped(self):
        # a huge/garbage context must be clamped, never crash the tool
        _write(os.path.join(self.d, "z.py"), "hit\n")
        out, ok = execute({"action": "grep", "pattern": "hit", "context": 999}, self.d)
        self.assertTrue(ok)
        self.assertIn("hit", out)

    def test_grep_single_file_path_preserves_count_and_name(self):
        # a grep scoped to ONE explicit file must still report the real count and
        # filename. rg omits the `file:` prefix for a single explicit file unless
        # forced with -H; without it the grouper reports a bogus '0 matches' and a
        # blank filename — the exact 'harness lies to the model' bug this item fixes.
        _write(os.path.join(self.d, "app.py"), "def foo():\n    return needle\n\nneedle_again = 1\n")
        out, ok = execute({"action": "grep", "pattern": "needle", "path": "app.py"}, self.d)
        self.assertTrue(ok)
        self.assertIn("2 matches", out)          # both hits counted, not swallowed to 0
        self.assertNotIn("0 matches", out)       # never under-report a real match
        self.assertIn("app.py", out)             # filename preserved, not blank

    @unittest.skipUnless(_which("rg"), "malformed-glob error is only surfaced by rg")
    def test_glob_real_error_not_faked_ok(self):
        # a malformed glob is a real error, not 'no files' — must report ok False
        _, ok = execute({"action": "glob", "pattern": "["}, self.d)
        self.assertFalse(ok)

    def test_group_grep_per_file_cap_and_count(self):
        raw = "\n".join(f"f1.py:{i}:match {i}" for i in range(1, 6))   # 5 matches, one file
        body, n = _group_grep(raw)
        self.assertEqual(n, 5)                           # full count reported
        self.assertIn("f1.py — 5 matches, showing 3:", body)   # capped to first 3 blocks

    def test_group_grep_overall_block_cap(self):
        raw = "\n".join(f"f{i}.py:1:x" for i in range(50))   # 50 files, 1 match each
        body, n = _group_grep(raw)
        self.assertEqual(n, 50)
        self.assertIn("more file(s)", body)              # >40 blocks → overflow is flagged, not dropped silently

    def test_bash_empty_command(self):
        _, ok = execute({"action": "bash", "command": "  "}, self.d)
        self.assertFalse(ok)

    def test_unknown_action(self):
        out, ok = execute({"action": "frobnicate"}, self.d)
        self.assertFalse(ok)

    def test_large_bash_output_offloaded(self):
        out, ok = execute({"action": "bash", "command": "for i in $(seq 1 5000); do echo line$i; done"}, self.d)
        self.assertTrue(ok)
        self.assertIn("saved to", out)   # big output goes to a file, not dumped

    # --- agent robustness ---
    def test_malformed_json_gives_up_gracefully(self):
        class Garbage:
            name = "g"
            def stream(self, m, schema=None, temperature=0.0):
                yield "not json at all"
            def chat(self, *a, **k):
                return "not json"
        a = Agent(Garbage(), sm.EphemeralSession(self.d, "g"), max_steps=10)
        r = a.send("do something")
        self.assertIn("could not hold", r)  # bails, doesn't hang

    def test_fleet_send_unknown_target(self):
        actions = [
            '{"thought":"msg","action":"fleet_send","target":"nobody-here","message":"hi"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        events = []
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(self.d, "s"), max_steps=4,
                  on_event=lambda k, **kw: events.append((k, kw.get("ok"))))
        r = a.send("message nobody")
        # the failed send is reported but the agent continues to completion
        self.assertEqual(r, "done")


class TestFailureAutopsy(unittest.TestCase):
    """P1.3 — a failed edit_file/bash diagnoses itself deterministically: multi-match
    edits list every location, a near-miss edit shows the CLOSEST region verbatim with
    real line numbers, and a failed bash appends an error-class recovery hint — so the
    model fixes in one step instead of burning a re-read/diagnose loop."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    # --- edit_file: exact multi-match lists locations, not just a count ---
    def test_exact_multi_match_lists_locations(self):
        _write(os.path.join(self.d, "m.py"), "a = 1\nb = 1\nc = 1\n")
        out, ok = execute({"action": "edit_file", "path": "m.py", "old": "= 1", "new": "= 9"}, self.d)
        self.assertFalse(ok)                       # still a failure → feeds the escalation signal
        self.assertIn("appears 3 times", out)
        for ln in ("line 1", "line 2", "line 3"):  # every location enumerated
            self.assertIn(ln, out)

    # --- edit_file: >1 whitespace-tolerant matches get the same treatment ---
    def test_fuzzy_multi_match_lists_locations(self):
        _write(os.path.join(self.d, "f.py"),
               "def a():\n    x = 1\n    return x\n\ndef b():\n        x = 1\n        return x\n")
        # no exact substring match (indentation differs), but two fuzzy matches
        out, ok = execute({"action": "edit_file", "path": "f.py",
                           "old": "x = 1\nreturn x", "new": "x = 2\nreturn x"}, self.d)
        self.assertFalse(ok)
        self.assertIn("2 places", out)
        self.assertIn("ignoring indentation", out)
        self.assertIn("line 2", out)               # def a()'s body
        self.assertIn("line 6", out)               # def b()'s body

    # --- edit_file: near-miss shows the CLOSEST region verbatim with real line numbers ---
    def test_close_but_not_exact_shows_closest_region(self):
        _write(os.path.join(self.d, "g.py"),
               "import os\n\n\ndef greet(name):\n    message = \"hello \" + name\n    return message\n")
        out, ok = execute({"action": "edit_file", "path": "g.py",
                           "old": "def greet(name):\n    message = \"hi \" + name\n    return message",
                           "new": "x"}, self.d)
        self.assertFalse(ok)
        self.assertIn("CLOSEST region", out)
        self.assertIn("lines 4-6", out)            # the real nearby line numbers
        self.assertIn("hello", out)                # the region is verbatim
        self.assertIn("copied EXACTLY", out)

    def test_no_close_region_falls_back_to_generic(self):
        _write(os.path.join(self.d, "h.py"), "alpha = 1\nbeta = 2\n")
        out, ok = execute({"action": "edit_file", "path": "h.py",
                           "old": "wildly unrelated tokens zzz qqq wibble", "new": "x"}, self.d)
        self.assertFalse(ok)
        self.assertNotIn("CLOSEST region", out)
        self.assertIn("copy the EXACT text", out)

    def test_closest_region_direct(self):
        text = "one\ntwo\nthree hundred\nfour\n"
        region = _closest_region(text, "three hundred and five")
        self.assertIsNotNone(region)
        i, j, block = region
        self.assertEqual((i, j), (3, 3))
        self.assertIn("three hundred", block)
        # nothing remotely similar → None (generic-message fallback)
        self.assertIsNone(_closest_region(text, "qqqqqq zzzzzz wibble wobble"))

    # --- failed bash: error-class recovery hints ---
    def test_error_hint_classifier(self):
        self.assertIn("venv", error_hint("ModuleNotFoundError: No module named 'foo'"))
        self.assertIn("PATH", error_hint("bash: frobnicate: command not found"))
        self.assertIn("port", error_hint("OSError: [Errno 48] Address already in use"))
        self.assertIn("path", error_hint("cat: nope.txt: No such file or directory"))
        self.assertIn("permission", error_hint("open: Permission denied").lower())
        self.assertEqual(error_hint("everything is fine, exit 0"), "")   # no false hint

    def test_failed_bash_appends_module_hint_via_agent(self):
        actions = [
            '{"thought":"probe","action":"bash","command":"python3 -c \'import nosuchmod_xyz123\'"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(self.d, "s"), max_steps=4)
        a.send("probe")
        # the hint rides the observation the model actually consumes
        blob = "\n".join(m["content"] for m in a.messages if m["role"] == "user")
        self.assertIn("ModuleNotFoundError", blob)
        self.assertIn(".venv/bin", blob)


class TestObservationShaping(unittest.TestCase):
    """P1.2 — one observation-shaping pipeline. shape() truncates head+TAIL with an
    explicit marker (never silent, never head-only), harness notes/pointers ride the
    tail so they can NEVER be sliced off, and offloads land under <cwd>/.forge/output
    where the model's own read_file/grep can actually reach them."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_shape_passes_short_text_through(self):
        self.assertEqual(shape("hello", 100), "hello")
        # a note on short text is still appended (a read-range note must always show)
        self.assertEqual(shape("hello", 100, note="\n[note]"), "hello\n[note]")

    def test_shape_keeps_head_and_tail_with_marker(self):
        text = "HEAD" + ("x" * 6000) + "TAIL"
        out = shape(text, 2000)
        self.assertLess(len(out), len(text))
        self.assertIn("omitted from the middle", out)     # explicit marker — never silent
        self.assertTrue(out.startswith("HEAD"))           # head retained
        self.assertTrue(out.endswith("TAIL"))             # TAIL retained — errors/summaries live here

    def test_shape_note_always_survives_truncation(self):
        text = "z" * 20000
        note = "\n[... output truncated. Full output saved to /w/.forge/output/x.txt]"
        out = shape(text, 2000, note=note)
        self.assertTrue(out.endswith(note))               # pointer is re-attached AFTER the cut
        self.assertIn("omitted from the middle", out)
        self.assertLess(len(out), len(text))

    def test_midsize_observation_not_cut_without_a_marker(self):
        # the core silent-truncation bug: a ~6000-char observation must never be cut
        # mid-content with no marker. Either it fits, or the cut is explicit.
        body = "".join(f"line{i:04d}-payload " for i in range(700))  # ~14k chars
        out = shape(body, 4000)
        self.assertIn("omitted from the middle", out)
        self.assertTrue(out.startswith("line0000"))       # nothing before the head is lost silently

    def test_offload_writes_under_cwd_and_read_file_can_reach_it(self):
        big = "".join(f"row {i}\n" for i in range(3000))   # well over MAX_OUTPUT
        self.assertGreater(len(big), MAX_OUTPUT)
        preview, note = _maybe_offload(big, "bash", self.d)
        self.assertIn("saved to", note)                    # literal phrase kept
        outdir = os.path.join(self.d, ".forge", "output")
        files = os.listdir(outdir)
        self.assertEqual(len(files), 1)                    # full text saved to exactly one file
        # the model can follow the pointer with its OWN read_file (confined to cwd)
        rel = os.path.join(".forge", "output", files[0])
        body, ok = execute({"action": "read_file", "path": rel, "limit": 5}, self.d)
        self.assertTrue(ok)
        self.assertIn("row 0", body)                       # got the real content back

    def test_offload_pointer_survives_the_agent_budget_shape(self):
        # the exact dead-code bug: the pointer used to sit past char ~12000, then
        # obs[:4000] sliced it off. Now it rides the tail, so a smaller agent budget
        # applied on top still keeps it.
        big = "".join(f"row {i}\n" for i in range(3000))
        preview, note = _maybe_offload(big, "bash", self.d)
        obs = preview + note
        shaped = shape(obs, 4000)                          # agent re-shapes to its budget
        self.assertIn("saved to", shaped)                  # pointer NOT sliced off
        self.assertIn("omitted from the middle", shaped)

    def test_read_range_note_survives_offload_and_shape(self):
        _write(os.path.join(self.d, "big.txt"), "".join("x" * 48 + f"{i}\n" for i in range(3000)))
        out, ok = execute({"action": "read_file", "path": "big.txt", "offset": 1, "limit": 2000}, self.d)
        self.assertTrue(ok)
        self.assertIn("saved to", out)                     # a big read offloads too
        self.assertIn("showing lines 1-2000 of 3000", out)
        shaped = shape(out, 4000)                          # agent budget on top
        self.assertIn("showing lines 1-2000 of 3000", shaped)   # range note NOT sliced off

    def test_forge_dir_is_gitignored(self):
        d = overflow_dir(self.d)
        self.assertTrue(os.path.isdir(d))
        gi = os.path.join(self.d, ".forge", ".gitignore")
        self.assertTrue(os.path.exists(gi))
        self.assertEqual(_read(gi).strip(), "*")

    def test_obs_budget_derives_from_window_and_caps_at_12000(self):
        class Small:
            name = "s"
            def effective_ctx(self): return 8192
            def chat(self, *a, **k): return '{"thought":"x","action":"say","message":"done"}'
        class Huge:
            name = "h"
            def effective_ctx(self): return 1_000_000
            def chat(self, *a, **k): return '{"thought":"x","action":"say","message":"done"}'
        small = Agent(Small(), sm.EphemeralSession(self.d, "s"))
        huge = Agent(Huge(), sm.EphemeralSession(self.d, "h"))
        self.assertEqual(small._obs_budget(), int(8192 * 4 * 0.08))   # ~8% of the real window
        self.assertEqual(huge._obs_budget(), 12000)                  # hard-capped, no split-brain


class TestInboxAuth(unittest.TestCase):
    def test_inbox_requires_token(self):
        import urllib.request
        import urllib.error
        s = sm.Session("authtest" + os.urandom(3).hex(), tempfile.mkdtemp(), "m", name="a")
        s.start_inbox()
        port = s.port
        # no token → 403
        req = urllib.request.Request(f"http://127.0.0.1:{port}/", data=b"hi", headers={"X-Forge-From": "x"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=3)
        self.assertEqual(cm.exception.code, 403)
        # correct token → delivered
        req = urllib.request.Request(f"http://127.0.0.1:{port}/", data=b"hi", headers={"X-Forge-Token": s.token})
        urllib.request.urlopen(req, timeout=3).read()
        self.assertEqual(s.drain()[0]["text"], "hi")


class TestEngineRouting(unittest.TestCase):
    def test_routing(self):
        self.assertIsInstance(make_backend("m", engine="ollama"), OllamaBackend)
        b = make_backend("m", engine="vllm")
        self.assertIsInstance(b, OpenAICompatBackend)
        self.assertIn(":8000", b.url)
        b2 = make_backend("m", engine="openai", base_url="http://cluster:9/v1")
        self.assertEqual(b2.url, "http://cluster:9/v1")
        # an explicit prefix always wins over the configured engine
        self.assertIsInstance(make_backend("ollama:m", engine="openai"), OllamaBackend)


class TestStreamingUnicode(unittest.TestCase):
    def test_partial_message_handles_unicode_escapes(self):
        from forge.agent import _partial_message
        raw = '{"action":"say","message":"caf\\u00e9 \\u2713 done"}'
        self.assertEqual(_partial_message(raw), "café ✓ done")

    def test_partial_message_incomplete_stream(self):
        from forge.agent import _partial_message
        self.assertEqual(_partial_message('{"message":"hel'), "hel")  # mid-stream, no closing quote


class TestCollisionGuard(unittest.TestCase):
    def test_edited_files_includes_edit_file(self):
        from forge import fleet
        sid = "collide" + os.urandom(3).hex()
        _write(os.path.join(sm.SESSIONS, f"{sid}.jsonl"),
               '{"type":"action","action":"edit_file","args":{"path":"a.py"}}\n')
        try:
            files = fleet.edited_files(sid, "/repo")
            self.assertIn(os.path.normpath("/repo/a.py"), files)
        finally:
            os.remove(os.path.join(sm.SESSIONS, f"{sid}.jsonl"))


class TestGrepConfinement(unittest.TestCase):
    def test_grep_path_escape_blocked(self):
        d = tempfile.mkdtemp()
        _, ok = execute({"action": "grep", "pattern": "x", "path": "/etc"}, d)
        self.assertFalse(ok)


class TestConfigEdge(unittest.TestCase):
    def test_corrupt_config_falls_back_to_defaults(self):
        from forge import config
        import unittest.mock as mock
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{ this is not valid json ")
            bad = f.name
        with mock.patch.object(config, "PATH", bad):
            cfg = config.load()
            self.assertIn("ladder", cfg)  # defaults, not a crash
        os.remove(bad)


class TestClaudeBridge(unittest.TestCase):
    """forge ↔ Claude Code fleet interop."""

    def setUp(self):
        import unittest.mock as mock
        from forge import bridge
        self.tmp = tempfile.mkdtemp()
        self.inbox = os.path.join(self.tmp, "inbox.json")
        self.tokf = os.path.join(self.tmp, "token")
        _write(self.tokf, "shared-secret")
        self.patches = [
            mock.patch.object(bridge, "DIR", self.tmp),
            mock.patch.object(bridge, "INBOX", self.inbox),
            mock.patch.object(bridge, "TOKEN_FILE", self.tokf),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        import shutil
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_claude_peers_skips_forge_and_dead(self):
        import json as _json
        from forge import bridge
        _write(self.inbox, _json.dumps([
            {"sessionId": "aaa", "name": "web", "cwd": "/x", "port": 1, "pid": os.getpid()},
            {"sessionId": "bbb", "name": "me", "cwd": "/y", "port": 2, "pid": os.getpid(), "kind": "forge"},
            {"sessionId": "ccc", "name": "dead", "cwd": "/z", "port": 3, "pid": 99999999},
        ]))
        peers = bridge.claude_peers()
        self.assertEqual([p["sid"] for p in peers], ["aaa"])
        self.assertEqual(peers[0]["kind"], "claude")

    def test_register_and_unregister(self):
        from forge import bridge

        class S:
            sid, name, cwd, port = "f1", "proj", "/p", 4242
        bridge.register(S())
        entries = bridge._read_inbox()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "forge")
        self.assertEqual(entries[0]["sessionId"], "f1")
        bridge.unregister()
        self.assertEqual(bridge._read_inbox(), [])

    def test_find_session_covers_claude_peers(self):
        import json as _json
        from forge import bridge, fleet
        _write(self.inbox, _json.dumps(
            [{"sessionId": "abc123", "name": "webapp", "cwd": "/w", "port": 5, "pid": os.getpid()}]))
        hits = fleet.find_session("webapp")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["kind"], "claude")

    def test_inbox_accepts_claude_fleet_protocol(self):
        import urllib.request
        from forge import session as sm2
        s = sm2.Session("brdg", self.tmp, "m")
        s.start_inbox()
        req = urllib.request.Request(
            f"http://127.0.0.1:{s.port}/send", data=b"hi from claude",
            headers={"X-Fleet-Token": "shared-secret", "X-Fleet-From": "claude-sess"})
        urllib.request.urlopen(req, timeout=5).read()
        msgs = s.drain()
        self.assertEqual(msgs, [{"from": "claude-sess", "text": "hi from claude"}])
        # wrong token still rejected
        bad = urllib.request.Request(
            f"http://127.0.0.1:{s.port}/send", data=b"x", headers={"X-Fleet-Token": "nope"})
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(bad, timeout=5)

    def test_doctor_without_claude(self):
        import unittest.mock as mock
        from forge import bridge
        with mock.patch("shutil.which", return_value=None), \
             mock.patch("os.path.isdir", return_value=False):
            lines = bridge.doctor(create_token=False)
        self.assertEqual(len(lines), 1)
        self.assertIn("not found", lines[0])


class TestInterrupt(unittest.TestCase):
    def test_bash_killed_when_stop_fires(self):
        import threading
        import time as _t
        stop = threading.Event()
        threading.Timer(0.4, stop.set).start()
        t0 = _t.monotonic()
        obs, ok = execute({"action": "bash", "command": "sleep 30"}, "/tmp", stop=stop)
        self.assertLess(_t.monotonic() - t0, 5)
        self.assertFalse(ok)
        self.assertIn("stopped", obs)

    def test_bash_timeout_still_works(self):
        import unittest.mock as mock
        from forge import tools
        with mock.patch.object(tools, "BASH_TIMEOUT", 1):
            obs, ok = execute({"action": "bash", "command": "sleep 30"}, "/tmp")
        self.assertFalse(ok)
        self.assertIn("timed out", obs)

    def test_run_interruptible_returns_worker_result(self):
        import threading
        from forge.tui import run_interruptible
        stop = threading.Event()
        self.assertEqual(run_interruptible(lambda: "done", stop), "done")  # non-tty path


class TestRoster(unittest.TestCase):
    def test_roster_lists_both_runtimes(self):
        import json as _json
        import unittest.mock as mock
        from forge import bridge, fleet
        tmp = tempfile.mkdtemp()
        inbox = os.path.join(tmp, "inbox.json")
        _write(inbox, _json.dumps(
            [{"sessionId": "cc1", "name": "webapp", "cwd": "/w", "port": 5, "pid": os.getpid()}]))
        with mock.patch.object(bridge, "INBOX", inbox):
            r = fleet.roster()
        self.assertIn("webapp", r)
        self.assertIn("claude", r)


class TestFindSession(unittest.TestCase):
    """Target matching quirks found in live use."""

    def _with_peers(self, peers):
        import contextlib
        import json as _json
        import unittest.mock as mock
        tmp = tempfile.mkdtemp()
        inbox = os.path.join(tmp, "inbox.json")
        _write(inbox, _json.dumps(peers))
        from forge import bridge
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(bridge, "INBOX", inbox))
        stack.enter_context(mock.patch("forge.session.registry", return_value=[]))  # isolate from real machine
        return stack

    def test_roster_display_format_works_as_target(self):
        from forge import fleet
        peers = [{"sessionId": "8fb90aba-1", "name": "ymp", "cwd": "/Users/ymp", "port": 1, "pid": os.getpid()},
                 {"sessionId": "a617c126-2", "name": "ai-grader", "cwd": "/g", "port": 2, "pid": os.getpid()}]
        with self._with_peers(peers):
            for t in ("ymp(8fb90aba)", "ymp(8fb90aba, claude)", "ymp (8fb90aba)"):
                hits = fleet.find_session(t)
                self.assertEqual([h["sid"] for h in hits], ["8fb90aba-1"], t)

    def test_sender_excluded_from_ambiguous_match(self):
        from forge import fleet
        peers = [{"sessionId": "claude-ymp", "name": "ymp", "cwd": "/Users/ymp", "port": 1, "pid": os.getpid()},
                 {"sessionId": "forge-ymp", "name": "ymp", "cwd": "/Users/ymp", "port": 2, "pid": os.getpid()}]
        with self._with_peers(peers):
            self.assertEqual(len(fleet.find_session("ymp")), 2)                       # ambiguous...
            hits = fleet.find_session("ymp", exclude_sid="forge-ymp")                 # ...unless you're one of them
            self.assertEqual([h["sid"] for h in hits], ["claude-ymp"])


class TestPathlessFileActions(unittest.TestCase):
    """Live failure: qwen3-coder emitted write_file without `path` — the empty
    path resolved to the cwd and the read-before-edit guard blocked every try
    with a nonsense message. Aliases must be honored; truly pathless actions
    must get an instructive error, not a confusing block."""

    class Scripted:
        name = "s"
        def __init__(self, replies):
            self.replies = list(replies)
            self.schemas = []           # P5.1: the schema requested for each call
        def chat(self, m, schema=None, temperature=0.0):
            self.schemas.append(schema)
            return self.replies.pop(0)

    def test_filename_alias_is_honored(self):
        import json as _json
        d = tempfile.mkdtemp()
        b = self.Scripted([
            _json.dumps({"thought": "t", "action": "write_file", "filename": "go/http_server.go",
                         "content": "package main\n"}),
            _json.dumps({"thought": "t", "action": "say", "message": "done"}),
        ])
        Agent(b, sm.EphemeralSession(d, "s"), max_steps=5).send("make it")
        with open(os.path.join(d, "go/http_server.go")) as f:
            self.assertEqual(f.read(), "package main\n")

    def test_missing_required_resends_with_variant(self):
        """P5.1: a write_file missing its `path` is not a wasted step — the harness
        re-asks with ONLY the write_file variant grammar-forced; the completed resend
        writes the file. The variant schema (const write_file) is what was requested."""
        import json as _json
        d = tempfile.mkdtemp()
        b = self.Scripted([
            _json.dumps({"thought": "t", "action": "write_file", "content": "package main\n"}),
            _json.dumps({"thought": "t", "action": "write_file", "path": "m.go", "content": "package main\n"}),
            _json.dumps({"thought": "t", "action": "say", "message": "ok"}),
        ])
        Agent(b, sm.EphemeralSession(d, "s"), max_steps=5).send("make it")
        with open(os.path.join(d, "m.go")) as f:
            self.assertEqual(f.read(), "package main\n")
        # the resend (2nd call) was constrained to the write_file variant only
        from forge.tools import ACTION_VARIANTS
        self.assertEqual(b.schemas[1], ACTION_VARIANTS["write_file"])

    def test_missing_required_falls_back_to_text_nudge(self):
        """When even the grammar-forced resend is still incomplete (advisory engine),
        the harness falls back to the instructive text nudge — no nonsense block."""
        import json as _json
        d = tempfile.mkdtemp()
        events = []
        b = self.Scripted([
            _json.dumps({"thought": "t", "action": "write_file", "content": "package main\n"}),
            _json.dumps({"thought": "t", "action": "write_file", "content": "still no path\n"}),
            _json.dumps({"thought": "t", "action": "say", "message": "ok"}),
        ])
        a = Agent(b, sm.EphemeralSession(d, "s"), max_steps=5,
                  on_event=lambda kind, **k: events.append((kind, k)))
        a.send("make it")
        obs = [k.get("text", "") for kind, k in events if kind == "observation"]
        self.assertTrue(any("missing required field(s): `path`" in t for t in obs), obs)
        self.assertFalse(any("Blocked: read" in t for t in obs), obs)   # the old nonsense block

    def test_dir_path_does_not_trigger_read_guard(self):
        import json as _json
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "pkg"))
        b = self.Scripted([
            _json.dumps({"thought": "t", "action": "write_file", "path": "pkg/x.go", "content": "x"}),
            _json.dumps({"thought": "t", "action": "say", "message": "ok"}),
        ])
        Agent(b, sm.EphemeralSession(d, "s"), max_steps=5).send("go")
        self.assertTrue(os.path.exists(os.path.join(d, "pkg/x.go")))


class TestSalvageAndProfiles(unittest.TestCase):
    """P5.4: a deterministic salvage pass recovers fenced / prose-wrapped /
    trailing-comma action JSON for free BEFORE it counts a malformed strike, and the
    path-field alias table now lives in forge/profiles.py resolved from backend.name.
    Mirrors the malformed-bailout tests (a genuinely-broken output still strikes)."""

    _SAY = '{"thought":"t","action":"say","message":"done"}'

    def _run(self, raws, max_steps=8):
        """Drive a send() over a scripted sequence of RAW model outputs; return
        (result, RecSession) so tests can inspect the salvage/malformed logs."""
        d = tempfile.mkdtemp()
        sess = _RecSession(d)
        a = Agent(ScriptBackend(raws), sess, max_steps=max_steps)
        return a.send("go"), sess, d

    def _kinds(self, sess, k):
        return [f for kind, f in sess.logs if kind == k]

    def test_fenced_json_salvages_without_strike(self):
        raw = "```json\n" + self._SAY + "\n```"
        result, sess, _ = self._run([raw])
        self.assertEqual(result, "done")                         # action executed
        salv = self._kinds(sess, "salvage")
        self.assertEqual([s["stage"] for s in salv], ["fence"])
        self.assertEqual(self._kinds(sess, "malformed"), [])     # NOT a strike

    def test_prose_prefixed_json_salvages_via_brace_scan(self):
        raw = "Sure, here is the action:\n" + self._SAY + "\nLet me know!"
        result, sess, _ = self._run([raw])
        self.assertEqual(result, "done")
        self.assertEqual([s["stage"] for s in self._kinds(sess, "salvage")], ["brace"])
        self.assertEqual(self._kinds(sess, "malformed"), [])

    def test_trailing_comma_json_salvages(self):
        raw = '{"thought":"t","action":"say","message":"done",}'
        result, sess, _ = self._run([raw])
        self.assertEqual(result, "done")
        self.assertEqual([s["stage"] for s in self._kinds(sess, "salvage")], ["trailing_comma"])
        self.assertEqual(self._kinds(sess, "malformed"), [])

    def test_salvaged_action_executes_and_writes_file(self):
        """A salvaged non-say action runs like a normal parsed action."""
        obj = '{"thought":"t","action":"write_file","path":"out.txt","content":"hi\\n"}'
        result, sess, d = self._run(["```\n" + obj + "\n```", self._SAY])
        self.assertEqual(result, "done")
        with open(os.path.join(d, "out.txt")) as f:
            self.assertEqual(f.read(), "hi\n")
        self.assertEqual([s["stage"] for s in self._kinds(sess, "salvage")], ["fence"])

    def test_salvage_does_not_advance_the_strike_counter(self):
        """Four genuinely-broken outputs (bad=4) then a fenced action: salvage recovers
        it instead of tripping the abort-at-5, proving salvage never counts a strike."""
        raws = ["not json"] * 4 + ["```json\n" + self._SAY + "\n```"]
        result, sess, _ = self._run(raws)
        self.assertEqual(result, "done")
        self.assertEqual(len(self._kinds(sess, "malformed")), 4)  # only the real ones
        self.assertEqual(len(self._kinds(sess, "salvage")), 1)

    def test_genuinely_broken_still_strikes_and_logs_malformed(self):
        from forge.agent import TRACE_V
        result, sess, _ = self._run(["this is not json at all"] * 6, max_steps=10)
        self.assertIn("could not hold", result)                  # bails, unchanged
        malformed = self._kinds(sess, "malformed")
        self.assertTrue(malformed and all(m["v"] == TRACE_V for m in malformed))
        self.assertEqual(self._kinds(sess, "salvage"), [])       # nothing salvaged

    def test_truncated_object_is_not_salvageable(self):
        """A NUM_PREDICT-truncated tail has no complete object to slice — it must fall
        through to the malformed strike, not be falsely recovered."""
        raw = '{"thought":"t","action":"write_file","path":"a.py","content":"def f():'
        result, sess, _ = self._run([raw] * 6, max_steps=10)
        self.assertIn("could not hold", result)
        self.assertEqual(self._kinds(sess, "salvage"), [])
        self.assertTrue(self._kinds(sess, "malformed"))

    def test_alias_table_lives_in_profiles_and_resolves(self):
        from forge import profiles
        default = profiles.resolve("script")["aliases"]
        self.assertEqual(default, ("filename", "file", "filepath", "file_path", "name"))
        # a qwen-named backend still resolves filename/file/etc (case-insensitive)
        for alias in ("filename", "file", "filepath", "file_path", "name"):
            self.assertIn(alias, profiles.resolve("ollama:qwen2.5-coder:3b")["aliases"])
            self.assertIn(alias, profiles.resolve("openai:Qwen-72B")["aliases"])
        self.assertEqual(profiles.resolve(None)["aliases"], default)

    def test_agent_resolves_alias_table_from_backend_name(self):
        """The Agent pulls its alias table from profiles.resolve(backend.name), and a
        qwen backend still honors a `filename` alias end-to-end (P3.2 behavior intact
        after the table moved to data)."""
        d = tempfile.mkdtemp()
        b = ScriptBackend([
            '{"thought":"t","action":"write_file","filename":"q.go","content":"package main\\n"}',
            self._SAY,
        ])
        b.name = "ollama:qwen2.5-coder:3b"
        a = Agent(b, sm.EphemeralSession(d, "s"), max_steps=6)
        self.assertIn("filename", a._aliases)
        a.send("make it")
        with open(os.path.join(d, "q.go")) as f:
            self.assertEqual(f.read(), "package main\n")


class TestActionGrammar(unittest.TestCase):
    """P5.1 state-dependent action grammar: per-action variants with the right
    required fields, and a per-step legal-action narrowing (mutating actions gone
    from the grammar in plan mode / when self.allowed excludes them)."""

    def test_variants_carry_per_action_required_fields(self):
        from forge.tools import ACTION_VARIANTS, required_fields
        cases = {
            "bash": ["command"],
            "read_file": ["path"],
            "write_file": ["path", "content"],
            "edit_file": ["path", "old", "new"],
            "grep": ["pattern"],
            "glob": ["pattern"],
            "list_files": [],
        }
        for action, req in cases.items():
            v = ACTION_VARIANTS[action]
            self.assertEqual(v["properties"]["action"], {"const": action})
            self.assertEqual(v["required"], ["thought", "action"] + req)
            self.assertEqual(required_fields(action), req)
            self.assertFalse(v["additionalProperties"])
            # `plan` and `note` are OPTIONAL in every variant — a plan/note update
            # must ride any action and must never be forced.
            self.assertIn("plan", v["properties"])
            self.assertIn("note", v["properties"])
            self.assertNotIn("plan", v["required"])
            self.assertNotIn("note", v["required"])

    def test_variant_advertises_only_its_own_fields(self):
        from forge.tools import ACTION_VARIANTS
        # a bash variant cannot legally carry edit fields; edit_file cannot carry command
        self.assertNotIn("old", ACTION_VARIANTS["bash"]["properties"])
        self.assertNotIn("command", ACTION_VARIANTS["edit_file"]["properties"])
        self.assertIn("command", ACTION_VARIANTS["bash"]["properties"])

    def test_build_schema_full_set_is_anyof_of_variants(self):
        from forge.tools import build_schema, ACTION_VARIANTS, ALL_ACTIONS
        s = build_schema(set(ALL_ACTIONS), "auto")
        self.assertIn("anyOf", s)
        consts = [v["properties"]["action"]["const"] for v in s["anyOf"]]
        self.assertEqual(consts, list(ALL_ACTIONS))          # canonical enum order
        self.assertEqual(s["anyOf"][0], ACTION_VARIANTS["bash"])

    def test_build_schema_single_action_is_bare_variant(self):
        from forge.tools import build_schema, ACTION_VARIANTS
        # one legal action → the single variant, NOT an anyOf-of-one (root anyOf is
        # exactly what OpenAI strict mode rejects).
        self.assertEqual(build_schema({"say"}, "auto"), ACTION_VARIANTS["say"])

    def test_build_schema_empty_falls_back_to_flat(self):
        from forge.tools import build_schema, ACTION_SCHEMA
        self.assertIs(build_schema(set(), "auto"), ACTION_SCHEMA)

    def test_plan_mode_legal_excludes_mutating(self):
        d = tempfile.mkdtemp()
        a = Agent(ScriptBackend([]), sm.EphemeralSession(d, "s"))
        a.mode = "plan"
        legal = a._legal_actions()
        for mut in ("bash", "write_file", "edit_file", "fleet_send"):
            self.assertNotIn(mut, legal)
        for ro in ("read_file", "list_files", "grep", "glob", "say"):
            self.assertIn(ro, legal)

    def test_allowed_intersects_legal(self):
        d = tempfile.mkdtemp()
        a = Agent(ScriptBackend([]), sm.EphemeralSession(d, "s"), allowed={"read_file", "say"})
        self.assertEqual(a._legal_actions(), {"read_file", "say"})

    def test_plan_mode_grammar_passed_to_backend(self):
        """End-to-end: in plan mode the schema handed to the backend is the anyOf of
        the read-only variants — no bash/write/edit const anywhere in it."""
        import json as _json
        from forge.tools import build_schema
        d = tempfile.mkdtemp()

        class Rec:
            name = "rec"
            def __init__(self):
                self.schema = None
            def stream(self, m, schema=None, temperature=0.0):
                self.schema = schema
                yield _json.dumps({"thought": "t", "action": "say", "message": "here is the plan"})
        b = Rec()
        a = Agent(b, sm.EphemeralSession(d, "s"), max_steps=2)
        a.mode = "plan"
        a.send("plan it")
        consts = [v["properties"]["action"]["const"] for v in b.schema["anyOf"]]
        self.assertNotIn("bash", consts)
        self.assertNotIn("edit_file", consts)
        self.assertEqual(b.schema, build_schema(a._legal_actions(), "plan"))


class TestDryRun(unittest.TestCase):
    """P5.2 deterministic pre-execution verifier: dry_run(act, cwd) -> (score, reason).
    A CLEAR miss is 0.0 (caught for free), an exact/unverifiable action is 1.0, a
    unique whitespace-tolerant edit is 0.7. Read-only — never writes."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def _p(self, name):
        return os.path.join(self.d, name)

    # ---- edit_file: exact hit / fuzzy / miss / ambiguous / no-file -------------
    def test_edit_exact_unique_match_scores_1(self):
        _write(self._p("f.py"), "x = 1\n")
        s, _ = dry_run({"action": "edit_file", "path": "f.py", "old": "x = 1", "new": "x = 2"}, self.d)
        self.assertEqual(s, 1.0)

    def test_edit_whitespace_only_diff_scores_fuzzy_0_7(self):
        # file is 4-space-indented; the model's `old` is 8-space-indented — not an
        # exact substring, but a unique whitespace-tolerant match.
        _write(self._p("f.py"), "    x = 1\n")
        s, _ = dry_run({"action": "edit_file", "path": "f.py", "old": "        x = 1", "new": "y"}, self.d)
        self.assertEqual(s, 0.7)

    def test_edit_old_absent_scores_0(self):
        _write(self._p("f.py"), "x = 1\n")
        s, why = dry_run({"action": "edit_file", "path": "f.py", "old": "NOPE", "new": "q"}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("not found", why)

    def test_edit_ambiguous_old_scores_0(self):
        _write(self._p("f.py"), "x = 1\nx = 1\n")
        s, why = dry_run({"action": "edit_file", "path": "f.py", "old": "x = 1", "new": "q"}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("2 times", why)

    def test_edit_missing_file_scores_0(self):
        s, why = dry_run({"action": "edit_file", "path": "gone.py", "old": "x", "new": "y"}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("no such file", why)

    def test_edit_empty_old_scores_0(self):
        _write(self._p("f.py"), "x = 1\n")
        s, _ = dry_run({"action": "edit_file", "path": "f.py", "old": "", "new": "y"}, self.d)
        self.assertEqual(s, 0.0)

    # ---- write_file .py: compile pass / fail ----------------------------------
    def test_write_py_compiles_scores_1(self):
        s, _ = dry_run({"action": "write_file", "path": "a.py", "content": "def f():\n    return 1\n"}, self.d)
        self.assertEqual(s, 1.0)

    def test_write_py_syntax_error_scores_0(self):
        s, why = dry_run({"action": "write_file", "path": "a.py", "content": "x = = 1\n"}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("SyntaxError", why)

    # ---- write_file .json: valid / invalid ------------------------------------
    def test_write_json_valid_scores_1(self):
        s, _ = dry_run({"action": "write_file", "path": "a.json", "content": '{"a": 1}'}, self.d)
        self.assertEqual(s, 1.0)

    def test_write_json_invalid_scores_0(self):
        s, why = dry_run({"action": "write_file", "path": "a.json", "content": '{"a": }'}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("invalid json", why)

    def test_write_other_extension_has_no_probe(self):
        s, _ = dry_run({"action": "write_file", "path": "a.txt", "content": "anything at all"}, self.d)
        self.assertEqual(s, 1.0)

    # ---- bash: builtin / assignment / path / on-PATH / missing / unparseable --
    def test_bash_builtin_head_scores_1(self):
        # `cd` is a shell builtin not on PATH — must NOT false-negative. Only the
        # head token is inspected, so `cd x && make` is fine too.
        s, _ = dry_run({"action": "bash", "command": "cd sub && ls"}, self.d)
        self.assertEqual(s, 1.0)

    def test_bash_var_assignment_prefix_scores_1(self):
        s, why = dry_run({"action": "bash", "command": "VAR=1 make target"}, self.d)
        self.assertEqual(s, 1.0)
        self.assertIn("assignment", why)

    def test_bash_on_path_scores_1(self):
        s, _ = dry_run({"action": "bash", "command": "ls -la"}, self.d)
        self.assertEqual(s, 1.0)

    def test_bash_path_like_head_scores_1(self):
        s, _ = dry_run({"action": "bash", "command": "./run.sh --now"}, self.d)
        self.assertEqual(s, 1.0)

    def test_bash_missing_command_scores_0(self):
        s, why = dry_run({"action": "bash", "command": "forge-no-such-cmd-xyzzy foo"}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("not found", why)

    def test_bash_unparseable_scores_0(self):
        s, why = dry_run({"action": "bash", "command": 'echo "unbalanced'}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("parse", why)

    def test_bash_empty_scores_0(self):
        s, _ = dry_run({"action": "bash", "command": "   "}, self.d)
        self.assertEqual(s, 0.0)

    # ---- read_file: exists / missing ------------------------------------------
    def test_read_existing_scores_1(self):
        _write(self._p("here.txt"), "hi")
        s, _ = dry_run({"action": "read_file", "path": "here.txt"}, self.d)
        self.assertEqual(s, 1.0)

    def test_read_missing_scores_0(self):
        s, why = dry_run({"action": "read_file", "path": "absent.txt"}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("no such file", why)

    def test_action_without_a_probe_scores_1(self):
        # list_files / grep / glob have no cheap check — never penalize them.
        for act in ({"action": "list_files"}, {"action": "grep", "pattern": "x"}):
            s, _ = dry_run(act, self.d)
            self.assertEqual(s, 1.0)


class TestResample(unittest.TestCase):
    """P5.2 best-of-N: a greedy action that dry_run scores 0 is not executed — the
    harness re-asks the SAME prompt at rising temperature, scores each candidate,
    and executes the argmax. The winner REPLACES messages[-1] (the transcript must
    match what ran); all-miss falls back to the greedy original; `say` is never
    dry-run; resamples never lengthen the message list."""

    def _assistants(self, agent):
        return [m["content"] for m in agent.messages if m["role"] == "assistant"]

    def test_greedy_miss_then_good_resample_executes_winner(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        miss = '{"thought":"blind","action":"edit_file","path":"r.py","old":"NOPE","new":"x = 2"}'
        win = '{"thought":"good","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}'
        miss2 = '{"thought":"bad2","action":"edit_file","path":"r.py","old":"ALSO NOPE","new":"z"}'
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',   # call 0 (greedy)
            miss,                                                        # call 1 (greedy edit — miss)
            win,                                                        # call 2 (resample t=0.5 — 1.0)
            miss2,                                                      # call 3 (resample t=0.8 — 0)
            '{"thought":"done","action":"say","message":"done"}',      # call 4 (greedy)
        ]
        sess = _RecSession(d)
        a = Agent(ScriptBackend(actions), sess, max_steps=8)
        a.send("change x")
        # the resampled winner ran, not the miss
        self.assertEqual(_read(os.path.join(d, "r.py")), "x = 2\n")
        # transcript matches what ran: the winner is an assistant echo, the miss is gone
        assistants = self._assistants(a)
        self.assertIn(win, assistants)
        self.assertNotIn(miss, assistants)
        # a resample telemetry record was logged with v=TRACE_V and the win
        recs = [f for k, f in sess.logs if k == "resample"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["v"], 1)
        self.assertTrue(recs[0]["replaced"])
        self.assertEqual(recs[0]["base_score"], 0.0)
        self.assertEqual(recs[0]["best_score"], 1.0)
        self.assertEqual(recs[0]["samples"], 2)

    def test_all_samples_miss_executes_greedy_original(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        miss = '{"thought":"blind","action":"edit_file","path":"r.py","old":"NOPE","new":"q"}'
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',
            miss,                                                                       # greedy miss
            '{"thought":"m2","action":"edit_file","path":"r.py","old":"NOPETWO","new":"q"}',   # resample miss
            '{"thought":"m3","action":"edit_file","path":"r.py","old":"NOPE3","new":"q"}',     # resample miss
            '{"thought":"done","action":"say","message":"done"}',
        ]
        sess = _RecSession(d)
        a = Agent(ScriptBackend(actions), sess, max_steps=8)
        a.send("change x")
        # nothing matched → the greedy original still ran (teaching failure), file unchanged
        self.assertEqual(_read(os.path.join(d, "r.py")), "x = 1\n")
        # the greedy miss stayed as the assistant echo (not replaced)
        self.assertIn(miss, self._assistants(a))
        recs = [f for k, f in sess.logs if k == "resample"]
        self.assertEqual(len(recs), 1)
        self.assertFalse(recs[0]["replaced"])
        self.assertEqual(recs[0]["best_score"], 0.0)

    def test_say_is_never_dry_run(self):
        import forge.agent as agent_mod
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        seen = []
        orig = agent_mod.dry_run

        def spy(act, cwd):
            seen.append(act.get("action"))
            return orig(act, cwd)

        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"done","action":"say","message":"all set"}',
        ]
        agent_mod.dry_run = spy
        try:
            a = Agent(ScriptBackend(actions), _RecSession(d), max_steps=6)
            a.send("look")
        finally:
            agent_mod.dry_run = orig
        # read_file was dry-run; say never reached dry_run
        self.assertIn("read_file", seen)
        self.assertNotIn("say", seen)

    def test_resample_does_not_lengthen_the_message_list(self):
        # a resampled step (greedy miss + 2 rejected/accepted candidates = 3 model
        # calls) must leave the same number of messages as a plain 3-action turn.
        def run(actions):
            d = tempfile.mkdtemp()
            _write(os.path.join(d, "r.py"), "x = 1\n")
            a = Agent(ScriptBackend(actions), _RecSession(d), max_steps=8)
            a.send("go")
            return a

        resample_turn = run([
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"blind","action":"edit_file","path":"r.py","old":"NOPE","new":"x = 2"}',  # miss
            '{"thought":"good","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',  # win
            '{"thought":"m","action":"edit_file","path":"r.py","old":"NOPE2","new":"z"}',         # miss
            '{"thought":"done","action":"say","message":"done"}',
        ])
        control_turn = run([
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"good","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',  # direct
            '{"thought":"done","action":"say","message":"done"}',
        ])
        # both executed exactly read+edit+say; the two extra resample generations
        # added no messages, so the transcripts are the same length.
        self.assertEqual(len(resample_turn.messages), len(control_turn.messages))
        self.assertEqual(len(self._assistants(resample_turn)), 3)


class _HeatBackend:
    """P5.5 fake backend: records the temperature of every generation and yields a
    scripted sequence of action JSONs (mirrors ScriptBackend, plus .temps)."""
    name = "heat"

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0
        self.temps = []

    def _next(self, temperature):
        self.temps.append(temperature)
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return act

    def stream(self, messages, schema=None, temperature=0.0):
        yield self._next(temperature)

    def chat(self, messages, schema=None, temperature=0.0):
        return self._next(temperature)


class TestRetryHeat(unittest.TestCase):
    """P5.5 retry-heat: every turn starts greedy (temperature 0.0); a malformed /
    loop / failed / declined retry bumps the sampling temperature +0.4 (capped at
    0.7); a clean execution resets it to 0.0. The first sample of each generation
    carries this heat, so a stuck 3B is perturbed instead of re-emitting the same
    greedy action."""

    def setUp(self):
        # P5.8: isolate passport telemetry PER TEST so accumulated malformed strikes
        # across this class's runs can't tip the "heat" model into malformed-prone and
        # change heat_bump (0.4 → 0.5) out from under these exact-schedule assertions.
        self._prev_pp = _profile.PROFILE_DIR
        _profile.PROFILE_DIR = tempfile.mkdtemp(prefix="forge-profile-heat-")

    def tearDown(self):
        _profile.PROFILE_DIR = self._prev_pp

    def test_schedule_malformed_fail_success_reset(self):
        # 0.0 (greedy) -> malformed bumps to 0.4 -> a failed action bumps to 0.7 ->
        # a clean read resets to 0.0 -> the final say is greedy again.
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "s.py"), "x = 1\n")
        actions = [
            "not json",                                                  # gen@0.0 malformed -> 0.4
            '{"thought":"t","action":"bash","command":"false"}',         # gen@0.4 fails -> 0.7
            '{"thought":"t","action":"read_file","path":"s.py"}',        # gen@0.7 ok -> reset 0.0
            '{"thought":"done","action":"say","message":"done"}',        # gen@0.0 clean
        ]
        b = _HeatBackend(actions)
        a = Agent(b, _RecSession(d), max_steps=8)
        a.send("go")
        self.assertEqual(b.temps, [0.0, 0.4, 0.7, 0.0])
        self.assertEqual(a._heat, 0.0)   # reset by the successful read

    def test_clean_turn_stays_greedy(self):
        # a turn with no retries never leaves temperature 0.0.
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "s.py"), "x = 1\n")
        actions = [
            '{"thought":"t","action":"read_file","path":"s.py"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        b = _HeatBackend(actions)
        Agent(b, _RecSession(d), max_steps=6).send("look")
        self.assertEqual(b.temps, [0.0, 0.0])

    def test_loop_break_bumps_heat(self):
        # the same edit is blocked (read-before-edit) three times; the 3rd trips the
        # loop detector, whose bump raises the NEXT generation off greedy.
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "u.py"), "a\n")   # exists but never read -> gate blocks the edit
        edit = '{"thought":"t","action":"edit_file","path":"u.py","old":"a","new":"b"}'
        actions = [edit, edit, edit, '{"thought":"done","action":"say","message":"done"}']
        b = _HeatBackend(actions)
        a = Agent(b, _RecSession(d), max_steps=8)
        a.send("go")
        # first three gens are greedy (blocks are not heat sites); the loop trip on the
        # 3rd bumps heat so the say samples at 0.4.
        self.assertEqual(b.temps, [0.0, 0.0, 0.0, 0.4])

    def test_declined_action_bumps_heat(self):
        # a manual-mode DECLINE perturbs the next generation; a plan-mode block does not.
        d = tempfile.mkdtemp()
        actions = [
            '{"thought":"t","action":"bash","command":"echo hi"}',   # gen@0.0 declined -> 0.4
            '{"thought":"done","action":"say","message":"ok"}',      # gen@0.4
        ]
        b = _HeatBackend(actions)
        a = Agent(b, _RecSession(d), max_steps=6)
        a.mode = "manual"
        a.approve = lambda desc: "no"
        a.send("go")
        self.assertEqual(b.temps, [0.0, 0.4])

    def test_plan_block_does_not_bump_heat(self):
        d = tempfile.mkdtemp()
        actions = [
            '{"thought":"t","action":"bash","command":"echo hi"}',   # plan-blocked, NOT a decline
            '{"thought":"done","action":"say","message":"here is my plan"}',
        ]
        b = _HeatBackend(actions)
        a = Agent(b, _RecSession(d), max_steps=6)
        a.mode = "plan"
        a.send("go")
        self.assertEqual(b.temps, [0.0, 0.0])

    def test_heat_caps_at_point_seven(self):
        # four consecutive malformed strikes never push heat past 0.7.
        d = tempfile.mkdtemp()
        b = _HeatBackend(["not json"])   # every gen is malformed
        a = Agent(b, _RecSession(d), max_steps=5)
        a.send("go")   # aborts after 5 malformed strikes
        self.assertEqual(b.temps[:5], [0.0, 0.4, 0.7, 0.7, 0.7])


class TestBackgroundBash(unittest.TestCase):
    """Servers must keep running while the agent continues, and die with forge."""

    def tearDown(self):
        from forge import tools
        tools._kill_background()
        tools._BG_PROCS.clear()

    def test_background_returns_immediately_and_keeps_running(self):
        import time as _t
        t0 = _t.monotonic()
        obs, ok = execute({"action": "bash", "command": "sleep 30", "background": True}, "/tmp")
        self.assertLess(_t.monotonic() - t0, 5)
        self.assertTrue(ok)
        self.assertIn("pid", obs)
        self.assertIn("KEEPS RUNNING", obs)
        from forge import tools
        self.assertIsNone(tools._BG_PROCS[-1].poll())    # still alive

    def test_instant_crash_is_reported(self):
        obs, ok = execute({"action": "bash", "command": "echo boom >&2; exit 3", "background": True}, "/tmp")
        self.assertFalse(ok)
        self.assertIn("exited immediately", obs)
        self.assertIn("boom", obs)

    def test_trailing_ampersand_heuristic(self):
        obs, ok = execute({"action": "bash", "command": "sleep 30 &"}, "/tmp")
        self.assertTrue(ok)
        self.assertIn("background", obs)
        # && must NOT trigger it
        obs2, ok2 = execute({"action": "bash", "command": "true && echo chained"}, "/tmp")
        self.assertTrue(ok2)
        self.assertIn("chained", obs2)

    def test_kill_background_cleans_up(self):
        from forge import tools
        execute({"action": "bash", "command": "sleep 30", "background": True}, "/tmp")
        p = tools._BG_PROCS[-1]
        tools._kill_background()
        p.wait(timeout=5)
        self.assertIsNotNone(p.poll())


class TestApprovalGate(unittest.TestCase):
    def test_request_answer_across_threads(self):
        import threading
        from forge.tui import ApprovalGate
        gate = ApprovalGate()
        got = []
        t = threading.Thread(target=lambda: got.append(gate.request("bash rm -rf junk")))
        t.start()
        for _ in range(50):
            if gate.pending():
                break
            import time as _t; _t.sleep(0.01)
        self.assertEqual(gate.pending(), "bash rm -rf junk")
        self.assertTrue(gate.answer("always"))
        t.join(timeout=5)
        self.assertEqual(got, ["always"])
        self.assertIsNone(gate.pending())

    def test_stop_event_resolves_to_no(self):
        import threading
        from forge.tui import ApprovalGate
        gate = ApprovalGate()
        stop = threading.Event(); stop.set()
        self.assertEqual(gate.request("bash x", stop_event=stop), "no")


class TestModeGate(unittest.TestCase):
    def _agent(self, mode, approve=None, approvals=()):
        a = Agent.__new__(Agent)
        a.mode = mode
        a.approvals = set(approvals)
        a.approve = approve or (lambda d: "yes")
        return a

    def test_auto_never_gates(self):
        a = self._agent("auto", approve=lambda d: self.fail("should not ask"))
        self.assertIsNone(a._gate("bash", {"action": "bash", "command": "rm -rf /"}))

    def test_plan_blocks_mutating_allows_readonly(self):
        a = self._agent("plan")
        self.assertIn("plan mode", a._gate("bash", {"action": "bash", "command": "ls"}))
        self.assertIn("plan mode", a._gate("edit_file", {"action": "edit_file", "path": "x"}))
        self.assertIsNone(a._gate("read_file", {"action": "read_file", "path": "x"}))
        self.assertIsNone(a._gate("fleet_send", {"action": "fleet_send", "target": "list"}))

    def test_manual_yes_no_always(self):
        asked = []
        a = self._agent("manual", approve=lambda d: asked.append(d) or "no")
        self.assertIn("DECLINED", a._gate("bash", {"action": "bash", "command": "git push"}))
        self.assertEqual(asked, ["bash git push"])
        a = self._agent("manual", approve=lambda d: "yes")
        self.assertIsNone(a._gate("edit_file", {"action": "edit_file", "path": "f.py"}))
        import unittest.mock as mock
        a = self._agent("manual", approve=lambda d: "always")
        with mock.patch("forge.config.set_key") as sk:
            self.assertIsNone(a._gate("bash", {"action": "bash", "command": "git status"}))
        self.assertIn("bash:git", a.approvals)
        sk.assert_called_once()
        # now pre-approved: approve callback must not be consulted again
        a.approve = lambda d: self.fail("already approved")
        self.assertIsNone(a._gate("bash", {"action": "bash", "command": "git diff"}))

    def test_approval_key_granularity(self):
        a = self._agent("manual")
        self.assertEqual(a._approval_key({"action": "bash", "command": "git push origin"}), "bash:git")
        self.assertEqual(a._approval_key({"action": "edit_file", "path": "x"}), "edit_file")


class TestExplorer(unittest.TestCase):
    def _ex(self, root):
        from forge.tui import Explorer, Screen
        s = Screen.__new__(Screen)
        s.w, s.h, s.rows, s.enabled = 80, 24, 2, False
        return Explorer(s, root)

    def _tree(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "src"))
        _write(os.path.join(d, "src", "main.go"), "package main\n")
        _write(os.path.join(d, "README.md"), "# hi\n")
        _write(os.path.join(d, ".hidden"), "x")
        return d

    def test_entries_dirs_first_hidden_toggle(self):
        d = self._tree()
        ex = self._ex(d)
        ents = ex._entries(d)
        self.assertEqual(ents[0], ("src", True))            # dirs first
        self.assertNotIn((".hidden", False), ents)          # hidden filtered
        ex.show_hidden = True
        self.assertIn((".hidden", False), ex._entries(d))

    def test_preview_file_and_binary(self):
        d = self._tree()
        ex = self._ex(d)
        pv = ex._preview(os.path.join(d, "README.md"), False, 40, 10)
        self.assertEqual(pv[0], "# hi")
        with open(os.path.join(d, "blob.bin"), "wb") as f:
            f.write(b"\x00\x01\x02")
        pv2 = ex._preview(os.path.join(d, "blob.bin"), False, 40, 10)
        self.assertIn("binary", pv2[0])


class TestLineEditor(unittest.TestCase):
    def test_editing_basics(self):
        from forge.tui import LineEditor
        ed = LineEditor()
        for c in "helo":
            ed.handle(c)
        ed.handle(b"\x1b[D")            # left
        ed.handle("l")
        self.assertEqual(ed.text(), "hello")
        ed.handle(b"\x05")              # ctrl-e end
        ed.handle(b"\x17")              # ctrl-w kills the word
        self.assertEqual(ed.text(), "")

    def test_history_navigation(self):
        from forge.tui import LineEditor
        ed = LineEditor(["first", "second"])
        ed.handle(b"\x1b[A")
        self.assertEqual(ed.text(), "second")
        ed.handle(b"\x1b[A")
        self.assertEqual(ed.text(), "first")
        ed.handle(b"\x1b[B"); ed.handle(b"\x1b[B")
        self.assertEqual(ed.text(), "")


class TestRenderWrap(unittest.TestCase):
    """Reply text wraps at word boundaries with a 2-space margin."""

    def _ui(self, width=40):
        from forge.repl import UI
        out = []
        return UI(out.append, width=lambda: width), out

    def test_stream_wraps_at_word_boundaries(self):
        ui, out = self._ui(40)
        ui("token", text="The files here are: .claude, .git, CONTRIBUTING.md plus several more words to overflow")
        ui("say", message="")
        text = "".join(out)
        self.assertTrue(all(len(l) <= 38 for l in text.splitlines()))
        flat = " ".join(text.split())
        self.assertIn("CONTRIBUTING.md", flat)          # never split mid-word

    def test_stream_word_split_across_chunks(self):
        ui, out = self._ui(40)
        for ch in ("Hel", "lo wor", "ld and more text that keeps going well past the width"):
            ui("token", text=ch)
        ui("say", message="")
        flat = " ".join("".join(out).split())
        self.assertIn("Hello world", flat)

    def test_block_wrap_indents_every_line(self):
        ui, out = self._ui(40)
        ui("say", message="one two three four five six seven eight nine ten eleven twelve thirteen")
        lines = [l for l in "".join(out).splitlines() if l]
        self.assertTrue(all(l.startswith("  ") and len(l) <= 38 for l in lines))


class TestBoxLayout(unittest.TestCase):
    """The input box wraps the logical line across its writable rows."""

    def _screen(self, w=24, rows=2):
        from forge.tui import Screen
        s = Screen.__new__(Screen)      # no tty / signal setup
        s.w, s.rows = w, rows
        return s

    def test_short_line_stays_on_row0(self):
        segs, crow, ccol = self._screen()._layout(2, "hello", 5)
        self.assertEqual(segs[0], "hello")
        self.assertEqual((crow, ccol), (0, 7))

    def test_wraps_to_second_row(self):
        # w=24 -> inner=20, caps=[18, 20]
        segs, crow, ccol = self._screen()._layout(2, "a" * 30, 30)
        self.assertEqual((segs[0], segs[1]), ("a" * 18, "a" * 12))
        self.assertEqual((crow, ccol), (1, 12))

    def test_scrolls_when_overflowing_the_box(self):
        segs, crow, ccol = self._screen()._layout(2, "b" * 50, 50)
        self.assertEqual(crow, 1)                       # cursor stays visible
        self.assertLessEqual(ccol, 19)


class TestTuiHelpers(unittest.TestCase):
    """Pure helpers from the TUI: ANSI-aware clipping and raw-mode key decoding."""

    def test_clip_plain(self):
        from forge.tui import _clip
        self.assertEqual(_clip("hello", 10), "hello")
        self.assertEqual(_clip("hello world", 5), "hello\033[0m")

    def test_clip_preserves_ansi(self):
        from forge.tui import _clip, _vis
        s = "\033[32mgreen\033[0m and more"
        out = _clip(s, 7)
        self.assertEqual(_vis(out), 7)
        self.assertIn("\033[32m", out)          # color codes survive, don't count
        self.assertEqual(_clip(s, 99), s)       # no truncation → untouched

    def _feed(self, data):
        from forge.tui import _read_key
        r, w = os.pipe()
        try:
            os.write(w, data)
            keys = []
            for _ in range(16):
                import select as _select
                if not _select.select([r], [], [], 0)[0]:
                    break
                keys.append(_read_key(r))
            return keys
        finally:
            os.close(r); os.close(w)

    def test_read_key_ascii_and_utf8(self):
        self.assertEqual(self._feed(b"a"), ["a"])
        self.assertEqual(self._feed("é".encode()), ["é"])     # 2-byte UTF-8
        self.assertEqual(self._feed("✓".encode()), ["✓"])     # 3-byte UTF-8

    def test_read_key_control_and_sequences(self):
        self.assertEqual(self._feed(b"\x01"), [b"\x01"])              # Ctrl-A
        self.assertEqual(self._feed(b"\x1b[A"), [b"\x1b[A"])          # arrow up
        self.assertEqual(self._feed(b"\x1b[3~"), [b"\x1b[3~"])        # Delete: whole seq, no stray ~
        self.assertEqual(self._feed(b"\x1b[3~x"), [b"\x1b[3~", "x"])  # nothing leaks into input


class _VoteBackend:
    """A fake backend whose .chat returns scripted judge outputs, recording the
    temperature it was asked to sample at."""
    name = "vote"

    def __init__(self, raws):
        self.raws = list(raws)
        self.i = 0
        self.temps = []

    def chat(self, messages, schema=None, temperature=0.0):
        self.temps.append(temperature)
        r = self.raws[min(self.i, len(self.raws) - 1)]
        self.i += 1
        return r


class TestVerifierV2(unittest.TestCase):
    """P2.2 — the rebuilt fleet verifier: worktree/rsync isolation with an
    uncommitted overlay, claim-scoped test detection, self-consistency verdict
    voting, and an honest UNKNOWN that never becomes a false REFUTED."""

    # ---- detect_test_cmd: new engines + claim scoping -----------------------
    def test_detect_test_cmd_root_level_test_file(self):
        from forge import fleet
        d = tempfile.mkdtemp()
        self.assertIsNone(fleet.detect_test_cmd(d))                 # nothing yet
        _write(os.path.join(d, "test_thing.py"), "import unittest\n")
        self.assertEqual(fleet.detect_test_cmd(d), "python3 -m unittest")   # bare test_*.py in root
        d2 = tempfile.mkdtemp()
        _write(os.path.join(d2, "thing_test.py"), "import unittest\n")
        self.assertEqual(fleet.detect_test_cmd(d2), "python3 -m unittest")  # *_test.py too

    def test_detect_test_cmd_go_and_package_managers(self):
        from forge import fleet
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "go.mod"), "module x\n")
        self.assertEqual(fleet.detect_test_cmd(d), "go test ./...")
        d2 = tempfile.mkdtemp()
        _write(os.path.join(d2, "package.json"), '{"scripts":{"test":"jest"}}')
        self.assertEqual(fleet.detect_test_cmd(d2), "npm test --silent")   # bare npm
        _write(os.path.join(d2, "yarn.lock"), "")
        self.assertEqual(fleet.detect_test_cmd(d2), "yarn test")           # lockfile picks the PM
        _write(os.path.join(d2, "pnpm-lock.yaml"), "")
        self.assertEqual(fleet.detect_test_cmd(d2), "pnpm test")           # pnpm wins over yarn

    def test_detect_test_cmd_scopes_to_edited_files(self):
        from forge import fleet
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "tests"))
        os.makedirs(os.path.join(d, "pkg"))
        _write(os.path.join(d, "pyproject.toml"), "[project]\nname='x'\n")
        _write(os.path.join(d, "tests", "test_foo.py"), "def test_x():\n    pass\n")
        _write(os.path.join(d, "pkg", "mod.py"), "x = 1\n")
        # files=None → whole suite, exactly as before (backward-compatible)
        self.assertEqual(fleet.detect_test_cmd(d), "pytest -q")
        self.assertEqual(fleet.detect_test_cmd(d, files=None), "pytest -q")
        # an edited test file is run directly
        self.assertEqual(fleet.detect_test_cmd(d, files=[os.path.join(d, "tests", "test_foo.py")]),
                         "pytest -q tests/test_foo.py")
        # a non-test edit maps to its nearest ancestor tests dir
        self.assertEqual(fleet.detect_test_cmd(d, files=[os.path.join(d, "pkg", "mod.py")]),
                         "pytest -q tests")
        # a non-.py edit contributes no scope → whole suite (no false narrowing)
        self.assertEqual(fleet.detect_test_cmd(d, files=[os.path.join(d, "README.md")]),
                         "pytest -q")

    # ---- deterministic verdict + pytest-exit-5 = model path -----------------
    def test_deterministic_verdict_and_pytest_exit5(self):
        from forge import fleet
        self.assertEqual(fleet._deterministic_verdict("npm test --silent", 0, "ok")["verdict"], "CONFIRMED")
        self.assertEqual(fleet._deterministic_verdict("pytest -q", 1, "1 failed")["verdict"], "REFUTED")
        # pytest exit 5 = "no tests collected" → defer to the model path, NOT REFUTED
        self.assertIsNone(fleet._deterministic_verdict("pytest -q tests/test_x.py", 5, "no tests ran"))
        # exit 5 from a non-pytest command is a genuine failure, not "no tests"
        self.assertEqual(fleet._deterministic_verdict("make test", 5, "boom")["verdict"], "REFUTED")

    # ---- self-consistency voting -------------------------------------------
    def test_majority_vote(self):
        from forge import fleet
        self.assertEqual(fleet._majority(["CONFIRMED", "CONFIRMED", "REFUTED"]), "CONFIRMED")
        self.assertEqual(fleet._majority(["REFUTED", "REFUTED", "CONFIRMED"]), "REFUTED")
        self.assertEqual(fleet._majority(["A", "B", "C"]), "UNKNOWN")   # 1-1-1 split
        self.assertEqual(fleet._majority([]), "UNKNOWN")
        self.assertEqual(fleet._majority(["CONFIRMED"]), "UNKNOWN")     # <2 agree

    def test_vote_samples_k3_at_temp_and_takes_mode(self):
        from forge import fleet
        b = _VoteBackend(["VERDICT: REFUTED", "VERDICT: REFUTED", "VERDICT: CONFIRMED"])
        self.assertEqual(fleet._vote(b, "evidence"), "REFUTED")         # mode of 3 wins
        self.assertEqual(b.temps, [0.8, 0.8, 0.8])                      # k=3 @ temp 0.8
        # unparseable samples are dropped; a lone parseable vote is not a majority
        b2 = _VoteBackend(["nonsense", "still nonsense", "VERDICT: CONFIRMED"])
        self.assertEqual(fleet._vote(b2, "e"), "UNKNOWN")

    # ---- isolation: worktree overlay + rsync fallback -----------------------
    @unittest.skipUnless(__import__("shutil").which("git"), "git required")
    def test_isolate_worktree_overlays_uncommitted_edit(self):
        import subprocess
        from forge import fleet
        d = tempfile.mkdtemp()
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
        subprocess.run(["git", "init", "-q"], cwd=d, env=env, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, env=env)
        subprocess.run(["git", "config", "user.name", "t"], cwd=d, env=env)
        _write(os.path.join(d, "tracked.py"), "V = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=d, env=env)
        subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-qm", "init"], cwd=d, env=env, check=True)
        # uncommitted working-tree edits — what the claim is actually about
        _write(os.path.join(d, "tracked.py"), "V = 2  # EDITED uncommitted\n")
        _write(os.path.join(d, "untracked.py"), "NEW = True\n")
        work, cleanup = fleet._isolate(d)
        try:
            self.assertNotEqual(work, d)                                    # isolated copy
            self.assertIn("EDITED", _read(os.path.join(work, "tracked.py")))  # overlay carried the edit
            self.assertTrue(os.path.exists(os.path.join(work, "untracked.py")))  # and the untracked file
            self.assertTrue(os.path.isdir(os.path.join(work, ".git")) or
                            os.path.exists(os.path.join(work, ".git")))      # .git preserved (worktree link)
        finally:
            cleanup()

    def test_isolate_rsync_fallback_includes_dotgit(self):
        from forge import fleet
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, ".git"))
        _write(os.path.join(d, ".git", "config"), "[core]\n")   # fake repo → worktree add fails
        _write(os.path.join(d, "a.py"), "VALUE = 1\n")
        work, cleanup = fleet._isolate(d)
        try:
            self.assertIn("VALUE = 1", _read(os.path.join(work, "a.py")))
            self.assertTrue(os.path.exists(os.path.join(work, ".git", "config")))  # fallback INCLUDES .git
        finally:
            cleanup()

    # ---- verify(): deterministic path + escalate-then-honest-UNKNOWN --------
    def test_verify_deterministic_confirms_and_refutes(self):
        from forge import fleet
        d = tempfile.mkdtemp()                  # non-git → rsync isolation
        _write(os.path.join(d, "x.py"), "x = 1\n")
        orig = fleet.detect_test_cmd
        try:
            fleet.detect_test_cmd = lambda cwd, files=None: "true"
            r = fleet.verify("all tests pass", d, ["m"])
            self.assertEqual(r["verdict"], "CONFIRMED")
            self.assertTrue(r["confirmed"])
            self.assertEqual(r["method"], "deterministic")
            fleet.detect_test_cmd = lambda cwd, files=None: "false"
            r = fleet.verify("all tests pass", d, ["m"], files=None)
            self.assertEqual(r["verdict"], "REFUTED")
            self.assertFalse(r["confirmed"])
        finally:
            fleet.detect_test_cmd = orig

    def test_verify_escalates_on_unknown_then_reports_unknown(self):
        from forge import fleet
        d = tempfile.mkdtemp()
        orig_det, orig_iso, orig_gv = fleet.detect_test_cmd, fleet._isolate, fleet._gather_and_vote
        try:
            fleet.detect_test_cmd = lambda cwd, files=None: None        # force the model path
            fleet._isolate = lambda cwd: (d, lambda: None)              # skip real isolation
            calls = []

            def fake_gv(model, work, claim):
                calls.append(model)
                return ("UNKNOWN" if len(calls) == 1 else "CONFIRMED"), "ev"
            fleet._gather_and_vote = fake_gv
            r = fleet.verify("done", d, ["small", "big"])
            self.assertEqual(calls, ["small", "big"])                   # escalated exactly one rung
            self.assertEqual(r["verdict"], "CONFIRMED")
            # a single-rung ladder that stays UNKNOWN reports UNKNOWN — never REFUTED
            calls.clear()
            fleet._gather_and_vote = lambda m, w, c: (calls.append(m) or ("UNKNOWN", "ev"))
            r = fleet.verify("done", d, ["only"])
            self.assertEqual(calls, ["only"])                          # no escalation available
            self.assertEqual(r["verdict"], "UNKNOWN")
            self.assertFalse(r["confirmed"])
        finally:
            fleet.detect_test_cmd, fleet._isolate, fleet._gather_and_vote = orig_det, orig_iso, orig_gv

    # ---- daemon: order only on REFUTED, UNKNOWN logged distinctly -----------
    def test_daemon_sends_order_only_on_refuted(self):
        from forge import daemon, fleet
        f = daemon.Forged("small,big", interval=1)
        self.assertEqual(f.models, ["small", "big"])                   # comma ladder → models list

        entry = {"sid": "s1", "name": "n", "cwd": "/repo", "status": "idle"}
        recs = tempfile.mkdtemp()
        saved = {"registry": daemon.sessmod.registry, "last_say": fleet.last_say,
                 "hv": fleet.harness_verified, "ef": fleet.edited_files,
                 "verify": fleet.verify, "send": fleet.send, "load": daemon._load,
                 "save": daemon._save, "receipts": fleet.RECEIPTS}

        def run_with(verdict):
            sent = []
            daemon.sessmod.registry = lambda: [entry]
            fleet.last_say = lambda sid: "all tests pass"
            fleet.harness_verified = lambda sid: False
            fleet.edited_files = lambda sid, cwd: set()
            fleet.verify = lambda claim, cwd, models, files=None: {
                "verdict": verdict, "confirmed": verdict == "CONFIRMED", "evidence": "e"}
            fleet.send = lambda *a, **k: sent.append((a, k))
            daemon._load = lambda fn, dflt: {}
            daemon._save = lambda fn, v: None
            fleet.RECEIPTS = os.path.join(recs, verdict + ".jsonl")
            f.verify_pass()
            return sent, _read(fleet.RECEIPTS)
        try:
            for v in ("CONFIRMED", "UNKNOWN"):
                sent, receipt = run_with(v)
                self.assertEqual(sent, [], f"{v} must not trigger a refutation order")
                self.assertIn(v, receipt)                              # verdict logged distinctly
            sent, receipt = run_with("REFUTED")
            self.assertEqual(len(sent), 1)                             # only REFUTED sends the order
            self.assertEqual(sent[0][1].get("sender"), "verifier")
            self.assertIn("REFUTED", receipt)
        finally:
            daemon.sessmod.registry = saved["registry"]; fleet.last_say = saved["last_say"]
            fleet.harness_verified = saved["hv"]; fleet.edited_files = saved["ef"]
            fleet.verify = saved["verify"]; fleet.send = saved["send"]
            daemon._load = saved["load"]; daemon._save = saved["save"]
            fleet.RECEIPTS = saved["receipts"]


class _FakeUI:
    """Records (kind, kwargs) events instead of painting a terminal."""
    def __init__(self): self.events = []
    def __call__(self, kind, **k): self.events.append((kind, k))


class _FakeAgent:
    def __init__(self, mode="auto"):
        self.mode = mode
        self.messages = []


class _WakeSession:
    """Minimal session for _on_wake: a scripted inbox drain, recording log()."""
    def __init__(self, msgs): self._msgs = list(msgs); self.logs = []
    def drain(self):
        m, self._msgs = self._msgs, []
        return m
    def log(self, kind, **k): self.logs.append((kind, k))


class TestWakeOnInbox(unittest.TestCase):
    """P2.3 — the wake pipe (idle sessions render/act on fleet traffic instantly),
    the select-or-wake key read, and the PURE auto-act policy predicate."""

    # ---- Session wake pipe: push writes a byte, drain empties it -------------
    def test_push_writes_wake_byte_and_drain_empties(self):
        import select
        s = sm.Session("wk1", tempfile.mkdtemp(), "m")
        try:
            self.assertIsNotNone(s.wake_fd)
            self.assertEqual(select.select([s.wake_fd], [], [], 0)[0], [])   # idle → not readable
            s.push("verifier", "[verify] fix it")
            self.assertEqual(select.select([s.wake_fd], [], [], 0)[0], [s.wake_fd])  # a byte woke it
            self.assertEqual(s.drain(), [{"from": "verifier", "text": "[verify] fix it"}])
            self.assertEqual(select.select([s.wake_fd], [], [], 0)[0], [])   # drain emptied the pipe too
        finally:
            os.close(s._wake_r); os.close(s._wake_w)

    def test_many_pushes_then_one_drain_clears_the_pipe(self):
        import select
        s = sm.Session("wk2", tempfile.mkdtemp(), "m")
        try:
            for i in range(5):
                s.push("peer", f"m{i}")
            self.assertEqual(select.select([s.wake_fd], [], [], 0)[0], [s.wake_fd])
            self.assertEqual(len(s.drain()), 5)
            self.assertEqual(select.select([s.wake_fd], [], [], 0)[0], [])   # no residue keeps it hot
        finally:
            os.close(s._wake_r); os.close(s._wake_w)

    # ---- _read_key_or_wake: fake key loop returns WAKE on the wake fd --------
    def test_read_key_or_wake_reads_the_key(self):
        from forge.tui import _read_key_or_wake
        kr, kw = os.pipe(); wr, ww = os.pipe()
        try:
            os.write(kw, b"a")
            self.assertEqual(_read_key_or_wake(kr, wr), "a")
        finally:
            for fd in (kr, kw, wr, ww): os.close(fd)

    def test_read_key_or_wake_returns_wake_sentinel(self):
        from forge.tui import _read_key_or_wake, WAKE
        kr, kw = os.pipe(); wr, ww = os.pipe()
        try:
            os.write(ww, b"x")
            self.assertIs(_read_key_or_wake(kr, wr), WAKE)
        finally:
            for fd in (kr, kw, wr, ww): os.close(fd)

    def test_read_key_or_wake_prefers_stdin_no_dropped_key(self):
        from forge.tui import _read_key_or_wake
        kr, kw = os.pipe(); wr, ww = os.pipe()
        try:
            os.write(kw, b"z"); os.write(ww, b"x")           # both ready
            self.assertEqual(_read_key_or_wake(kr, wr), "z")  # stdin serviced first
        finally:
            for fd in (kr, kw, wr, ww): os.close(fd)

    def test_read_key_or_wake_none_is_plain_blocking_read(self):
        from forge.tui import _read_key_or_wake
        kr, kw = os.pipe()
        try:
            os.write(kw, b"b")
            self.assertEqual(_read_key_or_wake(kr, None), "b")   # wake_fd None → unchanged
        finally:
            os.close(kr); os.close(kw)

    # ---- wake_should_act: the pure act-vs-render policy ----------------------
    def test_wake_policy_predicate(self):
        from forge.repl import wake_should_act as w, AUTO_ACT_SENDERS
        self.assertEqual(AUTO_ACT_SENDERS, {"verifier", "guard", "learn"})
        # act: auto mode + config 'act' + a system sender, budget available
        self.assertTrue(w("auto", "act", "verifier", "anything", 2))
        self.assertTrue(w("auto", "act", "guard", "x", 1))
        self.assertTrue(w("auto", "act", "learn", "x", 2))
        # act via the message tag, even from an unknown sender
        self.assertTrue(w("auto", "act", "randobox", "[verify] failed", 2))
        self.assertTrue(w("auto", "act", "randobox", "[task done] report", 2))
        self.assertTrue(w("auto", "act", "randobox", "[ask q1] status?", 2))
        # render-only: unknown sender, no matching tag
        self.assertFalse(w("auto", "act", "randobox", "just chatting", 2))
        self.assertFalse(w("auto", "act", "randobox", "[taskdone] no space", 2))
        # config gates: only 'act' auto-acts
        self.assertFalse(w("auto", "render", "verifier", "[verify]", 2))
        self.assertFalse(w("auto", "off", "verifier", "[verify]", 2))
        # mode gates: manual / plan only notify, never act
        self.assertFalse(w("manual", "act", "verifier", "[verify]", 2))
        self.assertFalse(w("plan", "act", "verifier", "[verify]", 2))
        # budget exhaustion stops the autonomous chain
        self.assertFalse(w("auto", "act", "verifier", "[verify]", 0))

    def test_wake_default_config_is_off(self):
        from forge import config
        self.assertEqual(config.DEFAULTS["wake"], "off")     # conservative: opt-in only

    # ---- _on_wake: render vs auto-act, context folding, budget --------------
    def _wake(self, agent, session, budget, wake_cfg):
        from forge import repl, config
        saved = config.get
        config.get = lambda k, d=None: wake_cfg if k == "wake" else saved(k, d)
        try:
            return repl._on_wake(self._ui, agent, session, budget)
        finally:
            config.get = saved

    def test_on_wake_acts_on_actionable_message(self):
        self._ui = _FakeUI()
        agent = _FakeAgent("auto")
        sess = _WakeSession([{"from": "verifier", "text": "[verify] fails"}])
        out = self._wake(agent, sess, 2, "act")
        self.assertEqual(out, "[fleet message from verifier]: [verify] fails")
        self.assertIn(("inbox", {"sender": "verifier", "text": "[verify] fails"}), self._ui.events)
        self.assertEqual(agent.messages, [])                 # actionable msg is the turn's user_text, not pre-appended

    def test_on_wake_render_only_folds_into_context(self):
        self._ui = _FakeUI()
        agent = _FakeAgent("auto")
        sess = _WakeSession([{"from": "otherbox", "text": "hey there"}])
        out = self._wake(agent, sess, 2, "render")
        self.assertIsNone(out)                               # nothing auto-acted under 'render'
        self.assertEqual(len(self._ui.events), 1)            # still rendered instantly
        self.assertEqual(len(agent.messages), 1)             # and kept for the next turn's context
        self.assertIn("[fleet message from otherbox]: hey there", agent.messages[0]["content"])
        self.assertEqual(sess.logs, [("inbox", {"sender": "otherbox", "text": "hey there"})])

    def test_on_wake_budget_zero_renders_only(self):
        self._ui = _FakeUI()
        agent = _FakeAgent("auto")
        sess = _WakeSession([{"from": "verifier", "text": "[verify] fails"}])
        out = self._wake(agent, sess, 0, "act")              # chain budget spent
        self.assertIsNone(out)
        self.assertEqual(len(agent.messages), 1)             # folded into context, not acted

    def test_on_wake_manual_mode_only_notifies(self):
        self._ui = _FakeUI()
        agent = _FakeAgent("manual")
        sess = _WakeSession([{"from": "verifier", "text": "[verify] fails"}])
        out = self._wake(agent, sess, 2, "act")
        self.assertIsNone(out)                               # manual/plan never auto-act
        self.assertEqual(len(self._ui.events), 1)            # but the message is rendered

    def test_on_wake_mixed_batch_acts_and_renders(self):
        self._ui = _FakeUI()
        agent = _FakeAgent("auto")
        sess = _WakeSession([
            {"from": "chatbox", "text": "fyi"},              # render-only
            {"from": "verifier", "text": "[verify] fails"},  # actionable
        ])
        out = self._wake(agent, sess, 2, "act")
        self.assertEqual(out, "[fleet message from verifier]: [verify] fails")
        self.assertEqual(len(self._ui.events), 2)            # both rendered
        self.assertEqual(len(agent.messages), 1)             # only the non-actionable one folded into context
        self.assertIn("chatbox", agent.messages[0]["content"])


class TestStepTrace(unittest.TestCase):
    """P3.1 — the transcript is a complete machine-readable trace: one 'meta'
    record at Agent.__init__, exactly one 'step' record per loop iteration carrying
    schema version + flags, and durable 'compact'/'malformed'/'loop' records for the
    events that used to leave no trace. None of this touches the observation stream."""

    def setUp(self):
        from forge import fleet
        self.fleet = fleet
        self._orig = fleet.detect_test_cmd
        self.fleet.detect_test_cmd = lambda cwd: None      # no suite → say is never gated

    def tearDown(self):
        self.fleet.detect_test_cmd = self._orig

    def test_meta_record_written_once_at_init(self):
        from forge import __version__
        from forge.agent import TRACE_V
        d = tempfile.mkdtemp()
        sess = _RecSession(d)
        ladder = [ScriptBackend([]), ScriptBackend([])]
        Agent(ladder, sess, workspace="hello world")
        metas = [f for k, f in sess.logs if k == "meta"]
        self.assertEqual(len(metas), 1)                     # exactly one header
        m = metas[0]
        self.assertEqual(m["v"], TRACE_V)
        self.assertEqual(m["forge"], __version__)
        self.assertEqual(m["model"], "script")
        self.assertEqual(m["ladder"], ["script", "script"])
        self.assertEqual(m["cwd"], d)
        self.assertEqual(m["mode"], "auto")
        self.assertEqual(m["briefing"],
                         __import__("hashlib").md5(b"hello world").hexdigest()[:12])
        # no workspace → briefing is None
        sess2 = _RecSession(d)
        Agent(ScriptBackend([]), sess2)
        self.assertIsNone([f for k, f in sess2.logs if k == "meta"][0]["briefing"])

    def test_one_step_per_iteration_with_flags(self):
        from forge.agent import TRACE_V
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        actions = [
            "this is not json",                                                    # malformed
            '{"thought":"a","action":"read_file","path":"r.py"}',                   # normal ok
            '{"thought":"b","action":"read_file","path":"r.py"}',                   # normal ok
            '{"thought":"c","action":"read_file","path":"r.py"}',                   # 3x → loop_trip
            '{"thought":"d","action":"say","message":"done"}',                      # ends the turn
        ]
        events = []
        sess = _RecSession(d)
        a = Agent(ScriptBackend(actions), sess, max_steps=8,
                  on_event=lambda k, **kw: events.append((k, kw)))
        result = a.send("go")
        self.assertEqual(result, "done")
        steps = [f for k, f in sess.logs if k == "step"]
        self.assertEqual(len(steps), 5)                    # exactly one per iteration
        self.assertTrue(all(s["v"] == TRACE_V and "elapsed_ms" in s for s in steps))
        # step 1: malformed, no action
        self.assertTrue(steps[0].get("malformed"))
        self.assertIsNone(steps[0].get("action"))
        # step 2: a normal action carries action + ok
        self.assertEqual(steps[1]["action"], "read_file")
        self.assertTrue(steps[1]["ok"])
        # step 4: the loop detector fired
        self.assertTrue(steps[3].get("loop_trip"))
        # the say step is present and flagged as the say action
        self.assertEqual(steps[4]["action"], "say")
        # the three blind events now leave durable records
        self.assertTrue(any(k == "malformed" for k, f in sess.logs))
        self.assertTrue(any(k == "loop" for k, f in sess.logs))
        # REGRESSION GUARD: none of this became an observation-ok event
        malformed_obs = [kw for k, kw in events if k == "observation" and kw.get("ok") is False]
        # (a read never fails here, so there are simply no failed observations)
        self.assertEqual(malformed_obs, [])

    def test_compaction_logs_record_and_flags_step(self):
        from forge.agent import TRACE_V

        class CompactBackend:
            name = "cb"
            last_prompt_tokens = 900                        # 90% of the window → over the 70% threshold
            def effective_ctx(self): return 1000
            def stream(self, messages, schema=None, temperature=0.0):
                yield '{"thought":"x","action":"say","message":"done"}'
            def chat(self, messages, schema=None, temperature=0.0):
                return "COMPACT SUMMARY"                    # _summarize uses the cheapest ladder model

        d = tempfile.mkdtemp()
        sess = _RecSession(d)
        a = Agent(CompactBackend(), sess, max_steps=3)
        for i in range(20):                                 # seed enough middle turns that len(middle) >= 4
            a.messages.append({"role": "user" if i % 2 else "assistant", "content": f"m{i}"})
        result = a.send("go")
        self.assertEqual(result, "done")
        compacts = [f for k, f in sess.logs if k == "compact"]
        self.assertEqual(len(compacts), 1)
        self.assertEqual(compacts[0]["v"], TRACE_V)
        self.assertEqual(compacts[0]["summary"], "COMPACT SUMMARY")
        self.assertEqual(compacts[0]["window"], 1000)
        steps = [f for k, f in sess.logs if k == "step"]
        self.assertTrue(steps[0].get("compacted"))         # the iteration that compacted is flagged
        self.assertFalse(a._compacted)                      # transient flag was cleared


class _FillBackend:
    """A fake backend with a controllable window + prompt-token count (like the
    TestObservationShaping fakes) for exercising the compaction triggers offline."""
    name = "fill"

    def __init__(self, window=8192, tokens=0):
        self._w = window
        self.last_prompt_tokens = tokens

    def effective_ctx(self):
        return self._w

    def stream(self, messages, schema=None, temperature=0.0):
        yield '{"thought":"x","action":"say","message":"done"}'

    def chat(self, messages, schema=None, temperature=0.0):
        return "SUMMARY"


class TestStructuralCompaction(unittest.TestCase):
    """P4.2 — the zero-model-call deterministic pass that reclaims mechanically
    redundant window BEFORE the LLM summarizer, plus the hard floor that escapes a
    full-window wedge, and the post-pass fill recompute so the 70% gate sees the
    reclaimed space instead of the stale token count."""

    def _agent(self, d, window=4000, tokens=3000):
        # tokens/window put fill above the 0.55 structural trigger by default
        return Agent(_FillBackend(window=window, tokens=tokens), _RecSession(d))

    def _pad(self, a, n=10):
        """Fill the middle so the target message is older than the recent-8 window."""
        for i in range(n):
            a.messages.append({"role": "assistant", "content": f"pad{i}"})
            a.meta.append({"kind": "msg"})

    def test_superseded_read_is_stubbed_live_copy_kept(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "f.py")
        _write(p, "a = 1\n" * 50)
        rp = os.path.realpath(p)
        a = self._agent(d)
        old = {"role": "user", "content": "Observation:\n" + "a = 1\n" * 50}
        a.messages.append(old)
        a.ledger.record_read(rp, 1)
        a.ledger.set_obs_msg(rp, old)
        a.meta.append({"kind": "obs", "action": "read_file", "path": rp, "step": 1})
        self._pad(a)
        # a later re-read/edit rebinds the ledger's live copy to a NEW observation
        new = {"role": "user", "content": "Observation:\n(fresh copy)"}
        a.messages.append(new)
        a.ledger.set_obs_msg(rp, new)
        a.meta.append({"kind": "obs", "action": "read_file", "path": rp, "step": 12})
        a._structural_compact(step=13)
        self.assertIn("superseded", old["content"])          # older read stubbed
        self.assertNotIn("a = 1\na = 1", old["content"])      # its full body is gone
        self.assertIn("(fresh copy)", new["content"])         # the live copy is untouched
        self.assertTrue(a._reclaimed)

    def test_live_read_is_never_stubbed(self):
        # the correctness guard: the bound (in-context) read must survive, else the
        # model passes read-before-edit on content it no longer holds.
        d = tempfile.mkdtemp()
        p = os.path.join(d, "f.py")
        _write(p, "a = 1\n" * 50)
        rp = os.path.realpath(p)
        a = self._agent(d)
        obs = {"role": "user", "content": "Observation:\n" + "a = 1\n" * 50}
        a.messages.append(obs)
        a.ledger.record_read(rp, 1)
        a.ledger.set_obs_msg(rp, obs)          # this IS the live binding
        a.meta.append({"kind": "obs", "action": "read_file", "path": rp, "step": 1})
        self._pad(a)
        a._structural_compact(step=13)
        self.assertIn("a = 1\na = 1", obs["content"])   # untouched — still the live copy
        self.assertFalse(a._reclaimed)

    def test_write_echo_collapses_to_path_bytes_sha1(self):
        d = tempfile.mkdtemp()
        a = self._agent(d)
        big = "x = 1\n" * 500
        echo = {"role": "assistant",
                "content": json.dumps({"thought": "w", "action": "write_file", "path": "f.py", "content": big})}
        a.messages.append(echo)
        a.meta.append({"kind": "write_echo", "action": "write_file", "path": os.path.join(d, "f.py"), "step": 1})
        self._pad(a)
        a._structural_compact(step=12)
        self.assertIn("[elided", echo["content"])
        self.assertNotIn(big, echo["content"])
        obj = json.loads(echo["content"])                # shape preserved → valid JSON
        self.assertEqual(obj["action"], "write_file")
        self.assertEqual(obj["path"], "f.py")
        self.assertIn("sha1", obj["content"])            # path+bytes+sha1 stub
        self.assertIn(str(len(big.encode())), obj["content"])
        self.assertTrue(a._reclaimed)

    def test_stale_failed_obs_shrinks_recent_one_survives(self):
        d = tempfile.mkdtemp()
        a = self._agent(d)
        err = "Traceback (most recent call last):\n" + "  frame line\n" * 100
        stale = {"role": "user",
                 "content": "  ⚠ this action FAILED — diagnose the cause before retrying.\nObservation:\n" + err}
        a.messages.append(stale)
        a.meta.append({"kind": "obs", "action": "bash", "path": None, "step": 1, "ok": False})
        recent_fail = {"role": "user",
                       "content": "  ⚠ this action FAILED — diagnose the cause before retrying.\nObservation:\n" + err}
        a.messages.append(recent_fail)
        a.meta.append({"kind": "obs", "action": "bash", "path": None, "step": 11, "ok": False})
        self._pad(a)
        a._structural_compact(step=13)              # step-1 fail is 12 old (>3); step-11 is 2 old (<=3)
        self.assertIn("first error line kept", stale["content"])
        self.assertIn("Traceback (most recent call last):", stale["content"])   # first line kept
        self.assertNotIn("frame line\n  frame line", stale["content"])          # tail elided
        self.assertNotIn("first error line kept", recent_fail["content"])       # too recent → untouched
        self.assertTrue(a._reclaimed)

    def test_wedge_above_window_is_escaped_by_floor(self):
        # head + tail alone exceed the window: _compact's len(middle) < 4 returns
        # silently and the session is permanently wedged. The floor is the escape.
        d = tempfile.mkdtemp()
        a = Agent(_FillBackend(window=8192, tokens=7000), _RecSession(d))
        for i in range(12):
            a.messages.append({"role": "user", "content": "Observation:\n" + "z" * 2500})
            a.meta.append({"kind": "obs", "action": "bash", "step": i, "ok": True})
        ceiling = 0.70 * 8192

        def used():
            return sum(len(m["content"]) for m in a.messages) // 4

        self.assertGreater(used(), ceiling)          # genuinely wedged
        a._compact()                                 # middle is empty → silent return, no relief
        self.assertGreater(used(), ceiling)
        a._floor()                                   # the hard floor truncates the oldest tail obs
        self.assertLessEqual(used(), ceiling)        # escaped
        self.assertTrue(a._reclaimed)
        # the head (system prompt) is NEVER truncated
        self.assertNotIn("hard-truncated", a.messages[0]["content"])

    def test_fill_recomputed_after_structural_pass(self):
        # judge correction: before the reclaimed flag, _fill returned the STALE
        # last_prompt_tokens, so reclaimed space was invisible until the next model
        # call and the 70% gate misfired. Under the P4.3 ledger, _fill is the sum of
        # per-message estimates (len//4 * tok_ratio); last_prompt_tokens is only a
        # cross-check floor, and the reclaimed flag drops it so the fresh ledger shows.
        d = tempfile.mkdtemp()
        a = self._agent(d, window=4000, tokens=3900)      # ~97% by the observed floor
        big = "x = 1\n" * 600
        echo = {"role": "assistant",
                "content": json.dumps({"action": "write_file", "path": "f.py", "content": big})}
        a.messages.append(echo)
        a.meta.append({"kind": "write_echo", "action": "write_file", "path": os.path.join(d, "f.py"), "step": 1})
        self._pad(a)
        self.assertEqual(a._fill()[0], 3900)              # BEFORE: the observed count is the floor
        a._structural_compact(step=12)
        fresh = sum(len(m["content"]) // 4 for m in a.messages)   # per-message ledger, tok_ratio=1.0
        self.assertEqual(a._fill()[0], fresh)             # AFTER: fresh ledger (reclaimed flag drops the stale floor)
        self.assertLess(a._fill()[0], 3900)               # reclaimed space is now visible

    def test_below_trigger_is_a_noop(self):
        d = tempfile.mkdtemp()
        a = self._agent(d, window=8000, tokens=100)       # fill well under the 0.55 trigger
        big = "x = 1\n" * 500
        echo = {"role": "assistant",
                "content": json.dumps({"action": "write_file", "path": "f.py", "content": big})}
        a.messages.append(echo)
        a.meta.append({"kind": "write_echo", "action": "write_file", "path": os.path.join(d, "f.py"), "step": 1})
        self._pad(a)
        before = echo["content"]
        a._structural_compact(step=12)
        self.assertEqual(echo["content"], before)         # nothing reclaimed below the trigger
        self.assertNotIn("[elided", echo["content"])
        self.assertFalse(a._reclaimed)

    def test_meta_stays_aligned_and_no_meta_key_leaks(self):
        # drive a REAL compaction rewrite and assert the parallel meta list stays
        # index-aligned and that no `_meta` key ever rides inside a sent message.
        class CB:
            name = "cb"
            last_prompt_tokens = 900
            def effective_ctx(self): return 1000
            def stream(self, m, schema=None, temperature=0.0):
                yield '{"thought":"x","action":"say","message":"done"}'
            def chat(self, m, schema=None, temperature=0.0): return "S"

        d = tempfile.mkdtemp()
        a = Agent(CB(), _RecSession(d), max_steps=3)
        for i in range(20):
            a.messages.append({"role": "user" if i % 2 else "assistant", "content": f"m{i}"})
        result = a.send("go")
        self.assertEqual(result, "done")
        self.assertEqual(len(a.meta), len(a.messages))                       # index-aligned across the rewrite
        self.assertTrue(all("_meta" not in m for m in a.messages))           # no _meta key ever sent
        self.assertTrue(all(set(m.keys()) <= {"role", "content"} for m in a.messages))

    def test_meta_tagged_at_funnel_points_over_a_real_turn(self):
        # a real read_file + write_file turn tags the ledger's obs and the write echo
        # in the parallel meta list, aligned with self.messages.
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        actions = [
            '{"thought":"r","action":"read_file","path":"r.py"}',
            '{"thought":"w","action":"write_file","path":"n.py","content":"y = 2\\n"}',
            '{"thought":"d","action":"say","message":"done"}',
        ]
        from forge import fleet
        orig = fleet.detect_test_cmd
        fleet.detect_test_cmd = lambda cwd: None            # no done-gate suite
        try:
            a = Agent(ScriptBackend(actions), _RecSession(d), max_steps=6)
            a.send("go")
        finally:
            fleet.detect_test_cmd = orig
        self.assertEqual(len(a.meta), len(a.messages))      # aligned the whole turn
        kinds = [r.get("kind") for r in a.meta]
        self.assertIn("write_echo", kinds)                  # the write echo was tagged
        self.assertTrue(any(r.get("kind") == "obs" and r.get("action") == "read_file" for r in a.meta))


class TestTokenLedger(unittest.TestCase):
    """P4.3 — the harness token ledger fixes WHAT the compaction gate measures
    (a warm KV-prefix cache makes Ollama's prompt_eval_count report only the newly
    evaluated suffix, collapsing fill toward zero) and maybe_compact fixes WHEN it
    fires (proactive 0.55 at the turn boundary; the in-turn gate stays 0.70)."""

    def _agent(self, d, window=8000, tokens=0):
        return Agent(_FillBackend(window=window, tokens=tokens), _RecSession(d))

    def test_fill_is_sum_of_msg_tokens(self):
        d = tempfile.mkdtemp()
        a = self._agent(d)
        a.messages.append({"role": "user", "content": "a" * 4000})
        expected = sum(len(m["content"]) // 4 for m in a.messages)   # tok_ratio=1.0, no plan
        self.assertEqual(a._fill()[0], expected)
        self.assertEqual(len(a.msg_tokens), len(a.messages))          # index-aligned per-message ledger

    def test_msg_tokens_tracks_appends_and_the_plan_pin(self):
        d = tempfile.mkdtemp()
        a = self._agent(d)
        base = a._fill()[0]                                           # system prompt only
        a.messages.append({"role": "user", "content": "z" * 400})     # +100 tokens
        self.assertEqual(a._fill()[0], base + 100)
        self.assertEqual(len(a.msg_tokens), len(a.messages))
        # the per-step plan pin is appended OUTSIDE self.messages but MUST be counted
        a.plan = ["find the bug", "write the failing test", "fix it"]
        pin_content = "[current plan]\n" + "\n".join(a.plan)
        self.assertEqual(a._fill()[0], base + 100 + len(pin_content) // 4)

    def test_warm_cache_suffix_does_not_collapse_fill(self):
        # the correctness bug: with keep_alive warm, prompt_eval_count reports only
        # the newly-evaluated SUFFIX, so the observed count collapses toward zero and
        # the 0.70 gate never fires. The ledger keeps fill honest.
        d = tempfile.mkdtemp()
        a = self._agent(d)
        for _ in range(10):
            a.messages.append({"role": "user", "content": "Observation:\n" + "y" * 2000})
        ledger = sum(len(m["content"]) // 4 for m in a.messages)
        self.assertGreater(ledger, 3000)
        a.backend.last_prompt_tokens = 40                            # warm-cache suffix, near-zero
        self.assertEqual(a._fill()[0], ledger)                       # ledger wins, not the 40-token suffix
        self.assertGreater(a._fill()[0], 40)

    def test_observed_count_is_an_upward_floor_only(self):
        # last_prompt_tokens is a cross-check floor: it can raise fill (a fuller,
        # uncached count) but never shrink it below the ledger.
        d = tempfile.mkdtemp()
        a = self._agent(d)
        a.messages.append({"role": "user", "content": "q" * 400})
        ledger = sum(len(m["content"]) // 4 for m in a.messages)
        a.backend.last_prompt_tokens = ledger + 5000                 # a bigger uncached count
        self.assertEqual(a._fill()[0], ledger + 5000)                # floor raises fill

    def test_calibration_rebases_ratio_on_uncached_call(self):
        d = tempfile.mkdtemp()
        a = self._agent(d)
        self.assertTrue(a._calibrate_pending)                        # armed at construction
        self.assertEqual(a.tok_ratio, 1.0)
        prompt = [{"role": "user", "content": "z" * 400}]            # 100 char-tokens
        a.backend.last_prompt_tokens = 150                           # true full-prompt count
        a._recalibrate(prompt)
        self.assertAlmostEqual(a.tok_ratio, 1.5)                     # 150 / 100
        self.assertFalse(a._calibrate_pending)                       # consumed

    def test_calibration_cross_check_is_up_only(self):
        # after the first calibration, a warm-cache (suffix-only) count must NOT
        # shrink the ratio; only a bigger observed ratio raises it.
        d = tempfile.mkdtemp()
        a = self._agent(d)
        a.tok_ratio = 1.5
        a._calibrate_pending = False
        prompt = [{"role": "user", "content": "z" * 400}]            # 100 char-tokens
        a.backend.last_prompt_tokens = 30                            # suffix only → ratio 0.3
        a._recalibrate(prompt)
        self.assertEqual(a.tok_ratio, 1.5)                           # NOT shrunk
        a.backend.last_prompt_tokens = 300                           # fuller → ratio 3.0
        a._recalibrate(prompt)
        self.assertAlmostEqual(a.tok_ratio, 3.0)                     # raised

    def test_compaction_rewrite_rearms_calibration(self):
        # a compaction rewrite breaks the KV prefix → the next generate is uncached,
        # so _compact re-arms calibration and a fresh full-prompt count rebases.
        d = tempfile.mkdtemp()
        a = Agent(_FillBackend(window=1000, tokens=900), _RecSession(d))
        for i in range(20):
            a.messages.append({"role": "user" if i % 2 else "assistant", "content": f"m{i}"})
        a._calibrate_pending = False
        a._compact()                                                 # system prompt alone > 70% → compacts
        self.assertTrue(any(k == "compact" for k, _ in a.session.logs))
        self.assertTrue(a._calibrate_pending)                        # re-armed by the rewrite
        prompt = [{"role": "user", "content": "q" * 800}]            # 200 char-tokens
        a.backend.last_prompt_tokens = 400
        a._recalibrate(prompt)
        self.assertAlmostEqual(a.tok_ratio, 2.0)                     # 400 / 200

    def test_maybe_compact_triggers_at_55_but_in_turn_gate_stays_70(self):
        d = tempfile.mkdtemp()
        a = Agent(_FillBackend(window=8000, tokens=0), _RecSession(d))
        for i in range(20):                                          # enough middle for len(middle) >= 4
            a.messages.append({"role": "user", "content": "Observation:\n" + "y" * 800})
            a.meta.append({"kind": "obs", "action": "bash", "step": i, "ok": True})
        used, window = a._fill()
        self.assertGreaterEqual(used, 0.55 * window)                 # over 55%
        self.assertLess(used, 0.70 * window)                         # but under 70%
        before = len(a.messages)
        a._compact()                                                 # in-turn gate is 0.70 → no-op
        self.assertEqual(len(a.messages), before)
        self.assertFalse(any(k == "compact" for k, _ in a.session.logs))
        a.maybe_compact(0.55)                                        # turn-end gate is 0.55 → compacts
        self.assertLess(len(a.messages), before)
        self.assertTrue(any(k == "compact" for k, _ in a.session.logs))
        self.assertTrue(any("summarized" in m["content"] for m in a.messages))

    def test_maybe_compact_below_threshold_is_a_noop(self):
        d = tempfile.mkdtemp()
        a = Agent(_FillBackend(window=8000, tokens=0), _RecSession(d))
        a.messages.append({"role": "user", "content": "small"})
        before = list(a.messages)
        a.maybe_compact(0.55)
        self.assertEqual(a.messages, before)                        # nothing to reclaim below threshold
        self.assertFalse(any(k == "compact" for k, _ in a.session.logs))

    def test_maybe_compact_off_when_lever_disabled(self):
        d = tempfile.mkdtemp()
        a = Agent(_FillBackend(window=8000, tokens=0), _RecSession(d), levers=frozenset())
        for i in range(20):
            a.messages.append({"role": "user", "content": "Observation:\n" + "y" * 800})
            a.meta.append({"kind": "obs", "action": "bash", "step": i, "ok": True})
        before = len(a.messages)
        a.maybe_compact(0.55)                                        # compaction lever off → no-op
        self.assertEqual(len(a.messages), before)

    def test_prefix_audit_off_by_default_and_warns_under_debug(self):
        d = tempfile.mkdtemp()
        a = self._agent(d)
        a._audit_prefix(step=1)                                     # FORGE_DEBUG unset → total no-op
        self.assertIsNone(a._prefix_hash)
        self.assertFalse(any(k == "prefix_mutation" for k, _ in a.session.logs))
        old = os.environ.get("FORGE_DEBUG")
        os.environ["FORGE_DEBUG"] = "1"
        try:
            a._audit_prefix(step=1)                                 # records the baseline head hash
            self.assertIsNotNone(a._prefix_hash)
            a.messages.append({"role": "user", "content": "appended after the head"})
            a._audit_prefix(step=2)                                 # head unchanged → no warning
            self.assertFalse(any(k == "prefix_mutation" for k, _ in a.session.logs))
            a.messages[0] = {"role": "system", "content": "MUTATED HEAD"}
            a._audit_prefix(step=3)                                 # head byte-changed → warn
            self.assertTrue(any(k == "prefix_mutation" for k, _ in a.session.logs))
        finally:
            if old is None:
                os.environ.pop("FORGE_DEBUG", None)
            else:
                os.environ["FORGE_DEBUG"] = old


class TestTraceCmd(unittest.TestCase):
    """P3.1 — `forge trace <sid|last>` renders the meta header + a step table."""

    def test_cmd_trace_pretty_prints_a_session(self):
        import io
        import types
        import contextlib
        from forge.__main__ import cmd_trace
        d = tempfile.mkdtemp()
        orig = sm.SESSIONS
        sm.SESSIONS = d
        try:
            sid = "tracetest"
            recs = [
                {"ts": 1, "type": "meta", "v": 1, "forge": "9.9.9", "model": "m",
                 "ladder": ["m", "n"], "cwd": "/x", "mode": "auto", "briefing": None},
                {"ts": 2, "type": "step", "v": 1, "step": 1, "tier": 0,
                 "action": "read_file", "ok": True, "used": 500, "window": 1000, "elapsed_ms": 12},
                {"ts": 3, "type": "step", "v": 1, "step": 2, "tier": 0,
                 "malformed": True, "used": 510, "window": 1000, "elapsed_ms": 8},
                {"ts": 4, "type": "step", "v": 1, "step": 3, "tier": 1, "action": "say",
                 "ok": None, "loop_trip": True, "used": 520, "window": 1000, "elapsed_ms": 5},
            ]
            with open(os.path.join(d, sid + ".jsonl"), "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cmd_trace(types.SimpleNamespace(sid="last"))   # default = newest file
                cmd_trace(types.SimpleNamespace(sid=sid))      # explicit sid
            out = buf.getvalue()
            self.assertIn("forge 9.9.9", out)                  # meta header
            self.assertIn("read_file", out)                    # a step row
            self.assertIn("50%", out)                          # fill 500/1000
            self.assertIn("malformed", out)                    # flags rendered
            self.assertIn("loop_trip", out)
        finally:
            sm.SESSIONS = orig


class TestLedgerUnit(unittest.TestCase):
    """P4.1 — the file-state Ledger in isolation: stat-tracked staleness, the
    diff-since-last-read baseline, explicit mutation marking, and the RAM caps
    that drop cached content but keep metadata."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def _f(self, name, body):
        p = os.path.join(self.d, name)
        _write(p, body)
        return os.path.realpath(p)

    def test_current_and_change_detection(self):
        led = Ledger()
        fp = self._f("a.py", "x = 1\n")
        led.record_read(fp, 1)
        self.assertTrue(led.current(fp))
        self.assertEqual(led.status(fp), "current")
        # change the file on disk (size differs) → stale
        _write(fp, "x = 999\n")
        led.refresh()
        self.assertFalse(led.current(fp))
        self.assertEqual(led.status(fp), "changed")

    def test_mark_mutated_poisons_until_reread(self):
        led = Ledger()
        fp = self._f("b.py", "a\n")
        led.record_read(fp, 1)
        self.assertTrue(led.current(fp))
        led.mark_mutated(fp)                      # harness knows it changed, even if stat wouldn't show it
        self.assertFalse(led.current(fp))
        self.assertEqual(led.status(fp), "changed")
        led.record_read(fp, 2)                    # a fresh read clears it
        self.assertTrue(led.current(fp))

    def test_diff_since_last_read(self):
        led = Ledger()
        fp = self._f("c.txt", "line1\nline2\n")
        led.record_read(fp, 1)
        _write(fp, "line1\nCHANGED\n")
        d = led.diff(fp)
        self.assertIn("-line2", d)
        self.assertIn("+CHANGED", d)
        # a re-record rebaselines; an identical-content touch diffs to empty string
        led.record_read(fp, 2)
        self.assertEqual(led.diff(fp), "")
        # no cached baseline → None (can't diff)
        led.evict(fp)
        self.assertIsNone(led.diff(fp))

    def test_ram_caps_drop_content_keep_metadata(self):
        led = Ledger()
        orig_total, orig_content = ledger_mod.TOTAL_CAP, ledger_mod.CONTENT_CAP
        ledger_mod.TOTAL_CAP, ledger_mod.CONTENT_CAP = 20, 100
        try:
            f1 = self._f("f1", "a" * 10)
            f2 = self._f("f2", "b" * 10)
            f3 = self._f("f3", "c" * 10)
            led.record_read(f1, 1)
            led.record_read(f2, 2)
            led.record_read(f3, 3)                # total 30 > 20 → LRU-drop oldest content
            self.assertIsNone(led.get(f1).content)        # content dropped (LRU)
            self.assertIsNotNone(led.get(f1).sha1)        # metadata kept
            self.assertTrue(led.get(f1).in_context)       # still counts as held
            self.assertTrue(led.current(f1))              # gate still passes (unchanged on disk)
            self.assertIsNotNone(led.get(f3).content)     # most-recent content retained
            # a single file over CONTENT_CAP is never cached, but is still tracked
            big = self._f("big", "z" * 200)
            led.record_read(big, 4)
            self.assertIsNone(led.get(big).content)
            self.assertIsNotNone(led.get(big).sha1)
            self.assertTrue(led.current(big))
        finally:
            ledger_mod.TOTAL_CAP, ledger_mod.CONTENT_CAP = orig_total, orig_content

    def test_partial_read_spans_not_whole(self):
        led = Ledger()
        fp = self._f("p.py", "".join(f"line{i:02d}\n" for i in range(1, 31)))
        led.record_read(fp, 1, offset=1, limit=3)         # only lines 1-3
        e = led.get(fp)
        self.assertFalse(e.whole)
        self.assertEqual(e.spans, [(1, 3)])
        self.assertTrue(led.covers(fp, 1, 3))
        self.assertFalse(led.covers(fp, 18, 5))           # a region it never read
        led.record_read(fp, 2, offset=18, limit=5)        # merge the new window
        self.assertTrue(led.covers(fp, 18, 5))
        # a whole-file read subsumes everything
        led.record_read(fp, 3)
        self.assertTrue(led.get(fp).whole)
        self.assertTrue(led.covers(fp, 25, 100))


class TestFileStateLedger(unittest.TestCase):
    """P4.1 — the ledger wired through Agent.send: honest read-before-edit that
    re-arms on disk change and on compaction eviction, served-from-cache repeat
    reads, and diff-on-reread of a changed file."""

    def _drive(self, d, actions, max_steps=10, agent=None):
        events = []
        a = agent or Agent(ScriptBackend(actions), sm.EphemeralSession(d, "s"),
                           max_steps=max_steps, on_event=lambda k, **kw: events.append((k, kw)))
        a.send("go")
        obs = [(kw.get("text", ""), kw.get("ok")) for k, kw in events if k == "observation"]
        return a, events, obs

    def test_edit_reblocked_after_ondisk_change(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "r.py")
        _write(fp, "x = 1\n")
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"mutate","action":"bash","command":"echo \\"x = 999\\" > r.py"}',
            '{"thought":"blind","action":"edit_file","path":"r.py","old":"x = 999","new":"x = 2"}',
            '{"thought":"reread","action":"read_file","path":"r.py"}',
            '{"thought":"edit","action":"edit_file","path":"r.py","old":"x = 999","new":"x = 2"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(d, actions)
        # step3: the edit is blocked because the file changed on disk since the read
        self.assertFalse(obs[2][1])
        self.assertIn("CHANGED on disk", obs[2][0])
        self.assertIn("step 1", obs[2][0])
        # step4: the re-read is answered with a DIFF, not the whole file
        self.assertTrue(obs[3][1])
        self.assertIn("CHANGED since you read it", obs[3][0])
        self.assertIn("+x = 999", obs[3][0])
        self.assertIn("-x = 1", obs[3][0])
        # step5: the edit now goes through
        self.assertTrue(obs[4][1])
        self.assertEqual(_read(fp), "x = 2\n")

    def test_repeat_read_served_from_cache(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\nbody line\n")
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"reread","action":"read_file","path":"r.py"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(d, actions)
        self.assertIn("body line", obs[0][0])              # first read shows the body
        self.assertTrue(obs[1][1])                          # served, ok
        self.assertIn("already in your context", obs[1][0])
        self.assertNotIn("body line", obs[1][0])            # the body is NOT re-injected

    def test_repeated_reads_still_trip_loop_detector(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        actions = [
            '{"thought":"1","action":"read_file","path":"r.py"}',
            '{"thought":"2","action":"read_file","path":"r.py"}',
            '{"thought":"3","action":"read_file","path":"r.py"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(d, actions)
        self.assertTrue(any(k == "loop" for k, kw in events))   # 3 identical reads still trip the loop

    def test_write_content_cached_then_diffed(self):
        d = tempfile.mkdtemp()
        actions = [
            '{"thought":"write","action":"write_file","path":"w.py","content":"a\\nb\\n"}',
            '{"thought":"mutate","action":"bash","command":"echo a > w.py"}',
            '{"thought":"reread","action":"read_file","path":"w.py"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(d, actions)
        # the read after the external change diffs against the WRITTEN content baseline
        self.assertTrue(obs[2][1])
        self.assertIn("CHANGED since you read it", obs[2][0])
        self.assertIn("-b", obs[2][0])

    def test_gate_reforces_read_after_eviction(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "r.py")
        _write(fp, "x = 1\n")
        real = os.path.realpath(fp)
        actions = [
            '{"thought":"blind","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"edit","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(d, "s"), max_steps=8)
        # simulate a read earlier in the session that compaction later dropped
        a.ledger.record_read(real, 0)
        a.ledger.evict(real)
        self.assertEqual(a.ledger.status(real), "evicted")
        events = []
        a.on_event = lambda k, **kw: events.append((k, kw))
        a.send("go")
        obs = [(kw.get("text", ""), kw.get("ok")) for k, kw in events if k == "observation"]
        self.assertFalse(obs[0][1])                         # edit blocked — read fell out of context
        self.assertIn("no longer in your context", obs[0][0])
        self.assertTrue(obs[2][1])                          # after re-read, the edit lands
        self.assertEqual(_read(fp), "x = 2\n")

    def test_compaction_evicts_read_observation(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "r.py")
        _write(fp, "x = 1\n")
        real = os.path.realpath(fp)
        a = Agent(ScriptBackend(['{"action":"say","message":"x"}']), sm.EphemeralSession(d, "s"))
        # a read whose observation lives in the message log
        obs_msg = {"role": "user", "content": "Observation:\nx = 1"}
        a.messages.append(obs_msg)
        a.ledger.record_read(real, 1)
        a.ledger.set_obs_msg(real, obs_msg)
        self.assertTrue(a.ledger.current(real))
        # compaction rewrites messages and drops that observation → eviction flips it
        a.messages = a.messages[:a.head_len]
        a._evict_compacted()
        self.assertFalse(a.ledger.current(real))
        self.assertEqual(a.ledger.status(real), "evicted")
        self.assertIsNone(a.ledger.get(real).content)       # cached content dropped too

    def test_partial_read_region_guard(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "p.py")
        _write(fp, "".join(f"line{i:02d}\n" for i in range(1, 31)))
        actions = [
            '{"thought":"read top","action":"read_file","path":"p.py","offset":1,"limit":3}',
            '{"thought":"edit far","action":"edit_file","path":"p.py","old":"line20","new":"LINE20"}',
            '{"thought":"read region","action":"read_file","path":"p.py","offset":18,"limit":6}',
            '{"thought":"edit","action":"edit_file","path":"p.py","old":"line20","new":"LINE20"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(d, actions)
        self.assertFalse(obs[1][1])                         # editing an unread region is blocked
        self.assertIn("only read PART", obs[1][0])
        self.assertTrue(obs[3][1])                          # after reading the region, the edit lands
        self.assertIn("LINE20", _read(fp))

    def test_plain_read_then_edit_unchanged(self):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "s.py")
        _write(fp, "value = 1\n")
        actions = [
            '{"thought":"read","action":"read_file","path":"s.py"}',
            '{"thought":"edit","action":"edit_file","path":"s.py","old":"value = 1","new":"value = 2"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(d, actions)
        self.assertTrue(obs[0][1])
        self.assertTrue(obs[1][1])                          # read → edit works with no re-read
        self.assertEqual(_read(fp), "value = 2\n")


class TestLineAnchoredEdit(unittest.TestCase):
    """P5.3 — line-numbered reads + the line-anchored edit dialect: number the
    reads, splice by range with a fail-closed anchor, and re-arm staleness after a
    splice (shifted numbers) so an out-of-window edit needs a fresh numbered read."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    # ---- numbered reads --------------------------------------------------------
    def test_read_prefixes_absolute_line_numbers(self):
        _write(os.path.join(self.d, "a.txt"), "alpha\nbeta\ngamma\n")
        body, ok = execute({"action": "read_file", "path": "a.txt"}, self.d)
        self.assertTrue(ok)
        self.assertIn("1\talpha", body)
        self.assertIn("2\tbeta", body)
        self.assertIn("3\tgamma", body)

    def test_read_numbers_honor_offset_limit(self):
        _write(os.path.join(self.d, "n.txt"), "".join(f"line{i}\n" for i in range(1, 21)))
        body, ok = execute({"action": "read_file", "path": "n.txt", "offset": 5, "limit": 3}, self.d)
        self.assertTrue(ok)
        self.assertIn("5\tline5", body)          # absolute number, not 1-based-within-window
        self.assertIn("7\tline7", body)
        self.assertNotIn("4\tline4", body)
        self.assertNotIn("8\tline8", body)

    # ---- anchored edit at the execute() layer (fail-closed) --------------------
    def test_anchored_edit_splices_and_echoes_numbered_window(self):
        _write(os.path.join(self.d, "f.txt"), "one\ntwo\nthree\nfour\nfive\n")
        out, ok = execute({"action": "edit_file", "path": "f.txt", "start_line": 3,
                           "end_line": 3, "anchor": "three", "new": "THREE"}, self.d)
        self.assertTrue(ok)
        self.assertEqual(_read(os.path.join(self.d, "f.txt")), "one\ntwo\nTHREE\nfour\nfive\n")
        self.assertIn("replaced lines 3-3", out)
        self.assertIn("3\tTHREE", out)           # echoed post-edit window, numbered
        self.assertIn("2\ttwo", out)             # ±3 lines of context
        self.assertIn("4\tfour", out)

    def test_anchored_edit_multi_line_range(self):
        _write(os.path.join(self.d, "f.txt"), "a\nb\nc\nd\ne\n")
        out, ok = execute({"action": "edit_file", "path": "f.txt", "start_line": 2,
                           "end_line": 4, "anchor": "b", "new": "X\nY"}, self.d)
        self.assertTrue(ok)
        self.assertEqual(_read(os.path.join(self.d, "f.txt")), "a\nX\nY\ne\n")
        self.assertIn("replaced lines 2-4", out)

    def test_anchor_mismatch_rejected_fail_closed(self):
        _write(os.path.join(self.d, "f.txt"), "one\ntwo\nthree\n")
        out, ok = execute({"action": "edit_file", "path": "f.txt", "start_line": 2,
                           "end_line": 2, "anchor": "WRONG", "new": "X"}, self.d)
        self.assertFalse(ok)
        self.assertIn("anchor mismatch", out)
        self.assertEqual(_read(os.path.join(self.d, "f.txt")), "one\ntwo\nthree\n")   # untouched

    def test_anchor_tolerates_indentation_slip(self):
        _write(os.path.join(self.d, "f.py"), "def f():\n    return 1\n")
        # anchor copied without the leading indentation still lands (whitespace-tolerant)
        out, ok = execute({"action": "edit_file", "path": "f.py", "start_line": 2,
                           "end_line": 2, "anchor": "return 1", "new": "    return 2"}, self.d)
        self.assertTrue(ok)
        self.assertEqual(_read(os.path.join(self.d, "f.py")), "def f():\n    return 2\n")

    def test_missing_anchor_rejected(self):
        _write(os.path.join(self.d, "f.txt"), "one\ntwo\n")
        out, ok = execute({"action": "edit_file", "path": "f.txt", "start_line": 1,
                           "end_line": 1, "new": "X"}, self.d)
        self.assertFalse(ok)
        self.assertIn("anchor", out)
        self.assertEqual(_read(os.path.join(self.d, "f.txt")), "one\ntwo\n")

    def test_out_of_range_line_rejected(self):
        _write(os.path.join(self.d, "f.txt"), "one\ntwo\n")
        out, ok = execute({"action": "edit_file", "path": "f.txt", "start_line": 99,
                           "end_line": 99, "anchor": "x", "new": "X"}, self.d)
        self.assertFalse(ok)
        self.assertIn("out of range", out)

    def test_old_not_found_offers_ready_anchored_template(self):
        # a near-miss `old` (word changed, so exact+fuzzy both fail) should return the
        # closest region AND a ready-to-use line-anchored edit with correct line numbers.
        _write(os.path.join(self.d, "m.py"),
               "def f():\n    return sum(xs) / len(xs)\n")
        out, ok = execute({"action": "edit_file", "path": "m.py",
                           "old": "return total(xs) / len(xs)", "new": "return 0"}, self.d)
        self.assertFalse(ok)
        self.assertIn("CLOSEST region (lines 2-2)", out)
        self.assertIn('"start_line": 2', out)          # anchored template offered
        self.assertIn('"end_line": 2', out)
        self.assertIn('"anchor": "    return sum(xs) / len(xs)"', out)

    def test_anchored_edit_never_turns_valid_file_invalid(self):
        _write(os.path.join(self.d, "g.py"), "x = 1\ny = 2\n")
        out, ok = execute({"action": "edit_file", "path": "g.py", "start_line": 1,
                           "end_line": 1, "anchor": "x = 1", "new": "def bad("}, self.d)
        self.assertFalse(ok)
        self.assertIn("invalid", out)
        self.assertEqual(_read(os.path.join(self.d, "g.py")), "x = 1\ny = 2\n")     # not written

    def test_old_new_dialect_still_works(self):
        _write(os.path.join(self.d, "f.txt"), "keep\ntarget\nkeep\n")
        out, ok = execute({"action": "edit_file", "path": "f.txt", "old": "target", "new": "hit"}, self.d)
        self.assertTrue(ok)
        self.assertEqual(_read(os.path.join(self.d, "f.txt")), "keep\nhit\nkeep\n")

    # ---- dry_run scores the anchored dialect -----------------------------------
    def test_dry_run_anchored_match_scores_1(self):
        _write(os.path.join(self.d, "f.txt"), "one\ntwo\nthree\n")
        s, _ = dry_run({"action": "edit_file", "path": "f.txt", "start_line": 2,
                        "end_line": 2, "anchor": "two", "new": "X"}, self.d)
        self.assertEqual(s, 1.0)

    def test_dry_run_anchored_mismatch_scores_0(self):
        _write(os.path.join(self.d, "f.txt"), "one\ntwo\nthree\n")
        s, why = dry_run({"action": "edit_file", "path": "f.txt", "start_line": 2,
                          "end_line": 2, "anchor": "NOPE", "new": "X"}, self.d)
        self.assertEqual(s, 0.0)
        self.assertIn("mismatch", why)

    # ---- agent-loop staleness (via the P4.1 ledger) ----------------------------
    def _drive(self, d, actions, max_steps=10):
        events = []
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(d, "s"),
                  max_steps=max_steps, on_event=lambda k, **kw: events.append((k, kw)))
        a.send("go")
        obs = [(kw.get("text", ""), kw.get("ok")) for k, kw in events if k == "observation"]
        return a, events, obs

    def test_anchored_edit_blocked_on_stale_file(self):
        fp = os.path.join(self.d, "r.py")
        _write(fp, "x = 1\ny = 2\n")
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"mutate","action":"bash","command":"echo \\"x = 9\\" > r.py"}',
            '{"thought":"blind","action":"edit_file","path":"r.py","start_line":1,"end_line":1,"anchor":"x = 9","new":"x = 3"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(self.d, actions)
        self.assertFalse(obs[2][1])                          # anchored edit blocked — file changed on disk
        self.assertIn("CHANGED on disk", obs[2][0])
        self.assertIn("Re-read lines 1-1", obs[2][0])        # anchored range echoed in the block

    def test_first_read_numbered_and_read_before_edit_still_gates(self):
        fp = os.path.join(self.d, "r.py")
        _write(fp, "alpha = 1\nbeta = 2\n")
        actions = [
            # a blind anchored edit with NO prior read is gated by read-before-edit
            '{"thought":"blind","action":"edit_file","path":"r.py","start_line":1,"end_line":1,"anchor":"alpha = 1","new":"alpha = 9"}',
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"edit","action":"edit_file","path":"r.py","start_line":1,"end_line":1,"anchor":"alpha = 1","new":"alpha = 9"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(self.d, actions)
        self.assertFalse(obs[0][1])                          # blind edit blocked
        self.assertTrue(obs[1][1])                           # the read shows numbered content
        self.assertIn("1\talpha = 1", obs[1][0])
        self.assertTrue(obs[2][1])                           # after the read, the anchored edit lands
        self.assertEqual(_read(fp), "alpha = 9\nbeta = 2\n")

    def test_splice_marks_file_stale_for_out_of_window_edit(self):
        fp = os.path.join(self.d, "r.py")
        _write(fp, "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n")
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"splice","action":"edit_file","path":"r.py","start_line":1,"end_line":1,"anchor":"a = 1","new":"a = 10"}',
            # a second edit WITHOUT a re-read is blocked: the splice shifted numbers, file is stale
            '{"thought":"again","action":"edit_file","path":"r.py","start_line":6,"end_line":6,"anchor":"f = 6","new":"f = 60"}',
            '{"thought":"reread","action":"read_file","path":"r.py"}',
            '{"thought":"edit2","action":"edit_file","path":"r.py","start_line":6,"end_line":6,"anchor":"f = 6","new":"f = 60"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(self.d, actions)
        self.assertTrue(obs[1][1])                           # first splice succeeds
        self.assertFalse(obs[2][1])                          # second edit blocked — file marked stale
        self.assertTrue(obs[3][1])                           # re-read re-injects fresh numbers
        self.assertTrue(obs[4][1])                           # now the edit lands
        self.assertEqual(_read(fp), "a = 10\nb = 2\nc = 3\nd = 4\ne = 5\nf = 60\n")

    # ---- review-fix regressions: end_line region guard, empty anchor, atomic writes --
    def test_anchored_end_line_beyond_read_span_is_blocked(self):
        # a PARTIAL read (lines 1-3), then an anchored edit whose end_line (8) overruns
        # the seen span, must be BLOCKED — not silently delete lines 4-8 (the anchor
        # only guards start_line).
        fp = os.path.join(self.d, "r.py")
        original = "".join(f"line{i}\n" for i in range(1, 11))   # 10 lines
        _write(fp, original)
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py","offset":1,"limit":3}',
            '{"thought":"overrun","action":"edit_file","path":"r.py","start_line":1,"end_line":8,"anchor":"line1","new":"X"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        a, events, obs = self._drive(self.d, actions)
        self.assertFalse(obs[1][1])                          # blocked
        self.assertIn("only read PART", obs[1][0])
        self.assertEqual(_read(fp), original)                # file untouched — no silent deletion

    def test_empty_anchor_only_matches_a_truly_identical_line(self):
        from forge.tools import _anchor_ok
        self.assertTrue(_anchor_ok("", ""))                  # exact blank match is fine
        self.assertFalse(_anchor_ok("", "   "))              # blank anchor must NOT match a whitespace line
        self.assertFalse(_anchor_ok("", "code"))
        self.assertTrue(_anchor_ok("}", "    }"))            # real content stays whitespace-tolerant
        self.assertFalse(_anchor_ok("x=1", "x = 2"))

    def test_write_file_ensures_a_single_final_newline(self):
        execute({"action": "write_file", "path": "a.py", "content": "x = 1"}, self.d)
        self.assertEqual(_read(os.path.join(self.d, "a.py")), "x = 1\n")     # newline added
        execute({"action": "write_file", "path": "b.py", "content": "y = 2\n"}, self.d)
        self.assertEqual(_read(os.path.join(self.d, "b.py")), "y = 2\n")     # not doubled
        execute({"action": "write_file", "path": "empty.txt", "content": ""}, self.d)
        self.assertEqual(_read(os.path.join(self.d, "empty.txt")), "")       # empty stays empty

    def test_writes_are_atomic_and_leave_no_temp_file(self):
        import glob
        _write(os.path.join(self.d, "f.txt"), "one\ntwo\n")
        execute({"action": "write_file", "path": "w.txt", "content": "hello\n"}, self.d)
        execute({"action": "edit_file", "path": "f.txt", "start_line": 1, "end_line": 1,
                 "anchor": "one", "new": "ONE"}, self.d)
        self.assertEqual(_read(os.path.join(self.d, "w.txt")), "hello\n")
        self.assertEqual(_read(os.path.join(self.d, "f.txt")), "ONE\ntwo\n")
        self.assertEqual(glob.glob(os.path.join(self.d, ".forge-tmp-*")), [])   # no leaked temp file

    def test_atomic_write_preserves_an_executable_files_mode(self):
        import stat
        sp = os.path.join(self.d, "s.sh")
        _write(sp, "#!/bin/sh\necho hi\n")
        os.chmod(sp, 0o755)
        execute({"action": "edit_file", "path": "s.sh", "start_line": 2, "end_line": 2,
                 "anchor": "echo hi", "new": "echo bye"}, self.d)
        self.assertEqual(_read(sp), "#!/bin/sh\necho bye\n")
        self.assertTrue(os.stat(sp).st_mode & stat.S_IXUSR)   # +x bit survived the edit

    def test_edit_preserves_crlf_line_endings(self):
        cp = os.path.join(self.d, "c.txt")
        with open(cp, "wb") as f:
            f.write(b"one\r\ntwo\r\nthree\r\n")               # CRLF file
        execute({"action": "edit_file", "path": "c.txt", "start_line": 2, "end_line": 2,
                 "anchor": "two", "new": "TWO"}, self.d)
        with open(cp, "rb") as f:
            self.assertEqual(f.read(), b"one\r\nTWO\r\nthree\r\n")   # endings preserved, not flattened to LF


class TestReadKeyDecode(unittest.TestCase):
    """tui._read_key: an undecodable stdin byte must NOT read as the b'' EOF sentinel
    (which the REPL treats as quit) — it is skipped and the next key is returned."""
    def test_undecodable_byte_is_skipped_not_treated_as_eof(self):
        from forge.tui import _read_key
        r, w = os.pipe()
        try:
            os.write(w, b"\x80a")                            # one bad byte, then 'a'
            self.assertEqual(_read_key(r), "a")              # bad byte skipped, 'a' returned
        finally:
            os.close(r); os.close(w)

    def test_real_eof_still_returns_empty_bytes(self):
        from forge.tui import _read_key
        r, w = os.pipe()
        os.close(w)                                          # writer closed → EOF
        try:
            self.assertEqual(_read_key(r), b"")
        finally:
            os.close(r)


class TestBatchEdit(unittest.TestCase):
    """P6.5 atomic multi-edit: an `edits:[{old,new},…]` array applied validate-first,
    all-or-nothing; and the loop-signature fix so distinct edits to one file don't trip."""
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_batch_applies_all_hunks_at_once(self):
        _write(os.path.join(self.d, "f.py"), "a = 1\nb = 2\nc = 3\n")
        out, ok = execute({"action": "edit_file", "path": "f.py", "edits": [
            {"old": "a = 1", "new": "a = 10"},
            {"old": "c = 3", "new": "c = 30"}]}, self.d)
        self.assertTrue(ok)
        self.assertIn("applied 2 edits", out)
        self.assertEqual(_read(os.path.join(self.d, "f.py")), "a = 10\nb = 2\nc = 30\n")

    def test_batch_is_atomic_on_any_failure(self):
        original = "a = 1\nb = 2\n"
        _write(os.path.join(self.d, "f.py"), original)
        out, ok = execute({"action": "edit_file", "path": "f.py", "edits": [
            {"old": "a = 1", "new": "a = 10"},        # would apply
            {"old": "NOPE", "new": "x"}]}, self.d)     # fails → whole batch aborts
        self.assertFalse(ok)
        self.assertIn("edit 2/2", out)
        self.assertIn("NOTHING was written", out)
        self.assertEqual(_read(os.path.join(self.d, "f.py")), original)   # file untouched

    def test_batch_hunks_apply_against_running_text(self):
        _write(os.path.join(self.d, "f.py"), "x = 1\n")
        out, ok = execute({"action": "edit_file", "path": "f.py", "edits": [
            {"old": "x = 1", "new": "x = 2"},          # then the next hunk sees x = 2
            {"old": "x = 2", "new": "x = 3"}]}, self.d)
        self.assertTrue(ok)
        self.assertEqual(_read(os.path.join(self.d, "f.py")), "x = 3\n")

    def test_batch_refused_if_it_would_break_syntax(self):
        original = "def f():\n    return 1\n"
        _write(os.path.join(self.d, "g.py"), original)
        out, ok = execute({"action": "edit_file", "path": "g.py", "edits": [
            {"old": "return 1", "new": "return ("}]}, self.d)    # invalid python
        self.assertFalse(ok)
        self.assertIn("invalid", out)
        self.assertEqual(_read(os.path.join(self.d, "g.py")), original)

    def test_distinct_edits_to_one_file_do_not_trip_the_loop(self):
        fp = os.path.join(self.d, "r.py")
        _write(fp, "a = 1\nb = 2\nc = 3\n")
        actions = [
            '{"thought":"read","action":"read_file","path":"r.py"}',
            '{"thought":"e1","action":"edit_file","path":"r.py","old":"a = 1","new":"a = 10"}',
            '{"thought":"e2","action":"edit_file","path":"r.py","old":"b = 2","new":"b = 20"}',
            '{"thought":"e3","action":"edit_file","path":"r.py","old":"c = 3","new":"c = 30"}',
            '{"thought":"done","action":"say","message":"done"}',
        ]
        events = []
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(self.d, "s"),
                  max_steps=10, on_event=lambda k, **kw: events.append((k, kw)))
        a.send("edit three lines")
        self.assertFalse(any(k == "loop" for k, _ in events))         # no false loop trip
        self.assertEqual(_read(fp), "a = 10\nb = 20\nc = 30\n")        # all three applied


class TestRepoMap(unittest.TestCase):
    """P4.4 — ranked symbol-aware repo map, token-budgeted briefing, outline reads."""

    _GENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}

    def _git_repo(self):
        d = tempfile.mkdtemp()
        subprocess.run(["git", "init", "-q"], cwd=d, env=self._GENV, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, env=self._GENV)
        subprocess.run(["git", "config", "user.name", "t"], cwd=d, env=self._GENV)
        return d

    def _commit(self, d, msg):
        subprocess.run(["git", "add", "-A"], cwd=d, env=self._GENV, check=True)
        subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-qm", msg],
                       cwd=d, env=self._GENV, check=True)

    # ---- 1. outline read mode ------------------------------------------------
    def test_outline_returns_signatures_with_linenos(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "m.py"),
               "import os\n\n\ndef alpha(a, b=2) -> int:\n    return a\n\n\n"
               "class Widget(Base):\n    def run(self, fast=False):\n        return 1\n")
        out, ok = execute({"action": "read_file", "path": "m.py", "outline": True}, d)
        self.assertTrue(ok)
        self.assertIn("def alpha(a, b=2) -> int", out)      # signature reconstructed
        self.assertIn("class Widget(Base)", out)
        self.assertIn("def run(self, fast=False)", out)     # a method too
        self.assertIn("4", out)                              # alpha's line number
        self.assertIn("OUTLINE", out)
        # and it is NOT the raw body
        self.assertNotIn("return a", out)

    def test_outline_skips_syntax_error_file_gracefully(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "bad.py"), "def broken(:\n    pass\n")
        out, ok = execute({"action": "read_file", "path": "bad.py", "outline": True}, d)
        self.assertTrue(ok)                                  # no crash, still a usable observation
        self.assertIn("no symbols", out)

    def test_outline_unsupported_language(self):
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "notes.txt"), "hello\nworld\n")
        out, ok = execute({"action": "read_file", "path": "notes.txt", "outline": True}, d)
        self.assertTrue(ok)
        self.assertIn("no symbols", out)

    def test_action_schema_and_tool_help_document_outline(self):
        from forge.tools import ACTION_SCHEMA, TOOL_HELP
        self.assertIn("outline", ACTION_SCHEMA["properties"])
        self.assertEqual(ACTION_SCHEMA["properties"]["outline"]["type"], "boolean")
        self.assertIn("outline", TOOL_HELP)

    # ---- 2. ranked tree + rollups -------------------------------------------
    def test_ranked_tree_by_recency_keeps_every_dir_via_rollup(self):
        from forge import workspace
        d = self._git_repo()
        for name in ("aaa", "mmm", "zzz"):
            os.makedirs(os.path.join(d, name))
        for i in range(6):
            _write(os.path.join(d, "aaa", f"old{i}.py"), f"x = {i}\n")
        _write(os.path.join(d, "mmm", "mid.py"), "y = 1\n")
        self._commit(d, "init")
        # a later commit touches a lexically-LAST file — recency must float it up
        _write(os.path.join(d, "zzz", "fresh.py"), "def hot():\n    return 1\n")
        self._commit(d, "hot")
        tree, n = workspace.build_tree(d, cap=3)
        self.assertEqual(n, 8)
        self.assertIn("fresh.py", tree)                      # recent file made the top-3
        for dirname in ("aaa/", "mmm/", "zzz/"):             # every top-level dir survives truncation
            self.assertIn(dirname, tree)
        self.assertIn("more files not shown", tree)

    def test_no_git_tree_degrades_gracefully(self):
        from forge import workspace
        d = tempfile.mkdtemp()                               # NOT a git repo
        os.makedirs(os.path.join(d, "pkg"))
        for i in range(5):
            _write(os.path.join(d, "pkg", f"f{i}.py"), "z = 1\n")
        self.assertEqual(workspace._recency_scores(d), {})   # no crash, empty scores
        tree, n = workspace.build_tree(d, cap=2)
        self.assertEqual(n, 5)
        self.assertIn("pkg/", tree)                          # rollup still lists the dir

    # ---- 3. symbol index -----------------------------------------------------
    def test_index_extracts_python_and_regex_langs(self):
        from forge import index
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "a.py"), "def f():\n    pass\n\nclass C:\n    def m(self):\n        pass\n")
        _write(os.path.join(d, "b.js"), "export function g(x){}\nclass D {}\n")
        _write(os.path.join(d, "c.go"), "package main\nfunc H() {}\ntype T struct {}\n")
        _write(os.path.join(d, "d.rs"), "pub fn r() {}\nstruct S {}\n")
        py = {s["name"] for s in index.extract_symbols(os.path.join(d, "a.py"))}
        self.assertEqual(py, {"f", "C", "C.m"})
        self.assertIn("g", {s["name"] for s in index.extract_symbols(os.path.join(d, "b.js"))})
        self.assertIn("H", {s["name"] for s in index.extract_symbols(os.path.join(d, "c.go"))})
        self.assertIn("r", {s["name"] for s in index.extract_symbols(os.path.join(d, "d.rs"))})

    def test_index_refresh_is_incremental_by_mtime_size(self):
        from forge import index
        d = tempfile.mkdtemp()
        idxdir = tempfile.mkdtemp()
        _write(os.path.join(d, "one.py"), "def a():\n    pass\n")
        _write(os.path.join(d, "two.py"), "def b():\n    pass\n")
        files = ["one.py", "two.py"]

        calls = {"n": 0}
        real = index.extract_symbols
        def spy(path, text=None):
            calls["n"] += 1
            return real(path, text)
        index.extract_symbols = spy
        try:
            syms = index.refresh(d, files=files, index_dir=idxdir)
            self.assertEqual({s["name"] for s in syms}, {"a", "b"})
            self.assertEqual(calls["n"], 2)                  # both parsed on the cold run
            index.refresh(d, files=files, index_dir=idxdir)  # nothing changed
            self.assertEqual(calls["n"], 2)                  # ZERO re-extraction (incremental)
            time.sleep(0.01)
            _write(os.path.join(d, "two.py"), "def b():\n    pass\n\ndef c():\n    pass\n")
            out = index.refresh(d, files=files, index_dir=idxdir)
            self.assertEqual(calls["n"], 3)                  # only the changed file re-parsed
            self.assertIn("c", {s["name"] for s in out})
        finally:
            index.extract_symbols = real
        # the persisted jsonl is keyed per file
        self.assertTrue(os.path.exists(index.index_path(d, idxdir)))

    # ---- 4. token-budgeted briefing -----------------------------------------
    def test_context_respects_small_vs_large_budget(self):
        from forge import workspace
        d = tempfile.mkdtemp()
        os.environ["FORGE_INDEX_DIR"] = tempfile.mkdtemp()
        try:
            _write(os.path.join(d, "svc.py"), "def handler():\n    return 1\n")
            small = workspace.context(d, budget=4096)
            large = workspace.context(d, budget=65536)
            self.assertNotIn("Key symbols", small)           # tiny window → no symbol map
            self.assertIn("Key symbols", large)              # roomy window → symbols included
            self.assertIn("handler", large)
        finally:
            os.environ.pop("FORGE_INDEX_DIR", None)


class TestJitRetrieval(unittest.TestCase):
    """P4.5 — turn-start '[retrieved context]' injection from the symbol index."""

    _SAY = '{"thought":"done","action":"say","message":"ok"}'

    def _repo(self, **files):
        d = tempfile.mkdtemp()
        for name, body in files.items():
            p = os.path.join(d, name)
            os.makedirs(os.path.dirname(p), exist_ok=True) if "/" in name else None
            _write(p, body)
        return d

    def _note(self, agent):
        """The single injected retrieval message, or None."""
        for m in agent.messages:
            if m["role"] == "user" and m["content"].startswith("[retrieved context"):
                return m["content"]
        return None

    def test_injects_bounded_note_for_named_file_and_symbol(self):
        # a BARE Agent.send (no repl, no @-expansion) — proves cmd_run/fleet get it too
        d = self._repo(**{"pyproject.toml": "[project]\nname='x'\n",
                          "auth.py": "def authenticate(user, pw):\n    return True\n"})
        os.environ["FORGE_INDEX_DIR"] = tempfile.mkdtemp()
        try:
            a = Agent(ScriptBackend([self._SAY]), sm.EphemeralSession(d, "jit"), max_steps=2)
            a.send("please fix authenticate in auth.py")
            note = self._note(a)
            self.assertIsNotNone(note)                                # a note was injected
            self.assertIn("auth.py", note)                            # the matched file
            self.assertIn("authenticate — auth.py:1", note)           # symbol name — file:line
            self.assertIn("def authenticate(user, pw)", note)         # reconstructed signature
            self.assertIn("Test: pytest -q", note)                    # the detected test command
            self.assertLessEqual(len(note), 2400 + 20)                # capped
        finally:
            os.environ.pop("FORGE_INDEX_DIR", None)

    def test_skips_when_nothing_matches(self):
        d = self._repo(**{"auth.py": "def authenticate():\n    return 1\n"})
        os.environ["FORGE_INDEX_DIR"] = tempfile.mkdtemp()
        try:
            a = Agent(ScriptBackend([self._SAY]), sm.EphemeralSession(d, "jit"), max_steps=2)
            a.send("please fix the frobnicate gizmo and refactor everything")
            self.assertIsNone(self._note(a))                          # nothing real named → NOTHING
        finally:
            os.environ.pop("FORGE_INDEX_DIR", None)

    def test_note_is_capped_and_truncated(self):
        # 10 symbols with very long signatures → the raw note blows past the cap
        args = ", ".join(f"parameter_number_{i}=0" for i in range(30))
        body = "".join(f"def handler{n}({args}):\n    return {n}\n\n" for n in range(10))
        d = self._repo(**{"big.py": body})
        os.environ["FORGE_INDEX_DIR"] = tempfile.mkdtemp()
        try:
            a = Agent(ScriptBackend([self._SAY]), sm.EphemeralSession(d, "jit"), max_steps=2)
            a.send("look at the handler functions in big.py")
            note = self._note(a)
            self.assertIsNotNone(note)
            self.assertLessEqual(len(note), 2400 + len("\n… (truncated)"))
            self.assertTrue(note.endswith("(truncated)"))
            # never more than the hard caps' worth of symbol lines
            self.assertLessEqual(sum(1 for ln in note.splitlines() if ln.startswith("- handler")), 10)
        finally:
            os.environ.pop("FORGE_INDEX_DIR", None)

    def test_gated_off_in_bare_mode(self):
        # levers=frozenset() (bench 'bare' harness) → workspace lever off → NO note,
        # even when the prompt names a real file. Injection lives in the workspace lever.
        d = self._repo(**{"auth.py": "def authenticate():\n    return 1\n"})
        os.environ["FORGE_INDEX_DIR"] = tempfile.mkdtemp()
        try:
            a = Agent(ScriptBackend([self._SAY]), sm.EphemeralSession(d, "jit"),
                      max_steps=2, levers=frozenset())
            a.send("fix authenticate in auth.py")
            self.assertIsNone(self._note(a))
        finally:
            os.environ.pop("FORGE_INDEX_DIR", None)


class TestResume(unittest.TestCase):
    """P4.7 — reconstruct working memory from a transcript for `forge --resume`.
    Transcripts are built with real Session.log into a temp SESSIONS dir; no model
    calls, no daemon, no network."""

    def setUp(self):
        self.sdir = tempfile.mkdtemp()
        self._orig_sessions = sm.SESSIONS
        self._orig_registry = sm.registry
        sm.SESSIONS = self.sdir
        sm.registry = lambda: []              # nothing live unless a test says so

    def tearDown(self):
        sm.SESSIONS = self._orig_sessions
        sm.registry = self._orig_registry

    def _sess(self, sid, cwd):
        return sm.Session(sid, cwd, "m")

    def test_load_reconstructs_summary_plan_and_tail(self):
        from forge import resume as rz
        d = tempfile.mkdtemp()
        s = self._sess("recon1", d)
        s.log("meta", v=1, forge="x", model="m", ladder=["m"], cwd=d, mode="auto", briefing="abc")
        s.log("user", text="do the thing")
        s.log("plan", items=["[x] step one", "[~] step two"])
        s.log("action", action="read_file", args={"path": "a.py"}, thought="reading a")
        s.log("observation", text="line1\nline2", ok=True)
        s.log("compact", v=1, summary="earlier we set up X and Y", window=8192)
        s.log("action", action="bash", args={"command": "ls"}, thought="listing")
        s.log("observation", text="a.py\nb.py", ok=True)
        s.log("assistant", text="all set", thought="done")

        data = rz.load("recon1")
        self.assertIsNotNone(data)
        # last compaction summary → the '[Earlier progress]' note
        self.assertIn("earlier we set up X and Y", data["summary_note"]["content"])
        self.assertIn("Earlier progress", data["summary_note"]["content"])
        # last plan restored
        self.assertEqual(data["plan"], ["[x] step one", "[~] step two"])
        # tail: action→assistant JSON, observation→user 'Observation:' message,
        # the final say synthesized back into an action:say object
        roles = [m["role"] for m in data["tail_msgs"]]
        self.assertEqual(roles[0], "user")                       # the user turn
        self.assertIn("assistant", roles)
        self.assertTrue(any(m["role"] == "user" and m["content"].startswith("Observation:")
                            for m in data["tail_msgs"]))
        last = data["tail_msgs"][-1]
        self.assertEqual(last["role"], "assistant")
        self.assertIn('"say"', last["content"])
        self.assertIn("all set", last["content"])
        # a synthesized action carries the command from args (lossy but present)
        self.assertTrue(any('"ls"' in m["content"] for m in data["tail_msgs"] if m["role"] == "assistant"))
        # read action captured for ledger seeding
        self.assertIn("a.py", data["read_ts"])

    def test_load_without_compact_or_plan_is_graceful(self):
        from forge import resume as rz
        d = tempfile.mkdtemp()
        s = self._sess("bare1", d)
        s.log("meta", cwd=d)
        s.log("user", text="hi")
        s.log("assistant", text="hello")
        data = rz.load("bare1")
        self.assertIsNone(data["summary_note"])
        self.assertEqual(data["plan"], [])
        self.assertEqual(data["read_ts"], {})
        self.assertEqual(rz.load("no-such-sid"), None)

    def test_apply_keeps_head_splices_summary_and_tail(self):
        from forge import resume as rz
        d = tempfile.mkdtemp()
        s = self._sess("ap1", d)
        s.log("meta", cwd=d)
        s.log("compact", summary="the state so far", window=8192)
        s.log("user", text="continue please")
        s.log("assistant", text="ok")
        data = rz.load("ap1")

        agent = Agent(ScriptBackend([]), sm.EphemeralSession(d, "m"))
        head = list(agent.messages)                              # system-only head (workspace=None)
        note = rz.apply(agent, data)
        self.assertEqual(agent.messages[:len(head)], head)       # fresh head untouched
        self.assertIn("the state so far", agent.messages[len(head)]["content"])
        self.assertEqual(len(agent.meta), len(agent.messages))   # parallel meta stays aligned
        self.assertIn("resumed ap1", note)

    def test_apply_seeds_only_unchanged_files(self):
        from forge import resume as rz
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "keep.py"), "x = 1\n")
        _write(os.path.join(d, "stale.py"), "y = 2\n")
        s = self._sess("seed1", d)
        s.log("meta", cwd=d)
        s.log("action", action="read_file", args={"path": "keep.py"}, thought="r")
        s.log("action", action="read_file", args={"path": "stale.py"}, thought="r")
        s.log("assistant", text="done")
        data = rz.load("seed1")
        # stale.py is modified AFTER its recorded read → its mtime is now newer
        future = time.time() + 1000
        os.utime(os.path.join(d, "stale.py"), (future, future))

        agent = Agent(ScriptBackend([]), sm.EphemeralSession(d, "m"))
        rz.apply(agent, data)
        keep_fp = os.path.realpath(os.path.join(d, "keep.py"))
        stale_fp = os.path.realpath(os.path.join(d, "stale.py"))
        self.assertTrue(agent.ledger.current(keep_fp))           # unchanged → seeded & current
        self.assertIsNone(agent.ledger.get(stale_fp))            # changed → NOT seeded (stays unread)

    def test_live_pid_session_refuses_resume(self):
        from forge import resume as rz
        sm.registry = lambda: [{"sid": "livesid", "pid": os.getpid()}]
        self.assertTrue(rz.is_live("livesid"))
        self.assertFalse(rz.is_live("deadsid"))

    def test_last_picks_newest_non_live_for_cwd(self):
        from forge import resume as rz
        d1 = tempfile.mkdtemp()
        d2 = tempfile.mkdtemp()
        self._sess("old1", d1).log("meta", cwd=d1)
        self._sess("new1", d1).log("meta", cwd=d1)
        self._sess("other1", d2).log("meta", cwd=d2)     # different cwd — must be ignored
        os.utime(os.path.join(self.sdir, "old1.jsonl"), (1000, 1000))
        os.utime(os.path.join(self.sdir, "new1.jsonl"), (2000, 2000))
        os.utime(os.path.join(self.sdir, "other1.jsonl"), (3000, 3000))

        self.assertEqual(rz.latest_sid(d1), "new1")              # newest for d1
        self.assertEqual(rz.resolve_sid("last", d1), "new1")
        # a live newest is skipped → the next-newest non-live wins
        sm.registry = lambda: [{"sid": "new1", "pid": os.getpid()}]
        self.assertEqual(rz.latest_sid(d1), "old1")

    def test_resolve_explicit_sid_and_prefix(self):
        from forge import resume as rz
        d = tempfile.mkdtemp()
        self._sess("abcd1234ef", d).log("meta", cwd=d)
        self.assertEqual(rz.resolve_sid("abcd1234ef", d), "abcd1234ef")
        self.assertEqual(rz.resolve_sid("abcd12", d), "abcd1234ef")   # unambiguous prefix
        self.assertIsNone(rz.resolve_sid("nomatch", d))

    def test_agent_logs_plan_records_for_resume(self):
        # DO item 1: the plan-update branch persists a 'plan' record so resume can
        # restore the living plan. A record fires only when the plan actually changes.
        d = tempfile.mkdtemp()
        sess = _RecSession(d)
        plan_a = '{"thought":"t","action":"list_files","path":".","plan":["[ ] a","[ ] b"]}'
        plan_a2 = '{"thought":"t","action":"list_files","path":".","plan":["[ ] a","[ ] b"]}'
        plan_b = '{"thought":"t","action":"say","message":"done","plan":["[x] a","[~] b"]}'
        a = Agent(ScriptBackend([plan_a, plan_a2, plan_b]), sess, max_steps=6)
        a.send("go")
        plans = [f["items"] for k, f in sess.logs if k == "plan"]
        self.assertEqual(plans[0], ["[ ] a", "[ ] b"])
        self.assertEqual(plans[-1], ["[x] a", "[~] b"])
        self.assertEqual(len(plans), 2)                          # unchanged repeat did not re-log


class TestPromptHistory(unittest.TestCase):
    """P4.7 — prompt history persists to ~/.forge/history across sessions."""

    def test_history_round_trip_and_flatten(self):
        from forge import repl
        d = tempfile.mkdtemp()
        orig = repl.HISTORY_PATH
        repl.HISTORY_PATH = os.path.join(d, "history")
        try:
            self.assertEqual(repl._load_history(), [])
            repl._append_history("first prompt")
            repl._append_history("second prompt")
            self.assertEqual(repl._load_history(), ["first prompt", "second prompt"])
            repl._append_history("multi\nline")             # newline flattened
            repl._append_history("   ")                     # blank ignored
            hist = repl._load_history()
            self.assertEqual(hist[-1], "multi line")
            self.assertNotIn("   ", hist)
        finally:
            repl.HISTORY_PATH = orig

    def test_history_trims_to_cap(self):
        from forge import repl
        d = tempfile.mkdtemp()
        orig_path, orig_cap = repl.HISTORY_PATH, repl.HISTORY_CAP
        repl.HISTORY_PATH = os.path.join(d, "history")
        repl.HISTORY_CAP = 5
        try:
            for i in range(12):
                repl._append_history(f"prompt {i}")
            hist = repl._load_history()                     # load trims to the cap
            self.assertEqual(len(hist), 5)
            self.assertEqual(hist[-1], "prompt 11")
        finally:
            repl.HISTORY_PATH, repl.HISTORY_CAP = orig_path, orig_cap


if __name__ == "__main__":
    unittest.main()
