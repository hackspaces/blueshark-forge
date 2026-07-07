"""Test suite for forge. Stdlib unittest (no deps), no model calls — covers the
harness invariants, tools, config, fleet, and workspace logic.

    python -m unittest discover -s tests      # or: python -m pytest tests
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import session as sm            # noqa: E402
from forge.agent import Agent              # noqa: E402
from forge.backends import make_backend, OllamaBackend, OpenAICompatBackend  # noqa: E402
from forge.tools import execute, _fuzzy_replace  # noqa: E402


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


class TestWorkspace(unittest.TestCase):
    def test_detect_project(self):
        from forge import workspace
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "package.json"), "{}")
        label, markers = workspace.detect_project(d)
        self.assertIn("Node", label)

    def test_detect_inferred_language(self):
        from forge import workspace
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "a.py"), "x=1")
        _write(os.path.join(d, "b.py"), "y=2")
        label, _ = workspace.detect_project(d)
        self.assertIn("Python", label)


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
        def chat(self, m, schema=None, temperature=0.0):
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

    def test_missing_path_gets_instructive_error(self):
        import json as _json
        d = tempfile.mkdtemp()
        events = []
        b = self.Scripted([
            _json.dumps({"thought": "t", "action": "write_file", "content": "package main\n"}),
            _json.dumps({"thought": "t", "action": "say", "message": "ok"}),
        ])
        a = Agent(b, sm.EphemeralSession(d, "s"), max_steps=5,
                  on_event=lambda kind, **k: events.append((kind, k)))
        a.send("make it")
        obs = [k.get("text", "") for kind, k in events if kind == "observation"]
        self.assertTrue(any("missing its `path`" in t for t in obs), obs)
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


if __name__ == "__main__":
    unittest.main()
