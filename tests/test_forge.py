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
