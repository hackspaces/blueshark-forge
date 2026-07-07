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
from forge.tools import (execute, _fuzzy_replace, _syntax_error, _which,  # noqa: E402
                         shape, overflow_dir, _maybe_offload, MAX_OUTPUT,
                         error_hint, _closest_region)


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


if __name__ == "__main__":
    unittest.main()
