"""Backend protocol-parser tests. Stdlib unittest (no deps), no live servers —
the pure generators (iter_sse / iter_ndjson) are the seam, so fixture byte
streams stand in for a real inference server.

    python -m unittest discover -s tests
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import backends                                       # noqa: E402
from forge.backends import (iter_sse, iter_ndjson, ctx_cap,      # noqa: E402
                            OllamaBackend, OpenAICompatBackend)


class _FakeResp:
    """A context-manager iterable over lines — stands in for an HTTP response."""
    def __init__(self, lines):
        self.lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)


def _sse(obj):
    return ("data: " + json.dumps(obj)).encode()


class TestIterSSE(unittest.TestCase):
    def test_content_then_usage_frame(self):
        lines = [
            _sse({"choices": [{"delta": {"content": "Hel"}}]}),
            _sse({"choices": [{"delta": {"content": "lo"}}]}),
            _sse({"choices": [], "usage": {"prompt_tokens": 123}}),
            b"data: [DONE]",
        ]
        out = list(iter_sse(lines))
        text = "".join(c for c, u in out)
        usages = [u for c, u in out if u]
        self.assertEqual(text, "Hello")
        self.assertEqual(len(usages), 1)
        self.assertEqual(usages[0]["prompt_tokens"], 123)

    def test_stops_at_done(self):
        lines = [
            _sse({"choices": [{"delta": {"content": "a"}}]}),
            b"data: [DONE]",
            _sse({"choices": [{"delta": {"content": "SHOULD-NOT-APPEAR"}}]}),
        ]
        text = "".join(c for c, u in iter_sse(lines))
        self.assertEqual(text, "a")

    def test_malformed_interleaved_lines_skipped(self):
        lines = [
            b": keepalive comment",                  # not a data: line
            _sse({"choices": [{"delta": {"content": "x"}}]}),
            b"data: {not valid json",                # JSONDecodeError
            b"data: {}",                             # KeyError/IndexError (no choices)
            _sse({"choices": [{"delta": {"content": "y"}}]}),
            _sse({"choices": [], "usage": {"prompt_tokens": 7}}),
            b"data: [DONE]",
        ]
        out = list(iter_sse(lines))          # must not raise
        text = "".join(c for c, u in out)
        self.assertEqual(text, "xy")
        self.assertEqual([u for c, u in out if u][0]["prompt_tokens"], 7)

    def test_accepts_str_lines(self):
        lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "z"}}]})]
        self.assertEqual("".join(c for c, u in iter_sse(lines)), "z")

    def test_hostile_frames_do_not_crash_and_content_is_always_str(self):
        # non-object frames, non-dict choices/delta, and non-string content must NOT
        # raise (AttributeError/TypeError previously escaped and killed the stream).
        lines = [
            b"data: 5", b"data: null", b'data: "x"', b"data: [1,2]",
            _sse({"choices": "nope"}),                        # choices not a list
            _sse({"choices": [5]}),                           # choices[0] not a dict
            _sse({"choices": [{"delta": None}]}),             # delta null
            _sse({"choices": [{"delta": "txt"}]}),            # delta a string
            _sse({"choices": [{"delta": {"content": 5}}]}),   # non-string content
            _sse({"choices": [{"delta": {"content": None}}]}),
            _sse({"choices": [{"delta": {"content": "good"}}]}),
            _sse({"choices": [], "usage": {"prompt_tokens": 9}}),
            b"data: [DONE]",
        ]
        out = list(iter_sse(lines))                            # must not raise
        self.assertEqual("".join(c for c, u in out), "good")
        self.assertTrue(all(isinstance(c, str) for c, u in out))
        self.assertEqual([u for c, u in out if u][0]["prompt_tokens"], 9)

    def test_usage_attached_to_a_content_chunk_is_surfaced(self):
        # some engines attach usage to the final CONTENT chunk, not a separate empty one
        lines = [_sse({"choices": [{"delta": {"content": "hi"}}],
                       "usage": {"prompt_tokens": 42}})]
        out = list(iter_sse(lines))
        self.assertEqual("".join(c for c, u in out), "hi")
        self.assertEqual([u for c, u in out if u][0]["prompt_tokens"], 42)


class TestIterNDJSON(unittest.TestCase):
    def test_content_then_done_frame(self):
        lines = [
            json.dumps({"message": {"content": "Hel"}, "done": False}).encode(),
            json.dumps({"message": {"content": "lo"}, "done": False}).encode(),
            json.dumps({"message": {"content": ""}, "done": True,
                        "prompt_eval_count": 77}).encode(),
        ]
        out = list(iter_ndjson(lines))
        text = "".join(c for c, u in out)
        usages = [u for c, u in out if u]
        self.assertEqual(text, "Hello")
        self.assertEqual(usages[0]["prompt_eval_count"], 77)

    def test_stops_after_done(self):
        lines = [
            json.dumps({"message": {"content": "a"}, "done": True,
                        "prompt_eval_count": 3}).encode(),
            json.dumps({"message": {"content": "SHOULD-NOT-APPEAR"}}).encode(),
        ]
        text = "".join(c for c, u in iter_ndjson(lines))
        self.assertEqual(text, "a")

    def test_blank_and_malformed_lines_skipped(self):
        lines = [
            b"",
            b"   ",
            b"{not json",
            json.dumps({"message": {"content": "ok"}, "done": True,
                        "prompt_eval_count": 5}).encode(),
        ]
        out = list(iter_ndjson(lines))       # must not raise
        self.assertEqual("".join(c for c, u in out), "ok")
        self.assertEqual([u for c, u in out if u][0]["prompt_eval_count"], 5)

    def test_non_object_and_null_message_frames_do_not_crash(self):
        # `{"message": null}` makes `.get("message", {})` return None (not the default);
        # a non-object frame or non-string content must not raise into the loop.
        lines = [
            b"5", b"null", b'"x"', b"[1]",
            json.dumps({"message": None}).encode(),            # present-but-null
            json.dumps({"message": {"content": 7}}).encode(),  # non-string content
            json.dumps({"message": "txt"}).encode(),           # message not a dict
            json.dumps({"message": {"content": "good"}, "done": True,
                        "prompt_eval_count": 11}).encode(),
        ]
        out = list(iter_ndjson(lines))       # must not raise
        self.assertEqual("".join(c for c, u in out), "good")
        self.assertTrue(all(isinstance(c, str) for c, u in out))
        self.assertEqual([u for c, u in out if u][0]["prompt_eval_count"], 11)


class TestStreamIntegration(unittest.TestCase):
    def test_openai_stream_captures_prompt_tokens(self):
        lines = [
            _sse({"choices": [{"delta": {"content": "Hi"}}]}),
            _sse({"choices": [{"delta": {"content": "!"}}]}),
            _sse({"choices": [], "usage": {"prompt_tokens": 123}}),
            b"data: [DONE]",
        ]
        b = OpenAICompatBackend("gpt-x", "https://h/v1")
        b._open = lambda *a, **k: _FakeResp(lines)
        chunks = list(b.stream([{"role": "user", "content": "hi"}]))
        self.assertEqual("".join(chunks), "Hi!")
        self.assertEqual(b.last_prompt_tokens, 123)   # the bug fix: no longer 0

    def test_ollama_stream_captures_prompt_eval_count(self):
        lines = [
            json.dumps({"message": {"content": "Hi"}, "done": False}).encode(),
            json.dumps({"message": {"content": ""}, "done": True,
                        "prompt_eval_count": 77}).encode(),
        ]
        b = OllamaBackend("gemma2:9b")
        b._open = lambda *a, **k: _FakeResp(lines)
        chunks = list(b.stream([{"role": "user", "content": "hi"}]))
        self.assertEqual("".join(chunks), "Hi")
        self.assertEqual(b.last_prompt_tokens, 77)


class TestBodyEnvelopes(unittest.TestCase):
    def test_ollama_body(self):
        b = OllamaBackend("gemma2:9b")
        b._window = 8192                       # avoid a live /api/show call
        schema = {"type": "object"}
        body = b._body([{"role": "user", "content": "x"}], schema, 0.0, True)
        self.assertEqual(body["format"], schema)
        self.assertIn("num_ctx", body["options"])
        self.assertEqual(body["options"]["num_predict"], backends.NUM_PREDICT)
        self.assertTrue(body["stream"])
        # no schema -> no format key
        self.assertNotIn("format", b._body([], None, 0.0, False))

    def test_openai_body(self):
        b = OpenAICompatBackend("gpt-x", "https://h/v1")
        schema = {"type": "object"}
        body = b._body([{"role": "user", "content": "x"}], schema, 0.0, True)
        rf = body["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["schema"], schema)
        self.assertFalse(rf["json_schema"]["strict"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        # non-stream -> no stream_options; no schema -> no response_format
        plain = b._body([], None, 0.0, False)
        self.assertNotIn("stream_options", plain)
        self.assertNotIn("response_format", plain)


class TestCtxCap(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop("FORGE_NUM_CTX", None)

    def tearDown(self):
        if self._env is None:
            os.environ.pop("FORGE_NUM_CTX", None)
        else:
            os.environ["FORGE_NUM_CTX"] = self._env

    def test_env_wins(self):
        os.environ["FORGE_NUM_CTX"] = "12345"
        self.assertEqual(ctx_cap(), 12345)

    def test_config_next(self):
        from forge import config
        orig = config.load
        config.load = lambda: {"num_ctx": 4096}
        try:
            self.assertEqual(ctx_cap(), 4096)
        finally:
            config.load = orig

    def test_default_last(self):
        from forge import config
        orig = config.get
        def _boom(*a, **k):
            raise RuntimeError("no config")
        config.get = _boom
        try:
            self.assertEqual(ctx_cap(), 32768)
        finally:
            config.get = orig


class _HTTPBody:
    def read(self):
        return b"server-body"

    def close(self):
        pass


class TestForgeError(unittest.TestCase):
    """_open translates raw urllib failures into clean, user-facing ForgeErrors."""
    def setUp(self):
        import urllib.request
        self._orig = urllib.request.urlopen

    def tearDown(self):
        import urllib.request
        urllib.request.urlopen = self._orig

    def _raise(self, exc):
        import urllib.request
        urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(exc)

    def test_ollama_404_is_clean_error(self):
        import urllib.error
        self._raise(urllib.error.HTTPError("u", 404, "nf", {}, _HTTPBody()))
        with self.assertRaises(backends.ForgeError) as cm:
            OllamaBackend("ghost-model")._open("req")
        self.assertIn("not installed", str(cm.exception))

    def test_openai_auth_failure(self):
        import urllib.error
        self._raise(urllib.error.HTTPError("u", 401, "no", {}, _HTTPBody()))
        with self.assertRaises(backends.ForgeError) as cm:
            OpenAICompatBackend("gpt-x", "https://h/v1")._open("req")
        self.assertIn("authentication failed", str(cm.exception))

    def test_openai_unreachable(self):
        import urllib.error
        self._raise(urllib.error.URLError("down"))
        with self.assertRaises(backends.ForgeError) as cm:
            OpenAICompatBackend("gpt-x", "https://h/v1")._open("req")
        self.assertIn("Can't reach", str(cm.exception))


class _ReadResp:
    """A context-manager HTTP response whose read() returns fixed JSON bytes."""
    def __init__(self, payload):
        self._payload = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


_COMPLETION = {"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": 9}}


class TestSchemaDialect(unittest.TestCase):
    """P5.1 schema-dialect negotiation for OpenAI-compat engines. No network — a fake
    _open dispatches on the request body so a fixture 400 stands in for a server that
    rejects response_format, and the dialect ladder is exercised deterministically."""

    def setUp(self):
        from forge import config
        self._orig_set, self._orig_get = config.set_key, config.get
        self.persisted = {}
        config.set_key = lambda k, v: self.persisted.__setitem__(k, v)
        config.get = lambda k, d=None: None          # no previously-cached dialect

    def tearDown(self):
        from forge import config
        config.set_key, config.get = self._orig_set, self._orig_get

    def _backend(self, dispatch):
        b = OpenAICompatBackend("m", "https://h/v1")
        b._schema_dialect = None                     # force negotiation
        b._open = lambda req, timeout=600: dispatch(json.loads(req.data.decode()))
        return b

    def test_body_dialects(self):
        b = OpenAICompatBackend("m", "https://h/v1")
        schema = {"type": "object"}
        self.assertIn("response_format", b._body([], schema, 0.0, False, "response_format"))
        self.assertEqual(b._body([], schema, 0.0, False, "guided_json")["guided_json"], schema)
        self.assertEqual(b._body([], schema, 0.0, False, "json_schema")["json_schema"], schema)
        none_body = b._body([], schema, 0.0, False, "none")
        for k in ("response_format", "guided_json", "json_schema"):
            self.assertNotIn(k, none_body)

    def test_negotiation_falls_back_to_guided_json(self):
        calls = []
        def dispatch(body):
            if "response_format" in body:
                calls.append("response_format")
                raise backends.ForgeError("https://h/v1 returned HTTP 400: unknown parameter response_format")
            if "guided_json" in body:
                calls.append("guided_json")
                return _ReadResp(_COMPLETION)
            calls.append("none")
            return _ReadResp(_COMPLETION)
        b = self._backend(dispatch)
        out = b.chat([{"role": "user", "content": "hi"}], schema={"type": "object"})
        self.assertEqual(out, "OK")
        self.assertEqual(calls, ["response_format", "guided_json"])
        self.assertEqual(b._schema_dialect, "guided_json")
        self.assertEqual(self.persisted.get("schema_dialect"), "guided_json")
        self.assertEqual(b.last_prompt_tokens, 9)

    def test_negotiation_first_dialect_wins(self):
        def dispatch(body):
            self.assertIn("response_format", body)   # accepted on the first try, never downgrades
            return _ReadResp(_COMPLETION)
        b = self._backend(dispatch)
        b.chat([{"role": "user", "content": "hi"}], schema={"type": "object"})
        self.assertEqual(b._schema_dialect, "response_format")
        self.assertEqual(self.persisted.get("schema_dialect"), "response_format")

    def test_negotiation_all_reject_falls_to_none(self):
        seen = []
        def dispatch(body):
            for k in ("response_format", "guided_json", "json_schema"):
                if k in body:
                    seen.append(k)
                    raise backends.ForgeError(f"https://h/v1 returned HTTP 400: bad {k}")
            seen.append("none")
            return _ReadResp(_COMPLETION)            # 'none' carries no schema → must open
        b = self._backend(dispatch)
        out = b.chat([{"role": "user", "content": "hi"}], schema={"type": "object"})
        self.assertEqual(out, "OK")
        self.assertEqual(seen, ["response_format", "guided_json", "json_schema", "none"])
        self.assertEqual(b._schema_dialect, "none")             # session-local fallback
        self.assertIsNone(self.persisted.get("schema_dialect"))  # NOT persisted: re-probe next run
                                                                 # (a transient/unrelated 400 must not
                                                                 #  permanently disable constrained decoding)

    def test_non_schema_400_propagates(self):
        def dispatch(body):
            raise backends.ForgeError("https://h/v1 returned HTTP 400: context length exceeded")
        b = self._backend(dispatch)
        with self.assertRaises(backends.ForgeError):
            b.chat([{"role": "user", "content": "hi"}], schema={"type": "object"})
        self.assertIsNone(b._schema_dialect)         # a real error caches nothing

    def test_cached_dialect_skips_negotiation(self):
        calls = []
        def dispatch(body):
            calls.append([k for k in ("response_format", "guided_json", "json_schema") if k in body])
            return _ReadResp(_COMPLETION)
        b = self._backend(dispatch)
        b._schema_dialect = "guided_json"            # already resolved
        b.chat([{"role": "user", "content": "hi"}], schema={"type": "object"})
        self.assertEqual(calls, [["guided_json"]])   # one call, guided_json only

    def test_no_schema_never_negotiates(self):
        calls = []
        def dispatch(body):
            calls.append(body)
            return _ReadResp(_COMPLETION)
        b = self._backend(dispatch)
        b.chat([{"role": "user", "content": "hi"}])  # schema=None
        self.assertEqual(len(calls), 1)
        self.assertIsNone(b._schema_dialect)         # unresolved — nothing to probe


class TestFrontierEngines(unittest.TestCase):
    """Frontier models (OpenAI, Anthropic) via the OpenAI-compatible backend + BYO key."""

    def test_anthropic_engine_points_at_the_compat_endpoint(self):
        from forge.backends import make_backend, OpenAICompatBackend
        b = make_backend("claude-sonnet-4", engine="anthropic", api_key="sk-ant-x")
        self.assertIsInstance(b, OpenAICompatBackend)
        self.assertEqual(b.url, "https://api.anthropic.com/v1")   # posts to /v1/chat/completions
        self.assertEqual(b.key, "sk-ant-x")

    def test_anthropic_falls_back_to_ANTHROPIC_API_KEY(self):
        from forge.backends import make_backend
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env", "OPENAI_API_KEY": "sk-openai"}, clear=True):
            b = make_backend("claude-x", engine="anthropic")     # no explicit key
        self.assertEqual(b.key, "sk-ant-env")                    # the RIGHT provider's env, not OpenAI's

    def test_openai_engine_still_works(self):
        from forge.backends import make_backend
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai"}, clear=True):
            b = make_backend("gpt-4o", engine="openai")
        self.assertEqual(b.url, "https://api.openai.com/v1")
        self.assertEqual(b.key, "sk-openai")

    def test_explicit_key_beats_env(self):
        from forge.backends import make_backend
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env"}, clear=True):
            b = make_backend("claude-x", engine="anthropic", api_key="explicit")
        self.assertEqual(b.key, "explicit")


if __name__ == "__main__":
    unittest.main()
