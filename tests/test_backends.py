"""Backend protocol-parser tests. Stdlib unittest (no deps), no live servers —
the pure generators (iter_sse / iter_ndjson) are the seam, so fixture byte
streams stand in for a real inference server.

    python -m unittest discover -s tests
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


if __name__ == "__main__":
    unittest.main()
