"""Model-agnostic backends. Any model, one interface, tuned to run well locally.

`make_backend("<spec>")`:
    gemma2:9b                     -> local Ollama model
    ollama:qwen3.6:latest         -> explicit Ollama
    openai:gpt-4o@https://host/v1 -> any OpenAI-compatible endpoint

Two calls: `.chat()` returns the full text; `.stream()` yields text chunks so the
reply types out live. Constrained decoding (schema) keeps small models reliable.

Mac/Apple-Silicon tuning (env-overridable):
    FORGE_KEEP_ALIVE  keep the model resident between turns (default 30m — avoids
                      multi-second reloads; the single biggest local speedup)
    FORGE_NUM_CTX     context-window cap (default 32768; setup sizes it to RAM)
    FORGE_NUM_PREDICT max tokens per turn (default 2048)
Ollama uses Metal automatically. Set OLLAMA_FLASH_ATTENTION=1 in the environment
for faster, lower-memory attention.
"""
import hashlib
import json
import os
import socket
import urllib.error
import urllib.request


class ForgeError(Exception):
    """A clean, user-facing error — shown as a message, not a stack trace."""


KEEP_ALIVE = os.environ.get("FORGE_KEEP_ALIVE", "30m")
# Memory-safe CAP on the context we actually run with. A model's real window may
# be far larger (qwen3-coder = 256K); we use min(real_window, cap) so a huge
# window doesn't blow up unified memory. Raise it if you have RAM to spare.
NUM_PREDICT = int(os.environ.get("FORGE_NUM_PREDICT", "2048"))


def ctx_cap():
    """Memory-safe cap on context, resolved live: env > config > default."""
    v = os.environ.get("FORGE_NUM_CTX")
    if v:
        return int(v)
    try:
        from . import config
        return int(config.get("num_ctx", 32768))
    except Exception:
        return 32768


NUM_CTX = ctx_cap()  # back-compat static alias for the char-estimate fallback


# --- Pure protocol parsers -------------------------------------------------
# The two stream dialects are parsed by these module-level generators, each
# yielding (text_chunk, usage_or_none). Keeping them pure (they take an iterable
# of lines, never a socket) is the test seam: fixtures replace live servers, and
# a parser that MUST surface usage makes the dropped-usage-frame bug (below)
# impossible to reintroduce.

def iter_sse(lines):
    """OpenAI-dialect Server-Sent Events. Yields (delta_text, usage_or_none).
    The final usage frame has choices == [] — surface it as ('', usage) instead
    of letting it fall into the choices[0] IndexError and get dropped (which is
    why last_prompt_tokens stayed 0 forever on streaming non-Ollama engines)."""
    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8", "replace")
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
            if obj.get("usage") and not obj.get("choices"):
                yield "", obj["usage"]
            else:
                yield obj["choices"][0]["delta"].get("content", ""), None
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


def iter_ndjson(lines):
    """Ollama-dialect newline-delimited JSON. Yields (chunk_text, usage_or_none);
    on the done frame yields ('', {'prompt_eval_count': ...}) then stops."""
    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8", "replace")
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        yield obj.get("message", {}).get("content", ""), None
        if obj.get("done"):
            yield "", {"prompt_eval_count": obj.get("prompt_eval_count")}
            return


class OllamaBackend:
    def __init__(self, model, url=None):
        self.model = model
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.name = f"ollama:{model}"
        self._window = None            # the model's real context length (from /api/show)
        self.last_prompt_tokens = 0     # exact tokens of the last prompt (from the response)

    def context_window(self):
        """The model's TRUE context length, queried once from Ollama."""
        if self._window is None:
            self._window = self._query_window() or 8192
        return self._window

    def _query_window(self):
        try:
            req = urllib.request.Request(f"{self.url}/api/show",
                                         data=json.dumps({"model": self.model}).encode(),
                                         headers={"Content-Type": "application/json"})
            info = json.loads(urllib.request.urlopen(req, timeout=30).read())
            for k, v in (info.get("model_info") or {}).items():
                if k.endswith("context_length"):
                    return int(v)
        except Exception:
            return None

    def effective_ctx(self):
        """What we actually run with: the real window, capped for memory."""
        return min(self.context_window(), ctx_cap())

    def _body(self, messages, schema, temperature, stream):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": temperature, "num_ctx": self.effective_ctx(), "num_predict": NUM_PREDICT},
        }
        if schema:
            body["format"] = schema
        return body

    def _req(self, body):
        return urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})

    def _open(self, req, timeout=600):
        """urlopen with errors translated to clean, actionable messages."""
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            if e.code == 404:
                raise ForgeError(f"Model '{self.model}' is not installed. Pull it: ollama pull {self.model}   (or run: forge setup)")
            raise ForgeError(f"Ollama returned HTTP {e.code}: {body}")
        except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
            raise ForgeError(f"Can't reach Ollama at {self.url} — is it running? Start it with:  ollama serve")

    def chat(self, messages, schema=None, temperature=0.0):
        with self._open(self._req(self._body(messages, schema, temperature, False))) as r:
            resp = json.loads(r.read())
        if resp.get("prompt_eval_count"):
            self.last_prompt_tokens = resp["prompt_eval_count"]
        return resp["message"]["content"]

    def stream(self, messages, schema=None, temperature=0.0):
        with self._open(self._req(self._body(messages, schema, temperature, True))) as r:
            for chunk, usage in iter_ndjson(r):
                if chunk:
                    yield chunk
                if usage and usage.get("prompt_eval_count"):
                    self.last_prompt_tokens = usage["prompt_eval_count"]

    def warm(self):
        """Load the model into memory now, so the first real turn is fast."""
        try:
            body = {"model": self.model, "messages": [{"role": "user", "content": "hi"}],
                    "stream": False, "keep_alive": KEEP_ALIVE, "options": {"num_predict": 1}}
            urllib.request.urlopen(self._req(body), timeout=120).read()
        except Exception:
            pass


class OpenAICompatBackend:
    def __init__(self, model, url="https://api.openai.com/v1", key=None):
        self.model = model
        self.url = url.rstrip("/")
        self.key = key or os.environ.get("OPENAI_API_KEY", "")
        self.name = f"openai:{model}"
        self.last_prompt_tokens = 0

    def context_window(self):
        return int(os.environ.get("FORGE_REMOTE_CTX", "128000"))  # most modern APIs; override if needed

    def effective_ctx(self):
        return min(self.context_window(), ctx_cap() * 8)  # remote windows are large; cap generously

    def _body(self, messages, schema, temperature, stream):
        body = {"model": self.model, "messages": messages, "temperature": temperature, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if schema:
            # strict:False — our action schema has optional fields (command/path/…),
            # which OpenAI strict mode forbids. Non-strict json_schema still guides
            # vLLM/llama.cpp/LM Studio/OpenAI toward valid JSON.
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "action", "schema": schema, "strict": False}}
        return body

    def _req(self, body):
        return urllib.request.Request(
            f"{self.url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})

    def _open(self, req, timeout=600):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code in (401, 403):
                raise ForgeError(f"{self.url}: authentication failed ({e.code}). Check your API key (OPENAI_API_KEY / config api_key).")
            raise ForgeError(f"{self.url} returned HTTP {e.code}: {body}")
        except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
            raise ForgeError(f"Can't reach the inference server at {self.url}. Is it running and is the URL correct?")

    def chat(self, messages, schema=None, temperature=0.0):
        with self._open(self._req(self._body(messages, schema, temperature, False))) as r:
            resp = json.loads(r.read())
        if resp.get("usage", {}).get("prompt_tokens"):
            self.last_prompt_tokens = resp["usage"]["prompt_tokens"]
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ForgeError(f"Unexpected response from {self.url}: {str(resp)[:200]}")

    def stream(self, messages, schema=None, temperature=0.0):
        with self._open(self._req(self._body(messages, schema, temperature, True))) as r:
            for chunk, usage in iter_sse(r):
                if chunk:
                    yield chunk
                if usage and usage.get("prompt_tokens"):
                    self.last_prompt_tokens = usage["prompt_tokens"]

    def warm(self):
        pass


# Every serious inference server speaks the OpenAI-compatible protocol, so one
# backend covers them all — these are just the usual default ports so users don't
# have to remember URLs. `forge setup` writes the chosen engine + base_url to config.
ENGINE_URLS = {
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
    "llamacpp": "http://localhost:8080/v1",
    "mlx": "http://localhost:8080/v1",
    "lmstudio": "http://localhost:1234/v1",
    "localai": "http://localhost:8080/v1",
    "tgi": "http://localhost:8080/v1",
    "openai": "https://api.openai.com/v1",
}
LOCAL_ENGINES = {"ollama"}  # engines forge can pull models for / that run on-box


def make_backend(spec, engine="ollama", base_url=None, api_key=None):
    """Build a backend for a model spec. An explicit `ollama:`/`openai:` prefix
    overrides; otherwise a bare spec is routed to the configured `engine`."""
    if spec.startswith("ollama:"):
        return OllamaBackend(spec[len("ollama:"):])
    if spec.startswith("openai:"):
        rest = spec[len("openai:"):]
        url = base_url or "https://api.openai.com/v1"
        if "@" in rest:
            model, url = rest.split("@", 1)
        else:
            model = rest
        return OpenAICompatBackend(model, url, api_key)
    if engine in (None, "ollama"):
        return OllamaBackend(spec)
    url = base_url or ENGINE_URLS.get(engine, "http://localhost:8000/v1")
    return OpenAICompatBackend(spec, url, api_key)


def record_digest(messages):
    """A stable fingerprint of a prompt: md5 of the LAST 2000 chars of its JSON.
    The tail is what changes turn-to-turn (fresh observations, appended turns),
    so it discriminates steps while staying robust to a giant unchanging head.
    Shared by RecordingBackend (record) and ReplayBackend strict mode (verify)."""
    return hashlib.md5(json.dumps(messages)[-2000:].encode("utf-8")).hexdigest()


class RecordingBackend:
    """P3.3 flight recorder. Wraps a real backend and, per chat/stream call,
    appends a cassette row {digest, raw, prompt_tokens} to the FORGE_RECORD file —
    turning a live run into a replayable transcript at zero extra inference. Every
    other attribute (name / effective_ctx / context_window / last_prompt_tokens /
    warm / …) delegates to the inner backend, so the Agent sees a normal backend."""

    def __init__(self, inner, path):
        self._inner = inner
        self._path = path

    def _record(self, messages, raw):
        row = {"digest": record_digest(messages), "raw": raw,
               "prompt_tokens": getattr(self._inner, "last_prompt_tokens", 0)}
        with open(self._path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def chat(self, messages, schema=None, temperature=0.0):
        raw = self._inner.chat(messages, schema=schema, temperature=temperature)
        self._record(messages, raw)
        return raw

    def stream(self, messages, schema=None, temperature=0.0):
        raw = ""
        for chunk in self._inner.stream(messages, schema=schema, temperature=temperature):
            raw += chunk
            yield chunk
        # last_prompt_tokens is populated by the inner stream's usage frame, which
        # arrives at the end — so record only once the stream is fully drained.
        self._record(messages, raw)

    def __getattr__(self, name):
        # Only reached for attributes NOT set on the wrapper (name/effective_ctx/
        # context_window/last_prompt_tokens/warm/…) — forward them to the inner
        # backend so it stays a drop-in. _inner is set first in __init__, so this
        # never recurses on the wrapper's own private attrs.
        return getattr(self._inner, name)
