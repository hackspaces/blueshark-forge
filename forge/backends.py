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
    FORGE_NUM_CTX     context window (default 8192)
    FORGE_NUM_PREDICT max tokens per turn (default 2048)
Ollama uses Metal automatically. Set OLLAMA_FLASH_ATTENTION=1 in the environment
for faster, lower-memory attention.
"""
import json
import os
import urllib.request

KEEP_ALIVE = os.environ.get("FORGE_KEEP_ALIVE", "30m")
NUM_CTX = int(os.environ.get("FORGE_NUM_CTX", "16384"))  # summarization compaction keeps us under this
NUM_PREDICT = int(os.environ.get("FORGE_NUM_PREDICT", "2048"))


class OllamaBackend:
    def __init__(self, model, url=None):
        self.model = model
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.name = f"ollama:{model}"

    def _body(self, messages, schema, temperature, stream):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": temperature, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        }
        if schema:
            body["format"] = schema
        return body

    def _req(self, body):
        return urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})

    def chat(self, messages, schema=None, temperature=0.0):
        with urllib.request.urlopen(self._req(self._body(messages, schema, temperature, False)), timeout=600) as r:
            return json.loads(r.read())["message"]["content"]

    def stream(self, messages, schema=None, temperature=0.0):
        with urllib.request.urlopen(self._req(self._body(messages, schema, temperature, True)), timeout=600) as r:
            for line in r:
                if not line.strip():
                    continue
                obj = json.loads(line)
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break

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

    def _body(self, messages, schema, temperature, stream):
        body = {"model": self.model, "messages": messages, "temperature": temperature, "stream": stream}
        if schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "action", "schema": schema, "strict": True}}
        return body

    def _req(self, body):
        return urllib.request.Request(
            f"{self.url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})

    def chat(self, messages, schema=None, temperature=0.0):
        with urllib.request.urlopen(self._req(self._body(messages, schema, temperature, False)), timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def stream(self, messages, schema=None, temperature=0.0):
        with urllib.request.urlopen(self._req(self._body(messages, schema, temperature, True)), timeout=600) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    def warm(self):
        pass


def make_backend(spec):
    if spec.startswith("openai:"):
        rest = spec[len("openai:"):]
        url = "https://api.openai.com/v1"
        if "@" in rest:
            model, url = rest.split("@", 1)
        else:
            model = rest
        return OpenAICompatBackend(model, url)
    if spec.startswith("ollama:"):
        return OllamaBackend(spec[len("ollama:"):])
    return OllamaBackend(spec)
