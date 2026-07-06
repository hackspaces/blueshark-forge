"""Per-machine config. Each computer gets its own ~/.forge/config.json — the model
ladder chosen for its hardware, plus runtime preferences. `forge setup` writes it;
the TUI edits it live; the CLI reads it (flags still override)."""
import json
import os

PATH = os.path.expanduser("~/.forge/config.json")

DEFAULTS = {
    "engine": "ollama",     # ollama | openai | vllm | llamacpp | mlx | lmstudio | tgi | sglang
    "base_url": "",         # for non-ollama engines (an OpenAI-compatible endpoint)
    "api_key": "",          # optional, for remote/authenticated endpoints
    "ladder": ["gemma2:9b", "qwen2.5-coder:7b"],   # overwritten by `forge setup`
    "num_ctx": 8192,
    "keep_alive": "30m",
    "num_predict": 2048,
    "stuck_threshold": 7,
    "verbose": False,
    "machine": {},                                  # detected hardware summary
}


def load():
    cfg = dict(DEFAULTS)
    try:
        with open(PATH) as f:
            cfg.update(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass
    return cfg


def save(cfg):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, PATH)
    return cfg


def get(key, default=None):
    return load().get(key, default)


def set_key(key, value):
    cfg = load()
    cfg[key] = value
    return save(cfg)


def exists():
    return os.path.exists(PATH)
