"""`forge setup` — the installer. Inspects the machine, recommends a model ladder
sized to its RAM, pulls the models via Ollama, and writes a per-machine config.
Run once per computer; each gets its own ~/.forge/config.json."""
import os
import platform
import shutil
import subprocess

from . import config
from .util import slurp

def detect_machine():
    info = {"os": platform.system(), "arch": platform.machine(), "cores": os.cpu_count() or 0, "ram_gb": 0, "chip": ""}
    try:
        if info["os"] == "Darwin":
            info["ram_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / (1024**3))
            info["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        elif info["os"] == "Linux":
            for line in slurp("/proc/meminfo").splitlines():
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / (1024**2))
                    break
            info["chip"] = platform.processor()
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return info


# RAM-tiered local ladders (cheapest → strongest), from the 2026 Apple-Silicon /
# local-model landscape. MoE coders run fast on Apple Silicon at large sizes.
TIERS = [
    (64, ["qwen3-coder:30b", "qwen3.6:35b-a3b"], "workstation"),
    (40, ["qwen3-coder:30b", "qwen3.6:35b-a3b"], "high (e.g. M-Pro/Max 48GB+)"),
    (28, ["qwen2.5-coder:7b", "qwen3-coder:30b"], "mid (≈32GB)"),
    (14, ["gemma2:9b", "qwen2.5-coder:7b"], "modest (≈16GB)"),
    (7, ["qwen2.5:3b", "gemma2:9b"], "light (≈8GB)"),
    (0, ["qwen2.5:3b"], "minimal"),
]


def recommend(ram_gb):
    for floor, ladder, label in TIERS:
        if ram_gb >= floor:
            return ladder, label
    return TIERS[-1][1], TIERS[-1][2]


def num_ctx_for(ram_gb):
    """How much context window to run with, sized to RAM (bigger = fewer
    compactions, but more unified memory for the KV cache)."""
    if ram_gb >= 64:
        return 65536
    if ram_gb >= 40:
        return 32768
    if ram_gb >= 28:
        return 24576
    if ram_gb >= 14:
        return 12288
    return 8192


def _ollama_ok():
    if not shutil.which("ollama"):
        return False, "Ollama is not installed. Install it from https://ollama.com, then re-run `forge setup`."
    try:
        subprocess.run(["ollama", "list"], capture_output=True, timeout=10, check=True)
        return True, ""
    except (subprocess.SubprocessError, OSError):
        return False, "Ollama is installed but not running. Start it (`ollama serve`) and re-run `forge setup`."


def _have_model(name):
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, timeout=10)
        base = name.split(":")[0]
        return any(line.split()[0].split(":")[0] == base and name.split(":")[-1] in line for line in out.splitlines()[1:]) \
            or any(name == line.split()[0] for line in out.splitlines()[1:])
    except (subprocess.SubprocessError, OSError):
        return False


def _ask_engine():
    print("  How do you run models?")
    print("    1) Ollama — local, laptop-friendly, auto-manages models  (default)")
    print("    2) An OpenAI-compatible server — vLLM · llama.cpp · MLX · LM Studio · TGI · SGLang · cloud API")
    print("       (you give the URL + model names; for a workstation/cluster with big models, or remote inference)")
    try:
        return "ollama" if input("  [1/2]: ").strip() != "2" else "server"
    except EOFError:
        return "ollama"


def _ask_server(url, models, api_key):
    from .backends import ENGINE_URLS
    print("\n  Which server? (vllm / llamacpp / mlx / lmstudio / tgi / sglang / openai)")
    try:
        eng = (input("  engine [openai]: ").strip() or "openai").lower()
    except EOFError:
        eng = "openai"
    default_url = ENGINE_URLS.get(eng, "http://localhost:8000/v1")
    if not url:
        try:
            url = input(f"  base URL [{default_url}]: ").strip() or default_url
        except EOFError:
            url = default_url
    if not models:
        try:
            models = [m.strip() for m in input("  model name(s), cheap→strong, comma-separated: ").split(",") if m.strip()]
        except EOFError:
            models = []
    if api_key is None:
        try:
            api_key = input("  API key (blank for none): ").strip()
        except EOFError:
            api_key = ""
    return eng, url, models, api_key


def run(auto=False, keep_models=None, engine=None, url=None, api_key=None, models=None):
    print("forge setup\n")
    hw = detect_machine()
    print(f"  machine: {hw['chip'] or hw['arch']} · {hw['ram_gb']}GB RAM · {hw['cores']} cores · {hw['os']}\n")

    if engine is None:
        engine = "ollama" if auto else _ask_engine()
    if engine == "server":
        engine, url, models, api_key = _ask_server(url, models, api_key)

    if engine == "ollama":
        return _setup_ollama(hw, auto, models or keep_models)
    return _setup_server(hw, engine, url, models, api_key)


def _setup_ollama(hw, auto, keep_models):
    ok, msg = _ollama_ok()
    if not ok:
        print(f"  ✗ {msg}")
        return 1
    ladder, label = recommend(hw["ram_gb"])
    if keep_models:
        ladder = keep_models
    print(f"  engine:  ollama (local)")
    print(f"  tier:    {label}")
    print(f"  ladder:  {' → '.join(ladder)}  (fast → strong, escalates when stuck)\n")
    if not auto:
        try:
            ans = input("  pull these models and save config? [Y/n] ").strip().lower()
        except EOFError:
            ans = "y"
        if ans and ans not in ("y", "yes"):
            print("  aborted."); return 1
    for m in ladder:
        if _have_model(m):
            print(f"  ✓ {m} already present"); continue
        print(f"  ↓ pulling {m} …")
        if subprocess.run(["ollama", "pull", m]).returncode != 0:
            print(f"  ✗ failed to pull {m} (name may differ on your Ollama — edit config.json). Continuing.")
    cfg = config.load()
    cfg.update({"engine": "ollama", "base_url": "", "api_key": "",
                "ladder": ladder, "num_ctx": num_ctx_for(hw["ram_gb"]), "machine": hw})
    config.save(cfg)
    print(f"\n  ✓ config written to {config.PATH}  ·  window {cfg['num_ctx']} tokens")
    print(f"  ✓ run `forge` to start (ladder: {' → '.join(ladder)})")
    return 0


def _setup_server(hw, engine, url, models, api_key):
    from .backends import ENGINE_URLS
    url = url or ENGINE_URLS.get(engine, "http://localhost:8000/v1")
    if not models:
        print("  ✗ no model names given. Re-run and provide at least one model your server serves."); return 1
    cfg = config.load()
    cfg.update({"engine": engine, "base_url": url, "api_key": api_key or "",
                "ladder": models, "num_ctx": max(num_ctx_for(hw["ram_gb"]), 16384), "machine": hw})
    config.save(cfg)
    print(f"\n  ✓ engine: {engine}  ·  {url}")
    print(f"  ✓ ladder: {' → '.join(models)}")
    print(f"  ✓ config written to {config.PATH}")
    print(f"  ✓ run `forge` to start (make sure your server is up)")
    return 0
