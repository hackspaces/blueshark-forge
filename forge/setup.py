"""`forge setup` — the installer. Inspects the machine, recommends a model ladder
sized to its RAM, pulls the models via Ollama, and writes a per-machine config.
Run once per computer; each gets its own ~/.forge/config.json."""
import os
import platform
import shutil
import subprocess

from . import config


def _slurp(path):
    with open(path, errors='replace') as f:
        return f.read()

def detect_machine():
    info = {"os": platform.system(), "arch": platform.machine(), "cores": os.cpu_count() or 0, "ram_gb": 0, "chip": ""}
    try:
        if info["os"] == "Darwin":
            info["ram_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / (1024**3))
            info["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        elif info["os"] == "Linux":
            for line in _slurp("/proc/meminfo").splitlines():
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


def run(auto=False, keep_models=None):
    print("forge setup\n")
    hw = detect_machine()
    print(f"  machine: {hw['chip'] or hw['arch']} · {hw['ram_gb']}GB RAM · {hw['cores']} cores · {hw['os']}")

    ok, msg = _ollama_ok()
    if not ok:
        print(f"\n  ✗ {msg}")
        return 1

    ladder, label = recommend(hw["ram_gb"])
    if keep_models:
        ladder = keep_models
    print(f"  tier:    {label}")
    print(f"  ladder:  {' → '.join(ladder)}  (fast → strong, escalates when stuck)\n")

    if not auto:
        try:
            ans = input("  pull these models and save config? [Y/n] ").strip().lower()
        except EOFError:
            ans = "y"
        if ans and ans not in ("y", "yes"):
            print("  aborted.")
            return 1

    for m in ladder:
        if _have_model(m):
            print(f"  ✓ {m} already present")
            continue
        print(f"  ↓ pulling {m} …")
        r = subprocess.run(["ollama", "pull", m])
        if r.returncode != 0:
            print(f"  ✗ failed to pull {m} (name may differ on your Ollama — edit config.json). Continuing.")

    cfg = config.load()
    cfg["ladder"] = ladder
    cfg["num_ctx"] = num_ctx_for(hw["ram_gb"])
    cfg["machine"] = hw
    config.save(cfg)
    print(f"\n  ✓ config written to {config.PATH}")
    print(f"  ✓ context window sized to {cfg['num_ctx']} tokens for {hw['ram_gb']}GB")
    print(f"  ✓ run `forge` to start (ladder: {' → '.join(ladder)})")
    return 0
