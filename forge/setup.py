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
        elif info["os"] == "Windows":
            # no sysctl / meminfo here — without this branch ram_gb stayed 0 and
            # every Windows machine silently fell to the minimal tier.
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                info["ram_gb"] = round(st.ullTotalPhys / (1024**3))
            info["chip"] = platform.processor()
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        pass
    # Discrete GPU(s) — the memory that actually matters for fast inference on a
    # workstation / multi-GPU rig / datacenter node. Apple Silicon has no separate
    # VRAM (unified memory = RAM). Best-effort: no nvidia-smi → gpus 0.
    info["gpus"] = 0
    info["vram_gb"] = 0
    info["gpu_name"] = ""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=6).decode("utf-8", "replace")
        rows = [r for r in out.splitlines() if r.strip()]
        if rows:
            info["gpus"] = len(rows)
            info["vram_gb"] = round(sum(int(r.split(",")[0]) for r in rows) / 1024)  # MiB → GB
            info["gpu_name"] = rows[0].split(",")[1].strip()
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return info


def _is_accelerated(hw):
    """True when this machine has a FAST memory path for inference: Apple Silicon
    (unified memory + Metal, ~120-800 GB/s) or a real NVIDIA dGPU (CUDA). Everything
    else — Intel/AMD laptops with integrated graphics — decodes CPU-only over shared
    DDR (~26-51 GB/s), where big models crawl (9B ≈ 4 tok/s) and can swap-thrash."""
    if hw.get("os") == "Darwin" and hw.get("arch") == "arm64":
        return True                                      # unified memory + Metal
    if hw.get("vram_gb"):
        return True                                      # a real discrete GPU was detected
    import shutil as _sh
    return _sh.which("nvidia-smi") is not None           # fallback when vram wasn't populated


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

# CPU-only ladders (no Apple Silicon, no dGPU): decode is memory-bandwidth-bound on
# shared DDR, so the ceiling is ~3B for a usable default — a 9B rung is ~4 tok/s at
# best and swap-thrashes an 8GB Windows laptop (the OS alone idles at 3-4GB). The 7B
# top rung on roomier machines is escalation-only: slow but smarter when stuck.
CPU_TIERS = [
    (14, ["qwen2.5:3b", "qwen2.5-coder:7b"], "cpu · modest (≥16GB, 7b rung is slow)"),
    (7, ["qwen2.5:1.5b", "qwen2.5:3b"], "cpu · light (≈8GB)"),
    (0, ["qwen2.5:1.5b"], "cpu · minimal"),
]


def recommend(ram_gb, hw=None):
    """RAM-tiered ladder for this machine. With `hw` (a detect_machine dict), a
    machine with no fast memory path gets the CPU-capped table — never a ladder
    whose escalation rung would swap-thrash it."""
    table = TIERS if (hw is None or _is_accelerated(hw)) else CPU_TIERS
    for floor, ladder, label in table:
        if ram_gb >= floor:
            return ladder, label
    return table[-1][1], table[-1][2]


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
        rows = [ln for ln in out.splitlines()[1:] if ln.split()]   # skip blank lines (else split()[0] IndexErrors)
        return any(line.split()[0].split(":")[0] == base and name.split(":")[-1] in line for line in rows) \
            or any(name == line.split()[0] for line in rows)
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
    print("\n  Which server? (vllm / llamacpp / mlx / lmstudio / tgi / sglang / openai / anthropic)")
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
        rc = _setup_ollama(hw, auto, models or keep_models)
    else:
        rc = _setup_server(hw, engine, url, models, api_key)
    if rc == 0:
        _offer_company(auto)
    return rc


def _offer_company(auto):
    """After a model is set up, offer to charter a default company — so a first-time user
    comes out of setup with an ORG ready, not just a single agent. Uses the configured
    ladder: the strongest rung manages, a cheaper rung works. Skipped in --auto (no prompts)
    and if a 'starter' company already exists."""
    from . import company, config
    from .render import paint
    if company.charter_exists("starter"):
        return
    ladder = config.load().get("ladder", []) or []
    if not ladder:
        return
    manager = ladder[-1]                              # strongest rung manages
    workers = [ladder[0], ladder[0]]                 # two cheap workers
    if not auto:
        try:
            ans = input(paint("\n  Charter a starter company (a manager + 2 workers)? [Y/n] ", "cyan")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if ans in ("n", "no"):
            return
    try:
        company.create_charter("starter", manager, workers)
    except Exception:
        return
    print(paint(f"  ✓ chartered company 'starter'", "green")
          + paint(f"   — manager {manager} · workers {ladder[0]} ×2", "dim"))
    print(paint("    put it to work:  forge company run starter \"<a goal>\"", "dim"))
    print(paint("    watch it live:   forge company watch starter", "dim"))


def _setup_ollama(hw, auto, keep_models):
    ok, msg = _ollama_ok()
    if not ok:
        print(f"  ✗ {msg}")
        return 1
    ladder, label = recommend(hw["ram_gb"], hw)
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
    if not auto:
        _probe_ladder(ladder, engine="ollama")
    _bridge_report()
    print(f"  ✓ run `forge` to start (ladder: {' → '.join(ladder)})")
    return 0


def passport(backend, verbose=True):
    """P5.8 active probe: drive ~10 canned micro-prompts through a REAL backend and
    write the initial passport, so a fresh install is tuned to this model BEFORE its
    first task. Scores format-holding / field completeness / exact-text reproduction
    (the scoring is offline in profile.score_probe). Best-effort — a probe against an
    unreachable or slow engine degrades to no passport, never an error. Returns the
    scores dict (or None on total failure)."""
    from . import profile
    from .tools import ACTION_SCHEMA
    specs = profile.probe_specs()
    sysmsg = ("You are a coding agent. Reply to each instruction with ONE JSON action "
              "object and nothing else.")
    raws = []
    for spec in specs:
        try:
            raw = backend.chat(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content": spec["prompt"]}],
                schema=ACTION_SCHEMA, temperature=0.0)
        except Exception:
            raw = ""
        raws.append(raw or "")
    scores = profile.score_probe(raws, specs)
    profile.write_passport(backend.name, scores)
    if verbose:
        print(f"    · {backend.name}: format {scores['format_hold']:.0%} · "
              f"fields {scores['field_complete']:.0%} · exact {scores['exact_repro']:.0%}")
    return scores


def _probe_ladder(ladder, engine="ollama", base_url="", api_key=""):
    """Probe every rung of a freshly-configured ladder and write its passport.
    Best-effort and self-contained; a single unreachable rung just prints a note."""
    from .backends import make_backend
    print("\n  measuring models (passport probe — tunes the harness per model):")
    for m in ladder:
        try:
            passport(make_backend(m, engine=engine, base_url=base_url, api_key=api_key))
        except Exception:
            print(f"    · {m}: probe skipped (backend not reachable)")


def _bridge_report():
    """Claude Code interop: check what's on this machine, prepare what's safe."""
    from . import bridge
    print("\n  Claude Code fleet interop:")
    for ln in bridge.doctor():
        print(f"    {ln}")
    print()


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
    _probe_ladder(models, engine=engine, base_url=url, api_key=api_key or "")
    _bridge_report()
    print(f"  ✓ run `forge` to start (make sure your server is up)")
    return 0
