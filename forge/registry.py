"""Model catalog — what forge can run, and what THIS machine can run.

Two jobs:
  1. Answer the question nobody has a clean tool for: "what open models can my
     computer actually run?" — `forge models` lists a spread of good open models
     from ~0.5B to ~70B, checked against this machine's RAM + whether it has a fast
     memory path (Apple Silicon / a real GPU) or is CPU-only.
  2. Run them — most entries are Ollama-native, so `forge models use <name>` is
     one command. A few carry hand-verified recipes for models the easy tools can't
     load (sarvam-30b's custom arch, BitNet's 1-bit runtime).

HONESTY:
  * The RAM figure and the runs-well/slow verdict are ESTIMATES (weights + overhead
    vs total RAM; speed from size + hardware class). A real "run it" is one command.
  * status "verified" means forge RAN this model on record (a docs/models report
    exists). "candidate" means a known-good model we haven't personally benched here
    — a fit estimate, not a promise.

A Python module (not JSON) on purpose: ships with the package automatically, diffs
cleanly in review, needs no loader.
"""

# ram_gb_needed = recommended TOTAL system RAM (weights + OS headroom + KV), an estimate.
# params_b drives the size sort + the "runs up to ~NB" ceiling + the CPU speed verdict.
MODELS = [
    # ---- the verified / special-runtime models --------------------------------
    {
        "name": "sarvam-30b", "repo": "sarvamai/sarvam-30b", "arch": "SarvamMoEForCausalLM",
        "kind": "moe", "params_b": 30, "engine": "llamacpp",
        "arch_probe": "sarvam-moe", "min_version": "b9960",
        "weights": {"gguf_repo": "Sumitc13/sarvam-30b-GGUF",
                    "file": "sarvam-30B-Q4_K_M.gguf", "quant": "Q4_K_M", "size_gb": 19.58},
        "serve": {"ctx": 16384, "ngl": 999, "jinja": True, "port": 8080},
        "ram_gb_needed": 22, "reasoning": True, "status": "verified",
        "report": "docs/models/sarvam-30b.md", "lift_pts": None,
        "notes": "India's open MoE. Stock Ollama can't load this arch — needs a llama.cpp "
                 "with sarvam-moe compiled in (forge does it for you).",
    },
    {
        "name": "bitnet-b1.58-2b", "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "arch": "BitNetForCausalLM", "kind": "1bit", "params_b": 2, "engine": "bitnet.cpp",
        "weights": {"file": "ggml-model-i2_s.gguf", "quant": "i2_s", "size_gb": 1.2},
        "ram_gb_needed": 3, "reasoning": False, "status": "candidate", "report": None, "lift_pts": None,
        "notes": "1-bit — purpose-built for fast CPU inference on low-memory machines. "
                 "Needs the bitnet.cpp runtime.",
    },
    {
        "name": "phi-2", "repo": "microsoft/phi-2", "arch": "PhiForCausalLM", "kind": "dense",
        "params_b": 2.7, "engine": "ollama", "ollama_tag": "phi",
        "weights": {"quant": "Q4_0", "size_gb": 1.6}, "ram_gb_needed": 4,
        "reasoning": False, "status": "candidate", "report": None, "lift_pts": None,
        "notes": "Small, CPU-friendly, standard arch — Ollama pulls it directly.",
    },

    # ---- the spread of good open models (Ollama-native → one-command runnable) --
    {"name": "qwen2.5:0.5b", "repo": "Qwen/Qwen2.5-0.5B-Instruct", "arch": "qwen2", "kind": "dense",
     "params_b": 0.5, "engine": "ollama", "ollama_tag": "qwen2.5:0.5b",
     "weights": {"quant": "Q4_K_M", "size_gb": 0.4}, "ram_gb_needed": 2,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Tiny — runs on almost anything."},
    {"name": "llama3.2:1b", "repo": "meta-llama/Llama-3.2-1B-Instruct", "arch": "llama", "kind": "dense",
     "params_b": 1, "engine": "ollama", "ollama_tag": "llama3.2:1b",
     "weights": {"quant": "Q4_K_M", "size_gb": 0.8}, "ram_gb_needed": 3,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Tiny general model, fast on CPU."},
    {"name": "qwen2.5:1.5b", "repo": "Qwen/Qwen2.5-1.5B-Instruct", "arch": "qwen2", "kind": "dense",
     "params_b": 1.5, "engine": "ollama", "ollama_tag": "qwen2.5:1.5b",
     "weights": {"quant": "Q4_K_M", "size_gb": 1.0}, "ram_gb_needed": 3,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Small, capable, quick."},
    {"name": "gemma2:2b", "repo": "google/gemma-2-2b-it", "arch": "gemma2", "kind": "dense",
     "params_b": 2, "engine": "ollama", "ollama_tag": "gemma2:2b",
     "weights": {"quant": "Q4_K_M", "size_gb": 1.6}, "ram_gb_needed": 4,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Strong for its size (Google)."},
    {"name": "llama3.2:3b", "repo": "meta-llama/Llama-3.2-3B-Instruct", "arch": "llama", "kind": "dense",
     "params_b": 3, "engine": "ollama", "ollama_tag": "llama3.2:3b",
     "weights": {"quant": "Q4_K_M", "size_gb": 2.0}, "ram_gb_needed": 5,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "The sweet spot for an 8GB machine."},
    {"name": "qwen2.5-coder:7b", "repo": "Qwen/Qwen2.5-Coder-7B-Instruct", "arch": "qwen2", "kind": "dense",
     "params_b": 7, "engine": "ollama", "ollama_tag": "qwen2.5-coder:7b",
     "weights": {"quant": "Q4_K_M", "size_gb": 4.7}, "ram_gb_needed": 8,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Strong small coding model — great with forge."},
    {"name": "mistral:7b", "repo": "mistralai/Mistral-7B-Instruct-v0.3", "arch": "mistral", "kind": "dense",
     "params_b": 7, "engine": "ollama", "ollama_tag": "mistral:7b",
     "weights": {"quant": "Q4_K_M", "size_gb": 4.4}, "ram_gb_needed": 8,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Popular, well-rounded 7B."},
    {"name": "llama3.1:8b", "repo": "meta-llama/Llama-3.1-8B-Instruct", "arch": "llama", "kind": "dense",
     "params_b": 8, "engine": "ollama", "ollama_tag": "llama3.1:8b",
     "weights": {"quant": "Q4_K_M", "size_gb": 4.9}, "ram_gb_needed": 8,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "The default general 8B a lot of people run."},
    {"name": "gemma2:9b", "repo": "google/gemma-2-9b-it", "arch": "gemma2", "kind": "dense",
     "params_b": 9, "engine": "ollama", "ollama_tag": "gemma2:9b",
     "weights": {"quant": "Q4_K_M", "size_gb": 5.4}, "ram_gb_needed": 10,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Punches above its size; wants ~16GB."},
    {"name": "qwen2.5:14b", "repo": "Qwen/Qwen2.5-14B-Instruct", "arch": "qwen2", "kind": "dense",
     "params_b": 14, "engine": "ollama", "ollama_tag": "qwen2.5:14b",
     "weights": {"quant": "Q4_K_M", "size_gb": 9.0}, "ram_gb_needed": 12,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Mid-size general; needs 16GB+ to be comfortable."},
    {"name": "gemma2:27b", "repo": "google/gemma-2-27b-it", "arch": "gemma2", "kind": "dense",
     "params_b": 27, "engine": "ollama", "ollama_tag": "gemma2:27b",
     "weights": {"quant": "Q4_K_M", "size_gb": 16.0}, "ram_gb_needed": 20,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Big, strong — a 32GB+ / Apple-Silicon model."},
    {"name": "qwen2.5-coder:32b", "repo": "Qwen/Qwen2.5-Coder-32B-Instruct", "arch": "qwen2", "kind": "dense",
     "params_b": 32, "engine": "ollama", "ollama_tag": "qwen2.5-coder:32b",
     "weights": {"quant": "Q4_K_M", "size_gb": 20.0}, "ram_gb_needed": 24,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Top open coding model — the one to run with forge if you can."},
    {"name": "mixtral:8x7b", "repo": "mistralai/Mixtral-8x7B-Instruct-v0.1", "arch": "mixtral", "kind": "moe",
     "params_b": 47, "engine": "ollama", "ollama_tag": "mixtral:8x7b",
     "weights": {"quant": "Q4_K_M", "size_gb": 26.0}, "ram_gb_needed": 32,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "MoE — big footprint, but fast per token for its size."},
    {"name": "llama3.1:70b", "repo": "meta-llama/Llama-3.1-70B-Instruct", "arch": "llama", "kind": "dense",
     "params_b": 70, "engine": "ollama", "ollama_tag": "llama3.1:70b",
     "weights": {"quant": "Q4_K_M", "size_gb": 40.0}, "ram_gb_needed": 48,
     "status": "candidate", "report": None, "lift_pts": None, "notes": "Frontier-class open weights — wants a workstation / high-RAM Mac."},
]


def get(name):
    for m in MODELS:
        if m["name"] == name:
            return m
    return None


def names():
    return [m["name"] for m in MODELS]


def fits(entry, ram_gb):
    """RAM ESTIMATE: does the machine have the recommended RAM? An estimate — other
    apps shrink real headroom; never presented as a guarantee."""
    need = entry.get("ram_gb_needed") or 0
    return bool(ram_gb) and ram_gb >= need


def runs(entry, ram_gb, accelerated):
    """How this model would run on this machine: (verdict, style). Factors RAM fit AND
    speed — a big model on a CPU-only machine 'fits' but crawls (decode is memory-
    bandwidth-bound). accelerated = Apple Silicon (Metal) or a real GPU."""
    if not fits(entry, ram_gb):
        return "won't fit", "faint"
    p = entry.get("params_b") or 0
    if accelerated or p <= 3.5:
        return "runs well", "good"
    if p <= 9:
        return "usable · slower", "warn"
    return "fits · slow on CPU", "warn"


def ceiling(ram_gb, accelerated):
    """The biggest model (by params) this machine can run WELL, for the headline.
    Returns (params_b, name) or (0, None) if nothing runs well."""
    ok = [m for m in MODELS if runs(m, ram_gb, accelerated)[0] == "runs well"]
    if not ok:
        return 0, None
    best = max(ok, key=lambda m: m.get("params_b") or 0)
    return best.get("params_b") or 0, best["name"]


def runbook(entry):
    """The copy-pasteable recipe for an entry, as printable lines. Verified entries
    reproduce their docs/models report; candidates are labeled as unproven."""
    lines = []
    eng = entry.get("engine")
    if eng == "llamacpp":
        w = entry["weights"]
        s = entry["serve"]
        lines += [
            "# 1. runtime — verify the arch is compiled in BEFORE downloading:",
            "brew install llama.cpp",
            f"strings \"$(brew --prefix)/lib/libllama.0.dylib\" | grep -i {entry['arch_probe']}",
            f"#    -> must print '{entry['arch_probe']}' (needs >= {entry['min_version']})",
            "",
            "# 2. weights (single-file, resumable):",
            f"mkdir -p ~/models/{entry['name']} && cd ~/models/{entry['name']}",
            f"curl -L -C - --retry 5 -o {w['file']} \\",
            f"  \"https://huggingface.co/{w['gguf_repo']}/resolve/main/{w['file']}\"",
            "",
            "# 3. serve (OpenAI-compatible):",
            f"llama-server -m ~/models/{entry['name']}/{w['file']} \\",
            f"  --host 127.0.0.1 --port {s['port']} -c {s['ctx']} -ngl {s['ngl']}"
            + (" --jinja" if s.get("jinja") else "") + f" --alias {entry['name']}",
            "",
            "# 4. point forge at it (ctx must match the server):",
            f"FORGE_REMOTE_CTX={s['ctx']} forge --model "
            f"\"openai:{entry['name']}@http://127.0.0.1:{s['port']}/v1\" run \"<task>\"",
        ]
    elif eng == "ollama":
        lines += [
            "# 1. pull (Ollama must be installed):",
            f"ollama pull {entry['ollama_tag']}",
            "",
            "# 2. run through forge:",
            f"forge --model {entry['ollama_tag']} run \"<task>\"",
        ]
    else:                                        # bespoke runtime (e.g. bitnet.cpp)
        lines += [
            f"# bespoke runtime: {eng}",
            "# build github.com/microsoft/BitNet (bitnet.cpp), serve its OpenAI-compatible",
            "# endpoint, then: forge --model \"openai:<alias>@http://127.0.0.1:<port>/v1\"",
        ]
    if entry.get("status") != "verified":
        lines += ["", "# ⚠ candidate recipe — a known-good model, but not personally",
                  "#   benched here. Treat the fit/speed as an estimate, not a guarantee."]
    return lines
