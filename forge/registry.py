"""Curated model registry — the recipes forge actually knows how to run.

Each entry is the machine-readable form of a docs/models report: which engine runs
this model, where the loadable weights come from, the serve flags, and how much
memory it really needs. `forge models` lists them against THIS machine's RAM;
`forge models show <name>` prints the runbook.

HONESTY RULES (from the catalog design review):
  * status "verified" means forge RAN this model end-to-end on record (a docs/models
    report exists). status "candidate" means the recipe is researched but not yet
    proven here — shown as a runbook to check, never a guarantee.
  * Curated stays SMALL and hand-verified. The long tail of HF is not mirrored here;
    an entry is added by running the model, not by scraping metadata.

A Python module (not JSON) on purpose: ships with the package automatically
(pyproject packages only `forge/`), diffs cleanly in review, needs no loader.
"""

MODELS = [
    {
        "name": "sarvam-30b",
        "repo": "sarvamai/sarvam-30b",
        "arch": "SarvamMoEForCausalLM",          # custom MoE: 128 experts, top-6, sigmoid routing
        "kind": "moe",
        "engine": "llamacpp",
        "arch_probe": "sarvam-moe",              # what `strings libllama | grep` must print
        "min_version": "b9960",                  # the build verified to carry the arch
        "weights": {"gguf_repo": "Sumitc13/sarvam-30b-GGUF",
                    "file": "sarvam-30B-Q4_K_M.gguf", "quant": "Q4_K_M", "size_gb": 19.58},
        "serve": {"ctx": 16384, "ngl": 999, "jinja": True, "port": 8080},
        "ram_gb_needed": 22,
        "reasoning": True,                       # emits reasoning_content + content
        "status": "verified",                    # ran a real agentic task; test passed
        "report": "docs/models/sarvam-30b.md",
        "lift_pts": None,                        # forge bench pending (see report)
        "notes": "Stock Ollama cannot load this arch (sharded official GGUF + custom "
                 "sigmoid-MoE routing). Needs a llama.cpp with sarvam-moe compiled in — "
                 "check the arch_probe BEFORE downloading 20GB.",
    },
    {
        "name": "phi-2",
        "repo": "microsoft/phi-2",
        "arch": "PhiForCausalLM",
        "kind": "dense",
        "engine": "ollama",
        "ollama_tag": "phi",                     # ollama pull phi == phi-2 2.7B
        "weights": {"quant": "Q4_0", "size_gb": 1.6},
        "ram_gb_needed": 4,
        "reasoning": False,
        "status": "candidate",                   # recipe researched; not yet run through forge here
        "report": None,
        "lift_pts": None,
        "notes": "The easy case: standard arch, Ollama-native, CPU-friendly (2.7B). "
                 "The cheap rung for the CPU_TIERS band.",
    },
    {
        "name": "bitnet-b1.58-2b",
        "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "arch": "BitNetForCausalLM",
        "kind": "1bit",                          # NATIVE ternary training — not a quantized FP model
        "engine": "bitnet.cpp",                  # bespoke runtime (github.com/microsoft/BitNet)
        "weights": {"file": "ggml-model-i2_s.gguf", "quant": "i2_s", "size_gb": 1.2},
        "ram_gb_needed": 3,
        "reasoning": False,
        "status": "candidate",
        "report": None,
        "lift_pts": None,
        "notes": "Purpose-built for fast CPU inference on low-memory machines — the model "
                 "class that matters most for budget hardware. Needs bitnet.cpp built from "
                 "source (stock llama.cpp/Ollama do NOT run its i2_s kernels efficiently).",
    },
]


def get(name):
    """The entry named `name`, or None."""
    for m in MODELS:
        if m["name"] == name:
            return m
    return None


def names():
    return [m["name"] for m in MODELS]


def fits(entry, ram_gb):
    """Honest RAM ESTIMATE (weights + KV + overhead vs total RAM). An estimate —
    other running apps shrink the real headroom; never presented as a guarantee."""
    need = entry.get("ram_gb_needed") or 0
    return bool(ram_gb) and ram_gb >= need


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
        lines += ["", "# ⚠ candidate recipe — researched but NOT yet verified through forge",
                  "#   on this machine. Treat as a runbook to check, not a guarantee."]
    return lines
