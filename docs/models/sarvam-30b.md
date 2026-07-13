# sarvam-30b — forge model report

> Running India's open MoE through the forge agentic harness by owning the one layer that was missing — the runtime. A model **stock Ollama can't load**, driving a real read/write/verify loop with **zero model-specific code**.
>
> Companion web page: <https://claude.ai/code/artifact/fa9f18e0-a341-4e3a-a021-173baa70fdc0>

| | |
|---|---|
| **Model** | [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) |
| **Architecture** | `SarvamMoEForCausalLM` — sparse MoE: 1 dense + 18 MoE layers, 128 routed experts, top-6, +1 shared |
| **Quant tested** | Q4_K_M — 19.58 GB, single-file ([Sumitc13/sarvam-30b-GGUF](https://huggingface.co/Sumitc13/sarvam-30b-GGUF)) |
| **Runtime** | llama.cpp `b9960` (Homebrew, Metal) — has `sarvam_moe` compiled in; stock Ollama does not |
| **Harness** | forge 0.8.3, OpenAI-compatible engine |
| **Host** | Apple Silicon, 48 GB unified memory |
| **Throughput** | ~185 tok/s prompt · ~65–69 tok/s generation |
| **Reasoning model** | Yes — emits `reasoning_content` + `content` |
| **Agentic run** | ✅ correct multi-file task via valid action-JSON |
| **Bench (base vs harness)** | ⏳ pending a full `forge bench` run (see note below) |

---

## Why this model is interesting

Most "run model X locally" guides assume your runtime already knows the architecture. The moment a lab ships something genuinely new, that assumption breaks — and you learn how much of the stack you actually control.

sarvam-30b is exactly that case. Because `sarvam_moe` is a custom architecture, `convert_hf_to_gguf` and any runtime built before support landed reject it with *"architecture not supported."* Ollama's tracking issue for these models ([#14319](https://github.com/ollama/ollama/issues/14319)) is still open, and its sharded official GGUF can't be pulled the usual way either.

The interesting question isn't "can I download it" — the quantized weights are on the Hub. It's: **when the easy button doesn't work, how thin is the layer you have to build yourself?** Here it was one layer — the runtime — and forge sat on top unchanged.

## Getting it running (reproduce)

### 1. A runtime that understands `sarvam_moe`

```sh
brew install llama.cpp        # prebuilt, Metal-enabled — no compile
strings "$(brew --prefix)/lib/libllama.0.dylib" | grep -i sarvam
# -> sarvam-moe        (this line is the whole gate — check it BEFORE downloading 20 GB)
```

If that prints `sarvam-moe`, the GGUF will load. If not, your build predates the support — update, or build mainline from source.

### 2. The weights (single-file matters)

```sh
mkdir -p ~/models/sarvam-30b && cd ~/models/sarvam-30b
curl -L -C - --retry 5 -o sarvam-30B-Q4_K_M.gguf \
  "https://huggingface.co/Sumitc13/sarvam-30b-GGUF/resolve/main/sarvam-30B-Q4_K_M.gguf"
```

`-C -` makes it resumable. A **single-file** GGUF is the point: Ollama can't pull the *sharded* official repo, which is the first wall most people hit.

### 3. Serve it — OpenAI-compatible on :8080

```sh
llama-server \
  -m ~/models/sarvam-30b/sarvam-30B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 16384 -ngl 999 --jinja --alias sarvam-30b
```

`-ngl 999` → all layers on Metal · `--jinja` → sarvam's own chat template · `-c 16384` → small KV cache.

### 4. Point forge at it — no config change

```sh
FORGE_REMOTE_CTX=16384 \
forge --model "openai:sarvam-30b@http://127.0.0.1:8080/v1" run "<your task>"
```

**The one gotcha:** forge assumes a 128k window and only compacts near it, but we capped the server at 16k. `FORGE_REMOTE_CTX=16384` tells forge the *real* window so it compacts in time instead of overflowing the server. Match the two numbers.

## Base vs harnessed (`forge bench`)

> ⏳ **Pending.** The whole point of this collection is the *lift* — the same weights bare vs. wrapped in the full harness, measured by `forge bench` with per-lever ablation. This first entry was cut short before the bench run (a ~30B MoE on a laptop draws real power). The qualitative agentic result below is real and verified; the quantitative base-vs-harness table will be filled from:
>
> ```sh
> FORGE_REMOTE_CTX=16384 forge bench --model "openai:sarvam-30b@http://127.0.0.1:8080/v1"
> ```

## Does it actually work? (real run + honest caveats)

**It loads and serves on Metal.** llama-server (`b9960-a935fbffe`) recognized the arch, loaded the 19.6 GB MoE in ~4.5 s, and served at ~185 tok/s prompt / ~65–69 tok/s generation. No "unsupported architecture."

**sarvam-30b is a reasoning model.** A raw completion emits chain-of-thought into `reasoning_content` and the answer into `content`:

```
prompt:    "What is the capital of India? Answer in one short sentence."
reasoning: "1. Analyze the user's request... 2. Identify the core information needed..."  (1852 chars)
content:   "The capital of India is New Delhi."
finish:    stop   (484 completion tokens for a one-line answer)
```

forge reads `message.content` (the answer, not the scratchpad) and doesn't cap `max_tokens`, so the model finishes reasoning and then acts — no special-casing.

**forge drove it through a real agentic task.** Task: *write `greet.py` with a `greet(name)` function, write `test_greet.py` with an assertion, run the test.* The loop turned cleanly — each step the model reasoned (~400 tokens) and emitted a valid forge action:

```
· write_file: greet.py        [ok]
· write_file: test_greet.py    [ok]
```

What it wrote — correct f-string, import, and assertion:

```python
# greet.py
def greet(name):
    return f"Hello, {name}!"

# test_greet.py
from greet import greet
assert greet('Sarvam') == 'Hello, Sarvam!'
```

`python3 test_greet.py` **passes**.

**Honest caveats:**

- The run was stopped **before** forge's own final "run the test" step (power). The artifacts are correct; the test was verified independently (no model required).
- It's **not fast, and it's token-hungry** — ~65 tok/s and a few hundred reasoning tokens per step means each agentic step is ~10 s. Fine for a demo, slow for a tight loop.
- The **reasoning-model + JSON-grammar combination held**: forge settled on `response_format` constrained decoding on the first call, and the model reasoned first, then produced valid action-JSON.

## What it demonstrates

None of these pieces is individually exotic. The point is the *shape*: when a new model outran the convenient tooling, we didn't need a new agent, a new protocol, or a line of model-specific code — we needed to own **one** layer (the runtime), and the harness above it never cared which weights answered. That's the property worth having as open models proliferate faster than any single tool can keep up.

---

*Verified on macOS · Apple Silicon · 48 GB · llama.cpp b9960 · forge 0.8.3. Your build numbers will differ; the `strings … | grep sarvam` check is the part that matters.*
