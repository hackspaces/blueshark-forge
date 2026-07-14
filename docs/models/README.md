# forge model reports

An open, growing record of how language models behave when they're driven by **forge** — not just *whether* they run locally, but how much the **harness** changes what they can do.

forge's premise is that **the intelligence lives in the harness, not the weights**. This collection is the evidence. For each model we publish two things:

1. **How to actually run it** locally through forge — including the models the one-line tools refuse.
2. **How it does bare vs. harnessed** — the same weights with no scaffolding versus the full forge loop, measured with `forge bench` (base vs. harness + per-lever ablation).

## Reports

| Model | Architecture | Runtime | Highlight | Report |
|---|---|---|---|---|
| **sarvam-30b** | sparse MoE (`sarvam_moe`) | llama.cpp `b9960` (Metal) | a real agentic loop on a model stock Ollama can't even load | [→](sarvam-30b.md) |

## Why base-vs-harness is the number that matters

A bare model call answers a prompt. A harness gives the same weights memory, tools, verification, record/replay, constrained output, and crash recovery. The interesting figure isn't a benchmark score in the abstract — it's the **lift**: how much better the identical weights get once the harness is doing its job.

`forge bench` measures exactly that, and ablates it lever by lever, so the lift is *attributable* rather than magic:

```sh
FORGE_REMOTE_CTX=<server-ctx> \
forge bench --model "openai:<name>@http://127.0.0.1:8080/v1"
```

## Add a model report

1. Get the model running through forge — see any existing report for the pattern (usually: a runtime that knows the arch → a GGUF → `forge --model openai:<name>@<url>`).
2. Run `forge bench` for the base-vs-harness numbers and the per-lever ablation.
3. Copy [`_TEMPLATE.md`](_TEMPLATE.md) to `docs/models/<model>.md`, fill it in, and add a row to the table above.
4. **Keep it honest** — record caveats, failures, throughput, and what you *didn't* test. A report that only shows wins isn't useful.

---

*This collection is plain Markdown (readable straight from the repo) and Pages-ready. To serve it as an open website, enable **Settings → Pages → source: `main` / `docs`**.*
