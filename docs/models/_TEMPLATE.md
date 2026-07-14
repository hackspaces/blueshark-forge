<!--
Copy this file to docs/models/<model>.md, fill EVERY section, and add a row to README.md.
Delete the sections that genuinely don't apply, but keep the report honest — caveats and
failures included. A report that only shows wins isn't useful.
-->

# <model> — forge model report

> One-line thesis: what's interesting about running this model through forge.

| | |
|---|---|
| **Model** | [<org>/<model>](https://huggingface.co/<org>/<model>) |
| **Architecture** | <arch> (dense / MoE / …) |
| **Quant tested** | <e.g. Q4_K_M — size, single-file?> |
| **Runtime** | <llama.cpp bXXXX / vLLM / Ollama / …> — note any arch-support gotcha |
| **Harness** | forge <version> |
| **Host** | <hardware, RAM/VRAM> |
| **Throughput** | ~<n> tok/s prompt · ~<n> tok/s generation |
| **Reasoning model** | yes / no |
| **Agentic run** | ✅ / ⚠️ / ❌ — one line |
| **Bench (base vs harness)** | fill from `forge bench`, or ⏳ pending |

---

## Why this model is interesting

## Getting it running (reproduce)

Numbered, copy-pasteable steps: runtime → weights → serve → point forge at it.

## Base vs harnessed (`forge bench`)

The lift table — bare weights vs. full harness, plus the per-lever ablation. This is the
point of the collection; if it's pending, say so and give the exact command to fill it.

## Does it actually work? (real run + honest caveats)

The real transcript, throughput, quirks (reasoning model? constrained-decoding behavior?),
and an honest list of what you stopped short of or didn't test.

## What it demonstrates
