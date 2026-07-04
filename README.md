# blueshark-forge

A model-agnostic agentic runtime for the terminal. Any model, frontier or a
small local one, becomes a capable agent, because the intelligence lives in the
harness, not the weights. And every forge session is part of a fleet: they verify
each other's work, coordinate, and share what they learn.

Not tied to any vendor. Runs on your machine, on your models.

## Quick start

```bash
# 1. install Ollama (https://ollama.com) and make sure it's running
# 2. install forge
pipx install blueshark-forge          # or: pip install blueshark-forge

# 3. let it configure itself for THIS machine
#    (detects your RAM/chip, picks a model ladder, pulls the models, writes config)
forge setup

# 4. go — open it in any repo and talk to it
cd your-project
forge
```

`forge setup` sizes the model ladder to your hardware automatically (e.g. 48GB
Apple Silicon → `qwen3-coder:30b → qwen3.6:35b-a3b`). Switch models any time from
the TUI with `/model`. Everything runs locally.

## Why

Claude Code, Codex, and the rest are excellent, but each locks you to one
provider's harness. forge is the harness itself, opened up: point it at Gemma,
Qwen, your own model, or a frontier API, and you get the same agentic loop, tools,
and multi-agent fabric.

The bet: move the agentic scaffolding out of the model's weights and into the
harness, and even a 9B becomes a real agent. The levers:

- **Constrained decoding** — every model output is grammar-forced to a valid tool
  call (Ollama `format` schema). A small model literally cannot emit a malformed
  call.
- **Bounded steps** — the harness holds the loop; the model does one thing per turn.
- **Loop detection** — repeated no-progress actions are broken automatically.
- **Autonomy scaffolding** — task mode tells the model to act, not ask.
- **Verify-on-done** — a claim of "done" is checked, never trusted.

**Workspace + computer awareness** (like a real coding assistant): on start, forge
builds a gitignore-aware map of the project, detects the language/project type,
reads the git state, and learns the machine it's on (OS, shell, tool versions), all
pinned into context. Say "fix the auth bug" or "read this @file" and it already
knows where things are. It also inherits whatever the fleet has learned about the repo.

**Frontier agent loop**: a living plan (todo list the agent maintains and the
harness pins each turn), surgical `edit_file` (not fragile full rewrites),
self-correction (failed actions are flagged so the model diagnoses), loop-breaking,
and context compaction for long sessions.

**Local model router (escalation ladder)**: `--model a,b,c` is a ladder of local
models, cheapest first. forge runs on the fast one and, when it detects it's stuck
(the same command failing repeatedly), automatically escalates to a stronger LOCAL
model with full context and keeps going — no cloud, no vendor. The default is
`gemma2:9b → qwen2.5-coder:7b → qwen3.6`. Threshold tunable via FORGE_STUCK_THRESHOLD.
This is the whole "local can be enough" bet: a smart harness routing across small
models beats one big call for most work, and stays on your machine.

**Alive terminal**: a spinner while it thinks, a live plan panel, and clean per-step
rendering with timing and pass/fail.

Proven: Gemma-9B, fully local, autonomously fixes a multi-bug repo through forge
(read → fix → run tests → confirm). The reliability tracks task crispness — a
clear verification signal (tests) makes small models solid; open-ended judgement
still wants a bigger model, which is why the fleet's verifier routes to one.

## Use

```
forge                          chat with an agent in the cwd (default model)
forge --model gemma2:9b        pick any Ollama model, or openai:model@url
forge run "<task>"             one-shot: run a task to completion, autonomous
forge status                   autopilot state + live sessions
```

The fleet (multi-agent) layer — native, because forge owns its own sessions:

```
forge up                       start the autopilot (TRUST + COORDINATE + LEARN)
forge down                     stop it
forge send <target> <msg>      message another session (it absorbs it mid-work)
forge receipts                 trust audit trail — verdicts on "done" claims
forge learnings [dir]          durable facts learned in a repo
```

## Architecture

```
forge (one per terminal)
  repl / run  →  agent loop (the harness brain)
     · backend:  any model (Ollama · OpenAI-compatible · your own)
     · tools:    bash / read_file / write_file / list_files
     · levers:   constrain · bounded steps · loop-break · autonomy
     · session:  transcript + registry + native inbox
        │  many forge sessions
        ▼
forged (the fleet autopilot, native to forge)
     TRUST      independent verifier agent disproves "done" claims (routes to
                a capable model; read-only, cannot edit what it judges)
     COORDINATE warns two sessions editing the same file
     LEARN      harvests durable repo facts, shares them across sessions
     MESSAGE    session-to-session, via each session's inbox
```

Because forge owns the transcript format, the registry, and the inbox, the fleet
is built in, no external channel API, no reading someone else's logs. This is the
same fleet system first prototyped on Claude Code, now native and vendor-free.

## Layout
```
~/forge/forge/
  backends.py   model-agnostic backends
  tools.py      tools + the constrained action schema
  session.py    transcript · registry · inbox · ephemeral sessions
  agent.py      the agent loop (harness brain) + levers
  repl.py       interactive chat
  fleet.py      verify · coordinate · learn · message primitives
  daemon.py     forged — the autopilot loop
  __main__.py   the CLI
~/.forge/       runtime: sessions/ · registry.json · learn/ · verdicts.jsonl
```

## Status

Working: the agentic terminal (chat + run), model-agnostic backends, the harness
levers, and the native fleet layer (send/verify/guard/learn + daemon). Verifier
routes to a capable local model (qwen-coder class) for reliable checking.

Next: streaming output, richer TUI, tool sandboxing, more backends (vLLM/MLX,
Anthropic), and training a model native to forge's protocol — the flywheel where
forge's own trajectories teach the model to be best *in forge*.
