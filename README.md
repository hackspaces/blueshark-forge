# blueshark-forge

A model-agnostic agentic runtime for the terminal. Any model, frontier or a
small local one, becomes a capable agent, because the intelligence lives in the
harness, not the weights. And every forge session is part of a fleet: they verify
each other's work, coordinate, and share what they learn.

Not tied to any vendor. Runs on your machine, on your models.

[![PyPI](https://img.shields.io/pypi/v/blueshark-forge)](https://pypi.org/project/blueshark-forge/)

---

## Install

**Requirements:** Python 3.10+ and an inference engine (Ollama is the easy default).

```bash
# 1. install forge
pipx install blueshark-forge          # recommended (isolated); or: pip install blueshark-forge

# 2. install an engine to run models locally — Ollama is the simplest
#    macOS/Linux:  https://ollama.com  (download, then it runs in the background)
#    check it's up:  ollama --version
```

## Set up (once per machine)

```bash
forge setup
```

This inspects your machine and configures forge for it:
- detects your **RAM / chip / cores**,
- picks a **model ladder** sized to your hardware (e.g. 8GB → a 3B; 16GB → 9B;
  48GB Apple Silicon → `qwen3-coder:30b → qwen3.6`),
- **pulls those models** via Ollama,
- sizes the **context window** to your RAM,
- writes it all to `~/.forge/config.json`.

Non-interactive: `forge setup --auto`.

### Using something other than Ollama

forge speaks the OpenAI-compatible protocol that **vLLM, llama.cpp, MLX, LM Studio,
TGI, SGLang, and cloud APIs** all serve — great for a workstation/cluster or remote
inference. Choose it interactively in `forge setup`, or configure directly:

```bash
# point at a vLLM server (or any OpenAI-compatible endpoint)
forge setup --engine vllm \
  --url http://your-server:8000/v1 \
  --models "Qwen/Qwen2.5-Coder-32B-Instruct"

# a cloud API
forge setup --engine openai --url https://api.openai.com/v1 \
  --api-key sk-... --models "gpt-4o-mini,gpt-4o"
```

Engines: `ollama` (default) · `vllm` · `llamacpp` · `mlx` · `lmstudio` · `tgi` ·
`sglang` · `openai`. Set `OPENAI_API_KEY` in your env instead of `--api-key` if you prefer.

## Use it

```bash
cd your-project
forge                       # interactive chat, oriented in this repo
```

Then just talk to it — it already knows your files, git state, and machine:

```
❯ what does this project do?
❯ read @src/auth.js and explain the login flow
❯ fix the failing tests
❯ add a --dry-run flag to the CLI and update the README
```

It works autonomously: it picks the files, makes the changes, runs the tests to
verify, and reports back — only asking when it genuinely needs you.

**In the chat:**
- `Esc` — clear the input line, or (mid-run) **stop the agent**
- `@path` — pull a file's contents into your message
- `/model` — switch models live · `/config` — show settings · `/plan` — current plan
- `Ctrl-D` — quit

**One-shot (non-interactive), great for scripts:**

```bash
forge run "fix the type errors in src/ and run the build"
```

## Commands

```
forge                       chat with an agent in the current repo
forge run "<task>"          run one task to completion, autonomously
forge setup                 detect hardware / choose engine / write config
forge status                show every live forge session and what it's doing
forge send <target> <msg>   message another running session
forge up  /  forge down     start / stop the fleet autopilot (verify + coordinate + learn)
forge receipts              trust audit trail — verdicts on "done" claims
forge learnings [dir]       durable facts forge has learned about a repo
forge --version
```

## One fleet with Claude Code

If Claude Code runs on the same machine with a fleet channel (`~/.claude/fleet`),
forge joins that network automatically — no configuration:

- **Unified board** — `forge status` lists Claude Code sessions alongside forge
  sessions (and Claude Code's fleet board sees forge sessions).
- **Cross-runtime messaging** — `forge send <target> <msg>` and the agent's
  `fleet_send` action reach Claude Code sessions; Claude Code's `fleet_send`
  reaches forge sessions. Messages land mid-work, as if from a teammate.

forge speaks the Claude fleet's wire protocol directly: every forge session
registers in the shared inbox registry (tagged `kind: "forge"`) and accepts the
fleet's authenticated `POST /send`. `forge setup` checks the interop on any
machine and prepares what's safe (shared token), reporting exactly what works.
Without Claude Code, forge's native fleet works standalone.

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
forge/
  backends.py   model-agnostic backends (Ollama + OpenAI-compatible) + routing
  tools.py      tools (bash/read/write/edit/grep/glob/fleet_send) + action schema
  agent.py      the agent loop (harness brain) + levers + context management
  workspace.py  workspace + machine awareness (file tree, project type, git, tools)
  session.py    transcript · registry · token-authed inbox · locking
  repl.py       interactive chat + slash menus
  tui.py        raw-mode line editor (Esc to clear/stop) + interrupt watcher
  fleet.py      verify · coordinate · learn · message primitives
  daemon.py     forged — the autopilot loop
  config.py     per-machine config (~/.forge/config.json)
  setup.py      the installer (hardware detection, engine choice, model pulls)
  __main__.py   the CLI
~/.forge/       runtime: sessions/ · registry.json · learn/ · verdicts.jsonl (mode 0700)
```

## Development

```bash
git clone https://github.com/hackspaces/blueshark-forge && cd blueshark-forge
python -m unittest discover -s tests    # 34 tests, stdlib only, no deps
./forge-cli                             # run from the checkout without installing
```

CI runs the suite on every push across Python 3.10–3.13. Contributions welcome.


## Security & trust model

forge runs on **your** machine with **your** privileges — treat it like any coding
assistant that can edit files and run commands.

- The **file tools** (`read/write/edit/grep/glob`) are confined to the working
  directory. The **`bash` tool is intentionally *not* sandboxed** — it runs
  arbitrary shell commands as you, on purpose (that's what a coding agent needs).
  Run forge in repos you trust, or use OS-level sandboxing for untrusted code.
- The **fleet inbox** (session-to-session messaging) is localhost-only and
  **token-authenticated**: only real forge sessions (which can read the private
  `~/.forge/registry.json`, mode 0600) can message each other. `~/.forge` is 0700.
- The **autopilot** (`forge up`) runs a repo's own test command to verify "done"
  claims. It does this on an isolated copy, but it *does* execute the project's
  test script — only run `forge up` over repos you trust.

Found a security issue? Please open an issue (or email the maintainer).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: fork, branch, add tests, open a
PR against `main`. `main` is protected — changes land through reviewed PRs with
green CI, not direct pushes.
