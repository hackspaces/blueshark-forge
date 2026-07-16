# blueshark-forge

A model-agnostic agentic runtime for the terminal. Any model, frontier or a
small local one, becomes a capable agent, because the intelligence lives in the
harness, not the weights. And every forge session is part of a fleet: they verify
each other's work, coordinate, and share what they learn.

Not tied to any vendor. Runs on your machine, on your models.

[![PyPI](https://img.shields.io/pypi/v/blueshark-forge)](https://pypi.org/project/blueshark-forge/)

---

## Install

**macOS / Linux** — one line; checks your environment and installs the `forge` CLI:

```bash
curl -fsSL https://topk1.com/forge/install.sh | sh
```

**Windows** (PowerShell or cmd) — requires Python 3.10+:

```bat
pip install blueshark-forge
```

<sub>Prefer to see the macOS/Linux script first? Append `FORGE_INSTALL_DRY_RUN=1` before `sh` to check your setup without installing anything — or just open [topk1.com/forge/install.sh](https://topk1.com/forge/install.sh) and read it. Fetching straight from the repo works too: `curl -fsSL https://raw.githubusercontent.com/hackspaces/blueshark-forge/main/site/install.sh | sh`.</sub>

**Or by hand** (any OS) — requires Python 3.10+:

```bash
pipx install blueshark-forge          # recommended (isolated); or: pip install blueshark-forge
```

**An engine is optional** — only local models need one. Install [Ollama](https://ollama.com) for the
simplest local setup, or bring a frontier model with your own key (OpenAI / Anthropic). forge drives any of them.

## Set up (once per machine)

**Just run `forge`.** On a fresh install it detects your hardware, tells you the
biggest model your machine can actually run, and points at the next step:

```
  ✦ forge  ·  run any model your machine can handle

  Nothing set up yet — but Apple M5 Pro · 48GB RAM · GPU / Metal (fast)
  can run models up to ~70B right here.

  forge models            see everything it can run
  forge models use phi-2  a quick starter — pulled + ready in ~2 min
  forge run "…"           then put it to work
```

`forge models` lists what fits this machine — every model sized and speed-checked
against your RAM, GPU/VRAM, or Apple unified memory. `forge models use <name>`
provisions it (pulls the weights, launches a server if it needs one) and points
forge's config at it. That's the whole setup.

<sub>`forge models --all` runs the same fit-math over the *full* downloadable catalog (Ollama library + HuggingFace GGUF + MLX), not just the curated spread.</sub>

**Or pick a ladder yourself** — `forge setup` inspects your machine and configures
forge for it:
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

# frontier models with your own key — OpenAI…
forge setup --engine openai --api-key sk-... --models "gpt-4o-mini,gpt-4o"

# …or Anthropic (its OpenAI-compatible endpoint; key also read from ANTHROPIC_API_KEY)
forge setup --engine anthropic --api-key sk-ant-... --models "claude-sonnet-4,claude-opus-4"
```

<sub>`openai` / `anthropic` default to the right URL, so `--url` is optional. Your key stays in `~/.forge/config.json` (or the `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env var) — forge itself hosts nothing; it just drives the API you point it at.</sub>

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

**Repo rules.** Drop a `FORGE.md` (or `AGENTS.md`, or `CLAUDE.md`) at the repo
root and forge pins it into every session as top-priority, user-authored
instructions — above anything the fleet has merely learned. The fleet's own
learned facts are validated before they stick: a claimed test command is run once
in an isolated copy (or matched against the detected one) and a ✓ marks the ones
the harness actually confirmed.

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
forge --resume <sid|last>   resume a prior session from its transcript
forge run "<task>"          run one task to completion, autonomously
forge setup                 detect hardware / choose engine / write config
forge status                show every live forge session and what it's doing
forge send <target> <msg>   message another running session
forge up  /  forge down     start / stop the fleet autopilot (verify + coordinate + learn)
forge receipts              trust audit trail — verdicts on "done" claims
forge learnings [dir]       durable facts forge has learned about a repo (✓ = harness-verified)
forge forget [pattern]      prune learned facts (substring match, or all)
forge trace [sid|last]      replay a session's step trace as a table
forge bench [--report]      harness-lift eval: same model bare vs full harness
forge replay [sid|last]     re-drive a recorded session through the harness, no model
forge passport [--probe]    per-model capability profile + the knobs it auto-tunes
forge --version
```

### Model passports

Every knob (loop threshold, output budget, retry temperature) is one-size-fits-all
by default — but failure modes are model-specific. forge keeps a **passport** per
model: it measures each one two ways — an active ~90s probe at `forge setup` (can it
hold the action format? reproduce exact text? stay valid at temp 0?) and passive
telemetry from live runs (malformed rate, loop-trip rate, fuzzy-edit rate, escalation
frequency) — then tunes itself to the model at hand: a tighter loop threshold for
loop-prone models, a bigger `num_predict` for write-file truncators, a hotter retry
schedule for models whose greedy retries come back identical. `forge passport` shows
each model's learned profile and the knobs it resolves to; `forge passport --probe`
re-runs the active probe now. An un-profiled model runs on the stock defaults.

### Resume a session

Every session's transcript (`~/.forge/sessions/<id>.jsonl`) is reconstructable
memory, not just telemetry. `forge --resume <sid>` (or `--resume last`, the newest
session for the current repo) rebuilds an agent from it: a **fresh** workspace
briefing as the head (re-oriented in the repo as it is now, not a stale snapshot),
the last compaction summary as `[Earlier progress]`, the recent turns replayed, the
living plan restored, and the read-file ledger seeded — but only for files still
unchanged on disk, so read-before-edit stays honest and anything touched since is
re-read. A session that is still running (live pid) is refused. Prompt history
persists to `~/.forge/history` across sessions.

### Flight recorder + replay

Every step's raw model output — malformed ones included — is logged into the
session transcript. `forge replay <sid>` re-drives a **real** agent loop from those
raws with **no model and no GPU**, then reports the first step where a changed
harness diverges (a different gate/action/compaction point) and the terminal state
— so a harness change is validated against real small-model behavior at zero
inference cost. `forge replay <sid> --to-fixture <name>` snapshots the session's
raws into `tests/fixtures/<name>.jsonl`, and `tests/test_replay.py` sweeps every
fixture as a regression test. `--strict` also asserts each recorded prompt digest
matches (loose, the default, is robust to prompt-wording changes). Set
`FORGE_RECORD=<path>` to additionally mirror every model call into a `{digest, raw,
prompt_tokens}` cassette. Replay reconstructs the harness-**decision** path; full
fidelity of the file-system half needs a workspace snapshot (a `setup.sh`, like the
bench fixtures), so replay runs in a throwaway dir and never touches your files.

**Fault injection.** Add repeatable `--fault` flags to replay a real trace through
adverse conditions without inference:

```bash
forge replay last --fault truncate_output --fault authority_violation
```

Available faults: `truncate_output`, `malformed_burst`, `wrong_edit_anchor`,
`force_compaction`, `authority_violation`, `repeat_storm` (duplicate an action to
provoke a no-progress loop), `stale_read` (re-read a file right after mutating it),
and `deceptive_completion` (an unverified "all tests pass" claim right after a
change). The report includes recovery, false-completion, action efficiency,
observation failures, loops, escalations, authority denials, completion rejections,
**verification precision** (of the completions the gate judged, the share truly
verified), **workspace-corruption rate** (mutations whose write/edit failed), and
context-token pressure. Injection changes a deep copy of the trace; the recorded
session and workspace stay untouched.

### Harness-lift benchmark

`forge bench` measures what the *harness* buys, not the weights: it runs each task
fixture (`bench/<task>/` — `prompt.txt` + optional `setup.sh`/`verify.sh`) through
the real agent loop twice, once **bare** (every scaffolding lever off) and once with
the **full** harness, and prints the pass-rate lift. Per-lever ablation flags
(`--no-compact`, `--no-loop-detect`, `--no-read-gate`, `--single-rung`) drop one
lever from the full set so you can see which lever earned its complexity. Results
append to `~/.forge/bench/results.jsonl`.

Honest framing: "bare" turns off constrained decoding, but the loop still demands a
JSON action every step and gives up after 5 malformed replies — so a bare pass-rate
substantially measures *format compliance* (can the raw model hold the action
contract at all). That is exactly the harness-lift story worth telling: the
scaffolding is what makes a small local model usable.

### In the chat

- **Modes** (`shift+tab` cycles, or `/mode auto|plan|manual`):
  - **auto** — acts freely, no questions (the default)
  - **plan** — read-only: investigates, then presents a plan for approval
  - **manual** — asks before every mutating action: `y` yes once ·
    `a` always (saved — that action type won't ask again) · `n` no
- **Authority** is separate from model capability: `FORGE_AUTHORITY=observe|contribute|operator|admin`
  (default `operator`). Observe can inspect; contribute can edit and test; operator can
  run normal shell commands and message peers; admin is required for destructive shell
  that can escape the workspace, privilege escalation, remote scripts, secret-store reads,
  and forced history changes. Invalid authority values fail closed to `observe`.
- **Queue messages while it works** — just keep typing; Enter delivers your
  message to the agent between steps (it steers mid-task). Anything not
  absorbed becomes the next turn.
- **`/files` — folder explorer** — a three-pane Miller-column browser
  (parent · current · preview) right in the terminal: `↑↓` move, `←→`
  navigate, `Enter` on a file attaches it to your next message as `@file`,
  `.` shows hidden files, `q` closes.
- `Esc` clears the line, or stops the agent mid-run (twice force-returns).

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
- **Loop detection** — repeated actions are broken automatically, but *semantically*:
  a repeat only counts as a loop when the workspace did not change between the repeats.
  Re-running the tests after an edit is a new hypothesis and stays healthy; re-running
  them with nothing changed is the loop.
- **Autonomy scaffolding** — task mode tells the model to act, not ask.
- **Verify-on-done** — a claim of "done" is checked, never trusted. Completion policy is
  deterministic: `FORGE_COMPLETION_POLICY=audit|balanced|strict` (default `balanced`).
  Balanced rejects a failed check once and records any second-claim escape as an explicit
  override; strict requires passing evidence for every changed workspace.

**Workspace + computer awareness** (like a real coding assistant): on start, forge
builds a gitignore-aware map of the project, detects the language/project type,
reads the git state, and learns the machine it's on (OS, shell, tool versions), all
pinned into context. Say "fix the auth bug" or "read this @file" and it already
knows where things are. It also inherits whatever the fleet has learned about the repo.

**Frontier agent loop**: a living plan (todo list the agent maintains and the
harness pins each turn), pinned `note` scratch facts (a durable fact worth
keeping — where something lives, a command that works — pinned alongside the plan
so it survives compaction verbatim; the harness seeds note #0 with the detected
test command), surgical `edit_file` (not fragile full rewrites), self-correction
(failed actions are flagged so the model diagnoses), loop-breaking, and context
compaction for long sessions.

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
forge learnings [dir]          durable facts learned in a repo (✓ = harness-verified)
forge forget [pattern]         prune learned facts (substring match, or all)
forge trace [sid|last]         replay a session's per-step trace as a table
forge bench [--report]         harness-lift eval: same model bare vs full harness
forge replay [sid|last]        re-drive a recorded session through the harness, no model
forge passport [--probe]       per-model capability profile + the knobs it auto-tunes
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
                (read_file numbers every line; edit_file takes {start_line,end_line,anchor,new}
                 to splice a range — no exact-text reproduction — or {old,new} as fallback;
                 read_file {outline:true} maps a big file's defs/classes → line numbers)
  agent.py      the agent loop (harness brain) + levers + context management
  workspace.py  workspace + machine awareness (recency-ranked repo map w/ per-dir
                rollups, symbol briefing, project type, git, tools)
  index.py      persistent symbol index (ast for .py, regex for js/ts/go/rs)
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
python -m unittest discover -s tests    # 400+ tests, stdlib only, no deps
./forge-cli                             # run from the checkout without installing
```

CI runs the suite on every push across Python 3.10–3.13. Contributions welcome.


## Security & trust model

forge runs on **your** machine with **your** privileges — treat it like any coding
assistant that can edit files and run commands.

- The **file tools** (`read/write/edit/grep/glob`) are confined to the working
  directory. The **`bash` tool is intentionally *not* sandboxed** — it runs with
  your OS privileges. Harness authority still gates access: operator sessions handle
  normal shell work, while recognized destructive/privileged/secret-sensitive commands
  require `FORGE_AUTHORITY=admin`. Run forge in repos you trust, or use OS-level
  sandboxing for untrusted code.
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
