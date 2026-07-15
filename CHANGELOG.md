# Changelog

Every published release of `blueshark-forge`, newest first.

Generated from the GitHub Releases by `tools/changelog.py` — the releases are
what the tag-gated publish actually shipped, so this cannot drift. Don't hand-edit.

## [v0.11.4](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.11.4) — the harness does the read you skipped, instead of bouncing once per file

*2026-07-15 · `pip install blueshark-forge==0.11.4`*

See the release commit for details.

## [v0.11.3](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.11.3) — a stuck model is now caught in 3 steps, not ground for 36

*2026-07-15 · `pip install blueshark-forge==0.11.3`*

See the release commit for details.

## [v0.11.2](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.11.2) — run_tests finds pytest-style tests, and a zero-test run is never evidence

*2026-07-15 · `pip install blueshark-forge==0.11.2`*

See the release commit for details.

## [v0.11.1](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.11.1) — the syntax gate no longer fails open

*2026-07-15 · `pip install blueshark-forge==0.11.1`*

Two real bugs in the post-write syntax check (P1.1's gate), found chasing a CI
flake rather than by reading code.

A timeout made the gate fail OPEN. The check ran with timeout=3 and swallowed
TimeoutExpired into "can't check", which the caller treats as permission to write.
That is right when the checker isn't installed and a hazard when it merely timed
out: node --check is ~0.02s warm but a cold start on a loaded machine is not, so
the gate reported "fine" on a file it had never checked and let invalid JS land.
Now 10s (FORGE_SYNTAX_TIMEOUT) plus one doubled retry; a genuine hang still reads
as "cannot check", never as "fine".

The error named a file that does not exist. The check runs on a temp copy, so its
error cites the temp path, rewritten back to the real basename — but tempfile
returns /var/folders/… while macOS reports /private/var/folders/…, so the replace
matched only the suffix and stranded the prefix: '/privatea.js:1' instead of
'a.js:1'. The model was being sent to fix the wrong file. Nothing covered it,
which is how a user-facing message stayed wrong unnoticed.

Both now pinned by tests. 725 tests, green on 3.10–3.13.

## [v0.11.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.11.0) — the thesis phases are done, and the harness now catches itself

*2026-07-15 · `pip install blueshark-forge==0.11.0`*

Phases P1–P5 of the breakthrough roadmap are COMPLETE (31/56 items, 55%). That is
the whole thesis spine:

  P1 Honest observations        5/5 ✅
  P2 The verified loop          3/3 ✅   "done" is a fact, not a claim
  P3 Measurement                4/4 ✅   bench proves harness-lift per lever
  P4 Context integrity & memory 8/8 ✅
  P5 Small-model amplification  8/8 ✅   the thesis, bench-validated
  P6 Loop architecture          3/8
  P7–P10                        0/20

The roadmap's own one-sentence breakthrough was: bench measures harness-lift per
lever per model (P3), the verified loop makes "done" a fact instead of a claim
(P2), and the swarm compiles goals into verified commits from laptop-sized models
(P9). Two of those three legs now stand. P9 — the moat — does not.

Headline feature: the claim guard (P5.9, #42). A model that claimed done, was
rejected, and claimed the identical thing again would do so until the step limit —
qwen2.5-coder:7b did it 33 times, burning 63s and the whole budget to re-send one
sentence. loop_detect watches ACTIONS; nothing watched the completion path. The
cause was starvation, not stubbornness: the bounce named the failing command while
the evidence contract kept only a digest of its output, so the model was told "it
failed" 33 times and never what failed. Now: claim → show the real failure → stop.
3 claims, 18s, an honest ending. It does not make models smarter; it converts a
confused failure into a fast truthful one — and it never opens the gate, because
repetition must not become an escape from the done-gate (H05).

P5.9 is the first roadmap item NOT produced by the 31-agent audit. Thirty-one
agents read every line and could not have found it: it only exists when a real
model runs a real task. A static audit finds what the code says; running it finds
what the code does.

Also in this line since v0.10.0: a real first-run experience, TTY-gated output,
one canonical install URL (topk1.com/forge/install.sh), a mobile-usable landing
site, grouped `forge --help`, and CLAUDE.md.

723 tests, green on 3.10–3.13.

## [v0.10.3](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.10.3) — a help you can actually read

*2026-07-15 · `pip install blueshark-forge==0.10.3`*

Re-cut: the first v0.10.3 tag pointed at a commit that SyntaxErrored on Python
3.10/3.11 (f-string backslash, PEP 701 — see #41). The tag-gated publish caught it
and refused to ship, so nothing ever reached PyPI; this tag replaces it on the
fixed tree. All four matrix jobs (3.10–3.13) are green.

`forge --help` was a wall: argparse spelled its 16-name brace blob twice (usage
line + positional header), then listed every command flat. `models` and `setup` —
the first-run path — sat at the bottom. `--model`, `--dir`, `--name`, `--verbose`
carried no help text at all. And bare `forge`, the most common invocation of all,
was never mentioned.

Help is now grouped by what you're trying to do, first-run first:

  Start here      models, setup
  Work            run, bench
  The fleet       up/down, status, send
  What happened   trace, receipts, learnings/forget, passport
  Data out        export, corpus, replay

It opens with the three invocations people actually type, every option is
described, and colour is TTY-gated so piping --help stays plain text. The
subparsers got a <command> metavar too — argparse had been spelling all 16 names
into every usage line, including the one printed on an error, where it buried the
actual message.

Hand-grouped help can drift from the parser in a way a generated list cannot, so
TestGroupedHelp pins the groups against the real subcommand set (read back out of
argparse's own invalid-choice error) and fails both ways.

Also adds CLAUDE.md — the repo rules an agent can't infer: stdlib-only is
load-bearing; a merge is not the finish line (merge → release → verify the curl
one-liner actually serves it); never tag a merge commit; Python 3.10 is the floor
and a green run on a newer interpreter proves nothing; and why the site's grid
needs minmax(0,1fr).

719 tests, green on 3.10 / 3.11 / 3.12 / 3.13.

## [v0.10.2](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.10.2) — one canonical install URL, recorded everywhere

*2026-07-15 · `pip install blueshark-forge==0.10.2`*

The domain is live, so the repo now says so — in one voice, everywhere.

### The real bug: the installer's own URL was a 404

`install.sh`'s header told readers to fetch `raw.githubusercontent.com/.../main/install.sh`. There is no root `install.sh` — it lives at `main/site/install.sh`. So anyone who **opened the script before piping it to their shell** (the security-conscious move) hit a dead URL. The header now names the canonical `https://topk1.com/forge/install.sh`, with the raw-GitHub path documented as a working mirror.

### One story, not two

The installer's **Next** steps led with `forge setup` — the path v0.10.1 deliberately demoted. It now mirrors the first-run welcome exactly:

```
Next
  forge                  see what this machine can run
  forge models use phi-2 a quick starter — pulled + ready in ~2 min
  forge run "<a task>"   then put it to work

  (or  forge setup  to pick a model ladder yourself)
```

The README's **Set up** section gets the same treatment — it leads with the real first-run output instead of `forge setup`.

### Install

```bash
curl -fsSL https://topk1.com/forge/install.sh | sh
```

`README.md` is the PyPI `long_description`, so the new instructions ship to the PyPI page too; `Homepage` now points at `https://topk1.com/forge/`.

**Verified against the live domain:** `/forge`, `/forge/`, and `/forge/install.sh` all 200; `install.sh` serves `application/x-sh` (not the rewritten HTML); the full one-liner runs clean end-to-end. 716 tests pass.

**Full changelog:** https://github.com/hackspaces/blueshark-forge/compare/v0.10.1...v0.10.2

## [v0.10.1](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.10.1) — a real first run + clean piped output

*2026-07-14 · `pip install blueshark-forge==0.10.1`*

Two terminal-UX fixes, both found by dogfooding a fresh `pip install blueshark-forge`.

First-run experience (#36)
  Bare `forge` on a fresh install used to spin a "loading model…" spinner for a
  few seconds and then drop into a chat pointed at a placeholder default model the
  user had never installed — no guidance, no sign of what the machine could run.
  Now it shows the machine's hardware, the model-size ceiling it can run, and the
  two commands to get going (`forge models`, `forge models use phi-2`, `forge run`),
  with `forge setup` as the manual path. `forge run "…"` exits 1 with the same
  pointer instead of the same dead end. Once a model is configured, nothing changes.

Clean piped output (#37)
  The thinking/loading spinner and the `forge models --all` scan progress wrote
  `\r`, ANSI color, and `\033[K` unconditionally — leaking escape-code spam into
  `forge run … > log`, `| tee`, and CI capture. Both now animate only on a real
  terminal (`sys.stdout.isatty()`); off-TTY they're silent and the results print
  clean.

No API changes. 716 tests.

## [v0.10.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.10.0) — the model catalog: find and run what your machine can handle

*2026-07-14 · `pip install blueshark-forge==0.10.0`*

forge learns to answer "which open models can MY computer actually run?" — and to
run them in one command, from a laptop to a datacenter.

The catalog (`forge models`):
  · forge models        — a spread of good open models, fit- AND speed-checked
    against your hardware: system RAM on a CPU laptop, unified memory on Apple
    Silicon, and VRAM on a GPU box (single card → multi-GPU / datacenter node).
    "runs well up to ~NB", honestly — a 7B "fits" 8GB but crawls on CPU; a small
    GPU is judged by its VRAM, not system RAM.
  · forge models --all  — run that same math on the WHOLE downloadable catalog:
    the Ollama library + curated HuggingFace GGUF (the pool LM Studio / llama.cpp /
    Jan draw from) + MLX (Apple Silicon), fetched concurrently and cached.
  · forge models use <name>  — turnkey. Ollama models are pulled + configured;
    llama.cpp models (even a custom arch like sarvam-30b) get their weights fetched,
    a server launched, and config wired — then `forge run` just works.
    forge models show/stop round it out.

Frontier models, your own key: `forge setup --engine openai|anthropic` — forge
hosts nothing, it drives the API you point it at.

Harness-lift: the catalog surfaces each model's base-vs-harness gain from
`forge bench` — the "weak weights + strong harness" number, made visible.

Getting it: a one-line installer (`curl -fsSL …/site/install.sh | sh`) and a
landing site that lives in the repo (Vercel-deployable).

Still stdlib-only, zero runtime deps. 712 tests.

## [v0.9.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.9.0) — the model catalog + hardware-honest local defaults

*2026-07-14 · `pip install blueshark-forge==0.9.0`*

forge learns to answer "which model, and will it run here?" — and to run
on more than a Mac.

The model catalog (new `forge models` command family):
  · forge models          — curated, hand-verified recipes checked against
    THIS machine's RAM: engine, size, a fit estimate, and an honest
    verified/candidate status (only models forge has actually run are verified).
  · forge models show <n> — the copy-pasteable runbook; for a custom-arch
    model like sarvam-30b, the `strings|grep` arch gate BEFORE the 20GB
    download, resumable weights, serve flags, the context-match gotcha.
  · forge models use <n>  — turnkey provisioning for Ollama-native models
    (pull, write config, smoke-test → `forge run` just works). Non-Ollama
    engines get the honest runbook, never a half-done job.

Hardware-honest local defaults:
  · CPU-only machines (no Apple Silicon, no dGPU) get a RAM-capped ladder
    instead of a 9B escalation rung that would swap-thrash an 8GB laptop.
  · Windows RAM is now detected — it was silently 0, dropping every Windows
    machine to the minimal tier.

Full-fidelity replay (H11): `forge replay --to-fixture` now also captures a
content-addressed workspace snapshot. Credentials are never archived,
captures are bounded, and everything excluded is an honest fidelity limitation.

docs/models: an open collection of how models run through forge, first entry
sarvam-30b.

Still stdlib-only, zero runtime deps. 667 tests.

## [v0.8.3](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.8.3) — automated, self-proving release pipeline

*2026-07-12 · `pip install blueshark-forge==0.8.3`*

The release process now proves itself. Pushing a `vX.Y.Z` tag runs one ordered,
fail-closed pipeline — no manual `gh release create`, no way to ship a mislabelled
or untested artifact.

A `v*` tag triggers: tests (3.10–3.13, the same reusable matrix every push to main
runs) → guard (assert the tag equals `forge.__version__`; build the wheel and
install it into a clean venv; smoke-test `forge --version`/`--help`) → pypi (publish
the VETTED artifact via OIDC trusted publishing) → github-release (create the
Release, titled and noted from the `release:` commit body). Each stage gates the
next, so a red test, a tag≠version mismatch, or a broken wheel stops the run before
anything publishes; the Release is created last, so a Release existing means the
version is already on PyPI. This closes the exact gap behind the v0.7.6–v0.7.11
"version bumped but never released" drift.

All actions are pinned by immutable commit SHA (verified commits, not tag objects);
the publish is idempotent (`skip-existing`) so a re-pushed tag recovers a failed
release; jobs carry timeouts and least-privilege permissions. `RELEASING.md`
documents the one-push flow. `tests/test_cli.py` adds in-repo checks that the CLI
starts (`--version`/`--help`) and `__version__` is clean semver — the coherence the
tag-gate depends on.

This release is itself the first cut driven through the new pipeline. Stdlib-only,
zero runtime deps.

## [v0.8.2](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.8.2) — readable output across every report command

*2026-07-11 · `pip install blueshark-forge==0.8.2`*

Extends the width-aware, colour-gated treatment that landed for `forge status`
in v0.8.1 to every other plain-print surface, via a shared forge/render.py
(paint · fit · tilde · term_width · strip_ansi). One theme everywhere: output is
collapsed to a line and fit to the ACTUAL terminal width with an ellipsis rather
than wrapping into an unreadable block, and colour is emitted only to a real TTY
(honours NO_COLOR and TERM=dumb) so piped or redirected output stays clean text.

- receipts   coloured ✓/✗ verdict, ~-shortened project, evidence fit to width
- learnings  coloured verified mark, facts fit to width
- trace      coloured header + action, FAIL in red, active flags in yellow,
             ~-shortened cwd, a box-drawn divider sized to the header
- run        one-shot event lines fit to width (were fixed [:80]/[:70]) with
             ok/fail, escalate/borrow and inbox lines coloured

The interactive chat TUI already has its own rich renderer (in-chat actions,
diffs, file explorer, approval gate) and is unchanged. Stdlib-only, zero runtime
deps. 526 tests green (+10 for render).

## [v0.8.1](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.8.1) — readable status screen + clean Ctrl-C everywhere

*2026-07-11 · `pip install blueshark-forge==0.8.1`*

Two CLI quality-of-life fixes.

`forge status` is now readable. The old renderer truncated at a fixed 120/150
chars with no terminal-width awareness, so long asks and replies wrapped into an
unreadable block. It now fits the actual terminal: one aligned header per session
(color glyph by runtime+state — green idle · yellow working · cyan claude), a
~-shortened path, and task/you/reply each collapsed to a single width-fitted line
with an ellipsis. forge and Claude Code peers render through one unified path with
a right-aligned "N forge · M claude" count. Colour is gated on a TTY (honours
NO_COLOR / TERM=dumb), so piped output stays clean plaintext.

Ctrl-C exits clean from any command. A KeyboardInterrupt at any prompt or during
any long command used to dump a raw traceback — most visibly at the `forge setup`
model-pull prompt, because setup ran entirely outside the top-level try and the
handler only caught ForgeError. setup now runs inside the guarded block, and the
CLI boundary catches KeyboardInterrupt/EOFError to exit 130 (the shell SIGINT
convention) with no traceback. The handler lives in main() so it also covers the
installed console-script entry point; the chat REPL still swallows its own Ctrl-C
to cancel a line and is unaffected.

516 tests green, including a real SIGINT delivered to a live setup prompt.

## [v0.8.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.8.0) — the reliability roadmap, finished (semantic loops + richer fault eval)

*2026-07-11 · `pip install blueshark-forge==0.8.0`*

Completes the two partially-shipped items from the evidence-driven-kernel roadmap:
deeper fault-injection evaluation and semantic loop detection. The harness now
distinguishes an action that changed the workspace from one that merely repeated it,
and stresses recovery across three more deterministic failure modes.

Semantic loop detection
  The loop breaker now keys on an action's EFFECT, not just its shape. A write_file
  signature folds a content hash (edit_file already folded its payload), so rewriting
  a file with NEW content is a distinct action — a changed hypothesis that stays
  healthy — while re-emitting the SAME bytes still trips the breaker. Re-running the
  tests after an edit was already healthy (the edit breaks the repeat window); this
  closes the remaining gap where a content-blind write signature falsely grouped
  genuinely-different writes, and where an identical-write loop could slip through.

Three new fault scenarios (deterministic, zero-inference)
  repeat_storm         duplicate an action to provoke a no-progress loop and exercise
                       loop recovery;
  stale_read           re-read a file immediately after mutating it, against the
                       harness's pre-edit ledger snapshot;
  deceptive_completion inject an unverified "all tests pass" claim right after a
                       change — the evidence-aware done-gate must catch it.
  All eight faults are surfaced through `forge replay <sid> --fault <name>`.

Two new evaluation metrics
  verification_precision   of the completions the gate judged, the share that were
                           truly verified (n/a when nothing was judged — a run that
                           verified nothing is not "perfectly precise");
  workspace_corruption_rate  mutations whose write/edit FAILED, over the mutations
                           that actually executed (gate-blocked actions count in
                           neither term).

Hardening from an adversarial review of this change: the fault finder no longer
crashes on a model row whose raw is valid non-object JSON, and the corruption metric
no longer inflates its denominator with mutations that never ran. Stdlib-only,
zero runtime deps. 512 tests green.

## [v0.7.12](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.12) — the corpus flywheel (sessions become training data)

*2026-07-11 · `pip install blueshark-forge==0.7.12`*

Turns forge's own operating record into training signal for the model that runs
inside forge. New `forge/corpus.py` + `forge corpus [sid|last|--all] [--out FILE]`
reconstruct a session transcript at the turn level (user → action → observation →
say) and emit two harness-native signals:

- SFT examples — for every action that actually executed, {messages: <context>,
  completion: <action>}: the context paired with the action the model should emit.
  Teaches the action protocol and the tool sequences that worked.
- Preference pairs — the CORRECTION moments, the highest-value signal because they
  are drawn straight from the failure modes forge already measures and repairs:
  {prompt, chosen, rejected, kind}. kind=grammar is a malformed strike recovered to
  valid action JSON; kind=narrate is an "I'll do it…" preamble that got bounced,
  recovered to real work (the act-don't-narrate reward). chosen > rejected is the
  DPO/ORPO reward.

Deterministic and stdlib-only. The per-session workspace briefing is intentionally
omitted from the reconstructed context — it isn't recoverable from a transcript and
the trainer prepends a consistent system prompt instead. `--out corpus.jsonl` writes
flat JSONL tagged by split (sft/pref) over one session or every recorded one.

This is turn one of the flywheel: every session forge runs is now labeled data for
the harness that produced it. 456 tests green.

## [v0.7.11](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.11) — zero-inference fault-injection replay + newline classifier fix

*2026-07-11 · `pip install blueshark-forge==0.7.11`*

Adds deterministic, zero-inference fault injection over recorded replay traces:
`forge replay <sid> --fault <name>` (repeatable) injects into a deep copy of the
trace — the recorded session and workspace stay untouched — and re-drives it
through the real Agent.send loop. Five scenarios ship: truncate_output,
malformed_burst, wrong_edit_anchor, force_compaction, and authority_violation.
The report covers recovery, false-completion, action/tool efficiency, observation
failures, loops, escalations, authority denials, completion rejections, and
context-token pressure; inapplicable and unknown injections are reported
explicitly rather than silently skipped.

Also rolls up the post-v0.7.10 fix that treats unquoted newlines as command
separators in the authority classifier, so a privileged later line can no longer
hide behind a benign first command (quoted multi-line data stays exempt).

## [v0.7.10](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.10) — token-aware authority classifier hardening

*2026-07-11 · `pip install blueshark-forge==0.7.10`*

Replaces the shell admin-classification regexes with shlex token-aware parsing
that inspects the actual parsed command and delete targets. Closes prior bypasses
and false positives: recursive-force deletes are detected across combined, split,
and GNU long options; targets are analyzed after quote removal so absolute, home,
variable, `.`/`..`, traversal-outside-workspace, and unprovable-wildcard deletes
are admin-only while workspace-relative cleanup (rm -rf build/./dist/cache) stays
at operator; remote installer pipes are matched structurally across segments;
wrappers and env assignments before a privileged command are handled; forced git
history includes --force-with-lease; secret stores match by path component with
.env.example/.sample/.template explicitly permitted. Invalid FORGE_AUTHORITY now
fails closed to observe, and malformed shell tokenization fails closed to admin.

## [v0.7.9](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.9) — runtime authority separate from model capability

*2026-07-11 · `pip install blueshark-forge==0.7.9`*

Adds harness-owned authority levels (observe/contribute/operator/admin, default
operator) independent of model passports and ladder tier, configured via
FORGE_AUTHORITY. Authority is checked before interaction mode and cannot be
expanded by a stronger escalated model. The constrained grammar is narrowed
before generation and re-enforced after parsing for advisory engines; denials
log actual/required authority and project into the canonical protocol as
AuthorityDenied → DIAGNOSE (recovery to PLAN). Admin-only shell is scoped to
genuinely dangerous patterns — sudo, remote-script pipes, secret-store reads,
forced git history, and recursive-force deletes aimed at root/home/an absolute
path — while everyday-safe commands (env, rm -rf of a project subdir) stay at
operator. Read-only `fleet_send target=list` remains available to observers.

## [v0.7.8](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.8) — deterministic completion policy

*2026-07-11 · `pip install blueshark-forge==0.7.8`*

Turns completion accept/reject from control-flow booleans into an explicit,
inspectable CompletionPolicy over harness-built EvidenceContracts. Three modes
via FORGE_COMPLETION_POLICY: audit (record, never block), balanced (default —
reject a failed check once, then permit the historical second-claim escape as a
named single_bounce_override), and strict (require passing verification for every
changed workspace). Each decision is logged as completion_policy with its
evidence and attempt number, and the final decision is attached to accepted
assistant records. Balanced mode is outcome-compatible with prior behavior while
making the former invisible escape hatch explicit. README documents the modes.

## [v0.7.7](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.7) — evidence-aware completion gate

*2026-07-11 · `pip install blueshark-forge==0.7.7`*

Connects the v0.7.6 evidence model to the real agent loop. A turn-scoped,
harness-owned EvidenceCollector records changed files (from write/edit and
opaque mutating bash) and verification results (command, exit code, output
digest) straight from tool outcomes — the model cannot forge it. Every accepted
completion carries an EvidenceContract with `verified`; a failed done-gate emits
a structured `completion_rejected` record with its evidence, and a missing/absent
runner is captured as an explicit unverified assumption. Evidence resets each
user turn. The done-gate's accept/reject decision and the action schema are
unchanged — the contract is additive on assistant/completion_rejected records.

## [v0.7.6](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.6) — evidence-driven execution kernel

*2026-07-11 · `pip install blueshark-forge==0.7.6`*

Formalizes a first-class execution state model (ORIENT→INVESTIGATE→PLAN→MUTATE→
VERIFY→DIAGNOSE→COMPLETE) and a versioned canonical runtime event protocol, and
projects existing session transcript records onto additive `runtime` envelopes
without changing or removing any legacy field. Adds structured evidence
contracts (changed files, verification, digests, unverified assumptions) plus
verification obligations and recovery transitions on state changes. A stuck
hand-off is no longer misprojected as a completion claim.

## [v0.7.5](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.5) — Phase 6 tools + environment fluency

*2026-07-10 · `pip install blueshark-forge==0.7.5`*

The harness grows better limbs, and learns the machine it's running on.

## New actions & mechanisms
- **run_tests** — one action runs the project's REAL test suite (auto-detected — no guessing pytest vs npm) and returns a COMPACT digest: the counts line, which tests failed, the assertion, and your files in the traceback. It parses pytest/unittest/go/cargo/jest and filters site-packages noise out of the frames — surfacing exactly the short summary that a raw-output cut buries. A known runner you invoke by hand via bash is digested too.
- **Atomic multi-edit** — `edit_file` learns an `edits:[{old,new},…]` array applied validate-first, all-or-nothing: a rename + its call sites in one turn instead of one edit per turn. Also fixes a loop-detector false-positive on consecutive distinct edits to one file.
- **Background-process supervisor** — a crashed background server is now visible the instant it dies (pushed into context unprompted), and a live roster of pids/logs rides bash observations so they survive compaction.

## Environment fluency
The harness knows the OS but now also its *behavioral* differences: on macOS it surfaces the BSD-vs-GNU userland quirks (`timeout`→`gtimeout`, `sed -i` needs a suffix arg, no `readlink -f`/`date -d`), naming the GNU tools actually installed, and error hints gain OS-gated recovery lines. It also detects the project's real interpreter (`.venv`/`uv`/`poetry`) so "use python3" becomes "use `.venv/bin/python`".

Stdlib-only, zero runtime deps. 450 tests green.

## [v0.7.4](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.4) — 'act, don't narrate' guard

*2026-07-10 · `pip install blueshark-forge==0.7.4`*

A harness fix surfaced by running forge through a real, standard agentic coding benchmark (bare vs full harness ablation).

## What the benchmark caught
Running the same model (qwen3-coder:30b) with the harness off vs on, the *full* harness scored **worse** (0/5 vs bare 2/5) — not because the model was worse, but because of a harness defect: the model would read the files, emit a `say` like *"I'll implement it… let me analyze"*, and the harness accepted that as the finished turn and stopped — before writing a single line. The done-gate only bounces a `say` after file mutations, so a read-then-narrate turn slipped through.

## The fix
In autonomous mode, a `say` that only *describes* upcoming file work after a zero-mutation turn is now bounced once — "do the work, don't narrate your plan." This took the full harness from **0/5 → 3/5** on the slice, eliminating the regression. It also helps small models, which are especially prone to narrate-and-bail.

This is the "test the harness, not the model" methodology in action: an ablation on a real benchmark found a genuine scaffolding bug that a raw pass-rate would have hidden.

Stdlib-only, zero runtime deps. 423 tests green.

## [v0.7.3](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.3) — Trial fixes + send() decomposition

*2026-07-10 · `pip install blueshark-forge==0.7.3`*

Fixes surfaced by a live end-to-end trial (a real 7B and 30B model fixing a buggy project through the CLI), plus a readability refactor of the core loop.

## Trial-driven fixes
- **edit_file** now hands back a ready-to-use line-anchored template when the `old` snippet doesn't match, so models don't have to reproduce it byte-exactly (they routinely can't).
- **detect_test_cmd** recognizes a bare root-level `test_*.py` / `*_test.py`, so the synchronous done-gate auto-verifies for the common small-project layout.
- **Accurate stuck message** on a single-model ladder ("no stronger local model to escalate to" instead of a false "even after escalating").
- **write_file** ensures a single final newline (quieter diffs).

## Refactor
- The 500-line `send()` loop is decomposed into named single-responsibility methods (`_begin_turn`, `_prepare_and_generate`, `_handle_malformed`, `_execute_and_record`, `_handle_failure`, `_escalate`, `_do_fleet_send`, `_log_step`). Control-altering phases use explicit signals so the loop's early-exit semantics are preserved exactly — verified by the full test suite AND real 7B/30B inference.

Stdlib-only, zero runtime deps. 419 tests green.

## [v0.7.2](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.2) — Review cleanup pass

*2026-07-10 · `pip install blueshark-forge==0.7.2`*

A bug-fix release clearing the remaining lower-severity findings from the adversarial review (20 fixes, one commit each).

## Fixes
- **Resource leaks:** background-log fds, session wake-pipe fds, and the throwaway replay tempdir are all released now.
- **Crash-safety:** the TRUST daemon's state files are written atomically (crash mid-write no longer resets them).
- **Fleet concurrency:** the Claude-bridge inbox read-modify-write is flock-serialized; the wake-pipe drain is now atomic with the inbox swap (no lost wakeups).
- **Verification accuracy:** interpreter inline-code runs (`python -c`, `node -e`) count as file-mutating; the read-before-edit gate falls back to a content sha1 when mtime+size look unchanged.
- **Resilience / graceful degradation:** deeply-nested Python no longer aborts the symbol index; a missing `git`, an empty model ladder, non-numeric `offset`/`limit`, non-string action fields, and a transient 400 all degrade cleanly instead of crashing or persisting bad state.
- **Fidelity:** edits preserve CRLF line endings and executable bits; a subset index refresh no longer clobbers the cache.

Stdlib-only, zero runtime deps. 416 tests green.

## [v0.7.1](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.1) — Adversarial-review hardening

*2026-07-10 · `pip install blueshark-forge==0.7.1`*

A bug-fix release hardening forge against the HIGH-tier and corruption-class findings from a six-agent adversarial review of the whole codebase.

## Fixes
- **Stream parsers** no longer crash on hostile frames — a non-object JSON frame (`{"message": null}`, a bare number/array/string) or a non-string `content` previously raised an uncaught error that killed the turn or silently truncated the stream.
- **Anchored edits** can no longer silently delete lines — `end_line` is now validated against the range actually read, and a blank anchor matches only a truly identical line.
- **TRUST daemon** no longer permanently suppresses a done-claim after a transient verify timeout/error (the claim is marked seen only after a verdict is produced).
- **Verify subprocess** is bounded and group-killed on timeout, so it can't abort the tick or orphan the real test process.
- **A bad paste no longer quits the REPL** — an undecodable stdin byte is skipped instead of being read as EOF.
- **Writes are atomic** (temp-file + `os.replace`, mode-preserving) so a mid-write failure can't truncate the original file.
- Plus: paths-with-spaces handling, unreadable-store resilience, concurrent-passport-write safety, and cleaner error surfacing.

Stdlib-only, zero runtime deps. 415 tests green.

## [v0.7.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.7.0) — Phase 5: Small-model amplification

*2026-07-08 · `pip install blueshark-forge==0.7.0`*

Phase 5 complete — the harness scaffolding that makes small local models reliable.

## Highlights
- **Model passports (P5.8)** — forge measures each model (active ~90s setup probe + passive live telemetry) and auto-tunes its own knobs: tighter loop threshold for loop-prone models, bigger `num_predict` for write-file truncators, hotter retry schedule for greedy-identical failers. New `forge passport`.
- **Step-scoped borrowing + unified stuck ledger (P5.7)** — one weighted per-turn stuck score; buy a single strong-rung generation at the exact decision points the cheap model is stuck, then decay back.
- **Self-harvested few-shot exemplars (P5.6)** — malformed-retry nudges quote the model's own past valid actions.
- **Retry-heat temperature (P5.5)** — greedy first try, perturb on each nudge, reset on success.
- **Deterministic JSON salvage (P5.4)** — recover fenced / prose-prefixed / trailing-comma action JSON before counting a malformed strike.
- **Line-anchored edit dialect (P5.3)**, **dry-run verifier + best-of-N resample (P5.2)**, **state-dependent action grammar (P5.1)**.

Stdlib-only, zero runtime deps. 403 tests green.

## [v0.6.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.6.0) — Phase 3: Measurement

*2026-07-07 · `pip install blueshark-forge==0.6.0`*

Phase 3 makes forge's core thesis — *the harness makes small models capable* — measurable and regressable.

## Highlights
- **`forge bench`** — task-eval harness reporting **harness-lift** (same model bare vs full) with per-lever ablation across 8 switchable levers (schema, workspace, plan-pin, loop-detect, read-gate, alias-repair, escalation, compaction). Default = all levers on = byte-for-byte unchanged.
- **Structured step trace** — one schema-versioned `meta` record per session + one `step` record per loop iteration; the formerly-invisible malformed/loop/compaction events now log durably. New **`forge trace <sid|last>`**.
- **Flight recorder + `forge replay`** — records raw model output per step (malformed raws included) into a `RecordingBackend` cassette; re-drives a real `Agent.send` with **no model**, turning any recorded session into a zero-inference regression fixture.
- **Agent-loop invariant battery** — direct tests for escalation, loop detection, compaction, plan pinning, and inbox absorption.

Suite: **215 tests green**, stdlib-only, zero runtime deps.

## [v0.5.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.5.0) — forge v0.5.0 — /files folder explorer

*2026-07-07 · `pip install blueshark-forge==0.5.0`*

## `/files` — a Miller-column folder explorer in the terminal

Three panes like ranger/Finder, dependency-free: **parent · current · preview** (file contents; directories show their entries; binaries show size). Runs on the alternate screen — your conversation is restored pixel-perfect on close.

- `↑↓` move · `←→` navigate
- `Enter` on a file **attaches it to your next message** as `@file` (prompt pre-filled)
- `.` toggles hidden files · `q`/`Esc` closes

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.4.1...v0.5.0

## [v0.4.1](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.4.1) — forge v0.4.1

*2026-07-07 · `pip install blueshark-forge==0.4.1`*

## Fix

- **No more guessed project types** — the workspace briefing and banner only claim a project type from a real manifest at the root (`go.mod`, `package.json`, `pyproject.toml`, …), naming the marker. A directory with stray source files (like a home dir with one `.go` file) gets no label at all.

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.4.0...v0.4.1

## [v0.4.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.4.0) — forge v0.4.0 — background processes

*2026-07-07 · `pip install blueshark-forge==0.4.0`*

## Servers keep running while the agent tests them

- **`bash {command, background: true}`** — starts the process detached, returns **immediately** with its pid + a live log file, and keeps it running while the agent continues: curl it, run a client, `tail` the log, `kill` the pid.
- A trailing `&` triggers the same path automatically (`&&` chains do not).
- An instant crash — taken port, compile error — is caught within ~1s and reported with its real output instead of pretending to run.
- All background processes are cleaned up when the forge session exits.

Also includes the v0.3.1 pathless write_file fix.

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.3.1...v0.4.0

## [v0.3.1](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.3.1) — forge v0.3.1

*2026-07-07 · `pip install blueshark-forge==0.3.1`*

## Fix: pathless write_file soft-lock

Small models sometimes emit `write_file`/`edit_file` with the path under a different key (`filename`, `file`, …) or missing entirely. That used to hit an internal guard with a confusing block, invisibly — the model would spiral into mkdir/touch/echo workarounds.

- path aliases honored: `filename` / `file` / `filepath` / `file_path` / `name`
- truly pathless actions get an instructive error showing the exact JSON to re-send
- the read-before-edit guard only triggers on real files and now logs to the transcript

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.3.0...v0.3.1

## [v0.3.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.3.0) — forge v0.3.0 — plan/auto/manual modes + live message queue

*2026-07-07 · `pip install blueshark-forge==0.3.0`*

## Modes — `shift+tab` cycles (or `/mode`)

- **auto** — acts freely, no questions (the default, unchanged)
- **plan** — read-only: investigates with read/grep/glob/list, presents a plan via `say`; mutating actions are blocked with guidance
- **manual** — every mutating action (`bash`, `write_file`, `edit_file`, `fleet_send`) asks first:
  - `y` — yes, once
  - `a` — always: saves the approval (bash by command head like `bash:git`, others by action type) so it never asks again
  - `n` — no: the model is told not to retry as-is

## Queue messages while it works

The input box stays live during a run — type and press Enter to deliver a message the agent absorbs **between steps** and steers by. Anything not absorbed becomes the next turn. The approval question paints in the activity row above the box.

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.2.3...v0.3.0

## [v0.2.3](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.2.3) — forge v0.2.3 — clean rendering

*2026-07-07 · `pip install blueshark-forge==0.2.3`*

## Clean rendering

- **Word-wrapped replies** — reply text wraps at word boundaries with a 2-space margin, streamed or whole; no more mid-word hard breaks (`CONTRIBU` / `TING.md`)
- **Rounded input box with two writable lines** — long input wraps onto a second row as you type instead of scrolling horizontally; overflow keeps the cursor visible
- Footer layout: activity spinner · boxed input · status line

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.2.2...v0.2.3

## [v0.2.2](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.2.2) — forge v0.2.2

*2026-07-06 · `pip install blueshark-forge==0.2.2`*

## Fixes

- **Working spinner in the output flow** — the `working…` indicator now sits just above the input (where output streams), not below it; footer is now activity · rule · input · status
- **Roster-format targets resolve** — `fleet_send` accepts `name(sid-prefix)` exactly as the session list displays it
- **Self-excluding target match** — messaging a name you share with another session picks the *other* session instead of failing as ambiguous
- Ambiguity errors now suggest id-prefix targeting

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.2.1...v0.2.2

## [v0.2.1](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.2.1) — forge v0.2.1

*2026-07-06 · `pip install blueshark-forge==0.2.1`*

## Fixes

- **Ctrl-C no longer crashes with a traceback** — it now stops the agent gracefully; pressing Esc/Ctrl-C a second time force-returns the prompt if a step is stuck
- **Esc interrupts long-running commands** — a slow `find`/build is killed immediately (whole process group) instead of running to completion
- **`fleet_send` target `list`** — the agent can now answer "what sessions are you connected to" with the full cross-runtime roster (forge + Claude Code)

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.2.0...v0.2.1

## [v0.2.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.2.0) — forge v0.2.0 — one fleet with Claude Code

*2026-07-06 · `pip install blueshark-forge==0.2.0`*

## One fleet with Claude Code

forge now speaks the Claude Code fleet wire protocol. On a machine running Claude Code with a fleet channel:

- **Unified board** — `forge status` lists Claude Code sessions alongside forge sessions (task, last prompt, last reply), and Claude Code's fleet board sees forge sessions.
- **Cross-runtime messaging, both directions** — `forge send` / the agent's `fleet_send` reach Claude Code sessions (arriving as channel events mid-conversation); Claude Code's `fleet_send` reaches forge sessions (arriving in the agent's inbox mid-work).
- **Zero config** — every forge session registers in the shared inbox registry and accepts the fleet's authenticated `POST /send`.
- **`forge setup` checks interop on any machine** — detects Claude Code, prepares the shared token, and reports exactly what works. Without Claude Code, forge's native fleet works standalone.

Verified live in both directions before release.

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.1.1...v0.2.0

## [v0.1.1](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.1.1) — forge v0.1.1

*2026-07-06 · `pip install blueshark-forge==0.1.1`*

## What's new

### Claude-Code-grade terminal
- Bottom-pinned input with a scroll-region transcript, live status line (model · context %), boxed diffs, and a welcome banner
- Live plan panel and clean tool-step rendering with pass/fail + timing

### Line-editor polish
- Delete/Home/End keys and readline-style Ctrl-A/E (home/end), Ctrl-U/K/W (kill line-start / line-end / word)
- Multi-byte UTF-8 input decoded correctly
- Delete key no longer inserts a stray `~`
- Terminal resize (SIGWINCH) re-pins the scroll region and repaints the footer
- Long input lines no longer wrap into the pinned footer
- Arrow keys pressed mid-run no longer stop the agent — bare Esc still does

**Full changelog**: https://github.com/hackspaces/blueshark-forge/compare/v0.1.0...v0.1.1

## [v0.1.0](https://github.com/hackspaces/blueshark-forge/releases/tag/v0.1.0) — forge v0.1.0

*2026-07-06 · `pip install blueshark-forge==0.1.0`*

A model-agnostic agentic runtime for the terminal — any local model becomes a capable agent, because the intelligence lives in the harness, not the weights.

**Install**
```bash
pipx install blueshark-forge   # or: pip install blueshark-forge
forge setup                    # detects your hardware / picks an engine + models
forge                          # open it in any repo and talk to it
```

**Highlights**
- Runs any local model via Ollama, or any OpenAI-compatible server (vLLM, llama.cpp, MLX, LM Studio, TGI, SGLang, cloud)
- Harness levers that make small models capable: constrained decoding, living plan, surgical edits, self-correction, read-before-edit, a local model ladder that escalates when stuck
- Workspace + machine awareness, streaming TUI (Esc to clear/stop), honest per-model context management
- Native multi-agent fleet: session tracking, token-authenticated messaging, independent verification, coordination, shared learnings
- Stdlib-only, 34 tests, CI across Python 3.10–3.13
