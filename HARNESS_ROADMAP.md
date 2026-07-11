# BlueShark Forge — True Harness Roadmap

> A post-v0.8 execution plan for turning Forge from a capable coding-agent loop into a
> durable, evidence-driven agent runtime. This file is written for agents: one slice is
> one branch, one bounded implementation, and one pull request.

## Why this document exists

Forge already owns constrained actions, tools, context management, model escalation,
authority, completion policy, replay, fault injection, fleet communication, and corpus
generation. The next step is not to add more prompt instructions. It is to move the
remaining truth, control, safety, and recovery decisions out of the model and into the
harness.

The target invariant is:

> The model proposes. The harness authorizes, executes, observes, verifies, records, and
> decides whether the task is complete.

`ROADMAP.md` remains the detailed historical backlog. This document is the focused
post-v0.8 architecture track. If the two documents overlap, re-check current `main` and
follow the implementation that already exists. Never reimplement a completed mechanism
only because an older roadmap entry says it is open.

## Non-negotiable constraints

- Python 3.10+ and no third-party runtime dependencies.
- Any model and any OpenAI-compatible backend must remain replaceable.
- The model must never grant itself authority or manufacture evidence.
- File, shell, network, fleet, and external-service effects must be explicit.
- A failed or missing verifier must never silently become a verified completion.
- Every important decision must be reconstructable from the event log.
- Old transcripts and fixtures remain readable across protocol changes.
- Untrusted project output, fleet messages, and model text are data, not authority.
- Normal coding workflows must keep working while stricter contracts are introduced.

## How agents take work in slices

1. Start from the latest `main`; do not branch from another roadmap PR.
2. Pick exactly one slice marked **ready** whose dependencies are done.
3. Re-read the named code areas. File names are guidance; current code is the source of
   truth.
4. Write a five-line implementation note in the PR before coding: invariant, files,
   compatibility risk, tests, and explicit non-goals.
5. Keep the branch limited to the slice. If another gap is discovered, document it as a
   follow-up instead of absorbing it.
6. Add focused unit tests, then run the complete suite:

   ```bash
   python -m unittest discover -s tests -v
   ```

7. Run relevant replay/fault/bench checks when the slice touches agent decisions.
8. Update this file's status in the implementation PR: `ready` → `in progress` →
   `done (PR #N)`.

Branch names use `agent/hXX-short-name`. PR titles use `harness: <outcome>`.

## Universal definition of done

A slice is complete only when:

- Its acceptance checks below pass.
- New public records have a version and backwards-compatible reader.
- Failure paths are tested, not only the happy path.
- The model cannot bypass the mechanism by changing its wording or output shape.
- User-visible behavior is documented.
- The full Python 3.10–3.13 test matrix is green.
- No unrelated refactor is bundled into the PR.

## Dependency and parallel-work map

| Lane | Ordered slices | Parallel guidance |
|---|---|---|
| Integrity | H00 → H04 → H05 | H00 can run beside H01 and H06 |
| Runtime kernel | H01 → H02 → H03 | Keep one agent on this chain at a time |
| Safe execution | H06 → H07 → H08 | Can run beside the runtime-kernel chain after H01 |
| Durability and fleet | H02 → H09 → H10 | H09 can start after the reducer API stabilizes |
| Replay and measurement | H03 → H11 → H12 | Can run beside H06–H10 |
| Process intelligence | H03 → H13 → H14 | H13 can run beside H11 |
| Learning | H04 + H12 → H15 → H18 | Do not train from unverified legacy rows by default |
| Extensibility | H06 → H16 → H17 | Domain work stays out of the core until H16 lands |

---

## Wave 0 — Make the harness prove itself

### H00 — CI and release evidence

**Status:** ready  
**Depends on:** none  
**Primary areas:** `.github/workflows/test.yml`, `.github/workflows/publish.yml`, packaging

**Outcome.** Every commit and release has machine-visible evidence matching the standard
Forge expects from its own agents.

**Scope.** Preserve the existing Python 3.10–3.13 test matrix. Add workflow timeouts and
concurrency cancellation; gate publishing on tests; assert release tag equals
`forge.__version__`; build a wheel and smoke-test `forge --version` and `forge --help`
before publishing. Pin actions by immutable commit where practical.

**Non-goals.** No product features and no broad lint/type migration.

**Acceptance.** A deliberately failing test blocks CI; a mismatched tag blocks publish;
the built wheel installs into a clean environment and starts the CLI.

### H01 — First-class task contract

**Status:** ready  
**Depends on:** none  
**Primary areas:** `forge/execution.py`, session metadata, CLI/run construction

**Outcome.** Each run has a harness-owned contract defining the goal, allowed effects,
invariants, verification obligations, approval requirements, and budgets.

**Scope.** Add a versioned `TaskContract` with backwards-compatible defaults. It must be
serializable into the session's initial metadata and recoverable on resume/replay. Begin
with fields the runtime can enforce deterministically; retain a structured extension map
for future domain adapters.

**Non-goals.** Do not enforce state transitions or implement sandboxing in this slice.

**Acceptance.** Contract round-trip tests; legacy sessions receive an equivalent default
contract; malformed or unknown contract versions fail closed with a useful error.

---

## Wave 1 — Make the runtime protocol authoritative

### H02 — Authoritative execution reducer

**Status:** blocked by H01  
**Depends on:** H01  
**Primary areas:** `forge/execution.py`, `forge/agent.py`, transcript logging

**Outcome.** The state machine controls the loop instead of only projecting legacy records
after decisions have happened.

**Scope.** Introduce a pure reducer from `(state, event, task_contract)` to a transition
decision. The agent loop asks the reducer before executing an action and before accepting
completion. Illegal transitions generate a versioned rejection event and deterministic
recovery state. Preserve the existing projector as a compatibility reader.

**Non-goals.** No tool redesign and no new user-facing mode.

**Acceptance.** Mutation cannot jump directly to verified completion; denied and failed
events take the documented recovery path; replay reconstructs the same state sequence.

### H03 — Complete action lifecycle

**Status:** blocked by H02  
**Depends on:** H02  
**Primary areas:** execution events, tool dispatch, background processes, replay

**Outcome.** Every attempted effect has a stable identity and exactly one terminal result.

**Scope.** Add run ID, turn ID, action ID, parent action ID, attempt number, timestamps,
and terminal outcome. Represent requested, authorized, started, succeeded, failed,
cancelled, timed out, and indeterminate outcomes. Make duplicate terminal events invalid.

**Non-goals.** Do not implement a durable scheduler yet.

**Acceptance.** A foreground action, background action, denial, timeout, cancellation, and
process crash each produce a complete lifecycle; retries receive new attempt IDs while
retaining their causal parent.

### H04 — Evidence receipt v2

**Status:** blocked by H03  
**Depends on:** H03  
**Primary areas:** `forge/execution.py`, verification, receipts, git/workspace inspection

**Outcome.** Completion evidence identifies what changed, what checked it, and the exact
workspace state that was judged.

**Scope.** Extend evidence with pre/post workspace identity, changed-path digest, verifier
identity/version, command, exit code, output digest, timestamp, relevant artifacts,
unverified assumptions, authority decision, and task-contract ID. Use the git tree when
available and a deterministic manifest otherwise. Keep old receipts readable.

**Non-goals.** No cryptographic signing service.

**Acceptance.** Changing a file after verification invalidates the receipt; identical
workspaces produce identical manifests; opaque bash mutations are never represented as a
specific verified file list unless the harness measured it.

### H05 — No unapproved failed-verification escape

**Status:** blocked by H04  
**Depends on:** H04  
**Primary areas:** `CompletionPolicy`, REPL approvals, receipts

**Outcome.** A repeated model claim cannot convert failed verification into success.

**Scope.** Retain audit/balanced/strict compatibility, but require a recorded human
approval or explicit contract policy for any failed-verification override. The resulting
state is `accepted_unverified`, never `verified`. Strict contracts cannot be overridden by
model repetition.

**Non-goals.** Do not remove audit mode.

**Acceptance.** Repeating `say` after failed tests remains rejected; approval survives
pause/resume and appears in the receipt; non-interactive strict runs exit non-zero when
proof is missing.

---

## Wave 2 — Replace guessed safety with explicit capabilities

### H06 — Tool effect and capability declarations

**Status:** ready  
**Depends on:** H01 recommended, not required  
**Primary areas:** `forge/tools.py`, `forge/authority.py`, action schema

**Outcome.** Policy evaluates declared effects rather than inferring all risk from an
action name or raw shell string.

**Scope.** Each tool declares possible filesystem, process, network, secret, external
message, and irreversible effects plus minimum authority and approval policy. The
contract and session authority produce a deterministic capability grant. Preserve the
shell classifier as defense-in-depth for undeclared shell details.

**Non-goals.** Do not claim containment before H07.

**Acceptance.** Tests cover every built-in tool; unknown tools/effects fail closed;
stronger models receive no extra capabilities; denials name the missing capability.

### H07 — Executor boundary and real isolation adapters

**Status:** blocked by H06  
**Depends on:** H06  
**Primary areas:** tool execution, subprocess handling, platform setup/doctor

**Outcome.** Agent decisions are separated from where and how effects execute.

**Scope.** Define an executor interface and keep the current local executor as an explicit
unsafe-compatible mode. Add a capability-aware isolated executor using available OS or
container primitives, with honest platform detection and graceful refusal when a required
guarantee is unavailable. Network and filesystem scopes must be enforceable, not prompts.

**Non-goals.** No bundled container runtime and no false claim of cross-platform parity.

**Acceptance.** Escape attempts cannot touch a protected fixture outside the workspace in
isolated mode; unavailable isolation produces a clear blocked result; local mode remains
backwards compatible and visibly labelled.

### H08 — Transactional workspace

**Status:** blocked by H07  
**Depends on:** H07  
**Primary areas:** workspace setup, git integration, receipts, cancellation

**Outcome.** A task changes an isolated candidate workspace and publishes only a verified
transaction.

**Scope.** Use a temporary git worktree/branch when possible and a copied manifest-backed
workspace otherwise. Support commit, discard, inspect diff, and controlled promotion.
Cancellation or crash leaves the user's starting tree untouched.

**Non-goals.** No automatic merge to `main` and no remote push.

**Acceptance.** Failed verification discards cleanly; successful promotion matches the
receipt's post-state; dirty starting trees are preserved; cleanup is idempotent.

---

## Wave 3 — Durable execution and trustworthy fleets

### H09 — Event-sourced checkpoints and crash recovery

**Status:** blocked by H02  
**Depends on:** H02, H03 recommended  
**Primary areas:** sessions, execution reducer, resume, process lifecycle

**Outcome.** A killed process can resume from committed events without guessing what
happened or duplicating an effect.

**Scope.** Add atomic event commits, reducer snapshots, schema migration, action
reconciliation, and explicit indeterminate recovery. Resume from the last committed
state; never treat an action that may have executed as safely retryable without checking
its idempotency/reconciliation rule.

**Non-goals.** No distributed queue.

**Acceptance.** Kill tests at each action lifecycle boundary; recovery neither loses a
committed result nor blindly repeats an indeterminate mutation; corrupt tail records are
quarantined with the last valid offset reported.

### H10 — Task graph, leases, and independent verification

**Status:** blocked by H08 and H09  
**Depends on:** H08, H09  
**Primary areas:** fleet, daemon, workspace ownership, verifier

**Outcome.** Multiple agents work on explicit tasks without silently racing on the same
state, and the verifier is separated from the executor.

**Scope.** Add task IDs, dependencies, ownership leases, heartbeats, expiry/recovery,
workspace or path claims, handoff records, and merge results. The verifier receives the
contract, candidate workspace, and receipt—not the executor's private reasoning—and has
no mutation capability.

**Non-goals.** No consensus protocol and no arbitrary internet-distributed fleet.

**Acceptance.** Conflicting leases are rejected; abandoned work is reclaimable; verifier
cannot mutate the candidate; a task completes only when dependencies and policy pass.

---

## Wave 4 — Replay, fault injection, and measurable harness lift

### H11 — Full-fidelity workspace replay

**Status:** blocked by H03  
**Depends on:** H03; benefits from H08  
**Primary areas:** `forge/replay.py`, fixtures, workspace manifests

**Outcome.** Replay can reproduce the environment half of a recorded run, not only the
model-output and harness-decision half.

**Scope.** Record a content-addressed workspace fixture: base git commit plus patch and
untracked fixture files, or a bounded manifest for non-git workspaces. Redact/exclude
secrets and large artifacts by policy. Restore into a throwaway workspace and compare
observations, effects, states, and terminal receipt.

**Non-goals.** Never archive an arbitrary home directory or credentials.

**Acceptance.** A mutation/test trace reproduces the same observations and receipt;
missing/redacted inputs yield an explicit fidelity limitation; replay never touches the
original workspace.

### H12 — Outcome, safety, cost, and recovery evaluation matrix

**Status:** blocked by H11  
**Depends on:** H11  
**Primary areas:** bench, faults, passports, reports

**Outcome.** Harness changes are accepted on measured outcomes rather than anecdotes.

**Scope.** Standardize task success, invariant violations, false completion, corruption,
recovery rate, tool efficiency, tokens, latency, escalations, approvals, and cost. Run
bare/full and per-lever ablations over deterministic fixtures and model/passport cohorts.
Add regression thresholds without pretending stochastic model runs are byte-identical.

**Non-goals.** No leaderboard claim from a tiny private fixture set.

**Acceptance.** Machine-readable and human reports agree; a deliberately broken lever is
caught; zero-inference replay results and live-model results are clearly distinguished.

---

## Wave 5 — Process intelligence from the event log

### H13 — XES/OCEL-compatible event export

**Status:** blocked by H03  
**Depends on:** H03  
**Primary areas:** event schema, CLI export, documentation

**Outcome.** Forge runs become process-mining data without log-scraping.

**Scope.** Define stable case, activity, timestamp, lifecycle, resource/model, tool,
workspace, task, state, outcome, parent, and evidence attributes. Export dependency-free
CSV/JSON plus standards-compatible XES or OCEL output. Apply deterministic secret
redaction before export.

**Non-goals.** No embedded process-mining library.

**Acceptance.** Export round-trips core identities; parallel actions preserve lifecycle
and causality; fixtures contain no raw secrets; schema version is explicit.

### H14 — Conformance and behavioral counterexamples

**Status:** blocked by H13 and H12  
**Depends on:** H13, H12  
**Primary areas:** trace analysis, reports, fault generation

**Outcome.** Forge can compare intended execution rules with actual agent behavior and
turn violations into reproducible regression cases.

**Scope.** Add deterministic transition/conformance checking against the authoritative
runtime protocol. Report unexpected paths, loops, skipped obligations, repeated recovery
failures, and high-cost variants. Convert a selected violation into a minimized replay or
fault fixture.

**Non-goals.** No claim that discovered behavior is automatically a correct specification.

**Acceptance.** Seeded violations are detected and localized to the first divergence;
valid alternate paths remain conforming; minimized counterexamples reproduce the failure.

---

## Wave 6 — A clean learning flywheel

### H15 — Verified trajectory corpus and credit assignment

**Status:** blocked by H04 and H12  
**Depends on:** H04, H12  
**Primary areas:** `forge/corpus.py`, receipts, redaction, dataset metadata

**Outcome.** Training data represents actions that contributed to verified outcomes, not
merely tool calls whose immediate observation said `ok`.

**Scope.** Join actions to final task receipts and classify verified success, accepted
unverified, failure, correction, recovery, redundant action, and policy violation. Export
SFT/preferences with provenance, model/passport, harness version, contract, outcome,
weights, and redaction metadata. Default training export to verified successful tasks.

**Non-goals.** No trainer or GPU framework dependency inside Forge.

**Acceptance.** Failed trajectories do not enter the default SFT split; useful recovery
pairs survive; a successful but causally irrelevant action is not labelled equally to the
verified repair; secrets are removed before writing JSONL.

---

## Wave 7 — General harness, domain-specific drivers

### H16 — Versioned tool-driver registry

**Status:** blocked by H06  
**Depends on:** H06  
**Primary areas:** action schema, tools, executor, configuration

**Outcome.** New capabilities can be added without editing a monolithic action switch and
still inherit schema validation, authority, tracing, replay, shaping, and receipts.

**Scope.** Define a tool driver contract: identity/version, input schema, declared effects,
capabilities, dry run, execute, reconcile, result schema, redaction, and verifier hooks.
Convert built-ins first. Load user drivers only through an explicit allowlist. Consider a
minimal stdlib MCP adapter only after the registry contract is proven.

**Non-goals.** No arbitrary auto-loading of executable files and no capability-by-name.

**Acceptance.** A fixture driver works end-to-end; malformed drivers fail closed; built-in
behavior and constrained schemas remain compatible; driver version appears in receipts.

### H17 — Domain-pack boundary and reference adapters

**Status:** blocked by H16 and H01  
**Depends on:** H16, H01  
**Primary areas:** extension documentation, examples, contract extensions

**Outcome.** Forge can safely host non-coding agents without contaminating the core with
domain-specific policy.

**Scope.** Specify how a domain pack contributes tools, task-contract fields, invariants,
verifiers, redactors, and examples. Build small offline reference packs for process-log
analysis and black-box automata experiments. Document a Canvas LMS pack contract using a
fake server/fixtures; do not require live credentials in core tests.

**Non-goals.** No production Canvas integration or bundled external SDK.

**Acceptance.** Each reference pack runs offline, cannot exceed its declared effects, and
produces a domain-specific evidence receipt through the unchanged core runtime.

### H18 — Gated self-improvement

**Status:** blocked by H08, H10, H12, and H15  
**Depends on:** H08, H10, H12, H15  
**Primary areas:** evolve workflow, worktrees, evaluation, fleet verification

**Outcome.** Forge may propose improvements to Forge, but only as reviewable branches with
deterministic evidence.

**Scope.** Consume measured regressions or curated backlog items, create an isolated
worktree, run an implementation agent, and gate the candidate on unit tests, replay,
benchmark non-regression, invariant checks, and independent verification. Produce a branch
and receipt; never merge, publish, or expand authority automatically.

**Non-goals.** No autonomous self-deployment.

**Acceptance.** A good fixture change produces a reviewable branch and complete receipt;
a benchmark regression or policy violation blocks it; interruption leaves `main` and the
user workspace untouched.

---

## Cross-cutting review checklist

Every slice reviewer should ask:

- Could the model forge or bypass this decision?
- What happens after kill, timeout, duplicate delivery, or partial execution?
- Is the failure explicit in the event log and CLI?
- Does replay reproduce it?
- Does the mechanism preserve old sessions and receipts?
- Does stronger model capability accidentally expand runtime authority?
- Are secrets removed before transcript, export, learning, or fleet transfer?
- Is the claimed guarantee actually enforced by the executor, or merely classified?
- Can success be checked without asking an LLM to judge its own work?

## Slice handoff template

Copy this into the PR body:

```markdown
Roadmap slice: HXX
Invariant introduced:
Dependencies confirmed:
Files intentionally changed:
Compatibility risk:
Failure cases tested:
Replay/bench evidence:
Explicit non-goals:
Follow-ups discovered:
```

When these slices are complete, Forge is no longer just a coding loop with good
scaffolding. It is a replaceable-model runtime whose behavior is governed by contracts,
contained capabilities, authoritative state, durable events, independent evidence, and
measurable recovery.
