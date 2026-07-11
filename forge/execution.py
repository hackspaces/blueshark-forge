"""Canonical execution states, runtime events, and completion evidence.

This module is intentionally independent from the agent loop.  Existing transcript
records remain valid while :class:`ExecutionTracker` projects them onto a stable
protocol that replay, policy, fleet, and UI consumers can share.
"""
import hashlib
import os

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


PROTOCOL_VERSION = 1


class ExecutionState(str, Enum):
    ORIENT = "ORIENT"
    INVESTIGATE = "INVESTIGATE"
    PLAN = "PLAN"
    MUTATE = "MUTATE"
    VERIFY = "VERIFY"
    DIAGNOSE = "DIAGNOSE"
    COMPLETE = "COMPLETE"


class RuntimeEvent(str, Enum):
    MODEL_REQUESTED_ACTION = "ModelRequestedAction"
    ACTION_STARTED = "ActionStarted"
    ACTION_COMPLETED = "ActionCompleted"
    WORKSPACE_CHANGED = "WorkspaceChanged"
    PROCESS_EXITED = "ProcessExited"
    VERIFICATION_PASSED = "VerificationPassed"
    VERIFICATION_FAILED = "VerificationFailed"
    LOOP_SUSPECTED = "LoopSuspected"
    MODEL_ESCALATED = "ModelEscalated"
    CONTEXT_COMPACTED = "ContextCompacted"
    COMPLETION_CLAIMED = "CompletionClaimed"
    COMPLETION_REJECTED = "CompletionRejected"
    AUTHORITY_DENIED = "AuthorityDenied"


READ_ACTIONS = frozenset(("read_file", "list_files", "grep", "glob"))
MUTATING_ACTIONS = frozenset(("write_file", "edit_file"))
VERIFY_ACTIONS = frozenset(("run_tests",))


@dataclass(frozen=True)
class VerificationEvidence:
    command: str
    exit_code: int
    artifact_digest: str = ""

    def validate(self) -> List[str]:
        errors = []
        if not self.command.strip():
            errors.append("verification command is required")
        if not isinstance(self.exit_code, int):
            errors.append("verification exit_code must be an integer")
        return errors


@dataclass(frozen=True)
class EvidenceContract:
    claim: str
    changed_files: List[str] = field(default_factory=list)
    verification: List[VerificationEvidence] = field(default_factory=list)
    unverified_assumptions: List[str] = field(default_factory=list)

    def validate(self, require_verification: bool = True) -> List[str]:
        errors = []
        if not self.claim.strip():
            errors.append("claim is required")
        if len(set(self.changed_files)) != len(self.changed_files):
            errors.append("changed_files must not contain duplicates")
        if require_verification and self.changed_files and not self.verification:
            errors.append("changed files require verification evidence")
        for item in self.verification:
            errors.extend(item.validate())
        return errors

    @property
    def verified(self) -> bool:
        return not self.validate() and all(v.exit_code == 0 for v in self.verification)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceContract":
        checks = [VerificationEvidence(**item) for item in value.get("verification", [])]
        return cls(
            claim=str(value.get("claim", "")),
            changed_files=list(value.get("changed_files", [])),
            verification=checks,
            unverified_assumptions=list(value.get("unverified_assumptions", [])),
        )


class EvidenceCollector:
    """Turn-scoped, harness-owned evidence used by the completion gate.

    Models never author this data. Mutating and verification tool outcomes feed the
    collector directly, so completion provenance is grounded in runtime facts.
    """

    def __init__(self, cwd: str):
        self.cwd = os.path.realpath(cwd)
        self.reset()

    def reset(self) -> None:
        self.changed_files = set()
        self.verification = []
        self.unverified_assumptions = []

    def record_change(self, path: str) -> None:
        if not path:
            return
        if path == "<bash>":
            self.changed_files.add(path)
            return
        real = os.path.realpath(path if os.path.isabs(path) else os.path.join(self.cwd, path))
        try:
            rel = os.path.relpath(real, self.cwd)
        except ValueError:
            rel = real
        self.changed_files.add(rel)

    def record_verification(self, command: str, ok: bool, artifact: str = "") -> None:
        digest = hashlib.sha256((artifact or "").encode("utf-8", "replace")).hexdigest()[:16]
        self.verification.append(VerificationEvidence(command, 0 if ok else 1, digest))

    def record_assumption(self, assumption: str) -> None:
        assumption = (assumption or "").strip()
        if assumption and assumption not in self.unverified_assumptions:
            self.unverified_assumptions.append(assumption)

    def contract(self, claim: str) -> EvidenceContract:
        return EvidenceContract(
            claim=claim,
            changed_files=sorted(self.changed_files),
            verification=list(self.verification),
            unverified_assumptions=list(self.unverified_assumptions),
        )


class PolicyOutcome(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ACCEPT_UNVERIFIED = "accept_unverified"


@dataclass(frozen=True)
class CompletionDecision:
    outcome: PolicyOutcome
    code: str
    reason: str
    recovery_state: Optional[ExecutionState] = None
    override_used: bool = False

    @property
    def allowed(self) -> bool:
        return self.outcome != PolicyOutcome.REJECT

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["outcome"] = self.outcome.value
        if self.recovery_state is not None:
            result["recovery_state"] = self.recovery_state.value
        return result


class CompletionPolicy:
    """Deterministic policy over harness-built completion evidence.

    Modes:
      audit     never blocks, but labels unsupported completion;
      balanced  rejects a failed check once, then permits the existing single-bounce
                escape hatch as an explicit override; accepts missing-runner cases as
                visibly unverified;
      strict    rejects every changed workspace without passing evidence.
    """

    MODES = frozenset(("audit", "balanced", "strict"))

    def __init__(self, mode: Optional[str] = None):
        selected = (mode or os.environ.get("FORGE_COMPLETION_POLICY", "balanced")).lower()
        self.mode = selected if selected in self.MODES else "balanced"

    def evaluate(self, contract: EvidenceContract, attempt: int = 1) -> CompletionDecision:
        if not contract.changed_files:
            return CompletionDecision(PolicyOutcome.ACCEPT, "no_changes",
                                      "completion did not change the workspace")
        if contract.verified:
            return CompletionDecision(PolicyOutcome.ACCEPT, "verified",
                                      "all recorded verification evidence passed")

        failed = any(item.exit_code != 0 for item in contract.verification)
        if self.mode == "audit":
            return CompletionDecision(PolicyOutcome.ACCEPT_UNVERIFIED, "audit_only",
                                      "policy is audit-only; unsupported completion was recorded")

        if failed:
            if self.mode == "balanced" and attempt > 1:
                return CompletionDecision(
                    PolicyOutcome.ACCEPT_UNVERIFIED, "single_bounce_override",
                    "verification failed, but the balanced policy permits the existing second-claim escape hatch",
                    override_used=True)
            return CompletionDecision(PolicyOutcome.REJECT, "verification_failed",
                                      "one or more verification commands failed",
                                      ExecutionState.DIAGNOSE)

        if self.mode == "strict":
            return CompletionDecision(PolicyOutcome.REJECT, "verification_missing",
                                      "changed files require passing verification evidence",
                                      ExecutionState.VERIFY)

        return CompletionDecision(PolicyOutcome.ACCEPT_UNVERIFIED, "verifier_unavailable",
                                  "no runnable verifier was available; assumptions are recorded")


@dataclass(frozen=True)
class EventEnvelope:
    event: RuntimeEvent
    source_type: str
    state_from: ExecutionState
    state_to: ExecutionState
    progress_evidence: List[str] = field(default_factory=list)
    verification_obligation: Optional[str] = None
    recovery_transition: Optional[ExecutionState] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["v"] = PROTOCOL_VERSION
        result["event"] = self.event.value
        result["state_from"] = self.state_from.value
        result["state_to"] = self.state_to.value
        if self.recovery_transition is not None:
            result["recovery_transition"] = self.recovery_transition.value
        return result


class ExecutionTracker:
    """Project legacy transcript records onto the canonical runtime protocol.

    The tracker is deterministic and side-effect free.  It remembers the active
    action only so an observation can be classified without changing the existing
    transcript schema or agent loop control flow.
    """

    def __init__(self, state: ExecutionState = ExecutionState.ORIENT):
        self.state = state
        self.active_action = ""

    def observe(self, kind: str, fields: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        event = self._classify(kind, fields)
        if event is None:
            return None
        previous = self.state
        target = self._target(event, kind, fields)
        evidence = self._evidence(event, fields)
        obligation = None
        recovery = None
        if event == RuntimeEvent.WORKSPACE_CHANGED:
            obligation = "verify the changed workspace before completion"
            recovery = ExecutionState.DIAGNOSE
        elif event == RuntimeEvent.VERIFICATION_FAILED:
            recovery = ExecutionState.DIAGNOSE
        elif event == RuntimeEvent.COMPLETION_REJECTED:
            recovery = ExecutionState.VERIFY
        elif event == RuntimeEvent.AUTHORITY_DENIED:
            recovery = ExecutionState.PLAN
        self.state = target
        return EventEnvelope(event, kind, previous, target, evidence,
                             obligation, recovery).to_dict()

    def _classify(self, kind: str, fields: Mapping[str, Any]) -> Optional[RuntimeEvent]:
        if kind == "model":
            return RuntimeEvent.MODEL_REQUESTED_ACTION
        if kind == "action":
            self.active_action = str(fields.get("action", ""))
            return RuntimeEvent.ACTION_STARTED
        if kind == "observation":
            if not fields.get("ok", False):
                return RuntimeEvent.VERIFICATION_FAILED if self.active_action in VERIFY_ACTIONS else RuntimeEvent.ACTION_COMPLETED
            if self.active_action in MUTATING_ACTIONS:
                return RuntimeEvent.WORKSPACE_CHANGED
            if self.active_action in VERIFY_ACTIONS:
                return RuntimeEvent.VERIFICATION_PASSED
            return RuntimeEvent.ACTION_COMPLETED
        if kind == "verified":
            return RuntimeEvent.VERIFICATION_PASSED if fields.get("ok", False) else RuntimeEvent.VERIFICATION_FAILED
        if kind == "loop":
            return RuntimeEvent.LOOP_SUSPECTED
        if kind in ("escalate", "borrow"):
            return RuntimeEvent.MODEL_ESCALATED
        if kind in ("compact", "struct_compact", "floor"):
            return RuntimeEvent.CONTEXT_COMPACTED
        if kind in ("narrate_bounce", "completion_rejected"):
            return RuntimeEvent.COMPLETION_REJECTED
        if kind == "authority_denied":
            return RuntimeEvent.AUTHORITY_DENIED
        if kind == "assistant":
            # A stuck/hand-off message is logged as an `assistant` record too, but the
            # agent is giving up — not claiming completion. Don't project it to COMPLETE.
            if fields.get("stuck"):
                return None
            return RuntimeEvent.COMPLETION_CLAIMED
        if kind == "inbox" and fields.get("sender") == "background" and "EXITED" in str(fields.get("text", "")):
            return RuntimeEvent.PROCESS_EXITED
        return None

    def _target(self, event: RuntimeEvent, kind: str,
                fields: Mapping[str, Any]) -> ExecutionState:
        if event == RuntimeEvent.ACTION_STARTED:
            if self.active_action in READ_ACTIONS:
                return ExecutionState.INVESTIGATE
            if self.active_action == "plan":
                return ExecutionState.PLAN
            if self.active_action in MUTATING_ACTIONS:
                return ExecutionState.MUTATE
            if self.active_action in VERIFY_ACTIONS:
                return ExecutionState.VERIFY
        mapping = {
            RuntimeEvent.WORKSPACE_CHANGED: ExecutionState.MUTATE,
            RuntimeEvent.VERIFICATION_PASSED: ExecutionState.VERIFY,
            RuntimeEvent.VERIFICATION_FAILED: ExecutionState.DIAGNOSE,
            RuntimeEvent.LOOP_SUSPECTED: ExecutionState.DIAGNOSE,
            RuntimeEvent.MODEL_ESCALATED: ExecutionState.DIAGNOSE,
            RuntimeEvent.COMPLETION_CLAIMED: ExecutionState.COMPLETE,
            RuntimeEvent.COMPLETION_REJECTED: ExecutionState.DIAGNOSE,
            RuntimeEvent.PROCESS_EXITED: ExecutionState.DIAGNOSE,
            RuntimeEvent.AUTHORITY_DENIED: ExecutionState.DIAGNOSE,
        }
        return mapping.get(event, self.state)

    def _evidence(self, event: RuntimeEvent, fields: Mapping[str, Any]) -> List[str]:
        if event == RuntimeEvent.WORKSPACE_CHANGED and self.active_action:
            return [self.active_action]
        if event in (RuntimeEvent.VERIFICATION_PASSED, RuntimeEvent.VERIFICATION_FAILED):
            cmd = fields.get("cmd") or self.active_action
            return [str(cmd)] if cmd else []
        if event == RuntimeEvent.COMPLETION_CLAIMED and fields.get("text"):
            return [str(fields["text"])[:200]]
        return []
