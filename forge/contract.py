"""Harness-owned task contract (H01).

The enforceable constraints a run operates under, captured once at run start,
serialized into the session's `meta` record, and recoverable on resume/replay.

The contract records ONLY what the runtime already enforces deterministically —
the run's authority (which action kinds are legal), the completion/verification
obligation, the approval mode, and the step budget — so it is a faithful,
inspectable statement of the run's guardrails rather than an aspiration. Later
slices (the execution reducer, evidence receipts) enforce AGAINST this contract;
here it is additive provenance and changes no behavior.

`extensions` is a structured, additive map reserved for future domain adapters.
Old transcripts with no `contract` field remain readable: `from_dict(None)` yields
a permissive default so resume/replay of a pre-contract session still works.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

CONTRACT_VERSION = 1


@dataclass(frozen=True)
class TaskContract:
    goal: str = ""                       # the run's task ("" for an open interactive session)
    mode: str = "auto"                   # approval requirement: auto | plan | manual
    authority: str = "operator"          # observe | contribute | operator | admin
    allowed_actions: List[str] = field(default_factory=list)   # action kinds legal at `authority`
    completion_policy: str = "balanced"  # audit | balanced | strict
    requires_verification: bool = True   # changed files need passing verification before "done"
    max_steps: int = 0                   # step budget for the run (0 = unset)
    version: int = CONTRACT_VERSION
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def permits(self, action_kind: str) -> bool:
        """Whether an action kind is within the contract's allowed effects."""
        return action_kind in self.allowed_actions

    @classmethod
    def from_dict(cls, value: Any, *, fallback_mode: str = "auto") -> "TaskContract":
        """Rebuild a contract from a meta record's `contract` field. A legacy meta
        with no contract yields a permissive default stamped with the legacy mode,
        so pre-contract transcripts stay readable and resumable."""
        if not value:
            return cls(mode=fallback_mode)
        return cls(
            goal=str(value.get("goal", "")),
            mode=str(value.get("mode", fallback_mode)),
            authority=str(value.get("authority", "operator")),
            allowed_actions=list(value.get("allowed_actions", [])),
            completion_policy=str(value.get("completion_policy", "balanced")),
            requires_verification=bool(value.get("requires_verification", True)),
            max_steps=int(value.get("max_steps", 0) or 0),
            version=int(value.get("version", 1) or 1),
            extensions=dict(value.get("extensions", {}) or {}),
        )


def from_runtime(*, goal: str, mode: str, authority_level: str, allowed_actions,
                 completion_policy_mode: str, max_steps: int,
                 extensions: Dict[str, Any] = None) -> TaskContract:
    """Build the contract from a run's ALREADY-RESOLVED runtime config — the values
    the harness will actually enforce. Kept free of authority/execution imports so
    the contract type stays a plain, serializable record."""
    return TaskContract(
        goal=goal or "",
        mode=mode or "auto",
        authority=authority_level,
        allowed_actions=sorted(allowed_actions),
        completion_policy=completion_policy_mode,
        requires_verification=(completion_policy_mode != "audit"),
        max_steps=int(max_steps or 0),
        extensions=extensions or {},
    )
