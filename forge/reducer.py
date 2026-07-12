"""Authoritative execution reducer (H02).

A PURE function from (state, event, task_contract) to a transition decision. The
agent loop consults it before executing an action and before accepting completion,
so the execution state machine CONTROLS the loop rather than only projecting legacy
records after decisions have already happened. Illegal transitions produce a
versioned rejection carrying a deterministic recovery state.

The projector in `execution.py` (`ExecutionTracker`) remains untouched as a
compatibility reader over existing transcripts. This reducer is deterministic:
the same event sequence always yields the same state sequence, so a replay
reconstructs it exactly.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .execution import (ExecutionState, RuntimeEvent,
                        READ_ACTIONS, MUTATING_ACTIONS, VERIFY_ACTIONS)

REDUCER_VERSION = 1

S = ExecutionState
E = RuntimeEvent

# A completion claim is legal only from states with nothing unverified pending.
_COMPLETION_READY = frozenset((S.ORIENT, S.INVESTIGATE, S.PLAN, S.VERIFY, S.COMPLETE))

# Observational events: (resulting state, recovery state or None). No gate — they
# record something that happened; a recovery state routes the loop out of trouble.
_OBSERVED = {
    E.MODEL_REQUESTED_ACTION: (None, None),        # None → stay in the current state
    E.ACTION_COMPLETED:       (None, None),
    E.WORKSPACE_CHANGED:      (S.MUTATE, S.DIAGNOSE),
    E.VERIFICATION_PASSED:    (S.VERIFY, None),
    E.VERIFICATION_FAILED:    (S.DIAGNOSE, S.DIAGNOSE),
    E.LOOP_SUSPECTED:         (S.DIAGNOSE, S.DIAGNOSE),
    E.MODEL_ESCALATED:        (S.DIAGNOSE, None),
    E.CONTEXT_COMPACTED:      (None, None),
    E.PROCESS_EXITED:         (S.DIAGNOSE, S.DIAGNOSE),
    E.COMPLETION_REJECTED:    (S.DIAGNOSE, S.VERIFY),
}


@dataclass(frozen=True)
class Transition:
    allowed: bool
    state_from: ExecutionState
    state_to: ExecutionState
    event: RuntimeEvent
    code: str
    reason: str
    recovery_state: Optional[ExecutionState] = None
    v: int = REDUCER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event"] = self.event.value
        d["state_from"] = self.state_from.value
        d["state_to"] = self.state_to.value
        d["recovery_state"] = self.recovery_state.value if self.recovery_state else None
        return d


def reduce(state: ExecutionState, event: RuntimeEvent, contract,
           action: str = "") -> Transition:
    """The pure transition. `contract` is the run's TaskContract (its
    `requires_verification` decides whether an unverified mutation may complete);
    `action` names the tool for an ACTION_STARTED event."""
    if event == E.ACTION_STARTED:
        if action in READ_ACTIONS:
            to = S.INVESTIGATE
        elif action == "plan":
            to = S.PLAN
        elif action in MUTATING_ACTIONS:
            to = S.MUTATE
        elif action in VERIFY_ACTIONS:
            to = S.VERIFY
        else:
            to = state
        return Transition(True, state, to, event, "action_started",
                          f"{action or 'action'} started")

    if event == E.AUTHORITY_DENIED:
        # a denied action is never permitted; route back to planning.
        return Transition(False, state, S.DIAGNOSE, event, "authority_denied",
                          "action denied by harness authority", recovery_state=S.PLAN)

    if event == E.COMPLETION_CLAIMED:
        if getattr(contract, "requires_verification", True) and state not in _COMPLETION_READY:
            # the defining invariant: a mutation (or an unresolved diagnose) cannot
            # jump straight to verified completion.
            reason = ("workspace changed and is not yet verified"
                      if state == S.MUTATE else f"cannot complete from {state.value}")
            return Transition(False, state, state, event, "unverified_completion",
                              f"completion rejected: {reason}", recovery_state=S.VERIFY)
        return Transition(True, state, S.COMPLETE, event, "completed",
                          "completion accepted")

    to, recovery = _OBSERVED.get(event, (None, None))
    return Transition(True, state, to if to is not None else state, event,
                      "transition", event.value, recovery_state=recovery)
