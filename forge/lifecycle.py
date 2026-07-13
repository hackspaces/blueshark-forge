"""Complete action lifecycle (H03).

Every attempted effect gets a stable identity — (run, turn, action, parent,
attempt) — and travels a lifecycle with EXACTLY ONE terminal outcome:

    requested → authorized → started → { succeeded | failed | denied
                                       | cancelled | timed_out | indeterminate }

A second terminal on the same action is rejected. A retry is a NEW action with a
fresh action id and an incremented attempt while retaining its causal parent, so
the provenance chain of a repeatedly-attempted effect stays intact.

Identities are deterministic monotonic counters (not uuids), so replaying a
recorded run reconstructs the same identities.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

LIFECYCLE_VERSION = 1


class Stage(str, Enum):
    # progress stages
    REQUESTED = "requested"
    AUTHORIZED = "authorized"
    STARTED = "started"
    # terminal outcomes
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    INDETERMINATE = "indeterminate"


TERMINAL = frozenset((Stage.SUCCEEDED, Stage.FAILED, Stage.DENIED,
                      Stage.CANCELLED, Stage.TIMED_OUT, Stage.INDETERMINATE))


class DuplicateTerminal(Exception):
    """A lifecycle that already reached a terminal cannot be terminated again."""


@dataclass
class ActionLifecycle:
    run_id: str
    turn_id: str
    action_id: str
    action_kind: str
    parent_action_id: Optional[str] = None
    attempt: int = 1
    stage: Stage = Stage.REQUESTED
    outcome: Optional[Stage] = None
    detail: str = ""
    timestamps: Dict[str, float] = field(default_factory=dict)
    v: int = LIFECYCLE_VERSION

    @property
    def terminal(self) -> bool:
        return self.outcome is not None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        d["outcome"] = self.outcome.value if self.outcome else None
        return d


class LifecycleTracker:
    """Mints action identities for one run and enforces the single-terminal rule.

    `clock` is injected (default a zero clock) so tests are deterministic; the agent
    passes `time.time`.
    """

    def __init__(self, run_id, clock=None):
        self.run_id = str(run_id)
        self._clock = clock or (lambda: 0.0)
        self._n = 0
        self.live: Dict[str, ActionLifecycle] = {}

    def request(self, turn_id, action_kind, parent_action_id=None, attempt=1) -> ActionLifecycle:
        self._n += 1
        lc = ActionLifecycle(self.run_id, str(turn_id), f"a{self._n}", str(action_kind),
                             parent_action_id, int(attempt), Stage.REQUESTED,
                             timestamps={"requested": self._clock()})
        self.live[lc.action_id] = lc
        return lc

    def _advance(self, lc: ActionLifecycle, stage: Stage) -> ActionLifecycle:
        if lc.terminal:
            raise DuplicateTerminal(f"{lc.action_id} already terminal ({lc.outcome.value})")
        lc.stage = stage
        lc.timestamps[stage.value] = self._clock()
        return lc

    def authorize(self, lc: ActionLifecycle) -> ActionLifecycle:
        return self._advance(lc, Stage.AUTHORIZED)

    def start(self, lc: ActionLifecycle) -> ActionLifecycle:
        return self._advance(lc, Stage.STARTED)

    def finish(self, lc: ActionLifecycle, outcome: Stage, detail: str = "") -> ActionLifecycle:
        if outcome not in TERMINAL:
            raise ValueError(f"{outcome} is not a terminal outcome")
        if lc.terminal:
            raise DuplicateTerminal(f"{lc.action_id} already terminal ({lc.outcome.value})")
        lc.stage = outcome
        lc.outcome = outcome
        lc.detail = detail or lc.detail
        lc.timestamps["terminal"] = self._clock()
        self.live.pop(lc.action_id, None)
        return lc

    def deny(self, lc: ActionLifecycle, reason: str = "") -> ActionLifecycle:
        return self.finish(lc, Stage.DENIED, reason)

    def retry_of(self, lc: ActionLifecycle, action_kind: str = None) -> ActionLifecycle:
        """A fresh lifecycle for a retry: new action id, attempt + 1, causal parent = lc."""
        return self.request(lc.turn_id, action_kind or lc.action_kind,
                            parent_action_id=lc.action_id, attempt=lc.attempt + 1)


def outcome_for(ok: bool, cancelled: bool = False, timed_out: bool = False,
                indeterminate: bool = False) -> Stage:
    """Map an execution result to its terminal outcome. Cancellation and timeout win
    over a bare ok/fail; a launched-but-unknown effect (e.g. a background process) is
    indeterminate."""
    if cancelled:
        return Stage.CANCELLED
    if timed_out:
        return Stage.TIMED_OUT
    if indeterminate:
        return Stage.INDETERMINATE
    return Stage.SUCCEEDED if ok else Stage.FAILED
