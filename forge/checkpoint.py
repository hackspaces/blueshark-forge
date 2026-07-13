"""Event-sourced checkpoints and crash recovery (H09).

A forge session is an append-only event log — the `.jsonl` transcript. This module
makes reading it crash-safe and resuming it correct:

  * `read_committed` keeps only fully-committed records. A process killed mid-append
    leaves a torn final line; it is QUARANTINED (not silently dropped) and the byte
    offset of the last valid record is reported, so a resume starts from the last
    committed state and a recovery tool can truncate the garbage precisely.
  * `recovery_state` replays the committed events through the H02 reducer to get the
    execution state to resume from — no guessing.
  * `needs_reconciliation` refuses to treat an action that MAY have executed (an
    INDETERMINATE lifecycle terminal from H03) as blindly retryable — its idempotency
    must be checked first.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

CHECKPOINT_VERSION = 1


@dataclass
class RecoveryReport:
    records: List[Dict[str, Any]] = field(default_factory=list)  # committed (valid) records, in order
    last_valid_offset: int = 0        # byte offset just past the last committed record
    quarantined: List[str] = field(default_factory=list)         # raw text of dropped corrupt lines
    corrupt_tail: bool = False        # the final line was torn — the signature of a crash

    @property
    def clean(self) -> bool:
        return not self.quarantined


def read_committed(path: str) -> RecoveryReport:
    """Read a transcript, keeping only complete, valid-JSON records. A trailing line
    that is not newline-terminated or does not parse is quarantined as a torn tail;
    any other unparseable line is quarantined too. The last valid byte offset lets a
    caller truncate exactly at the last committed record."""
    if not os.path.exists(path):
        return RecoveryReport()
    with open(path, "rb") as f:
        lines = f.readlines()

    report = RecoveryReport()
    offset = 0
    n = len(lines)
    for i, raw in enumerate(lines):
        end = offset + len(raw)
        is_last = (i == n - 1)
        text = raw.decode("utf-8", "replace")
        stripped = text.strip()
        if not stripped:
            offset = end
            if not is_last:              # a blank line inside the log is inert, not corruption
                report.last_valid_offset = end
            continue
        # a committed record is newline-terminated (the append finished) AND valid JSON.
        complete = raw.endswith(b"\n")
        try:
            rec = json.loads(stripped)
            if complete:
                report.records.append(rec)
                report.last_valid_offset = end
            else:
                report.quarantined.append(text)      # parsed but not newline-terminated → torn
                report.corrupt_tail = True
        except (json.JSONDecodeError, UnicodeDecodeError):
            report.quarantined.append(text)
            if is_last:
                report.corrupt_tail = True
        offset = end
    return report


def recovery_state(records: List[Dict[str, Any]]):
    """The execution state to resume from, reconstructed by replaying the committed
    events through the authoritative reducer (H02). Deterministic."""
    from .execution import ExecutionState, RuntimeEvent
    from . import reducer as _reducer
    from .contract import TaskContract

    meta = next((r for r in records if r.get("type") == "meta"), {})
    contract = TaskContract.from_dict(meta.get("contract"), fallback_mode=meta.get("mode", "auto"))
    state = ExecutionState.ORIENT
    active = ""
    for r in records:
        kind = r.get("type")
        if kind == "action":
            active = str(r.get("action", ""))
            state = _reducer.reduce(state, RuntimeEvent.ACTION_STARTED, contract, action=active).state_to
        elif kind == "observation":
            if not r.get("ok", True):
                state = _reducer.reduce(state, RuntimeEvent.VERIFICATION_FAILED, contract).state_to
            elif active in ("write_file", "edit_file"):
                state = _reducer.reduce(state, RuntimeEvent.WORKSPACE_CHANGED, contract).state_to
        elif kind == "assistant" and r.get("text") is not None and not r.get("stuck"):
            state = _reducer.reduce(state, RuntimeEvent.COMPLETION_CLAIMED, contract).state_to
    return state


def last_lifecycle(records: List[Dict[str, Any]]):
    """The most recent action_lifecycle record, or None."""
    for r in reversed(records):
        if r.get("type") == "action_lifecycle":
            return r
    return None


def _executed_action(record: Dict[str, Any]) -> bool:
    """An `action` record for an action that actually RAN — not a harness pre-execution
    rejection (read-before-edit, missing/invalid path, region gate), which never has an
    effect and never gets a lifecycle even in a clean run."""
    if record.get("type") != "action":
        return False
    args = record.get("args") or {}
    return not (args.get("blocked") or args.get("invalid"))


def reconciliation_action(records: List[Dict[str, Any]]):
    """The action_kind whose effect is UNKNOWN after a crash and must be reconciled
    before any retry — or None if the last committed state is safe to resume.

    Two crash signatures qualify:
      * the last lifecycle terminal is INDETERMINATE (e.g. a launched background process);
      * a DANGLING executed action — a committed `action` record with no `action_lifecycle`
        terminal after it, i.e. a hard kill (kill -9 / OOM / power loss) BETWEEN committing
        the action and finishing it. The mutation may or may not have landed.
    """
    lc = last_lifecycle(records)
    if lc is not None and lc.get("outcome") == "indeterminate":
        return lc.get("action_kind") or "an action"
    last_idx, last_action = -1, None
    for i, r in enumerate(records):
        if _executed_action(r):
            last_idx, last_action = i, r
    if last_action is None:
        return None
    # a terminal after the last executed action means that action completed cleanly
    if any(r.get("type") == "action_lifecycle" for r in records[last_idx + 1:]):
        return None
    return last_action.get("action") or "an action"


def needs_reconciliation(records: List[Dict[str, Any]]) -> bool:
    """True if the last recorded action MAY have executed but its result is unknown, so
    it must NOT be blindly retried on resume."""
    return reconciliation_action(records) is not None
