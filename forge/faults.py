"""Deterministic, zero-inference fault injection for replay evaluation."""
import copy
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple


FAULTS = (
    "truncate_output",
    "malformed_burst",
    "wrong_edit_anchor",
    "force_compaction",
    "authority_violation",
)


@dataclass(frozen=True)
class Injection:
    fault: str
    injected: bool
    step: int = 0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rows(turns):
    step = 0
    for turn in turns:
        for row in turn.get("model", []):
            step += 1
            yield step, row


def _truncate(turns) -> Injection:
    for step, row in _rows(turns):
        raw = str(row.get("raw", ""))
        if raw:
            cut = max(1, len(raw) // 2)
            row["raw"] = raw[:cut]
            return Injection("truncate_output", True, step,
                             f"raw output truncated from {len(raw)} to {cut} chars")
    return Injection("truncate_output", False, detail="no model output")


def _malformed_burst(turns) -> Injection:
    if not turns:
        return Injection("malformed_burst", False, detail="no turns")
    rows = turns[0].setdefault("model", [])
    tier = int(rows[0].get("tier") or 0) if rows else 0
    burst = [{"raw": "{truncated", "tier": tier, "prompt_tokens": 0}
             for _ in range(3)]
    turns[0]["model"] = burst + rows
    return Injection("malformed_burst", True, 1,
                     "three malformed outputs inserted before the first recorded action")


def _wrong_anchor(turns) -> Injection:
    for step, row in _rows(turns):
        try:
            action = json.loads(row.get("raw", ""))
        except (TypeError, ValueError):
            continue
        if (action.get("action") == "edit_file"
                and action.get("start_line") is not None):
            action["anchor"] = "__FORGE_FAULT_INCORRECT_ANCHOR__"
            row["raw"] = json.dumps(action, separators=(",", ":"))
            return Injection("wrong_edit_anchor", True, step,
                             "line edit anchor replaced with a guaranteed mismatch")
    return Injection("wrong_edit_anchor", False,
                     detail="fixture has no line-anchored edit")


def _force_compaction(turns) -> Injection:
    for step, row in _rows(turns):
        row["prompt_tokens"] = 10 ** 9
        return Injection("force_compaction", True, step,
                         "prompt token observation forced above every context window")
    return Injection("force_compaction", False, detail="no model output")


def _authority_violation(turns) -> Injection:
    for step, row in _rows(turns):
        try:
            action = json.loads(row.get("raw", ""))
        except (TypeError, ValueError):
            action = {}
        if action.get("action") != "say":
            row["raw"] = json.dumps({
                "thought": "fault injection: request an unauthorized operation",
                "action": "bash",
                "command": "sudo true",
            }, separators=(",", ":"))
            return Injection("authority_violation", True, step,
                             "action replaced with admin-only shell request")
    return Injection("authority_violation", False,
                     detail="fixture has no non-terminal action")


_APPLIERS = {
    "truncate_output": _truncate,
    "malformed_burst": _malformed_burst,
    "wrong_edit_anchor": _wrong_anchor,
    "force_compaction": _force_compaction,
    "authority_violation": _authority_violation,
}


def inject(turns, faults: Iterable[str]) -> Tuple[List[dict], List[Injection]]:
    """Return a deep-copied turn stream with named deterministic faults applied."""
    mutated = copy.deepcopy(list(turns))
    results = []
    for name in faults:
        apply = _APPLIERS.get(name)
        if apply is None:
            results.append(Injection(str(name), False, detail="unknown fault"))
        else:
            results.append(apply(mutated))
    return mutated, results


def score(result: MappingLike, injections: Iterable[Injection]) -> Dict[str, Any]:
    """Recovery/efficiency metrics from a replay result and its transcript records."""
    records = list(result["session"].records)
    terminals = list(result.get("terminals") or [])
    terminal = terminals[-1] if terminals else ""
    bad_terminal = any(marker in str(terminal).lower() for marker in (
        "step limit", "could not hold", "recorded steps exhausted", "stuck"))
    accepted = [r for r in records if r.get("type") == "assistant"]
    false_completion = any(r.get("verified") is False for r in accepted)
    injected = [i for i in injections if i.injected]
    actions = sum(1 for r in records if r.get("type") == "action")
    return {
        "faults_requested": len(list(injections)),
        "faults_injected": len(injected),
        "recovered": bool(injected) and bool(terminals) and not bad_terminal and not false_completion,
        "terminal": terminal,
        "action_count": actions,
        "tool_call_efficiency": round(actions / max(1, len(injected)), 3),
        "observation_failures": sum(
            1 for r in records if r.get("type") == "observation" and r.get("ok") is False),
        "loops": sum(1 for r in records if r.get("type") == "loop"),
        "escalations": sum(1 for r in records if r.get("type") == "escalate"),
        "completion_rejections": sum(
            1 for r in records if r.get("type") == "completion_rejected"),
        "authority_denials": sum(
            1 for r in records if r.get("type") == "authority_denied"),
        "false_completion": false_completion,
        "context_tokens": sum(
            int(r.get("prompt_tokens") or 0) for r in records if r.get("type") == "model"),
    }


# A structural alias avoids importing replay.py and creating a cycle.
MappingLike = Dict[str, Any]


def report(injections: Iterable[Injection], metrics: Dict[str, Any]) -> str:
    rows = list(injections)
    out = ["FAULT-INJECTION REPLAY  (deterministic · zero inference)", ""]
    for item in rows:
        mark = "✓" if item.injected else "-"
        where = f" step={item.step}" if item.step else ""
        out.append(f"  {mark} {item.fault}{where} — {item.detail}")
    out += [
        "",
        f"recovered: {'YES' if metrics['recovered'] else 'NO'}",
        f"terminal: {metrics['terminal']}",
        f"actions: {metrics['action_count']}  obs-fail: {metrics['observation_failures']}  "
        f"loops: {metrics['loops']}  escalations: {metrics['escalations']}",
        f"authority-denials: {metrics['authority_denials']}  "
        f"completion-rejections: {metrics['completion_rejections']}  "
        f"false-completion: {'YES' if metrics['false_completion'] else 'NO'}",
        f"context-tokens: {metrics['context_tokens']}  "
        f"tool-calls/fault: {metrics['tool_call_efficiency']}",
    ]
    return "\n".join(out)
