"""XES/OCEL-compatible event export (H13).

Turns a forge session's transcript into process-mining data without log-scraping.
The H03 action-lifecycle records already carry stable identities — run, turn, action,
parent, attempt, activity, timestamps, outcome — so this is a deterministic projection
onto a normalized event schema, exportable as dependency-free CSV/JSON or standards-
shaped OCEL 2.0. Secrets are deterministically redacted before anything leaves.

No third-party process-mining library — stdlib only.
"""
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter

EXPORT_SCHEMA_VERSION = 1

# Deterministic secret redaction. Each match becomes a stable <redacted:sha8> token,
# so the same secret always maps to the same placeholder (useful for correlation) while
# the raw value never leaves. Layered: armored keys and structured shapes first, then a
# high-entropy fallback for bare tokens that carry no keyword context.
_STANDALONE = [
    # any armored private key (RSA/EC/OPENSSH/PGP/…), tolerant of the armor label
    re.compile(r"-----BEGIN[^\n-]*PRIVATE KEY[^\n-]*-----[\s\S]+?-----END[^\n-]*-----"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{4,}"),   # JWT
    re.compile(r"\b(sk|rk)-[A-Za-z0-9]{20,}\b"),                 # OpenAI-style keys
    re.compile(r"\bgh[posru]_[A-Za-z0-9]{30,}\b"),             # GitHub tokens
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),           # Slack tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                        # AWS access key id
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*"),    # bearer tokens
]
# scheme://user:password@host — mask ONLY the password segment
_URL_CRED = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:@/]+):([^\s:@/]+)@")
# KEY=value / "token": "value with spaces" — keep the key, mask a quoted OR unquoted value
_KV = re.compile(r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIAL|AUTH)[A-Z0-9_]*)"
                 r"(\s*[=:]\s*)(?:(['\"])(.+?)\3|([^\s'\"]{3,}))")
# bare high-entropy runs (context-free tokens: AWS secret keys, hex/base64 blobs)
_BLOB = re.compile(r"\b[A-Za-z0-9+/_-]{32,}={0,2}\b")


def _mask(value: str) -> str:
    return "<redacted:" + hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:8] + ">"


def _entropy(s: str) -> float:
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values()) if n else 0.0


def _looks_secret(s: str) -> bool:
    # a long, mixed-class, high-entropy run is a token, not a word or a path.
    return (any(c.isalpha() for c in s) and any(c.isdigit() for c in s)
            and _entropy(s) >= 4.0)


def _kv_sub(m):
    key, delim, q, qval, uval = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    if qval is not None:
        return f"{key}{delim}{q}{_mask(qval)}{q}"
    return f"{key}{delim}{_mask(uval)}"


def redact(text):
    """Return `text` with any secret-shaped substring replaced by a stable placeholder.
    Non-strings pass through unchanged."""
    if not isinstance(text, str) or not text:
        return text
    out = _URL_CRED.sub(lambda m: m.group(1) + ":" + _mask(m.group(2)) + "@", text)
    out = _KV.sub(_kv_sub, out)
    for pat in _STANDALONE:
        out = pat.sub(lambda m: _mask(m.group(0)), out)
    out = _BLOB.sub(lambda m: _mask(m.group(0)) if _looks_secret(m.group(0)) else m.group(0), out)
    return out


def _csv_safe(value) -> str:
    """Neutralize CSV formula-injection: a cell that a spreadsheet would evaluate
    (leading = + - @ or control chars) is prefixed with a single quote."""
    s = "" if value is None else str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


# The stable, versioned event schema. Every exporter emits exactly these columns.
FIELDS = ["schema_version", "case_id", "turn_id", "action_id", "parent_action_id",
          "attempt", "activity", "lifecycle", "outcome", "timestamp", "resource",
          "task", "detail"]


def to_events(records):
    """Project a transcript's records onto normalized process-mining events — one per
    action lifecycle. Session context (model, task) comes from the meta header."""
    meta = next((r for r in records if r.get("type") == "meta"), {}) or {}
    contract = meta.get("contract") or {}
    resource = meta.get("model", "")
    task = contract.get("goal", "")
    events = []
    for r in records:
        if r.get("type") != "action_lifecycle":
            continue
        ts = r.get("timestamps") or {}
        events.append({
            "schema_version": EXPORT_SCHEMA_VERSION,
            "case_id": r.get("run_id", ""),
            "turn_id": r.get("turn_id", ""),
            "action_id": r.get("action_id", ""),
            "parent_action_id": r.get("parent_action_id") or "",
            "attempt": r.get("attempt", 1),
            "activity": r.get("action_kind", ""),
            "lifecycle": r.get("stage", ""),
            "outcome": r.get("outcome") or "",
            "timestamp": ts.get("terminal") or ts.get("requested") or r.get("ts", ""),
            "resource": resource,
            "task": redact(task),
            "detail": redact(r.get("detail", "")),
        })
    return events


def to_csv(events) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for e in events:
        w.writerow({k: _csv_safe(e.get(k)) for k in FIELDS})   # neutralize formula injection
    return buf.getvalue()


def to_json(events) -> str:
    return json.dumps({"schema_version": EXPORT_SCHEMA_VERSION, "events": events},
                      indent=2, sort_keys=True)


def to_ocel(events) -> str:
    """OCEL 2.0-shaped object-centric log: `run` and `action` objects, one event per
    lifecycle referencing both — so parallel actions keep their identity and causality."""
    runs, actions, activities = {}, {}, set()
    ocel_events = []
    for i, e in enumerate(events):
        runs.setdefault(e["case_id"], {"id": e["case_id"], "type": "run", "attributes": []})
        actions.setdefault(e["action_id"], {
            "id": e["action_id"], "type": "action",
            "attributes": [{"name": "activity", "value": e["activity"]},
                           {"name": "attempt", "value": e["attempt"]},
                           {"name": "parent_action_id", "value": e["parent_action_id"]}]})
        activities.add(e["activity"])
        ocel_events.append({
            "id": f"e{i}", "type": e["activity"], "time": e["timestamp"],
            "attributes": [{"name": "lifecycle", "value": e["lifecycle"]},
                           {"name": "outcome", "value": e["outcome"]},
                           {"name": "turn_id", "value": e["turn_id"]},
                           {"name": "detail", "value": e["detail"]}],
            "relationships": [{"objectId": e["case_id"], "qualifier": "run"},
                              {"objectId": e["action_id"], "qualifier": "action"}],
        })
    doc = {
        "ocel:version": "2.0",
        "schema_version": EXPORT_SCHEMA_VERSION,
        "objectTypes": [{"name": "run", "attributes": []}, {"name": "action", "attributes": []}],
        "eventTypes": [{"name": a, "attributes": []} for a in sorted(activities)],
        "objects": list(runs.values()) + list(actions.values()),
        "events": ocel_events,
    }
    return json.dumps(doc, indent=2, sort_keys=True)


FORMATS = {"csv": to_csv, "json": to_json, "ocel": to_ocel}


def export(records, fmt: str = "json") -> str:
    """Records → the chosen format's text. Unknown format falls back to json."""
    return FORMATS.get(fmt, to_json)(to_events(records))
