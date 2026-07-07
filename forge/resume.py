"""P4.7 — session resume from the transcript.

Turn a session's write-only JSONL transcript (~/.forge/sessions/<sid>.jsonl) back
into reconstructable working memory so `forge --resume <sid|last>` can continue a
crashed, Ctrl-C'd, or overnight-abandoned session instead of starting cold.

Almost every primitive already exists and is REUSED here:
  - P3.1 logs a `meta` header (model ladder, cwd, briefing hash, mode) as line 1.
  - P3.1 persists each compaction SUMMARY as a `compact` record.
  - P4.7 (this item) persists plan updates as `plan` records.
  - action / observation records carry the tool history; observations were logged
    at the SAME shaped budget the live model saw, so replaying them is exact. Action
    records only keep {command,path} (not the raw assistant JSON), so the assistant
    side of the tail is SYNTHESIZED lossily from (action, args, thought).

load(sid) folds the records into (summary_note, tail_msgs, plan, read_ts); apply()
splices them onto a freshly-built Agent (keeping its re-oriented workspace head) and
seeds the P4.1 read-ledger ONLY with paths still unchanged on disk — a file touched
since it was read stays unseeded so read-before-edit stays honest.

Pure stdlib, no model calls.
"""
import glob
import json
import os

from . import session as sessmod
from . import fleet

# How many trailing user/assistant/action/observation records to replay verbatim
# into the reconstructed message list. The last compact summary carries everything
# older; this is the recent, high-fidelity window.
TAIL_RECORDS = 30


# ---- transcript discovery ---------------------------------------------------
def _read_meta(path):
    """The `meta` header record for a transcript file (logged first at
    Agent.__init__), or None. Reads only the top of the file — meta is line 1."""
    try:
        with open(path, "r", errors="replace") as f:
            for _ in range(5):                      # meta is at/near the very top
                line = f.readline()
                if not line:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") == "meta":
                    return r
    except OSError:
        pass
    return None


def _live_sids():
    """The sids of sessions currently registered with a live pid — these must not
    be resumed (their transcript is still being written)."""
    return {e.get("sid") for e in sessmod.registry()}


def latest_sid(cwd):
    """The newest NON-LIVE transcript whose meta.cwd matches `cwd` — how
    `--resume last` chooses. Returns a sid, or None if there is nothing to resume
    for this directory."""
    cwd = os.path.abspath(cwd)
    live = _live_sids()
    files = glob.glob(os.path.join(sessmod.SESSIONS, "*.jsonl"))
    for path in sorted(files, key=os.path.getmtime, reverse=True):
        sid = os.path.basename(path)[:-len(".jsonl")]
        if sid in live:
            continue
        m = _read_meta(path)
        if m and os.path.abspath(m.get("cwd", "") or "") == cwd:
            return sid
    return None


def resolve_sid(spec, cwd):
    """Map the CLI argument to a concrete sid: 'last' → newest for cwd, anything
    else is taken as an explicit sid (prefix-matched against existing transcripts
    so a short id works). Returns a sid or None."""
    if spec and spec != "last":
        path = os.path.join(sessmod.SESSIONS, spec + ".jsonl")
        if os.path.exists(path):
            return spec
        # tolerate an unambiguous sid PREFIX (mirrors how the fleet addresses sessions)
        hits = [os.path.basename(p)[:-len(".jsonl")]
                for p in glob.glob(os.path.join(sessmod.SESSIONS, spec + "*.jsonl"))]
        if len(hits) == 1:
            return hits[0]
        return None
    return latest_sid(cwd)


def is_live(sid):
    """True if `sid` is a session currently running (registered with a live pid);
    resuming it would fork a second writer onto a live transcript."""
    return sid in _live_sids()


# ---- record → message synthesis --------------------------------------------
def _assistant_say(r):
    """Rebuild the assistant JSON for a persisted `say` (assistant record)."""
    obj = {}
    if r.get("thought"):
        obj["thought"] = r["thought"]
    obj["action"] = "say"
    obj["message"] = r.get("text", "")
    return json.dumps(obj)


def _assistant_action(r):
    """SYNTHESIZE the assistant JSON for a tool step from its action record. Lossy
    by design: action records keep only {command,path,target} in args (not old/new/
    pattern), so a replayed edit/grep loses those fields — accepted per the P4.7
    brief. The thought and the command/path — what a continuing model most needs —
    survive."""
    obj = {}
    if r.get("thought"):
        obj["thought"] = r["thought"]
    obj["action"] = r.get("action")
    args = r.get("args") or {}
    for k in ("command", "path", "target"):
        v = args.get(k)
        if isinstance(v, str) and v:
            obj[k] = v
    return json.dumps(obj)


def _observation_msg(r):
    """Rebuild the observation user-message exactly as send() fed it to the model:
    the shaped text after an 'Observation:\\n' header, with the failed-action tag
    prepended for a failing step."""
    text = r.get("text", "")
    tag = "" if r.get("ok", True) else "  ⚠ this action FAILED — diagnose the cause before retrying.\n"
    return {"role": "user", "content": f"{tag}Observation:\n{text}"}


def _tail_messages(recs, n=TAIL_RECORDS):
    """The last `n` conversational records rebuilt into messages, in order.
    user → user message; assistant(say) and action → assistant message; observation
    → user message. meta/step/model/compact/plan/verified/… are structural and skipped."""
    conv = [r for r in recs if r.get("type") in ("user", "assistant", "action", "observation")]
    msgs = []
    for r in conv[-n:]:
        t = r["type"]
        if t == "user":
            msgs.append({"role": "user", "content": r.get("text", "")})
        elif t == "assistant":
            msgs.append({"role": "assistant", "content": _assistant_say(r)})
        elif t == "action":
            msgs.append({"role": "assistant", "content": _assistant_action(r)})
        elif t == "observation":
            msgs.append(_observation_msg(r))
    return msgs


# ---- load + apply -----------------------------------------------------------
def load(sid):
    """Fold a transcript into reconstruction material, or None if it has no records.

    Reads the WHOLE file (no 300KB tail cap) so the last compaction summary — often
    near the top of a long session — is never lost. Returns a dict with:
      sid, meta, summary_note (a '[Earlier progress]' user message or None),
      plan (the last logged plan items), tail_msgs (replayed recent turns),
      read_ts ({relpath: latest read/write/edit timestamp} for ledger seeding).
    """
    recs = fleet._records(sid, tail_bytes=10 ** 9)     # whole file
    if not recs:
        return None
    meta = next((r for r in recs if r.get("type") == "meta"), None)

    summary = None
    plan = []
    read_ts = {}
    for r in recs:
        t = r.get("type")
        if t == "compact" and r.get("summary"):
            summary = r["summary"]                     # last one wins
        elif t == "plan" and isinstance(r.get("items"), list):
            plan = r["items"]                          # last one wins
        elif t == "action" and r.get("action") in ("read_file", "write_file", "edit_file"):
            p = (r.get("args") or {}).get("path")
            if isinstance(p, str) and p:
                read_ts[p] = max(read_ts.get(p, 0.0), r.get("ts", 0.0) or 0.0)

    summary_note = None
    if summary:
        summary_note = {"role": "user",
                        "content": "[Earlier progress, summarized to save context:]\n" + summary}

    return {"sid": sid, "meta": meta, "summary_note": summary_note,
            "plan": list(plan), "tail_msgs": _tail_messages(recs), "read_ts": read_ts}


def _seed_ledger(agent, read_ts):
    """Seed the P4.1 read-ledger from the recorded reads, but ONLY for files still
    unchanged on disk — the current-mtime staleness check. A file whose on-disk
    mtime is newer than when it was last read was modified since (by us or anything
    else), so it stays UNSEEDED and read-before-edit will force a fresh read. Returns
    the list of realpaths actually seeded."""
    cwd = getattr(agent.session, "cwd", None) or ""
    seeded = []
    for rel, ts in read_ts.items():
        fp = os.path.realpath(os.path.join(cwd, rel))
        if not os.path.isfile(fp):
            continue
        try:
            mtime = os.stat(fp).st_mtime
        except OSError:
            continue
        if ts and mtime <= ts:                         # unchanged since we read it → safe to seed
            if agent.ledger.record_read(fp, 0) is not None:
                seeded.append(fp)
    return seeded


def apply(agent, data):
    """Splice reconstructed memory onto a freshly-built Agent.

    The agent's head (system + re-oriented workspace briefing) is KEPT as-is — a
    resume must re-orient in the CURRENT repo, not replay a stale briefing. Onto it
    we append the earlier-progress summary and the recent-turn tail, restore the
    living plan, and seed the read-ledger for still-fresh files. Returns a short
    human note describing what was restored."""
    head = agent.messages[:agent.head_len]
    spliced = list(head)
    if data.get("summary_note"):
        spliced.append(data["summary_note"])
    tail = data.get("tail_msgs") or []
    spliced.extend(tail)
    agent.messages = spliced
    agent._sync_meta()                                 # P4.2: keep the parallel meta list aligned

    plan = data.get("plan") or []
    if plan:
        agent.plan = list(plan)

    seeded = _seed_ledger(agent, data.get("read_ts") or {})

    note = (f"resumed {data.get('sid', '?')[:8]} · {len(tail)} msg(s) replayed · "
            f"plan {len(plan)} item(s) · summary "
            f"{'restored' if data.get('summary_note') else 'none'} · "
            f"{len(seeded)} file(s) still in context")
    agent._resume_info = note
    return note
