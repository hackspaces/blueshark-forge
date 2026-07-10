"""Turn one of the flywheel: forge sessions → harness-native training data.

Every session forge records is labeled data. The transcript holds the model's RAW output
at each step, whether it executed, and — uniquely — the moments the HARNESS had to correct
the model. This module turns a transcript into two training signals:

  SFT examples  — {"messages": <context>, "completion": <action>}: for each action that
                  executed successfully, the context and the action the model should emit.
                  Teaches the action protocol and the tool sequences that actually worked.

  Preference pairs — {"prompt": <context>, "chosen": <good>, "rejected": <bad>, "kind"}:
                  the CORRECTION moments — the highest-value signal, because they encode the
                  exact failure modes forge measures and repairs:
                    kind=grammar  a malformed strike, recovered to valid action JSON
                    kind=narrate  an "I'll do it…" preamble that got bounced, recovered to
                                  real work (the act-don't-narrate reward)
                  `chosen` > `rejected` is the reward signal (DPO/ORPO-ready).

Deterministic, stdlib-only. Reconstructs the conversation faithfully at the turn level
(user → action → observation → say); the trainer prepends a consistent system prompt, so
the per-session workspace briefing (not recoverable from a transcript) is intentionally
omitted. `forge corpus [sid|last|--all]` writes JSONL.
"""
import json


def _obs_msg(text):
    return {"role": "user", "content": "Observation:\n" + (text or "")}


def build(records, sid="", system=None):
    """Extract SFT examples + preference pairs from one session's transcript records.
    Returns {"sft": [...], "pref": [...]}. `system`, if given, is prepended as the
    system message of every example's context."""
    sft, pref = [], []
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    last_raw = None            # the raw output of the most recent `model` record (pending outcome)
    pending_reject = None      # (context_snapshot, rejected_raw, kind) awaiting the recovered action

    def snapshot():
        return [dict(m) for m in messages]

    for r in records:
        t = r.get("type")
        if t == "user":
            messages.append({"role": "user", "content": r.get("text", "")})
        elif t == "model":
            last_raw = r.get("raw", "")
        elif t == "malformed":
            # the current model output was invalid JSON — a rejected example; the recovered
            # (valid) action that follows becomes its `chosen`.
            pending_reject = (snapshot(), r.get("raw", "") or (last_raw or ""), "grammar")
        elif t == "narrate_bounce":
            # the model said an "I'll do it…" preamble and was bounced — rejected; the real
            # work action that follows is its `chosen`. Record the bounce as the harness reply.
            pending_reject = (snapshot(), last_raw or r.get("msg", ""), "narrate")
        elif t == "action":
            if last_raw is None:
                continue
            ctx = snapshot()                       # context BEFORE this action
            messages.append({"role": "assistant", "content": last_raw})
            # the observation record follows and carries ok; stash for it
            _pending_action = (ctx, last_raw)
            last_raw = ("__ACTION__", _pending_action)
        elif t == "observation":
            ok = r.get("ok")
            messages.append(_obs_msg(r.get("text", "")))
            if isinstance(last_raw, tuple) and last_raw and last_raw[0] == "__ACTION__":
                ctx, raw = last_raw[1]
                if ok:
                    sft.append({"messages": ctx, "completion": raw,
                                "meta": {"session": sid, "kind": "action"}})
                    if pending_reject:
                        rctx, rej, kind = pending_reject
                        pref.append({"prompt": rctx, "chosen": raw, "rejected": rej,
                                     "kind": kind, "meta": {"session": sid}})
                        pending_reject = None
            last_raw = None
        elif t == "assistant":                      # a `say` (ends the turn)
            if last_raw is not None and not isinstance(last_raw, tuple):
                sft.append({"messages": snapshot(), "completion": last_raw,
                            "meta": {"session": sid, "kind": "say"}})
                messages.append({"role": "assistant", "content": last_raw})
            last_raw = None
    return {"sft": sft, "pref": pref}


def build_jsonl(records, sid="", system=None):
    """Flat JSONL rows tagged by split — one dict per line, ready to write. SFT rows carry
    `split:"sft"`, preference rows `split:"pref"`."""
    b = build(records, sid=sid, system=system)
    rows = []
    for e in b["sft"]:
        rows.append({"split": "sft", **e})
    for e in b["pref"]:
        rows.append({"split": "pref", **e})
    return rows
