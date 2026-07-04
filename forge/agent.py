"""The agent loop — the harness brain.

Frontier-quality scaffolding so any model, even a small one, works well:

  CONSTRAINED DECODING  every output is grammar-forced to a valid action.
  LIVING PLAN           the agent maintains a todo list the harness pins into
                        context every turn, so long-horizon work stays coherent.
  SELF-CORRECTION       failed actions are flagged so the model diagnoses instead
                        of blindly retrying; repeated no-progress loops are broken.
  CONTEXT COMPACTION    old tool output is summarized so long sessions don't blow
                        the window.
  VERIFY-BEFORE-DONE    the agent is pushed to actually check its work before `say`.
"""
import json
import os
import re
import threading

from .tools import ACTION_SCHEMA, TOOL_HELP, execute

STUCK_AT = int(os.environ.get("FORGE_STUCK_THRESHOLD", "7"))  # failures before escalating a rung

_MSG_OPEN = re.compile(r'"message"\s*:\s*"')


def _partial_message(raw):
    """Return the current (possibly incomplete) value of the JSON `message` field,
    unescaped, as it streams. Used to type the reply out live."""
    m = _MSG_OPEN.search(raw)
    if not m:
        return None
    i, out = m.end(), []
    esc = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/", "r": "\r"}
    while i < len(raw):
        c = raw[i]
        if c == "\\":
            if i + 1 >= len(raw):
                break
            out.append(esc.get(raw[i + 1], raw[i + 1]))
            i += 2
            continue
        if c == '"':
            break
        out.append(c)
        i += 1
    return "".join(out)

SYSTEM = f"""You are Forge, a sharp, autonomous coding and shell agent working in a terminal on the user's machine. You get real work done with tools, one concrete step at a time.

{TOOL_HELP}

How you work:
- Keep a `plan` for multi-step work: break the request into a short todo list and update item states ([ ]/[~]/[x]) as you go. Think before the first action.
- Inspect before you change: read/list/bash to understand, then edit_file for surgical changes (prefer it over rewriting whole files).
- Verify with reality: run tests/commands to confirm things actually work. Never claim success you have not checked.
- Large repos: NEVER scan everything with `find . -exec` — it is pathologically slow with node_modules present. For repo-wide operations use `git ls-files` (lists exactly the real project files, node_modules excluded) piped to your command, e.g. `git ls-files | xargs wc -l | sort -rn | head`. Use `rg` for content search. The file tree in your workspace briefing is already the real files.

When you `say`: answer the user's question fully and clearly in natural prose. Be concise, but never clipped or truncated — give the actual information, finish your lists and sentences, and don't trail off with "...". A one-word answer to a real question is not enough. Only stop the turn to `say` when you have genuinely finished the work or need the user's input."""

AUTONOMOUS = """

BE AUTONOMOUS — this is the core of how you work. When the user asks for something, DO it end to end: make the reasonable choice yourself (pick the file, read it, make the change), use your tools, verify the result, and report what you actually did. Do NOT ask for permission or confirmation to take normal steps. Do NOT stop just to narrate what you are about to do — do it, then tell them the outcome. If the user says "any/you pick/you decide", that means choose and proceed immediately. Only come back to the user before finishing when you hit a genuine blocker you cannot resolve yourself, a real ambiguity where guessing would waste real work, or an action that is destructive or irreversible. A request like "read a file and add a comment" should end with the comment added and verified, not with a question."""

# Rough char budget before we compact old observations (keeps small-context models alive)
_COMPACT_AT = 24000


class Agent:
    def __init__(self, backend, session, max_steps=60, on_event=None, autonomous=False,
                 system=None, allowed=None, workspace=None):
        # `backend` may be a single backend or a LADDER (list, cheapest→strongest
        # local models). The harness starts cheap and escalates a rung when stuck.
        self.ladder = backend if isinstance(backend, list) else [backend]
        self.tier = 0
        self.backend = self.ladder[0]
        self.session = session
        self.max_steps = max_steps
        self.on_event = on_event or (lambda *a, **k: None)
        self.allowed = allowed
        base = (system if system is not None else SYSTEM) + (AUTONOMOUS if autonomous else "")
        self.messages = [{"role": "system", "content": base}]
        if workspace:
            self.messages.append({"role": "user", "content": workspace})
            self.messages.append({"role": "assistant", "content": '{"thought":"Oriented in the workspace. Ready.","action":"say","message":"Ready."}'})
        self.plan = []
        self.stop = threading.Event()  # set from the UI (Esc) to interrupt mid-run

    def set_ladder(self, ladder):
        """Swap the model ladder live (conversation preserved)."""
        self.ladder = ladder
        self.tier = 0
        self.backend = ladder[0]

    # ---- context hygiene ----
    def _compact(self):
        """Summarize old observation turns so the window stays lean."""
        size = sum(len(m["content"]) for m in self.messages)
        if size < _COMPACT_AT:
            return
        # keep system + last 8 turns verbatim; collapse the middle
        head, tail = self.messages[:1], self.messages[-8:]
        middle = self.messages[1:-8]
        if not middle:
            return
        note = f"[{len(middle)} earlier steps compacted. Current plan is pinned below.]"
        self.messages = head + [{"role": "user", "content": note}] + tail
        self.on_event("compact", n=len(middle))

    def _pin_plan(self):
        if self.plan:
            return {"role": "user", "content": "[current plan]\n" + "\n".join(self.plan)}
        return None

    def _generate(self, prompt):
        """Stream the model's action. When it turns out to be a `say`, emit the
        message text live (token by token) via on_event('token')."""
        if not hasattr(self.backend, "stream"):
            return self.backend.chat(prompt, schema=ACTION_SCHEMA)
        raw, emitted, is_say = "", 0, False
        try:
            for chunk in self.backend.stream(prompt, schema=ACTION_SCHEMA):
                if self.stop.is_set():
                    break
                raw += chunk
                if not is_say and '"say"' in raw:
                    is_say = True
                if is_say:
                    msg = _partial_message(raw)
                    if msg is not None and len(msg) > emitted:
                        self.on_event("token", text=msg[emitted:])
                        emitted = len(msg)
        except Exception:
            if not raw:
                raise
        return raw

    def _absorb_inbox(self):
        for m in self.session.drain():
            self.messages.append({"role": "user", "content": f"[fleet message from {m['from']}]: {m['text']}"})
            self.session.log("inbox", sender=m["from"], text=m["text"])
            self.on_event("inbox", sender=m["from"], text=m["text"])

    def send(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        self.session.log("user", text=user_text)
        self.session.set_status("working")
        bad = 0
        recent = []
        fail_counts = {}
        total_fails = 0
        try:
            for step in range(1, self.max_steps + 1):
                if self.stop.is_set():
                    self.on_event("stopped")
                    return "(stopped)"
                self._absorb_inbox()
                self._compact()
                pin = self._pin_plan()
                prompt = self.messages + ([pin] if pin else [])
                self.on_event("thinking")
                raw = self._generate(prompt)

                try:
                    act = json.loads(raw)
                except json.JSONDecodeError:
                    bad += 1
                    self.on_event("malformed")
                    self.messages.append({"role": "user", "content": "That was not valid action JSON. Reply with one JSON action object only."})
                    if bad >= 5:
                        return "(the model could not hold the action format)"
                    continue
                bad = 0
                self.messages.append({"role": "assistant", "content": raw})

                # plan update
                if isinstance(act.get("plan"), list) and act["plan"]:
                    if act["plan"] != self.plan:
                        self.plan = act["plan"]
                        self.on_event("plan", plan=self.plan)

                kind = act.get("action")
                if kind == "say":
                    msg = act.get("message", "")
                    self.session.log("assistant", text=msg, thought=act.get("thought", ""))
                    self.on_event("say", message=msg)
                    return msg

                if self.allowed is not None and kind not in self.allowed:
                    self.messages.append({"role": "user", "content": f"'{kind}' not permitted. Allowed: {sorted(self.allowed)}."})
                    continue

                sig = f"{kind}:{act.get('command') or act.get('path') or ''}"
                recent.append(sig)
                if recent[-3:].count(sig) >= 3:
                    self.on_event("loop")
                    self.messages.append({"role": "user", "content": "You repeated the same action 3x with no progress. Do something different, or `say` if the task is already done."})
                    recent.clear()
                    continue

                self.session.log("action", action=kind, args={k: act.get(k) for k in ("command", "path") if act.get(k)}, thought=act.get("thought", ""))
                self.on_event("action", action=kind, thought=act.get("thought", ""),
                              detail=act.get("command") or act.get("path") or "")
                obs, ok = execute(act, self.session.cwd)
                self.session.log("observation", text=obs[:4000], ok=ok)
                self.on_event("observation", text=obs, ok=ok)

                tag = ""
                if not ok:
                    fail_counts[sig] = fail_counts.get(sig, 0) + 1
                    total_fails += 1
                    tag = "  ⚠ this action FAILED — diagnose the cause before retrying.\n"
                    # per-command repeat (survives interleaved successes, unlike a consecutive counter)
                    if fail_counts[sig] >= 3:
                        self.on_event("loop")
                        tag = (f"  ⚠ `{sig}` has now failed {fail_counts[sig]} times. STOP retrying this exact thing. "
                               "Change approach entirely: re-read the real file/error, rewrite with write_file instead of edit_file, "
                               "or `say` to tell the user you're stuck and exactly what failed.\n")
                    # stuck: escalate to a stronger LOCAL model (same task, same
                    # context) rather than grinding or giving up. All still local.
                    if total_fails >= STUCK_AT:
                        if self.tier < len(self.ladder) - 1:
                            self.tier += 1
                            self.backend = self.ladder[self.tier]
                            self.on_event("escalate", model=self.backend.name)
                            self.session.log("escalate", model=self.backend.name)
                            if hasattr(self.backend, "warm"):
                                self.backend.warm()
                            self.messages.append({"role": "user", "content":
                                f"[The previous model kept failing. You are now a stronger model taking over the SAME task with full context above. Step back, re-diagnose from the real errors, and solve it. Last error: {obs[:200].strip()}]"})
                            fail_counts.clear()
                            total_fails = 0
                            continue
                        stuck = (f"I'm stuck even after escalating through the local models. Last error: {obs[:200].strip()}. "
                                 "This needs a different approach — want me to try one, or take it yourself?")
                        self.session.log("assistant", text=stuck)
                        self.on_event("say", message=stuck)
                        return stuck
                self.messages.append({"role": "user", "content": f"{tag}Observation:\n{obs[:4000]}"})

            return "(hit the step limit — ask me to continue)"
        finally:
            self.session.set_status("idle")
