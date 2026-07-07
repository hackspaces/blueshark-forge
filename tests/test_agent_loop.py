"""P3.4 — Agent-loop invariant battery.

Direct, offline ScriptBackend/FakeBackend tests for the five flagship harness
levers that had ZERO coverage before this file: the escalation ladder, 3x loop
detection, context compaction (head/tail pinning + summarizer-failure fallback),
plan pin/update semantics, and mid-run inbox absorption. These are the product
thesis — any refactor of Agent.send() can silently break them with the rest of
the suite green — so each test is written to FAIL if its mechanism is disabled
(e.g. tier must actually flip, `recent` must actually clear, the pin must be
transient, an empty plan must NOT wipe a live one).

Stdlib unittest, no deps, no network, no model calls, no real daemon.
"""
import os
import tempfile
import unittest
import unittest.mock as mock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import agent as agent_mod          # noqa: E402
from forge import session as sm               # noqa: E402
from forge.agent import Agent                 # noqa: E402


def _write(p, s):
    with open(p, "w") as f:
        f.write(s)


def _bash(cmd):
    return '{"thought":"t","action":"bash","command":"%s"}' % cmd


_READ_R = '{"thought":"r","action":"read_file","path":"r.py"}'
_SAY = '{"thought":"d","action":"say","message":"done"}'


class _ScriptBackend:
    """Yields a scripted sequence of raw action JSONs (clamps to the last when
    exhausted). chat() is only ever the summarizer here — returns a canned say."""
    name = "script"

    def __init__(self, actions, name=None):
        self.actions = list(actions)
        self.i = 0
        if name:
            self.name = name

    def stream(self, messages, schema=None, temperature=0.0):
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        yield act

    def chat(self, messages, schema=None, temperature=0.0):
        return '{"thought":"x","action":"say","message":"done"}'


class _FakeBackend:
    """Records the message lists it is asked to generate from (for pin/inbox
    assertions), exposes effective_ctx/last_prompt_tokens so Agent._fill can drive
    compaction deterministically, and can make the summarizer (chat) raise so the
    fallback branch is exercised."""
    name = "fake"

    def __init__(self, actions, ctx=200000, prompt_tokens=0, summary="SUMMARY", raise_chat=False):
        self.actions = list(actions)
        self.i = 0
        self._ctx = ctx
        self.last_prompt_tokens = prompt_tokens
        self._summary = summary
        self._raise_chat = raise_chat
        self.received = []

    def effective_ctx(self):
        return self._ctx

    def stream(self, messages, schema=None, temperature=0.0):
        self.received.append(list(messages))
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        yield act

    def chat(self, messages, schema=None, temperature=0.0):
        if self._raise_chat:
            raise RuntimeError("summarizer offline")
        return self._summary


class _InboxSession:
    """A stub session with a REAL push/drain (unlike EphemeralSession, whose push
    is absent and drain always returns []). Records log() so nothing hits disk."""

    def __init__(self, cwd, sid="inbox"):
        self.cwd, self.sid, self.name = cwd, sid, "inbox"
        self.status = "idle"
        self.logs = []
        self._inbox = []

    def log(self, kind, **fields):
        self.logs.append((kind, fields))

    def push(self, sender, text):
        self._inbox.append({"from": sender, "text": text})

    def drain(self):
        msgs, self._inbox = self._inbox, []
        return msgs

    def set_status(self, s):
        self.status = s


class _PushingBackend:
    """Simulates a fleet/user message ARRIVING mid-run: on its first generation it
    pushes into the session (so the message is not yet in context), then serves the
    scripted actions. Records the message lists it receives so a test can prove the
    pushed message was absorbed BEFORE the next generation, not after."""
    name = "pusher"

    def __init__(self, session, actions, push_from="daemon", push_text="hold off"):
        self.session = session
        self.actions = list(actions)
        self.i = 0
        self._pushed = False
        self.received = []

    def stream(self, messages, schema=None, temperature=0.0):
        self.received.append(list(messages))
        if not self._pushed:
            self.session.push(self._pf, self._pt)
            self._pushed = True
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        yield act

    def chat(self, messages, schema=None, temperature=0.0):
        return '{"thought":"x","action":"say","message":"done"}'


# stash the push args on the instance (kept off __init__ signature noise)
def _pushing(session, actions, push_from="daemon", push_text="hold off"):
    b = _PushingBackend(session, actions)
    b._pf, b._pt = push_from, push_text
    return b


class TestEscalationLadder(unittest.TestCase):
    """The stuck-escalation lever (agent.py send(): total_fails >= STUCK_AT)."""

    def test_escalation_flips_tier_lands_takeover_and_resets_counters(self):
        # 2-rung ladder, STUCK_AT patched to 3 (it is read from env at import time).
        # rung0 emits 3 DISTINCT failing bash actions (distinct sigs so the loop
        # detector never fires and each fail counts) -> the harness must promote to
        # tier 1. Then rung1 fails 2 MORE times before saying done: if the fail
        # counters were NOT reset on escalation, the very first rung1 fail would be
        # total_fails==4>=3 with the ladder maxed and bail with the stuck `say`
        # instead of reaching "done" — so result=="done" proves the reset.
        d = tempfile.mkdtemp()
        rung0 = _ScriptBackend([_bash("false #a1"), _bash("false #a2"), _bash("false #a3")], name="weak")
        rung1 = _ScriptBackend([_bash("false #b1"), _bash("false #b2"), _SAY], name="strong")
        events = []
        a = Agent([rung0, rung1], sm.EphemeralSession(d, "esc"), max_steps=20,
                  on_event=lambda k, **kw: events.append((k, kw)))
        with mock.patch.object(agent_mod, "STUCK_AT", 3):
            result = a.send("do it")
        self.assertEqual(a.tier, 1)                               # promoted a rung
        self.assertEqual(result, "done")                         # counters reset -> rung1 got 2 fresh fails
        escs = [kw for k, kw in events if k == "escalate"]
        self.assertEqual(len(escs), 1)                           # escalated exactly once
        self.assertEqual(escs[0].get("model"), "strong")         # to the stronger rung
        self.assertTrue(any("stronger model taking over" in m.get("content", "") for m in a.messages),
                        "the takeover user-message must be injected with full context")

    def test_single_rung_ladder_returns_the_stuck_say(self):
        # a lone backend has no stronger rung: after STUCK_AT fails the loop must
        # NOT escalate (tier stays 0) and must return the honest "stuck" say.
        d = tempfile.mkdtemp()
        b = _ScriptBackend([_bash("false #1"), _bash("false #2"), _bash("false #3")])
        events = []
        a = Agent(b, sm.EphemeralSession(d, "stuck"), max_steps=20,
                  on_event=lambda k, **kw: events.append((k, kw)))
        with mock.patch.object(agent_mod, "STUCK_AT", 3):
            result = a.send("do it")
        self.assertEqual(a.tier, 0)                              # never escalated
        self.assertFalse(any(k == "escalate" for k, kw in events))
        self.assertIn("stuck", result.lower())                  # bailed with the stuck say
        self.assertTrue(any(k == "say" for k, kw in events))


class TestLoopDetection(unittest.TestCase):
    """The 3x-repeat loop breaker (agent.py send(): recent[-3:].count(sig) >= 3)."""

    def test_repeat_action_trips_nudge_and_clears_recent(self):
        # SIX identical successful reads then a say. If `recent` is cleared on each
        # trip, the loop fires exactly TWICE (steps 3 and 6). If it were NOT cleared,
        # it would fire on steps 3,4,5,6 (four times) — so ==2 pins the clear.
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        events = []
        a = Agent(_ScriptBackend([_READ_R] * 6 + [_SAY]), sm.EphemeralSession(d, "loop"),
                  max_steps=12, on_event=lambda k, **kw: events.append((k, kw)))
        result = a.send("read it")
        loops = [1 for k, kw in events if k == "loop"]
        self.assertEqual(sum(loops), 2)                          # cleared after each trip
        nudges = [m for m in a.messages
                  if "repeated the same action 3x" in m.get("content", "")]
        self.assertEqual(len(nudges), 2)                         # the nudge was appended each trip
        self.assertEqual(result, "done")                         # the say still lands afterwards


class TestCompaction(unittest.TestCase):
    """Context compaction (agent.py _compact/_summarize): head pinned, recent tail
    kept, older middle replaced by a summary note."""

    def _seed(self, a, n=20):
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            a.messages.append({"role": role, "content": "<<seed %d>>" % i})

    def test_compaction_pins_head_and_tail_and_inserts_summary(self):
        d = tempfile.mkdtemp()
        # effective_ctx=1000, last_prompt_tokens=900 -> 900 >= 0.70*1000 -> compact.
        be = _FakeBackend([_SAY], ctx=1000, prompt_tokens=900, summary="STATE rebuilt from middle")
        events = []
        a = Agent(be, sm.EphemeralSession(d, "comp"), max_steps=4,
                  on_event=lambda k, **kw: events.append((k, kw)))
        orig_head = a.messages[0]                                # the system message
        self.assertEqual(a.head_len, 1)
        self._seed(a, 20)
        result = a.send("go")

        self.assertIs(a.messages[0], orig_head)                 # head (system) untouched
        note = a.messages[1]                                    # summary sits right after the head
        self.assertTrue(note["content"].startswith("[Earlier progress, summarized"))
        self.assertIn("STATE rebuilt from middle", note["content"])
        self.assertTrue(any(k == "compact" for k, kw in events))
        # older middle turns are gone, the recent tail (last ~12) survives
        contents = [m.get("content", "") for m in a.messages]
        self.assertFalse(any("<<seed 0>>" in c for c in contents))   # oldest middle: summarized away
        self.assertFalse(any("<<seed 8>>" in c for c in contents))   # still middle: gone
        self.assertTrue(any("<<seed 9>>" in c for c in contents))    # tail boundary: kept
        self.assertTrue(any("<<seed 19>>" in c for c in contents))   # newest: kept
        self.assertEqual(result, "done")

    def test_summarizer_failure_falls_back_to_a_plain_note(self):
        d = tempfile.mkdtemp()
        # same trigger, but the summarizer (ladder[0].chat) raises -> _summarize must
        # swallow it and substitute the deterministic fallback string, NOT crash.
        be = _FakeBackend([_SAY], ctx=1000, prompt_tokens=900, raise_chat=True)
        a = Agent(be, sm.EphemeralSession(d, "compfb"), max_steps=4)
        orig_head = a.messages[0]
        self._seed(a, 20)
        result = a.send("go")
        self.assertIs(a.messages[0], orig_head)                 # head still pinned
        note = a.messages[1]
        self.assertTrue(note["content"].startswith("[Earlier progress, summarized"))
        self.assertIn("earlier steps omitted", note["content"])  # the fallback, not a traceback
        self.assertEqual(result, "done")


class TestPlanPin(unittest.TestCase):
    """Plan pin/update semantics (agent.py _pin_plan + the plan-update guard)."""

    def test_plan_pinned_transiently_and_empty_list_does_not_clear(self):
        d = tempfile.mkdtemp()
        # step1 carries a non-empty plan -> self.plan is set. step2 carries an EMPTY
        # plan list -> must NOT wipe the live plan. The pinned copy must appear in the
        # backend's RECEIVED prompt (step2 onward) but NEVER in self.messages.
        step1 = '{"thought":"p","action":"bash","command":"true","plan":["[ ] one","[ ] two"]}'
        step2 = '{"thought":"n","action":"bash","command":"true","plan":[]}'
        be = _FakeBackend([step1, step2, _SAY])
        a = Agent(be, sm.EphemeralSession(d, "plan"), max_steps=6)
        result = a.send("go")

        self.assertEqual(a.plan, ["[ ] one", "[ ] two"])        # set, and NOT cleared by the empty list
        # step2's generation prompt (received[1]) must carry the transient pin
        pins = [m for m in be.received[1] if m.get("content", "").startswith("[current plan]")]
        self.assertEqual(len(pins), 1)
        self.assertEqual(pins[0]["content"], "[current plan]\n[ ] one\n[ ] two")
        # but the pin is transient — it must never be persisted into the transcript
        self.assertFalse(any("[current plan]" in m.get("content", "") for m in a.messages))
        self.assertEqual(result, "done")


class TestInboxAbsorption(unittest.TestCase):
    """Mid-run inbox absorption (agent.py _absorb_inbox)."""

    def test_fleet_message_absorbed_before_next_generation(self):
        d = tempfile.mkdtemp()
        sess = _InboxSession(d)
        be = _pushing(sess, [_bash("true"), _SAY], push_from="daemon", push_text="hold off on the refactor")
        events = []
        a = Agent(be, sess, max_steps=6, on_event=lambda k, **kw: events.append((k, kw)))
        result = a.send("work")

        tagged = "[fleet message from daemon]: hold off on the refactor"
        self.assertTrue(any(m.get("content") == tagged for m in a.messages),
                        "the mid-run fleet message must be tagged and appended")
        # it was pushed DURING step1's generation; it must be present in step2's
        # prompt (received[1]) — i.e. absorbed BEFORE the next generation, not after
        self.assertTrue(any(m.get("content") == tagged for m in be.received[1]))
        self.assertTrue(any(k == "inbox" and kw.get("sender") == "daemon" for k, kw in events))
        self.assertEqual(result, "done")

    def test_user_mid_run_message_tagged_as_steer(self):
        d = tempfile.mkdtemp()
        sess = _InboxSession(d)
        be = _pushing(sess, [_bash("true"), _SAY], push_from="user", push_text="actually stop")
        a = Agent(be, sess, max_steps=6)
        a.send("work")
        self.assertTrue(
            any(m.get("content") == "[user (mid-run — steer accordingly)]: actually stop"
                for m in a.messages),
            "a mid-run message from the user must get the steer tag, not the fleet tag")


if __name__ == "__main__":
    unittest.main()
