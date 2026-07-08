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
from forge import profile as _profile         # noqa: E402
from forge import session as sm               # noqa: E402
from forge.agent import Agent                 # noqa: E402

# P5.8 hermetic redirect: keep passport telemetry (and the per-model knobs it tunes,
# e.g. loop_threshold, which the 3x-loop test below asserts on) off the real ~/.forge.
_profile.PROFILE_DIR = tempfile.mkdtemp(prefix="forge-profile-loop-suite-")


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
        a.stuck_at = 3          # P5.7: the loop reads self.stuck_at (revived config key)
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
        a.stuck_at = 3          # P5.7: the loop reads self.stuck_at (revived config key)
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
    """Plan pin/update semantics (agent.py _pin_state + the plan-update guard)."""

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


class TestNotePin(unittest.TestCase):
    """P4.8 pinned scratch notes: a `note` field the harness pins alongside the
    plan (agent.py _pin_state / _add_note / the note-update guard). Notes are
    harness-owned state, so they survive compaction verbatim — the whole point."""

    def test_note_action_pins_a_fact_into_the_prompt(self):
        d = tempfile.mkdtemp()
        # step1 carries a `note`; it must land in self.notes and appear in step2's
        # generation prompt (received[1]) as a pinned [notes] block, but the pin is
        # transient — it must NEVER be persisted into self.messages.
        step1 = '{"thought":"n","action":"bash","command":"true","note":"config lives at src/config.py"}'
        be = _FakeBackend([step1, _SAY])
        a = Agent(be, sm.EphemeralSession(d, "note"), max_steps=6)
        result = a.send("go")

        self.assertIn("config lives at src/config.py", a.notes)
        notes_pins = [m for m in be.received[1] if "[notes]" in m.get("content", "")]
        self.assertEqual(len(notes_pins), 1)
        self.assertIn("- config lives at src/config.py", notes_pins[0]["content"])
        # transient: the pin is appended OUTSIDE self.messages every step
        self.assertFalse(any("[notes]" in m.get("content", "") for m in a.messages))
        self.assertEqual(result, "done")

    def test_notes_deduped_and_fifo_capped(self):
        d = tempfile.mkdtemp()
        be = _FakeBackend([_SAY])
        a = Agent(be, sm.EphemeralSession(d, "cap"), max_steps=2)
        a.notes = []
        # dedup: exact repeat + whitespace/case variant collapse to one
        self.assertTrue(a._add_note("fact one"))
        self.assertFalse(a._add_note("fact one"))
        self.assertFalse(a._add_note("  FACT   one "))
        self.assertEqual(a.notes, ["fact one"])
        # FIFO cap on COUNT: past NOTES_CAP the oldest is evicted
        a.notes = []
        for i in range(agent_mod.NOTES_CAP + 3):
            a._add_note("n%d" % i)
        self.assertEqual(len(a.notes), agent_mod.NOTES_CAP)
        self.assertNotIn("n0", a.notes)                                  # oldest gone
        self.assertIn("n%d" % (agent_mod.NOTES_CAP + 2), a.notes)        # newest kept
        # FIFO cap on CHARS: one big note pushes older ones out
        a.notes = []
        a._add_note("x" * 100)
        a._add_note("y" * agent_mod.NOTES_CHARS)
        self.assertNotIn("x" * 100, a.notes)
        self.assertIn("y" * agent_mod.NOTES_CHARS, a.notes)

    def test_pin_state_emits_notes_with_an_empty_plan(self):
        d = tempfile.mkdtemp()
        be = _FakeBackend([_SAY])
        a = Agent(be, sm.EphemeralSession(d, "pin"), max_steps=2)
        # notes-only (empty plan) must STILL pin — _pin_plan returned None here
        a.plan = []
        a.notes = ["test command: pytest -q", "config at src/config.py"]
        pin = a._pin_state()
        self.assertIsNotNone(pin)
        self.assertNotIn("[current plan]", pin["content"])
        self.assertIn("[notes]", pin["content"])
        self.assertIn("- test command: pytest -q", pin["content"])
        self.assertIn("- config at src/config.py", pin["content"])
        # plan AND notes present → both blocks, plan first
        a.plan = ["[ ] do it"]
        both = a._pin_state()["content"]
        self.assertTrue(both.startswith("[current plan]\n[ ] do it"))
        self.assertIn("[notes]", both)
        # both empty → None (unchanged from the old _pin_plan)
        a.plan, a.notes = [], []
        self.assertIsNone(a._pin_state())

    def test_seeded_test_command_note_appears_once(self):
        d = tempfile.mkdtemp()
        os.mkdir(os.path.join(d, "tests"))     # detect_test_cmd -> "pytest -q"
        be = _FakeBackend([_SAY])
        a = Agent(be, sm.EphemeralSession(d, "seed"), max_steps=2)
        self.assertEqual(a.notes, ["test command: pytest -q"])   # seeded ONCE at init
        # the model re-noting the same fact does not duplicate the seed
        self.assertFalse(a._add_note("test command: pytest -q"))
        self.assertEqual(a.notes.count("test command: pytest -q"), 1)

    def test_notes_survive_compaction(self):
        d = tempfile.mkdtemp()
        # same compaction trigger as TestCompaction (900 >= 0.70*1000)
        be = _FakeBackend([_SAY], ctx=1000, prompt_tokens=900, summary="STATE")
        a = Agent(be, sm.EphemeralSession(d, "notecomp"), max_steps=4)
        a.notes = ["config lives at src/config.py"]
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            a.messages.append({"role": role, "content": "<<seed %d>>" % i})
        result = a.send("go")

        contents = [m.get("content", "") for m in a.messages]
        self.assertFalse(any("<<seed 0>>" in c for c in contents))   # middle summarized away
        # the note is NOT stored in self.messages (harness state), yet it STILL rides
        # the post-compaction prompt because the pin is rebuilt from self.notes
        self.assertFalse(any("[notes]" in c for c in contents))
        self.assertTrue(any("- config lives at src/config.py" in m.get("content", "")
                            for m in be.received[-1]))
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


class _BorrowRung:
    """A stronger ladder rung: its chat() — the _borrow entry point — records every
    call (so a test can prove a borrow happened, and how many times) and returns a
    scripted action. stream() serves the same action when this rung is driven directly;
    warm() is a counted no-op."""
    def __init__(self, action, name="strong"):
        self.name = name
        self._action = action
        self.chat_calls = []
        self.warmed = 0

    def stream(self, messages, schema=None, temperature=0.0):
        yield self._action

    def chat(self, messages, schema=None, temperature=0.0):
        self.chat_calls.append(list(messages))
        return self._action

    def warm(self):
        self.warmed += 1


_BASH_OK = '{"thought":"b","action":"bash","command":"true"}'


class TestBorrowingAndDecay(unittest.TestCase):
    """P5.7 — step-scoped borrowing (buy ONE strong-rung generation at the exact stuck
    points, WITHOUT swapping self.backend) + unified stuck ledger + tier decay. Written
    to FAIL if a mechanism is disabled: the 3rd malformed strike must borrow before the
    abort-at-5, a repeated per-sig failure must borrow, the escalation lever OFF must
    disable borrowing entirely, and an escalated tier must decay back."""

    def test_third_malformed_strike_borrows_before_abort(self):
        # three unsalvageable outputs → the 3rd routes through _borrow (rung1.chat) and
        # the borrowed action executes, so the turn reaches "done" instead of aborting.
        d = tempfile.mkdtemp()
        cheap = _ScriptBackend(["not json", "not json", "not json", _SAY], name="weak")
        strong = _BorrowRung(_BASH_OK, name="strong")
        events = []
        a = Agent([cheap, strong], sm.EphemeralSession(d, "mb"), max_steps=12,
                  on_event=lambda k, **kw: events.append((k, kw)))
        result = a.send("go")
        self.assertEqual(len(strong.chat_calls), 1)          # borrowed exactly once, at strike 3
        self.assertNotIn("could not hold", result)           # the borrow averted the abort
        self.assertEqual(result, "done")
        self.assertEqual(a.tier, 0)                          # borrow does NOT stick the tier
        self.assertTrue(any(k == "borrow" and kw.get("model") == "strong" for k, kw in events))

    def test_escalation_lever_off_disables_borrow(self):
        # same stuck sequence but the escalation lever is OFF: no borrow ever, and the
        # turn falls through to the unchanged malformed abort at 5 strikes.
        d = tempfile.mkdtemp()
        cheap = _ScriptBackend(["not json"] * 5, name="weak")
        strong = _BorrowRung(_BASH_OK, name="strong")
        levers = frozenset(agent_mod.ALL_LEVERS - {"escalation"})
        a = Agent([cheap, strong], sm.EphemeralSession(d, "off"), max_steps=12, levers=levers)
        result = a.send("go")
        self.assertEqual(len(strong.chat_calls), 0)          # lever off → never borrows
        self.assertIn("could not hold", result)              # abort-at-5 preserved

    def test_repeated_sig_failure_borrows(self):
        # the SAME failing action, repeated until its signature has failed 3x, is the
        # second borrow trigger. stuck_at is pinned high so this is isolated from the
        # score-driven escalation (which would otherwise fire on the same step).
        d = tempfile.mkdtemp()
        bx = _bash("false #x")
        cheap = _ScriptBackend([bx, bx, bx, bx, _SAY], name="weak")
        strong = _BorrowRung(_BASH_OK, name="strong")
        events = []
        a = Agent([cheap, strong], sm.EphemeralSession(d, "sf"), max_steps=12,
                  on_event=lambda k, **kw: events.append((k, kw)))
        a.stuck_at = 100
        result = a.send("go")
        self.assertEqual(len(strong.chat_calls), 1)          # borrowed when sig_fails hit 3
        self.assertEqual(a.tier, 0)                          # borrow does NOT escalate
        self.assertEqual(result, "done")
        self.assertTrue(any(k == "borrow" for k, kw in events))

    def test_tier_decays_after_clean_steps(self):
        # forced to tier 1 (sticky, so the turn-boundary decay is suppressed); CLEAN_DECAY
        # clean steps in a row must relax exactly one rung back to tier 0.
        d = tempfile.mkdtemp()
        cheap = _ScriptBackend([_SAY], name="weak")
        clean = [_bash("true #%d" % i) for i in range(agent_mod.CLEAN_DECAY)]
        strong = _ScriptBackend(clean + [_SAY], name="strong")
        events = []
        a = Agent([cheap, strong], sm.EphemeralSession(d, "decay"), max_steps=20,
                  on_event=lambda k, **kw: events.append((k, kw)))
        a.sticky_escalation = True
        a.tier, a.backend = 1, a.ladder[1]
        result = a.send("go")
        self.assertEqual(a.tier, 0)                          # decayed one rung after clean steps
        self.assertTrue(any(k == "deescalate" for k, kw in events))
        self.assertEqual(result, "done")

    def test_turn_boundary_decays_tier_unless_sticky(self):
        # a fresh turn relaxes an escalated tier by one rung at send() entry…
        d = tempfile.mkdtemp()
        cheap = _ScriptBackend([_SAY], name="weak")
        strong = _ScriptBackend([_SAY], name="strong")
        events = []
        a = Agent([cheap, strong], sm.EphemeralSession(d, "tb"), max_steps=4,
                  on_event=lambda k, **kw: events.append((k, kw)))
        a.sticky_escalation = False
        a.tier, a.backend = 1, a.ladder[1]
        a.send("go")
        self.assertEqual(a.tier, 0)
        self.assertTrue(any(k == "deescalate" for k, kw in events))

    def test_sticky_escalation_pins_the_tier(self):
        # …but sticky_escalation keeps it across the turn boundary.
        d = tempfile.mkdtemp()
        cheap = _ScriptBackend([_SAY], name="weak")
        strong = _ScriptBackend([_SAY], name="strong")
        events = []
        a = Agent([cheap, strong], sm.EphemeralSession(d, "sticky"), max_steps=4,
                  on_event=lambda k, **kw: events.append((k, kw)))
        a.sticky_escalation = True
        a.tier, a.backend = 1, a.ladder[1]
        a.send("go")
        self.assertEqual(a.tier, 1)                          # sticky → no decay
        self.assertFalse(any(k == "deescalate" for k, kw in events))

    def test_step_trace_flags_survive_borrowing(self):
        # the P3.1 flight recorder must keep firing across a borrow: the malformed strike,
        # the per-step trace, and the new `borrowed` provenance flag all land in the log.
        d = tempfile.mkdtemp()
        sess = _InboxSession(d)
        cheap = _ScriptBackend(["not json", "not json", "not json", _SAY], name="weak")
        strong = _BorrowRung(_BASH_OK, name="strong")
        result = Agent([cheap, strong], sess, max_steps=12).send("go")
        kinds = [k for k, f in sess.logs]
        self.assertIn("malformed", kinds)
        self.assertIn("borrow", kinds)
        steps = [f for k, f in sess.logs if k == "step"]
        self.assertTrue(steps)
        self.assertTrue(all("v" in s and "step" in s and "tier" in s for s in steps))
        self.assertTrue(any(s.get("malformed") for s in steps))          # P3.1 malformed flag
        self.assertTrue(any(s.get("borrowed") == "strong" for s in steps))  # P5.7 borrow flag
        self.assertEqual(result, "done")

    def test_stuck_threshold_config_key_is_revived(self):
        # the previously-dead config key now sets self.stuck_at at construction.
        d = tempfile.mkdtemp()
        from forge import config as cfg
        with mock.patch.dict(os.environ, {}), \
                mock.patch.object(cfg, "get", lambda k, *a: 4 if k == "stuck_threshold" else None):
            os.environ.pop("FORGE_STUCK_THRESHOLD", None)   # env would otherwise win
            a = Agent(_ScriptBackend([_SAY]), sm.EphemeralSession(d, "cfg"), max_steps=2)
        self.assertEqual(a.stuck_at, 4)


if __name__ == "__main__":
    unittest.main()
