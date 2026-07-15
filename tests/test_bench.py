"""P3.2 tests — harness levers (ablation switches) + the `forge bench` runner.

Stdlib unittest only. No network, no real model, no real daemon: every Agent runs
against a fake/scripted backend and an EphemeralSession, and every bench fixture is
a throwaway tempdir whose verdict is a hardcoded verify.sh exit code.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import session as sm            # noqa: E402
from forge import bench                    # noqa: E402
from forge.agent import Agent, ALL_LEVERS  # noqa: E402


def _write(p, s):
    with open(p, "w") as f:
        f.write(s)


def _read(p):
    with open(p) as f:
        return f.read()


class ScriptBackend:
    """Yields a scripted sequence of action JSONs (one per stream call)."""
    name = "script"

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def stream(self, messages, schema=None, temperature=0.0):
        act = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        yield act

    def chat(self, messages, schema=None, temperature=0.0):
        return '{"thought":"x","action":"say","message":"done"}'


class RecordingBackend:
    """Records the exact prompt (message list) handed to it each step."""
    name = "rec"

    def __init__(self, action):
        self.action = action
        self.prompts = []

    def stream(self, messages, schema=None, temperature=0.0):
        self.prompts.append([dict(m) for m in messages])
        self.schema = schema
        yield self.action

    def chat(self, messages, schema=None, temperature=0.0):
        self.schema = schema
        return self.action


SAY = '{"thought":"x","action":"say","message":"done"}'


def _make_fixture(prompt="do it", verify=None, setup=None, extra=None):
    """A throwaway bench fixture dir. Returns its path."""
    d = tempfile.mkdtemp(prefix="fixture-")
    _write(os.path.join(d, "prompt.txt"), prompt)
    if verify is not None:
        _write(os.path.join(d, "verify.sh"), verify)
    if setup is not None:
        _write(os.path.join(d, "setup.sh"), setup)
    for name, body in (extra or {}).items():
        _write(os.path.join(d, name), body)
    return d


# ----------------------------------------------------------------------------
# Part A — lever gating
# ----------------------------------------------------------------------------

class TestLeverGating(unittest.TestCase):
    def test_default_levers_is_all(self):
        a = Agent(ScriptBackend([SAY]), sm.EphemeralSession(tempfile.mkdtemp(), "s"))
        self.assertEqual(a.levers, ALL_LEVERS)
        self.assertEqual(len(ALL_LEVERS), 9)

    def test_bare_does_not_block_read_before_edit(self):
        """With NO levers, a blind edit of an unread existing file is allowed
        (the read-before-edit guard is off)."""
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        actions = [
            '{"thought":"blind","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',
            SAY,
        ]
        events = []
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(d, "s"), max_steps=6,
                  levers=frozenset(),
                  on_event=lambda k, **kw: events.append((k, kw.get("ok"))))
        a.send("change x")
        obs = [ok for k, ok in events if k == "observation"]
        self.assertTrue(obs[0])                       # blind edit was NOT blocked
        self.assertIn("x = 2", _read(os.path.join(d, "r.py")))

    def test_default_still_guards_read_before_edit(self):
        """levers=None keeps the guard: the SAME blind edit never writes from memory.
        The harness serves the file's real content instead, and only the retry lands —
        the exact contrast with the lever-off case above, which writes blind at step 0."""
        d = tempfile.mkdtemp()
        _write(os.path.join(d, "r.py"), "x = 1\n")
        actions = [
            '{"thought":"blind","action":"edit_file","path":"r.py","old":"x = 1","new":"x = 2"}',
            SAY,
        ]
        events = []
        a = Agent(ScriptBackend(actions), sm.EphemeralSession(d, "s"), max_steps=6,
                  levers=None,
                  on_event=lambda k, **kw: events.append((k, kw)))
        a.send("change x")
        obs = [(kw.get("text", ""), kw.get("ok")) for k, kw in events if k == "observation"]
        self.assertIn("without reading it", obs[0][0])   # served the read rather than writing blind
        self.assertEqual(_read(os.path.join(d, "r.py")), "x = 1\n")   # guard held: nothing written

    def test_compaction_off_never_calls_compact(self):
        d = tempfile.mkdtemp()
        a = Agent(ScriptBackend([SAY]), sm.EphemeralSession(d, "s"), max_steps=3,
                  levers=frozenset())
        called = []
        a._compact = lambda: called.append(1)
        a.send("go")
        self.assertEqual(called, [])                  # compaction lever off → never called

    def test_compaction_on_calls_compact(self):
        d = tempfile.mkdtemp()
        a = Agent(ScriptBackend([SAY]), sm.EphemeralSession(d, "s"), max_steps=3,
                  levers=None)
        called = []
        a._compact = lambda: called.append(1)
        a.send("go")
        self.assertTrue(called)                       # default → compaction called each step

    def test_plan_pin_off_not_in_prompt(self):
        d = tempfile.mkdtemp()
        b = RecordingBackend(SAY)
        a = Agent(b, sm.EphemeralSession(d, "s"), max_steps=3, levers=frozenset())
        a.plan = ["[ ] do the thing"]
        a.send("go")
        joined = " ".join(m.get("content", "") for m in b.prompts[0])
        self.assertNotIn("[current plan]", joined)

    def test_plan_pin_on_in_prompt(self):
        d = tempfile.mkdtemp()
        b = RecordingBackend(SAY)
        a = Agent(b, sm.EphemeralSession(d, "s"), max_steps=3, levers=None)
        a.plan = ["[ ] do the thing"]
        a.send("go")
        joined = " ".join(m.get("content", "") for m in b.prompts[0])
        self.assertIn("[current plan]", joined)

    def test_schema_lever_controls_constrained_decoding(self):
        """schema lever on → the P5.1 per-action grammar (anyOf of the legal action
        variants) is passed; off → None."""
        from forge.tools import build_schema, ALL_ACTIONS
        d = tempfile.mkdtemp()
        b_on = RecordingBackend(SAY)
        Agent(b_on, sm.EphemeralSession(d, "s"), max_steps=2, levers=None).send("go")
        # auto mode, no allow-list → every action is legal this step
        self.assertIn("anyOf", b_on.schema)
        self.assertEqual(b_on.schema, build_schema(set(ALL_ACTIONS), "auto"))
        b_off = RecordingBackend(SAY)
        Agent(b_off, sm.EphemeralSession(d, "s"), max_steps=2, levers=frozenset()).send("go")
        self.assertIsNone(b_off.schema)

    def test_workspace_lever_controls_briefing_injection(self):
        d = tempfile.mkdtemp()
        on = Agent(ScriptBackend([SAY]), sm.EphemeralSession(d, "s"),
                   workspace="BRIEFING TEXT", levers=None)
        self.assertTrue(any("BRIEFING TEXT" in m["content"] for m in on.messages))
        self.assertEqual(on.head_len, 3)              # system + 2 workspace msgs
        off = Agent(ScriptBackend([SAY]), sm.EphemeralSession(d, "s"),
                    workspace="BRIEFING TEXT", levers=frozenset())
        self.assertFalse(any("BRIEFING TEXT" in m["content"] for m in off.messages))
        self.assertEqual(off.head_len, 1)             # system only, head_len still correct


# ----------------------------------------------------------------------------
# Part B — bench runner + report
# ----------------------------------------------------------------------------

class TestBenchRunner(unittest.TestCase):
    def test_run_task_pass_verdict(self):
        fx = _make_fixture(verify="exit 0\n")
        row = bench.run_task(fx, ScriptBackend([SAY]), ALL_LEVERS, max_steps=3, model="m")
        self.assertIs(row["pass"], True)
        for key in ("model", "levers", "task", "pass", "steps", "seconds",
                    "escalations", "malformed", "loops"):
            self.assertIn(key, row)
        self.assertEqual(row["levers"], sorted(ALL_LEVERS))
        self.assertEqual(row["model"], "m")

    def test_run_task_fail_verdict(self):
        fx = _make_fixture(verify="exit 1\n")
        row = bench.run_task(fx, ScriptBackend([SAY]), frozenset(), max_steps=3, model="m")
        self.assertIs(row["pass"], False)
        self.assertEqual(row["levers"], [])

    def test_run_task_counts_steps(self):
        fx = _make_fixture(verify="exit 0\n")
        actions = [
            '{"thought":"w","action":"write_file","path":"out.txt","content":"hi"}',
            SAY,
        ]
        row = bench.run_task(fx, ScriptBackend(actions), ALL_LEVERS, max_steps=5, model="m")
        self.assertEqual(row["steps"], 1)             # one concrete action before say

    def test_run_task_setup_sh_runs(self):
        """setup.sh runs in the workspace before the agent; verify.sh checks it."""
        fx = _make_fixture(setup="echo ready > marker\n", verify="test -f marker\n")
        row = bench.run_task(fx, ScriptBackend([SAY]), ALL_LEVERS, max_steps=3, model="m")
        self.assertIs(row["pass"], True)

    def test_run_task_no_verdict_is_none(self):
        """No verify.sh and no detectable test command → no verdict (not a fail)."""
        fx = _make_fixture(verify=None)
        row = bench.run_task(fx, ScriptBackend([SAY]), ALL_LEVERS, max_steps=3, model="m")
        self.assertIsNone(row["pass"])

    def test_ephemeral_session_not_in_registry(self):
        d = tempfile.mkdtemp()
        s = sm.EphemeralSession(d, "m")
        s.register()                                  # no-op
        self.assertNotIn(s.sid, {e["sid"] for e in sm.registry()})
        before = {e["sid"] for e in sm.registry()}
        bench.run_task(_make_fixture(verify="exit 0\n"), ScriptBackend([SAY]),
                       ALL_LEVERS, max_steps=3, model="m")
        after = {e["sid"] for e in sm.registry()}
        self.assertFalse(after - before)              # bench registered nothing


class TestBenchReport(unittest.TestCase):
    def _rows(self, model, levers, verdicts):
        return [{"model": model, "levers": list(levers), "task": f"t{i}", "pass": v}
                for i, v in enumerate(verdicts)]

    def test_lift_table_shows_bare_vs_full(self):
        full = sorted(ALL_LEVERS)
        rows = (self._rows("qwen", [], [False, False, True]) +      # bare: 1/3
                self._rows("qwen", full, [True, True, True]))       # full: 3/3
        out = bench.report(rows)
        self.assertIn("HARNESS-LIFT", out)
        self.assertIn("bare", out)
        self.assertIn("full", out)
        self.assertIn("1/3 33%", out)
        self.assertIn("3/3 100%", out)
        self.assertIn("+67pts", out)

    def test_ablation_table(self):
        full = sorted(ALL_LEVERS)
        ablate = sorted(ALL_LEVERS - {"compaction"})
        rows = (self._rows("qwen", full, [True, True]) +
                self._rows("qwen", ablate, [True, False]))
        out = bench.report(rows)
        self.assertIn("ABLATION", out)
        self.assertIn("compaction", out)

    def test_config_label(self):
        self.assertEqual(bench.config_label([]), "bare")
        self.assertEqual(bench.config_label(sorted(ALL_LEVERS)), "full")
        self.assertEqual(bench.config_label(sorted(ALL_LEVERS - {"read_gate"})), "full-read_gate")


class TestBenchConfigs(unittest.TestCase):
    class _Args:
        def __init__(self, **k):
            self.bare = self.no_compact = self.no_loop_detect = False
            self.no_read_gate = self.single_rung = False
            for key, v in k.items():
                setattr(self, key, v)

    def test_default_is_bare_and_full(self):
        labels = [lab for lab, _ in bench.configs_for(self._Args())]
        self.assertIn("bare", labels)
        self.assertIn("full", labels)

    def test_ablation_flag_adds_config(self):
        cfgs = bench.configs_for(self._Args(no_compact=True))
        labels = [lab for lab, _ in cfgs]
        self.assertIn("full", labels)
        self.assertIn("full-compaction", labels)
        # the ablation config really has compaction removed
        levers = dict(cfgs)["full-compaction"]
        self.assertNotIn("compaction", levers)
        self.assertEqual(levers, ALL_LEVERS - {"compaction"})

    def test_list_tasks_finds_seed_fixtures(self):
        tasks = bench.list_tasks()
        self.assertIn("fix-failing-test", tasks)
        self.assertIn("implement-function", tasks)


if __name__ == "__main__":
    unittest.main()
