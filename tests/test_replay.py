"""P3.3 — flight recorder + forge replay.

Every step's RAW model output (malformed ones included) is logged into the
transcript; RecordingBackend mirrors chat/stream into a {digest, raw,
prompt_tokens} cassette; ReplayBackend re-drives a REAL Agent.send from recorded
raws with NO model. The fixture sweep turns tests/fixtures/*.jsonl into
zero-inference regression tests. Stdlib unittest, offline, deterministic.
"""
import glob
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.agent import Agent                                    # noqa: E402
from forge.backends import RecordingBackend, record_digest       # noqa: E402
from forge import replay as replaymod                            # noqa: E402


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class _RecSession:
    """A minimal session that records log() calls — enough to drive Agent.send
    and inspect the transcript."""
    def __init__(self, cwd, sid="rec"):
        self.cwd, self.sid, self.name = cwd, sid, "rec"
        self.status = "idle"
        self.logs = []
    def log(self, kind, **fields): self.logs.append((kind, fields))
    def drain(self): return []
    def set_status(self, s): self.status = s
    def push(self, sender, text): pass


class _FakeBackend:
    """A fake model: chat returns a fixed action, stream yields it in chunks, and
    it exposes the same surface (name / effective_ctx / context_window /
    last_prompt_tokens / warm) a real backend does."""
    name = "fake"

    def __init__(self, chat_raw='{"action":"say","message":"c"}',
                 stream_chunks=('{"action":', '"say","message":"s"}')):
        self.last_prompt_tokens = 0
        self._chat_raw = chat_raw
        self._chunks = list(stream_chunks)

    def context_window(self): return 4096
    def effective_ctx(self): return 4096
    def warm(self): return "warmed"

    def chat(self, messages, schema=None, temperature=0.0):
        self.last_prompt_tokens = 111
        return self._chat_raw

    def stream(self, messages, schema=None, temperature=0.0):
        for ch in self._chunks:
            yield ch
        self.last_prompt_tokens = 222


class _StreamScript:
    """Streams a scripted sequence of raw action strings, one per step (the last
    repeats). Reports a fixed prompt-token count so the recorder has something to
    capture."""
    name = "script"

    def __init__(self, raws, ptoks=42):
        self.raws = list(raws)
        self.i = 0
        self.last_prompt_tokens = ptoks
    def stream(self, messages, schema=None, temperature=0.0):
        yield self.raws[min(self.i, len(self.raws) - 1)]
        self.i += 1
    def chat(self, messages, schema=None, temperature=0.0):
        return '{"action":"say","message":"fallback"}'


class TestRecordingBackend(unittest.TestCase):
    def test_records_chat_and_stream_rows(self):
        d = tempfile.mkdtemp()
        cass = os.path.join(d, "cassette.jsonl")
        rb = RecordingBackend(_FakeBackend(), cass)
        raw_chat = rb.chat([{"role": "user", "content": "hi"}])
        raw_stream = "".join(rb.stream([{"role": "user", "content": "yo"}]))
        self.assertEqual(raw_chat, '{"action":"say","message":"c"}')
        self.assertEqual(raw_stream, '{"action":"say","message":"s"}')
        rows = _read_jsonl(cass)
        self.assertEqual(len(rows), 2)
        # each row: {digest, raw, prompt_tokens}
        for row in rows:
            self.assertEqual(set(row), {"digest", "raw", "prompt_tokens"})
        self.assertEqual(rows[0]["raw"], raw_chat)
        self.assertEqual(rows[0]["prompt_tokens"], 111)      # captured after chat()
        self.assertEqual(rows[1]["raw"], raw_stream)
        self.assertEqual(rows[1]["prompt_tokens"], 222)      # from the stream's usage frame
        # digest matches the shared helper on the same messages
        self.assertEqual(rows[0]["digest"], record_digest([{"role": "user", "content": "hi"}]))

    def test_forwards_backend_surface(self):
        rb = RecordingBackend(_FakeBackend(), os.path.join(tempfile.mkdtemp(), "c"))
        self.assertEqual(rb.name, "fake")
        self.assertEqual(rb.effective_ctx(), 4096)
        self.assertEqual(rb.context_window(), 4096)
        self.assertEqual(rb.warm(), "warmed")
        rb.chat([{"role": "user", "content": "x"}])
        self.assertEqual(rb.last_prompt_tokens, 111)         # delegated to the inner backend


class TestReplayBackend(unittest.TestCase):
    def test_pops_raws_in_order_loose(self):
        recs = [{"raw": "a"}, {"raw": "b"}, {"raw": "c"}]
        ladder = replaymod.build_ladder(recs)
        b = ladder[0]
        self.assertEqual("".join(b.stream([])), "a")
        self.assertEqual("".join(b.stream([])), "b")
        self.assertEqual("".join(b.stream([])), "c")
        # exhausted → a clean terminal say, not an IndexError
        out = "".join(b.stream([]))
        self.assertEqual(json.loads(out)["action"], "say")

    def test_replays_prompt_tokens_for_compaction_timing(self):
        ladder = replaymod.build_ladder([{"raw": "x", "prompt_tokens": 1234}])
        b = ladder[0]
        "".join(b.stream([]))
        self.assertEqual(b.last_prompt_tokens, 1234)

    def test_strict_mode_raises_on_digest_mismatch(self):
        recs = [{"raw": '{"action":"say","message":"x"}', "digest": "deadbeef"}]
        ladder = replaymod.build_ladder(recs, strict=True)
        with self.assertRaises(replaymod.ReplayDivergence):
            list(ladder[0].stream([{"role": "user", "content": "anything"}]))

    def test_strict_mode_passes_on_matching_digest(self):
        msgs = [{"role": "user", "content": "hi"}]
        recs = [{"raw": '{"action":"say","message":"ok"}', "digest": record_digest(msgs)}]
        ladder = replaymod.build_ladder(recs, strict=True)
        out = "".join(ladder[0].stream(msgs))
        self.assertEqual(json.loads(out)["message"], "ok")

    def test_ladder_has_a_rung_per_recorded_tier(self):
        recs = [{"raw": "a", "tier": 0}, {"raw": "b", "tier": 2}]
        ladder = replaymod.build_ladder(recs)
        self.assertEqual(len(ladder), 3)                     # tiers 0,1,2 all present

    def test_drives_agent_send_to_terminal_say(self):
        raws = [
            '{"thought":"greet","action":"bash","command":"echo hi"}',
            '{"thought":"done","action":"say","message":"all set"}',
        ]
        recs = [{"raw": r, "tier": 0, "prompt_tokens": 0} for r in raws]
        result = replaymod.replay_records(
            {"model": "fake", "mode": "auto"},
            [{"user": "do it", "model": recs}])
        self.assertEqual(result["terminals"], ["all set"])


class TestRawLogged(unittest.TestCase):
    """The raw model output of EVERY step is logged — including the malformed
    ones the parser discards (the valuable ones)."""

    def test_valid_and_malformed_raws_both_logged(self):
        d = tempfile.mkdtemp()
        sess = _RecSession(d)
        b = _StreamScript([
            "this is not valid action json",                 # malformed
            '{"thought":"t","action":"say","message":"done"}',
        ])
        Agent(b, sess, max_steps=6).send("go")
        models = [f for k, f in sess.logs if k == "model"]
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["raw"], "this is not valid action json")   # malformed kept
        self.assertEqual(models[1]["raw"], '{"thought":"t","action":"say","message":"done"}')
        for m in models:
            self.assertEqual(m["tier"], 0)
            self.assertEqual(m["prompt_tokens"], 42)          # replayable for compaction
        # and the malformed branch still fired its own record
        self.assertEqual(len([f for k, f in sess.logs if k == "malformed"]), 1)


class TestRecordReplayRoundTrip(unittest.TestCase):
    """Record a run through RecordingBackend, then replay the recorded raws
    through ReplayBackend into a fresh Agent — same terminal, zero inference."""

    def test_cassette_raws_replay_to_same_terminal(self):
        raws = [
            '{"thought":"look","action":"bash","command":"echo hi"}',
            '{"thought":"ok","action":"say","message":"finished"}',
        ]
        d = tempfile.mkdtemp()
        cass = os.path.join(d, "cassette.jsonl")
        rec = RecordingBackend(_StreamScript(raws), cass)
        term1 = Agent(rec, _RecSession(d), max_steps=6).send("go")
        self.assertEqual(term1, "finished")
        rows = _read_jsonl(cass)
        # feed the recorded raws back through ReplayBackend
        result = replaymod.replay_records(
            {"model": "script", "mode": "auto"},
            [{"user": "go", "model": [{"raw": r["raw"]} for r in rows]}])
        self.assertEqual(result["terminals"], [term1])


class TestFixtureRoundTrip(unittest.TestCase):
    """write_fixture(sid) distills a transcript into tests/fixtures/<name>.jsonl;
    the fixture then replays to the recorded terminal — via a monkeypatched record
    source so the test never touches ~/.forge."""

    def test_transcript_to_fixture_to_replay(self):
        transcript = [
            {"type": "meta", "model": "ollama:x", "cwd": "/rec", "mode": "auto"},
            {"type": "user", "text": "make it"},
            {"type": "model", "raw": '{"action":"bash","command":"echo ok"}', "tier": 0, "prompt_tokens": 5},
            {"type": "step", "step": 1, "action": "bash", "window": 8192, "used": 5},
            {"type": "model", "raw": '{"action":"say","message":"shipped"}', "tier": 0, "prompt_tokens": 7},
            {"type": "assistant", "text": "shipped"},
            {"type": "step", "step": 2, "action": "say", "window": 8192, "used": 7},
        ]
        orig = replaymod._records_for
        replaymod._records_for = lambda sid: transcript
        try:
            path = replaymod.write_fixture("somesid", "_rt_tmp_fixture")
        finally:
            replaymod._records_for = orig
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        rows = _read_jsonl(path)
        meta = rows[0]
        self.assertEqual(meta["type"], "meta")
        self.assertEqual(meta["expected"], "shipped")
        self.assertEqual(meta["window"], 8192)
        result = replaymod.replay_fixture(path)
        self.assertEqual(result["terminals"][-1], "shipped")


class TestDivergenceReport(unittest.TestCase):
    def test_first_divergence_spots_action_change(self):
        rec = [{"action": "bash"}, {"action": "read_file"}, {"action": "say"}]
        rep = [{"action": "bash"}, {"action": "write_file"}, {"action": "say"}]
        self.assertEqual(replaymod.first_divergence(rec, rep), 2)

    def test_matching_paths_report_no_divergence(self):
        steps = [{"action": "bash"}, {"action": "say"}]
        self.assertIsNone(replaymod.first_divergence(steps, list(steps)))

    def test_replay_reports_match_on_identical_harness(self):
        transcript = [
            {"type": "meta", "model": "m", "cwd": "/x", "mode": "auto"},
            {"type": "user", "text": "go"},
            {"type": "model", "raw": '{"action":"say","message":"hi"}', "tier": 0, "prompt_tokens": 3},
            {"type": "step", "step": 1, "action": "say", "window": 8192, "used": 3},
        ]
        orig = replaymod._records_for
        replaymod._records_for = lambda sid: transcript
        try:
            report = replaymod.replay("sid1234")
        finally:
            replaymod._records_for = orig
        self.assertIn("terminal[1]: hi", report)
        self.assertIn("MATCHES", report)


class TestFixtureSweep(unittest.TestCase):
    """Every tests/fixtures/*.jsonl is a zero-inference regression test: replay it
    through a real Agent.send and assert the recorded terminal state."""

    def test_all_fixtures_replay_to_recorded_terminal(self):
        fixtures = sorted(glob.glob(os.path.join(replaymod.fixtures_dir(), "*.jsonl")))
        self.assertTrue(fixtures, "no fixtures found under tests/fixtures/")
        for path in fixtures:
            with self.subTest(fixture=os.path.basename(path)):
                meta, turns, window = replaymod.load_fixture(path)
                result = replaymod.replay_fixture(path)
                self.assertTrue(result["terminals"], "replay produced no terminal state")
                if meta.get("expected") is not None:
                    self.assertEqual(result["terminals"][-1], meta["expected"])


if __name__ == "__main__":
    unittest.main()
