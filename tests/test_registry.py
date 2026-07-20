"""Curated model registry: entry shape, honest status, fit math, runbooks."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import registry as REG                                  # noqa: E402


class TestRegistryShape(unittest.TestCase):
    REQUIRED = ("name", "repo", "arch", "kind", "engine", "ram_gb_needed", "status", "notes")

    def test_every_entry_is_well_formed(self):
        self.assertTrue(REG.MODELS)
        for m in REG.MODELS:
            for k in self.REQUIRED:
                self.assertIn(k, m, f"{m.get('name', '?')} missing {k}")
            self.assertIn(m["status"], ("verified", "candidate"))

    def test_verified_entries_carry_a_report(self):
        # "verified" is a claim — it must point at the docs/models evidence.
        for m in REG.MODELS:
            if m["status"] == "verified":
                self.assertTrue(m.get("report"), f"{m['name']} verified but has no report")

    def test_engine_specific_fields_present(self):
        # a bad entry would crash runbook()/use() at print/pull time — assert up front.
        for m in REG.MODELS:
            if m["engine"] == "ollama":
                self.assertTrue(m.get("ollama_tag"), f"{m['name']} ollama entry needs ollama_tag")
            elif m["engine"] == "llamacpp":
                self.assertTrue(m.get("arch_probe"), f"{m['name']} llamacpp entry needs arch_probe")
                self.assertTrue((m.get("weights") or {}).get("gguf_repo"),
                                f"{m['name']} llamacpp entry needs weights.gguf_repo")

    def test_names_are_unique(self):
        ns = REG.names()
        self.assertEqual(len(ns), len(set(ns)))

    def test_get(self):
        self.assertIsNotNone(REG.get("sarvam-30b"))
        self.assertIsNone(REG.get("nope"))


class TestFit(unittest.TestCase):
    def test_fit_is_ram_vs_need(self):
        sarvam = REG.get("sarvam-30b")
        self.assertTrue(REG.fits(sarvam, 48))                  # the machine it ran on
        self.assertFalse(REG.fits(sarvam, 8))                  # budget laptop: honest no
        self.assertFalse(REG.fits(sarvam, 0))                  # unknown RAM → no claim

    def test_small_models_fit_small_machines(self):
        self.assertTrue(REG.fits(REG.get("phi-2"), 8))
        self.assertTrue(REG.fits(REG.get("bitnet-b1.58-2b"), 8))


class TestRunsAndCeiling(unittest.TestCase):
    """The 'what can this machine run' logic: fit + speed by hardware class."""

    def test_runs_verdicts(self):
        big = REG.get("llama3.1:70b")          # 70B, needs 48GB
        mid = REG.get("qwen2.5-coder:7b")      # 7B, needs 8GB
        small = REG.get("llama3.2:3b")         # 3B, needs 5GB
        # won't fit — RAM too small (regardless of hardware)
        self.assertEqual(REG.runs(big, 8, True)[0], "won't fit")
        # fits + accelerated → runs well even for big
        self.assertEqual(REG.runs(big, 48, True)[0], "runs well")
        # fits + CPU-only + 7B → usable but slow
        self.assertEqual(REG.runs(mid, 8, False)[0], "usable · slower")
        # fits + CPU-only + small → runs well
        self.assertEqual(REG.runs(small, 8, False)[0], "runs well")
        # fits + GPU/Metal + 7B → runs well (accel beats the CPU speed cap)
        self.assertEqual(REG.runs(mid, 16, True)[0], "runs well")

    def test_ceiling_scales_with_machine(self):
        cap_p, cap_name = REG.ceiling(48, True)                # 48GB Apple Silicon
        self.assertGreaterEqual(cap_p, 30)                     # can run big models
        # 8GB CPU-only: ceiling is a small model (~3B), not a 7B that would crawl
        cap_p8, cap_name8 = REG.ceiling(8, False)
        self.assertLessEqual(cap_p8, 3.5)
        self.assertIsNotNone(cap_name8)
        # 1GB: essentially nothing runs well
        self.assertEqual(REG.ceiling(1, False), (0, None))

    def test_catalog_spans_a_real_range(self):
        params = sorted(m.get("params_b") or 0 for m in REG.MODELS)
        self.assertLessEqual(params[0], 1)                     # something tiny
        self.assertGreaterEqual(params[-1], 30)                # something big
        self.assertGreater(len(REG.MODELS), 10)                # a real spread, not 3

    def test_gpu_uses_vram_not_system_ram(self):
        big = REG.get("llama3.1:70b")      # ~40GB weights, ram_gb_needed 48
        mid = REG.get("gemma2:9b")         # ~5.4GB weights
        # small GPU (8GB VRAM) even with lots of RAM: 70B does NOT fit VRAM → not "runs well"
        self.assertNotEqual(REG.runs(big, 64, True, vram_gb=8)[0], "runs well")
        self.assertEqual(REG.runs(mid, 64, True, vram_gb=8)[0], "runs well")      # 9B fits 8GB
        # datacenter (640GB VRAM): 70B runs well
        self.assertEqual(REG.runs(big, 512, True, vram_gb=640)[0], "runs well")

    def test_gpu_spill_then_wont_fit(self):
        big = REG.get("llama3.1:70b")      # ram_gb_needed 48
        self.assertEqual(REG.runs(big, 64, True, vram_gb=8)[0], "usable · slower")  # 8+64 ≥ 48 → spills
        self.assertEqual(REG.runs(big, 16, True, vram_gb=8)[0], "won't fit")        # 8+16 < 48 → no

    def test_ceiling_respects_vram(self):
        cap_small, _ = REG.ceiling(64, True, vram_gb=8)        # tiny GPU, big RAM → VRAM-bound
        self.assertLessEqual(cap_small, 13)
        cap_dc, _ = REG.ceiling(512, True, vram_gb=640)        # datacenter → up to catalog max
        self.assertGreaterEqual(cap_dc, 30)


class TestRunbook(unittest.TestCase):
    def test_sarvam_runbook_reproduces_the_verified_recipe(self):
        text = "\n".join(REG.runbook(REG.get("sarvam-30b")))
        for must in ("strings", "sarvam-moe",                  # the arch gate, BEFORE the download
                     "curl -L -C -",                           # resumable weights
                     "llama-server", "--jinja",                # serve flags
                     "FORGE_REMOTE_CTX=16384",                 # the ctx-match gotcha
                     "openai:sarvam-30b@http://127.0.0.1:8080/v1"):
            self.assertIn(must, text)

    def test_candidate_runbooks_are_labeled_unproven(self):
        for name in ("phi-2", "bitnet-b1.58-2b"):
            text = "\n".join(REG.runbook(REG.get(name)))
            self.assertIn("candidate recipe", text)            # never sold as a guarantee

    def test_verified_runbook_carries_no_candidate_warning(self):
        text = "\n".join(REG.runbook(REG.get("sarvam-30b")))
        self.assertNotIn("candidate recipe", text)

    def test_ollama_runbook_is_two_steps(self):
        text = "\n".join(REG.runbook(REG.get("phi-2")))
        self.assertIn("ollama pull phi", text)
        self.assertIn("forge --model phi run", text)


class TestModelsUse(unittest.TestCase):
    """`forge models use` — turnkey provisioning (Ollama path), fully mocked (no network)."""

    def setUp(self):
        import tempfile
        from forge import config as C
        self._orig_path = C.PATH
        C.PATH = os.path.join(tempfile.mkdtemp(), "config.json")   # isolate config writes
        self.C = C

    def tearDown(self):
        self.C.PATH = self._orig_path

    def _use(self, name, hw=None):
        from forge import models_cmd as MC
        return MC._use(hw or {"ram_gb": 8, "arch": "arm64", "os": "Darwin"}, name)

    def test_use_ollama_pulls_writes_config_and_smokes(self):
        from forge import setup as S, models_cmd as MC
        import forge.backends as B
        calls = {}

        def fake_run(cmd, *a, **k):
            calls["pull"] = cmd
            return mock.Mock(returncode=0)

        with mock.patch.object(S, "_ollama_ok", return_value=(True, "")), \
             mock.patch.object(S, "_have_model", return_value=False), \
             mock.patch.object(MC.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(B, "make_backend") as mkb:
            mkb.return_value.chat.return_value = "OK"
            rc = self._use("phi-2")
        self.assertEqual(rc, 0)
        self.assertEqual(calls["pull"], ["ollama", "pull", "phi"])   # pulled the right tag
        cfg = self.C.load()
        self.assertEqual(cfg["engine"], "ollama")
        self.assertEqual(cfg["ladder"], ["phi"])                     # config points at it

    def test_use_ollama_skips_pull_when_present(self):
        from forge import setup as S, models_cmd as MC
        import forge.backends as B
        with mock.patch.object(S, "_ollama_ok", return_value=(True, "")), \
             mock.patch.object(S, "_have_model", return_value=True), \
             mock.patch.object(MC.subprocess, "run") as run, \
             mock.patch.object(B, "make_backend") as mkb:
            mkb.return_value.chat.return_value = "OK"
            rc = self._use("phi-2")
        self.assertEqual(rc, 0)
        run.assert_not_called()                                      # already present → no pull

    def test_use_errors_when_ollama_absent(self):
        from forge import setup as S
        with mock.patch.object(S, "_ollama_ok", return_value=(False, "Ollama is not installed.")):
            self.assertEqual(self._use("phi-2"), 1)                  # honest failure, no config write
        self.assertFalse(os.path.exists(self.C.PATH))

    def test_use_bespoke_runtime_is_honest_not_faked(self):
        # bitnet.cpp is a bespoke runtime forge can't auto-provision — it must hand
        # over the runbook (rc=2) and write NO config, with zero side effects.
        rc = self._use("bitnet-b1.58-2b", hw={"ram_gb": 16, "arch": "arm64", "os": "Darwin"})
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(self.C.PATH))

    def test_use_unknown_name(self):
        self.assertEqual(self._use("nope"), 1)


class TestBenchLift(unittest.TestCase):
    """The catalog's base→full harness-lift column, read from bench results."""

    def _results(self, rows):
        import json as _j
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "results.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(_j.dumps(r) + "\n")
        return p

    def test_lift_is_full_minus_bare_points(self):
        from forge import models_cmd as MC, registry as REG
        # bare 1/2 = 50%, full 2/2 = 100% → +50 pts
        rows = [
            {"model": "openai:sarvam-30b@x", "levers": [], "task": "a", "pass": True},
            {"model": "openai:sarvam-30b@x", "levers": [], "task": "b", "pass": False},
            {"model": "openai:sarvam-30b@x", "levers": sorted(list(_all_levers())), "task": "a", "pass": True},
            {"model": "openai:sarvam-30b@x", "levers": sorted(list(_all_levers())), "task": "b", "pass": True},
        ]
        self.assertEqual(MC.bench_lift(REG.get("sarvam-30b"), self._results(rows)), 50)

    def test_matches_by_ollama_tag(self):
        from forge import models_cmd as MC, registry as REG
        rows = [
            {"model": "phi", "levers": [], "task": "a", "pass": False},
            {"model": "phi", "levers": sorted(list(_all_levers())), "task": "a", "pass": True},
        ]
        # phi-2's ollama_tag is "phi" — a bench run addressed by tag still matches.
        self.assertEqual(MC.bench_lift(REG.get("phi-2"), self._results(rows)), 100)

    def test_no_data_is_none_then_pending(self):
        from forge import models_cmd as MC, registry as REG
        self.assertIsNone(MC.bench_lift(REG.get("sarvam-30b"), self._results([])))
        self.assertEqual(MC._lift(REG.get("sarvam-30b"))[0], "pending")   # no live data, no stored

    def test_missing_file_is_none(self):
        from forge import models_cmd as MC, registry as REG
        self.assertIsNone(MC.bench_lift(REG.get("sarvam-30b"), "/no/such/results.jsonl"))


def _all_levers():
    from forge.agent import ALL_LEVERS
    return ALL_LEVERS


class TestModelsUseLlamacpp(unittest.TestCase):
    """`forge models use` — llama.cpp turnkey path, fully mocked (no server, no power)."""

    def setUp(self):
        import tempfile
        from forge import config as C
        self._orig_path = C.PATH
        C.PATH = os.path.join(tempfile.mkdtemp(), "config.json")
        self.C = C
        self.hw = {"ram_gb": 48, "arch": "arm64", "os": "Darwin"}

    def tearDown(self):
        self.C.PATH = self._orig_path

    def _use(self):
        from forge import models_cmd as MC
        return MC._use(self.hw, "sarvam-30b")

    def test_happy_path_launches_and_writes_matched_ctx_config(self):
        from forge import models_cmd as MC
        with mock.patch.object(MC.shutil, "which", return_value="/usr/bin/llama-server"), \
             mock.patch.object(MC, "_arch_missing", return_value=False), \
             mock.patch.object(MC, "_ensure_gguf", return_value="/models/sarvam.gguf"), \
             mock.patch.object(MC, "_launch_server", return_value=True), \
             mock.patch.object(MC, "_wait_health", return_value=True):
            rc = self._use()
        self.assertEqual(rc, 0)
        cfg = self.C.load()
        self.assertEqual(cfg["engine"], "llamacpp")
        self.assertEqual(cfg["base_url"], "http://127.0.0.1:8080/v1")
        self.assertEqual(cfg["ladder"], ["sarvam-30b"])
        self.assertEqual(cfg["remote_ctx"], 16384)          # ctx matched to the server

    def test_missing_runtime_errors_without_config(self):
        from forge import models_cmd as MC
        with mock.patch.object(MC.shutil, "which", return_value=None):   # no llama-server, no brew
            self.assertEqual(self._use(), 1)
        self.assertFalse(os.path.exists(self.C.PATH))

    def test_arch_absent_is_a_hard_stop(self):
        from forge import models_cmd as MC
        with mock.patch.object(MC.shutil, "which", return_value="/usr/bin/llama-server"), \
             mock.patch.object(MC, "_arch_missing", return_value=True), \
             mock.patch.object(MC, "_launch_server") as launch:
            self.assertEqual(self._use(), 1)               # refuse — build lacks the arch
            launch.assert_not_called()                     # never even tries to serve
        self.assertFalse(os.path.exists(self.C.PATH))

    def test_unhealthy_server_fails_before_config(self):
        from forge import models_cmd as MC
        with mock.patch.object(MC.shutil, "which", return_value="/usr/bin/llama-server"), \
             mock.patch.object(MC, "_arch_missing", return_value=False), \
             mock.patch.object(MC, "_ensure_gguf", return_value="/models/sarvam.gguf"), \
             mock.patch.object(MC, "_launch_server", return_value=True), \
             mock.patch.object(MC, "_wait_health", return_value=False):
            self.assertEqual(self._use(), 1)
        self.assertFalse(os.path.exists(self.C.PATH))

    def test_arch_missing_verdicts(self):
        from forge import models_cmd as MC
        with mock.patch.object(MC, "_libllama_paths", return_value=[]):
            self.assertIsNone(MC._arch_missing("sarvam-moe"))     # can't locate → None (warn)
        with mock.patch.object(MC, "_libllama_paths", return_value=["/x/libllama.dylib"]), \
             mock.patch.object(MC.subprocess, "run", return_value=mock.Mock(stdout="qwen2\nsarvam-moe\n")):
            self.assertFalse(MC._arch_missing("sarvam-moe"))      # present
        with mock.patch.object(MC, "_libllama_paths", return_value=["/x/libllama.dylib"]), \
             mock.patch.object(MC.subprocess, "run", return_value=mock.Mock(stdout="qwen2\nllama\n")):
            self.assertTrue(MC._arch_missing("sarvam-moe"))       # found libs, arch absent


class TestModelsStop(unittest.TestCase):
    def test_stop_kills_a_launched_server(self):
        import tempfile
        from forge import models_cmd as MC
        srv = tempfile.mkdtemp()
        with mock.patch.object(MC, "SERVERS", srv):
            pidf, _ = MC._server_files("sarvam-30b")
            with open(pidf, "w") as f:
                f.write("4242")
            with mock.patch.object(MC.os, "kill") as kill:
                rc = MC._stop("sarvam-30b")
        self.assertEqual(rc, 0)
        kill.assert_called_once_with(4242, MC.signal.SIGTERM)
        self.assertFalse(os.path.exists(pidf))               # pidfile cleaned up

    def test_stop_ollama_is_a_noop(self):
        from forge import models_cmd as MC
        with mock.patch.object(MC.os, "kill") as kill:
            self.assertEqual(MC._stop("phi-2"), 0)           # Ollama manages its own process
            kill.assert_not_called()

    def test_stop_with_no_server(self):
        import tempfile
        from forge import models_cmd as MC
        with mock.patch.object(MC, "SERVERS", tempfile.mkdtemp()):
            self.assertEqual(MC._stop("sarvam-30b"), 0)      # nothing to stop → clean no-op


class TestRemoteCtxConfig(unittest.TestCase):
    """The llama.cpp config is self-sufficient: context_window reads remote_ctx."""

    def setUp(self):
        import tempfile
        from forge import config as C
        self._orig = C.PATH
        C.PATH = os.path.join(tempfile.mkdtemp(), "config.json")
        self.C = C

    def tearDown(self):
        self.C.PATH = self._orig

    def test_context_window_reads_config_remote_ctx(self):
        from forge.backends import OpenAICompatBackend
        self.C.save({"remote_ctx": 16384})
        b = OpenAICompatBackend("m", "http://x/v1")
        with mock.patch.dict(os.environ, {}, clear=True):     # no FORGE_REMOTE_CTX
            self.assertEqual(b.context_window(), 16384)        # config value used

    def test_env_still_overrides_config(self):
        from forge.backends import OpenAICompatBackend
        self.C.save({"remote_ctx": 16384})
        b = OpenAICompatBackend("m", "http://x/v1")
        with mock.patch.dict(os.environ, {"FORGE_REMOTE_CTX": "4096"}, clear=True):
            self.assertEqual(b.context_window(), 4096)         # env wins

    def test_default_when_unset(self):
        from forge.backends import OpenAICompatBackend
        self.C.save({"remote_ctx": 0})
        b = OpenAICompatBackend("m", "http://x/v1")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(b.context_window(), 128000)       # large-window fallback


class TestInstalledMarker(unittest.TestCase):
    """`forge models` now marks which catalog models are actually pulled locally
    (● installed / ○ available) — the install-state it knew but never showed."""

    def test_matcher_by_tag_and_base_name(self):
        from forge.models_cmd import _is_installed
        self.assertTrue(_is_installed("gemma2:9b", {"gemma2:9b"}))
        self.assertTrue(_is_installed("phi-2", {"phi-2:latest"}))      # bare name → tagged
        self.assertFalse(_is_installed("mistral:7b", {"gemma2:9b"}))
        self.assertFalse(_is_installed("qwen2.5:14b", set()))

    def test_installed_tags_empty_off_ollama(self):
        # non-Ollama engine (or absent Ollama) shows no install-state rather than guessing
        from forge.models_cmd import _installed_tags
        self.assertEqual(_installed_tags("llamacpp"), set())


if __name__ == "__main__":
    unittest.main()
