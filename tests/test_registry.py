"""Curated model registry: entry shape, honest status, fit math, runbooks."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


if __name__ == "__main__":
    unittest.main()
