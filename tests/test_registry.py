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

    def test_use_non_ollama_is_honest_not_faked(self):
        # sarvam is llamacpp — turnkey not built yet; must hand over the runbook, rc=2, no config.
        rc = self._use("sarvam-30b", hw={"ram_gb": 48, "arch": "arm64", "os": "Darwin"})
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(self.C.PATH))

    def test_use_unknown_name(self):
        self.assertEqual(self._use("nope"), 1)


if __name__ == "__main__":
    unittest.main()
