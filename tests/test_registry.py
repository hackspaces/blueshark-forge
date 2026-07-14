"""Curated model registry: entry shape, honest status, fit math, runbooks."""
import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
