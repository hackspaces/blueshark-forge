"""The full downloadable catalog: param parsing, sources, fit-math, cache — mocked (no live net)."""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import catalog as C                                     # noqa: E402


class TestParams(unittest.TestCase):
    def test_name_params(self):
        self.assertEqual(C.name_params("7b"), 7.0)
        self.assertEqual(C.name_params("1.5b"), 1.5)
        self.assertEqual(C.name_params("qwen2.5:0.5b"), 0.5)
        self.assertEqual(C.name_params("8x7b"), 56.0)                # MoE nominal
        self.assertEqual(C.name_params("Meta-Llama-3.1-8B-Instruct-GGUF"), 8.0)  # 8, not 3.1
        self.assertIsNone(C.name_params("bge-m3"))                   # no size → None
        self.assertIsNone(C.name_params("latest"))

    def test_estimates_scale_with_params(self):
        self.assertLess(C.est_size_gb(1), C.est_size_gb(7))
        self.assertLess(C.est_ram_gb(1), C.est_ram_gb(70))
        self.assertGreaterEqual(C.est_ram_gb(0.5), 2)               # a floor


class TestSources(unittest.TestCase):
    def test_ollama_index_scrape(self):
        html = '<a href="/library/llama3.2">x</a> <a href="/library/qwen2.5">y</a> <a href="/other">z</a>'
        self.assertEqual(C.ollama_models(html), ["llama3.2", "qwen2.5"])

    def test_ollama_entries_dedup_by_size(self):
        e = C._ollama_entries_for("llama3.2", tags=["1b", "3b", "latest", "1b"])
        self.assertEqual([x["params_b"] for x in e], [1.0, 3.0])    # sizeless + dup dropped
        self.assertEqual(e[0]["engine"], "ollama")
        self.assertEqual(e[0]["ollama_tag"], "llama3.2:1b")         # turnkey-usable

    def test_ollama_tags_scraped_from_html(self):
        # the registry tags/list 404s — tags come from the HTML tags page.
        html = 'x <a>llama3.2:1b</a> y llama3.2:3b z llama3.2:latest other:9b'
        self.assertEqual(C._ollama_tags("llama3.2", html=html), ["1b", "3b", "latest"])

    def test_hf_junk_is_filtered(self):
        fake = json.dumps([{"id": "good/Qwen2.5-7B-GGUF"},
                           {"id": "spam/Llama-8B-Uncensored-RP-GGUF"},   # junk → dropped
                           {"id": "spam/Model-13B-abliterated"}])         # junk → dropped
        with mock.patch.object(C, "_get", return_value=fake):
            out = C.source_huggingface(limit=5)
        self.assertEqual([e["params_b"] for e in out], [7.0])            # only the clean one

    def test_hf_mlx_repo_gets_the_mlx_engine(self):
        fake = json.dumps([{"id": "lmstudio-community/Gemma-2-9B-MLX-4bit"}])
        with mock.patch.object(C, "_get", return_value=fake):
            out = C.source_huggingface(limit=3)
        self.assertEqual(out[0]["engine"], "mlx")                        # run it with mlx, not llama.cpp

    def test_hf_query_parses_ids_and_tags_engine(self):
        fake = json.dumps([{"id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"},
                           {"id": "unsloth/Qwen2.5-14B-GGUF"},
                           {"id": "someone/not-a-model"}])           # no size → dropped
        with mock.patch.object(C, "_get", return_value=fake):
            out = C.source_huggingface(limit=5)
        self.assertEqual({e["engine"] for e in out}, {"llamacpp"})
        self.assertIn(8.0, [e["params_b"] for e in out])
        self.assertIn(14.0, [e["params_b"] for e in out])
        self.assertTrue(all(e["params_b"] for e in out))            # sizeless dropped

    def test_mlx_source_tags_engine(self):
        fake = json.dumps([{"id": "mlx-community/Llama-3.2-3B-Instruct-4bit"}])
        with mock.patch.object(C, "_get", return_value=fake):
            out = C.source_mlx(limit=3)
        self.assertEqual(out[0]["engine"], "mlx")
        self.assertEqual(out[0]["params_b"], 3.0)


class TestApplicableSources(unittest.TestCase):
    def test_mlx_only_on_apple_silicon(self):
        self.assertIn("mlx", C._applicable_sources({"os": "Darwin", "arch": "arm64"}))
        self.assertNotIn("mlx", C._applicable_sources({"os": "Windows", "arch": "AMD64"}))
        self.assertIn("ollama", C._applicable_sources({"os": "Linux", "arch": "x86_64"}))


class TestFetchAndCache(unittest.TestCase):
    def setUp(self):
        self._orig = C.CACHE
        C.CACHE = os.path.join(tempfile.mkdtemp(), "catalog.json")

    def tearDown(self):
        C.CACHE = self._orig

    def test_fetch_aggregates_sources_and_caches(self):
        with mock.patch.dict(C.SOURCES, {
            "ollama": lambda on_progress=None: [C._entry("a:7b", 7, "ollama", "ollama")],
            "huggingface": lambda: [C._entry("x/13B", 13, "llamacpp", "huggingface")],
        }, clear=True):
            entries = C.fetch_catalog(["ollama", "huggingface"])
        self.assertEqual(sorted(e["params_b"] for e in entries), [7.0, 13.0])  # aggregated + sorted
        self.assertTrue(os.path.exists(C.CACHE))                     # cached

    def test_fetch_dedups_variants_preferring_the_turnkey_source(self):
        # the same model+size from ollama and HF (as a quant variant) → one entry, ollama wins.
        with mock.patch.dict(C.SOURCES, {
            "ollama": lambda on_progress=None: [C._entry("gemma2:9b", 9, "ollama", "ollama")],
            "huggingface": lambda: [C._entry("lmstudio-community/gemma-2-9b-it-GGUF", 9, "llamacpp", "huggingface"),
                                    C._entry("lmstudio-community/gemma-2-9b-it-MLX-4bit", 9, "mlx", "huggingface")],
        }, clear=True):
            entries = C.fetch_catalog(["ollama", "huggingface"])
        self.assertEqual(len(entries), 1)                            # 3 variants → 1
        self.assertEqual(entries[0]["source"], "ollama")            # turnkey source kept

    def test_base_normalizes_for_dedup(self):
        self.assertEqual(C._base("gemma2:9b"), C._base("lmstudio-community/gemma-2-9b-it-GGUF"))
        self.assertNotEqual(C._base("llama3.2:1b"), C._base("llama3.1:8b"))   # versions preserved

    def test_a_dead_source_never_breaks_the_scan(self):
        def boom():
            raise RuntimeError("network down")
        with mock.patch.dict(C.SOURCES, {
            "ollama": lambda on_progress=None: [C._entry("a:1b", 1, "ollama", "ollama")],
            "huggingface": boom,
        }, clear=True):
            entries = C.fetch_catalog(["ollama", "huggingface"])
        self.assertEqual([e["params_b"] for e in entries], [1.0])    # ollama survived

    def test_load_uses_fresh_cache_without_fetching(self):
        with open(C.CACHE, "w") as f:
            json.dump({"fetched": time.time(), "entries": [C._entry("z:3b", 3, "ollama", "ollama")]}, f)
        with mock.patch.object(C, "fetch_catalog", side_effect=AssertionError("should not fetch")):
            entries, cached = C.load_catalog({"os": "Linux", "arch": "x86_64"})
        self.assertTrue(cached)
        self.assertEqual(entries[0]["params_b"], 3.0)

    def test_load_refetches_when_stale(self):
        with open(C.CACHE, "w") as f:
            json.dump({"fetched": time.time() - C._MAX_AGE - 1, "entries": [C._entry("old:1b", 1, "ollama", "ollama")]}, f)
        with mock.patch.object(C, "fetch_catalog", return_value=[C._entry("new:2b", 2, "ollama", "ollama")]) as fc:
            entries, cached = C.load_catalog({"os": "Linux", "arch": "x86_64"})
        fc.assert_called_once()
        self.assertFalse(cached)
        self.assertEqual(entries[0]["params_b"], 2.0)


class TestCatalogEntriesWorkWithFitMath(unittest.TestCase):
    def test_registry_runs_accepts_a_catalog_entry(self):
        from forge import registry as REG
        e = C._entry("big/70B", 70, "llamacpp", "huggingface")
        self.assertEqual(REG.runs(e, 8, True)[0], "won't fit")
        self.assertEqual(REG.runs(e, 64, True)[0], "runs well")


if __name__ == "__main__":
    unittest.main()
