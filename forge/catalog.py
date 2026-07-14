"""The full downloadable catalog — run the fit-math on every model people can grab.

`forge models` shows a fast, offline CURATED spread. `forge models --all` fetches the
real libraries people download from, sizes each model, caches it, and runs the SAME
fit/speed math on all of it — so you see everything THIS machine can run, across
sources, not a hand-picked few.

Sources (each maps a place-you-download to the forge engine that runs it):
  · ollama       — the Ollama library (~236 models), one-command via `forge models use`
  · huggingface  — top GGUF repos (the pool LM Studio / llama.cpp / Jan all draw from) → llama.cpp
  · mlx          — mlx-community (Apple-Silicon native) → mlx

Honesty: sizes are ESTIMATES from each model's parameter count (parsed from its name/
tag) at a 4-bit quant — a fit estimate, not a promise. Ollama entries are turnkey;
HF/MLX entries are "runnable, here's the recipe". Network on first fetch; cached to
~/.forge/catalog.json (refresh with `--refresh`). stdlib only.
"""
import concurrent.futures
import json
import os
import re
import time
import urllib.request

CACHE = os.path.expanduser("~/.forge/catalog.json")
_MAX_AGE = 7 * 24 * 3600            # a week — libraries change slowly
_OLLAMA_LIB = "https://ollama.com/library"
_OLLAMA_REG = "https://registry.ollama.ai/v2/library"
_HF_API = "https://huggingface.co/api/models"

# a param size inside a name/tag: "7b" / "1.5b" / "8x7b" (MoE) / "70b". \b so "m3" ≠ size.
_PARAMS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?:x(\d+))?b\b", re.I)
# keep the catalog PG — drop uncensored/roleplay/etc. finetunes that dominate raw HF sorts.
_JUNK = re.compile(r"uncensor|abliterat|heretic|roleplay|\brp\b|nsfw|erotic|horny|waifu|smut|hentai|deckard", re.I)


def _ok(name):
    return not _JUNK.search(name or "")


# strip org + quant/format/variant tokens so the same model dedups across sources/quants.
_STRIP = re.compile(r"[-_.:](gguf|mlx|i?q\d[_a-z0-9]*|\d+bit|fp16|bf16|f16|instruct|it|chat|base|text|"
                    r"\d+(?:\.\d+)?(?:x\d+)?b)\b", re.I)


def _base(name):
    n = name.split("/")[-1]
    prev = None
    while prev != n:
        prev, n = n, _STRIP.sub("", n)
    return re.sub(r"[^a-z0-9]", "", n.lower())


def name_params(s):
    """Billions of params implied by a model name or tag, or None. Takes the largest
    match ('Llama-3.1-8B' → 8, not 3.1; '8x7b' → 56 MoE nominal)."""
    best = None
    for m in _PARAMS.finditer(s):
        base = float(m.group(1))
        val = base * int(m.group(2)) if m.group(2) else base
        if best is None or val > best:
            best = val
    return best


def est_size_gb(params_b):
    """Rough 4-bit GGUF size — ~0.6 GB per billion params."""
    return round(params_b * 0.6, 1)


def est_ram_gb(params_b):
    """Recommended total RAM (weights + OS headroom + KV), an estimate."""
    return max(2, round(params_b * 0.65) + 3)


def _entry(name, params_b, engine, source, tag=None):
    """A catalog entry shaped like a registry entry, so registry.runs/fits work on it."""
    e = {"name": name, "params_b": params_b, "engine": engine, "source": source,
         "weights": {"size_gb": est_size_gb(params_b)}, "ram_gb_needed": est_ram_gb(params_b),
         "status": "catalog", "notes": ""}
    if tag:
        e["ollama_tag"] = tag
    return e


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "forge-catalog"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ---- source: Ollama library ------------------------------------------------

def ollama_models(html=None):
    html = html if html is not None else _get(_OLLAMA_LIB)
    return sorted(set(re.findall(r'href="/library/([a-z0-9][a-z0-9._-]*)"', html)))


def _ollama_tags(name, html=None):
    """A library model's tags — scraped from its tags page. (The registry's Docker-v2
    tags/list 404s; the HTML page is the reliable source.)"""
    try:
        html = html if html is not None else _get(f"{_OLLAMA_LIB.rsplit('/', 1)[0]}/library/{name}/tags")
    except Exception:
        return []
    return sorted(set(re.findall(rf"{re.escape(name)}:([a-z0-9][a-z0-9._-]*)", html)))


def _ollama_entries_for(name, tags=None):
    out, seen = [], set()
    for tag in (tags if tags is not None else _ollama_tags(name)):
        p = name_params(tag)
        if p is None or p in seen:
            continue
        seen.add(p)
        out.append(_entry(f"{name}:{tag}", p, "ollama", "ollama", tag=f"{name}:{tag}"))
    return out


def source_ollama(on_progress=None):
    models = ollama_models()
    entries, done = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(_ollama_entries_for, models):
            entries.extend(res)
            done += 1
            if on_progress:
                on_progress("ollama", done, len(models))
    return entries


# ---- sources: HuggingFace (GGUF) and MLX -----------------------------------

def _hf_query(params_qs, engine, source, limit):
    """Top HF repos matching a query → catalog entries, params parsed from the repo id.
    Query REPUTABLE, curated authors (not raw sort-by-downloads, which surfaces junk):
    lmstudio-community is the pool LM Studio/llama.cpp users trust; mlx-community feeds MLX."""
    try:
        data = json.loads(_get(f"{_HF_API}?{params_qs}&sort=downloads&direction=-1&limit={limit}"))
    except Exception:
        return []
    out, seen = [], set()
    for m in data:
        rid = m.get("id") or m.get("modelId") or ""
        tail = rid.split("/")[-1]
        p = name_params(tail)
        key = (tail.lower(), p)
        if p is None or p > 200 or key in seen or not _ok(rid):   # sizeless / absurd / junk → drop
            continue
        seen.add(key)
        eng = "mlx" if "mlx" in rid.lower() else engine           # run it with the right engine
        out.append(_entry(rid, p, eng, source))
    return out


def source_huggingface(limit=200):
    # lmstudio-community = curated GGUF quants of real models (what LM Studio surfaces).
    return _hf_query("author=lmstudio-community", "llamacpp", "huggingface", limit)


def source_mlx(limit=120):
    return _hf_query("author=mlx-community", "mlx", "mlx", limit)


SOURCES = {
    "ollama": source_ollama,
    "huggingface": source_huggingface,
    "mlx": source_mlx,
}


def _applicable_sources(hw):
    """Which sources make sense for this machine — MLX only on Apple Silicon."""
    names = ["ollama", "huggingface"]
    if (hw or {}).get("os") == "Darwin" and (hw or {}).get("arch") == "arm64":
        names.append("mlx")
    return names


# ---- fetch + cache ---------------------------------------------------------

_SRC_PREF = {"ollama": 0, "mlx": 1, "huggingface": 2}   # prefer the turnkey source on a tie


def fetch_catalog(sources, on_progress=None):
    raw = []
    for name in sources:
        fn = SOURCES.get(name)
        if not fn:
            continue
        try:
            raw.extend(fn(on_progress=on_progress) if name == "ollama" else fn())
        except Exception:
            continue                                  # a dead source never breaks the scan
    # dedup: one entry per (model, size) — collapse quant variants + cross-source dups,
    # keeping the most turnkey source (ollama > mlx > huggingface).
    best = {}
    for e in raw:
        key = (_base(e["name"]), round(e.get("params_b") or 0, 1))
        cur = best.get(key)
        if cur is None or _SRC_PREF.get(e.get("source"), 9) < _SRC_PREF.get(cur.get("source"), 9):
            best[key] = e
    entries = sorted(best.values(), key=lambda e: e.get("params_b") or 0)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"fetched": time.time(), "sources": sources, "entries": entries}, f)
    os.replace(tmp, CACHE)
    return entries


def load_catalog(hw=None, refresh=False, max_age=_MAX_AGE, on_progress=None):
    """Cached catalog if fresh, else fetch the sources applicable to `hw`.
    Returns (entries, from_cache)."""
    if not refresh:
        try:
            with open(CACHE) as f:
                c = json.load(f)
            if time.time() - c.get("fetched", 0) < max_age and c.get("entries"):
                return c["entries"], True
        except (OSError, ValueError):
            pass
    return fetch_catalog(_applicable_sources(hw), on_progress), False
