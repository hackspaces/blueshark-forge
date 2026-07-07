"""Symbol index — a persistent, incremental map of where things are DEFINED.

stdlib `ast` for Python; compiled-regex extractors for js/ts/go/rs (no parser
dependency, ever). Persisted as one JSONL line per file at
``~/.forge/index/<cwd-slug>.jsonl`` keyed by (path, mtime, size) so ``refresh()``
re-parses only the files that actually changed. Feeds the workspace briefing's
"Key symbols" section and read_file's outline mode — so a small model reads a
symbol map instead of groping through a 2000-line file.

Zero third-party imports. Everything degrades to an empty list on error; a broken
file never crashes the harness."""
import ast
import json
import os
import re

FORGE_HOME = os.path.expanduser("~/.forge")
INDEX_DIR = os.path.join(FORGE_HOME, "index")

_PY_EXT = {".py", ".pyi"}

# Regex extractors for languages we index without a real parser. Each entry maps
# an extension to a list of (kind, compiled-regex) — the regex must expose a
# named group `name`. First matching spec per line wins; the whole (stripped)
# line becomes the signature. Deliberately conservative: better to miss a symbol
# than to hallucinate one into a small model's context.
_JS_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
_RE_SPECS = {
    "js": [
        ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)")),
    ],
    ".go": [
        ("function", re.compile(r"^func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_]\w*)")),
        ("type", re.compile(r"^type\s+(?P<name>[A-Za-z_]\w*)\s+(?:struct|interface)\b")),
    ],
    ".rs": [
        ("function", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(?P<name>[A-Za-z_]\w*)")),
        ("type", re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(?P<name>[A-Za-z_]\w*)")),
    ],
}


def _specs_for(ext):
    if ext in _JS_EXT:
        return _RE_SPECS["js"]
    return _RE_SPECS.get(ext)


def supported(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in _PY_EXT or _specs_for(ext) is not None


def _unparse(node):
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _func_sig(node):
    aw = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    args = _unparse(node.args)
    ret = f" -> {_unparse(node.returns)}" if node.returns is not None else ""
    return f"{aw}def {node.name}({args}){ret}"


def _class_sig(node):
    bases = [b for b in (_unparse(x) for x in node.bases) if b]
    return f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")


def _py_symbols(text):
    """Top-level defs/classes plus one level of methods, via ast (syntax errors
    skipped). lineno is the `def`/`class` line (not the decorator) so the model
    can read the body with a precise offset."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({"name": node.name, "kind": "function",
                        "lineno": node.lineno, "signature": _func_sig(node)})
        elif isinstance(node, ast.ClassDef):
            out.append({"name": node.name, "kind": "class",
                        "lineno": node.lineno, "signature": _class_sig(node)})
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append({"name": f"{node.name}.{sub.name}", "kind": "method",
                                "lineno": sub.lineno, "signature": _func_sig(sub)})
    return out


def _regex_symbols(text, specs):
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if len(line) > 400:          # skip pathological minified lines
            continue
        for kind, pat in specs:
            m = pat.match(line)
            if m:
                out.append({"name": m.group("name"), "kind": kind,
                            "lineno": i, "signature": line.strip()[:200]})
                break
    return out


def extract_symbols(path, text=None):
    """Return [{name, kind, lineno, signature}, ...] for one file (empty on any
    error or unsupported language)."""
    ext = os.path.splitext(path)[1].lower()
    specs = None if ext in _PY_EXT else _specs_for(ext)
    if ext not in _PY_EXT and specs is None:
        return []
    if text is None:
        try:
            with open(path, errors="replace") as f:
                text = f.read()
        except OSError:
            return []
    if ext in _PY_EXT:
        return _py_symbols(text)
    return _regex_symbols(text, specs)


# ---- persistence -------------------------------------------------------------
def index_path(cwd, index_dir=None):
    d = index_dir or os.environ.get("FORGE_INDEX_DIR") or INDEX_DIR
    slug = re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(cwd)).strip("-") or "root"
    return os.path.join(d, slug + ".jsonl")


def _load(path):
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out[d["path"]] = {"mtime": d.get("mtime"), "size": d.get("size"),
                                  "symbols": d.get("symbols", [])}
    except (OSError, ValueError, KeyError):
        return {}
    return out


def _save(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for rel in sorted(records):
            rec = records[rel]
            f.write(json.dumps({"path": rel, "mtime": rec["mtime"],
                                "size": rec["size"], "symbols": rec["symbols"]}) + "\n")
    os.replace(tmp, path)


def refresh(root, files=None, index_dir=None):
    """Bring the on-disk index up to date and return a flat list of symbol dicts
    (each with a `path` relative to `root`). STATS every candidate file but only
    re-extracts the ones whose (mtime, size) changed since the last run."""
    if files is None:
        from . import workspace
        files = workspace._source_files(root, 5000)
    path = index_path(root, index_dir)
    cache = _load(path)
    fresh = {}
    out = []
    for rel in files:
        if not supported(rel):
            continue
        ab = os.path.join(root, rel)
        try:
            st = os.stat(ab)
        except OSError:
            continue
        rec = cache.get(rel)
        if rec and rec.get("mtime") == st.st_mtime and rec.get("size") == st.st_size:
            syms = rec["symbols"]
        else:
            syms = extract_symbols(ab)
        fresh[rel] = {"mtime": st.st_mtime, "size": st.st_size, "symbols": syms}
        for s in syms:
            d = dict(s)
            d["path"] = rel
            out.append(d)
    if fresh != cache:
        try:
            _save(path, fresh)
        except OSError:
            pass
    return out
