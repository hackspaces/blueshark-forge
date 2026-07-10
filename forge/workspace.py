"""Workspace awareness — forge orients in the project before it acts.

On session start it builds a map of the working directory (gitignore-aware),
detects the project type, notes key files, and pulls in any facts the fleet has
already learned about this repo. All of it is pinned into the agent's context so
it knows the layout without groping — you can say "read this" or "fix the auth
bug" and it already knows where things are."""
import os
import platform
import shutil
import subprocess
import time

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
               "build", ".next", "target", ".cache", ".pytest_cache", ".mypy_cache",
               "vendor", ".idea", ".vscode", "coverage", ".turbo"}
IGNORE_EXT = {".pyc", ".pyo", ".so", ".o", ".class", ".lock", ".log", ".map",
              ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz",
              ".woff", ".woff2", ".ttf", ".mp4", ".mov"}

PROJECT_MARKERS = [
    ("package.json", "Node.js"), ("pyproject.toml", "Python"), ("setup.py", "Python"),
    ("requirements.txt", "Python"), ("Cargo.toml", "Rust"), ("go.mod", "Go"),
    ("pom.xml", "Java/Maven"), ("build.gradle", "Java/Gradle"), ("Gemfile", "Ruby"),
    ("composer.json", "PHP"), ("Makefile", "Make"),
]


def _git_files(root):
    try:
        r = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            # tracked + untracked-but-not-ignored
            u = subprocess.run(["git", "-C", root, "ls-files", "--others", "--exclude-standard"],
                               capture_output=True, text=True, timeout=10)
            # splitlines(), NOT split(): a path with a space must stay one token.
            return sorted(set(r.stdout.splitlines() + (u.stdout.splitlines() if u.returncode == 0 else [])))
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _walk_files(root, cap):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in IGNORE_EXT or fn.startswith("."):
                continue
            out.append(os.path.relpath(os.path.join(dp, fn), root))
            if len(out) > cap * 3:
                return out
    return sorted(out)


def _source_files(root, cap):
    """The raw candidate file list (git-tracked, else walked), IGNORE_EXT-filtered.
    Shared by build_tree and the symbol index so both see the same universe."""
    files = _git_files(root)
    if files is None:
        files = _walk_files(root, cap)
    return [f for f in files if os.path.splitext(f)[1].lower() not in IGNORE_EXT]


def _recency_scores(root, n=100):
    """path -> recency rank (0 = touched by the newest commit) from one
    `git log --name-only`. Empty dict when this isn't a git repo — callers then
    fall back to manifest-proximity + depth, so no-git degrades gracefully."""
    try:
        r = subprocess.run(["git", "-C", root, "log", "--name-only", "--pretty=format:", "-n", str(n)],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return {}
    except (OSError, subprocess.SubprocessError):
        return {}
    scores, rank = {}, 0
    for line in r.stdout.splitlines():
        p = line.strip()
        if p and p not in scores:
            scores[p] = rank
            rank += 1
    return scores


_MANIFEST_NAMES = {name for name, _ in PROJECT_MARKERS}


def _rank(files, root):
    """Order files by importance: (git recency, near a manifest, then deeper
    files first, then name) — so the top-K shown when truncated is the useful
    sample, not the lexically-first one."""
    scores = _recency_scores(root)
    manifest_dirs = {os.path.dirname(f) for f in files if os.path.basename(f) in _MANIFEST_NAMES}
    unseen = len(scores) + len(files) + 1   # files no commit touched rank after all seen files
    def key(f):
        near = 0 if os.path.dirname(f) in manifest_dirs else 1
        return (scores.get(f, unseen), near, -f.count("/"), f)
    return sorted(files, key=key)


def _rollup(files):
    """One line per top-level directory with its true file count — so every
    directory stays visible in the briefing even when the file tree is truncated."""
    counts = {}
    for f in files:
        top = f.split("/", 1)[0] if "/" in f else "(root)"
        counts[top] = counts.get(top, 0) + 1
    return [f"  {d}{'' if d == '(root)' else '/'} — {counts[d]} file(s)" for d in sorted(counts)]


def build_tree(root, cap=180):
    files = _source_files(root, cap)
    total = len(files)                      # true count (before capping the display)
    shown = sorted(_rank(files, root)[:cap])   # SELECT by rank, render alphabetically
    lines, seen = [], set()
    for f in shown:
        parts = f.split("/")
        for d in range(len(parts) - 1):
            prefix = "/".join(parts[:d + 1])
            if prefix not in seen:
                seen.add(prefix)
                lines.append("  " * d + parts[d] + "/")
        lines.append("  " * (len(parts) - 1) + parts[-1])
    out = "\n".join(lines)
    if total > cap:
        out += (f"\n  … +{total - cap} more files not shown (tree is the top {cap} by recency; "
                "every directory is summarized below)")
        out += "\n\nAll directories (file counts):\n" + "\n".join(_rollup(files))
    return out, total


def detect_project(root):
    """Project type from HARD EVIDENCE only — a manifest at the root (go.mod,
    package.json, pyproject.toml, ...). No markers → no claim: guessing a
    language from stray source files mislabels non-project dirs (a home
    directory with one .go file is not 'a Go project')."""
    hits = [(label, name) for name, label in PROJECT_MARKERS if os.path.exists(os.path.join(root, name))]
    labels = []
    for label, _ in hits:
        if label not in labels:
            labels.append(label)
    if labels:
        return ", ".join(labels), [name for _, name in hits]
    return "", []


def _ver(tool, arg="--version"):
    try:
        r = subprocess.run([tool, arg], capture_output=True, text=True, timeout=5)
        return (r.stdout + r.stderr).strip().splitlines()[0][:40]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def environment(cwd):
    """Base computer awareness: OS, shell, tools, git state — so the agent uses
    the right commands for THIS machine."""
    lines = [
        f"OS: {platform.platform()}  ({platform.machine()})",
        f"Shell: {os.environ.get('SHELL', 'unknown')}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M %Z')}",
        f"CWD: {cwd}",
    ]
    # available tools (presence, with versions for the common runtimes)
    present = []
    for t in ("git", "node", "npm", "pnpm", "yarn", "python3", "pytest", "cargo", "go", "rg", "docker", "make"):
        if shutil.which(t):
            present.append(t)
    lines.append("Tools available: " + ", ".join(present))
    # machine intelligence (from `forge setup`), so the model knows its constraints
    try:
        from . import config as _cfg
        m = _cfg.get("machine", {})
        if m:
            lines.insert(1, f"Machine: {m.get('chip','')} · {m.get('ram_gb','?')}GB RAM · {m.get('cores','?')} cores")
    except Exception:
        pass
    for t in ("node", "python3"):
        v = _ver(t)
        if v:
            lines.append(f"  {t}: {v}")
    # git state
    if shutil.which("git"):
        try:
            r = subprocess.run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                branch = r.stdout.strip()
                st = subprocess.run(["git", "-C", cwd, "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=5)
                dirty = len(st.stdout.split("\n")) - 1 if st.stdout.strip() else 0
                lines.append(f"Git: on branch {branch}" + (f", {dirty} uncommitted change(s)" if dirty else ", clean"))
        except (OSError, subprocess.SubprocessError):
            pass
    # OS-userland quirks: macOS ships BSD tools, not GNU — surface the differences so the
    # model uses the right command for THIS box instead of a Linux habit that fails here.
    if platform.system() == "Darwin":
        note = ("Userland: BSD (macOS), not GNU — `timeout`→use `gtimeout`, `sed -i` needs a "
                "backup-suffix arg (`sed -i '' …`), no `readlink -f` / `date -d`.")
        note += (" GNU coreutils installed (g-prefixed: gsed, gtimeout, gdate…)."
                 if (shutil.which("gtimeout") or shutil.which("gsed"))
                 else " GNU coreutils NOT installed — use the BSD forms above.")
        lines.append(note)
    return "ENVIRONMENT\n" + "\n".join(lines)


SMALL_CTX = 8192          # below this the briefing drops to env + rollup tree, no symbols

INSTRUCTION_FILES = ("FORGE.md", "AGENTS.md", "CLAUDE.md")  # first-found wins
INSTRUCTIONS_CAP = 3000    # chars of user-authored instructions pinned into the briefing
LEARN_RENDER_CAP = 12      # top-N learnings shown (already verified-first from fleet.learnings)


def _instructions(root, cap=INSTRUCTIONS_CAP):
    """The first-found per-repo instructions file at the repo root — FORGE.md,
    else AGENTS.md, else CLAUDE.md — capped to `cap` chars. Returns (name, text)
    or (None, None). These are USER-authored rules and outrank anything the fleet
    merely learned, so context() pins them above the learnings section."""
    for name in INSTRUCTION_FILES:
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read(cap + 1)
        except OSError:
            continue
        text = text.strip()
        if not text:
            continue
        if len(text) > cap:
            text = text[:cap].rstrip() + "\n…(instructions truncated)"
        return name, text
    return None, None


def _render_learning(x):
    """Render one learning for the briefing. A LEARN v2 record (dict) gets a ✓ when
    the harness verified it; a bare string (legacy / caller-supplied) prints plain."""
    if isinstance(x, dict):
        mark = "✓ " if x.get("verified") else ""
        return f"- {mark}{x.get('fact', '')}"
    return f"- {x}"


def _key_symbols(root, limit=40):
    """The top ~`limit` defined symbols, recency-first, for large-ctx briefings."""
    try:
        from . import index
        syms = index.refresh(root)
    except Exception:
        return []
    scores = _recency_scores(root)
    unseen = len(scores) + len(syms) + 1
    syms.sort(key=lambda s: (scores.get(s["path"], unseen), s.get("lineno", 0)))
    return syms[:limit]


def context(root, learnings=None, budget=None):
    """The workspace briefing pinned into the agent's context at session start.

    `budget` is the model's effective context window (tokens). Small windows get a
    compact env + directory-rollup tree; large windows get the ranked file tree
    plus a "Key symbols" map from the symbol index."""
    small = budget is not None and budget < SMALL_CTX
    cap = 25 if small else 180
    tree, n = build_tree(root, cap=cap)
    ptype, markers = detect_project(root)
    parts = [
        environment(root),
        "",
        f"WORKSPACE: {root}",
    ]
    if ptype:
        parts.append(f"Project type: {ptype} (markers: {', '.join(markers)})")
    shown_note = f" (top {cap} of {n} by recency; every dir in the rollup below)" if n > cap else ""
    parts.append(f"\n{n} project files (git-tracked, node_modules excluded). File tree{shown_note}:\n{tree}")
    name, instr = _instructions(root)
    if instr:
        parts.append(f"\nPROJECT INSTRUCTIONS (user-authored — follow these):\n{instr}")
    if learnings:
        parts.append("\nWhat the fleet has already learned about this repo:\n"
                     + "\n".join(_render_learning(x) for x in learnings[:LEARN_RENDER_CAP]))
    if not small:
        syms = _key_symbols(root)
        if syms:
            parts.append("\nKey symbols (name — file:line · signature):\n"
                         + "\n".join(f"- {s['name']} — {s['path']}:{s['lineno']}  {s['signature']}" for s in syms))
    parts.append("\nYou already know this layout. Work on the right files directly; only list/read to confirm details. "
                 "For a big file, read_file {path, outline:true} maps its defs/classes before you read exact ranges. "
                 "For any repo-wide command, use `git ls-files` (these files, node_modules excluded) — never `find . -exec`.")
    return "\n".join(parts)
