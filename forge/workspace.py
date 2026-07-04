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
            return sorted(set(r.stdout.split() + (u.stdout.split() if u.returncode == 0 else [])))
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


def build_tree(root, cap=180):
    files = _git_files(root)
    if files is None:
        files = _walk_files(root, cap)
    files = [f for f in files if os.path.splitext(f)[1].lower() not in IGNORE_EXT]
    total = len(files)                      # true count (before capping the display)
    shown = sorted(files)[:cap]
    lines, seen = [], set()
    for f in shown:
        parts = f.split("/")
        for d in range(len(parts) - 1):
            prefix = "/".join(parts[:d + 1])
            if prefix not in seen:
                seen.add(prefix)
                lines.append("  " * d + parts[d] + "/")
        lines.append("  " * (len(parts) - 1) + parts[-1])
    truncated = "" if total <= cap else f"\n  … +{total - cap} more files (tree truncated for display)"
    return "\n".join(lines) + truncated, total


EXT_LANG = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".jsx": "JavaScript",
            ".tsx": "TypeScript", ".rs": "Rust", ".go": "Go", ".java": "Java", ".rb": "Ruby",
            ".php": "PHP", ".swift": "Swift", ".c": "C", ".cpp": "C++", ".sh": "Shell"}


def detect_project(root):
    hits = [(label, name) for name, label in PROJECT_MARKERS if os.path.exists(os.path.join(root, name))]
    labels = []
    for label, _ in hits:
        if label not in labels:
            labels.append(label)
    if labels:
        return ", ".join(labels), [name for _, name in hits]
    # fallback: infer language from the most common source extension
    counts = {}
    files = _git_files(root) or _walk_files(root, 200)
    for f in files:
        lang = EXT_LANG.get(os.path.splitext(f)[1].lower())
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    if counts:
        top = max(counts, key=counts.get)
        return f"{top} (inferred)", []
    return "unknown", []


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
    return "ENVIRONMENT\n" + "\n".join(lines)


def context(root, learnings=None):
    """The workspace briefing pinned into the agent's context at session start."""
    tree, n = build_tree(root)
    ptype, markers = detect_project(root)
    parts = [
        environment(root),
        "",
        f"WORKSPACE: {root}",
        f"Project type: {ptype}" + (f" (markers: {', '.join(markers)})" if markers else ""),
        f"\n{n} project files (git-tracked, node_modules excluded). File tree"
        + (f" (first 180 of {n} shown)" if n > 180 else "") + f":\n{tree}",
    ]
    if learnings:
        parts.append("\nWhat the fleet has already learned about this repo:\n" + "\n".join(f"- {f}" for f in learnings))
    parts.append("\nYou already know this layout. Work on the right files directly; only list/read to confirm details. "
                 "For any repo-wide command, use `git ls-files` (these files, node_modules excluded) — never `find . -exec`.")
    return "\n".join(parts)
