"""Tools + the constrained action schema that gates them.

The schema is the contract every model output is forced to match — a small model
cannot emit a malformed or hallucinated call. It also carries the agent's living
PLAN, which keeps long-horizon work coherent (the single biggest quality lever for
weaker models: hold the plan in the harness, not in the model's head)."""
import os
import shlex
import shutil
import subprocess

BASH_TIMEOUT = int(os.environ.get("FORGE_BASH_TIMEOUT", "60"))  # fail fast, don't hang for minutes


def _which(t):
    return shutil.which(t)


def _q(s):
    return shlex.quote(str(s))

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "items": {"type": "string"},
            "description": "your current todo list; each item prefixed [ ] todo, [~] doing, [x] done. Update it as you progress.",
        },
        "thought": {"type": "string", "description": "brief reasoning, one line"},
        "action": {
            "type": "string",
            "enum": ["bash", "read_file", "write_file", "edit_file", "list_files", "grep", "glob", "fleet_send", "say"],
        },
        "command": {"type": "string", "description": "shell command (bash)"},
        "path": {"type": "string", "description": "file path (read/write/edit/list)"},
        "content": {"type": "string", "description": "full file content (write_file)"},
        "old": {"type": "string", "description": "exact text to replace (edit_file)"},
        "new": {"type": "string", "description": "replacement text (edit_file)"},
        "pattern": {"type": "string", "description": "search pattern (grep = regex in file contents; glob = filename pattern like **/*.py)"},
        "offset": {"type": "integer", "description": "start line for read_file (1-based)"},
        "limit": {"type": "integer", "description": "max lines for read_file"},
        "target": {"type": "string", "description": "which session to message (fleet_send): its project name, dir, or id"},
        "message": {"type": "string", "description": "the message text (say, or fleet_send)"},
    },
    "required": ["thought", "action"],
}

TOOL_HELP = """Each turn, output ONE JSON action. Maintain a `plan` (todo list) and keep it updated as you work.
Actions:
  bash        {command}            run a shell command, see its output
  read_file   {path}              read a file
  write_file  {path, content}     create/overwrite a file with full content
  edit_file   {path, old, new}    surgically replace an exact snippet (preferred for changes)
  list_files  {path?}             list a directory
  grep        {pattern, path?}    search file CONTENTS by regex (ripgrep; structured, fast)
  glob        {pattern}           find files by name (e.g. **/*.py); use this over `find`
  fleet_send  {target, message}   message another session — forge or Claude Code (it receives it mid-work)
  say         {message}           talk to the user (ends your turn)"""


def _fuzzy_replace(text, old, new):
    """Match `old` ignoring per-line leading/trailing whitespace, so a model that
    gets indentation slightly wrong can still edit. Only acts on a UNIQUE match."""
    tlines = text.split("\n")
    olines = [ln.strip() for ln in old.strip("\n").split("\n")]
    if not olines or not any(olines):
        return text, False, "empty"
    hits = []
    for i in range(len(tlines) - len(olines) + 1):
        if [tlines[i + j].strip() for j in range(len(olines))] == olines:
            hits.append(i)
    if len(hits) != 1:
        return text, False, f"{len(hits)} fuzzy matches"
    i = hits[0]
    result = tlines[:i] + new.split("\n") + tlines[i + len(olines):]
    return "\n".join(result), True, "fuzzy"


MAX_OUTPUT = 12000  # chars kept inline; larger output is saved to a file with a preview
_OVERFLOW_DIR = os.path.expanduser("~/.forge/output")


def _maybe_offload(text, label):
    """Keep big outputs out of context: save the full text to a file and return a
    preview + path the model can grep/read, instead of dumping (or silently truncating)."""
    if len(text) <= MAX_OUTPUT:
        return text
    os.makedirs(_OVERFLOW_DIR, exist_ok=True)
    import hashlib
    fn = os.path.join(_OVERFLOW_DIR, f"{label}-{hashlib.md5(text.encode()).hexdigest()[:8]}.txt")
    with open(fn, "w") as f:
        f.write(text)
    head = text[:MAX_OUTPUT]
    return (f"{head}\n\n[... output truncated: {len(text)} chars total. Full output saved to {fn} — "
            f"read a range or grep it for what you need.]")


def _run(cmd, cwd, timeout=None):
    timeout = timeout or BASH_TIMEOUT
    try:
        p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout + p.stderr).strip()
        ok = p.returncode == 0
        body = out if out else f"(no output, exit {p.returncode})"
        return _maybe_offload(body, "bash"), ok
    except subprocess.TimeoutExpired:
        return (f"(timed out after {timeout}s — the command was too slow. Scope it down: exclude node_modules/.git, "
                "or use `git ls-files` (lists only real project files), `rg`, or a narrower path instead of scanning "
                "everything with `find . -exec`.)", False)


def _resolve(cwd, path):
    """Resolve a file path and CONFINE it to the workspace. Returns the absolute
    path, or None if it escapes (absolute path outside cwd, or ../ traversal).
    This keeps the file tools from writing/reading anywhere on the machine."""
    if not path:
        return None
    full = os.path.realpath(os.path.join(cwd, path))
    root = os.path.realpath(cwd)
    if full == root:
        return full
    try:
        if os.path.commonpath([full, root]) != root:
            return None
    except ValueError:  # different drives (Windows) etc.
        return None
    return full


def execute(action, cwd):
    """Run one action. Returns (observation, ok)."""
    a = action.get("action")
    try:
        if a == "bash":
            cmd = (action.get("command") or "").strip()
            if not cmd:
                return "(no command provided)", False
            return _run(cmd, cwd)
        if a == "list_files":
            lp = _resolve(cwd, action.get("path", "."))
            if not lp:
                return "path escapes the workspace — use a path inside the project", False
            return _run(f"ls -la {_q(lp)}", cwd)
        if a == "grep":
            pat = action.get("pattern", "")
            if not pat:
                return "grep needs a `pattern` (regex to search in file contents)", False
            where = _resolve(cwd, action.get("path", "."))   # confine to the workspace, like read_file
            if not where:
                return "path escapes the workspace — search inside the project", False
            tool = "rg -n --no-heading" if _which("rg") else "grep -rn"
            out, _ = _run(f"{tool} -e {_q(pat)} {_q(where)}", cwd)
            return out, True   # no matches is a valid result, not a failure
        if a == "glob":
            pat = action.get("pattern", "")
            if not pat:
                return "glob needs a `pattern` (e.g. **/*.py)", False
            if _which("rg"):
                out, _ = _run(f"rg --files -g {_q(pat)}", cwd)
            else:
                out, _ = _run(f"git ls-files {_q(pat)} 2>/dev/null || find . -path {_q('*/'+pat)}", cwd)
            return out or "(no files match)", True
        if a == "read_file":
            path = _resolve(cwd, action.get("path", ""))
            if not path:
                return "path escapes the workspace — use a path inside the project", False
            if not os.path.exists(path):
                return f"no such file: {action.get('path')}", False
            if os.path.isdir(path):
                return f"{action.get('path')} is a directory — use list_files or glob", False
            with open(path, errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            offset = max(1, int(action.get("offset", 1)))
            if total and offset > total:
                return f"offset {offset} is past the end — {action.get('path')} has only {total} lines", False
            limit = max(1, int(action.get("limit", 800)))
            chunk = lines[offset - 1: offset - 1 + limit]
            body = "".join(chunk)
            note = ""
            if offset > 1 or offset - 1 + limit < total:
                shown_to = min(offset - 1 + limit, total)
                note = f"\n[showing lines {offset}-{shown_to} of {total}. Use offset/limit to read more.]"
            return _maybe_offload(body, "read") + note, True
        if a == "write_file":
            p = _resolve(cwd, action.get("path", ""))
            if not p:
                return "path escapes the workspace — use a path inside the project", False
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f:
                f.write(action.get("content", ""))
            return f"wrote {action['path']} ({len(action.get('content',''))} bytes)", True
        if a == "edit_file":
            p = _resolve(cwd, action.get("path", ""))
            if not p:
                return "path escapes the workspace — use a path inside the project", False
            if not os.path.exists(p):
                return f"no such file: {action.get('path')} — use write_file to create it", False
            old, new = action.get("old", ""), action.get("new", "")
            with open(p, errors="replace") as f:
                text = f.read()
            if not old:
                return f"edit failed: provide the `old` snippet to replace.", False
            n = text.count(old)
            if n == 1:
                with open(p, "w") as f:
                    f.write(text.replace(old, new, 1))
                return f"edited {action['path']} (exact)", True
            if n > 1:
                return f"edit failed: `old` appears {n} times — add surrounding context to make it unique.", False
            # exact miss → whitespace-tolerant match (small models rarely reproduce indentation exactly)
            newtext, ok, how = _fuzzy_replace(text, old, new)
            if ok:
                with open(p, "w") as f:
                    f.write(newtext)
                return f"edited {action['path']} ({how})", True
            return f"edit failed: `old` not found in {action['path']} ({how}). Read the file and copy the EXACT text, or use write_file to rewrite it.", False
        return f"(unknown action: {a})", False
    except Exception as e:
        return f"(error running {a}: {e})", False
