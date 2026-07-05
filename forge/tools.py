"""Tools + the constrained action schema that gates them.

The schema is the contract every model output is forced to match — a small model
cannot emit a malformed or hallucinated call. It also carries the agent's living
PLAN, which keeps long-horizon work coherent (the single biggest quality lever for
weaker models: hold the plan in the harness, not in the model's head)."""
import os
import subprocess

BASH_TIMEOUT = int(os.environ.get("FORGE_BASH_TIMEOUT", "60"))  # fail fast, don't hang for minutes

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
            "enum": ["bash", "read_file", "write_file", "edit_file", "list_files", "fleet_send", "say"],
        },
        "command": {"type": "string", "description": "shell command (bash)"},
        "path": {"type": "string", "description": "file path (read/write/edit/list)"},
        "content": {"type": "string", "description": "full file content (write_file)"},
        "old": {"type": "string", "description": "exact text to replace (edit_file)"},
        "new": {"type": "string", "description": "replacement text (edit_file)"},
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
  fleet_send  {target, message}   message another forge session (it receives it mid-work)
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


def _run(cmd, cwd, timeout=None):
    timeout = timeout or BASH_TIMEOUT
    try:
        p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout + p.stderr).strip()
        ok = p.returncode == 0
        body = out if out else f"(no output, exit {p.returncode})"
        return body[:6000], ok
    except subprocess.TimeoutExpired:
        return (f"(timed out after {timeout}s — the command was too slow. Scope it down: exclude node_modules/.git, "
                "or use `git ls-files` (lists only real project files), `rg`, or a narrower path instead of scanning "
                "everything with `find . -exec`.)", False)


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
            return _run(f"ls -la {action.get('path', '.')}", cwd)
        if a == "read_file":
            with open(os.path.join(cwd, action["path"])) as f:
                return f.read()[:8000], True
        if a == "write_file":
            p = os.path.join(cwd, action["path"])
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f:
                f.write(action.get("content", ""))
            return f"wrote {action['path']} ({len(action.get('content',''))} bytes)", True
        if a == "edit_file":
            p = os.path.join(cwd, action["path"])
            old, new = action.get("old", ""), action.get("new", "")
            with open(p) as f:
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
