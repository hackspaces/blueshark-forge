"""Tools + the constrained action schema that gates them.

The schema is the contract every model output is forced to match — a small model
cannot emit a malformed or hallucinated call. It also carries the agent's living
PLAN, which keeps long-horizon work coherent (the single biggest quality lever for
weaker models: hold the plan in the harness, not in the model's head)."""
import json
import os
import shlex
import shutil
import subprocess
import time

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
        "background": {"type": "boolean", "description": "bash: run as a background process (servers, watchers) — returns pid + log file immediately and keeps running while you continue"},
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
  bash        {command, background:true}   start a SERVER or long-lived process: returns pid + log file
                                  immediately and keeps running — then test it (curl, client, ...),
                                  check output with `tail <log>`, stop with `kill <pid>`
  read_file   {path}              read a file
  write_file  {path, content}     create/overwrite a file with full content
  edit_file   {path, old, new}    surgically replace an exact snippet (preferred for changes)
  list_files  {path?}             list a directory
  grep        {pattern, path?}    search file CONTENTS by regex (ripgrep; structured, fast)
  glob        {pattern}           find files by name (e.g. **/*.py); use this over `find`
  fleet_send  {target, message}   message another session — forge or Claude Code (it receives it mid-work);
                                  target "list" (no message) lists every reachable session
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


# ---- background processes (servers, watchers) --------------------------------
# The agent starts one, gets a pid + live log file back IMMEDIATELY, and keeps
# working — test with curl, tail the log, kill the pid. All are cleaned up when
# the forge session exits.
_BG_PROCS = []
_BG_TRAILING_AMP = __import__("re").compile(r"(?<![&|])&\s*$")


def _run_background(cmd, cwd):
    import time
    os.makedirs(_OVERFLOW_DIR, exist_ok=True)
    log = os.path.join(_OVERFLOW_DIR, f"bg-{len(_BG_PROCS) + 1}-{os.getpid()}.log")
    lf = open(log, "w")
    p = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    _BG_PROCS.append(p)
    if len(_BG_PROCS) == 1:
        import atexit
        atexit.register(_kill_background)
    time.sleep(1.2)                      # long enough to catch an instant crash
    if p.poll() is not None:
        lf.flush()
        try:
            with open(log, errors="replace") as f:
                tail = f.read()[-800:]
        except OSError:
            tail = ""
        return (f"(background command exited immediately, code {p.returncode}) output:\n{tail}",
                p.returncode == 0)
    return (f"✓ running in the background: pid {p.pid}, output → {log}\n"
            f"It KEEPS RUNNING while you continue — test it now (curl, run a client, ...). "
            f"Check its output: bash `tail -n 40 {log}`. Stop it: bash `kill {p.pid}`. "
            f"It is stopped automatically when this forge session ends.", True)


def _kill_background():
    import signal
    for p in _BG_PROCS:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass


def _run(cmd, cwd, timeout=None, stop=None):
    """Run a shell command. If `stop` (a threading.Event) fires mid-run — the
    user hit Esc — the whole process group is killed so control returns
    immediately instead of waiting out a slow command."""
    import signal
    timeout = timeout or BASH_TIMEOUT
    p = subprocess.Popen(cmd, cwd=cwd, shell=True, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            out = p.communicate(timeout=0.2)[0]
            break
        except subprocess.TimeoutExpired:
            if stop is not None and stop.is_set():
                _kill_group(p, signal)
                return "(stopped by user)", False
            if time.monotonic() > deadline:
                _kill_group(p, signal)
                return (f"(timed out after {timeout}s — the command was too slow. Scope it down: exclude node_modules/.git, "
                        "or use `git ls-files` (lists only real project files), `rg`, or a narrower path instead of scanning "
                        "everything with `find . -exec`.)", False)
    out = (out or "").strip()
    ok = p.returncode == 0
    body = out if out else f"(no output, exit {p.returncode})"
    return _maybe_offload(body, "bash"), ok


def _kill_group(p, signal):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        p.kill()
    try:
        p.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        pass


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


def _external_check(cmd, text, path):
    """Run an external syntax checker (bash -n, node --check) on CANDIDATE text
    written to a temp file with the real extension — the real file is never
    touched. Returns "" on pass, the first error line (with the temp path
    rewritten to the real basename) on failure, or None if the check can't run."""
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(path)[1])
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        r = subprocess.run(cmd + [tmp], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            return ""
        msg = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
        line = msg[0] if msg else f"exit {r.returncode}"
        return line.replace(tmp, os.path.basename(path))
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# config files that legitimately use JSONC (comments / trailing commas) — strict json.loads would wrongly reject them
_JSONC_NAMES = ("tsconfig.json", "jsconfig.json", "devcontainer.json")


def _syntax_error(path, text):
    """Check CANDIDATE file content before it's written, per extension. Returns
    None if no checker applies, "" if a check ran and PASSED, or a human error
    string if it FAILED. Wrapped so it can NEVER raise into execute()."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".py":
            try:
                compile(text, path, "exec")
                return ""
            except SyntaxError as e:   # IndentationError subclasses SyntaxError
                return f"{e.msg} at line {e.lineno}: {(e.text or '').strip()}"
        if ext == ".json":
            jsonc = os.path.basename(path) in _JSONC_NAMES or ".vscode" in path.replace("\\", "/").split("/")
            if jsonc or not text.strip():   # JSONC config, or an empty placeholder — don't strict-parse
                return None
            try:
                json.loads(text)
                return ""
            except ValueError as e:
                return str(e)
        if ext == ".toml":
            try:
                import tomllib          # 3.11+ only; py3.10 has no stdlib toml
            except ModuleNotFoundError:
                return None
            try:
                tomllib.loads(text)
                return ""
            except Exception as e:
                return str(e)
        if ext in (".sh", ".bash") and _which("bash"):
            return _external_check(["bash", "-n"], text, path)
        if ext in (".js", ".mjs", ".cjs") and _which("node"):
            return _external_check(["node", "--check"], text, path)
        return None
    except Exception:
        return None


def _gate(path, newtext, oldtext):
    """Decide whether to refuse a write/edit. Returns (err, block): err is
    _syntax_error(path, newtext) verbatim (for the observation tail); block is
    True ONLY on a valid→invalid regression — newtext fails a check that the
    current on-disk content (oldtext, or None for a new file) passes. A file that
    is ALREADY broken can still be saved with partial progress, so multi-error
    files and merge conflicts can be repaired one hunk at a time."""
    err = _syntax_error(path, newtext)
    if not err:                                      # candidate is clean, or no checker applies
        return err, False
    if oldtext is not None and _syntax_error(path, oldtext):
        return err, False                            # was already broken → don't block partial progress
    return err, True                                 # valid (or new) → invalid → block


def _syntax_tail(err):
    """Observation suffix: '' when no checker ran, ', syntax OK' when it passed,
    ', still has errors — <msg>' when we saved a file that is still broken."""
    if err == "":
        return ", syntax OK"
    return f", still has errors — {err}" if err else ""


def execute(action, cwd, stop=None):
    """Run one action. Returns (observation, ok). `stop` interrupts a running
    bash command when the user hits Esc."""
    a = action.get("action")
    try:
        if a == "bash":
            cmd = (action.get("command") or "").strip()
            if not cmd:
                return "(no command provided)", False
            if action.get("background") or _BG_TRAILING_AMP.search(cmd):
                return _run_background(_BG_TRAILING_AMP.sub("", cmd).strip(), cwd)
            return _run(cmd, cwd, stop=stop)
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
            content = action.get("content", "")
            prev = None
            if os.path.isfile(p):                 # only a valid→invalid regression is refused; an
                with open(p, errors="replace") as f:   # already-broken file can be saved with partial progress
                    prev = f.read()
            err, block = _gate(p, content, prev)  # check the CANDIDATE — never touch the real file on failure
            if block:
                return (f"that content would make {action['path']} invalid — {err}. "
                        "The file was NOT written. Fix it and retry.", False)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f:
                f.write(content)
            return f"wrote {action['path']} ({len(content)} bytes)" + _syntax_tail(err), True
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
            if n > 1:
                return f"edit failed: `old` appears {n} times — add surrounding context to make it unique.", False
            if n == 1:
                newtext, how = text.replace(old, new, 1), "exact"
            else:
                # exact miss → whitespace-tolerant match (small models rarely reproduce indentation exactly)
                newtext, matched, how = _fuzzy_replace(text, old, new)
                if not matched:
                    return f"edit failed: `old` not found in {action['path']} ({how}). Read the file and copy the EXACT text, or use write_file to rewrite it.", False
            # check the CANDIDATE text before writing — a failing edit never touches the file, but
            # a file that's ALREADY broken can be repaired one hunk at a time (block only valid→invalid)
            err, block = _gate(p, newtext, text)
            if block:
                return (f"that edit would make {action['path']} invalid — {err}. "
                        "The file was NOT changed. Fix the snippet and retry.", False)
            with open(p, "w") as f:
                f.write(newtext)
            return f"edited {action['path']} ({how})" + _syntax_tail(err), True
        return f"(unknown action: {a})", False
    except Exception as e:
        return f"(error running {a}: {e})", False
