"""Tools + the constrained action schema that gates them.

The schema is the contract every model output is forced to match — a small model
cannot emit a malformed or hallucinated call. It also carries the agent's living
PLAN, which keeps long-horizon work coherent (the single biggest quality lever for
weaker models: hold the plan in the harness, not in the model's head)."""
import difflib
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
        "context": {"type": "integer", "description": "grep: lines of context to show around each match (default 2, max 5)"},
        "offset": {"type": "integer", "description": "start line for read_file (1-based)"},
        "limit": {"type": "integer", "description": "max lines for read_file"},
        "outline": {"type": "boolean", "description": "read_file: return only the file's symbol map (defs/classes with line numbers) instead of its text — turns a 2000-line file into a ~60-line map; then read exact ranges with offset/limit"},
        "target": {"type": "string", "description": "which session to message (fleet_send): its project name, dir, or id"},
        "message": {"type": "string", "description": "the message text (say, or fleet_send)"},
        "note": {"type": "string", "description": "a durable fact worth keeping: where something lives, a command that works, a decision made"},
    },
    "required": ["thought", "action"],
}

TOOL_HELP = """Each turn, output ONE JSON action. Maintain a `plan` (todo list) and keep it updated as you work.
Optionally set `note` to pin a durable fact worth keeping: where something lives, a command that works, a decision made (survives compaction).
Actions:
  bash        {command}            run a shell command, see its output
  bash        {command, background:true}   start a SERVER or long-lived process: returns pid + log file
                                  immediately and keeps running — then test it (curl, client, ...),
                                  check output with `tail <log>`, stop with `kill <pid>`
  read_file   {path}              read a file
  read_file   {path, outline:true}  map a big file: its defs/classes with line numbers (.py/.js/.ts/.go/.rs),
                                  then read a symbol's body with offset/limit
  write_file  {path, content}     create/overwrite a file with full content
  edit_file   {path, old, new}    surgically replace an exact snippet (preferred for changes)
  list_files  {path?}             list a directory
  grep        {pattern, path?, context?}   search file CONTENTS by regex (ripgrep; ±context lines, grouped by file)
  glob        {pattern}           find files by name (e.g. **/*.py); use this over `find`
  fleet_send  {target, message}   message another session — forge or Claude Code (it receives it mid-work);
                                  target "list" (no message) lists every reachable session
  say         {message}           talk to the user (ends your turn)"""


def _fuzzy_hits(tlines, olines):
    """0-based line indices where the stripped block `olines` matches `tlines`
    ignoring each line's leading/trailing whitespace."""
    return [i for i in range(len(tlines) - len(olines) + 1)
            if [tlines[i + j].strip() for j in range(len(olines))] == olines]


def _fuzzy_replace(text, old, new):
    """Match `old` ignoring per-line leading/trailing whitespace, so a model that
    gets indentation slightly wrong can still edit. Only acts on a UNIQUE match."""
    tlines = text.split("\n")
    olines = [ln.strip() for ln in old.strip("\n").split("\n")]
    if not olines or not any(olines):
        return text, False, "empty"
    hits = _fuzzy_hits(tlines, olines)
    if len(hits) != 1:
        return text, False, f"{len(hits)} fuzzy matches"
    i = hits[0]
    result = tlines[:i] + new.split("\n") + tlines[i + len(olines):]
    return "\n".join(result), True, "fuzzy"


def _exact_starts(text, old):
    """0-based line index of the start of each exact substring occurrence of `old`."""
    starts, idx = [], text.find(old)
    while idx != -1:
        starts.append(text.count("\n", 0, idx))
        idx = text.find(old, idx + 1)
    return starts


def _match_report(tlines, starts):
    """One line per match: '  line N: <that line, stripped>' — gives the model the
    exact locations to disambiguate with instead of just a count."""
    return "\n".join(f"  line {i + 1}: {tlines[i].strip()}" for i in starts)


def _closest_region(text, old, max_lines=4000):
    """Find the file region most similar to `old` via difflib, for the 0-match edit
    case. Slides a window of len(old)±2 lines over the file, scores each with
    SequenceMatcher.ratio() — pre-filtered by the cheap real_quick_ratio/quick_ratio
    upper bounds so it stays fast — and returns (start_lineno, end_lineno, verbatim
    region) for the best window clearing a 0.5 ratio floor, else None. Files past
    max_lines are skipped (fall back to the generic message) to bound the scan."""
    tlines = text.split("\n")
    if not tlines or len(tlines) > max_lines:
        return None
    oldblock = old.strip("\n")
    if not oldblock:
        return None
    onum = len(oldblock.split("\n"))
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(oldblock)
    floor, best_ratio, best = 0.5, 0.0, None
    for size in range(max(1, onum - 2), onum + 3):
        if size > len(tlines):
            break
        for i in range(len(tlines) - size + 1):
            sm.set_seq1("\n".join(tlines[i:i + size]))
            if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
                continue
            r = sm.ratio()
            if r >= floor and r > best_ratio:
                best_ratio, best = r, (i, size)
    if best is None:
        return None
    i, size = best
    return i + 1, i + size, "\n".join(tlines[i:i + size])


# ---- failed-bash error classifier --------------------------------------------
# When a bash action FAILS, the first pattern that matches its output appends a
# one-line, deterministic recovery hint — turning a generic failure into an
# actionable next step without spending a model turn to diagnose it.
_BASH_HINTS = [
    (r"(?:ModuleNotFoundError|ImportError:.*No module named)",
     "hint: a Python module is missing — check for a venv (ls .venv/bin) and use its python, or `pip install` the missing module."),
    (r"command not found|not found: ",
     "hint: that executable isn't on PATH — check the name/spelling, install it, or use its full path (which <cmd>)."),
    (r"address already in use|EADDRINUSE",
     "hint: that port is already taken — pick another port, or find and kill the holder (lsof -i :<port>)."),
    (r"No such file or directory",
     "hint: a path doesn't exist — verify it (ls the parent dir), mkdir -p a missing dir, or fix a wrong relative path."),
    (r"Permission denied|EACCES",
     "hint: permission denied — check the file mode (ls -l), chmod +x a script, or you're writing outside a writable dir."),
    (r"npm ERR!.*(?:ENOENT|missing script)|Missing script",
     "hint: npm can't find that — run from the dir with package.json and check the script name in its \"scripts\"."),
    (r"error: pathspec .* did not match|not a git repository",
     "hint: git can't resolve that — check the ref/path exists, or run inside a git repo (git status)."),
    (r"SyntaxError|IndentationError",
     "hint: the code has a syntax error — read the cited line/column and fix it before re-running."),
    (r"unbound variable|: unbound",
     "hint: a shell variable is unset — quote it or provide a default (${VAR:-default})."),
    (r"connection refused|Could not connect|ECONNREFUSED",
     "hint: nothing is listening there — start the server first (try it as a background bash), or fix the host/port."),
    (r"No space left on device|ENOSPC",
     "hint: the disk is full — free space or write somewhere else."),
]
_BASH_HINTS = [(__import__("re").compile(p, __import__("re").I), h) for p, h in _BASH_HINTS]


def error_hint(text):
    """Return the first matching recovery hint for a failed bash observation, or ''.
    Pure and deterministic — the caller appends it to the failure observation."""
    for rx, hint in _BASH_HINTS:
        if rx.search(text or ""):
            return hint
    return ""


MAX_OUTPUT = 12000  # chars kept inline; larger output is saved to a file with a preview


def shape(text, budget, note=""):
    """Fit `text` to a char budget WITHOUT silently dropping the tail, then append
    `note` verbatim AFTER the cut so a harness pointer/summary can NEVER be sliced
    off. If it fits, return it unchanged (plus any note). If not, keep a HEAD
    (~60% of budget) + an explicit '[… N chars omitted …]' marker + a TAIL
    (remaining budget). TAIL-retention is mandatory: errors, test summaries and
    pointers live at the END of output, so head-only truncation is the worst
    possible policy for a coding agent."""
    if len(text) <= budget:
        return text + note
    head = int(budget * 0.6)
    tail = budget - head
    omitted = len(text) - head - tail
    marker = f"\n[… {omitted} chars omitted from the middle …]\n"
    return text[:head] + marker + text[len(text) - tail:] + note


def overflow_dir(cwd):
    """Where big outputs and background logs live: <cwd>/.forge/output — INSIDE the
    workspace so the model's own read_file/grep (confined to cwd by _resolve) can
    follow the pointer it is handed. On first creation, drop a .gitignore ('*') so
    the whole .forge dir stays out of version control."""
    d = os.path.join(cwd, ".forge", "output")
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        gi = os.path.join(cwd, ".forge", ".gitignore")
        if not os.path.exists(gi):
            try:
                with open(gi, "w") as f:
                    f.write("*\n")
            except OSError:
                pass
    return d


def _maybe_offload(text, label, cwd):
    """Keep big outputs out of context: save the FULL text to a file under
    <cwd>/.forge/output and return (preview, note) — the preview is shape()'d, the
    note is a pointer the model can read_file/grep. The note is returned SEPARATELY
    (not baked in) so the caller re-attaches it AFTER any further truncation."""
    if len(text) <= MAX_OUTPUT:
        return text, ""
    d = overflow_dir(cwd)
    import hashlib
    fn = os.path.join(d, f"{label}-{hashlib.md5(text.encode()).hexdigest()[:8]}.txt")
    with open(fn, "w") as f:
        f.write(text)
    note = (f"\n[... output truncated: {len(text)} chars total. Full output saved to {fn} — "
            f"read a range or grep it.]")
    return shape(text, MAX_OUTPUT), note


# ---- background processes (servers, watchers) --------------------------------
# The agent starts one, gets a pid + live log file back IMMEDIATELY, and keeps
# working — test with curl, tail the log, kill the pid. All are cleaned up when
# the forge session exits.
_BG_PROCS = []
_BG_TRAILING_AMP = __import__("re").compile(r"(?<![&|])&\s*$")


def _run_background(cmd, cwd):
    import time
    log = os.path.join(overflow_dir(cwd), f"bg-{len(_BG_PROCS) + 1}-{os.getpid()}.log")
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
    preview, note = _maybe_offload(body, "bash", cwd)
    return preview + note, ok


def _kill_group(p, signal):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        p.kill()
    try:
        p.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def _run_rc(cmd, cwd, timeout=None, stop=None):
    """Like _run but returns (raw_output, returncode) and does NOT offload.
    grep/glob must inspect the tool's exit code (rg/grep: 1=no match, 2=bad
    regex) and group/cap the RAW output BEFORE any offload, so they cannot use
    _run (which merges stderr, swallows rc, and offloads first)."""
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
                return "(stopped by user)", 130
            if time.monotonic() > deadline:
                _kill_group(p, signal)
                return (f"(timed out after {timeout}s — narrow the path or pattern.)", 124)
    return (out or "").strip(), p.returncode


_GREP_MATCH = __import__("re").compile(r"^(.+?):(\d+):")     # a match line: file:lineno:text
_GREP_ANY = __import__("re").compile(r"^(.+?)[:-](\d+)[:-]")  # a match OR context line


def _group_grep(raw, per_file=3, total=40):
    """Turn rg/grep output (file:line:text match lines, file-line-text context
    lines, `--` block separators) into a per-file digest: a '<file> — <M>
    matches, showing <k>:' header followed by the first few match blocks, capped
    per file and overall so a big grep stays actionable without a follow-up read.
    Returns (assembled_text, total_match_count). Filenames and matched text are
    always preserved."""
    # 1) split on rg's `--` separators into blocks
    raw_blocks, cur = [], []
    for ln in raw.split("\n"):
        if ln.strip() == "--":
            if cur:
                raw_blocks.append(cur); cur = []
        elif ln != "":
            cur.append(ln)
    if cur:
        raw_blocks.append(cur)
    # 2) a contextless block (context=0) is really one block per match line
    blocks = []
    for b in raw_blocks:
        if any(not _GREP_MATCH.match(ln) for ln in b):
            blocks.append(b)
        else:
            blocks.extend([ln] for ln in b)
    # 3) group blocks by file in first-appearance order
    order, groups = [], {}
    for b in blocks:
        f = next((_GREP_ANY.match(ln).group(1) for ln in b if _GREP_ANY.match(ln)), "")
        if f not in groups:
            groups[f] = []; order.append(f)
        groups[f].append(b)
    # 4) assemble with per-file (~3) and overall (~40) block caps
    parts, shown, grand = [], 0, 0
    omitted_files = 0
    for f in order:
        fb = groups[f]
        matches = sum(1 for b in fb for ln in b if _GREP_MATCH.match(ln))
        grand += matches
        if shown >= total:
            omitted_files += 1
            continue
        take = fb[:per_file][: total - shown]
        shown += len(take)
        head = f"{f} — {matches} matches, showing {len(take)}:"
        parts.append(head + "\n" + "\n".join("\n".join(b) for b in take))
    body = "\n\n".join(parts)
    if omitted_files:
        body += f"\n\n[... {omitted_files} more file(s) with matches omitted — narrow the pattern or path.]"
    return body, grand


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
            ctx = min(max(int(action.get("context", 2) or 0), 0), 5)
            # -H/--with-filename: force the `file:` prefix even for a single explicit
            # file, else rg drops it and _group_grep can't recover the filename or count.
            tool = "rg -nH --no-heading" if _which("rg") else "grep -rnH"
            out, rc = _run_rc(f"{tool} -C {ctx} -e {_q(pat)} {_q(where)}", cwd, stop=stop)
            if rc == 2:   # regex parse error — retry as a LITERAL string (-F) so a pasted snippet still lands
                # never echo the raw `rg: regex parse error:` line — a small model reads it as 'no results'
                lit, lrc = _run_rc(f"{tool} -F -C {ctx} -e {_q(pat)} {_q(where)}", cwd, stop=stop)
                if lrc == 0 and lit:
                    grouped, n = _group_grep(lit)
                    body, note = _maybe_offload(grouped, "grep", cwd)
                    return (f"your regex was invalid — searched literally instead: {n} matches\n{body}" + note, True)
                return (f"your regex was invalid — searched literally instead: no matches for {pat}", True)
            if rc == 1:   # rg/grep both exit 1 on no match — a valid result, not a failure
                return f"no matches for {pat} (searched under {action.get('path', '.')})", True
            if rc == 0:
                grouped, _ = _group_grep(out)
                body, note = _maybe_offload(grouped, "grep", cwd)
                return body + note, True
            return out, False   # timeout / real error — surface it, don't fake success
        if a == "glob":
            pat = action.get("pattern", "")
            if not pat:
                return "glob needs a `pattern` (e.g. **/*.py)", False
            if _which("rg"):
                out, rc = _run_rc(f"rg --files -g {_q(pat)}", cwd, stop=stop)
            else:
                out, rc = _run_rc(f"git ls-files {_q(pat)} 2>/dev/null || find . -path {_q('*/'+pat)}", cwd, stop=stop)
            if rc == 1 or not out:   # rg --files exits 1 when nothing matches the glob
                return "(no files match)", True
            if rc == 0:
                body, note = _maybe_offload(out, "glob", cwd)
                return body + note, True
            return out, False   # e.g. a malformed glob — a real error, not "no files"
        if a == "read_file":
            path = _resolve(cwd, action.get("path", ""))
            if not path:
                return "path escapes the workspace — use a path inside the project", False
            if not os.path.exists(path):
                return f"no such file: {action.get('path')}", False
            if os.path.isdir(path):
                return f"{action.get('path')} is a directory — use list_files or glob", False
            if action.get("outline"):
                from . import index
                with open(path, errors="replace") as f:
                    text = f.read()
                nlines = text.count("\n") + 1
                syms = index.extract_symbols(path, text)
                if not syms:
                    return (f"{action.get('path')}: no symbols to outline "
                            f"(outline supports .py/.js/.ts/.go/.rs) — {nlines} lines; "
                            "read it directly with offset/limit.", True)
                rows = []
                for s in syms:
                    ind = "    " if s["kind"] == "method" else "  "
                    rows.append(f"{ind}{s['lineno']:>5}  {s['signature']}")
                head = f"OUTLINE {action.get('path')} — {len(syms)} symbols in {nlines} lines:\n"
                tail = "\n[outline only — read a symbol's body with offset/limit, e.g. {offset:<lineno>, limit:40}.]"
                preview, off_note = _maybe_offload(head + "\n".join(rows), "outline", cwd)
                return preview + off_note + tail, True
            with open(path, errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            offset = max(1, int(action.get("offset", 1)))
            if total and offset > total:
                return f"offset {offset} is past the end — {action.get('path')} has only {total} lines", False
            limit = max(1, int(action.get("limit", 800)))
            chunk = lines[offset - 1: offset - 1 + limit]
            body = "".join(chunk)
            range_note = ""
            if offset > 1 or offset - 1 + limit < total:
                shown_to = min(offset - 1 + limit, total)
                range_note = f"\n[showing lines {offset}-{shown_to} of {total}. Use offset/limit to read more.]"
            preview, off_note = _maybe_offload(body, "read", cwd)
            # both notes ride at the TAIL so they survive any further shape() truncation
            return preview + off_note + range_note, True
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
                # enumerate the locations so the model can add context, not just a count
                locs = _match_report(text.split("\n"), _exact_starts(text, old))
                return (f"edit failed: `old` appears {n} times in {action['path']} — add surrounding "
                        f"context to make it unique. Matches at:\n{locs}", False)
            if n == 1:
                newtext, how = text.replace(old, new, 1), "exact"
            else:
                # exact miss → whitespace-tolerant match (small models rarely reproduce indentation exactly)
                newtext, matched, how = _fuzzy_replace(text, old, new)
                if not matched:
                    tlines = text.split("\n")
                    olines = [ln.strip() for ln in old.strip("\n").split("\n")]
                    hits = _fuzzy_hits(tlines, olines) if any(olines) else []
                    if len(hits) > 1:
                        # >1 whitespace-tolerant matches → same enumerate treatment as the exact case
                        locs = _match_report(tlines, hits)
                        return (f"edit failed: `old` matches {len(hits)} places in {action['path']} "
                                f"(ignoring indentation) — add surrounding context to make it unique. "
                                f"Matches at:\n{locs}", False)
                    # zero matches → point at the closest region verbatim so the retry is a copy, not a re-read
                    region = _closest_region(text, old)
                    if region:
                        i, j, block = region
                        return (f"edit failed: `old` not found in {action['path']}. CLOSEST region "
                                f"(lines {i}-{j}):\n{block}\n"
                                f"Re-send edit_file with old copied EXACTLY from the region above.", False)
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
