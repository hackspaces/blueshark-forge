"""Fleet — the native multi-agent nervous system for forge.

Because forge owns its transcripts (~/.forge/sessions/*.jsonl), its registry, and
a native inbox on every session, fleet is not a bolt-on here — it is built in:

  MESSAGE    post into any session's inbox; the agent absorbs it between steps
  TRUST      verify a session's "done" claim with an INDEPENDENT forge agent
             (model-agnostic — the checker can be any model, even a local one)
  COORDINATE warn two sessions editing the same file, before the conflict
  LEARN      harvest durable repo facts a session discovers and share them

All of it runs on whatever model you point it at. No vendor API.
"""
import json
import os
import re
import time
import urllib.request
from collections import Counter

from . import session as sessmod
from .util import slurp
from .backends import make_backend

FORGE = os.path.expanduser("~/.forge")
STATE = os.path.join(FORGE, "state"); os.makedirs(STATE, exist_ok=True)
RECEIPTS = os.path.join(FORGE, "verdicts.jsonl")
LEARN_DIR = os.path.join(FORGE, "learn"); os.makedirs(LEARN_DIR, exist_ok=True)


# ---- transcript reading -----------------------------------------------------
def _records(sid, tail_bytes=300000):
    path = os.path.join(sessmod.SESSIONS, f"{sid}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        size = os.path.getsize(path)
        if size > tail_bytes:
            f.seek(size - tail_bytes); f.readline()
        recs = []
        for line in f:
            try:
                recs.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    return recs


def last_say(sid):
    text = None
    for r in _records(sid):
        if r.get("type") == "assistant" and r.get("text"):
            text = r["text"]
    return text


def harness_verified(sid):
    """True if the P2.1 done-gate already verified the latest state: a passing
    'verified' record no older than the most recent file mutation, so nothing has
    changed on disk since the harness ran the tests. Mutations count write_file /
    edit_file AND file-touching bash (echo > f, sed -i …) — otherwise a bash edit
    after the verified record would let the daemon skip a now-stale claim without
    ever re-running the tests. Lets the daemon skip its expensive rsync-and-run
    verify pass for a claim the harness already grounded."""
    last_mut = last_ver = 0.0
    for r in _records(sid):
        t = r.get("ts", 0) or 0
        typ = r.get("type")
        if typ == "action":
            a = r.get("action")
            if a in ("write_file", "edit_file"):
                last_mut = t
            elif a == "bash" and bash_mutates((r.get("args") or {}).get("command", "")):
                last_mut = t
        elif typ == "verified" and r.get("ok"):
            last_ver = t
    return last_ver > 0 and last_ver >= last_mut


def edited_files(sid, cwd):
    files = set()
    for r in _records(sid):
        if r.get("type") == "action" and r.get("action") in ("write_file", "edit_file"):
            p = (r.get("args") or {}).get("path")
            if p:
                files.add(os.path.normpath(os.path.join(cwd, p)))
    return files


def recent_work(sid, max_chars=9000):
    chunks = []
    for r in _records(sid):
        if r.get("type") == "assistant" and r.get("text"):
            chunks.append(r["text"])
        elif r.get("type") == "action":
            chunks.append(f"[{r.get('action')} {(r.get('args') or {})}]")
    return "\n".join(chunks)[-max_chars:]


# ---- MESSAGE ----------------------------------------------------------------
def find_session(target, exclude_sid=None):
    """Match a target across BOTH fleets: forge sessions and Claude Code
    sessions (via the bridge). Forge entries have no 'kind'; claude peers
    carry kind='claude'. `exclude_sid` drops the sender's own session, so
    "message ymp" from a session named ymp means the OTHER ymp."""
    from . import bridge
    live = sessmod.registry() + bridge.claude_peers()
    if exclude_sid:
        live = [e for e in live if e["sid"] != exclude_sid]
    t = target.strip().lower()
    # roster display format pasted back as a target: "name(sidprefix)" or
    # "name(sidprefix, claude)" — honor the sid prefix inside the parens
    m = re.match(r"^(.*?)\s*\(\s*([a-z0-9-]+)\s*(?:,[^)]*)?\)$", t)
    if m:
        by_sid = [e for e in live if e["sid"].lower().startswith(m.group(2))]
        if by_sid:
            return by_sid
        t = m.group(1).strip() or t
    # exact session id wins (that's how the daemon addresses a session)
    exact = [e for e in live if e["sid"].lower() == t]
    if exact:
        return exact
    return [e for e in live if e["name"].lower() == t or e["sid"].lower().startswith(t)
            or t in e["cwd"].lower() or t in e["name"].lower()]


def roster():
    """One-line-per-session view of everything reachable, both runtimes."""
    from . import bridge
    forge_s = [f"{e['name']}({e['sid'][:8]})" for e in sessmod.registry()]
    claude_s = [f"{e['name']}({e['sid'][:8]}, claude)" for e in bridge.claude_peers()]
    return ", ".join(forge_s + claude_s) or "(none)"


def send(target, text, sender="fleet", sender_cwd="", sender_sid=""):
    hits = find_session(target, exclude_sid=sender_sid or None)
    if len(hits) != 1:
        raise SystemExit(f"target '{target}' matched {len(hits)} sessions — "
                         f"use a session id prefix (e.g. '5fa56b39') to pick one; live: {roster()}")
    e = hits[0]
    if not e.get("port"):
        raise SystemExit(f"session {e['name']} has no reachable inbox")
    if e.get("kind") == "claude":       # a Claude Code session — speak its protocol
        from . import bridge
        try:
            bridge.send(e, text, sender, sender_cwd, sender_sid)
        except (urllib.error.URLError, OSError) as ex:
            raise SystemExit(f"couldn't deliver to {e['name']} (claude): {ex}")
        return e
    headers = {"X-Forge-From": sender}
    if e.get("token"):
        headers["X-Forge-Token"] = e["token"]   # authenticate to the peer's inbox
    req = urllib.request.Request(f"http://127.0.0.1:{e['port']}/", data=text.encode(), headers=headers)
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except urllib.error.URLError as ex:
        raise SystemExit(f"couldn't deliver to {e['name']}: {ex}")
    return e


# ---- TRUST ------------------------------------------------------------------
VERIFY_SYSTEM = """You are an INDEPENDENT verification agent. Another agent, working in this repository, claims it finished a task. Your job is to DISPROVE the claim if you can. Do not trust it.

Follow this discipline exactly — do NOT skip steps or assume anything:
  1. Read the actual files shown to you. Identify the REAL test command from package.json "scripts", a Makefile, or the project layout. Never assume the language or framework.
  2. Run that exact test command with bash and read its real output and exit status.
  3. If needed, read the source file the tests exercise to confirm.
You may read and run commands but you must NOT edit anything.

Base your verdict ONLY on commands you actually ran and their real output. Never guess. When done, `say` EXACTLY one of:
  VERDICT: CONFIRMED — <the test command you ran and its passing result>
  VERDICT: REFUTED — <the test command you ran and the concrete failure>"""


def _seed(cwd):
    """Ground the verifier in reality: the file listing + package manifest, so a
    small model cannot hallucinate the project type."""
    import subprocess
    try:
        ls = subprocess.run("ls -la", cwd=cwd, shell=True, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        ls = ""
    manifest = ""
    for name in ("package.json", "Makefile", "pyproject.toml", "Cargo.toml"):
        p = os.path.join(cwd, name)
        if os.path.exists(p):
            try:
                manifest += f"\n--- {name} ---\n" + slurp(p)[:1500]
            except OSError:
                pass
    return f"Repository files:\n{ls}\n{manifest}"

CLAIM_RE = re.compile(r"\b(all tests?\s+(pass|passing|green)|tests?\s+(all\s+)?(pass|passing|green)|all green|ready to (merge|review)|task complete|feature (is )?complete|done and (tested|verified)|passing)\b", re.I)


def detect_test_cmd(cwd, files=None):
    """Find the project's real test command deterministically.

    When `files` (edited absolute paths) is given, SCOPE a pytest run to the
    nearest test target of those edits so ONE unrelated pre-existing red test
    can't refute every claim forever. files=None returns the whole-suite command
    exactly as before (fully backward-compatible)."""
    pj = os.path.join(cwd, "package.json")
    if os.path.exists(pj):
        try:
            scripts = json.loads(slurp(pj)).get("scripts", {})
            if scripts.get("test") and "no test specified" not in scripts["test"]:
                if os.path.exists(os.path.join(cwd, "pnpm-lock.yaml")):
                    return "pnpm test"
                if os.path.exists(os.path.join(cwd, "yarn.lock")):
                    return "yarn test"
                return "npm test --silent"
        except (OSError, json.JSONDecodeError):
            pass
    mk = os.path.join(cwd, "Makefile")
    if os.path.exists(mk):
        try:
            if re.search(r"^test:", slurp(mk), re.M):
                return "make test"
        except OSError:
            pass
    if os.path.exists(os.path.join(cwd, "go.mod")):
        return "go test ./..."
    if os.path.exists(os.path.join(cwd, "pyproject.toml")) or os.path.isdir(os.path.join(cwd, "tests")):
        scope = _pytest_scope(cwd, files)
        return ("pytest -q " + scope).rstrip() if scope else "pytest -q"
    if os.path.exists(os.path.join(cwd, "Cargo.toml")):
        return "cargo test"
    return None


def _pytest_scope(cwd, files):
    """The nearest pytest target(s) for the edited files: an edited test file is
    run directly; a non-test edit maps to its nearest ancestor `tests` dir. Paths
    are returned relative to cwd so `pytest -q <scope>` runs only the claim's
    neighborhood. Empty string => fall back to the whole suite."""
    if not files:
        return ""
    cwd = os.path.normpath(cwd)
    targets = []
    for f in files:
        f = os.path.normpath(f)
        if not f.endswith(".py"):
            continue
        base = os.path.basename(f)
        target = None
        if base.startswith("test_") or base.endswith("_test.py"):
            target = f
        else:
            d = os.path.dirname(f)
            while d and d.startswith(cwd):
                cand = os.path.join(d, "tests")
                if os.path.isdir(cand):
                    target = cand
                    break
                if d == cwd:
                    break
                d = os.path.dirname(d)
        if target and target not in targets:
            targets.append(target)
    rels = []
    for t in targets:
        try:
            rel = os.path.relpath(t, cwd)
        except ValueError:
            continue
        if not rel.startswith(".."):
            rels.append(rel)
    return " ".join(rels)


# ---- bash mutation heuristic (done-gate) ------------------------------------
_REDIR_WRITE_RE = re.compile(r">(?!&)")          # > / >> to a file (not 2>&1, >&2)
_INPLACE_RE = re.compile(r"\b(?:sed|gsed|perl|ruby)\b\s+(?:-\S+\s+)*-\S*i\b")  # in-place edit
_SHELL_SEP_RE = re.compile(r"[;&|\n]+")
_MUTATING_HEADS = frozenset((
    "cp", "mv", "rm", "rmdir", "mkdir", "touch", "tee", "ln", "dd", "install",
    "patch", "truncate", "chmod", "chown", "rsync", "unzip", "shred", "mktemp"))
_GIT_WRITE = frozenset((
    "apply", "checkout", "restore", "reset", "stash", "clean", "revert",
    "cherry-pick", "merge", "rebase", "pull"))
# An interpreter running INLINE code can write files with no shell redirect
# (`python -c "open('f','w')..."`, `node -e "fs.writeFileSync(...)"`). We gate on the
# inline-code FLAG, not the head — `python app.py` (running the app) stays read-only.
_INTERP_HEADS = frozenset((
    "python", "python2", "python3", "node", "nodejs", "deno", "bun",
    "ruby", "perl", "php"))
_INTERP_INLINE = frozenset(("-c", "-e", "-p", "-n", "--eval", "--exec"))


def bash_mutates(command):
    """Heuristic: could this bash command change files on disk? Deliberately
    OVER-inclusive — a write that slips past the done-gate is the failure we must
    not have, so read-only bash (ls/cat/grep/git status…) stays ungated while
    anything that redirects to a file, edits in place, or runs a file-mutating
    command counts. Used both to trigger the gate live and to invalidate a stale
    'verified' record for the daemon."""
    if not command:
        return False
    if _REDIR_WRITE_RE.search(command) or _INPLACE_RE.search(command):
        return True
    for seg in _SHELL_SEP_RE.split(command):
        toks = seg.split()
        while toks and ("=" in toks[0] or toks[0] in ("sudo", "env", "time", "nice", "command")):
            toks = toks[1:]                       # strip leading env-assignment / wrapper
        if not toks:
            continue
        base = toks[0].rsplit("/", 1)[-1]
        if base in _MUTATING_HEADS:
            return True
        if base == "git" and len(toks) > 1 and toks[1] in _GIT_WRITE:
            return True
        if base in _INTERP_HEADS and _INTERP_INLINE.intersection(toks):
            return True                           # interpreter running INLINE code can write files
    return False


JUDGE_SYSTEM = """You are a strict verification judge. You are given ONLY the raw command output and file contents an independent verifier gathered while checking another agent's completion claim. Decide, FROM THIS EVIDENCE ALONE, whether the claim is CONFIRMED (the evidence positively shows the work is done and its tests/commands pass) or REFUTED (the evidence shows failures, errors, or that the work is not done). Assume nothing that is not present in the evidence. Reply with a single line, exactly one of:
  VERDICT: CONFIRMED
  VERDICT: REFUTED"""


def _overlay_uncommitted(cwd, dest):
    """A detached worktree checks out HEAD — the COMMITTED state — but done-claims
    are almost always about UNCOMMITTED working-tree edits. Copy every modified /
    untracked path reported by `git status --porcelain` from the live cwd on top of
    the worktree so the verifier sees the ACTUAL edits, not the last commit."""
    import shutil
    import subprocess
    try:
        out = subprocess.run(["git", "-C", cwd, "status", "--porcelain"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return
    for line in out.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:]
        if " -> " in path:                 # rename/copy — the destination is what exists now
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        src, dst = os.path.join(cwd, path), os.path.join(dest, path)
        if "D" in status and not os.path.exists(src):   # deletion — mirror it into the copy
            try:
                if os.path.isdir(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                elif os.path.exists(dst):
                    os.remove(dst)
            except OSError:
                pass
            continue
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=True)
            elif os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        except OSError:
            pass


def _rsync_copy(cwd, dest):
    """Fallback isolation for a non-git cwd: an rsync copy that now INCLUDES .git
    (the old code excluded it, breaking git-dependent tests). copytree covers the
    no-rsync case; node_modules is excluded and re-linked by the caller."""
    import shutil
    import subprocess
    try:
        r = subprocess.run(["rsync", "-a", "--exclude", "node_modules",
                            cwd.rstrip("/") + "/", dest + "/"],
                           capture_output=True, timeout=120)
        if r.returncode == 0:
            return
    except Exception:
        pass
    try:
        shutil.copytree(cwd, dest, ignore=shutil.ignore_patterns("node_modules"),
                        symlinks=True, dirs_exist_ok=True)
    except Exception:
        pass


def _isolate(cwd):
    """Make an isolated working copy of `cwd` and return (workdir, cleanup).

    Prefer a detached git worktree (keeps .git, ~instant, no writes through to the
    live repo) with the uncommitted working-tree edits overlaid on top of HEAD.
    Fall back to an rsync/copytree copy (INCLUDING .git) when cwd is not a git repo
    or the worktree can't be created."""
    import shutil
    import subprocess
    import tempfile
    base = tempfile.mkdtemp(prefix="forge-verify-")
    work = None
    cleanup = lambda: shutil.rmtree(base, ignore_errors=True)
    if os.path.isdir(os.path.join(cwd, ".git")):
        wt = os.path.join(base, "wt")
        try:
            r = subprocess.run(["git", "-C", cwd, "worktree", "add", "--detach", wt],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and os.path.isdir(wt):
                _overlay_uncommitted(cwd, wt)
                work = wt
                def cleanup():
                    subprocess.run(["git", "-C", cwd, "worktree", "remove", "--force", wt],
                                   capture_output=True, timeout=30)
                    shutil.rmtree(base, ignore_errors=True)
        except Exception:
            work = None
    if work is None:
        work = os.path.join(base, "copy")
        _rsync_copy(cwd, work)
    nm = os.path.join(cwd, "node_modules")   # read-only module resolution; never copied
    if os.path.isdir(nm) and not os.path.exists(os.path.join(work, "node_modules")):
        try:
            os.symlink(nm, os.path.join(work, "node_modules"))
        except OSError:
            pass
    return work, cleanup


def _deterministic_verdict(cmd, rc, out):
    """Turn a test-command result into a verdict dict — or None when we must defer
    to model reasoning instead of falsely REFUTING. pytest exit code 5 means "no
    tests were collected" (e.g. the claim's scope has no tests) — that is the model
    path, NOT a refutation."""
    toks = cmd.split()
    if toks and toks[0] == "pytest" and rc == 5:
        return None
    tail = " ".join(out.strip().splitlines()[-4:])[:220]
    ok = rc == 0
    return {"verdict": "CONFIRMED" if ok else "REFUTED", "confirmed": ok,
            "evidence": f"`{cmd}` exited {rc} — {tail}", "method": "deterministic"}


def _observations(agent):
    """The verifier's OWN command outputs — the only evidence the judge may see.
    Last ~6000 chars of the agent's Observation messages."""
    obs = [m.get("content", "") for m in agent.messages
           if m.get("role") == "user" and "Observation:" in m.get("content", "")]
    return "\n\n".join(obs)[-6000:]


def _majority(votes):
    """The mode of the votes, but only if at least 2 agree (self-consistency); a
    1-1-1 split (or too few votes) is honestly UNKNOWN, never silently REFUTED."""
    if not votes:
        return "UNKNOWN"
    tok, n = Counter(votes).most_common(1)[0]
    return tok if n >= 2 else "UNKNOWN"


def _vote(backend, evidence, k=3):
    """Sample the verdict k times at temperature 0.8 over the SAME distilled
    evidence and majority-vote the token — self-consistency gives small models
    their largest accuracy lift exactly here, over evidence already gathered."""
    prompt = [{"role": "system", "content": JUDGE_SYSTEM},
              {"role": "user", "content": f"EVIDENCE (the verifier's own command outputs):\n{evidence or '(no evidence was gathered)'}\n\nGiven ONLY this evidence, output your one-line verdict."}]
    votes = []
    for _ in range(k):
        try:
            raw = backend.chat(prompt, temperature=0.8)
        except Exception:
            continue
        m = re.search(r"VERDICT:\s*(CONFIRMED|REFUTED)", raw, re.I)
        if m:
            votes.append(m.group(1).upper())
    return _majority(votes)


def _gather_and_vote(model, work, claim):
    """Run an independent verify agent to gather evidence, then vote a verdict from
    its own observations. Returns (verdict, evidence)."""
    from .agent import Agent
    backend = make_backend(model)
    sess = sessmod.EphemeralSession(work, model)
    agent = Agent(backend, sess, max_steps=18, autonomous=True,
                  system=VERIFY_SYSTEM, allowed={"bash", "read_file", "list_files", "say"})
    try:
        agent.send(f'{_seed(work)}\n\nThe claim to verify: "{claim[:800]}"\n\n'
                   f'There is no test suite the harness could detect and scope — inspect and reason carefully.')
    except Exception:
        pass
    evidence = _observations(agent)
    return _vote(backend, evidence), evidence


def verify(claim, cwd, models, files=None):
    """Verify a completion claim on an ISOLATED COPY of the repo.

    Deterministic core: the harness detects and RUNS the real test command —
    scoped to the claim's edited files — and grounds the verdict in the actual
    exit code, never the model's judgment.
    Model fallback: when there is no runnable/scoped suite, an independent agent
    gathers evidence and the verdict is decided by self-consistency voting
    (k=3 @ 0.8) over that evidence, escalating one ladder rung on UNKNOWN. An
    honest UNKNOWN is reported as UNKNOWN — never collapsed into REFUTED.

    `models` is the checker ladder (cheapest→strongest); `files` scopes the run.
    """
    import subprocess

    if isinstance(models, str):          # tolerate a single spec / comma ladder
        models = [m.strip() for m in models.split(",") if m.strip()]
    models = models or ["gemma2:9b"]

    work, cleanup = _isolate(cwd)
    try:
        scoped = None
        if files:
            scoped = []
            for f in files:
                try:
                    rel = os.path.relpath(f, cwd)
                except ValueError:
                    continue
                if not rel.startswith(".."):
                    scoped.append(os.path.join(work, rel))
        cmd = detect_test_cmd(work, files=scoped)
        if cmd:
            # start_new_session so a timeout kills the whole PROCESS GROUP — with
            # shell=True the `timeout=` kill would otherwise hit only `sh -c` and orphan
            # the real test process (pytest/node/go), leaking CPU. On timeout, fall
            # through to model reasoning rather than crash the verify pass.
            import signal
            p = subprocess.Popen(cmd, cwd=work, shell=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                out, err = p.communicate(timeout=180)
                det = _deterministic_verdict(cmd, p.returncode, (out or "") + (err or ""))
                if det is not None:
                    return det
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                p.wait()
            # pytest collected nothing / timed out → fall through to model reasoning

        verdict, evidence = _gather_and_vote(models[0], work, claim)
        if verdict == "UNKNOWN" and len(models) > 1:
            verdict, evidence = _gather_and_vote(models[1], work, claim)
    finally:
        cleanup()
    return {"verdict": verdict, "confirmed": verdict == "CONFIRMED",
            "evidence": (f"model vote → {verdict}; " + (evidence or "").strip())[:300],
            "method": "model"}


# ---- LEARN ------------------------------------------------------------------
EXTRACT_SYSTEM = """You mine one coding session's work for DURABLE facts a DIFFERENT agent in this same repo would benefit from and could NOT easily guess: the real test/build command, a convention, a gotcha, where a key thing lives, a required order of steps. Drop generic advice, task restatements, and obvious things. 0-5 facts, each one concrete sentence. Output ONLY a JSON array of strings."""


def _learn_path(cwd):
    return os.path.join(LEARN_DIR, re.sub(r"[^A-Za-z0-9]", "-", cwd) + ".jsonl")


MAX_FACTS = 50            # per-repo cap; oldest-unverified evicted first


def _load_records(cwd):
    """Every stored LEARN v2 record for this repo, unsorted, as dicts."""
    p = _learn_path(cwd)
    if not os.path.exists(p):
        return []
    out = []
    for line in slurp(p).splitlines():
        try:
            d = json.loads(line)
            if d.get("fact"):
                out.append(d)
        except json.JSONDecodeError:
            pass
    return out


def _write_records(cwd, records):
    with open(_learn_path(cwd), "w") as f:
        for d in records:
            f.write(json.dumps(d) + "\n")


def learnings(cwd):
    """Learned facts for this repo as LEARN v2 records, VERIFIED-FIRST then newest.
    (Records are dicts: {fact, kind, sid, ts, verified, last_confirmed, ...}.)"""
    recs = _load_records(cwd)
    recs.sort(key=lambda d: (0 if d.get("verified") else 1, -(d.get("ts") or 0)))
    return recs


# ---- LEARN v2: classify → validate → supersede ------------------------------
_CMD_RUNNERS = frozenset((
    "npm", "pnpm", "yarn", "pytest", "py.test", "python", "python3", "make", "go",
    "cargo", "ruby", "rake", "node", "tox", "bash", "sh", "mvn", "gradle", "poetry",
    "pip", "deno", "bun", "phpunit", "dotnet", "ctest", "cmake", "just", "task"))


def _is_command(s):
    s = (s or "").strip()
    if not s:
        return False
    head = s.split()[0].rsplit("/", 1)[-1]
    return head in _CMD_RUNNERS or s.startswith("./")


def _extract_command(fact):
    """The runnable command a fact asserts, or None. Prefers a backticked command
    (`npm test`); falls back to an explicit 'command is X' phrasing."""
    for m in re.finditer(r"`([^`]+)`", fact):
        if _is_command(m.group(1)):
            return m.group(1).strip()
    m = re.search(r"\b(?:command|tests?|build|lint|check)\b[^`\n]{0,40}?\bis\b[:\s]+([^.;\n]+)", fact, re.I)
    if m:
        cand = m.group(1).strip().strip("`\"'")
        if _is_command(cand):
            return cand
    return None


def _extract_path(fact):
    """A filesystem path the fact references, or None (a dir/file with a real
    extension — version numbers like 3.10 are excluded by the alpha-led suffix)."""
    m = re.search(r"([\w.\-/]*/[\w.\-]+\.[A-Za-z][\w]*)", fact)     # dir/.../file.ext
    if m:
        return m.group(1)
    m = re.search(r"\b([\w\-]+\.[A-Za-z][\w]*)\b", fact)            # bare file.ext
    if m:
        return m.group(1)
    return None


def _classify(fact):
    """(kind, payload): 'executable'+cmd, 'path'+path, or 'note'+None."""
    cmd = _extract_command(fact)
    if cmd:
        return "executable", cmd
    path = _extract_path(fact)
    if path:
        return "path", path
    return "note", None


def _norm_cmd(c):
    return re.sub(r"\s+", " ", (c or "").strip())


def _fact_key(fact, kind=None, payload=None):
    """Normalized supersede key. Two executable/path facts that share the same
    surrounding phrasing but a DIFFERENT command/path contradict → same key, so the
    newer one replaces the older (e.g. 'the test command is `npm test`' vs
    '… `pytest -q`')."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    if kind == "executable" and payload:
        return "cmd:" + norm(fact.replace(payload, ""))
    if kind == "path" and payload:
        return "path:" + norm(fact.replace(payload, ""))
    return "note:" + norm(fact)


def _run_fact(cmd, cwd):
    """Run a claimed command ONCE in an isolated copy of the repo (reusing verify()'s
    _isolate). Returns True on exit 0, False on any nonzero/timeout — never raises.
    A False here means UNVERIFIED (the suite may be legitimately red), not 'discard'."""
    import subprocess
    if not os.path.isdir(cwd):
        return None
    work, cleanup = _isolate(cwd)
    try:
        p = subprocess.run(cmd, cwd=work, shell=True, capture_output=True, text=True, timeout=60)
        return p.returncode == 0
    except Exception:
        return False
    finally:
        cleanup()


def _validate_fact(fact, cwd, detected):
    """Classify a fact and, when it is checkable, set its verified bit deterministically.
    Executable facts: cross-check against detect_test_cmd for FREE first (agreement →
    verified without ever running); otherwise run once. Path facts: existence check.
    Notes stay verified=None."""
    kind, payload = _classify(fact)
    verified, method = None, None
    if kind == "executable":
        if detected and _norm_cmd(payload) == _norm_cmd(detected):
            verified, method = True, "detect"          # agrees with the deterministic detector
        else:
            verified, method = _run_fact(payload, cwd), "run"
    elif kind == "path":
        verified, method = os.path.exists(os.path.join(cwd, payload)), "exists"
    return kind, payload, verified, method


def _cap_records(records):
    """Enforce MAX_FACTS, evicting oldest-UNVERIFIED first so verified facts survive."""
    if len(records) <= MAX_FACTS:
        return records
    verified = [r for r in records if r.get("verified")]
    rest = [r for r in records if not r.get("verified")]
    rest.sort(key=lambda r: r.get("ts") or 0, reverse=True)   # newest unverified first
    keep_rest = rest[:max(0, MAX_FACTS - len(verified))]
    return (verified + keep_rest)[:MAX_FACTS]


def _store_learnings(cwd, facts, sid):
    """Validate, dedupe, and supersede fresh facts into the repo's LEARN store.
    Returns the list of fresh fact STRINGS (the daemon's learn_pass shares these)."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    existing = _load_records(cwd)
    have = {norm(d["fact"]) for d in existing}
    try:
        detected = detect_test_cmd(cwd) if os.path.isdir(cwd) else None
    except Exception:
        detected = None
    by_key, order = {}, []
    for d in existing:
        k = d.get("key") or _fact_key(d["fact"], d.get("kind"), d.get("cmd") or d.get("path"))
        if k not in by_key:
            order.append(k)
        by_key[k] = d
    fresh, now = [], time.time()
    for fact in facts:
        if not fact or norm(fact) in have:
            continue
        have.add(norm(fact))
        kind, payload, verified, method = _validate_fact(fact, cwd, detected)
        k = _fact_key(fact, kind, payload)
        rec = {"fact": fact, "kind": kind, "sid": sid, "ts": now,
               "verified": verified, "last_confirmed": now if verified else None,
               "method": method, "key": k}
        if kind == "executable" and payload:
            rec["cmd"] = payload
        elif kind == "path" and payload:
            rec["path"] = payload
        if k not in by_key:
            order.append(k)
        by_key[k] = rec                # supersede any prior fact under this key
        fresh.append(fact)
    if fresh:
        _write_records(cwd, _cap_records([by_key[k] for k in order]))
    return fresh


def forget(cwd, pattern=None):
    """Prune learned facts. With `pattern`, drop facts whose text contains it
    (case-insensitive); without, clear the repo's store. Returns the count removed."""
    recs = _load_records(cwd)
    if not recs:
        return 0
    if pattern:
        pat = pattern.lower()
        kept = [r for r in recs if pat not in r.get("fact", "").lower()]
    else:
        kept = []
    removed = len(recs) - len(kept)
    if removed:
        _write_records(cwd, kept)
    return removed


def harvest(sid, cwd, model):
    work = recent_work(sid)
    if len(work) < 200:
        return []
    from .agent import Agent
    backend = make_backend(model)
    sess = sessmod.EphemeralSession(cwd, model)
    agent = Agent(backend, sess, max_steps=2, system=EXTRACT_SYSTEM, allowed={"say"})
    out = agent.send(f"The session's recent work:\n\"\"\"\n{work}\n\"\"\"")
    m = re.search(r"\[[\s\S]*\]", out)
    if not m:
        return []
    try:
        facts = [x for x in json.loads(m.group(0)) if isinstance(x, str) and x.strip()][:5]
    except json.JSONDecodeError:
        return []
    return _store_learnings(cwd, facts, sid)
