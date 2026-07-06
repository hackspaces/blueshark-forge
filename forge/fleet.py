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
def find_session(target):
    """Match a target across BOTH fleets: forge sessions and Claude Code
    sessions (via the bridge). Forge entries have no 'kind'; claude peers
    carry kind='claude'."""
    from . import bridge
    live = sessmod.registry() + bridge.claude_peers()
    t = target.lower()
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
    hits = find_session(target)
    if len(hits) != 1:
        raise SystemExit(f"target '{target}' matched {len(hits)}; live: {roster()}")
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


def detect_test_cmd(cwd):
    """Find the project's real test command deterministically."""
    pj = os.path.join(cwd, "package.json")
    if os.path.exists(pj):
        try:
            scripts = json.loads(slurp(pj)).get("scripts", {})
            if scripts.get("test") and "no test specified" not in scripts["test"]:
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
    if os.path.exists(os.path.join(cwd, "pyproject.toml")) or os.path.isdir(os.path.join(cwd, "tests")):
        return "pytest -q"
    if os.path.exists(os.path.join(cwd, "Cargo.toml")):
        return "cargo test"
    return None


def verify(claim, cwd, model):
    """Verify a completion claim on an ISOLATED COPY of the repo.

    Deterministic core: the harness detects and RUNS the real test command and
    grounds the verdict in the actual exit code — never the model's judgment.
    Model fallback: only when there is no detectable test suite does an
    independent agent reason about the claim.
    """
    import shutil
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp(prefix="forge-verify-")
    try:
        # argv form (no shell) — a repo path containing quotes can't inject
        subprocess.run(["rsync", "-a", "--exclude", ".git", "--exclude", "node_modules",
                        cwd.rstrip("/") + "/", tmp + "/"], timeout=120)
        nm = os.path.join(cwd, "node_modules")
        if os.path.isdir(nm):
            try:
                os.symlink(nm, os.path.join(tmp, "node_modules"))
            except OSError:
                pass

        cmd = detect_test_cmd(tmp)
        if cmd:
            p = subprocess.run(cmd, cwd=tmp, shell=True, capture_output=True, text=True, timeout=180)
            out = (p.stdout + p.stderr).strip()
            tail = " ".join(out.splitlines()[-4:])[:220]
            ok = p.returncode == 0
            return {
                "verdict": "CONFIRMED" if ok else "REFUTED",
                "confirmed": ok,
                "evidence": f"`{cmd}` exited {p.returncode} — {tail}",
                "method": "deterministic",
            }

        # no test suite → independent model reasoning as fallback
        from .agent import Agent
        backend = make_backend(model)
        sess = sessmod.EphemeralSession(tmp, model)
        agent = Agent(backend, sess, max_steps=18, autonomous=True,
                      system=VERIFY_SYSTEM, allowed={"bash", "read_file", "list_files", "say"})
        out = agent.send(f'{_seed(tmp)}\n\nThe claim to verify: "{claim[:800]}"\n\nThere is no obvious test suite — inspect and reason carefully.')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    m = re.search(r"VERDICT:\s*(CONFIRMED|REFUTED)", out, re.I)
    verdict = m.group(1).upper() if m else "UNKNOWN"
    return {"verdict": verdict, "confirmed": verdict == "CONFIRMED",
            "evidence": out.strip()[:300], "method": "model"}


# ---- LEARN ------------------------------------------------------------------
EXTRACT_SYSTEM = """You mine one coding session's work for DURABLE facts a DIFFERENT agent in this same repo would benefit from and could NOT easily guess: the real test/build command, a convention, a gotcha, where a key thing lives, a required order of steps. Drop generic advice, task restatements, and obvious things. 0-5 facts, each one concrete sentence. Output ONLY a JSON array of strings."""


def _learn_path(cwd):
    return os.path.join(LEARN_DIR, re.sub(r"[^A-Za-z0-9]", "-", cwd) + ".jsonl")


def learnings(cwd):
    p = _learn_path(cwd)
    if not os.path.exists(p):
        return []
    out = []
    for line in slurp(p).splitlines(keepends=True):
        try:
            d = json.loads(line)
            if d.get("fact"):
                out.append(d["fact"])
        except json.JSONDecodeError:
            pass
    return out


def _store_learnings(cwd, facts, sid):
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    have = {norm(f) for f in learnings(cwd)}
    fresh = [f for f in facts if f and norm(f) not in have]
    if fresh:
        with open(_learn_path(cwd), "a") as f:
            for fact in fresh:
                f.write(json.dumps({"fact": fact, "ts": time.time(), "sid": sid}) + "\n")
    return fresh


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
