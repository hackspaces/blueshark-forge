"""Bridge to the Claude Code fleet — one network across both runtimes.

Claude Code sessions on this machine coordinate through ~/.claude/fleet:
a registry (inbox.json), a shared token, and a localhost HTTP inbox per
session (POST /send with X-Fleet-* headers). Forge speaks that exact wire
protocol, in both directions:

  OUT  forge's fleet_send / `forge send` can target a Claude Code session —
       we POST into its inbox and the message lands in its conversation as a
       <channel source="fleet"> event.
  IN   every forge session registers itself in inbox.json (kind: "forge")
       and accepts the Claude fleet's POST /send, so Claude Code sessions
       discover and message forge sessions with their normal fleet_send.

If ~/.claude/fleet doesn't exist (no Claude Code fleet on this machine),
every function degrades to a no-op and forge's native fleet works alone.
"""
import contextlib
import json
import os
import time
import urllib.request

try:
    import fcntl
except ImportError:
    fcntl = None

DIR = os.path.expanduser("~/.claude/fleet")
INBOX = os.path.join(DIR, "inbox.json")
TOKEN_FILE = os.path.join(DIR, "token")
CLAUDE_PROJECTS = os.path.expanduser("~/.claude/projects")


def available():
    return os.path.exists(TOKEN_FILE)


def token():
    try:
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    except OSError:
        return None


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _read_inbox():
    try:
        with open(INBOX) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _write_inbox(entries):
    tmp = INBOX + f".{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=1)
    os.replace(tmp, INBOX)


@contextlib.contextmanager
def _inbox_lock():
    """Serialize the inbox read-modify-write across processes (forge + Claude Code),
    so concurrent registrations don't lose an entry. os.replace prevents a torn READ;
    this flock prevents a lost UPDATE. No-op where fcntl is unavailable / dir missing."""
    if fcntl is None:
        yield
        return
    try:
        os.makedirs(DIR, exist_ok=True)
        f = open(INBOX + ".lock", "w")
    except OSError:
        yield
        return
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def claude_peers():
    """Live Claude Code sessions, normalized to forge registry shape."""
    peers = []
    for e in _read_inbox():
        if e.get("kind") == "forge" or not _pid_alive(e.get("pid")):
            continue
        peers.append({
            "sid": str(e.get("sessionId")), "name": e.get("name") or "?",
            "cwd": e.get("cwd") or "", "port": e.get("port"), "pid": e.get("pid"),
            "kind": "claude",
        })
    return peers


def register(session):
    """Announce a forge session to the Claude Code fleet (best-effort)."""
    if not available() or not session.port:
        return
    try:
        with _inbox_lock():
            entries = [e for e in _read_inbox() if e.get("pid") != os.getpid() and _pid_alive(e.get("pid"))]
            entries.append({
                "pid": os.getpid(), "cwd": session.cwd, "sessionId": session.sid,
                "name": session.name, "port": session.port, "startedAt": time.time() * 1000,
                "kind": "forge",
            })
            _write_inbox(entries)
    except OSError:
        pass


def unregister():
    if not available():
        return
    try:
        with _inbox_lock():
            _write_inbox([e for e in _read_inbox() if e.get("pid") != os.getpid()])
    except OSError:
        pass


def send(peer, text, sender_name, sender_cwd="", sender_sid=""):
    """POST into a Claude Code session's fleet inbox. Raises on failure."""
    tok = token()
    if not tok:
        raise SystemExit("Claude Code fleet not available on this machine")
    req = urllib.request.Request(
        f"http://127.0.0.1:{peer['port']}/send", data=text.encode(),
        headers={"X-Fleet-Token": tok, "X-Fleet-From": f"forge/{sender_name}",
                 "X-Fleet-From-Cwd": sender_cwd, "X-Fleet-From-Session": sender_sid})
    urllib.request.urlopen(req, timeout=5).read()


def doctor(create_token=True):
    """Check — and where safe, prepare — Claude Code interop on this machine.
    Returns printable status lines. Run by `forge setup`; degrades gracefully
    on computers without Claude Code."""
    import shutil
    lines = []
    claude_cli = shutil.which("claude")
    if not claude_cli and not os.path.isdir(os.path.expanduser("~/.claude")):
        return ["○ Claude Code not found — cross-runtime fleet off (forge's native fleet still works)"]
    lines.append(f"✓ Claude Code detected ({claude_cli or '~/.claude'})")
    if create_token and not os.path.exists(TOKEN_FILE):
        try:
            import secrets
            os.makedirs(DIR, exist_ok=True)
            fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_hex(24))   # same shape the channel server generates
        except OSError:
            pass
    lines.append("✓ shared fleet token ready (~/.claude/fleet/token)" if os.path.exists(TOKEN_FILE)
                 else "✗ couldn't create ~/.claude/fleet/token — check permissions")
    channel = os.path.join(DIR, "fleet-channel.mjs")
    if os.path.exists(channel):
        registered = False
        try:
            with open(os.path.expanduser("~/.claude.json")) as f:
                registered = "fleet-channel" in f.read()
        except OSError:
            pass
        lines.append("✓ Claude fleet channel installed" if registered else
                     "△ fleet-channel.mjs present but not registered — run: claude mcp add fleet -- node ~/.claude/fleet/fleet-channel.mjs")
    else:
        lines.append("△ Claude fleet channel not installed — Claude Code sessions can't exchange fleet messages yet;")
        lines.append("  forge sessions still register in the shared inbox and become reachable the moment it's added")
    n = len(claude_peers())
    lines.append(f"✓ {n} Claude Code session(s) reachable right now" if n
                 else "○ no Claude Code sessions reachable right now")
    return lines


def summarize(peer):
    """Tail a Claude Code session's transcript for board display:
    (task title, user's last words, Claude's last words)."""
    import re
    path = os.path.join(CLAUDE_PROJECTS, re.sub(r"[^A-Za-z0-9]", "-", peer["cwd"]),
                        f"{peer['sid']}.jsonl")
    info = {"title": None, "prompt": None, "claude": None}
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > 400_000:
                f.seek(size - 400_000); f.readline()
            for line in f:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                t = d.get("type")
                if t == "ai-title":
                    info["title"] = d.get("aiTitle")
                elif t == "last-prompt":
                    info["prompt"] = d.get("lastPrompt")
                elif t == "assistant":
                    content = d.get("message", {}).get("content", [])
                    texts = [c.get("text", "") for c in content
                             if isinstance(c, dict) and c.get("type") == "text"]
                    if texts and texts[-1].strip():
                        info["claude"] = " ".join(texts[-1].split())
    except OSError:
        pass
    return info
