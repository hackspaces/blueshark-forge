"""Session state: transcript, registry, and inbox — the surfaces fleet plugs into.

Every forge session:
  - appends a transcript to ~/.forge/sessions/<id>.jsonl (fleet reads this)
  - registers itself in ~/.forge/registry.json while alive (fleet discovers it)
  - runs a localhost inbox HTTP server so other sessions / the daemon can inject
    messages, which the agent loop picks up between steps.
"""
import contextlib
import json
import secrets
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import fcntl
except ImportError:  # non-Unix; locking becomes a no-op
    fcntl = None

FORGE = os.path.expanduser("~/.forge")
SESSIONS = os.path.join(FORGE, "sessions")
REGISTRY = os.path.join(FORGE, "registry.json")
_LOCK = os.path.join(FORGE, "registry.lock")
os.makedirs(SESSIONS, exist_ok=True)
try:
    os.chmod(FORGE, 0o700)   # transcripts + tokens are private
except OSError:
    pass


def _nonblock(fd):
    """Set O_NONBLOCK on a fd, so reads raise EAGAIN instead of blocking and
    writes never stall the caller. No-op where fcntl is unavailable (non-Unix)."""
    if fcntl is None:
        return
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except OSError:
        pass


@contextlib.contextmanager
def _registry_lock():
    """Serialize the registry read-modify-write across processes, so concurrent
    sessions don't clobber each other's entries."""
    if fcntl is None:
        yield
        return
    f = open(_LOCK, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _load_registry():
    try:
        with open(REGISTRY) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _save_registry(entries):
    tmp = REGISTRY + f".{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=1)
    os.chmod(tmp, 0o600)          # holds per-session inbox tokens — keep it private
    os.replace(tmp, REGISTRY)


class Session:
    def __init__(self, sid, cwd, model, name=None):
        self.sid = sid
        self.cwd = cwd
        self.model = model
        self.name = name or os.path.basename(cwd) or "forge"
        self.path = os.path.join(SESSIONS, f"{sid}.jsonl")
        self.status = "idle"
        self.port = None
        self.token = secrets.token_hex(16)   # gates the inbox against other local processes
        self._inbox = []
        self._lock = threading.Lock()
        # Wake pipe: push() writes a byte so an idle REPL blocked in select() wakes
        # the instant a fleet message lands, instead of rotting until the human types.
        try:
            self._wake_r, self._wake_w = os.pipe()
            _nonblock(self._wake_r); _nonblock(self._wake_w)
        except OSError:
            self._wake_r = self._wake_w = None

    @property
    def wake_fd(self):
        """The read end of the wake pipe — select() on it to learn an inbox
        message arrived. None where os.pipe() was unavailable."""
        return self._wake_r

    # ---- transcript ----
    def log(self, kind, **fields):
        rec = {"ts": time.time(), "type": kind, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- inbox ----
    def push(self, sender, text):
        # May be called from the inbox HTTP thread — append under the lock, then
        # wake outside it. A full/broken pipe is harmless: the reader wakes anyway.
        with self._lock:
            self._inbox.append({"from": sender, "text": text})
        if self._wake_w is not None:
            try:
                os.write(self._wake_w, b"x")
            except (BlockingIOError, BrokenPipeError, OSError):
                pass

    def drain(self):
        with self._lock:
            msgs, self._inbox = self._inbox, []
        self._drain_wake()
        return msgs

    def _drain_wake(self):
        """Empty the wake pipe (read until EAGAIN) so a stale byte can't keep a
        drained, empty inbox spuriously readable."""
        if self._wake_r is None:
            return
        try:
            while os.read(self._wake_r, 4096):
                pass
        except (BlockingIOError, OSError):
            pass

    def start_inbox(self):
        session = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/ping":
                    ident = json.dumps({"pid": os.getpid(), "cwd": session.cwd,
                                        "sessionId": session.sid, "name": session.name,
                                        "kind": "forge"}).encode()
                    self.send_response(200); self.end_headers(); self.wfile.write(ident)
                else:
                    self.send_response(404); self.end_headers()

            def _authed(self):
                # forge protocol: this session's own token (from the 0600 registry).
                if self.headers.get("X-Forge-Token", "") == session.token:
                    return "X-Forge-From"
                # Claude Code fleet protocol: the machine-wide shared token, on /send.
                from . import bridge
                tok = bridge.token()
                if (self.path == "/send" and tok
                        and self.headers.get("X-Fleet-Token", "") == tok):
                    return "X-Fleet-From"
                return None

            def do_POST(self):
                from_header = self._authed()
                if not from_header:
                    self.send_response(403); self.end_headers(); self.wfile.write(b"forbidden"); return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                except (TypeError, ValueError):
                    self.send_response(400); self.end_headers(); return
                if n < 0 or n > 1_000_000:            # bound the read
                    self.send_response(413); self.end_headers(); return
                body = self.rfile.read(n).decode("utf-8", "replace")
                sender = "".join(c for c in self.headers.get(from_header, "peer") if c.isalnum() or c in "-_.:")[:64]
                session.push(sender, body)
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    # ---- registry ----
    def register(self):
        with _registry_lock():
            entries = [e for e in _load_registry() if e.get("pid") != os.getpid() and _pid_alive(e.get("pid", -1))]
            entries.append({
                "sid": self.sid, "cwd": self.cwd, "name": self.name, "model": self.model,
                "pid": os.getpid(), "port": self.port, "status": self.status,
                "token": self.token, "startedAt": time.time(),
            })
            _save_registry(entries)
        from . import bridge
        bridge.register(self)   # visible to Claude Code's fleet too (no-op without one)

    def set_status(self, status):
        self.status = status
        self.register()

    def deregister(self):
        try:
            with _registry_lock():
                _save_registry([e for e in _load_registry() if e.get("pid") != os.getpid()])
        except OSError:
            pass
        from . import bridge
        bridge.unregister()


class EphemeralSession:
    """A throwaway session for internal agents (e.g. the verifier). Never
    registers, serves no inbox, and stays invisible to the fleet registry so
    it can never be watched or verified itself."""
    def __init__(self, cwd, model, sid=None):
        self.sid = sid or "eph-" + uuid_hex()
        self.cwd = cwd
        self.model = model
        self.name = "ephemeral"
        self.status = "idle"
        self.port = None

    def log(self, *a, **k): pass
    def drain(self): return []
    def set_status(self, s): self.status = s
    def register(self): pass
    def deregister(self): pass


def uuid_hex():
    import uuid
    return uuid.uuid4().hex[:10]


def registry():
    """Live sessions (prunes dead pids)."""
    return [e for e in _load_registry() if _pid_alive(e.get("pid", -1))]
