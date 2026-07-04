"""Session state: transcript, registry, and inbox — the surfaces fleet plugs into.

Every forge session:
  - appends a transcript to ~/.forge/sessions/<id>.jsonl (fleet reads this)
  - registers itself in ~/.forge/registry.json while alive (fleet discovers it)
  - runs a localhost inbox HTTP server so other sessions / the daemon can inject
    messages, which the agent loop picks up between steps.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FORGE = os.path.expanduser("~/.forge")
SESSIONS = os.path.join(FORGE, "sessions")
REGISTRY = os.path.join(FORGE, "registry.json")
os.makedirs(SESSIONS, exist_ok=True)


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
        self._inbox = []
        self._lock = threading.Lock()

    # ---- transcript ----
    def log(self, kind, **fields):
        rec = {"ts": time.time(), "type": kind, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- inbox ----
    def push(self, sender, text):
        with self._lock:
            self._inbox.append({"from": sender, "text": text})

    def drain(self):
        with self._lock:
            msgs, self._inbox = self._inbox, []
        return msgs

    def start_inbox(self):
        session = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/ping":
                    self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n).decode("utf-8", "replace")
                sender = self.headers.get("X-Forge-From", "peer")
                session.push(sender, body)
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    # ---- registry ----
    def register(self):
        entries = [e for e in _load_registry() if e.get("pid") != os.getpid() and _pid_alive(e.get("pid", -1))]
        entries.append({
            "sid": self.sid, "cwd": self.cwd, "name": self.name, "model": self.model,
            "pid": os.getpid(), "port": self.port, "status": self.status,
            "startedAt": time.time(),
        })
        _save_registry(entries)

    def set_status(self, status):
        self.status = status
        self.register()

    def deregister(self):
        entries = [e for e in _load_registry() if e.get("pid") != os.getpid()]
        try:
            _save_registry(entries)
        except OSError:
            pass


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
