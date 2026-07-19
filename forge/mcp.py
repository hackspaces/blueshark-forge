"""Minimal stdlib stdio MCP client (P9.1).

One subprocess per configured server, newline-delimited JSON-RPC 2.0 over
stdin/stdout — no third-party deps, matching forge's stdlib-only rule. This module
owns only the transport, the initialize→tools/list handshake, and tools/call. Turning
the discovered tools into grammar-forced `mcp__<server>__<tool>` forge actions is the
next step (the agent integration); this layer stays agent-agnostic.

v1 scope: stdio transport only — no HTTP/SSE, no sampling, no resources. Servers spawn
lazily on first use and are terminated at interpreter exit, like tools._BG_PROCS. A
single reader thread routes responses by id, so the transport is correct without
select() on a pipe (which does not work on Windows — and forge now targets native
Windows).
"""
import json
import os
import subprocess
import threading

from . import __version__

# Per-RPC wall-clock deadline. Mirrors tools.BASH_TIMEOUT: a hung server fails fast
# instead of blocking the agent loop for minutes.
MCP_TIMEOUT = int(os.environ.get("FORGE_MCP_TIMEOUT", "60"))
PROTOCOL_VERSION = "2025-06-18"   # MCP protocol revision forge advertises on initialize

_SERVERS = []          # every started MCPServer, for one shared atexit cleanup
_ATEXIT_REGISTERED = False


class MCPError(Exception):
    """A server failed to start, timed out, died, or returned a protocol error."""


class MCPServer:
    """A lazily-spawned MCP server over stdio. Thread-safe for serialized use by the
    agent loop (one tool call per step)."""

    def __init__(self, name, command, env=None, allow=None):
        self.name = name
        self.command = list(command)
        self.env = env or {}
        self.allow = set(allow) if allow else None   # optional per-server tool allowlist
        self.proc = None
        self.tools = []          # [{name, description, inputSchema, annotations}], post-allowlist
        self._id = 0
        self._pending = {}       # request id -> (Event, result-holder dict)
        self._start_lock = threading.Lock()
        self._started = False
        self._alive = False
        self._reader = None      # the stdout reader thread, joined on stop()

    # ---- lifecycle -------------------------------------------------------------
    def start(self):
        """Spawn the subprocess and run the handshake. Idempotent and thread-safe;
        a failure raises MCPError and leaves the server unstarted."""
        with self._start_lock:
            if self._started:
                return
            try:
                self.proc = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,      # servers chatter on stderr; ignore it
                    env={**os.environ, **self.env},
                    text=True, bufsize=1,           # line-buffered text pipes
                    start_new_session=True,         # own process group for a clean group-kill
                )
            except (OSError, ValueError) as e:
                raise MCPError(f"could not start MCP server {self.name!r}: {e}")
            self._alive = True
            self._reader = threading.Thread(
                target=self._read_loop, name=f"mcp-{self.name}", daemon=True)
            self._reader.start()
            _register(self)
            try:
                self._handshake()
            except MCPError:
                self.stop()
                raise
            self._started = True

    def _handshake(self):
        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "forge", "version": __version__},
        })
        self._notify("notifications/initialized")
        listed = self._request("tools/list", {}) or {}
        tools = listed.get("tools", []) or []
        if self.allow is not None:
            tools = [t for t in tools if t.get("name") in self.allow]
        self.tools = tools

    def stop(self):
        """Terminate the server. Cross-platform: POSIX group-kill, else proc.terminate()
        (Windows has no os.killpg/os.getpgid — the exact gap that crashed forge's own
        teardown before it was guarded)."""
        p = self.proc
        self._alive = False
        for ev, holder in list(self._pending.values()):   # unblock any in-flight callers
            holder.setdefault("err", f"MCP server {self.name!r} stopped")
            ev.set()
        if not p or p.poll() is not None:
            return
        try:
            if p.stdin:
                p.stdin.close()
        except OSError:
            pass
        try:
            import signal
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)   # POSIX process group
        except AttributeError:
            p.terminate()                                   # native Windows
        except (OSError, ProcessLookupError):
            pass
        # Reap the child so it isn't left unwaited (Python warns "subprocess still
        # running" at interpreter exit otherwise), then release the pipe fds. The
        # reader thread sees EOF once the process dies; join it before closing its
        # stream so the close can't race an in-flight readline().
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        r = self._reader
        if r is not None:
            r.join(timeout=2)
        if p.stdout:
            try:
                p.stdout.close()
            except OSError:
                pass

    # ---- JSON-RPC transport ----------------------------------------------------
    def _read_loop(self):
        """One reader for the server's lifetime: parse each stdout line, route a
        response to its waiting caller by id. Non-JSON lines (servers that log to
        stdout) and unmatched ids (notifications) are skipped, not fatal."""
        out = self.proc.stdout
        while True:
            line = out.readline()
            if line == "":                       # EOF — the server exited
                self._alive = False
                for ev, holder in list(self._pending.values()):
                    holder.setdefault("err", f"MCP server {self.name!r} exited")
                    ev.set()
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue                         # a log line, not JSON-RPC — skip
            if not isinstance(msg, dict):
                continue
            rid = msg.get("id")
            entry = self._pending.get(rid)
            if entry is None:
                continue                         # a notification or an id we're not awaiting
            ev, holder = entry
            if "error" in msg:
                err = msg.get("error") or {}
                holder["err"] = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            else:
                holder["ok"] = msg.get("result", {})
            ev.set()

    def _notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _request(self, method, params):
        """Send one request and block for its response up to MCP_TIMEOUT."""
        if not self._alive:
            raise MCPError(f"MCP server {self.name!r} is not running")
        self._id += 1
        rid = self._id
        ev, holder = threading.Event(), {}
        self._pending[rid] = (ev, holder)
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            if not ev.wait(timeout=MCP_TIMEOUT):
                raise MCPError(f"MCP server {self.name!r} timed out on {method} after {MCP_TIMEOUT}s")
            if "err" in holder:
                raise MCPError(holder["err"])
            return holder.get("ok", {})
        finally:
            self._pending.pop(rid, None)

    def _send(self, msg):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._alive = False
            raise MCPError(f"MCP server {self.name!r} pipe closed: {e}")

    # ---- tool call -------------------------------------------------------------
    def call(self, tool, args):
        """Invoke a tool. Returns (text, is_error): text is the concatenation of the
        response's text content blocks (non-text blocks are noted by type); is_error
        maps the protocol's isError so the agent's fail-accounting/escalation sees it."""
        self.start()
        res = self._request("tools/call", {"name": tool, "arguments": args or {}})
        if not isinstance(res, dict):
            return str(res), False
        parts = []
        for b in res.get("content", []) or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            else:
                parts.append(f"[{b.get('type', 'unknown')} content]")
        text = "\n".join(p for p in parts if p) or "(no content)"
        return text, bool(res.get("isError"))


def load_servers(cfg):
    """Build (unstarted) MCPServer instances from the config `mcp` block:
        {"mcp": {"github": {"command": ["npx","-y","@modelcontextprotocol/server-github"],
                            "env": {...}, "allow": ["create_issue", ...]}}}
    A spec without a command is skipped."""
    servers = {}
    for name, spec in (cfg.get("mcp") or {}).items():
        if not isinstance(spec, dict):
            continue
        command = spec.get("command")
        if not command:
            continue
        servers[name] = MCPServer(name, command, env=spec.get("env"), allow=spec.get("allow"))
    return servers


def connect(cfg, warn=None):
    """Load the configured servers, start + discover each, and return the ones that came
    up ({name: started MCPServer}) — ready to hand to Agent(mcp_servers=...). A server
    that fails to start is skipped (with an optional warn(name, msg) callback), never
    fatal: a broken MCP config must not stop the agent from running at all."""
    up = {}
    for name, server in load_servers(cfg).items():
        try:
            server.start()
            up[name] = server
        except MCPError as e:
            if warn:
                warn(name, str(e))
    return up


def _register(server):
    global _ATEXIT_REGISTERED
    _SERVERS.append(server)
    if not _ATEXIT_REGISTERED:
        import atexit
        atexit.register(_kill_all)
        _ATEXIT_REGISTERED = True


def _kill_all():
    for s in list(_SERVERS):
        try:
            s.stop()
        except Exception:
            pass
