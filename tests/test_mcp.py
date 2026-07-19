"""P9.1 stdlib MCP client — transport, discovery, tools/call, and the robustness
the judge review called out (skip non-JSON stdout lines, per-server allowlist, fail
fast on a hung/dead server).

Offline and stdlib-only, like the runtime: the "server" is a tiny pure-Python script
that speaks JSON-RPC 2.0 over stdin/stdout — never npx, never the network (CLAUDE.md:
tests must not hit the network or launch a real server)."""
import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import mcp   # noqa: E402


# A minimal but real stdio MCP server. It deliberately prints a NON-JSON log line to
# stdout before its first response (many real servers do) — discovery only works if the
# client skips it. `slow` sleeps so a lowered timeout trips; `boom` returns isError.
FIXTURE = textwrap.dedent('''
    import sys, json, time
    print("fixture-mcp: booting", flush=True)          # a log line on stdout, not JSON-RPC
    def send(o): sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
    TOOLS = [
        {"name":"echo","description":"echo text","inputSchema":{"type":"object"},
         "annotations":{"readOnlyHint":True}},
        {"name":"boom","description":"always errors","inputSchema":{"type":"object"}},
        {"name":"slow","description":"sleeps 5s","inputSchema":{"type":"object"}},
    ]
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        msg = json.loads(line); m = msg.get("method"); rid = msg.get("id")
        if m == "initialize":
            send({"jsonrpc":"2.0","id":rid,"result":{"protocolVersion":"2025-06-18",
                  "capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"0"}}})
        elif m == "notifications/initialized":
            pass                                          # a notification: no response
        elif m == "tools/list":
            send({"jsonrpc":"2.0","id":rid,"result":{"tools":TOOLS}})
        elif m == "tools/call":
            name = msg["params"]["name"]; args = msg["params"].get("arguments",{})
            if name == "echo":
                send({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text",
                      "text":"echo: " + args.get("text","")}]}})
            elif name == "boom":
                send({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text",
                      "text":"kaboom"}],"isError":True}})
            elif name == "slow":
                time.sleep(5); send({"jsonrpc":"2.0","id":rid,"result":{"content":[]}})
            else:
                send({"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"no tool "+name}})
        elif rid is not None:
            send({"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"method not found"}})
''')


class TestMCPClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.path = tempfile.mkstemp(suffix="_mcpfix.py")
        with os.fdopen(fd, "w") as f:
            f.write(FIXTURE)
        cls.cmd = [sys.executable, cls.path]

    @classmethod
    def tearDownClass(cls):
        try: os.remove(cls.path)
        except OSError: pass

    def setUp(self):
        self._servers = []

    def tearDown(self):
        for s in self._servers:
            s.stop()

    def _server(self, allow=None):
        s = mcp.MCPServer("fixture", self.cmd, allow=allow)
        self._servers.append(s)
        return s

    def test_handshake_and_discovery_skipping_log_lines(self):
        s = self._server()
        s.start()                                          # initialize + tools/list handshake
        names = {t["name"] for t in s.tools}
        self.assertEqual(names, {"echo", "boom", "slow"})  # the leading log line was skipped
        # readOnlyHint survived discovery (it drives mutating-vs-readonly gating in PR 2)
        echo = next(t for t in s.tools if t["name"] == "echo")
        self.assertTrue(echo["annotations"]["readOnlyHint"])

    def test_call_returns_text(self):
        s = self._server()
        text, is_error = s.call("echo", {"text": "hi"})
        self.assertEqual(text, "echo: hi")
        self.assertFalse(is_error)

    def test_is_error_is_surfaced(self):
        # protocol isError -> ok=False, so the agent's fail-accounting/escalation sees it
        s = self._server()
        text, is_error = s.call("boom", {})
        self.assertTrue(is_error)
        self.assertIn("kaboom", text)

    def test_allowlist_filters_tools(self):
        s = self._server(allow=["echo"])
        s.start()
        self.assertEqual([t["name"] for t in s.tools], ["echo"])

    def test_unknown_tool_is_an_error(self):
        s = self._server()
        s.start()
        with self.assertRaises(mcp.MCPError):
            s.call("nope", {})

    def test_hung_server_times_out_fast(self):
        s = self._server()
        s.start()
        orig = mcp.MCP_TIMEOUT
        mcp.MCP_TIMEOUT = 0.4                              # don't wait the fixture's full 5s
        try:
            with self.assertRaises(mcp.MCPError):
                s.call("slow", {})
        finally:
            mcp.MCP_TIMEOUT = orig

    def test_dead_server_raises_not_hangs(self):
        s = self._server()
        s.start()
        s.stop()
        with self.assertRaises(mcp.MCPError):
            s.call("echo", {"text": "x"})

    def test_start_is_idempotent(self):
        s = self._server()
        s.start()
        proc1 = s.proc
        s.start()                                          # second start is a no-op
        self.assertIs(s.proc, proc1)

    def test_load_servers_from_config(self):
        servers = mcp.load_servers({"mcp": {
            "fix": {"command": self.cmd, "allow": ["echo"]},
            "nocmd": {"env": {}},                          # no command -> skipped
        }})
        self.assertEqual(set(servers), {"fix"})
        self.assertEqual(servers["fix"].allow, {"echo"})
        self.assertFalse(servers["fix"]._started)          # lazy: not spawned yet


if __name__ == "__main__":
    unittest.main()
