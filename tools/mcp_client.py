#!/usr/bin/env python3
"""A minimal MCP stdio client for `ucas --mcp` (standard library only).

    python3 tools/mcp_client.py [path/to/ucas] [--flat]
    python3 tools/mcp_client.py ssh host /path/ucas      # a remote server

It does the handshake (initialize, notifications/initialized), lists the
tools, calls a few, answers an ambiguity from the error's readings, and
saves an image the server returns.  The Mcp class is reusable.
"""

import base64
import json
import os
import subprocess
import sys


class McpError(Exception):
    pass


class Mcp:
    def __init__(self, argv):
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        self.next_id = 1

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None):
        rid = self.next_id
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise McpError("the server exited")
            resp = json.loads(line)
            if resp.get("id") == rid:
                break
        if "error" in resp:
            raise McpError("%(code)s: %(message)s" % resp["error"])
        return resp["result"]

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def initialize(self):
        r = self.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "mcp_client.py", "version": "1"}})
        self.notify("notifications/initialized")
        return r

    def call(self, name, arguments, answer=None):
        """tools/call; an ambiguity is answered by answer(error) (a reading
        or None) and the call repeated with args.assume."""
        for _ in range(4):
            r = self.request("tools/call", {"name": name,
                                            "arguments": arguments})
            err = (r.get("structuredContent") or {}).get("error")
            if r.get("isError") and err and err.get("code") == "ambiguous" \
                    and answer:
                pick = answer(err)
                if pick is not None:
                    args = arguments.get("args", arguments)
                    args.setdefault("assume", {})[err["key"]] = pick
                    continue
            return r
        return r

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    argv = sys.argv[1:] or [os.path.join(here, "..", "ucas")]
    flat = "--flat" in argv
    argv = [a for a in argv if a != "--flat"] + ["--mcp"] + \
        (["--flat"] if flat else [])
    m = Mcp(argv)
    info = m.initialize()
    print("server", info["serverInfo"]["name"], "protocol",
          info["protocolVersion"])
    tools = m.request("tools/list")["tools"]
    print(len(tools), "tools:", ", ".join(t["name"] for t in tools[:16]))

    def show(title, r):
        print("--", title, "(error)" if r.get("isError") else "")
        print("  " + r["content"][0]["text"].strip().replace("\n", "\n  ")
              [:600])

    def grouped(group, op, args):
        return (op, args) if flat else (group, {"op": op, "args": args})

    show("factor", m.call(*grouped("algebra", "factor", {"f": "x^6 - 1"})))
    show("range", m.call(*grouped("analyze", "range",
                                  {"f": "isqrt(x: u4)"})))

    def default(err):
        for r in err["readings"]:
            if r["default"]:
                return r["index"]
    show("zeros of z^2 + 1, answering the ambiguity",
         m.call(*grouped("properties", "zeros", {"f": "z^2 + 1"}),
                answer=default))
    r = m.call(*grouped("plot", "plot", {"fs": "sin(x)/x"}))
    for c in r["content"]:
        if c["type"] == "image":
            with open("mcp_plot.gif", "wb") as f:
                f.write(base64.b64decode(c["data"]))
            print("-- plot: wrote mcp_plot.gif (%s)" % c["mimeType"])
    m.close()


if __name__ == "__main__":
    main()
