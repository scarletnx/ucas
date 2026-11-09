#!/usr/bin/env python3
"""A small client for `ucas --json` (standard library only).

    from ucas_client import Ucas
    with Ucas("./ucas") as u:
        r = u.call("integrate", f="x^2 sin(x)", var="x", frm="0", to="pi")
        print(r["result"]["definite"]["value"])

The process is started once and keeps its state (definitions) until it
is closed or `reset`.  Each call sends one JSON line and reads one back.

An "ambiguous" error lists the readings of the input and the key that
picks one (args.assume).  `call` can resolve these itself through a
policy: a function from the error object to a reading (or None to give
up), which is how an agent loop answers the questions a terminal user
would be asked.  Run this file to see a short session:

    python3 tools/ucas_client.py [path/to/ucas]
"""

import json
import os
import subprocess
import sys


class UcasError(Exception):
    def __init__(self, response):
        self.response = response
        err = response.get("error", {})
        super().__init__("%s: %s" % (err.get("code"), err.get("message")))


class Ucas:
    def __init__(self, binary="ucas", outdir=None, files="write"):
        argv = [binary, "--json", "--files", files]
        if outdir:
            argv += ["--dir", outdir]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True,
                                     encoding="utf-8")
        self.next_id = 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait()

    def request(self, cmd, args=None):
        """One request and its response, as dictionaries."""
        req = {"id": self.next_id, "cmd": cmd}
        if args:
            req["args"] = args
        self.next_id += 1
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("ucas exited")
        resp = json.loads(line)
        assert resp["id"] == req["id"], "responses come back in order"
        return resp

    def call(self, cmd, policy=None, max_questions=4, **args):
        """A request; ambiguities are answered by policy(error) and the
        request is repeated with args.assume.  Raises UcasError when the
        answer is an error.  `frm` stands for the keyword-like `from`."""
        if "frm" in args:
            args["from"] = args.pop("frm")
        args = {k: v for k, v in args.items() if v is not None}
        for _ in range(max_questions + 1):
            resp = self.request(cmd, args)
            err = resp.get("error")
            if resp["ok"]:
                return resp
            if err and err["code"] == "ambiguous" and policy:
                pick = policy(err)
                if pick is not None:
                    args.setdefault("assume", {})[err["key"]] = pick
                    continue
            raise UcasError(resp)
        raise UcasError(resp)

    def schema(self):
        return self.request("schema")["result"]


def take_default(err):
    """A policy: the reading the text frontends would take."""
    for r in err.get("readings", []):
        if r.get("default"):
            return r["index"]
    return None


def demo(binary):
    def show(title, resp):
        print("--", title)
        print(json.dumps(resp.get("result"), indent=2)[:1200])
        for c in resp.get("conditions", []):
            print("   holds where", c)
        for c in resp.get("checks", []):
            print("   check:", c["kind"], "passed" if c["passed"] else "FAILED")

    with Ucas(binary, files="none") as u:
        cmds = [c["cmd"] for c in u.schema()["commands"]]
        print("%d commands, e.g. %s" % (len(cmds), ", ".join(cmds[:8])))

        show("integrate", u.call("integrate", f="x^2 sin(x)", var="x",
                                 frm="0", to="pi"))
        show("solve", u.call("solve", equations=["x + y = 3", "x - y = 1"]))

        # define a function once, use it by name afterwards
        u.call("define", name="isqrt4",
               params=[{"name": "x", "type": "u4"}], body="isqrt(x)")
        show("gates", u.call("gates", f="isqrt4"))

        # an ambiguity: "z" is complex unless x or y appears.  Without a
        # policy the error says how to choose; with one it is answered.
        try:
            u.call("zeros", f="z^2 + 1")
        except UcasError as e:
            err = e.response["error"]
            print("-- ambiguous:", err["message"])
            for r in err["readings"]:
                print("   %d) %s%s" % (r["index"], r["reading"],
                                      "  (default)" if r["default"] else ""))
        show("zeros, z complex", u.call("zeros", f="z^2 + 1",
                                        policy=take_default))

        # the AST form needs no precedence or juxtaposition rules
        expr = {"op": "-", "args": [{"op": "^", "args": [{"var": "x"}, 6]},
                                    {"num": "1"}]}
        show("factor", u.call("factor", f=expr))
        print("-- state:", [f["signature"] for f in
                             u.call("state")["result"]["functions"]])


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    demo(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(here, "..", "ucas"))
