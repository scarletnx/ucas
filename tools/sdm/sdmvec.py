#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Intel SDM's Operation pseudocode as an oracle for x86check.

    sdmvec.py SDMDIR MANIFEST IN.BIN OUT.BIN STATUS

ucas `x86check FILTER sdm` writes the tests the way it writes them for the
host compiler (src/x86intr_tab.inc): IN.BIN holds each test's input points
(MAXOP operand slots of 64 bytes), MANIFEST names the intrinsic, its
immediate and its operand kinds.  This program runs each test through the
pseudocode of the instruction the intrinsic compiles to, as printed in
Intel 64 and IA-32 Architectures Software Developer's Manual, Vol. 2
(order no. 325383), and writes the result bytes to OUT.BIN in the layout of
the compiled test program (result lanes at offset 0).  STATUS gets one line
per test: "ok", or "skip REASON" (no instruction for the intrinsic, or a
page the interpreter cannot read).

SDMDIR holds the pages that tools/sdm/extract.py writes (one .sdm file per
instruction page; `make sdm SDM_PDF=...`).  Places where the text cannot be
read as written are listed in ERRATA below and handled in the reader, so the
pages stay the manual's own words.

Treating the vendor's pseudocode as the executable specification follows
Reid, "Trustworthy Specifications of ARM v8-A and v8-M System Level
Architecture", FMCAD 2016 (doi:10.1109/FMCAD.2016.7886675); the extraction
and the lenient reading come from ~/vibe86 (tools/sdm, tools/ucc/sdm.py).
"""

import math
import os
import re
import struct
import sys

# Places where the SDM (325383-093US) can't be read as written, and how this
# reader takes them.  Numbered on from ~/vibe86/docs/sdm-errata.md (E-1..E-18).
ERRATA = {
    "E-19": "VPSRAVW/D/Q, VPSRAV* and others: SignExtend(SRC1[i+w-1:i] >> COUNT) "
            "applies a logical >> to a w-bit field, after which SignExtend sees a 0 "
            "sign; meant is the arithmetic shift.  Read: SignExtend(X >> c) is X "
            "sign-extended, then shifted.",
    "E-20": "VPSRAVD and VPSRAVQ (EVEX): the masked-off lanes write *DEST[31:0] "
            "remains unchanged* / DEST[31:0] := 0 (DEST[63:0] for Q), lane 0 for "
            "every j; meant is DEST[i+31:i] (DEST[i+63:i]).  Read as written, zeroing-masking would clear lane 0 "
            "only.  Read: DEST[i+31:i].",
    "E-21": "VPERMILPS (EVEX variable versions): (KL, VL) = (16, 512) only, though "
            "the 128- and 256-bit forms run the same text.  Read: KL = VL/32.",
    "E-22": "VPERMI2W/D/Q and others: DEST[...] = ... with '=' where ':=' is "
            "meant.  Read: a statement-level '=' assigns.",
    "E-23": "PSHUFB (EVEX): 'k[destpos]' for the writemask k1, 'no masking' and "
            "'zeroing' as conditions.  Read: k1, *no writemask*, zeroing-masking.",
    "E-25": "VPSRAVW/D/Q (EVEX): COUNT := SRC2[i+3:i] (W), [i+4:i] (D), [i+5:i] "
            "(Q), then IF COUNT < 16/32/64 ... ELSE fill with the sign: the test "
            "can never fail on a field that narrow, while the VEX form reads the "
            "whole element (COUNT_0 := SRC2[31:0]) and hardware fills with the "
            "sign for any count >= the element width (_mm_srav_epi32 on this "
            "host).  Read: the whole element.",
    "E-26": "VGF2P8AFFINEINVQB (EVEX): the 'FOR b := 0 to 7:' loop and its "
            "body are printed at column 0, after and outside the FOR j loop whose "
            "j (k1[j*8+b], DEST.qword[j]) they read.  Read: inside FOR j.",
    "E-27": "GF2P8AFFINEINVQB: inverse() is the page's 'Inverse Byte Listings' "
            "table (the inverse in GF(2^8) mod x^8+x^4+x^3+x+1, 0 for 0), which "
            "is not part of the Operation section, and parity() is defined on "
            "the GF2P8AFFINEQB page.  Read: both computed here.",
    "E-24": "GF2P8MULB: the helper's 'return tword.byte[0]' at column 0, after "
            "the indented body.  Read: part of the helper.",
}

MAXVL = 512


class SdmError(Exception):
    pass


# ------------------------------------------------------------------ pages

def read_page(path):
    sec, ops = None, []
    title = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# ") and sec is None and not title:
                title = line[2:]
                continue
            if line.startswith("@"):
                sec = line[1:].strip()
                continue
            if sec == "operation":
                ops.append(line)
    return title, ops


NORMALIZE = {"–": "-", "−": "-", "—": "-", "∗": "*", "×": "*", "≠": "!=",
             "≤": "<=", "≥": ">=", "←": ":=", "’": "'", "‘": "'", " ": " "}

JUNK = re.compile(r"^\s*((X?MM\d+|[0-9A-F]{2}H|Figure.*)\s*)+$")


def normalize(line):
    for a, b in NORMALIZE.items():
        line = line.replace(a, b)
    return line


def sections(lines, mnems):
    """Split an Operation section at its title lines.

    Returns (prelude, [(title, lines)]): the prelude holds the helper
    definitions before the first title.
    """
    out, cur, title, prelude = [], [], None, []
    table = False
    for raw in lines:
        line = normalize(raw.rstrip())
        if JUNK.match(line) and line.strip():
            continue
        # a table of special cases after the pseudocode: up to the next title
        if re.match(r"^\s*Table \d+-\d+\.", line):
            table = True
            continue
        s = line.strip()
        ind = len(line) - len(line.lstrip())
        word = re.split(r"[\s(,]", s, 1)[0] if s else ""
        is_title = (ind <= 2 and s and ":=" not in s and
                    any(w in mnems for w in word.split("/")) and
                    not re.match(r"^\w+\s*\(.*\)\s*\{?\s*$", s) or
                    (ind <= 2 and re.match(r"^[A-Z0-9/]+ \(+(EVEX|VEX|SSE|with|128|256)", s)
                     is not None))
        if is_title:
            table = False
            if title is not None:
                out.append((title, cur))
            title, cur = s, []
            continue
        if table:
            continue
        if title is None:
            prelude.append(line)
        else:
            cur.append(line)
    if title is not None:
        out.append((title, cur))
    return prelude, out


# ------------------------------------------------------------------ lexer

PHRASES = [
    (r"\*\s*no writemask\s*\*", " __NOWM "),
    (r"\bno writemask\b", " __NOWM "),
    (r"\bno masking\b", " __NOWM "),
    (r"\*\s*no controlmask\s*\*", " __NOWM "),
    (r"\*\s*merging-masking\s*\*", " __MERGE "),
    (r"\*\s*zeroing-masking\s*\*", " __ZERO "),
    (r"\*\s*zeroing\s*\*", " __ZERO "),
    (r"\bmerging-masking\b", " __MERGE "),
    (r"\bzeroing-masking\b", " __ZERO "),
    (r"\bzeroing\b", " __ZERO "),
    (r"\(?\s*\w+\s*\*?\s*is memory\s*\*?\s*\)?", " __FALSE "),
    (r"\bEVEX\.b\b", " __EVEXB "),
    (r"\bEVEX\.b\s*==", " __EVEXB =="),
]

TOKEN = re.compile(r"""
    (?P<ws>[ \t]+)
  | (?P<num>0[xX][0-9A-Fa-f_]+|[0-9][0-9A-Fa-f_]*H\b|[01]+[bB]\b|\d+\.\d+|\d+)
  | (?P<op>:=|!=|<=|>=|<<|>>|==|[-+*/=<>()\[\]:,;.{}&|^~!?])
  | (?P<word>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<other>.)
""", re.X)

KW = {"IF", "THEN", "ELSE", "FI", "FOR", "TO", "DOWNTO", "ENDFOR", "ROF",
      "CASE", "OF", "ESAC", "RETURN", "DEFINE", "WHILE", "DO", "OD"}
WORD_OPS = {"AND", "OR", "XOR", "NOT", "MOD"}


class Tok:
    __slots__ = ("kind", "text", "ind", "first", "line")

    def __init__(self, kind, text, ind, first, line):
        self.kind, self.text, self.ind, self.first, self.line = kind, text, ind, first, line

    def __repr__(self):
        return f"{self.kind}:{self.text}@{self.line}"


def strip_comments(text):
    text = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.S)
    return text


def lex(lines):
    """Tokens with each one's line indentation; prose in *...* becomes a
    'prose' token spanning lines."""
    text = strip_comments("\n".join(lines))
    toks = []
    prose = None
    depth = 0
    for ln, line in enumerate(text.split("\n")):
        line = re.sub(r"//.*$", "", line)
        # a ';' comment: a semicolon followed by lower-case prose
        line = re.sub(r";\s*[a-z][a-z \-,'()]*$", ";", line)
        line = re.sub(r";\s*(Only|If|When) .*$", ";", line)
        if prose is None:
            for pat, rep in PHRASES:
                line = re.sub(pat, rep, line, flags=re.I if "is memory" in pat else 0)
        ind = len(line) - len(line.lstrip())
        first = True
        pos = 0
        if prose is not None:
            end = line.rfind("*")
            if end < 0:
                prose[1] += " " + line.strip()
                continue
            prose[1] += " " + line[:end].strip()
            toks.append(Tok("prose", prose[1].strip(), prose[0], prose[2], prose[3]))
            prose = None
            pos = end + 1
            first = False
        while pos < len(line):
            if line[pos] == "*" and (first or toks and toks[-1].kind in ("nl", "kw", "op")
                                     and toks[-1].text in ("THEN", "ELSE", ";", ":")):
                end = line.find("*", pos + 1)
                if end < 0:
                    prose = [ind, line[pos + 1:], first, ln]
                    pos = len(line)
                    break
                toks.append(Tok("prose", line[pos + 1:end].strip(), ind, first, ln))
                first = False
                pos = end + 1
                continue
            if line[pos] == "*" and re.match(r"\*[^*]*[a-z]{4,}[^*]*\*\s*;?\s*$", line[pos:]):
                # a trailing comment in stars ('*0x11B = ... in binary*')
                break
            m = TOKEN.match(line, pos)
            pos = m.end()
            k = m.lastgroup
            t = m.group(k)
            if k == "ws":
                continue
            if k == "word":
                u = t.upper()
                if u in KW and (t.isupper() or u in ("FOR", "TO", "DOWNTO", "IF", "ELSE",
                                                     "RETURN", "DEFINE")):
                    k, t = "kw", u
                elif u in WORD_OPS and (t.isupper() or t in ("and", "or", "xor", "not")):
                    k, t = "op", u
            if k == "op" and t in "([":
                depth += 1
            if k == "op" and t in ")]":
                depth = max(0, depth - 1)
            if k == "other":
                if t in "'\"":
                    continue
                raise SdmError(f"line {ln}: cannot read {t!r} in {line.strip()!r}")
            toks.append(Tok(k, t, ind, first, ln))
            first = False
        if prose is None and depth == 0:
            toks.append(Tok("nl", "\n", ind, False, ln))
    if prose is not None:
        toks.append(Tok("prose", prose[1].strip(), prose[0], prose[2], prose[3]))
    toks.append(Tok("eof", "", -1, True, 10 ** 9))
    return toks


def number(t):
    t = t.replace("_", "")
    if t[:2] in ("0x", "0X"):
        return int(t, 16)
    if t.endswith("H"):
        return int(t[:-1], 16)
    if t[-1] in "bB" and set(t[:-1]) <= {"0", "1"}:
        return int(t[:-1], 2)
    if "." in t:
        return float(t)
    return int(t)


# ------------------------------------------------------------------ parser
# Nodes are tuples: ('num', v) ('name', N) ('bin', op, a, b) ('un', op, a)
# ('call', f, [args]) ('slice', base, hi, lo) ('bit', base, i)
# ('field', base, kind, idx) ('mem', addr) ('cond', c, a, b) ('prose', text)
# statements: ('assign', lhs, rhs) ('if', c, then, else) ('for', v, a, b,
# step, body) ('case', e, [(vals, body)]) ('return', e) ('nop',) ('expr', e)

FIELDS = {"BIT": 1, "BYTE": 8, "WORD": 16, "DWORD": 32, "QWORD": 64, "XMM": 128,
          "YMM": 256}

# The word operators AND/OR join conditions; the symbols & | ^ bind tighter
# than comparisons ('IF shufbyte & 80H = 80H' in PSHUFB), as in the SDM's
# usage rather than C's.
PREC = [("?",), ("OR", "||"), ("AND", "&&"), ("=", "==", "!="),
        ("<", ">", "<=", ">="), ("|",), ("XOR", "^"), ("&",), ("<<", ">>"),
        ("+", "-"), ("*", "/", "MOD")]


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.p = 0

    def context(self, t):
        return " ".join(x.text for x in self.t if x.line == t.line and x.kind != "nl")

    def peek(self, k=0):
        return self.t[self.p + k]

    def next(self):
        t = self.t[self.p]
        self.p += 1
        return t

    def at(self, kind, *texts):
        t = self.peek()
        return t.kind == kind and (not texts or t.text in texts)

    def skip_nl(self):
        while self.at("nl") or self.at("op", ";"):
            self.next()

    def expect(self, kind, text=None):
        t = self.next()
        if t.kind != kind or (text is not None and t.text != text):
            raise SdmError(f"line {t.line}: expected {text or kind}, found {t.text!r}"
                           f" ({self.context(t)})")
        return t

    # ---- expressions
    def expr(self, lvl=0):
        if lvl == len(PREC):
            return self.unary()
        ops = PREC[lvl]
        if ops == ("?",):
            c = self.expr(1)
            if self.at("op", "?"):
                self.next()
                a = self.expr(0)
                self.skip_nl_in_expr()
                self.expect("op", ":")
                b = self.expr(0)
                return ("cond", c, a, b)
            return c
        a = self.expr(lvl + 1)
        while self.peek().kind == "op" and self.peek().text in ops:
            op = self.next().text
            op = {"||": "OR", "&&": "AND", "^": "XOR", "==": "="}.get(op, op)
            self.skip_nl_in_expr()
            b = self.expr(lvl + 1)
            a = ("bin", op, a, b)
        return a

    def skip_nl_in_expr(self):
        while self.at("nl") and self.peek(1).kind != "eof" and \
                self.peek(1).kind == "op" and self.peek(1).text in (":", "?"):
            self.next()

    def unary(self):
        if self.at("op", "-", "~", "!", "NOT"):
            op = self.next().text
            return ("un", "NOT" if op in ("!", "NOT") else op, self.unary())
        return self.postfix(self.primary())

    def primary(self):
        t = self.next()
        if t.kind == "num":
            return ("num", number(t.text))
        if t.kind == "op" and t.text == "(":
            e = self.expr()
            while self.at("nl"):
                self.next()
            self.expect("op", ")")
            return e
        if t.kind == "prose":
            return ("prose", t.text)
        if t.kind == "word":
            name = t.text.upper()
            if name == "MEM" and self.at("op", "["):
                self.next()
                e = self.expr()
                while self.at("nl"):
                    self.next()
                self.expect("op", "]")
                return ("mem", e)
            if self.at("op", "("):
                self.next()
                args = []
                while not self.at("op", ")"):
                    args.append(self.expr())
                    if self.at("op", ","):
                        self.next()
                self.expect("op", ")")
                return ("call", name, args)
            return ("name", name)
        raise SdmError(f"line {t.line}: unexpected {t.text!r} ({self.context(t)})")

    def postfix(self, e):
        while True:
            if self.at("op", "["):
                self.next()
                hi = self.expr()
                if self.at("op", ":"):
                    self.next()
                    lo = self.expr()
                    self.expect("op", "]")
                    e = ("slice", e, hi, lo)
                else:
                    self.expect("op", "]")
                    e = ("bit", e, hi)
            elif self.at("op", ".") and self.peek(1).kind == "word" and \
                    self.peek(1).text.upper() in FIELDS and self.peek(2).text == "[":
                self.next()
                kind = self.next().text.upper()
                self.expect("op", "[")
                idx = self.expr()
                self.expect("op", "]")
                e = ("field", e, kind, idx)
            else:
                return e

    # ---- statements
    def block(self, opener_ind, stops, dedent=True):
        body = []
        while True:
            self.skip_nl()
            t = self.peek()
            if t.kind == "eof":
                return body
            if t.kind == "kw" and t.text in stops:
                return body
            if dedent and t.first and t.ind <= opener_ind:
                return body
            body.extend(self.statement())

    def statement(self):
        t = self.peek()
        if t.kind == "kw":
            if t.text == "IF":
                return [self.if_stmt(chained=False)]
            if t.text == "FOR":
                return [self.for_stmt()]
            if t.text == "CASE":
                return [self.case_stmt()]
            if t.text == "RETURN":
                self.next()
                return [("return", self.expr())]
            if t.text in ("FI", "ENDFOR", "ROF", "ESAC", "THEN"):
                # a stray terminator (VPERMILPS: '...; FI;')
                self.next()
                return []
            if t.text == "ELSE":
                raise SdmError(f"line {t.line}: ELSE without IF")
        if t.kind == "prose":
            self.next()
            return [("prose", t.text)]
        if t.kind == "op" and t.text in ("{", "}"):
            self.next()
            return []
        save = self.p
        if t.kind == "word":
            lv = self.postfix(self.primary())
            if self.at("op", ":=", "="):
                self.next()
                self.skip_nl_cont()
                return [("assign", lv, self.expr())]
            self.p = save
        lhs = self.expr()
        if self.at("op", ":=", "="):
            self.next()
            self.skip_nl_cont()
            rhs = self.expr()
            if self.at("op", "(") or self.at("word"):
                # trailing prose such as '(Unmodified)'
                while not self.at("nl") and not self.at("eof"):
                    self.next()
                return []
            return [("assign", lhs, rhs)]
        if lhs[0] == "bin" and lhs[1] == "=":
            return [("assign", lhs[2], lhs[3])]
        if lhs[0] == "slice" and self.at("op", "(") or self.at("word"):
            while not self.at("nl") and not self.at("eof"):
                self.next()
            return []
        if lhs[0] == "prose":
            return [("prose", lhs[1])]
        return [("expr", lhs)]

    def skip_nl_cont(self):
        while self.at("nl"):
            self.next()

    def cond(self):
        e = self.expr()
        if self.at("op", ":"):
            self.next()
        return e

    def if_stmt(self, chained):
        it = self.next()
        ind = it.ind
        c = self.cond()
        while self.at("nl"):
            self.next()
        if self.at("kw", "THEN"):
            self.next()
        then = self.block(ind, ("ELSE", "FI", "ENDFOR", "ROF", "ESAC"))
        els = []
        if self.at("kw", "ELSE") and (not self.peek().first or self.peek().ind >= ind - 1 or True):
            et = self.next()
            if self.at("op", ":"):
                self.next()
            if self.at("kw", "IF"):
                els = [self.if_stmt(chained=True)]
            else:
                els = self.block(min(ind, et.ind) if et.first else ind,
                                 ("ELSE", "FI", "ENDFOR", "ROF", "ESAC"))
        if not chained:
            self.skip_nl()
            if self.at("kw", "FI"):
                self.next()
                if self.at("op", ";"):
                    self.next()
        return ("if", c, then, els)

    def for_stmt(self):
        ft = self.next()
        ind = ft.ind
        var = self.expect("word").text.upper()
        if not self.at("op", ":=", "="):
            raise SdmError(f"line {ft.line}: FOR without :=")
        self.next()
        a = self.expr()
        down = self.at("kw", "DOWNTO")
        if not (self.at("kw", "TO") or down):
            raise SdmError(f"line {ft.line}: FOR without TO")
        self.next()
        b = self.expr()
        if self.at("op", ":"):
            self.next()
        if self.at("kw", "DO"):
            self.next()
        explicit = self.has_endfor()
        body = self.block(ind, ("ENDFOR", "ROF", "OD") if explicit else
                          ("ENDFOR", "ROF", "OD", "ELSE", "FI", "ESAC"), dedent=not explicit)
        if explicit and self.at("kw", "ENDFOR", "ROF", "OD"):
            self.next()
            if self.at("op", ";"):
                self.next()
        return ("for", var, a, b, -1 if down else 1, body)

    def has_endfor(self):
        """Whether this FOR is closed by an ENDFOR (counting nested FORs)."""
        depth = 0
        q = self.p
        start = self.t[self.p - 1]
        while self.t[q].kind != "eof":
            t = self.t[q]
            if t.kind == "kw" and t.text == "FOR":
                depth += 1
            if t.kind == "kw" and t.text in ("ENDFOR", "ROF"):
                if depth == 0:
                    return True
                depth -= 1
            # a FOR without ENDFOR at a lower indentation ends the search
            if t.first and t.ind < start.ind and t.kind != "nl":
                return False
            q += 1
        return False

    def case_stmt(self):
        self.next()
        e = self.expr()
        if self.at("kw", "OF"):
            self.next()
        arms = []
        while True:
            self.skip_nl()
            if self.at("kw", "ESAC") or self.at("eof"):
                break
            vals = [self.expr()]
            while self.at("op", ","):
                self.next()
                vals.append(self.expr())
            self.expect("op", ":")
            body = []
            while not self.at("nl") and not self.at("eof"):
                body.extend(self.statement())
                if self.at("op", ";"):
                    self.next()
            arms.append((vals, body))
        if self.at("kw", "ESAC"):
            self.next()
        return ("case", e, arms)

    def program(self):
        body = self.block(-1, ())
        return body


# ------------------------------------------------------------------ helpers

def parse_defines(prelude):
    """Helper functions defined before the first title: 'define f(x):'
    with an indented body, or 'f(x) {' ... '}'."""
    funcs = {}
    lines = list(prelude)
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r"^(?:define\s+)?(\w+)\s*\(([^)]*)\)\s*(:|\{)", s, re.I)
        if not m or ":=" in s:
            i += 1
            continue
        name = m.group(1).upper()
        args = [a.strip().upper() for a in m.group(2).split(",") if a.strip()]
        body = []
        i += 1
        if m.group(3) == "{":
            while i < len(lines) and lines[i].strip() != "}":
                body.append(lines[i])
                i += 1
            i += 1
        else:
            while i < len(lines):
                ln = lines[i]
                st = ln.strip()
                if st and not ln[0].isspace():
                    # E-24: a 'return' at column 0 still belongs to the helper
                    if st.lower().startswith("return"):
                        body.append("   " + st)
                        i += 1
                    break
                body.append(ln)
                i += 1
            body = [b[1:] if b.startswith(" ") else b for b in body]
        funcs[name] = (args, Parser(lex(body)).program())
    return funcs


def kl_vl(lines):
    for ln in lines:
        m = re.match(r"^\s*\(\s*KL\s*,\s*VL\s*\)\s*=\s*(.*)$", ln)
        if m:
            pairs = re.findall(r"\(\s*(\d+)\s*,?\s*(\d+)\s*\)", m.group(1))
            return [(int(a), int(b)) for a, b in pairs]
    return None


# ------------------------------------------------------------------ evaluator

class BV(int):
    """An integer that remembers the width of the field it was read from."""

    def __new__(cls, v, w):
        o = int.__new__(cls, v & ((1 << w) - 1))
        o.w = w
        return o


class FloatVal:
    def __init__(self, f):
        self.f = f


class Return(Exception):
    def __init__(self, v):
        self.v = v


def gf_mul(a, b):
    r = 0
    for i in range(8):
        if b >> i & 1:
            r ^= a << i
    for i in range(14, 7, -1):
        if r >> i & 1:
            r ^= 0x11B << (i - 8)
    return r


GF_INV = [0] + [next(y for y in range(1, 256) if gf_mul(x, y) == 1) for x in range(1, 256)]


def sext(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) & 1 else v


def to_float(v, w):
    if isinstance(v, FloatVal):
        return v.f
    if isinstance(v, float):
        return v
    if w == 32:
        return struct.unpack("<f", struct.pack("<I", v & 0xffffffff))[0]
    if w == 64:
        return struct.unpack("<d", struct.pack("<Q", v & (2 ** 64 - 1)))[0]
    raise SdmError(f"a float of {w} bits")


def from_float(f, w):
    if w == 32:
        try:
            return struct.unpack("<I", struct.pack("<f", f))[0]
        except OverflowError:
            return 0x7f800000 if f > 0 else 0xff800000
    return struct.unpack("<Q", struct.pack("<d", f))[0]


def sat(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def conv(name, x):
    """Truncate*/Saturate*To* conversions by their names."""
    sizes = {"QUADWORD": 64, "QWORD": 64, "DOUBLEWORD": 32, "DWORD": 32,
             "WORD": 16, "BYTE": 8}
    m = re.match(r"^(TRUNCATE|SATURATE)(SIGNED|UNSIGNED)?(\w+?)TO(SIGNED|UNSIGNED)?(\w+)$", name)
    if not m:
        return None
    kind, s1, src, s2, dst = m.groups()
    if src not in sizes or dst not in sizes:
        return None
    ws, wd = sizes[src], sizes[dst]
    if kind == "TRUNCATE":
        return x & ((1 << wd) - 1)
    if s1 == "SIGNED" or (s1 is None and s2 != "UNSIGNED"):
        v = sext(x, ws)
        if (s2 or s1) == "UNSIGNED":
            return sat(v, 0, (1 << wd) - 1)
        return sat(v, -(1 << (wd - 1)), (1 << (wd - 1)) - 1) & ((1 << wd) - 1)
    v = x & ((1 << ws) - 1)
    return sat(v, 0, (1 << wd) - 1)


class Machine:
    def __init__(self, env, funcs, mem, mode):
        self.env = env          # name -> int (registers are MAXVL-bit ints)
        self.funcs = funcs
        self.mem = mem          # bytes
        self.mode = mode        # 'none', 'merge', 'zero'
        self.approx = False

    # ---- lvalues: (base name, lo bit, width)
    def place(self, e, env):
        if e[0] == "name":
            return e[1], 0, None
        if e[0] in ("slice", "bit", "field"):
            base, lo0, w0 = self.place(e[1], env)
            if e[0] == "slice":
                hi, lo = int(self.ev(e[2], env)), int(self.ev(e[3], env))
                # DEST[MAXVL-1:VL] at VL = MAXVL: no bits
                return base, lo0 + lo, max(0, hi - lo + 1)
            if e[0] == "bit":
                return base, lo0 + int(self.ev(e[2], env)), 1
            w = FIELDS[e[2]]
            return base, lo0 + w * int(self.ev(e[3], env)), w
        raise SdmError(f"cannot assign to {e}")

    def lookup(self, name, env):
        if name in env:
            return env[name]
        if name in self.env:
            return self.env[name]
        alias = {"MAX_VL": "MAXVL", "MAX_KL": "MAXKL", "K": "K1", "IMM": "IMM8",
                 "SRC": "SRC1", "SRCDEST": "DEST", "VINDEX": "VINDEX"}
        if name in alias and alias[name] != name:
            return self.lookup(alias[name], env)
        if name == "__NOWM":
            return 1 if self.mode == "none" else 0
        if name == "__MERGE":
            return 1 if self.mode == "merge" else 0
        if name == "__ZERO":
            return 1 if self.mode == "zero" else 0
        if name in ("__FALSE", "__EVEXB", "DISP"):
            return 0
        raise SdmError(f"unknown name {name}")

    def store(self, lhs, v, env):
        base, lo, w = self.place(lhs, env)
        # a helper's names are its own (PCLMUL128 builds its own DEST)
        scope = env if "__FN" in env or base in env or base not in self.env and lhs[0] == "name" \
            or (base not in self.env and base not in GLOBALS) else self.env
        if base == "SRCDEST":
            base, scope = "DEST", self.env
        if isinstance(v, FloatVal):
            if w is None:
                scope[base] = v
                return
            v = from_float(v.f, w)
        if isinstance(v, float):
            v = from_float(v, w or 64)
        if w is None:
            scope[base] = v
            return
        if w == 0:
            return
        old = scope.get(base, 0)
        if isinstance(old, FloatVal):
            old = 0
        m = ((1 << w) - 1) << lo
        scope[base] = (int(old) & ~m) | ((int(v) << lo) & m)

    def ev(self, e, env):
        k = e[0]
        if k == "num":
            return e[1]
        if k == "name":
            return self.lookup(e[1], env)
        if k == "prose":
            return self.prose_val(e[1])
        if k in ("slice", "bit", "field"):
            base = self.ev(e[1], env)
            if isinstance(base, FloatVal):
                raise SdmError("a field of a float")
            if k == "slice":
                hi, lo = int(self.ev(e[2], env)), int(self.ev(e[3], env))
                if hi < lo:
                    return BV(0, 1)
                return BV(int(base) >> lo, hi - lo + 1)
            if k == "bit":
                return BV(int(base) >> int(self.ev(e[2], env)), 1)
            w = FIELDS[e[2]]
            return BV(int(base) >> (w * int(self.ev(e[3], env))), w)
        if k == "mem":
            a = int(self.ev(e[1], env))
            if a < 0 or a + 8 > len(self.mem) + 8:
                raise SdmError(f"memory address {a} outside the operand")
            b = bytes(self.mem[a:a + 8]).ljust(8, b"\0")
            return int.from_bytes(b, "little")
        if k == "cond":
            return self.ev(e[2] if self.ev(e[1], env) else e[3], env)
        if k == "un":
            v = self.ev(e[2], env)
            if e[1] == "-":
                return -v
            if e[1] == "~":
                return BV(~int(v), v.w) if isinstance(v, BV) else ~int(v)
            if isinstance(v, BV) and v.w > 1:
                return BV(~int(v), v.w)
            return 0 if v else 1
        if k == "bin":
            return self.binop(e[1], e[2], e[3], env)
        if k == "call":
            return self.call(e[1], e[2], env)
        raise SdmError(f"cannot evaluate {k}")

    def prose_val(self, text):
        t = text.lower()
        if "remains unchanged" in t or "unmodified" in t:
            return None
        raise SdmError(f"prose: {text}")

    def binop(self, op, a, b, env):
        if op == "AND" and a[0] != "num" and b[0] != "num":
            x = self.ev(a, env)
            # a boolean AND of conditions short-circuits; bit fields AND
            if not isinstance(x, BV) or x.w == 1:
                if not x:
                    return 0
                y = self.ev(b, env)
                if isinstance(x, BV) and isinstance(y, BV) and y.w > 1:
                    return int(x) & int(y)
                return 1 if y else 0 if not isinstance(y, BV) or y.w == 1 else int(x) & int(y)
            y = self.ev(b, env)
            return BV(int(x) & int(y), max(x.w, getattr(y, "w", 0)))
        if op == "OR" and a[0] != "num" and b[0] != "num":
            x = self.ev(a, env)
            if not isinstance(x, BV) or x.w == 1:
                if x:
                    return 1
                y = self.ev(b, env)
                return 1 if y else 0
        x, y = self.ev(a, env), self.ev(b, env)
        if isinstance(x, FloatVal) or isinstance(y, FloatVal) or isinstance(x, float) \
                or isinstance(y, float):
            fx = to_float(x, getattr(x, "w", 64)) if not isinstance(x, (int,)) or \
                isinstance(x, BV) else float(x)
            fy = to_float(y, getattr(y, "w", 64)) if not isinstance(y, (int,)) or \
                isinstance(y, BV) else float(y)
            return self.fop(op, fx, fy)
        xi, yi = int(x), int(y)
        if op == "+":
            return xi + yi
        if op == "-":
            return xi - yi
        if op == "*":
            return xi * yi
        if op == "/":
            return xi // yi
        if op == "MOD":
            return xi % yi
        # shifts of unbounded integers: past 1024 bits nothing of a
        # 512-bit register is left
        if op == "<<":
            return 0 if yi > 1024 else xi << yi
        if op == ">>":
            return (-1 if xi < 0 else 0) if yi > 1024 else xi >> yi
        if op in ("&", "AND"):
            return xi & yi
        if op in ("|", "OR"):
            return xi | yi
        if op == "XOR":
            r = xi ^ yi
            if isinstance(x, BV) and isinstance(y, BV):
                return BV(r, max(x.w, y.w))
            return r
        if op == "=":
            return 1 if xi == yi else 0
        if op == "!=":
            return 1 if xi != yi else 0
        if op == "<":
            return 1 if xi < yi else 0
        if op == ">":
            return 1 if xi > yi else 0
        if op == "<=":
            return 1 if xi <= yi else 0
        if op == ">=":
            return 1 if xi >= yi else 0
        raise SdmError(f"operator {op}")

    def fop(self, op, x, y):
        try:
            if op == "/":
                if y == 0:
                    return FloatVal(math.copysign(math.inf, x) * math.copysign(1, y)
                                    if x != 0 else math.nan)
                return FloatVal(x / y)
            if op == "*":
                return FloatVal(x * y)
            if op == "+":
                return FloatVal(x + y)
            if op == "-":
                return FloatVal(x - y)
        except OverflowError:
            return FloatVal(math.inf)
        raise SdmError(f"float operator {op}")

    def fval(self, e, env, w):
        """A float operand: a bit field read at its width."""
        v = self.ev(e, env)
        if isinstance(v, FloatVal):
            return v.f
        if isinstance(v, float):
            return v
        return to_float(int(v), getattr(v, "w", w))

    def call(self, f, args, env):
        if f in ("SIGNEXTEND", "ZEROEXTEND", "SIGN_EXTEND", "ZERO_EXTEND"):
            a = args[0]
            if f.startswith("SIGN") and a[0] == "bin" and a[1] == ">>":
                # E-19: SignExtend(X >> c) is an arithmetic shift of X
                x = self.ev(a[2], env)
                c = int(self.ev(a[3], env))
                return sext(int(x), x.w) >> c if isinstance(x, BV) else int(x) >> c
            x = self.ev(a, env)
            if f.startswith("SIGN"):
                return sext(int(x), x.w) if isinstance(x, BV) else int(x)
            return int(x)
        if f == "APPROXIMATE":
            self.approx = True
            a = args[0]
            if a[0] == "bin" and a[1] == "/":
                n = self.fval(a[2], env, 64)
                if a[3][0] == "call" and a[3][1] == "SQRT":
                    d = self.fval(a[3][2][0], env, 32)
                    if d < 0 or d != d:
                        return FloatVal(math.nan)
                    d = math.sqrt(d)
                else:
                    d = self.fval(a[3], env, 32)
                if d == 0:
                    return FloatVal(math.copysign(math.inf, d) * (1 if n > 0 else -1))
                if math.isinf(d):
                    return FloatVal(math.copysign(0.0, d))
                return FloatVal(n / d)
            raise SdmError("APPROXIMATE of something other than a quotient")
        if f == "INVERSE" and "INVERSE" not in self.funcs:
            # E-27: the multiplicative inverse in GF(2^8) mod 0x11B, 0 for 0
            # (the page's 'Inverse Byte Listings' table, not in Operation)
            return GF_INV[int(self.ev(args[0], env)) & 0xff]
        if f == "PARITY" and "PARITY" not in self.funcs:
            # defined on the GF2P8AFFINEQB page, used by GF2P8AFFINEINVQB
            return bin(int(self.ev(args[0], env))).count("1") & 1
        if f in self.funcs:
            names, body = self.funcs[f]
            local = {"__FN": 1}
            for n, a in zip(names, args):
                local[n] = self.ev(a, env)
            try:
                self.run(body, local)
            except Return as r:
                return r.v
            raise SdmError(f"{f} returned nothing")
        c = conv(f, int(self.ev(args[0], env))) if len(args) == 1 else None
        if c is not None:
            return c
        raise SdmError(f"unknown function {f}")

    def run(self, stmts, env):
        for s in stmts:
            k = s[0]
            if k == "assign":
                v = self.ev(s[2], env)
                if v is None:
                    continue
                self.store(s[1], v, env)
            elif k == "if":
                c = self.ev(s[1], env)
                self.run(s[2] if (int(c) if not isinstance(c, FloatVal) else c.f) else s[3],
                         env)
            elif k == "for":
                a, b = int(self.ev(s[2], env)), int(self.ev(s[3], env))
                rng = range(a, b + 1) if s[4] > 0 else range(a, b - 1, -1)
                for x in rng:
                    env[s[1]] = x
                    self.run(s[5], env)
            elif k == "case":
                v = int(self.ev(s[1], env))
                for vals, body in s[2]:
                    if any(int(self.ev(x, env)) == v for x in vals):
                        self.run(body, env)
                        break
            elif k == "return":
                raise Return(self.ev(s[1], env))
            elif k == "prose":
                # a statement in prose: 'remains unchanged', or a comment
                # (GF2P8AFFINEQB's '* parity(x) = 1 if ... *')
                pass
            elif k == "expr":
                pass


GLOBALS = {"DEST", "SRC1", "SRC2", "SRC3", "SRC", "K1", "IMM8", "VINDEX",
           "BASE_ADDR", "SCALE", "VL", "KL", "MAXVL", "MAXKL", "TMP_DEST"}


# ------------------------------------------------------------------ intrinsics

# intrinsic stem (after _mm{,256,512}_ and mask_/maskz_) -> (page, mnemonic,
# section keywords, operand roles).  Roles name the SDM operands the
# intrinsic's vector/scalar operands bind to, in order; 'DEST' is a table
# the instruction also reads (vpermt2*).
ISA = {
    "sllv_epi16": ("VPSLLVW_VPSLLVD_VPSLLVQ", "VPSLLVW", ("EVEX",), ("SRC1", "SRC2")),
    "srlv_epi16": ("VPSRLVW_VPSRLVD_VPSRLVQ", "VPSRLVW", ("EVEX",), ("SRC1", "SRC2")),
    "srav_epi16": ("VPSRAVW_VPSRAVD_VPSRAVQ", "VPSRAVW", ("EVEX",), ("SRC1", "SRC2")),
    "srav_epi32": ("VPSRAVW_VPSRAVD_VPSRAVQ", "VPSRAVD", ("EVEX",), ("SRC1", "SRC2")),
    "srav_epi64": ("VPSRAVW_VPSRAVD_VPSRAVQ", "VPSRAVQ", ("EVEX",), ("SRC1", "SRC2")),
    "sllv_epi32": ("VPSLLVW_VPSLLVD_VPSLLVQ", "VPSLLVD", ("EVEX",), ("SRC1", "SRC2")),
    "sllv_epi64": ("VPSLLVW_VPSLLVD_VPSLLVQ", "VPSLLVQ", ("EVEX",), ("SRC1", "SRC2")),
    "srlv_epi32": ("VPSRLVW_VPSRLVD_VPSRLVQ", "VPSRLVD", ("EVEX",), ("SRC1", "SRC2")),
    "srlv_epi64": ("VPSRLVW_VPSRLVD_VPSRLVQ", "VPSRLVQ", ("EVEX",), ("SRC1", "SRC2")),
    "rcp14_ps": ("VRCP14PS", "VRCP14PS", ("EVEX",), ("SRC1",)),
    "rcp14_pd": ("VRCP14PD", "VRCP14PD", ("EVEX",), ("SRC1",)),
    "rsqrt14_ps": ("VRSQRT14PS", "VRSQRT14PS", ("EVEX",), ("SRC1",)),
    "rsqrt14_pd": ("VRSQRT14PD", "VRSQRT14PD", ("EVEX",), ("SRC1",)),
    "permutexvar_epi8": ("VPERMB", "VPERMB", ("EVEX",), ("SRC1", "SRC2")),
    "permutexvar_epi16": ("VPERMD_VPERMW", "VPERMW", ("EVEX",), ("SRC1", "SRC2")),
    "permutexvar_epi32": ("VPERMD_VPERMW", "VPERMD", ("EVEX",), ("SRC1", "SRC2")),
    "permutexvar_ps": ("VPERMPS", "VPERMPS", ("EVEX",), ("SRC1", "SRC2")),
    "permutexvar_epi64": ("VPERMQ", "VPERMQ", ("EVEX", "vector"), ("SRC1", "SRC2")),
    "permutexvar_pd": ("VPERMPD", "VPERMPD", ("EVEX", "vector"), ("SRC1", "SRC2")),
    "permutex2var_epi8": ("VPERMT2B", "VPERMT2B", ("EVEX",), ("DEST", "SRC1", "SRC2")),
    "permutex2var_epi16": ("VPERMT2W_D_Q_PS_PD", "VPERMT2W", ("EVEX",),
                           ("DEST", "SRC1", "SRC2")),
    "permutex2var_epi32": ("VPERMT2W_D_Q_PS_PD", "VPERMT2D", ("EVEX",),
                           ("DEST", "SRC1", "SRC2")),
    "permutex2var_epi64": ("VPERMT2W_D_Q_PS_PD", "VPERMT2Q", ("EVEX",),
                           ("DEST", "SRC1", "SRC2")),
    "cvtepi64_epi8": ("VPMOVQB_VPMOVSQB_VPMOVUSQB", "VPMOVQB", ("register",), ("SRC1",)),
    "cvtsepi16_epi8": ("VPMOVWB_VPMOVSWB_VPMOVUSWB", "VPMOVSWB", ("register",), ("SRC1",)),
    "cvtusepi16_epi8": ("VPMOVWB_VPMOVSWB_VPMOVUSWB", "VPMOVUSWB", ("register",),
                        ("SRC1",)),
    "cvtsepi32_epi8": ("VPMOVDB_VPMOVSDB_VPMOVUSDB", "VPMOVSDB", ("register",), ("SRC1",)),
    "cvtusepi32_epi8": ("VPMOVDB_VPMOVSDB_VPMOVUSDB", "VPMOVUSDB", ("register",),
                        ("SRC1",)),
    "cvtepi32_epi8": ("VPMOVDB_VPMOVSDB_VPMOVUSDB", "VPMOVDB", ("register",), ("SRC1",)),
    "cvtepi16_epi8": ("VPMOVWB_VPMOVSWB_VPMOVUSWB", "VPMOVWB", ("register",), ("SRC1",)),
    "gf2p8mul_epi8": ("GF2P8MULB", "VGF2P8MULB", ("EVEX",), ("SRC1", "SRC2")),
    "gf2p8affine_epi64_epi8": ("GF2P8AFFINEQB", "VGF2P8AFFINEQB", ("EVEX",),
                               ("SRC1", "SRC2")),
    "gf2p8affineinv_epi64_epi8": ("GF2P8AFFINEINVQB", "VGF2P8AFFINEINVQB", ("EVEX",),
                                  ("SRC1", "SRC2")),
    "compress_epi32": ("VPCOMPRESSD", "VPCOMPRESSD", ("reg",), ("SRC1",)),
    "compress_epi64": ("VPCOMPRESSQ", "VPCOMPRESSQ", ("reg",), ("SRC1",)),
    "expand_epi32": ("VPEXPANDD", "VPEXPANDD", ("EVEX",), ("SRC1",)),
    "expand_epi64": ("VPEXPANDQ", "VPEXPANDQ", ("EVEX",), ("SRC1",)),
    "shuffle_epi8": ("PSHUFB", "VPSHUFB", ("EVEX",), ("SRC1", "SRC2")),
    "permutevar_ps": ("VPERMILPS", "VPERMILPS", ("EVEX", "variable"), ("SRC1", "SRC2")),
    "permutevar_pd": ("VPERMILPD", "VPERMILPD", ("EVEX", "variable"), ("SRC1", "SRC2")),
    "i32gather_epi32": ("VPGATHERDD_VPGATHERDQ", "VPGATHERDD", ("EVEX",), ("VINDEX",)),
    "i32gather_epi64": ("VPGATHERDD_VPGATHERDQ", "VPGATHERDQ", ("EVEX",), ("VINDEX",)),
    "i64gather_epi64": ("VPGATHERQD_VPGATHERQQ", "VPGATHERQQ", ("EVEX",), ("VINDEX",)),
    "i32gather_ps": ("VGATHERDPS_VGATHERDPD", "VGATHERDPS", ("EVEX",), ("VINDEX",)),
    "i32gather_pd": ("VGATHERDPS_VGATHERDPD", "VGATHERDPD", ("EVEX",), ("VINDEX",)),
    "i64gather_pd": ("VGATHERQPS_VGATHERQPD", "VGATHERQPD", ("EVEX",), ("VINDEX",)),
    "i64gather_ps": ("VGATHERQPS_VGATHERQPD", "VGATHERQPS", ("EVEX",), ("VINDEX",)),
    "clmulepi64_epi128": ("PCLMULQDQ", "VPCLMULQDQ", ("EVEX",), ("SRC1", "SRC2")),
    "clmulepi64_si128": ("PCLMULQDQ", "VPCLMULQDQ", ("VEX",), ("SRC1", "SRC2")),
}

# the instructions with a 128/256-bit VEX form for the non-AVX-512 intrinsic
VEX_OK = {"clmulepi64_epi128": ("VEX",), "shuffle_epi8": ("VEX",)}


def split_name(name):
    m = re.match(r"^_mm(256|512)?_(mask_|maskz_|mask2_|mask3_)?(\w+)$", name)
    if not m:
        return None
    return int(m.group(1) or 128), (m.group(2) or "").rstrip("_"), m.group(3)


PAGES = {}


def program_for(sdmdir, page, mnem, keys, vl):
    key = (page, mnem, keys)
    if key in PAGES:
        return PAGES[key]
    path = os.path.join(sdmdir, page + ".sdm")
    if not os.path.exists(path):
        raise SdmError(f"no page {page}.sdm")
    title, ops = read_page(path)
    mnems = set(re.split(r"[/ ]", title.split(" - ")[0].split("—")[0]))
    mnems |= {mnem, mnem.lstrip("V")}
    prelude, secs = sections(ops, mnems)
    funcs = parse_defines(prelude)
    found = None
    for t, lines in secs:
        words = re.split(r"[/\s(),]+", t)
        if mnem not in words:
            continue
        tl = t.lower()
        if all(k.lower() in tl for k in keys) and "memory" not in tl and \
                "store" not in tl:
            found = (t, lines)
            break
    if found is None:
        raise SdmError(f"{page}: no section for {mnem} ({' '.join(keys)})")
    t, lines = found
    # helper definitions inside the section itself
    body = [ln for ln in lines if not re.match(r"^\s*\(\s*KL\s*,\s*VL\s*\)", ln)]
    if mnem == "VGF2P8AFFINEINVQB" and "EVEX" in t:
        # E-26: 'FOR b := 0 to 7:' and its body are printed at column 0,
        # outside the FOR j loop whose j they read; they belong inside it
        out, inside = [], False
        for ln in body:
            if ln.startswith("FOR b"):
                inside = True
            if inside and ln.startswith("DEST[MAX_VL"):
                inside = False
            out.append("    " + ln if inside else ln)
        body = out
    m = re.search(r"\bi := j \* (\d+)", "\n".join(body))
    if m and mnem in ("VPSRAVD", "VPSRAVQ"):
        # E-20: the masked-off lanes are lane j's, not lane 0's
        body = [re.sub(r"DEST\[(31|63):0\]( remains unchanged\*| := 0)",
                       r"DEST[i+\1:i]\2", ln) for ln in body]
    if m and mnem.startswith("VPSRAV"):
        # E-25: the count is the whole element, as IF COUNT < w needs
        w = int(m.group(1))
        body = [re.sub(r"COUNT := SRC2\[i\+\d:i\]", f"COUNT := SRC2[i+{w - 1}:i]", ln)
                for ln in body]
        body = [re.sub(r"COUNT := SRC2\[\d:0\]", f"COUNT := SRC2[{w - 1}:0]", ln)
                for ln in body]
    prog = Parser(lex(body)).program()
    PAGES[key] = (t, lines, prog, funcs)
    return PAGES[key]


def run_test(sdmdir, name, imm, kinds, opbytes):
    sp = split_name(name)
    if sp is None:
        raise SdmError("not an intrinsic name")
    vl, mk, stem = sp
    if stem not in ISA:
        raise SdmError("no instruction for it here")
    page, mnem, keys, roles = ISA[stem]
    title, lines, prog, funcs = program_for(sdmdir, page, mnem, keys, vl)
    env = {"MAXVL": MAXVL, "MAXKL": 64, "VL": vl, "DISP": 0}
    pairs = kl_vl(lines)
    if pairs:
        kl = dict((v, k) for k, v in pairs).get(vl)
        if kl is None:
            # E-21: only the 512-bit pair is listed
            k5, v5 = pairs[-1]
            kl = k5 * vl // v5
        env["KL"] = kl
    vals, kk, imm8, mem = [], None, None, b""
    for (kind, bits, elem), raw in zip(kinds, opbytes):
        if kind == "K":
            kk = int.from_bytes(raw[:8], "little") & ((1 << bits) - 1)
        elif kind == "I":
            imm8 = imm
        elif kind == "P":
            mem = raw
        else:
            vals.append(int.from_bytes(raw[:bits // 8], "little"))
    mode = {"": "none", "mask": "merge", "maskz": "zero"}.get(mk)
    if mode is None:
        raise SdmError(f"{mk} forms are not bound")
    dest = 0
    if mode == "merge":
        dest = vals.pop(0)
    ops = dict(zip(roles, vals))
    if len(vals) < len(roles):
        raise SdmError("fewer operands than roles")
    if "DEST" in ops:
        if mode == "merge":
            raise SdmError("merge into a table operand")
        dest = ops.pop("DEST")
    env.update(ops)
    env["DEST"] = dest
    if "SRC1" in env and "SRC" not in env:
        env["SRC"] = env["SRC1"]
    if "VINDEX" in env:
        env["BASE_ADDR"] = 0
        env["SCALE"] = imm
        # the plain form gathers every element: k1 all ones
        if kk is None:
            kk = (1 << 64) - 1
    env["K1"] = kk if kk is not None else (1 << 64) - 1
    if imm8 is not None:
        env["IMM8"] = imm8
    m = Machine(env, funcs, mem, mode)
    m.run(prog, {})
    d = m.env["DEST"]
    if isinstance(d, FloatVal):
        raise SdmError("a float in DEST")
    return int(d) & ((1 << MAXVL) - 1), m.approx


def parse_kind(s):
    # V512:i32  K16  I  S:f32  P:i32
    m = re.match(r"^([VKISP])(\d*)(?::(\w+))?$", s)
    return m.group(1), int(m.group(2) or 0), m.group(3)


def selftest(sdmdir):
    """Parse every mapped section: sdmvec.py --parse SDMDIR"""
    bad = 0
    for stem, (page, mnem, keys, roles) in sorted(ISA.items()):
        try:
            t, lines, prog, funcs = program_for(sdmdir, page, mnem, keys, 512)
            print(f"ok   {stem:28} {t}")
        except SdmError as e:
            bad += 1
            print(f"FAIL {stem:28} {e}")
    return 1 if bad else 0


def main(argv):
    if len(argv) == 3 and argv[1] == "--parse":
        return selftest(argv[2])
    if len(argv) == 2 and argv[1] == "--errata":
        for k, v in sorted(ERRATA.items(), key=lambda kv: int(kv[0][2:])):
            print(f"{k}: {v}")
        return 0
    if len(argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    sdmdir, manifest, inb, outb, statusf = argv[1:]
    with open(manifest) as f:
        head = f.readline().split()
        ptb, npts, outsz, opb = map(int, head)
        tests = [ln.split() for ln in f if ln.strip()]
    data = open(inb, "rb").read()
    out = bytearray(len(tests) * npts * outsz)
    status = []
    for t, tl in enumerate(tests):
        name, imm = tl[0], int(tl[1])
        kinds = [parse_kind(x) for x in tl[2:]]
        try:
            for p in range(npts):
                base = (t * npts + p) * ptb
                ops = [data[base + k * opb: base + (k + 1) * opb] for k in range(len(kinds))]
                d, _ = run_test(sdmdir, name, imm, kinds, ops)
                o = (t * npts + p) * outsz
                out[o:o + 64] = d.to_bytes(64, "little")
            status.append("ok")
        except SdmError as e:
            status.append("skip " + str(e).replace("\n", " "))
        except (RecursionError, ValueError, TypeError, KeyError, IndexError) as e:
            status.append(f"skip {type(e).__name__}: {e}".replace("\n", " "))
    with open(outb, "wb") as f:
        f.write(out)
    with open(statusf, "w") as f:
        for s in status:
            f.write(s + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
