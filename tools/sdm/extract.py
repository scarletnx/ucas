#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract instruction specifications from the Intel SDM, Volume 2.

(From ~/vibe86, tools/sdm/extract.py, by the same author; ucas reads the
pages with tools/sdm/sdmvec.py, vibe86 with its tools/ucc.)

    extract.py SDM.pdf OUTDIR [MNEMONIC-PAGE-TITLE ...]

Runs `pdftotext -layout` on the combined Volume 2 (2A-2D, "Instruction Set
Reference, A-Z", order number 325383) and writes one OUTDIR/<NAME>.sdm per
instruction page: the opcode table, the operand-encoding table, the
"Operation" pseudocode and the "Flags Affected" text, verbatim apart from
page furniture (running footers, form feeds) and pdftotext column artefacts.

Without names, every instruction page in the manual is written; with names
(e.g. "ADD", "SAL/SAR/SHL/SHR"), only those.

The .sdm format (read by tools/ucc):

    # <title> -- Intel SDM 325383-<rev>, <page>
    @opcodes
    <opcode> | <instruction> | <op/en> | <64-bit mode> | <compat/leg mode>
    @encoding
    <op/en> | <operand 1> | <operand 2> | <operand 3> | <operand 4>
    @operation
    <pseudocode>
    @flags
    <prose>

The approach of treating the vendor's pseudocode as the executable
specification follows Reid, "Trustworthy Specifications of ARM v8-A and v8-M
System Level Architecture", FMCAD 2016 (doi:10.1109/FMCAD.2016.7886675), who
extracted ARM's ASL from the architecture manual in the same way.
"""

import os
import re
import subprocess
import sys

FOOTER = re.compile(r"^\S.*\s{2,}Vol\.\s+2[A-D]\s+[0-9A-Z]+-[0-9]+\s*$")
HEADER_LINE = re.compile(r"^([A-Z][A-Za-z0-9/ ]*?[A-Za-z0-9])\s*—\s*(\S.*?)\s*$")

# the headings that end a section of an instruction page
HEADINGS = (
    "Opcode", "Instruction Operand Encoding", "Description", "Operation",
    "Flags Affected", "Intel C/C++ Compiler Intrinsic Equivalent",
    "Intel C/C++ Compiler Intrinsic Equivalents",
    "Protected Mode Exceptions", "Real-Address Mode Exceptions",
    "Virtual-8086 Mode Exceptions", "Compatibility Mode Exceptions",
    "64-Bit Mode Exceptions", "Exceptions (All Operating Modes)",
    "Exceptions (All Modes of Operation)",
    "SIMD Floating-Point Exceptions", "Numeric Exceptions", "Other Exceptions",
    "FPU Flags Affected", "Floating-Point Exceptions", "IA-32 Architecture Compatibility",
    "Instruction Operand Encoding1", "NOTES:", "NOTE:",
)
END_OF_OPERATION = ("Flags Affected", "Intel C/C++ Compiler Intrinsic Equivalent",
                    "Intel C/C++ Compiler Intrinsic Equivalents", "Protected Mode Exceptions",
                    "Exceptions (All Operating Modes)", "Exceptions (All Modes of Operation)",
                    "SIMD Floating-Point Exceptions", "Numeric Exceptions", "Other Exceptions",
                    "FPU Flags Affected", "Real-Address Mode Exceptions",
                    "64-Bit Mode Exceptions", "Floating-Point Exceptions",
                    "IA-32 Architecture Compatibility", "Virtual-8086 Mode Exceptions",
                    "Compatibility Mode Exceptions")


def pdf_text(pdf):
    out = subprocess.run(["pdftotext", "-layout", pdf, "-"], check=True,
                         capture_output=True).stdout.decode("utf-8", "replace")
    return out


def clean(text):
    """Drop running footers and form feeds; keep a page marker per footer."""
    lines = []
    page = None
    for raw in text.split("\n"):
        line = raw.replace("\f", "").rstrip()
        m = FOOTER.match(line)
        if m:
            pg = re.search(r"Vol\.\s+(2[A-D])\s+([0-9A-Z]+-[0-9]+)\s*$", line)
            page = f"Vol. {pg.group(1)} {pg.group(2)}"
            lines.append(("\0page", page))
            continue
        lines.append((line, None))
    return lines


def revision(text):
    m = re.search(r"Order Number:\s*325383-(\d+US)", text)
    return m.group(1) if m else "?"


def split_pages(lines):
    """Yield (title, first page, body lines) for every instruction section.

    A section starts at a 'NAME—Title' line followed (within a few lines) by
    an opcode-table header, and runs to the next one.
    """
    starts = []
    n = len(lines)
    for i, (line, _) in enumerate(lines):
        if line == "\0page" or not line or line[0] == " ":
            continue
        m = HEADER_LINE.match(line)
        if not m or "Vol." in line or ". . ." in line or "...." in line:
            continue
        for j in range(i + 1, min(i + 9, n)):
            nxt = lines[j][0].strip()
            if nxt.startswith("Opcode") or nxt.startswith("Opcode/"):
                starts.append(i)
                break
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        body = lines[i + 1:end]
        page = None
        for line, pg in body:
            if line == "\0page":
                page = pg
                break
        yield lines[i][0].strip(), page, body


def is_heading(s, names):
    s = s.strip()
    return any(s == h or (s.startswith(h) and h.endswith(":")) for h in names)


def section(body, name, stops, indented=False):
    """The lines of section `name` up to the next heading in `stops`.

    Headings start in column 0, except the centred table titles
    (indented=True).
    """
    out = None
    for line, _ in body:
        if line == "\0page":
            continue
        if out is None:
            if line.strip() == name and (indented or not line[:1].isspace()):
                out = []
            continue
        if line and not line[0].isspace() and is_heading(line, stops):
            break
        out.append(line)
    if out is None:
        return None
    while out and not out[-1].strip():
        out.pop()
    while out and not out[0].strip():
        out.pop(0)
    return out


OPC_START = re.compile(
    r"^\s{0,6}((?:NP\s+|NFx\s+)?(?:REX(?:\.[WRwr])?\s*\+?\s*)?(?:[0-9A-F]{2}\b|VEX\.|EVEX\.|"
    r"[0-9A-F]{2}\+))")


def columns(line):
    return [c.strip() for c in re.split(r"\s{2,}", line.strip())]


FOOTNOTE = re.compile(r"(r/m8|r8|m8)\*?\d\b")


MODE2 = re.compile(r"^(V|N\.E\.|N\.S\.|I|Inv\.?)/(V|N\.E\.|N\.S\.|I|Inv\.?)$")
STACKED_ROW = re.compile(
    r"^\s{0,6}(?P<op>(?:NP\s+|NFx\s+)?(?:[0-9A-F]{2}\b|REX|VEX\.|EVEX\.|MAP|NP)\S*(?:\s\S+)*?)"
    r"\s{2,}(?P<open>[A-Z][A-Za-z0-9-]{0,5})\s+(?P<mode>\S+/\S+)(?:\s+(?P<rest>.*))?$")


def mode_cols(m):
    """'V/N.E.' (64-bit / 32-bit mode support) -> ('Valid', 'N.E.')."""
    full = {"V": "Valid", "I": "Invalid", "Inv.": "Invalid", "Inv": "Invalid"}
    a, b = m.split("/", 1)
    return full.get(a, a), full.get(b, b)


def opcode_rows_stacked(body):
    """The 'Opcode/Instruction | Op/En | 64/32 bit Mode Support | CPUID
    Feature Flag | Description' layout of the SSE/AVX pages: the instruction
    is on the line(s) under its opcode."""
    rows = []
    in_table = False
    cur = None
    for line, _ in body:
        if line == "\0page":
            continue
        s = line.strip()
        if re.match(r"^Opcode\s?\*?\s?/", s):
            in_table = True
            cur = None
            continue
        if not in_table:
            continue
        if s.startswith("NOTES") or s.startswith("NOTE:") or "Instruction Operand Encoding" in s \
                or s == "Description":
            in_table = False
            continue
        if not s or s.startswith("Instruction") or s in ("Support", "Flag", "Mode"):
            continue
        m = STACKED_ROW.match(line)
        if m and MODE2.match(m.group("mode")):
            m64, mleg = mode_cols(m.group("mode"))
            cur = [m.group("op").strip(), "", m.group("open"), m64, mleg]
            rows.append(cur)
            continue
        indent = len(line) - len(line.lstrip())
        if cur is not None and indent <= 3:
            frag = re.split(r"\s{2,}", s)[0]
            cur[1] = (cur[1] + " " + frag).strip() if cur[1] else frag
    out = []
    for r in rows:
        if r[1]:
            r[1] = re.sub(r"\s+,", ",", re.sub(r",(\S)", r", \1", r[1]))
            out.append(r)
    return out


def opcode_rows(body):
    """The legacy-encoded rows of the opcode table(s), as 5-tuples.

    Only the 'Opcode | Instruction | Op/En | 64-Bit Mode | Compat/Leg Mode |
    Description' layout used by the general-purpose instructions is read;
    pdftotext sometimes splits the operands of a row onto the next line
    (where a footnote superscript sat), which is repaired here.
    """
    rows = []
    in_table = False
    pending = None
    for line, _ in body:
        if line == "\0page":
            continue
        s = line.strip()
        if s.startswith("Opcode"):
            in_table = True
            continue
        if not in_table:
            continue
        if s.startswith("NOTES") or s.startswith("NOTE:") or "Instruction Operand Encoding" in s \
                or s == "Description":
            in_table = False
            continue
        if not s or s.startswith("En ") or s in ("Mode", "Leg Mode", "En"):
            continue
        if OPC_START.match(line):
            cols = columns(line)
            # 'REX.W + 81 /0 id   ADD r/m64, imm32   MI   Valid   N.E.   Desc'
            if len(cols) >= 5:
                row = [cols[0], FOOTNOTE.sub(r"\1", cols[1]), cols[2], cols[3], cols[4]]
            else:
                row = cols + [""] * (5 - len(cols))
            # operands pushed onto the line above by a footnote superscript:
            # '            1, r81' then '00 /r    ADD r/m8    MR ...'
            # an operand's footnote mark inline: 'ROL r/m8,2 1' -> 'ROL r/m8, 1'
            row[1] = re.sub(r",\d (\S)", r", \1", row[1])
            if pending and row[1].endswith(","):
                row[1] += pending.lstrip(",")
            elif pending and "," not in row[1] and " " in row[1]:
                row[1] += pending
            pending = None
            rows.append(row)
            continue
        pending = None
        if re.match(r"^\s{6,}\d+,?\s*\S", line):
            frag = FOOTNOTE.sub(r"\1", columns(line)[0])
            pending = re.sub(r"^\d+,?\s*", ", ", frag)
    return [r for r in rows if r[0] and r[1]]


def encoding_rows(body):
    rows = []
    sec = section(body, "Instruction Operand Encoding", HEADINGS, indented=True)
    if sec is None:
        sec = section(body, "Instruction Operand Encoding1", HEADINGS, indented=True)
    if sec is None:
        return rows
    for line in sec:
        s = line.strip()
        if not s or s.startswith("Op/En") or s.startswith("Op / En"):
            continue
        cols = columns(line)
        if len(cols) >= 2:
            rows.append(cols)
    return rows


def name_of(title):
    return title.split("—")[0].strip()


def fname(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    pdf, outdir, want = argv[1], argv[2], set(argv[3:])
    text = pdf_text(pdf)
    rev = revision(text)
    lines = clean(text)
    os.makedirs(outdir, exist_ok=True)
    seen = set()
    for title, page, body in split_pages(lines):
        name = name_of(title)
        if name in seen:
            # a second page with the same mnemonic (MOV to/from control
            # registers, the string MOVSD next to the SSE one): named by title
            name = title.replace("—", " ")
        if want and name not in want and name_of(title) not in want:
            continue
        if name in seen:
            continue
        seen.add(name)
        op = section(body, "Operation", END_OF_OPERATION)
        fl = section(body, "Flags Affected", HEADINGS)
        with open(os.path.join(outdir, fname(name) + ".sdm"), "w") as f:
            f.write(f"# {title} -- Intel SDM 325383-{rev}, {page or '?'}\n")
            f.write("@opcodes\n")
            rows = opcode_rows(body)
            if any(re.match(r"^Opcode\s?\*?\s?/", l.strip()) for l, _ in body if l != "\0page"):
                rows = [r for r in rows if len(r) == 5 and r[1] and r[3] in
                        ("Valid", "Invalid", "N.E.", "N.S.", "Inv.")] + opcode_rows_stacked(body)
            for r in rows:
                f.write(" | ".join(r) + "\n")
            f.write("@encoding\n")
            for r in encoding_rows(body):
                f.write(" | ".join(r) + "\n")
            f.write("@operation\n")
            for line in op or []:
                f.write(line + "\n")
            f.write("@flags\n")
            for line in fl or []:
                f.write(line.strip() + "\n")
    missing = want - seen
    for m in sorted(missing):
        print(f"extract: {m}: not found", file=sys.stderr)
    print(f"extract: {len(seen)} instruction pages from 325383-{rev}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
