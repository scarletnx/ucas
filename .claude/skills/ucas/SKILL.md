---
name: ucas
version: 1.0.0
description: Run ucas, the computer algebra system for programmers, from the shell (no MCP needed) to prove two integer formulas equal on their whole domain (enumerated or by SAT), find where an expression wraps or hits C undefined behaviour, get the exact range of a value, search cheaper equivalent forms, price an expression under a CPU or GPU cost model, rewrite C99 into proved faster C for an x86-64 target, and simplify a SPIR-V or GLSL shader with each rewrite proved. Use when a change rests on integer arithmetic that must be right for every input (an encoder, a decoder, a recompiler's lowering, a bit trick, a clamp), when a hot path needs its cost on a given core, or when a sampled test is all there is and a proof is wanted.
triggers:
  - ucas
  - usa o ucas
  - prova com o ucas
  - prove these are equal
  - is this formula right for every input
  - where does this overflow
  - cheaper form of this expression
  - cost of this on zen4
allowed-tools:
  - Bash
  - Read
---

# ucas: proofs and costs for integer and float formulas, from the shell

DraVee, 2026-11-11: "E precisa usar o ucas sem ser por mcp, faca a skill de acordo com o readme e tals (seria ate bom
colocar como PR la)". The MCP server (`ucas --mcp`) is there only when a client loads it; the command line always is.
Everything here follows ucas's README.md, which is the reference: this file is the short way in.

## Use

`ucas.sh`, beside this file, finds the binary and runs it with the same arguments: `$UCAS`, else `ucas` on PATH,
else `UCAS_BIN` in the nearest `.claude/skills.env` above the working folder, else this repository's `./ucas`
(after `make`). `ucas.sh which` prints the one it found; none is an error that says how to get one.

Three ways to talk to it:

```sh
U=.claude/skills/ucas/ucas.sh           # or the path of this folder
$U -e 'range (x: u8) / 3'               # one statement; -e again for the next, in order
$U file.ucas                            # a file of statements, each echoed before its answer
$U --json -e '{"cmd": "range", "args": {"f": "isqrt(x: u4)"}}'   # one JSON request, one JSON answer
```

The language (README, "Language"): `f(x: u32, y: i16 in -5..5) = expr`; machine types `u1..u64`, `i1..i64` wrap, `int`
and `real` are the ideal ones, `f32`/`f64` round; `^` is power (`xor` is exclusive or); `[t0, t1][i]` is a table;
`match`, `?:`. A parameter's range (`in a..b`) narrows the domain, and a domain of up to 2^22 points is enumerated.

## Recipes (each ran as written)

Two formulas equal for every input:

```sh
$U -e 'avg(x: u32, y: u32) = (x & y) + ((x xor y) >> 1)' \
   -e 'avg2(x: u32, y: u32) = u32((u64(x) + u64(y)) >> 1)' -e 'equiv avg avg2'
# avg and avg2 are equal on the domain of avg (proved by SAT: 745 AIG nodes, 818 conflicts)
```

A difference comes back with an input that shows it (`differ at ... e.g. x=..., y=...`).

The same as data, for a script or an agent (a function given as an object with typed parameters):

```sh
$U --json -e '{"id": 1, "cmd": "equiv", "args": {"f": {"name": "avg", "body": "(x & y) + ((x xor y) >> 1)",
  "params": [{"name": "x", "type": "u32"}, {"name": "y", "type": "u32"}]}, "g": "u32((u64(x) + u64(y)) >> 1)"}}'
# {"ok": true, ..., "proof": {"kind": "sat", "text": "proved by SAT: 745 AIG nodes, 818 conflicts"}, ...}
```

The exact range, and whether it fits a smaller type:

```sh
$U -e 'r(c: u32 in 0..65535) = (c * 255 + 32895) >> 16' -e 'range r'    # 0..255, fits in u8, all 65536 points
```

Where an operation wraps, or C assumes it does not:

```sh
$U -e 'bad(x: u32, y: u32) = (x + y) >> 1' -e 'wrap bad'     # nuw/nsw per operation, with a witness
$U -e 'f(x: i32) = (x * 2) / 2' -e 'domain f'                 # "C assumes 2x does not overflow i32"
```

Cheaper forms, checked on the whole domain (`more` continues the search):

```sh
$U -e 'q(x: u32 in 0..1000) = (x * 4) % 16' -e 'find q 3'    # x << 2 & 0xfffff00f (mask), cost 3
```

A cost under a model (README, "Cost models" and "x86-64"): `gt1030` (a GPU through SPIR-V) is active by default and
is what `find` prices with; `x86 models` lists the cores (`skl`, `skx`, `glc`, `rpl`, `zen4`, `x86v2`..`x86v4`), each
usable as `model NAME` or `use NAME`; `mca F on zen4` gives a cycle-level bound (critical path, ports). The x86 numbers
were typed in from uops.info, Agner Fog and the vendors' manuals and are approximate: a cost ranks forms, a benchmark
decides.

```sh
$U -e 'cost (x * 3) >> 1 model zen4'                          # 4: shift 1, imul 3
```

C99 in, proved faster C out for one x86-64 target (README, "C frontend"): `nofile` prints instead of writing FILE.opt.c,
`check` compiles both with the host cc and compares them on edge and random inputs, `bench` times both, `assume x 0 1000`
is a contract on an input, `wrapv` exploits no signed overflow. Each rewrite names its proof (exhaustive, sat, algebra,
induction, structural, lanes); the vectoriser's loops are proposed, not applied, unless `bench` measures them faster
than clang's or `vectorize` is given.

```sh
$U -e 'c optimize file.c target zen4 nofile'
```

A shader's integer instructions simplified, each rewrite proved by SAT (README, "Commands", `simplify`; GLSL needs
glslangValidator and spirv-opt on PATH, and what it finds is beyond `spirv-opt -O`). On the command line it is the
JSON command `spirv_simplify` (`code` is GLSL, SPIR-V assembly or C; `ranges` narrows an input; `nofile` skips
writing `simplified.spvasm`):

```sh
jq -nc --rawfile c shader.comp '{"cmd": "spirv_simplify", "args": {"code": $c, "lang": "glsl", "nofile": true}}' \
  | $U --json --files none
# ... 2 rewrites proved by SAT ...: min(x, 300) where x = v & 255 "equals its operand"
```

For GLSL it also works on whole functions (README, "Function level"): a branch the callers never take, a mask the
arguments never need, a loop iteration that never wins, a call the same on every pass, each proved by `equiv` over the
domain an abstract interpretation from `main` gives, and listed with what it relies on (`result.function_level` in
JSON). The module is not changed by that part; the GLSL to use is in the report. Float rewrites (README, "Float
simplify") are derived, not sampled, and kept within the original's Vulkan precision unless `exact` or `ulp N` asks
for less.

## Reading an answer

- What the answer rests on is said with it: `over all N points of the domain` (exhaustive), `proved by SAT`,
  `by algebra`, or `sampled 65536 points ... not exhaustive`. In JSON, `proof.kind` is `exhaustive`, `sat`,
  `symbolic` or `numeric`. A sampled answer is evidence, not a proof: say "sampled" when you pass it on.
- SAT gives up on hard multipliers (100000 conflicts or 2 s): the answer then stays "sampled, not proved" and says
  why. Narrow the inputs with ranges (`in a..b`) or a `where` clause, or split the formula, and ask again.
- JSON never guesses: an `ambiguous` error lists the readings and the `args.assume` key that picks one; answer it,
  never retry with `defaults: true` to make it go away unless the default is the reading you mean.
- Its own limits (README, "Limits"): `gates` stops at 6 input bits, `find` builds terms of about 5 or 6 operations,
  `approx` measures on a sample. A "no cheaper form found" there is the end of a search, not a proof that none exists.

## Rules

- Quote the statement that ran and the proof kind with any result you report; the reader must be able to run it.
- English mode (`english`, or prose to the Discord bot) translates guesses into statements; for a proof, write the
  statements yourself.
- A long search gets `"timeout_s"` in JSON (it runs in a forked child, the session is untouched on a timeout).
- `--dir DIR` (or `--files none`) keeps the GIFs and modules some commands write out of the working folder.

## Test

`sh .claude/skills/ucas/test.sh`: runs the recipes above against the binary `ucas.sh` finds and checks their answers
(the SAT proof, the counterexample, the range, the JSON proof kind, the wrap witness), and that `ucas.sh` refuses
with no binary. Offline, a few seconds. With no binary at all it fails with ucas.sh's message.
