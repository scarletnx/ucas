# cfront: C in, uber-optimised C out

Goal: `ucas` takes C99 source (a function, a file, a translation unit) and
rewrites it into the fastest equivalent C it can prove equivalent, for a
chosen x86-64 target.  The output is allowed to be unreadable (intrinsics,
branchless bit tricks, magic constants, tables, unrolled/vectorised loops,
strength-reduced divisions) but it must be correct: every rewrite is proved
(exhaustively, by SAT/SMT via src/sat.c and nodesat.c, or by algebra), or
reported as unproved and not applied unless the user opts in (`approx`,
`ulp N` as in float simplify).  It is the C counterpart of what `simplify`
already does for SPIR-V/GLSL (README "Float simplify", the SPIR-V simplify
row) and of `find`, `gates`, `approx`, `range`, `wrap`, `equiv`.

House rules as everywhere: C99, BSD make, libm only, data-oriented (SoA
tokens/AST/IR, arenas, handles), no external compiler needed for the core
path (the host `cc` may be used, like glslc.c uses glslangValidator, only for
optional cross-checks: compile both versions, run them on random and edge
inputs, benchmark with rdtsc).  Each area: tests, JSON commands, MCP group
(e.g. `c`), README section; `make test` stays green.

## Worktrees

| worktree | branch | owner |
|---|---|---|
| `~/ucas-cfront` | `cfront` (integration) | F1: frontend, IR, semantics, optimiser, C emitter |
| `~/ucas-cfront-x86` | `cfront-x86` | F2: x86-64 cost models, intrinsics catalogue, vectoriser |

## F1: the frontend and the rewrite pipeline

- Preprocessor (enough of it: #define/function macros, #if/#ifdef, #include
  of the user's files; system headers recognised by name with built-in
  declarations of the standard integer types, stdint, math.h, string.h
  basics, immintrin.h types).
- C99 parser (SoA AST), types (integer promotions, usual arithmetic
  conversions, exact-width types, pointers, arrays, structs, unions, enums,
  typedefs, const/restrict/volatile), declarations, statements, expressions;
  errors with line/column.
- Semantics faithful to C99 on LP64 x86-64: UB (signed overflow, shifts,
  division by zero, out-of-bounds, strict aliasing as an option), unsigned
  wrap, implementation-defined choices stated.  Lower to the ucas node DAG
  (src/node.c) for straight-line code, to calc loops (calc.h) for loops,
  plus a memory model for arrays/pointers (SSA over memory; restrict and
  noalias facts; loads/stores).  What is outside the model (I/O, volatile,
  calls to unknown functions) stays as an opaque barrier, never miscompiled.
- The pipeline: per function, per loop, per expression run the existing
  engines (simplify, find, range/where narrowing, wrap analysis, closed
  forms of loops, approx when allowed, gates for bit functions, table
  packing, division by constants, popcount/clz idioms, select vs branch by
  cost) plus classic optimisations ucas can prove (CSE, LICM, strength
  reduction, loop closed forms, unrolling, bounds-check elimination by
  range proofs, branch-free conversion) and F2's vectoriser.  Choose by
  F2's cost model.
- Emitter: valid, compilable C99 (+ <immintrin.h> intrinsics for the chosen
  target, with a scalar fallback guarded by feature macros), keeping the
  function's signature/ABI; a report of each rewrite with its proof kind.
- Verification: SAT/exhaustive/algebraic equivalence per rewrite; end-to-end
  differential testing by compiling original and output with the host cc
  (clang 19) on random+edge inputs; optional rdtsc benchmark.

## F2: x86-64 cost models and vector code

- Cost models as `model` declarations (README "Cost models") for x86-64
  µarchs: at least Zen 4, Golden Cove / Raptor Lake P-core, Skylake(-X),
  plus a generic x86-64-v2/v3/v4 level: latency, reciprocal throughput,
  ports/uops per instruction form (from uops.info / Agner Fog / Intel &
  AMD optimisation manuals as the cited source of the numbers), a simple
  throughput-vs-latency scheduler estimate for straight-line code and loop
  bodies (critical path vs port pressure, à la llvm-mca, simplified).
- An intrinsics catalogue (SSE2..SSE4.2, AVX, AVX2, FMA, BMI1/2, POPCNT,
  LZCNT, AVX-512F/BW/DQ/VL/VBMI/VPOPCNTDQ/GFNI/VPTERNLOG where it matters,
  PCLMUL, AES-NI), each with exact semantics as ucas expressions over lanes
  (so rewrites into them are provable), its required feature flag and its
  costs per µarch.
- The vectoriser: SLP and loop vectorisation over the IR from F1 (reductions,
  masked tails, gathers only when they pay, shuffles), choosing widths by
  cost; idiom recognition into single intrinsics (pshufb tables, vpternlog
  for 3-input logic — basis.md's LOP3 result applies directly —, pmaddubsw,
  psadbw, pdep/pext, mulx/adcx chains, lzcnt/tzcnt, vpopcnt, gf2p8affine).
- `gates`/`find` bases for x86 (add `x86` bases: andn, blsr, blsi, bzhi,
  pdep, pext, vpternlog as a 1-op 3-input gate) so the existing searches
  pick x86 instructions.
