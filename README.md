# ucas

A computer algebra system for programmers.  It is built for questions
like these:

- what is the cheapest program for this function on *this* machine (a
  gate network for a 4-bit square root, a LOP3 table, a switch that is
  secretly `2x + 3`)?
- which inputs does this function actually work on, where does it wrap,
  and what does a C compiler get to assume about it?
- where are the critical points, what is the range?
- how good is this IEEE bit hack, and what is its best magic constant?

It is not a better assembler (that is Souper's job).  It works on the
mathematical function: it rewrites it, measures it on its domain, and
searches for other implementations.

C99, BSD make, no dependencies beyond libm.

```sh
make            # FreeBSD make or bmake
make test       # tests/*.ucas must print tests/*.out
./ucas file.ucas    # run a file (each statement is echoed)
./ucas -e 'range isqrt(x: u4)'
./ucas          # interactive
```

## A tour

```
> isqrt4(x: u4) = isqrt(x)
> gates isqrt4
gates for isqrt4: 4 input bits -> 4 output bits (2 constant), basis c
  minimal: cost 5, 5 steps (no program of cost <= 4 exists; 660213 candidates searched)
  t0 = or(x0, x1)                = x0 | x1
  ...
> gates isqrt4 basis a64sha
  minimal: cost 3, 3 steps ...
  t2 = bsl(x2, x3, t0)           = x2 & x3 | ~x2 & t0

> days(m: u8 in 1..12) = match m { 2 => 28, 4 | 6 | 9 | 11 => 30, _ => 31 }
> find days
  cost 6   u8(0x3bbeecc >> (u32(m) << 1) & 3) + 28  packed table in one constant
  cost 9   [31, 28, 31, ...][m - 1]                lookup table (baseline)

> rsqrt(x: f32 in 0.01..100) = 1/sqrt(x)
> guess(x: f32, K: u32) = asfloat(K - (asuint(x) >> 1))
> fit guess ~ rsqrt
  K = 1597465581  (0x5f3763ed)
  max rel error 0.0342 ...
> quake(x: f32) = y * (1.5f - 0.5f * x * y * y)
    where y = asfloat(0x5f3759df - (asuint(x) >> 1))
> error quake ~ rsqrt
  max rel error 0.00175224 (0.175%) ...

> s(x: i8) = x * 2 + 1
> wrap s
  2x   (i8)
      nsw fails: signed wrap at 128 points, e.g. x=-128  <- UB in C; ...
  end to end: differs from the mathematical result at 128 points
      exact for: -64..63
```

`tests/` has one file per area; they double as examples.

## Language

```
f(x: u8, y: i16 in -5..5) = expr  [where name = expr, condition, ...]
g = x^2 + 1                  # free variables become parameters, in order
```

Types:

| type | meaning |
|------|---------|
| `real` | the real line; `sqrt(-1)`, `1/0` are *undefined* (outside the domain) |
| `f32`, `f64` (`float`, `double`) | IEEE binary32/64: every result rounds, `1/0` is inf, `sqrt(-1)` is NaN |
| `int` | unbounded integers (checked against the int64 range) |
| `uN`, `iN` (1..64; also `uint8_t`...) | wrapping machine integers |
| `bool` | 0 or 1 |

`real` and `int` are the *ideal* twins of the machine types: `wrap`
compares a function against itself evaluated with every `uN`/`iN` as
`int` and every `f32`/`f64` as `real`.

A parameter can carry a range (`x: u8 in 1..12`, or just `x in 0..9`),
and `uniform` (its value is the same across a GPU warp; see `cost`).
Untyped parameters are `real`, or `int` when the body uses integer-only
operators (`& | xor << >> % ~ [] match`), with a note.

Expressions: `+ - * / % ^` (power), `mod`, `& | xor ⊕ ~ << >>`,
comparisons (chains allowed), `&& || ! and or not`, `c ? a : b`,
`x in a..b`, `|x|`, `[t0, t1, ...][i]` (lookup table), `x²`,
functions `sin cos tan exp log ln sqrt isqrt abs floor ceil sign min max
clz ctz popcount`, conversions `u8(x) f32(x) real(n) int(x)`, bit casts
`asuint asint asfloat` (`bits` = `asuint`), effects `barrier(e)`,
`atomic_{add,sub,min,max,and,or,xor,xchg}[_shared](x, v)`.  C float
literals (`1.5f`) are `f32`.

```
match x { 0 => 3, 1 | 2 => 5, 3..9 => 7, _ => 9 }   # also switch, ':' or '->'
{ x if x >= 0, -x otherwise }                         # piecewise
```

A `match` or piecewise without a default arm is *partial*: the other
inputs are outside the domain, and `gates` treats them as don't-cares.

### How ambiguities are resolved

Each of these prints a `note:` when it applies.

| input | read as |
|-------|---------|
| `2x`, `3(x+1)`, `2 sin x` | multiplication, binding tighter than `*` `/` |
| `1/2x` | `1/(2x)` |
| `sin x^2`, `sin 2x` | `sin(x^2)`, `sin(2x)`: the juxtaposed product is the argument |
| `sin x cos x` | `sin(x) * cos(x)` |
| `x(y+1)` | `x * (y+1)` when `x` is not a function |
| `xy` | `x*y` when `xy` is undefined and `x`, `y` are known (or, in an anonymous expression, a two-letter name) |
| `\|x\|` | absolute value where an operand is expected; `\|` is bitwise or after an operand |
| `a < b < c` | `a < b && b < c` |
| `x = 0` inside an expression | `x == 0` |
| `a & b == c` | `(a & b) == c` (bitwise binds tighter, unlike C) |
| `sqrt(n)` with integer `n` | `isqrt(n)` |
| `1/2` | `0.5`: two integer literals that do not divide give a real |
| `^` | power; `xor` / `⊕` is exclusive or |
| `2e` vs `2e5` | `2*e` vs the number |
| a newline inside `( )` | continues the statement only if the next line is indented |

## Commands

A command's function argument is a name or an anonymous expression with
inline annotations: `range isqrt(x: u4)`.

| command | |
|---------|--|
| `show F`, `table F [max n]`, `parse EXPR`, `funcs`, `help` | look |
| `range F` | min/max (exhaustive for finite domains, distinct values, smallest type); for one real parameter a grid refined at the critical points; IEEE NaN/inf counts |
| `domain F` | the conditions each partial operation imposes, the C assumptions (signed overflow, shift amounts, `__builtin_clz(0)`), and the inputs where the function is defined |
| `wrap F` | per operation: whether LLVM's `nuw`/`nsw` hold (with a witness), C UB, truncating conversions; for floats `nnan`/`ninf` (fast-math) per operation; end to end: where the machine result equals the mathematical one, or the rounding error in ulp |
| `crit F` | real: derivative, roots (bisection), min/max/inflection, kinks; integer: the piecewise-affine segments and local extrema; several reals: the gradient |
| `diff F [x]` | symbolic derivative, or the forward difference for an integer parameter |
| `equiv F G` | equal on F's domain? (with a counterexample) |
| `gates F [basis b] [max k] [as c]` | minimal gate network over a basis, with proof of minimality; `as c` also prints a C function |
| `find F [n] [max cost]`, `more [n]` | other implementations, cheapest first under the active cost model; `more` resumes the search |
| `fit G ~ F [metric maxrel\|maxabs\|rms]` | G's extra parameters are constants: choose them to minimise the error against F |
| `error G ~ F` | max/mean relative error, max absolute error, ulp |
| `cost F [model m]`, `use m`, `models` | price F under a cost model |

Finite domains of up to 2^22 points are enumerated; larger ones are
sampled (65536 points, edge values first), and the output says so.
`wrap` and `range` add interval bounds when sampling, which can prove
that an operation never wraps.

## Bases and exact synthesis

```
basis c      { and(x, y) = x & y; or(x, y) = x | y; xor(x, y) = x xor y; not(x) = ~x }
basis spirv : c {}
basis a64 : c    { bic(x, y) = x & ~y; orn(x, y) = x | ~y; eon(x, y) = x xor ~y }
basis a64sha : a64 { bsl(s, x, y) = (s & x) | (~s & y); eor3(...); bcax(...) }
basis x86bmi : c { andn(x, y) = ~x & y }
basis nand   { nand(x, y) = ~(x & y) }
```

A gate is declared by its expression over 1-bit inputs, optionally with
`cost n`.  ucas derives its truth table and which arguments commute from
the expression; nothing else has to be stated.  `gates` takes functions
of up to 6 input bits.  It runs iterative deepening on cost with the
pruning rules of basis.md §5.2 (no duplicate or constant values, ordered
interchangeable arguments, canonical order of independent steps), using
the domain's don't-cares.  The search budget is 2^27 candidate steps
(`nodes N` changes it).  When it runs out, ucas falls back to Shannon
decomposition: split on an input, solve both halves, join them with the
cheapest combination the basis allows.  That gives a checked program and
an upper bound, but not a good one (17 gates for a 3-bit adder that
needs about 9).

## Cost models

```
model gt1030 { iadd: 1, logic: 1, shift: 2, imul: 3, idiv: 24, idivk: 5,
  cmp: 2, select: 1, minmax: 2, bitcnt: 4, conv: 4, bitcast: 0,
  fadd: 1, fmul: 1, fdiv: 10, fsqrt: 8, ftrans: 4, fpow: 10, f64: 32,
  load: 8, branch: 2, diverge: 6, barrier: 20,
  atomic_shared: 8, atomic_global: 64 }
model mine : gt1030 { imul: 1 }
```

`gt1030` (NVIDIA GP108, sm_61, through SPIR-V) is built in and active.
Its numbers are relative issue costs from the sm_61 throughput table:
fp32 is full rate, 32-bit integer multiply is an XMAD sequence, integer
division is emulated, the SFU ops are quarter rate, fp64 is 1/32 rate
(`f64` multiplies the cost of every f64 float operation).  They are
estimates, not measurements.

`cost F` counts each shared subterm once and prices every `?:` three
ways:

- as a select (both arms run);
- as a uniform branch (branch cost plus the dearer arm);
- as a divergent branch, where the warp runs both arms plus the
  reconvergence penalty: the "keep-up" cost.

A condition is uniform when it depends only on `uniform` parameters.
Barriers and atomics have value semantics for one invocation
(`barrier(e)` is `e`, an atomic returns the old value), so they can
appear in expressions and be priced.  They are never merged by
hash-consing.

`find` takes its operation costs from the active model.

## Design

Data-oriented throughout (see `src/ucas.h`):

- **Tokens** are four parallel arrays (`kind`, `flags`, `offset`,
  `length`); text is decoded only when a token is used.
- **Nodes** are a hash-consed DAG in parallel arrays.  The constructors
  infer types, insert conversions, simplify locally (only identities
  that are exact: no `x+0 → x` on IEEE floats, nothing that drops a
  partial subterm) and fold constants with the evaluator's own
  semantics.  An id is always larger than its operands', so id order is
  a topological order.
- **Plans** copy the nodes one query needs into dense local arrays and
  evaluate a chunk of 256 domain points one node at a time: every
  operation is a loop over a column.  Each value carries a flag byte:
  undefined, wrapped unsigned, wrapped signed, C UB, truncated, NaN,
  inf.  A backward liveness pass over the same columns ignores flags
  from `?:` arms and `&&` operands that C would not evaluate.
- **find's term bank** is struct-of-arrays with all value columns in
  one flat array, deduplicated by column hash (observational
  equivalence).
- Float domains are enumerated in IEEE key order, so `x: f32 in 1..2`
  is every float in that interval.

## Limits (for now)

- `gates` stops at 6 input bits (64-entry truth tables), and its
  fallback beyond the exact search is weak.  The next steps are
  bit-blasting the word-level expression as a starting upper bound and
  SAT-based exact synthesis.
- `crit`/`domain` solve numerically for one real parameter; several
  real parameters get the gradient only.
- `find` builds terms of up to about 5-6 operations before the bank
  (128 MB) fills; constants come from a small pool plus solved holes.
- `fit` is a scan followed by a pattern search, not a global optimiser.
