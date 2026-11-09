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
clz ctz popcount sec csc cot atan copysign fma`, Lambert W (`W(x)`, also `W0`,
`lambertw`, `productlog`: the principal branch; `Wm1(x)`: the -1 branch; a
bare `W` without a call stays a variable), conversions `u8(x) f32(x) real(n) int(x)`, bit casts
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
| `range F [nocrit]` | min/max (exhaustive for finite domains, distinct values, smallest type); for one real parameter a grid refined at the critical points; IEEE NaN/inf counts |
| `domain F` | the conditions each partial operation imposes, the C assumptions (signed overflow, shift amounts, `__builtin_clz(0)`), and the inputs where the function is defined |
| `wrap F` | per operation: whether LLVM's `nuw`/`nsw` hold (with a witness), C UB, truncating conversions; for floats `nnan`/`ninf` (fast-math) per operation; end to end: where the machine result equals the mathematical one, or the rounding error in ulp |
| `crit F` | real: derivative, roots (bisection), min/max/inflection, kinks; integer: the piecewise-affine segments and local extrema; several reals: the gradient |
| `integrate F [wrt x] [from a to b]` | quotients are cancelled first (x/x, (x^2-1)/(x-1), x^2/x^3), and the answer says where that holds (x != 0); removable points do not affect a definite integral.  A constant integrand names no variable: asked (terminal, bot) or x with a note; `wrt t` picks one.  A unary function's name alone means f(x): `integrate W0` |
| `diff F [x]` | symbolic derivative, or the forward difference for an integer parameter |
| `equiv F G` | equal on F's domain? (with a counterexample) |
| `gates F [basis b] [max k] [as c]` | minimal gate network over a basis, with proof of minimality; `as c` also prints a C function |
| `find F [n] [max cost]`, `more [n]` | other implementations, cheapest first under the active cost model; `more` resumes the search |
| `fit G ~ F [metric maxrel\|maxabs\|rms]` | G's extra parameters are constants: choose them to minimise the error against F |
| `error G ~ F` | max/mean relative error, max absolute error, ulp |
| `cost F [model m]`, `use m`, `models` | price F under a cost model |
| `animate [hillis\|blelloch\|sequential] scan [n N] [fps F] [seconds S] [random [seed K] \| values ...] [f32] [MODEL] [to NAME]` | an animated GIF of a prefix-sum scan (n capped at 100): one row per level, values updating, the critical-path time and the work under the cost model advancing, with barriers and divergence on a timeline; also a table of the three algorithms' costs.  Filler words are skipped, so `animate random sample prefix sum scan n=8 at 30 fps using gt1030` works as written.  GIF delays are whole hundredths, so 30 fps alternates 3/3/4 cs.  On the command line it writes NAME.gif (default scan.gif); in Discord it is attached |
| `approx F in a..b [options]`, `approx more [n]` | polynomial, rational and range-reduced approximations, measured as they run (see below) |

Finite domains of up to 2^22 points are enumerated; larger ones are
sampled (65536 points, edge values first), and the output says so.
`wrap` and `range` add interval bounds when sampling, which can prove
that an operation never wraps.

## Approximations

```
> approx sin(x) in -pi/2..pi/2 degree 5..9 keep endpoints norational nosplit noreduce
approx sin(x) on [-1.57079637, 1.57079637] as f32, fma
  checked: sin(x) is odd (f(-x) = -f(x) at 64 points): fitting x*P(x^2) on [0, 1.57079637]
  metric: relative error
  exact at: x = -1.57079637 -> -1, x = 1.57079637 -> 1
  fits in real arithmetic (max relative error of x*P(x^2)):
    deg  cheb interp  cheb econ    minimax      no exact pts
    5    0.000348     -            0.000131     0.000108
    7    3.02e-06     -            1.09e-06     9.39e-07
    9    1.71e-08     -            5.98e-09     5.31e-09
  ...
  #   kind      deg   form        error (naive)     max rel    max abs    cost
  3   minimax   9     horner+fma  1.53 ulp (1.81)   9.1e-08    9.09e-08   6    at x=-1.5408833  exact pts yes  keep +12%
      x*P(x^2): fma(x*s, fma(fma(fma(2.59965395e-06, s, -0.000198068723), s,
                0.00833303016), s, -0.166666582), x) where s = x*x
```

`approx F in a..b` looks for cheap approximations of F on the interval
and ranks them by error and by cost under the active model.  F is any
one-variable function ucas can evaluate: a builtin, an expression, or a
named function (whose range and type are used when `in` is omitted).
Every candidate is written as ucas source, parsed back, priced with
`cost`'s rules (each shared subterm once, `?:` as a select) and measured
with the IEEE evaluator against a long double reference, so the numbers
are those of the expression as printed.  The listed candidates are
defined as `ap1`, `ap2`, ... (`name p` changes the prefix), so `error`,
`cost`, `wrap` and `show` work on them.

| option | |
|--------|--|
| `as f32`, `as f64`, `as real` | target format (default: the parameter's float type, else f32); `real` fits without rounding |
| `as c` | also print each listed candidate as a C function (`fmaf`, `copysignf`, `memcpy` bit casts) |
| `degree n`, `degree a..b` | degree range (in x) for every kind of candidate |
| `error e`, `ulp u` | a target: candidates that meet it are listed first, cheapest first, and the search stops at the first degree that meets it |
| `relative`, `absolute` | the error to minimise; by default relative unless F has a zero that a polynomial factor cannot match |
| `odd`, `even`, `nosym` | symmetry; by default detected numerically (64 points) |
| `keep endpoints`, `keep x1, x2, ...` | points where the approximation must be exact (in real arithmetic; after rounding the list says whether the machine result equals the rounded value there) |
| `keep crit` | also exact at the critical points of F inside the interval (found from the derivative), so the extrema survive |
| `method cheb`, `method econ`, `method minimax` | which real-arithmetic fit seeds the machine candidates: Chebyshev interpolation, Chebyshev-economised Taylor, or minimax (the default) |
| `compare` | with a named `method`: also search the other shapes (rational, piecewise, range reductions); without it a named method gets a focused report: its polynomial as an expression and in the Chebyshev basis, for the requested degree or the lowest one reaching the precision |
| `free` | do not ask about (or note) special points at the interval's ends |
| `weight W` | an extra weight function of x for the minimax error |
| `at c` | expansion point for a non-symmetric fit (default: 0 when inside, a zero at an endpoint, else the midpoint) |
| `fma`, `nofma` | Horner/Estrin with or without `fma` (default with) |
| `norefine` | keep the naively rounded coefficients |
| `norational`, `nosplit`, `noreduce` | skip rational fits, two-piece splits, range reductions |
| `list n`, `points n` | how many to list (default: the Pareto front, at most 8); sample size (default 131072) |
| `approx more [n]` | the next candidates of the last `approx` |

What it does, in order:

- **Shape.** Odd and even functions (checked at 64 points, not
  assumed) are fitted as `x*P(x^2)` / `P(x^2)` on the folded interval.
  Otherwise P is expanded at 0, a zero at an endpoint, or the midpoint;
  a simple zero there (checked: f(x)/(x-c) settles) becomes a factor,
  `(x-1)*P(x-1)` for log on [1, 2], so the relative error stays bounded.
- **Fits in real arithmetic** (`src/remez.c`, long double): Chebyshev
  interpolation, Chebyshev economisation of the Taylor series (computed
  exactly by Taylor-jet arithmetic over the expression), Padé, and the
  Remez exchange for weighted polynomial and rational minimax (absolute,
  relative, user weight).  The table shows the starting points beside
  the minimax result.  Exact points are folded in as
  `P = L + Z*Q` (L interpolates, Z vanishes there), so the constrained
  problem is an ordinary weighted one; the `keep +12%` column is the
  extra minimax error the constraint costs.
- **Machine form.** Coefficients are rounded to the format, then
  re-optimised: one coefficient is rounded at a time and the higher ones
  refitted, a leading coefficient near 1 is pinned to 1 (written as
  `x + x*s*Q(s)`, which makes the leading term exact), then each
  coefficient is walked by a few ulps while the measured error drops.
  The naive rounding is measured the same way and shown in parentheses;
  if refinement did not pay off on the full sample the naive one is
  kept.  Horner and Estrin forms (with or without fma) are priced; under
  a throughput model like gt1030 Estrin never costs less (it wins on
  latency, which the model does not price), so it appears further down.
- **Rational** fits P/Q compete at equal footing; they win only when
  division is cheap (`model m : gt1030 { fdiv: 2 }` makes tan on
  [-1.4, 1.4] rational).
- **Piecewise:** two pieces with coefficients chosen by a select, on
  |x| with `copysign(.., x)` for odd functions.
- **Range reductions and identities** apply after the target is checked
  numerically to be the builtin (33 points), and each is measured end
  to end:
  sin/cos by quadrant (`k` and the quadrant from the bits of
  `x*2/pi + 1.5*2^23`, Cody-Waite constants in 2 or 3 parts, the sign
  flipped by xor-ing bit 31 of `asuint`); `cos x = sin(pi/2 - x)` with
  pi/2 in two parts; `cos x = 1 - 2 sin(x/2)^2`; `tan = sin/cos`,
  `sec = 1/cos`, `csc = 1/sin`, `cot = cos/sin`; `tan x = 2t/(1-t^2)`
  with `t = tan(x/2)`; exp as `2^k * P(r)` with 2^k built in the
  exponent field (`asfloat((asuint(j) << 23) + 0x3f800000)`); log by
  splitting exponent and mantissa (`m` in [sqrt(1/2), sqrt(2))) as
  `e*ln2 + m*P(m)`; sqrt and 1/sqrt by halving the exponent, with a
  polynomial or a short polynomial seed and one Newton step, and the
  bit-hack seed with one or two Newton steps; atan of |x| > 1 as
  `pi/2 - atan(1/|x|)` with `copysign`.

Errors are measured on a sample of the format's values in the interval
(half evenly spaced in value, half evenly spaced in bit pattern, plus
the floats around each zero of F, the exact points and 0); the ulp is
that of the exact value, so a correctly rounded result has at most 0.5.
`error G ~ F` compares against F rounded to the format, so its ulp count
can differ by about one half.

## English

A line that reads as English is translated into ucas statements, which
are printed (`ucas: ...`) and run, so every answer also shows the syntax
behind it.  English only; the translator is a hand-written parser
(`src/nlp.c`) with no dictionary beyond its own phrase tables.

```
> what is the range of the integer square root of a 4-bit unsigned x?
  ucas: it(x: u4) = isqrt(x)
  ucas: range it
> is avg comutative?
  note: read "comutative" as "commutative"
  ucas: lhs(a: u8, b: u8) = avg(a, b)
  ucas: rhs(a: u8, b: u8) = avg(b, a)
  ucas: equiv lhs rhs
> cos of pi/2 to -pi/2 w/o crit. points
  note: read the interval as [-pi/2, pi/2]
  ucas: it(x in -pi/2..pi/2) = cos(x)
  ucas: range it nocrit
> cos from -90deg to 90deg using cheby poly, yes critical
  ucas: it(x in -(90*pi/180)..90*pi/180) = cos(x)
  ucas: approx it method cheb keep crit
> alt ((1 << 3) & x) + x*5
  ucas: it(x) = (1 << 3 & x) + x*5
  ucas: find it 10
```

- **When a line is English**: it ends in `?`, starts with `?`, starts
  like a sentence (`what`, `is`, `does`, `let`, `alt`, ...), a command
  word is followed by an English one (`range of`, `find cheaper`), or
  it has two or more English words ucas has no use for.  Definitions
  (`f(x) = ...`) always stay ucas.
- **Requests**: range, domain, wrap (`does it overflow`), crit, diff
  (`with respect to y`), equiv (`is A the same as B`), error (`how
  accurate is G as an approximation of F`), fit (`the best constant
  for G to approximate F`), gates (`using arm64 with sha3`, `in C`),
  find (`cheaper implementations`, `alt`: 10 by default), cost, table,
  approx (`cheby poly`, `minimax`, `taylor`, `degree 4`, `f64`),
  integrate (`the integral of ... from 0 to pi`, `antiderivative`),
  closed (`the closed form of ...`), scheme (`the best evaluation
  scheme for ...`), continuous, dsolve (`solve y from 0 to 2 in 4
  steps`), sums and products in words (`the sum of i squared for i
  from 1 to n`, `the product over k from 1 to 5 of k`),
  properties checked with equiv (commutative, associative, additive,
  idempotent, odd/even function), definitions (`let f of x be ...`,
  `call it g`), `assume`, `forget`.
- **Expressions**: `x squared plus one`, `the sum of a, b and c`, `half
  of x`, `20 percent of y`, `the number of set bits in x`, `x shifted
  left by 3`, `e to the power of 2`, mixed freely with ucas notation.
  Number words, `90deg`, `90 degrees`, `90°` (angles, converted to
  radians), `1.5r`/`rad`.  Greek letter names (`theta`, `alpha`, or
  `θ`) are variables; one or two letters are variables; `it`/`its`
  refer to the last function.
- **Variables**: `for a signed byte x`, `where x is between 1 and 12`,
  `x and y are unsigned bytes`, `when x is 3` (substituted), `on [a, b]`
  or `from a to b` for the only variable, `cos of a to b` (an implicit
  x); a reversed interval is put in order.
- **Ambiguity**: `byte` (u8/i8), `char`, `int` (i32 or unbounded),
  `integer`, `word` (16/32/64), N-bit without a sign, `log` (ln, log2,
  log10), `the square root` of an integer (isqrt or real), `and`/`or`/
  `not` between numbers, and scope (`the square root of x plus 1`, `the
  sum of x and y squared`).  On a terminal ucas asks; term meanings are
  kept for the session (`assume log means base 2` sets one).  Otherwise
  the default is taken and a `note:` says which and how to say the
  other.  `-i` always asks, `-y` never does.
- **Follow-up questions**: approx without an interval asks for one; the
  reply is read as the rest of the sentence, so options compound:
  `cheby poly nocrit`, `0 to pi/4 degree 4 crit`.  An empty reply takes
  the defaults (a usual interval for the function).
- **Misspellings**: words are corrected towards the words `src/nlp.c`
  knows (generated into `nlp_words.h` by `src/nlp_words.awk`) by
  optimal-string-alignment distance (1 edit for 5-6 letters, 2 for
  longer), including two halves of one word (`add ivite`).  Ties are
  asked about.  Names being defined, function and variable names are
  never corrected.

Two ucas additions serve it: `range F nocrit` (the grid only, no
refinement at critical points) and, for approx, `method cheb|econ|
minimax` (which fit seeds the machine candidates; minimax by default)
and `keep crit` (exact at the function's critical points inside the
interval, so its extrema survive).  `it`, `that`, `lhs` and `rhs` are
the names the translator defines.

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

## Loops and calculus

Iterative code is turned into mathematics that can be reduced, and back.

```
> closed sum(i, 1, n, i^2)
  note: n is a bound of sum: taken as int in 0..100
sum(i, 1, n, i^2)
  = n*(n + 1)*(2n + 1)/6
  method: Faulhaber (polynomial in the index)
  check: equals direct iteration at all 101 points of the domain

> fib[0] = 0; fib[1] = 1; fib[n] = fib[n-1] + fib[n-2]
> fb(n: int in 0..90) = fib[n]
> scheme fb
fib[n]
  iterating: 3 per iteration, up to 90 iterations: 270
  closed form: 36 (9 ops): int(floor(0.4472135955*1.618033989^real(n) - 0.4472135955*(-0.6180339887)^real(n) + 0.5))
    but it differs from the loop at 20 of 91 points (e.g. n=71: 308061521170129 by iteration vs 308061521170130): not used
  matrix form: state' = M state, M = [0 1; 1 1]; the answer is (M^k v0)[0]
    M^k by squaring: at most 14 matrix products of 28 each: 392 (O(log k))
    check: equals the loop at 91 points (in exact int64 arithmetic: no rounding, unlike the closed form's real powers)
    cheaper than iterating once k > about 131

> c(n: int in 1..1000) = { x = n; k = 0; while x > 1 { x = x / 2; k += 1 }; k }
> closed c
loop(while x > 1; x = n then x/2; k = 0 then k + 1).k
  = n > 1 ? 63 - int(clz(u64(n))) : 0
  method: constant body, accumulator, trip count
  trip count n > 1 ? 63 - int(clz(u64(n))) : 0 (checked against the loop at 1000 points)
  check: equals direct iteration at all 1000 points of the domain
```

### Syntax

| form | meaning |
|------|---------|
| `sum(i, lo, hi, e)`, `sum(i in lo..hi, e)`, `sum(i in lo..hi) e`, `Σ(...)` | the index runs `lo..hi` inclusive; an empty range is 0 |
| `prod(...)`, `∏(...)` | the same for products (empty is 1) |
| `x[0] = 1` then `x[k+1] = a*x[k] + b` (or `x[n] = x[n-1] + x[n-2]`) | a recurrence: initial values and one step rule, in any order; `x` becomes a function `x(n, a, b)` with `n in 0..60`, and `x[e]` can be used in expressions |
| `{ s = 0; for i in 1..n { s += i }; s }` | a code block: `=`, `+= -= *= /= %=`, `++ --`, `a, b = b, a + b`, `x: u32 = 0`, `for i in lo..hi { }`, `while c { }`, `if c { } else { }`, ending with an expression or `return e` |
| `y' = -2y + 1`, `y(0) = 3` | an ODE (any order: `y''`, systems `x' = y; y' = -x`) and its initial values |
| `pde name { u_t = 0.1 u_xx; u(x, 0) = sin(pi x); u(0, t) = 0; x in 0..1; t in 0..0.5 }` | a 1-D linear PDE (`u_t` or `u_tt`); also `periodic`, `grid n`, `u_t(x, 0) = ...` for waves |

Inference, with notes: a free variable used as a bound is an `int` in
`0..100`; a body that would divide integers by the index makes the index
`real` (`sum(i, 1, n, 1/i)`); in recurrences the type comes from the
initial values, and an integer division of the sequence makes it `real`.
Code blocks are C-like: no such rewriting, variables first assigned
inside a branch or loop body are local to it, and the loop body may not
assign the index.  A loop is one node (`O_LOOP`) in the DAG, so `table`,
`range`, `equiv`, `wrap`, `diff` (forward mode: the loop carries its
derivative) and `cost` work on loops as on any expression.

### Commands

| command | |
|---------|--|
| `closed F` | closed forms of the loops in F, innermost first, each checked against direct iteration on F's domain; a form that disagrees is rejected and shown as such |
| `scheme F [model m]` | evaluation schemes priced with the cost model: naive/products/Horner/Estrin for polynomials (cost and critical path), forward differences for tabulating, powers by squaring, and for loops: iteration vs the closed form vs the matrix power (O(log n), exact in int64) with the crossover trip count, and strength reduction of polynomial loop bodies |
| `continuous F` | sums as integrals plus Euler-Maclaurin corrections (products through the logarithm: Stirling), and loops that step `x += h*g(x)` as the ODE `x' = g(x)`; the measured difference is printed |
| `integrate F [wrt x] [from a to b]` (also `∫`) | a symbolic antiderivative, checked by differentiating it back; definite integrals also numerically (adaptive Gauss-Kronrod 7-15 with an error estimate; `inf` bounds allowed) |
| `dsolve y [from a] [to b] [steps n]` | ODEs: linear systems with constant coefficients symbolically, otherwise numerically (Dormand-Prince 5(4), adaptive); PDEs by finite differences |

What is found:

- sums: polynomial bodies (Faulhaber, factored over the rationals),
  `r^i` times a polynomial (geometric and arithmetico-geometric, symbolic
  `r` guarded by `r == 1 ? ...`), telescoping `g(i) - g(i+k)`, rational
  bodies with integer poles whose residues cancel (partial fractions),
  branches on `i % m` and on `i < c` (the sum is split), linearity;
- products: constants, `r^e(i)`, `exp(e(i))`, telescoping ratios, ratios
  of shifted linear factors;
- recurrences and loops: affine steps with constant coefficients in any
  number of variables (characteristic roots, including repeated and
  complex ones; exact rational arithmetic when the roots are rational;
  symbolic initial values and constants), `x' = a x + p(k)` with a
  polynomial `p` (particular solutions; symbolic `a` for constant `p`),
  accumulators (`s += f(i, other states)` becomes a sum), power maps
  `x' = c x^p`, Mobius maps `x' = (a x + b)/(c x + d)` (2x2 matrix power),
  quadratic maps conjugate to `y' = y^2` (e.g. `x' = 2x(1 - x)`);
- while loops: the trip count of `v += d`, `v *= c`, `v /= c`, `v >>= k`
  against a bound, and `v*v <= n` (isqrt); the other variables then
  follow from the counted loop.  The trip count is checked on its own.

### What is checked, and what is approximate

Every closed form (`closed`, `scheme`) is evaluated against the loop
itself on the function's domain, exhaustively up to 2^16 points and
sampled beyond; integer forms must be equal, real forms equal within
1e-9 relative (the report gives the largest difference).  When the closed
form needs a guard for empty ranges, the unguarded form is tried first.
Points where only one side leaves the 64-bit range are reported apart
(`2^(n+1) - 1` overflows at `n = 62` where the sum still fits).  A loop
needing more than 2^20 iterations at a point is not evaluated there, and
the report counts those points.  Antiderivatives are differentiated back
and compared on a grid; ODE solutions are checked by their residual on a
grid, their initial values, and an independent numeric solution.

Approximate by nature, with measured error: numeric integrals (error
estimate from the Kronrod extension), numeric ODE solutions (the
difference to a 1000x looser solve), PDEs (the difference between two
grids, and against the series/exact solution where one applies),
Euler-Maclaurin and Euler-step ODEs (the difference to the sum or loop
is measured at domain points).  Closed forms of integer sequences with
irrational roots (Fibonacci) are real-valued and rounded; they are exact
only while doubles are (the check says where they stop matching; the
matrix power is exact).

### Limits

- Loop bodies are compiled once per entry and run lane-parallel over the
  chunk of domain points; deep nesting (more than ~50 bound variables) is
  refused.
- Recurrences have at most 8 terms of state; offsets in `x[k+d]` are
  limited to -16..7; initial values must be `x[0]..x[m-1]`.  Evaluating
  `x[n]` also computes `x[n+1]..x[n+m-1]`, so the iteration overflows a
  little before the value it returns would.
- No closed form for harmonic numbers, factorials (both are covered by
  `continuous`), Collatz-like loops, or nonlinear maps other than those
  listed.  Products of a polynomial with integer roots are reported as
  Gamma-like.
- `integrate` knows the rules listed above; it does not do substitution
  beyond linear arguments, rational functions other than `c/u^n`, or
  `exp(x) sin(x)`-type products.  Unknown cases fall back to numbers.
- `dsolve` is symbolic only for linear systems with constant
  coefficients and forcing by polynomials, exponentials and sines; the
  independent variable is the one free name in the equations (`t` if
  none).
- PDEs: one space dimension, linear, constant coefficients; Dirichlet,
  periodic or (for advection) outflow boundaries.

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
- `approx` measures on a sample, not on every float (the C output can
  show a slightly larger maximum); the coefficient search after rounding
  is a local walk, not a lattice-based (fpminimax) search; range
  reductions are offered for builtins only, and the cost model prices
  throughput, not latency.  Piecewise approximations use two pieces.

## Discord bot (optional)

`discord/` holds `ucas-discord`, a bot that runs messages addressed to
it as if they were typed at the `ucas>` prompt and replies with the
output.  A message is addressed to the bot when it mentions it anywhere
(`<@id>`, or `@Name#1234` / `@Name` typed as text) or is a Discord reply
to one of its messages; the text without the mention is the input, after
leading greetings and punctuation (`hey @Wimsy, ...`) are dropped.  It is a separate program with its own BSD Makefile; the `ucas`
binary and `make test` do not depend on it.

```
@Wimsy range isqrt(x: u4)
@Wimsy ```
sq(x: u4) = x*x
table sq max 4
```
@Wimsy reset
hey @Wimsy, what is the integral of x/x from -1 to 1?
```

The rest of the message goes to the engine unchanged (a surrounding
```` ``` ```` fence with an optional language tag, or single backticks,
is removed), so new commands work without changes to the bot.  When the
input has more than one line, each statement is echoed as `> stmt`
before its output, as with `ucas file`.  Output comes back as a reply
in a code block that mentions nobody.  Up to 3 messages are used,
split at line breaks under Discord's 2000-character limit; anything
longer is sent as a preview plus the whole output attached as
`ucas-output.txt`.  Messages from bots (including itself), webhooks and
system users are ignored.

### Setup

1. In the Discord Developer Portal, create an application, add a Bot,
   and copy its token.
2. Under Bot → Privileged Gateway Intents, enable **Message Content
   Intent** (without it the bot sees empty messages in servers).
3. Invite it (OAuth2 → URL Generator, scope `bot`, permissions *View
   Channels*, *Send Messages*, *Send Messages in Threads*, *Read Message
   History*, *Attach Files*).
4. Build and run on the host:

```sh
pkg install curl git gmake    # libcurl; git and gmake build Concord
make discord                  # or: cd discord && make
make discord-test             # no network or token needed
mkdir -p secrets && chmod 700 secrets
$EDITOR secrets/token.txt     # the bot token, one line; chmod 600 it
discord/ucas-discord --check-token   # where it was found; never prints it
discord/ucas-discord
```

English works in Discord too: `@Wimsy what is the range of x squared
for a signed byte x?`.  When the translator needs a decision, it asks in the
channel instead of on a terminal.  Two kinds of question:

- **Choice:** which `log`, what a `byte` is.  Answer `2` or with the
  option's name; no mention is needed for this short answer.  The
  choice is kept for the channel's session.
- **Missing details:** an interval for `approx`.  Answer with a mention
  or a Discord reply, e.g. `@Wimsy 0 to pi/4 degree 5 crit`.

A mention alone takes the default, and `cancel` drops the question.
A reply that looks like a new request instead of an answer (`approx
…`, `what is …`, a definition, a command), or a long reply to a
numbered choice, drops the open question and runs as a new request;
the reply says so.  While a
question is open, the asker's next message to the bot in that channel
(or short reply, for a numbered choice) is the answer; other users are
unaffected.  Questions expire after 15 minutes.
Each answer re-runs the original message with all answers so far.
Translation is deterministic, so the same questions come up in the same
order, and nothing waits inside the sandboxed evaluator meanwhile.
(Engine side: `\x1edefer` / `\x1eanswer` directive lines, see `nlp.c`;
the bot strips `\x1e` from user text.)

`--log PATH` appends a JSON line for every message addressed to the bot
(only those)
(author, channel, text, and whether it was a request, an answer, a
cancel or a reset) and for every reply (the engine's full output,
including the `ucas:` translations and any question asked).  It holds
user messages; `logs/` is in `.gitignore`.  `--one-session` shuts the bot
down once a message and its follow-up questions are done (no requests
or questions left open), for reviewing usage one session at a time.

The bot keeps running until SIGINT/SIGTERM/SIGHUP.  If the gateway
connection ends it reconnects with exponential backoff (2 s doubling to
5 minutes).  It gives up only if Discord rejects the token (4004) or the
evaluator process dies.  If the Message Content intent is not enabled
(close code 4014), it prints how to enable it and continues without it.
It then sees DMs and server messages that mention it (`@bot $1 + 2`).
Concord's log goes to stderr at `info`; `UCAS_DISCORD_LOG=debug|warn|...`
changes the level.  Avoid `trace` outside debugging: it can include
HTTP details.

The token is looked up in this order:

1. `--token-file PATH`;
2. the `DISCORD_TOKEN` environment variable (wiped from the environment
   once read);
3. `secrets/token.txt` next to the executable;
4. `../secrets/token.txt` from the executable.  The in-tree binary is in
   `discord/`, so this finds `secrets/token.txt` at the top of the tree.

The executable's path comes from the kernel (`KERN_PROC_PATHNAME` on
FreeBSD, `/proc/self/exe` elsewhere), not the working directory.
`secrets/` is in `.gitignore`.  The bot warns if the file is readable
by group or others.  The token is read after the evaluation supervisor
has forked, so the sandboxed workers never have it in memory.  It is
never logged: errors name the file, and `--check-token` reports only
its length and whether it has a bot token's three dot-separated parts.

The Gateway and REST client is [Concord](https://github.com/Cogmasters/concord)
v2.4.0 (commit `fb0f834d07d65048a6afde177e318fe9659a9494`).  The first
`make` clones it into `discord/third_party/concord` (ignored by git)
and builds its static library with gmake.  Concord handles identify,
heartbeats, resume/reconnect and the REST rate limits (429
`retry_after`).

### State and limits

State is **per channel** (a thread is its own channel): functions,
bases, models, the active model and the `find`/`more` search belong to
the channel and persist across messages.  `$reset` drops them.

Each channel has a worker process that holds its state.  Every message
is run in a fresh `fork` of that worker; when the evaluation ends
normally the child *becomes* the channel's worker and the old one
exits.  So whatever a message defines persists, with no list of
"defining" commands, and a message that times out, runs out of memory
or crashes leaves the state exactly as it was before it.  Messages in
one channel run in order (up to 4 waiting); channels run in parallel.
At most 64 channels have a worker; beyond that the least recently used
idle one is dropped (its state is lost).  If the bot restarts, all state
is lost.

| variable | default | |
|----------|---------|--|
| `UCAS_DISCORD_CPU` | 10 | CPU seconds per message (`RLIMIT_CPU`) |
| `UCAS_DISCORD_WALL` | 20 | wall-clock seconds per message, then SIGKILL |
| `UCAS_DISCORD_MEM` | 1024 | MiB of address space (`RLIMIT_AS`) per worker |
| `UCAS_DISCORD_CHANNELS` | 64 | channel workers kept |
| `UCAS_DISCORD_QUEUE` | 4 | messages waiting per channel |
| `UCAS_DISCORD_MAXOUT` | 1048576 | output bytes kept per message |

A message over a limit is answered with e.g. `timed out after 20 s;
this channel's state is unchanged` and the output it printed so far.

Sandbox: the language has no command that reads or writes files or
runs programs (only the CLI's `main.c` opens files, and the bot does
not link it).  The worker and its children also run in Capsicum
capability mode (`cap_enter`, checked at start-up: no file system
access, no new sockets, no `exec`), with no core dumps, a zero file-size
limit, stdin on `/dev/null` and no descriptors except their output pipe
and control socket.  The process that talks to Discord never runs
engine code.

### Test mode

`ucas-discord --stdin` reads gateway events, one JSON object per line
(`{"op":0,"t":"MESSAGE_CREATE","d":{...}}` or a bare message object),
passes them through the same handler as the live bot and prints each
REST request it would make.  `ucas-discord --selftest` checks the
prefix and fence handling, message splitting and the JSON codecs.
`make discord-test` runs both against `discord/tests/*.jsonl`.
