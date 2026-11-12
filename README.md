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

License: GPL-3.0-or-later (COPYING).  src/snxglslc/ is ScarletNX's GLSL
compiler, GPL-3.0-or-later as well; it is why ucas is GPLv3.

```sh
make            # FreeBSD make or bmake
make test       # tests/*.ucas must print tests/*.out
                # (tests run with UCAS_SPIRV_CROSS=none; NAME.env replaces
                # that, and a NAME.needs test is skipped without its tool)
./ucas file.ucas    # run a file (each statement is echoed)
./ucas -e 'range isqrt(x: u4)'
./ucas          # interactive
./ucas --json   # JSON Lines for programs and agents (see "JSON frontend")
./ucas --mcp    # an MCP server for AI clients (see "MCP server")
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
| `f16`, `f32`, `f64` (`float`, `double` for the last two) | IEEE binary16/32/64: every result rounds (to nearest, ties to even; subnormals and all), `1/0` is inf, `sqrt(-1)` is NaN |
| `int` | unbounded integers (checked against the int64 range) |
| `uN`, `iN` (1..64; also `uint8_t`...) | wrapping machine integers |
| `bool` | 0 or 1 |
| `complex` (`c128`), `c64` | complex numbers; `c64` rounds both parts of every result to f32.  `z`, `w`, `z1`.., `w1`.. are complex by default unless `x` or `y` is present (see "Complex and hypercomplex numbers") |
| `quat` (`quaternion`), `oct` (`octonion`), `sed` (`sedenion`), `cd(N)` | the Cayley-Dickson algebras of dimension 4, 8, 16 and N = 2..32 |

`real` and `int` are the *ideal* twins of the machine types: `wrap`
compares a function against itself evaluated with every `uN`/`iN` as
`int` and every `f16`/`f32`/`f64` as `real`.

A parameter can carry a range (`x: u8 in 1..12`, or just `x in 0..9`),
and `uniform` (its value is the same across a GPU warp; see `cost`).
Untyped parameters are `real`, or `int` when the body uses integer-only
operators (`& | xor << >> % ~ [] match`), with a note.

Expressions: `+ - * / % ^` (power), `mod`, `& | xor ⊕ ~ << >>`,
comparisons (chains allowed), `&& || ! and or not`, `c ? a : b`,
`x in a..b`, `|x|`, `[t0, t1, ...][i]` (lookup table), `x²`,
functions `sin cos tan exp log ln sqrt isqrt abs floor ceil sign min max
clz ctz popcount sec csc cot atan copysign fma asin acos atan2 sinh cosh
tanh asinh acosh atanh` (also `arcsin arccos arctan arsinh arcosh artanh`;
`45°` is `45*pi/180`), Lambert W (`W(x)`, also `W0`,
`lambertw`, `productlog`: the principal branch; `Wm1(x)`: the -1 branch; a
bare `W` without a call stays a variable), conversions `u8(x) f16(x) f32(x) real(n) int(x)`, bit casts
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
| `range F [nocrit]` | min/max (exhaustive for finite domains, distinct values, smallest type); for one real parameter a grid refined at the critical points; IEEE NaN/inf counts.  A sampled integer range (and a real one in several parameters) also gets a proved bound: where the body does not decrease (increase) in a variable, by the structure of its operations (the derivative's sign by the chain rule), the maximum takes that variable at its largest, which is its range's end or the other side of a where relation (`t <= dd`: t = dd); integer quotients then simplify exactly, floor((k*d + r)/d) = k when 0 <= r < d (r < d proved from d/c, d >> s, d - c, x % d or intervals).  `(t*64 + dd/2)/dd where t <= dd` is proved to stay in 0..64, with the steps; when a proved end meets the sample's extreme, that end is exact.  Integer types must not wrap anywhere on the box, or nothing is claimed |
| `domain F` | the conditions each partial operation imposes, the C assumptions (signed overflow, shift amounts, `__builtin_clz(0)`), and the inputs where the function is defined |
| `wrap F` | per operation: whether LLVM's `nuw`/`nsw` hold (with a witness), C UB, truncating conversions; for floats `nnan`/`ninf` (fast-math) per operation; end to end: where the machine result equals the mathematical one, or the rounding error in ulp |
| `crit F` | real: derivative, roots (bisection), min/max/inflection, kinks; integer: the piecewise-affine segments and local extrema; several reals: the gradient |
| `integrate F [wrt x] [from a to b]` | quotients are cancelled first (x/x, (x^2-1)/(x-1), x^2/x^3), and the answer says where that holds (x != 0); removable points do not affect a definite integral.  A constant integrand names no variable: asked (terminal, bot) or x with a note; `wrt t` picks one.  A unary function's name alone means f(x): `integrate W0` |
| `diff F [x]` | symbolic derivative, or the forward difference for an integer parameter; higher, partial and mixed derivatives, gradient, limits, series: see "Derivatives, limits and series" |
| `equiv F G` | equal on F's domain? (with a counterexample).  G must be defined and equal wherever F is defined (points where F is undefined are outside its domain).  Enumerated when the domain is small enough; otherwise sampled, then, on machine integers, bit-blasted and proved by SAT ("proved by SAT: N AIG nodes, M conflicts"; JSON proof kind `sat`), or a counterexample found by SAT and re-checked by the evaluator; when the solver runs out of its budget (100000 conflicts or 2 s) the answer stays "sampled, not proved" and says why.  See "Proofs on machine integers" |
| `monotonic F [x] [in a..b]` | increasing/decreasing (strictly or not), constant, or the maximal intervals of monotonicity with a witness pair x1 < x2; polynomials proved by the real roots of f', integer types exhaustively (and whether wrapping is what breaks it) |
| `period F [x]`, `periodic F [x]` | the fundamental period: trigonometric subterms with linear arguments (2π/\|k\|, π/\|k\| for tan), `x % m`, `x & m`, combined by their lcm (proved), then the smaller fractions p/k (k <= 64) checked numerically; incommensurate periods (sin x + sin(√2 x)) are almost periodic, not periodic; integer domains searched exhaustively; otherwise a numeric search (not proved) |
| `identify EXPR`, `identify LHS = RHS`, `identify F` | classify and list properties, each tagged proved/exhaustive/numeric: polynomials (degree, factors, roots), rational functions (poles, asymptotes), algebraic/transcendental/piecewise; equations in one variable (solutions), conics by B²-4AC and the 3x3 determinant (circle, ellipse, parabola, hyperbola, degenerate), lines, implicit curves; ODEs (`y'' + y = 0`, or `identify y` after `y' = ...`): order, linear, homogeneous, constant coefficients with the characteristic roots and solution basis, autonomous, separable, exact, Riccati |
| `parity F`, `symmetry F` | even/odd (parity rules proved symbolically, machine integers exhaustively in two's complement), axes x = c and centres of symmetry, f(x, y) = f(y, x) |
| `bounded F`, `zeros F` (`roots`) | bounds by interval arithmetic or polynomial degree, poles and growth at large \|x\|; zeros exactly for polynomials (with multiplicity and complex ones), numerically otherwise |
| `injective F`, `surjective F [onto a..b]`, `bijective F` | with a witness f(x1) = f(x2) or a value never taken; the codomain is the result type for machine integers, the reals otherwise |
| `continuity F`, `discontinuities F` | removable, jump, infinite (pole), essential; ends of the domain |
| `convexity F` (`convex`, `concave`) | intervals of convexity, inflection points, a chord witness |
| `asymptotes F` | vertical, horizontal and oblique; exact for rational functions, from limits otherwise |
| `gates F [basis b] [max k] [as c]` | minimal gate network over a basis, with proof of minimality; `as c` also prints a C function |
| `find F [n] [max cost]`, `more [n]` | other implementations, cheapest first under the active cost model; `more` resumes the search.  First the function's own operations that never bite are dropped (a mask, min, max, clamp or ?: replaced by an operand, a conversion removed), each checked on the whole domain (exhaustively or by SAT) and listed with what it drops, then applied repeatedly to the smallest form.  Identities the model prices lower join them (a signed division of non-negative operands as unsigned, compare and select as min/max, clamps, a division by a variable through f32), each with its derivation and the conditions it holds under (see [Rewrites the prices make worthwhile](#rewrites-the-prices-make-worthwhile)); floats get these too.  With two integer parameters, one of which has at most 8 values (by its range and the where clause), find splits on it and fits `(x*K + C) >> S` exactly per case (the smallest S with one C for every case; S = 0 is `x*K + C`, C = 0 is `(x*K) >> S`); K is offered as a function of the case when one fits (`A / ((1 << c) - 1)`, `A >> c`, `A / c`) and as a select chain, each verified on the whole domain and marked "K depends only on c: compute it once where c is set" with its cost per use once K is known.  BC7's `(i*64 + m/2)/m` with m = (1 << ib) - 1 becomes `(i*K + 4) >> 3` with K = 170, 73, 34 (or 511/m); `x / d` for d in 1..6 becomes a multiply and a shift per d |
| `fit G ~ F [metric maxrel\|maxabs\|rms]` | G's extra parameters are constants: choose them to minimise the error against F |
| `error G ~ F` | max/mean relative error, max absolute error, ulp |
| `cost F [model m]`, `use m`, `models` | price F under a cost model |
| `animate exec of [N-bit\|signed N-bit\|uN\|f32] EXPR [lanes 4\|8\|16\|32] [fps F] [seconds S] [random [seed K]] [MODEL] [to NAME]` | EXPR lowered to its serial form (one instruction per operation, with its class and cost under the model) and run in lockstep on a 2x2 quad of lanes (up to a warp of 32), each lane with its own inputs and registers.  A `?:` whose lanes disagree is a branch: the "then" side runs with the other lanes masked, then the "else" side, then reconvergence, so every lane pays for both (keep-up); a uniform condition runs one side.  `barrier()` stalls every lane; an atomic serializes the lanes.  The type phrase applies to every free variable (`32-bit` is u32 unless `signed`); without one it asks, defaulting to f32 when EXPR is real (sin, decimals, pi) and u32 otherwise.  A ranged real input (an approximation's domain) is spread over its range across the lanes |
| `simplify` + a ```` ```spv ```` block or an attached shader (also `SAT simplify`, `optimize this`, `peephole with smt` ...) [`budget S`] | cheaper forms of the module's integer instructions, each proved by SAT: one of its operands (a clamp, min or mask that never bites), a constant (a compare that always holds), or one cheaper instruction (a shift for a multiply or divide by a power of two, a mask for a remainder or an extract at bit 0, a min or max for a clamp with one side never reached, a bitfield extract for a shift and a mask).  A candidate is run first on random inputs with the interpreter's arithmetic, then both sides are bit-blasted (src/sat.c: an and-inverter graph and a CDCL solver) and no input may make them differ.  The operands' definitions are expanded up to 16 levels, phis at selection merges as "one of the incoming values"; loads, loop phis and specialization constants are free inputs, so a proof holds for every value they could take.  Proofs are refinements: where the original is defined, the replacement must be defined and equal, so nothing leans on what SPIR-V leaves undefined (a shift by the width or more, a bitfield past it, a division by zero).  Assumptions in the request narrow the inputs: `n in [0, 32)`, `n in (a, b]`, `n in 0..31`, `1 <= n < 32`, `n < 32` (0 <= n) apply to the value named n or every load of it.  Two-instruction rewrites merge shift chains ((y << 1) << (31 - n) is y << (32 - n) when n >= 1).  A rewrite that fails is shown with the solver's counterexample when it is a near miss ("fails at %n = 0").  GLSL works too (a ```` ```glsl ```` block or a .comp/.frag/.glsl file): a snippet is completed (a #version, undeclared globals declared from their use as buffer members so the optimizer keeps their writes, a compute main() calling each function with its arguments from a buffer), compiled by glslangValidator with debug lines and optimized with `spirv-opt -O` as a real pipeline would (so what simplify finds is beyond that), and each rewrite names the GLSL line it came from; an assumption on a parameter follows it through inlining.  Chains of shifts by constants, adds, subtracts and constant multiplies fold into one multiply (an integer hash's `p -= p << 17; p += p << 7; p += p << 4` is `p *= 4007528593`); those are proved by algebra mod 2^w (a multiplier against a shift-and-add network is slow for SAT) and checked on 1024 random inputs, and the report says which proof each rewrite has.  With an assumption that bounds the only input of a stored or returned value to at most 2^20 values, a range search checks every input: what the value is (floor, round or ceil of a ratio: `(c * 255 + 32895) >> 16` for c in [0, 65535] is round(c * 255 / 65535)) and forms from the families (x + K) >> S, (x +- (x >> S1) + K) >> S (K solved exactly) and t = x + K; (t +- (t >> S1)) >> S that are cheaper, or as cheap without a multiply (`t = c + 128; (t - (t >> 8)) >> 8`); when none is, it says how many forms it tried.  Unoptimized input gets store-to-load forwarding and dead-store removal in each block for per-invocation variables first.  Proved rewrites apply in order, dead instructions go, and the module is written back (`simplified.spvasm`, attached in Discord), with the cost under the model before and after.  Rewrites any driver's back end makes on its own (a multiply, unsigned divide or remainder by a power of two, an extract at bit 0, a shift then a mask, two constant shifts, an identity operand such as x + 0 or x & ~0, constant folding, x - x) are kept in the module but listed apart in one line with their count; the headline cost is what is left beyond spirv-opt -O and those, with the total shown next to it (JSON: `cost_driver_baseline`, `driver_trivial_count`, `driver_trivial_gain`, and `driver_trivial` on each rewrite).  The time budget defaults to 6 s.  Float instructions follow (see [Float simplify](#float-simplify)): each float rewrite comes with a derivation, not a sample; `float`, `exact` (`bit-exact only`), `ulp N` (`within 2 ulp`), `approx` (`allow approximations`), `per component`.  GLSL with functions that call each other also gets a function-level section (see [Function level](#function-level)): `functions` (`function level`, `per function`, `functions only`) runs it alone, `nofunctions` (`without the function level`) skips it.  With spirv-cross installed, the rewritten module and the one spirv-opt -O made are cross-compiled back to GLSL and diffed (see [GLSL view](#glsl-view)).  C goes to the C frontend instead (see [C frontend](#c-frontend)) |
| `animate exec of best found cos [-pi/2, pi/2] with crit + cheby poly` | an approximation request (best, found, approx, cheby, minimax, taylor ...): it is read as `approximate ...` in English, `approx` runs (asking precision and endpoints as usual), and the best candidate `ap1` is animated.  Options can be joined with `+`, `&`, `and` |
| `animate exec` + a ```` ```spv ```` block | runs SPIR-V assembly (spirv-dis text) on the lanes in SIMT lockstep and animates it like the expression form, the serial form being the executed SPIR-V lines with each lane's values.  Structured control flow with masks: a selection runs both sides when the lanes disagree, then reconverges; a loop iterates while any lane runs; switch, calls, return/kill; OpPhi per lane.  Uniform/StorageBuffer/Workgroup memory is shared (random contents unless `seed`), Function/Private/Input/Output per lane; FragCoord is the quad's pixel centres, invocation ids the lane.  Barriers stall, atomics serialize.  Bitfield extract/insert/reverse, find LSB/MSB, OpSpecConstantOp (folded with the spec constants' defaults) and OpUndef (zero) are handled.  An access outside a variable reads 0 and drops the write, as with robustBufferAccess, and is counted in a note (random inputs are rarely valid data for a decoder).  Long runs (up to 100000 executed instructions) print the first 160 and last 20 lines plus the time by cost class; when the workgroup is wider than the lanes run, a note says so.  In the chat a GIF over 6 MiB is re-rendered at half the frame rate until it fits.  A snippet is completed: types and constants from glslang's names (`%int`, `%v4float`, `%uint_1`, `%float_0_5`, `%int_n1`), the GLSL.std.450 import, an entry point, the function and its return; any other undefined operand is an input typed from its use (its value asked for: random per lane, the lane index, 0, 1; or given as `%19 = 2.5` / `%19 = lane`), or an output when it is only stored to.  A shader with images (textures, samplers, storage images) also runs over a whole image, with image inputs you choose: see [Shaders on images](#shaders-on-images) |
| `animate [hillis\|blelloch\|sequential] scan [n N] [fps F] [seconds S] [random [seed K] \| values ...] [f32] [MODEL] [to NAME]` | an animated GIF of a prefix-sum scan (n capped at 100): one row per level, values updating, the critical-path time and the work under the cost model advancing, with barriers and divergence on a timeline; also a table of the three algorithms' costs.  Filler words are skipped, so `animate random sample prefix sum scan n=8 at 30 fps using gt1030` works as written.  GIF delays are whole hundredths, so 30 fps alternates 3/3/4 cs.  On the command line it writes NAME.gif (default scan.gif); in Discord it is attached |
| `plot F, ...`, `plot3d F`, `neighbourhood F [at a]` | GIF plots, bounds chosen from the function when not given (see Plots) |
| `area F [and G]`, `revolve F [and G]`, `volume`, `surfacearea`, `arclength`, `centroid` | integrals shaded (Riemann sums, the accumulation function) and solids of revolution, measured and drawn (see "Integrals and solids of revolution") |
| `approx F in a..b [options]`, `approx more [n]` | polynomial, rational and range-reduced approximations, measured as they run (see below) |
| `complex F` (`rect F`), `re F`, `im F`, `conj F`, `modulus F`, `arg F` | F with `i` as the imaginary unit, as a + bi (exact where the constants are recognised), each part checked (see "Complex numbers and trig forms") |
| `polar F` | r e^(i theta) and r (cos theta + i sin theta), theta the principal argument in (-pi, pi], with the conditions it holds under |
| `nthroots N F`, `unity N` | the N complex N-th roots (of 1), each raised back to the N-th power |
| `holomorphic F`, `residue F at a`, `poles F`, `integrate F around \|z\| = r`, `table quaternion`, `algebra octonion`, `rotate V by Q`, `slerp` ... | complex functions and the quaternions, octonions, sedenions (see "Complex and hypercomplex numbers") |
| `expand F`, `factor F`, `collect F [in x]`, `simplify F`, `together F`, `cancel F`, `apart F [in x]` | polynomials and rational functions over Q (see "Symbolic algebra"); a cancelled factor is stated as a condition (`holds where x - 1 != 0`) |
| `gcd P, Q`, `lcm P, Q`, `resultant P, Q [in x]`, `discriminant P [in x]`, `quo P, Q [in x]`, `rem P, Q [in x]`, `divide P by Q [in x]` | polynomial gcd (multivariate), resultants and discriminants with symbolic coefficients, division with remainder |
| `solve L = R [, L2 = R2 ...] [for x [, y]]` | polynomial equations (exact roots: rational, surds, cubics in trigonometric or hyperbolic form, quartics that split over Q(sqrt d); otherwise numeric with a note), rational equations (poles excluded), `sin u = c` and other invertible functions with the general solution in k, one square root (squared, extraneous roots named), linear systems (unique, a family with free unknowns, or none), two polynomial equations by the resultant; every solution substituted back |
| `substitute x = a [, y = b] in F` (`subst`) | F at those values, simplified |
| `laplace`, `ilaplace`, `ztrans`, `iztrans`, `zsolve`, `tf`, `fourier`, `ifourier`, `dft`, `dtft`, `genfunc`, `egf`, `sum(n, 1, inf, ...)`, `prod(...)`, `converge`, `radius`, `powerseries`, `asymptotic` | transforms, series and sums (see "Transforms, series and sums") |
| `trig F [simplify\|sincos\|exp\|reduce\|product\|sum\|expand\|harmonic\|half\|weierstrass\|inverse\|double\|all]` | equivalent trig, hyperbolic and inverse-trig forms, each with the identity used, its conditions and a numeric check; the simplest is named |
| `ring R [EXPR]`, `inverse A mod N`, `crt ...`, `factor P over GF(q)`, `magic D u32`, `gf2matrix EXPR` ... | rings and finite fields: Z/n, GF(p^n), polynomials and matrices over them (see "Rings and finite fields") |

Finite domains of up to 2^22 points are enumerated; larger ones are
sampled (65536 points, edge values first), and the output says so.
The where clause narrows an integer parameter first when a conjunct
compares it with an expression whose interval is known: in
`f(v: u32, n: u32 in 5..8) = ... where v < 1 << n`, v runs over 0..255
(1024 points, enumerated).  `wrap` and `range` add interval bounds when
sampling, which can prove that an operation never wraps.

### Proofs on machine integers

When a domain is too big to enumerate, `equiv` (and `find`, for its
candidates) bit-blasts both expressions (src/nodesat.c) into the
and-inverter graph of src/sat.c and asks the CDCL solver for an input
within the ranges and the where clause where F is defined and G is
undefined or different.  The circuit follows the evaluator exactly:
`uN`/`iN` wrap; a shift by a negative amount or by the width or more,
a division or remainder by zero, a table index out of range and `int`
overflow are undefined; `INT_MIN / -1` in `iN` wraps; `?:`, `&&` and
`||` look only at the taken side; loops with constant bounds are
unrolled (up to 256 iterations).  Not bit-blasted: floats and reals,
`int` multiplication, `isqrt`, while loops; those stay sampled.
Multipliers are hard for SAT (x*y against y*x gives up), so the budget
is 100000 conflicts or 2 s of CPU.

## Symbolic algebra

```
> factor x^6 - 1
  = (x - 1)*(x + 1)*(x^2 + x + 1)*(x^2 - x + 1)
  check: exact (both sides have the same normal form)
> simplify (x^2-1)/(x-1)
  = x + 1
  holds where x - 1 != 0 (cancelled there)
> integrate 1/(x^3+1)
  = log(abs(x + 1))/3 - log(x^2 - x + 1)/6 + atan((2x - 1)/sqrt(3))/sqrt(3) + C
> integrate 1/(x^4+1)
  = sqrt(2)*log(x^2 + sqrt(2)*x + 1)/8 - sqrt(2)*log(x^2 - sqrt(2)*x + 1)/8
    + sqrt(2)*atan(sqrt(2)*x - 1)/4 + sqrt(2)*atan(sqrt(2)*x + 1)/4 + C
> solve sin x = 1/2
  x = pi/6 + 2pi*k   (k any integer)
  x = 5*pi/6 + 2pi*k   (k any integer)
  check: each solution substituted (10 values, k = -2..2): residual 9.4e-16
> solve sqrt(x+1) = x - 1
  isolating sqrt(x + 1) and squaring: x^2 - 3x = 0
  extraneous (from squaring, the original equation fails): x = 0
  x = 3
```

The core (`src/poly.c`) is exact: rationals with 64-bit numerators and
denominators and 128-bit intermediates (a result that does not fit
stops the computation and says so), sparse multivariate polynomials over
Q in *atoms* (variables, and opaque subterms such as `sin(x)` or
`sqrt(x + 1)`), their gcd (primitive pseudo-remainder sequences with
recursive contents), exact division, resultants (fraction-free
elimination of the Sylvester matrix) and discriminants, factorisation
(content and monomial factors, square-free decomposition, rational
roots, cyclotomic factors, Kronecker's method for the other factors of
small degree; in two variables homogeneous polynomials through x/y;
otherwise several variables get their contents and square-free parts,
with a note), and rational functions kept as p/q with the gcd
cancelled.  An expression is taken apart over sums, products, integer
powers and quotients (`x^(3/2)` is `x*sqrt(x)`); a constant that is
exactly a fraction as a double (0.25, 1/3) is that fraction, and `pi`,
`e` are atoms.  The product of every denominator met is kept, so a
factor that cancels is reported as a condition.  Printing stays exact:
`x^3/3`, `1/(2(x + 1))`, `sqrt(3)` (an unevaluated surd), `x^(1/3)`;
coefficients beyond 2^53 cannot be represented in an expression and are
printed rounded, with a note.

`simplify` simplifies the arguments of functions (`sqrt(u^2) = |u|`,
`sqrt` of any perfect square, `exp(log u) = u` for `u > 0`, `log(exp u)
= u`, `exp(u)^k = exp(k u)`, products of exponentials joined), then
takes the shortest of the expanded and factored normal forms, also with
`sin^2 = 1 - cos^2` and `cosh^2 = 1 + sinh^2` applied when both functions
of the same argument occur.  `log(a b)` is not split (that holds only
for a, b > 0).  Trigonometric rewriting is `trig`'s (forms.c): in
English, "simplify" or "expand" of a trig expression goes there.

Checks: the normal forms of input and result are compared exactly when
both convert (`check: exact`), otherwise at 64 sample points; gcd and lcm
by exact division (and coprime cofactors); resultants and discriminants
numerically through the roots; every solution of `solve` by
substitution (for k = -2..2 in a general solution).

### Integration

`integrate` tries, in order: rational functions (exact, below), the
older rules (polynomials, linear arguments, by parts with exp/sin/cos,
atan forms), and then the methods of `src/rint.c`, each result
differentiated back and compared before it is used:

- **rational functions**, completely: the polynomial part, Hermite
  reduction for repeated factors (the rational part), then partial
  fractions over the irreducible factors of the square-free rest:
  `c log|x - r|`, `c log|f|` when the numerator is `c f'`, log and atan
  for quadratics (a log of a quotient with surds when the roots are real
  but irrational), quartics that split into quadratics over Q(sqrt d)
  (`x^4 + 1`), and otherwise numeric roots (residues paired into log and
  atan terms, with a note that the coefficients are decimals);
- linearity; `sinh`, `cosh`, `tanh`;
- `R(e^(a x))` through `u = e^(g x)`;
- products of sines and cosines (multiple angles rewritten first: `sin x
  cos 2x`), times a polynomial: product to sum, then by parts;
- rational functions of `sin u`, `cos u` (`tan`, `sec`, ... rewritten):
  even powers by power reduction, else Bioche's rules (`t = cos u`, `sin
  u`, `tan u`), else the Weierstrass substitution `t = tan(u/2)` (with a
  note: it holds between the poles of the tangent);
- `R(x, sqrt(a x + b))` by `t = sqrt(a x + b)`; `R(x, sqrt(quadratic))`:
  a rational part plus `P(x)/sqrt(q)` and terms `c/((x - a)^k sqrt(q))`;
  the first is `Q(x) sqrt(q) + c * integral of dx/sqrt(q)` (Ostrogradsky:
  a linear system for Q), the base integral `asin`, `asinh` or `log|2a x
  + b + 2 sqrt(a) sqrt(q)|`, the others through `t = 1/(x - a)` (hence
  `sign(x - a)`);
- by parts: `log`, `atan`, `asin`, `acos`, `asinh`, `acosh`, `atanh`
  (and their powers) times an integrable factor; `e^(a x) sin|cos(b x)`
  by the closed form of parts twice;
- substitution `u = g(x)` for every subterm g whose derivative divides
  the integrand (up to a constant; `x^k` in the rest is rewritten through
  `u = a x^k + b`).

A definite integral uses F(b) - F(a) only after F is scanned on the
interval: where F jumps but the integrand is continuous (the Weierstrass
form at x = pi), the jump is located by bisection, measured by one-sided
values extrapolated to the point, printed and taken out.

Not done: integrands with symbolic parameters in a denominator
(`1/(x^2 + a^2)`), algebraic functions beyond one square root of a
polynomial of degree 1 or 2, the Risch algorithm (`exp(x^2)` and
`sin(x)/x` are reported as not found; a definite integral is still
computed numerically), denesting radicals beyond Q(sqrt d).

### Solving

`solve` takes one or more equations (`F` alone means `F = 0`) and the
unknowns after `for` (default: all variables; for one equation in
several variables, x, or the question is asked on a terminal).  One
equation in one unknown: the numerator of the rational normal form is
factored over Q and each factor solved exactly when it can be (linear
and quadratic factors, `x^n = c`, cubics with three real roots as `2
sqrt(..) cos(acos(..)/3 - 2k pi/3)` and with one as a `cosh`/`sinh` form,
quartics that split over Q(sqrt d)); higher degrees are numeric
(Durand-Kerner), marked `(numeric)`.  Roots of a denominator are
excluded and named.  Complex roots are counted, not listed (`nthroots`
lists them).  A polynomial in one function of x, `f(u) = c` with f in
sin cos tan exp log sqrt abs sinh cosh tanh atan asin acos and `b^u`, is
inverted: periodic functions give `x = x0 + p*k (k any integer)` with
exact angles where they exist (pi/6).  One square root: isolated and
squared, and the roots that squaring added are named as extraneous.
Symbolic coefficients: degree 1 and 2 by formula, with `a != 0` and the
discriminant's sign stated.  Systems: linear ones by exact Gauss-Jordan
elimination (a unique solution, a family in the free unknowns, or "no
solution" with the ranks); two polynomial equations in two unknowns by
eliminating the second unknown with the resultant.  Anything else: sign
changes on [-100, 100], refined by bisection, with a note that
solutions may be missed.

## Complex numbers and trig forms

```
> complex (1+2i)/(3-i)
(1+2i)/(3-i) = 1/10 + 7i/10
> polar x + i y
  note: i is the imaginary unit; x, y are taken as real
x + i y = sqrt(x^2 + y^2)*e^(i*atan2(y, x))
  theta is the principal argument, -pi < theta <= pi; undefined where x = y = 0
  check: equal at 64 of 64 sample points (max relative difference 4e-16)
> trig sin(x) cos(x)
  product to sum: sin a cos b = (sin(a + b) + sin(a - b))/2, ...
    = sin(2x)/2
    check: equal at 64 of 64 sample points (max relative difference 6e-17)
  simplest: sin(2x)/2  (product to sum)
```

The DAG is real-valued; these commands (`src/forms.c`) map an
expression to a pair of real nodes (Re, Im).  `i` is the imaginary unit,
also `j` (with a note), `I` and `im`; `cis(t)` is `e^(i t)` and `r∠t` is
`r e^(i t)`.  A function's parameter named `i` is asked about (default:
the unit, with a note).  Every other variable is taken as real, and the
output says so.  Products, quotients (through the conjugate), integer
powers, `exp`, `sin cos tan sec csc cot sinh cosh tanh` of complex
arguments are exact; `sqrt`, `log`, non-integer powers and the inverse
functions take the principal branch (`log z = log|z| + i arg z`, `z^w =
e^(w log z)`, `asin z = -i log(iz + sqrt(1 - z^2))` ...), and a real
argument on a branch cut (`asin(2)`) says which value it takes.  Each
part is normalised: sums of monomials over atoms, `sin^2 + cos^2 = 1`
and `cosh^2 - sinh^2 = 1` folded, `sqrt(x^2) = |x|`, `exp(a + b) = exp(a)
exp(b)`.  Constants are printed exactly when they are rationals,
rational multiples of pi, square roots, `(p + k sqrt(m))/q`, their square
roots (`sqrt((5 + sqrt(5))/8)`), logarithms of small integers and surds,
or `exp` of those; otherwise as decimals, and an angle that is not nice
is kept as `atan(4/3)` with its value.  A cancelled factor (x/x) is
reported as a condition.

`trig F` lists the forms that apply, each with the identity used:
simplification (also `sin(asin u) = u`, `asin(sin u) = u` for |u| <=
pi/2 ...), tan/sec/csc/cot in sin and cos, power reduction and product to
sum (every sin/cos/sinh/cosh as exponentials, multiplied out and paired
back, so `sin^3 x = (3 sin x - sin 3x)/4`), sum to product, compound and
multiple angles (`sin 3x = 3 sin x - 4 sin^3 x`, also only in sin or only
in cos, `tan(a + b)`), `a sin u + b cos u = R sin(u + phi) = R cos(u -
psi)`, half angles (`sin(u/2) = sqrt((1 - cos u)/2)` where `sin(u/2) >=
0`), the tan(u/2) substitution (`weierstrass` or `half`), inverse
functions in terms of each other (`atan u = asin(u/sqrt(1 + u^2))`,
`acos u = pi/2 - asin u`, `atan2` via `atan` by quadrant, `atan a + atan
b` for `a b < 1`, `asinh u = log(u + sqrt(u^2 + 1))` ...), and the
exponential/logarithmic (Euler) form.  A mode word picks one kind.  With
no mode, a simplification that removes a composition is applied first.
The simplest form is the shortest, counting each function call as 6
characters.

What is checked: every printed result is parsed back as ucas text and
evaluated at sample points (64, seeded, inside the parameters' ranges;
one point for constants) against the original, which is evaluated by C's
complex functions (`cexp`, `clog`, `csqrt`, `csin` ...; real semantics
for `trig`, where a point the original is undefined at is skipped).  A
form that holds only under a condition (`atan u = pi/2 - atan(1/u)` for
`u > 0`) is checked where the condition holds, and the points outside
where it differs are counted, which shows that the condition is needed.
Limits: variables are real here, except that a complex variable (`z`,
see the next section) is written `x + iy` first; `polar` of a symbolic
z uses `atan2`, recognising `R cos u + i R sin u` with R > 0 (then theta
= u for -pi < u <= pi).

## Complex and hypercomplex numbers

```
> (1+2i)*(3-i)
5 + 5i
> i^i
exp(-pi/2)
> sqrt(-4)
  note: sqrt(-4) is undefined in the reals; its complex principal value follows
  note: on the branch cut of sqrt: the principal value, sqrt z = e^(log(z)/2): ...
2i
> j*i
  note: i, j are quaternion units: ij = k = -ji
-k
> (e1*e2)*e4
  note: octonion products are not associative: each product is taken as grouped, ...
e7
> f(z) = z^2 + 1
  note: z is taken as complex (no x or y here; say z: real for a real variable)
> zeros f
  2 roots (a polynomial of degree 2):
    z = -i
    z = i
> residue 1/(z^2+1) at i
  = -i/2
  a pole of order 1 (|f| grows like |z - a|^-1)
> integrate 1/z around |z| = 1
  = 2pi*i
  residue theorem: 2 pi i times the sum of the residues at the 1 pole inside = 2pi*i: agrees
> rotate (1, 0, 0) by 90 degrees about the z axis
  q = sqrt(2)/2 + k*sqrt(2)/2: a rotation by pi/2 about (0, 0, 1)
  q v q^-1 = j
  rotated: (0, 1, 0)
```

`src/hyper.c`.  **Design.**  The DAG stays real-valued.  A
hypercomplex expression is an ordinary DAG whose units (`i`; `i j k`;
`e1`..`e31`) are variables in reserved slots and whose complex variables
are ordinary parameters; a function records the algebra of its value
and of each parameter (`struct func`: `halg`, `hk[]`).  Nothing in the
constructors reorders a product of non-constants (only real constants
move, and real scalars commute with everything), so the DAG is faithful
for non-commutative and non-associative algebras: `i*j` and `j*i` are
different nodes and a product is evaluated as it is grouped.  A small
plan (the reachable nodes in id order, one value of dimension 2^k per
node) evaluates it.  This keeps every symbolic tool working unchanged:
printing, substitution, inlining of calls, `diff_node` (the complex
derivative of a holomorphic expression is the formal one), and forms.c's
split into real and imaginary parts once `z = x + iy`.  A separate
complex node kind would have had to be taught to every evaluator,
simplifier and printer; the price of this choice is that the real
evaluators must not see these functions, so the real-only commands
(`range`, `crit`, ...) refuse a complex function and say how to get a
real one.

Multiplication is the Cayley-Dickson doubling `(p, q)(s, t) = (ps -
conj(t) q, tp + q conj(s))`, with `e_{m+d/2} = (0, e_m)`: `ij = k`,
`e1 e2 = e3`, `e1 e4 = e5` (`table octonion` prints the table and the
Fano plane).  Every element of a Cayley-Dickson algebra lies in the
complex plane spanned by 1 and its vector part, so an analytic function
with real coefficients is evaluated by C's complex function at `a + |v|i`
and mapped back: `f(a + v) = Re f(a + |v|i) + (v/|v|) Im f(a + |v|i)`.
That is exact for quaternions and octonions (one element generates an
associative subalgebra) and gives `exp`, `log`, `sqrt`, powers and the
trig and hyperbolic functions and their inverses everywhere.

**Syntax.**

| form | meaning |
|------|---------|
| `3 + 4i`, `i`, `I`, `j` (alone, with a note), `cis(t)` | complex; `j` with `i` or `k` is a quaternion unit |
| `1 + 2i + 3j + 4k` | a quaternion (`k` is a unit only next to `i` or `j`, or in a list of quaternions) |
| `e1`..`e7`, `e8`..`e15`, up to `e31`; `e0` is 1 | octonion, sedenion, cd(32) units; `i j k` are then `e1 e2 e3`; `2e1` is read as `2*e1` there (with a note), never elsewhere |
| `z`, `w`, `z1`, `w_2`, `zeta` | complex variables unless `x` or `y` appears in the statement (a point or a surface z = f(x, y), implicit equations in x, y, z): then real, as before.  ODEs (`z'' = ...`), sums, code blocks and `where` clauses are never read this way |
| `z: complex`, `z: c64`, `q: quat`, `o: oct`, `s: sed`, `s: cd(16)`, `z: real` | annotations win over every default |
| `conj(z)`, `re(z)`, `im(z)`, `arg(z)`, `abs(z)`, `|z|`, `norm(z)` | conjugate, parts (for a quaternion `im` is the vector part), the principal argument in (-pi, pi] (for a quaternion the angle in [0, pi]), the Euclidean norm; `conj re im arg` are functions only when called, so a variable named `re` still works |
| `a/b` | `a b^-1` (right division); a note says when `b^-1 a` differs |
| `a^n` | repeated multiplication (power-associative, so well defined); a real exponent in the plane of `a`; a non-real one is `e^(log(a) b)`, with a note where `e^(b log a)` differs |

A defined function named `i` (`i = 3`) is used as that, with a note.
In a definition the translator makes (`it(i) = i^i`), an untyped `i`
is the unit.  An explicit parameter `f(i) = ...` stays a real parameter.

**Principal values and branch cuts.**  `log z = log|z| + i arg z` with
`-pi < arg z <= pi` (the cut (-inf, 0] takes the value from above);
`sqrt z = e^(log(z)/2)` (i sqrt|z| on the negative reals); `z^w = e^(w log
z)`; `asin z = -i log(iz + sqrt(1 - z^2))`, `acos z = pi/2 - asin z`
(cuts (-inf, -1), (1, inf)); `acosh z = log(z + sqrt(z + 1) sqrt(z - 1))`
(cut (-inf, 1)); `atanh z = (log(1 + z) - log(1 - z))/2`; `atan` and
`asinh` are cut on the imaginary axis beyond +-i.  These match forms.c.
A real argument on a cut is noted.  In the quaternions and beyond a
function of a negative real has a whole sphere of values (every unit u
with u^2 = -1); `i` is taken, with a note.  A constant expression that
is undefined in the reals (`sqrt(-4)`, `log(-1)`, `asin(2)`,
`(-8)^(1/3)`) prints its complex principal value with a note; with a
variable the real function stays as it was.

**Commands** on a complex function of one complex variable (a name, or
an expression):

| command | |
|---------|--|
| `show F`, `table F [max n]` | the function with its kinds; values at 0, +-1, +-i, +-1 +- i, 2, 2i (quaternions: 0, 1, the units, 1 + a unit, their mean) |
| `zeros F` (`roots`, `solve`, `L = R`) | a polynomial (its coefficients from 64 samples on a circle by the DFT, checked at 8 other points) gets every complex root by Durand-Kerner, multiple roots grouped and polished on the derivative, exact where recognised (`-1/2 + i*sqrt(3)/2`); otherwise Newton from a 21x21 grid in \|Re\|, \|Im\| <= 7, each zero checked |
| `diff F`, `holomorphic F` (`analytic`) | the complex derivative (the formal derivative, checked against difference quotients), when the Cauchy-Riemann equations hold at 64 sample points; where it can fail (branch cuts of log, sqrt, non-integer powers; poles); otherwise not holomorphic, with the point where df/dx != df/(i dy), the non-holomorphic parts used (conj, re, im, \|.\|, arg) and the Wirtinger derivatives there |
| `limit F at a`, `limit F as z -> a`, `lim z -> a F` | along 8 directions, z -> a by factors of 4 (Richardson-extrapolated); equal: the limit (and whether F is continuous or a removable singularity there); else the values per direction; growing from every side: infinity; `a = inf` |
| `residue F at a`, `poles F` | the residue as (1/2 pi i) times circle integrals of radius 0.01 and 0.005, which must agree; the order of a pole from the growth of \|F\| on circles of radius 1e-2..1e-4 (or: regular, essential); poles are the zeros of 1/F by Newton |
| `integrate F around \|z - c\| = r [clockwise]`, `integrate F from a to b` | the trapezoid rule on the circle (1024 and 2048 points), cross-checked by the residue theorem with the poles found inside, or Cauchy's theorem when there are none; the straight segment by Simpson's rule |
| `complex F`, `re F`, `im F`, `conj F`, `modulus F`, `arg F`, `polar F` | forms.c with each complex variable written `x + iy` (w = u + iv, z1 = x1 + i y1) and a note: `complex z^2` is `x^2 - y^2 + 2i*x*y` |
| `plot F [radius R] [nofile] [to NAME]` | domain colouring, a GIF: hue = arg F (24 hues), brightness steps each time \|F\| doubles, axes dotted, undefined points black; it lists the zeros (all hues meet) and poles it finds |
| `plot \|F\|`, `plot re F`, `plot im F` | a surface over (x, y) by plot.c |

The algebras:

| command | |
|---------|--|
| `table complex\|quaternion\|octonion\|sedenion\|cd N` | the multiplication table (row times column); for octonions the 7 oriented lines of the Fano plane |
| `algebra NAME [property]` | commutative, associative, alternative, flexible/power-associative, normed (composition), division algebra, inverses, conjugation: each with a counterexample from the units or a check on 200 random elements, and the theorems that bound them (Frobenius, Artin, Hurwitz) |
| `zerodivisors [NAME]` | a pair (e_a +- e_b)(e_c +- e_d) = 0: from the sedenions on |
| `associator a, b, c`, `commutator a, b` | (ab)c - a(bc) and ab - ba |
| `rotate V by Q`, `rotate V by ANGLE about AXIS` | q v q^-1 with v the pure quaternion of V (q normalised, with a note); the angle in radians, or degrees (`90 degrees`, `90°`, or a whole number above 2pi, with a note); the axis a vector or `x`, `y`, `z` (`the z axis`).  Checked against the rotation matrix, which is printed |
| `axisangle (x, y, z), ANGLE`, `axisangle Q` | the unit quaternion cos(theta/2) + sin(theta/2) u, and back |
| `rotmatrix Q`, `frommatrix m11, ..., m33` | to and from the 3x3 rotation matrix (Shepperd's method; the matrix must be orthogonal with det 1; q and -q are the same rotation) |
| `slerp Q1, Q2, T` | spherical linear interpolation on the shorter arc, checked against q1 (q1^-1 q2)^t |

In English (`src/nlp_hyper.inc`): `multiply (1+2i) by (3-i)`, `divide 1
by i`, `what is (1+2i)*(3-i)?`, `what is e to the i pi`, `the square root
of -4`, `roots of z^3 = 1`, `is z^2 holomorphic`, `is conj(z) analytic?`,
`where is F holomorphic`, `the complex derivative of F`, `the residue of
1/(z^2+1) at i`, `the poles of F`, `the contour integral of F around the
unit circle`, `integrate F around the circle of radius 2 about i`,
`rotate (1, 0, 0) by 90 degrees about the z axis using quaternions`, `the
rotation matrix of q`, `the quaternion for 90 degrees about the z axis`,
`slerp between 1 and k at 1/2`, `quaternion product of i and j`, `is
octonion multiplication associative`, `are the quaternions commutative?`,
`is the sedenion algebra a division algebra`, `the octonion
multiplication table`, `the Fano plane`, `zero divisors of the
sedenions`, `the associator of e1, e2 and e4`, `the commutator of i and
j`, `domain colouring of (z^2-1)/(z^2+1)`, and the calc2 phrases (`the
limit of sin(z)/z as z approaches 0`) with complex variables.  A line
that is a formula with units (`i^i`, `i*j`) is not English.

**Checked, and approximate.**  Every derivative is compared with
difference quotients; residues and contour integrals are computed twice
and cross-checked (residue theorem); zeros are evaluated back; rotations
are checked against the matrix; algebra properties are counterexamples
(exact, from the units) or checks on random elements (numeric).  Zeros
of non-polynomials, poles, limits and orders are numeric and say so.
Not done: residues at infinity, Laurent series, branch-cut-aware
contours (an integrand undefined on the contour is refused), several
complex variables in the analysis commands, quaternion calculus
(`diff`, `zeros` work on complex functions only), exact symbolic
quaternion simplification (values are numeric with recognised
constants).

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
  `pi/2 - atan(1/|x|)` with `copysign`; cosh as `(e + 1/e)/2` with `e =
  exp(|x|)`, sinh and tanh as an odd polynomial below a cut and `(e -
  1/e)/2`, `1 - 2/(exp(2|x|) + 1)` above it.  asin acos atan2 asinh
  acosh atanh (and sinh cosh tanh) are fitted like any builtin: the long
  double reference, Taylor jets (so `cheb econ` and Padé) and the
  symmetry check know them.

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
  integrals drawn and solids of revolution (`area under x^2 from 0 to
  4`, `slice of integral ...`, `revolve x^2 about the y axis`, `the volume
  of revolution of ...`; see "Integrals and solids of revolution"),
  derivatives, limits, series (`the second derivative of`, `d/dx`,
  `partial derivative of x^2 y wrt y`, `∂f/∂x`, `fx`, `the gradient of
  ... at (1, 2)`, `hessian`, `laplacian`, `jacobian`, `directional
  derivative of ... along (3, 4)`, `the wronskian of sin x, cos x`, `are
  sin x and cos x linearly independent`, `the limit of sin x / x as x
  approaches 0`, `lim x->0+ ...`, `what does 1/x tend to as x goes to
  infinity`, `the taylor series of ... about 1 to order 3`, `dy/dx for
  x^2 + y^2 = 1`, `the primitive of ...`; see "Derivatives, limits and
  series"),
  closed (`the closed form of ...`), scheme (`the best evaluation
  scheme for ...`), continuous, dsolve (`solve y from 0 to 2 in 4
  steps`), sums and products in words (`the sum of i squared for i
  from 1 to n`, `the product over k from 1 to 5 of k`),
  properties checked with equiv (commutative, associative, additive,
  idempotent), definitions (`let f of x be ...`,
  `call it g`), `assume`, `forget`; complex and trig forms (`the polar
  form of 1 + i`, `write (1+2i)/(3-i) in a+bi form`, `real and imaginary
  parts of e^(ix)`, `modulus and argument of 3 + 4i`, `convert 2 cis(pi/3)
  to rectangular form`, `the 5th roots of unity`, `solve z^4 = -16`,
  `rewrite sin^2 x using double angle`, `express tan x in terms of sin
  and cos`, `simplify sin x cos x`, `expand sin(a+b)`, `product to sum
  of sin x cos y`, `write 3 sin x + 4 cos x in the form R sin(x + phi)`,
  `rewrite acos x in terms of asin`; see "Complex numbers and trig
  forms"); complex functions, quaternions and octonions (`is z^2
  holomorphic`, `the residue of 1/(z^2+1) at i`, `rotate (1, 0, 0) by 90
  degrees about the z axis using quaternions`, `is octonion
  multiplication associative`; see "Complex and hypercomplex numbers");
  algebra (`factor x^4 - 1`, `factorise ...`, `what are the
  factors of x^2 - 4`, `expand (x+1)^5`, `simplify (x^2-1)/(x-1)` (not
  trig: that is `trig`'s), `cancel ...`, `partial fraction decomposition
  of 1/(x^2-1)`, `decompose F into partial fractions`, `put 1/x + 1/(x+1)
  over a common denominator`, `combine F into a single fraction`,
  `collect the terms of F in x`, `solve x^2 - 5x + 6 = 0`, `solve sin x =
  1/2`, `solve the system x + y = 3, x - y = 1 [for x and y]`, `gcd of
  x^2-1 and x^2+2x+1`, `the lcm of ...`, `the resultant of P and Q in x`,
  `the discriminant of a x^2 + b x + c`, `divide P by Q`, `the quotient
  of P divided by Q`, `the remainder when P is divided by Q`, `plug in x
  = 2 into F`, `substitute x = 2 in F`; see "Symbolic algebra");
  properties answered by props.c (`is f monotonic`, `is
  it strictly increasing`, `where is f decreasing`, `is sin(2x)
  periodic`, `find the period of`, `does f repeat`, `is x squared even`,
  `is f bounded`, `one to one`, `a bijection`, `where is 1/x
  discontinuous`, `where is x^3 concave`, `the asymptotes of`, `the
  roots of`, `where does f cross zero`, `identify the equation x^2 + y^2
  = 1`, `what kind of equation is y'' + y = 0`, `classify`, `what type
  of function is`, `what properties does f have`); rings and finite
  fields (`the inverse of 3 mod 256`, `x ≡ 2 mod 3 and x ≡ 3 mod 5`, `is
  Z/12 a field`, `in GF(2^8) with the AES polynomial multiply 0x57 by
  0x83`, `the magic number for dividing a u32 by 7`; see "Rings and
  finite fields"); transforms, series
  and sums (src/nlp_xform.inc: `the laplace transform of t^2 e^-t`,
  `inverse laplace of 1/(s^2+1)`, `solve y'' + y = 0 with y(0) = 1,
  y'(0) = 0 using laplace`, `the z transform of a^n u[n]`, `inverse z
  transform of z/(z - 0.5)`, `solve y[n+1] = 2y[n], y[0] = 1 using the z
  transform`, `transfer function of y[n] = 0.9 y[n-1] + x[n]`, `the
  frequency response of ...`, `is the filter ... stable`, `fourier series
  of a square wave` (also sawtooth, triangle wave; `on -1 to 1`),
  `fourier transform of e^(-x^2)`, `the dft of [1, 2, 3, 4]`, `does the
  sum of 1/n^2 converge`, `what is the sum of 1/n^2 from 1 to infinity`,
  `the infinite sum of 1/2^n`, `the product of 1 - 1/n^2 for n from 2 to
  infinity`, `radius of convergence of the sum of x^n/n!`, `the
  generating function of fibonacci`, `the exponential generating
  function of 2^n`, `the power series of e^(2x)`, `the asymptotic
  expansion of the sum of 1/k`; see "Transforms, series and sums").
- **Code blocks**: a request followed by a fenced block is about that
  block.  `SAT simplify`, `simplify this`, `optimize this`, `peephole
  with smt`, `make it faster`, `superoptimize`, `find something
  cheaper`, with `budget 20`, `n in [0, 32)` or `n < 32`, `nofile`,
  simplify a SPIR-V or GLSL block (and show spirv-cross's GLSL of the
  result, see "GLSL view") or run the C frontend on a C block (`for
  zen4`, `and test it` too; see "C frontend"); which language the block
  is comes from its tag, or from its content when it has none.
- **Expressions**: `x squared plus one`, `the sum of a, b and c`, `half
  of x`, `20 percent of y`, `the number of set bits in x`, `x shifted
  left by 3`, `e to the power of 2`, mixed freely with ucas notation.
  Number words, `90deg`, `90 degrees`, `90°` (angles, converted to
  radians), `1.5r`/`rad`.  Primes after a name are ucas notation and
  pass through: `what is f'(3)`, `f''`, `the slope field of y' = t -
  y`, `solve y' = -2y + 1 with y(0) = 3` (defines the ODE and its
  initial values, then `dsolve`; also systems, `x' = y and y' = -x`);
  contractions (`what's`, `f's`) are words.  A comma before a sign
  starts a new item: `(y, -x)`, `at (1, -2)`.  Greek letter names (`theta`, `alpha`, or
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

## JSON frontend for agents and programs

```sh
ucas --json [--dir DIR] [--files write|base64|both|none]   # JSON Lines
ucas --json -e '{"cmd": "range", "args": {"f": "isqrt(x: u4)"}}'
ucas --json-schema          # the interface as JSON Schema, every command
python3 tools/ucas_client.py # a client and a short agent session
```

`ucas --json` reads one request per line on stdin and writes one
response per line on stdout, in order, nothing else (diagnostics are
inside the responses; stderr is untouched).  It never prompts, never
parses English unless asked to, and never guesses silently: where the
text frontends take a reading and print a `note:`, JSON mode either
follows a fixed rule (listed below) or answers with an `ambiguous` error
that lists the readings and the key that picks one.

### Requests

```json
{"id": 7, "cmd": "integrate", "args": {"f": "x^2 sin(x)", "var": "x", "from": "0", "to": "pi"}}
```

`id` is any JSON value and comes back as it was; `cmd` names a command;
`args` holds its named, typed arguments.  Unknown fields, unknown
arguments, missing required ones and wrong types are `bad_request`
errors.  `{"cmd": "schema"}` (or `--json-schema`) lists every command
with its parameters (`commands`), the request and response schemas
(`$defs`) and the request-wide options; `tools/ucas_client.py` reads it.

Each command is a template: the arguments become one ucas statement,
which is run as if typed and returned as `ucas` in the response (error
positions refer to it).  So the language is the one documented above,
and what ran is never in doubt.  Argument types:

| type | JSON | becomes |
|------|------|---------|
| expression | a string of ucas syntax (one line: no newline or `#`), a number, or the AST form below | the text, or the AST fully parenthesised |
| function | a defined name, an expression with inline types (`"isqrt(x: u4)"`), or `{"body": expr, "params": [...], "name"?}` | the name; an object is defined first (as `it`, `that`, or its `name`) |
| expressions, names | an array (a single value is one element) | joined with `, ` |
| number | a JSON number, or a constant expression as a string (`"pi/2"`, `"inf"`) | the number |
| interval | `[lo, hi]` | `lo..hi` |
| params (define) | `[{"name": "x", "type": "u8", "range": [0, 10], "uniform": false}]` | `x: u8 in 0..10` |

The AST form needs no precedence or juxtaposition rules:
`{"op": "+", "args": [a, b, ...]}` (binary `+ - * / ^ % mod & | xor <<
>> == != < <= > >= && || =`, n-ary for `+ * & | xor && ||`), `{"op":
"neg"|"not"|"~", "args": [a]}`, `{"op": "?:", "args": [c, a, b]}`, any
other `op` a function call (`{"op": "atan2", "args": [y, x]}`), `{"var":
"x"}`, `{"num": "1/3"}` (an exact fraction), `{"expr": "ucas text"}`.

Options accepted by every command, in `args`:

| option | |
|--------|--|
| `assume` | answers to ambiguities: `{"caret": "power", "z": "complex", "log": "ln"}`; a reading's text, a unique prefix or word of it, or its 1-based index |
| `defaults` | `true`: take the reading the CLI would take (noted as kind `default`) instead of an error |
| `timeout_s` | wall-clock limit; the request runs in a forked child, and on a timeout or a crash the state is as it was |
| `files` | `write` (to `--dir`, default the working directory), `base64` (inline, nothing written), `both`, `none` |
| `answers` | replies to follow-up questions (`english` asking for an interval), in order |

### Responses

```json
{"id": 7, "ok": true, "cmd": "integrate", "ucas": "integrate x/x from -1 to 1", "structured": true,
 "result": {"kind": "integrate", "var": "x", "integrand": "x/x", "antiderivative": "x",
            "definite": {"from": -1, "to": 1, "symbolic": 2, "numeric": 2, "error_estimate": 0,
                         "evaluations": 15, "value": 2}},
 "notes": [{"kind": "info", "text": "x/x = 1 where x != 0; the result holds there"}],
 "conditions": ["x != 0"],
 "checks": [{"kind": "derivative", "points": 400, "max_diff": 0, "passed": true, "text": "..."},
            {"kind": "numeric_integral", "points": 15, "max_diff": 0, "passed": true, "text": "..."}],
 "files": [], "text": "∫ x/x dx\n  note: x/x = 1 where x != 0; ...\n  = x + C\n  ..."}
```

| field | |
|-------|--|
| `ok` | false when there is an `error` |
| `ucas` | the statements that ran |
| `result` | the command's fields, with `kind`; exact values as `{"value": 0.333.., "exact": "1/3"}`; expressions as ucas strings that parse back; a command not (yet) structured gives `{"text": ...}` and `"structured": false` |
| `results` | every result, when the request ran several commands (`ucas`, `funcs`) |
| `notes` | `{"kind", "text"}`: `rule` (a fixed reading of the syntax, e.g. `1/2x read as 1/(2x)`), `info`, `assumed` (args.assume answered it), `default` (args.defaults) |
| `conditions` | where the answer holds (`x != 0`, `a != 0`, `b^2 - 4a*c >= 0 (real solutions)`) |
| `checks` | independent checks: `{"kind", "points", "max_diff", "passed", "text"}` |
| `proof` | `{"kind": "symbolic"/"exhaustive"/"sat"/"numeric", "text"}`: how far the answer is established |
| `files` | `{"name", "mime", "bytes", "path"?, "base64"?}` (plots, animations, rewritten SPIR-V) |
| `text` | the output as the CLI prints it |
| `error` | `{"code", "message", "pos"?: {"line", "col"}, "key"?, "readings"?, "how"?}` |

Error codes: `bad_json`, `bad_request`, `unknown_command`,
`parse_error`, `domain_error`, `unsupported`, `failed` (the command
reported an error), `ambiguous`, `needs_input`, `not_understood`
(English), `timeout`, `internal` (the evaluation crashed; the state is
unchanged when it ran under `timeout_s`).  A `fail()` inside a command
becomes the error and the process carries on; a failure that a command
catches and works around is not an error.  Numbers that are not finite
are `null` (an unbounded interval end, a limit's value is a `kind`).

```json
{"id": 4, "cmd": "ucas", "args": {"source": "k = 3 ^ 5 & 1"}}
{"id": 4, "ok": false, "cmd": "ucas", "ucas": "k = 3 ^ 5 & 1", ..., "error": {"code": "ambiguous",
 "message": "\"^\" between integers, next to bitwise operators", "key": "caret",
 "readings": [{"index": 1, "reading": "xor (bitwise exclusive or)", "default": true},
              {"index": 2, "reading": "power", "default": false}],
 "how": "repeat the request with args.assume.caret set to a reading (or its index), or args.defaults = true ..."}}
```

### Ambiguity: rules, and what is asked

Expressions are ucas syntax, so its grammar applies as written, and the
readings in "How ambiguities are resolved" are rules, not guesses: `2x`,
`sin x^2`, `1/2x` = `1/(2x)`, `^` is power (`xor` is exclusive or), `a <
b < c`, `x = 0` inside an expression, `a & b == c`, `|x|`, `log` is the
natural log (English `log` is asked), `i` is the imaginary unit in the
complex commands and the other variables are real.  They come back as
notes of kind `rule`.  Anything else is a decision, and JSON mode makes
none: it answers `ambiguous` with the readings and an `args.assume` key.
Two sources:

- **Questions** the terminal would ask (`nlp_can_ask` is true in JSON
  mode; the answer comes from `args.assume`, never stdin): `caret` (`^`
  between integers next to bitwise operators), `solve_for` (one equation,
  several variables), `wrt` (an integrand with no variable), `keep_ends`
  (approx: special points at the interval's ends), `anim_type`,
  `spv_type`, `spv_value`, `i_param` (a parameter named `i`), and in
  English `log`, `byte`, `char`, `int`, `integer`, `word`, `sqrt`, `not`,
  `andor`, `scope` (what "of" applies to), `unknown_word`.  Others have
  the key `question`.
- **Noted guesses** (a `note:` in the CLI): `z` (z, w are complex unless
  x or y appear), `untyped` (a parameter's type inferred from integer
  operators), `juxtaposition` (`xy` read as `x*y`), `isqrt` (`sqrt` of an
  integer), `j` (j as the imaginary unit), `angle` (degrees), `free_vars`
  (a definition without parameters), `e_units` (`2e1` as `2*e1`).  The
  assume value can only confirm the reading taken; the other readings
  carry a `how` (annotate `z: real`, give `params`), since they need
  different input rather than a flag.
- `define` needs a type for every parameter (or a range): otherwise
  `param_type` (`real` or `int`) is asked.

`{"cmd": "assume", "args": {"set": {...}, "clear": true}}` keeps answers
for the session (each request's `args.assume` comes first).

### Sessions

Definitions (functions, models, bases, the `find`/`approx` searches)
persist across requests.  `define` returns the function as data
(`signature`, `params`, `body`); `state` lists the functions, models,
the active model, bases and the session's assumptions; `reset` answers
and then re-executes the binary (so nothing survives; stdin is read
unbuffered, so no request is lost).  `ucas` runs raw statements (several
lines allowed, with an optional `code` block for SPIR-V/GLSL): the escape
hatch for recurrences, ODEs, bases, models and everything without a
template.  `english` is the explicit opt-in to the translator: its result
has the translated `statements` and the `results` they produced; a
question is `ambiguous` (`assume`) or `needs_input` (`answers`).

`timeout_s` forks: the child answers and, if it finishes, carries on as
the server with the new state; on a timeout it is killed and the parent
answers `timeout` with its state untouched.  The first such request turns
the original process into a waiter that exits when the last worker has
(so a client sees one process, one pipe, throughout).

### Structured commands

Every command in the schema returns `"structured": true` except `help`,
`funcs`, `models`, `use`, `scheme`, `continuous`, `animate` (its GIF is
still in `files`) and raw `ucas`/`english` statements of those.  By area:

| area | commands |
|------|----------|
| look, analysis | `show table parse range domain wrap crit equiv fit error cost gates find more approx approx_more` |
| calculus | `diff grad hessian laplacian jacobian wronskian limit taylor implicit integrate antiderivative closed dsolve` |
| algebra | `expand factor simplify together cancel collect apart gcd lcm resultant discriminant quo rem divide solve substitute` |
| properties | `monotonic period identify zeros parity symmetry bounded injective surjective bijective continuity convexity asymptotes` |
| complex | `complex re im conj modulus arg polar trig nthroots unity holomorphic residue poles` and `integrate` with `around` (`kind: contour_integral`) |
| hypercomplex | `multiplication_table algebra zerodivisors associator commutator rotate axisangle rotmatrix frommatrix slerp` |
| rings, finite fields | `ring inverse_mod order_mod phi carmichael factorint legendre jacobi sqrtmod qr primroot dlog crt congruence magic ring_factor irreducible primitive ring_roots ring_poly ring_gcd ring_norm matrix gf2matrix` |
| transforms, series | `laplace ilaplace laplace_solve ztrans iztrans zsolve tf fourier ifourier fourier_series dft idft dtft genfunc egf infinite_sum infinite_product converge radius powerseries asymptotic` |
| solids | `area revolve volume surfacearea arclength centroid` |
| pictures, shaders | `plot plot3d neighbourhood animate spirv_simplify` |
| abstract algebra | `bigint bigint_factor snf hnf zmatrix qmatrix abgroup abiso abhom abtensor abtor abext abmap abelem absub abexact permgroup permelem permsub permhom cayley groebner idealmember idealeq fxmodule pmsnf` |
| elliptic curves | `elliptic diophantine weierstrass congruent mordell cubesum` |
| session | `define state assume reset english schema` |
| homological | `space homology cohomology les bockstein bockstein_trace prop01` |
| C frontend | `c_optimize c_lower c_targets` |

A ring argument is typed: `{"ring": "GF(2^8)", "modulus": "aes"}` (the
reduction polynomial or its name: `aes`, `crc32` ...), `{"ring": "Z/12"}`,
`{"ring": "Z[i]"}`, or `{"modulus": 256}` alone for Z/256 (`mod 256`).
Other typed arguments: `matrix` (`[[1, 2], [3, 4]]`), `values` (`[1,
"-2+2i"]`), `congruences` (`[{"residue": 2, "modulus": 3}, ...]`).
Examples:

```json
{"cmd": "inverse_mod", "args": {"a": "0x53", "ring": "GF(2^8)", "modulus": "aes"}}
{"cmd": "crt", "args": {"congruences": [{"residue": 2, "modulus": 3}, {"residue": 3, "modulus": 5}]}}
{"cmd": "matrix", "args": {"matrix": [[1, 2], [3, 4]], "op": "solve", "rhs": [1, 0], "modulus": 7}}
{"cmd": "laplace", "args": {"f": "t^2 e^-t"}}
{"cmd": "infinite_sum", "args": {"index": "n", "f": "1/n^2"}}
{"cmd": "revolve", "args": {"f": "sqrt(x)", "from": 0, "to": 4, "about": "x"}}
{"cmd": "spirv_simplify", "args": {"code": "vec3 decode(vec3 c) { ... }", "float_only": true, "float_ranges": {"c": [0, 1]}}}
{"cmd": "spirv_simplify", "args": {"code": "unsigned shr(unsigned x, unsigned n) { ... }", "lang": "c", "ranges": {"n": [0, 31]}}}
{"cmd": "english", "args": {"text": "SAT simplify", "code": "unsigned f(unsigned x) { return (x * 8) / 4; }", "lang": "c"}}
{"cmd": "animate", "args": {"code": "#version 450 ...", "inputs": [{"png_base64": "iVBORw0..."}, {"color": "black"}, "checker"], "output_format": "RG8", "size": "128x128", "sampler": "nearest clamp", "seed": 1}}
```

`animate` with `code` (SPIR-V assembly or GLSL) runs `exec` on it; its
image arguments (see [Shaders on images](#shaders-on-images)): `inputs`
in binding order, each a string (a colour name, `#rrggbb`, `noise`,
`gradient`, `checker`, a format), `null` (noise) or an object with
`color`, `pattern`, `png_base64` (any PNG, JPEG, GIF, BMP or PPM file in
base64, with an optional `name`), `file` (a path), `format`, `width`,
`height`, `filter` and `address`; `output_format`, `size`, `sampler`,
`seed`, `lanes`, `to`, `nofile`.  The GIF comes back in `files`.

What the results hold beyond the common fields:

- **Rings**: the answer's parts from its headline (`lhs`, `value`,
  `value_int` when it is an integer that fits a double exactly,
  `modulus`/`in`/`over`), typed lists where there are several answers
  (`roots` of sqrtmod, `solutions` and `modulus` of crt/congruence, dlog's
  `value`/`modulus`/`base`, primroot's `primitive_root`/`order`/`phi`),
  and the ring's properties as fields (`units`, `unit_group`,
  `zero_divisors`, `ideals` ...).
- **Transforms**: `lhs`, `value`, `roc` (the region of convergence),
  `poles`, `rules`, `valid_for` (`t >= 0`), lines such as `Y(s) = ...`,
  `y(t) = ...`, `H(z) = ...`, `G(x) = ...` as fields under their own
  names, DFT `values` as `{exact, re, im, abs, arg}`, radius `radius` and
  `center`; assumptions (`assuming w > 0`) are `conditions`, and each
  check (transformed back, Parseval, Gauss-Kronrod) is in `checks`.
- **Solids**: `area`, `signed_integral`, `volume`, `surface_area`,
  `arc_length`, `region_area`, `centroid` as `{value, exact, is_exact}`
  (`exact` the recognised form, `8pi`, `64/3`); `integral` with the
  antiderivative, symbolic and numeric values and the error estimate;
  `method`; the GIF in `files`; the image size in `image`.
- **Float simplify** (`result.float`): `cost_before`, `cost_after`,
  `mode` (`vulkan_precision`, `exact`, `ulp`), and `rewrites`,
  `approximations`, `not_applied`, each with the `instruction`, its
  `replacement` lines, `kind`, `cost_delta`, `proof` (`bit_exact`,
  `bit_exact_where_defined`, `vulkan_precision`, `approximation`) and
  per lane group `lanes`: `exact`, `proof`, `ulp_bound`, `rel_error`,
  `not_exact_at`, `witness`, `fail`, `cost`, `gain`, `derivation` (the
  proof's lines).  The arguments `exact`, `ulp`, `approx`, `percomp`,
  `float_only` and `float_ranges` are the CLI's options; `model` makes
  a cost model the active one first.  Sums made dot products are in
  `dot_products` (`from`, `to`, `line`, `cost_before`, `cost_after`,
  `relies_on`, `proof`), and their `rewrites` entries have proof `dot`.
- **Analysis** (stress test 1): `equiv` has `method` (`exhaustive`,
  `sat`, `sat_counterexample`, `sampled`), `sat` (`result`, `aig_nodes`,
  `conflicts`, `why`) and `narrowed` (the where clause's bounds); `find`
  has `dot_products` (`model`, `available`; when there is one,
  `cost_before`, `cost`, `expr`, `relies_on`, `proof`: `exhaustive`,
  `sat` or `lanes`; or `candidate` and `not_proved`), `peephole` (each
  `{cost, expr, drops, proof}`), `peephole_fixpoint`
  and `parametric` (`{cost, cost_per_use, expr, K, why, proof}`), and
  each implementation's `proof`; `cost` has `shared` (the named
  subterms), `loops` (`body`, `overhead`, `once`, `trips_worst`,
  `trips_average`, `trips_known`, `trips_bound_sound`, `total_worst`,
  `total_average`), `average_total`, `selects_not_listed` and
  `unavailable` (dot forms the model does not have); `range` has
  `proved_bound` (`min`, `max`, `min_exact`, `max_exact` and the steps).
  Any result list is cut at 500 entries (`KEY_omitted`).
- **Identify**: `properties` with a field per property, each `{text,
  tag, proof}` when it carries a tag (`proved`, `numeric`, `witness`).
- **Plots**: `axes` (`{var, lo, hi, why, values_lo, values_hi}`),
  `image` (`{width, height, frames, fps, bytes}`), the GIF in `files`.
- **Property family, hypercomplex algebras, nthroots, implicit**: the
  findings as fields (`injective`, `witness`, `vertical`, `oblique`,
  `commutative`, `associative` ...); nthroots' `roots` typed (`re`, `im`,
  `exact`, `polar`), implicit's `derivative`.

### For developers: the report API

`src/report.h`.  A command calls `rep_begin("kind")` and a field per
printed result next to its `printf`: `rep_str`, `rep_num`, `rep_int`,
`rep_bool`, `rep_exact(key, value, "1/3")`, `rep_expr(key, nid)`,
`rep_list`/`rep_obj` ... `rep_end` for nesting (key `NULL` inside a
list; a `rep_list` under a key already there continues it), `rep_cond`,
`rep_check`, `rep_proof`, `rep_note`, `rep_file`, `rep_text_list(key)`
for lines as a list, and `rep_text_fields()`: the lines the command
prints from there on become fields when the request ends (`  label:
value` -> `"label": value`, a trailing `[proved]` tag -> `{text, tag,
proof}`, `check:` -> `checks`, `how:` -> `method`, `assuming ...` ->
`conditions`, `y(t) = ...` -> `"y(t)"`, the first unindented line ->
`headline` and its `lhs = value mod m` parts).  Most of the commands
added after the first JSON release use it at their dispatcher (ring.c,
xform.c, series.c, viz.c, hyper.c, props.c) and typed calls for their
main values; it is the quickest way to make a new command structured.  Outside a JSON request every call returns
at once, so the text output is unchanged (`make test`).  A new command
gets a template line in `src/jsonmode.c`'s table (name, statement words,
typed parameters with their fragments) and works at once, unstructured;
a few `rep_` calls make it structured.  Hooks: `fail()` records its
message (util.c) and the statement loop commits it only when the
statement ended in it (parse.c); `note()` checks the guesses table
(report.c); `clarify` and the follow-up question in nlp.c ask
`rep_answer`; `deliver` in anim.c hands files over.  Text is captured by
pointing fd 1 at a temporary file for the request, so child tools
(glslangValidator, spirv-dis) are captured too.

Why a template per command rather than a second parser: one grammar, one
implementation, and the response shows the exact statement, so an agent
can learn the text language from the JSON one.  Why JSON Lines: one
request in flight, no framing beyond newlines, trivially driven from any
language.  The Discord bot still uses its `\x1e` directives (`defer`,
`ask`, `answer`, `file`); JSON mode carries the same events as data
(`ambiguous`/`needs_input` errors, `files`) and could replace them.

`tests/json/*.jsonl` (with `.out`) cover every structured command,
the errors, ambiguities, state, reset, timeouts and the schema;
`make json-test` (part of `make test`), `make json-accept`.

## MCP server

```sh
ucas --mcp              # a Model Context Protocol server on stdin/stdout
ucas --mcp --flat       # one tool per command instead of one per area
ucas --mcp --dir DIR    # also write the GIFs and modules into DIR
```

A client configuration (Claude Desktop, Claude Code, or any MCP client
that starts stdio servers):

```json
{"mcpServers": {"ucas": {"command": "/path/to/ucas", "args": ["--mcp"]}}}
```

On another machine, let ssh carry the stdio:

```json
{"mcpServers": {"ucas": {"command": "ssh", "args": ["host", "/path/to/ucas", "--mcp"]}}}
```

(`claude mcp add ucas -- /path/to/ucas --mcp` in Claude Code.)

`src/mcp.c` speaks JSON-RPC 2.0, one message per line, as MCP's stdio
transport specifies: `initialize` (protocol versions 2025-06-18,
2025-03-26 and 2024-11-05; another version is answered with the latest),
`notifications/initialized`, `ping`, `tools/list`, `tools/call`,
`resources/list`, `resources/read`, `prompts/list`, `prompts/get`.
Notifications are not answered; errors are JSON-RPC's (-32700 parse
error, -32600 invalid request or a batch, -32601 unknown method, -32602
an unknown tool, op or prompt, or arguments that are not an object).
stdout carries only protocol messages.

It is a layer over the JSON frontend, not beside it: a tool call becomes
a JSON request (`{"cmd": op, "args": ...}`) served by the same code, so
everything in "JSON frontend for agents and programs" holds: typed
arguments, sessions (definitions persist across calls; `session` op
`reset` restarts the process image), `assume` answers, `timeout_s` (the
call runs in a forked child; the state survives a timeout).  The result
of a call:

- `content`: the text the CLI prints (for an error, the code, message,
  the readings of an ambiguity with their `assume` key, then the text);
  each GIF as `{"type": "image", "mimeType": "image/gif", "data": ...}`;
  other files (a rewritten SPIR-V module) as embedded resources.  Images
  are kept under 4 MiB of base64: animations are re-rendered at a lower
  frame rate first, as for the Discord bot.
- `structuredContent` (protocol 2025-06-18): the JSON frontend's
  response, with the typed `result`, `checks`, `proof`, `conditions`,
  `notes` and `error` (files without their base64).  The grouped tools
  declare it as their `outputSchema`.
- `isError`: true when the response's `ok` is false.  An ambiguity is an
  error whose `structuredContent.error` has `key` and `readings`: call
  again with `args.assume.KEY`.

**Tools: grouped by default.**  There are about 150 commands.  As one
tool each they make a 340 KB `tools/list`, and several clients cap or
truncate the tool list, so by default there is one tool per area, with
an `op` and the op's `args`:

| tool | ops |
|------|-----|
| `analyze` | show table parse range domain wrap crit equiv fit error cost gates find more approx approx_more |
| `calculus` | diff grad hessian laplacian jacobian wronskian limit taylor implicit integrate antiderivative closed scheme continuous dsolve |
| `algebra` | expand factor simplify together cancel collect apart gcd lcm resultant discriminant quo rem divide solve substitute |
| `properties` | monotonic period identify zeros parity symmetry bounded injective surjective bijective continuity convexity asymptotes |
| `complex` | complex re im conj modulus arg polar trig nthroots unity holomorphic residue poles |
| `hypercomplex` | multiplication_table algebra zerodivisors associator commutator rotate axisangle rotmatrix frommatrix slerp |
| `rings` | ring inverse_mod order_mod phi carmichael factorint legendre jacobi sqrtmod qr primroot dlog crt congruence magic ring_factor irreducible primitive ring_roots ring_poly ring_gcd ring_norm matrix gf2matrix |
| `transforms` | laplace ilaplace laplace_solve ztrans iztrans zsolve tf fourier ifourier fourier_series dft idft dtft genfunc egf infinite_sum infinite_product converge radius powerseries asymptotic |
| `solids` | area revolve volume surfacearea arclength centroid |
| `plot` | plot plot3d neighbourhood animate |
| `abstract_algebra` | bigint bigint_factor snf hnf zmatrix qmatrix abgroup abiso abhom abtensor abtor abext abmap abelem absub abexact permgroup permelem permsub permhom cayley groebner idealmember idealeq fxmodule pmsnf |
| `elliptic` | elliptic diophantine weierstrass congruent mordell cubesum |
| `session` | define state assume reset schema funcs models use help |
| `homological` | space homology cohomology les bockstein bockstein_trace prop01 |
| `c` | c_optimize c_lower c_targets |
| `ucas`, `english`, `spirv_simplify` | tools of their own (full input schema) |

```json
{"name": "calculus", "arguments": {"op": "integrate", "args": {"f": "x^2 sin(x)", "var": "x", "from": "0", "to": "pi"}}}
```

Each grouped tool's description lists its ops with their parameters
(`integrate(f*, var, from, to, around)`, `*` required) and what they
compute, about 4 KB a tool and 60 KB in all; the exact schema of every
op is the `session` tool's `schema` op or the `ucas://schema` resource.
The trade: a model picks among 15 tools and reads one description,
instead of scanning 120; the price is that an op's arguments are checked
by ucas (a precise `bad_request` error) rather than by the client.
`--flat` gives one tool per command with its exact `inputSchema` (no
`outputSchema` there, to keep the list at 340 KB rather than 690 KB),
for clients that prefer it.  A command added to `src/jsonmode.c`'s table
appears in both forms (in `--mcp` it is a tool of its own until it is
named in a group in `src/mcp.c`).

**Resources:** `ucas://state` (the session's functions, models, bases
and assumptions, JSON), `ucas://help` (the command summary),
`ucas://schema` (the JSON Schema of every command).  **Prompts:**
`cheaper_implementation` (f), `prove_identity` (identity),
`explore_function` (f): each a short plan in terms of the tools.

Tests: `tests/mcp/*.jsonl` with `.out` (the handshake, every error,
`tools/list`, calls with images, an ambiguity and its answer, a timeout,
reset, resources, prompts, the 2024-11-05 protocol, `--flat`), `make
mcp-test` (part of `make test`), `make mcp-accept`.
`tools/mcp_client.py` is a stdlib client (handshake, calls, an ambiguity
answered, an image saved), and `tools/mcp_sdk_check.mjs` drives the
server with the official TypeScript SDK client (`npm install
@modelcontextprotocol/sdk`), which also validates `structuredContent`
against `outputSchema`.

## Bases and exact synthesis

```
basis c      { and(x, y) = x & y; or(x, y) = x | y; xor(x, y) = x xor y; not(x) = ~x }
basis spirv : c {}
basis a64 : c    { bic(x, y) = x & ~y; orn(x, y) = x | ~y; eon(x, y) = x xor ~y }
basis a64sha : a64 { bsl(s, x, y) = (s & x) | (~s & y); eor3(...); bcax(...) }
basis x86bmi : c { andn(x, y) = ~x & y }
basis avx512 : x86bmi { ternlog(x, y, z) = lut }
basis nand   { nand(x, y) = ~(x & y) }
```

A gate is declared by its expression over 1-bit inputs, optionally with
`cost n`.  `= lut` declares a truth-table instruction (vpternlog, LOP3):
any function of its inputs in one step, the table chosen per use and
printed as vpternlog's immediate (bit `a<<2 | b<<1 | c`).  So `gates maj
basis avx512` is one `ternlog(a, b, c, 0xe8)`, a 2-bit adder 3 steps and
a 4:1 multiplexer 3 steps, each proved minimal (README "x86-64").  ucas derives its truth table and which arguments commute from
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
model gt1030 { iadd: 1, logic: 1, shift: 2, imul: 3, idiv: 28.5, idivk: 8.75,
  cmp: 2.375, select: 1.625, minmax: 1, bitcnt: 3, conv: 3, bitcast: 0,
  fadd: 1, fmul: 1, fdiv: 3, fsqrt: 4, ftrans: 4, fpow: 8, f64: 32.375,
  load: 6.875, branch: 6.25, diverge: 14.375, barrier: 9.125,
  atomic_shared: 6.625, atomic_global: 50, store: 15,
  dot4x8: 1, dot4x8sat: 6.5
  sdiv: 36, sdivk: 21, irem: 32.25, iremk: 13.5
  fuse lop3 = lut 3 : logic
  fuse imad(a: int, b: int, c: int) = a * b + c : imul
  ... }
model mine : gt1030 { imul: 1 }
```

`gt1030` (NVIDIA GP108, sm_61, through SPIR-V) is built in and active.
Its numbers are issue costs relative to fadd, measured on a GT 1030 by
the SPIR-V corpus in tests/gt1030 (snx-shaderperf, throughput with the
SMs full; the report is tests/gt1030/results/2026-11-11.md): fp32 is
full rate, 32-bit integer multiply is an XMAD sequence, integer
division is emulated, fdiv, sqrt and the SFU ops are quarter rate, fp64
is 1/32 rate (`f64` multiplies the cost of every f64 float operation;
f64 division and sqrt are library calls, about 300 and 570).  load,
store and atomic_global are cache-resident accesses; streams from DRAM
cost 90 to 140.  DP4A is the dot4x8 class (see Dot products below).
The report also has what the classes do not show: subgroup operations,
textures; the fusions it measured are in the model (below).  `cd
tests/gt1030 && ./run.py` measures again (it needs the GPU).

Integer division has a class for each kind: `idiv`/`idivk` unsigned
(by a variable, by a constant), `sdiv`/`sdivk` signed, `irem`/`iremk`
and `srem`/`sremk` the remainders.  A model that does not give them
prices them as its `idiv`/`idivk` (so a model written before keeps its
prices), and a signed remainder it does not give is the signed
division plus what a remainder adds to the unsigned one.  `uN` and
`bool` divide unsigned, `iN` and `int` signed.  Whatever the model, an
unsigned division by 2^k is priced as the shift (its remainder as the
mask) that every compiler makes of it, and a signed one as its three
shifts and an add.  On the GT 1030 (both runs, less the add each test
step has): udiv 28.5, by 7 8.75; umod 32.25, by 7 13.5; sdiv 36, by 7
21; a signed remainder is not measured (39.75 and 25.75 by the rule).

### Fusions

Several operations that one instruction does are declared in the
model, not in ucas:

```
fuse NAME(a: int, b: int, c: int) = a * b + c : imul
fuse NAME(a: int, k: const, b: int) = (a << k) + b : shift
fuse NAME(a: float, b: float, c: float) = a * b + c : fmul contract
fuse NAME(a: float) = -a : 0 in fadd fmul
fuse NAME = lut 3 : logic
nofuse NAME
```

A fusion's inputs are `int`, `float`, `const` (a constant) or `any`;
its pattern is an expression over them, matched with commutative
operands either way round (and `x << k` also matches `x * 2^k`, `x >>
k` an unsigned `x / 2^k`); its price is a sum of numbers and classes
(`2*shift + iadd`).  `contract` marks one that rounds once where the
operations round twice (a*b + c as FFMA): it is not used on a SPIR-V
result decorated NoContraction (`precise`).  `in CLASS ...` makes it
a modifier: the operation costs that much where every use of it is an
operation of those classes (a negation or abs folded into the add that
reads it).  `lut K` is any tree of bitwise logic (and, or, xor, not)
with at most K inputs, LOP3 or vpternlog.  A derived model inherits
its parent's fusions; `nofuse NAME` drops one.  `models` lists them.

Pricing covers the expression's DAG with them, the instruction
selection of a compiler: a value used more than once is computed once
and is an input to everything that uses it, which cuts the DAG into
trees, and in each tree dynamic programming takes the cheapest of each
node by itself, each pattern rooted there (its inner nodes used only
inside it) and each k-feasible cut of its logic cone for a lut.  That is
the cheapest cover that never computes a value twice.  `cost` lists the
fusions it used, `find` and `simplify` (SPIR-V) price with them.

gt1030 has the fusions its corpus measured, each from a test that
fails the per-class sum: `lop3` (lop3_select, 4 logic ops and an add,
2.00; lop3_xor3 2.00), `imad` (imad and imul_const, 3.0: the add folds
into the XMAD chain), `shl_add` and `shr_add` (shl_const_add,
shr_const_add, imul_pow2, udiv_pow2: 2.0, a shift by a constant and an
add cost one shift; the corpus cannot tell that from a constant shift
costing 1), `ffma`, `ffms`, `ffmr` (fmad_contract 1.0; fmad_precise,
NoContraction, 2.0), `fneg`, `fabs` (fneg_add, fabs_add: 1.0).  rotl,
two shifts and two logic ops, is priced 5 (two shifts and a LOP3)
against 4.41 measured.  The x86 models keep their bases (vpternlog,
andn) for `gates` and `find`, not fusions.

```
> cost lop3(a: u32, b: u32, c: u32, k: u32) = ((a & b) | (~a & c)) + k
cost of lop3 under gt1030: 2 (5 operations, shared subterms counted once)
  iadd             1 x 1     = 1
  fused by the model (one instruction for several operations):
    lop3             1 x 1     = 1  (bitwise logic of up to 3 inputs; saves 3)
> cost sh(a: u32, b: u32, c: u32) = a * b + (a * b) * c
cost of sh under gt1030: 6 (3 operations, shared subterms counted once)
  imul             1 x 3     = 3
  fused by the model (one instruction for several operations):
    imad             1 x 3     = 3  (a*b + c; saves 1)
```

### Rewrites the prices make worthwhile

`find` (and `simplify` on SPIR-V) also try identities that hold on the
domain the user gave, and keep one only when the active model prices
it lower; each comes with its derivation and the conditions it holds
under, proved from the parameters' ranges and the where clause
(interval arithmetic, or bound.c's monotonicity proofs), then checked
again on the domain as a second opinion (every point, SAT, or a sample
when floats are in it).  src/irw.c:

- a signed division or remainder whose dividend is never negative and
  divisor at least 1 is the unsigned one (trunc = floor there, and the
  bits read as unsigned are the same numbers);
- `x < y ? x : y` is `min(x, y)`, and the other compares and arms
  (integers: everywhere; floats: where neither value is NaN, the sign
  of a zero may change, both allowed by Vulkan's default float
  controls);
- a clamp written with selects, `x < lo ? lo : min(x, hi)`, is `max(lo,
  min(x, hi))` when lo <= hi is proved (and the mirror image);
- a division or remainder by a variable through f32: q0 = trunc(f32(a)
  / f32(b)), r0 = a - q0 b, then q0 + (r0 >= b) - (r0 < 0).  f32(a),
  f32(b) are exact below 2^24, Vulkan's division is within 2.5 ulp and
  ulp(x) <= 2^-23 x, so |p - x| <= 5*2^-24 a/b; below 1, q0 is within
  one of floor(a/b) and r0 in [-b, 2b) says which.  The bound is
  printed with the domain's numbers; when it is not below 1 the
  rewrite is shown as not applied, with why.  Its check evaluates with
  ucas's correctly rounded division, so after it no operation is
  dropped on an evaluation's word (correctly rounded, the correction
  is never needed; at 2.5 ulp it is).

```
> sd(a: i32 in 0..1000, b: i32) = a / 7 + b
> find sd 1
  rewrites the model prices lower (each derived, with what it holds under):
  cost 10  i32(u32(a)/7) + b                        the signed division a/7 as u32 (its operands are never negative) (derived; checked by SAT)
      holds where a >= 0 (a in [0, 1000]: its range) and 7 >= 1 (7 in [7, 7]: a constant)
      ...
> ud(a: u32 in 0..60000, b: u32 in 30..4000) = a / b
> find ud 1
  cost 23  u32(t1 + i32(t2 >= i32(b)) - i32(t2 < 0)) where t1 = i32(f32(i32(a))/f32(i32(b))), t2 = i32(a) - t1*i32(b) the division a/b through f32 with one correction step ...
      holds where a in [0, 60000] (its range), b in [30, 4000] (its range): both below 2^24, the divisor >= 1, and 5*2^-24 * 60000/30 = 0.000596 < 1
```

Under gt1030 the signed division by 7 (21) becomes the unsigned one
(8.75), min costs 1 against a compare and a select (4), and the float
path 23 against 28.5.  Under a model where they are not cheaper
(`model selcheap : gt1030 { cmp: 1, select: 1, minmax: 4 }`, or one
where conversions are dear) find says "derived, but not cheaper" and
leaves the function as it is.  In SPIR-V, OpSDiv/OpSRem/OpSMod become
OpUDiv/OpUMod on the bits (bitcasts in and out: OpUDiv wants an
unsigned result type) and OpSelect of a compare of its own arms
SMin/UMin/SMax/UMax (the compare goes too when nothing else uses it),
each proved by SAT under the request's assumptions and kept when the
model's prices say so; the totals are priced with the fusions and the
fusions used are listed.  The function level (GLSL) offers the same
signed-to-unsigned division where the calls' domains keep the operands
non-negative, checked on the whole inferred domain and kept when the
model prices it lower.  A request's `use MODEL` and model definitions
run before its simplify.

`cost F` counts each shared subterm once (and prints the body's shared
subterms once, named, so the rest of the report can refer to them) and
prices every `?:` three ways (at most 12 are listed; the rest are
counted in the estimate):

- as a select (both arms run);
- as a uniform branch (branch cost plus the dearer arm);
- as a divergent branch, where the warp runs both arms plus the
  reconvergence penalty: the "keep-up" cost.

A condition is uniform when it depends only on `uniform` parameters.
Barriers and atomics have value semantics for one invocation
(`barrier(e)` is `e`, an atomic returns the old value), so they can
appear in expressions and be priced.  They are never merged by
hash-consing.

A loop is priced as it runs: its initial values and the loop-invariant
subterms of its body once (hoisted), the rest of the body plus the loop
overhead per iteration (a `for`: the index add, the compare and the
branch; a `while`: the branch, its compare being part of the condition)
times the trip count.  The trip count is exact for constant bounds;
otherwise the worst case over the parameters' ranges (a sound interval
bound, or the maximum when the domain is enumerated) and the average over
the domain; a `while` loop's trip count is measured by running it.

```
> quant(v: i32 in 0..255) = { best = 0; berr = 100000; q0 = (v*31 + 127)/255
    for d in -2..2 { q = q0 + d; qc = q < 0 ? 0 : q > 31 ? 31 : q
      r = (qc << 3) | (qc >> 2); e = (r - v)*(r - v)
      if e < berr { berr = e; best = qc } }; best }
> cost quant
cost of quant under gt1030: 134 (0 operations outside loops and 1 loop, ...)
  shared subterms (named once, used below):
    q = int((31v + 127)/255) + d
    qc = q < 0 ? 0 : q > 31 ? 31 : q
    ...
  loop d in -2..2:
    per iteration: body 20 + overhead 5 (index add, compare, branch) = 25
    trip count: 5 (constant bounds)
    once: 9 (initial values and loop-invariant subterms, hoisted)
    loop total: 134
  ?: on q > 31  (arms cost 0 and 0)  [in a loop body: per iteration]
  ...
```

`find` takes its operation costs from the active model.

### Identities solved for the function

Some rewrites have no fixed right-hand side: their constants are solved
from the divisor, or from the values a subterm takes on its domain
(src/idrw.c).  They reach `find` and `simplify` the same way as the
rewrites above: each is a candidate with its derivation and the
conditions it holds under, kept only where the active model prices it
lower, and otherwise listed as "derived, but not cheaper".

- **A divisibility test by a constant**, `n % d == r` (or `!=`), as a
  multiply and a compare.  With d = 2^k e, e odd, the map n ->
  rotr((n - r) e^-1, k) is a bijection of the machine word, and it takes
  the n with that remainder onto 0..B, B = floor((2^w - 1 - r)/d); so
  the test is `rotr(n*einv + c, k) <= B`.  Signed n (r = 0) the same,
  with the multiples on both sides of 0 moved onto 0..B by the added
  constant.  u8 to u32 and i8 to i32; the derivation prints e^-1, c and
  B.  SAT does not finish on a multiplier against a divider, so the
  second opinion here is the domain's points (all of them when there
  are few, a sample otherwise) and the derivation is the proof.
- **A division or remainder by a constant whose dividend has a proved
  bound** N, as one multiply and one shift in u32: M = ceil(2^s/d) with
  N (M d - 2^s) < 2^s (the excess stays below 1/d) and N M < 2^32 (the
  product does not wrap).  The bound comes from the parameters' ranges,
  the where clause, interval arithmetic or bound.c's proofs.  When N is
  too large for every shift, find says what the dividend would have to
  stay below (`not applied ... needs x <= 57343`).
- **A map of one variable with a small range** (compares and selects, a
  table, quotients; at most 4096 values), read at every value of the
  variable:
  - as a table in one constant, `(C >> b*(x - lo)) & (2^b - 1)`, when
    its entries fit 32 bits (b a power of two when that fits, so that
    the index is a shift; a bool is the low bit; the least value is
    added back when it is not 0);
  - as a multiply and a shift, `(x*M + A) >> s`: for each s, the offsets
    A that serve a multiplier M are an interval whose width is concave
    in M, so the least s with a pair is found exactly, and the
    derivation prints the two values of x where the fit is tightest.

  The largest such subterms are tried first, their operands where
  neither form fits.

```
> dv(n: u32) = n % 7 == 0
> find dv 1
  cost 5   3067833783n <= 613566756                 the divisibility test n % 7 == 0 as a multiply and a compare (derived; checked on a sample)
      holds where n is any u32 (the multiply wraps mod 2^32)
      7 is odd, and 7 * 0xb6db6db7 = 1 mod 2^32
      t(n) = n*0xb6db6db7 takes each 32-bit word to a different one (a bijection): a multiply by an odd number does
      the multiples of 7 are q*7 for q = 0..613566756 (floor((2^32 - 1)/7)); there n * 0xb6db6db7 = q mod 2^32: t takes them onto 0..613566756
      a bijection takes every other n outside 0..613566756: t(n) <= 613566756 exactly where n % 7 == 0
> d(x: u32 in 0..99, y: u32) = x / 10 + y
> find d 1
  cost 5   (103x >> 10) + y                         the division x/10 as a multiply and a shift in u32 (derived; checked on a sample)
      holds where x in [0, 99] (its range)
      M = ceil(2^10/10) = 103, so M*10 = 2^10 + 6
      ...
      x <= 99: x*6 <= 594 < 2^10, and x*M <= 10197 < 2^32, so the product does not wrap in u32
> sw(x: u32 in 0..7, y: u32) = (x == 0 ? 3 : x == 1 ? 7 : x == 2 ? 1 : x == 5 ? 9 : 0) + y
> find sw 1
  cost 6   (0x900173 >> (x << 2) & 15) + y          the map ... of x as a table in one constant (8 entries of 4 bits) (derived; checked by SAT)
> q(v: u32 in 0..255) = (v*31 + 127)/255
> find q 1
  cost 5   249v + 1014 >> 11                        the map (31v + 127)/255 of v as a multiply and a shift in u32 (derived; checked on every point)
```

Under gt1030 the divisibility test costs 5.375 against 15.875 (a
remainder by a constant is 13.5, a multiply 3), the division and its add 5 against
9.75, the table 6 against 17 for four compares and selects, and the
quantisation 5 against 11.75.  Under a model where the multiply is dear
(`model muldear : gt1030 { imul: 20 }`) or a shift by a variable is
(`{ shift: 20 }`) find says "derived, but not cheaper" and leaves the
function as it is.

In SPIR-V, `simplify` reads each OpUDiv, OpUMod, OpSDiv, OpSRem and
OpSMod by a constant, with what its dividend is computed from, into a
ucas function whose inputs carry the request's assumptions (`simplify x
in [0, 1000)`), derives the multiply and the shift for the proved bound
and writes them back (OpIMul, OpShiftRightLogical, bitcasts for a
signed type) where the model prices them lower: the report's
"divisions by a constant" section has each one's derivation, and the
JSON result an `identities` list.  Without a bound on the dividend
nothing is derived.  The translation is checked against the
interpreter on 64 samples; the written module passes spirv-val.

### Dot products

Packed integer dot products are operations of the language, with the
semantics of SPV_KHR_integer_dot_product (src/dot.c).  The operands are
u32 words of lanes; lane k is bits [k*b, (k+1)*b), zero-extended for an
unsigned operand and sign-extended for a signed one:

| op | lanes | operands | result |
|---|---|---|---|
| `udot4x8(a, b)`, `sdot4x8`, `sudot4x8` | 4 x 8 bits (OpUDot/OpSDot/OpSUDot, PackedVectorFormat4x8Bit) | unsigned, signed, a signed x b unsigned | u32 for udot, i32 otherwise: the exact sum's low 32 bits |
| `udot2x16(a, b)`, `sdot2x16`, `sudot2x16` | 2 x 16 bits (the 2-vector forms) | as above | as above (2 x 16 can wrap: 2 * 65535^2 > 2^32) |
| `udot8x4(a, b)`, `sdot8x4`, `sudot8x4` | 8 x 4 bits (AMD's v_dot8; SPIR-V has no form for them) | as above | as above |
| `udot4x8sat(a, b, acc)` ... `sudot8x4sat` | the same | plus an accumulator of the result's type | the exact sum plus acc, saturated to the result's range (OpUDotAccSat ...) |

```
> show sudot4x8(0xff000001, 0x01000002)     lanes 1*2 + (-1)*1
_() = 1
> show udot4x8sat(0xffffffff, 0xffffffff, 0xfffffff0)
_() = 4294967295
> range sdot4x8(a: u8, b: u8)
  -16256..16384
```

They fold, evaluate, have interval bounds (lane by lane from the
operands' ranges) and are bit-blasted for SAT.

**Cost classes.**  One per form: `udot4x8`, `sdot4x8`, `sudot4x8`, the
same for `2x16` and `8x4`, and the `...sat` ones; `dot4x8: 1` (and
`dot2x16`, `dot8x4`, `dot4x8sat` ...) sets a form's three signednesses
at once.  Unlike the other classes, which default to 1, a model that
does not list a form cannot use it: it costs infinity, `models` leaves
it out, `cost` says it is not available, and `all: v` does not set it;
`dot4x8: none` takes a form away from a model it inherits.

| model | forms | source |
|---|---|---|
| gt1030 | dot4x8 1, dot4x8sat 6.5 | measured (tests/gt1030: udot4x8 + iadd 2.14, sdot4x8 + iadd 2.00, udot4x8sat 6.47: DP4A has no saturation, the driver emulates it); sudot4x8 is the same instruction, not measured apart |
| sm75 (Turing) | dot4x8 1, dot4x8sat 6.5 | estimates: the CUDA guide's throughput table for the classes it covers (int 64/clk as fp32, so imul 1; SFU and conversions 16), the rest gt1030's; IDP4A at the IMAD rate, no saturating form |
| sm86 (Ampere GA10x) | dot4x8 2, dot4x8sat 13 | estimates: fp32 128/clk, the integer pipe 64, so every integer class is 2 |
| rdna2 (AMD gfx10.3) | udot4x8, sdot4x8, udot8x4, sdot8x4 1, their sat forms 1 | estimates from AMD's ISA guide: full-rate v_dot4/v_dot8 with a clamp bit; no mixed signedness, no 2 x 16 |
| glc, rpl, zen4 | sudot4x8, sdot2x16 and their sat forms | estimates: vpdpbusd (u8 x s8) and vpdpwssd (s16 x s16) of AVX-VNNI / AVX512-VNNI, priced as movd + the published latency + movd |

**The rewrite.**  Nothing in the search knows a target: each sum in a
function is normalised to sum s_i f_i g_i + c (through +, -, negation,
t << k as t * 2^k, a lone term as t * 1, and conversions that keep the
value mod 2^W or exactly), and its terms are laid into the lanes of the
forms the active model has.  A factor is a lane value when its proved
range fits a lane (the parameters' ranges narrowed by the where clause,
then intervals); a bit field (w >> 8k) & 255, w >> 24, u8(w >> 8k) is
lane k of w already, so the dot takes w and the other side's zero lanes
cancel w's other lanes; other values are packed (constants into a
constant word for free, a value into lane 0 as it is, into lane k by a
shift and an or, a negative one masked when its sign bits would meet a
used lane), each instruction priced.  Groups are chosen greedily,
cheapest first, and a rewrite is kept only when the whole function is
cheaper under the model; a `...sat` form takes the rest of the sum as
its accumulator only when the intervals show nothing can saturate.

Each rewrite is proved: exhaustively or by SAT (the parameters cut to
their ranges' bits); when SAT gives up (a multiplier against a
differently built one), by lanes: each used lane of each word is shown
to be its factor and each unused lane 0 on one side (small circuits
without a multiplier), a dot is by definition the exact sum of its lane
products, which the intervals keep inside its result (no wrap), and the
rest of the sum is the same terms regrouped by the ring laws mod 2^W.
The report says what each lane relies on.

```
> f(x: u32) = (x & 255)*3 + ((x >> 8) & 255)*5 + ((x >> 16) & 255)*7 + (x >> 24)*9
> find f 1                                  (gt1030)
  dot products under gt1030 (sums of products laid into lanes; cost 24 as written):
  cost 1   udot4x8(x, 0x9070503)
           x & 0xff is bits 0..7 of x: lane 0 as it is
           ...
           udot4x8's sum is in 0..6120: it does not wrap
           proved by SAT: 755 AIG nodes, 0 conflicts
> g(a: u32 in 0..255, b: u32 in 0..255, c: u32) = a*b + c
> find g 1
  cost 2   udot4x8(a, b) + c
           a in 0..255 (proved range)
           b in 0..255 (proved range)
```

With a model without dot forms (`model nodot : gt1030 { dot4x8: none,
dot4x8sat: none }`, or x86v3) nothing is rewritten and `find` says so;
under sm86 a single product is no cheaper as a dot (2 + 2 against
2 + 2) and stays; under zen4 the four bytes go into vpdpbusd's signed
side as constants, but `a*b` of two bytes does not (u8 x u8 is no VNNI
form); under rdna2 `a*b + c` with c in 0..1000000 becomes
`udot4x8sat(a, b, c)` (the clamp is free there and cannot act), and
with c a full u32 it stays `udot4x8(a, b) + c`.  Note that ucas types
`a*b` of two u8 as a u8 (it wraps): `h(a: u8, b: u8, c: u32) = a*b + c`
becomes `u32(u8(udot4x8(a, b))) + c`.  And since gt1030 measured DP4A
at 1 and a shift at 2, a product with a small power of two can come out
as a dot too: that is the model speaking.

**SPIR-V and GLSL.**  `animate exec` runs OpUDot, OpSDot, OpSUDot and
their AccSat forms, packed or on vectors, priced by the model's class
(as the multiplies and adds they stand for when it has none).
`simplify` reads each maximal integer sum of a module into a ucas
function (its loads and other free inputs are the parameters, with the
assumed ranges), searches it under the active model's 4 x 8-bit forms,
checks the translation against its own interpreter, and writes the
result back: OpUDot/OpSDot/OpSUDot(AccSat) with PackedVectorFormat4x8Bit
and the shifts and ors that pack, adding `OpCapability DotProduct`,
`DotProductInput4x8BitPacked` and `OpExtension
"SPV_KHR_integer_dot_product"` when the module lacks them; spirv-val
accepts it, and spirv-cross writes the instruction through
GL_EXT_spirv_intrinsics (tests/dot_view.ucas).  "simplify under sm86"
(for, on, model NAME) makes that model the active one first; the JSON
`spirv_simplify` takes `model`.  The function level proposes the same
for a GLSL expression over the values its leaves take at its calls,
with the declaration GLSL needs (GLSL has no built-in for these).
Neither cfront nor the 2 x 16 and 8 x 4 forms write code yet.

### Long expressions

Reports print an expression with its shared subterms named: a subterm
used more than once and larger than a few nodes becomes a binding, `e
where t1 = ..., t2 = ...` (the source's own name from a `where` or a
block assignment when it has one).  A reported expression is cut at
2000 characters (a note at 600, a line of `cost` or `find` at a few
hundred) with "… (N more characters)".  In JSON and MCP responses a list
longer than 500 entries is cut (`KEY_omitted` says how many) and the
`text` field is cut past 64 KiB, keeping the head and the last lines;
when files are written (`--dir`), the full text goes to a file named in
the marker.

## Float simplify

After the integer part, `simplify` reads every float value the module
stores or returns as one scalar function per component (src/spirv.c
builds the graphs, src/fsimp.c does the mathematics).  Components with
the same function (splat constants) are analysed once and the rewrite
applies to every lane; a vec2 or vec3 is just fewer lanes of a vec4, so
one scalar search covers every width.  When the constants differ per
component, each lane is analysed and a rewrite applies only when every
lane allows it (`per component` forces one analysis per lane).

Two equivalence notions are reported for every rewrite:

- **bit-exact** against the reference: IEEE binary32, round to nearest
  even, `pow`/`exp2`/`log2` correctly rounded (long double, then f32),
  NaN payloads ignored, -0 and +0 distinct;
- **within the original's Vulkan precision**: the rewrite's derived
  error is at most what the original was already allowed, using the
  Vulkan 1.3 table (add, sub, mul correctly rounded; `OpFDiv` 2.5 ulp;
  `fma` inherited from a multiply then an add; `exp2` 3 + 2|x| ulp;
  `log2` 3 ulp outside [0.5, 2], absolute 2^-21 inside; `pow`
  inherited from exp2(y * log2 x); `sqrt` from 1/inversesqrt, which is
  2 ulp; `mix` from x*(1 - a) + y*a), and the default float controls
  (without `SignedZeroInfNanPreserve` the sign of a zero, infinities and
  NaNs need not be preserved; the module's execution modes are read).

Rewrites are refinements: where GLSL leaves the result undefined
(`pow` of x < 0 or of 0 with y <= 0, `log2` and `inversesqrt` of
x <= 0, `sqrt` of x < 0), the replacement may give anything.

Every claim is derived, not sampled:

- interval arithmetic on f32 in the total order -inf < ... < -0 < +0 <
  ... < +inf (rounding is monotone, so an interval's image is spanned by
  its corners);
- monotonicity rules (x + K, x * K for K > 0, fma(x, K, C), pow(x, y > 0),
  exp2, log2, sqrt; inversesqrt and negation reverse), so the inputs
  where a value is -0, +inf or undefined are a preimage of an interval:
  found by bisection on the order (32 steps, a solution, not a search);
- the standard model fl(a op b) = (a op b)(1 + d), |d| <= u = 2^-24, with
  the exact cases (Sterbenz, scaling by 2^k, exact reciprocals) and
  |v| < 2^24 ulp(v);
- witnesses of a difference are built from the analysis (a quotient
  just off a rounding midpoint, the end of a domain where an error term
  is largest, x = -A where a sum cancels) and then evaluated.

| rewrite | when | derivation |
|---|---|---|
| mix(a, b, step(e, x)) -> `x >= e or NaN ? b : a` (`OpFUnordGreaterThanEqual` + `OpSelect`) | always offered | step gives 0 or 1 (NaN fails x < e: 1), and a*1 + b*0 is a except for b = +-inf/NaN or a = -0 with b*0 = +0 (and symmetrically); the inputs where that happens are solved by monotonicity, the ones where the original is undefined excused |
| x / d -> x * fl(1/d) | a divide by a constant (spirv-opt -O already does it for GLSL: reported as such) | k d = 1 + delta exactly; bit-exact when delta = 0, else at most 1 ulp from x / d when 2^24 |delta| < 1, and 0.5 + 2^24 |delta| <= 2.5 ulp is within the division's precision |
| (x + A) * K -> fma(x, K1, K2) | an add feeding a constant multiply | K1 = fl(1/d), K2 = fl(A/d) re-optimised against the source's (x + A)/d; no cancellation (x, A of one sign where it is used) gives (1 + 2^24 max(e1, e2)) ulp, plus the add's rounding, against the division's 2.5; bit-exact when K1 is a power of two and K2 = A K1 exactly; with cancellation, a counterexample |
| x * K + B -> fma(x, K, B) | a multiply-add | a contraction Vulkan already allows (no `NoContraction`) |
| pow(q, y) -> exp2(y * log2 q) | q > 0 where it is used | the expression pow's precision is defined by; against the correctly rounded pow, 2^(|t|(2u + u^2))(1 + u) - 1 with |t| = |y log2 q| bounded on q's range |
| sqrt(x) -> x * inversesqrt(x) | x > 0 where it is used | both carry inversesqrt's error; the original divides (2.5 ulp), the rewrite multiplies (0.5 ulp) |
| pow(q, y) ~ P(q - c) | q bounded where it is used | relative minimax (remez.c) with f32 coefficients; its real error certified on 4096 geometric pieces by a second-order Taylor bound with explicit derivative bounds, plus gamma_2n for Horner with fma; compared with pow's allowance piece by piece: listed as approximations unless within |

"Where it is used" matters: after `mix(a, b, step(e, x))` becomes a
select, `b` is only used where x >= e (or NaN), so its pow's base, a
sqrt's operand or an add's sign are analysed on that part only, and the
rewrites that rely on it apply only with the select.  Assumptions narrow
an input (`c in [0, 1]`, `c in (0, 0.5]`, `c in 0..1`; each component,
no NaN; 0 includes -0).

Options: `exact` applies bit-exact rewrites only; `ulp N` applies those
whose derived bound against the reference is at most N ulp (a select is
0 ulp: its finite results are identical); `approx` with `ulp N` also
applies an approximation within N ulp.  Without an option, rewrites
within the original's Vulkan precision apply.  Costs count every
component, with step as a compare and a select and mix as a subtract
and an fma (as lowered).  The report lists the rewrites by savings with
their proofs, what spirv-opt -O already did (with the same
derivation), approximations with their certified errors, near misses
with their counterexamples, and the dead-side note: a select still runs
both sides; a branch would skip one where the lanes agree.

The user's sRGB decode, `vec3 decode(vec3 c) { return mix(c / 12.92,
pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c)); }`:
spirv-opt -O had already made both divisions multiplies (within 0.53
and 0.71 ulp, inside the 2.5 ulp of a division); float simplify makes
the mix a select (bit-exact except c in [-8.4e-45, -0], where the
original gives +0, and c = +inf, where it gives NaN: both allowed by the
default float controls; c < -0.055 is undefined in the original), the
pow `exp2(2.4 * log2 q)` (its own Vulkan definition; 13 ulp from the
correctly rounded pow for c in [0, 1]) and the add and multiply one fma
(2.16 <= 2.5 ulp): 18 -> 14 per component under gt1030.  No polynomial
cheaper than the pow is within its precision on the base's range
[0.0905, 1] (the best, degree 7, is 3.3e-4 relative).

Not done yet: proofs for two or more float inputs that interact (a
second input is kept as an interval, so its sets are "possibly"); bit-
level SAT for small float pieces; joining two vec2 operations into one
vec4 (the gt1030 model prices components, so it would save nothing
there).

## Function level

The instruction-level rewrites see one SPIR-V instruction at a time, and
spirv-opt and the driver have already done the easy ones.  What is left
in a hand-optimised shader is usually a function-level fact: a branch
that the callers never take, a mask that the arguments never need, a
loop iteration that never wins.  For GLSL, `simplify` therefore also
reads the source (the OpSource text `glslangValidator -g` keeps) and
works on whole functions (src/glslfn.c).

**Parsing.** The source goes through ScarletNX's GLSL ES compiler
(src/snxglslc), which gives a typed syntax tree with source positions.
Vulkan GLSL 4.50 is brought to GLSL ES 3.10 first where the two only
spell things differently, keeping every line in place: the `#version`
line, `layout(push_constant)`, `T a[N] = { ... }` (as `T[N](...)`),
`#extension` lines, and `const` on a local (GLSL ES wants a constant
expression there; the word is remembered for the suggestions).  A
declaration or function the parser still refuses is left out, and the
report names it and why.  Integer scalars, arrays and constant tables
are understood; floats, vectors and structs are opaque.

**Domains.** An abstract interpretation runs from `main` (or from every
function nobody calls).  Its state is a set of environments, one per
value of the *key* variables: globals only ever assigned constants or
selects of constants (`sh_ib = mode == 6u ? 4u : mode == 1u ? 3u : 2u`
gives one environment for each of 2, 3, 4), bool locals, and locals
assigned only such selects.  Each variable is an interval with the bits
that may be one (`(w >> 8c) & 0xFF` of a word whose bytes are below 2^n
is below 2^n) and an optional symbolic upper bound (`t <= dd` after
`clamp(x, 0, dd)`).  Conditions narrow the variables they compare; loops
iterate to a fixed point (widened after three passes).  A function with
a loop is run once per value of the keys it reads and of its
single-valued arguments (its summary); a loop-free one is run on each
call's own values.  An expression whose leaves repeat (`a + (b - a) /
16` with a, b in 0..255) is run on every combination of its leaves when
there are at most 65536; one that grows with a leaf bounded by another
(`(t * 64 + dd / 2) / dd` with t <= dd) is run at the bound over every
value of the rest, which gives `t64 <= 64` exactly.  The domains are
over-approximations: what is proved on them holds for every run.

**Searches**, each candidate checked by the interpreter first and then
proved by `equiv_check` (ucas's own `equiv`) on a ucas translation:

- a *pure helper* (scalar parameters and result, writes nothing, reads
  scalar globals) whose domain has at most 2^20 points is checked whole:
  each `if` removed or reduced to one side, each min, max, clamp, mask,
  remainder, select and compare replaced by an operand or a constant,
  the first and last iterations of each constant-bound loop dropped, as
  long as the function's value stays the same at every point of its
  domain; what is kept is proved together;
- elsewhere, each such operation is checked on the values its leaves
  took (recorded during the analysis), and so is the interpolation
  `((2^S - w) a + w b + C) >> S`, which is `a + ((w (b - a) + C) >> S)`
  with one multiply (arithmetic shift; the identity floor((a 2^S + X) /
  2^S) = a + floor(X / 2^S) and the intervals showing no wrap are why,
  the exhaustive check is the proof);
- a pure helper of one wide input and inputs of a few values (cases) is
  fitted by `(x * K + C) >> S` per case: the least S, C = 2^(S-1) first,
  covering the most of each case's domain; K is written as one formula of
  the case when one fits (`A / ((1 << g) - 1)`, `A / (1 << g)`, `A / g`),
  else listed.  Where the form holds on less than the inferred domain
  (BC7's weight tables repeat past 2^ib), what must hold is stated (`i <=
  2^sh_ib - 1`) with the call sites where the analysis shows it and
  those where it does not;
- invariant calls: a call in a loop whose arguments and everything the
  callee reads (globals, arrays, transitively) are not written in the
  loop, and whose callee writes nothing, is the same on every pass; also
  a call one level down (in a function the loop calls, outside that
  function's own loops) whose arguments come from parameters bound to
  such values.  That is a dataflow fact; no proof beyond the read and
  write sets.  Calls costing under 8 are listed in one line.

**The translation.** A function becomes a ucas code block: its
parameters and the globals it reads are the ucas function's parameters
(u32, i32, bool), ranged by the domain's bounding box with a `where`
clause for the union of its rows; locals are typed variables, every
literal and conversion explicit; a call is the callee inlined as a
nested block; loops are `while` loops; `return`, `break` and `continue`
set flags that guard the statements after them.  A domain of more than
2^22 points is proved row by row (a row split on its widest input);
nothing is reported on sampling.

Each proposal says what never bites or changes, the GLSL to use (the
function, or the expression before and after), the proof and its number
of points, the cost under the active model before and after (a loop
priced per iteration times its trip count, a branch by its dearer side),
and what it relies on: the inferred domain and the calls it came from,
so that what must stay true is in view.  The module itself is not
changed by this section.  JSON: `result.function_level`, a list of
`{kind, function, line, what, replacement, proof, points, cost_before,
cost_after, domain, relies_on}` (`kind`: `function`, `expression`,
`case_fit`; `invariant_call` with `cost_per_call` and `passes`).

On a BC7 encoder (ScarletNX's bc7.comp), with every helper already hand
tuned, it finds on its own: `Widen(v, n)`'s `if (n == 8u) return v;` and
`& 0xFFu` dead (n in 5..8, v < 2^n, from Decoded and Quant); `Weight(i)`
= `(i * K + 32u) >> 6u` with K = 4095 / ((1u << sh_ib) - 1u), relying on
i < 2^sh_ib (shown at Eval's calls, not at Fit's, where the index comes
from a packed word); Quant's `for d in -2..2` needing only -1..1;
Eval's `min(..., top)` dead (t64 <= 64); TexelError's interpolation
with one multiply; and `PAllowed(s)` the same on all four passes of
FitSubset's loop.  It does not yet go further with Quant (two of guess
- 1, guess, guess + 1 are left by the parity rules, so a branch-free
comparison of two candidates would do).

## GLSL view

SPIR-V is hard to read, so after `simplify` writes `simplified.spvasm`
it shows what changed as GLSL: the rewritten module is assembled
(`spirv-as`), validated (`spirv-val`) and cross-compiled with
`spirv-cross --vulkan-semantics` at the source's `#version` (ES with
`--es` when the source was ES), into `simplified.glsl`; the module as
`spirv-opt -O` left it, before the rewrites, goes the same way into
`original.glsl`, and the report shows a unified diff of the two (at most
40 lines):

```
  GLSL view (spirv-cross's reconstruction of the SPIR-V, with its own names, not your source):
    --- original.glsl (spirv-opt -O)
    +++ simplified.glsl
    @@ -14,7 +14,7 @@
     void main()
     {
    -    uint _25 = ucas_args.Skip_n;
    -    _5.s.x = (_5.s.x >> _25) | ((_5.s.y << 1u) << (31u - _25));
    -    _5.s.w = _5.s.w >> _25;
    +    uint _26 = ucas_args.Skip_n;
    +    _5.s.x = (_5.s.x >> _26) | (_5.s.y << (32u - _26));
    +    _5.s.w = _5.s.w >> _26;
     }
```

It is spirv-cross's reconstruction of the SPIR-V, not the user's
source: functions spirv-opt inlined are in `main`, temporaries have
spirv-cross's names (`_26`), and an id the rewrites renumbered changes
its name.  The command line writes both files; JSON and MCP put the
diff in the result's `glsl_view` and the GLSL in `glsl`, and list both
files in `files`; the Discord bot attaches `simplified.glsl` next to
`simplified.spvasm` (its engine runs in a sandbox that cannot start
programs, so the bot makes the view, with the same code).  A SPIR-V
snippet completed by ucas leaves its inputs undefined, so spirv-val
rejects it and the view is skipped with a note.

spirv-cross is an external tool, like glslangValidator: ucas looks for
`spirv-cross` on `PATH` (`UCAS_SPIRV_CROSS` names another one, a path or
a name; `none` turns the view off silently).  When it is missing, or a
step fails, a note says so and simplify carries on without the view.
The one used here is FreeBSD's package `spirv-cross-1.4.357.0`
(`pkg install spirv-cross`, Khronos tag `vulkan-sdk-1.4.357.0`, which
pairs with the installed spirv-tools and spirv-headers 1.4.357.0); the
expected outputs of the tests that need it (`tests/glslview.ucas`,
`tests/json/glslview.jsonl`, `tests/mcp/glslview.jsonl`,
`discord/tests/glslview.jsonl`, each with a `.needs` file) were made
with it, and they are skipped where it is not installed.
`tests/glslview_absent.ucas` is the same run without it.

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

### Derivatives, limits and series

`src/calc2.c` (syntax in `src/calc2_syntax.inc`, English in
`src/nlp_calc2.inc`).  Every answer says where it holds and is checked
numerically.

| command | |
|---------|--|
| `diff F [x y ...] [n]`, `diff F wrt x, y`, `partial F ...` | derivatives of any order: `diff F x 3`, mixed `diff F x y` (∂²/∂y∂x, x first).  The variables after F are told apart by spacing (`diff x^2*y x`); a name that is not a variable of F is part of the product (with a note).  F with several variables and none named: the gradient.  `diff F x n` (a literal n): the n-th derivative of exp/sin/cos/log/powers of a linear argument, checked for n = 1..6 |
| `F''`, `F'(a)`, `fx`, `fxy`, `f_x` | a defined function's derivatives; `F'(a)` first checks that the derivative exists at a |
| `grad F [at a, b]`, `hessian F [at ...]`, `laplacian F [at ...]` | the gradient (and its value and length at a point); the Hessian, its determinant, symmetry of the mixed partials checked at sample points, and at a point the second-derivative test (leading minors: minimum, maximum, saddle, inconclusive); the Laplacian, harmonic when it is 0 |
| `jacobian F, G, ... [wrt x, y] [at ...]` | the Jacobian matrix, its determinant when square, invertibility at a point |
| `ddir F along v1, v2 [at ...] [raw]` | ∇F·v/|v| (`raw`: ∇F·v), checked against (f(p + hu) - f(p - hu))/2h |
| `wronskian F, G, ... [wrt x]` | row k holds the k-th derivatives; W simplified, also recognised from its values (a constant, c·exp(ax), a polynomial) and checked against the numeric determinant.  W(x0) != 0: independent.  W = 0 at every sample: a linear relation is fitted to the values; for analytic functions W = 0 means dependence, otherwise (x^2, x·abs(x)) it does not, and the values decide |
| `limit F [as] x -> a[+|-] [from left|right]`, `lim x->a F`, `lim(x -> a) F`, `limit F at a` | a may be `inf`, `-inf`, `oo`, `∞`.  Symbolically: substitution where continuous, L'Hôpital on 0/0 and inf/inf (up to 12 steps, simplified in between), 0·inf as a quotient (both ways), inf - inf by the conjugate (sqrt) or a common quotient, 1^inf/0^0/inf^0 through exp(B log A), leading powers at infinity, exp beating powers, squeeze for 0·bounded, oscillation (sin(1/x)), one-sided limits that differ (the limit does not exist), a side outside the domain (the other side's limit).  Numerically: f at a ± 0.1·2^-k (or x = ±10·2^k), Richardson-extrapolated, with the verdict "agrees" or "DISAGREES" |
| `taylor F [x] [at a] [order n]`, `maclaurin F [order n]` | coefficients from the derivatives, checked against Taylor-jet arithmetic (long double), and |f - P| measured at a ± h for three h (it shrinks like h^(n+1)); refused where a derivative does not exist (sqrt, abs at 0) |
| `implicit L = R [for y] [wrt x] [order 2]` | dy/dx = -F_x/F_y (and d²y/dx²) where F_y != 0, checked on points of the curve found by bisection and Newton |
| `antiderivative F [wrt x]`, `primitive F` | `integrate` without bounds (+ C, differentiated back); bounds are refused (use integrate) |

Where a derivative holds: the conditions of F's partial operations
(`log(u)`: u > 0, `sqrt(u)`: u >= 0, `u/v`: v != 0, ...), solved for x
when linear, minus the points where it fails.  Those are looked for
(one variable, in the parameter's range or [-10, 10]) at the zeros of
the arguments of abs/sign, sqrt and fractional powers, min/max ties,
piece boundaries and the integers of floor/ceil, and each is examined by
one-sided difference quotients: a corner (both slopes given), a vertical
tangent, a jump, an end of the domain.  With several variables they are
listed as "may fail where u = 0".  Derivatives are simplified (like
terms, coefficients, exp(a)exp(b)), never by cancelling a factor that
could be 0; limits do cancel (a limit does not see one point).

`integrate` also knows rational functions completely, substitution,
by parts, trigonometric and square-root integrands (see "Symbolic
algebra").

Not done: limits of several variables, Taylor series of several variables
beyond symbolic coefficients, partial fractions (the transforms carry a
small one, xfrat.c), the n-th derivative of products.

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
- `integrate` knows the methods of "Symbolic algebra"; it is not the
  Risch algorithm, and unknown cases fall back to numbers.
- `dsolve` is symbolic only for linear systems with constant
  coefficients and forcing by polynomials, exponentials and sines; the
  independent variable is the one free name in the equations (`t` if
  none).
- PDEs: one space dimension, linear, constant coefficients; Dirichlet,
  periodic or (for advection) outflow boundaries.

## Transforms, series and sums

```
> laplace t^2 e^-t
L{t^2*e^(-t)}(s) = 2/(s + 1)^3
  region of convergence: Re(s) > -1
  check: matches int_0^inf f(t) e^(-st) dt (Gauss-Kronrod) at s = -1/2, 1/2, 3: ...
> ilaplace 1/(s^2 + 2s + 5)
L^-1{1/(s^2 + 2s + 5)}(t) = e^(-t)*sin(2t)/2
  poles: s = -1 +- 2i
> laplace solve y'' + y = sin(t), y(0) = 0, y'(0) = 1
  Y(s) = (s^2 + 2)/(s^2 + 1)^2
  y(t) = -t*cos(t)/2 + 3sin(t)/2    for t >= 0
> tf y[n] = 0.9 y[n-1] + x[n]
  H(z) = 10z/(10z - 9)
       = 1/(1 - (9/10)z^-1)
  stable (BIBO): every pole inside the unit circle (max |p| = 9/10)
  -3 dB (gain 10/sqrt(2)) at w = 0.105458 rad/sample
    float y = x + 0.9f*y1;
> sum(n, 1, inf, 1/n^2)
  converges absolutely: p-series 1/n^2 with p = 2 > 1
  = pi^2/6
  numeric: 1.644934066848226 (the first 128 terms plus an Euler-Maclaurin tail ...)
> asymptotic sum(k, 1, n, 1/k)
sum(k, 1, n, 1/k) ~ log(n) + gamma + 1/(2n) - 1/(12n^2) + 1/(120n^4) + ...
```

`src/xform.c` (transforms), `src/series.c` (sums and products),
`src/xfrat.c` (shared: exact-looking numbers, the printer, polynomials
and rational functions, partial fractions), English in
`src/nlp_xform.inc`.  `tests/xform.ucas` runs every command.

| command | |
|---------|--|
| `laplace F [t] [s] [period T]` | L{f}(s) = int_0^inf f(t) e^(-st) dt, one-sided.  Table: t^n, t^p (Gamma), e^(at), sin, cos, sinh, cosh (of bt + c); rules: linearity, e^(at) f -> F(s - a), t^n f, product to sum, powers of sin/cos, polynomials expanded, u(t - a) (also `H`, `step`, `heaviside`, `theta`) as e^(-as) shifts, delta(t - a), piecewise `{... if t < 1, ...}` and \|t - a\| as steps, `period T` (F_T(s)/(1 - e^(-Ts))).  Symbolic constants are allowed (`e^(a t)`, `sin(w t)`).  Prints the region of convergence Re(s) > a and the rules used |
| `ilaplace F [s] [t]` | partial fractions (complex pairs as e^(at)(cos, sin), ±r pairs as cosh/sinh when that is shorter), e^(-as) factors as u(t - a) shifts, a constant part as delta(t), (s - a)^(-p) with p not an integer (Gamma), and symbolic c/(s - a)^m and (b s + c)/(A s^2 + B s + C) (with the assumption printed).  The right-sided inverse: Re(s) > the largest pole's real part |
| `laplace solve ODE, ICs` | `y'' + 3y' + 2y = u(t - 1), y(0) = 1, y'(0) = 0` (also `dy/dt`, `d^2y/dt^2`, `y(t)`; conditions joined by `,` `and` `with`): linear, constant coefficients; prints the transformed equation, Y(s) and y(t); missing initial values are 0 with a note |
| `ztrans X [n] [z]` | Z{x}(z) = sum_{n>=0} x[n] z^-n.  a^n, n^k a^n (Eulerian numbers), cos/sin(bn + c), delays u[n - k] and delta[n - k], 1/n! (e^(1/z)), piecewise; the z^-1 form; ROC \|z\| > r |
| `iztrans X [z] [n]` | partial fractions of X(z)/z: p^n times binomials in n, complex pairs as r^n (cos, sin), poles at 0 as delta[n - k]; e^(a/z); causal (ROC outside every pole, stated); the first terms by long division in z^-1 |
| `zsolve REC, ICs` | `y[n+2] = y[n+1] + y[n], y[0] = 0, y[1] = 1` (backward shifts y[n-1] too): Y(z), y[n] |
| `tf EQ` \| `tf H` | the transfer function of a difference equation (`y[n] = 0.9 y[n-1] + x[n]`, input x, output y) or of an ODE (`y'' + 3y' + 2y = x`), or H(z)/H(s) itself: poles, zeros, BIBO stability, causality, impulse response, DC gain, a frequency response table, the -3 dB point, and (discrete) the filter as C in direct form I |
| `fourier series F [x] [on a..b \| on [a, b] \| on a to b \| period P] [terms N]` | default [-pi, pi]; a_n, b_n with closed forms (A + B(-1)^n)/n^k recognised, finite trigonometric series, the first terms, the L2 error of S_N, the midpoint at a jump and the Gibbs overshoot (measured; it tends to 8.95% of the jump) |
| `fourier F [x] [w]`, `ifourier F [w] [x]` | F(w) = int f(x) e^(-iwx) dx (angular frequency, no 1/sqrt(2pi); the inverse has 1/(2pi)).  e^(-ax^2), e^(-a\|x\|), e^(-ax) u(x), 1/(x^2 + a^2), boxes, sin(ax)/x, delta(x - c), 1 and cos/sin as deltas, x^n f (derivatives), cos/sin(bx) f (shifts).  Complex results print as `a + i*b` |
| `dft [x0, x1, ...]`, `fft`, `idft`, `ifft` | a radix-2 FFT when N is a power of two, else the direct sum; `a + bi` entries; the table of X[k], \|X[k]\| and arg; the inverse and Parseval as checks |
| `dtft [x0, ...]` \| `dtft X` | a finite list in closed form; a sequence through its Z transform on the unit circle (refused when the ROC excludes it) |
| `genfunc A [n]`, `egf A` | the ordinary generating function sum a_n x^n = X(1/x) from the Z transform, the power-series library for the rest (Catalan, central binomials), recurrences (`genfunc a[n+1] = 3a[n] + 1, a[0] = 0`) and named sequences (fibonacci, lucas, pell, tribonacci, catalan, squares, cubes, triangular, powers of 2, factorials); the EGF sum a_n x^n/n! for sums of n^k r^n (Stirling numbers), r^n cos/sin, deltas, n! |
| `sum(n, lo, inf, a)`, `sum(n in lo..inf, a)` | an infinite sum (a finite `sum(...)` is evaluated as before): the convergence test that decides, a closed form, an accelerated numeric value, and the two compared |
| `prod(n, lo, inf, a)` | an infinite product: rational factors as Gamma quotients (Wallis: pi/2, prod (1 + 1/n^2) = sinh(pi)/pi), divergence from the factors' expansion |
| `converge A \| sum(...)` | the convergence tests alone, and partial sums |
| `radius A \| sum(...)` | a power series in one variable, `x^n/n!`, `(x - 2)^n/(n 3^n)`: center, radius (ratio or root test), both endpoints tested, the interval, and the sum when the library knows it |
| `powerseries F [x] [at a]` (`coeffs`) | the general term: rational F exactly (partial fractions), others by Taylor coefficients matched against the library; the radius |
| `asymptotic sum(k, 1, n, f)` | Euler-Maclaurin: int f + C + f(n)/2 + B2/2! f'(n) + B4/4! f'''(n), the constant C computed and recognised (gamma, log(2pi)/2) |

Input words: `u(t - a)` and `u[n - k]` are steps, `delta(t)` is Dirac's
delta (a marker that evaluates to 0 except at 0), `delta[n]` the
Kronecker delta; `n!` and `binom(n, k)` (also `choose`, `C(n, k)`) work
in these commands (the helpers `heaviside`, `dirac`, `kron`, `fact`,
`binom` are defined, hidden, on first use).  The variables are guessed
(t, then x, for time; n, k, i for an index) or named after the
expression: `laplace e^(a x) x p`.

How sums are decided, in order: a rational term exactly (the degree
difference: nth-term test, the harmonic series, limit comparison with a
p-series); the nth-term test numerically; the ratio test (a steady ratio,
Richardson-extrapolated; a ratio tending to 0 or infinity is recognised);
the alternating series test with the absolute series tested apart
(conditional convergence); comparison with p-series through the envelope
max \|a_n\| over windows at 10^4, 10^5, 10^6 (so sin(n)/n^2 compares with
1/n^2); the integral test, symbolic (an antiderivative and its limit) or
numeric (over x = e^u, whether the pieces shrink per doubling of u: this
separates 1/(n log n) from 1/(n log^2 n)); the root test.

Closed forms: polynomial times r^n (Eulerian numbers); rational terms by
partial fractions with poles at integers and half-integers: zeta(k),
eta(k) = (1 - 2^(1-k)) zeta(k), Dirichlet beta (pi/4, Catalan's G,
pi^3/32), digamma differences (the first-order residues must cancel);
1/(n^2 + a^2) as pi a coth(pi a); a library of power series (exp, cosh,
sinh, geometric 1/(1 - u)^k, -log(1 - u), atanh/atan, 1/sqrt(1 - 4u),
Catalan, sqrt(1 + u)) matched with a scale and a shift (`sum 2^k/k!` =
e^2 - 1); telescoping g(n) - g(n + k).  Numeric values: geometric-like
terms summed until negligible (Aitken as the check), alternating ones by
the Cohen-Villegas-Zagier transform, the rest by the first terms plus an
Euler-Maclaurin tail (the integral, if need be over x = e^u), or
Richardson on partial sums; each with an error estimate.  A value with no
closed form is matched against constants (pi, gamma, zeta(3), Catalan,
log 2, ...) and reported as "recognised numerically (not proved)".

What is checked: a Laplace transform against the integral computed at
three s right of the abscissa (symbols get sample values, printed); an
inverse by transforming the result back numerically; a Z transform by
summing its series at three z outside the ROC; an inverse Z transform
against long division (32 terms); ODE and recurrence solutions by their
residual and initial values, by RK4 and by iterating the recurrence (the
test file also runs `dsolve` and `closed` on the same problems); Fourier
coefficients' closed forms against the quadrature for n <= 48; a Fourier
transform against the integral (with its truncation error for slowly
decaying f); generating functions by summing the series inside the
radius; sums and products by the accelerated numeric value.

Seams: these commands work on real nodes: s, z and w are real variables,
complex values are computed in C99 `double complex` and printed as
`a + i*b`; the complex type being added elsewhere can take them over.
Every inverse transform goes through `xf_apart()` (xfrat.c), a small
partial-fraction routine with double coefficients and exact rational or
quadratic-surd roots (cnum.c's `poly_roots` on the square-free part); it
is meant to be replaced by alg.c's `apart` once that is merged.

Limits: numeric coefficients for the inverse transforms and partial
fractions (symbolic only for first and second order denominators); no
Laplace rule for t^p times sin/cos with p not an integer, log t, or
general products; Fourier transforms by table (no shift rule for f(x - c)
beyond deltas and boxes, no convolution); the power-series library is
small and matches one family at a time (sums of two families are not
split); asymptotic expansions need an antiderivative of the term;
infinite products are exact for rational factors only; Laurent series
and Mellin or two-sided Laplace transforms are not done.

## Abstract algebra, exactly

```
bigint factor 2^200 + 1
snf [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
abiso Z/4 + Z/6, Z/2 + Z/12
abtor Z/12 + Z, Z/8 + Z/3
abmap Z/12 -> Z/4 by [[1]]
permgroup A5
groebner x + y + z, x*y + y*z + z*x, x*y*z - 1 order lex
idealmember x^3 - 1 in x + y + z, x*y + y*z + z*x, x*y*z - 1
fxmodule [[2, 1, 0], [0, 2, 0], [0, 0, 2]]
```

An exact framework (`src/absalg.h`, the API that the homological
algebra and proof layers build on): no floating point anywhere,
integers that never wrap, and every claimed fact computed with a
certificate that is checked before it is printed (`check:` lines, the
`checks` of a JSON response).

| command | what | certificate |
|---------|------|-------------|
| `bigint EXPR`, `bigint factor N` | integers of any size (`+ - * ^ !`, `/` exact, `div`, `%`, `gcd lcm abs isprime binomial`); factoring by trial division and Pollard-Brent rho | the product of the factors; Miller-Rabin (deterministic below 2^64, probable above) |
| `snf M`, `hnf M` | Smith form U A V = D with U, V, U^-1, V^-1; row Hermite form H = U A | the products multiplied out; U U^-1 = I |
| `zmat [det\|rank\|kernel\|image\|inverse\|cokernel\|solve] M[, b]` | integer linear algebra: saturated kernel bases, lattice bases, A x = b over Z (over Q when Z has no solution) | A K = 0, A x = b, the lattice equality |
| `qmat [rref\|det\|rank\|inverse\|nullspace\|solve] M[, b]` | matrices over Q (`1/2`) | A A^-1 = I, A x = b |
| `abgroup G` | invariant factors, primary decomposition, order, rank, exponent, torsion, generators | the Smith form of the relations |
| `abiso G, H` | isomorphic?, with an explicit isomorphism and its inverse | both well defined, both composites the identity |
| `abhom`, `abtensor`, `abtor`, `abext` `G, H` | Hom, tensor, Tor, Ext, computed from presentations (not from formulas) | `abselftest N` compares them with the cyclic formulas |
| `abmap G -> H by M` | well defined?, kernel, image, cokernel, injective, surjective, inverse | relations to relations; G/ker = im |
| `abelem G, x`, `absub G, x1, x2, ...` | canonical form and order of an element; a subgroup, its quotient and index | n x is a combination of relations; membership witnesses |
| `abexact G1 -> G2 -> G3 by f, g` | exactness at G2 | g f = 0 and preimages of generators of ker g |
| `permgroup G`, `permelem G, p`, `permsub H in G`, `permhom G -> H by ...` | permutation groups by Schreier-Sims: order, base, orbits, derived and lower central series, solvable, nilpotent, centre, membership, normality, cosets, homomorphisms | sifting words that rebuild the permutation; orbit-stabiliser; the graph subgroup has order abs(G) |
| `cayley T` | a group by its Cayley table | all axioms (n^3 associativity) |
| `groebner F [order lex\|grlex\|grevlex] [mod p] [vars x, y]` | reduced Groebner basis over Q or F_p, each element as a combination of the generators | cofactors multiplied out; every S-pair of the basis reduces to 0 |
| `idealmember h in F`, `idealeq F = G` | ideal membership, inclusion, equality | h = sum c_j f_j multiplied out; else the nonzero normal form |
| `fxmodule M`, `pmsnf [[x, 1], [0, x]]` | modules over the PID F[x]: invariant factors of xI - M (minimal and characteristic polynomial, rational canonical form), Smith form of polynomial matrices | U A V = D with U, V inverted |

Groups are written `Z/4 + Z/6`, `Z^2 + Z/3`, `(Z/2)^3`, `Z/2^3` (= Z/8),
`0`, or as a presentation: `coker [[2, 0], [0, 3]]` is Z^rows modulo
the columns.  Permutation groups are `S4`, `A5`, `C6`, `D8` (order 8)
or generators in cycle notation `(1 2 3 4), (1 2)` (1-based).  JSON:
the commands of the same names (`zmatrix`, `qmatrix` for `zmat`,
`qmat`; `bigint_factor`), and the `abstract_algebra` MCP tool.

How it is built (data-oriented, like the rest of ucas):

- **zint** (`zint.c`): a 64-bit handle; values below 2^61 are inline,
  others index one arena of three columns (sign, length, offset) over
  one array of 32-bit limbs.  Zeroed memory is the integer 0.  Knuth's
  division; `z_mark`/`z_release_keep` scope temporaries, and every
  elimination compacts the arena to its live entries after each pivot.
- **zmat** (`zmat.c`): dense row-major handle arrays.  Hermite and Smith
  forms by Euclid across rows and columns (the smallest entry as pivot,
  rounded quotients), applied to the matrix and its transforms and
  their inverses together; Smith starts from Hermite.  A random 30x30
  integer matrix takes 0.03 s.
- **fgab** (`fgab.c`): a group is a subquotient A/B of Z^n (a
  presentation Z^g/R is A = Z^g; homology is cycles/boundaries), A and
  B kept as column-Hermite bases so coordinates are a triangular solve;
  R = B in A-coordinates and its Smith form give the invariants and
  canonical coordinates.  Homomorphisms are coordinate matrices; kernels,
  images, cokernels, sums, meets, m G, G[m], T_p are lattice operations;
  Hom is the lattice of well-defined matrices modulo the zero maps;
  Tor and Ext come from the free resolution 0 -> Z^b -R-> Z^a.
- **pgrp** (`pgrp.c`): deterministic Schreier-Sims (every Schreier
  generator sifted), explicit transversals in one flat array per level.
  The centre and centralisers enumerate (groups up to 10^5 elements in
  the command).
- **ideal** (`ideal.c`): polynomials as SoA term arrays (exponents,
  numerator and denominator columns); Buchberger with the normal
  strategy, the product and chain criteria, cofactors carried through
  every S-polynomial and reduction.

Limits: factoring above 2^64 is Pollard rho only (a composite with two
large factors may stay unsplit, and is reported as such); polynomial
arithmetic is schoolbook; Groebner bases use Buchberger, not F4; the
centre of a permutation group is found by enumeration.  English covers
only a few phrasings: "is A5 solvable?", "is Z/4 + Z/6 isomorphic to
Z/2 + Z/12?", "smith normal form of [[...]]", "groebner basis of ...",
"is h in the ideal generated by ...?", "tensor product of G and H",
"order of S5".

## Homological algebra

```
cohomology RP^4                     # Z, 0, Z/2, 0, Z/2
cohomology RP^4 with Z/2            # Z/2 in every degree
homology K_s generators             # the 9-vertex Klein bottle, with cycles
les L(4) mod 2                      # 0 -> C -2-> C -> C (x) Z/2 -> 0, delta by the snake lemma
bockstein L(9) mod 3                # pages E_r, D_r, d_r until stable
bockstein L(9) mod 3 trace 2        # j_r(3^(r-1) a): in ker d_r? in im d_r?
prop01 S^2 x RP^3 mod 2             # Proposition 0.1 of bockstein.pdf, every n and r
```

Exact homology over Z on top of the abstract algebra core (`src/homalg.h`,
built on `absalg.h`'s zint, zmat and fgab): every group is an fgab
subquotient of the cochains, so an element is the same integer vector on
every page of a spectral sequence, and every claim is checked before it
is printed (the `check:` line counts each kind; JSON `checks`).

| command | what | checked |
|---------|------|---------|
| `space X` | cells per dimension, Euler characteristic, small boundaries (not reduced) | d o d = 0 |
| `homology X [with G] [generators]` | H_n(X; G), G any f.g. abelian group (`Z/2`, `Z/4 + Z`) | universal coefficients: H_n(X; G) = H_n (x) G + Tor(H_{n-1}, G) |
| `cohomology X [with G] [generators]` | H^n(X; G) = H^n(Hom(C_*, Z) (x) G) | both UCTs: Hom(H_n, G) + Ext(H_{n-1}, G), and H^n (x) G + Tor(H^{n+1}, G) |
| `les X mod m` | the long exact sequence of 0 -> C^* -m-> C^* -> C^* (x) Z/m -> 0, maps in canonical generators | short exactness per degree, each snake-lemma lift solved and verified, exactness at every spot (with 0 at both ends) |
| `bockstein X mod p [pages N]` | the Bockstein spectral sequence: D_r, E_r, d_r per page, the stable page, E_inf | each couple exact, i' j' k' well defined (and j' independent of the preimage), im d <= ker d, the stable page has d = 0 and the next page equal |
| `bockstein X mod p trace n [a] [page r]` | for a in H^n(X; Z) (canonical coordinates; default each generator): j_r(p^(r-1) a) in E_r^n, in ker d_r?, in im d_r^(n-1)? (with a preimage) | the preimage mapped back |
| `prop01 X mod p` (= `bockstein X mod p check`) | Proposition 0.1 for every n and every r | see below |

Spaces: `S^n`, `T^n`, `RP^n`, `CP^n`, `K` (Klein bottle), `L(m)` or
`L(m,q)` (lens spaces; write `,` since `;` ends a ucas statement), `pt`,
cellular by default; `_s` for a simplicial model: `S^n_s` (the boundary
of the (n+1)-simplex), `T^2_s` (Moebius' 7 vertices), `RP^2_s` (6
vertices), `RP^n_s` for n <= 4 (the barycentric subdivision of the
cross-polytope's boundary modulo -1, checked to be a 2-to-1 covering),
`K_s` (a 3x3 grid, checked not to collapse).  Constructions: `X x Y`
(product cells, Koszul signs), `X v Y` (wedge at the first 0-cells),
`susp(X)`; your own: `cells([[0, 0]], [[0], [2]])` gives the boundary
matrices d_1, d_2, ... (rows: the (k-1)-cells; `cells(1, 1, 1: [],
[[3]])` with counts when a matrix is empty) and `facets([[0,1,2],
[0,2,3], [3,4]])` a simplicial complex.  JSON: `space homology
cohomology les bockstein bockstein_trace prop01`; MCP tool `homological`.

**Proposition 0.1, checked completely on a space.**  For a in H = H^n(X;
Z) let psi_r(a) = [j_r(p^(r-1) a)] in E_r^n / im d_r^(n-1).  a generates
a Z summand of H iff its class in F = H/T(H) is primitive; psi_r(a) = 0
iff a lies in K = ker psi_r; and a sublattice of F contains a primitive
vector iff the gcd of its coordinates is 1 (in a Smith basis it is d_1
Z + ...).  So "abar is not in im d_r for every such a" is exactly
"content(image of K in F) != 1", one finite computation; "abar is in ker
d_r" is d_r psi_r = 0 on all of H.  This is done for every n and every r
up to the first page where i_r is injective in every degree.  There k_r
= 0 (im k = ker i), so d_r = 0, and i_r with the identity on E_r is an
isomorphism of exact couples onto the next page with j_{r+1}(p^r a) =
j_r(p^(r-1) a); so every later page, and every abar, is the one already
checked (the next page is also computed and compared).  The general
proof (all complexes, all p, n, r) is the proof layer's (`prove.h`);
this is its concrete oracle.

For programs (`src/homalg.h`): `hs_*` build spaces (SoA simplices,
sparse boundary triplets); `hcx_*` cochain complexes of presented groups
Z^g/R (so C (x) Z/m is a complex like any other), `hcx_cohomology` an
fgab; `hmap_*` chain maps and their induced maps; `hles_compute` any
short exact sequence of complexes to its long exact sequence;
`struct hcouple` an exact couple with degrees (di, dj, dk), `hc_derive`
its derived couple, `hc_stable`; `struct hss` the pages of a spectral
sequence, `hss_bockstein`, `hss_trace`, `hss_prop01`.

**Reduction first.**  Every command except `space` works on a reduced
complex: each boundary entry +-1 cancels its pair of cells (an elementary
collapse; algebraic Morse theory), sparsely, until none is left, keeping
the chain homotopy equivalence g: C' -> C, f: C -> C', h with f g = 1 and
1 - g f = d h + h d.  All four identities (f and g chain maps, f g = 1,
the homotopy) are checked on every cell of the original complex before
anything is computed, and the output says `reduced: cells 40 232 384 192
-> 1 1 1 1`.  So H^*(C') is H^*(X) with f^*, g^* inverse to each other,
and cocycles and cycles are printed back on the space's own cells (f^*
for cochains, g for chains: `generators`, the cocycle of a trace).  The
Bockstein sequence 0 -> C' -p-> C' -> C' (x) Z/p -> 0 is built on the
reduced cochain complex, which is still free of finite type: the
statement of Proposition 0.1 is about H^*(X; Z) and H^*(X; Z/p), which
the checked equivalence identifies with those of C' (and, since f is a
chain map of free complexes, with the whole exact couple: the couple is
natural).  `unreduced` (JSON `"unreduced": true`) uses the space's own
cells.

| (FreeBSD, -O2) | unreduced | reduced |
|---|---|---|
| `homology RP^3_s` (40 232 384 192 cells) | 2.0 s | < 0.01 s |
| `cohomology RP^3_s with Z/2` | 16.6 s | < 0.01 s |
| `bockstein RP^3_s mod 2`, `prop01 RP^3_s mod 2` | > 10 min (stopped) | < 0.01 s each |
| `bockstein`, `prop01`, `cohomology` of `RP^4_s` (121 1320 4080 4800 1920 cells) | not tried | 0.8 s for the three |
| `tests/homalg.ucas` (about 70 statements) | | 1 s |

Limits: the reduction cancels unit pivots only, so a complex whose
remaining boundaries have no +-1 entry (RP^n's cellular 2s) stays as it
is (they are small); no cup products (so no Sq^1 = d_1 identification
beyond the numbers); only cellular models for lens spaces and CP^n.

## Elliptic curves and Diophantine equations

```
ec [0,0,1,-1,0]                       # the summary of 37a1
ec y^2 + y = x^3 - x^2 - 10*x - 20 local
ec [0,0,1,-1,0] mul 7 (0,0)
diophantine a/(b+c) + b/(a+c) + c/(a+b) = 4 positive
congruent 157
cubesum 7 positive
mordell 17 bound 6000
```

`src/ec.h` is the shared core (the finite-field side, branch `elip-ff`,
builds on it): a field behind a table of operations with elements as
fixed-size values (`fe`): Q with `zint` rationals, F_p for any p, Z/n (a
failed inversion yields a factor), GF(p^k) through `ring.h`, and room for
other implementations; curves in general Weierstrass form with b2..b8,
c4, c6, the discriminant and j; points and the chord-and-tangent law; a
struct-of-arrays point table; the isomorphisms [u, r, s, t] with
composition and inverse; models (short, Montgomery, Edwards, twisted
Edwards, Legendre, twisted Hessian) with their maps to and from
Weierstrass form in both directions; plane cubics with a rational flex
to Weierstrass form.

Over Q (`ecq.c`): integral and global minimal models (Laska-Kraus-
Connell), Tate's algorithm at every bad prime (Kodaira symbol, c_p,
conductor exponent, split or non-split), torsion (bounded by gcd
#E(F_p), found by Nagell-Lutz, checked against Mazur's list), naive and
canonical heights (Tate's series at infinity, Silverman's formulas at the
bad primes; normalised as h(x(nP))/n^2, the BSD regulator's convention),
a sieved point search, 2-isogeny descent (local solubility of the
quartics over Q_p and R, both Selmer groups, global points on the
quartics mapped back through the dual isogeny) giving proved rank
bounds, independence by the height pairing, the lattice enlarged by any
found point outside it (Hermite form), saturation at p <= 7 (by
reduction, and at 2 exactly by the descent map when the 2-torsion is
rational).

The analytic side (`ecan.c`, floating point with the bound stated):
periods by AGM, Weierstrass p by q-series (checked against p'^2 = 4p^3 -
g2 p - g3), a_n, the root number from the functional equation (the
mismatch for both signs is printed: a wrong conductor or a_p would show
there), L(E, 1) up to L'''(E, 1), the analytic rank, and #Sha_an =
L^(r)(1)/r! #T^2 / (Omega R prod c_p).  With analytic rank 0 or 1 the
rank is a theorem (Gross-Zagier, Kolyvagin).  On 80 random curves the
functional-equation mismatch stayed below 1e-9 and #Sha_an came out an
integer for every curve whose generators were found.

`diophantine` (`ecdio.c`) parses a rational equation in up to six
variables (denominators cleared, their spurious factors divided out and
remembered: a solution must not make one vanish) and recognises a plane
cubic (homogeneous in 3 variables, or affine in 2 and homogenised), a
curve quadratic in one variable with a quartic discriminant (s = 2Ay +
B, s^2 = B^2 - 4AC), or z^2 = a quartic form.  It finds a rational
point; from a flex, the tangent goes to the line at infinity; otherwise
the pencil of lines through the point turns the cubic into w^2 =
quartic(t) with a known point (the tangent direction), and Washington's
map (or the root case) gives Weierstrass form.  Each step is data with a
forward and a backward map; then the minimal model and the group.
Combinations sum n_i P_i + T are taken in order of height, mapped back,
the composed maps checked to return the same point, the side conditions
(`positive`, `integer`, `nonzero`) applied, and the solution substituted
into the original equation exactly.  With no solution it says why when
it can: E(Q) finite (rank 0 by descent) and every point mapped; or, for
positivity on a cubic, the positive real points lie on the bounded
component of E(R) and the 2-isogeny Selmer group at the largest
2-torsion point is all positive, so no rational point lies there.

For a/(b+c) + b/(a+c) + c/(a+b) = N: N = 4 gives the 80-digit solution
(from -9P + T on y^2 + xy + y = x^3 - 234x + 1352, conductor 910), N = 6
a 134-digit and N = 10 a 190-digit one, N = 2 gives (3, 1, 1); N = 1, 3,
5, 7, 9, 11, 13, 15, 17 have rank 0 (no solution, proved), and N = 19,
21, 23, 29, 35, 39 have rank 1 but no positive solution, proved by the
component argument.  `x^4 + y^4 = z^2 nonzero` reproves Fermat's case.
`congruent n` gives a triangle (checked: a^2 + b^2 = c^2, ab/2 = n) or a
proof that none exists (rank 0), else the bounds and Tunnell's counts.
`cubesum k` solves x^3 + y^3 = k z^3; `mordell k` gives the summary and
the integral points of y^2 = x^3 + k up to a bound.

JSON: `elliptic` (curve, op, args), `diophantine` (equation, positive,
integer, nonzero, count, maxmult, search), `weierstrass`, `congruent`,
`mordell`, `cubesum`; the MCP tool `elliptic`.  Tests: `tests/elip.ucas`,
`tests/json/elip.jsonl`, `tests/mcp/elliptic.jsonl`.

Not done yet: a full 2-descent for curves without rational 2-torsion
(their rank's upper bound is the analytic rank, a theorem only up to
rank 1), second descents (`congruent 157`, whose generator is huge, is
left open with its bounds), integral points beyond a search bound (no
elliptic logarithms or Baker bounds), saturation only at p <= 7 and
without an index bound, Hessian models from Weierstrass form (they need
a 3-torsion point), theta-congruent numbers.

## Elliptic curves over finite fields

```
> ecorder y^2 = x^3 + 2x + 3 mod 97
#E = 100
  trace: t = p + 1 - #E = -2
  how: naive: 1 + sum over x of (1 + (x^3 + ax + b | p)), with a table of squares
  check: Hasse: |t| <= 2 sqrt(p)
  check: (73, 83) has order 10 (N P = O, and (N/q) P != O stepwise for each prime q of N)
  check: (46, 72) has order 25 ...
  check: N = 100 is the only multiple of 50 in the Hasse interval [79, 117]
> ecpp 2^255 - 19
57896...19949 is prime
  how: Atkin-Morain ECPP, 6 steps down to 3425690484314565209
> ecm 2^128 + 1
340282366920938463463374607431768211457 = 59649589127497217 * 5704689200685129054721
> ecvectors
  check: RFC 8032 7.1 test 3 (ed25519): pass        (30 of 30)
```

Curves over F_p (fixed-width Montgomery arithmetic up to 521 bits),
F_{p^k} (k <= 12) and binary fields GF(2^m), with their own arithmetic
(`src/ecff.h`).  A curve is a name (`secp256k1`, `P-256`, `P-384`,
`P-521`, `Curve25519`, `Ed25519`, `Curve448`, `BN254`, `BLS12-381`,
`K-163`, `B-163`, `K-233`), an equation (`y^2 = x^3 + 2x + 3 mod 97`, also
`over F_97`, `over GF(97)`; a general Weierstrass equation is moved to
short form and the map printed) or coefficients (`[2, 3] mod 97`); a
point is `(x, y)`, `G`, `O` or `random`.  Every answer prints how it was
found and `check:` lines; JSON fields come from its `label: value` lines.

| command | |
|---------|--|
| `ec C` | the curve: discriminant, j, order (named curves: standard), trace, embedding degree, supersingular / anomalous, twist |
| `ecorder C [method naive\|bsgs\|schoof\|cm]` | #E: naive, baby-step giant-step with Mestre, CM for j = 0 or 1728, Schoof; certified by point orders (the only multiple of the exponent in the Hasse interval) |
| `ecpoints C`, `ecadd C P Q`, `ecmul C k P` | points, the group law, scalar multiples (group structure Z/m x Z/mn by the Weil pairing) |
| `ecdh`, `ecdsa`, `x25519`, `x448`, `ed25519`, `eccompress`, `ecdecompress`, `ecvalidate` | the protocols on the named curves; RFC 6979 nonces for ECDSA |
| `ecvectors` | the known-answer tests: SHA-2 (FIPS), RFC 6979 (P-256), RFC 5903, RFC 7748 (X25519, X448), RFC 8032 (Ed25519); ECDSA on secp256k1, P-384, P-521 cross-checked against OpenSSL |
| `ecdlog C Q P`, `ecweak anomalous\|supersingular BITS` | discrete logs: Pohlig-Hellman, BSGS, rho (orders up to 64 bits), MOV, Smart's anomalous attack; weak curves to try them on |
| `ecpairing C` | Weil and reduced Tate pairings (Miller), including BN254 and BLS12-381 in F_p^12, with non-degeneracy, e^r = 1 and bilinearity checked |
| `hilbert D`, `eccm D BITS`, `modpoly l`, `ecisogeny C l`, `ecvolcano C l`, `ecdivpoly C n` | Hilbert class polynomials (coefficients up to 1152 bits), prime-order CM curves, modular polynomials, Velu and Kohel isogenies, isogeny graphs, division polynomials |
| `ecb2 C [mul k]` | binary curves over GF(2^m): Koblitz counting, FIPS parameters checked (non-Koblitz counted only for m <= 24) |
| `ecm N`, `ecpp N` | Lenstra's ECM (stage 1 + 2); Atkin-Morain ECPP with an independent checker of the certificate |

Append `timed` to any of them for timings.  MCP tool `elliptic_ff`.  This
is variable-time mathematics, not hardened cryptography: use it to check
and explore, not to protect secrets.

Limits: no SEA (no Elkies or Atkin primes), so Schoof takes about 40 s at
160 bits and random 256-bit curves are impractical (named and CM curves
are fine); no Ed448; rho only for orders up to 64 bits; the `ec.h`
`ECF_USER` adapter for these fields is not done.

## C frontend

`c optimize` takes C99 (a file or a ```` ```c ```` block) and writes the
same file with the functions it could prove faster forms for rewritten,
for one x86-64 target.  Everything else stays byte for byte.

```
> c optimize tests/cfront/k1.c nofile
c optimize: 4 functions, 4 rewrites proved (1 exhaustive, 2 by SAT, 1 by algebra) for x86-64-v3
  popcount_loop (line 4): popcount(x)
    proof: sat (proved by SAT: 2017 AIG nodes, 0 conflicts); cost 94.0 -> 1.0
  classify (line 12): switch on c: a range check and the bit mask 0x1104111
    proof: sat (proved by SAT: 586 AIG nodes, 57 conflicts); cost 25.5 -> 6.0
  ...
> c optimize tests/cfront/mem.c nofile
  brighten (line 8): store 0: a table of 256 values indexed by src[i]
    proof: sat (proved by SAT: 2960 AIG nodes, 349 conflicts; per stored value, over the loads); cost 9.8 -> 4.2 per element (loads and stores kept, in order)
  my_strlen (line 34): a scan for the terminating zero: n += strlen(s + n)
    proof: structural (C99 7.21.6.3: strlen counts the characters before the first null; ...)
```

| command | |
|---|---|
| `c optimize FILE\|block [target T] [budget S] [assume NAME LO HI] [wrapv] [check] [bench] [vectorize] [nofile] [timed] [trace] [vecstub] [keep]` | rewrite; `FILE.opt.c` is written unless `nofile` (or a block) |
| `c lower FILE\|block [wrapv]` | each function as one ucas value of its parameters, with C's UB as a domain; for the others, each counted loop's IR |
| `c targets` | the targets and their instruction-set features |

Options: `target` (default `x86-64-v3`; `zen3`, `zen4`, `skylake`,
`skylake-avx512`, `icelake`, `alderlake`, `raptorlake`,
`sapphirerapids`, `x86-64`, `-v2`, `-v4`; it picks the costs and the
intrinsics the output may use under feature macros); `budget` (CPU
seconds for proofs, 10); `assume x 0 1000` (a contract on an input,
reported with each rewrite that uses it); `wrapv` (signed overflow
wraps: no UB is exploited); `check` (compile both versions with the host
cc and compare them on edge and random inputs, inputs on which the
original has UB skipped by UBSan); `bench` (also time both with rdtsc);
`timed` (print the seconds); `trace` (every candidate and why it was not
taken); `vecstub` (the stub vectoriser, below); `keep` (keep `check`'s
build directory); `engines all|none|find,gates,...` (which of ucas's
engines generate candidates, below; all by default), `ebudget S` (their
CPU seconds in total; default `budget`), `steps N` (sketches the
synthesis may try per value; default 1000), `approx` (apply
approximations with a derived error bound; by default they are proposed).  JSON `c_optimize`, `c_lower`, `c_targets` (an input
range is `range`: `"x 0 1000"`); MCP tool `c`.

**In English, and "SAT simplify" on C.**  Any of simplify's phrasings
before a C block runs `c optimize` on it: "SAT simplify", "simplify
this", "optimize this C for zen4 and test it", "peephole with smt",
"make it faster".  The options that mean the same carry over: `budget
20`, a target named (`for zen4`), ranges (`n in [0, 32)`, `n in 0..31`,
`n < 32` become `assume n 0 31`), `check` ("test it", "verify"),
`bench`, `wrapv`, `trace`, `nofile`; float options are SPIR-V's only.
"lower this C" runs `c lower`.  A bare block with no words is not run,
as a bare SPIR-V or GLSL block is not.

````
SAT simplify
```c
unsigned shr(unsigned x, unsigned n) { return n < 32 ? x >> n : 0; }
```
make it faster for zen4 with n in [0, 32)
```c
...
```
````

Which language a block is (`src/codelang.c`): the tag decides when there
is one: ```` ```glsl ```` (and `comp`, `frag`, `vert` ...) is GLSL,
```` ```c ```` / ```` ```h ```` is C, ```` ```spv ```` is SPIR-V.  A
```` ```c ```` block with GLSL's marks (`uint`, `vec3`, `layout(`,
`gl_`, `#version`) and none of C's is GLSL tagged for its colours, and a
note says so.  An untagged block is read from its content,
conservatively: marks only C has (`#include`, `unsigned`, `uint32_t`,
`char`, `static`, `->`, pointer declarations) make it C, marks only GLSL
has make it GLSL; with both or neither (`int twice(int x) { return x +
x; }`) it is GLSL, as before, with a note to tag it ```` ```c ```` for
the C frontend.  In Discord a C block's rewritten code comes back as an
attached `simplified.c` (with a preview), and an attached `.c`/`.h`
file, or a text file that is C by its content, is read as a C block.
Over JSON, `english` takes the block as `code` (and `lang`), and
`spirv_simplify` given C (`"lang": "c"`, or C by its content) runs it
through the C frontend as `c optimize` with its `budget`, `ranges` and
`nofile` (the result is a `c_optimize` one; a note points to
`c_optimize` for a target and `check`).

**Pipeline** (`src/cfront.h`, `cf_*.c`):

1. Preprocess (macros, `#if`, the user's includes; system headers are
   built-in declarations) and parse C99 with LP64 types.
2. Lower each function to one node value of its parameters by a
   predicated symbolic execution: promotions and the usual conversions
   explicit, unsigned wrap, signed overflow, division by zero,
   `INT_MIN / -1`, oversized shifts and float-to-integer conversions
   out of range as a domain condition (C UB), `>>` of a negative value
   arithmetic and narrowing conversions modulo 2^N (GCC's and clang's
   implementation-defined choices).  `float` and `double` are IEEE
   binary32/64, each operation rounded in its own type (x86-64 SSE:
   `FLT_EVAL_METHOD` 0).  Loops with a constant trip count unroll (up to
   66 iterations); other loops leave the obligation "still running after
   66 iterations", which must be proved false.  Calls of functions of the
   same file are inlined.
3. Generate candidates and keep the cheapest proved one under the
   target's costs (the x86 side's scheduler replaces the default costs
   when it is built in): bit idioms (popcount, clz, ctz, bit length,
   x & -x, parity, abs), a switch as a range check plus a table, a bit
   mask or a line, small-input tables, polynomials of a bounded input,
   dropped operations that never bite (masks, clamps, `?:` arms,
   conversions), divisions narrowed or made unsigned or turned into a
   multiply and shift on a bounded dividend, and IEEE rewrites.
4. Emit C with no UB of its own (wrapping arithmetic through unsigned
   types, a signed division only by a constant other than -1, `?:`,
   `&&`, `||` kept lazy, partial operations never hoisted) and
   feature-guarded intrinsics with portable fallbacks; only the replaced
   spans change.  Float output gets `#pragma STDC FP_CONTRACT OFF`: the
   proofs are about each operation rounded on its own.
5. Optionally, `check` and `bench`.  A rewrite that compiles to the same
   instructions as `clang -O2` makes of the original is listed with
   `[same code as clang -O2]`.

**Proofs.**  Each rewrite says which:

| proof | what was shown |
|---|---|
| exhaustive | every input of the (possibly assumed or call-site) domain evaluated, both sides |
| sat | the miter bit-blasted, the SAT solver found it unsatisfiable |
| algebra | a stated lemma: a derivation (IEEE rewrites below) or an interval argument (a 64-bit division whose operands fit 32 bits, a magic-number division) when SAT gives up; the sample is then only a check of the lemma's premises |
| induction | a loop summed in closed form (below), then the rest proved as above |
| structural | a rewrite of the program's structure: a strlen scan, a loop rebuilt from its IR |
| lanes | the vectoriser (below and "x86-64"): every lane of every stored vector proved equal to the scalar value of its iteration, from the intrinsics' lane semantics; applied only with `bench` (where measured faster than clang) or `vectorize` |
| tested | reserved for opted-in, only tested rewrites (none are applied now) |

A static function whose address does not escape is entered only from
this file's calls, so its parameters range over what the calls pass
(bounded by intervals over each argument's own inputs); its rewrites are
proved over that domain and say so (`over: v in [0, 510] (from the
call sites)`).

**Loops in closed form.**  `for (i = lo; i < hi; i++) { s += P(i); t -=
Q(i); }`, with P and Q polynomials of degree at most 2 in i over the
loop's invariants and `hi` read once (no side effects, nothing the loop
writes), becomes s + d0 T + d1 C(T,2) + d2 C(T,3), T = max(0, hi - lo),
d_k the forward differences of P at lo.  The derivation is the proof:
P(lo + j) = sum_k d_k C(j, k) in any commutative ring when the third
difference vanishes (Newton's forward formula), sum_{j<T} C(j, k) =
C(T, k+1) (the hockey stick), and C(T, k) is computed exactly mod 2^64
by halving the even one of T and T - 1 and multiplying by 3^-1 mod
2^64.  The index must be at least `int`, compared in its own type, and
step by 1, so it never wraps; the summand may only use `+ - *` and
conversions that keep its value mod 2^w (w the accumulator's width).
Signed overflow inside the sum is not a domain condition there (the
closed form is defined where the loop is; that is a refinement).
`clang -O2` already turns such simple sums into closed forms itself
(SCEV), so this mostly matters for what follows it (the closed form is
then optimised further) and for compilers without it.

**Functions with memory.**  Pointers and stores are not lowered into a
function value; their loops are handled one iteration at a time
(`cf_lower_iter`: the variables an iteration reads before writing and
its loads are inputs, its stores recorded in order):

- *element-wise bodies*: a loop whose body only stores (no other
  variable written, no store under a condition), each stored value a
  function of what the same iteration loaded, has each stored value
  optimised and proved as a function of those loads (`brighten`: a
  256-byte table).  Loads and stores keep their addresses and their
  order: a load that a later store now uses is read into a temporary at
  its original place.  So the rewrite holds whatever the arrays alias,
  and the loop's control is not touched.  A rewrite that leaves a table
  lookup indexed by data or a helper call in a loop the vectoriser can
  prove (so clang vectorises the original as written) is "proposed, not
  applied" by default: `brighten`'s table runs 0.39 -> 0.80 and
  `invert_low`'s 0.05 -> 0.80 cycles per element against clang's
  vectorised loop (i5-9400; brighten's timed by hand, invert_low's by
  `bench`).  `bench` measures it like the vector loops (below)
  and keeps it where it wins; rewrites that keep the body vectorisable
  (an unsigned division, `x * 0.5`), and table rewrites of loops the
  vectoriser cannot prove, are applied as before.
- *strlen scans*: `while (s[n]) n++;`, `while (*p != '\0') ++p;` and the
  same as a `for` with an empty body become `n += strlen(s + n)` and
  `p += strlen(p)` (C99 7.21.6.3), for `char`, `signed char` and
  `unsigned char` data and a 64-bit or signed counter (an unsigned
  32-bit counter could wrap and rescan; it is left alone).
- *the vectoriser's IR*: a counted `for` (`i = lo; i < hi; i += c`)
  becomes a `struct cf_loop`: index, bounds, each array with its alias
  fact (`restrict`, or two distinct arrays: "no alias"), loads and
  stores with their element index, type and order, reductions
  `acc = next(acc, ...)`.  `c lower` prints it.  The x86 vectoriser
  (`src/cfx86.c`, README "x86-64") is offered every such loop and
  proposes intrinsics when it can prove every lane (applied with `bench`
  where they beat clang, or with `vectorize`), or says why not; `vecstub` registers a stub that rebuilds the loop from the IR as
  scalar C instead, so `check` validates the extraction.
  Whether a load is from before or after a store is part of the IR, so
  aliasing a vectoriser cannot rule out is visible to it.

**Floats.**  Float-valued results are rewritten only by derivations
(the proof engines compare floats by value, not by bits), and a bitwise
sample checks each derivation's premises: `float(double(x) op
double(y)) = x op y` for `+ - * /` when both are float values (double
rounding is innocuous since 53 >= 2 * 24 + 2; so `x * 0.5` with a float
x is a float multiply), `x / 2^k = x * 2^-k` when 2^-k is representable,
float to double and back, an integer through a float that holds it
exactly and back, and comparisons of such integers.  Integer-valued
functions with float inside are proved exhaustively or by SAT as usual
when their inputs allow (`(int)(float)c` for a byte `c` is `c`).

Measured on this host (i5-9400, Skylake; cycles per call through a
non-inlined wrapper, about 5 of which is the call; median of 15 runs,
other builds were running):

| function | before | after | |
|---|---|---|---|
| popcount loop (k1) | 21.7 | 5.5 | x3.9 |
| clz loop | 24.8 | 5.6 | x4.4 |
| lowest set bit loop | 27.5 | 5.6 | x4.9 |
| parity loop | 42.9 | 5.7 | x7.5 |
| `tri` (capped loop -> table) | 8.5 | 5.9 | x1.4 |
| `table_month` (dropped mask) | 6.7 | 5.8 | x1.2 |
| `use2` (static callee over its call sites -> table) | 6.6 | 5.7 | x1.2 |
| `classify` (switch -> bit mask) | 5.7 | 6.3 | x0.9 |
| `ilog2_loop` (31 - clz) | 5.9 | 6.3 | x0.9 |

The last two are slower: clang already makes a lookup of the switch and
recognises the shift loop, and the proved form costs less only under the
default cost model.  `check` agreed on every call (k1: 80000 calls; k2:
379687 calls, 313 inputs skipped for UB in the original; k3, mem.c).
`bench` times functions that take pointers over 4096-element buffers
(lengths 4096) and the original without the UBSan checks `check` adds
(they kept clang from vectorising it), in 3 runs of the harness, each
the median of 15; the harness does not call
functions with float parameters (the vectoriser's float loops were
checked by hand, "x86-64").

Not lowered (the code stays as written, and the report says why): K&R
definitions, `_Complex`, `long double`, `asm`, `goto`, case labels
inside nested statements (Duff's device), VLAs, bit-field layout,
`volatile`, calls to functions outside the file, and more than 8 inputs
per value.  Loops that are neither unrollable, nor sums of the form
above, nor element-wise or strlen scans keep their code (a reduction
such as `s += a[i] & 1` is in the IR but not rewritten here); stores
under a condition, pointers that move, `break` and `return` inside a
loop stop the iteration lowering.

Limits: a `while` loop whose proved form is long (more than 96
operations unrolled) is kept, and SAT may give up on a while loop's
equivalence (a `while (x)` popcount is sampled, not proved, and kept);
induction over loop invariants is not done.  Closed forms stop at
degree 7 and at geometric sums with a constant ratio (see "Engines").
Float rewrites are the listed exact identities; there is no
`fma` contraction or reassociation (they change results).
`check`'s harness passes random values to parameters other than
lengths, so loops whose trip count is a random parameter take too long
for it (tests/cfront/loops.c is checked by hand).

### Engines

The candidates of step 3 also come from the rest of ucas
(`src/cf_bridge.c`, `src/cf_synth.c`): each lowered value (a node over
typed leaves, with the call sites' and assumed ranges and C's UB as its
domain) is offered to every engine, each candidate carries its engine's
own certificate, and then goes through the same pipeline as cfront's
own: proved against the value (exhaustively, by SAT, or, when SAT gives
up, by the certificate's derivation with a sample checking its
premises), priced by the target's costs, applied only when cheaper, and
under the same apply policy (vector code and table lookups proposed
unless `bench` measures them faster or `vectorize`; approximations
proposed unless `approx`).  The report names the engine and its
certificate:

```
> c optimize cfront/bridge.c nofile budget 120          (tests/cfbridge.ucas)
c optimize: 22 functions, 21 rewrites proved (6 exhaustive, 8 by SAT, 5 by algebra, 2 by induction) for x86-64-v3
  engines: find 5 gates 1 alg 1 ring 4 dio 3 gf2 2 loops 2 (+1 proposed) synth 1
  cong (line 7): (3 x + 5) % 7 == 2 as x % 7 == 6
    engine: ring; derivation: 3*6 + 5 = 23 = 2 (mod 7); gcd(3, 7) = 1 and 3^-1 = 5 (mod 7) (3*5 = 1, checked), so the solutions are x = 6 (mod 7); no wrap (intervals)
  two_sq (line 19): an equation without solutions mod q
    engine: dio; Diophantine certificate: a^2 + b^2 - 21 takes only the residues {1, 2, 4, 5, 6, 7, 8} mod 9 (every residue of its 2 inputs tried), never 0; the values are exact integers where defined
  ec_pts (line 22): an elliptic curve without such integer points
    engine: dio; Diophantine certificate: E: y^2 = x^3 - x has rank 0 (2-isogeny descent, ecq.c) and torsion of order 4: E(Q) = {O, (-1, 0), (0, 0), (1, 0)}, each with y = 0; ...
  sum4 (line 31): loop in closed form: s += sum of degree 4 in i
    engine: loops; derivation: Newton: P(lo+j) = sum_k D^k P(lo) C(j,k) in any commutative ring (degree 4), ...
  rev_each (line 62): intrinsic: synthesised: nibble LUT pair (two pshufb, xor_si128)
    engine: synth; synthesis: CEGIS: 1000 of 80421 sketches under cost 11.0 tried, 1015 SAT calls, 741 counterexamples, 0 linear solves; winner nibble LUT pair (two pshufb, xor_si128) (cost 10.0 with the moves); beat the value as written (cost 15.0)
  gather_bits (line 115): intrinsic: _pext_u32(x, 0xe38)
    engine: gf2; GF(2) linearity proof: affine by structure (...); its matrix moves single bits in order: a gather with mask 0xe38
```

| engine | what it offers | certificate |
|---|---|---|
| `find` | find.c's forms (closed forms, peephole drops, per-case `(x*K + C) >> S` fits, the enumerated bank to cost 4) under the target's µarch model | none of its own (`search`): cfront's proof |
| `gates` | a bitwise function of up to 6 words: its per-bit truth table, the minimal network over `x86bmi` (`c` without BMI); vpternlog for 3 words on AVX-512VL | minimality of the exhaustive search (gates.c) |
| `alg` | the value's polynomial over Z (alg.c): expanded, by Horner, factored; `P / Q` cancelled when Q divides P | the factorisation multiplied back; only + - * (an identity of Z holds in Z/2^w); a division only where neither side wraps (sound intervals or signed overflow as UB) |
| `ring` | `x * K == C` as `x == C K^-1` (mod 2^w, also for even K); `(a x + b) % m == c` as `x % m' == x0`; two congruences joined by the CRT; magic numbers (ring.c's `magic`) for u64 and i32 division | the inverse multiplied back; the linear congruence theorem with gcd and inverse; the CRT residues; Granlund-Montgomery's `m d = 2^k + e`, `e <= 2^s` |
| `dio` | equations and remainders a polynomial cannot reach: its residues mod q (q a power of two survives wrapping; any q on exact values); `u^2 + v^2 == C` by the two-squares theorem; `y^2 == x^3 + a x + b` with side conditions when E has rank 0 (ecq.c's 2-isogeny descent) and its torsion points | the residue table; the factorisation of C; rank and torsion |
| `gf2` | GF(2)-affine maps of the inputs: bit gathers and scatters (pext, pdep); CRC steps `(crc >> k) ^ T((crc ^ data) mod 2^k)`: Sarwate's table for k = 8, two pclmul (a Barrett reduction, mu = t^(32+k) div P in the ring (Z/2)[t]) | linearity by structure (xor, shifts and masks by constants, or/add of bit-disjoint operands, conversions), or `f(a ^ b) = f(a) ^ f(b) ^ f(0)` for all pairs (8 bits), or `f = M x ^ c` exhaustively (16 bits); for pclmul both sides affine and equal on every basis vector |
| `loops` | closed forms of counted loops beyond degree 2 (cf_lower.c) and geometric sums; float sums proposed as their exact real value (below) | derivations |
| `bound` | comparisons and clamps decided by bound.c's proved bounds where intervals do not decide them | bound.c's steps |
| `synth` | counterexample-guided synthesis over the intrinsics catalogue (below) | the search's statistics; the proof is cfront's (lanes) |

**Synthesis.**  `cf_synth.c` searches programs of up to 3 instructions on
128-bit registers over templates from the catalogue whose constant
operands are holes: gf2p8affine (matrix and xor byte), gf2p8mul,
pshufb (a table, or the nibble-LUT pair: two 16-entry lookups of the low
and high nibbles combined by xor, or or add), and/or/xor/add with a
constant, vpternlogd (imm8 and a constant operand), shifts and rotates
(amounts enumerated), binary and/or/xor/andnot/add.  Sketches are tried
shortest first, then cheapest under the µarch's costs (x86data.c), below
the value's cost (a solution then bounds the rest); holes are solved by
SAT against a growing set of counterexamples (gf2p8affine of the input
by GF(2) linear algebra), each solution verified for every input
(exhaustively up to 16 bits, else by SAT over the register) and a
failure adds its counterexample.  The budget is counted in sketches
(`steps`), so the results do not depend on the machine.  Scalar values of
one word go through it (moves into and out of xmm counted), and the
vectoriser asks it for every store of u8 lanes that is a function of one
u8 load, below the 8-bit translation's cost: what wins is whatever is
cheapest at the target, nothing is recognised by pattern:

| kernel | x86-64-v3 | icelake (GFNI, AVX-512) | zen4, `steps 3000` |
|---|---|---|---|
| `d[i] = s[i] << 3` | and; vpsllw 3 (32 lanes) | and; vpsllw 3 (64 lanes) | |
| `d[i] = s[i] >> 2 \| s[i] << 6` | nibble LUT pair | gf2p8affineqb | |
| bit reversal of each byte | nibble LUT pair | gf2p8affineqb | |
| `(x & ror(x, 2)) \| (~x & 0x5a)` | none within 1000 sketches (16-bit lanes) | the same | gf2p8affineqb; vpternlogd |
| bit reversal in each byte of a u64 (scalar) | nibble LUT pair | gf2p8affineqb | |
| popcount of each byte of a u64 (scalar) | none below the value's cost | nibble LUT pair (add) | |

The tests are `tests/cfbridge.ucas` (all engines, x86-64-v3; approx;
engine selection), `cfbridge_icl.ucas` and `cfbridge_zen4.ucas`, each in
a process of its own: cfront's proof budget is CPU time and the
synthesis grows the node table, which slows what runs after it.  The
kernels are `tests/cfront/bridge.c`.

The 8x8 matrix of gf2p8affine is packed as Intel defines it (the row of
output bit i in byte 7 - i, as the catalogue's lane semantics `b[7 - k]`
and wunkolo's post have it): bit reversal is 0x8040201008040201.

**Floats as integrals.**  A float loop `s += t(i)` whose term is a
polynomial in i over the loop's invariants (`s += x * x * h`, `x = i *
h`) is a left Riemann sum of a polynomial F, for which Euler-Maclaurin
terminates: its exact real value is Faulhaber's sums.  It is proposed
with its error bound, never applied by default (`approx` applies it): the
loop's own rounding differs from the exact sum by at most g(N + k)
(|s0| + sum |terms|), g(k) = k u/(1 - k u), u = 2^-53, N the trip count
and k the term's operations; the closed form's own rounding is a few
ulps of its terms.  For `riemann` (tests/cfront/bridge.c) the two
differed by 2.2e-14 relative at n = 797161 and 0 for n <= 121.  Integer
sums get exact closed forms instead (`loops`).

**What is validated where.**  Candidates are proved like every other
(the proof column).  Intrinsic code runs on this host (i5-9400) where it
can: `check` ran the SSSE3/AVX2 programs (nibble LUTs, and+shift
vectors) and the pclmul CRC on 40000+ calls, all agreeing.  GFNI and
AVX-512 do not exist here (QEMU 11.1's TCG has no GFNI either: it traps
on gf2p8affineqb), so the gf2p8affine and vpternlogd programs are proved
against the catalogue's lane semantics only, the same semantics the
vectoriser proves its lanes with; the catalogue's gf2p8affine is
Intel's pseudo-code (`parity(b[7 - k] & a) ^ imm`) and agrees with the
post's constants.

Not done: fsimp's float rewrites (none is bit-exact beyond cfront's own
IEEE identities) and minimax approximations (cfront does not lower libm
calls, so there is nothing to approximate yet); synthesis of values of
more than one input word, of sequences longer than 3, and with pext,
pclmul or vpermb as searched templates (the gf2 engine supplies pext,
pdep and pclmul forms directly); `error E` is accepted but only proposes
(no numeric bound is derived for a symbolic trip count).

## x86-64

Cost models for x86-64 cores, a throughput simulator, an intrinsics
catalogue with exact lane semantics, x86 gate bases, and the C
frontend's vectoriser.  Code: `src/x86data.c` (µarchs and costs),
`x86mca.c`, `x86intr.c` + `x86intr_tab.inc` (the catalogue and
`x86check`), `x86.c` (commands), `cfx86.c` (the vectoriser),
`tools/x86bench.c` (the measurements).

| command | |
|---|---|
| `x86 models` | the µarchs: `skl` (Skylake client; this host), `skx` (Skylake-X), `glc` (Golden Cove, Sapphire Rapids), `rpl` (Raptor/Alder Lake P-core), `zen4`, and the levels `x86v2`, `x86v3`, `x86v4` |
| `model NAME` | one µarch: features, ports, and per instruction class (77) latency, reciprocal throughput, uops and ports at xmm, ymm and zmm, each row tagged with its source |
| `mca F [on M]` | a cycle-level estimate of F's x86 code: critical path, port pressure, front end; loops per iteration |
| `intrinsic NAME [on M]` | signature, features, class, lane semantics (as ucas expressions, the first and last lane shown), costs |
| `intrinsics [FILTER] [on M]` | count per feature, or the names matching a substring, a feature or a class |
| `x86check [FILTER] [n N] [avx512]` | the lane semantics against the host cc's intrinsics |

Each µarch is also an ordinary cost model (`use zen4`, `cost F model
skl`, `find` under it; its scalar classes are the latencies).  JSON:
`x86_models`, `x86_model`, `mca`, `intrinsic`, `intrinsics`,
`x86check`; MCP group `x86`.

**Where the numbers come from.**  The cost tables were typed in from
uops.info, Agner Fog's instruction tables and the Intel and AMD
optimisation manuals **from memory, not scraped or copied**: treat
every number as approximate.  Each row carries a tag (`U` uops.info,
`A` Agner Fog, `I` Intel, `D` AMD's Zen 4 guide, `M` measured, `E`
estimate, `V` derived from the ymm row, `G` the worst of the level's
cores).  About 10 rows were measured on this host (i5-9400, Skylake)
with `tools/x86bench.c`; Zen 4 and Golden Cove rows tagged `E` are
estimates.  `mca` does not model caches, mispredictions or memory
dependences: it is a bound, not a measurement.

**The intrinsics catalogue**: 2585 intrinsics (SSE2 to AVX-512
F/BW/DQ/VL/CD/VBMI/VPOPCNTDQ/BITALG, FMA, BMI1/2, POPCNT, LZCNT, GFNI,
PCLMUL, AES), generated from family rows plus irregular ones.  Each has
its semantics per result lane in a small lane language compiled to ucas
nodes (so rewrites into intrinsics can be proved), its features and its
instruction class (whose costs are the µarch's).  AVX-512 masked forms
are `k[i] ? base : src` (or 0) over the unmasked one.  Memory forms
(loads, stores, masked ones, gathers) see the pointer as the 64 bytes
it points at, lanes of its element type; a store's result is that
memory after it (so the lanes it must not touch are checked too); a
gather reads element `index * scale / size` (scales below the element
size are not modelled).  Floating point is IEEE round to nearest even;
`rcp`/`rsqrt` are the exact values checked within their documented
error (and `rsqrtps` reads a denormal as a signed zero).

**x86check** compiles every intrinsic with the host cc (one test per
immediate) and compares its lanes with the semantics evaluated on 64
random and edge points.  What is validated where:

| | how | result |
|---|---|---|
| non-AVX-512 intrinsics | run on this host's hardware (AVX2) | 579 intrinsics, 1955 tests: all agree |
| AVX-512, most | `x86check avx512`: compiled for every feature, clang's header definitions lowered to generic vector IR and run as AVX2 code here | 1468 intrinsics agree with **clang's definitions**, not with AVX-512 hardware |
| AVX-512 that call an `llvm.x86.avx512.*` builtin | proved equal, half by half, to the run-validated narrower form of the same name | 42 proved (equal DAGs), 2 only sampled |
| their mask/maskz forms | composed from a validated base | 268 |
| the rest | | **201 not validated** (compress/expand, permutexvar, 16-bit and 64-bit variable shifts, 512-bit gathers, ...) |

No AVX-512 hardware or emulator is available: this host is an i5-9400,
QEMU 11.1's TCG has no AVX-512, and Intel SDE is not installed.  Before
this work the f32 FMA forms differed in 7 tests: ucas's evaluator
rounded an f32 fma twice.  `fma32` (src/eval.c) now rounds once (the
exact product in double, the sum rounded to odd through an exact
TwoSum, then to float; the derivation is in the comment), and so does
`fma64`, in integer arithmetic: FreeBSD's libm `fma()` is not
correctly rounded (`x86check avx512` found
`fma(-0x1.5af1d78b58c4p+66, -0x1.b428a02b5f643p-121, -1)`, and 2213 of
10^8 hard cases differ from `vfmadd`).

**Gate bases and find.**  `basis avx512` has vpternlog as a one-step
gate of 3 inputs (Bases above; basis.md has the LOP3 analysis this
generalises).  Under an x86 model with BMI1, `find` counts `andn`,
`blsr`, `blsi`, `blsmsk` as one operation each, and with BMI2 it solves
`y = pext(x, K)` and `y = pdep(x, K)` for a parameter exactly (a
subsequence match; a dynamic program), each match proved like every
other:

```
> e(x: u16) = ((x >> 3) & 7) | ((x >> 6) & 0x38)
> use skl
> find e
  cost 3   x >> 3 & 7 | x >> 6 & 0x38               pext(x, 0xe38)
```

**The vectoriser** (`src/cfx86.c`; `c optimize` offers it every
counted loop the C frontend extracts).  It must not make code slower,
and `clang -O2` vectorises such loops itself, so:

- by default its loops are **proposed, not applied**: the report gives
  the proof and the code ("proposed, not applied"), and the loop keeps
  what the rest of the pipeline does with it;
- with `bench`, every proposed loop is applied (and, in a build of their
  own, the loop-body rewrites that block vectorisation), the functions
  are timed
  against the original compiled by clang at the same target and flags
  (3 runs, each the median of 15), and a function keeps its vector loops
  (or its body rewrite: the faster of the two) only if it was faster by
  more than 5% in every run; otherwise the
  report says "proved, not applied: clang's own vectorisation is as fast
  or faster (N vs M cycles/element)" and the loop stays as written (a
  scalar body rewrite there would only undo clang's vectorisation);
  functions the harness cannot call (float parameters) are not applied;
- `vectorize` applies them all.

There is no default estimate of "the scalar loop as clang would
vectorise it": the scheduler (`mca`) prices ucas's own code, and
guessing clang's would not be honest.

- *What it takes*: `for (i = lo; i < hi; i++)` with an int-or-wider
  index and a bound of the index's type that the loop does not change;
  loads and stores at `i + c`; element-wise values of integer or
  float/double type; integer reductions `acc = acc op g` (`+ - ^ | &`,
  min, max, and the `?:` forms of min and max); AVX2/AVX-512 gathers
  `t[e(i)]` of 32-bit elements.
- *How*: one intrinsic per node, in lanes of W bits, the narrowest W
  for which every lane is proved (bytes computed in 16-bit lanes when
  the C int arithmetic needs more than 8 bits), loads widened with
  pmovzx/pmovsx and stores narrowed with packs or vpmov; clamps become
  min/max, a clamp of `a + b`, `a - b` or `2a` to the 8/16-bit range a
  saturating add or subtract (padds, paddus, psubs, psubus),
  `(a + b + 1) >> 1` pavg, multiplies by 2^k±1 shifts and
  adds.  The vector width is the target's widest (512 with AVX-512F,
  and BW for 8/16-bit lanes; 256 with AVX2; 128 otherwise); the code is
  guarded by the features' macros, unrolled by 4, and followed by the
  scalar loop (rebuilt from the IR) for the remainder and for builds
  without the features.
- *Proof* (`lanes`): each vector value carries its lanes as ucas nodes
  built from the catalogue's lane semantics, operand by operand (the
  bitwise si forms act on every lane view; reinterpreting a register as
  other lane widths is a bit cut).  Loads give every lane its own input,
  so data moving between lanes would show.  Lane j of every store and
  reduction operand is proved equal to the scalar value at iteration
  `i + j`: equal DAGs, exhaustively, or by SAT.  A gather's index lanes
  are proved equal to the scalar index; a min or max read off a `?:` is
  proved to be one.  Integer reductions are reassociated, which is exact
  mod 2^n (signed sums are computed unsigned; where C's sum is defined it
  is that sum).
- *Memory*: arrays are disjoint by `restrict` or by being distinct
  arrays (the IR's facts); otherwise the output tests the byte ranges at
  run time and falls back to the scalar loop.  A store that feeds a
  later iteration's load, a load after a store to its array, two stores
  to one array, strided loads, and gathers from an array that may
  overlap a written one are refused.
- *Refused, with the reason in the report*: float reductions
  (reassociation changes the rounding; there is no opt-in), conditional
  loads and stores (not in the IR), loop-carried dependences, strided
  access.

`check` compared every vectorised function with the original on 20000
calls each (tests/cfront/mem.c, vec.c) at x86-64-v3 and x86-64-v2;
the float loops (`check`'s harness skips float parameters) were
compared by hand on 20000 random calls with bit-exact results, the
original compiled with `-ffp-contract=off`: ucas proves each operation
rounded on its own and its output says `FP_CONTRACT OFF`, while clang's
default fuses `a * x[i] + y[i]` in the original into an fma (C99 allows
either), so with the default flags the two differ in the last bit.  For
AVX-512 targets this host only compiles the vector code (`check` says
so) and tests the scalar fallback.

Measured with `bench` (i5-9400, cycles per element over 4096 elements,
3 runs each the median of 15, other builds were running).  "clang" is the
original compiled by clang 19 `-O2 -march=x86-64-v3`, so it is clang's
own autovectoriser; "applied" is what `bench` decides:

| loop | clang | vectoriser | applied |
|---|---|---|---|
| `sat16` (`s*2` saturated to int16) | 0.18 | 0.06 | yes, x3.3: one `vpaddsw` |
| `brighten` (u8, `*3/2+10`, clamp) | 0.39 | 0.14 | yes, x2.7: 16-bit lanes where clang uses 32 |
| `gath` (`t[idx[i] & 63]`) | 0.83 | 0.63 | yes, x1.3: clang does not gather on Skylake |
| `avg8` (`(a+b+1)>>1`) | 0.05 | 0.04 | no (not 5% faster in every run; both use pavgb) |
| `add32` (`a + b*3`) | 0.33 | 0.38 | no |
| `iota` (`2i + 1`) | 0.16 | 0.22 | no |
| `sum8`, `xorsum` (reductions) | 0.09 | 0.11 | no |
| `invert_low` (`p < 16 ? 15 - p : p`) | 0.05 | 0.06 | no (its table rewrite, 0.80, is not applied either: kept as written) |
| `maxv`, `sum64`, `shift_left` | | the same | no |
| `saxpy`, `dscale` (float parameters) | | not timed | no |

So it is mostly no better than `clang -O2 -march=native`'s
autovectoriser, sometimes worse, and those loops are not applied; it
wins where its proofs let it pick narrower lanes than C's promotions
suggest, on saturating clamps, and on gathers.  Code placement alone
moved one unchanged function by up to 25% between the two builds, which
is why one run is not enough to decide.  Its cost
estimates (`cost ... -> ...` in the report) are reciprocal throughputs
from the model, not measurements.

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
  is every float in that interval.  An `f16` parameter has 63487 finite
  values, so a function of one is analysed on all of them: a float form of
  an integer formula (`u8(floor(f16(x) * f16(1 / 3)))` for `x / 3`, the
  fp16 arithmetic a mobile GPU does fastest) is proved by `equiv`, not
  sampled, while the integer inputs stay within 2^22 points.
- `f16` has no host type in C99: `fround`, `fbits` and `frombits`
  (src/eval.c) round and convert on the double, and `fma16` rounds once by
  the same round-to-odd step as `fma32`.  `asuint`/`asfloat` take and give
  16 bits and keep a NaN's payload.

## Limits (for now)

- `gates` stops at 6 input bits (64-entry truth tables), and its
  fallback beyond the exact search is weak.  The next steps are
  bit-blasting the word-level expression as a starting upper bound and
  SAT-based exact synthesis.
- `crit`/`domain` solve numerically for one real parameter; several
  real parameters get the gradient only.
- `find` builds terms of up to about 5-6 operations before the bank
  (128 MB) fills; constants come from a small pool plus solved holes.
- `f16` is priced as `f32` by the cost models (none has a class for half
  floats yet) and `approx` has no f16 format.  A float `equiv` is by
  enumeration: two unranged f16 parameters are 2^32 points, past the 2^22
  that are enumerated, and are sampled (SAT covers the integers only).
- `fit` is a scan followed by a pattern search, not a global optimiser.
- `approx` measures on a sample, not on every float (the C output can
  show a slightly larger maximum); the coefficient search after rounding
  is a local walk, not a lattice-based (fpminimax) search; range
  reductions are offered for builtins only, and the cost model prices
  throughput, not latency.  Piecewise approximations use two pieces.

## Plots

```
> plot sin x, cos x
plot of 2 functions
  x in [-6.283 (-2pi), 6.283 (2pi)]: periodic with period 6.283 (2pi): two periods
  y in [-1.1, 1.1]: the values
  ...
  image: 640x440, 13072 bytes
  wrote plot.gif
> plot tan x
  y in [-13.45, 13.4]: the bulk of the values (they reach [-191, 191])
    poles at x ≈ -1.571 (-pi/2), 1.571 (pi/2)
> neighbourhood sin(x)/x at 3
  x in [1.714, 4.286]: around x = 3: features on both sides within 0.9184 (an inflection point at 2.082)
```

`plot` draws functions into a GIF (a still plot is one frame): NAME.gif
on the command line (default `plot.gif`), an attachment in the bot, as
`animate` does.  It also prints what it chose and found (bounds and why,
samples, poles, jumps, undefined stretches), so the text is a test.

| form | |
|------|--|
| `plot F, G, ...` | functions of one variable overlaid, with a legend.  The line breaks where F is undefined, at jumps and at poles (a jump that stays when the interval is halved 60 times; a pole when the values grow; poles get a dashed asymptote), and the y range keeps the bulk of the values when there are poles or far outliers.  Integer inputs are points (up to 160) or steps |
| `plot x^2 + y^2 = 1`, `plot y = x^2`, `plot x = 3` | an equation: `y = ...` is a curve, anything else lhs - rhs = 0, drawn by marching squares (equal scales on both axes) |
| `plot x^2 + y^2 < 4` | a region, shaded, with its boundary |
| `plot (x(t), y(t))`, `plot r = f(theta)`, `plot polar f` | parametric and polar curves (t in [0, 2pi] by default; a polar curve runs until it closes) |
| `plot (x(t), y(t), z(t))` | a space curve in a box, coloured along t |
| `plot (P, Q)` | a vector field: arrows coloured by length |
| `plot y' = f(t, y)`, `plot y` (a defined ODE), `plot F field` | a slope field with nine RK4 solutions; through the initial value `y(0) = ...` when the ODE has one or the plot says `with y(0) = 1` |
| `plot F` of two variables, `plot3d F` | a surface: a z-buffered, shaded and height-coloured 65x65 grid with grid lines, back panes, labelled axes and a colour bar.  `contour`, `heatmap`, `wireframe` are the other views; `rotate` turns it once (36 frames) |
| three or more variables | one is animated (`t` if there is one, else the third), x and y form a heat map (`3d`/`plot3d`: a surface); with four, small multiples (3x3) over a third and frames over the fourth; the rest are fixed at the middle of their range, with a note.  `slices`: small multiples instead of frames; `parallel`: parallel coordinates of 400 random points |
| `plot F animate t` | frames over t of whatever the other variables make |
| `deriv`, `tangent a` | also F' (symbolic, `diff`'s) and the tangent at a |
| `logx`, `logy`, `loglog` | logarithmic axes |
| `x in a..b`, `x in [a, b]`, `x from a to b`, `y in c..d`, `z in ...` | ranges; `x` is the horizontal axis whatever the curve's variable is called |
| `fps F`, `frames N`, `seconds S`, `points N`, `to NAME`, `nofile` | animation, sampling, the file (`nofile`: a checksum, for tests) |

Bounds that are not given come from the parameter's declared range,
then its type (`x: u8` is 0..255), then the function: a period found
numerically (`guess_period`: multiples of pi/6, pi/4 and quarter units
first, then a scan; two periods are shown, one period loops an
animation); else the zeros, extrema and inflection points
(`crit_roots` on F and F'), poles, jumps, holes and domain edges in
[-10, 10] (else [-100, 100]), with a margin, not past a domain edge, the
eight nearest 0 when there are many, rounded to a twentieth of the
span.  With several variables each is sliced at a few values of the
others, and a range from the features is widened to another variable's
when f is symmetric in the two up to sign (checked at 32 points: `x^2 -
y^2` gets a square) or when both are around 0 with spans within 1.5x
(the features do not set them apart).  Integer inputs (and integer
values) get integer ticks.  A log axis whose range reaches 0 starts two
decades below the first positive zero or extremum (sampled on a log
grid), else three decades below the top decade.  `neighbourhood F` (also `plot F around a`, `at a`) prints the
features it found; `at a` takes a window reaching the nearest feature
on each side, or the radius of curvature when there is none.

In English: `plot|graph|draw|sketch|visualize|chart|render ...`, `show
me a surface plot of ...`, `a contour plot of ...`, `F as a
neighbourhood`, `plot F near its interesting part`, `what does F look
like`, `and|with its derivative`, `and its tangent at 2`, `in 3d`, `as a
heatmap`, `rotating`, `animated over t`, `as t goes from 0 to 1`, `on a
log scale` (y, with a note), `log-log`, `in polar coordinates`, `the
slope field of ...` (or `the direction field y' = ...`, `... with y(0) = 1`),
`the vector field (P, Q)`, `(P, Q) as arrows`, `as small multiples`,
`in parallel coordinates`, `at 30 fps`, `save it as NAME`.

## Integrals and solids of revolution

```
> slice of integral f(x) = x^2 at 0 till 4
  ucas: area x^2 from 0 to 4
area under x^2
  x in [0, 4]: given
  ∫ x^2 dx from 0 to 4
  antiderivative: x^3/3
  exact: F(4) - F(0) = 21.3333333333333 = 64/3
  numeric (adaptive Gauss-Kronrod 7-15, 15 evaluations): 21.3333333333333, estimated error 2.3e-13
  ...
  wrote area.gif
> revolve f(x) = x^2 in x-axis
  ucas: revolve x^2 about x
solid of revolution: the region between x^2 and the x-axis about the x-axis
  x in [-2, 2]: plot's bounds (covering the 2 features found in [-10, 10])
  method: disks, V = pi ∫ (x^2)^2 dx from -2 to 2
    exact: V = 40.2123859659494 = 64pi/5
  check: shells in the radius on a grid: V ≈ 40.2148 (relative difference 6e-05, the grid's resolution)
  ...
  animation: 600x500, 37 frames (5.2 s), ... bytes
```

`area` draws F with the region between it and the x-axis (or between F
and G) shaded on [a, b], the parts above and below in their own colours,
a and b marked, and prints the integral both ways: F(b) - F(a) from the
antiderivative when integ.c finds one (shown as an exact form when the
value is p/q, times pi, pi^2, sqrt 2, 3 or 5, q <= 1000), and adaptive
Gauss-Kronrod always, with how far they agree.  The zeros (or crossings)
inside split it into signed parts; the total area is ∫ |f|, the area
between two curves ∫ |f - g| next to the signed ∫ (f - g).  An end where
the antiderivative is undefined takes its limit (log x on [0, 1] is -1),
or the integral diverges there (1/x on [0, 1]) and no area is claimed; an
integrable singular end (arc length of a quarter circle) is trimmed by
1e-8 and 1e-12 of the interval and the difference is the error stated.

| form | |
|------|--|
| `area F from a to b`, `area F x in a..b` | the shaded integral, one frame (`area.gif`) |
| `area F and G` | the region between them; without bounds, between their outermost crossings |
| `area F` | without bounds: one lobe between two zeros (the one nearest 0), else plot's bounds (the neighbourhood search), with a note |
| `riemann N [left\|right\|mid\|trapezoid\|simpson]` | the sum's rectangles, trapezoids or parabolas over the region, the sum, the integral and the error; without N an animation of n = 1, 2, 4 ... 64 converging (midpoint by default; Simpson takes an even n) |
| `accumulate` | A(t) = ∫_a^t F: a sweeping line over F with the running area, and A's graph below traced as it goes (its formula when F has an antiderivative) |
| `frames N`, `fps F`, `to NAME`, `nofile` | as plot's |

`revolve` turns a region around an axis: one curve about a horizontal
axis, the region between F and the axis (`revolve x^2 and 0 about y = -1`
for the region under F); about a vertical axis, the region between F and
the x-axis; two curves, the region between them (washers).  The default
axis is the x-axis (with a note), and bounds not given are found as for
`area`, about a vertical axis on the side x >= c.  The volume is by
washers (pi ∫ (R^2 - r^2) dx, R and r the distances to the axis) about a
horizontal axis and by shells (2pi ∫ |x - c| |f - g| dx) about a vertical
one, exact when the integrand has an antiderivative (squares are expanded
first: sqrt(x)^2 = x, sin^2 u = (1 - cos 2u)/2), numeric always, and
checked by the other method on a grid (shells in the radius, washers in
y).  A region that crosses a horizontal axis turns into disks out to the
further side; one on both sides of a vertical axis keeps the shell
integral when the two sides sweep separate parts, else the union is
measured on the grid.  Also printed: the region's area and centroid, and
Pappus' theorem V = 2pi d A as a check when the region stays on one side;
the lateral surface area 2pi ∫ r sqrt(1 + F'^2) dx of each curve (r its
distance to the axis, without the end disks), exact or numeric.

The picture is the classic one: the region, in the plane, sweeps from 0
to 2pi about the axis and leaves the surface of the solid behind (the
region's boundary, revolved, is that surface: the curves and the ends),
drawn through plot's z-buffer, lit per vertex and dithered, the outside
and inside of a surface in different shades, the region and its curve
showing through the solid while it turns, the axis dashed where hidden,
and a ruler beside it.  It is seen from the narrow end, so the curved
surface faces the eye.

| form | |
|------|--|
| `revolve F [and G] [from a to b]` | the solid and its numbers; 36 steps of 10 degrees (`revolve.gif`) |
| `about x`, `about the x-axis`, `about y`, `about y = c`, `about x = c` | the axis |
| `disk`, `washer`, `shell` | the method asked for; when it would integrate in y, the dx method is used and the asked one is the check (with a note) |
| `still`, `rotate` | the finished solid only; or then turned once (the light turns with the eye) |
| `volume ...`, `surfacearea ...` | the numbers without the image |
| `arclength F from a to b` | ∫ sqrt(1 + F'^2) dx, with the chord as a lower bound |
| `centroid F [and G] [from a to b]` | the area and centroid of the region (between F and the x-axis, or F and G), and the volumes Pappus gives about the axes |

In English: `slice of integral f(x) = x^2 at 0 till 4`, `area under x^2
from 0 to 4`, `shade the integral of sin x from 0 to pi`, `show the area
between x^2 and x from 0 to 1`, `a left riemann sum of x^2 from 0 to 4
with 8 rectangles`, `the midpoint rule for sin x`, `the trapezoid rule
...`, `simpson's rule ...`, `the accumulation function of cos x from 0
to 2pi`, `revolve f(x) = x^2 in x-axis`, `rotate x^2 around the y axis`,
`solid of revolution of sqrt x from 0 to 4`, `spin sin x about the
x-axis`, `rotate the region between x and x^2 about the line y = 2`,
`using washers|shells`, `and rotate it`, `as a still image`, `the
volume of revolution of ...`, `the volume of the solid obtained by
rotating ...`, `the surface area of revolution of ...`, `the arc length
of ...`, `the centroid of the region between ...`, `in 12 frames`, `at
20 fps`.  A definition in the sentence (`f(x) = x^2`, `y = x^2`) gives
the function and is not kept.

## Shaders on images

`animate exec` runs a shader's image instructions, and a shader with
images is shown working on whole images: its inputs, its outputs filling
in, and the 2x2 quad's trace.  The user's own pictures can be the
inputs.

```
animate exec of this seed 1                     (a ```glsl block follows)
animate exec of this input 1 gradient input 2 #ff8800 output R8 size 128x128
animate exec of this input 1 file photo.jpg sampler nearest clamp
run this on my images, for input 1 use pure black      (English)
use a red image for the second input, with an RG8 output
```

**Instructions.**  `OpTypeImage` (1D, 2D, 3D, Cube, Rect, Buffer;
arrayed; depth; sampled or storage; its format), `OpTypeSampler`,
`OpTypeSampledImage`; `OpSampledImage`, `OpImage`, `OpImageSample*`
(implicit and explicit LOD, with `Bias`, `Lod`, `Grad`, `ConstOffset`,
`Offset`, `MinLod`; the `Dref` and `Proj` forms), `OpImageFetch`,
`OpImageGather`, `OpImageDrefGather`, `OpImageRead`, `OpImageWrite`,
`OpImageQuerySize`, `SizeLod`, `Levels`, `Samples`, `Lod`,
`OpImageTexelPointer` with the atomics on it.  `OpDPdx`, `OpDPdy`,
their `Fine` and `Coarse` forms and `OpFwidth*` come from the 2x2 quad,
as the hardware does: the lanes are laid out in rows, the x neighbour of
lane l is l ^ 1 and the y neighbour l ^ cols; `Fine` uses the lane's own
row (column), `Coarse` the quad's top-left pixel's, and the plain forms
are `Fine`.  A neighbour that is masked off (divergent control flow, a
discarded pixel) counts as equal (a difference of 0), counted in a note.
Matrices (`OpTypeMatrix`, `OpTranspose`, the matrix products,
`OpOuterProduct`, composites of up to 16 components) and the
GLSL.std.450 vector functions (`Length`, `Distance`, `Normalize`,
`Cross`, `Reflect`, `Refract`, `FaceForward`, `Determinant`,
`MatrixInverse`, the `Pack`/`Unpack` family) are there too.  An image
handle is a value like any other (its variable's memory holds the
texture's and sampler's numbers), so arrays of textures, phis and calls
carry images.  Texture operations are priced as loads, derivatives as
float adds.

**Inputs.**  Every image, sampled image and texture binding is an input
slot, in (set, binding) order, then as declared; each element of an
array is one.  A separate `sampler` binding is a sampler, not an input;
a write-only storage image (`writeonly`, NonReadable) is an output that
starts clear.  An input is, by default, random noise in every channel
(RGBA8, 64x64, seeded: `seed K`, so tests are deterministic), or what
the request says:

| option | meaning |
|--------|---------|
| `input N noise` / `gradient` / `checker` | a pattern (the gradient: red across, green down) |
| `input N black` / `#rrggbb` / `#rrggbbaa` / `rgba(1, 0, 0, 0.5)` | one colour; names are CSS's (`purple`, `orange`, `teal` ...), except `green`, which is pure green (CSS's green is `darkgreen` here) |
| `input N file PATH` | a PNG, JPEG, GIF (the first frame), BMP or PPM/PGM file (the command line and JSON only) |
| `input N gallery K` | the K-th picture that came with the request |
| `input N R8` (`RG8`, `RGB8`, `RGBA8`, `SRGB8_A8`, `RGBA16F`, `R32F`, `R32UI` ...) | its format; the pixels are converted to it as an upload would |
| `input N 32x32`, `input N nearest mirror` | a pattern's size; this input's sampler |
| `inputs white` | every input |
| `output RG8` | the colour target's format (RGBA8 by default, unorm, clamped) |
| `size WxH` | the image run's size |
| `sampler nearest clamp` | the sampler state (also `linear`, `trilinear`, `repeat`, `mirror`, `border white`) |
| `images` | the request's pictures, asked for |

Pictures that come with the request (Discord attachments, the replied-to
message's gallery, ```` ```image NAME ```` blocks of base64 after the
shader, JSON's `png_base64`) fill the inputs not given otherwise, in
order; the rest are noise.  A storage image declared with a format keeps
it; an integer sampled type (`usampler2D`) gets `RGBA8UI` by default.
Pictures larger than 1024 pixels a side are box-filtered down; files are
decoded by stb_image v2.30 (`src/third_party`, unmodified, compiled once
in `src/img.c` from memory, PNG, JPEG baseline and progressive, GIF, BMP
and PNM only), with the size, dimensions (16384 a side) and pixel count
(24 M) checked before anything is decoded.  QOI and WebP are refused.

In English (`src/nlp_img.inc`), the line before a shader's code block is
read as a request to run it when it says so ("run", "execute", "apply",
"show", "render" ...) or names its inputs: "for input 1 use pure black",
"use a red image for the second input", "input 2 is a checkerboard",
"use white for all inputs", "with an RG8 output", "show it on a 128x128
image", "use my images", "on these pictures", "with nearest sampling".
Clauses (split at commas and "and") pair the inputs they name with what
they say; a colour alone is for every input; a format next to "output",
or with no input named, is the output's.  Misspelt words are corrected
("purpel" is "purple").  The statement it becomes is shown (`note: read
as: animate exec of this input 1 black`).

**The image run.**  A fragment shader (or a compute shader with storage
images) is also run over every pixel of an image: the first input's
size, scaled down to at most 256x256, or `size WxH` (up to 1024x1024),
8x4 lanes at a time so that every 2x2 quad is together for derivatives
(lanes past the edge run as helpers and write nothing).  `FragCoord` is
the pixel's centre; the `location 0` input that is a `vec2` or `vec3`
float (else the first such input) is the pixel's uv, its centre over
the image's size, 0..1 (a note says which input was taken); other inputs
are random per pixel, or set (`%name = 0.5`).  Each `Output` variable is
an image: `location 0` in the chosen format, integer outputs as RGBA32I
or RGBA32UI; a discarded pixel stays clear.  A compute shader runs one
invocation per pixel (`GlobalInvocationId` = the pixel,
`LocalInvocationId` and `WorkgroupId` from its `LocalSize`); shared
memory and barriers across more than 32 invocations are not modelled.
The trace (as before) is of the lanes at the image's centre.  The report
gives each input's and output's checksum (FNV-1a over the stored
texels).  The run stops, with a note, after 20 million instructions.

**The GIF.**  Above the trace: each input ("input 1: noise RGBA8
64x64", or the file's name), then each output, and each storage image
the run wrote ("..., after").  An image is shown at 192 pixels on its
longer side (whole multiples for small ones), alpha over a checkerboard,
in a 6x8x5 colour cube with a 4x4 ordered dither.  The outputs fill in
quad by quad, in rows, as the trace's time runs, an orange box on the
quad being filled; a blue box marks the traced lanes' pixels on every
panel.  The GIF uses 256 colours and writes the pixels a frame does not
change as transparent, so it stays small; the 6 MiB cap and its lower
frame rates apply as before.

**Formats and rounding.**  Texels are stored as the format holds them,
then read as Vulkan specifies ("Fixed-Point Data Conversions", "sRGB"):

- unorm: c / (2^b - 1); a write clamps to [0, 1] and rounds to nearest
  (floor(v * (2^b - 1) + 0.5)).  snorm: max(c / (2^(b-1) - 1), -1).
- sRGB: decoded per texel before filtering, c <= 0.04045 ? c / 12.92 :
  ((c + 0.055) / 1.055)^2.4; alpha linear.  A file's bytes are taken as
  encoded sRGB for an sRGB format and as linear values for a unorm one
  (as an upload of the same bytes would).
- 16-bit floats round to nearest even (overflow to infinity); R11G11B10F
  has no sign; integer writes keep the low bits.
- a format's missing channels read as 0, 0, 0 and 1.
- outputs: unorm clamped and rounded as above.

**Sampling.**  The sampler's state is not in SPIR-V, so it is assumed:
linear filtering, linear mipmaps, repeat (a note says so, and how to
change it).  Mipmaps are made by a 2x2 (2x2x2) box filter of the level
above, in linear space for sRGB.  The LOD: lambda = log2(rho) plus the
sampler's and the instruction's bias, rho the larger of the lengths of
the coordinate's x and y derivatives in texels (Vulkan's scale factor,
from the quad's fine derivatives for implicit LOD), clamped to [minLod,
maxLod]; lambda <= 0 magnifies.  Nearest mip: level ceil(d + 0.5) - 1;
linear: a blend of floor(d) and the next.  Nearest filtering takes
floor(u); linear the 2, 4 or 8 texels around u - 0.5, weighted by the
fraction.  Addressing: repeat, mirrored repeat, clamp to edge, clamp to
border (transparent black, opaque black or white), mirror clamp to
edge.  Integer formats are never filtered (nearest).  Gather returns the
footprint's texels in Vulkan's order (i0,j1) (i1,j1) (i1,j0) (i0,j0)
from level 0.  Dref compares (less or equal by default) each texel
before filtering.  A fetch, read or write outside the image (or its
levels and layers) reads 0, 0, 0 and 1 (alpha 0 when the format has
alpha) and drops the write, as robustImageAccess2 does, counted in a
note.

**Differences from a GPU.**  Filter weights and coordinates are in
double precision; GPUs use 8 bits of sub-texel precision for the
weights (and fixed point for the coordinates), so their results can
differ in the last bit of an 8-bit value, more at texel boundaries.
GPUs pick the LOD per quad or per pixel pair with approximations of
rho (and may use anisotropic filtering, off here); the LOD here is per
lane and exact.  Cube maps are not filtered across faces (no seamless
filtering), and their LOD uses the face's 2D derivatives.  Mipmaps a
driver or an application generates may use other filters.

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

The bot reacts to each message it handles: a question mark when it
asks a follow-up question (or when a mention arrives with nothing
else), a check mark when it understood and answered, and a cross when it
could not answer (an error, "sorry" from the translator, a failed check,
or a timeout, the memory limit or a crash).  A `$` typed after the
mention, from the old trigger, is ignored.

The rest of the message goes to the engine unchanged (a surrounding
```` ``` ```` fence with an optional language tag, or single backticks,
is removed), so new commands work without changes to the bot.  When the
input has more than one line, each statement is echoed as `> stmt`
before its output, as with `ucas file`.  Output comes back as a reply
in a code block that mentions nobody.  Up to 3 messages are used,
split at line breaks under Discord's 2000-character limit; anything
longer is sent as a preview plus the whole output attached as
`ucas-output.txt`.  Messages from itself, webhooks and system users are
ignored, and so are other bots' except those on an allowlist (see [Bots
and programs](#bots-and-programs)).

Attachments and replies count as input.  A message's attached files,
and for a Discord reply to someone else's message that message's
```` ``` ```` blocks and files, are added to the request as code blocks,
so `@Wimsy animate exec of this` works on a posted shader.  Binary
SPIR-V goes through `spirv-dis`; GLSL (`.comp`, `.frag`, `.vert`,
`.glsl` ..., the stage from the extension or what the code uses)
through `glslangValidator -V` and then `spirv-dis`; C (`.c`, `.h`, or
text that is C by its content) as a ```` ```c ```` block for the C
frontend; SPIR-V assembly and other text are added as they are.  Images (PNG, JPEG, GIF, BMP, PPM)
are the shader's image inputs, in attachment order (see [Shaders on
images](#shaders-on-images)): a gallery posted by someone, replied to
with `@Wimsy run this on these, for input 1 use pure black` and the
shader in a code block (or the shader posted, replied to with the
pictures attached).  They travel to the engine as ```` ```image NAME ````
blocks of base64.  Files come only from Discord's CDN, code at most
256 KiB and images at most 8 MiB (decoded, at most 24 Mpixels, and
scaled down to 1024 pixels a side), at most 10 per message, and the
tools run with CPU and memory limits.  A short note says what was done
with each file.

### Setup

1. In the Discord Developer Portal, create an application, add a Bot,
   and copy its token.
2. Under Bot → Privileged Gateway Intents, enable **Message Content
   Intent** (without it the bot sees empty messages in servers).
3. Invite it (OAuth2 → URL Generator, scope `bot`, permissions *View
   Channels*, *Send Messages*, *Send Messages in Threads*, *Read Message
   History*, *Attach Files*, *Add Reactions*).
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

### Bots and programs

Another bot (an AI agent on Discord, say) can use the bot as a person
does, and has a JSON path made for programs.  `@Wimsy explain AI usage`
(also "explain ai use", "how do bots use you", "AI usage", "usage for
AI", "help for bots", "bot usage", misspellings tolerated) answers anyone
with the guide written for such a bot: a short version in the message
and the whole of it attached as `wimsy-ai-usage.md`.  Its source is
`discord/ai_usage.md` (and `ai_usage_short.md`), embedded at build time;
`{MENTION}`, `{ID}`, `{RATE}`, `{BURST}`, `{CHAIN}`, `{CPU}` and `{WALL}`
are filled in at run time.  `@Wimsy help` points at it.

**Which bots.**  Only bot accounts on the allowlist are served, by
default the two bots `1562594665109393468` and
`1566587484866093137`; every other bot and every
webhook is ignored, silently.  There is no setting that serves every
bot.  A bot must address the bot explicitly: a real mention (`<@id>`,
not the name typed as text) or a Discord reply to one of its messages.
A bare reply to one of its notices (the rate limit, the chain limit,
the help) is not a request.

**Safeguards**, for loops and abuse:

- a rate limit per bot (a token bucket): `UCAS_DISCORD_BOT_BURST`
  requests at once, refilled at `UCAS_DISCORD_BOT_RATE` a minute.  The
  first request over it is answered with `rate limited: retry in N s
  (...)`; the rest are ignored until a request is taken again.
- a cap on reply chains: a bot's request that is a Discord reply to the
  bot deepens its chain in that channel, a fresh mention starts a new
  one, and any human message in the channel ends them all.  Past
  `UCAS_DISCORD_BOT_CHAIN` the bot says `reply-chain limit: ...` once
  and ignores the chain.
- replies never ping: `allowed_mentions` is empty and the reply
  reference does not notify (as for everyone).
- the log (`--log`) marks bot authors (`"bot":true`) and records each
  refusal (`bot-rate-limited`, `bot-chain-cap`, `bot-notice-reply`) and
  each JSON request (`json-request`, `json-reset`, `json-schema`,
  `json-refused`).

| variable | default | |
|----------|---------|--|
| `UCAS_DISCORD_BOTS` | `1562594665109393468,1566587484866093137` | bot user ids served, comma separated; `deny` (or empty) for none |
| `UCAS_DISCORD_BOT_RATE` | 6 | requests a minute per bot |
| `UCAS_DISCORD_BOT_BURST` | 3 | requests at once per bot |
| `UCAS_DISCORD_BOT_CHAIN` | 5 | replies in a row to the bot without a human message |

**JSON over Discord.**  A message addressed to the bot that holds a
```` ```json ```` block is a request of the [JSON frontend](#json-frontend-for-agents-and-programs)
(the protocol of `ucas --json`: `{"id", "cmd", "args"}`, or an array of
up to 8 run in order); text outside the block is ignored.  Anyone may
use it, not only bots.

````
@Wimsy ```json
{"id": 1, "cmd": "integrate", "args": {"f": "x^2 sin(x)"}}
```
````

The reply is a ```` ```json ```` block with the response (or the array
of them), as `ucas --json` gives it, and the usual reaction (a check
mark when every response is `ok`, else a cross):

```json
{"id": 1, "ok": true, "cmd": "integrate", "ucas": "integrate x^2 sin(x)", "structured": true,
 "result": {"kind": "integrate", "var": "x", "integrand": "x^2*sin(x)",
 "antiderivative": "2x*sin(x) + 2cos(x) - x^2*cos(x)"}, "notes": [], "conditions": [],
 "checks": [{"kind": "derivative", "points": 401, "max_diff": 7.105427357601002e-15, "passed": true, ...}],
 "files": [], "text": "∫ x^2*sin(x) dx\n  = 2x*sin(x) + 2cos(x) - x^2*cos(x) + C\n  check: ..."}
```

- Requests run in the channel's worker, through the JSON frontend's own
  request handling (`json_bot_serve` in `jsonmode.c`, the same `handle`
  as `ucas --json`), so they share the channel's state with the text
  path.  The answers are deterministic: an ambiguity is an error
  (`"code": "ambiguous"`, `key`, `readings`), never a question; repeat
  the request with `args.assume` or `args.defaults`.
- What the sandbox cannot do happens in the bot, around the worker:
  GLSL in `args.code` is compiled to SPIR-V (a note says so),
  spirv-cross's GLSL view goes into `text` and `result.glsl_view`, and
  `{"attachment": N}` (the message's N-th attachment, from 1) can stand
  for `args.code` or an entry of `args.inputs` (an image).
- Files (`exec.gif`, `simplified.spvasm`, `simplified.glsl`,
  `simplified.c`, plots) are attached to the reply and listed in `files`
  as `{"name", "mime", "bytes", "attached"}`, without base64.
  `args.files` other than `"none"` and `timeout_s` are not used (the
  limits below apply).
- A response that does not fit in a message is attached whole as
  `response.json`; the message holds it with `text` cut
  (`"text_truncated": true`) or, if that is still too long, a summary
  (`id`, `ok`, `cmd`, `headline`: the first lines of the text, `error`,
  `files`), with `"full": "response.json"` either way.
- `{"cmd": "schema"}` attaches `schema.json`; `{"cmd": "reset"}` resets
  the channel.  Both go alone in a message.  Malformed JSON, a missing
  attachment or GLSL that does not compile is answered with an error
  response (`bad_json`, `bad_attachment`, `bad_code`); in an array, the
  refused request has the error and the others `not_run`.

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
| `UCAS_DISCORD_MAXOUT` | 8388608 | output bytes kept per message (a GIF travels in it as base64) |

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
REST request it would make (and the content of an attached
`response.json`).  An event's `author` may carry `"bot": true`, and its
`timestamp` is the clock of the bots' rate limit in this mode.
`ucas-discord --selftest` checks the prefix and fence handling, message
splitting, the JSON codecs, the allowlist, the rate limit, the chain
cap and the "explain AI usage" phrases.
`make discord-test` runs both against `discord/tests/*.jsonl`, with
`UCAS_SPIRV_CROSS=none` unless `NAME.env` sets it; a test with a
`NAME.needs` file is skipped when that tool is missing.

## Proofs

```
theorem bockstein: forall C:cx, p n r:int, a:el: fintype(C) and prime(p) and 0 <= n and 1 <= r and a in H(C,n) and zsummand(a,H(C,n)) -> Zr(bock(C,p),r,j(bock(C,p),a)) and dzero(bock(C,p),r,j(bock(C,p),a)) and not dimg(bock(C,p),r,n-1,j(bock(C,p),a))
prove bockstein      # two proofs (the paper's, and by coordinates), each re-checked
proof bockstein      # each route's script, the tree of lemmas, the axioms used
check bockstein      # the independent checker again
axioms               # every axiom; lemmas: the library
```

`theorem NAME: FORMULA` states a theorem in many-sorted first-order logic
(`forall x y:el, n:int: ...`, `exists`, `and or not -> <->`, `= != <= <
in`, terms with `+ - *` and `^` on integers; sorts `int`, `el`, `set`,
`cpl`, `cx`, `hom`).  `prove NAME [using L ...]` proves it: by a library
theorem with the same statement (every such route, each re-checked), else
by replaying the library's theorem proofs on it (a false variant fails at
the step that needs the missing hypothesis, which is printed), else by the
automation; when all fail it looks for a counterexample in a concrete
model (standard spaces' cochains, homalg's exact Bockstein spectral
sequences, small integers) and says that this is a computation, not a
proof.  JSON: `theorem prove proof proof_check axioms lemmas`; MCP tool
`proofs`.

**Kernel** (`src/prove.h`, `kernel.c`).  Terms and formulas are one
hash-consed DAG in SoA columns with de Bruijn binders; theorems are
sequents in kernel.c's own SoA table and only its rules make them:
assume, axiom, imp/and/or/not/iff introduction and elimination,
false_elim, raa, all_intro (eigenvariable checked), all_elim, ex_intro,
ex_elim (freshness checked), refl, subst (Leibniz), conv (equal normal
forms: commutative-ring polynomials over Z, Z-module linear combinations,
symbols marked hom additive and Z-linear, numerals compared), lin (a
linear combination of equations, with nonnegative numerals on
inequalities), induct (over Z from a base) and hom_decl (a symbol becomes
a homomorphism once both laws are theorems without hypotheses).  Every
theorem records its rule, premises and terms, so a proof is a DAG.
`kcheck.c` replays it in id order with its own substitution, hypothesis
sets and normal forms (canonical strings, not the kernel's id-ordered
DAG).  Integers in normal forms are int64 with overflow checks (an
overflow fails the step; nothing wraps).

**Axioms** (`src/thy_absalg.thy`, the only `def`/`arith` lines;
`axioms` lists them, `axioms NAME` the ones a proof rests on): the
ordered ring Z (`int_discrete` n <= 0 or 1 <= n, `le_antisym`,
`mul_nonneg`, `int_domain`, `pow_zero`, `pow_succ`), and definitions
only: subgroups, sums and intersections of subgroups, direct summands, a
generating a Z (or Z/p^k) summand; exact couples, i^m, the derived couple
construction (D_1..E_1 and page r+1 from page r: D' = i D, E' = ker d /
im d with representatives in E, j'(i x) = [j x], k'[e] = k e), d_r [e] =
0 and [e] in im d_r^m; cochain complexes (Ch, dif, d d = 0), their
cohomology as the quotient Z/B given with its projection (the relation
Cls: five lines), the grading (projections onto C^n, d of degree +1),
H^n, chain maps, short exact sequences of complexes, the induced map
H(f) and the connecting map (lift, apply d, pull back) as relations;
multiplication by p, the quotient complex C/pC (its cochains, kernel and
differential), "of finite type" (here: torsion-free cochains), primes;
and the Bockstein couple bock(C,p): D = H(C), E = H(C/p), i = H(p),
j = H(rho), k = delta (each map named as the value of the relation, whose
totality and single-valuedness are lemmas), E's grading.  Proved, not
assumed: the long exact sequence of any short exact sequence of
complexes (`les_exact`, with H(f) and the connecting map well defined),
that 0 -> C -p-> C -> C/p -> 0 is short exact for a torsion-free C, that
the Bockstein couple is exact (`bock_couple`), i = p on classes,
H^n a direct summand of H* (`H_summand`), the degrees of i, j, k
(`deg_i`, `deg_j`, `deg_k`), and everything about derived couples.

**Automation** (`kauto.c`, results are kernel theorems): linear integer
arithmetic by Fourier-Motzkin with the multipliers as a `lin`
certificate, products of two hypotheses and case splits n <= 0 or 1 <= n
when the rational relaxation is not enough; abelian-group equalities by
elimination on unit pivots (monomials divided, equations lifted through
applications by congruence, integer coefficients that vanish modulo the
integer equations); subgroup membership of linear combinations; and
goal-directed chaining through facts and lemmas with unification modulo
the normal forms (metavariables solved through linear terms), with
existentials opened and arguments rewritten by congruence: enough for
diagram chases.

**Library** (`thy_absalg.thy`, scripts in an Isar-like language: `fix
intro have obtain show contra cases induct`): the closed forms of the
r-th derived couple by induction on r (D_r = i^(r-1) D, Z_r =
k^-1(i^(r-1) D), B_r = j(ker i^(r-1)), j_r(i^(r-1) x) = [j x]); every
derived couple is exact (`derived_exact`: ker j_r = im i_r, im k_r =
ker i_r, ...), d_r d_r = 0, j_r additive and well defined; i^m = p^m on
the Bockstein couple; `bockstein_paper` (the paper's argument: D_r =
p^(r-1)H, j_r(p^(r-1)a) = a-bar, ker j_r = im i_r, im k_r = ker i_r =
p-torsion of D_r, and the valuation argument p^(r-1) = p^r n is
impossible) and `bockstein_coords` (B_(r+1) = j(H[p^r]), so a-bar in
im d_r puts a in H[p^r] + pH, and a's coordinate in H = <a> (+) M gives
1 in pZ); `bockstein_torsion` (a Z/p^k summand's a-bar is nonzero on
E_1..E_k and zero on E_(k+1)); `cyclic_order` (symbolic exponents);
`modular_law`; `short_five_mono`; `snake_ker_exact`.

The statement's d_r and im d_r^(n-1) are those of the total graded
couple, with Edeg(bock(C,p), m, e) meaning e in H^m(C/p); i_r and k_r are
by the definition of the derived couple the restrictions of i and k (only
j_r is new, as a relation on representatives).  Route (b) is a coordinate
argument that shares the closed forms of the derived couples with route
(a); the reduction to elementary complexes (splitting a complex of finite
type, additivity of the spectral sequence) is not formalised.

Tests: `tests/prove.ucas` (the two routes, re-checks, the other theorems,
automation, and four false variants: no summand hypothesis, a Z/p
summand, a Z/p^k summand dying on E_k, an integer inequality; each fails
with the failing step and a counterexample), `tests/json/prove.jsonl`,
`tests/mcp/prove.jsonl`.
