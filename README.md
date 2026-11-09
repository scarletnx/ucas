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
| `complex` (`c128`), `c64` | complex numbers; `c64` rounds both parts of every result to f32.  `z`, `w`, `z1`.., `w1`.. are complex by default unless `x` or `y` is present (see "Complex and hypercomplex numbers") |
| `quat` (`quaternion`), `oct` (`octonion`), `sed` (`sedenion`), `cd(N)` | the Cayley-Dickson algebras of dimension 4, 8, 16 and N = 2..32 |

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
clz ctz popcount sec csc cot atan copysign fma asin acos atan2 sinh cosh
tanh asinh acosh atanh` (also `arcsin arccos arctan arsinh arcosh artanh`;
`45°` is `45*pi/180`), Lambert W (`W(x)`, also `W0`,
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
| `diff F [x]` | symbolic derivative, or the forward difference for an integer parameter; higher, partial and mixed derivatives, gradient, limits, series: see "Derivatives, limits and series" |
| `equiv F G` | equal on F's domain? (with a counterexample) |
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
| `find F [n] [max cost]`, `more [n]` | other implementations, cheapest first under the active cost model; `more` resumes the search |
| `fit G ~ F [metric maxrel\|maxabs\|rms]` | G's extra parameters are constants: choose them to minimise the error against F |
| `error G ~ F` | max/mean relative error, max absolute error, ulp |
| `cost F [model m]`, `use m`, `models` | price F under a cost model |
| `animate exec of [N-bit\|signed N-bit\|uN\|f32] EXPR [lanes 4\|8\|16\|32] [fps F] [seconds S] [random [seed K]] [MODEL] [to NAME]` | EXPR lowered to its serial form (one instruction per operation, with its class and cost under the model) and run in lockstep on a 2x2 quad of lanes (up to a warp of 32), each lane with its own inputs and registers.  A `?:` whose lanes disagree is a branch: the "then" side runs with the other lanes masked, then the "else" side, then reconvergence, so every lane pays for both (keep-up); a uniform condition runs one side.  `barrier()` stalls every lane; an atomic serializes the lanes.  The type phrase applies to every free variable (`32-bit` is u32 unless `signed`); without one it asks, defaulting to f32 when EXPR is real (sin, decimals, pi) and u32 otherwise.  A ranged real input (an approximation's domain) is spread over its range across the lanes |
| `simplify` + a ```` ```spv ```` block or an attached shader (also `SAT simplify`, `optimize this`, `peephole with smt` ...) [`budget S`] | cheaper forms of the module's integer instructions, each proved by SAT: one of its operands (a clamp, min or mask that never bites), a constant (a compare that always holds), or one cheaper instruction (a shift for a multiply or divide by a power of two, a mask for a remainder or an extract at bit 0, a min or max for a clamp with one side never reached, a bitfield extract for a shift and a mask).  A candidate is run first on random inputs with the interpreter's arithmetic, then both sides are bit-blasted (src/sat.c: an and-inverter graph and a CDCL solver) and no input may make them differ.  The operands' definitions are expanded up to 16 levels, phis at selection merges as "one of the incoming values"; loads, loop phis and specialization constants are free inputs, so a proof holds for every value they could take.  Proofs are refinements: where the original is defined, the replacement must be defined and equal, so nothing leans on what SPIR-V leaves undefined (a shift by the width or more, a bitfield past it, a division by zero).  Assumptions in the request narrow the inputs: `n in [0, 32)`, `n in (a, b]`, `n in 0..31`, `1 <= n < 32`, `n < 32` (0 <= n) apply to the value named n or every load of it.  Two-instruction rewrites merge shift chains ((y << 1) << (31 - n) is y << (32 - n) when n >= 1).  A rewrite that fails is shown with the solver's counterexample when it is a near miss ("fails at %n = 0").  GLSL works too (a ```` ```glsl ```` block or a .comp/.frag/.glsl file): a snippet is completed (a #version, undeclared globals declared from their use as buffer members so the optimizer keeps their writes, a compute main() calling each function with its arguments from a buffer), compiled by glslangValidator with debug lines and optimized with `spirv-opt -O` as a real pipeline would (so what simplify finds is beyond that), and each rewrite names the GLSL line it came from; an assumption on a parameter follows it through inlining.  Chains of shifts by constants, adds, subtracts and constant multiplies fold into one multiply (an integer hash's `p -= p << 17; p += p << 7; p += p << 4` is `p *= 4007528593`); those are proved by algebra mod 2^w (a multiplier against a shift-and-add network is slow for SAT) and checked on 1024 random inputs, and the report says which proof each rewrite has.  With an assumption that bounds the only input of a stored or returned value to at most 2^20 values, a range search checks every input: what the value is (floor, round or ceil of a ratio: `(c * 255 + 32895) >> 16` for c in [0, 65535] is round(c * 255 / 65535)) and forms from the families (x + K) >> S, (x +- (x >> S1) + K) >> S (K solved exactly) and t = x + K; (t +- (t >> S1)) >> S that are cheaper, or as cheap without a multiply (`t = c + 128; (t - (t >> 8)) >> 8`); when none is, it says how many forms it tried.  Unoptimized input gets store-to-load forwarding and dead-store removal in each block for per-invocation variables first.  Proved rewrites apply in order, dead instructions go, and the module is written back (`simplified.spvasm`, attached in Discord), with the cost under the model before and after.  The time budget defaults to 6 s |
| `animate exec of best found cos [-pi/2, pi/2] with crit + cheby poly` | an approximation request (best, found, approx, cheby, minimax, taylor ...): it is read as `approximate ...` in English, `approx` runs (asking precision and endpoints as usual), and the best candidate `ap1` is animated.  Options can be joined with `+`, `&`, `and` |
| `animate exec` + a ```` ```spv ```` block | runs SPIR-V assembly (spirv-dis text) on the lanes in SIMT lockstep and animates it like the expression form, the serial form being the executed SPIR-V lines with each lane's values.  Structured control flow with masks: a selection runs both sides when the lanes disagree, then reconverges; a loop iterates while any lane runs; switch, calls, return/kill; OpPhi per lane.  Uniform/StorageBuffer/Workgroup memory is shared (random contents unless `seed`), Function/Private/Input/Output per lane; FragCoord is the quad's pixel centres, invocation ids the lane.  Barriers stall, atomics serialize.  Bitfield extract/insert/reverse, find LSB/MSB, OpSpecConstantOp (folded with the spec constants' defaults) and OpUndef (zero) are handled.  An access outside a variable reads 0 and drops the write, as with robustBufferAccess, and is counted in a note (random inputs are rarely valid data for a decoder).  Long runs (up to 100000 executed instructions) print the first 160 and last 20 lines plus the time by cost class; when the workgroup is wider than the lanes run, a note says so.  In the chat a GIF over 6 MiB is re-rendered at half the frame rate until it fits.  A snippet is completed: types and constants from glslang's names (`%int`, `%v4float`, `%uint_1`, `%float_0_5`, `%int_n1`), the GLSL.std.450 import, an entry point, the function and its return; any other undefined operand is an input typed from its use (its value asked for: random per lane, the lane index, 0, 1; or given as `%19 = 2.5` / `%19 = lane`), or an output when it is only stored to |
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
`wrap` and `range` add interval bounds when sampling, which can prove
that an operation never wraps.

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
`ucas-output.txt`.  Messages from bots (including itself), webhooks and
system users are ignored.

Attachments and replies count as input.  A message's attached files,
and for a Discord reply to someone else's message that message's
```` ``` ```` blocks and files, are added to the request as code blocks,
so `@Wimsy animate exec of this` works on a posted shader.  Binary
SPIR-V goes through `spirv-dis`; GLSL (`.comp`, `.frag`, `.vert`,
`.glsl` ..., the stage from the extension or what the code uses)
through `glslangValidator -V` and then `spirv-dis`; SPIR-V assembly and
other text are added as they are.  Images can't be read (the bot asks
for the code as text).  Files come only from Discord's CDN, at most
256 KiB and 4 per message, and the tools run with CPU and memory
limits.  A short note says what was done with each file.

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
REST request it would make.  `ucas-discord --selftest` checks the
prefix and fence handling, message splitting and the JSON codecs.
`make discord-test` runs both against `discord/tests/*.jsonl`.
