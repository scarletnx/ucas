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
| `error G ~ F` | max/mean relative error, max absolute error, ulp; for a float G also how much of it rounding can be, to first order: the worst case, the high-probability bound and the expected size (see "Probabilistic rounding error") |
| `cost F [model m] [consecutive i]`, `use m`, `models` | price F under a cost model; `consecutive i` names an integer parameter whose values run along the lanes of a warp (see "Cost models") |
| `rop F [model m]` | the graphics fixed-function states (blending, logic ops, stencil ops) that compute the per-cell update F(s, d) of a scatter, each proved and priced: see [Fixed function as compute](#fixed-function-as-compute) |
| `stoke F [steps N] [restarts R] [len L] [seed K] [model m]` | MCMC (Metropolis-Hastings) over straight-line integer programs priced by the model; what reaches distance 0 on the tests is verified on the whole domain (see "Stochastic search") |
| `pit F G [trials N] [seed K]` | polynomial identity testing (Schwartz-Zippel): a probabilistic answer with its error bound (see "Stochastic search") |
| `phash F [seed K] [model m]` | a switch's keys hashed injectively into a table, when the model prices the lookup below the compare chain (see "Stochastic search") |
| `dword F [tol R \| atol T \| ulp X] [model m]` | F's f64 arithmetic over f32, alone or as double-word pairs, when the model prices that below its f64 operations and the derived error bound meets what the result needs (see "fp64 without fp64") |
| `foldproof F [model m]` | the float round trips the model says its driver folds, and their rounding as integer arithmetic no fold reaches, proved equal (see "Driver folds") |
| `banks F [, G ...] [size N] [elem B] [model m]` | the bank-conflict degree of shared-memory index expressions over a warp, and an index swizzle that lowers it, proved to keep the indices apart (see "Memory kinds") |
| `intatomic F [count N] [tol T] [global] [model m]` | a float atomic add as an integer atomic add on fixed point, when its addend is proved exact there (or within tol) and the model prices that lower (see "Memory kinds") |
| `texalu F [tol E] [under m]` | table reads and weighted sums of neighbouring entries as reads through the texture unit, where the model prices them lower, each proved, with what the pipeline needs for it (see "The texture unit as an ALU") |
| `texenc T [rows W] [tol E]` | a constant table as BC4 blocks: endpoints, selectors and bytes, exact or within E, every pair of endpoints tried (see "Tables as compressed textures") |
| `animate exec of [N-bit\|signed N-bit\|uN\|f32] EXPR [lanes 4\|8\|16\|32] [fps F] [seconds S] [random [seed K]] [MODEL] [to NAME]` | EXPR lowered to its serial form (one instruction per operation, with its class and cost under the model) and run in lockstep on a 2x2 quad of lanes (up to a warp of 32), each lane with its own inputs and registers.  A `?:` whose lanes disagree is a branch: the "then" side runs with the other lanes masked, then the "else" side, then reconvergence, so every lane pays for both (keep-up); a uniform condition runs one side.  `barrier()` stalls every lane; an atomic serializes the lanes.  The type phrase applies to every free variable (`32-bit` is u32 unless `signed`); without one it asks, defaulting to f32 when EXPR is real (sin, decimals, pi) and u32 otherwise.  A ranged real input (an approximation's domain) is spread over its range across the lanes |
| `simplify` + a ```` ```spv ```` block or an attached shader (also `SAT simplify`, `optimize this`, `peephole with smt` ...) [`budget S`] | cheaper forms of the module's integer instructions, each proved by SAT: one of its operands (a clamp, min or mask that never bites), a constant (a compare that always holds), or one cheaper instruction (a shift for a multiply or divide by a power of two, a mask for a remainder or an extract at bit 0, a min or max for a clamp with one side never reached, a bitfield extract for a shift and a mask).  A candidate is run first on random inputs with the interpreter's arithmetic, then both sides are bit-blasted (src/sat.c: an and-inverter graph and a CDCL solver) and no input may make them differ.  The operands' definitions are expanded up to 16 levels, phis at selection merges as "one of the incoming values"; loads, loop phis and specialization constants are free inputs, so a proof holds for every value they could take.  Proofs are refinements: where the original is defined, the replacement must be defined and equal, so nothing leans on what SPIR-V leaves undefined (a shift by the width or more, a bitfield past it, a division by zero).  Assumptions in the request narrow the inputs: `n in [0, 32)`, `n in (a, b]`, `n in 0..31`, `1 <= n < 32`, `n < 32` (0 <= n) apply to the value named n or every load of it.  Two-instruction rewrites merge shift chains ((y << 1) << (31 - n) is y << (32 - n) when n >= 1).  A rewrite that fails is shown with the solver's counterexample when it is a near miss ("fails at %n = 0").  GLSL works too (a ```` ```glsl ```` block or a .comp/.frag/.glsl file): a snippet is completed (a #version, undeclared globals declared from their use as buffer members so the optimizer keeps their writes, a compute main() calling each function with its arguments from a buffer), compiled by glslangValidator with debug lines and optimized with `spirv-opt -O` as a real pipeline would (so what simplify finds is beyond that), and each rewrite names the GLSL line it came from; an assumption on a parameter follows it through inlining.  Chains of shifts by constants, adds, subtracts and constant multiplies fold into one multiply (an integer hash's `p -= p << 17; p += p << 7; p += p << 4` is `p *= 4007528593`); those are proved by algebra mod 2^w (a multiplier against a shift-and-add network is slow for SAT) and checked on 1024 random inputs, and the report says which proof each rewrite has.  With an assumption that bounds the only input of a stored or returned value to at most 2^20 values, a range search checks every input: what the value is (floor, round or ceil of a ratio: `(c * 255 + 32895) >> 16` for c in [0, 65535] is round(c * 255 / 65535)) and forms from the families (x + K) >> S, (x +- (x >> S1) + K) >> S (K solved exactly) and t = x + K; (t +- (t >> S1)) >> S that are cheaper, or as cheap without a multiply (`t = c + 128; (t - (t >> 8)) >> 8`); when none is, it says how many forms it tried.  Unoptimized input gets store-to-load forwarding and dead-store removal in each block for per-invocation variables first.  Proved rewrites apply in order, dead instructions go, and the module is written back (`simplified.spvasm`, attached in Discord), with the cost under the model before and after.  Rewrites any driver's back end makes on its own (a multiply, unsigned divide or remainder by a power of two, an extract at bit 0, a shift then a mask, two constant shifts, an identity operand such as x + 0 or x & ~0, constant folding, x - x) are kept in the module but listed apart in one line with their count; the headline cost is what is left beyond spirv-opt -O and those, with the total shown next to it (JSON: `cost_driver_baseline`, `driver_trivial_count`, `driver_trivial_gain`, and `driver_trivial` on each rewrite).  The time budget defaults to 6 s.  Float instructions follow (see [Float simplify](#float-simplify)): each float rewrite comes with a derivation, not a sample; `float`, `exact` (`bit-exact only`), `ulp N` (`within 2 ulp`), `approx` (`allow approximations`), `per component`.  GLSL with functions that call each other also gets a function-level section (see [Function level](#function-level)): `functions` (`function level`, `per function`, `functions only`) runs it alone, `nofunctions` (`without the function level`) skips it.  With spirv-cross installed, the rewritten module and the one spirv-opt -O made are cross-compiled back to GLSL and diffed (see [GLSL view](#glsl-view)).  C goes to the C frontend instead (see [C frontend](#c-frontend)) |
| `animate exec of best found cos [-pi/2, pi/2] with crit + cheby poly` | an approximation request (best, found, approx, cheby, minimax, taylor ...): it is read as `approximate ...` in English, `approx` runs (asking precision and endpoints as usual), and the best candidate `ap1` is animated.  Options can be joined with `+`, `&`, `and` |
| `animate exec` + a ```` ```spv ```` block | runs SPIR-V assembly (spirv-dis text) on the lanes in SIMT lockstep and animates it like the expression form, the serial form being the executed SPIR-V lines with each lane's values.  Structured control flow with masks: a selection runs both sides when the lanes disagree, then reconverges; a loop iterates while any lane runs; switch, calls, return/kill; OpPhi per lane.  Uniform/StorageBuffer/Workgroup memory is shared (random contents unless `seed`), Function/Private/Input/Output per lane; FragCoord is the quad's pixel centres, invocation ids the lane.  Barriers stall, atomics serialize: the integer ones, exchange, compare-exchange, load, store and the float add (priced `atomic_*_f` where the model lists it); another atomic is refused by name; the lanes run are one subgroup for the OpGroupNonUniform instructions (see [Subgroups](#subgroups)).  A dot or matrix product is priced as its multiply-adds (one multiply and K - 1 multiply-adds for each K-term dot), each `fmul`, times `f64` for f64; a transpose only moves.  A texel fetch is priced `tex_fetch`, a storage image's read `image_load`, a filtered read `tex_sample` (`tex_sample_f32` for a texture of f32 texels), each its fallback where the model lists none.  A load or a store of Workgroup memory is priced as its kind (`load_shared`, `store_shared`) plus `bank_conflict` for each address after the first that the lanes put on one bank; under a model that lists no memory kinds it is a `load` or a `store`.  Bitfield extract/insert/reverse, find LSB/MSB, OpSpecConstantOp (folded with the spec constants' defaults) and OpUndef (zero) are handled.  An access outside a variable reads 0 and drops the write, as with robustBufferAccess, and is counted in a note (random inputs are rarely valid data for a decoder).  Long runs (up to 100000 executed instructions) print the first 160 and last 20 lines plus the time by cost class; when the workgroup is wider than the lanes run, a note says so.  In the chat a GIF over 6 MiB is re-rendered at half the frame rate until it fits.  A snippet is completed: types and constants from glslang's names (`%int`, `%v4float`, `%uint_1`, `%float_0_5`, `%int_n1`), the GLSL.std.450 import, an entry point, the function and its return; any other undefined operand is an input typed from its use (its value asked for: random per lane, the lane index, 0, 1; or given as `%19 = 2.5` / `%19 = lane`), or an output when it is only stored to.  A shader with images (textures, samplers, storage images) also runs over a whole image, with image inputs you choose: see [Shaders on images](#shaders-on-images) |
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
| `search walk\|anneal\|cmaes` | the coefficient search after rounding: the ulp walk alone (the default), then simulated annealing, or CMA-ES, over the coefficients' ulp offsets (see "Coefficient search") |
| `budget N`, `seed K` | the search's objective evaluations (2000) and seed (`UCAS_SEED`, else 1) |
| `temp T0[..T1]` | anneal: the temperature's ends, as fractions of the starting error (0.1..0.001) |
| `pop P`, `sigma S` | cmaes: the population (8 + the number of coefficients) and the first step in ulps (2) |
| `norational`, `nosplit`, `noreduce` | skip rational fits, two-piece splits, range reductions |
| `list n`, `points n` | how many to list (default: the Pareto front, at most 8); sample size (default 131072) |
| `approx more [n]` | the next candidates of the last `approx` |

### Coefficient search

Rounding a fit's coefficients to f32 or f64 moves the error off the
optimum, so approx walks each coefficient by a few ulps while the
measured error drops.  That is a coordinate descent: it stops where no
single small step helps.  `search anneal` and `search cmaes` start from
there and move several coefficients at once (src/numsearch.c):

- **anneal**: one coefficient moves by a few ulps; a worse point is
  accepted with probability exp(-dE/T), T falling geometrically from
  `temp` T0 to T1 times the starting error over `budget` evaluations.
- **cmaes**: CMA-ES, a Gaussian over the offsets whose mean, step and
  covariance follow the better half of each generation of `pop` samples,
  each rounded to whole ulps before it is measured.  Linear
  recombination weights and sums of twelve uniforms stand in for the
  usual logarithmic weights and normal deviates, and exp is computed in
  place: no libm function is in the loop, so a seed gives the same run
  everywhere.

The objective is the largest error on the core's sample (3000 points).
What a search finds is measured again on the table's sample, like every
candidate, and kept only when that measure agrees it is better; the
candidate's `search:` line gives the method, the seed, the evaluations,
the error before and after on both samples, and the label: measured
(on every float when the table says "all of them"), not certified.

```
> approx sin(x) in 0..0.7853981 degree 5 as f32 list 2 search anneal
  1   minimax   5     horner+fma  24.2 ulp (25.6) ...
      search: simulated annealing over the coefficients' ulp offsets from the walk's, seed 1, 2000 evaluations on 3000 sample points: 24.94 -> 23.94 ulp there; on the table's measure 25.07 -> 24.16 ulp: kept (measured on that sample, not certified)
  2   rational  5/2   horner+fma  2.38 ulp ...
      search: ... 1.706 -> 1.623 ulp there; on the table's measure 2.491 against the walk's 2.382 ulp: not kept
```

The searches run on the polynomial and rational forms; two-piece splits
and range reductions keep the walk.

### Probabilistic rounding error

`error G ~ F`, for a float G, adds how much of the error rounding can
be (src/rounderr.c); approx gives the same three numbers, in units of u
and in one line, for its best candidate.  Every rounded
operation i returns its exact result times (1 + d_i), |d_i| <= u; to
first order the computed value is the exact one plus sum c_i d_i, where
c_i is the operation's value times the sensitivity of the result to it
(one reverse sweep of the expression at each input).  From the same
c_i, relative to the value and the largest over a sample of inputs:

| line | formula | holds |
|---|---|---|
| worst case | u sum \|c_i\| | always, to first order: every error at its bound and aligned.  For n operations of equal weight it is n u, the classical gamma_n = n u/(1 - n u) |
| probable | lambda u sqrt(sum c_i^2), lambda = 4 | with probability at least 1 - 2 exp(-lambda^2/2) = 0.99933 at one input, if the d_i are independent with mean zero (Hoeffding's inequality; for equal weights Higham and Mary's gamma~_n = lambda sqrt(n) u) |
| expected | u sqrt(sum c_i^2 / 3) | the root mean square if the d_i are independent and uniform in [-u, u] |

The independence is a model, not a theorem (rounding errors are
deterministic), and the probability is over that model at one input,
not over inputs: the report says so.  The probable bound only gains
from about lambda^2 = 16 operations of similar weight on; below that the
line says it is no better than the worst case.  Operations are taken as
correctly rounded (a libm's or a GPU's functions are not always).
Not analysed, and said: bit operations, a bit cast whose bits are used
as an integer or changed before they are read back as a float, a
variable exponent.  A float's bits read back (`asfloat(asuint(x))`),
floor, ceil and copysign are exact.

Choices are where a small error can make a large one: floor, ceil and a
conversion to an integer jump by 1 at the integers, copysign by twice
its magnitude where its sign operand crosses 0, and min, max and `?:`
(on a comparison of floats) take one side or the other.  The sweep
follows the side the exact values take; at each input, where an operand
is within its own first-order error of the point where the choice
changes (a sweep rooted at that operand), the worst case adds what the
other side brings: the jump times the result's sensitivity to it, or
the distance between the sides plus the other side's own error.  The
report counts those inputs and names the first; the probable and
expected lines stay those of the side taken.

```
> error qz ~ qzx       # f32(i32(x * 0.1)) * 10, x in 0..100
    choices: 1 conversion to an integer; at 1 of the 121 inputs one can go the other way, and the worst case adds what the other side brings there (the probable and expected lines do not); the first: the operand of a conversion to an integer, 10, is within its error 5.96e-07 of 10, at x=100
    worst case  0.1 (1.68e+06 u): every error at its bound and aligned, u sum |c_i|
```

tests/roundchoice.ucas has one of each.

Error-free transformations break the independence on purpose: 2Sum
(Knuth's six operations), Fast2Sum (Dekker's three) and 2Prod
(`fma(a, b, -(a*b))`) compute the rounding error of a sum or a product
exactly.  Each one found is taken as exact as a whole (s + t = a + b,
p + e = a b), so its operations add no error, and since the sweep
differentiates through them, the error of the sum or product it
corrects cancels to first order wherever the result adds the error
term back.  They are matched as written, up to negations, and only when
nothing outside them reads their inner values; Fast2Sum's precondition
(the operand taken first has the larger exponent) and 2Prod's (the
product at least 2^-102 for f32, where its error is representable) are
checked at each input of the sample, and where one fails its operations
count as rounded there, with the first such input named.  What the pairs
leave is of order u^2, outside a first-order analysis: `dword` derives
it.  `dsq_dw` above gives 1 u, the last rounding to f32 (its pair is
within 6.2 u^2 of the exact value); counted as independent roundings,
its 13 operations gave 2.8 u.

```
> error dot2 ~ dot      # Dot2 of a*b + c*d: two 2Prod and a 2Sum, the error terms added back
  rounding, first order in u = 2^-24: 6 rounded operations, each taken as correctly rounded, and 7 in error-free transformations; ...
    error-free transformations: 1 2Sum (s + t = a + b, Knuth), 2 2Prod (p + e = a b by an fma while |a b| >= 2^-102); taken as exact, ...
    the preconditions hold at every input of the sample (checked, not proved)
    worst case  5.96e-08 (1 u): every error at its bound and aligned, u sum |c_i|
> error naive ~ dot     # a*b + c*d as written: it cancels
    worst case  0.000862 (1.45e+04 u): every error at its bound and aligned, u sum |c_i|
```

tests/roundeft.ucas has these, a Fast2Sum whose precondition fails at
some inputs, one whose inner value is read outside it (not taken) and a
wrong sign (no transformation).

```
> s20(x: f32 in 1..2) = x+x*x+x+x*x+ ... +x      # twenty additions of x and x*x
> error s20 ~ e20
  max rel error 3.23201e-07 (3.23e-05%) at x=1.533
  mean rel error 9.30717e-08 over 1001 points
  rounding, first order in u = 2^-24: 21 rounded operations, each taken as correctly rounded; relative to the value, the largest over 4096 inputs
    worst case  6.92e-07 (11.6 u): every error at its bound and aligned, u sum |c_i|
    probable    6.74e-07 (11.3 u): with probability >= 0.99933 at one input, 4 u sqrt(sum c_i^2), if the errors are independent with mean 0 (a model; Hoeffding, Higham-Mary)
    expected    9.72e-08 (1.63 u): the root mean square if they are independent and uniform in [-u, u]
```

(The measured maximum, 3.2e-07, is under both bounds, and the measured
mean, 9.3e-08, is where the expected line puts it.)

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
| look, analysis | `show table parse range domain wrap crit equiv fit error cost gates find texalu texenc more approx approx_more` |
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
  `total_average`, and under a model with a warp `warp`: `lanes`,
  `lanes_are`, `points` or `warps`, `all_counted`, `trips_most_average`,
  `p_split`, `exit_per_lane`, `exit_on_vote` (null without `sg_vote`),
  `vote`, `selects_kept`, `best`; or `leave_together`), `average_total`, `selects_not_listed` and
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
| `analyze` | show table parse range domain wrap crit equiv fit error cost gates find texalu texenc more approx approx_more |
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
  sg_ballot: 0, sg_vote: 0, sg_add: 32.625, sg_shuffle: 4, sg_scan: 60.25
  sdiv: 36, sdivk: 21, irem: 32.25, iremk: 13.5
  fuse lop3 = lut 3 : logic
  fuse imad(a: int, b: int, c: int) = a * b + c : imul
  ...
  folds f32 -> f16 -> f32 }
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
cost 90 to 140.  DP4A is the dot4x8 class (see Dot products below), the
subgroup operations the `sg_` classes (see Subgroups below).
The report also has what the classes do not show: textures, the
clustered and partitioned subgroup operations; the fusions it measured
are in the model (below).  `cd
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

Two more forms say what the classes alone cannot:

```
fuse NAME(a: u16, b: u16, c: int) = a * b + c : mul16
fuse NAME(a: float) = min(max(a, 0.0), 1.0) : 0 on fadd fmul
```

An input `uN` (`sN`), N from 1 to 32, is an integer proved inside
[0, 2^N) (inside [-2^(N-1), 2^(N-1))): its range comes from the
function being priced (`cost` and `find`: the parameters' ranges, the
where clause, then intervals over the operations) and from the types
(a `u16`, `x & 0xffff`, `x >> 16` of a u32).  That is how a multiply
of narrow operands gets its own price: the class `mul16` is there for
it, and like the dot forms a model that does not list it cannot use
it (a fusion priced by it is then unavailable).  `on CLASS ...` makes
the fusion a result modifier, the mirror image of `in`: the pattern
folds into the instruction that makes its input when that is an
operation of those classes used only there (a clamp to [0, 1] into a
saturating add, multiply or fma).  SPIR-V simplify prices without
ranges, so a fusion with a proved width does not apply there yet.

gt1030 lists neither: no test in the corpus measures a 16 x 16
multiply-add (XMAD alone) against IMUL or a saturating result against
the clamp, and nothing is priced before it is measured.  With a model
that has them:

```
> model xmad : gt1030 { mul16: 1
    fuse xmad(a: u16, b: u16, c: int) = a * b + c : mul16
    fuse mul16(a: u16, b: u16) = a * b : mul16 }
> m1(a: u32 in 0..60000, b: u32 in 0..60000, c: u32) = a * b + c
> cost m1                                   (3 under gt1030: an IMAD)
cost of m1 under xmad: 1 (2 operations, shared subterms counted once)
  fused by the model (one instruction for several operations):
    xmad             1 x 1     = 1  (a*b + c; saves 3)
> model sat : gt1030 { fuse fsat(a: float) = min(max(a, 0.0), 1.0) : 0 on fadd fmul }
> cost s1(x: f32, y: f32, z: f32) = min(max(x * y + z, 0.0), 1.0)
cost of s1 under sat: 1 (4 operations, shared subterms counted once)
  fused by the model (one instruction for several operations):
    ffma             1 x 1     = 1  (a*b + c, contracted: one rounding; saves 1)
    fsat             1 x 0     = 0  (min(max(a, 0), 1), a result modifier; saves 2)
```

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
For a condition that varies, a model with a `warp` (gt1030: 32 lanes
share a program counter) also gets the branch's *expected* price: the
warps that agree skip the arm they do not take, the ones that split pay
the divergence.

```
      select: +1.625   uniform branch: +6.25   divergent branch: +20.625
      expected branch: -8.73566   (P(true) = 1.526e-05 over all 65536 points; 32 independent lanes split with probability 1 - p^32 - (1 - p)^32 = 0.0004882)
      -> branch (its expected cost)
```

P(true) is the condition's frequency over the domain, uniform, counted
over the parameters it reads (every point when there are at most 65536,
a sample otherwise; a sample that never saw it true uses the bound 3/n,
at 95% confidence, and says so).  The lanes are taken as independent
draws, so a warp splits with probability 1 - p^w - (1 - p)^w, unless
`cost F consecutive i` names an index that runs along the lanes: the
aligned runs of w consecutive values of i are then counted, all of them
when they are few enough (`i < 16000` over 0..32767 splits no warp:
+1.64 against +20.6).  The choice between select and branch follows the
expected price; the divergent price stays in the line as the worst
case.  Loops' selects and models without a warp are priced as before.
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

Under a model with a `warp`, a loop whose lanes can run different trip
counts (its bounds, or its condition through the states it reads, vary
per invocation, and the count varies over the domain) is also priced as
a warp runs it: until its last lane leaves, two ways.  Leaving per lane,
the lanes that left are masked off and wait for the rest (the keep-up),
and a warp whose lanes leave at different counts pays `diverge` once,
reconverging after the loop.  Leaving on a vote, `while sg_any(c)`,
every lane runs every iteration and nothing diverges, but each state the
loop changes takes a select to keep its value in the lanes where c
fails, and the vote is an `sg_vote`.  The warp's count is the most of
its lanes': for independent lanes, from the counts over the domain (the
most of w draws is at most t with probability F(t)^w, and the lanes
agree with probability the sum of P(t)^w), or counted over the aligned
runs of the index `cost F consecutive i` names.  The cheaper way is
marked, so the vote is offered where it is cheaper: clearing the low
bit of 0..15 until none is left takes 0 to 4 trips, few enough that the
keep-up's divergence costs more than a select a trip.

```
> pc(c: u32 in 0..15) = { x = c; while x != 0 { x = x & (x - 1) }; x }
> cost pc
  ...
    in a warp of 32 lanes (independent draws from the domain, all 16 points): it runs until its last lane leaves, 3.873 iterations on average, and its lanes leave at different counts with probability 1
      exit per lane: 0 + 3.873 x 10.625 + 1 x 14.375 diverge (the lanes that left keep up) = 55.5278
      exit on a vote: 0 + 3.873 x (10.625 + 0 sg_vote + 1 x 1.625 select) = 47.4468  (while sg_any(x != 0), a select keeping x where it fails)
      -> exit on a vote
```

`x = n; k = 0; while x > 1 { x = x / 2; k += 1 }` over 1..1000 runs 9
trips in nearly every warp of independent lanes, and leaving per lane
is cheaper (164 against 178.9).  A loop whose condition is uniform, a
vote among them, says that its lanes leave together.

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
  neither form fits.  The key need not be a parameter: a subterm with a
  small proved range (`(v >> 3) & 7`, `v & 255`) is one too, for the
  largest subterms that read the parameters only through it; they are
  evaluated with a fresh variable in its place and the table or the
  multiply-shift is written on the key itself (innermost keys first, so
  the largest map that fits is the one offered).  A map with literal
  arms (`match m { 2 => 28, ... }`) is read as an int where it is the
  whole function.

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

The same pass takes two more roots, reading compares, selects and logic
operations under them as well.  A divisibility test, OpIEqual or
OpINotEqual of a remainder by a constant (one nothing else reads) and a
constant, becomes the multiply and the compare (`x % 3u == 0u` 15.875 ->
5.375 under gt1030; GLSL's `%` on an int is OpSMod, whose remainder is 0
exactly where OpSRem's is, so `sx % 5 == 0` goes too).  A map of one
small-range value, the top of a chain of OpSelect or an OpLogicalOr or
OpLogicalAnd of compares, becomes a table in one constant or a multiply
and a shift, the key's range from the module (`x & 7`) or the request's
assumptions: four compares and selects on `x & 7` 17 -> 7, the
membership test `k == 1u || k == 4u || k == 6u` one bit of 0x52.  What
the rewrite drops but something else reads (a compare the optimizer
shares between two chains) stays, and its price is added to the new
form's.  A chain written as nested `?:` becomes branches and phis, which
this does not read.  These go in the report's "divisibility tests and
maps of one value" section (tests/idmaps_spv.ucas,
tests/idmaps2_spv.ucas).

### Integers as floats, floats as integers

The identities between the two number systems of a 32-bit word are
rewrites of the same kind (src/ifx.c): each is derived, with the
ranges it holds on, and `find` keeps one only when the active model
prices it lower.  All of them are exact, bit for bit (the two floor
forms up to the sign of a zero); f32 only.

| as written | rewritten | holds where | derivation |
|---|---|---|---|
| `f32(i)` | `asfloat(i \| 0x4B000000) - 2^23` | 0 <= i < 2^23 | the bits are those of 2^23 + i (exponent field 150, mantissa i); the subtraction is exact |
| `f32(i)` | `asfloat(i + 0x4B400000) - 1.5*2^23` | -2^22 <= i < 2^22 | the same from 1.5*2^23: the add stays in the mantissa |
| `iN(x)` | `asint(x + 1.5*2^23) - 0x4B400000` | x an integer in [-2^22, 2^22) inside iN | x + 1.5*2^23 is an integer in [2^23, 2^24), exact; its mantissa holds it |
| `floor(x)`, `ceil(x)` | `r > x ? r - 1 : r` (`r < x ? r + 1 : r`), r = (x + 1.5*2^23) - 1.5*2^23 | \|x\| <= 2^22; a zero result may change sign | the add rounds to the nearest integer, ties to even; one compare corrects it |
| `floor(x)`, `ceil(x)` | `x` | x an integer | |
| `f32(iN(x))`, `iN(f32(i))` | `x`, `i` | x an integer inside iN; \|i\| <= 2^24 | both conversions are exact |
| `f32(1 << k)` | `asfloat((k + 127) << 23)` | 1 << k does not wrap | 2^k is an f32: sign 0, exponent field k + 127, mantissa 0 |
| `x * f32(1 << k)` | `asfloat(asint(x) + (k << 23))` | x normal and never 0, x * 2^k < 2^128 | the product keeps the mantissa and is exact; the add reaches only the exponent field |
| `31 - clz(x)` (W-1 for a W-bit x) | `(asint(f32(x)) >> 23) - 127` | 1 <= x < 2^24 | f32(x) is exact there, and its exponent field is floor(log2 x) + 127 |
| integer `+`, `-`, `*` | the same on f32, conversions at the ends | every value in it inside (-2^24, 2^24) | f32 holds those integers: no operation rounds, contracted into an fma or not |

A float is known to be an integer when it is a floor, a ceil, a
converted integer, or a sum, difference, product, min, max or select
of such; its range comes from the parameters' ranges (`x: f32 in
0..1000`) by intervals, widened by one f32 step where an operation
can round.  The last row takes the largest piece of integer
arithmetic whose values all stay below 2^24 (intervals of its own,
which see through a float converted to an integer), uses an operand
that is a converted integer-valued float as the float it was, and
offers its conversions both ways, plain and by the bits; since almost
every function has such a piece, it is listed only when cheaper (the
others say "derived, but not cheaper" as the rewrites above do).

```
> cv(i: u32 in 0..1000) = f32(i)
> find cv 1
find cv: rewrites by identities the model prices lower (cost 3 as written, under gt1030)
  cost 2   asfloat(i | 0x4b000000) - 8388608        the conversion f32(i) by the bits of 2^23 (derived)
      holds where i in [0, 1000], inside [0, 2^23)
      0 <= i < 2^23: i | 0x4B000000 is the bit pattern with sign 0, exponent field 150 and mantissa i, the f32 2^23 + i exactly (one unit in the last place there is 1)
      2^23 + i - 2^23 = i is an integer below 2^24, an f32: the subtraction does not round
      f32(i) is exact below 2^24: both are i
> ld(x: f32 in 1..2, k: u32 in 0..20) = x * f32(1 << k)
> find ld 1
  cost 2   asfloat(asint(x) + (i32(k) << 23))       x*f32(1 << k) as an add of k to the exponent field of x (derived)
      holds where x in [1, 2] (normal, never zero), k in [0, 20], and 2 * 2^20 < 2^128
      ...
> tx(x: f32 in 0..1000, y: f32 in 0..100) = i32(floor(x)) * i32(floor(y)) + 7
> find tx 1
  cost 10  i32(floor(x)*floor(y) + 7)               the integer arithmetic i32(floor(x))*i32(floor(y)) + 7 on f32 (derived)
      holds where every value in it is an integer in [0, 100007], inside (-2^24, 2^24) (interval arithmetic over the parameters' ranges)
      ...
```

Under gt1030 a conversion (3) becomes a logic op and an add (2), a
multiply by 2^k (shift, conversion and multiply: 6) one shift-add on
the bits (2), and integer arithmetic on values that were floats goes
back to floats (15 to 9 above: an FFMA, and the conversions it no
longer needs).  The floor by the magic add costs 7 against 3 and the
exponent field for floor(log2 x) 5 against clz's 4: derived, not
applied.  Under sm86, where conversions and bit counts are quarter
rate on the integer pipe (8), both apply (the log 7 against 10, the
floor 7.4 against 8); under a model whose conversions cost an add
(`model convcheap : gt1030 { conv: 1 }`) none of the conversions
does; integer multiply-adds of integers stay IMADs under gt1030 (3
against 10 with the conversions) and move to FFMA under a model
whose multiply is dear enough (`imul: 12`).  A function with float
parameters and an integer result gets these rewrites from `find`
although its search needs integer parameters.  Not yet: the same in
SPIR-V and GLSL (find only), f16 and f64, a round-to-nearest
operation for the magic add to stand for (ucas has floor, ceil and
the truncating conversion), 48-bit products by 2Prod.

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
  dot products under gt1030 (sums of products laid into lanes; cost 21 as written):
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
becomes `u32(u8(udot4x8(a, b))) + c`.  The cost as written, and the
cost that a dot product has to beat, is the cover's, with the model's
fusions (what `cost` prints): `a*b + c` is one IMAD (3) under gt1030
and the dot with its add (2) still wins, `a*8 + b` is one shift-add
(2) and stays (the dot and the add are 2 too), and under a model with
a 16 x 16 multiply-add at 1 (`mul16`) the product of two bytes stays
an XMAD.  The search itself steers class by class, where a group's
gain shows before the cover hides it inside a multiply-add it breaks.

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

A sum with one lane of a word in it is searched too: `((x >> 8) & 255)
+ y` is `udot4x8(x, 0x100) + y` under gt1030 (2 against 4), the dot
with a one-hot constant word being that byte and the add the sum's own.
A byte read signed is a signed lane as it is, however the sign is
extended: `i32(i8(x >> 8)) + y`, the C idiom `((x << 16) >> 24) + y` on
an i32, `x >> 24` on an i32, and in SPIR-V OpBitFieldSExtract
(`bitfieldExtract` of an int) are `sudot4x8(x, 0x100) + y` and the like:
the sign extension for the price of the dot, and the sum of a word's four
signed bytes is `sudot4x8(x, 0x1010101)` (14 against 1 under gt1030).
The field is read from the word outward (the shifts, a mask, then each
conversion), so a conversion that changes how its bits read (`u8` of an
`i8`) is followed, and one that cannot keep a negative value (`u16` of
an `i8`) ends the field.  Signed and unsigned bytes of one word do not
share a word's side.
With `0x01010101` it is the sum of a word's bytes, with powers of two a
weighted one.  A lone field is not a sum and stays as it is.

### Memory kinds

A model can price memory by kind, and say how its shared memory is
banked:

```
model gt1030 { ...,
  load_shared: 3, store_shared: 3, bank_conflict: 4,
  banks: 32, bank_bytes: 4, warp: 32,
  load_ubo: 4.125, load_ubo_div: 148.125,
  load_stream: 80.625, store_stream: 111,
  atomic_shared_same: 63.75, atomic_shared_f: 24.25,
  atomic_global_f: 138.75, atomic_global_64: 264.625,
  atomic_global_cas: 143.5, atomic_global_stream: 137.5,
  tex_fetch: 43.125, tex_sample: 21.625, tex_sample_f32: 54.75,
  image_load: 31.875 }
```

| class | what | priced as, when the model does not list it |
|---|---|---|
| `load_shared`, `store_shared` | workgroup memory | `load`, `store` |
| `bank_conflict` | each address after the first that a warp reaches on one bank | 0 |
| `load_ubo`, `load_ubo_div` | a uniform block read at a uniform address, at an address that differs per lane | `load`, `load_ubo` |
| `load_stream`, `store_stream` | a DRAM-bound stream | `load`, `store` |
| `atomic_shared_same`, `atomic_shared_f` | every lane on one address; a float add | `atomic_shared` |
| `atomic_global_f`, `_64`, `_cas`, `_stream` | float add, 64-bit add, compare-and-swap, DRAM | `atomic_global` |
| `tex_fetch`, `tex_sample`, `tex_sample_f32`, `image_load` | texelFetch, an explicit-LOD sample (rgba8, rgba32f), imageLoad | `load` (`tex_sample_f32`: `tex_sample`) |

`banks`, `bank_bytes` and `warp` are counts, not prices.  gt1030's values
are the measurements of tests/gt1030 (a step minus the add it contains).
A model that lists none of this (the x86 models) keeps its `load`,
`store` and atomic prices for everything, has no banks, and none of what
follows fires under it; `models` prints only what a model lists.  `cost`
prices a float `atomic_add[_shared]` as `atomic_*_f` where listed.

**Bank conflicts.**  `banks F [, G ...]` takes the index expressions of
the accesses to one shared array.  The first integer parameter that is
not `uniform` is the lane: a warp is `warp` consecutive values of it
(aligned windows of its range), the other parameters are the same for
the whole warp.  The conflict degree is the largest number of distinct
addresses on one bank: for an index a·lane + b two lanes share a bank
when `banks` divides a (l - l'), so warp·gcd(a, banks)/banks lanes do
(printed, and enumerated too); any other index is enumerated over every
warp and every value of the uniform parameters (sampled past 65536
cases, and then nothing is rewritten).  An access costs `load_shared +
(degree - 1) bank_conflict`: 32 lanes on one bank are 3 + 31·4 = 127
under gt1030, the 126.94 measured.  `animate exec` prices a workgroup
access the same way, from the addresses the lanes run with.

```
> row(lane: u32 in 0..31, r: u32 uniform in 0..31) = r * 32 + lane
> col(lane: u32 in 0..31, c: u32 uniform in 0..31) = lane * 32 + c
> banks row, col
  row = 32r + lane
    lane parameter lane; conflict degree 1: ...
    price 3
  col = 32lane + c
    lane parameter lane; conflict degree 32: index = 32 lane + b: two lanes share a bank when 32 divides 32 (l - l'), ...
    price 127
  -> swizzle every index i of the array: i + (i >> 5)
     it is strictly increasing (i + 1 lands after i), so no two indices meet; the array grows to 1055 elements: every index enumerated
     row_sw = ...  conflict degree 1 ...: price 3 + 2 for the swizzle
     col_sw = ...  conflict degree 1 (index = 33 lane + b: ...): price 3 + 2 for the swizzle
     total 10 against 130 (-92.3%); every access of the array takes the swizzle, reads and writes alike
```

A swizzle is searched over the accesses together: the row number `i >>
s` xored into the bank bits, added into them, or one element of padding
after every 2^s (the array grows).  Each candidate's degrees are
enumerated again and its instructions priced by the model's cover with
its fusions (gt1030's shift-and-add makes the padded row cost 2); the
cheapest total is taken when it beats the price as written.  It must keep indices apart,
or a read would no longer find what a write put there: the xor is
proved an involution and the add inverted (bijections of [0, size)),
the padding strictly increasing, by enumeration of every index or by
SAT.  `F_sw` is defined for each access.  Without banks in the model,
or when no candidate is cheaper, nothing changes.

**Integer atomics.**  `intatomic F` takes a function with a float
`atomic_add` (or the addend itself).  Where the model prices a float
atomic add above an integer one (on sm_61 a shared float add is a
compare-and-swap loop), the addend is put on fixed point: its values are
proved to be k·2^g from the structure of the expression (integers and
binary constants as leaves; + - *, and powers of two, keep a grid) with
|k| bounded by the ranges and no f32 value on the way past 24 bits, so
nothing rounds; `count N` addends (1024 by default) must fit 32 bits.
The integer sum is then the exact sum, in any order of arrival, where
the float sum rounds past 24 bits in an order-dependent way.  An addend
on no grid needs `tol T`: it is scaled by the largest 2^s whose sum
fits and rounded to nearest, off by at most N·2^-(s+1) in total.  The
report gives the integer addend (the conversions dropped where the
structure allows: `f32(i) * 0.25` adds `i`), both costs, and what every
other access of the cell becomes; `F_ia` is defined.

```
> w(acc: f32, i: u32 in 0..255) = atomic_add_shared(acc, f32(i) * 0.25)
> intatomic w
  the float add costs 24.25, an integer one 6.625
  the addend is k 2^-2 with |k| <= 255, exactly: ...
  the sum of 1024 of them is at most 261120 2^-2: it fits 32 bits ...
  -> atomic_add_shared(acc, i32(i)): addend and atomic cost 6.625 against 28.25 (-76.5%)
```

**In a shader**, `simplify` adds a memory part after the float part,
silent when it has nothing to say.  An address is uniform when it
depends only on constants, uniform blocks read at uniform addresses and
the workgroup's builtins; invocation builtins, buffer contents and
function parameters vary per lane.  A branch on a varying condition
makes varying what depends on the side a lane took: the phis of the
blocks where two paths from the branch meet that share no other block,
and, when the branch lets lanes leave a loop at different iterations,
every value read after the loop that it changes as it goes round (inside
the loop the lanes still there are on the same iteration).  A phi after a varying branch that it
does not depend on stays uniform when its inputs are: a loop counter
under a varying `if` in its body, an index a uniform `if` chooses.

- A uniform block read at a per-lane address, where `load_ubo_div` is
  above `load`: moving the block to a storage buffer is *proposed*, with
  the host's side (descriptor type, buffer usage) and every member whose
  offset, stride or size changes from std140 to std430.  It is an
  interface change and is never applied.
- A texel fetch, where `tex_sample` is below `tex_fetch`: `textureLod`
  through a nearest sampler at the texel's centre is offered as a
  candidate worth testing, with the proof that the float coordinates
  select the same texel for every size up to 2^20 (p + 0.5 exact, the
  quotient within OpFDiv's 2.5 ulp, the sampler's sub-texel rounding)
  and the sampler state it needs.  Not applied.  A fetch of an image
  whose type holds f32 texels is not offered: its sampled read is
  `tex_sample_f32`.
- A float atomic add on workgroup memory (`OpAtomicFAddEXT`):
  `intatomic`'s analysis of its addend, with the workgroup's size as
  the count when the add runs once an invocation (in the entry point,
  in no loop).  Workgroup memory is the module's own, so the variable
  (an f32 or arrays of them) is rewritten when every add is exact on a
  common grid 2^g, every store puts a value on that grid (a constant,
  or a value `intatomic` proves on one), the largest stored value plus
  every add stays within 32 bits, and the module costs less with each
  load converted back (f32(cell) * 2^g) and each store converted
  (i32(v * 2^-g)): it becomes i32, each add an `OpAtomicIAdd` of the
  integer addend, and with no f32 add left the module stops declaring
  `AtomicFloat32AddEXT`.  Any other access (an exchange, a whole-array
  load) leaves it f32, with the reason.  A stored -0.0 reads back as
  +0.0.  An add into a storage buffer is a *proposal*: the buffer is
  read outside the module, the count is the dispatch's, so the report
  gives the most adds into one cell that stay exact and what the other
  readers and the host convert.
- A workgroup array whose accesses conflict: `banks` on their index
  expressions (the lane is the linear invocation index,
  gl_LocalInvocationIndex, in a workgroup of any shape, and
  gl_LocalInvocationID's components are taken from it: x = i % X,
  y = i / X % Y, z = i / (X Y), x the index itself when the workgroup is
  one-dimensional; a loop counter is uniform with the range its loop
  gives it).  An element of a nested array is read by its index laid
  flat; up to 32 access chains of an array are searched together.  A
  proved swizzle is written into every access chain of the array (the
  array's length grows for a padded row; a nested array's indices are
  taken apart again level by level, a division or remainder by a length
  that is no power of two as one multiply and one shift (the index is
  below the array's size, so idrw.c's M and s serve it), and a padded row
  there must be its own innermost row, which grows by one element while
  the indices stay), and spirv-val is run on the result; the price of
  laying the indices flat and taking them apart is that of what is
  written.  An array read otherwise (structs
  as elements, an access to a row, a chain through another chain, more
  than 32 chains) is said to be not analysed, with the reason.
### Driver folds

A model can say which float round trips its driver drops:

```
model gt1030 { ..., folds f32 -> f16 -> f32 }
model mine : zen4 { folds f32 -> f16 -> f32 }
```

`folds T -> M -> T` (M narrower than T), a line of the model's block like
its `fuse` lines, declares that the conversion to M and back costs
nothing and does not round (`folds none` drops what the parent
declared: sm75, sm86 and rdna2 do, the fold being a measurement of
GP108's driver and not an estimate to carry over): measured on the GT 1030, where
the f32 -> f16 -> f32 round trip is dropped (tests/gt1030, "f16 round
trip").  Nothing in the code names a driver; a model without the line
prices the conversions as written.

`cost` prices F as the driver runs it, the folded round trips gone
before anything is counted (so what they sat between fuses as it
would), and says whether
the program relies on its rounding: proved harmless when the operand is
an M value already (an integer that fits the significand, a value
converted from M), checked on every point of a small domain, and
otherwise searched for an input where the folded program differs.

```
> q(x: f32 in 0.5..8) = f32(f16(x)) * 2
> cost q
cost of q under gt1030: 1 (1 operations, shared subterms counted once)
  fmul             1 x 1     = 1
  the driver folds f32 -> f16 -> f32 (model gt1030): f32(f16(x)) is priced 0 and computes x itself
      warning: q relies on that rounding (quantisation): at x=0.717493892: the program expects 0.717285156, the folded one gives 0.717493892
      `foldproof q` gives the rounding in a form no float fold reaches
> foldproof q
  ...
    fold-proof: asfloat(asuint(x) + 4095 + (asuint(x) >> 13 & 1) & 0xffffe000)
      cost 6 under gt1030 (the round trip as written would be 6; folded it is 0 and does not round)
    proof:
      1. between 2^-14 and 2^16 the f16 values of a binade are the f32 values whose 13 low mantissa
         bits are 0; ... rounding the value to f16 (nearest, ties to even) is rounding
         the pattern to a multiple of 2^13 (nearest, ties to the even quotient)
      2. (b + 4095 + (b >> 13 & 1)) & ~8191 is that rounding of b, for every 32-bit b: against
         q = b >> 13, r = b mod 2^13, (q + (r > 2^12 or r = 2^12 and q odd)) << 13: proved by SAT
      3. the range: every operand is 0 or has 2^-14 <= |x| < 65520 ...
    check (not the proof): 726 bit patterns ... all equal to the evaluator's round trip
  defined q_fp = ...
```

`foldproof F` rewrites every folded round trip of F: the identity where
nothing is lost, otherwise integer rounding of the mantissa bits, which
is IEEE round-to-nearest-even to M bit for bit.  The short form above is
taken when the operand's range keeps every result a normal M value; with
no such range the full form also gives subnormal results (the grid fixed
at 2^(emin - mant), the quotient by a variable shift), infinity past the
largest M value, NaN for NaN and the sign of zero.  The grid argument is
in words; the add-and-mask rounding is proved against its definition by
SAT on every bit pattern, and a labelled secondary check compares the
form with the evaluator's own round trip on ties and their neighbours at
every mantissa bit of every binade.  The function `F_fp` is defined.

In a shader, `simplify` reports a folded `OpFConvert` pair the same way
and, asked with `foldproof` ("simplify, fold-proof"), writes the integer
instructions into the module.

## fp64 without fp64

`dword F` takes a function whose value is an f64 computation (or
`f32(...)` of one) and builds it over f32 in two ways: with f32 alone
(every operation rounded to f32), and as double-word arithmetic, where a
value is an unevaluated sum hi + lo of two f32 with |lo| <= u |hi|, u =
2^-24, on the error-free transformations 2Sum, Fast2Sum and 2Prod (with
fma).  Each candidate gets a bound on its error against the exact value,
and is priced under the model; the cheapest one whose bound meets the
requirement replaces the f64 operations when it costs less than they do.
Under gt1030 (f64 at 1/32 rate, division and sqrt library calls) the
pairs usually win; under an x86 model, where f64 costs what f32 does,
nothing is rewritten.  Both sides are priced by the model's cover with
its fusions: a lone f64 multiply-add that gt1030 fuses (38.375) stays,
its pair and the recombination costing more (50.375); in the pairs a
negation feeding an fma is the free source modifier it is on the card.
No fusion absorbs an operation of an error-free transformation: each is
read more than once.

```
> dsq(x: f32 in 1..2, y: f32 in 3..4) = f32(f64(x) * f64(x) - f64(y) * f64(y))
> dword dsq
dword dsq under gt1030: the f64 arithmetic over f32 (u = 2^-24, u64 = 2^-53)
  as written  cost 73.75   error against the exact value <= 3.997e-16 (3.6 u64) relative
  f32 alone   cost 2       error <= 2.146e-07 (3.6 u) relative: as f32 within 0.5 + 3.6 ulp
  f32 pairs   cost 13      error <= 2.203e-14 (6.2 u^2) relative: as f32 within 0.5 + 3.7e-07 ulp
  required: the result feeds an f32: within 0.5 + 2^-10 ulp of the exact value (ten guard bits; ...)
  -> f32 pairs: cost 13 against 73.75 (-82.4%), 15 f32 operations for 6
  the form:
    t1 = x * x                               !
    ...
    t15 = t3 + t14                           !
    result: t15
    ! 11 operations of the error-free transformations: exact only as written, so they are
      NoContraction in SPIR-V, `precise` in GLSL (no fused multiply-add, no reassociation)
  each value: its range, the derived error bound, the algorithm:
      f64(x)*f64(x)                            [1, 4]                   0 u^2          2Prod (exact)
      f64(y)*f64(y)                            [9, 16]                  0 u^2          2Prod (exact)
      f64(x)*f64(x) - f64(y)*f64(y)            [-15, -5]                6.2 u^2        SloppyDWPlusDW
  derivation ...
  conditions:
    finite inputs (an infinity or a NaN gives NaN where f64 would carry it)
    no overflow: the largest value is 16, below 2^127 (derived from the ranges)
    no underflow: the 2 error-free products stay above 2^-102 (derived from the ranges)
    Fast2Sum's precondition |a| >= |b|: derived in each use from the magnitudes above
  check (sampled, not a proof): 4096 points of the domain, the largest error against the exact value
    (long double) 2.1 u^2 (the bound: 6.2); the f64 program's 0 u64
  defined dsq_dw: the form as a function (cost dsq_dw, error dsq_dw ~ dsq)
> dword dsq model zen4
  -> not rewritten: the form that meets the requirement costs 45, the f64 operations 21 under zen4
```

The requirement is what the result feeds: for `f32(...)` the default is
ten guard bits (within 0.5 + 2^-10 ulp of the exact value), `ulp X`
changes it, and a loose one is met by f32 alone (`dword dsq ulp 4`: cost
2); `tol R` is a relative bound and `atol T` an absolute one, and an f64
result needs one of them (the pair is then recombined as f64(hi) +
f64(lo), or handed over as two f32: an interface change that is
proposed, not made).

The algorithms are those of Joldes, Muller and Popescu, "Tight and
rigorous error bounds for basic building blocks of double-word
arithmetic" (ACM TOMS 44(2), 2017):

| operation | algorithm | f32 operations | derived here | the paper |
|---|---|---:|---:|---:|
| FP + FP, FP x FP | 2Sum, 2Prod | 6, 3 | exact | exact |
| pair + FP | DWPlusFP | 10 | (1 + kappa) u^2 | 2 u^2 |
| pair + pair | SloppyDWPlusDW | 11 | (1 + 2 kappa) u^2 | 3 u^2, same signs |
| pair + pair | AccurateDWPlusDW | 20 | (2 + kappa) u^2 | 3 u^2 |
| pair x FP | DWTimesFP3 | 7 | 2 u^2 | 2 u^2 |
| pair x pair | DWTimesDW3 | 10 | 6 u^2 | 5 u^2 |
| pair / FP | DWDivFP3 | 11 | 7 u^2 | 3 u^2 |
| pair / pair | DWDivDW2 | 15 | 18 u^2 | 15 u^2 |
| sqrt(pair) | one Newton step | 9 | 5.125 u^2 | |
| min, max of pairs | the pair by the sign of (xh - yh) + (xl - yl) | 6 | the larger operand's + 3 u^2 | |

The bounds used are the ones the standard model fl(a op b) = (a op b)(1 +
d), |d| <= u, gives line by line with the transformations exact; the
report prints each derivation.  They are a little above the paper's,
whose proofs are case analyses (an exact subtraction here, a Sterbenz
case there), and an addition's depends on kappa = (|x| + |y|) / |x + y|,
the largest over the parameters' ranges: 1 for operands of one sign.
Where kappa is not bounded (the ranges let a sum cancel) no relative
bound exists, for the f64 program either; the absolute bound, which
every value also carries, then answers `atol`.  The cheaper sloppy
addition is taken while its bound meets the requirement and its
Fast2Sum precondition follows from the ranges; the accurate one
otherwise.

What is stated with a result: the ranges every value takes; no overflow
(derived from the ranges, or left to the caller when a parameter has
none); no underflow of an error term (a product below 2^-102 adds at
most 2^-149); divisors away from 0; a positive operand for sqrt (at 0
the Newton step is 0/0); finite inputs.  Scaling by a power of two,
negation and abs are exact.  min and max choose one of their two pairs:
xh - yh is exact when the high words are within a factor of two
(Sterbenz), and then the difference's sign is the pairs' unless they are
within u^2 (|xh| + |yh|) of each other, where either one is that close
to the right one; farther apart the high words decide.  Two single f32
are compared by the instruction itself.  An f64 parameter is split in place (hi =
f32(v), lo = f32(v - hi), one f64 subtraction), a constant when it is
read; integers up to 24 bits convert exactly and a 32-bit one in two
exact parts; an operation with no pair form (sin, pow ...) stays in f64
and its value is split.  A sampled check against the exact value in long
double is printed as that: a check, not the proof.

`find F` mentions the form in a line when one is cheaper.  In a shader,
`simplify` rewrites the f64 computations that end in a conversion to
f32, of a scalar or of a whole vector (their inputs f32 values and
integers converted to f64, f64 constants; ranges from the request,
`x in [1, 2]`, a vector's range for each of its components): the f32
instructions replace the f64 ones, the operations of the error-free
transformations are decorated NoContraction, the Float64 capability and
type go when nothing uses them, and spirv-as and spirv-val are run on the
result ("spirv-val accepts the written module").  `ulp N` in the request
is the requirement; `exact` rewrites nothing (a pair is not bit-exact).

f64 vectors and matrices are read component by component: the
componentwise operations, a vector times a scalar, `dot`, `length`,
`distance`, `normalize`, a matrix times a vector, a vector or a scalar
times a matrix, matrix products and `transpose`, constructors, swizzles
and extractions.  A result is still one function of at most 8 inputs.

The f64 instructions under a result go only when nothing else reads
them, and the price is of what goes:

- Results that share f64 values are one group with one set of inputs.
  The pair of a shared value is written once where the first result's
  block is the second's too, again where it is not, and the group is
  rewritten when its forms together cost less than the f64 instructions
  they replace, what is shared counted once on both sides ("with %75,
  which shares its f64 values: cost 14 against 109.125").  A group can be
  cheaper where each of its results alone is not.
- An f64 value that something else keeps reading (a store, a result
  that is not rewritten) stays.  The results above it either split it
  (hi = f32(v), lo = f32(v - hi): one f64 subtraction) or compute it
  again in f32 from its own inputs; both plans are priced for the whole
  module and the one that saves more is written ("3 of the 5 f64
  instructions under it stay for what else reads them").
- A group with more than 8 inputs is cut at the values its results
  share: they stay f64 and are split.

`min`, `max` and `clamp` of f64 values are read too (FMin, FMax, NMin,
NMax, FClamp).  An f64 comparison is a result of its own (`dword F`
with a comparison for F; in a shader the ordered compares and
FUnordNotEqual): a bool, not a value with an error to spend, so it is
rewritten only where the answer cannot change.  Either both sides are
exact in f64 (no operation on the way rounds: f32 values converted and
products of two of them, 48 significant bits) and in the form (a
derived error of 0, the error-free products), and the pairs are compared
high word first, then low where the high words tie (a normalized pair's
high word is its value rounded, and rounding keeps order); or the
sides' ranges, widened by the f64 program's errors, do not meet, and the
comparison is a constant.  A sampled check counts the points where the
form and the f64 comparison differ, inside the conditions.  Not
rewritten: results that stay f64.

### Fixed function as compute

A scatter (a histogram, a bitset, a splat) updates a cell once per
contribution: d becomes F(s, d).  As compute that is an atomic, or a
compare-and-swap loop when F is no atomic the target has.  The
graphics pipeline applies such updates in fixed function, a fragment
each: the blender, the logic op, the stencil op.  `rop F` (src/rop.c)
finds the fixed-function states that compute F:

```
> acc(s: u32 in 0..255, d: u32 in 0..1000000) = d + s
> rop acc
rop acc: acc = d + s, applied once per fragment (s what a fragment brings, d the cell's value)
  order: the result does not depend on the order of the fragments: acc(s2, acc(s1, d)) = acc(s1, acc(s2, d)) (proved by SAT)
  fixed-function states that compute it (each a proposal: the pipeline changes):
  blending: VK_BLEND_OP_ADD, source factor ONE, destination factor ONE on a float target (R32_SFLOAT): s + d
      holds where the operands, the factors, both products and the result stay below 2^24 in magnitude (at most 1000255, the result in [0, 1000255], from the parameters' ranges)
      acc and s + d expand to the same polynomial; every value is an integer below 2^24, which f32 holds, so no step of the blender rounds whatever its arithmetic, and the integer result (inside u32: no wrap) is the polynomial's
      price: rop_blend is not priced by gt1030 (not measured): unavailable
      pipeline: VkPipelineColorBlendAttachmentState: blendEnable, colorBlendOp = VK_BLEND_OP_ADD, srcColorBlendFactor = VK_BLEND_FACTOR_ONE, dstColorBlendFactor = VK_BLEND_FACTOR_ONE; an R32_SFLOAT colour attachment; one point per contribution, its position the cell, its colour s
  not in any price: the draw's setup (a render pass, the pipeline, one vertex per contribution) and reading the target back
```

F's parameters are the cell's value (the one named `d` or `dst`, else
the last), what a fragment brings (the first other one), optionally a
second value of the fragment (dual-source blending) and one `uniform`
parameter, a pipeline constant (the blend constant, the stencil
reference); all of one type, u8, u16, u32 or f32.  Ranges come from
the parameters (`s: u32 in 0..255`).

**The catalogue**, each state with its exact result:

| family | states | on | class |
|---|---|---|---|
| logic op | the 16 VkLogicOp: bitwise functions of (s, d) | unsigned integer targets | `rop_logic` |
| stencil op | the 8 VkStencilOp on d (and the reference), under a write mask of the low or the high bits: counters mod 256 or saturating, an invert, a replace | the 8-bit stencil value | `rop_stencil` |
| blending | VK_BLEND_OP_ADD, SUBTRACT, REVERSE_SUBTRACT with every pair of factors among 0, 1, s, d, the constant, the second source and their 1 - x; MIN, MAX | float targets (R32_SFLOAT); with factors ONE on R8/R16_UNORM, the integer n standing for n/(2^w - 1), where the clamp to [0, 1] is a saturation | `rop_blend` |
| advanced blending | MULTIPLY, SCREEN, DARKEN, LIGHTEN, DIFFERENCE, EXCLUSION of VK_EXT_blend_operation_advanced, on opaque colours in [0, 1] | UNORM or float targets | `rop_adv` |

**Nothing is recognised by name.**  Every state is compared with F on
F's domain: integer states on every point or by SAT; a blend equation
by expanding it and F to the same polynomial (so `s + d - s*d` is
found as ADD with ONE and ONE_MINUS_SRC_COLOR, `s + d - 2*s*d` as
ONE_MINUS_DST_COLOR and ONE_MINUS_SRC_COLOR, `d + s*k` with a
uniform k as CONSTANT_COLOR, `s*t + d*(1 - t)` as dual-source
blending), the fewest operations first.  What the equality is worth is
said with it:

- integer data on a float target is exact when the ranges keep the
  operands, the factors, both products and the result below 2^24,
  where no f32 operation can round (and the result inside its uN):
  then the blender's arithmetic does not matter.  Without that bound
  the state is shown as not exact, with the numbers; integer targets
  do not blend;
- one f32 operation (ADD with ONE and ONE, MIN, MAX) is exact if the
  blender rounds to nearest; several are equal as real functions only,
  since Vulkan fixes neither their rounding nor whether the multiply
  and the add are fused;
- on UNORM targets, exact if the conversion rounds to nearest, which
  Vulkan prefers and does not require.

**The order.**  Atomics arrive in any order, fragments in primitive
order.  `rop` decides whether it matters: F(s2, F(s1, d)) against
F(s1, F(s2, d)), on every point or by SAT for integers; for floats
min and max exactly, the rest as real functions (in f32 a sum depends
on the order unless every step is exact, which the integer bound
gives).  A state for an update that depends on the order stands for a
sequential loop over the fragments, not for atomics.

**Prices.**  Each family has a cost class, the price of one fragment
through that state.  No built-in model lists them (the corpus measures
none: tests/gt1030 needs blend, logic-op and stencil state first), and
an unpriced form is unavailable; `rop` says so and still shows the
proof.  With a model that prices one, it is compared with the update
as a global atomic, by the model's memory kinds: `atomic_global` for
an integer add, min, max, and, or, xor, `atomic_global_f` for a float
add, and for anything else `atomic_global_cas` plus the update's own
operations (a compare-and-swap loop); the draw's setup and the
read-back are in no price and are said so.

```
> model fixed : gt1030 { rop_blend: 2, rop_logic: 60 }
> rop acc model fixed
      price: rop_blend 2 a fragment under fixed, against 50 (one global atomic, atomic_global): cheaper
> rop bits model fixed                      (bits(s: u32, d: u32) = d | s)
  logic op: VK_LOGIC_OP_OR on an unsigned integer target (R32_UINT): s | d
      price: rop_logic 60 a fragment under fixed, against 50 (one global atomic, atomic_global): not cheaper
```

A state is always a proposal: it replaces a compute dispatch by a
draw (a render target, the blend or stencil state printed, one point
per contribution whose position is the cell), which the host has to
do.  English: "the blend state for acc", "the logic op for a | b with
u32 a and b", "the fixed-function forms of cnt on fixed", "show me
the blending for acc".  Not yet: simplify does not propose a state
for a module's atomics; the depth test as an argmin, occlusion
queries as counts, MSAA resolve, blits and the other fixed-function
ideas are not in the catalogue; separate alpha, sRGB and the
piecewise advanced operations (overlay, the lights, dodge, burn) are
not either.

### Subgroups

A subgroup is W invocations that run in lockstep (a warp, a wave), lane
0 to lane W - 1.  A value is per lane or shared by the lanes (a
`uniform` parameter, a constant).  These operations read a value in
every lane at once (src/sg.c), each defined exactly for a full
subgroup, all W lanes active:

| op | value | SPIR-V, GLSL |
|---|---|---|
| `sg_ballot(p)` | the word whose bit l is p in lane l: u32, u64 when W > 32 | OpGroupNonUniformBallot, `subgroupBallot` |
| `sg_any(p)`, `sg_all(p)` | p in some lane, in every lane | OpGroupNonUniformAny/All |
| `sg_add(v)` | the sum of v over the lanes, in v's type (wrapping) | OpGroupNonUniformIAdd Reduce, `subgroupAdd` |
| `sg_shuffle(v, i)` | v in lane i; undefined when i is not in 0..W-1 | OpGroupNonUniformShuffle |
| `sg_scan(v)` | in lane l, the sum of v over lanes 0..l | OpGroupNonUniformIAdd InclusiveScan, `subgroupInclusiveAdd` |

and, written with them, `sg_exclusive(v)` = `sg_scan(v) - v`,
`sg_shuffle_xor(v, k)` = `sg_shuffle(v, lane xor k)`, `sg_broadcast(v,
i)`, `sg_broadcast_first(v)` (lane 0), `sg_rotate(v, d)` =
`sg_shuffle(v, (lane + d) % W)`, and the segmented scan `sg_segscan(v,
h)`, which restarts at the lanes where h holds (the heads): in lane l
the sum of v over the lanes from the last head at or below l (lane 0
when none is) to l, written `sg_scan(v) - sg_shuffle(sg_exclusive(v),
s)` with s the highest set bit, at or below l, of the heads' ballot
with lane 0's bit added, and `sg_sort(v)`, in lane l the l-th smallest
of v over the lanes (integers, W a power of two), written as Batcher's
bitonic network: for k = 2, 4 ... W and then j = k/2 ... 1, a lane and
lane xor j compare and exchange, the lane keeping the smaller when
`(lane & j) == 0` and `(lane & k) == 0` agree; m(m + 1)/2 steps of a
shuffle, a min and a max for W = 2^m (3 for 4 lanes, 15 for 32).
GLSL's names are read as well (`subgroupBallot` ... `subgroupRotate`).
Sums are of integers: the order of a float sum is the implementation's,
so it has no exact value and is refused.

In an operand that is a value per lane, `lane` is the lane's index and
everything else is shared by the lanes, so the lanes' inputs are
written as a function of the lane: a bit or a field of a shared word, a
table's entry.  That is what makes the operations exact here: spelled
out lane by lane they are ordinary expressions, which is how they are
evaluated, bounded and bit-blasted, so `range`, `equiv`, `domain`,
`wrap` and `find` work on them as on anything else; the node keeps the
operation for printing and for the price.  A parameter that is not
`uniform` has a value in each lane, which ucas does not know for the
other lanes: read across the lanes (`sg_add(c)`, `sg_shuffle(x, 3)`)
it is taken as the value in the lane read, priced at the worst (each of
the W lanes read through a shuffle, a load under a model without one,
and a `?:` on what it gives as a divergent branch, the report naming
the parameter), and it has no exact value: `show`, `range`, `equiv`,
`domain`, `wrap` and `find` refuse it saying so.  A call passes it on:
`q(5)` with q's x read across the lanes is 5 in every lane, and a
uniform argument is the same in each.

`sg_active(m)`, m a uniform parameter of the function, says which lanes
run (bit l for lane l) for the function's subgroup operations after it;
its value is m's low W bits, the ballot of true.  A ballot has no bit
of an inactive lane, `sg_any`, `sg_all`, a sum and a scan skip it, and
a shuffle that reads one is undefined (`domain` says which lane it
needs active); a segmented scan is refused under it for now.  The
operations are written with the mask (a test in the operand, a select
around a shuffle), so `find`'s forms are checked for every mask and say
so ("holds with the lanes active in m"), and cost prices those tests as
written, where the hardware masks the lanes for nothing.  A mask that
is not a uniform parameter, or one after the operations it would mask,
is refused.

```
> ac3(m: u8 uniform, lane: u32 in 0..7) = { sg_active(m); sg_shuffle(lane * 10, 2) }
> show ac3(0x04, 5)
_() = 20
> show ac3(0x01, 5)
_() = undefined
> domain ac3
  requires lane 2 is active in u32(m)  (sg_active: a shuffle reads it)
```

W is the range of the function's
`lane` parameter (`lane: u32 in 0..7` is 8 lanes; such a function is
itself a value per lane), else the active model's `warp: N` (the count
the memory kinds use, above); without either the operations are
refused.

```
> show sg_ballot(lane % 3 == 0)
_() = 1227133513
> b(x: u32 uniform) = sg_ballot(((x >> lane) & 1) != 0)
> equiv b id                                 (id(x) = x)
b and id are equal on the domain of b (proved by SAT: 33 AIG nodes, 0 conflicts)
> s8(x: u8 uniform, lane: u32 in 0..7) = sg_scan((x >> lane) & 3)
> show s8(0xb5, 3)
_() = 6
> sh(i: u32 in 0..40) = sg_shuffle(lane + 100, i)
> domain sh
  requires 0 <= i < 32  (a lane of the subgroup)
```

**Cost classes.**  `sg_ballot`, `sg_vote` (any, all), `sg_add`,
`sg_shuffle`, `sg_scan`; like the dot forms, a model that does not list
one cannot use it (`sg_scan: none` takes one away).  An operation costs its class once and its operand
once in each lane: a value computed in every lane is priced once,
whether an operation reads it across the lanes or the lane uses it
itself.  A ballot, a vote and a sum are one value for the subgroup, so
a `?:` on them is priced as a uniform branch, and the lanes of a loop
that leaves on a vote leave together (see [Rewrites the prices make
worthwhile](#rewrites-the-prices-make-worthwhile) for a loop priced
both ways).  gt1030, measured
(tests/gt1030, each step less its add): a shuffle 4.00 (by xor,
broadcast, rotate and the quad swap 4.00 to 4.07 with their add, so
those are priced one logic or add above their measurement), the sum
32.63, the inclusive scan 60.24.  The ballot and the votes fit below
zero (`ballot((a & 1) != 0) + b` is 3.65 where the and, the compare and
the add alone are 4.375: the compare's bool is never made), so they are
priced 0 and their compare as a compare, 0.7 above the measurement.
The estimated models (sm75, sm86, rdna2) inherit these unchanged.

**What find derives.**  Each form is built by a recogniser, priced as
one lane runs it, kept only when the whole function gets cheaper under
the active model, and printed with its derivation and what it holds
under; the whole is then checked against the function on its domain
(every point, SAT, or a sample).

- A sum or a scan of a value that is 0 or 1 in every lane (its range,
  by intervals) counts lanes: `popcount(sg_ballot(v != 0))`, and for
  the scan the ballot's bits 0..lane, `& ((2 << lane) - 1)`.
- A segmented scan of a value that is 0 or 1 in every lane counts the
  ballot's bits from the head's to the lane's, `popcount(sg_ballot(v
  != 0) & ((2 << lane) - 1) & (~0 << s))`: no scan and no shuffle
  (82.625 -> 25.75 under gt1030, 8 lanes); of anything else it keeps
  its scan and shuffle, and the scan the forms below.
- A scan of anything is a ladder of ceil(log2 W) shuffles and adds
  (Hillis and Steele): after the step for d a lane holds the sum of the
  2d lanes ending at it.  Proved for any value per lane: with a free
  word in each lane, every lane's result is the words of lanes 0..lane,
  each once.
- A constant table of at most W entries can sit one entry per lane (the
  table read once in each lane, at the lane's index) and be read by a
  shuffle: no memory, no bank conflicts.  Every lookup of the table or
  none.
- A GF(2)-linear function of one shared word, bit by bit: lane i works
  out bit i of the result, the parity of x & row[i], and the ballot
  puts the 32 bits together.  Linearity is derived from the structure
  (xor, not, shifts and masks by constants, conversions, an or or a sum
  of fields that share no bit).  When every row has one bit the lane
  takes that bit of the word, its position `lane + b`, `b - lane` or
  `lane xor b` when the rows follow one of those and a table otherwise.
- A table read at the lane's index (the register above, a bit slice's
  rows or shifts) whose entries are a closed form of the lane, `c + k
  lane` (every entry checked) or one of find's maps of one small-range
  value (a table in one constant, read at `lane & (2^b - 1)` so that
  every lane shifts inside the word; a multiply and a shift), is that
  form: the lane's index costs no load.  The odd bits of a word into a
  byte shift by `2 lane + 1` (8.375 under a model with 8 lanes, 13.25
  with the table).  For a register the same form at each lookup's own
  index needs no shuffle at all; where that is cheaper the report says
  so with its price, and the register stays the load.
- One function of a small-range value read at several keys, `h(a) +
  h(b) + ...` with h written out at each (the subterms that read one
  parameter each and become the same node with one variable put for
  it), is tabulated at every point of the keys' range and then held one
  entry per lane as above: five reads of `(x*x + 3) % 11`, x in 0..15,
  86.5 -> 30.875 under gt1030.  The dearest such function is tried, and
  kept only when the register makes the whole cheaper.
- An order statistic of a value per lane v, however it is written (a
  loop over the lanes counting with ballots, a chain of selections):
  the least, a median, the greatest or any k-th smallest, or v sorted
  ascending or descending along the lanes (the top k in the first k
  lanes).  It is told by evaluating the function beside the lanes'
  values sorted, at the domain's points (all of them when there are at
  most 1024, a sample otherwise), and written as `sg_sort`'s network,
  read in the lane itself or by one shuffle: a median of 4 nibbles
  counted with ballots, 133.5 -> 43 under gt1030, of 32 lanes 1196 ->
  172.  The network sorts every input: by the 0-1 principle it is
  enough that it sorts every input of 0s and 1s, and after the stage
  for k/2 the halves of a block of k lanes are sorted in opposite
  directions, so such a block is 0^a 1^b 0^c; the stage's merge is run
  on every one of those, at a block of each direction (27 blocks for 4
  lanes, 999 for 32).

```
> f1(x: u8 uniform, lane: u32 in 0..7) = sg_scan((x >> lane) & 1)
> find f1 1                                  (gt1030)
  subgroup forms under gt1030 (each derived, with what it holds under; cost 63.25 as written):
  cost 13.375 u8(popcount(sg_ballot(x >> lane & 1 != 0) & (2 << lane) - 1))
           sg_scan(x >> lane & 1): x >> lane & 1 is 0 or 1 in every lane (its range, by intervals), so ...
           holds in a full subgroup (all 8 lanes active); needs ballot (GL_KHR_shader_subgroup_ballot)
           checked on every point of the domain
> find s8 1
  cost 33  (x >> lane & 3) + (lane >= 1 ? sg_shuffle(t1, lane - 1) : 0) + (lane >= 2 ? sg_shuffle(t2, lane - 2) : 0) + ...
> t3(i: u32 in 0..7, j: u32 in 0..7, k: u32 in 0..7) = T(i) + T(j) * T(k)       (T an 8-entry table)
> find t3 1
  cost 21.875 sg_shuffle(t1, i) + sg_shuffle(t1, j)*sg_shuffle(t1, k) where t1 = [3, 1, 4, 1, 5, 9, 2, 6, 0, ...][lane]
> rev32(x: u32 uniform) = { a = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1); ... (d >> 16) | (d << 16) }
> find rev32 1                               (29 as written: five swaps)
  cost 6.375 sg_ballot(x >> 31 - lane & 1 != 0)
```

The table read once (6.875) is dearer as a register (10.875) and stays;
under a model without the operations nothing is proposed; under one
where they are dear (`model dear : gt1030 { sg_shuffle: 20 }`) find
says "derived, but not cheaper" with both prices; a word that is not
`uniform` is not sliced.

The carry word of a wide add, one word per lane, is one add of two
ballots: with g the lanes whose sum carries out and p those that pass
a carry on, the carries into the lanes are `((g << 1) + p) xor p`
(`equiv` against the bit-serial recurrence proves it, where g & p ==
0).  tests/subgroup.ucas adds two words as lanes of limbs this way and
proves the result equal to `a + b`.

**SPIR-V** (src/spv_sg.inc).  `animate exec` runs the OpGroupNonUniform
instructions of Subgroup scope: the lanes it runs are one subgroup
(`SubgroupSize` is their number, `SubgroupLocalInvocationId` and the
`Subgroup*Mask` built-ins follow the lane), and an instruction reads
its operand in the lanes that are active, as the specification says, so
a lane a branch left out takes no part.  A read of a lane that is not
active or not in the subgroup (a shuffle's, a broadcast's, a bit search
in an empty ballot) is undefined; it reads 0 and a note counts it.
Each instruction is priced as the operation above it stands for, the
way find prices that operation: the ballot, any, all, the sum, the
inclusive scan and the shuffle are their classes; the exclusive scan is
the inclusive one less the lane's value (one add more); a shuffle by
xor, up or down is the shuffle and its index; all-equal is a broadcast,
a compare and all; elect is the lane against the ballot's lowest bit; a
ballot's bit count is popcount, with the mask `(2 << lane) - 1` or `(1
<< lane) - 1` for the scans.  A product, a min, a max, a bitwise or a
clustered sum over the lanes has no operation yet: it runs, at cost 0,
and a note names it; so does one whose class the model does not list.

`simplify` writes them.  A sum or a scan over the lanes
(OpGroupNonUniformIAdd, Reduce, InclusiveScan or ExclusiveScan) of a
value that is 0 or 1 in every lane (a select of 1 and 0 by a condition,
or anything whose range by intervals is [0, 1]: `x & 1`, a one-bit
field) is the number of the lanes it covers where the value is not 0:
OpGroupNonUniformBallotBitCount, with the same group operation, of the
ballot of `v != 0` (of the condition itself for a select).  That holds
for any subgroup size and any set of active lanes (both sides count the
active lanes only), so nothing about the pipeline changes; the module
gets `OpCapability GroupNonUniformBallot` (and a uvec4 type) when it
lacks them.  It is written when the model prices the compare, the
ballot and the count below the sum (and the select that goes dead);
under gt1030 a stream compaction's offset, `subgroupExclusiveAdd(x > 5u
? 1u : 0u)`, goes from 62.875 to 7.  Otherwise each is listed with both
prices, or with the class the model lacks, and left
(tests/subgroup_spv.ucas).

Not there yet: the forms that need the subgroup's size fixed and all
its lanes active (a scan as a ladder of shuffles, a table in a
register) are not written into a module, since a Vulkan pipeline fixes
the size (`requiredSubgroupSize`) outside it; a segmented scan under
`sg_active`; an exact value for inputs read in the other lanes (they
need more than a function's 8 parameters).

### A catalogue of instructions

Targets have single instructions for things the base operations take
several steps to say.  src/inst.c lists them as operations of the
language, each defined by an expression over the base operations: that
expression is its exact semantics, and it is how the operation is
evaluated, bounded and bit-blasted (it is spelled out first, as the
subgroup operations are), so `range`, `equiv`, `domain` and `find`
handle it like its definition.  The node keeps the operation for
printing and for the price.

| op | value (32-bit words) | instructions |
|---|---|---|
| `bfe(x, p, n)` | bits p..p+n-1 of x, zero-extended; n = 0 is 0; undefined when p + n > 32 | OpBitFieldUExtract; BFE; v_bfe_u32 |
| `sbfe(x, p, n)` | the same field sign-extended (i32) | OpBitFieldSExtract; BFE; v_bfe_i32 |
| `brev(x)` | bit i is bit 31 - i of x | OpBitReverse; BREV; v_bfrev_b32 |
| `mulhi(a, b)`, `smulhi(a, b)` | the high 32 bits of the 64-bit product of two u32, of two i32 | OpUMulExtended, OpSMulExtended; IMAD.HI; v_mul_hi_u32/i32 |
| `funl(hi, lo, k)` | the high word of (hi:lo) << k, k in 0..31; `funl(x, x, k)` is a rotate | SHF.L |
| `funr(hi, lo, k)` | the low word of (hi:lo) >> k, k in 0..31 | SHF.R; v_alignbit_b32 (v_alignbyte with k = 8n) |
| `med3(a, b, c)` | the median of three integers; with lo <= hi, `med3(x, lo, hi)` is x clamped | v_med3_i32/u32; SClamp, UClamp where lo <= hi |
| `usad4x8(a, b, c)` | c plus the sum of the absolute differences of the four bytes of a and b | v_sad_u8 |
| `byteperm(a, b, s)` | each byte one of the eight of b:a: nibble i of s picks result byte i by its low three bits, and its top bit asks for eight copies of that byte's sign bit | PRMT (the form without a mode) |
| `bfi(base, ins, p, n)` | base with bits p..p+n-1 replaced by the low n bits of ins; p and n constant, p + n <= 32 | OpBitFieldInsert; BFI; v_bfi_b32 with its mask |
| `mulwide(a, b)`, `madwide(a, b, c)` | the 64-bit product of two u32 (u64), and that plus the u64 c (wrapping) | IMAD.WIDE.U32; v_mad_u64_u32 |

**Cost classes.**  `bfe` (both fields), `brev`, `mulhi` (both),
`funnel` (both), `med3`, `sad4x8`, `byteperm`, `bfi`, `mulwide` (both).  A model that does not
list one cannot use it, and no built-in model lists any: none is
measured yet (the GT 1030 corpus has no test for them), so under
gt1030, sm75, sm86 and rdna2 they are unavailable and nothing is
rewritten into them.  `model mine : gt1030 { bfe: 1, funnel: 1 }` is
how a measured price gets in.

**What find offers.**  The patterns the instructions are, each replaced
where the active model has the instruction and the whole function gets
cheaper, then checked on the domain (src/inst.c, `pattern`):

- `(w >> k) & (2^n - 1)` and `(w << j) >> k`: `bfe`; in i32, where the
  right shift keeps the sign: `sbfe`;
- `(a << k) | (b >> (32 - k))`, or with xor or + (the two fields share
  no bit): `funl(a, b, k)`;
- `u32((u64(a) * u64(b)) >> k)` with k >= 32, and the signed one:
  `mulhi(a, b) >> (k - 32)`: a division by a constant as compilers
  write it;
- `max(lo, min(v, hi))` and `min(hi, max(v, lo))` where the ranges
  prove lo <= hi: `med3(v, lo, hi)`;
- a subterm that permutes the bits of one word, read off its GF(2)
  matrix (derived from its structure, as for the subgroups): the
  reversal is `brev`, a rotation `funl(w, w, k)`, whole bytes moved
  `byteperm(w, w, s)`;
- a u32 of one or two words whose every byte is a byte of b:a, a byte
  of zeros (b is then 0) or eight copies of a byte's sign: `byteperm(a,
  b, s)`, each byte's source told at 48 random points (the check on the
  domain then decides): bytes of two words, `(a & 0xff) | (b & 0xff00)
  | ...`, and a byte sign-extended;
- `(base & ~m) | field`, or with xor or +, m the bits p..p+n-1 and the
  field `(ins << p) & m`, `(ins & (2^n - 1)) << p`, or `ins << p` where
  ins's range keeps it in n bits: `bfi(base, ins, p, n)`;
- `u64(a) * u64(b)`, each operand a word or narrower widened, a
  constant below 2^32 or a value whose range stays below it:
  `mulwide(a, b)`, and plus a u64 `madwide(a, b, c)`; the low word `a *
  b` of a product whose high word the function also reads (`mulhi(a,
  b)` or its pattern): both words from one `mulwide`;
- a sum with the four bytes' differences of two words in it, each
  `max(p, q) - min(p, q)` or `abs(p - q)` of the bytes as shifts and
  masks or `bfe`: `usad4x8(a, b, c)`, c the rest of the sum;
- a word of a 64-bit shift of two words, `u32((hi:lo) >> k)` and
  `u32(((hi:lo) << k) >> 32)` with hi:lo written `(u64(hi) << 32) |
  u64(lo)` and k in 1..31: `funr(hi, lo, k)`, `funl(hi, lo, k)`.

```
> model cat : gt1030 { bfe: 1, brev: 1, mulhi: 1, funnel: 1, med3: 1, sad4x8: 1, byteperm: 1 }
> use cat
> mix(x: u32, y: u32) = ((x << 4) >> 20) + rev32(y)      (rev32: the reversal as five swaps)
> find mix 1
  instructions under cat (each for the pattern it is; cost 33 as written):
  cost 3   bfe(x, 16, 12) + brev(y)
           ...: bit i is bit 31 - i of y (linear over GF(2) by its structure, every row one bit)
           x << 4 >> 0x14: bits 16..27 of x
           checked by SAT
> d5(a: u32) = u32((u64(a) * 0xcccccccd) >> 34)
> find d5 1
  cost 3   mulhi(3435973837, a) >> 2
> cl(x: i32, lo: i32 in -10..5, hi: i32 in 5..90) = max(lo, min(x, hi))
> find cl 1
  cost 1   med3(x, lo, hi)
           max(lo, min(x, hi)): x clamped to lo..hi, which is their median since lo <= hi (5 <= 5, by intervals)
```

With hi in 4..90 the bounds may cross and the clamp stays; under gt1030
nothing is offered.

The rest of the instructions a model may want, and where each stands:

| instruction | here |
|---|---|
| ISCADD, LEA, IADD3, XMAD, LOP3, the `.SAT`/`\|x\|`/`-x` operand modifiers | fusions, declared by the model (`fuse iadd3(a: int, b: int, c: int) = a + b + c : iadd`); gt1030 has the ones its corpus measured |
| POPC, FLO, v_ffbh/v_ffbl | `popcount`, `clz`, `ctz` (class `bitcnt`) |
| DP4A, v_dot4, v_dot8 | the dot forms |
| REDUX, DPP row shifts and broadcasts, ds_swizzle, ds_bpermute, v_permlane16 | the subgroup operations' classes; reductions other than add, and a DPP shuffle folded into an operand, are not there yet |
| IMAD.WIDE.S32, v_mad_i64_i32 | the signed wide product: not an operation yet |
| v_perm_b32 | its selector is a byte per result byte with its own encoding, not `byteperm`'s nibbles: no operation yet |
| VADD4, VMNMX, VABSDIFF4 without the sum, v_msad, v_qsad, v_lerp_u8, v_pk_* | not yet |
| v_cubeid/sc/tc/ma, v_trig_preop, v_ldexp, v_frexp, med3 of floats (its NaN rule) | float instructions: not yet |
| FSWZADD, the quad swaps | subgroup (quad) operations: not yet |
| the SALU as a parallel pipe, Intel's region addressing | need pipes and operand modes in the model, which prices issue cost per class only |

`simplify` on SPIR-V has its own bitfield-extract rewrite (priced by
its instruction classes); it does not write these operations yet.

### Functions as tables

A function of few input bits can be one load.  find's closed forms
already offer the plain lookup table for one parameter of at most 256
values; src/tab.c is the general case.  When the parameters the function
reads have, by their ranges and the where clause, at most 2^16 points
between them, the function is evaluated at every one and becomes a
table read at an index packed from the parameters: fields side by side
when every range is a power of two (`a | b << 4`), mixed radix otherwise
(`i + 16(m - 1)`).  It is offered only where the active model prices
the load and the index below the function as written.  The proof is
the tabulation: entry k is the function at point k, every point
evaluated (0 where the function is undefined or the where clause
fails, which a refinement allows), and the check on the domain
afterwards is exhaustive.

```
> bc(i: u32 in 0..15, m: u32 in 1..15) = (i*64 + m/2)/m
> find bc 1                                  (gt1030: load 6.875)
  a table under gt1030 (the function at every point of its domain; cost 32.5 as written):
  cost 9.875 [0, 64, 128, 192, 256, 320, ... (859 more characters)
           240 entries of u32 (960 bytes of constant data), one for each point of i in 0..15, m in 1..15; entry k is the function at the point with index k = i + 16(m - 1), all 240 evaluated
           priced as gt1030's load, a cache-resident access: the table has to stay in cache for that
           checked on every point of the domain
> big(x: u16) = (x/7 + 3)/5
> find big 1
  cost 6.875 [0, 0, 0, ...
           65536 entries of u16 (131072 bytes of constant data), ...
```

The price is the model's `load`, which gt1030 measured on a
cache-resident access (a stream from DRAM costs 90 there): the report
gives the table's size so that it can be judged against that.  `a + b`
stays (1 against 9.875), a domain of more than 2^16 points is not
tabulated, and under a model whose loads are dear (`model dearload :
gt1030 { load: 60 }`) nothing is offered.

A table of at most W entries can also sit in a register of the
subgroup (Subgroups, above); that form is offered for a table the
function reads several times, since read once the load into the lanes
costs more than the lookup it saves.

A function of one parameter whose range is too large for a table can
have one for part of it: `x < K ? T[x - lo] : f(x)`, T the function at
every point of lo..K-1 (each evaluated, so the select is the function
everywhere).  Its price is an expectation over the range, each value as
likely: a select computes both sides; on a `uniform` parameter the warp
takes one side, the table's with probability p = P(x < K); on one with
a value in each lane a warp splits with the probability `cost` uses for
a ?: (stoch.c) and then pays both sides and the divergence.  K is
searched over the table sizes 2^k up to 2^16, and the cheapest is
offered when it is below the function as written: `(x*x % 1000003)/7 +
x % 13` with a uniform x in 0..99999 is 26.83 expected against 39.75
(p = 0.655), and with a value in each lane nothing (a warp of 32 almost
always splits).  Not there yet: other distributions than the uniform
one, a texture fetch as the table (no measured price), wide loads, a
shared-memory table read by broadcast.

### The texture unit as an ALU

A sampler addresses, converts and filters in fixed function.  Those
are operations of the language, with exact semantics (src/texalu.c),
on a table `T = [t0, t1, ...]` of n texels:

| op | value |
|---|---|
| `fetch_repeat(T, i)` | `T[i mod n]` (floored: REPEAT) |
| `fetch_mirror(T, i)` | `T[min(r, 2n - 1 - r)]`, r = i mod 2n (MIRRORED_REPEAT) |
| `fetch_clamp(T, i)` | `T[min(max(i, 0), n - 1)]` (CLAMP_TO_EDGE) |
| `fetch_mirror_clamp(T, i)` | `T[min(max(i, -1 - i), n - 1)]` (MIRROR_CLAMP_TO_EDGE) |
| `fetch_border(T, i, k)` | `0 <= i < n ? T[i] : k` (CLAMP_TO_BORDER) |
| `tex_linearB(T, x)` | s = x - 1/2, i0 = floor(s), a = floor((s - i0) 2^B + 1/2) / 2^B: `(1 - a) T[c(i0)] + a T[c(i0 + 1)]`, c the clamp to the table |
| `tex_bilinearB(T, W, x, y)` | the same on both axes of the table read as rows of W texels: four texels, their weights the products of the two axes' |

The `fetch_` reads take an integer texel index of any type and are
total (Vulkan's wrapping of a nearest read's integer coordinate).  The
filters take texel coordinates (texel j's centre at j + 1/2,
unnormalizedCoordinates), their texels are f32, their weights are on
the sub-texel grid of B bits (1 to 14; `tex_linear8`: 1/256), and the
weighted sum is exact and rounded once to f32.  They fold, evaluate,
have interval bounds, and a `fetch_` read is bit-blasted for `equiv`
through its definition.

```
> show fetch_mirror([10, 20, 30, 40, 50, 60, 70], -1)
_() = 10
> show tex_linear8([0, 1, 4, 9], 2.25)      (3/4 of 1 and 1/4 of 4... of texels 1 and 2)
_() = 3.25
> show tex_bilinear8([0, 1, 4, 9], 2, 1, 1)  (the corner of four texels: their mean)
_() = 3.5
```

**Cost classes and parameters.**  `tex_nearest` prices a `fetch_` read
(its coordinate, (i + 1/2)/n made from the index, costs a `conv` and an
`fmul` more), `tex_sample` a filtered read of an 8-bit normalised
texture (every texel k/255), `tex_sample_f32` one of a 32-bit float
texture.  They are memory kinds (above) in a model's list, but a read
through the unit has no fallback: a model that does not list the
class cannot use the operation (`none` takes it away).  Two parameters, which are not prices,
say what the hardware's unit does beyond the definition above:
`subtexel: B`, the bits of a texel it resolves (Vulkan's
subTexelPrecisionBits: a coordinate or a weight is snapped to that
grid, by less than 2^-B either way), and `tex_within: U`, how many ulps
of the result its filter may be from the exact weighted sum.  A
rewrite is offered only where it holds whichever way the unit snaps
and rounds within those two; a model that does not state them gets
nothing that depends on them.

gt1030 lists what the 2026-11-11 run measured: `tex_sample: 21.625`
(x_tex_bilinear, rgba8, explicit LOD: 21.65 and 21.31),
`tex_sample_f32: 54.75` (rgba32f: 54.80) and `subtexel: 8` (the
driver's limit).  A nearest sampled read was not measured, and neither
was the filter's exactness, so under gt1030 as it stands nothing below
is offered: `texalu` says which number is missing.

**The rewrites.**  `texalu F [tol E] [under M]` (and a section of
`find`, and the function level of `simplify` on GLSL) looks for:

- a table read `T[e]`, or a select between one and a constant, whose
  index is an address mode of one of its subterms i: `T[i % 7]`,
  `T[i < 7 ? i : 13 - i]`, `T[min(max(i, 0), 6)]`, `0 <= i && i < 7 ?
  T[i] : 0`.  The recogniser only proposes (every subterm, every
  mode); the identity `e = mode(i)` wherever the read is defined is
  between integers and is proved by enumeration or SAT.  The
  coordinate is (i + 1/2)/n made in f32, within (|i| + 1/2) 2^-23 of a
  texel's centre, and the unit snaps by less than 2^-subtexel: the
  rewrite needs their sum below 1/2 on i's proved range (|i| up to
  about 4 million at 8 bits), and says so when the range is not known;
- a weighted sum of table entries.  The sum is normalised to
  sum w_k T[e_k] + rest (through +, -, products with weights, a
  division by a constant, fma; an integer sum where intervals show it
  does not wrap), the weights as forms c0 + c1 s + c2 t + c3 st over at
  most two float variables (constants, `1 - t` and `t`, `(1-s)(1-t)`
  ...), the indices as affine forms over the same variables.  The
  table is read as rows of W when every index variable's coefficient
  is 1 or W and W divides the table.  Entries that are neighbours are
  covered by blocks (2 x 2, a pair along x, a pair along y), each one
  filtered read: the block's weights must sum to a constant S, be
  S(1-u)(1-v), S u(1-v), S(1-u)v, S uv for fractions u, v in [0, 1]
  (each linear in one variable, shown on the variables' ranges), and
  its texels must be inside the table on the indices' proved ranges
  (a filter clamps at the edge, an index does not).  The cover with
  the lowest price under the model is taken, by dynamic programming
  over the entries (greedily past 16).

The texture's format comes from the table: bytes (R8_UNORM, priced
`tex_sample`) when every entry is k/255, or a whole number 0..255
(stored as k/255, the read scaled by 255), or one of evenly spaced
levels lo + step m with at most 256 of them (the bytes are m, spread
over 0..255 when the largest m divides 255; the read is mapped back by
one multiply-add, whose rounding is in the bound); 32-bit floats
(`tex_sample_f32`) otherwise.  A table of levels is read as f32 when
the map back would put the bound over what is allowed.

A filtered read's error against the exact weighted sum is derived:
|S| times (the weight's move times the largest step between
neighbouring texels, plus `tex_within` ulps, plus the texels' rounding
to f32) and the f32 products and adds around it.  A fraction that is a
constant on the 2^-subtexel grid, with a coordinate that f32 holds
exactly, does not move; any other is snapped by less than
2^-subtexel.  The rewrite is taken when that bound is within `tol E`,
or, without `tol`, within the bound of the sum's own f32 rounding as
written (gamma_n times the sum of |w t|).  An integer sum of entries
0..255 (stored as k/255 in an R8_UNORM texture) is taken only when the
bound is below 1/2: then `floor(v + 1/2)` is the sum exactly.  The
result is evaluated again against the original on the domain with
ucas's own filter, as a second opinion that is labelled as one.

Every rewrite is kept only when the function is cheaper under the
active model, and comes with what the pipeline needs for it, as a
proposal that is never applied: the texture (format, size, contents),
the sampler state, the binding and the GLSL that reads it.  The setup
(the descriptor, the upload) is not priced: it is not measured.

```
> model cal : gt1030 { tex_nearest: 5, tex_within: 2 }
> f(i: u16 in 0..1000) = [10, 20, 30, 40, 50, 60, 70][i % 7]
> texalu f under cal
texalu f: reads through the texture unit under cal (cost 20.375 as written):
  cost 9   fetch_repeat([10, 20, 30, 40, 50, 60, 70], i)
      ...
      holds where the read is defined: there i % 7 = i mod 7
      that identity between integers is proved: exhaustive
      the coordinate (i + 1/2)/n is made in f32: within (|i| + 1/2) 2^-23 <= 0.000119 of a texel's centre for i in [0, 1000] (its range), and the unit snaps by less than 2^-8: 0.000119 + 0.00391 < 1/2, the texel is i's
      the pipeline needs (a proposal: nothing is applied):
        a texture R8_UINT, 7 x 1, one level, its texels the table's entries
        a sampler: NEAREST filters, REPEAT, normalised coordinates, lod 0
        one combined image sampler more in the layout; the shader reads textureLod(s, (float(i) + 0.5) * (1.0 / 7.0), 0.0).r
> s4(x: u8 in 0..2, y: u8 in 0..2) = B[y*4 + x] + B[y*4 + x + 1] + B[y*4 + x + 4] + B[y*4 + x + 5]
> texalu s4 under cal                       (B: 16 bytes)
  cost 33.625 int(1020tex_bilinear8([...], 4, f32(x) + 1, f32(y) + 1) + 0.5)      (35.5 as written)
      4 entries of the 4 x 4 table from offset (0, 0), weights' sum 4: one read at fraction 0.5 along x, 0.5 along y
      ...
      the entries' sum is a whole number and the reads are within 0.000456 of it (with the f32 products and adds): below 1/2, so floor(v + 1/2) is the sum
> texalu f                                  (gt1030)
texalu f: the texture unit under gt1030 (cost 20.375 as written): nothing taken
  gt1030 lists no tex_nearest: a nearest read with an address mode is not available
```

Under `cal`, `T[min(max(i, 0), 6)]` stays (8.875 against 9: "derived,
but not cheaper"), and with `tex_sample: 40` so does the sum of four.
A mix `(1 - t) T[i] + t T[i + 1]` is one read at fraction t, within
2^-8 times the largest step, so it needs `tol`; a fixed pair at
fraction 3/4 is within the filter's own rounding.  A 3 x 3 kernel is
three taps and one entry.  Weights of both signs, a footprint that can
leave the table, an index beyond f32's reach are refused, each with
its reason.  In English: "what can the texture unit compute of f",
"use the sampler for mx within 0.01 under texel".  JSON and MCP:
`texalu` (f, tol, model).

Not done: texels split between taps (a 3 x 3 kernel in four reads),
trilinear and anisotropic filters, mip levels as means, min/max and
compare samplers, gather, array layers, sRGB and the other formats
(R16, half floats), normalised coordinates for the filters, and SPIR-V
written for a read (a module cannot add its own binding).  None of
those has a measured price yet either.

### Tables as compressed textures

A block-compressed format decodes in fixed function: a few parameters
a block, interpolated by the decoder.  A table the decoder's palette
reproduces is stored in less than its bytes and read like any other
texture.  `texenc T [rows W] [tol E]` (src/texenc.c) searches such an
encoding with the table as its target, not an image.

BC4 (RGTC1, unsigned): a 4 x 4 block is 8 bytes, two 8-bit endpoints
r0, r1 and sixteen 3-bit selectors, least significant first, texel
(x, y) at bits 3(4y + x).  With r0 > r1 the palette is r0, r1 and
((8 - i) r0 + (i - 1) r1)/7 for i = 2..7; otherwise r0, r1, ((6 - i) r0
+ (i - 1) r1)/5 for i = 2..5, then 0 and 255.  A texel is its palette
entry over 255, a rational, and a read returns its f32.  The encoder
tries every pair of endpoints (65536) with each texel on its nearest
palette entry, and keeps the smallest largest error, then the smallest
sum of squares, then the first found: the answer depends on the table
alone.  The table is read as rows of W (4 unless given); texels beyond
it in a block are don't-cares; whole numbers 0..255 are stored as
k/255.

```
> R = [255, 219, 183, 147, 111, 75, 39, 3, 255, 3, 39, 75, 111, 147, 183, 219]
> texenc R
texenc R: 16 entries as BC4, rows of 4 (4 x 4 texels): 1 block, 8 bytes (16 as R8_UNORM, 64 as R32_SFLOAT); whole numbers k are stored as k/255
  block (0, 0): endpoints 255, 3 (six steps of 1/7 between them), selectors 0 2 3 4 5 6 7 1 0 1 7 6 5 4 3 2, bytes ff 03 d0 58 3f c8 5d 4e
  decodes to [1, 0.858823538, 0.717647076, ...]
  exact: every texel is its entry's f32 over 255 (16 texels a block against 8 palette values, every pair of endpoints tried)
> texenc Q tol 0.01                         (sixteen different values)
  not exact: the closest blocks are within 0.0514 of the table (entry 3: 0.300000012 as 0.248627454), every pair of endpoints tried
  more than the 0.01 allowed (tol)
```

An 8-level ramp, or one of 6 levels with 0 and 1, is exact in half the
bytes of R8; the decoded table is what a nearest or filtered read of
the texture returns, so the `fetch_` and `tex_` operations take it as
their table, and `texalu` says when the 8-bit texture it proposes is
exact as BC4 blocks.  The bytes for R and for a 6-level table were
decoded by ffmpeg's RGTC1 decoder to the same values.  In English:
"compress R as BC4", "encode Q as bc4 within 0.02, rows of 8".  JSON
and MCP: `texenc` (table, rows, tol).

Not priced: no model has a class for a read of a compressed texture
apart from its uncompressed one (none is measured), so nothing is
rewritten on BC4's account; `texenc` gives the asset and its proof.
Not done: BC5 (two BC4 blocks, one a channel), ASTC (weight grids,
dual planes, partitions, HDR endpoints), BC6H and BC7, ETC2/EAC, the
sRGB variants, and a polynomial split between a table's knots and a
residual in the ALU.

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

## Stochastic search

Searches that draw random numbers are seeded: `seed K`, else the
environment's `UCAS_SEED`, else 1.  The same seed is the same run, on
any libm (acceptance and bounds use + * / only).  What a search finds
is verified or labelled; nothing sampled is called a proof.

**stoke.**  `stoke F` is Markov-chain Monte Carlo over straight-line
programs, after STOKE (Schkufza, Sharma, Aiken 2013).  A program is up
to `len` (8) instructions from add sub mul and or xor shl shr not neg
on the inputs, earlier results and constants.  What F has joins the
set: its compares, selects or signed shifts bring ltu lts eq, sel and
sar; its widths, each a width an instruction can take (its operands
read at that width, zero-extended or cut), bring zext and sext; each of
its tables of integer constants a load; a value 32 bits wide the dot
forms the model prices (and 0x01010101 among the constants).  F as
written lowers to such a program, and without any of them the search
is the one it always was, draw for draw.  A proposal changes an
opcode or a width, an operand, a constant, a whole instruction (or
removes it), or swaps two; it is accepted by Metropolis-Hastings on

    cost = Hamming distance to F on the test inputs + the price under the model

(the tests: the ends of the ranges and 36 random points).  The price is
the report's own, the cover of the program's expression by the model's
fusions: an instruction the result does not read costs nothing, and a
multiply-add or a shift-add counts as the one instruction it is, so the
fusions steer the chain.  Prices are cached by the live instructions.  `restarts`
(4) chains of `steps` (200000) proposals run: the first starts from F
itself when F is straight-line in those operations (optimisation), the
others from random programs (synthesis).  A program at distance 0 is a
candidate: it is checked on the whole domain (enumerated, or by SAT;
for polynomials SAT gives up on, by identity testing, printed as
"probably"), and a counterexample joins the tests.  The chain's own
price is the sum of its instructions' classes; what it finds is priced
by the model's cover with its fusions, as `cost` does, and listed
cheapest first, the fewer operations first at one price.

```
> xr(x: u8, y: u8) = (x | y) - (x & y)
> stoke xr
stoke xr under gt1030: Metropolis-Hastings over programs of up to 8 instructions (...),
  4 chains of 200000 proposals, seed 1; xr as written costs 3
  800000 proposals, 666113 accepted, 664981 at distance 0 on the 40 tests (0 counterexamples joined them), 4 verified
  cost 1      y xor x
      every input enumerated
  -> 1 against 3 (-66.7%)
> m9(x: u8) = x * 9
> stoke m9                  # gt1030 fuses a shift and an add: 2 against the multiply's 3
  cost 2      x + 8x
> stoke m9 model sm75       # sm75's multiply costs 1
  -> nothing cheaper than 1 was found and verified (the search is stochastic: more `steps`, another `seed`)
```

`find F [n] [max c] stoke` runs it after find's own enumeration.  It is
for machine integers; "nothing cheaper was found" is not "nothing
cheaper exists" (find's enumeration, up to its cost bound, is).

**pit.**  `pit F G` is Schwartz-Zippel identity testing.  Both must be
polynomials over the integers in integer parameters (+ - *, constant
powers and shifts, conversions that do not widen).  They are evaluated
at `trials` (4) points drawn uniformly mod p = 2^61 - 1; two different
polynomials of total degree at most d agree at one point with
probability at most d/p, so "equal" is wrong with probability at most
(d/p)^trials, which is printed.  Equal over the integers they are equal
in a wrapping type, reduction mod 2^w being a ring homomorphism.  The
converse fails: 128 x (x + 1) is not the zero polynomial and is 0 for
every u8, so "they differ over the integers" (which is certain) is not a
verdict on machine functions; pit then compares those on the domain and
says what it saw.  `equiv` adds the test's line when enumeration and SAT
leave it at "sampled, not proved" (a multiplier against a multiplier):

```
> equiv pa pb
pa and pb are equal on the domain of pa (sampled, not proved; SAT gave up on 19200 AIG nodes ...)
  polynomials of total degree at most 2, equal at 4 random points mod 2^61 - 1: two that differ agree at one with probability at most 2/(2^61 - 1) = 8.67e-19 (Schwartz-Zippel), at all 4 with at most 5.66e-73: equal over the integers except with that probability, and then for every input here (a probabilistic answer, not a proof)
```

**phash.**  `phash F` takes a function of one machine integer written
as a `match` or an if-chain on constant keys (`x == k`, their `||`,
small ranges) and looks for a hash that sends the keys to distinct
slots of a table of 2^b entries, b from the smallest that fits up to
three more: `(x >> s) & mask`, `(x * K) >> s`, `((x xor x >> a) * K) >>
s` with K drawn at random, and a two-level hash with a displacement
table for the smallest b.  Injectivity is checked on the key set, slot
by slot.  Other inputs are handled by the parameter's range when every
value is a key, else by a second table holding each slot's key,
compared with x (an empty slot holds a key that hashes elsewhere).  The
lookup replaces the chain when the model prices it lower, after it is
proved equal to F for every input (enumerated or SAT); `F_ph` is
defined.

```
> sw8(x: u32) = match x { 3 => 10, 17 => 20, 100 => 30, 1000 => 40, 65537 => 50, 99999 => 60, 123456 => 70, 7777777 => 80, _ => 0 }
> phash sw8
phash sw8 under gt1030: 8 keys, the compare chain costs 32; seed 1
  injective on the 8 keys (each slot checked): h(x) = (x * 1671120851) >> 29, a table of 8 slots
  -> [1000, 3, 123456, 99999, 17, 65537, 7777777, 100][t1] == x ? [40, 10, 70, 60, 20, 50, 80, 30][t1] : 0 where t1 = 1671120851x >> 0x1d
     cost 22.75 against 32 (-28.9%); equal to the switch for every input: proved by SAT
> few(x: u32) = match x { 5 => 1, 900 => 2, 70000 => 3, _ => 0 }
> phash few
  -> not rewritten: the lookup costs 18.75, the compare chain 12 under gt1030
```

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

The float part is binary32's: values of another width (f64, f16) are
inputs to it, and their own instructions are not priced or rewritten
here (f64 has its own part, see "fp64 without fp64").

### The sink: precision from the stored format backwards

A float that ends in a format with fewer bits than f32 does not need
f32's precision on the way.  `simplify ... sink FORMAT` says what the
module's stored and returned floats end in, and an image store's
format (`layout(rgba8)`, in its OpTypeImage) is read from the module:

| sink | one step |
|---|---|
| `unormN` (`rgba8`, `r8g8b8a8_unorm`, `rgb10a2`: its widest channel) | 1/(2^N - 1), absolute |
| `snormN` (`rgba8_snorm`) | 1/(2^(N-1) - 1), absolute |
| an sRGB format (`b8g8r8a8_srgb`) | 1/(12.92 (2^N - 1)): 8 bits of the encoded value, whose slope is at most 12.92 |
| `f16` (`rgba16f`, `r16g16b16a16_sfloat`) | 2^-11 of the value, for magnitudes from 2^-14 |

The argument of `packUnorm4x8`, `packSnorm4x8`, `packUnorm2x16` and
`packSnorm2x16` has the function's own width as its sink (GLSL defines
them as round(clamp(c, 0, 1) * 255) and the like), with no word in the
request.  A stored or returned vector put together from its lanes
(`o[i] = vec4(enc, a)`) is a root like any other when float arithmetic
is under the construct.

Integer formats and 32-bit floats are not sinks (`sink rgba32f` is an
error).  In English: "for an RGBA8_UNORM target", "the format is
rgb10a2", `sink unorm8`.

One step is the budget at the root: the UNORM conversion clamps to
[0, 1], scales by 2^N - 1 and rounds; rounding to nearest (what Vulkan
asks of an implementation; it allows either neighbour) two values
within one step are stored as the same integer or as neighbours.  Exact
equality of the stored integer cannot be promised on a continuous
domain (any change moves some value across a rounding boundary), so the
budget is "within one step", against the reference, wherever the
reference's value is a number.

The budget is carried from the root to each rewrite (src/fsimp.c,
sink_pass).  A rewrite's own bound is an absolute deviation E at its
node, and

    dev(n) = E(n) + sum over operands k of max|dn/dk| dev(k) + 4 ulp(n)

with max|dn/dk| bounded on the operands' ranges where the node is used
(1 for add, min, max and a select's sides; the other factor's magnitude
for a multiply; |y| q^(y-1) for pow; the polynomial's own sum i |a_i|
T^(i-1) where one replaces a pow), and 4 ulp where an operation whose
operand moved rounds again.  A compare or a step whose operand moves
has no bound.  The ranges are the unperturbed program's, and a
perturbed operand leaves its range by at most its deviation: a second
pass bounds the derivatives on each range widened by twice the first
pass's deviation and must find every deviation inside that widening,
which makes the bound hold (the deviations grow continuously with the
rewrites' errors and would have to cross the widening first).

The rewrites the other rules accept (within the original's Vulkan
precision) move the root too and are counted first; then each node's
cheapest remaining rewrite is taken, the largest savings first, while
the root stays within the budget.  What the active model does not
price lower is never a candidate, so under a model where pow is cheap
the sink changes nothing.

```
simplify float c in [0, 1] sink unorm8
  float cost under gt1030: 51 -> 33 (-35.3%)
  sink unorm8: one step is 0.00392157. ...
    budget 0.00392, the rewrites accepted use at most 0.00162 of it
    %61 = OpExtInst %v3float %2 Pow %60 %26
      -> ... four Fma ...   (pow as a polynomial; -9)
          relative minimax of q^2.4 on [0.090473935, 1] ...: 0.00162 relative in all
          the sink: its own bound moves this value by at most 0.00162; carried to the stored value ... at most 0.00162 of the 0.00392 one step allows
```

That is the sRGB decode above: with the 8-bit sink the pow is a
degree-4 polynomial (the select keeps it on [0.0905, 1]); without a
sink the polynomials are only listed, and `sink unorm16` refuses every
degree cheaper than the pow ("it would move the stored value by up to
0.00135, more than the 1.53e-05 a step of the sink allows").  When the
value is scaled after the pow (`pow(...) * 3.0 - 1.0`) the pow's share
of the step is a third and the degree that fitted no longer does.  An
`imageStore` into an rgba8 image of `pow(c.rgb * 0.75 + 0.125,
vec3(0.45))`, c clamped to [0, 1], needs no request word: 34 -> 19, a
degree-3 polynomial using 0.00383 of the 0.00392.  A `clamp` is read as
min(max(x, lo), hi), so what follows it knows its range.

exp2, log2, sqrt and inversesqrt get the same treatment where their
argument's range is bounded (above 0 for the last three): an absolute
minimax polynomial of each degree cheaper than the function (under
gt1030 each costs 4, a degree-1 polynomial 2 and a degree-2 one 3),
certified the same way with the function's own second derivative
(ln2^2 2^x, 1/(ln2 x^2), x^-1.5/4, 3 x^-2.5/4) and compared with its
Vulkan precision (exp2 3 + 2|x| ulp, log2 3 ulp or 2^-21 absolute in
[0.5, 2], inversesqrt and sqrt 2 ulp, lower bounds).  The bound is
absolute, so log2 over a range holding 1 (where it is 0) has no
relative one and says so.  Under an 8-bit sink `sqrt(c) * 0.5` on
[0.25, 1] becomes a degree-2 polynomial (0.00175 of the step) while
`inversesqrt(c) * 0.25` beside it, steeper there, is refused
(tests/sinkpoly.ucas).

Left to the other parts of ucas: an integer function of few inputs
whose value must be the same integer (`u32(floor(x*255 + 0.5))` of an
8-bit x) is `find`'s, which checks every point and offers tables and
multiply-shifts; a branch never taken on the domain and a mask that
drops nothing are `find`'s peephole and `simplify`'s SAT rewrites.

Not done yet: proofs for two or more float inputs that interact (a
second input is kept as an interval, so its sets are "possibly"); bit-
level SAT for small float pieces; joining two vec2 operations into one
vec4 (the gt1030 model prices components, so it would save nothing
there).

### Advice: what could leave the shader

`simplify` adds a report of what the module computes that changes less
often than it runs, whenever it finds some; `advice` ("and advise",
"what could be hoisted") also has it say when it finds none.  Nothing
is rewritten: the change is to the pipeline's interface
(a push constant, a uniform, a varying), which is the host's.  Each
instruction gets the lowest frequency its operands allow (src/spirv.c,
advice_pass):

- **constant**;
- **draw**: reads only constants, uniform blocks, push constants and
  specialization constants;
- **primitive** (fragment shaders): reads only flat inputs and values
  of the draw.  A flat input is the provoking vertex's output, the same
  at every pixel of the primitive, so what is computed from it could be
  computed in the vertex shader and passed flat: each pixel reads the
  value it computes now (no interpolation);
- **vertex** (fragment shaders): affine in the float inputs that are
  not flat, centroid, sample or built-in: sums, differences, and
  products, quotients, dots, matrix products and mixes whose other
  operands are of the draw or the primitive.  A varying is v = sum b_i v_i over the
  vertices with sum b_i = 1 (perspective-correct barycentrics), and an
  affine f has f(v) = sum b_i f(v_i), so f computed at the vertices and
  interpolated is the same real number at every pixel;
- **every run**: the rest (textures, storage buffers, built-ins, a
  function's own variables, anything not affine of a varying).

The values of a lower frequency that something of a higher one uses
are what would be passed in.  For the draw the report prices both
sides under the active model: what the instructions and their loads
cost every run, against one load a value passed in; where passing
costs more it says "derived, but not cheaper" (three values of one
uniform are three loads against one load and three operations).  It is
the same instructions on the same values: adds, subtracts, multiplies,
conversions and selections are exact or correctly rounded on a host as
in the shader, and the report counts the divisions, square roots and
transcendentals, which are only within Vulkan's precision in the
shader and may differ from a host's in the last places.  For the
primitive and the vertex the report gives the derivation and the cost
a pixel, and no verdict: no model has a price for a vertex or a varying yet (the
GT 1030 run measured them: a vertex about 300 fadd steps, a pixel's
inputs free up to two vec4, about 3.4 a vec4 for the third and fourth
and 15 from the fifth; tests/gt1030 results, "Stage I/O"), and an
unpriced form is not offered.

```
simplify advice
advice (changes outside the module; nothing here is rewritten), priced under gt1030:
  once a draw: 5 instructions that read only constants, uniform blocks and push constants cost 45.5 every run (18 of arithmetic, 4 loads at 6.875)
    computed once a draw and passed in as 2 values (push constants or a uniform block: 2 loads, 13.75), that saves 31.75 every run
    ...; 3 of them (divisions, square roots, transcendentals) are only within Vulkan's precision in the shader ...
    %27 = OpVectorTimesScalar %v3float %21 %26   (19.75)
        (line 9: vec3 l = normalize(pc.light.xyz) * pc.light.w;)
    %40 = OpFDiv %float %34 %39   (25.75)
        (line 10: float k = pow(2.0, pc.exposure) / (pc.gamma + 1.0);)
  once a vertex: 3 instructions costing 3 every pixel are affine in the varyings (sums, and products with values of the draw or the primitive)
    not priced: gt1030 has no price for a vertex or for a varying, ...
    %79 = OpFAdd %float %75 %78   (3)
        (line 13: float fogk = clamp(-vp.z * u.fog.x + u.fog.y, 0.0, 1.0);)
```

JSON: the result gains `advice_draw`, `advice_primitive` and
`advice_vertex` where there are such values (`advice: true` also
says when there are none).

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
| `integrate F [from a to b] numerically qmc [points N] [replicates R] [seed K] [noscramble]` | a quasi-Monte Carlo estimate on Sobol points (src/qmc.c): over from..to, or without them over the box of every parameter's range, in up to 8 dimensions.  The direction numbers are Joe and Kuo's; by default 8 replicates of 4096 points (N a power of two) run, each under a random digital shift, and the answer is their mean with its standard error: an estimate from the replicates' spread, labelled as one, with "not converged" when they differ by more than 5% of the value (a pole between the points looks like that), and "a spike" when the largest value met is over 50 times the mean size: around an integrable singularity the replicates agree on a value that is off and their spread understates the error (1/sqrt(|x|) on [-1, 1]: 3.9796 with a standard error of 0.0018, for 4).  `noscramble` runs the one unshifted sequence and gives no error estimate.  A coordinate is the centre of its 2^-32 cell, so the box's ends are never evaluated; points where the integrand is undefined are left out and counted.  It is also the fallback of a definite integral with no antiderivative here that Gauss-Kronrod refuses (an undefined or infinite point inside): the estimate when the replicates agree, one line saying it settles nothing when they do not |
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
- Subgroup operations are exact for a full subgroup whose per-lane
  values are functions of the lane and of shared values; clustered,
  partitioned and quad operations, min/max/bitwise reductions and float
  sums are not operations yet.
- The catalogue's instructions (bfe, brev, mulhi, funnel shifts, med3,
  usad4x8, byteperm, bfi, mulwide) are priced by no built-in model: none
  is measured.
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
