# Reducing LOP3's truth tables: the process, and what a CAS would have done

This note records how ScarletNX's Maxwell `LOP3` translation was reduced from a sum of minterms to the shortest
program for each truth table (worktree `~/ScarletNX-lop3`, branch `lop3-lut`). It then asks what a general computer
algebra system for this class of problem would have given: a system that takes a set of gates and a cost model, and
maps any expression to its reduced form.

The numbers below come from the scripts in `~/ScarletNX-lop3/work/` (`crosscheck.py`, `forms.py`, `classes.py`,
`bases.py`) and from `tools/lop3gen.py` itself.

---

## 1. The problem

`LOP3` takes three 32-bit operands `a`, `b`, `c` and an 8-bit truth table `lut`. Each result bit is

```
result[i] = lut[ a[i] << 2 | b[i] << 1 | c[i] ]
```

so one instruction computes any of the 256 boolean functions of three inputs, bit by bit.

SPIR-V has no such instruction. The old emitter wrote the function's disjunctive normal form directly. For each set
bit of `lut` it emitted one minterm: up to three `OpNot`, two `OpBitwiseAnd` and one `OpBitwiseOr` into the
accumulator. Neither spirv-opt nor the drivers fold this back. Over all 256 tables that is **4608 instructions**, up
to **36** for a single table (`0xff` built from 8 minterms). Even the select `a ? b : c` (`0xca`) took 16.

The goal was to find, ahead of time, a short program for every table, store each as a `u64`, and have a small
emitter follow it.

## 2. Representation: the truth table is the canonical form

The key fact behind everything else: **for three variables, the 8-bit truth table is a complete canonical form.**
Two expressions are equivalent if and only if they have the same truth table, and computing that table is cheap.

The generator evaluates on all 8 input combinations at once by bit slicing. Each variable is replaced by the truth
table of the projection onto it:

| variable | table  | meaning                                    |
|----------|--------|--------------------------------------------|
| `a`      | `0xF0` | bit `i` is set when `i & 4` (the `a` bit)  |
| `b`      | `0xCC` | bit `i` is set when `i & 2`                |
| `c`      | `0xAA` | bit `i` is set when `i & 1`                |

Then any expression built from `& | ^ ~`, evaluated on these bytes with `~x` taken as `x ^ 0xFF`, yields its own
truth table. For example, `c ^ (a & (b ^ c))` gives `0xAA ^ (0xF0 & (0xCC ^ 0xAA)) = 0xCA`.

So **an equivalence check is one byte comparison**. Deciding equivalence, the usual hard core of a CAS, costs nothing
here. The work is all in the other direction: finding the cheapest expression for a given table.

## 3. The cost model and the gate basis

- **Basis:** `OpBitwiseAnd`, `OpBitwiseOr`, `OpBitwiseXor`, `OpNot`. These are the only bitwise logic instructions
  SPIR-V offers on integers.
- **Cost:** one per instruction. Constants and the operands are free.
- **Tie-breaks**, in order: fewer `OpNot` (a driver targeting hardware with a native LOP3 or and-not can absorb
  them), then a shorter chain of dependent steps (depth), then the first program in search order. The last rule makes
  the output deterministic.

Operations considered and rejected:

- **Integer add with carry.** LOP3 works on each bit independently, and an add moves information into the next bit
  position. Where the two operands share no set bit, `x + y == x | y == x ^ y`, so an add can stand in for OR but
  never beats it. Where they do share a bit, the carry corrupts the neighbouring bit.
- **`OpSelect`.** It chooses whole values on a boolean condition, not individual bits, so it cannot express a
  bitwise mux.
- **`OpBitFieldInsert`.** It inserts a contiguous field at a fixed offset, not under an arbitrary mask.
- **And-not / or-not.** SPIR-V has no such instructions. That absence is why NOT costs a full instruction here, and
  §7 shows it is also why equivalence classes are smaller than one might hope.

## 4. Why rewriting toward a normal form is not enough

The task statement said "reduce by boolean algebra". The classical algebraic route is to rewrite toward a normal form
and minimise within it. Two such forms were measured on the same cost model (`work/forms.py`). In both, each
subterm is counted every time it appears, with no sharing:

- **Minimal sum of products** (what Quine–McCluskey or Espresso produce): the cheapest cover by implicants, with
  shared negated inputs counted once.
- **Algebraic normal form** (Zhegalkin polynomial, XOR of AND monomials), which is canonical and handles XOR well.

| form                            | total for 256 tables | worst single table |
|---------------------------------|---------------------:|-------------------:|
| minterms (the old emitter)      | 4608                 | 36                 |
| minimal sum of products         | 1489                 | 14                 |
| algebraic normal form (ANF)     | 1409                 | 12                 |
| best of the two, per table      | 1158                 | 9                  |
| **exhaustive search (shipped)** | **784**              | **5**              |

The search is strictly cheaper than **both** normal forms on 177 of the 256 tables. Typical reasons:

| table  | function           | SOP | ANF | search | program found                  |
|--------|--------------------|----:|----:|-------:|--------------------------------|
| `0x96` | 3-input parity     | 14  | 2   | 2      | `c ^ (a ^ b)`                  |
| `0x7e` | not all equal      | 8   | 8   | 3      | `(a ^ b) \| (a ^ c)`           |
| `0xca` | mux `a ? b : c`    | 4   | 4   | 3      | `c ^ (a & (b ^ c))`            |
| `0xe8` | majority           | 5   | 5   | 4      | `(a & b) \| (c & (a \| b))`    |
| `0x16` | exactly one        | 11  | 5   | 4      | `(a \| b) ^ (c \| (a & b))`    |
| `0x17` | at most one        | 8   | 6   | 5      | `(c & (a ^ b)) ^ ~(a & b)`     |

The pattern:

1. **SOP has no XOR.** Parity costs 14 as a sum of products and 2 as XORs.
2. **ANF has XOR but no OR.** Majority and "not all equal" pay for that.
3. **Both are trees, not graphs.** The optimal programs reuse an intermediate value (a straight-line program is a
   DAG), and they mix AND, OR and XOR freely. `0x7e` uses `a` in two XORs and joins them with an OR, which neither
   normal form can express.
4. **Rewriting with identities is a local search.** Starting from either form, a rewrite system (absorption,
   distribution, `x ^ y = (x | y) & ~(x & y)` and so on) has to go uphill before it can reach these programs. Without
   an exhaustive or saturating strategy it stops at a local minimum, and you cannot tell whether that minimum is
   global.

So the shipped generator does not rewrite. It **searches**, and minimality follows from the search being complete.

## 5. The search

### 5.1 Straight-line programs

A program is a sequence of steps. Each step applies one gate to earlier values: slots 0–2 are `a`, `b`, `c`, and
slot `3 + i` is step `i`'s result. The program's value is its last step. Evaluating on the bytes from §2 gives each
prefix's truth table directly.

The generator runs a depth-first search over all such programs up to `MAX_STEPS = 5`. Every time a step produces
a truth table, the program so far is a candidate for that table, and the best-scoring candidate is kept. Scores are
compared lexicographically as (steps, NOTs, depth, program).

### 5.2 Pruning, and why each rule is safe

1. **No repeated value.** A step that produces a table already present (an operand or an earlier step) is never on a
   shortest path: any later use could refer to the earlier copy, so the step can be deleted.
2. **No constants.** A step that produces `0x00` or `0xFF` is never useful. The constant tables themselves are
   handled as zero-step programs.
3. **Commutative operands are ordered:** only `(x, y)` with `x < y`. NOT takes one operand.
4. **Independent steps in canonical order.** If step `k+1` does not use step `k`'s result, its key `(op, x, y)`
   must be greater than step `k`'s.

   Why this is safe: two adjacent independent steps can be swapped without changing any value. The swap renumbers
   their two result slots, but their own operands come from earlier slots, so their keys do not change. Bubble-sorting
   adjacent independent pairs by key therefore turns any program into one that satisfies the rule, with the same
   length, the same gates and the same depth. Each topological order is visited once instead of up to `k!` times.
5. **Recording any prefix is enough.** A shortest program for `f` never contains a dead step: removing one would give
   a shorter program for `f`. So a program found as some other program's prefix is still a valid candidate.

With these rules the full search takes **about 4 s in Python**.

### 5.3 Verifying the search

- **Unpruned cross-check** (`work/crosscheck.py`): the same search with rules 3 and 4 removed (ordered pairs, every
  order) took 6 minutes. It gave **the same (steps, NOTs, depth) for all 251 non-trivial tables**. So the pruning
  loses nothing, not even on the tie-breaks.
- **Self-check:** before writing the header, the generator evaluates every chosen program and compares it with its
  table.
- **Against the old emitter, in SPIR-V** (`tests/shader_lop3.c`): for all 256 tables, both emitters write real SPIR-V
  through `SpvEmitter`. A small interpreter for the four opcodes runs the emitted words. It feeds them
  `(0xF0F0F0F0, 0xCCCCCCCC, 0xAAAAAAAA)`, which covers all 8 combinations in every bit, plus all-zero, all-one and
  61 random operand sets. It checks old = new = the definition, and that the new code never has more instructions:
  34,048 checks in all.
- **Mutation check:** corrupting the `0xca` entry made both the test (124 failures) and `lop3gen.py --check` fail.

### 5.4 Results

| ops     | 0 | 1  | 2  | 3   | 4  | 5  |
|---------|---|----|----|-----|----|----|
| tables  | 5 | 12 | 42 | 112 | 73 | 12 |

The 5 zero-op tables are the two constants and the three operands. The 12 five-op tables are minority (`0x17`) and the
permutation families of `0x29`, `0x2b`, `0x6b` and `0x97`:

```
0x17 (c & (a ^ b)) ^ ~(a & b)      0x29 (a | b) ^ ((a & b) | ~c)     0x2b (c & (a ^ b)) | ~(a | b)
0x6b (c ^ (a & b)) | ~(a | b)      0x97 (c & (a | b)) ^ ~(a & b)     0xe9 (c | (a & b)) ^ ~(a | b)
```

Each of these programs contains exactly one NOT, and in this basis it cannot be merged into a neighbouring gate. With
AArch64's `BIC`/`ORN`/`EON`, no table needs more than 4 operations (§8.3).

## 6. Packaging

The `u64` format (`src/nx/gpu/shader/lop3_table.h`, generated):

```
bits 0-2   number of steps n (0-5)
n == 0:    bits 3-5 the result: 0-2 operand a/b/c, 3 zero, 4 all ones
step i:    the byte at bit 8 + 8*i
             bits 0-1  operation: AND, OR, XOR, NOT
             bits 2-4  first operand's slot
             bits 5-7  second operand's slot (NOT ignores it)
           slot 0-2 = a, b, c; slot 3+i = step i's result
result:    the last step's slot
```

Five steps at one byte each, plus the header byte, fit in 48 bits. The 3-bit slot fields reach slot 7, which is
exactly the fifth step's result (`3 + 4`), so the format is tight for `MAX_STEPS = 5`.

- `shader_lop3` (`emit_int.c`) decodes the steps and calls `spv_emit_*` for each one.
- `tools/lop3gen.py --check` runs in CTest as `lop3_table_current`, following the same pattern as
  `build-controls.py --check`.

## 7. Symmetry: what equivalence classes buy

The 256 functions fall into far fewer classes under relabelling (`work/classes.py`):

| group acting on the function                        | classes | classes whose members differ in cost |
|-----------------------------------------------------|--------:|-------------------------------------:|
| P: permute the inputs                               | 80      | 0                                    |
| NP: also negate inputs                              | 22      | 20                                   |
| NPN: also negate the output                         | 14      | 13                                   |

- **Permuting inputs never changes cost.** A permutation only relabels the leaves of a program. So 80 searches
  plus a renaming of operands would have produced the whole table.
- **Negation does change cost, because NOT costs an instruction in this basis.** The textbook NPN reduction to 14
  classes is valid only for a basis where complementing an input or output is free. Examples are AArch64's
  `BIC`/`ORN`/`EON`, or the hardware LOP3 itself. For SPIR-V, the classes are the 80 P-classes.

This is a direct example of something a CAS needs to know: **which symmetries the cost model respects.** Applying
NPN canonicalisation blindly here would have produced wrong costs.

---

## 8. What a general CAS for this class of problem would have given

### 8.1 The general problem

Stripped of Maxwell and SPIR-V, the task was:

> Given a **domain** (boolean functions of `n` inputs, bitwise), a **basis** (gates, their arities, commutativity
> and costs) and a **tie-break policy**, compute for every function a minimum-cost program, prove it minimal, and
> emit the result as a table plus an emitter.

Every piece of this was written by hand for one instance: the bit-sliced evaluator, the canonical form, the search,
the pruning proofs, the symmetry analysis, the encoding, the verification, and the comparison against the normal
forms. A CAS specialised to *mapping gates to their reduced form* would provide these as reusable parts.

### 8.2 The components

1. **Canonical forms and equivalence.**
   - Truth tables for small `n` (as here).
   - Reduced ordered BDDs or AIGs for larger `n`.
   - ANF for XOR-heavy algebra.
   - NPN or P canonicalisation, restricted to the symmetries the declared cost model respects (§7).

   Equivalence of two expressions becomes "same canonical form", which is the "process of finding equivalence" in
   its general shape.

2. **A declared basis and cost model.** For example:

   ```
   basis spirv  { and/2 comm cost 1; or/2 comm cost 1; xor/2 comm cost 1; not/1 cost 1 }
   basis a64    { and; orr; eor; mvn; bic/2 (x & ~y); orn/2 (x | ~y); eon/2 comm (x ^ ~y) }
   basis a64sha { a64 + bsl/3 (s ? x : y); eor3/3 comm; bcax/3 (x ^ (y & ~z)) }
   ```

   From the declaration the system would derive which pruning rules hold (commutativity, which symmetry group is
   cost-invariant) instead of the author proving them again.

3. **An exact-synthesis engine with a choice of strategy:**
   - **Enumeration** of straight-line programs for `n <= 3` (seconds; this note).
   - **Symmetry-reduced enumeration** for `n = 4`: 65,536 functions in 222 NPN classes. Knuth computed exact costs
     for all of them this way (TAOCP 4A, §7.1.2).
   - **SAT-based exact synthesis** for `n >= 5`, or for single functions: "is there a program of `k` steps?" is a SAT
     instance, as in ABC, mockturtle or percy.
   - **Equality saturation** (e-graphs, as in `egg`) when the input is an *expression*, not a truth table. It applies
     every identity at once without the local-minimum problem of §4, then extracts the cheapest term under the cost
     model.

4. **Certificates.** A proof of minimality for each table: the search was exhaustive to `k - 1` steps, or the SAT
   instance for `k - 1` was UNSAT. Plus a checker that the chosen program evaluates to the table. Here this took a
   separate 6-minute script.

5. **Code generation.** The table (any packing), the emitter skeleton and a `--check` mode, so the generated header
   stays reproducible in CI.

### 8.3 Concrete payoffs, using this codebase

**a) Retargeting is a change of declaration, not a rewrite.** The same search run against three bases
(`work/bases.py`):

| basis                                   | total ops, 256 tables | distribution (ops: tables)              |
|-----------------------------------------|----------------------:|-----------------------------------------|
| SPIR-V (AND OR XOR NOT)                 | 784                   | 0:5 1:12 2:42 3:112 4:73 5:12           |
| AArch64 scalar (+BIC ORN EON)           | 611                   | 0:5 1:27 2:112 3:88 4:24                |
| AArch64 SIMD/SHA3 (+BSL EOR3 BCAX)      | 510                   | 0:5 1:40 2:163 3:48                     |

Each row has a different optimal program per table, a different worst case (5, 4, 3 ops) and so a different packing.
With a CAS these are three declarations. By hand, each is a new generator.

- **x86 with AVX-512 `vpternlog`** is the degenerate case: every table costs one instruction, and the "reduced form"
  is the table itself.
- **The reverse direction** (lowering guest 3-input logic such as A64's `EOR3`/`BCAX` to a host without them) is the
  same problem with the bases swapped.

**b) Cross-instruction reduction (peephole).** The table above reduces one LOP3. A guest shader often chains
`LOP`, `LOP32I` and `LOP3` instructions whose combined inputs are still at most three registers. With the truth table
as canonical form, composing a chain is just evaluating it on `0xF0`/`0xCC`/`0xAA`. The whole chain then becomes one
table, and one table lookup gives its optimal program. A CAS that treats "expression over at most 3 variables →
canonical form → optimal program" as a primitive makes this a few lines. The same idea extends to four inputs with
the `n = 4` table.

**c) Analyses derived from the same algebra.** `dataflow.c`'s known-bits analysis handles `LOP` (AND/OR/XOR/pass)
but not `LOP3`, so a LOP3 result is treated as unknown. Given known bits for `a`, `b`, `c`, a result bit is known
exactly when the table is constant over the input combinations still possible at that bit. That transfer function
falls out of the truth-table representation. A CAS could generate it for any table-defined instruction instead of
leaving it out.

**d) Trust.** The minimality claim in this work rests on a hand-written pruning argument plus a second, slower
search to back it up. A CAS that emits a minimality certificate with each entry would make the claim checkable
without reading the generator.

### 8.4 Where a CAS would not help

- **The cost model is a proxy.** Instruction count in SPIR-V is not what the GPU runs. A driver with a native LOP3
  (NVIDIA) may collapse any of these programs to one instruction anyway. An AMD driver may or may not recognise
  `c ^ (a & (b ^ c))` as `v_bfi_b32`. Measuring that needs the drivers themselves (spirv-opt, then per-vendor
  disassembly), not algebra.
- **Tie-breaking is policy.** "Fewer NOTs, then less depth" was a judgment about drivers. A CAS can expose the knob
  but cannot choose it.
- **Beyond small `n`, optimality gets expensive.** At `n = 5` there are 2^32 functions, and exact synthesis is per
  function via SAT. The tabulate-everything approach used here stops scaling around `n = 4`.

## 9. Reproducing

On `fbsd-srv`:

```sh
cd ~/ScarletNX-lop3
python3 tools/lop3gen.py            # regenerate src/nx/gpu/shader/lop3_table.h (~4 s)
python3 tools/lop3gen.py --check    # what the lop3_table_current test runs
build/rel/tests/snx-test-shader-lop3
cd work
python3 crosscheck.py               # unpruned search, ~6 min
python3 forms.py                    # SOP / ANF / search comparison
python3 classes.py                  # P / NP / NPN classes and cost invariance
python3 bases.py                    # SPIR-V vs AArch64 bases
```
