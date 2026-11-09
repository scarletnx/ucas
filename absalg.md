# absalg: exact abstract algebra, homological algebra and proofs in ucas

Goal: ucas gains an exact abstract-algebra framework broad enough to state and
*prove* (not just evaluate on samples) the decidable / constructively provable
parts of algebra, and uses it to prove Proposition 0.1 of `~/mb/bockstein.pdf`
(text: `pdftotext -layout ~/mb/bockstein.pdf -`) from general machinery.
Nothing about the Bockstein proposition may be hard-coded beyond its
*statement*; every step must come from general definitions, algorithms and
proof rules that also prove other theorems.

House rules (same as the rest of ucas, see README "Design" and `src/ucas.h`):

- C99 only (`CSTD=c99`), BSD make, no dependencies beyond libm.  The
  existing code already uses `__int128` in ring.h; allowed, nothing else
  non-standard.
- Data-oriented design: struct-of-arrays for every hot table, arenas and
  integer handles instead of pointer graphs, dense column loops, no per-
  object malloc in inner loops.  Hash-cons where identity matters.
- Exact: no floating point anywhere in this framework.  Integers overflow
  into an arbitrary-precision representation, never silently wrap.
- Every answer that claims a fact carries a checkable certificate (a
  witness, a transform matrix, or a proof tree checked by the kernel).
- Each area: `tests/NAME.ucas` + `.out`, JSON commands in `src/jsonmode.c`'s
  table (+ `tests/json/NAME.jsonl`), the commands grouped into an MCP tool in
  `src/mcp.c`, a README section.  `make test` must stay green.

## Branches / worktrees on fbsd-srv

| worktree | branch | owner | contents |
|---|---|---|---|
| `~/ucas-absalg` | `absalg` | agent A (core) — integration branch | `src/absalg.h` (public API), `zint.c`, `zmat.c`, `fgab.c`, `pgrp.c`, `ideal.c`, ... |
| `~/ucas-absalg-hom` | `absalg-hom` | agent B (homological) | `homalg.h`, `cx.c`, `simp.c`, `excouple.c`, ... |
| `~/ucas-absalg-prove` | `absalg-prove` | agent C (proofs) | `prove.h`, `kernel.c`, `pab.c` (parametric groups), `thm_*.c`, ... |

B and C `git merge absalg` whenever A pushes API they need.  Everyone
commits only in their own worktree.  Shared files (Makefile, parse.c,
nlp.c, jsonmode.c, mcp.c, README.md) get *small, appended* hooks only
(one `if (xxx_command(p))` line, one SRCS+= line, table rows at the end of
the area's block), so the final merges are trivial.

## Layering (API contract)

A — core (`src/absalg.h`), first commit = the header, then implementations:

1. `zint`: arbitrary-precision integers in an arena (SoA: sign/len/offset
   columns + one limb array), with a small-int fast path; add, sub, mul,
   divmod (floor and Euclidean), gcd/xgcd, pow, p-adic valuation, compare,
   parse/print.
2. `zmat`: dense integer matrices over zint; mul, transpose, rank, det,
   Hermite and Smith normal forms *with unimodular transforms* (U A V = D,
   verified), kernel/image/cokernel bases, solving A x = b over Z.
   Also matrices over Q and over Z/p (rref, nullspace, inverse).
3. `fgab`: finitely generated abelian groups as presentations Z^g / R;
   invariant factors and primary decomposition; elements as vectors;
   homomorphisms as matrices (well-definedness checked); kernel, image,
   cokernel, quotient, sum/intersection of subgroups, m·G, G[m] (m-torsion),
   T_p(G), subgroup membership with witness, element order, isomorphism test
   with explicit isomorphism, direct sums, Hom and ⊗ and Tor/Ext of f.g.
   abelian groups, exactness test of a sequence.  Subquotients A/B with
   B ⊆ A ⊆ Z^g as the canonical object for (co)homology.
4. Beyond abelian groups: finite permutation groups (Schreier–Sims: order,
   membership, orbits/stabilisers, subgroup tests, normal closure, centre,
   derived series, solvable/nilpotent, cosets), finite groups by Cayley
   table, homomorphism checks; rings and ideals: Z, Q, Z/n, F_q (reuse
   `ring.h`), polynomial rings over Q and F_p with Gröbner bases (Buchberger
   with criteria), ideal membership with cofactor certificate, ideal
   equality/inclusion; modules over PIDs (Z and F[x]): presentation → SNF.

B — homological (`src/homalg.h`), on top of A:

1. (Co)chain complexes of f.g. free Z-modules (matrices d^n), check d∘d=0;
   chain maps; (co)homology as fgab subquotients; coefficients Z/m (⊗) and
   any f.g. abelian group; universal coefficients as a check.
2. Spaces: simplicial complexes (SoA simplices), standard models (S^n, T^n,
   RP^n, Klein bottle, lens spaces L(m;1), CP^2 via a known triangulation or
   a cellular model, products/wedges/suspensions), cellular complexes given
   by matrices.
3. Short exact sequences of complexes → long exact sequence with the
   connecting homomorphism (snake lemma, computed), exactness verified.
4. Exact couples (D, E, i, j, k with degrees), derived couples computed
   exactly (D' = im i, E' = ker d / im d, i', j', k' well defined and
   checked), the spectral sequence pages E_r, d_r, stabilisation detection
   (E_∞), and the Bockstein spectral sequence of a cochain complex for a
   prime p (exactly the construction in the paper).  Element tracing:
   given a ∈ H^n(X;Z) report j_r(p^{r-1}a), whether it is in ker d_r / im d_r.

C — proofs (`src/prove.h`), on top of A and B:

1. A small LCF-style kernel: terms (hash-consed DAG, SoA) for objects
   (groups, maps, elements, subgroups, integers, primes), formulas
   (=, ∈, ⊆, exact, iso, ¬, ∧, ∨, →, ∀, ∃ over typed sorts with symbolic
   integer parameters n, r, p, e...), and theorems that can only be made by
   kernel rules.  Every proof is a DAG of rule applications that can be
   re-checked by an independent checker and printed.
2. Decision procedures that return kernel proofs: Presburger arithmetic
   (at least linear arithmetic with valuations / divisibility over Z, for
   symbolic r, exponents), the theory of f.g. abelian groups with symbolic
   exponents ("parametric abelian groups": Z, Z/p^e with e symbolic,
   p^k·G, G[p], quotients — normal forms by case splits on Presburger
   conditions), and subobject-lattice reasoning in abelian categories
   (diagram chasing: exactness, kernels/images under composition, the
   lemmas about derived couples proved *from the exact couple axioms*,
   e.g. D_r = im i^{r-1}, im k_r = ker i_r, ker j_r = im i_r, d_r = j_r k_r).
3. Structural reductions as theorems: structure theorem for f.g. abelian
   groups (from SNF certificates), splitting of finite-type free cochain
   complexes over a PID into elementary complexes (Z[n] and Z --m--> Z),
   additivity/naturality of the exact couple and Bockstein spectral
   sequence, transitivity of Aut(G) on generators of Z summands.
4. A statement language in ucas syntax (`theorem NAME: ...`, `prove NAME`,
   `proof NAME` prints the tree, `check NAME` re-checks) plus JSON/MCP ops.
5. Target: `prove bockstein` states Prop 0.1 for all cochain complexes of
   finite type, all primes p, all n ≥ 0, all r ≥ 1, and proves it, both
   following the paper's argument (lemmas about derived couples + the
   final valuation argument) and by an independent route (reduction to
   elementary complexes).  Cross-validated by B's concrete BSS computations
   on RP^n, lens spaces, tori etc. for every r until stabilisation.
