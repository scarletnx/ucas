# elip: elliptic curves in ucas

Goal: elliptic curves as a first-class area of ucas: over Q, number fields
where cheap, Z/n, F_p and F_{p^k}; their arithmetic, their number theory,
and everything they are used for.  The user's headline example: solve
Diophantine problems in several variables by reducing them to an elliptic
curve, e.g. find positive integers with

    a/(b+c) + b/(a+c) + c/(a+b) = 4

(the classical answer has 80-digit coordinates): homogenise, find a rational
point, map birationally to Weierstrass form, compute the group (torsion,
generators), walk multiples/combinations of generators until a point maps
back to a solution meeting the side conditions (positivity), map back and
verify exactly.  That is an example, not the scope.

House rules: as in README "Design" and absalg.md: C99, BSD make, libm only,
data-oriented (SoA, arenas, handles), exact arithmetic (use `zint` from
`src/absalg.h`, branch `absalg`; `git merge absalg` to pick up the
implementation as agent A lands it; fixed-width multiprecision for crypto
fields is fine as its own fast path).  Floats only where the mathematics is
analytic (periods, L-values, heights' archimedean part, canonical-height
estimates), and then with stated error bounds.  Every claim carries a
certificate (the point checks on the curve, the map's inverse composes to
the identity, the group order factors and the point order is verified, a
primality certificate, ...).  Each area: tests/NAME.ucas + .out, JSON
commands + tests/json, an MCP tool group in src/mcp.c, a README section;
`make test` stays green.

## Worktrees

| worktree | branch | owner |
|---|---|---|
| `~/ucas-elip` | `elip` (integration) | E1: curves over Q and Diophantine equations |
| `~/ucas-elip-ff` | `elip-ff` | E2: curves over finite fields, crypto and algorithms |

E2 merges `elip` for the shared curve core (E1 commits `src/ec.h` first:
curve model, points, group law over a generic field interface).  Shared
files get small appended hooks only.

## E1: over Q, Diophantine equations

- Curve models: general Weierstrass a1..a6, short, Montgomery, Edwards,
  twisted Edwards, Legendre, Hessian/cubic; b2..b8, c4, c6, Δ, j;
  conversions between models with the maps (and their inverses) explicit.
- Any plane cubic with a rational point → Weierstrass (Nagell's algorithm),
  quartics y^2 = quartic with a point → cubic; homogenising rational
  equations in several variables (the a/(b+c) family, sums of cubes
  x^3 + y^3 = k z^3, Mordell y^2 = x^3 + k, congruent numbers n, θ-congruent,
  rational triangles/boxes problems, x^4 ± y^4 = z^2 style descents) with
  the birational map and its inverse kept as data.
- Group law over Q (exact rationals with zint), multiples, torsion
  (Nagell–Lutz + reduction mod good primes, Mazur's list as a check),
  minimal model, Tate's algorithm (Kodaira types, c_p, conductor),
  canonical and naive heights, point search (ratpoints-style sieve), 2-descent
  (and 2-isogeny descent) for rank bounds, Sha[2] where it shows,
  saturation, generators; integral points on Mordell curves.
- The solve loop for the headline example and others: find all solutions in
  a height bound, or the smallest satisfying side conditions, verified by
  substitution into the original equation.
- Analytic side (floats, error-bounded): periods (AGM), the complex
  lattice/Weierstrass ℘, a_p and L(E,1) / analytic rank estimates, BSD
  quotient check on small curves, modularity data (a_n of the newform).

## E2: over finite fields, cryptography, algorithms

- F_p (any size: fixed-width Montgomery limbs up to 521 bits + zint) and
  F_{p^k} (reuse ring.h's GF(p^n) where it fits), binary curves over GF(2^m).
- Point counting: naive, baby-step giant-step / Mestre, Schoof (and SEA if
  feasible); group structure Z/m × Z/mn; Hasse bound; Frobenius trace;
  supersingular tests; twists; embedding degree.
- Scalar multiplication: affine, Jacobian, projective, Montgomery ladder,
  wNAF, constant-time ladder note; complete Edwards formulas.
- Named curves: secp256k1, P-256, P-384, P-521, Curve25519/X25519,
  Ed25519, Curve448, BN254/BLS12-381 parameters; validated against RFC/SEC
  test vectors.  ECDH, ECDSA (sign/verify, RFC 6979 nonces), EdDSA, point
  compression, validation; discrete log attacks for small/weak cases:
  Pollard rho, Pohlig–Hellman, MOV (pairing transfer), anomalous (Smart).
- Pairings: Weil and Tate (Miller's algorithm), degeneracy/bilinearity
  checks.
- Isogenies: Vélu's formulas, kernel polynomial, isogeny graphs (ℓ-volcano
  walk), modular polynomials for small ℓ, j-invariant tables, CM curves
  (Hilbert class polynomials for small D) and CM point counting.
- Division polynomials ψ_n, torsion fields.
- Algorithms that use curves: Lenstra ECM factoring (stage 1 + 2) and
  ECPP-style primality certificates (Atkin–Morain / Goldwasser–Kilian,
  with a checker); random curve generation with prime order.
