/*
 * ecw_impl.inc - short Weierstrass curves y^2 = x^3 + a x + b over a
 * field, as a template: ecw.c includes it once for F_p (mf, prefix ecp_)
 * and once for F_{p^k} (fk, prefix eck_).  The includer defines
 *   FIELD, FE_N, PFX(name), FE_ADD/SUB/MUL/SQR/NEG/COPY/ZERO/ONE,
 *   FE_ISZERO, FE_ISONE, FE_EQ, FE_INV (int), FE_SQRT (int), FE_SMALL
 *   (set from an int64), FE_RAND
 * Affine points carry inf; Jacobian (X : Y : Z) means (X/Z^2, Y/Z^3)
 * with Z = 0 the point at infinity.
 */

void
PFX(set)(struct PFX(curve) *E, const FIELD *K, const u64 *a, const u64 *b)
{
	u64 t[FE_N];

	memset(E, 0, sizeof *E);
	E->K = K;
	FE_COPY(K, E->a, a);
	FE_COPY(K, E->b, b);
	E->a_zero = FE_ISZERO(K, a);
	FE_SMALL(K, t, -3);
	E->a_m3 = FE_EQ(K, a, t);
}

/* 4 a^3 + 27 b^2 (the discriminant is -16 times it) */
void
PFX(disc)(const struct PFX(curve) *E, u64 *d)
{
	const FIELD *K = E->K;
	u64 t[FE_N], u[FE_N], c[FE_N];

	FE_SQR(K, t, E->a);
	FE_MUL(K, t, t, E->a);
	FE_SMALL(K, c, 4);
	FE_MUL(K, t, t, c);
	FE_SQR(K, u, E->b);
	FE_SMALL(K, c, 27);
	FE_MUL(K, u, u, c);
	FE_ADD(K, d, t, u);
}

/* j = 1728 * 4a^3 / (4a^3 + 27b^2); 0: singular */
int
PFX(jinv)(const struct PFX(curve) *E, u64 *j)
{
	const FIELD *K = E->K;
	u64 t[FE_N], d[FE_N], c[FE_N];

	PFX(disc)(E, d);
	if (FE_ISZERO(K, d))
		return 0;
	FE_SQR(K, t, E->a);
	FE_MUL(K, t, t, E->a);
	FE_SMALL(K, c, 6912);
	FE_MUL(K, t, t, c);
	FE_INV(K, d, d);
	FE_MUL(K, j, t, d);
	return 1;
}

/* x^3 + a x + b */
void
PFX(rhs)(const struct PFX(curve) *E, u64 *r, const u64 *x)
{
	const FIELD *K = E->K;
	u64 t[FE_N];

	FE_SQR(K, t, x);
	FE_ADD(K, t, t, E->a);
	FE_MUL(K, t, t, x);
	FE_ADD(K, r, t, E->b);
}

int
PFX(on_curve)(const struct PFX(curve) *E, const struct PFX(pt) *P)
{
	u64 l[FE_N], r[FE_N];

	if (P->inf)
		return 1;
	FE_SQR(E->K, l, P->y);
	PFX(rhs)(E, r, P->x);
	return FE_EQ(E->K, l, r);
}

void
PFX(set_inf)(const struct PFX(curve) *E, struct PFX(pt) *P)
{
	FE_ZERO(E->K, P->x);
	FE_ZERO(E->K, P->y);
	P->inf = 1;
}

int
PFX(eq)(const struct PFX(curve) *E, const struct PFX(pt) *P,
    const struct PFX(pt) *Q)
{
	if (P->inf || Q->inf)
		return P->inf == Q->inf;
	return FE_EQ(E->K, P->x, Q->x) && FE_EQ(E->K, P->y, Q->y);
}

void
PFX(neg)(const struct PFX(curve) *E, struct PFX(pt) *R,
    const struct PFX(pt) *P)
{
	FE_COPY(E->K, R->x, P->x);
	FE_NEG(E->K, R->y, P->y);
	R->inf = P->inf;
}

/* affine addition (chord and tangent) */
void
PFX(add)(const struct PFX(curve) *E, struct PFX(pt) *R,
    const struct PFX(pt) *P, const struct PFX(pt) *Q)
{
	const FIELD *K = E->K;
	u64 l[FE_N], t[FE_N], x3[FE_N], y3[FE_N];

	if (P->inf) {
		*R = *Q;
		return;
	}
	if (Q->inf) {
		*R = *P;
		return;
	}
	if (FE_EQ(K, P->x, Q->x)) {
		FE_ADD(K, t, P->y, Q->y);
		if (FE_ISZERO(K, t)) {
			PFX(set_inf)(E, R);
			return;
		}
		/* tangent: (3x^2 + a) / 2y */
		FE_SQR(K, l, P->x);
		FE_ADD(K, t, l, l);
		FE_ADD(K, l, l, t);
		FE_ADD(K, l, l, E->a);
		FE_ADD(K, t, P->y, P->y);
	} else {
		FE_SUB(K, l, Q->y, P->y);
		FE_SUB(K, t, Q->x, P->x);
	}
	if (!FE_INV(K, t, t))
		fail("ecff: a non-invertible denominator in the group law");
	FE_MUL(K, l, l, t);
	FE_SQR(K, x3, l);
	FE_SUB(K, x3, x3, P->x);
	FE_SUB(K, x3, x3, Q->x);
	FE_SUB(K, t, P->x, x3);
	FE_MUL(K, y3, l, t);
	FE_SUB(K, y3, y3, P->y);
	FE_COPY(K, R->x, x3);
	FE_COPY(K, R->y, y3);
	R->inf = 0;
}

void
PFX(sub)(const struct PFX(curve) *E, struct PFX(pt) *R,
    const struct PFX(pt) *P, const struct PFX(pt) *Q)
{
	struct PFX(pt) N;

	PFX(neg)(E, &N, Q);
	PFX(add)(E, R, P, &N);
}

/* ---- Jacobian ---- */

void
PFX(to_jac)(const struct PFX(curve) *E, struct PFX(jpt) *J,
    const struct PFX(pt) *P)
{
	const FIELD *K = E->K;

	if (P->inf) {
		FE_ONE(K, J->X);
		FE_ONE(K, J->Y);
		FE_ZERO(K, J->Z);
		return;
	}
	FE_COPY(K, J->X, P->x);
	FE_COPY(K, J->Y, P->y);
	FE_ONE(K, J->Z);
}

void
PFX(from_jac)(const struct PFX(curve) *E, struct PFX(pt) *P,
    const struct PFX(jpt) *J)
{
	const FIELD *K = E->K;
	u64 zi[FE_N], z2[FE_N];

	if (FE_ISZERO(K, J->Z)) {
		PFX(set_inf)(E, P);
		return;
	}
	if (!FE_INV(K, zi, J->Z))
		fail("ecff: Z not invertible");
	FE_SQR(K, z2, zi);
	FE_MUL(K, P->x, J->X, z2);
	FE_MUL(K, z2, z2, zi);
	FE_MUL(K, P->y, J->Y, z2);
	P->inf = 0;
}

void
PFX(jdbl)(const struct PFX(curve) *E, struct PFX(jpt) *R,
    const struct PFX(jpt) *P)
{
	const FIELD *K = E->K;
	u64 XX[FE_N], YY[FE_N], YYYY[FE_N], ZZ[FE_N], S[FE_N], M[FE_N],
	    T[FE_N], t[FE_N];

	if (FE_ISZERO(K, P->Z) || FE_ISZERO(K, P->Y)) {
		FE_ONE(K, R->X);
		FE_ONE(K, R->Y);
		FE_ZERO(K, R->Z);
		return;
	}
	FE_SQR(K, XX, P->X);
	FE_SQR(K, YY, P->Y);
	FE_SQR(K, YYYY, YY);
	FE_SQR(K, ZZ, P->Z);
	FE_ADD(K, S, P->X, YY);
	FE_SQR(K, S, S);
	FE_SUB(K, S, S, XX);
	FE_SUB(K, S, S, YYYY);
	FE_ADD(K, S, S, S);
	if (E->a_m3) {
		FE_SUB(K, M, P->X, ZZ);
		FE_ADD(K, t, P->X, ZZ);
		FE_MUL(K, M, M, t);
		FE_ADD(K, t, M, M);
		FE_ADD(K, M, M, t);
	} else {
		FE_ADD(K, M, XX, XX);
		FE_ADD(K, M, M, XX);
		if (!E->a_zero) {
			FE_SQR(K, t, ZZ);
			FE_MUL(K, t, t, E->a);
			FE_ADD(K, M, M, t);
		}
	}
	FE_SQR(K, T, M);
	FE_SUB(K, T, T, S);
	FE_SUB(K, T, T, S);
	/* Z3 before Y, X are overwritten (R may be P) */
	FE_ADD(K, t, P->Y, P->Z);
	FE_SQR(K, t, t);
	FE_SUB(K, t, t, YY);
	FE_SUB(K, R->Z, t, ZZ);
	FE_COPY(K, R->X, T);
	FE_SUB(K, t, S, T);
	FE_MUL(K, t, M, t);
	FE_ADD(K, YYYY, YYYY, YYYY);
	FE_ADD(K, YYYY, YYYY, YYYY);
	FE_ADD(K, YYYY, YYYY, YYYY);
	FE_SUB(K, R->Y, t, YYYY);
}

void
PFX(jadd)(const struct PFX(curve) *E, struct PFX(jpt) *R,
    const struct PFX(jpt) *P, const struct PFX(jpt) *Q)
{
	const FIELD *K = E->K;
	u64 Z1Z1[FE_N], Z2Z2[FE_N], U1[FE_N], U2[FE_N], S1[FE_N], S2[FE_N],
	    H[FE_N], r[FE_N], HH[FE_N], HHH[FE_N], V[FE_N], t[FE_N];

	if (FE_ISZERO(K, P->Z)) {
		*R = *Q;
		return;
	}
	if (FE_ISZERO(K, Q->Z)) {
		*R = *P;
		return;
	}
	FE_SQR(K, Z1Z1, P->Z);
	FE_SQR(K, Z2Z2, Q->Z);
	FE_MUL(K, U1, P->X, Z2Z2);
	FE_MUL(K, U2, Q->X, Z1Z1);
	FE_MUL(K, S1, P->Y, Q->Z);
	FE_MUL(K, S1, S1, Z2Z2);
	FE_MUL(K, S2, Q->Y, P->Z);
	FE_MUL(K, S2, S2, Z1Z1);
	FE_SUB(K, H, U2, U1);
	FE_SUB(K, r, S2, S1);
	if (FE_ISZERO(K, H)) {
		if (FE_ISZERO(K, r)) {
			PFX(jdbl)(E, R, P);
			return;
		}
		FE_ONE(K, R->X);
		FE_ONE(K, R->Y);
		FE_ZERO(K, R->Z);
		return;
	}
	FE_SQR(K, HH, H);
	FE_MUL(K, HHH, H, HH);
	FE_MUL(K, V, U1, HH);
	FE_MUL(K, t, P->Z, Q->Z);
	FE_MUL(K, R->Z, t, H);
	FE_SQR(K, t, r);
	FE_SUB(K, t, t, HHH);
	FE_SUB(K, t, t, V);
	FE_SUB(K, t, t, V);
	FE_COPY(K, R->X, t);
	FE_SUB(K, t, V, t);
	FE_MUL(K, t, r, t);
	FE_MUL(K, S1, S1, HHH);
	FE_SUB(K, R->Y, t, S1);
}

int
PFX(jis_inf)(const struct PFX(curve) *E, const struct PFX(jpt) *P)
{
	return FE_ISZERO(E->K, P->Z);
}

/* ---- scalar multiplication ---- */

/* affine double-and-add (an inversion per step: the textbook way) */
void
PFX(mul_affine)(const struct PFX(curve) *E, struct PFX(pt) *R,
    const struct PFX(pt) *P, const nn *k)
{
	struct PFX(pt) A, B = *P;
	int i, nb = nn_bits(k);

	PFX(set_inf)(E, &A);
	for (i = nb - 1; i >= 0; i--) {
		PFX(add)(E, &A, &A, &A);
		if (nn_bit(k, i))
			PFX(add)(E, &A, &A, &B);
	}
	*R = A;
}

/* Montgomery ladder on Jacobian points: the same add + double for every
 * bit (regular, but the field code is not constant-time) */
void
PFX(mul_ladder)(const struct PFX(curve) *E, struct PFX(pt) *R,
    const struct PFX(pt) *P, const nn *k)
{
	struct PFX(jpt) R0, R1;
	int i, nb = nn_bits(k);

	PFX(to_jac)(E, &R1, P);
	FE_ONE(E->K, R0.X);
	FE_ONE(E->K, R0.Y);
	FE_ZERO(E->K, R0.Z);
	for (i = nb - 1; i >= 0; i--) {
		if (nn_bit(k, i)) {
			PFX(jadd)(E, &R0, &R0, &R1);
			PFX(jdbl)(E, &R1, &R1);
		} else {
			PFX(jadd)(E, &R1, &R0, &R1);
			PFX(jdbl)(E, &R0, &R0);
		}
	}
	PFX(from_jac)(E, R, &R0);
}

/* width-w NAF digits of k (least significant first); returns the count */
static int
PFX(wnaf)(const nn *k0, int w, signed char *d, int max)
{
	nn k = *k0;
	int n = 0;
	u64 mod = (u64)1 << w;

	while (!nn_is_zero(&k)) {
		int di = 0;
		if (n >= max)
			fail("ecff: scalar too long");
		if (k.w[0] & 1) {
			u64 r = k.w[0] & (mod - 1);
			if (r >= mod / 2) {
				di = (int)r - (int)mod;
				nn_add_u64(&k, &k, (u64)(-di));
			} else {
				di = (int)r;
				nn_sub_u64(&k, &k, (u64)di);
			}
		}
		d[n++] = (signed char)di;
		nn_shr(&k, &k, 1);
	}
	return n;
}

void
PFX(mul_wnaf)(const struct PFX(curve) *E, struct PFX(pt) *R,
    const struct PFX(pt) *P, const nn *k, int w)
{
	struct PFX(jpt) tab[16], A, P2, N;
	signed char d[64 * NW + 2];
	int n, i, j;

	if (w < 2)
		w = 2;
	if (w > 5)
		w = 5;
	PFX(to_jac)(E, &tab[0], P);
	PFX(jdbl)(E, &P2, &tab[0]);
	for (j = 1; j < (1 << (w - 2)); j++)
		PFX(jadd)(E, &tab[j], &tab[j - 1], &P2);
	n = PFX(wnaf)(k, w, d, (int)sizeof d);
	FE_ONE(E->K, A.X);
	FE_ONE(E->K, A.Y);
	FE_ZERO(E->K, A.Z);
	for (i = n - 1; i >= 0; i--) {
		PFX(jdbl)(E, &A, &A);
		if (d[i] > 0)
			PFX(jadd)(E, &A, &A, &tab[d[i] / 2]);
		else if (d[i] < 0) {
			N = tab[(-d[i]) / 2];
			FE_NEG(E->K, N.Y, N.Y);
			PFX(jadd)(E, &A, &A, &N);
		}
	}
	PFX(from_jac)(E, R, &A);
}

void
PFX(mul)(const struct PFX(curve) *E, struct PFX(pt) *R,
    const struct PFX(pt) *P, const nn *k)
{
	if (P->inf || nn_is_zero(k)) {
		PFX(set_inf)(E, R);
		return;
	}
	PFX(mul_wnaf)(E, R, P, k, 4);
}

void
PFX(mul_u64)(const struct PFX(curve) *E, struct PFX(pt) *R,
    const struct PFX(pt) *P, u64 k)
{
	nn kk;

	nn_set(&kk, k);
	PFX(mul)(E, R, P, &kk);
}

/* a point with the given x, if x^3 + a x + b is a square */
int
PFX(lift_x)(const struct PFX(curve) *E, struct PFX(pt) *P, const u64 *x)
{
	u64 r[FE_N];

	PFX(rhs)(E, r, x);
	if (!FE_SQRT(E->K, P->y, r))
		return 0;
	FE_COPY(E->K, P->x, x);
	P->inf = 0;
	return 1;
}

/* a pseudo-random point (deterministic PRNG) */
void
PFX(random_point)(const struct PFX(curve) *E, struct PFX(pt) *P)
{
	u64 x[FE_N];
	int tries;

	for (tries = 0; tries < 100000; tries++) {
		FE_RAND(E->K, x);
		if (PFX(lift_x)(E, P, x)) {
			if (nn_rand64() & 1)
				FE_NEG(E->K, P->y, P->y);
			return;
		}
	}
	fail("ecff: no point found");
}
