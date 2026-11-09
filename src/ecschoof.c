/*
 * ecschoof.c - division polynomials and Schoof's algorithm: for small
 * primes l, t mod l from the Frobenius (x^p, y^p) on the l-torsion,
 * computed in F_p[x]/(h) with h = psi_l (or a Frobenius-stable factor of
 * it found along the way), points written (X(x), Y(x) y) and added in
 * Jacobian coordinates.  The residues are combined by CRT; once the
 * remaining candidates are few, a baby-step giant-step match on random
 * points finishes, and the count is certified like every other (eccount.c).
 * SEA (Elkies primes) is in ecsea.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

struct ecsch_log ecsch_last;

/* ---- division polynomials ---- */

void
ec_divpolys(const struct ecp_curve *E, struct px *f, int nmax)
{
	const struct mf *F = E->K;
	struct px FF, F2, t, u, v;
	u64 c[FW], a2[FW], ab[FW], b2[FW], a3[FW];
	int n;

	px_init(&FF);
	px_init(&F2);
	px_init(&t);
	px_init(&u);
	px_init(&v);
	for (n = 0; n <= nmax; n++)
		px_init(&f[n]);
	if (nmax >= 1)
		px_set_u64(F, &f[1], 1);
	if (nmax >= 2)
		px_set_u64(F, &f[2], 1);
	/* F = 4(x^3 + a x + b) = (2y)^2 */
	px_reserve(F, &FF, 3);
	mf_mul_u64(F, px_coef(F, &FF, 0), E->b, 4);
	mf_mul_u64(F, px_coef(F, &FF, 1), E->a, 4);
	mf_zero(F, px_coef(F, &FF, 2));
	mf_from_u64(F, px_coef(F, &FF, 3), 4);
	FF.d = 3;
	px_norm(F, &FF);
	px_sqr(F, &F2, &FF);
	mf_sqr(F, a2, E->a);
	mf_mul(F, ab, E->a, E->b);
	mf_sqr(F, b2, E->b);
	mf_mul(F, a3, a2, E->a);
	if (nmax >= 3) {
		/* 3x^4 + 6a x^2 + 12b x - a^2 */
		struct px *g = &f[3];
		px_reserve(F, g, 4);
		mf_neg(F, px_coef(F, g, 0), a2);
		mf_mul_u64(F, px_coef(F, g, 1), E->b, 12);
		mf_mul_u64(F, px_coef(F, g, 2), E->a, 6);
		mf_zero(F, px_coef(F, g, 3));
		mf_from_u64(F, px_coef(F, g, 4), 3);
		g->d = 4;
		px_norm(F, g);
	}
	if (nmax >= 4) {
		/* 2(x^6 + 5a x^4 + 20b x^3 - 5a^2 x^2 - 4ab x - 8b^2 - a^3) */
		struct px *g = &f[4];
		px_reserve(F, g, 6);
		mf_mul_u64(F, c, b2, 8);
		mf_add(F, c, c, a3);
		mf_neg(F, px_coef(F, g, 0), c);
		mf_mul_u64(F, c, ab, 4);
		mf_neg(F, px_coef(F, g, 1), c);
		mf_mul_u64(F, c, a2, 5);
		mf_neg(F, px_coef(F, g, 2), c);
		mf_mul_u64(F, px_coef(F, g, 3), E->b, 20);
		mf_mul_u64(F, px_coef(F, g, 4), E->a, 5);
		mf_zero(F, px_coef(F, g, 5));
		mf_one(F, px_coef(F, g, 6));
		g->d = 6;
		px_norm(F, g);
		mf_from_u64(F, c, 2);
		px_scale(F, g, g, c);
	}
	for (n = 5; n <= nmax; n++) {
		int m = n / 2;
		if (n & 1) {
			/* f_{2m+1} = f_{m+2} f_m^3 - f_{m-1} f_{m+1}^3, F^2 on the
			 * even-index pair */
			px_sqr(F, &t, &f[m]);
			px_mul(F, &t, &t, &f[m]);
			px_mul(F, &t, &t, &f[m + 2]);
			px_sqr(F, &u, &f[m + 1]);
			px_mul(F, &u, &u, &f[m + 1]);
			px_mul(F, &u, &u, &f[m - 1]);
			if (m % 2 == 0)
				px_mul(F, &t, &t, &F2);
			else
				px_mul(F, &u, &u, &F2);
			px_sub(F, &f[n], &t, &u);
		} else {
			/* f_{2m} = f_m (f_{m+2} f_{m-1}^2 - f_{m-2} f_{m+1}^2) */
			px_sqr(F, &t, &f[m - 1]);
			px_mul(F, &t, &t, &f[m + 2]);
			px_sqr(F, &u, &f[m + 1]);
			px_mul(F, &u, &u, &f[m - 2]);
			px_sub(F, &v, &t, &u);
			px_mul(F, &f[n], &v, &f[m]);
		}
	}
	px_free(&FF);
	px_free(&F2);
	px_free(&t);
	px_free(&u);
	px_free(&v);
}

/* ---- points of E[l] over F_p[x]/(h): (X : Y y : Z) ---- */

struct sctx {
	const struct mf *F;
	const struct ecp_curve *E;
	struct pxmod M;
	struct px f;        /* x^3 + a x + b mod h */
	struct px g;        /* a factor found (split) */
	int split;
};

struct spt {
	struct px X, Y, Z;  /* Y is the coefficient of y */
};

static void
spt_init(struct spt *P)
{
	px_init(&P->X);
	px_init(&P->Y);
	px_init(&P->Z);
}

static void
spt_free(struct spt *P)
{
	px_free(&P->X);
	px_free(&P->Y);
	px_free(&P->Z);
}

static void
spt_copy(const struct mf *F, struct spt *R, const struct spt *P)
{
	px_copy(F, &R->X, &P->X);
	px_copy(F, &R->Y, &P->Y);
	px_copy(F, &R->Z, &P->Z);
}

#define MM(r, a, b) px_mulmod(F, &C->M, (r), (a), (b))

/* doubling: a Z^4 term, y^2 = f */
static void
sdbl(struct sctx *C, struct spt *R, const struct spt *P)
{
	const struct mf *F = C->F;
	struct px XX, YY, YYYY, ZZ, S, Mv, T, t;

	px_init(&XX);
	px_init(&YY);
	px_init(&YYYY);
	px_init(&ZZ);
	px_init(&S);
	px_init(&Mv);
	px_init(&T);
	px_init(&t);
	MM(&XX, &P->X, &P->X);
	MM(&YY, &P->Y, &P->Y);
	MM(&YY, &YY, &C->f);              /* (Y y)^2 = Y^2 f */
	MM(&YYYY, &YY, &YY);
	MM(&ZZ, &P->Z, &P->Z);
	/* S = 4 X YY */
	MM(&S, &P->X, &YY);
	px_add(F, &S, &S, &S);
	px_add(F, &S, &S, &S);
	/* M = 3 XX + a ZZ^2 */
	px_add(F, &Mv, &XX, &XX);
	px_add(F, &Mv, &Mv, &XX);
	if (!C->E->a_zero) {
		MM(&t, &ZZ, &ZZ);
		px_scale(F, &t, &t, C->E->a);
		px_add(F, &Mv, &Mv, &t);
	}
	MM(&T, &Mv, &Mv);
	px_sub(F, &T, &T, &S);
	px_sub(F, &T, &T, &S);
	/* Z3 = 2 Y Z (a y-multiple); Y3 = M (S - X3) - 8 YYYY (plain): then
	 * rescale by y: X f, Y3 f y, Z3' = 2 Y Z f */
	MM(&t, &P->Y, &P->Z);
	px_add(F, &t, &t, &t);
	MM(&R->Z, &t, &C->f);
	px_sub(F, &S, &S, &T);
	MM(&S, &Mv, &S);
	px_add(F, &YYYY, &YYYY, &YYYY);
	px_add(F, &YYYY, &YYYY, &YYYY);
	px_add(F, &YYYY, &YYYY, &YYYY);
	px_sub(F, &S, &S, &YYYY);
	MM(&R->Y, &S, &C->f);
	MM(&R->X, &T, &C->f);             /* X f, scaled by y^2 = f */
	px_free(&XX);
	px_free(&YY);
	px_free(&YYYY);
	px_free(&ZZ);
	px_free(&S);
	px_free(&Mv);
	px_free(&T);
	px_free(&t);
}

/* is the polynomial zero mod h */
static int
iszero(const struct px *a)
{
	return a->d < 0;
}

/* R = P + Q; check: test H and r for a partial vanishing (split) */
static void
sadd(struct sctx *C, struct spt *R, const struct spt *P, const struct spt *Q,
    int check)
{
	const struct mf *F = C->F;
	struct px Z1Z1, Z2Z2, U1, U2, S1, S2, H, r, HH, HHH, V, t;

	if (iszero(&P->Z)) {
		spt_copy(F, R, Q);
		return;
	}
	if (iszero(&Q->Z)) {
		spt_copy(F, R, P);
		return;
	}
	px_init(&Z1Z1);
	px_init(&Z2Z2);
	px_init(&U1);
	px_init(&U2);
	px_init(&S1);
	px_init(&S2);
	px_init(&H);
	px_init(&r);
	px_init(&HH);
	px_init(&HHH);
	px_init(&V);
	px_init(&t);
	MM(&Z1Z1, &P->Z, &P->Z);
	MM(&Z2Z2, &Q->Z, &Q->Z);
	MM(&U1, &P->X, &Z2Z2);
	MM(&U2, &Q->X, &Z1Z1);
	MM(&S1, &P->Y, &Q->Z);
	MM(&S1, &S1, &Z2Z2);
	MM(&S2, &Q->Y, &P->Z);
	MM(&S2, &S2, &Z1Z1);
	px_sub(F, &H, &U2, &U1);
	px_sub(F, &r, &S2, &S1);
	if (iszero(&H)) {
		if (iszero(&r))
			sdbl(C, R, P);
		else {
			px_add(F, &t, &S1, &S2);
			if (!iszero(&t) && check) {
				/* equal on some roots, opposite on others */
				px_gcd(F, &C->g, &r, &C->M.m);
				C->split = 1;
			}
			px_set_u64(F, &R->X, 1);
			px_set_u64(F, &R->Y, 1);
			R->Z.d = -1;
		}
		goto out;
	}
	if (check) {
		px_gcd(F, &t, &H, &C->M.m);
		if (t.d > 0) {
			px_copy(F, &C->g, &t);
			C->split = 1;
			goto out;
		}
	}
	MM(&HH, &H, &H);
	MM(&HHH, &H, &HH);
	MM(&V, &U1, &HH);
	MM(&t, &P->Z, &Q->Z);
	MM(&R->Z, &t, &H);
	MM(&t, &r, &r);
	MM(&t, &t, &C->f);                /* (r y)^2 */
	px_sub(F, &t, &t, &HHH);
	px_sub(F, &t, &t, &V);
	px_sub(F, &t, &t, &V);
	px_copy(F, &R->X, &t);
	px_sub(F, &t, &V, &t);
	MM(&t, &r, &t);
	MM(&S1, &S1, &HHH);
	px_sub(F, &R->Y, &t, &S1);
out:
	px_free(&Z1Z1);
	px_free(&Z2Z2);
	px_free(&U1);
	px_free(&U2);
	px_free(&S1);
	px_free(&S2);
	px_free(&H);
	px_free(&r);
	px_free(&HH);
	px_free(&HHH);
	px_free(&V);
	px_free(&t);
}

static void
smul(struct sctx *C, struct spt *R, const struct spt *P, u64 k)
{
	const struct mf *F = C->F;
	struct spt A;
	int i;

	spt_init(&A);
	px_set_u64(F, &A.X, 1);
	px_set_u64(F, &A.Y, 1);
	A.Z.d = -1;
	for (i = 63 - __builtin_clzll(k); i >= 0; i--) {
		sdbl(C, &A, &A);
		if ((k >> i) & 1)
			sadd(C, &A, &A, P, 0);
	}
	spt_copy(F, R, &A);
	spt_free(&A);
}

/* P == +-Q (Jacobian, cross-multiplied): 1 equal, -1 opposite, 0 neither */
static int
scmp(struct sctx *C, const struct spt *P, const struct spt *Q)
{
	const struct mf *F = C->F;
	struct px a, b, z1, z2;
	int res = 0;

	px_init(&a);
	px_init(&b);
	px_init(&z1);
	px_init(&z2);
	MM(&z1, &P->Z, &P->Z);
	MM(&z2, &Q->Z, &Q->Z);
	MM(&a, &P->X, &z2);
	MM(&b, &Q->X, &z1);
	if (px_eq(F, &a, &b)) {
		MM(&z1, &z1, &P->Z);
		MM(&z2, &z2, &Q->Z);
		MM(&a, &P->Y, &z2);
		MM(&b, &Q->Y, &z1);
		if (px_eq(F, &a, &b))
			res = 1;
		else {
			px_add(F, &a, &a, &b);
			if (iszero(&a))
				res = -1;
		}
	}
	px_free(&a);
	px_free(&b);
	px_free(&z1);
	px_free(&z2);
	return res;
}

/*
 * t mod l for an odd prime l, working modulo h (psi_l, or a factor).
 * Returns t mod l, or -1 on failure; *deg the degree of the final h.
 */
static int
schoof_l(const struct ecp_curve *E, int l, const struct px *psi, int *deg)
{
	const struct mf *F = E->K;
	struct sctx C;
	struct px h, xp, t;
	struct spt pi, pi2, P, Q, S, T;
	nn e;
	int q = (int)nn_mod_u64(&F->P, (u64)l), tau = -1, k, round;

	px_init(&h);
	px_init(&xp);
	px_init(&t);
	px_copy(F, &h, psi);
	px_monic(F, &h);
	spt_init(&pi);
	spt_init(&pi2);
	spt_init(&P);
	spt_init(&Q);
	spt_init(&S);
	spt_init(&T);
	memset(&C, 0, sizeof C);
	C.F = F;
	C.E = E;
	px_init(&C.f);
	px_init(&C.g);
	for (round = 0; round < 40 && tau < 0; round++) {
		struct px fx;
		pxmod_init(F, &C.M, &h);
		C.split = 0;
		*deg = h.d;
		/* f = x^3 + a x + b */
		px_init(&fx);
		px_reserve(F, &fx, 3);
		mf_copy(F, px_coef(F, &fx, 0), E->b);
		mf_copy(F, px_coef(F, &fx, 1), E->a);
		mf_zero(F, px_coef(F, &fx, 2));
		mf_one(F, px_coef(F, &fx, 3));
		fx.d = 3;
		px_norm(F, &fx);
		px_rem(F, &C.f, &fx, &h);
		/* pi = (x^p, f^((p-1)/2) y), pi^2 */
		px_set_x(F, &t);
		px_powmod(F, &C.M, &pi.X, &t, &F->P);
		nn_sub_u64(&e, &F->P, 1);
		nn_shr(&e, &e, 1);
		px_powmod(F, &C.M, &pi.Y, &C.f, &e);
		px_set_u64(F, &pi.Z, 1);
		px_powmod(F, &C.M, &pi2.X, &pi.X, &F->P);
		px_powmod(F, &C.M, &pi2.Y, &pi.Y, &F->P);
		px_mulmod(F, &C.M, &pi2.Y, &pi2.Y, &pi.Y);
		px_set_u64(F, &pi2.Z, 1);
		px_free(&fx);
		/* P = (x, y); Q = q P */
		px_set_x(F, &P.X);
		px_rem(F, &P.X, &P.X, &h);
		px_set_u64(F, &P.Y, 1);
		px_set_u64(F, &P.Z, 1);
		smul(&C, &Q, &P, (u64)q);
		/* S = pi^2 + q P, watching for pi^2 = +-q P on part of E[l] */
		sadd(&C, &S, &pi2, &Q, 1);
		if (C.split) {
			goto split;
		}
		if (iszero(&S.Z)) {
			tau = 0;
			break;
		}
		/* tau pi for tau = 1 .. (l-1)/2 */
		spt_copy(F, &T, &pi);
		for (k = 1; k <= (l - 1) / 2; k++) {
			int c = scmp(&C, &T, &S);
			if (c == 1) {
				tau = k;
				break;
			}
			if (c == -1) {
				tau = l - k;
				break;
			}
			if (k == 1)
				sdbl(&C, &T, &T);
			else
				sadd(&C, &T, &T, &pi, 0);
		}
		if (tau < 0) {
			/* no match: a hidden partial coincidence; look for a
			 * factor through the gcd checks */
			spt_copy(F, &T, &pi);
			for (k = 1; k <= (l - 1) / 2 && !C.split; k++)
				sadd(&C, &T, &T, &pi, 1);
			if (!C.split)
				break;
			goto split;
		}
		break;
split:
		/* continue modulo the smaller of g and h / g */
		{
			struct px g2;
			px_init(&g2);
			px_divmod(F, &g2, NULL, &h, &C.g);
			px_monic(F, &g2);
			if (C.g.d > 0 && (g2.d <= 0 || C.g.d <= g2.d))
				px_copy(F, &h, &C.g);
			else if (g2.d > 0)
				px_copy(F, &h, &g2);
			px_free(&g2);
		}
		pxmod_free(&C.M);
	}
	if (C.M.m.c)
		pxmod_free(&C.M);
	spt_free(&pi);
	spt_free(&pi2);
	spt_free(&P);
	spt_free(&Q);
	spt_free(&S);
	spt_free(&T);
	px_free(&C.f);
	px_free(&C.g);
	px_free(&h);
	px_free(&xp);
	px_free(&t);
	return tau;
}

/* t mod 2: is there a 2-torsion point (gcd(x^p - x, x^3 + a x + b)) */
static int
schoof_2(const struct ecp_curve *E)
{
	const struct mf *F = E->K;
	struct px f, x, h, g;
	struct pxmod M;
	int r;

	px_init(&f);
	px_init(&x);
	px_init(&h);
	px_init(&g);
	px_reserve(F, &f, 3);
	mf_copy(F, px_coef(F, &f, 0), E->b);
	mf_copy(F, px_coef(F, &f, 1), E->a);
	mf_zero(F, px_coef(F, &f, 2));
	mf_one(F, px_coef(F, &f, 3));
	f.d = 3;
	pxmod_init(F, &M, &f);
	px_set_x(F, &x);
	px_powmod(F, &M, &h, &x, &F->P);
	px_sub(F, &h, &h, &x);
	px_gcd(F, &g, &h, &f);
	r = g.d > 0 ? 0 : 1;      /* a root: #E even, t = p + 1 = 0 mod 2 */
	pxmod_free(&M);
	px_free(&f);
	px_free(&x);
	px_free(&h);
	px_free(&g);
	return r;
}

/* CRT: (c mod M) and (r mod l) -> c mod M l */
static void
crt_step(nn *c, nn *M, u64 r, u64 l)
{
	/* c + M k = r mod l: k = (r - c) M^-1 mod l */
	u64 cm = nn_mod_u64(c, l), mm = nn_mod_u64(M, l), g;
	u64 inv = nt_inv(mm, l, &g);
	u64 k = nt_mulmod((r + l - cm) % l, inv, l);
	nn t;

	nn_mul_u64(&t, M, k);
	nn_add(c, c, &t);
	nn_mul_u64(M, M, l);
}

int
ec_count_schoof(const struct ecp_curve *E, nn *N, int verbose)
{
	const struct mf *F = E->K;
	struct px *f;
	nn M, c, bound, w2, lo, hi, p1;
	int l, lmax, i, tl;
	static const int primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
		41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
		107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 0};
	int need[64], nl = 0;
	u64 t0;

	memset(&ecsch_last, 0, sizeof ecsch_last);
	ec_hasse(&F->P, &lo, &hi, &w2);
	/* candidates: 4 sqrt(p) + 1 values of t; stop when M exceeds
	 * 4 sqrt(p) / 2^38 and let a BSGS match finish */
	nn_shl(&bound, &w2, 1);
	nn_shr(&bound, &bound, 38);
	nn_set(&M, 2);
	nn_set(&c, (u64)schoof_2(E));     /* c = t mod 2 */
	ecsch_last.l[0] = 2;
	ecsch_last.t[0] = (int)c.w[0];
	ecsch_last.kind[0] = '2';
	ecsch_last.n = 1;
	{
		nn m = M;
		for (i = 0; primes[i] && nn_cmp(&m, &bound) <= 0; i++) {
			if (nn_mod_u64(&F->P, (u64)primes[i]) == 0)
				continue;
			need[nl++] = primes[i];
			nn_mul_u64(&m, &m, (u64)primes[i]);
		}
		if (nn_cmp(&m, &bound) <= 0)
			fail("schoof: p too large for the prime table");
	}
	lmax = nl ? need[nl - 1] : 3;
	f = xmalloc(sizeof(struct px) * (size_t)(lmax + 3));
	ec_divpolys(E, f, lmax + 2);
	for (i = 0; i < nl; i++) {
		int deg = 0;
		l = need[i];
		t0 = ecff_ms();
		tl = schoof_l(E, l, &f[l], &deg);
		if (tl < 0) {
			for (int k = 0; k <= lmax + 2; k++)
				px_free(&f[k]);
			free(f);
			return 0;
		}
		crt_step(&c, &M, (u64)tl, (u64)l);
		ecsch_last.l[ecsch_last.n] = l;
		ecsch_last.t[ecsch_last.n] = tl;
		ecsch_last.kind[ecsch_last.n] = 's';
		ecsch_last.deg[ecsch_last.n] = deg;
		ecsch_last.ms[ecsch_last.n] = ecff_ms() - t0;
		ecsch_last.n++;
		if (verbose)
			printf("  l = %d: t = %d mod %d (degree %d, %llu ms)\n", l,
			    tl, l, deg, (unsigned long long)(ecff_ms() - t0));
	}
	for (i = 0; i <= lmax + 2; i++)
		px_free(&f[i]);
	free(f);
	ecsch_last.M = M;
	/* N = p + 1 - t = p + 1 - c mod M */
	nn_add_u64(&p1, &F->P, 1);
	{
		nn cm, Nc;
		nn_mod(&cm, &p1, &M);
		if (nn_cmp(&cm, &c) >= 0)
			nn_sub(&Nc, &cm, &c);
		else {
			nn_add(&Nc, &cm, &M);
			nn_sub(&Nc, &Nc, &c);
		}
		ecsch_last.bsgs = 1;
		return ec_match_bsgs(E, &Nc, &M, N);
	}
}

int ec_count_sea_impl(const struct ecp_curve *E, nn *N, int verbose);

int
ec_count_sea(const struct ecp_curve *E, nn *N, int verbose)
{
	return ec_count_schoof(E, N, verbose);
}
