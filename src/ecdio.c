/*
 * ecdio.c - Diophantine equations through elliptic curves.
 *
 *   parse     a rational equation in up to QP_MAXV variables becomes a
 *             polynomial F = 0 (denominators cleared, their spurious
 *             factors divided out again, and remembered: a solution must
 *             not make one vanish)
 *   classify  a ternary cubic form (projective plane cubic), an affine
 *             cubic in two variables (homogenised), y-quadratic curves
 *             s^2 = quartic(x), weighted forms z^2 = quartic(x, y)
 *   points    a small rational point by search
 *   maps      flex -> Weierstrass (ec.c), or the pencil of lines through
 *             the point -> s^2 = quartic(t) -> Weierstrass; every step is
 *             data with a forward and a backward map, and the composition
 *             is checked on every point carried across
 *   group     ecq_mw_compute: minimal model, torsion, 2-isogeny descent,
 *             generators
 *   search    combinations of the generators and the torsion, mapped back
 *             to the equation, tested against the side conditions, and
 *             verified by exact substitution into the original equation
 */
#include "ecq.h"
#include "ucas.h"
#include "report.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define I(v) z_from_i64(v)

/* ================================================================== */
/* polynomials                                                         */
/* ================================================================== */

void
qp_init(struct qpoly *f)
{
	memset(f, 0, sizeof *f);
}

void
qp_free(struct qpoly *f)
{
	free(f->e);
	free(f->c);
	memset(f, 0, sizeof *f);
}

static void
qp_reserve(struct qpoly *f, int n)
{
	if (n <= f->cap)
		return;
	f->cap = n + 8;
	f->e = xrealloc(f->e, (size_t)f->cap * QP_MAXV);
	f->c = xrealloc(f->c, (size_t)f->cap * sizeof *f->c);
}

void
qp_copy(struct qpoly *d, const struct qpoly *s)
{
	struct qpoly t;

	qp_init(&t);
	qp_reserve(&t, s->n);
	if (s->n) {
		memcpy(t.e, s->e, (size_t)s->n * QP_MAXV);
		memcpy(t.c, s->c, (size_t)s->n * sizeof *s->c);
	}
	t.n = s->n;
	qp_free(d);
	*d = t;
}

/* add c * x^e into f (combining) */
static void
qp_addterm(struct qpoly *f, const uint8_t *e, struct zq c)
{
	int i;

	if (zq_is_zero(c))
		return;
	for (i = 0; i < f->n; i++)
		if (!memcmp(f->e + i * QP_MAXV, e, QP_MAXV)) {
			f->c[i] = zq_add(f->c[i], c);
			if (zq_is_zero(f->c[i])) {
				f->n--;
				memmove(f->e + i * QP_MAXV, f->e + f->n * QP_MAXV,
				    QP_MAXV);
				f->c[i] = f->c[f->n];
			}
			return;
		}
	qp_reserve(f, f->n + 1);
	memcpy(f->e + f->n * QP_MAXV, e, QP_MAXV);
	f->c[f->n++] = c;
}

void
qp_const(struct qpoly *f, struct zq c)
{
	uint8_t e[QP_MAXV] = {0};

	f->n = 0;
	qp_addterm(f, e, c);
}

void
qp_var(struct qpoly *f, int v)
{
	uint8_t e[QP_MAXV] = {0};

	f->n = 0;
	e[v] = 1;
	qp_addterm(f, e, zq_int(Z_ONE));
}

void
qp_add(struct qpoly *r, const struct qpoly *a, const struct qpoly *b)
{
	struct qpoly t;
	int i;

	qp_init(&t);
	for (i = 0; i < a->n; i++)
		qp_addterm(&t, a->e + i * QP_MAXV, a->c[i]);
	for (i = 0; i < b->n; i++)
		qp_addterm(&t, b->e + i * QP_MAXV, b->c[i]);
	qp_free(r);
	*r = t;
}

void
qp_sub(struct qpoly *r, const struct qpoly *a, const struct qpoly *b)
{
	struct qpoly t;
	int i;

	qp_init(&t);
	for (i = 0; i < a->n; i++)
		qp_addterm(&t, a->e + i * QP_MAXV, a->c[i]);
	for (i = 0; i < b->n; i++)
		qp_addterm(&t, b->e + i * QP_MAXV, zq_neg(b->c[i]));
	qp_free(r);
	*r = t;
}

void
qp_mul(struct qpoly *r, const struct qpoly *a, const struct qpoly *b)
{
	struct qpoly t;
	int i, j, k;

	qp_init(&t);
	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++) {
			uint8_t e[QP_MAXV];
			for (k = 0; k < QP_MAXV; k++) {
				int s = a->e[i * QP_MAXV + k] + b->e[j * QP_MAXV + k];
				if (s > 250)
					fail("polynomial degree too high");
				e[k] = (uint8_t)s;
			}
			qp_addterm(&t, e, zq_mul(a->c[i], b->c[j]));
		}
	qp_free(r);
	*r = t;
}

void
qp_scale(struct qpoly *r, const struct qpoly *a, struct zq c)
{
	struct qpoly t;
	int i;

	qp_init(&t);
	for (i = 0; i < a->n; i++)
		qp_addterm(&t, a->e + i * QP_MAXV, zq_mul(a->c[i], c));
	qp_free(r);
	*r = t;
}

int
qp_is_zero(const struct qpoly *f)
{
	return f->n == 0;
}

static int
tdeg(const uint8_t *e)
{
	int k, s = 0;

	for (k = 0; k < QP_MAXV; k++)
		s += e[k];
	return s;
}

int
qp_deg(const struct qpoly *f)
{
	int i, d = -1;

	for (i = 0; i < f->n; i++)
		if (tdeg(f->e + i * QP_MAXV) > d)
			d = tdeg(f->e + i * QP_MAXV);
	return d;
}

int
qp_deg_in(const struct qpoly *f, int v)
{
	int i, d = -1;

	for (i = 0; i < f->n; i++)
		if (f->e[i * QP_MAXV + v] > d)
			d = f->e[i * QP_MAXV + v];
	return d;
}

int
qp_homogeneous(const struct qpoly *f)
{
	int i, d = qp_deg(f);

	for (i = 0; i < f->n; i++)
		if (tdeg(f->e + i * QP_MAXV) != d)
			return 0;
	return 1;
}

struct zq
qp_eval(const struct qpoly *f, const struct zq *x)
{
	struct zq s = zq_int(0);
	int i, k, j;

	for (i = 0; i < f->n; i++) {
		struct zq t = f->c[i];
		for (k = 0; k < QP_MAXV; k++)
			for (j = 0; j < f->e[i * QP_MAXV + k]; j++)
				t = zq_mul(t, x[k]);
		s = zq_add(s, t);
	}
	return s;
}

/* the leading term in lex order (index), -1 for 0 */
static int
qp_lead(const struct qpoly *f)
{
	int i, best = -1;

	for (i = 0; i < f->n; i++)
		if (best < 0 || memcmp(f->e + i * QP_MAXV, f->e + best * QP_MAXV,
		    QP_MAXV) > 0)
			best = i;
	return best;
}

int
qp_divexact(struct qpoly *q, const struct qpoly *a, const struct qpoly *b)
{
	struct qpoly r, t, m, Q;
	int lb = qp_lead(b), guard = 0;

	if (lb < 0)
		fail("division by the zero polynomial");
	qp_init(&r);
	qp_init(&t);
	qp_init(&m);
	qp_init(&Q);
	qp_copy(&r, a);
	while (r.n && guard++ < 10000) {
		int lr = qp_lead(&r), k;
		uint8_t e[QP_MAXV];
		for (k = 0; k < QP_MAXV; k++) {
			int d = r.e[lr * QP_MAXV + k] - b->e[lb * QP_MAXV + k];
			if (d < 0) {
				qp_free(&r);
				qp_free(&t);
				qp_free(&m);
				qp_free(&Q);
				return 0;
			}
			e[k] = (uint8_t)d;
		}
		m.n = 0;
		qp_addterm(&m, e, zq_div(r.c[lr], b->c[lb]));
		qp_addterm(&Q, e, zq_div(r.c[lr], b->c[lb]));
		qp_mul(&t, &m, b);
		qp_sub(&r, &r, &t);
	}
	qp_free(&r);
	qp_free(&t);
	qp_free(&m);
	qp_free(q);
	*q = Q;
	return 1;
}

void
qp_print(struct sbuf *s, const struct qpoly *f, const struct qvars *V)
{
	int i, k, first = 1, *ord;

	if (!f->n) {
		sb_puts(s, "0");
		return;
	}
	/* by degree, then lex */
	ord = xmalloc(sizeof(int) * (size_t)f->n);
	for (i = 0; i < f->n; i++)
		ord[i] = i;
	for (i = 1; i < f->n; i++)
		for (k = i; k > 0; k--) {
			const uint8_t *x = f->e + ord[k - 1] * QP_MAXV,
			    *y = f->e + ord[k] * QP_MAXV;
			if (tdeg(x) > tdeg(y) || (tdeg(x) == tdeg(y) &&
			    memcmp(x, y, QP_MAXV) >= 0))
				break;
			{
				int t = ord[k];
				ord[k] = ord[k - 1];
				ord[k - 1] = t;
			}
		}
	for (i = 0; i < f->n; i++) {
		int j = ord[i], neg, mono = 0;
		struct zq c = f->c[j];
		neg = z_sign(c.n) < 0;
		if (neg)
			c = zq_neg(c);
		sb_puts(s, first ? (neg ? "-" : "") : (neg ? " - " : " + "));
		first = 0;
		for (k = 0; k < V->nv; k++)
			if (f->e[j * QP_MAXV + k])
				mono = 1;
		if (!mono || !z_is_one(c.n) || !z_is_one(c.d)) {
			zq_print(s, c);
			if (mono)
				sb_putc(s, '*');
		}
		{
			int f1 = 1;
			for (k = 0; k < V->nv; k++) {
				int ex = f->e[j * QP_MAXV + k];
				if (!ex)
					continue;
				if (!f1)
					sb_putc(s, '*');
				f1 = 0;
				sb_puts(s, V->name[k]);
				if (ex > 1)
					sb_printf(s, "^%d", ex);
			}
		}
	}
	free(ord);
}

/* ================================================================== */
/* the equation parser                                                 */
/* ================================================================== */

struct ps {
	const char *s;
	struct dio_eq *D;
	const char *err;
	jmp_buf jb;
};

static void
qr_init(struct qrat *r)
{
	qp_init(&r->n);
	qp_init(&r->d);
}

static void
qr_free(struct qrat *r)
{
	qp_free(&r->n);
	qp_free(&r->d);
}

static void
perr(struct ps *P, const char *m)
{
	P->err = m;
	longjmp(P->jb, 1);
}

static void
sp(struct ps *P)
{
	while (*P->s == ' ' || *P->s == '\t')
		P->s++;
}

static void pexpr(struct ps *P, struct qrat *r);

static void
note_den(struct ps *P, const struct qpoly *d)
{
	struct dio_eq *D = P->D;
	int i;

	if (qp_deg(d) <= 0)
		return;
	for (i = 0; i < D->nden; i++) {
		struct qpoly q;
		int same;
		qp_init(&q);
		qp_sub(&q, &D->den[i], d);
		same = qp_is_zero(&q);
		qp_free(&q);
		if (same)
			return;
	}
	if (D->nden < 16) {
		qp_init(&D->den[D->nden]);
		qp_copy(&D->den[D->nden++], d);
	}
}

static void
patom(struct ps *P, struct qrat *r)
{
	sp(P);
	if (*P->s == '(') {
		P->s++;
		pexpr(P, r);
		sp(P);
		if (*P->s != ')')
			perr(P, "expected )");
		P->s++;
		return;
	}
	if (isdigit((unsigned char)*P->s)) {
		zint q;
		const char *e;
		if (!z_parse(P->s, &q, &e))
			perr(P, "a number");
		P->s = e;
		qp_const(&r->n, zq_int(q));
		qp_const(&r->d, zq_int(Z_ONE));
		return;
	}
	if (isalpha((unsigned char)*P->s) || *P->s == '_') {
		char nm[16];
		int k = 0, v;
		struct qvars *V = &P->D->V;
		while ((isalnum((unsigned char)*P->s) || *P->s == '_') && k < 15)
			nm[k++] = *P->s++;
		nm[k] = 0;
		for (v = 0; v < V->nv && strcmp(V->name[v], nm); v++)
			;
		if (v == V->nv) {
			if (V->nv == QP_MAXV)
				perr(P, "too many variables");
			strcpy(V->name[V->nv++], nm);
		}
		qp_var(&r->n, v);
		qp_const(&r->d, zq_int(Z_ONE));
		return;
	}
	perr(P, "a number, a variable or (");
}

static void
ppow(struct ps *P, struct qrat *r)
{
	patom(P, r);
	sp(P);
	if (*P->s == '^') {
		long e;
		char *end;
		struct qrat t;
		P->s++;
		sp(P);
		e = strtol(P->s, &end, 10);
		if (end == P->s || e < 0 || e > 64)
			perr(P, "a small exponent");
		P->s = end;
		qr_init(&t);
		qp_const(&t.n, zq_int(Z_ONE));
		qp_const(&t.d, zq_int(Z_ONE));
		while (e-- > 0) {
			qp_mul(&t.n, &t.n, &r->n);
			qp_mul(&t.d, &t.d, &r->d);
		}
		qr_free(r);
		*r = t;
	}
}

static void
punary(struct ps *P, struct qrat *r)
{
	sp(P);
	if (*P->s == '-') {
		P->s++;
		punary(P, r);
		qp_scale(&r->n, &r->n, zq_int(I(-1)));
		return;
	}
	if (*P->s == '+') {
		P->s++;
		punary(P, r);
		return;
	}
	ppow(P, r);
}

static void
pterm(struct ps *P, struct qrat *r)
{
	punary(P, r);
	for (;;) {
		struct qrat t;
		int div;
		sp(P);
		if (*P->s == '*' || *P->s == '/') {
			div = *P->s == '/';
			P->s++;
		} else if (*P->s == '(' || isalpha((unsigned char)*P->s))
			div = 0;   /* juxtaposition */
		else
			break;
		qr_init(&t);
		punary(P, &t);
		if (div) {
			if (qp_is_zero(&t.n))
				perr(P, "division by zero");
			note_den(P, &t.n);
			qp_mul(&r->n, &r->n, &t.d);
			qp_mul(&r->d, &r->d, &t.n);
		} else {
			qp_mul(&r->n, &r->n, &t.n);
			qp_mul(&r->d, &r->d, &t.d);
		}
		qr_free(&t);
	}
}

static void
pexpr(struct ps *P, struct qrat *r)
{
	pterm(P, r);
	for (;;) {
		struct qrat t;
		struct qpoly a, b;
		int sub;
		sp(P);
		if (*P->s != '+' && *P->s != '-')
			break;
		sub = *P->s == '-';
		P->s++;
		qr_init(&t);
		pterm(P, &t);
		qp_init(&a);
		qp_init(&b);
		/* n/d +- tn/td: the same denominator stays */
		{
			struct qpoly diff;
			int same;
			qp_init(&diff);
			qp_sub(&diff, &r->d, &t.d);
			same = qp_is_zero(&diff);
			qp_free(&diff);
			if (same) {
				if (sub)
					qp_sub(&r->n, &r->n, &t.n);
				else
					qp_add(&r->n, &r->n, &t.n);
			} else {
				qp_mul(&a, &r->n, &t.d);
				qp_mul(&b, &t.n, &r->d);
				if (sub)
					qp_sub(&r->n, &a, &b);
				else
					qp_add(&r->n, &a, &b);
				qp_mul(&r->d, &r->d, &t.d);
			}
		}
		qp_free(&a);
		qp_free(&b);
		qr_free(&t);
	}
}

/* clear the denominators of a polynomial's rational coefficients and
 * divide by the content: an integral primitive polynomial, leading
 * coefficient (lex) positive */
static void
qp_primitive(struct qpoly *f)
{
	zint L = Z_ONE, g = 0;
	int i, lead;

	for (i = 0; i < f->n; i++)
		L = z_lcm(L, f->c[i].d);
	for (i = 0; i < f->n; i++) {
		f->c[i] = zq_mul(f->c[i], zq_int(L));
		g = z_gcd(g, f->c[i].n);
	}
	lead = qp_lead(f);
	if (lead >= 0 && z_sign(f->c[lead].n) < 0)
		g = z_neg(g);
	if (!z_is_zero(g))
		for (i = 0; i < f->n; i++)
			f->c[i] = zq_make(z_divexact(f->c[i].n, g), Z_ONE);
}

int
dio_parse(const char *s, struct dio_eq *D, const char **why)
{
	struct ps P;
	struct qpoly a, b;
	int i;

	memset(D, 0, sizeof *D);
	qr_init(&D->lhs);
	qr_init(&D->rhs);
	qp_init(&D->F);
	P.s = s;
	P.D = D;
	P.err = NULL;
	if (setjmp(P.jb)) {
		if (why)
			*why = P.err;
		return 0;
	}
	pexpr(&P, &D->lhs);
	sp(&P);
	if (*P.s == '=') {
		P.s++;
		pexpr(&P, &D->rhs);
	} else {
		qp_const(&D->rhs.n, zq_int(0));
		qp_const(&D->rhs.d, zq_int(Z_ONE));
	}
	sp(&P);
	if (*P.s)
		perr(&P, "unexpected text after the equation");
	qp_init(&a);
	qp_init(&b);
	qp_mul(&a, &D->lhs.n, &D->rhs.d);
	qp_mul(&b, &D->rhs.n, &D->lhs.d);
	qp_sub(&D->F, &a, &b);
	qp_free(&a);
	qp_free(&b);
	/* divide out factors that came from denominators */
	for (i = 0; i < D->nden; i++) {
		struct qpoly q;
		qp_init(&q);
		while (qp_deg(&D->den[i]) > 0 && qp_divexact(&q, &D->F,
		    &D->den[i]))
			qp_copy(&D->F, &q);
		qp_free(&q);
	}
	qp_primitive(&D->F);
	if (qp_is_zero(&D->F)) {
		if (why)
			*why = "the equation holds identically";
		return 0;
	}
	return 1;
}

void
dio_free(struct dio_eq *D)
{
	int i;

	qr_free(&D->lhs);
	qr_free(&D->rhs);
	qp_free(&D->F);
	for (i = 0; i < D->nden; i++)
		qp_free(&D->den[i]);
	D->nden = 0;
}

/* ================================================================== */
/* quartic -> Weierstrass (Washington, Thm 2.17, and the root case)    */
/* ================================================================== */

/* w^2 = a t'^4 + b t'^3 + c t'^2 + d t' + e, t' = t - t0 */
struct q2w {
	int kind;          /* 1: e = q^2, q != 0; 2: e = 0 (t0 a root) */
	struct zq t0, a, b, c, d, e, q;
};

static struct zq
Zq(int64_t v)
{
	return zq_int(I(v));
}

#define QADD zq_add
#define QSUB zq_sub
#define QMUL zq_mul
#define QDIV zq_div

/* the Weierstrass coefficients */
static void
q2w_curve(const struct q2w *m, struct ec *W)
{
	struct ecf F;
	fe a[5];

	ecf_q(&F);
	if (m->kind == 1) {
		struct zq q = m->q, q2 = QMUL(q, q), a2, a4;
		a2 = QSUB(m->c, QDIV(QMUL(m->d, m->d), QMUL(Zq(4), q2)));
		a4 = QMUL(Zq(-4), QMUL(q2, m->a));
		a[0] = fe_from_q(QDIV(m->d, q));
		a[1] = fe_from_q(a2);
		a[2] = fe_from_q(QMUL(Zq(2), QMUL(q, m->b)));
		a[3] = fe_from_q(a4);
		a[4] = fe_from_q(QMUL(a2, a4));
	} else {
		/* Y^2 = X^3 + c X^2 + b d X + a d^2 */
		a[0] = a[2] = fe_from_q(Zq(0));
		a[1] = fe_from_q(m->c);
		a[3] = fe_from_q(QMUL(m->b, m->d));
		a[4] = fe_from_q(QMUL(m->a, QMUL(m->d, m->d)));
	}
	{
		static struct ecf QF;
		static int init;
		if (!init) {
			ecf_q(&QF);
			init = 1;
		}
		ec_init(W, &QF, a);
	}
}

/* (t, w) -> W; 0 when the point is exceptional for the formula */
static int
q2w_fwd(const struct q2w *m, struct zq t, struct zq w, struct ecpt *R)
{
	struct zq u = QSUB(t, m->t0);

	if (m->kind == 1) {
		struct zq q = m->q, q2 = QMUL(q, q), X, Y;
		if (zq_is_zero(u)) {
			if (zq_eq(w, q)) {
				ec_O(R);
				return 1;
			}
			return 0;
		}
		X = QDIV(QADD(QMUL(QMUL(Zq(2), q), QADD(w, q)), QMUL(m->d, u)),
		    QMUL(u, u));
		Y = QADD(QMUL(QMUL(Zq(4), q2), QADD(w, q)), QMUL(QMUL(Zq(2), q),
		    QADD(QMUL(m->d, u), QMUL(m->c, QMUL(u, u)))));
		Y = QSUB(Y, QDIV(QMUL(QMUL(m->d, m->d), QMUL(u, u)), QMUL(Zq(2),
		    q)));
		Y = QDIV(Y, QMUL(u, QMUL(u, u)));
		R->x = fe_from_q(X);
		R->y = fe_from_q(Y);
		R->inf = 0;
		return 1;
	}
	if (zq_is_zero(u)) {
		ec_O(R);
		return 1;
	}
	{
		/* U = 1/u, W' = w U^2, X = d U, Y = d W' */
		struct zq U = zq_inv(u), Wp = QMUL(w, QMUL(U, U));
		R->x = fe_from_q(QMUL(m->d, U));
		R->y = fe_from_q(QMUL(m->d, Wp));
		R->inf = 0;
		return 1;
	}
}

static int
q2w_back(const struct q2w *m, const struct ecpt *P, struct zq *t,
    struct zq *w)
{
	struct zq X, Y, u;

	if (P->inf) {
		*t = m->t0;
		*w = m->kind == 1 ? m->q : Zq(0);
		return 1;
	}
	X = fe_q(&P->x);
	Y = fe_q(&P->y);
	if (m->kind == 1) {
		struct zq q = m->q;
		if (zq_is_zero(Y))
			return 0;
		u = QDIV(QSUB(QMUL(QMUL(Zq(2), q), QADD(X, m->c)),
		    QDIV(QMUL(m->d, m->d), QMUL(Zq(2), q))), Y);
		*w = QADD(zq_neg(q), QDIV(QMUL(u, QSUB(QMUL(u, X), m->d)),
		    QMUL(Zq(2), q)));
		*t = QADD(u, m->t0);
		return 1;
	}
	if (zq_is_zero(X))
		return 0;
	u = QDIV(m->d, X);
	*w = QDIV(QMUL(Y, m->d), QMUL(X, X));
	*t = QADD(u, m->t0);
	return 1;
}

/* the quartic D(t) (coefficients d[0..4] of t^0..t^4) with the point
 * (t0, w0) -> the map */
static int
q2w_make(struct q2w *m, const struct zq *D, struct zq t0, struct zq w0)
{
	struct zq c[5];
	int k, j;
	static const int bin[5][5] = {{1}, {1, 1}, {1, 2, 1}, {1, 3, 3, 1},
	    {1, 4, 6, 4, 1}};

	/* Taylor at t0 */
	for (k = 0; k <= 4; k++) {
		struct zq s = Zq(0);
		for (j = 4; j >= k; j--) {
			struct zq p = Zq(1);
			int i;
			for (i = 0; i < j - k; i++)
				p = QMUL(p, t0);
			s = QADD(s, QMUL(QMUL(Zq(bin[j][k]), D[j]), p));
		}
		c[k] = s;
	}
	if (!zq_eq(c[0], QMUL(w0, w0)))
		fail("internal: the quartic's point");
	memset(m, 0, sizeof *m);
	m->t0 = t0;
	m->a = c[4];
	m->b = c[3];
	m->c = c[2];
	m->d = c[1];
	m->e = c[0];
	if (!zq_is_zero(w0)) {
		m->kind = 1;
		m->q = w0;
		return 1;
	}
	if (zq_is_zero(c[1]))
		return 0;    /* a double root: singular */
	m->kind = 2;
	return 1;
}

/* ================================================================== */
/* the maps from the equation's curve to Weierstrass form              */
/* ================================================================== */

enum { DK_FLEX = 1, DK_LINES, DK_YQUAD };

struct dio_map {
	int kind;
	int proj;            /* the variables are projective coordinates */
	int affine;          /* homogenised: the third coordinate is 1 */
	int vmap[3];         /* which variable is coordinate 0, 1, 2 */
	fe cub[10];
	fe P0[3];
	struct ec_flexmap fm;
	fe A[9], Ainv[9];
	struct zq F1[2], F2[3], F3[4];     /* high degree first */
	struct zq D[5];                    /* low first */
	struct q2w qw;
	/* DK_YQUAD: F = PA(x) y^2 + PB(x) y + PC(x) (low first) */
	int vx, vy, vw, weighted;
	struct zq PA[5], PB[5], PC[5];
	struct ec W;
	struct ec Emin;
	struct eciso tomin;
};

static struct ecf QFd;

static const struct ecf *
QQd(void)
{
	static int init;

	if (!init) {
		ecf_q(&QFd);
		init = 1;
	}
	return &QFd;
}

static struct zq
evalh(const struct zq *c, int n, struct zq t)
{
	/* c[0] t^(n-1) + ... + c[n-1] */
	struct zq s = Zq(0);
	int i;

	for (i = 0; i < n; i++)
		s = QADD(QMUL(s, t), c[i]);
	return s;
}

static struct zq
evall(const struct zq *c, int n, struct zq t)
{
	struct zq s = Zq(0);
	int i;

	for (i = n - 1; i >= 0; i--)
		s = QADD(QMUL(s, t), c[i]);
	return s;
}

static int
mono_idx(int a, int b, int c)
{
	int i;

	for (i = 0; i < 10; i++)
		if (ec_cub_mono[i][0] == a && ec_cub_mono[i][1] == b &&
		    ec_cub_mono[i][2] == c)
			return i;
	return -1;
}

static struct zq
feq(const fe *a)
{
	return fe_q(a);
}

/* the lines through P0: the forms and the quartic */
static int
lines_setup(struct dio_map *m)
{
	fe G[10];
	int swap, k, i;

	for (swap = 0; swap < 2; swap++) {
		/* Ainv's columns: two unit vectors completing P0, then P0 */
		int c0, c1, ok = 0;
		for (c0 = 0; c0 < 3 && !ok; c0++)
			for (c1 = 0; c1 < 3 && !ok; c1++) {
				fe det, t;
				if (c0 == c1)
					continue;
				for (k = 0; k < 9; k++)
					fe_zero(QQd(), &m->Ainv[k]);
				for (k = 0; k < 3; k++) {
					fe_set_i64(QQd(), &m->Ainv[k * 3 + (swap ? 1 : 0)],
					    k == c0);
					fe_set_i64(QQd(), &m->Ainv[k * 3 + (swap ? 0 : 1)],
					    k == c1);
					m->Ainv[k * 3 + 2] = m->P0[k];
				}
				/* det != 0: P0's coordinate outside {c0, c1} */
				t = m->P0[3 - c0 - c1];
				det = t;
				ok = !fe_is_zero(QQd(), &det);
			}
		if (!ok)
			return 0;
		/* A = Ainv^-1 by the adjugate */
		{
			struct zq M[9], det = Zq(0);
			int r, c;
			for (k = 0; k < 9; k++)
				M[k] = feq(&m->Ainv[k]);
			for (r = 0; r < 3; r++)
				for (c = 0; c < 3; c++) {
					int r0 = (c + 1) % 3, r1 = (c + 2) % 3;
					int c0b = (r + 1) % 3, c1b = (r + 2) % 3;
					struct zq v = QSUB(QMUL(M[r0 * 3 + c0b],
					    M[r1 * 3 + c1b]), QMUL(M[r0 * 3 + c1b],
					    M[r1 * 3 + c0b]));
					m->A[r * 3 + c] = fe_from_q(v);
				}
			for (c = 0; c < 3; c++)
				det = QADD(det, QMUL(M[c], feq(&m->A[c * 3])));
			if (zq_is_zero(det))
				return 0;
			for (k = 0; k < 9; k++)
				m->A[k] = fe_from_q(QDIV(feq(&m->A[k]), det));
		}
		ec_cub_transform(QQd(), m->cub, m->Ainv, G);
		if (!fe_is_zero(QQd(), &G[mono_idx(0, 0, 3)]))
			fail("internal: the base point is not at (0 : 0 : 1)");
		m->F1[0] = feq(&G[mono_idx(1, 0, 2)]);
		m->F1[1] = feq(&G[mono_idx(0, 1, 2)]);
		m->F2[0] = feq(&G[mono_idx(2, 0, 1)]);
		m->F2[1] = feq(&G[mono_idx(1, 1, 1)]);
		m->F2[2] = feq(&G[mono_idx(0, 2, 1)]);
		m->F3[0] = feq(&G[mono_idx(3, 0, 0)]);
		m->F3[1] = feq(&G[mono_idx(2, 1, 0)]);
		m->F3[2] = feq(&G[mono_idx(1, 2, 0)]);
		m->F3[3] = feq(&G[mono_idx(0, 3, 0)]);
		if (zq_is_zero(m->F1[0]) && zq_is_zero(m->F1[1]))
			return 0;    /* singular point */
		if (!zq_is_zero(m->F1[0]))
			break;      /* the tangent direction has v0 != 0 */
	}
	if (zq_is_zero(m->F1[0]))
		return 0;
	/* D(t) = F2(t,1)^2 - 4 F1(t,1) F3(t,1), low first */
	{
		struct zq f1[2], f2[3], f3[4];
		for (i = 0; i < 2; i++)
			f1[i] = m->F1[1 - i];
		for (i = 0; i < 3; i++)
			f2[i] = m->F2[2 - i];
		for (i = 0; i < 4; i++)
			f3[i] = m->F3[3 - i];
		for (i = 0; i < 5; i++)
			m->D[i] = Zq(0);
		for (i = 0; i < 3; i++)
			for (k = 0; k < 3; k++)
				m->D[i + k] = QADD(m->D[i + k], QMUL(f2[i], f2[k]));
		for (i = 0; i < 2; i++)
			for (k = 0; k < 4; k++)
				if (i + k <= 4)
					m->D[i + k] = QSUB(m->D[i + k], QMUL(Zq(4),
					    QMUL(f1[i], f3[k])));
	}
	{
		/* the tangent direction t0 = -F1[1]/F1[0], w0 = F2(t0, 1) */
		struct zq t0 = zq_neg(QDIV(m->F1[1], m->F1[0]));
		struct zq w0 = evalh(m->F2, 3, t0);
		if (!q2w_make(&m->qw, m->D, t0, w0))
			return 0;
	}
	return 1;
}

/* C point (projective X[3], cubic coordinates) -> W; 0: exceptional */
static int
map_fwd(const struct dio_map *m, const fe *X, struct ecpt *R)
{
	if (m->kind == DK_FLEX)
		return ec_flexmap_fwd(&m->W, &m->fm, X, R);
	if (m->kind == DK_LINES) {
		fe U[3];
		int i, j;
		struct zq t, lam, w;
		for (i = 0; i < 3; i++) {
			fe s, p;
			fe_zero(QQd(), &s);
			for (j = 0; j < 3; j++) {
				fe_mul(QQd(), &p, &m->A[i * 3 + j], &X[j]);
				fe_add(QQd(), &s, &s, &p);
			}
			U[i] = s;
		}
		if (fe_is_zero(QQd(), &U[1]) || fe_is_zero(QQd(), &U[2]))
			return 0;
		if (fe_is_zero(QQd(), &U[0]) && fe_is_zero(QQd(), &U[1]))
			return 0;
		t = QDIV(feq(&U[0]), feq(&U[1]));
		lam = QDIV(feq(&U[1]), feq(&U[2]));
		w = QADD(QMUL(QMUL(Zq(2), evalh(m->F3, 4, t)), lam),
		    evalh(m->F2, 3, t));
		return q2w_fwd(&m->qw, t, w, R);
	}
	{
		/* DK_YQUAD: X = (x, y) affine (X[2] = 1) */
		struct zq x = QDIV(feq(&X[0]), feq(&X[2]));
		struct zq y = QDIV(feq(&X[1]), feq(&X[2]));
		struct zq s = QADD(QMUL(QMUL(Zq(2), evall(m->PA, 5, x)), y),
		    evall(m->PB, 5, x));
		return q2w_fwd(&m->qw, x, s, R);
	}
}

/* W point -> C point (projective, cubic coordinates); 0: exceptional */
static int
map_back(const struct dio_map *m, const struct ecpt *P, fe *X)
{
	struct zq t, w;

	if (m->kind == DK_FLEX) {
		ec_flexmap_back(&m->W, &m->fm, P, X);
		return 1;
	}
	if (!q2w_back(&m->qw, P, &t, &w))
		return 0;
	if (m->kind == DK_LINES) {
		struct zq f1 = evalh(m->F1, 2, t), f2 = evalh(m->F2, 3, t),
		    f3 = evalh(m->F3, 4, t), lam, U[3];
		int i, j;
		if (!zq_is_zero(f3))
			lam = QDIV(QSUB(w, f2), QMUL(Zq(2), f3));
		else if (!zq_is_zero(QADD(w, f2)))
			lam = QDIV(QMUL(Zq(-2), f1), QADD(w, f2));
		else
			return 0;
		if (zq_is_zero(lam))
			return 0;
		U[0] = QMUL(lam, t);
		U[1] = lam;
		U[2] = Zq(1);
		for (i = 0; i < 3; i++) {
			struct zq s = Zq(0);
			for (j = 0; j < 3; j++)
				s = QADD(s, QMUL(feq(&m->Ainv[i * 3 + j]), U[j]));
			X[i] = fe_from_q(s);
		}
		return 1;
	}
	{
		struct zq a = evall(m->PA, 5, t), b = evall(m->PB, 5, t);
		if (zq_is_zero(a))
			return 0;
		X[0] = fe_from_q(t);
		X[1] = fe_from_q(QDIV(QSUB(w, b), QMUL(Zq(2), a)));
		X[2] = fe_from_q(Zq(1));
		return 1;
	}
}

/* floating-point x on the minimal model of a real point of C (for the
 * real components) */
static int
map_fwd_real(const struct dio_map *m, const double *X, double *xmin)
{
	double x, u2, r;

	if (m->kind == DK_FLEX) {
		double U[3];
		int i, j;
		for (i = 0; i < 3; i++) {
			U[i] = 0;
			for (j = 0; j < 3; j++)
				U[i] += zq_to_double(feq(&m->fm.M[i * 3 + j])) * X[j];
		}
		if (U[2] == 0)
			return 0;
		x = U[0] / (U[2] * zq_to_double(feq(&m->fm.lam)));
	} else if (m->kind == DK_LINES) {
		double U[3], t, lam, w, tp, q, d;
		int i, j;
		for (i = 0; i < 3; i++) {
			U[i] = 0;
			for (j = 0; j < 3; j++)
				U[i] += zq_to_double(feq(&m->A[i * 3 + j])) * X[j];
		}
		if (U[1] == 0 || U[2] == 0)
			return 0;
		t = U[0] / U[1];
		lam = U[1] / U[2];
		{
			double f2 = 0, f3 = 0;
			for (i = 0; i < 3; i++)
				f2 = f2 * t + zq_to_double(m->F2[i]);
			for (i = 0; i < 4; i++)
				f3 = f3 * t + zq_to_double(m->F3[i]);
			w = 2 * f3 * lam + f2;
		}
		tp = t - zq_to_double(m->qw.t0);
		if (tp == 0)
			return 0;
		d = zq_to_double(m->qw.d);
		if (m->qw.kind == 1) {
			q = zq_to_double(m->qw.q);
			x = (2 * q * (w + q) + d * tp) / (tp * tp);
		} else
			x = d / tp;
	} else
		return 0;
	/* to the minimal model: x' = (x - r) / u^2 */
	r = zq_to_double(feq(&m->tomin.r));
	u2 = zq_to_double(feq(&m->tomin.u));
	u2 *= u2;
	*xmin = (x - r) / u2;
	return 1;
}

/* ================================================================== */
/* classification and base points                                      */
/* ================================================================== */

/* the cubic form of a homogeneous F in the variables v[0..2] */
static void
cubic_coeffs(const struct qpoly *F, const int *v, fe *c)
{
	int i, k;

	for (k = 0; k < 10; k++)
		fe_zero(QQd(), &c[k]);
	for (i = 0; i < F->n; i++) {
		const uint8_t *e = F->e + i * QP_MAXV;
		k = mono_idx(e[v[0]], e[v[1]], e[v[2]]);
		if (k < 0)
			fail("internal: not a cubic form");
		c[k] = fe_from_q(F->c[i]);
	}
}

static int
on_cubic(const fe *c, const fe *X)
{
	fe r;

	ec_cub_eval(QQd(), c, X, &r);
	return fe_is_zero(QQd(), &r);
}

static int
smooth_on_cubic(const fe *c, const fe *X)
{
	fe g[3];

	ec_cub_grad(QQd(), c, X, g);
	return !(fe_is_zero(QQd(), &g[0]) && fe_is_zero(QQd(), &g[1]) &&
	    fe_is_zero(QQd(), &g[2]));
}

/* small points of the projective cubic: |coordinates| <= B */
static int
cubic_points(const fe *c, int B, fe (*out)[3], int max)
{
	int64_t ci[10];
	int k, n = 0, small = 1;
	int64_t x, y, z;

	for (k = 0; k < 10; k++) {
		struct zq q = feq(&c[k]);
		if (!z_is_one(q.d) || !z_fits_i64(q.n) || z_bits(q.n) > 50)
			small = 0;
		else
			ci[k] = z_to_i64(q.n);
	}
	for (z = 0; z <= B && n < max; z++)
		for (y = -B; y <= B && n < max; y++)
			for (x = -B; x <= B && n < max; x++) {
				int64_t v[3] = {x, y, z};
				fe X[3];
				int ok;
				/* primitive, first nonzero positive */
				if (z == 0 && (y < 0 || (y == 0 && x <= 0)))
					continue;
				if (nt_gcd(nt_gcd((u64)(x < 0 ? -x : x), (u64)(y < 0 ?
				    -y : y)), (u64)z) != 1)
					continue;
				if (small) {
					i128 s = 0;
					for (k = 0; k < 10; k++) {
						i128 t = ci[k];
						int a;
						if (!t)
							continue;
						for (a = 0; a < 3; a++) {
							int e;
							for (e = 0; e < ec_cub_mono[k][a]; e++)
								t *= v[a];
						}
						s += t;
					}
					ok = s == 0;
				} else {
					for (k = 0; k < 3; k++)
						fe_set_i64(QQd(), &X[k], v[k]);
					ok = on_cubic(c, X);
				}
				if (!ok)
					continue;
				for (k = 0; k < 3; k++)
					fe_set_i64(QQd(), &out[n][k], v[k]);
				n++;
			}
	return n;
}

/* the coefficients (low first, degree <= 4) of F in x, at y^k */
static int
coeffs_in(const struct qpoly *F, int vx, int vy, int k, struct zq *c,
    int weighted_y1)
{
	int i, j;

	for (j = 0; j < 5; j++)
		c[j] = Zq(0);
	for (i = 0; i < F->n; i++) {
		const uint8_t *e = F->e + i * QP_MAXV;
		int d;
		if (e[vy] != k)
			continue;
		d = e[vx];
		if (d > 4)
			return 0;
		c[d] = QADD(c[d], F->c[i]);
	}
	(void)weighted_y1;
	return 1;
}

static void
print_zq_poly(struct sbuf *s, const struct zq *c, int n, const char *v)
{
	struct qvars V;
	struct qpoly p;
	int i;

	memset(&V, 0, sizeof V);
	V.nv = 1;
	snprintf(V.name[0], sizeof V.name[0], "%s", v);
	qp_init(&p);
	for (i = 0; i < n; i++) {
		uint8_t e[QP_MAXV] = {0};
		e[0] = (uint8_t)i;
		qp_addterm(&p, e, c[i]);
	}
	qp_print(s, &p, &V);
	qp_free(&p);
}

static void
print_point3(struct sbuf *s, const fe *X)
{
	sb_printf(s, "(%s : %s : %s)", fe_str(QQd(), &X[0]),
	    fe_str(QQd(), &X[1]), fe_str(QQd(), &X[2]));
}

/* ================================================================== */
/* the pipeline                                                        */
/* ================================================================== */

/* values of the variables from a point of the curve model; 0 when the
 * point gives no affine solution */
static int
point_values(const struct dio_map *m, const struct dio_eq *D, const fe *X,
    struct zq *val)
{
	int i;

	for (i = 0; i < QP_MAXV; i++)
		val[i] = Zq(0);
	if (m->kind == DK_YQUAD) {
		if (m->weighted) {
			/* (x : 1 : z) -> (p, q, z q^2) */
			struct zq x = feq(&X[0]), z = feq(&X[1]);
			val[m->vx] = zq_int(x.n);
			val[m->vy] = zq_int(x.d);
			val[m->vw] = QMUL(z, zq_int(z_mul(x.d, x.d)));
			return 1;
		}
		val[m->vx] = feq(&X[0]);
		val[m->vy] = feq(&X[1]);
		return 1;
	}
	if (m->affine) {
		if (fe_is_zero(QQd(), &X[2]))
			return 0;
		val[m->vmap[0]] = QDIV(feq(&X[0]), feq(&X[2]));
		val[m->vmap[1]] = QDIV(feq(&X[1]), feq(&X[2]));
		return 1;
	}
	/* projective: scale to coprime integers */
	{
		zint L = Z_ONE, g = 0;
		struct zq q[3];
		for (i = 0; i < 3; i++) {
			q[i] = feq(&X[i]);
			L = z_lcm(L, q[i].d);
		}
		for (i = 0; i < 3; i++) {
			q[i] = QMUL(q[i], zq_int(L));
			g = z_gcd(g, q[i].n);
		}
		if (z_is_zero(g))
			return 0;
		for (i = 0; i < 3; i++)
			val[m->vmap[i]] = zq_int(z_divexact(q[i].n, g));
	}
	(void)D;
	return 1;
}

/* the side conditions and the exact substitution */
static int
check_solution(const struct dio_map *m, const struct dio_eq *D,
    const struct dio_opts *o, struct zq *val, const char **why)
{
	int i, nv = D->V.nv;
	struct zq l, r, dl, dr;

	if (m->proj && !m->affine && o->positive) {
		/* projective: an overall sign is free */
		int neg = 0;
		for (i = 0; i < nv; i++)
			if (z_sign(val[i].n) < 0)
				neg++;
		if (neg == nv)
			for (i = 0; i < nv; i++)
				val[i] = zq_neg(val[i]);
	}
	for (i = 0; i < D->nden; i++)
		if (zq_is_zero(qp_eval(&D->den[i], val))) {
			*why = "a denominator vanishes";
			return 0;
		}
	for (i = 0; i < nv; i++) {
		if (o->positive && z_sign(val[i].n) <= 0) {
			*why = "not positive";
			return 0;
		}
		if (o->nonzero && z_is_zero(val[i].n)) {
			*why = "zero";
			return 0;
		}
		if (o->integral && !z_is_one(val[i].d)) {
			*why = "not integral";
			return 0;
		}
	}
	dl = qp_eval(&D->lhs.d, val);
	dr = qp_eval(&D->rhs.d, val);
	if (zq_is_zero(dl) || zq_is_zero(dr)) {
		*why = "a denominator vanishes";
		return 0;
	}
	l = QDIV(qp_eval(&D->lhs.n, val), dl);
	r = QDIV(qp_eval(&D->rhs.n, val), dr);
	if (!zq_eq(l, r))
		fail("internal: a point mapped back does not satisfy the "
		    "equation (%s != %s)", zq_str(l), zq_str(r));
	return 1;
}

static int
ndigits(zint a)
{
	return (int)strlen(z_str(z_abs(a)));
}

/* build the maps for the equation; prints the derivation */
static int
dio_maps(struct dio_eq *D, struct dio_map *m, const struct dio_opts *o,
    struct sbuf *out)
{
	const struct qpoly *F = &D->F;
	int nv = D->V.nv, deg = qp_deg(F), k;

	memset(m, 0, sizeof *m);
	if (nv == 3 && deg == 3 && qp_homogeneous(F)) {
		m->proj = 1;
		m->vmap[0] = 0;
		m->vmap[1] = 1;
		m->vmap[2] = 2;
		cubic_coeffs(F, m->vmap, m->cub);
		sb_printf(out, "a plane cubic in P^2 (homogeneous of degree 3 in "
		    "%s, %s, %s)\n", D->V.name[0], D->V.name[1], D->V.name[2]);
	} else if (nv == 2 && deg == 3) {
		/* homogenise with a third coordinate */
		struct qpoly H;
		int i;
		qp_init(&H);
		for (i = 0; i < F->n; i++) {
			uint8_t e[QP_MAXV];
			memcpy(e, F->e + i * QP_MAXV, QP_MAXV);
			e[2] = (uint8_t)(3 - tdeg(F->e + i * QP_MAXV));
			qp_addterm(&H, e, F->c[i]);
		}
		m->proj = 1;
		m->affine = 1;
		m->vmap[0] = 0;
		m->vmap[1] = 1;
		m->vmap[2] = 2;
		cubic_coeffs(&H, m->vmap, m->cub);
		qp_free(&H);
		sb_printf(out, "an affine cubic in %s, %s: homogenised with z "
		    "(solutions have z != 0)\n", D->V.name[0], D->V.name[1]);
	} else if (nv == 2 || nv == 3) {
		/* y-quadratic: F = A(x) y^2 + B(x) y + C(x), or weighted
		 * F = a z^2 + G(x, y) with G a quartic form */
		int vx, vy, vw = -1, found = 0;
		for (vy = 0; vy < nv && !found; vy++)
			for (vx = 0; vx < nv && !found; vx++) {
				struct qpoly G;
				int i, ok = 1;
				if (vx == vy)
					continue;
				if (qp_deg_in(F, vy) != 2)
					continue;
				if (nv == 3) {
					vw = 3 - vx - vy;
					/* weighted: z = vy only as z^2, G(x, w) of
					 * degree 4 homogeneous */
					for (i = 0; i < F->n && ok; i++) {
						const uint8_t *e = F->e + i * QP_MAXV;
						if (e[vy] == 1)
							ok = 0;
						else if (e[vy] == 2 && (e[vx] || e[vw]))
							ok = 0;
						else if (e[vy] == 0 && e[vx] + e[vw] != 4)
							ok = 0;
					}
					if (!ok)
						continue;
					m->weighted = 1;
					/* x := vx, y := vw = 1 */
					qp_init(&G);
					for (i = 0; i < F->n; i++) {
						const uint8_t *e = F->e + i * QP_MAXV;
						if (e[vy] == 2)
							m->PA[0] = QADD(m->PA[0], F->c[i]);
						else
							m->PC[e[vx]] = QADD(m->PC[e[vx]], F->c[i]);
					}
					qp_free(&G);
				} else {
					if (!coeffs_in(F, vx, vy, 2, m->PA, 0) ||
					    !coeffs_in(F, vx, vy, 1, m->PB, 0) ||
					    !coeffs_in(F, vx, vy, 0, m->PC, 0))
						continue;
				}
				/* D = B^2 - 4AC of degree <= 4 */
				{
					struct zq Dq[9];
					int a, b, hi = 0;
					for (a = 0; a < 9; a++)
						Dq[a] = Zq(0);
					for (a = 0; a < 5; a++)
						for (b = 0; b < 5; b++) {
							Dq[a + b] = QADD(Dq[a + b], QMUL(m->PB[a],
							    m->PB[b]));
							Dq[a + b] = QSUB(Dq[a + b], QMUL(Zq(4),
							    QMUL(m->PA[a], m->PC[b])));
						}
					for (a = 5; a < 9; a++)
						if (!zq_is_zero(Dq[a]))
							hi = 1;
					if (hi)
						continue;
					for (a = 0; a < 5; a++)
						m->D[a] = Dq[a];
				}
				m->kind = DK_YQUAD;
				m->vx = vx;
				m->vy = vy;
				m->vw = vw;
				found = 1;
			}
		if (!found) {
			sb_printf(out, "not a curve this pipeline handles: it needs "
			    "a cubic (in 3 homogeneous or 2 affine variables), a "
			    "curve quadratic in one variable with a quartic "
			    "discriminant, or a z^2 = quartic form\n");
			return 0;
		}
		{
			struct sbuf s = {0};
			print_zq_poly(&s, m->D, 5, D->V.name[m->vx]);
			if (m->weighted)
				sb_printf(out, "weighted homogeneous (%s, %s of weight "
				    "1, %s of weight 2): set %s = 1, then %s^2 = %s / %s\n",
				    D->V.name[m->vx], D->V.name[m->vw],
				    D->V.name[m->vy], D->V.name[m->vw], "s", sb_str(&s),
				    "1");
			else
				sb_printf(out, "quadratic in %s: s = 2 A(%s) %s + "
				    "B(%s) gives s^2 = B^2 - 4AC = %s\n",
				    D->V.name[m->vy], D->V.name[m->vx], D->V.name[m->vy],
				    D->V.name[m->vx], sb_str(&s));
			free(s.p);
		}
		/* a rational point on s^2 = D(x) */
		{
			int64_t n, d, Bd = o->search > 0 && o->search < 200 ?
			    o->search : 60;
			int got = 0;
			for (d = 1; d <= Bd && !got; d++)
				for (n = 0; n <= Bd && !got; n++) {
					int sg;
					for (sg = 0; sg < 2 && !got; sg++) {
						struct zq x, v, s;
						if (nt_gcd((u64)n, (u64)d) != 1 || (sg && n == 0))
							continue;
						x = zq_make(I(sg ? -n : n), I(d));
						v = evall(m->D, 5, x);
						if (!zq_is_square(v, &s))
							continue;
						if (!q2w_make(&m->qw, m->D, x, s))
							continue;
						got = 1;
						sb_printf(out, "a point on the quartic: (%s, %s) = "
						    "(%s, %s)\n", D->V.name[m->vx], "s", zq_str(x),
						    zq_str(s));
					}
				}
			if (!got) {
				sb_printf(out, "no rational point on the quartic with "
				    "height <= %lld was found (it may have none: then the "
				    "equation has no rational solutions with these "
				    "variables)\n", (long long)Bd);
				return 0;
			}
		}
		q2w_curve(&m->qw, &m->W);
		goto weier;
	} else {
		sb_printf(out, "not a curve this pipeline handles (%d variables, "
		    "degree %d)\n", nv, deg);
		return 0;
	}
	/* projective cubic: a base point */
	{
		fe pts[32][3];
		int np = cubic_points(m->cub, 12, pts, 32), best = -1;
		if (np == 0)
			np = cubic_points(m->cub, 40, pts, 32);
		for (k = 0; k < np; k++)
			if (smooth_on_cubic(m->cub, pts[k]) &&
			    ec_cub_is_flex(QQd(), m->cub, pts[k])) {
				best = k;
				break;
			}
		if (best < 0)
			for (k = 0; k < np; k++)
				if (smooth_on_cubic(m->cub, pts[k])) {
					best = k;
					break;
				}
		if (best < 0) {
			sb_printf(out, "no smooth rational point with coordinates of "
			    "size <= 40 was found on the cubic\n");
			return 0;
		}
		for (k = 0; k < 3; k++)
			m->P0[k] = pts[best][k];
		sb_printf(out, "rational points found by search: %d; base point "
		    "P0 = ", np);
		print_point3(out, m->P0);
		if (ec_cub_flex_to_w(QQd(), m->cub, m->P0, &m->W, &m->fm)) {
			m->kind = DK_FLEX;
			sb_printf(out, ", a flex: its tangent line to infinity and "
			    "P0 to O give Weierstrass form directly\n");
		} else {
			struct sbuf s = {0};
			if (!lines_setup(m)) {
				sb_printf(out, "\nthe cubic is singular at P0 (genus 0 "
				    "or reducible)\n");
				return 0;
			}
			m->kind = DK_LINES;
			print_zq_poly(&s, m->D, 5, "t");
			sb_printf(out, " (not a flex)\nlines through P0 with slope t "
			    "meet the cubic in two more points, real or conjugate: "
			    "the discriminant gives w^2 = %s\n", sb_str(&s));
			sb_printf(out, "the tangent at P0 (t = %s) gives the point "
			    "(t, w) = (%s, %s)\n", zq_str(m->qw.t0), zq_str(m->qw.t0),
			    zq_str(m->qw.q));
			free(s.p);
			q2w_curve(&m->qw, &m->W);
		}
	}
weier:
	if (m->W.singular) {
		sb_printf(out, "the curve is singular (genus 0): not an elliptic "
		    "curve\n");
		return 0;
	}
	{
		struct sbuf s = {0};
		ec_print(&s, &m->W);
		sb_printf(out, "Weierstrass form: %s\n", sb_str(&s));
		free(s.p);
	}
	ecq_minimal(&m->W, &m->Emin, &m->tomin);
	{
		struct sbuf s = {0};
		ec_print(&s, &m->Emin);
		sb_printf(out, "minimal model E: %s, by ", sb_str(&s));
		eciso_print(&s, QQd(), &m->tomin);
		sb_printf(out, "%s\n", strstr(sb_str(&s), "[u"));
		free(s.p);
	}
	return 1;
}

static void
sort_vectors(int64_t *v, double *q, int n, int r)
{
	int i, j, k;

	for (i = 1; i < n; i++)
		for (j = i; j > 0 && q[j - 1] > q[j]; j--) {
			double t = q[j];
			q[j] = q[j - 1];
			q[j - 1] = t;
			for (k = 0; k < r; k++) {
				int64_t s = v[j * r + k];
				v[j * r + k] = v[(j - 1) * r + k];
				v[(j - 1) * r + k] = s;
			}
		}
}

/* the real points of the cubic with every coordinate > 0 (the plane
 * a + b + c = 1), mapped to the minimal model: how many land on the
 * bounded component and how many on the other */
static void
real_positive_samples(const struct dio_map *m, const struct ec *E,
    int *n_egg, int *n_inf, int *n_tot)
{
	double c[10], b2 = zq_to_double(feq(&E->b2)), b4 = zq_to_double(feq(
	    &E->b4)), b6 = zq_to_double(feq(&E->b6));
	long double rt[3];
	int nr, k, i, j, G = 300;

	*n_egg = *n_inf = *n_tot = 0;
	nr = ecq_real_roots(b2, b4, b6, rt);
	for (k = 0; k < 10; k++)
		c[k] = zq_to_double(feq(&m->cub[k]));
	for (i = 1; i < G; i++) {
		double a = (double)i / G, prev = 0;
		for (j = 0; j <= 4 * G; j++) {
			double bb = (1 - a) * j / (4.0 * G), X[3], v = 0;
			X[0] = a;
			X[1] = bb;
			X[2] = 1 - a - bb;
			for (k = 0; k < 10; k++)
				v += c[k] * pow(X[0], ec_cub_mono[k][0]) * pow(X[1],
				    ec_cub_mono[k][1]) * pow(X[2], ec_cub_mono[k][2]);
			if (j > 0 && ((prev < 0) != (v < 0)) && j < 4 * G) {
				/* a sign change: bisect for the root */
				double lo = (1 - a) * (j - 1) / (4.0 * G), hi = bb, x;
				int it;
				for (it = 0; it < 60; it++) {
					double mid = (lo + hi) / 2, w = 0, Y[3];
					Y[0] = a;
					Y[1] = mid;
					Y[2] = 1 - a - mid;
					for (k = 0; k < 10; k++)
						w += c[k] * pow(Y[0], ec_cub_mono[k][0]) *
						    pow(Y[1], ec_cub_mono[k][1]) *
						    pow(Y[2], ec_cub_mono[k][2]);
					if ((w < 0) == (prev < 0))
						lo = mid;
					else
						hi = mid;
				}
				X[1] = (lo + hi) / 2;
				X[2] = 1 - a - X[1];
				if (X[1] > 0 && X[2] > 0 && map_fwd_real(m, X, &x)) {
					(*n_tot)++;
					if (nr == 3 && x >= rt[0] - 1e-9 * (1 + fabsl(rt[0])) &&
					    x <= rt[1] + 1e-9 * (1 + fabsl(rt[1])))
						(*n_egg)++;
					else
						(*n_inf)++;
				}
			}
			prev = v;
		}
	}
}

int
dio_solve_eq(struct dio_eq *D, const struct dio_opts *o)
{
	struct dio_map m;
	struct ecq_mw M;
	struct sbuf out = {0};
	struct eciso toE;
	int r, i, nsol = 0, tried = 0, unmapped = 0;
	zmark mk;

	{
		struct sbuf s = {0};
		qp_print(&s, &D->F, &D->V);
		printf("equation: %s = 0", sb_str(&s));
		rep_str("polynomial", sb_str(&s));
		free(s.p);
		if (D->nden) {
			printf("  (with ");
			for (i = 0; i < D->nden; i++) {
				struct sbuf s2 = {0};
				qp_print(&s2, &D->den[i], &D->V);
				printf("%s%s != 0", i ? ", " : "", sb_str(&s2));
				free(s2.p);
			}
			printf(")");
		}
		printf("\n");
	}
	if (!dio_maps(D, &m, o, &out)) {
		printf("%s", sb_str(&out));
		free(out.p);
		rep_bool("elliptic", 0);
		return -1;
	}
	printf("%s", sb_str(&out));
	sb_clear(&out);
	ecq_mw_compute(&m.Emin, &M, o->search > 0 ? o->search : 2000);
	eciso_compose(QQd(), &toE, &m.tomin, &M.tomin);
	{
		struct sbuf s = {0};
		ec_print_ainv(&s, &M.E);
		rep_str("curve", sb_str(&s));
		free(s.p);
	}
	printf("torsion: ");
	if (M.T.n == 1)
		printf("trivial");
	else if (M.T.n1 == 1)
		printf("Z/%d", M.T.n2);
	else
		printf("Z/2 x Z/%d", M.T.n2);
	printf(" (%d points)\n", M.T.n);
	if (M.have_descent)
		printf("2-isogeny descent: Selmer ranks %d + %d, images found %d "
		    "+ %d: %d <= rank <= %d\n", M.D.nsel, M.D.nsel2, M.D.nimg,
		    M.D.nimg2, M.D.rank_lo, M.D.rank_hi);
	r = (int)M.gens.n;
	if (M.rank_hi >= 0 && M.rank_lo == M.rank_hi)
		printf("rank %d (proved: descent bound met by %d independent "
		    "points)\n", M.rank_lo, r);
	else if (M.rank_hi >= 0)
		printf("%d <= rank <= %d\n", M.rank_lo, M.rank_hi);
	else
		printf("rank >= %d (no 2-torsion: no descent bound here)\n",
		    M.rank_lo);
	rep_int("rank_lo", M.rank_lo);
	if (M.rank_hi >= 0)
		rep_int("rank_hi", M.rank_hi);
	for (i = 0; i < r; i++) {
		struct ecpt P;
		ecpts_get(&M.gens, (uint32_t)i, &P);
		printf("generator P%d = %s, height %.10f\n", i + 1,
		    ec_pt_str(&M.E, &P), M.h[i]);
	}
	if (r && M.saturated_to)
		printf("saturated at the primes <= %d\n", M.saturated_to);
	/* the search */
	{
		int K = o->maxmult > 0 ? o->maxmult : (r <= 1 ? 30 : r == 2 ? 10 :
		    4);
		int64_t nvec = 1, *vec;
		double *qv;
		int64_t idx;
		double B[64];
		int a, b, want = o->want > 0 ? o->want : 1;
		if (o->want < 0)
			nvec = 0;
		for (i = 0; i < r; i++)
			nvec *= 2 * K + 1;
		if (nvec > 200000)
			fail("too many combinations (lower maxmult)");
		vec = xmalloc(sizeof(int64_t) * (size_t)(nvec * (r ? r : 1) + 1));
		qv = xmalloc(sizeof(double) * ((size_t)nvec + 1));
		for (a = 0; a < r; a++)
			for (b = 0; b < r; b++) {
				struct ecpt P, Q;
				ecpts_get(&M.gens, (uint32_t)a, &P);
				ecpts_get(&M.gens, (uint32_t)b, &Q);
				B[a * r + b] = a == b ? M.h[a] : ecq_height_pairing(&M.E,
				    &P, &Q);
			}
		for (idx = 0; idx < nvec; idx++) {
			int64_t t = idx;
			double q = 0;
			for (a = 0; a < r; a++) {
				vec[idx * r + a] = t % (2 * K + 1) - K;
				t /= 2 * K + 1;
			}
			for (a = 0; a < r; a++)
				for (b = 0; b < r; b++)
					q += B[a * r + b] * (double)vec[idx * r + a] *
					    (double)vec[idx * r + b];
			qv[idx] = q;
		}
		if (r)
			sort_vectors(vec, qv, (int)nvec, r);
		rep_list("solutions");
		for (idx = 0; idx < nvec && nsol < want; idx++) {
			struct ecpt S;
			int ti;
			mk = z_mark();
			ec_O(&S);
			for (a = 0; a < r; a++) {
				struct ecpt P, Q;
				if (!vec[idx * r + a])
					continue;
				ecpts_get(&M.gens, (uint32_t)a, &P);
				ec_mul_i64(&M.E, &Q, vec[idx * r + a], &P);
				ec_add(&M.E, &S, &S, &Q);
			}
			for (ti = 0; ti < M.T.n && nsol < want; ti++) {
				struct ecpt R, Wp, back;
				fe X[3];
				struct zq val[QP_MAXV];
				const char *why = NULL;
				ec_add(&M.E, &R, &S, &M.T.pts[ti]);
				tried++;
				eciso_pt_inv(&m.W, &toE, &Wp, &R);
				if (!ec_on(&m.W, &Wp))
					fail("internal: the point is not on W");
				if (!map_back(&m, &Wp, X)) {
					unmapped++;
					continue;
				}
				/* the point is on the model, and the forward map
				 * returns it */
				if (m.kind != DK_YQUAD && !on_cubic(m.cub, X))
					fail("internal: mapped back off the cubic");
				if (map_fwd(&m, X, &back) && !ec_pt_eq(&m.W, &back, &Wp))
					fail("internal: the maps do not compose to the "
					    "identity at %s", ec_pt_str(&m.W, &Wp));
				if (!point_values(&m, D, X, val))
					continue;
				if (!check_solution(&m, D, o, val, &why))
					continue;
				nsol++;
				printf("solution %d, from ", nsol);
				{
					int first = 1;
					for (a = 0; a < r; a++)
						if (vec[idx * r + a]) {
							printf("%s%lld*P%d", first ? "" : " + ",
							    (long long)vec[idx * r + a], a + 1);
							first = 0;
						}
					if (M.T.pts[ti].inf) {
						if (first)
							printf("O");
					} else
						printf("%s%s", first ? "" : " + ",
						    ec_pt_str(&M.E, &M.T.pts[ti]));
				}
				printf(":\n");
				rep_obj(NULL);
				for (a = 0; a < D->V.nv; a++) {
					printf("  %s = %s", D->V.name[a], zq_str(val[a]));
					if (z_is_one(val[a].d) && ndigits(val[a].n) > 12)
						printf("  (%d digits)", ndigits(val[a].n));
					printf("\n");
					rep_str(D->V.name[a], zq_str(val[a]));
				}
				rep_str("from", "");
				rep_end();
				printf("  check: substituted into the equation, both sides "
				    "are equal\n");
			}
			z_release(mk);
		}
		rep_end();
		free(vec);
		free(qv);
		if (o->want >= 0) {
		printf("points tried: %d (%d at exceptional points of the maps)\n",
		    tried, unmapped);
		rep_int("solutions_found", nsol);
		}
		if (nsol == 0 && o->want >= 0) {
			if (M.rank_hi == 0 && unmapped == 0) {
				printf("no solutions: E(Q) is finite (rank 0 proved) and "
				    "every point of it was mapped back and fails the "
				    "conditions\n");
				rep_str("conclusion", "none (proved)");
			} else if (o->positive && m.proj && !m.affine &&
			    m.kind != DK_YQUAD) {
				int ne, ni, nt;
				char why[512];
				real_positive_samples(&m, &M.E, &ne, &ni, &nt);
				if (nt == 0)
					printf("the real curve has no point with every "
					    "coordinate positive (none at %d x %d samples of "
					    "the triangle)\n", 300, 1200);
				else if (ni == 0) {
					printf("the positive real points lie on the bounded "
					    "component (egg) of E(R) (all %d sample points "
					    "of the positive region)\n", nt);
					if (ecq_egg_empty(&M.E, why, sizeof why)) {
						printf("no positive solutions: no rational point "
						    "lies on the egg: %s\n", why);
						rep_str("conclusion", "no positive solutions "
						    "(egg empty by 2-descent)");
					} else
						printf("whether the egg has rational points is "
						    "open here: %s\n", why);
				} else
					printf("positive real points: %d on the egg, %d on "
					    "the unbounded component; none found among the "
					    "combinations tried: raise maxmult\n", ne, ni);
			} else
				printf("none found among the combinations tried\n");
		}
	}
	ecq_mw_free(&M);
	free(out.p);
	return nsol;
}

int
dio_solve(const char *eq, const struct dio_opts *o)
{
	struct dio_eq D;
	const char *why = NULL;
	int n;

	if (!dio_parse(eq, &D, &why))
		fail("diophantine: %s", why ? why : "cannot parse the equation");
	n = dio_solve_eq(&D, o);
	dio_free(&D);
	return n;
}
