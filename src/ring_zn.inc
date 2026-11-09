/* ring.c, part 2: expressions over a ring, and the commands on Z/n */

/* ---- polynomials over R in one variable ---- */

static const struct ring *PR;
static const char *pvar = "x";
static int p_rem_note;

static void p_sum(struct rpoly *r);

static int
at_mult_start(void)
{
	if (rti >= rlim)
		return 0;
	if (at_sym('('))
		return 1;
	if (rt[rti].k != RK_ID)
		return 0;
	return !strcmp(rt[rti].s, pvar) || (PR->deg > 1 && (at_id("a") ||
	    at_id("alpha"))) || at_id("gcd");
}

static u64
code_arg(const struct ring *R, const struct rtok *t)
{
	if (t->big || (t->v >> 64))
		rfail("%s is too large", u128s(t->v));
	if (R->deg == 1)
		return (u64)(t->v % R->n);
	if ((u64)t->v > R->qm1)
		rfail("%s is not an element of a ring with %s elements (codes 0..%s)",
		    u128s(t->v), u128s((u128)R->qm1 + 1), u128s(R->qm1));
	if ((u64)t->v >= R->n)
		note("integers are elements in integer notation: %s = %s",
		    el_code(R, (u64)t->v), el_pstr(R, (u64)t->v, "x"));
	return (u64)t->v;
}

static void
p_atom(struct rpoly *r)
{
	if (acc_sym('(')) {
		p_sum(r);
		expect_sym(')');
		return;
	}
	if (rti < rlim && rt[rti].k == RK_NUM) {
		rp_const(PR, r, code_arg(PR, &rt[rti]));
		rti++;
		return;
	}
	if (rti < rlim && rt[rti].k == RK_ID && !strcmp(rt[rti].s, pvar)) {
		rti++;
		rp_x(r);
		return;
	}
	if (PR->deg > 1 && (acc_id("a") || acc_id("alpha"))) {
		rp_const(PR, r, r_x(PR));
		return;
	}
	if (acc_id("gcd")) {
		struct rpoly b;
		expect_sym('(');
		p_sum(r);
		expect_sym(',');
		p_sum(&b);
		expect_sym(')');
		if (!rp_gcd(PR, r, r, &b))
			rfail("gcd over a ring that is not a field (a leading "
			    "coefficient is not a unit)");
		return;
	}
	rfail("expected a polynomial in %s, found %s", pvar, here());
}

static void p_unary(struct rpoly *r);

static void
p_pow(struct rpoly *r)
{
	struct rpoly b;
	i128 e;

	p_atom(r);
	if (!acc_sym('^'))
		return;
	e = iexpr_unary();
	if (e < 0)
		rfail("negative powers of polynomials");
	if (r->d > 0 && e * r->d > RMAXDEG)
		rfail("degree above %d", RMAXDEG);
	b = *r;
	rp_const(PR, r, r_const(PR, 1));
	while (e-- > 0)
		rp_mul(PR, r, r, &b);
}

static void
p_unary(struct rpoly *r)
{
	if (acc_sym('-')) {
		struct rpoly z;
		p_unary(r);
		rp_zero(&z);
		rp_sub(PR, r, &z, r);
		return;
	}
	acc_sym('+');
	p_pow(r);
}

static void
p_term(struct rpoly *r)
{
	struct rpoly b, q, m;

	p_unary(r);
	for (;;) {
		if (acc_sym('*') || at_mult_start()) {
			p_unary(&b);
			if (!rp_mul(PR, r, r, &b) || r->d > RMAXDEG)
				rfail("degree above %d", RMAXDEG);
		} else if (acc_sym('/')) {
			p_unary(&b);
			if (b.d < 0)
				rfail("division by zero");
			if (!rp_divmod(PR, r, &b, &q, &m))
				rfail("the divisor's leading coefficient is not a unit");
			if (m.d >= 0 && !p_rem_note++)
				note("/ is the quotient; the remainder is %s",
				    rp_str(PR, &m));
			*r = q;
		} else if (acc_sym('%')) {
			p_unary(&b);
			if (b.d < 0)
				rfail("division by zero");
			if (!rp_divmod(PR, r, &b, NULL, &m))
				rfail("the divisor's leading coefficient is not a unit");
			*r = m;
		} else
			return;
	}
}

static void
p_sum(struct rpoly *r)
{
	struct rpoly b;

	p_term(r);
	for (;;) {
		if (acc_sym('+')) {
			p_term(&b);
			rp_add(PR, r, r, &b);
		} else if (acc_sym('-')) {
			p_term(&b);
			rp_sub(PR, r, r, &b);
		} else
			return;
	}
}

static void
pexpr_into(const struct ring *R, struct rpoly *p, const char *var)
{
	const struct ring *sr = PR;
	const char *sv = pvar;

	PR = R;
	pvar = var;
	p_rem_note = 0;
	p_sum(p);
	PR = sr;
	pvar = sv;
}

/* a whole polynomial argument [rti, rlim) */
static void
poly_arg(const struct ring *R, struct rpoly *p)
{
	int nk;

	/* a named polynomial: crc32, aes ... */
	if (R->deg == 1 && R->n == 2 && rti + 1 == rlim && rt[rti].k == RK_ID &&
	    (nk = named_index(rt[rti].s)) >= 0) {
		int i;
		p->d = named_mod[nk].deg;
		for (i = 0; i <= p->d; i++)
			p->c[i] = (u64)(named_mod[nk].code >> i & 1);
		note("%s is the %s polynomial %s", named_mod[nk].name,
		    named_mod[nk].what, rp_str(R, p));
		rti++;
		return;
	}
	/* GF(2): a lone hex or binary literal is a polynomial in bits */
	if (R->deg == 1 && R->n == 2 && rti + 1 == rlim &&
	    rt[rti].k == RK_NUM && rt[rti].base != 10) {
		u128 v = rt[rti].v;
		int i;
		p->d = -1;
		for (i = 0; v; i++, v >>= 1) {
			if (i > RMAXDEG)
				rfail("degree above %d", RMAXDEG);
			p->c[i] = (u64)(v & 1);
		}
		p->d = i - 1;
		note("%s is read as a polynomial in bit notation: %s",
		    u128x(rt[rti].v), rp_str(R, p));
		rti++;
		return;
	}
	pexpr_into(R, p, "x");
	if (!at_end())
		rfail("unexpected %s in the polynomial", here());
}

/* ---- elements: + - * / ^ (negative exponents), inv() ---- */

static const struct ring *ER;
static int e_alt;          /* multiply with the other algorithm */
static int e_ninv, e_nops;
static u64 e_inv_a[64], e_inv_r[64];
static int e_last_op;
static u64 e_last_a, e_last_b;

static u64 e_sum(void);

static u64
e_mul(u64 a, u64 b)
{
	e_nops++;
	e_last_op = '*';
	e_last_a = a;
	e_last_b = b;
	return e_alt ? r_mul_alt(ER, a, b) : r_mul(ER, a, b);
}

static u64
e_inverse(u64 a)
{
	u64 r, g;

	if (!r_inv(ER, a, &r)) {
		if (ER->deg == 1) {
			nt_inv(a, ER->n, &g);
			rfail("%llu has no inverse mod %llu: gcd(%llu, %llu) = %llu "
			    "(%llu * %llu = 0)", U(a), U(ER->n), U(a), U(ER->n), U(g),
			    U(a), U(ER->n / g));
		}
		rfail("%s has no inverse%s", el_code(ER, a), a == 0 ? "" :
		    " (it is a zero divisor)");
	}
	if (e_ninv < 64) {
		e_inv_a[e_ninv] = a;
		e_inv_r[e_ninv++] = r;
	}
	return r;
}

static u64
e_atom(void)
{
	u64 v;

	if (acc_sym('(')) {
		v = e_sum();
		expect_sym(')');
		return v;
	}
	if (rti < rlim && rt[rti].k == RK_NUM)
		return code_arg(ER, &rt[rti++]);
	if (ER->deg > 1 && (acc_id("x") || acc_id("a") || acc_id("alpha")))
		return r_x(ER);
	if (acc_id("inv") || acc_id("inverse")) {
		expect_sym('(');
		v = e_sum();
		expect_sym(')');
		return e_inverse(v);
	}
	if (ER->deg == 1 && at_id("x"))
		rfail("x is not an element of Z/%llu (polynomials: poly ... mod %llu)",
		    U(ER->n), U(ER->n));
	rfail("expected an element, found %s", here());
	return 0;
}

static u64 e_unary(void);

static u64
e_pow(void)
{
	u64 b = e_atom(), r;
	i128 e;

	if (!acc_sym('^'))
		return b;
	e = iexpr_unary();
	if (e < 0) {
		b = e_inverse(b);
		e = -e;
	}
	if (e >> 64)
		rfail("exponent too large");
	r = r_pow(ER, b, (u64)e);
	e_nops++;
	e_last_op = '^';
	return r;
}

static u64
e_unary(void)
{
	if (acc_sym('-'))
		return r_neg(ER, e_unary());
	acc_sym('+');
	return e_pow();
}

static u64
e_term(void)
{
	u64 v = e_unary();

	for (;;) {
		if (acc_sym('*') || (rti < rlim && (at_sym('(') ||
		    (ER->deg > 1 && (at_id("x") || at_id("a"))))))
			v = e_mul(v, e_unary());
		else if (acc_sym('/')) {
			u64 b = e_unary();
			v = e_mul(v, e_inverse(b));
			e_last_op = '/';
		} else
			return v;
	}
}

static u64
e_sum(void)
{
	u64 v = e_term();

	for (;;) {
		if (acc_sym('+')) {
			v = r_add(ER, v, e_term());
			e_nops++;
			e_last_op = '+';
		} else if (acc_sym('-')) {
			v = r_sub(ER, v, e_term());
			e_nops++;
			e_last_op = '-';
		} else
			return v;
	}
}

static u64
eexpr(const struct ring *R, int alt)
{
	u64 v;

	ER = R;
	e_alt = alt;
	e_ninv = e_nops = 0;
	e_last_op = 0;
	v = e_sum();
	if (!at_end())
		rfail("unexpected %s", here());
	return v;
}

/* ---- lists ---- */

static void
print_u64s(const char *label, const u64 *v, u64 n, u64 total, int max)
{
	u64 i;

	printf("  %s", label);
	for (i = 0; i < n && i < (u64)max; i++)
		printf("%s%llu", i ? ", " : "", U(v[i]));
	if (total > (u64)max || n > (u64)max)
		printf(", ... (%llu in all)", U(total));
	putchar('\n');
}

static const char *
fac_text(u64 n)
{
	static char buf[4][256];
	static int k;
	char *b = buf[k++ & 3];
	struct nt_fac f;
	size_t o = 0;
	int i;

	nt_factor(n, &f);
	b[0] = 0;
	if (f.k == 0)
		snprintf(b, 256, "%llu", U(n));
	for (i = 0; i < f.k; i++) {
		o += (size_t)snprintf(b + o, 256 - o, "%s%llu", i ? " * " : "",
		    U(f.p[i]));
		if (f.e[i] > 1)
			o += (size_t)snprintf(b + o, 256 - o, "^%d", f.e[i]);
	}
	return b;
}

static u64
upow(u64 p, int e)
{
	u64 r = 1;

	while (e-- > 0)
		r *= p;
	return r;
}

/* ---- running ucas text through the evaluator ---- */

/*
 * src: a ucas expression in x with its type annotated; f(x) is compared
 * with want(x) at every x (w <= 16) or at the ends and 65536 random x.
 * Returns the number of inputs tried, *bad the mismatches.
 */
static u64
engine_check(const char *src, int w, u64 (*want)(u64, const void *),
    const void *arg, u64 *bad, int *all)
{
	struct func *f = cmd_func_text(src);
	struct plan pl;
	int64_t *ii;
	double *ir;
	u64 mask = w == 64 ? ~(u64)0 : (((u64)1 << w) - 1), x = 0, n = 0;
	u64 st = 0x243f6a8885a308d3ULL, lim;

	*all = w <= 16;
	lim = *all ? mask + 1 : 65536 + 512;
	*bad = 0;
	plan_build(&pl, &f->body, 1, 0);
	ii = xcalloc((size_t)MAXP * CH, sizeof *ii);
	ir = xcalloc((size_t)MAXP * CH, sizeof *ir);
	while (x < lim) {
		uint32_t k, m = 0;
		u64 xs[CH];
		for (k = 0; k < CH && x < lim; k++, x++) {
			u64 v = x;
			if (!*all) {
				if (x < 256)
					v = x;
				else if (x < 512)
					v = mask - (x - 256);
				else {
					st ^= st << 13;
					st ^= st >> 7;
					st ^= st << 17;
					v = st & mask;
				}
			}
			xs[k] = v;
			ii[k] = (int64_t)v;
			m++;
		}
		plan_run(&pl, m, ii, ir);
		for (k = 0; k < m; k++) {
			n++;
			if (((u64)COLI(&pl, pl.root[0])[k] & mask) != (want(xs[k], arg) &
			    mask))
				(*bad)++;
		}
	}
	plan_free(&pl);
	free(ii);
	free(ir);
	return n;
}

static u64
want_identity(u64 x, const void *arg)
{
	(void)arg;
	return x;
}

/* ---- inverse mod n, including 2^k up to 2^128 ---- */

static void
cmd_inverse_pow2(i128 a, int k)
{
	u128 mask = k == 128 ? ~(u128)0 : (((u128)1 << k) - 1), x, y;
	int it = 0, bits = 3;

	x = (u128)a & mask;
	if (!(x & 1)) {
		int z = 0;
		while (!(x >> z & 1) && z < k)
			z++;
		printf("%s has no inverse mod 2^%d: it is even (gcd = 2^%d), and "
		    "x * %s = 0 mod 2^%d\n", u128x(x), k, z, u128x(
		    k == 128 && z == 0 ? 1 : ((u128)1 << (k - z)) & mask), k);
		printf("  as u%d arithmetic: multiplying by an even constant loses "
		    "the top bit%s, so it cannot be undone\n", k, z > 1 ? "s" : "");
		return;
	}
	y = x;   /* correct to 3 bits: x*x = 1 mod 8 for odd x */
	while (bits < k) {
		y = (y * (2 - x * y)) & mask;
		bits *= 2;
		it++;
	}
	printf("%s^-1 = %s mod 2^%d", u128s(x), u128x(y), k);
	if (k <= 64)
		printf(" (%s)", u128s(y));
	putchar('\n');
	printf("  how: Newton's iteration y = y*(2 - a*y), from y = a (right to 3 "
	    "bits: a*a = 1 mod 8 for odd a); each step doubles the correct bits: "
	    "%d step%s\n", it, it == 1 ? "" : "s");
	printf("  check: %s * %s = 1 mod 2^%d (the product's low %d bits are "
	    "%s)\n", u128x(x), u128x(y), k, k, u128x((x * y) & mask));
	if (k == 8 || k == 16 || k == 32 || k == 64) {
		char src[200];
		u64 bad, n;
		int all;
		printf("  as u%d arithmetic: x * %s undoes x * %s (multiplying by an "
		    "odd constant is a bijection of u%d), and x * %s = x / %s "
		    "exactly when %s divides x\n", k, u128x(y), u128x(x), k,
		    u128x(y), u128s(x), u128s(x));
		snprintf(src, sizeof src, "(x: u%d) * %s * %s", k, u128x(x),
		    u128x(y));
		n = engine_check(src, k, want_identity, NULL, &bad, &all);
		printf("  check in ucas's u%d evaluator: x * %s * %s = x for %s "
		    "(%llu inputs)%s\n", k, u128x(x), u128x(y), all ? "every x" :
		    "the 256 smallest and largest x and 65536 random x", U(n),
		    bad ? ": FAILED" : "");
	}
}

static void
cmd_inverse_int(i128 a0, u64 n)
{
	u64 a, g, inv;
	u128 prod;

	if (n == 1) {
		printf("Z/1 is the zero ring: 0 = 1, and every element is its own "
		    "inverse\n");
		return;
	}
	a = (u64)(((a0 % (i128)n) + (i128)n) % (i128)n);
	if ((i128)a != a0)
		note("%s = %llu mod %llu", i128s(a0), U(a), U(n));
	inv = nt_inv(a, n, &g);
	if (g != 1) {
		printf("%llu has no inverse mod %llu: gcd(%llu, %llu) = %llu > 1\n",
		    U(a), U(n), U(a), U(n), U(g));
		if (a)
			printf("  %llu * %llu = %s = 0 mod %llu, so %llu is a zero "
			    "divisor; a*x = 1 would give %llu = %llu * %llu * x = 0\n",
			    U(a), U(n / g), u128s((u128)a * (n / g)), U(n), U(a),
			    U(n / g), U(n / g), U(a));
		printf("  the units mod %llu are the residues coprime to %llu "
		    "(phi(%llu) = %llu of them)\n", U(n), U(n), U(n), U(nt_phi(n)));
		return;
	}
	prod = (u128)a * inv;
	printf("%llu^-1 = %llu mod %llu", U(a), U(inv), U(n));
	if (n > 1000)
		printf(" (0x%llx)", U(inv));
	putchar('\n');
	printf("  how: extended Euclid: %llu*%llu - %s*%llu = 1\n", U(inv), U(a),
	    u128s((prod - 1) / n), U(n));
	printf("  check: %llu * %llu = %s = %s*%llu + %llu\n", U(a), U(inv),
	    u128s(prod), u128s(prod / n), U(n), U((u64)(prod % n)));
}

static void
cmd_inverse(void)
{
	struct rspec s;
	int end;
	i128 a;

	if (!find_ring(&s, rti, &end, 1))
		rfail("inverse A mod N (or in GF(q), Z/n[x]/(f), ...)");
	/* mod 2^k with k up to 128: Newton */
	if (s.kind == RS_ZN) {
		rlim = end;
		a = iexpr_sum();
		if (!at_end())
			rfail("unexpected %s", here());
		cmd_inverse_int(a, s.R.n);
		return;
	}
	if (s.kind == RS_EXT) {
		u64 v, r;
		rlim = end;
		v = eexpr(&s.R, 0);
		if (!r_inv(&s.R, v, &r)) {
			printf("%s has no inverse in %s%s\n", el_str(&s.R, v), s.name,
			    v ? ": it is a zero divisor" : "");
			return;
		}
		printf("%s^-1 = %s in %s\n", el_code(&s.R, v), el_str(&s.R, r),
		    s.name);
		printf("  how: %s\n", s.R.field ? "a^(q-2) (a^(q-1) = 1 for every "
		    "nonzero a)" : "extended Euclid on the polynomials");
		printf("  check: %s * %s = %s (shift-and-add multiplication)\n",
		    el_code(&s.R, v), el_code(&s.R, r), el_code(&s.R,
		    r_mul_alt(&s.R, v, r)));
		return;
	}
	rfail("inverse: Z/n, GF(q) or a quotient ring");
}

/* the 2^k forms that do not fit u64: "mod 2^64", "mod 2^128" */
static int
pow2_modulus(int from, int *k, int *end)
{
	int i;

	for (i = from; i + 4 <= nrt; i++)
		if (rt[i].k == RK_ID && (!strcasecmp(rt[i].s, "mod") ||
		    !strcasecmp(rt[i].s, "modulo")) && i + 4 == nrt &&
		    rt[i + 1].k == RK_NUM && rt[i + 1].v == 2 &&
		    rt[i + 2].k == RK_SYM && rt[i + 2].s[0] == '^' &&
		    rt[i + 3].k == RK_NUM && rt[i + 3].v >= 1 &&
		    rt[i + 3].v <= 128) {
			*k = (int)rt[i + 3].v;
			*end = i;
			return 1;
		}
	return 0;
}

/* ---- order, phi, lambda ---- */

static void
cmd_order_pow2(i128 a0, int k)
{
	u128 mask = k == 128 ? ~(u128)0 : (((u128)1 << k) - 1), a = (u128)a0 &
	    mask, y;
	int j;

	if (!(a & 1)) {
		printf("%s has no multiplicative order mod 2^%d: it is even, so no "
		    "power of it is 1\n", u128x(a), k);
		return;
	}
	/* the order of an odd a divides 2^(k-2) (k >= 3): square until 1 */
	for (j = 0, y = a; y != 1 && j < 130; j++)
		y = (y * y) & mask;
	printf("ord(%s) = 2^%d mod 2^%d\n", u128x(a), j, k);
	printf("  how: the units mod 2^%d form C2 x C2^%d, so the order is a "
	    "power of two: square until the result is 1 (%d squarings)\n", k,
	    k >= 3 ? k - 2 : 0, j);
	printf("  as u%d arithmetic: x * %s^(2^%d) = x, so multiplying by %s "
	    "2^%d times is the identity\n", k, u128x(a), j, u128x(a), j);
}

static void
cmd_order_int(i128 a0, u64 n)
{
	u64 a = (u64)(((a0 % (i128)n) + (i128)n) % (i128)n), o, lam, ph, k, x;
	struct nt_fac f;
	int i;

	o = nt_order(a, n);
	if (o == 0) {
		u64 g = nt_gcd(a, n);
		printf("%llu has no multiplicative order mod %llu: gcd(%llu, %llu) "
		    "= %llu, so no power of it is 1\n", U(a), U(n), U(a), U(n), U(g));
		printf("  additive order: %llu (%llu / gcd(%llu, %llu))\n",
		    U(n / g), U(n), U(a), U(n));
		return;
	}
	lam = nt_lambda(n);
	ph = nt_phi(n);
	printf("ord(%llu) = %llu mod %llu\n", U(a), U(o), U(n));
	nt_factor(o, &f);
	printf("  how: the order divides lambda(%llu) = %llu; %llu^%llu = 1",
	    U(n), U(lam), U(a), U(o));
	for (i = 0; i < f.k; i++)
		printf("%s %llu^(%llu/%llu) = %llu", i ? "," : " and", U(a), U(o),
		    U(f.p[i]), U(nt_powmod(a, o / f.p[i], n)));
	printf("\n");
	if (o == ph)
		printf("  %llu is a primitive root mod %llu (its order is phi(%llu) "
		    "= %llu)\n", U(a), U(n), U(n), U(ph));
	else
		printf("  not a primitive root: phi(%llu) = %llu%s\n", U(n), U(ph),
		    nt_has_primroot(n) ? "" : " (and there is none: the unit group "
		    "is not cyclic)");
	printf("  additive order: %llu\n", U(n / nt_gcd(a, n)));
	if (o <= (1u << 22)) {
		x = a % n;
		for (k = 1; x != 1 % n; k++)
			x = nt_mulmod(x, a, n);
		printf("  check: stepping through the powers, %llu^k = 1 first at "
		    "k = %llu\n", U(a), U(k));
	} else
		printf("  check: %llu^%llu = %llu by repeated squaring\n", U(a),
		    U(o), U(nt_powmod(a, o, n)));
}

static void
cmd_phi(int lambda)
{
	i128 v = iexpr_sum();
	struct nt_fac f;
	u64 n, r;
	int i;

	if (!at_end())
		rfail("unexpected %s", here());
	if (v < 1 || (v >> 64))
		rfail("%s(n) for 1 <= n < 2^64", lambda ? "lambda" : "phi");
	n = (u64)v;
	nt_factor(n, &f);
	if (!lambda) {
		r = nt_phi(n);
		printf("phi(%llu) = %llu\n", U(n), U(r));
		printf("  how: %llu = %s; phi = product of p^(e-1) (p - 1):", U(n),
		    fac_text(n));
		for (i = 0; i < f.k; i++) {
			printf(" %s", i ? "* " : "");
			if (f.e[i] > 1)
				printf("%llu^%d*", U(f.p[i]), f.e[i] - 1);
			printf("%llu", U(f.p[i] - 1));
		}
		if (n == 1)
			printf(" 1 (the empty product)");
		printf("\n");
		printf("  the order of the unit group (Z/%llu)*; lambda(%llu) = %llu "
		    "is its exponent\n", U(n), U(n), U(nt_lambda(n)));
		if (n <= (1u << 24)) {
			u64 k, c = 0;
			for (k = 1; k <= n; k++)
				c += nt_gcd(k, n) == 1;
			printf("  check: counted the k in 1..%llu with gcd(k, %llu) = 1: "
			    "%llu\n", U(n), U(n), U(c));
		}
		return;
	}
	r = nt_lambda(n);
	printf("lambda(%llu) = %llu (Carmichael: the least m with a^m = 1 for "
	    "every unit a)\n", U(n), U(r));
	printf("  how: %llu = %s; lambda is the lcm of", U(n), fac_text(n));
	for (i = 0; i < f.k; i++) {
		u64 p = f.p[i];
		int e = f.e[i];
		if (p == 2)
			printf(" %s%llu", i ? "" : "", U(e == 1 ? 1 : e == 2 ? 2 :
			    upow(2, e - 2)));
		else
			printf(" %llu", U(upow(p, e - 1) * (p - 1)));
		printf(" (%llu^%d)%s", U(p), e, i + 1 < f.k ? "," : "");
	}
	printf("%s\n", f.k ? "" : " nothing: 1");
	printf("  lambda(2^e) = 2^(e-2) for e >= 3; lambda(p^e) = phi(p^e) "
	    "otherwise; phi(%llu) = %llu\n", U(n), U(nt_phi(n)));
	if (n <= (1u << 14)) {
		u64 k, m = 1;
		for (k = 1; k < n; k++)
			if (nt_gcd(k, n) == 1) {
				u64 o = nt_order(k, n), c = 1, x = k;
				while (x != 1 % n) {
					x = nt_mulmod(x, k, n);
					c++;
				}
				(void)o;
				m = m / nt_gcd(m, c) * c;
			}
		printf("  check: the lcm of the orders of all units, by stepping "
		    "through their powers: %llu\n", U(m));
	}
}

static void
cmd_factorint(void)
{
	i128 v = iexpr_sum();
	struct nt_fac f;
	u128 prod = 1;
	int i;

	if (!at_end())
		rfail("unexpected %s", here());
	if (v < 1 || (v >> 64))
		rfail("factorint n for 1 <= n < 2^64");
	nt_factor((u64)v, &f);
	printf("%s = %s%s\n", i128s(v), fac_text((u64)v), f.k == 1 &&
	    f.e[0] == 1 ? " (prime)" : "");
	printf("  how: trial division by the primes below 50, then Pollard-Brent "
	    "rho; primality by Miller-Rabin with the first 12 prime bases "
	    "(deterministic below 2^64)\n");
	for (i = 0; i < f.k; i++)
		prod *= (u128)upow(f.p[i], f.e[i]);
	printf("  check: the product is %s; each factor is prime\n", u128s(prod));
}

/* ---- Legendre and Jacobi symbols, square roots ---- */

static void
cmd_symbol(int legendre)
{
	i128 a0, n0;
	u64 a, n;
	int j;

	a0 = iexpr_sum();
	acc_sym(',') || acc_sym('|') || acc_sym('/') || acc_id("mod") ||
	    acc_id("over");
	n0 = iexpr_sum();
	if (!at_end())
		rfail("unexpected %s", here());
	if (n0 < 3 || !(n0 & 1) || (n0 >> 64))
		rfail("the bottom of a %s symbol is an odd number >= 3",
		    legendre ? "Legendre" : "Jacobi");
	n = (u64)n0;
	a = (u64)(((a0 % n0) + n0) % n0);
	if (legendre && !nt_isprime(n)) {
		note("%llu is not prime: the Legendre symbol is not defined; this "
		    "is the Jacobi symbol", U(n));
		legendre = 0;
	}
	j = nt_jacobi(a, n);
	printf("(%s/%llu) = %d\n", i128s(a0), U(n), j);
	if (nt_isprime(n)) {
		u64 e = nt_powmod(a, (n - 1) / 2, n), r;
		printf("  %s\n", j == 0 ? "the top is divisible by the bottom" :
		    j == 1 ? "a quadratic residue: x^2 = a has a solution" :
		    "a quadratic non-residue: x^2 = a has no solution");
		if (j == 1 && nt_sqrt_prime(a, n, &r))
			printf("  (%llu^2 = %llu mod %llu)\n", U(r), U(a), U(n));
		printf("  how: quadratic reciprocity (the binary Jacobi algorithm)\n");
		printf("  check: Euler's criterion %llu^((%llu-1)/2) = %lld mod %llu\n",
		    U(a), U(n), e == n - 1 ? -1LL : (long long)e, U(n));
	} else {
		u64 roots[4];
		int k = nt_sqrt_all(a, n, roots, 4);
		printf("  how: quadratic reciprocity; the product of the Legendre "
		    "symbols over %llu = %s\n", U(n), fac_text(n));
		printf("  %llu is composite: (a/n) = 1 does not make a a square mod n"
		    "; here %llu %s a square mod %llu\n", U(n), U(a), k ? "is" :
		    "is not", U(n));
	}
}

static void
cmd_sqrtmod_int(i128 a0, u64 n)
{
	u64 a = (u64)(((a0 % (i128)n) + (i128)n) % (i128)n), roots[256];
	int k, i;

	k = nt_sqrt_all(a, n, roots, 256);
	if (k < 0) {
		printf("x^2 = %llu mod %llu has more than 256 solutions\n", U(a),
		    U(n));
		return;
	}
	printf("x^2 = %llu mod %llu: ", U(a), U(n));
	if (k == 0)
		printf("no solution (%llu is not a square mod %llu)\n", U(a), U(n));
	else {
		printf("x = ");
		for (i = 0; i < k && i < 32; i++)
			printf("%s%llu", i ? ", " : "", U(roots[i]));
		if (k > 32)
			printf(", ... (%d in all)", k);
		printf("\n");
	}
	if (nt_isprime(n))
		printf("  how: %s\n", n == 2 || a == 0 ? "directly" : n % 4 == 3 ?
		    "p = 3 mod 4: x = a^((p+1)/4)" : "Tonelli-Shanks");
	else
		printf("  how: %llu = %s: roots mod each prime (Tonelli-Shanks), "
		    "lifted to the prime powers (Hensel), combined by CRT\n", U(n),
		    fac_text(n));
	if (k > 0) {
		printf("  check:");
		for (i = 0; i < k && i < 4; i++)
			printf(" %llu^2 = %llu%s", U(roots[i]), U(nt_mulmod(roots[i],
			    roots[i], n)), i + 1 < k && i < 3 ? "," : "");
		printf("%s (mod %llu)", k > 4 ? ", ..." : "", U(n));
	} else if (nt_isprime(n) && n > 2)
		printf("  check: Euler's criterion %llu^((%llu-1)/2) = %llu != 1",
		    U(a), U(n), U(nt_powmod(a, (n - 1) / 2, n)));
	else
		printf("  check:");
	if (n <= (1u << 22)) {
		u64 x, c = 0;
		for (x = 0; x < n; x++)
			c += nt_mulmod(x, x, n) == a;
		printf("; brute force over 0..%llu: %llu solution%s", U(n - 1), U(c),
		    c == 1 ? "" : "s");
	}
	printf("\n");
}

static void
cmd_qr(u64 n)
{
	u64 *seen, x, c = 0, list[64];
	int k = 0;

	if (n > (1u << 24))
		rfail("qr: n up to 2^24");
	seen = xcalloc(n, 1);
	for (x = 0; x < n; x++)
		seen[nt_mulmod(x, x, n)] = 1;
	for (x = 0; x < n; x++)
		if (seen[x]) {
			if (k < 64)
				list[k++] = x;
			c++;
		}
	printf("the squares mod %llu: %llu of %llu residues\n", U(n), U(c), U(n));
	print_u64s("", list, (u64)k, c, 64);
	if (nt_isprime(n) && n > 2)
		printf("  %llu is prime: the nonzero squares are (p-1)/2 = %llu, "
		    "the a with (a/p) = 1\n", U(n), U((n - 1) / 2));
	printf("  how: x^2 mod %llu for every x\n", U(n));
	free(seen);
}

/* ---- primitive roots and discrete logs ---- */

static void
cmd_primroot_int(u64 n, int test, u64 a)
{
	u64 ph = nt_phi(n), g, list[32], c = 0, x;
	struct nt_fac f;
	int i, k = 0;

	if (!nt_has_primroot(n)) {
		printf("there is no primitive root mod %llu: the unit group has "
		    "exponent lambda(%llu) = %llu < phi(%llu) = %llu, so it is not "
		    "cyclic\n", U(n), U(n), U(nt_lambda(n)), U(n), U(ph));
		printf("  primitive roots exist only mod 1, 2, 4, p^k and 2p^k (p an "
		    "odd prime)\n");
		if (test)
			printf("  ord(%llu) = %llu\n", U(a % n), U(nt_order(a, n)));
		return;
	}
	nt_factor(ph, &f);
	if (test) {
		u64 o = nt_order(a, n);
		a %= n;
		printf("%s: %llu is %sa primitive root mod %llu (its order is %llu, "
		    "phi(%llu) = %llu)\n", o == ph ? "yes" : "no", U(a),
		    o == ph ? "" : "not ", U(n), U(o), U(n), U(ph));
		if (o) {
			printf("  how:");
			for (i = 0; i < f.k; i++)
				printf("%s %llu^(%llu/%llu) = %llu", i ? "," : "", U(a),
				    U(ph), U(f.p[i]), U(nt_powmod(a, ph / f.p[i], n)));
			printf(" (a primitive root has none of these equal to 1)\n");
		}
		g = a;
	} else {
		g = nt_primroot(n);
		printf("the smallest primitive root mod %llu is %llu\n", U(n), U(g));
		printf("  how: (Z/%llu)* is cyclic of order phi = %llu = %s; g "
		    "generates it iff g^(%llu/p) != 1 for each prime p | %llu\n",
		    U(n), U(ph), fac_text(ph), U(ph), U(ph));
		if (ph <= (1u << 20)) {
			for (x = 1; x < n; x++)
				if (nt_gcd(x, n) == 1 && nt_order(x, n) == ph) {
					if (k < 32)
						list[k++] = x;
					c++;
				}
			printf("  phi(%llu) = %llu primitive roots:", U(ph),
			    U(nt_phi(ph)));
			for (i = 0; i < k; i++)
				printf(" %llu", U(list[i]));
			printf("%s\n", c > (u64)k ? " ..." : "");
		} else
			printf("  there are phi(%llu) = %llu primitive roots\n", U(ph),
			    U(nt_phi(ph)));
	}
	if (nt_order(g, n) == ph && ph <= (1u << 22)) {
		u64 cnt = 0, e = 1;
		uint8_t *seen = xcalloc(n, 1);
		for (x = 0; x < ph; x++) {
			if (!seen[e])
				cnt++;
			seen[e] = 1;
			e = nt_mulmod(e, g, n);
		}
		printf("  check: the powers %llu^0 .. %llu^%llu are %llu distinct "
		    "units\n", U(g), U(g), U(ph - 1), U(cnt));
		free(seen);
	}
}

static void
cmd_dlog_ring(const struct rspec *s, u64 a, u64 g, int gauto)
{
	const struct ring *R = &s->R;
	u64 x, ord;
	int r;

	if (gauto)
		note("no base given: the generator %s (base 2 or the smallest "
		    "primitive root)", el_code(R, g));
	r = r_dlog(R, g, a, &x, &ord);
	if (r < 0)
		rfail("the order of %s has a prime factor too large for "
		    "baby-step giant-step", el_code(R, g));
	if (r == 0) {
		if (ord == 0)
			printf("%s is not a unit in %s: it has no powers equal to 1\n",
			    el_code(R, g), s->name);
		else
			printf("no x: %s is not a power of %s in %s (%s generates "
			    "%llu elements; %s^%llu = %s != 1)\n", el_code(R, a),
			    el_code(R, g), s->name, el_code(R, g), U(ord), el_code(R, a),
			    U(ord), el_code(R, r_pow(R, a, ord)));
		return;
	}
	printf("%s^x = %s in %s: x = %llu (mod %llu, the order of %s)\n",
	    el_code(R, g), el_code(R, a), s->name, U(x), U(ord), el_code(R, g));
	printf("  how: Pohlig-Hellman over %llu = %s, baby-step giant-step for "
	    "each prime\n", U(ord), fac_text(ord));
	printf("  check: %s^%llu = %s\n", el_code(R, g), U(x), el_code(R,
	    r_pow(R, g, x)));
}

/* ---- linear and polynomial congruences, CRT ---- */

#define MAXCLS 4096
struct cls {
	u64 m;        /* modulus */
	int n;        /* residues */
	u64 *r;
};

static void
cls_free(struct cls *c)
{
	free(c->r);
	c->r = NULL;
}

static int
cmpu(const void *a, const void *b)
{
	u64 x = *(const u64 *)a, y = *(const u64 *)b;

	return x < y ? -1 : x > y;
}

/* S and T: every pair by CRT, modulo lcm */
static int
cls_meet(struct cls *a, const struct cls *b)
{
	struct cls c;
	int i, j, r;
	u64 m = 0;

	c.r = xcalloc(MAXCLS, sizeof(u64));
	c.n = 0;
	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++) {
			u64 x;
			r = nt_crt(a->r[i], a->m, b->r[j], b->m, &x, &m);
			if (r < 0) {
				free(c.r);
				return -1;
			}
			if (r == 0)
				continue;
			if (c.n >= MAXCLS) {
				free(c.r);
				return -2;
			}
			c.r[c.n++] = x;
		}
	if (m == 0) {
		/* no pair met: the modulus is still the lcm */
		u64 g = nt_gcd(a->m, b->m);
		u128 l = (u128)(a->m / g) * b->m;
		m = l >> 64 ? 0 : (u64)l;
	}
	qsort(c.r, (size_t)c.n, sizeof(u64), cmpu);
	cls_free(a);
	a->r = c.r;
	a->n = c.n;
	a->m = m;
	return 0;
}

/* P(x) = 0 mod m: the solutions mod m (or mod m/g for linear) */
static void
solve_poly(const struct rpoly *P, u64 m, struct cls *out, struct sbuf *how)
{
	struct ring R;

	ring_zn(&R, m);
	out->r = xcalloc(MAXCLS, sizeof(u64));
	out->n = 0;
	out->m = m;
	if (P->d < 0) {
		out->m = 1;
		out->r[out->n++] = 0;
		sb_printf(how, "every x (the equation is 0 = 0 mod %llu)", U(m));
		return;
	}
	if (P->d == 0) {
		sb_printf(how, "no x: it says %llu = 0 mod %llu", U(P->c[0]), U(m));
		return;
	}
	if (P->d == 1) {
		u64 a = P->c[1], b = nt_submod(0, P->c[0], m), g, mg, inv, x0;
		nt_inv(a, m, &g);
		if (b % g) {
			sb_printf(how, "%llux = %llu: gcd(%llu, %llu) = %llu does not "
			    "divide %llu, so no x", U(a), U(b), U(a), U(m), U(g), U(b));
			return;
		}
		mg = m / g;
		inv = nt_inv((a / g) % mg, mg, NULL);
		x0 = nt_mulmod((b / g) % mg, inv, mg ? mg : 1);
		if (a == 1)
			sb_printf(how, "x = %llu", U(b));
		else if (g == 1)
			sb_printf(how, "%llux = %llu: x = %llu * %llu^-1 = %llu * %llu "
			    "= %llu", U(a), U(b), U(b), U(a), U(b), U(inv), U(x0));
		else
			sb_printf(how, "%llux = %llu: gcd(%llu, %llu) = %llu divides "
			    "%llu: %llux = %llu mod %llu, x = %llu (%llu solutions mod "
			    "%llu)", U(a), U(b), U(a), U(m), U(g), U(b), U(a / g),
			    U(b / g), U(mg), U(x0), U(g), U(m));
		out->m = mg;
		out->r[out->n++] = x0;
		return;
	}
	if (P->d == 2 && P->c[1] == 0 && nt_gcd(P->c[2], m) == 1) {
		/* x^2 = c */
		u64 inv = nt_inv(P->c[2], m, NULL);
		u64 c = nt_mulmod(nt_submod(0, P->c[0], m), inv, m);
		int k = nt_sqrt_all(c, m, out->r, MAXCLS);
		if (k < 0)
			rfail("more than %d solutions", MAXCLS);
		out->n = k;
		sb_printf(how, "x^2 = %llu: square roots mod %llu (Tonelli-Shanks, "
		    "Hensel, CRT)", U(c), U(m));
		return;
	}
	if (m <= (1u << 22)) {
		u64 x;
		for (x = 0; x < m; x++)
			if (rp_eval(&R, P, x) == 0) {
				if (out->n >= MAXCLS)
					rfail("more than %d solutions", MAXCLS);
				out->r[out->n++] = x;
			}
		sb_printf(how, "degree %d: tried every x mod %llu", P->d, U(m));
		return;
	}
	{
		/* by prime powers, each searched, then CRT */
		struct nt_fac f;
		struct cls acc = {1, 1, NULL};
		int i;
		nt_factor(m, &f);
		acc.r = xcalloc(MAXCLS, sizeof(u64));
		acc.r[0] = 0;
		for (i = 0; i < f.k; i++) {
			u64 pe = upow(f.p[i], f.e[i]), x;
			struct ring Rp;
			struct cls c;
			struct rpoly Q;
			int j;
			if (pe > (1u << 22))
				rfail("degree %d modulo %llu: a prime power factor %llu is "
				    "too large to search", P->d, U(m), U(pe));
			ring_zn(&Rp, pe);
			Q = *P;
			for (j = 0; j <= Q.d; j++)
				Q.c[j] %= pe;
			rp_norm(&Q);
			c.m = pe;
			c.n = 0;
			c.r = xcalloc(MAXCLS, sizeof(u64));
			for (x = 0; x < pe; x++)
				if (rp_eval(&Rp, &Q, x) == 0 && c.n < MAXCLS)
					c.r[c.n++] = x;
			if (cls_meet(&acc, &c) < 0)
				rfail("too many solutions");
			cls_free(&c);
		}
		free(out->r);
		*out = acc;
		sb_printf(how, "degree %d: searched mod each prime power of %llu = "
		    "%s, combined by CRT", P->d, U(m), fac_text(m));
	}
}

/* one equation [a, b): LHS = RHS mod m, or "r mod m" */
struct ceq {
	struct rpoly P;
	u64 m;
	int a, b;
};

static void
parse_congruence(int a, int b, struct ceq *e, char *var)
{
	int i, eq = -1, md = -1, depth = 0, par = 0;
	struct ring R;
	struct rpoly L, Rh;

	for (i = a; i < b; i++) {
		if (rt[i].k == RK_SYM && rt[i].s[0] == '(') {
			if (i + 1 < b && rt[i + 1].k == RK_ID &&
			    (!strcasecmp(rt[i + 1].s, "mod") ||
			    !strcasecmp(rt[i + 1].s, "modulo")) && depth == 0) {
				md = i + 1;
				par = 1;
				break;
			}
			depth++;
		}
		if (rt[i].k == RK_SYM && rt[i].s[0] == ')')
			depth--;
		if (depth == 0 && rt[i].k == RK_SYM && rt[i].s[0] == '=' && eq < 0)
			eq = i;
		if (depth == 0 && rt[i].k == RK_ID && (!strcasecmp(rt[i].s, "mod") ||
		    !strcasecmp(rt[i].s, "modulo")))
			md = i;
		if (rt[i].k == RK_ID && !var[0] && strcasecmp(rt[i].s, "mod") &&
		    strcasecmp(rt[i].s, "modulo") && strlen(rt[i].s) <= 2)
			snprintf(var, 8, "%s", rt[i].s);
	}
	if (md < 0)
		rfail("a congruence needs a modulus: A = B mod m");
	rti = md + 1;
	rlim = par ? b - 1 : b;
	e->m = modulus_arg();
	if (par)
		expect_sym(')');
	if (rti != rlim)
		rfail("unexpected %s after the modulus", here());
	if (!var[0])
		snprintf(var, 8, "x");
	ring_zn(&R, e->m);
	if (eq < 0) {
		/* "r mod m": x = r */
		rti = a;
		rlim = par ? md - 1 : md;
		pexpr_into(&R, &Rh, var);
		if (!at_end())
			rfail("unexpected %s", here());
		rp_x(&L);
	} else {
		rti = a;
		rlim = eq;
		pexpr_into(&R, &L, var);
		if (!at_end())
			rfail("unexpected %s", here());
		rti = eq + 1;
		rlim = par ? md - 1 : md;
		pexpr_into(&R, &Rh, var);
		if (!at_end())
			rfail("unexpected %s", here());
	}
	rp_sub(&R, &e->P, &L, &Rh);
	e->a = a;
	e->b = b;
}

static const char *
eq_text(int a, int b)
{
	static char buf[4][256];
	static int k;
	char *s = buf[k++ & 3];
	size_t o = 0;
	int i;

	s[0] = 0;
	for (i = a; i < b && o < 200; i++) {
		const struct rtok *t = &rt[i];
		int sp = i > a && !(t->k == RK_SYM && strchr(")^,", t->s[0])) &&
		    !(t->k == RK_SYM && t->s[0] == '(' && rt[i - 1].k == RK_ID) &&
		    !(rt[i - 1].k == RK_SYM && rt[i - 1].s[0] == '-' && (i - 1 == a ||
		    (rt[i - 2].k == RK_SYM && strchr("^(=,*/", rt[i - 2].s[0])))) &&
		    !(rt[i - 1].k == RK_SYM && strchr("(^", rt[i - 1].s[0])) &&
		    !(t->k == RK_ID && rt[i - 1].k == RK_NUM && strlen(t->s) <= 2);
		if (t->k == RK_NUM)
			o += (size_t)snprintf(s + o, 256 - o, "%s%s", sp ? " " : "",
			    t->base == 16 ? u128x(t->v) : u128s(t->v));
		else
			o += (size_t)snprintf(s + o, 256 - o, "%s%s", sp ? " " : "",
			    t->s);
	}
	return s;
}

static void
cmd_congruence(void)
{
	struct ceq eq[16];
	struct cls acc, c;
	char var[8] = "";
	int ne = 0, a = rti, i, j, depth = 0, lim = rlim;
	struct sbuf how = {0};

	/* split at top-level , ; "and" */
	for (i = a; i <= lim; i++) {
		int brk = i == lim;
		if (!brk && rt[i].k == RK_SYM && rt[i].s[0] == '(')
			depth++;
		if (!brk && rt[i].k == RK_SYM && rt[i].s[0] == ')')
			depth--;
		if (!brk && depth == 0 && ((rt[i].k == RK_SYM && (rt[i].s[0] == ',' ||
		    rt[i].s[0] == ';')) || (rt[i].k == RK_ID &&
		    !strcasecmp(rt[i].s, "and"))))
			brk = 1;
		if (brk) {
			if (i > a) {
				if (ne == 16)
					rfail("at most 16 congruences");
				parse_congruence(a, i, &eq[ne++], var);
			}
			a = i + 1;
		}
	}
	if (ne == 0)
		rfail("congruence A = B mod m [, ...]");
	acc.m = 1;
	acc.n = 1;
	acc.r = xcalloc(MAXCLS, sizeof(u64));
	acc.r[0] = 0;
	printf("%s", ne > 1 ? "the system\n" : "");
	for (i = 0; i < ne; i++) {
		sb_clear(&how);
		sb_puts(&how, "");
		solve_poly(&eq[i].P, eq[i].m, &c, &how);
		printf("  %s: %s\n", eq_text(eq[i].a, eq[i].b), how.p);
		if (ne > 1) {
			printf("    %s", c.n ? "" : "no solution\n");
			for (j = 0; j < c.n && j < 16; j++)
				printf("%s%s = %llu", j ? ", " : "", var, U(c.r[j]));
			if (c.n > 16)
				printf(", ... (%d)", c.n);
			if (c.n)
				printf(" (mod %llu)\n", U(c.m));
		}
		if (cls_meet(&acc, &c) < 0)
			rfail("the combined modulus is too large (or too many "
			    "solutions)");
		cls_free(&c);
	}
	if (acc.n == 0)
		printf("no solution%s\n", ne > 1 ? ": the congruences contradict "
		    "each other (CRT: the residues differ mod the gcd of the "
		    "moduli)" : "");
	else {
		printf("%s = ", var);
		for (j = 0; j < acc.n && j < 32; j++)
			printf("%s%llu", j ? ", " : "", U(acc.r[j]));
		if (acc.n > 32)
			printf(", ... (%d in all)", acc.n);
		printf(" (mod %llu)\n", U(acc.m));
		if (ne > 1)
			printf("  how: the Chinese remainder theorem, pairwise (moduli "
			    "that share factors must agree mod their gcd)\n");
		if (ne == 1 && acc.m != eq[0].m && acc.m && eq[0].m / acc.m <= 16 &&
		    acc.n == 1) {
			u64 x;
			printf("  that is, mod %llu: %s =", U(eq[0].m), var);
			for (x = acc.r[0]; x < eq[0].m; x += acc.m)
				printf(" %llu", U(x));
			printf("\n");
		}
	}
	/* check */
	{
		int bad = 0;
		for (j = 0; j < acc.n; j++)
			for (i = 0; i < ne; i++) {
				struct ring R;
				ring_zn(&R, eq[i].m);
				if (rp_eval(&R, &eq[i].P, acc.r[j] % eq[i].m) != 0)
					bad++;
			}
		if (acc.n)
			printf("  check: %s substituted into %s: %s", acc.n == 1 ?
			    "the solution" : "each solution", ne == 1 ?
			    "the congruence" : "every congruence", bad ? "FAILED" :
			    "it holds");
		else
			printf("  check:");
		if (acc.m <= (1u << 22) && acc.m > 0) {
			u64 x, cnt = 0;
			for (x = 0; x < acc.m; x++) {
				for (i = 0; i < ne; i++) {
					struct ring R;
					ring_zn(&R, eq[i].m);
					if (rp_eval(&R, &eq[i].P, x % eq[i].m))
						break;
				}
				cnt += i == ne;
			}
			printf("; brute force over 0..%llu: %llu solution%s", U(acc.m - 1),
			    U(cnt), cnt == 1 ? "" : "s");
		}
		printf("\n");
	}
	cls_free(&acc);
	free(how.p);
}

/* "crt 2 mod 3, 3 mod 5" or "crt 2, 3 mod 3, 5" (residues, moduli) */
static void
cmd_crt(void)
{
	int i, hasmod = 0, haseq = 0;

	for (i = rti; i < rlim; i++) {
		if (rt[i].k == RK_ID && (!strcasecmp(rt[i].s, "mod") ||
		    !strcasecmp(rt[i].s, "modulo")))
			hasmod++;
		if (rt[i].k == RK_SYM && rt[i].s[0] == '=')
			haseq++;
	}
	cmd_congruence();
	(void)hasmod;
	(void)haseq;
}
