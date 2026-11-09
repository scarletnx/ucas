/*
 * Rings and finite fields: the commands.  A command's arguments are taken
 * as text (the rest of the statement) and read here with a small scanner,
 * so ring syntax (Z/12, GF(2^8, aes), 3x = 4 mod 7, [[1,2],[3,4]]) does
 * not touch the expression grammar.  Each answer says how it was found
 * and is checked another way: by brute force over small rings, by
 * multiplying inverses back, by a second multiplication algorithm, by
 * multiplying factors together and testing each with Rabin's test.
 */
#include "ring.h"
#include "ucas.h"
#include "calc.h"

#include <ctype.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* notes (from parsing) come before the answer */
#define printf(...) (notes_flush(), printf(__VA_ARGS__))
#undef putchar
#define putchar(c) (notes_flush(), putchar(c))

/* ---- errors: fail(), or quietly while trying a reading ---- */

static int spec_depth;
static jmp_buf spec_jmp;

static void
rfail(const char *fmt, ...)
{
	char buf[512];
	va_list ap;

	if (spec_depth)
		longjmp(spec_jmp, 1);
	va_start(ap, fmt);
	vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	fail("%s", buf);
}

/* ---- the scanner ---- */

enum { RK_END, RK_NUM, RK_ID, RK_SYM };
struct rtok {
	int k;
	u128 v;
	int big;        /* the literal does not fit 128 bits */
	int base;       /* 10, 16 or 2 */
	char s[40];
};
#define MAXRT 2048
static struct rtok rt[MAXRT + 1];
static int nrt, rti, rlim;

static void
rtok_push(int k, const char *s, size_t n)
{
	struct rtok *t;

	if (nrt >= MAXRT)
		rfail("that is too long for me");
	t = &rt[nrt++];
	memset(t, 0, sizeof *t);
	t->k = k;
	if (n >= sizeof t->s)
		n = sizeof t->s - 1;
	memcpy(t->s, s, n);
	t->s[n] = 0;
}

static void
rtokenize(const char *s)
{
	nrt = 0;
	while (*s) {
		const unsigned char *u = (const unsigned char *)s;
		if (isspace(*u)) {
			s++;
			continue;
		}
		if (isdigit(*u)) {
			struct rtok *t;
			int base = 10, d;
			u128 v = 0;
			rtok_push(RK_NUM, "", 0);
			t = &rt[nrt - 1];
			if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X') &&
			    isxdigit((unsigned char)s[2])) {
				base = 16;
				s += 2;
			} else if (s[0] == '0' && (s[1] == 'b' || s[1] == 'B') &&
			    (s[2] == '0' || s[2] == '1')) {
				base = 2;
				s += 2;
			}
			for (;; s++) {
				if (isdigit((unsigned char)*s))
					d = *s - '0';
				else if (base == 16 && isxdigit((unsigned char)*s))
					d = tolower((unsigned char)*s) - 'a' + 10;
				else if (*s == '_' && isxdigit((unsigned char)s[1]))
					continue;
				else
					break;
				if (d >= base)
					break;
				if (v > (~(u128)0 - (u128)d) / (u128)base)
					t->big = 1;
				v = v * (u128)base + (u128)d;
			}
			t->v = v;
			t->base = base;
			continue;
		}
		if (isalpha(*u) || *u == '_') {
			const char *b = s;
			while (isalnum((unsigned char)*s) || *s == '_')
				s++;
			rtok_push(RK_ID, b, (size_t)(s - b));
			continue;
		}
		if (u[0] == 0xe2 && u[1] == 0x89 && u[2] == 0xa1) {   /* ≡ */
			rtok_push(RK_SYM, "=", 1);
			s += 3;
		} else if (u[0] == 0xe2 && u[1] == 0x88 && u[2] == 0x9a) {  /* √ */
			rtok_push(RK_ID, "sqrt", 4);
			s += 3;
		} else if (u[0] == 0xce && u[1] == 0xb1) {   /* α */
			rtok_push(RK_ID, "a", 1);
			s += 2;
		} else if ((u[0] == 0xc2 && u[1] == 0xb7) ||
		    (u[0] == 0xc3 && u[1] == 0x97)) {     /* · × */
			rtok_push(RK_SYM, "*", 1);
			s += 2;
		} else if (u[0] == 0xe2 && u[1] == 0x88 && u[2] == 0x92) {  /* − */
			rtok_push(RK_SYM, "-", 1);
			s += 3;
		} else if (u[0] == 0xe2 && u[1] == 0x8a && u[2] == 0x95) {  /* ⊕ */
			rtok_push(RK_SYM, "+", 1);
			s += 3;
		} else if (u[0] == 0xe2 && u[1] == 0x84 && u[2] == 0xa4) {  /* ℤ */
			rtok_push(RK_ID, "Z", 1);
			s += 3;
		} else if (u[0] == 0xe2 && u[1] == 0x84 && u[2] == 0x9a) {  /* ℚ */
			rtok_push(RK_ID, "Q", 1);
			s += 3;
		} else if (u[0] == 0xc2 && (u[1] == 0xb2 || u[1] == 0xb3)) { /* ² ³ */
			rtok_push(RK_SYM, "^", 1);
			rtok_push(RK_NUM, "", 0);
			rt[nrt - 1].v = u[1] == 0xb2 ? 2 : 3;
			rt[nrt - 1].base = 10;
			s += 2;
		} else if (s[0] == '*' && s[1] == '*') {
			rtok_push(RK_SYM, "^", 1);
			s += 2;
		} else if (s[0] == '=' && s[1] == '=') {
			rtok_push(RK_SYM, "=", 1);
			s += 2;
		} else if (*u >= 0x80) {
			rfail("unexpected character in \"%s\"", s);
		} else {
			rtok_push(RK_SYM, s, 1);
			s++;
		}
	}
	rt[nrt].k = RK_END;
	rt[nrt].s[0] = 0;
	rti = 0;
	rlim = nrt;
}

static int
at_end(void)
{
	return rti >= rlim;
}

static int
at_sym(int c)
{
	return rti < rlim && rt[rti].k == RK_SYM && rt[rti].s[0] == c;
}

static int
acc_sym(int c)
{
	if (at_sym(c)) {
		rti++;
		return 1;
	}
	return 0;
}

static int
at_id(const char *w)
{
	return rti < rlim && rt[rti].k == RK_ID && !strcasecmp(rt[rti].s, w);
}

static int
acc_id(const char *w)
{
	if (at_id(w)) {
		rti++;
		return 1;
	}
	return 0;
}

static const char *
here(void)
{
	return rti < rlim ? (rt[rti].k == RK_NUM ? "a number" : rt[rti].s) :
	    "the end";
}

static void
expect_sym(int c)
{
	if (!acc_sym(c))
		rfail("expected '%c', found %s", c, here());
}

/* ---- numbers ---- */

static const char *
u128s(u128 v)
{
	static char buf[8][48];
	static int k;
	char *b = buf[k++ & 7], t[48];
	int n = 0, i;

	do {
		t[n++] = (char)('0' + (int)(v % 10));
		v /= 10;
	} while (v);
	for (i = 0; i < n; i++)
		b[i] = t[n - 1 - i];
	b[n] = 0;
	return b;
}

static const char *
u128x(u128 v)
{
	static char buf[8][48];
	static int k;
	char *b = buf[k++ & 7];

	if (v >> 64)
		snprintf(b, 48, "0x%llx%016llx", (unsigned long long)(u64)(v >> 64),
		    (unsigned long long)(u64)v);
	else
		snprintf(b, 48, "0x%llx", (unsigned long long)(u64)v);
	return b;
}

static const char *
i128s(i128 v)
{
	static char buf[4][50];
	static int k;
	char *b = buf[k++ & 3];

	if (v < 0)
		snprintf(b, 50, "-%s", u128s((u128)(-v)));
	else
		snprintf(b, 50, "%s", u128s((u128)v));
	return b;
}

#define U(v) ((unsigned long long)(v))

/* integer expressions: + - * / % ^, parentheses, in 127 bits */
static i128 iexpr_sum(void);

static i128
iexpr_atom(void)
{
	i128 v;

	if (acc_sym('(')) {
		v = iexpr_sum();
		expect_sym(')');
		return v;
	}
	if (rti < rlim && rt[rti].k == RK_NUM) {
		if (rt[rti].big || rt[rti].v >> 127)
			rfail("%s is too large", u128s(rt[rti].v));
		return (i128)rt[rti++].v;
	}
	rfail("expected a number, found %s", here());
	return 0;
}

static i128 iexpr_unary(void);

static i128
iexpr_pow(void)
{
	i128 b = iexpr_atom(), r = 1;
	i128 e;

	if (!acc_sym('^'))
		return b;
	e = iexpr_unary();
	if (e < 0)
		rfail("a negative exponent in an integer");
	while (e-- > 0)
		if (__builtin_mul_overflow(r, b, &r))
			rfail("the integer is too large");
	return r;
}

static i128
iexpr_unary(void)
{
	if (acc_sym('-'))
		return -iexpr_unary();
	if (acc_sym('+'))
		return iexpr_unary();
	return iexpr_pow();
}

static i128
iexpr_term(void)
{
	i128 v = iexpr_unary(), w;

	for (;;) {
		if (acc_sym('*')) {
			if (__builtin_mul_overflow(v, iexpr_unary(), &v))
				rfail("the integer is too large");
		} else if (acc_sym('/')) {
			if ((w = iexpr_unary()) == 0)
				rfail("division by zero");
			v /= w;
		} else if (acc_sym('%')) {
			if ((w = iexpr_unary()) == 0)
				rfail("division by zero");
			v %= w;
		} else
			return v;
	}
}

static i128
iexpr_sum(void)
{
	i128 v = iexpr_term();

	for (;;) {
		if (acc_sym('+')) {
			if (__builtin_add_overflow(v, iexpr_term(), &v))
				rfail("the integer is too large");
		} else if (acc_sym('-')) {
			if (__builtin_sub_overflow(v, iexpr_term(), &v))
				rfail("the integer is too large");
		} else
			return v;
	}
}

/* a modulus n >= 2 (< 2^64) */
static u64
modulus_arg(void)
{
	i128 n = iexpr_unary();

	if (n < 1)
		rfail("the modulus must be positive");
	if (n >> 64)
		rfail("moduli up to 2^64 - 1 (this one is %s)", i128s(n));
	return (u64)n;
}

static int
starts_number(void)
{
	return rti < rlim && (rt[rti].k == RK_NUM || at_sym('(') || at_sym('-'));
}

/* ---- printing elements and polynomials ---- */


static const char *
el_code(const struct ring *R, u64 a)
{
	static char buf[8][40];
	static int k;
	char *b = buf[k++ & 7];

	if (R->deg > 1 && R->n == 2)
		snprintf(b, 40, "0x%0*llx", (R->deg + 3) / 4, U(a));
	else
		snprintf(b, 40, "%llu", U(a));
	return b;
}

/* an element of an extension as a polynomial in var */
static void
el_poly(struct sbuf *s, const struct ring *R, u64 a, const char *var)
{
	u64 d[RMAXF];
	int i, first = 1;

	r_digits(R, a, d);
	for (i = R->deg - 1; i >= 0; i--) {
		if (d[i] == 0)
			continue;
		if (!first)
			sb_puts(s, " + ");
		first = 0;
		if (d[i] != 1 || i == 0)
			sb_printf(s, "%llu", U(d[i]));
		if (i > 0)
			sb_printf(s, "%s", var);
		if (i > 1)
			sb_printf(s, "^%d", i);
	}
	if (first)
		sb_puts(s, "0");
}

/* "0x57 = x^6 + x^4 + x^2 + x + 1", or "87 = ...", or a residue */
static const char *
el_str(const struct ring *R, u64 a)
{
	static struct sbuf b[6];
	static int k;
	struct sbuf *s = &b[k++ % 6];

	sb_clear(s);
	sb_puts(s, "");
	if (R->deg == 1) {
		sb_printf(s, "%llu", U(a));
		return sb_str(s);
	}
	sb_printf(s, "%s = ", el_code(R, a));
	el_poly(s, R, a, "x");
	return sb_str(s);
}

static const char *
el_pstr(const struct ring *R, u64 a, const char *var)
{
	static struct sbuf b[4];
	static int k;
	struct sbuf *s = &b[k++ & 3];

	sb_clear(s);
	sb_puts(s, "");
	el_poly(s, R, a, var);
	return sb_str(s);
}

/* a polynomial over R in var; coefficients of an extension as codes */
static void
rp_print(struct sbuf *s, const struct ring *R, const struct rpoly *p,
    const char *var)
{
	int i, first = 1;
	u64 one = r_const(R, 1);

	for (i = p->d; i >= 0; i--) {
		u64 c = p->c[i];
		if (c == 0)
			continue;
		if (!first)
			sb_puts(s, " + ");
		first = 0;
		if (c != one || i == 0) {
			sb_puts(s, el_code(R, c));
			if (i > 0 && R->deg > 1)
				sb_putc(s, '*');
		}
		if (i > 0)
			sb_puts(s, var);
		if (i > 1)
			sb_printf(s, "^%d", i);
	}
	if (first)
		sb_puts(s, "0");
}

static const char *
rp_str(const struct ring *R, const struct rpoly *p)
{
	static struct sbuf b[8];
	static int k;
	struct sbuf *s = &b[k++ & 7];

	sb_clear(s);
	sb_puts(s, "");
	rp_print(s, R, p, "x");
	return sb_str(s);
}

/* factors as (f1)^e1 (f2) ... */
static const char *
fac_str(const struct ring *R, const struct rpoly *fac, const int *mult, int k,
    u64 lead)
{
	static struct sbuf s;
	int i;

	sb_clear(&s);
	sb_puts(&s, "");
	if (lead != r_const(R, 1) || k == 0)
		sb_printf(&s, "%s", el_code(R, lead));
	for (i = 0; i < k; i++) {
		int paren = k > 1 || mult[i] > 1 || lead != r_const(R, 1);
		if (s.n)
			sb_putc(&s, paren && fac[i].d >= 1 ? ' ' : '*');
		if (paren)
			sb_putc(&s, '(');
		rp_print(&s, R, &fac[i], "x");
		if (paren)
			sb_putc(&s, ')');
		if (mult[i] > 1)
			sb_printf(&s, "^%d", mult[i]);
	}
	return sb_str(&s);
}

/* ---- ring descriptions ---- */

enum { RS_ZN = 1, RS_EXT, RS_ZI, RS_ZSQRT, RS_QX };
struct rspec {
	int kind;
	struct ring R;
	int gf;           /* written as a field GF(q) */
	int automod;      /* the modulus was chosen here */
	int64_t d;        /* Z[sqrt d] (Z[i]: -1) */
	int qd;           /* Q[x]/(f): f's degree and coefficients */
	struct q qf[RMAXF + 1];
	char name[200];
};

static const struct {
	const char *name;
	int deg;
	u128 code;
	const char *what;
} named_mod[] = {
	{"aes", 8, 0x11b, "AES (Rijndael)"},
	{"rijndael", 8, 0x11b, "AES (Rijndael)"},
	{"reedsolomon", 8, 0x11d, "Reed-Solomon (QR codes)"},
	{"rs", 8, 0x11d, "Reed-Solomon (QR codes)"},
	{"crc8", 8, 0x107, "CRC-8 (ATM)"},
	{"crc16", 16, 0x18005, "CRC-16 (IBM/ANSI)"},
	{"ccitt", 16, 0x11021, "CRC-16-CCITT"},
	{"crc16ccitt", 16, 0x11021, "CRC-16-CCITT"},
	{"crc32", 32, 0x104c11db7ULL, "CRC-32 (IEEE 802.3)"},
	{"crc32c", 32, 0x11edc6f41ULL, "CRC-32C (Castagnoli)"},
	{"crc64", 64, ((u128)1 << 64) | 0x42f0e1eba9ea3693ULL, "CRC-64 (ECMA-182)"},
	{NULL, 0, 0, NULL}
};

static int
named_index(const char *w)
{
	int k;

	for (k = 0; named_mod[k].name; k++)
		if (!strcasecmp(named_mod[k].name, w))
			return k;
	return -1;
}

static void pexpr_into(const struct ring *R, struct rpoly *p, const char *var);

/* a modulus over Z/p of degree want (0: any): a polynomial, a code, a name */
static int
parse_modulus(u64 p, int want, u64 *f, int *deg, const char **fname)
{
	struct ring B;
	struct rpoly m;
	int k, i;

	*fname = NULL;
	if (rti < rlim && rt[rti].k == RK_ID && (k = named_index(rt[rti].s)) >= 0) {
		if (p != 2)
			rfail("%s is a polynomial over GF(2)", named_mod[k].name);
		rti++;
		acc_id("polynomial");
		*deg = named_mod[k].deg;
		for (i = 0; i <= *deg; i++)
			f[i] = (u64)(named_mod[k].code >> i & 1);
		*fname = named_mod[k].name;
		return 1;
	}
	/* a lone integer: the code, digits base p (bits for p = 2) */
	if (rti < rlim && rt[rti].k == RK_NUM && (rti + 1 >= rlim ||
	    at_end() || rt[rti + 1].k == RK_SYM) && !(rti + 1 < rlim &&
	    rt[rti + 1].k == RK_SYM && strchr("+-*^", rt[rti + 1].s[0]))) {
		u128 v = rt[rti].v;
		if (rt[rti].big)
			rfail("that modulus is too large");
		rti++;
		for (i = 0; v; i++) {
			if (i > RMAXF)
				rfail("the modulus has degree above %d", RMAXF);
			f[i] = (u64)(v % p);
			v /= p;
		}
		*deg = i - 1;
		if (*deg < 1 || f[*deg] != 1)
			rfail("a modulus code must give a monic polynomial of degree >= 1");
		return 1;
	}
	ring_zn(&B, p);
	pexpr_into(&B, &m, "x");
	if (m.d < 1)
		rfail("the modulus must have degree >= 1");
	if (m.d > RMAXF)
		rfail("the modulus has degree above %d", RMAXF);
	if (!rp_monic(&B, &m))
		rfail("the modulus' leading coefficient is not a unit mod %llu", U(p));
	*deg = m.d;
	for (i = 0; i <= m.d; i++)
		f[i] = m.c[i];
	(void)want;
	return 1;
}

/* the smallest primitive polynomial of degree n over GF(p) */
static void
default_modulus(u64 p, int n, u64 *f)
{
	u64 c, lim, d[RMAXF + 1];
	struct ring R;
	int i;

	lim = n < 64 ? ((u64)1 << 62) : ~(u64)0;
	for (c = 1; c < lim; c++) {
		u64 v = c;
		if (c % p == 0)
			continue;
		for (i = 0; i < n; i++) {
			d[i] = v % p;
			v /= p;
		}
		if (v)
			break;
		d[n] = 1;
		if (!ring_ext(&R, p, n, d) || !R.field)
			continue;
		if (r_order(&R, r_x(&R)) == R.qm1) {
			memcpy(f, d, sizeof(u64) * (size_t)(n + 1));
			return;
		}
	}
	rfail("no primitive polynomial found");
}

static void
spec_name(struct rspec *s)
{
	struct sbuf b = {0};
	struct ring B;
	struct rpoly m;
	int i;

	if (s->kind == RS_QX)
		return;
	sb_puts(&b, "");
	if (s->kind == RS_ZN)
		sb_printf(&b, "Z/%llu", U(s->R.n));
	else if (s->kind == RS_EXT) {
		ring_zn(&B, s->R.n);
		m.d = s->R.deg;
		for (i = 0; i <= m.d; i++)
			m.c[i] = s->R.f[i];
		if (s->gf && s->R.field) {
			sb_printf(&b, "GF(%llu^%d)", U(s->R.n), s->R.deg);
		} else
			sb_printf(&b, "%s%llu[x]/(%s)", s->R.nprime ? "GF(" : "Z/",
			    U(s->R.n), "");
		if (!(s->gf && s->R.field)) {
			/* rewrite: GF(p)[x]/(f) or Z/n[x]/(f) */
			sb_clear(&b);
			sb_printf(&b, "%s%llu%s[x]/(%s)", s->R.nprime ? "GF(" : "Z/",
			    U(s->R.n), s->R.nprime ? ")" : "", rp_str(&B, &m));
		}
	} else if (s->kind == RS_ZI)
		sb_puts(&b, "Z[i]");
	else if (s->kind == RS_ZSQRT)
		sb_printf(&b, "Z[sqrt(%lld)]", (long long)s->d);
	snprintf(s->name, sizeof s->name, "%s", b.p ? b.p : "");
	free(b.p);
}

static int parse_qx(struct rspec *s);

/* [x]/(f) after Z/n or GF(p) */
static int
ext_suffix(struct rspec *s)
{
	int save = rti, deg;
	u64 f[RMAXF + 1];
	const char *fn;

	if (!acc_sym('['))
		return 0;
	if (!acc_id("x") || !acc_sym(']')) {
		rti = save;
		return 0;
	}
	if (!acc_sym('/'))
		rfail("expected /(f) after %s[x]", s->name);
	{
		int par = acc_sym('(');
		parse_modulus(s->R.n, 0, f, &deg, &fn);
		if (par)
			expect_sym(')');
	}
	if (!ring_ext(&s->R, s->R.n, deg, f))
		rfail("that ring has more than 2^64 elements");
	s->R.fname = fn;
	s->kind = RS_EXT;
	s->gf = 0;
	return 1;
}

static int
gf_prefix(const char *w, u64 *q)
{
	const char *d = NULL;

	if (!strncasecmp(w, "GF", 2) && isdigit((unsigned char)w[2]))
		d = w + 2;
	else if ((w[0] == 'F' || w[0] == 'f') && w[1] == '_' &&
	    isdigit((unsigned char)w[2]))
		d = w + 2;
	if (d == NULL)
		return 0;
	*q = strtoull(d, NULL, 10);
	return 1;
}

static int
zn_prefix(const char *w, u64 *n)
{
	if ((w[0] == 'Z' || w[0] == 'z') && w[1] == '_' &&
	    isdigit((unsigned char)w[2])) {
		*n = strtoull(w + 2, NULL, 10);
		return 1;
	}
	return 0;
}

/* GF(q[, modulus]), GF q, GF256, F_q, then an optional modulus */
static void
finish_gf(struct rspec *s, u64 q, int have_mod, u64 *f, int deg,
    const char *fn)
{
	struct nt_fac nf;
	u64 p;
	int n;

	if (q < 2)
		rfail("GF(%llu) is not a field: q must be a prime power", U(q));
	nt_factor(q, &nf);
	if (nf.k != 1)
		rfail("GF(%llu): %llu is not a prime power, so there is no field "
		    "with %llu elements (Z/%llu is a ring)", U(q), U(q), U(q), U(q));
	p = nf.p[0];
	n = nf.e[0];
	if (have_mod) {
		if (deg != n && n > 1)
			rfail("GF(%llu) = GF(%llu^%d) needs a modulus of degree %d, "
			    "not %d", U(q), U(p), n, n, deg);
		if (n == 1 && deg > 1) {
			/* GF(2, x^8 + ...): read as GF(2^8) with that modulus */
			n = deg;
		}
	}
	s->gf = 1;
	if (n == 1 && !have_mod) {
		ring_zn(&s->R, p);
		s->kind = RS_ZN;
		return;
	}
	if (!have_mod) {
		default_modulus(p, n, f);
		s->automod = 1;
		fn = NULL;
	}
	if (!ring_ext(&s->R, p, n, f))
		rfail("GF(%llu^%d) has more than 2^64 elements", U(p), n);
	s->R.fname = fn;
	s->kind = RS_EXT;
}

/*
 * A ring at rti: Z/n, Z/nZ, Zmod(n), Z_n, mod n, GF(q), GF(q, f), GF q f,
 * F_q, GF(p)[x]/(f), Z/n[x]/(f), Z[i], Z[sqrt(d)], Q[x]/(f).
 */
static int
parse_ring(struct rspec *s)
{
	int save = rti;
	u64 q, f[RMAXF + 1];
	int deg = 0, have = 0;
	const char *fn = NULL;

	memset(s, 0, sizeof *s);
	if (rti >= rlim || (rt[rti].k != RK_ID && !at_sym('(')))
		return 0;
	if (at_sym('(')) {
		/* (mod n) */
		rti++;
		if (!at_id("mod") && !at_id("modulo")) {
			rti = save;
			return 0;
		}
		rti++;
		ring_zn(&s->R, modulus_arg());
		expect_sym(')');
		s->kind = RS_ZN;
		goto done;
	}
	if (at_id("Z") || at_id("ZZ") || at_id("Integers")) {
		rti++;
		if (acc_sym('/')) {
			int par = acc_sym('(');
			ring_zn(&s->R, modulus_arg());
			if (par)
				expect_sym(')');
			acc_id("Z") || acc_id("ZZ");
			s->kind = RS_ZN;
			spec_name(s);
			ext_suffix(s);
			goto done;
		}
		if (acc_sym('(') && !at_end()) {
			ring_zn(&s->R, modulus_arg());
			expect_sym(')');
			s->kind = RS_ZN;
			goto done;
		}
		if (acc_sym('[')) {
			if (acc_id("i") && acc_sym(']')) {
				s->kind = RS_ZI;
				s->d = -1;
				goto done;
			}
			rti = save + 2;
			if (acc_id("sqrt")) {
				int par = acc_sym('(');
				i128 d = iexpr_unary();
				if (par)
					expect_sym(')');
				if (!acc_sym(']'))
					rfail("expected ] after Z[sqrt(d)");
				if (d == 0 || d > 1000000000 || d < -1000000000)
					rfail("Z[sqrt(d)]: d must be a nonzero integer "
					    "(|d| <= 10^9)");
				s->d = (int64_t)d;
				s->kind = d == -1 ? RS_ZI : RS_ZSQRT;
				goto done;
			}
			rfail("Z[...]: Z[i] or Z[sqrt(d)] (polynomial rings: "
			    "Z/n[x]/(f), GF(p)[x]/(f), Q[x]/(f))");
		}
		rti = save;
		return 0;
	}
	if (at_id("Zmod") || at_id("IntegerModRing") || at_id("Zn")) {
		rti++;
		expect_sym('(');
		ring_zn(&s->R, modulus_arg());
		expect_sym(')');
		s->kind = RS_ZN;
		goto done;
	}
	if (rt[rti].k == RK_ID && zn_prefix(rt[rti].s, &q)) {
		rti++;
		ring_zn(&s->R, q);
		s->kind = RS_ZN;
		goto done;
	}
	if (at_id("mod") || at_id("modulo")) {
		rti++;
		if (!starts_number()) {
			rti = save;
			return 0;
		}
		ring_zn(&s->R, modulus_arg());
		s->kind = RS_ZN;
		goto done;
	}
	if (at_id("Q") || at_id("QQ")) {
		rti++;
		if (parse_qx(s))
			goto done;
		rti = save;
		return 0;
	}
	if (rt[rti].k == RK_ID && gf_prefix(rt[rti].s, &q)) {
		rti++;
		goto gfmod;
	}
	if (at_id("GF") || at_id("F") || at_id("FF") || at_id("GaloisField") ||
	    at_id("Fq")) {
		rti++;
		if (acc_sym('(')) {
			i128 v = iexpr_sum();
			if (v < 2 || (v >> 64))
				rfail("GF(q): q must be a prime power below 2^64");
			q = (u64)v;
			if (acc_sym(',')) {
				u128 dummy = 0;
				(void)dummy;
				{
					struct nt_fac nf;
					nt_factor(q, &nf);
					if (nf.k != 1)
						rfail("GF(%llu): not a prime power", U(q));
					parse_modulus(nf.p[0], nf.e[0], f, &deg, &fn);
				}
				have = 1;
			}
			expect_sym(')');
		} else if (acc_sym('_')) {
			q = modulus_arg();
		} else if (starts_number()) {
			i128 v = iexpr_unary();
			if (v < 2 || (v >> 64))
				rfail("GF(q): q must be a prime power below 2^64");
			q = (u64)v;
		} else {
			rti = save;
			return 0;
		}
		goto gfmod;
	}
	return 0;
gfmod:
	if (!have) {
		/* GF(2)[x]/(f): an explicit quotient */
		if (at_sym('[')) {
			struct nt_fac nf;
			nt_factor(q, &nf);
			if (nf.k != 1 || nf.e[0] != 1)
				rfail("%llu[x]/(f): the coefficients must be a prime field",
				    U(q));
			ring_zn(&s->R, q);
			s->kind = RS_ZN;
			s->gf = 1;
			spec_name(s);
			if (ext_suffix(s))
				goto done;
		}
		{
			int s2 = rti;
			int named = rti < rlim && rt[rti].k == RK_ID &&
			    named_index(rt[rti].s) >= 0;
			if (named || acc_id("mod") || acc_id("modulo") ||
			    acc_id("with") || acc_id("using") || acc_sym(',')) {
				struct nt_fac nf;
				acc_id("the");
				acc_id("modulus");
				acc_id("polynomial");
				nt_factor(q, &nf);
				if (nf.k == 1 && !at_end()) {
					parse_modulus(nf.p[0], nf.e[0], f, &deg, &fn);
					have = 1;
				} else
					rti = s2;
			}
		}
	}
	finish_gf(s, q, have, f, deg, fn);
done:
	spec_name(s);
	return 1;
}

/*
 * The ring a command works in: the first "mod R", "over R", "in R" (or a
 * bare GF(...), Z/n) that reads to the end.  [from, *end) is left for the
 * command's own arguments.  report: when nothing reads, show why the last
 * candidate did not.
 */
static int
find_ring(struct rspec *s, int from, int *end, int report)
{
	int i, depth = 0, best = -1, last = -1, save = rti, slim = rlim;
	static struct rspec t;
	jmp_buf outer;

	memcpy(outer, spec_jmp, sizeof outer);

	for (i = from; i < nrt; i++) {
		int w = rt[i].k == RK_ID, kw, st, ok = 0;
		if (rt[i].k == RK_SYM && (rt[i].s[0] == ')' || rt[i].s[0] == ']'))
			depth--;
		if (rt[i].k == RK_SYM && (rt[i].s[0] == '(' || rt[i].s[0] == '[')) {
			depth++;
			if (!(rt[i].s[0] == '(' && i + 1 < nrt && rt[i + 1].k == RK_ID &&
			    (!strcasecmp(rt[i + 1].s, "mod") ||
			    !strcasecmp(rt[i + 1].s, "modulo"))))
				continue;
			if (depth > 1)
				continue;
		} else if (!w || depth > 0)
			continue;
		kw = w && (!strcasecmp(rt[i].s, "over") ||
		    !strcasecmp(rt[i].s, "in"));
		st = kw ? i + 1 : i;
		if (!kw && w && !(!strcasecmp(rt[i].s, "mod") ||
		    !strcasecmp(rt[i].s, "modulo") ||
		    !strncasecmp(rt[i].s, "GF", 2) || !strcasecmp(rt[i].s, "F") ||
		    !strcasecmp(rt[i].s, "Z") || !strcasecmp(rt[i].s, "Zmod") ||
		    !strncasecmp(rt[i].s, "Z_", 2) ||
		    !strncasecmp(rt[i].s, "F_", 2)))
			continue;
		last = st;
		spec_depth++;
		if (setjmp(spec_jmp) == 0) {
			rti = st;
			rlim = nrt;
			ok = parse_ring(&t) && rti == nrt;
		}
		spec_depth--;
		notes_drop();
		if (ok) {
			best = i;
			*s = t;
			break;
		}
	}
	rti = save;
	rlim = slim;
	memcpy(spec_jmp, outer, sizeof outer);
	if (best < 0) {
		if (report && last >= 0) {
			rti = last;
			rlim = nrt;
			if (parse_ring(&t) && rti < nrt)
				rfail("unexpected %s after the ring", here());
		}
		return 0;
	}
	*end = best;
	return 1;
}

#include "ring_zn.inc"
#include "ring_gf.inc"
#include "ring_misc.inc"
#include "ring_cmd.inc"
