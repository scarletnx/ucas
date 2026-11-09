/* ring.c, part 4: division by constants, GF(2) maps, Z[i], Z[sqrt d], Q[x]/(f) */

/* ---- magic numbers for division by a constant ---- */

/* floor(2^k / d) and the remainder, k <= 128 */
static void
pow2div(int k, u128 d, u128 *q, u128 *r)
{
	if (k < 128) {
		*q = ((u128)1 << k) / d;
		*r = ((u128)1 << k) % d;
		return;
	}
	*q = (~(u128)0) / d;
	*r = (~(u128)0) % d + 1;
	if (*r == d) {
		*q += 1;
		*r = 0;
	}
}

static u64
mulhi(u64 a, u64 b, int w)
{
	return (u64)(((u128)a * b) >> w);
}

static int64_t
sext(u64 v, int w)
{
	if (w == 64)
		return (int64_t)v;
	v &= ((u64)1 << w) - 1;
	if (v >> (w - 1))
		return (int64_t)(v | ~(((u64)1 << w) - 1));
	return (int64_t)v;
}

/* the generated sequence, emulated in w-bit arithmetic */
static u64
magic_u(u64 x, u64 m, int s, int add, int w)
{
	u64 mask = w == 64 ? ~(u64)0 : (((u64)1 << w) - 1), t;

	if (!add)
		return mulhi(x, m, w) >> s;
	t = mulhi(x, m, w);   /* m is the low w bits of the w+1-bit magic */
	return ((((x - t) & mask) >> 1) + t) >> (s - 1);
}

static int64_t
magic_s(int64_t x, int64_t m, int s, int add, int w)
{
	i128 p = (i128)x * m;
	int64_t t = (int64_t)(p >> w);

	if (add)
		t += x;
	t = sext((u64)t, w);
	return sext((u64)((t >> (s - 1)) + (x < 0 ? 1 : 0)), w);
}

/*
 * The magic number of `magic D uW|iW` for d >= 3 not a power of two: the
 * least s with m = ceil(2^(w' + s) / d), e = m d - 2^(w' + s) <= 2^s
 * (w' = w, or w - 1 signed).  Unsigned: *add when m has w + 1 bits, and
 * *m is then its low w bits (the add sequence of magic_u); signed: *m is
 * m itself (< 2^w).  *e is the error term of the proof.
 */
int
ring_magic_number(u64 d, int w, int sign, u64 *mo, int *so, int *addo,
    u64 *eo)
{
	u128 q, r, m = 0, e = 0;
	int s, k, wp = sign ? w - 1 : w;

	if (w < 2 || w > 64 || d < 3 || !(d & (d - 1)) ||
	    (u128)d >= ((u128)1 << wp))
		return 0;
	for (s = sign; s <= 64; s++) {
		k = wp + s;
		pow2div(k, d, &q, &r);
		m = q + (r != 0);
		e = r ? d - r : 0;
		if (e <= ((u128)1 << s))
			break;
	}
	if (s > 64)
		return 0;
	*so = s;
	*eo = (u64)e;
	if (!sign) {
		*addo = (m >> w) != 0;
		*mo = *addo ? (w == 64 ? (u64)m : (u64)(m - ((u128)1 << w))) :
		    (u64)m;
	} else {
		*addo = m >= ((u128)1 << (w - 1));
		*mo = (u64)m;
	}
	return 1;
}

static u64
want_div(u64 x, const void *arg)
{
	return x / *(const u64 *)arg;
}

static void
cmd_magic(void)
{
	i128 d0;
	char usrc[400] = "";
	int w = 0, sign = -1, s, add = 0, k;
	u128 m = 0, e = 0, q, r;
	u64 d;

	d0 = iexpr_unary();
	while (!at_end()) {
		ty_t t;
		if (rt[rti].k == RK_ID && ty_parse(rt[rti].s, strlen(rt[rti].s), &t) &&
		    (TYK(t) == K_UINT || TYK(t) == K_SINT)) {
			w = TYW(t);
			sign = TYK(t) == K_SINT;
			rti++;
		} else if (acc_id("signed"))
			sign = 1;
		else if (acc_id("unsigned"))
			sign = 0;
		else if (rti < rlim && rt[rti].k == RK_NUM && rti + 1 < rlim &&
		    (!strcasecmp(rt[rti + 1].s, "bit") ||
		    !strcasecmp(rt[rti + 1].s, "bits"))) {
			w = (int)rt[rti].v;
			rti += 2;
		} else if (acc_sym('-') || acc_id("for") || acc_id("as") ||
		    acc_id("in") || acc_id("a") || acc_id("an"))
			;
		else
			rfail("unexpected %s (magic D [u32|i32|uN|iN])", here());
	}
	if (w == 0) {
		static const char *const opt[] = {"u32", "i32", "u64", "i64", "u16",
			"u8"};
		static const int ow[] = {32, 32, 64, 64, 16, 8}, os[] = {0, 1, 0, 1,
			0, 0};
		int c = nlp_ask_choice(sign == 1 ? "which signed type" :
		    "which integer type", opt, 6, sign == 1 ? 1 : 0);
		w = ow[c];
		if (sign < 0)
			sign = os[c];
	}
	if (sign < 0) {
		static const char *const opt[] = {"unsigned", "signed"};
		char what[64];
		snprintf(what, sizeof what, "%d-bit: signed or unsigned", w);
		sign = nlp_ask_choice(what, opt, 2, 0);
	}
	if (w < 2 || w > 64)
		rfail("widths 2..64");
	if (d0 == 0)
		rfail("division by zero");
	if (d0 < 0 && !sign)
		rfail("an unsigned division by a negative number");
	d = (u64)(d0 < 0 ? -d0 : d0);
	if (d0 < 0)
		note("x / %s = -(x / %llu): the magic number for %llu, then negate",
		    i128s(d0), U(d), U(d));
	printf("x / %llu for %s x: %s%d\n", U(d), sign ? "signed" : "unsigned",
	    sign ? "i" : "u", w);
	if (d == 1) {
		printf("  x itself\n");
		return;
	}
	if (!sign && (u128)d >= ((u128)1 << w)) {
		printf("  q = x >= %llu ? 1 : 0 (the divisor is at least 2^%d / 2)\n",
		    U(d), w);
		return;
	}
	if (!(d & (d - 1))) {
		int sh = ilog2u(d);
		if (!sign)
			printf("  q = x >> %d (a power of two)\n", sh);
		else
			printf("  q = (x + ((x >> %d) & %llu)) >> %d (arithmetic shifts): "
			    "a shift rounds toward -inf, so negative x first get %llu - 1 "
			    "added (C's / rounds toward 0)\n", w - 1, U(d - 1), sh, U(d));
		return;
	}
	/* the least s with m = ceil(2^(w' + s)/d), e = m d - 2^(w' + s) <= 2^s */
	{
		int wp = sign ? w - 1 : w;
		for (s = sign; s <= 64; s++) {
			k = wp + s;
			pow2div(k, d, &q, &r);
			m = q + (r != 0);
			e = r ? d - r : 0;
			if (s < 128 && e <= ((u128)1 << s))
				break;
		}
		printf("  m = ceil(2^%d / %llu) = %s (%s, %d bits), shift s = %d\n", k,
		    U(d), u128s(m), u128x(m), ilog2u((u64)(m >> 1)) + 2 > 0 &&
		    (m >> 64) == 0 ? ilog2u((u64)m) + 1 : 65, s);
		printf("  q = floor(x * m / 2^%d)%s\n", k, sign ?
		    " for x >= 0, plus 1 for x < 0" : "");
		printf("  proof (Granlund-Montgomery): m*%llu = 2^%d + e with e = %s "
		    "<= 2^%d.  Write x = q*%llu + r, 0 <= r < %llu.  Then x*m/2^%d = "
		    "x/%llu + x*e/(%llu*2^%d), and x*e/2^%d %s 1 because |x| %s 2^%d, "
		    "so the extra term is %s 1/%llu: it cannot carry r/%llu <= "
		    "%llu/%llu past the next integer\n", U(d), k, u128s(e), s, U(d),
		    U(d), k, U(d), U(d), k, k, sign ? "<=" : "<", sign ? "<=" : "<",
		    wp, sign ? "at most" : "below", U(d), U(d), U(d - 1), U(d));
		if (sign)
			printf("  (for x < 0 the floor lands one below the truncated "
			    "quotient, since e > 0: hence the + 1)\n");
		if (!sign) {
			u64 lo = (u64)m;
			add = (m >> w) != 0;
			if (!add) {
				printf("  m fits u%d:\n", w);
				printf("    C:    q = (uint%d_t)(((uint%d_t)x * 0x%llx) >> %d);"
				    "\n", w, w <= 32 ? 64 : 128, U(lo), w + s);
				if (w <= 32) {
					printf("    in ucas: u%d((u64(x) * 0x%llx) >> %d)\n", w,
					    U(lo), w + s);
					snprintf(usrc, sizeof usrc, "u%d((u64((x: u%d)) * 0x%llx) "
					    ">> %d)", w, w, U(lo), w + s);
				}
			} else {
				lo = w == 64 ? (u64)m : (u64)(m - ((u128)1 << w));
				printf("  m has %d bits, one more than u%d: the usual sequence "
				    "uses m - 2^%d = 0x%llx and an add that cannot overflow:\n",
				    w + 1, w, w, U(lo));
				printf("    C:    t = mulhi(x, 0x%llx); q = (t + ((x - t) >> 1)) "
				    ">> %d;\n", U(lo), s - 1);
				if (w <= 32) {
					char t[120];
					snprintf(t, sizeof t, "u%d((u64(x) * 0x%llx) >> %d)", w,
					    U(lo), w);
					printf("    in ucas: (%s + ((x - %s) >> 1)) >> %d\n", t, t,
					    s - 1);
					snprintf(usrc, sizeof usrc, "(u%d((u64((x: u%d)) * 0x%llx) "
					    ">> %d) + ((x - %s) >> 1)) >> %d", w, w, U(lo), w, t,
					    s - 1);
				}
				m = lo;
			}
		} else {
			add = m >= ((u128)1 << (w - 1));
			printf("  as i%d, m = %lld%s:\n", w, (long long)sext((u64)m, w),
			    add ? " (negative: the sequence adds x back)" : "");
			printf("    C:    t = mulhs(x, %lld)%s; q = (t >> %d) + "
			    "((uint%d_t)x >> %d);   (mulhs: the high %d bits of the "
			    "signed product, which already divides by 2^%d)\n",
			    (long long)sext((u64)m, w), add ? " + x" : "", s - 1, w, w - 1,
			    w, w);
		}
	}
	/* check */
	{
		u64 n = 0, bad = 0, x, top = w == 64 ? ~(u64)0 : (((u64)1 << w) - 1);
		int exh = w <= 22;
		if (!sign) {
#define CHK(xx) do { u64 x_ = (xx) & top; n++; \
	if (magic_u(x_, (u64)m, s, add, w) != x_ / d) { bad++; } } while (0)
			if (exh)
				for (x = 0; x <= top; x++)
					CHK(x);
			else {
				u64 kk, st = 0x9e3779b97f4a7c15ULL;
				for (kk = 0; kk < 4096; kk++) {
					CHK(kk);
					CHK(top - kk);
					CHK((top / d - kk) * d - 1);
					CHK((top / d - kk) * d);
					CHK(kk * d - 1);
					CHK(kk * d);
				}
				for (kk = 0; kk < (1u << 20); kk++) {
					st ^= st << 13;
					st ^= st >> 7;
					st ^= st << 17;
					CHK(st);
				}
			}
#undef CHK
		} else {
			int64_t lo = w == 64 ? INT64_MIN : -((int64_t)1 << (w - 1));
			int64_t hi = w == 64 ? INT64_MAX : ((int64_t)1 << (w - 1)) - 1;
			int64_t ms = sext((u64)m, w);
#define CHKS(xx) do { int64_t x_ = sext((u64)(xx), w); n++; \
	if (magic_s(x_, ms, s, add, w) != x_ / (int64_t)d) { bad++; } } while (0)
			if (exh) {
				int64_t xs;
				for (xs = lo;; xs++) {
					CHKS(xs);
					if (xs == hi)
						break;
				}
			} else {
				u64 kk, st = 0x9e3779b97f4a7c15ULL;
				for (kk = 0; kk < 4096; kk++) {
					CHKS(lo + (int64_t)kk);
					CHKS(hi - (int64_t)kk);
					CHKS((int64_t)kk * (int64_t)d - 1);
					CHKS(-(int64_t)kk * (int64_t)d + 1);
					CHKS((hi / (int64_t)d - (int64_t)kk) * (int64_t)d);
					CHKS((lo / (int64_t)d + (int64_t)kk) * (int64_t)d - 1);
				}
				for (kk = 0; kk < (1u << 20); kk++) {
					st ^= st << 13;
					st ^= st >> 7;
					st ^= st << 17;
					CHKS(st);
				}
			}
#undef CHKS
		}
		if (usrc[0]) {
			u64 eb, en;
			int all;
			en = engine_check(usrc, w, want_div, &d, &eb, &all);
			printf("  check: the ucas form above, run by ucas's evaluator in "
			    "u%d, equals x / %llu for %s (%llu inputs)%s\n", w, U(d),
			    all ? "every x" : "the 256 smallest and largest x and 65536 "
			    "random x", U(en), eb ? ": FAILED" : "");
		}
		if (bad)
			printf("  check: FAILED: the sequence differs from x / %llu at "
			    "%llu of %llu inputs\n", U(d), U(bad), U(n));
		else
			printf("  check: the sequence above, emulated in %d-bit "
			    "arithmetic, equals x / %llu (C's truncating division) for "
			    "%s (%llu inputs)\n", w, U(d), exh ? "every input" :
			    "the ends of the range, both sides of the multiples of d "
			    "near 0 and near the top, and 2^20 random inputs", U(n));
	}
}

/* ---- the GF(2) matrix of a bit expression ---- */

static int
mat_pow_is_identity(const struct ring *R, const struct rmat *m, u64 e)
{
	struct rmat acc, b, t;
	int i, n = m->r, id;

	rm_init(&acc, n, n);
	for (i = 0; i < n; i++)
		RM(&acc, i, i) = 1;
	rm_init(&b, n, n);
	memcpy(b.a, m->a, sizeof(u64) * (size_t)n * n);
	while (e) {
		if (e & 1) {
			rm_mul(R, &t, &acc, &b);
			rm_free(&acc);
			acc = t;
		}
		e >>= 1;
		if (e) {
			rm_mul(R, &t, &b, &b);
			rm_free(&b);
			b = t;
		}
	}
	id = rm_is_identity(R, &acc);
	rm_free(&acc);
	rm_free(&b);
	return id;
}

static void
cmd_gf2matrix(const char *text)
{
	char src[1200], var[32] = "", tyw[16] = "", shown[1200];
	const char *p;
	size_t o = 0;
	struct func *f;
	struct plan pl;
	int64_t *ii;
	double *ir;
	int w = 0, i, j, slot = 0, annotated = 0;
	u64 col[64], c0, mask;
	struct ring B;
	struct rmat M;

	/* a trailing type: "u8", "as u16", "for 32 bits" */
	{
		char buf[1200];
		size_t n;
		snprintf(buf, sizeof buf, "%s", text);
		n = strlen(buf);
		while (n && isspace((unsigned char)buf[n - 1]))
			buf[--n] = 0;
		for (;;) {
			size_t k = n;
			ty_t t;
			while (k && isalnum((unsigned char)buf[k - 1]))
				k--;
			if (k < n && ty_parse(buf + k, n - k, &t) && TYK(t) == K_UINT &&
			    (k == 0 || buf[k - 1] == ' ')) {
				w = TYW(t);
				buf[k] = 0;
				n = k;
				while (n && (isspace((unsigned char)buf[n - 1])))
					buf[--n] = 0;
				if (n >= 2 && !strcmp(buf + n - 2, "as"))
					buf[n -= 2] = 0;
				else if (n >= 3 && !strcmp(buf + n - 3, "for"))
					buf[n -= 3] = 0;
				break;
			}
			break;
		}
		snprintf(shown, sizeof shown, "%s", buf);
		/* ^ is xor here */
		for (p = buf; *p && o + 6 < sizeof src; p++) {
			if (*p == '^') {
				memcpy(src + o, " xor ", 5);
				o += 5;
			} else
				src[o++] = *p;
		}
		src[o] = 0;
		if (strchr(buf, '^'))
			note("^ is xor here (addition in GF(2))");
	}
	/* the variable: the first identifier that is not an operator word */
	for (p = src; *p; p++)
		if ((isalpha((unsigned char)*p) || *p == '_') && (p == src ||
		    !isalnum((unsigned char)p[-1]))) {
			size_t n = 0;
			while (isalnum((unsigned char)p[n]) || p[n] == '_')
				n++;
			if (!(n == 3 && !strncmp(p, "xor", 3)) && n < sizeof var &&
			    !(p[n] == '(')) {
				memcpy(var, p, n);
				var[n] = 0;
				{
					const char *q = p + n;
					while (*q == ' ')
						q++;
					annotated = *q == ':';
				}
				break;
			}
			p += n - 1;
		}
	if (!var[0])
		rfail("gf2matrix: an expression in one variable, e.g. x ^ (x << 3)");
	if (!annotated) {
		if (w == 0) {
			static const char *const opt[] = {"u32", "u8", "u16", "u64"};
			static const int ow[] = {32, 8, 16, 64};
			w = ow[nlp_ask_choice("the width of x", opt, 4, 0)];
		}
		snprintf(tyw, sizeof tyw, "u%d", w);
		/* annotate the first use */
		{
			char t2[1300];
			const char *q = strstr(src, var);
			while (q && ((q > src && (isalnum((unsigned char)q[-1]) ||
			    q[-1] == '_')) || isalnum((unsigned char)q[strlen(var)])))
				q = strstr(q + 1, var);
			if (q == NULL)
				rfail("cannot find %s", var);
			snprintf(t2, sizeof t2, "%.*s(%s: %s)%s", (int)(q - src), src, var,
			    tyw, q + strlen(var));
			snprintf(src, sizeof src, "%s", t2);
		}
	}
	f = cmd_func_text(src);
	if (f->np != 1)
		rfail("gf2matrix: one variable (this has %d)", f->np);
	if (TYK(f->p[0].ty) != K_UINT && TYK(f->p[0].ty) != K_SINT)
		rfail("gf2matrix: the variable must be an integer type");
	if (TYK(N.ty[f->body]) != K_UINT && TYK(N.ty[f->body]) != K_SINT)
		rfail("gf2matrix: the result must be an integer");
	w = TYW(f->p[0].ty);
	{
		int wo = TYW(N.ty[f->body]);
		if (wo != w)
			rfail("gf2matrix: %d input bits and %d output bits: the matrix "
			    "is not square (cast the result)", w, wo);
	}
	mask = w == 64 ? ~(u64)0 : (((u64)1 << w) - 1);
	plan_build(&pl, &f->body, 1, 0);
	ii = xcalloc((size_t)MAXP * CH, sizeof *ii);
	ir = xcalloc((size_t)MAXP * CH, sizeof *ir);
	/* f(0), f(e_j) */
	ii[slot * CH] = 0;
	for (j = 0; j < w; j++)
		ii[slot * CH + 1 + j] = (int64_t)((u64)1 << j);
	plan_run(&pl, (uint32_t)w + 1, ii, ir);
	c0 = (u64)COLI(&pl, pl.root[0])[0] & mask;
	for (j = 0; j < w; j++)
		col[j] = ((u64)COLI(&pl, pl.root[0])[1 + j] & mask) ^ c0;
	ring_zn(&B, 2);
	rm_init(&M, w, w);
	for (i = 0; i < w; i++)
		for (j = 0; j < w; j++)
			RM(&M, i, j) = col[j] >> i & 1;
	/* is it affine: f(x) = M x + c for every x tried */
	{
		u64 n = 0, bad = 0, x, top = mask, st = 0x2545f4914f6cdd1dULL, wit = 0;
		int exh = w <= 20;
		u64 lim = exh ? top + 1 : 65536;
		for (x = 0; x < lim;) {
			uint32_t k, m = 0;
			u64 xs[CH];
			for (k = 0; k < CH && x < lim; k++, x++) {
				u64 v = x;
				if (!exh) {
					st ^= st << 13;
					st ^= st >> 7;
					st ^= st << 17;
					v = st & mask;
				}
				xs[k] = v;
				ii[slot * CH + k] = (int64_t)v;
				m++;
			}
			plan_run(&pl, m, ii, ir);
			for (k = 0; k < m; k++) {
				u64 y = c0, v = xs[k], got;
				for (j = 0; j < w; j++)
					if (v >> j & 1)
						y ^= col[j];
				got = (u64)COLI(&pl, pl.root[0])[k] & mask;
				if (got != y && !bad++)
					wit = v;
				n++;
			}
		}
		if (bad) {
			printf("%s is not GF(2)-linear (nor affine) on u%d: at x = 0x%llx "
			    "it differs from the linear map through its values at the "
			    "unit vectors\n", shown, w, U(wit));
			printf("  (linear maps are made of xor, shifts, rotations, "
			    "and with constants, bit permutations, and multiplication by "
			    "a constant bit)\n");
			plan_free(&pl);
			free(ii);
			free(ir);
			rm_free(&M);
			return;
		}
		printf("the GF(2) matrix of %s on u%d (row i = output bit i, column "
		    "j = input bit j, bit 0 first)\n", shown, w);
		for (i = 0; i < w; i++) {
			printf("  [");
			for (j = 0; j < w; j++)
				printf("%c", RM(&M, i, j) ? '1' : '.');
			printf("]\n");
		}
		if (c0)
			printf("  plus the constant 0x%llx: the map is affine, f(x) = M x "
			    "xor 0x%llx\n", U(c0), U(c0));
		printf("  how: the columns are f(1 << j)%s; check: f(x) = M x%s for "
		    "%s (%llu inputs)\n", c0 ? " xor f(0)" : "", c0 ? " xor f(0)" :
		    "", exh ? "every x" : "65536 random x, not every x", U(n));
	}
	/* rank, inverse, characteristic polynomial, order */
	{
		struct rmat t, inv;
		struct rpoly cp;
		static struct rpoly fac[RMAXDEG + 1];
		int mult[RMAXDEG + 1], rk, k, sq = 1, kk;
		u64 lead;
		rm_init(&t, w, w);
		memcpy(t.a, M.a, sizeof(u64) * (size_t)w * w);
		rk = rm_rref(&B, &t, NULL);
		rm_free(&t);
		printf("  rank %d: %s\n", rk, rk == w ? "invertible (a bijection of "
		    "the values)" : "singular (some values collide; the kernel has "
		    "dimension w - rank)");
		if (rk == w && rm_inverse(&B, &M, &inv)) {
			/* as xor of masked shifts, by diagonals */
			struct sbuf e = {0};
			int dlt;
			sb_puts(&e, "");
			for (dlt = -(w - 1); dlt <= w - 1; dlt++) {
				u64 msk = 0, full;
				for (i = 0; i < w; i++) {
					j = i - dlt;
					if (j >= 0 && j < w && RM(&inv, i, j))
						msk |= (u64)1 << i;
				}
				if (!msk)
					continue;
				if (e.n)
					sb_puts(&e, " ^ ");
				full = dlt >= 0 ? (mask << dlt) & mask : mask >> -dlt;
				if (dlt == 0)
					sb_printf(&e, msk == mask ? "x" : "(x & 0x%llx)", U(msk));
				else if (msk == full)
					sb_printf(&e, "(x %s %d)", dlt > 0 ? "<<" : ">>",
					    dlt > 0 ? dlt : -dlt);
				else
					sb_printf(&e, "((x %s %d) & 0x%llx)", dlt > 0 ? "<<" :
					    ">>", dlt > 0 ? dlt : -dlt, U(msk));
			}
			if (!e.n)
				sb_puts(&e, "0");
			printf("  inverse%s: %s\n", c0 ? " of the linear part" : "", e.p);
			/* check by running the printed inverse through the evaluator */
			{
				char s2[4096], *q2;
				struct func *g;
				struct plan pg;
				u64 bad = 0, tot = 0, x;
				size_t oo = 0;
				int first_y = 0;
				for (q2 = e.p; *q2 && oo + 8 < sizeof s2; q2++) {
					if (*q2 == '^') {
						memcpy(s2 + oo, "xor", 3);
						oo += 3;
					} else if (*q2 == 'x' && (q2 == e.p || q2[-1] != '0') &&
					    oo + 16 < sizeof s2) {
						if (!first_y++)
							oo += (size_t)snprintf(s2 + oo, sizeof s2 - oo,
							    "(y: u%d)", w);
						else
							s2[oo++] = 'y';
					} else
						s2[oo++] = *q2;
				}
				s2[oo] = 0;
				g = cmd_func_text(s2);
				plan_build(&pg, &g->body, 1, 0);
				for (x = 0; x < (w <= 16 ? mask + 1 : (u64)w + 4096);) {
					uint32_t k2, m2 = 0;
					u64 xs[CH];
					for (k2 = 0; k2 < CH && x < (w <= 16 ? mask + 1 :
					    (u64)w + 4096); k2++, x++) {
						u64 v = w <= 16 ? x : x < (u64)w ? (u64)1 << x :
						    (x * 0x9e3779b97f4a7c15ULL) & mask;
						xs[k2] = v;
						ii[k2] = (int64_t)v;
						m2++;
					}
					plan_run(&pl, m2, ii, ir);
					for (k2 = 0; k2 < m2; k2++)
						ii[k2] = (int64_t)(((u64)COLI(&pl, pl.root[0])[k2] ^
						    c0) & mask);
					plan_run(&pg, m2, ii, ir);
					for (k2 = 0; k2 < m2; k2++) {
						tot++;
						bad += ((u64)COLI(&pg, pg.root[0])[k2] & mask) !=
						    xs[k2];
					}
				}
				printf("  check: the inverse as printed, evaluated after f%s: "
				    "gives back x for %s (%llu inputs)%s\n", c0 ? " (and xor "
				    "f(0))" : "", w <= 16 ? "every x" : "the unit vectors "
				    "(which proves it for a linear map) and 4096 others",
				    U(tot), bad ? ": FAILED" : "");
				plan_free(&pg);
			}
			free(e.p);
			rm_free(&inv);
		}
		rm_charpoly(&B, &M, &cp);
		k = rp_factor(&B, &cp, fac, mult, RMAXDEG + 1, &lead);
		for (kk = 0; kk < k; kk++)
			if (mult[kk] > 1)
				sq = 0;
		if (k == 1 && mult[0] == 1)
			printf("  characteristic polynomial: %s (irreducible)\n",
			    rp_str(&B, &cp));
		else
			printf("  characteristic polynomial: %s = %s\n", rp_str(&B, &cp),
			    fac_str(&B, fac, mult, k, lead));
		if (rk == w) {
			int of;
			u64 o = rp_xorder(&B, &cp, &of);
			if (!of) {
				/* M^o = I; remove prime factors while it stays I */
				struct nt_fac nf;
				int q;
				nt_factor(o, &nf);
				for (q = 0; q < nf.k; q++)
					while (o % nf.p[q] == 0 &&
					    mat_pow_is_identity(&B, &M, o / nf.p[q]))
						o /= nf.p[q];
				printf("  order %llu: applying the map %llu times gives the "
				    "identity (M^%llu = I and no M^(%llu/r) = I, by matrix "
				    "powers; it divides the order of x mod the characteristic "
				    "polynomial)\n", U(o), U(o), U(o), U(o));
				(void)sq;
			}
		}
	}
	plan_free(&pl);
	free(ii);
	free(ir);
	rm_free(&M);
}

/* ---- Z[i] and Z[sqrt d] ---- */

struct qi {
	i128 a, b;     /* a + b sqrt(d) */
};
static int64_t QD;

static i128
ck(i128 v)
{
	if (v > ((i128)1 << 100) || v < -((i128)1 << 100))
		rfail("the numbers grow too large");
	return v;
}

static struct qi
qi_mul(struct qi x, struct qi y)
{
	struct qi r;

	r.a = ck(x.a * y.a + (i128)QD * x.b * y.b);
	r.b = ck(x.a * y.b + x.b * y.a);
	return r;
}

static i128
qi_norm(struct qi x)
{
	return ck(x.a * x.a - (i128)QD * x.b * x.b);
}

static i128
rdiv(i128 n, i128 d)
{
	/* n/d rounded to nearest */
	i128 q, r;

	if (d < 0) {
		n = -n;
		d = -d;
	}
	q = n / d;
	r = n % d;
	if (r < 0) {
		q--;
		r += d;
	}
	if (2 * r > d)
		q++;
	return q;
}

static int
qi_euclid(void)
{
	return QD == -1 || QD == -2 || QD == 2 || QD == 3;
}

/* x = q y + r with |N(r)| < |N(y)| (rounded quotient); exact: r = 0 */
static int
qi_divmod(struct qi x, struct qi y, struct qi *q, struct qi *r)
{
	i128 n = qi_norm(y);
	struct qi c = {y.a, -y.b}, t;

	if (n == 0)
		rfail("division by an element of norm 0");
	t = qi_mul(x, c);
	q->a = rdiv(t.a, n);
	q->b = rdiv(t.b, n);
	t = qi_mul(*q, y);
	r->a = x.a - t.a;
	r->b = x.b - t.b;
	return r->a == 0 && r->b == 0;
}

static const char *
qi_str(struct qi x)
{
	static char buf[8][120];
	static int k;
	char *s = buf[k++ & 7];
	const char *u = QD == -1 ? "i" : NULL;
	char us[40];

	if (!u) {
		snprintf(us, sizeof us, "sqrt(%lld)", (long long)QD);
		u = us;
	}
	if (x.b == 0)
		snprintf(s, 120, "%s", i128s(x.a));
	else if (x.a == 0)
		snprintf(s, 120, "%s%s%s", x.b == 1 ? "" : x.b == -1 ? "-" :
		    i128s(x.b), x.b == 1 || x.b == -1 || QD == -1 ? "" : "*", u);
	else
		snprintf(s, 120, "%s %s %s%s%s", i128s(x.a), x.b < 0 ? "-" : "+",
		    (x.b == 1 || x.b == -1) ? "" : i128s(x.b < 0 ? -x.b : x.b),
		    (x.b == 1 || x.b == -1 || QD == -1) ? "" : "*", u);
	return s;
}

static struct qi qi_sum(void);

static struct qi
qi_gcd(struct qi x, struct qi y)
{
	struct qi q, r;

	if (!qi_euclid())
		rfail("gcd in Z[sqrt(%lld)]: not norm-Euclidean here", (long long)QD);
	while (y.a || y.b) {
		qi_divmod(x, y, &q, &r);
		x = y;
		y = r;
	}
	return x;
}

static int qi_nops;
static char qi_how[256];

static struct qi
qi_atom(void)
{
	struct qi r = {0, 0};

	if (acc_sym('(')) {
		r = qi_sum();
		expect_sym(')');
		return r;
	}
	if (rti < rlim && rt[rti].k == RK_NUM) {
		if (rt[rti].big || rt[rti].v >> 90)
			rfail("too large");
		r.a = (i128)rt[rti++].v;
		return r;
	}
	if (QD == -1 && (acc_id("i") || acc_id("I"))) {
		r.b = 1;
		return r;
	}
	if (acc_id("sqrt")) {
		int par = acc_sym('(');
		i128 d = iexpr_unary();
		if (par)
			expect_sym(')');
		if (d != QD)
			rfail("sqrt(%s) is not in Z[sqrt(%lld)]", i128s(d), (long long)QD);
		r.b = 1;
		return r;
	}
	if (acc_id("norm") || acc_id("N")) {
		expect_sym('(');
		r = qi_sum();
		expect_sym(')');
		r.a = qi_norm(r);
		r.b = 0;
		snprintf(qi_how, sizeof qi_how, "N(a + b*%s) = a^2 - (%lld) b^2",
		    QD == -1 ? "i" : "sqrt(d)", (long long)QD);
		return r;
	}
	if (acc_id("conj") || acc_id("conjugate")) {
		expect_sym('(');
		r = qi_sum();
		expect_sym(')');
		r.b = -r.b;
		return r;
	}
	if (acc_id("gcd")) {
		struct qi y;
		expect_sym('(');
		r = qi_sum();
		expect_sym(',');
		y = qi_sum();
		expect_sym(')');
		snprintf(qi_how, sizeof qi_how, "Euclid with the quotient rounded to "
		    "the nearest lattice point (|N(r)| < |N(y)|)");
		qi_nops += 100;
		return qi_gcd(r, y);
	}
	rfail("expected an element of %s, found %s", QD == -1 ? "Z[i]" :
	    "Z[sqrt(d)]", here());
	return r;
}

static struct qi qi_unary(void);

static struct qi
qi_pow(void)
{
	struct qi b = qi_atom(), r = {1, 0};
	i128 e;

	if (!acc_sym('^'))
		return b;
	e = iexpr_unary();
	if (e < 0)
		rfail("negative powers: units only");
	while (e-- > 0)
		r = qi_mul(r, b);
	return r;
}

static struct qi
qi_unary(void)
{
	struct qi r;

	if (acc_sym('-')) {
		r = qi_unary();
		r.a = -r.a;
		r.b = -r.b;
		return r;
	}
	acc_sym('+');
	return qi_pow();
}

static struct qi
qi_term(void)
{
	struct qi x = qi_unary(), y, q, r;

	for (;;) {
		if (acc_sym('*') || (rti < rlim && (at_sym('(') || at_id("i") ||
		    at_id("sqrt")))) {
			x = qi_mul(x, qi_unary());
			qi_nops++;
		} else if (acc_sym('/')) {
			y = qi_unary();
			if (!qi_divmod(x, y, &q, &r)) {
				if (!qi_euclid())
					rfail("%s does not divide %s", qi_str(y), qi_str(x));
				note("%s / %s is not exact: the quotient %s (rounded), "
				    "remainder %s", qi_str(x), qi_str(y), qi_str(q), qi_str(r));
			}
			x = q;
			qi_nops++;
		} else if (acc_sym('%')) {
			y = qi_unary();
			if (!qi_euclid())
				rfail("remainders need a Euclidean ring");
			qi_divmod(x, y, &q, &r);
			x = r;
			qi_nops++;
		} else
			return x;
	}
}

static struct qi
qi_sum(void)
{
	struct qi x = qi_term(), y;

	for (;;) {
		if (acc_sym('+')) {
			y = qi_term();
			x.a = ck(x.a + y.a);
			x.b = ck(x.b + y.b);
		} else if (acc_sym('-')) {
			y = qi_term();
			x.a = ck(x.a - y.a);
			x.b = ck(x.b - y.b);
		} else
			return x;
	}
}

/* the fundamental unit of Z[sqrt d], d > 0 not a square: continued fraction */
static int
fund_unit(int64_t d, struct qi *u)
{
	i128 m = 0, dd = 1, a0 = (i128)nt_isqrt((u64)d), a = a0;
	i128 p0 = 1, p1 = a0, q0 = 0, q1 = 1;
	int k;

	for (k = 0; k < 2000; k++) {
		i128 n1 = p1 * p1 - (i128)d * q1 * q1;
		if (n1 == 1 || n1 == -1) {
			u->a = p1;
			u->b = q1;
			return 1;
		}
		m = dd * a - m;
		dd = ((i128)d - m * m) / dd;
		a = (a0 + m) / dd;
		{
			i128 p2 = a * p1 + p0, q2 = a * q1 + q0;
			if (p2 > ((i128)1 << 60) || q2 > ((i128)1 << 60))
				return 0;
			p0 = p1;
			p1 = p2;
			q0 = q1;
			q1 = q2;
		}
	}
	return 0;
}

static void
qi_factor_gauss(struct qi z)
{
	struct nt_fac f;
	struct qi rest = z, fac[64], q, r, prod = {1, 0};
	int e[64], k = 0, i, j;
	i128 n = qi_norm(z);

	if (n == 0)
		rfail("0 has no factorisation");
	if (n >> 63)
		rfail("the norm is too large to factor");
	nt_factor((u64)n, &f);
	for (i = 0; i < f.k; i++) {
		u64 p = f.p[i];
		struct qi pi;
		if (p == 2) {
			pi.a = 1;
			pi.b = 1;
		} else if (p % 4 == 3) {
			pi.a = (i128)p;
			pi.b = 0;
		} else {
			u64 t;
			nt_sqrt_prime(p - 1, p, &t);
			{
				struct qi P = {(i128)p, 0}, T = {(i128)t, 1};
				pi = qi_gcd(P, T);
			}
		}
		/* pi and its conjugate (for p = 1 mod 4) */
		for (j = 0; j < 2; j++) {
			struct qi c = j ? (struct qi){pi.a, -pi.b} : pi;
			int ex = 0;
			if (j && p % 4 != 1)
				break;
			while (qi_divmod(rest, c, &q, &r)) {
				rest = q;
				ex++;
			}
			if (ex) {
				fac[k] = c;
				e[k++] = ex;
			}
		}
	}
	printf("%s = ", qi_str(z));
	if (!(rest.a == 1 && rest.b == 0))
		printf("%s%s", qi_str(rest), k ? " * " : "");
	for (i = 0; i < k; i++) {
		printf("%s(%s)", i ? " * " : "", qi_str(fac[i]));
		if (e[i] > 1)
			printf("^%d", e[i]);
		for (j = 0; j < e[i]; j++)
			prod = qi_mul(prod, fac[i]);
	}
	printf(" in Z[i]\n");
	printf("  how: the norm %s = %s; 2 = -i (1 + i)^2 ramifies, p = 3 mod 4 "
	    "stays prime, p = 1 mod 4 splits as gcd(p, t + i) with t^2 = -1 mod "
	    "p and its conjugate\n", i128s(n), fac_text((u64)n));
	prod = qi_mul(prod, rest);
	printf("  check: the product is %s; %s is a unit (norm %s); each factor's "
	    "norm is a prime or the square of a prime = 3 mod 4\n", prod.a == z.a &&
	    prod.b == z.b ? "the element" : "NOT the element", qi_str(rest),
	    i128s(qi_norm(rest)));
}

static void
cmd_quadratic(struct rspec *s)
{
	int64_t d = s->d;
	int sq = d > 0 && (int64_t)nt_isqrt((u64)d) * (int64_t)nt_isqrt((u64)d)
	    == d;

	QD = d;
	if (at_end()) {
		if (d == -1) {
			printf("Z[i]: the Gaussian integers a + bi\n");
			printf("  norm N(a + bi) = a^2 + b^2 (multiplicative); units: 1, "
			    "-1, i, -i (norm 1)\n");
			printf("  a Euclidean domain (divide and round to the nearest "
			    "lattice point: |N(r)| <= N(y)/2), hence a PID and a UFD\n");
			printf("  primes: 1 + i (norm 2), p = 3 mod 4 (norm p^2), and a + bi "
			    "with a^2 + b^2 = p for p = 1 mod 4\n");
			printf("  characteristic 0; Z[i] = Z[x]/(x^2 + 1)\n");
			return;
		}
		printf("Z[sqrt(%lld)]: a + b sqrt(%lld)\n", (long long)d, (long long)d);
		if (sq) {
			printf("  %lld is a square: not an integral domain, (sqrt(%lld) - "
			    "%lld)(sqrt(%lld) + %lld) = 0\n", (long long)d, (long long)d,
			    (long long)nt_isqrt((u64)d), (long long)d,
			    (long long)nt_isqrt((u64)d));
			return;
		}
		printf("  norm N(a + b sqrt(%lld)) = a^2 - (%lld) b^2 "
		    "(multiplicative); an integral domain\n", (long long)d,
		    (long long)d);
		if (d < 0)
			printf("  units: 1, -1 only (a^2 + %lld b^2 = 1)\n",
			    (long long)-d);
		else {
			struct qi u;
			if (fund_unit(d, &u))
				printf("  units: +-u^k for the fundamental unit u = %s (norm "
				    "%s; from the continued fraction of sqrt(%lld), Pell's "
				    "equation)\n", qi_str(u), i128s(qi_norm(u)), (long long)d);
			else
				printf("  units: infinitely many (+-u^k); the fundamental unit "
				    "is too large to print\n");
		}
		if ((d % 4 + 4) % 4 == 1)
			printf("  not integrally closed: (1 + sqrt(%lld))/2 is a root of "
			    "x^2 - x - %lld but not in the ring, so not a UFD%s\n",
			    (long long)d, (long long)((d - 1) / 4), d == 5 ?
			    " (4 = 2*2 = (1 + sqrt(5))(-1 + sqrt(5)), all irreducible: "
			    "no element has norm +-2)" : "");
		else if (qi_euclid() || d == 6 || d == 7 || d == 11 || d == 19)
			printf("  norm-Euclidean, hence a PID and a UFD\n");
		else if (d < -2)
			printf("  not a UFD: the only imaginary quadratic Z[sqrt(d)] with "
			    "unique factorisation are d = -1 and -2%s\n", d == -5 ?
			    " (6 = 2*3 = (1 + sqrt(-5))(1 - sqrt(-5)))" : "");
		return;
	}
	if (d == -1 && acc_id("factor")) {
		struct qi z;
		int par = acc_sym('(');
		z = qi_sum();
		if (par)
			expect_sym(')');
		if (!at_end())
			rfail("unexpected %s", here());
		qi_factor_gauss(z);
		return;
	}
	{
		int a = rti;
		struct qi v;
		qi_nops = 0;
		qi_how[0] = 0;
		v = qi_sum();
		if (!at_end())
			rfail("unexpected %s", here());
		printf("%s = %s in %s\n", eq_text(a, rlim), qi_str(v), s->name);
		if (qi_how[0])
			printf("  how: %s\n", qi_how);
		printf("  norm %s\n", i128s(qi_norm(v)));
		/* check: the gcd divides both */
		rti = a;
		if (acc_id("gcd")) {
			struct qi x, y, q, r1, r2;
			expect_sym('(');
			x = qi_sum();
			expect_sym(',');
			y = qi_sum();
			printf("  check: %s divides %s (remainder %s) and %s (remainder "
			    "%s)\n", qi_str(v), qi_str(x), (qi_divmod(x, v, &q, &r1),
			    qi_str(r1)), qi_str(y), (qi_divmod(y, v, &q, &r2),
			    qi_str(r2)));
		}
		rti = rlim;
	}
}

/* ---- Q[x]/(f) ---- */

static int QN;            /* the modulus' degree */
static struct q QF[RMAXF + 1];

static void
qq(int ok)
{
	if (!ok)
		rfail("a rational number overflows 64 bits");
}

static const char *
qs(struct q a)
{
	static char buf[8][48];
	static int k;
	char *b = buf[k++ & 7];

	if (a.d == 1)
		snprintf(b, 48, "%lld", (long long)a.n);
	else
		snprintf(b, 48, "%lld/%lld", (long long)a.n, (long long)a.d);
	return b;
}

struct qe {
	struct q c[RMAXF];   /* coefficients of 1, x, ..., x^(QN-1) */
};

static void
qe_print(struct sbuf *s, const struct qe *e)
{
	int i, first = 1;

	for (i = QN - 1; i >= 0; i--) {
		struct q c = e->c[i];
		if (c.n == 0)
			continue;
		if (!first)
			sb_puts(s, c.n < 0 ? " - " : " + ");
		else if (c.n < 0)
			sb_puts(s, "-");
		first = 0;
		if (c.n < 0)
			c.n = -c.n;
		if (!(c.n == 1 && c.d == 1) || i == 0)
			sb_printf(s, "%s%s", qs(c), i ? "*" : "");
		if (i)
			sb_puts(s, "x");
		if (i > 1)
			sb_printf(s, "^%d", i);
	}
	if (first)
		sb_puts(s, "0");
}

static const char *
qe_str(const struct qe *e)
{
	static struct sbuf b[4];
	static int k;
	struct sbuf *s = &b[k++ & 3];

	sb_clear(s);
	sb_puts(s, "");
	qe_print(s, e);
	return sb_str(s);
}

static void
qe_zero(struct qe *e)
{
	int i;

	for (i = 0; i < RMAXF; i++) {
		e->c[i].n = 0;
		e->c[i].d = 1;
	}
}

static void
qe_mul(struct qe *r, const struct qe *a, const struct qe *b)
{
	struct q t[2 * RMAXF], p;
	int i, j;

	for (i = 0; i < 2 * QN; i++) {
		t[i].n = 0;
		t[i].d = 1;
	}
	for (i = 0; i < QN; i++)
		for (j = 0; j < QN; j++) {
			qq(q_mul(a->c[i], b->c[j], &p));
			qq(q_add(t[i + j], p, &t[i + j]));
		}
	/* reduce: x^QN = -(f0 + ... + f_{n-1} x^{n-1}) (f monic) */
	for (i = 2 * QN - 2; i >= QN; i--) {
		if (t[i].n == 0)
			continue;
		for (j = 0; j < QN; j++) {
			qq(q_mul(t[i], QF[j], &p));
			qq(q_sub(t[i - QN + j], p, &t[i - QN + j]));
		}
		t[i].n = 0;
	}
	for (i = 0; i < QN; i++)
		r->c[i] = t[i];
}

/* rational polynomials (degree <= QN) for the inverse */
struct qp {
	int d;
	struct q c[RMAXF + 1];
};

static void
qp_norm(struct qp *a)
{
	while (a->d >= 0 && a->c[a->d].n == 0)
		a->d--;
}

static void
qp_divmod(const struct qp *a, const struct qp *b, struct qp *q, struct qp *r)
{
	struct qp rr = *a, qq0;
	struct q li, c, p;
	int i, j;

	qq(q_div((struct q){1, 1}, b->c[b->d], &li));
	qq0.d = a->d - b->d;
	for (i = 0; i <= qq0.d; i++)
		qq0.c[i] = (struct q){0, 1};
	for (i = rr.d; i >= b->d; i--) {
		if (rr.c[i].n == 0)
			continue;
		qq(q_mul(rr.c[i], li, &c));
		qq0.c[i - b->d] = c;
		for (j = 0; j <= b->d; j++) {
			qq(q_mul(c, b->c[j], &p));
			qq(q_sub(rr.c[i - b->d + j], p, &rr.c[i - b->d + j]));
		}
	}
	rr.d = b->d - 1 < rr.d ? b->d - 1 : rr.d;
	qp_norm(&rr);
	if (qq0.d < 0)
		qq0.d = -1;
	qp_norm(&qq0);
	*q = qq0;
	*r = rr;
}

static void
qp_mulsub(struct qp *r, const struct qp *a, const struct qp *q,
    const struct qp *b)
{
	/* r = a - q b */
	struct qp t;
	struct q p;
	int i, j;

	t.d = q->d + b->d;
	if (q->d < 0 || b->d < 0)
		t.d = -1;
	for (i = 0; i <= t.d; i++)
		t.c[i] = (struct q){0, 1};
	for (i = 0; i <= q->d; i++)
		for (j = 0; j <= b->d; j++) {
			qq(q_mul(q->c[i], b->c[j], &p));
			qq(q_add(t.c[i + j], p, &t.c[i + j]));
		}
	r->d = a->d > t.d ? a->d : t.d;
	for (i = 0; i <= r->d; i++) {
		struct q x = i <= a->d ? a->c[i] : (struct q){0, 1};
		struct q y = i <= t.d ? t.c[i] : (struct q){0, 1};
		qq(q_sub(x, y, &r->c[i]));
	}
	qp_norm(r);
}

static int
qe_inv(const struct qe *a, struct qe *out)
{
	struct qp r0, r1, s0, s1, q, t;
	struct q li;
	int i;

	r0.d = QN;
	for (i = 0; i <= QN; i++)
		r0.c[i] = QF[i];
	r1.d = QN - 1;
	for (i = 0; i < QN; i++)
		r1.c[i] = a->c[i];
	qp_norm(&r1);
	if (r1.d < 0)
		return 0;
	s0.d = -1;
	s1.d = 0;
	s1.c[0] = (struct q){1, 1};
	while (r1.d > 0) {
		qp_divmod(&r0, &r1, &q, &t);
		r0 = r1;
		r1 = t;
		qp_mulsub(&t, &s0, &q, &s1);
		s0 = s1;
		s1 = t;
		if (r1.d < 0)
			return 0;   /* a common factor with f */
	}
	qq(q_div((struct q){1, 1}, r1.c[0], &li));
	qe_zero(out);
	for (i = 0; i <= s1.d; i++)
		qq(q_mul(s1.c[i], li, &out->c[i]));
	return 1;
}

static struct qe qe_sum(void);

static struct qe
qe_atom(void)
{
	struct qe r;

	qe_zero(&r);
	if (acc_sym('(')) {
		r = qe_sum();
		expect_sym(')');
		return r;
	}
	if (rti < rlim && rt[rti].k == RK_NUM) {
		if (rt[rti].big || (rt[rti].v >> 62))
			rfail("too large");
		r.c[0].n = (int64_t)rt[rti++].v;
		return r;
	}
	if (acc_id("x")) {
		if (QN == 1) {
			r.c[0].n = -QF[0].n;
			r.c[0].d = QF[0].d;
		} else
			r.c[1].n = 1;
		return r;
	}
	rfail("expected an element of Q[x]/(f), found %s", here());
	return r;
}

static struct qe
qe_pow(void)
{
	struct qe b = qe_atom(), r;
	i128 e;

	if (!acc_sym('^'))
		return b;
	e = iexpr_unary();
	if (e < 0) {
		if (!qe_inv(&b, &b))
			rfail("%s has no inverse: it shares a factor with the modulus",
			    qe_str(&b));
		e = -e;
	}
	qe_zero(&r);
	r.c[0].n = 1;
	while (e-- > 0)
		qe_mul(&r, &r, &b);
	return r;
}

static struct qe
qe_unary(void)
{
	struct qe r;
	int i;

	if (acc_sym('-')) {
		r = qe_unary();
		for (i = 0; i < QN; i++)
			r.c[i].n = -r.c[i].n;
		return r;
	}
	acc_sym('+');
	return qe_pow();
}

static struct qe
qe_term(void)
{
	struct qe x = qe_unary(), y;

	for (;;) {
		if (acc_sym('*') || (rti < rlim && (at_sym('(') || at_id("x")))) {
			y = qe_unary();
			qe_mul(&x, &x, &y);
		} else if (acc_sym('/')) {
			y = qe_unary();
			if (!qe_inv(&y, &y))
				rfail("%s has no inverse", qe_str(&y));
			qe_mul(&x, &x, &y);
		} else
			return x;
	}
}

static struct qe
qe_sum(void)
{
	struct qe x = qe_term(), y;
	int i;

	for (;;) {
		int sub;
		if (acc_sym('+'))
			sub = 0;
		else if (acc_sym('-'))
			sub = 1;
		else
			return x;
		y = qe_term();
		for (i = 0; i < QN; i++)
			qq((sub ? q_sub : q_add)(x.c[i], y.c[i], &x.c[i]));
	}
}

/* Q[x]/(f): f with rational (integer) coefficients, made monic */
static int
parse_qx(struct rspec *s)
{
	struct ring B;
	struct rpoly m;
	int save = rti, i;

	if (!acc_sym('[') || !acc_id("x") || !acc_sym(']') || !acc_sym('/'))
		return 0;
	(void)save;
	{
		/* integer coefficients: read over Z/(2^61 - 1) and map back */
		u64 P = 2305843009213693951ULL;
		int par = acc_sym('(');
		ring_zn(&B, P);
		pexpr_into(&B, &m, "x");
		if (par)
			expect_sym(')');
		if (m.d < 1 || m.d > RMAXF)
			rfail("Q[x]/(f): f of degree 1..%d", RMAXF);
		s->qd = m.d;
		for (i = 0; i <= m.d; i++) {
			int64_t v = m.c[i] > P / 2 ? -(int64_t)(P - m.c[i]) :
			    (int64_t)m.c[i];
			if (v > ((int64_t)1 << 40) || v < -((int64_t)1 << 40))
				rfail("Q[x]/(f): coefficients up to 2^40");
			s->qf[i].n = v;
			s->qf[i].d = 1;
		}
		for (i = 0; i <= m.d; i++)
			qq(q_div(s->qf[i], (struct q){s->qf[m.d].n, 1}, &s->qf[i]));
	}
	s->kind = RS_QX;
	{
		struct qe f;
		QN = s->qd + 1;
		qe_zero(&f);
		for (i = 0; i <= s->qd; i++)
			f.c[i] = s->qf[i];
		QN = s->qd + 1;
		snprintf(s->name, sizeof s->name, "Q[x]/(%s)", qe_str(&f));
	}
	return 1;
}

/* is f irreducible over Q: rational roots, then factor degrees mod p */
static void
qx_irreducible(const struct rspec *s)
{
	int n = s->qd, i, d, allowed[RMAXF + 1], nprimes = 0;
	int64_t z[RMAXF + 1], l = 1;
	int untested = 0;
	u64 p;

	for (i = 0; i <= n; i++)
		l = l / (int64_t)gcd64(l, s->qf[i].d) * s->qf[i].d;
	for (i = 0; i <= n; i++)
		z[i] = s->qf[i].n * (l / s->qf[i].d);
	for (d = 0; d <= n; d++)
		allowed[d] = 1;
	/* rational roots p/q: p | z0, q | zn */
	if (z[0] == 0) {
		printf("  a field: no: x divides the modulus, so x is a zero divisor\n");
		return;
	}
	{
		u64 a0 = (u64)(z[0] < 0 ? -z[0] : z[0]), an = (u64)(z[n] < 0 ? -z[n] :
		    z[n]), pp, qq2;
		if (a0 > (1u << 20) || an > (1u << 20))
			untested = 1;
		else
			for (pp = 1; pp <= a0; pp++)
				if (a0 % pp == 0)
					for (qq2 = 1; qq2 <= an; qq2++)
						if (an % qq2 == 0 && nt_gcd(pp, qq2) == 1) {
							int sg;
							for (sg = -1; sg <= 1; sg += 2) {
								int ok = 1;
								{
									i128 v = 0;
									for (i = 0; i <= n && ok; i++) {
										i128 t = z[i];
										int j;
										for (j = 0; j < i && ok; j++)
											ok = !__builtin_mul_overflow(t,
											    sg * (i128)pp, &t);
										for (j = 0; j < n - i && ok; j++)
											ok = !__builtin_mul_overflow(t,
											    (i128)qq2, &t);
										if (ok)
											ok = !__builtin_add_overflow(v,
											    t, &v);
									}
									if (!ok)
										untested = 1;
									if (ok && v == 0) {
										char rs[64];
										if (qq2 > 1)
											snprintf(rs, sizeof rs, "%s%llu/"
											    "%llu", sg < 0 ? "-" : "",
											    U(pp), U(qq2));
										else
											snprintf(rs, sizeof rs, "%s%llu",
											    sg < 0 ? "-" : "", U(pp));
										printf("  a field: no: x = %s is a "
										    "rational root, so x - (%s) "
										    "divides the modulus\n", rs, rs);
										return;
									}
								}
							}
						}
	}
	if (untested) {
		printf("  a field: not decided: the rational root test overflows "
		    "for these coefficients\n");
		return;
	}
	if (n <= 3) {
		printf("  a field: yes: the modulus has degree %d and no rational "
		    "root, so it is irreducible over Q\n", n);
		return;
	}
	/* factor degrees mod small primes: a factor over Q of degree d needs a
	 * subset of the factor degrees mod p summing to d, for every p */
	for (p = 3; p < 400 && nprimes < 12; p += 2) {
		struct ring B;
		struct rpoly f, fd, g;
		static struct rpoly fac[RMAXDEG + 1];
		int mult[RMAXDEG + 1], k, j, can[RMAXF + 1];
		u64 lead;
		if (!nt_isprime(p) || (u64)((z[n] % (int64_t)p + (int64_t)p) %
		    (int64_t)p) == 0)
			continue;
		ring_zn(&B, p);
		f.d = n;
		for (i = 0; i <= n; i++)
			f.c[i] = (u64)((z[i] % (int64_t)p + (int64_t)p) % (int64_t)p);
		rp_deriv(&B, &fd, &f);
		rp_gcd(&B, &g, &f, &fd);
		if (g.d != 0)
			continue;
		nprimes++;
		k = rp_factor(&B, &f, fac, mult, RMAXDEG + 1, &lead);
		memset(can, 0, sizeof can);
		can[0] = 1;
		for (j = 0; j < k; j++)
			for (d = n; d >= fac[j].d; d--)
				if (can[d - fac[j].d])
					can[d] = 1;
		for (d = 1; d < n; d++)
			allowed[d] &= can[d];
		if (k == 1) {
			printf("  a field: yes: the modulus is irreducible mod %llu (a "
			    "prime not dividing its leading coefficient), so it is "
			    "irreducible over Q\n", U(p));
			return;
		}
	}
	for (d = 1; d < n; d++)
		if (allowed[d])
			break;
	if (d == n)
		printf("  a field: yes: the modulus is irreducible over Q (no factor "
		    "degree is possible mod all of %d primes)\n", nprimes);
	else
		printf("  a field: not decided: no rational root, but a factor of "
		    "degree %d is possible mod each of %d primes (x^4 + 1 is like "
		    "this and irreducible; a full factorisation over Q is not done "
		    "here)\n", d, nprimes);
}

static void
cmd_qx(struct rspec *s)
{
	int i;

	QN = s->qd;
	for (i = 0; i <= s->qd; i++)
		QF[i] = s->qf[i];
	if (at_end() || at_id("field") || at_id("domain")) {
		printf("%s\n", s->name);
		printf("  a vector space of dimension %d over Q, basis 1, x, ..., "
		    "x^%d; characteristic 0\n", QN, QN - 1);
		qx_irreducible(s);
		rti = rlim;
		return;
	}
	{
		int a = rti;
		struct qe v, w;
		v = qe_sum();
		if (!at_end())
			rfail("unexpected %s", here());
		printf("%s = %s in %s\n", eq_text(a, rlim), qe_str(&v), s->name);
		printf("  how: polynomial arithmetic, reduced with x^%d = -(the lower "
		    "terms of the modulus); inverses by extended Euclid over Q\n", QN);
		{
			int j, nz = 0;
			for (j = 0; j < QN; j++)
				nz |= v.c[j].n != 0;
			if (nz && qe_inv(&v, &w)) {
				struct qe one;
				qe_mul(&one, &v, &w);
				printf("  its inverse: %s; check: the product is %s\n",
				    qe_str(&w), qe_str(&one));
			}
		}
	}
}
