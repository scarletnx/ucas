/* ecff_cmd.c, part 4: CM, Hilbert class polynomials, prime-order curves */

static int64_t
get_disc(void)
{
	zz D;
	int ok;
	int64_t d;

	skipws();
	if (takeword("D"))
		skipws();
	if (*sp == '=')
		sp++;
	if (!getzz(&D))
		fail("expected a discriminant D < 0");
	d = zz_to_i64(&D, &ok);
	if (!ok || d >= 0 || (((d % 4) + 4) % 4) > 1 || d < -10000000)
		fail("D must be a negative discriminant (D = 0 or 1 mod 4)");
	return d;
}

static void
cmd_hilbert(void)
{
	int64_t D = get_disc(), fa[64], fb[64];
	zz c[260];
	int h, e, i;
	size_t o = 0;
	static char buf[60000];

	h = cm_hilbert(D, c, 260, &e);
	if (h <= 0)
		fail("hilbert: class number too large or D not a discriminant");
	for (i = h; i >= 0; i--) {
		const char *s;
		if (zz_is_zero(&c[i]))
			continue;
		s = zz_str(&c[i]);
		if (i == h)
			o += (size_t)snprintf(buf + o, sizeof buf - o, "x^%d", h);
		else {
			zz a = c[i];
			a.neg = 0;
			o += (size_t)snprintf(buf + o, sizeof buf - o, " %c %s",
			    c[i].neg ? '-' : '+', zz_str(&a));
			if (i > 0)
				o += (size_t)snprintf(buf + o, sizeof buf - o, "%s",
				    i > 1 ? "" : "x");
			if (i > 1)
				o += (size_t)snprintf(buf + o, sizeof buf - o, "x^%d",
				    i);
		}
		(void)s;
	}
	if (h == 1 && buf[0] == 'x' && buf[1] == '^')
		memmove(buf + 1, buf + 3, strlen(buf + 3) + 1);
	printf("H_%lld(x) = %s\n", (long long)D, buf);
	printf("  class number: %d\n", h);
	{
		int n = cm_forms(D, fa, fb, 64);
		o = 0;
		for (i = 0; i < n && i < 64; i++)
			o += (size_t)snprintf(buf + o, sizeof buf - o, "%s(%lld, %lld, "
			    "%lld)", i ? ", " : "", (long long)fa[i], (long long)fb[i],
			    (long long)((fb[i] * fb[i] - D) / (4 * fa[i])));
		printf("  reduced forms: %s\n", buf);
	}
	printf("  how: the product of x - j((-b + sqrt D)/(2a)) over the reduced "
	    "forms, j = E4^3 / Delta by q-series in multiprecision fixed point\n");
	printf("  check: every coefficient within 2^-%d of an integer (imaginary "
	    "parts too)%s\n", e, e >= 20 ? "" : " -- NOT RELIABLE");
}

/* a CM curve over F_p: 4p = t^2 + |D| v^2, H_D(j) = 0, the twist with
 * p + 1 - t points */
static int
cm_build(int64_t D, const nn *p, const zz *want, zz *tout, char *why,
    size_t wn)
{
	nn x, y;
	struct px H;
	u64 roots[64 * FW];
	int nr, i, sgn;

	memset(&cc, 0, sizeof cc);
	if (!mf_init(CF, p) || !mf_prime_setup(CF)) {
		snprintf(why, wn, "p is not prime");
		return 0;
	}
	if (!cornacchia4(p, (u64)-D, &x, &y)) {
		snprintf(why, wn, "4p = t^2 + %lld v^2 has no solution",
		    (long long)-D);
		return 0;
	}
	px_init(&H);
	if (!cm_hilbert_mod(CF, D, &H)) {
		snprintf(why, wn, "no Hilbert class polynomial");
		return 0;
	}
	nr = px_roots(CF, &H, roots, 64);
	px_free(&H);
	if (nr == 0) {
		snprintf(why, wn, "H_D has no root mod p");
		return 0;
	}
	for (i = 0; i < nr; i++)
		for (sgn = 0; sgn < 2; sgn++) {
			zz t;
			zz_from_nn(&t, &x, sgn);
			if (want && zz_cmp(want, &t))
				continue;
			if (cm_curve_for(CF, roots + (size_t)i * FW, &t, CE)) {
				*tout = t;
				curve_desc();
				return 1;
			}
		}
	snprintf(why, wn, "no twist with trace +-%s", nn_str(&x));
	return 0;
}

static void
print_cm_result(int64_t D, const zz *t, const char *how)
{
	nn N, p1;
	zz P1, Nz;
	struct eccert C;
	struct nnfac f;
	char fb[2000];
	u64 j[FW];

	nn_add_u64(&p1, &CF->P, 1);
	zz_from_nn(&P1, &p1, 0);
	zz_sub(&Nz, &P1, t);
	N = Nz.m;
	ecp_jinv(CE, j);
	printf("%s\n", cc.desc);
	print_short_curve();
	printf("  CM: discriminant %lld (class number %d), j = %s a root of "
	    "H_D mod p\n", (long long)D, cm_classno(D), fe(j));
	printf("  trace: %s\n  order: #E = %s\n", zz_str(t), nn_str(&N));
	nn_factor(&N, &f, 1);
	nnfac_print(fb, sizeof fb, &f);
	printf("  factored: %s\n", fb);
	printf("  how: %s\n", how);
	if (ec_certify(CE, &N, &C))
		print_cert(&C, &N);
	else
		printf("  check: certification FAILED\n");
}

static void
cmd_eccm(void)
{
	int64_t D = get_disc();
	nn p;
	zz t;
	char why[300];

	skipws();
	if (takeword("bits")) {
		nn b, tb, vb, two;
		int bits, tries;
		need_nn(&b, "the size in bits");
		bits = (int)b.w[0];
		if (bits < 16 || bits > 512)
			fail("eccm: 16 to 512 bits");
		if (D % 2 == 0 || ((-D) % 8) == 7)
			fail("eccm: with D = %lld every order p + 1 - t is even (2 "
			    "divides it); prime orders need D = 5 mod 8 (-3, -11, -19, "
			    "-35, -43, -51, -67, ...)", (long long)D);
		/* t < 2^(bits/2 + 1), v < sqrt(2^(bits + 2) / |D|): p = (t^2 +
		 * |D| v^2)/4 of the given size, p and p + 1 -+ t prime */
		nn_set(&two, 1);
		nn_shl(&tb, &two, bits / 2 + 1);
		nn_shl(&vb, &two, bits + 2);
		nn_div_u64(&vb, &vb, (u64)-D);
		nn_sqrt(&vb, &vb);
		for (tries = 0; tries < 3000000; tries++) {
			nn tt, v, q, N, p1;
			zz want;
			int sgn;
			nn_rand_below(&tt, &tb);
			nn_rand_below(&v, &vb);
			nn_mul(&p, &tt, &tt);
			nn_mul(&q, &v, &v);
			nn_mul_u64(&q, &q, (u64)-D);
			nn_add(&p, &p, &q);
			if (p.w[0] & 3)
				continue;
			nn_shr(&p, &p, 2);
			if (nn_bits(&p) != bits || !(p.w[0] & 1) || !nn_isprime(&p))
				continue;
			nn_add_u64(&p1, &p, 1);
			for (sgn = 0; sgn < 2; sgn++) {
				if (sgn)
					nn_add(&N, &p1, &tt);
				else
					nn_sub(&N, &p1, &tt);
				if (nn_isprime(&N))
					break;
			}
			if (sgn == 2)
				continue;
			zz_from_nn(&want, &tt, sgn);
			if (!cm_build(D, &p, &want, &t, why, sizeof why))
				continue;
			print_cm_result(D, &t, "CM method: random t, v with p = (t^2 + "
			    "|D| v^2)/4 and #E = p + 1 - t both prime, then a root of "
			    "H_D mod p and the twist with that trace");
			return;
		}
		fail("eccm: no prime-order curve found");
	}
	takeword("mod") || takeword("over") || takeword("p");
	need_nn(&p, "p (or bits B)");
	if (!cm_build(D, &p, NULL, &t, why, sizeof why))
		fail("eccm: %s", why);
	print_cm_result(D, &t, "4p = t^2 + |D| v^2 (Cornacchia), j a root of "
	    "H_D mod p, the twist with p + 1 - t points");
}

static void
cmd_ecpp(void)
{
	static struct ecppcert C;
	nn N;
	char why[600];
	int s, ok;
	u64 t0 = ecff_ms();

	need_nn(&N, "the number to prove prime");
	if (nn_cmp_u64(&N, 2) < 0)
		fail("ecpp: N >= 2");
	if (nn_fits_u64(&N)) {
		printf("%s is %s\n", nn_str(&N), nt_isprime(N.w[0]) ? "prime" :
		    "composite");
		printf("  how: deterministic Miller-Rabin (exact below 2^64)\n");
		return;
	}
	if (!nn_isprime(&N)) {
		printf("%s is composite\n", nn_str(&N));
		printf("  how: BPSW (a failed Fermat or Lucas test is a proof of "
		    "compositeness)\n");
		return;
	}
	if (!ecpp_prove(&N, &C, ECPPMAX))
		fail("ecpp: no certificate found (no suitable discriminant with "
		    "class number <= 10)");
	printf("%s is prime\n", nn_str(&N));
	printf("  how: Atkin-Morain ECPP, %d steps down to %s\n", C.n,
	    nn_str(&C.last));
	for (s = 0; s < C.n; s++) {
		const struct ecppstep *S = &C.step[s];
		printf("  step %d: N = %s, D = %lld, E: y^2 = x^3 + %s x + %s, m = "
		    "%s, q = %s, P = (%s, %s)\n", s + 1, nn_str(&S->N),
		    (long long)S->D, nn_str(&S->a), nn_str(&S->b), nn_str(&S->m),
		    nn_str(&S->q), nn_str(&S->x), nn_str(&S->y));
	}
	ok = ecpp_check(&C, &N, why, sizeof why);
	printf("  check: the certificate verifies (independent checker): %s\n",
	    ok ? why : "NO");
	if (!ok)
		printf("  why: %s\n", why);
	if (timed)
		printf("  time: %llu ms\n", (unsigned long long)(ecff_ms() - t0));
}
