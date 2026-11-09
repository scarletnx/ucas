/* SPDX-License-Identifier: GPL-3.0-or-later */
/* ecff_cmd.c, part 6: binary curves */

static const char *
b2str(const struct b2f *K, const u64 *a)
{
	nn v;

	nn_zero(&v);
	memcpy(v.w, a, sizeof(u64) * (size_t)K->n);
	return nn_hex(&v);
}

/* ecb2 NAME [mul k] | ecb2 [a, b] over GF(2^m) poly e1 e2 ... [point (x, y)] */
static void
cmd_ecb2(void)
{
	static struct b2f K;
	struct b2c E;
	struct b2pt G, T;
	nn n, h, N, k;
	int named = 0, haveg = 0, i;
	char w[64];
	const char *save;
	const struct b2named *B = NULL;
	zz tm;

	skipws();
	save = sp;
	if (getword(w, sizeof w) && (B = b2named_find(w)) != NULL) {
		nn v;
		b2_init(&K, B->m, B->ex, B->nex);
		E.K = &K;
		nn_parse(B->a, &v, NULL);
		b2_from_nn(&K, E.a, &v);
		nn_parse(B->b, &v, NULL);
		b2_from_nn(&K, E.b, &v);
		nn_parse(B->gx, &v, NULL);
		b2_from_nn(&K, G.x, &v);
		nn_parse(B->gy, &v, NULL);
		b2_from_nn(&K, G.y, &v);
		G.inf = 0;
		nn_parse(B->n, &n, NULL);
		nn_parse(B->h, &h, NULL);
		named = haveg = 1;
	} else {
		nn a, b, m;
		int ex[5], nex = 0;
		sp = save;
		skipws();
		if (*sp != '[')
			fail("ecb2 NAME (K-163, B-163, K-233) or ecb2 [a, b] over "
			    "GF(2^m) poly e1 e2 ...");
		sp++;
		need_nn(&a, "a");
		need_nn(&b, "b");
		skipws();
		if (*sp == ']')
			sp++;
		takeword("over");
		skipws();
		if (strncasecmp(sp, "GF(2^", 5))
			fail("expected GF(2^m)");
		sp += 5;
		need_nn(&m, "m");
		skipws();
		if (*sp == ')')
			sp++;
		if (!takeword("poly"))
			fail("expected poly e1 e2 ... (f = x^m + x^e1 + x^e2 + ...)");
		while (nex < 5) {
			nn e;
			skipws();
			if (!isdigit((unsigned char)*sp))
				break;
			need_nn(&e, "an exponent");
			ex[nex++] = (int)e.w[0];
		}
		b2_init(&K, (int)m.w[0], ex, nex);
		E.K = &K;
		b2_from_nn(&K, E.a, &a);
		b2_from_nn(&K, E.b, &b);
	}
	printf("y^2 + x y = x^3 + %s x^2 + %s over GF(2^%d)\n", b2str(&K, E.a),
	    b2str(&K, E.b), K.m);
	{
		char fb[200];
		size_t o = (size_t)snprintf(fb, sizeof fb, "x^%d", K.m);
		for (i = 0; i < K.nex; i++)
			o += (size_t)snprintf(fb + o, sizeof fb - o, K.ex[i] ? " + x^%d"
			    : " + 1", K.ex[i]);
		printf("  field: GF(2)[x]/(%s)\n", fb);
	}
	if (named)
		printf("  standard: FIPS 186-4 %s\n", B->name);
	if (b2_is_zero(&K, E.b))
		fail("b = 0: singular");
	if (b2_count_koblitz(&E, &N, &tm)) {
		printf("  order: #E = %s\n", nn_str(&N));
		printf("  how: Koblitz curve (a, b in GF(2)): #E = 2^m + 1 - t_m, "
		    "t_k+1 = t_1 t_k - 2 t_k-1 from #E(F_2); t_m = %s\n",
		    zz_str(&tm));
		if (K.m <= 20) {
			nn N2;
			b2_count_naive(&E, &N2);
			printf("  check: the naive count agrees: %s\n", nn_eq(&N, &N2) ?
			    "yes" : "NO");
		}
	} else if (b2_count_naive(&E, &N)) {
		printf("  order: #E = %s\n", nn_str(&N));
		printf("  how: naive: 2 + 2 #{x != 0 : Tr(x + a + b/x^2) = 0}\n");
	} else
		nn_zero(&N);
	if (haveg) {
		printf("  G: (%s, %s)\n", b2str(&K, G.x), b2str(&K, G.y));
		printf("  check: G on the curve: %s\n", b2_on(&E, &G) ? "yes" :
		    "NO");
		printf("  check: n = %s prime (BPSW): %s\n", nn_str(&n),
		    nn_isprime(&n) ? "yes" : "NO");
		b2_mul_pt(&E, &T, &G, &n);
		printf("  check: n G = O: %s\n", T.inf ? "yes" : "NO");
		if (!nn_is_zero(&N)) {
			nn hn;
			nn_mul(&hn, &h, &n);
			printf("  check: #E = h n (h = %s): %s\n", nn_str(&h),
			    nn_eq(&hn, &N) ? "yes" : "NO");
		}
	}
	skipws();
	if (haveg && takeword("mul")) {
		need_nn(&k, "k");
		b2_mul_pt(&E, &T, &G, &k);
		if (T.inf)
			printf("  k G: O\n");
		else
			printf("  k G: (%s, %s)\n", b2str(&K, T.x), b2str(&K, T.y));
		printf("  check: k G on the curve: %s\n", b2_on(&E, &T) ? "yes" :
		    "NO");
	}
}
