/* SPDX-License-Identifier: GPL-3.0-or-later */
/* ecff_cmd.c, part 3: discrete logs, weak curves, pairings */

/* class-number-1 discriminants and their j (for anomalous curves) */
static const struct {
	int D;
	const char *j;
} cm1[] = {
	{-11, "-32768"}, {-19, "-884736"}, {-43, "-884736000"},
	{-67, "-147197952000"}, {-163, "-262537412640768000"}, {0, NULL}
};

/* y^2 = x^3 + 3k x + 2k, k = j / (1728 - j): a curve with invariant j */
static void
curve_from_j(const struct mf *F, const u64 *j, u64 *a, u64 *b)
{
	u64 t[FW], k[FW];

	mf_from_u64(F, t, 1728);
	mf_sub(F, t, t, j);
	mf_inv(F, t, t, NULL);
	mf_mul(F, k, j, t);
	mf_mul_u64(F, a, k, 3);
	mf_mul_u64(F, b, k, 2);
}

/* the curve "anomalous BITS": CM by D = -11 ..., p = (1 + |D| v^2)/4,
 * the twist with trace 1 (#E = p) */
static void
gen_anomalous(int bits)
{
	int di;
	nn v, p, t;

	if (bits < 16 || bits > 280)
		fail("anomalous: 16 to 280 bits");
	for (di = 0; cm1[di].D; di++) {
		u64 D = (u64)-cm1[di].D;
		/* v^2 = 3 * 2^bits / D puts p = (1 + D v^2)/4 near 0.75 * 2^bits */
		nn_set(&v, 3);
		nn_shl(&v, &v, bits);
		nn_div_u64(&v, &v, D);
		nn_sqrt(&v, &v);
		v.w[0] |= 1;
		(void)t;
		for (int tries = 0; tries < 4000; tries++, nn_add_u64(&v, &v, 2)) {
			struct ecp_curve T;
			struct ecp_pt P, R;
			u64 j[FW], a[FW], b[FW];
			zz jz;
			int tw;
			nn_mul(&p, &v, &v);
			nn_mul_u64(&p, &p, D);
			nn_add_u64(&p, &p, 1);
			nn_shr(&p, &p, 2);
			if (nn_bits(&p) != bits || !nn_isprime(&p))
				continue;
			memset(&cc, 0, sizeof cc);
			mf_init(CF, &p);
			mf_prime_setup(CF);
			zz_parse(cm1[di].j, &jz, NULL);
			mf_from_zz(CF, j, &jz);
			curve_from_j(CF, j, a, b);
			ecp_set(CE, CF, a, b);
			for (tw = 0; tw < 2; tw++) {
				if (tw) {
					ec_twist(CE, &T, NULL);
					ecp_set(CE, CF, T.a, T.b);
				}
				ecp_random_point(CE, &P);
				ecp_mul(CE, &R, &P, &p);
				if (R.inf) {
					curve_desc();
					cc.L.order = p;
					cc.known = 1;
					return;
				}
			}
		}
	}
	fail("anomalous: no prime found");
}

/* "supersingular BITS": y^2 = x^3 + x over p = 4 r - 1 (p, r prime):
 * #E = p + 1 = 4 r, embedding degree 2 */
static void
gen_supersingular(int bits)
{
	nn r, p;

	if (bits < 8 || bits > 500)
		fail("supersingular: 8 to 500 bits");
	nn_set(&r, 1);
	nn_shl(&r, &r, bits - 3);
	nn_add_u64(&r, &r, 1);
	for (;;) {
		nn_nextprime(&r, &r);
		nn_shl(&p, &r, 2);
		nn_sub_u64(&p, &p, 1);
		if (nn_isprime(&p))
			break;
		nn_add_u64(&r, &r, 2);
	}
	{
		zz o, z;
		zz_set_i64(&o, 1);
		zz_set_i64(&z, 0);
		custom_curve(&z, &z, &z, &o, &z, &p);
	}
	curve_desc();
	nn_add_u64(&cc.L.order, &p, 1);
	cc.known = 1;
}

static int
curve_generated(void)
{
	char w[32];
	const char *save = sp;
	nn b;

	if (getword(w, sizeof w) && (!strcasecmp(w, "anomalous") ||
	    !strcasecmp(w, "supersingular"))) {
		need_nn(&b, "the size in bits");
		if (w[0] == 'a')
			gen_anomalous((int)b.w[0]);
		else
			gen_supersingular((int)b.w[0]);
		return 1;
	}
	sp = save;
	return 0;
}

static void
print_short_curve(void)
{
	printf("  curve: y^2 = x^3 + %s x + %s mod %s\n", mf_str(CF, CE->a),
	    mf_str(CF, CE->b), nn_str(&CF->P));
}

static void
cmd_ecweak(void)
{
	nn N, p1;
	struct eccert C;
	char how[300];

	if (!curve_generated())
		fail("ecweak anomalous BITS | ecweak supersingular BITS");
	if (!count(CM_AUTO, &N, &C, how, sizeof how))
		fail("could not count the points");
	printf("%s\n", cc.desc);
	print_short_curve();
	printf("  order: #E = %s\n", nn_str(&N));
	nn_add_u64(&p1, &CF->P, 1);
	if (nn_eq(&N, &CF->P))
		printf("  weak: anomalous (#E = p): Smart's attack maps E(F_p) "
		    "onto (F_p, +) through the p-adic logarithm\n");
	else if (nn_eq(&N, &p1))
		printf("  weak: supersingular (#E = p + 1): embedding degree 2, "
		    "MOV maps the group into F_p^2^*\n");
	print_cert(&C, &N);
}

static void
cmd_ecdlog(void)
{
	struct ecp_pt P, Q, T;
	nn N, ord, k, p1, sec;
	struct nnfac fo;
	struct eccert C;
	char how[300], w[16], info[600], plog[2000];
	int method = 0, hasq = 1;    /* 0 auto 1 bsgs 2 rho 3 ph 4 mov 5 smart */
	zz t, P1, NN;
	u64 iters = 0;

	if (!curve_generated())
		parse_curve();
	parse_point(&P, 1);
	skipws();
	if (takeword("secret")) {
		need_nn(&sec, "the secret multiplier");
		ecp_mul(CE, &Q, &P, &sec);
		hasq = 0;
	} else
		parse_point(&Q, 1);
	skipws();
	if (takeword("using") && getword(w, sizeof w)) {
		static const char *const mn[] = {"auto", "bsgs", "rho", "ph",
			"mov", "smart", NULL};
		for (method = 0; mn[method] && strcasecmp(mn[method], w); method++)
			;
		if (!mn[method])
			fail("methods: bsgs, rho, ph (Pohlig-Hellman), mov, smart");
	}
	if (!count(CM_AUTO, &N, &C, how, sizeof how))
		fail("could not count the points");
	{
		struct nnfac fN;
		nn_factor(&N, &fN, 1);
		ec_point_order(CE, &P, &N, &fN, &ord);
	}
	nn_factor(&ord, &fo, 1);
	nn_add_u64(&p1, &CF->P, 1);
	zz_from_nn(&P1, &p1, 0);
	zz_from_nn(&NN, &N, 0);
	zz_sub(&t, &P1, &NN);
	if (method == 0) {
		if (nn_eq(&N, &CF->P))
			method = 5;
		else if (fo.k && nn_bits(&fo.p[fo.k - 1]) > 60 &&
		    ec_embedding_degree(&CF->P, &fo.p[fo.k - 1], 6) > 0)
			method = 4;
		else
			method = 3;
	}
	info[0] = 0;
	if ((method == 2 || method == 4) && (fo.k != 1 || fo.e[0] != 1)) {
		/* the largest prime's subgroup: P' = (ord/r) P, k mod r */
		nn c, r = fo.p[fo.k - 1];
		nn_divmod(&c, NULL, &ord, &r);
		ecp_mul(CE, &P, &P, &c);
		ecp_mul(CE, &Q, &Q, &c);
		printf("  note: projected to the subgroup of prime order r = %s "
		    "(P, Q times %s): k is found mod r\n", nn_str(&r), nn_str(&c));
		ord = r;
		nn_factor(&ord, &fo, 1);
	}
	switch (method) {
	case 1:
		if (!ec_dlog_bsgs(CE, &P, &Q, &ord, &k))
			fail("no k with k P = Q");
		snprintf(how, sizeof how, "baby-step giant-step: %s babies jP, "
		    "giants Q - i m P", "sqrt(n)");
		break;
	case 2:
		if (fo.k != 1 || fo.e[0] != 1)
			fail("rho here wants ord(P) prime (use ph)");
		if (!ec_dlog_rho(CE, &P, &Q, &ord, &k, &iters))
			fail("rho found no useful collision");
		snprintf(how, sizeof how, "Pollard rho: 20-adding walk, Floyd "
		    "cycle finding, a collision after %llu steps",
		    (unsigned long long)iters);
		break;
	case 3:
		if (!ec_dlog_ph(CE, &P, &Q, &ord, &fo, &k, plog, sizeof plog))
			fail("Pohlig-Hellman failed (no k, or a factor too large)");
		snprintf(how, sizeof how, "Pohlig-Hellman over ord(P) = %s: %s, "
		    "then CRT", nn_str(&ord), plog);
		break;
	case 4: {
		int kd;
		if (fo.k != 1 || fo.e[0] != 1)
			fail("mov here wants ord(P) prime");
		if (!ec_mov(CE, &P, &Q, &ord, &t, &k, &kd, info, sizeof info))
			fail("mov: the embedding degree of %s is not small (or the "
			    "log in F_p^k failed)", nn_str(&ord));
		snprintf(how, sizeof how, "MOV: the Weil pairing e_r(., R) maps "
		    "<P> into the r-th roots of unity of F_p^%d (embedding degree "
		    "%d), where baby-step giant-step finds the log", kd, kd);
		break;
	}
	case 5:
		if (!nn_eq(&N, &CF->P))
			fail("smart: the curve is not anomalous (#E = %s != p)",
			    nn_str(&N));
		if (!ec_smart(CE, &P, &Q, &k))
			fail("smart: no suitable lift found");
		snprintf(how, sizeof how, "Smart: lift to Z/p^2, p P' lies in "
		    "the kernel of reduction, psi(P) = -x/(p y) of p P' mod p is "
		    "a homomorphism E(F_p) -> F_p; k = psi(Q) / psi(P)");
		break;
	}
	ecp_mul(CE, &T, &P, &k);
	printf("k = %s\n", nn_str(&k));
	printf("  curve: %s\n", cc.desc);
	printf("  P: %s\n  Q: %s\n", pt(&P), pt(&Q));
	printf("  order of P: %s\n", nn_str(&ord));
	printf("  how: %s\n", how);
	if (info[0])
		printf("  transfer: %s\n", info);
	printf("  check: k P = Q: %s\n", ecp_eq(CE, &T, &Q) ? "yes" : "NO");
	if (!hasq)
		printf("  check: k = the secret mod ord(P): %s\n",
		    (nn_mod(&sec, &sec, &ord), nn_eq(&sec, &k)) ? "yes" : "NO");
}

/* ---- pairings ---- */

/* BN254 / BLS12-381: F_p^12 = F_p[w]/(w^12 - 2c w^6 + c^2 + 1) with
 * u = w^6 - c, u^2 = -1, xi = c + u = w^6; G2 untwisted into E(F_p^12) */
static void
pairing_named(void)
{
	static struct fk K;
	struct eck_curve Ek;
	struct eck_pt P, Q, P3, Q5, T;
	u64 m[13 * FW], w2[FKE], w3[FKE], u[FKE], x[FKE], y[FKE], t[FKE],
	    e[FKE], e1[FKE], e2[FKE], c0[FKE], c1[FKE];
	const struct ecnamed *N = cc.L.N;
	int c = atoi(N->xi), bls = N->name[1] == 'L', i;
	nn r = cc.L.n, three, five;
	char buf[2][400];

	memset(m, 0, sizeof m);
	mf_from_i64(CF, m, (int64_t)c * c + 1);
	mf_from_i64(CF, m + 6 * FW, -2 * (int64_t)c);
	mf_one(CF, m + 12 * FW);
	fk_init(&K, CF, 12, m);
	if (!fk_irreducible(&K))
		fail("pairing: w^12 - %d w^6 + %d is not irreducible", 2 * c,
		    c * c + 1);
	eck_from_ecp(&Ek, &K, CE);
	eck_pt_from_ecp(&K, &P, &cc.L.G);
	/* u = w^6 - c */
	fk_zero(&K, u);
	mf_one(CF, u + 6 * FW);
	mf_from_i64(CF, u, -c);
	fk_zero(&K, w2);
	mf_one(CF, w2 + 2 * FW);
	fk_zero(&K, w3);
	mf_one(CF, w3 + 3 * FW);
	if (bls) {
		fk_inv(&K, w2, w2);
		fk_inv(&K, w3, w3);
	}
	for (i = 0; i < 2; i++) {
		const char *s = i ? N->g2y : N->g2x, *comma = strchr(s, ',');
		char a[200], b[200];
		zz z0, z1;
		memcpy(a, s, (size_t)(comma - s));
		a[comma - s] = 0;
		snprintf(b, sizeof b, "%s", comma + 1);
		zz_parse(a, &z0, NULL);
		zz_parse(b, &z1, NULL);
		fk_zero(&K, c0);
		mf_from_zz(CF, c0, &z0);
		fk_zero(&K, c1);
		mf_from_zz(CF, c1, &z1);
		fk_mul(&K, t, c1, u);
		fk_add(&K, i ? y : x, c0, t);
	}
	fk_mul(&K, Q.x, x, w2);
	fk_mul(&K, Q.y, y, w3);
	Q.inf = 0;
	printf("e(G1, G2) on %s: reduced Tate pairing in F_p^12\n", N->name);
	printf("  field: F_p^12 = F_p[w]/(w^12 - %d w^6 + %d), u = w^6 - %d, "
	    "xi = %d + u = w^6\n", 2 * c, c * c + 1, c, c);
	printf("  twist: %s\n", bls ? "M-type E': y^2 = x^3 + 4 xi; psi(x, y) "
	    "= (x / w^2, y / w^3)" : "D-type E': y^2 = x^3 + 3 / xi; psi(x, y) "
	    "= (x w^2, y w^3)");
	printf("  check: G2 untwisted is on y^2 = x^3 + %s over F_p^12: %s\n",
	    bls ? "4" : "3", eck_on_curve(&Ek, &Q) ? "yes" : "NO");
	eck_mul(&Ek, &T, &Q, &r);
	printf("  check: r G2 = O: %s\n", T.inf ? "yes" : "NO");
	if (!eck_tate(&Ek, &P, &Q, &r, e, 1))
		fail("pairing: Miller's loop failed");
	fk_print(&K, buf[0], sizeof buf[0], e);
	printf("  value: %.300s%s\n", buf[0], strlen(buf[0]) > 300 ? " ..." : "");
	printf("  check: e != 1 (non-degenerate): %s\n", fk_is_one(&K, e) ?
	    "NO" : "yes");
	fk_pow_l(&K, t, e, r.w, NW);
	printf("  check: e^r = 1: %s\n", fk_is_one(&K, t) ? "yes" : "NO");
	nn_set(&three, 3);
	nn_set(&five, 5);
	eck_mul(&Ek, &P3, &P, &three);
	eck_mul(&Ek, &Q5, &Q, &five);
	eck_tate(&Ek, &P3, &Q, &r, e1, 1);
	eck_tate(&Ek, &P, &Q5, &r, e2, 1);
	fk_pow_l(&K, t, e, three.w, 1);
	printf("  check: e(3 G1, G2) = e^3: %s\n", fk_eq(&K, t, e1) ? "yes" :
	    "NO");
	fk_pow_l(&K, t, e, five.w, 1);
	printf("  check: e(G1, 5 G2) = e^5: %s\n", fk_eq(&K, t, e2) ? "yes" :
	    "NO");
	(void)buf[1];
}

static void
cmd_ecpairing(void)
{
	struct ecp_pt P, Q, P2, Q3;
	nn n, two, three;
	char w[16], b1[1200];
	int weil = 1, kd;
	static struct fk K;
	struct eck_curve Ek;
	struct eck_pt Pk, Qk, P2k, Q3k;
	u64 e[FKE], e1[FKE], e2[FKE], t[FKE], v[FKE];

	parse_curve();
	skipws();
	if (cc.named && cc.L.N->g2x && !*sp) {
		pairing_named();
		return;
	}
	parse_point(&P, 1);
	parse_point(&Q, 1);
	need_nn(&n, "n (the order of the points)");
	skipws();
	if (getword(w, sizeof w))
		weil = strcasecmp(w, "tate") != 0;
	kd = ec_embedding_degree(&CF->P, &n, FKMAX);
	if (kd <= 0)
		fail("pairing: n must be prime to p with a small embedding degree");
	if (!fk_init_auto(&K, CF, kd))
		fail("pairing: no field F_p^%d", kd);
	eck_from_ecp(&Ek, &K, CE);
	eck_pt_from_ecp(&K, &Pk, &P);
	eck_pt_from_ecp(&K, &Qk, &Q);
	{
		struct ecp_pt T1, T2;
		ecp_mul(CE, &T1, &P, &n);
		ecp_mul(CE, &T2, &Q, &n);
		if (!T1.inf || !T2.inf)
			fail("pairing: n P and n Q must be O");
	}
	if (weil ? !eck_weil(&Ek, &Pk, &Qk, &n, e) :
	    !eck_tate(&Ek, &Pk, &Qk, &n, e, 0))
		fail("pairing: Miller's algorithm failed");
	fk_print(&K, b1, sizeof b1, e);
	printf("e_%s(P, Q) = %s\n", nn_str(&n), b1);
	printf("  curve: %s\n  pairing: %s\n", cc.desc, weil ? "Weil (Miller, "
	    "randomised by an auxiliary point S)" : "reduced Tate, f_P(Q + S) "
	    "/ f_P(S) to the (q - 1)/n");
	printf("  field: F_p^%d (embedding degree of n)\n", kd);
	fk_pow_l(&K, t, e, n.w, NW);
	printf("  check: e^n = 1: %s\n", fk_is_one(&K, t) ? "yes" : "NO");
	nn_set(&two, 2);
	nn_set(&three, 3);
	ecp_mul(CE, &P2, &P, &two);
	ecp_mul(CE, &Q3, &Q, &three);
	eck_pt_from_ecp(&K, &P2k, &P2);
	eck_pt_from_ecp(&K, &Q3k, &Q3);
	if (weil) {
		eck_weil(&Ek, &P2k, &Q3k, &n, e1);
		eck_weil(&Ek, &Qk, &Pk, &n, e2);
	} else {
		eck_tate(&Ek, &P2k, &Q3k, &n, e1, 0);
		fk_one(&K, e2);
	}
	nn_set(&two, 6);
	fk_pow_l(&K, t, e, two.w, 1);
	printf("  check: e(2P, 3Q) = e(P, Q)^6: %s\n", fk_eq(&K, t, e1) ?
	    "yes" : "NO");
	if (weil) {
		fk_mul(&K, v, e, e2);
		printf("  check: e(P, Q) e(Q, P) = 1 (alternating): %s\n",
		    fk_is_one(&K, v) ? "yes" : "NO");
	}
	printf("  degenerate: %s\n", fk_is_one(&K, e) ? "yes (e = 1: P, Q "
	    "dependent, or the pairing is trivial on these)" : "no");
}

/* E(F_p) = Z/n1 x Z/n2 (n1 the exponent): P1 of order n1 and a point R
 * with e_n1(P1, R) of exact order n2; the pairing depends on R modulo
 * <P1>, so <P1, R> / <P1> has at least n2 elements: n1 n2 = #E */
static int
struct_certify(const nn *n1, const nn *n2, char *out, size_t on)
{
	static struct fk K;
	struct eck_curve Ek;
	struct nnfac f1, f2;
	nn q;
	int it, i;

	if (nn_is_one(n2)) {
		snprintf(out, on, "a point of order n1 = #E exists (above)");
		return 1;
	}
	fk_init_auto(&K, CF, 1);
	eck_from_ecp(&Ek, &K, CE);
	nn_factor(n1, &f1, 1);
	nn_factor(n2, &f2, 1);
	for (it = 0; it < 60; it++) {
		struct ecp_pt P1, R;
		struct eck_pt a, b;
		u64 e[FKE], t[FKE];
		nn ord;
		int ok = 1;
		ecp_random_point(CE, &P1);
		if (!ec_point_order(CE, &P1, n1, &f1, &ord) || !nn_eq(&ord, n1))
			continue;
		ecp_random_point(CE, &R);
		eck_pt_from_ecp(&K, &a, &P1);
		eck_pt_from_ecp(&K, &b, &R);
		if (!eck_weil(&Ek, &a, &b, n1, e))
			continue;
		fk_pow_l(&K, t, e, n2->w, NW);
		if (!fk_is_one(&K, t))
			continue;
		for (i = 0; i < f2.k && ok; i++) {
			nn_divmod(&q, NULL, n2, &f2.p[i]);
			fk_pow_l(&K, t, e, q.w, NW);
			if (fk_is_one(&K, t))
				ok = 0;
		}
		if (!ok)
			continue;
		snprintf(out, on, "P1 = %s has order n1; R = %s; the Weil "
		    "pairing e_n1(P1, R) = %s has order n2, so <P1, R> has at "
		    "least n1 n2 = #E points", pt(&P1), pt(&R), mf_str(CF, e));
		return 1;
	}
	snprintf(out, on, "NOT certified (no pairing witness found)");
	return 0;
}
