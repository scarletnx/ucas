/* ring.c, part 3: rings and fields, polynomials and matrices over them */

/* the unit group of Z/n as invariant factors: "C2 x C4" */
static const char *
unit_group(u64 n, int *cyclic)
{
	static char out[256];
	struct nt_fac f, g;
	u64 cyc[64], pr[64];
	int ex[64][64], cnt[64], ncyc = 0, np = 0, i, j, k, maxc = 0;
	size_t o = 0;

	nt_factor(n, &f);
	for (i = 0; i < f.k; i++) {
		u64 p = f.p[i];
		int e = f.e[i];
		if (p == 2) {
			if (e == 2)
				cyc[ncyc++] = 2;
			if (e >= 3) {
				cyc[ncyc++] = 2;
				cyc[ncyc++] = upow(2, e - 2);
			}
		} else
			cyc[ncyc++] = upow(p, e - 1) * (p - 1);
	}
	memset(cnt, 0, sizeof cnt);
	for (i = 0; i < ncyc; i++) {
		nt_factor(cyc[i], &g);
		for (j = 0; j < g.k; j++) {
			for (k = 0; k < np && pr[k] != g.p[j]; k++)
				;
			if (k == np)
				pr[np++] = g.p[j];
			ex[k][cnt[k]++] = g.e[j];
		}
	}
	for (k = 0; k < np; k++) {
		/* descending */
		for (i = 1; i < cnt[k]; i++)
			for (j = i; j > 0 && ex[k][j - 1] < ex[k][j]; j--) {
				int t = ex[k][j];
				ex[k][j] = ex[k][j - 1];
				ex[k][j - 1] = t;
			}
		if (cnt[k] > maxc)
			maxc = cnt[k];
	}
	*cyclic = maxc <= 1;
	out[0] = 0;
	if (maxc == 0)
		snprintf(out, sizeof out, "trivial");
	for (j = maxc - 1; j >= 0; j--) {
		u64 d = 1;
		for (k = 0; k < np; k++)
			if (j < cnt[k])
				d *= upow(pr[k], ex[k][j]);
		o += (size_t)snprintf(out + o, sizeof out - o, "%sC%llu",
		    j == maxc - 1 ? "" : " x ", U(d));
	}
	return out;
}

enum { SUB_ALL, SUB_UNITS, SUB_ZD, SUB_NIL, SUB_IDEM, SUB_IDEALS, SUB_FIELD,
	SUB_DOMAIN, SUB_CHAR, SUB_ORDER, SUB_GEN, SUB_SUBF, SUB_TABLE,
	SUB_FROB, SUB_ELEMS };

static const struct {
	const char *w;
	int sub;
} subwords[] = {
	{"units", SUB_UNITS}, {"unit", SUB_UNITS}, {"invertible", SUB_UNITS},
	{"zerodivisors", SUB_ZD}, {"zero_divisors", SUB_ZD},
	{"divisors", SUB_ZD}, {"nilpotents", SUB_NIL}, {"nilpotent", SUB_NIL},
	{"idempotents", SUB_IDEM}, {"idempotent", SUB_IDEM},
	{"ideals", SUB_IDEALS}, {"ideal", SUB_IDEALS}, {"field", SUB_FIELD},
	{"domain", SUB_DOMAIN}, {"integral", SUB_DOMAIN},
	{"characteristic", SUB_CHAR}, {"char", SUB_CHAR}, {"order", SUB_ORDER},
	{"size", SUB_ORDER}, {"cardinality", SUB_ORDER},
	{"generators", SUB_GEN}, {"generator", SUB_GEN},
	{"primitive", SUB_GEN}, {"primroots", SUB_GEN}, {"subfields", SUB_SUBF},
	{"subfield", SUB_SUBF}, {"table", SUB_TABLE}, {"tables", SUB_TABLE},
	{"logs", SUB_TABLE}, {"frobenius", SUB_FROB}, {"elements", SUB_ELEMS},
	{NULL, 0}
};

static int
sub_word(void)
{
	int k;

	if (rti >= rlim || rt[rti].k != RK_ID)
		return -1;
	if (at_id("zero") && rti + 2 == rlim && rt[rti + 1].k == RK_ID &&
	    !strncasecmp(rt[rti + 1].s, "divisor", 7)) {
		rti = rlim;
		return SUB_ZD;
	}
	if (at_id("integral") && rti + 2 == rlim && rt[rti + 1].k == RK_ID &&
	    !strcasecmp(rt[rti + 1].s, "domain")) {
		rti = rlim;
		return SUB_DOMAIN;
	}
	if (rti + 1 != rlim)
		return -1;
	for (k = 0; subwords[k].w; k++)
		if (!strcasecmp(subwords[k].w, rt[rti].s)) {
			rti = rlim;
			return subwords[k].sub;
		}
	return -1;
}

/* ---- Z/n ---- */

static void
zn_info(u64 n, int sub)
{
	struct nt_fac f;
	u64 ph = nt_phi(n), rad = 1, list[64], k, x;
	int i, cyc, nl;
	const char *ug;

	nt_factor(n, &f);
	for (i = 0; i < f.k; i++)
		rad *= f.p[i];
	ug = unit_group(n, &cyc);
	if (sub == SUB_ALL)
		printf("Z/%llu: the integers mod %llu%s%s\n", U(n), U(n), f.k > 1 ||
		    (f.k == 1 && f.e[0] > 1) ? " = " : "", f.k > 1 || (f.k == 1 &&
		    f.e[0] > 1) ? fac_text(n) : "");
	if (sub == SUB_ALL || sub == SUB_ORDER || sub == SUB_CHAR)
		printf("  %llu elements, characteristic %llu\n", U(n), U(n));
	if (sub == SUB_ALL || sub == SUB_FIELD || sub == SUB_DOMAIN) {
		if (n == 1)
			printf("  the zero ring (0 = 1): not a field, not an integral "
			    "domain\n");
		else if (f.k == 1 && f.e[0] == 1)
			printf("  a field: yes (%llu is prime: every nonzero a has an "
			    "inverse, a^(%llu-2)); so also an integral domain\n", U(n),
			    U(n));
		else {
			u64 a = f.p[0], b = n / f.p[0];
			printf("  a field: no; an integral domain: no (%llu is not prime: "
			    "%llu * %llu = 0 with both nonzero)\n", U(n), U(a), U(b));
		}
		if (sub == SUB_ALL && f.k == 1 && f.e[0] > 1)
			printf("  a local ring: its only maximal ideal is (%llu)\n",
			    U(f.p[0]));
	}
	if (sub == SUB_ALL && f.k > 1) {
		printf("  CRT: Z/%llu = ", U(n));
		for (i = 0; i < f.k; i++)
			printf("%sZ/%llu", i ? " x " : "", U(upow(f.p[i], f.e[i])));
		printf(" (a -> (a mod ");
		for (i = 0; i < f.k; i++)
			printf("%s%llu", i ? ", a mod " : "", U(upow(f.p[i], f.e[i])));
		printf("))\n");
	}
	if (sub == SUB_ALL || sub == SUB_UNITS) {
		nl = 0;
		for (x = 0; x < n && nl < (sub == SUB_UNITS ? 64 : 24); x++)
			if (nt_gcd(x, n) == 1)
				list[nl++] = x;
		{
			char lab[64];
			snprintf(lab, sizeof lab, "units: phi(%llu) = %llu: ", U(n),
			    U(ph));
			print_u64s(lab, list, (u64)nl, ph, sub == SUB_UNITS ? 64 : 24);
		}
		printf("  unit group: %s (order %llu, exponent lambda(%llu) = %llu)",
		    ug, U(ph), U(n), U(nt_lambda(n)));
		if (cyc && n > 2)
			printf("; cyclic, generated by the primitive root %llu\n",
			    U(nt_primroot(n)));
		else if (!cyc)
			printf("; not cyclic, so there is no primitive root\n");
		else
			printf("\n");
	}
	if (sub == SUB_GEN) {
		if (!nt_has_primroot(n))
			printf("  no generators: the unit group %s is not cyclic\n", ug);
		else
			cmd_primroot_int(n, 0, 0);
	}
	if (sub == SUB_ALL || sub == SUB_ZD) {
		u64 zd = n - ph - 1;
		nl = 0;
		for (x = 1; x < n && nl < (sub == SUB_ZD ? 64 : 24); x++)
			if (nt_gcd(x, n) != 1)
				list[nl++] = x;
		if (n == 1)
			zd = 0;
		if (zd == 0)
			printf("  zero divisors: none besides 0\n");
		else {
			char lab[96];
			snprintf(lab, sizeof lab, "zero divisors: %llu nonzero (every "
			    "nonzero non-unit of a finite ring): ", U(zd));
			print_u64s(lab, list, (u64)nl, zd, sub == SUB_ZD ? 64 : 24);
		}
	}
	if (sub == SUB_ALL || sub == SUB_NIL) {
		u64 cnt = n / rad;
		nl = 0;
		for (x = 0; x < n && nl < 24; x += rad)
			list[nl++] = x;
		{
			char lab[96];
			snprintf(lab, sizeof lab, "nilpotents: the multiples of rad(%llu) "
			    "= %llu, %llu of them: ", U(n), U(rad), U(cnt));
			print_u64s(lab, list, (u64)nl, cnt, 24);
		}
	}
	if (sub == SUB_ALL || sub == SUB_IDEM) {
		/* one per subset of the prime powers, by CRT */
		u64 m, cnt = (u64)1 << f.k;
		nl = 0;
		for (m = 0; m < cnt && nl < 64; m++) {
			u64 r = 0, mm = 1;
			for (i = 0; i < f.k; i++) {
				u64 pe = upow(f.p[i], f.e[i]);
				nt_crt(r, mm, (m >> i & 1) ? 1 % pe : 0, pe, &r, &mm);
			}
			list[nl++] = r;
		}
		qsort(list, (size_t)nl, sizeof(u64), cmpu);
		{
			char lab[96];
			snprintf(lab, sizeof lab, "idempotents: 2^%d (e = 0 or 1 mod "
			    "each prime power): ", f.k);
			print_u64s(lab, list, (u64)nl, cnt, 24);
		}
	}
	if (sub == SUB_ALL || sub == SUB_IDEALS) {
		u64 d, cnt = 0;
		nl = 0;
		for (d = 1; d <= n && d <= (1u << 26); d++)
			if (n % d == 0) {
				if (nl < 64)
					list[nl++] = d;
				cnt++;
			}
		printf("  ideals: (d) for each d | %llu (%llu of them; Z/%llu is a "
		    "principal ideal ring):", U(n), U(cnt), U(n));
		for (i = 0; i < nl && i < 24; i++)
			printf(" (%llu)", U(list[i] == n ? 0 : list[i]));
		printf("%s\n", cnt > 24 ? " ..." : "");
		printf("  quotients: Z/%llu / (d) = Z/d (a -> a mod d)\n", U(n));
		if (n > 1) {
			printf("  maximal = prime ideals: ");
			for (i = 0; i < f.k; i++)
				printf("%s(%llu)", i ? ", " : "", U(f.p[i]));
			printf(" (quotients Z/p, fields)\n");
		}
	}
	if (sub == SUB_ALL && n <= 2048 && n > 1) {
		/* brute force: inverses by search, zero divisors, powers */
		u64 cu = 0, cz = 0, cn = 0, ci = 0, a, b;
		for (a = 0; a < n; a++) {
			int unit = 0, zd = 0;
			u64 p = a;
			for (b = 0; b < n; b++) {
				u64 ab = a * b % n;
				if (ab == 1 % n)
					unit = 1;
				if (ab == 0 && b && a)
					zd = 1;
			}
			cu += unit;
			cz += zd;
			for (k = 0; k < 64 && p; k++)
				p = p * a % n;
			cn += p == 0;
			ci += a * a % n == a;
		}
		printf("  check: by brute force over the %llu elements: %llu units, "
		    "%llu zero divisors, %llu nilpotents, %llu idempotents\n", U(n),
		    U(cu), U(cz), U(cn), U(ci));
	}
	if (sub == SUB_SUBF || sub == SUB_TABLE || sub == SUB_FROB ||
	    sub == SUB_ELEMS) {
		if (sub == SUB_ELEMS && n <= 256) {
			printf("  elements: 0 .. %llu\n", U(n - 1));
			return;
		}
		rfail("that applies to fields GF(p^n), n > 1");
	}
}

static void
zn_element(u64 n, i128 a0)
{
	u64 a = (u64)(((a0 % (i128)n) + (i128)n) % (i128)n), g, inv, o;
	u64 roots[8];
	int k;

	printf("%llu in Z/%llu\n", U(a), U(n));
	inv = nt_inv(a, n, &g);
	if (g == 1) {
		printf("  a unit: inverse %llu (%llu * %llu = %s = 1 mod %llu)\n",
		    U(inv), U(a), U(inv), u128s((u128)a * inv), U(n));
		o = nt_order(a, n);
		printf("  multiplicative order %llu%s\n", U(o), o == nt_phi(n) &&
		    n > 2 ? " (a primitive root)" : "");
	} else if (a == 0)
		printf("  zero\n");
	else
		printf("  not a unit: gcd(%llu, %llu) = %llu; a zero divisor: %llu * "
		    "%llu = 0\n", U(a), U(n), U(g), U(a), U(n / g));
	printf("  additive order %llu\n", U(n / nt_gcd(a, n)));
	{
		u64 p = a;
		int e;
		for (e = 1; e < 64 && p; e++)
			p = nt_mulmod(p, a, n);
		if (a && p == 0)
			printf("  nilpotent: %llu^%d = 0\n", U(a), e);
	}
	if (nt_mulmod(a, a, n) == a)
		printf("  idempotent: %llu^2 = %llu\n", U(a), U(a));
	k = nt_sqrt_all(a, n, roots, 8);
	if (k > 0) {
		int i;
		printf("  a square: ");
		for (i = 0; i < k && i < 4; i++)
			printf("%s%llu^2", i ? ", " : "", U(roots[i]));
		printf("%s = %llu\n", k > 4 ? ", ..." : "", U(a));
	} else if (k == 0)
		printf("  not a square mod %llu\n", U(n));
}

/* ---- GF(q) and Z/n[x]/(f) ---- */

static void
modulus_poly(const struct ring *R, struct ring *B, struct rpoly *m)
{
	int i;

	ring_zn(B, R->n);
	m->d = R->deg;
	for (i = 0; i <= R->deg; i++)
		m->c[i] = R->f[i];
}

static const char *
mod_desc(const struct rspec *s)
{
	static char buf[256];
	struct ring B;
	struct rpoly m;
	int k;

	modulus_poly(&s->R, &B, &m);
	if (s->R.fname) {
		k = named_index(s->R.fname);
		snprintf(buf, sizeof buf, "%s), the %s polynomial %s", rp_str(&B, &m),
		    named_mod[k].what, u128x(named_mod[k].code));
	} else if (s->R.n == 2 && s->R.deg < 64)
		snprintf(buf, sizeof buf, "%s), %s0x%llx", rp_str(&B, &m),
		    s->automod ? "" : "",
		    U(((u64)1 << s->R.deg) | r_pack(&s->R, s->R.f)));
	else
		snprintf(buf, sizeof buf, "%s)", rp_str(&B, &m));
	return buf;
}

static void
subfield_list(const struct ring *R)
{
	int d;

	printf("  subfields: ");
	for (d = 1; d <= R->deg; d++)
		if (R->deg % d == 0) {
			if (d == 1)
				printf("GF(%llu)", U(R->n));
			else
				printf(", GF(%llu^%d)", U(R->n), d);
		}
	printf(" (GF(%llu^d) for each d | %d: the a with a^(%llu^d) = a)\n",
	    U(R->n), R->deg, U(R->n));
}

static void
gf_table(struct ring *R)
{
	u64 i;

	if (R->qm1 > 255)
		rfail("log/antilog tables are printed for fields of up to 256 "
		    "elements");
	ring_tables(R);
	printf("  generator g = %s\n", el_str(R, R->gen));
	printf("  antilog: g^k for k = 0, 1, ...\n   ");
	for (i = 0; i < R->qm1; i++)
		printf(" %s%s", el_code(R, R->ex[i]), (i % 16 == 15 && i + 1 <
		    R->qm1) ? "\n   " : "");
	printf("\n  log: log_g(a) for a = 1, 2, ...\n   ");
	for (i = 1; i <= R->qm1; i++)
		printf(" %3u%s", R->lg[i], (i % 16 == 15 && i < R->qm1) ? "\n   " :
		    "");
	printf("\n  check: g^log(a) = a for every nonzero a; a*b = g^(log a + "
	    "log b mod %llu)\n", U(R->qm1));
}

static void
ext_info(struct rspec *s, int sub)
{
	struct ring *R = &s->R, B;
	struct rpoly m;
	u64 q1 = R->qm1;
	int cyc;

	(void)cyc;
	modulus_poly(R, &B, &m);
	if (sub == SUB_ALL) {
		if (R->field)
			printf("%s = GF(%llu)[x]/(%s\n", s->name, U(R->n), mod_desc(s));
		else
			printf("%s\n", s->name);
		printf("  %s elements (%llu^%d), characteristic %llu", u128s((u128)q1 +
		    1), U(R->n), R->deg, U(R->n));
		if (R->nprime)
			printf(", dimension %d over GF(%llu)", R->deg, U(R->n));
		printf("\n");
		if (s->automod)
			printf("  modulus: the smallest primitive polynomial of degree "
			    "%d over GF(%llu) (by its coefficients read as a base-%llu "
			    "number; not necessarily the Conway polynomial)\n", R->deg,
			    U(R->n), U(R->n));
	}
	if (sub == SUB_ORDER || sub == SUB_CHAR) {
		printf("  %s elements, characteristic %llu\n", u128s((u128)q1 + 1),
		    U(R->n));
		return;
	}
	if (!R->field) {
		if (sub != SUB_ALL && sub != SUB_FIELD && sub != SUB_DOMAIN &&
		    sub != SUB_UNITS && sub != SUB_ZD)
			rfail("%s is not a field", s->name);
		if (!R->nprime) {
			printf("  a field: no; an integral domain: no (%llu is not "
			    "prime, so Z/%llu already has zero divisors)\n", U(R->n),
			    U(R->n));
			if (q1 < 4096) {
				u64 a, b, nu = 0, nz = 0, one = r_const(R, 1);
				for (a = 1; a <= q1; a++) {
					int u = 0, z = 0;
					for (b = 1; b <= q1; b++) {
						u64 p = r_mul(R, a, b);
						u |= p == one;
						z |= p == 0;
					}
					nu += u;
					nz += z;
				}
				printf("  units: %llu; zero divisors: %llu nonzero (by brute "
				    "force over all products)\n", U(nu), U(nz));
			}
		}
		else {
			static struct rpoly fac[RMAXF];
			int mult[RMAXF], k, i;
			u64 lead;
			u128 units = 1;
			k = rp_factor(&B, &m, fac, mult, RMAXF, &lead);
			printf("  a field: no; an integral domain: no: the modulus "
			    "factors, %s = %s\n", rp_str(&B, &m), fac_str(&B, fac, mult, k,
			    lead));
			for (i = 0; i < k; i++) {
				u128 qd = 1, qd1 = 1;
				int j;
				for (j = 0; j < fac[i].d * mult[i]; j++)
					qd *= R->n;
				for (j = 0; j < fac[i].d * (mult[i] - 1); j++)
					qd1 *= R->n;
				units *= qd - qd1;
			}
			printf("  units: %s (by CRT over the factors: q^(d e) - "
			    "q^(d (e-1)) for each factor of degree d to the power e); "
			    "zero divisors: %s nonzero\n", u128s(units),
			    u128s((u128)q1 + 1 - units - 1));
			if (k >= 1) {
				struct rpoly a, b;
				rp_divmod(&B, &m, &fac[0], &b, NULL);
				a = fac[0];
				printf("  e.g. (%s) * (%s) = 0\n", rp_str(&B, &a),
				    rp_str(&B, &b));
			}
		}
		return;
	}
	ring_tables(R);
	if (sub == SUB_ALL || sub == SUB_FIELD || sub == SUB_DOMAIN) {
		struct ring BB;
		int rabin, benor, berl;
		modulus_poly(R, &BB, &m);
		rabin = rp_irreducible(&BB, &m);
		benor = rp_benor(&BB, &m);
		berl = rp_berlekamp_count(&BB, &m);
		printf("  a field: yes, the modulus is irreducible (Rabin: x^(%llu^%d) "
		    "= x mod f and gcd(x^(%llu^(%d/r)) - x, f) = 1 for each prime r "
		    "| %d)\n", U(R->n), R->deg, U(R->n), R->deg, R->deg);
		printf("  check: Ben-Or's test %s; Berlekamp's matrix has nullity %d "
		    "(%s)\n", benor ? "agrees" : "DISAGREES", berl, berl == 1 ?
		    "one irreducible factor" : "DISAGREES");
		(void)rabin;
		if (q1 < 256) {
			/* the field axioms that matter, by brute force */
			u64 a, b, noinv = 0, zd = 0, dis = 0, one = r_const(R, 1);
			for (a = 1; a <= q1; a++) {
				int has = 0;
				for (b = 1; b <= q1; b++) {
					u64 p = r_mul(R, a, b);
					if (p == one)
						has = 1;
					if (p == 0)
						zd++;
					if (p != r_mul_alt(R, a, b) || p != r_mul(R, b, a))
						dis++;
				}
				noinv += !has;
			}
			printf("  check by brute force over all %llu x %llu products: "
			    "every nonzero element has an inverse%s, no zero divisors%s, "
			    "commutative and equal to shift-and-add%s\n", U(q1 + 1),
			    U(q1 + 1), noinv ? " (FAILED)" : "", zd ? " (FAILED)" : "",
			    dis ? " (FAILED)" : "");
		}
	}
	if (sub == SUB_ALL || sub == SUB_GEN || sub == SUB_UNITS) {
		u64 ox = r_order(R, r_x(R));
		printf("  x has order %llu: the modulus is %sprimitive", U(ox),
		    ox == q1 ? "" : "not ");
		if (ox != q1)
			printf("; generator %s", el_str(R, R->gen));
		printf("\n  unit group: cyclic of order %llu = %s; phi(%llu) = %llu "
		    "generators\n", U(q1), fac_text(q1), U(q1), U(nt_phi(q1)));
	}
	if (sub == SUB_GEN && q1 <= (1u << 20)) {
		u64 a, c = 0;
		printf("  generators:");
		for (a = 1; a <= q1; a++)
			if (r_order(R, a) == q1) {
				if (c < 24)
					printf(" %s", el_code(R, a));
				c++;
			}
		printf("%s\n", c > 24 ? " ..." : "");
		printf("  check: counted %llu elements of order %llu\n", U(c), U(q1));
	}
	if (sub == SUB_ALL || sub == SUB_SUBF)
		subfield_list(R);
	if (sub == SUB_SUBF) {
		int d;
		for (d = 1; d < R->deg; d++)
			if (R->deg % d == 0) {
				u128 pd = 1;
				int j;
				u64 h;
				for (j = 0; j < d; j++)
					pd *= R->n;
				h = r_pow(R, R->gen, (u64)(q1 / (u64)(pd - 1)));
				printf("    GF(%llu^%d) = {0} and the powers of g^%llu = %s\n",
				    U(R->n), d, U((u64)(q1 / (u64)(pd - 1))), el_code(R, h));
			}
	}
	if (sub == SUB_ALL || sub == SUB_FROB)
		printf("  Frobenius a -> a^%llu: an automorphism of order %d fixing "
		    "GF(%llu); the Galois group is cyclic of order %d\n", U(R->n),
		    R->deg, U(R->n), R->deg);
	if (sub == SUB_FROB && q1 <= (1u << 16)) {
		u64 a, bad = 0;
		for (a = 0; a <= q1; a++)
			for (u64 b = 0; b <= q1 && b < 64; b++)
				if (r_frob(R, r_add(R, a, b)) != r_add(R, r_frob(R, a),
				    r_frob(R, b)))
					bad++;
		printf("  check: (a + b)^%llu = a^%llu + b^%llu for every a and 64 "
		    "b: %s\n", U(R->n), U(R->n), U(R->n), bad ? "FAILED" : "holds");
	}
	if (sub == SUB_TABLE)
		gf_table(R);
	if (sub == SUB_ELEMS) {
		u64 a;
		if (q1 > 255)
			rfail("listing elements for fields of up to 256 elements");
		for (a = 0; a <= q1; a++)
			printf("  %s\n", el_str(R, a));
	}
	if (sub == SUB_IDEALS)
		printf("  ideals: (0) and the field itself\n");
	if (sub == SUB_ZD)
		printf("  zero divisors: none (a field)\n");
	if (sub == SUB_NIL)
		printf("  nilpotents: 0 only\n");
	if (sub == SUB_IDEM)
		printf("  idempotents: 0 and 1\n");
}

/* minimal polynomial of a over the prime field: prod (x - a^(p^i)) */
static int
minpoly(const struct ring *R, u64 a, struct rpoly *mp, u64 *conj)
{
	struct rpoly t, lin;
	struct ring B;
	u64 c = a;
	int k = 0, i;

	rp_const(R, &t, r_const(R, 1));
	do {
		conj[k++] = c;
		rp_x(&lin);
		lin.c[0] = r_neg(R, c);
		rp_norm(&lin);
		rp_mul(R, &t, &t, &lin);
		c = r_frob(R, c);
	} while (c != a && k < RMAXF);
	ring_zn(&B, R->n);
	mp->d = t.d;
	for (i = 0; i <= t.d; i++) {
		if (t.c[i] >= R->n)
			return -1;   /* not in the prime field: a bug */
		mp->c[i] = t.c[i];
	}
	(void)B;
	return k;
}

static void
gf_element(struct rspec *s, u64 a)
{
	struct ring *R = &s->R, B;
	struct rpoly mp;
	u64 conj[RMAXF], inv, o, tr, nm;
	int k, i;

	ring_tables(R);
	printf("%s in %s\n", el_str(R, a), s->name);
	if (a == 0) {
		printf("  zero: no inverse; minimal polynomial x\n");
		return;
	}
	if (!R->field) {
		if (r_inv(R, a, &inv))
			printf("  a unit: inverse %s; check: product %s\n", el_str(R,
			    inv), el_code(R, r_mul_alt(R, a, inv)));
		else
			printf("  not a unit (a zero divisor)\n");
		return;
	}
	r_inv(R, a, &inv);
	printf("  inverse: %s (a^(q-2))\n", el_str(R, inv));
	printf("  check: %s * %s = %s (%s)\n", el_code(R, a), el_code(R, inv),
	    el_code(R, r_mul_alt(R, a, inv)), R->lg ? "by log tables" :
	    "shift-and-add");
	o = r_order(R, a);
	printf("  order %llu (divides %llu)%s\n", U(o), U(R->qm1), o == R->qm1 ?
	    ": a generator" : "");
	{
		u64 x, ord;
		if (r_dlog(R, R->gen, a, &x, &ord) == 1)
			printf("  log base %s: %llu (%s^%llu = %s)\n", el_code(R, R->gen),
			    U(x), el_code(R, R->gen), U(x), el_code(R, r_pow(R, R->gen,
			    x)));
	}
	k = minpoly(R, a, &mp, conj);
	ring_zn(&B, R->n);
	if (k > 0) {
		printf("  minimal polynomial over GF(%llu): %s (degree %d: a lies in "
		    "GF(%llu^%d))\n", U(R->n), rp_str(&B, &mp), k, U(R->n), k);
		printf("  conjugates a^(%llu^i):", U(R->n));
		for (i = 0; i < k; i++)
			printf(" %s", el_code(R, conj[i]));
		printf("\n");
		printf("  check: the minimal polynomial is %sirreducible (Rabin) and "
		    "vanishes at a: %s\n", rp_irreducible(&B, &mp) ? "" : "NOT ",
		    el_code(R, rp_eval(R, &mp, a)));
	}
	tr = 0;
	{
		u64 c = a;
		for (i = 0; i < R->deg; i++) {
			tr = r_add(R, tr, c);
			c = r_frob(R, c);
		}
	}
	nm = r_pow(R, a, R->qm1 / (R->n - 1));
	printf("  trace %s, norm %s (to GF(%llu))\n", el_code(R, tr), el_code(R,
	    nm), U(R->n));
	if (R->n == 2)
		printf("  square root: %s (a^(2^%d))\n", el_code(R,
		    r_pow(R, a, (R->qm1 + 1) / 2)), R->deg - 1);
}

/* the value of an expression, how, and a second evaluation */
static void
ring_eval(struct rspec *s)
{
	const struct ring *R = &s->R;
	int a = rti, i;
	u64 v, w;

	v = eexpr(R, 0);
	{
		int ops = e_nops, lop = e_last_op;
		u64 la = e_last_a, lb = e_last_b;
		int ni = e_ninv;
		u64 ia[64], ir[64];
		memcpy(ia, e_inv_a, sizeof ia);
		memcpy(ir, e_inv_r, sizeof ir);
		printf("%s = %s in %s\n", eq_text(a, rlim), el_str(R, v), s->name);
		if (ops == 1 && lop == '*' && R->deg > 1 && R->n == 2) {
			u128 p = 0;
			int j;
			for (j = 0; j < 64; j++)
				if (lb >> j & 1)
					p ^= (u128)la << j;
			printf("  how: carry-less product %s, reduced mod (%s\n",
			    u128x(p), mod_desc(s));
		} else if (R->deg > 1)
			printf("  how: polynomial arithmetic in x, reduced mod (%s%s\n",
			    mod_desc(s), ni ? "; inverses as a^(q-2)" : "");
		else if (ni)
			printf("  how: arithmetic mod %llu; inverses by extended Euclid\n",
			    U(R->n));
		rti = a;
		w = eexpr(R, 1);
		notes_drop();   /* the second reading repeats the first's notes */
		printf("  check: %s%s", w == v ? "the same with a second "
		    "multiplication algorithm" : "a second multiplication algorithm "
		    "DISAGREES", R->lg ? " (log tables)" : R->deg > 1 ?
		    " (shift-and-add, xtime)" : " (double-and-add)");
		for (i = 0; i < ni && i < 3; i++)
			printf("; %s * %s = %s", el_code(R, ia[i]), el_code(R, ir[i]),
			    el_code(R, r_mul_alt(R, ia[i], ir[i])));
		printf("\n");
	}
}

static void
ring_log(struct rspec *s)
{
	u64 a, g;
	int gauto = 1, save, j;

	/* log A [base G] */
	save = rti;
	for (j = rti; j < rlim; j++)
		if (rt[j].k == RK_ID && !strcasecmp(rt[j].s, "base"))
			break;
	if (j < rlim) {
		int lim = rlim;
		rlim = j;
		a = eexpr(&s->R, 0);
		rti = j + 1;
		rlim = lim;
		g = eexpr(&s->R, 0);
		gauto = 0;
	} else {
		a = eexpr(&s->R, 0);
		if (s->R.deg == 1 && !s->R.field) {
			g = nt_primroot(s->R.n);
			if (!g)
				rfail("no base given and Z/%llu has no primitive root",
				    U(s->R.n));
		} else {
			ring_tables(&s->R);
			g = s->R.gen;
		}
	}
	(void)save;
	cmd_dlog_ring(s, a, g, gauto);
}

/* ---- polynomial commands ---- */

/* GF(2) when the coefficients are bits and no ring is named */
static int
default_gf2(struct rspec *s, const char *cmd)
{
	int i;

	for (i = rti; i < rlim; i++) {
		if (rt[i].k == RK_NUM && rt[i].v > 1 && !(i > 0 &&
		    rt[i - 1].k == RK_SYM && rt[i - 1].s[0] == '^') &&
		    rt[i].base == 10)
			rfail("%s: over which field? (%s P mod p, or over GF(q))", cmd,
			    cmd);
		if (rt[i].k == RK_ID && strcmp(rt[i].s, "x"))
			rfail("%s: over which field? (%s P mod p, or over GF(q))", cmd,
			    cmd);
	}
	note("no field given: over GF(2), as the coefficients are bits (say "
	    "\"over GF(p)\" or \"mod p\" for another)");
	memset(s, 0, sizeof *s);
	ring_zn(&s->R, 2);
	s->kind = RS_ZN;
	s->gf = 1;
	spec_name(s);
	return 1;
}

static const char *
field_name(const struct rspec *s)
{
	static char b[200];

	if (s->kind == RS_ZN && s->R.nprime)
		snprintf(b, sizeof b, "GF(%llu)", U(s->R.n));
	else
		snprintf(b, sizeof b, "%s", s->name);
	return b;
}

static void
need_field(const struct rspec *s, const char *what)
{
	if (s->kind != RS_ZN && s->kind != RS_EXT)
		rfail("%s: polynomials over Z/n or GF(q)", what);
	if (!s->R.field)
		rfail("%s over %s: %s is not a field%s", what, s->name, s->name,
		    s->kind == RS_ZN ? " (Z/n[x] has no unique factorisation when n "
		    "is composite; work mod each prime factor)" : "");
}

/* trial division by every monic polynomial of degree <= n/2 */
static int
brute_irreducible(const struct ring *R, const struct rpoly *f, int *ok)
{
	u128 q = (u128)R->qm1 + 1, cnt = 0, lim;
	int d;

	*ok = 0;
	for (d = 1; 2 * d <= f->d; d++) {
		u128 t = 1;
		int j;
		for (j = 0; j < d; j++) {
			t *= q;
			if (t > (1u << 16))
				return 0;
		}
		cnt += t;
	}
	if (cnt > (1u << 16))
		return 0;
	for (d = 1; 2 * d <= f->d; d++) {
		u64 c;
		lim = 1;
		for (int j = 0; j < d; j++)
			lim *= q;
		for (c = 0; c < (u64)lim; c++) {
			struct rpoly g, r;
			u64 v = c;
			int j;
			g.d = d;
			for (j = 0; j < d; j++) {
				g.c[j] = q > ((u128)1 << 64) ? v : (u64)(v % (u64)q);
				v = (u64)(v / (u64)q);
			}
			g.c[d] = r_const(R, 1);
			rp_divmod(R, f, &g, NULL, &r);
			if (r.d < 0) {
				*ok = 1;
				return 2;   /* a factor */
			}
		}
	}
	*ok = 1;
	return 1;
}

static void
cmd_irreducible(struct rspec *s, int primitive)
{
	struct rpoly f;
	static struct rpoly fac[RMAXDEG + 1];
	int mult[RMAXDEG + 1], k, irr, ben, berl, bf, bok;
	u64 lead;
	const struct ring *R = &s->R;

	need_field(s, primitive ? "primitive" : "irreducible");
	poly_arg(R, &f);
	if (f.d < 1)
		rfail("a polynomial of degree >= 1");
	irr = rp_irreducible(R, &f);
	printf("%s is %sirreducible over %s\n", rp_str(R, &f), irr ? "" : "not ",
	    field_name(s));
	printf("  how: Rabin's test (degree %d: x^(q^%d) = x mod f, and gcd(x^(q^"
	    "(%d/r)) - x, f) = 1 for each prime r | %d; q = %s)\n", f.d, f.d, f.d,
	    f.d, u128s((u128)R->qm1 + 1));
	if (!irr) {
		k = rp_factor(R, &f, fac, mult, RMAXDEG + 1, &lead);
		printf("  %s = %s\n", rp_str(R, &f), fac_str(R, fac, mult, k, lead));
	}
	ben = rp_benor(R, &f);
	{
		struct rpoly fd, g;
		rp_deriv(R, &fd, &f);
		rp_gcd(R, &g, &f, &fd);
		berl = g.d == 0 ? rp_berlekamp_count(R, &f) : -1;
	}
	bf = brute_irreducible(R, &f, &bok);
	printf("  check: Ben-Or's test %s", ben == irr ? "agrees" : "DISAGREES");
	if (berl >= 0)
		printf("; Berlekamp: %d distinct irreducible factor%s%s", berl,
		    berl == 1 ? "" : "s", (berl == 1) == irr ? "" : " (DISAGREES)");
	else
		printf("; not square-free (gcd(f, f') != 1)");
	if (bok)
		printf("; trial division by every monic polynomial of degree <= %d: "
		    "%s", f.d / 2, bf == 2 ? "a factor found" : "none divides");
	printf("\n");
	if (primitive) {
		int of;
		u64 o;
		u128 qn = 1;
		int j;
		for (j = 0; j < f.d; j++)
			qn *= (u128)R->qm1 + 1;
		if (!irr) {
			printf("not primitive: a primitive polynomial is irreducible\n");
			return;
		}
		if (f.c[0] == 0)
			rfail("x divides it");
		o = rp_xorder(R, &f, &of);
		if (of) {
			printf("  the order of x is not computed: q^%d - 1 does not fit "
			    "64 bits\n", f.d);
			return;
		}
		printf("%s: x has order %llu %s q^%d - 1 = %s\n", (u128)o + 1 == qn ?
		    "primitive" : "not primitive", U(o), (u128)o + 1 == qn ? "=" :
		    "<", f.d, u128s(qn - 1));
		printf("  how: x^((q^n - 1)/r) != 1 mod f for each prime r | q^n - 1 "
		    "= %s\n", qn - 1 < ((u128)1 << 64) ? fac_text((u64)(qn - 1)) :
		    "?");
		printf("  so an LFSR with this feedback polynomial has period %llu%s\n",
		    U(o), (u128)o + 1 == qn ? " (maximal: every nonzero state)" : "");
		if (o <= (1u << 22)) {
			struct rpoly x, h, one;
			u64 kk;
			rp_x(&x);
			rp_const(R, &one, r_const(R, 1));
			rp_divmod(R, &x, &f, NULL, &h);
			for (kk = 1; !rp_eq(&h, &one); kk++)
				rp_mulmod(R, &h, &h, &x, &f);
			printf("  check: stepping x, x^2, ... mod f: x^k = 1 first at k = "
			    "%llu\n", U(kk));
		}
	}
}

static void
cmd_factor(struct rspec *s)
{
	struct rpoly f, prod, t;
	static struct rpoly fac[RMAXDEG + 1];
	int mult[RMAXDEG + 1], k, i, j, allirr = 1;
	u64 lead;
	const struct ring *R = &s->R;

	need_field(s, "factor");
	poly_arg(R, &f);
	if (f.d < 1) {
		printf("%s is a constant\n", rp_str(R, &f));
		return;
	}
	k = rp_factor(R, &f, fac, mult, RMAXDEG + 1, &lead);
	if (k == 1 && mult[0] == 1 && lead == r_const(R, 1))
		printf("%s is irreducible over %s\n", rp_str(R, &f), field_name(s));
	else
		printf("%s = %s over %s\n", rp_str(R, &f), fac_str(R, fac, mult, k,
		    lead), field_name(s));
	printf("  how: square-free split (gcd(f, f'), p-th roots where f' = 0), "
	    "distinct-degree split (gcd(x^(q^i) - x, f)), Cantor-Zassenhaus "
	    "equal-degree split%s\n", R->n == 2 ? " (by the trace map)" : "");
	rp_const(R, &prod, lead);
	for (i = 0; i < k; i++) {
		for (j = 0; j < mult[i]; j++)
			rp_mul(R, &prod, &prod, &fac[i]);
		if (!rp_irreducible(R, &fac[i]))
			allirr = 0;
	}
	printf("  check: the product of the factors is %s; each factor is %s "
	    "(Rabin)", rp_eq(&prod, &f) ? "f" : "NOT f", allirr ? "irreducible" :
	    "NOT irreducible");
	{
		struct rpoly fd, g;
		rp_deriv(R, &fd, &f);
		rp_gcd(R, &g, &f, &fd);
		if (g.d == 0)
		{
			int bc = rp_berlekamp_count(R, &f);
			printf("; Berlekamp counts %d distinct factor%s", bc,
			    bc == 1 ? "" : "s");
		}
	}
	printf("\n");
	if (f.d >= 1 && R->qm1 < 64) {
		/* roots by evaluation */
		u64 x, c = 0;
		for (x = 0; x <= R->qm1; x++)
			if (rp_eval(R, &f, x) == 0)
				c++;
		for (i = 0, j = 0; i < k; i++)
			j += fac[i].d == 1;
		printf("  roots in %s: %llu by evaluation at every element, the "
		    "number of distinct linear factors (%d)\n", field_name(s), U(c), j);
	}
	(void)t;
}

static void
cmd_roots(struct rspec *s)
{
	struct rpoly f;
	const struct ring *R = &s->R;
	u64 x, list[64];
	int n = 0;

	poly_arg(R, &f);
	if (f.d < 1)
		rfail("a polynomial of degree >= 1");
	if (R->field) {
		static struct rpoly fac[RMAXDEG + 1];
		int mult[RMAXDEG + 1], k, i;
		u64 lead;
		k = rp_factor(R, &f, fac, mult, RMAXDEG + 1, &lead);
		for (i = 0; i < k; i++)
			if (fac[i].d == 1 && n < 64)
				list[n++] = r_neg(R, fac[i].c[0]);
		{
			int mm[64], j2;
			n = 0;
			for (i = 0; i < k; i++)
				if (fac[i].d == 1 && n < 64) {
					mm[n] = mult[i];
					list[n++] = r_neg(R, fac[i].c[0]);
				}
			for (i = 1; i < n; i++)
				for (j2 = i; j2 > 0 && list[j2 - 1] > list[j2]; j2--) {
					u64 t = list[j2];
					int tm = mm[j2];
					list[j2] = list[j2 - 1];
					mm[j2] = mm[j2 - 1];
					list[j2 - 1] = t;
					mm[j2 - 1] = tm;
				}
			printf("roots of %s in %s:", rp_str(R, &f), field_name(s));
			for (i = 0; i < n; i++) {
				printf(" %s", el_code(R, list[i]));
				if (mm[i] > 1)
					printf(" (multiplicity %d)", mm[i]);
			}
			printf("%s\n", n ? "" : " none");
		}
		printf("  how: the linear factors (gcd with x^q - x, then "
		    "Cantor-Zassenhaus)\n");
		if (R->qm1 < (1u << 20)) {
			u64 c = 0;
			for (x = 0; x <= R->qm1; x++)
				c += rp_eval(R, &f, x) == 0;
			printf("  check: evaluation at all %s elements: %llu root%s\n",
			    u128s((u128)R->qm1 + 1), U(c), c == 1 ? "" : "s");
		}
		return;
	}
	if (R->qm1 >= (1u << 22))
		rfail("roots over %s (not a field) by search: at most 2^22 elements",
		    s->name);
	for (x = 0; x <= R->qm1; x++)
		if (rp_eval(R, &f, x) == 0) {
			if (n < 64)
				list[n] = x;
			n++;
		}
	printf("roots of %s in %s: %d", rp_str(R, &f), s->name, n);
	if (n) {
		int i;
		printf(":");
		for (i = 0; i < n && i < 64; i++)
			printf(" %s", el_code(R, list[i]));
	}
	printf("\n  how: evaluation at every element (%s is not a field: a "
	    "polynomial can have more roots than its degree)\n", s->name);
}

static void
cmd_xorder(struct rspec *s)
{
	struct rpoly f;
	const struct ring *R = &s->R;
	u64 o;
	int of;

	need_field(s, "order");
	poly_arg(R, &f);
	if (f.d < 1)
		rfail("a polynomial of degree >= 1");
	if (f.c[0] == 0)
		rfail("x divides %s, so x has no order mod it", rp_str(R, &f));
	o = rp_xorder(R, &f, &of);
	if (of)
		rfail("the order does not fit 64 bits");
	printf("the order of x mod %s over %s: %llu\n", rp_str(R, &f),
	    field_name(s), U(o));
	printf("  the period of an LFSR (or of a CRC's register) with this "
	    "feedback polynomial: x^%llu = 1 mod f\n", U(o));
	printf("  how: factor f; for an irreducible factor of degree d the order "
	    "divides q^d - 1; a factor to the power e multiplies it by the least "
	    "p^t >= e; the lcm over the factors\n");
	if (o <= (1u << 22)) {
		struct rpoly x, h, one;
		u64 kk;
		rp_x(&x);
		rp_const(R, &one, r_const(R, 1));
		rp_divmod(R, &x, &f, NULL, &h);
		for (kk = 1; !rp_eq(&h, &one); kk++)
			rp_mulmod(R, &h, &h, &x, &f);
		printf("  check: stepping x, x^2, ... mod f: x^k = 1 first at k = "
		    "%llu\n", U(kk));
	} else {
		struct rpoly x, h, one;
		struct nt_fac nf;
		int i, ok;
		rp_x(&x);
		rp_const(R, &one, r_const(R, 1));
		rp_powmod(R, &h, &x, o, &f);
		ok = rp_eq(&h, &one);
		nt_factor(o, &nf);
		for (i = 0; i < nf.k; i++) {
			rp_powmod(R, &h, &x, o / nf.p[i], &f);
			ok &= !rp_eq(&h, &one);
		}
		printf("  check: x^%llu = 1 mod f and x^(%llu/r) != 1 for each prime r "
		    "| %llu = %s (repeated squaring): %s\n", U(o), U(o), U(o),
		    fac_text(o), ok ? "so the order is exactly that" : "FAILED");
	}
}

static void
cmd_poly(struct rspec *s)
{
	struct rpoly p;
	const struct ring *R = &s->R;
	int a = rti;

	if (s->kind != RS_ZN && s->kind != RS_EXT)
		rfail("poly: over Z/n or GF(q)");
	pexpr_into(R, &p, "x");
	if (!at_end())
		rfail("unexpected %s", here());
	printf("%s = %s over %s\n", eq_text(a, rlim), rp_str(R, &p),
	    field_name(s));
	printf("  how: polynomial arithmetic with coefficients in %s\n",
	    field_name(s));
}

static void
cmd_pgcd(struct rspec *s)
{
	struct rpoly a, b, g, u, v, t1, t2;
	const struct ring *R = &s->R;
	int i, lim = rlim, depth = 0;

	need_field(s, "gcd");
	for (i = rti; i < lim; i++) {
		if (rt[i].k == RK_SYM && rt[i].s[0] == '(')
			depth++;
		if (rt[i].k == RK_SYM && rt[i].s[0] == ')')
			depth--;
		if (depth == 0 && ((rt[i].k == RK_SYM && rt[i].s[0] == ',') ||
		    (rt[i].k == RK_ID && !strcasecmp(rt[i].s, "and"))))
			break;
	}
	if (i == lim)
		rfail("gcd A, B over a field");
	rlim = i;
	pexpr_into(R, &a, "x");
	rti = i + 1;
	rlim = lim;
	pexpr_into(R, &b, "x");
	rp_xgcd(R, &g, &u, &v, &a, &b);
	printf("gcd(%s, %s) = %s over %s\n", rp_str(R, &a), rp_str(R, &b),
	    rp_str(R, &g), field_name(s));
	printf("  how: Euclid; Bezout: (%s)(%s) + (%s)(%s) = %s\n", rp_str(R, &u),
	    rp_str(R, &a), rp_str(R, &v), rp_str(R, &b), rp_str(R, &g));
	rp_mul(R, &t1, &u, &a);
	rp_mul(R, &t2, &v, &b);
	rp_add(R, &t1, &t1, &t2);
	{
		struct rpoly r1, r2;
		int d1 = g.d < 0 || (rp_divmod(R, &a, &g, NULL, &r1) && r1.d < 0);
		int d2 = g.d < 0 || (rp_divmod(R, &b, &g, NULL, &r2) && r2.d < 0);
		printf("  check: the Bezout sum is %s; the gcd divides %s\n",
		    rp_eq(&t1, &g) ? "the gcd" : "NOT the gcd", d1 && d2 ? "both" :
		    "NOT both");
	}
}

/* ---- matrices ---- */

static u64
m_entry(const struct ring *R)
{
	ER = R;
	e_alt = 0;
	return e_sum();
}

static void
parse_matrix(const struct ring *R, struct rmat *m)
{
	static u64 v[64 * 64];
	int r = 0, c = 0, cols = -1, n = 0;

	expect_sym('[');
	if (at_sym('[')) {
		while (acc_sym('[')) {
			c = 0;
			while (!at_sym(']')) {
				if (n >= 64 * 64)
					rfail("matrices up to 64 x 64");
				v[n++] = m_entry(R);
				c++;
				acc_sym(',');
				if (at_end())
					rfail("expected ]");
			}
			expect_sym(']');
			if (cols >= 0 && c != cols)
				rfail("row %d has %d entries, not %d", r + 1, c, cols);
			cols = c;
			r++;
			acc_sym(',') || acc_sym(';');
		}
		expect_sym(']');
	} else {
		r = 1;
		c = 0;
		while (!at_sym(']')) {
			if (n >= 64 * 64)
				rfail("matrices up to 64 x 64");
			v[n++] = m_entry(R);
			c++;
			if (acc_sym(';')) {
				if (cols >= 0 && c != cols)
					rfail("row %d has %d entries, not %d", r, c, cols);
				cols = c;
				c = 0;
				r++;
			} else
				acc_sym(',');
			if (at_end())
				rfail("expected ]");
		}
		expect_sym(']');
		if (cols >= 0 && c != cols)
			rfail("row %d has %d entries, not %d", r, c, cols);
		cols = c;
	}
	if (r == 0 || cols <= 0)
		rfail("an empty matrix");
	rm_init(m, r, cols);
	memcpy(m->a, v, sizeof(u64) * (size_t)n);
}

static void
rm_print(const struct ring *R, const struct rmat *m, const char *ind)
{
	int i, j, w = 1;

	for (i = 0; i < m->r * m->c; i++) {
		int l = (int)strlen(el_code(R, m->a[i]));
		if (l > w)
			w = l;
	}
	for (i = 0; i < m->r; i++) {
		printf("%s[", ind);
		for (j = 0; j < m->c; j++)
			printf("%s%*s", j ? " " : "", w, el_code(R, RM(m, i, j)));
		printf("]\n");
	}
}

static int
rm_is_identity(const struct ring *R, const struct rmat *m)
{
	int i, j;

	for (i = 0; i < m->r; i++)
		for (j = 0; j < m->c; j++)
			if (RM(m, i, j) != (i == j ? r_const(R, 1) : 0))
				return 0;
	return 1;
}

enum { MX_ALL, MX_DET, MX_RANK, MX_INV, MX_NULL, MX_SOLVE };

static void
cmd_matrix(struct rspec *s, int what)
{
	const struct ring *R = &s->R;
	struct rmat m, inv, ns, prod, b;
	u64 det;
	int i, j, has_b = 0;

	if (s->kind != RS_ZN && s->kind != RS_EXT)
		rfail("matrices over Z/n or GF(q)");
	parse_matrix(R, &m);
	if (what == MX_SOLVE) {
		acc_sym(',');
		acc_id("x");
		acc_sym('=');
		parse_matrix(R, &b);
		if (b.r == 1 && b.c == m.r) {
			b.r = b.c;
			b.c = 1;
		}
		if (b.r != m.r || b.c != 1)
			rfail("b must have %d entries", m.r);
		has_b = 1;
	}
	if (!at_end())
		rfail("unexpected %s after the matrix", here());
	if (what == MX_ALL || what == MX_DET || what == MX_INV) {
		printf("the %dx%d matrix over %s\n", m.r, m.c, field_name(s));
		rm_print(R, &m, "  ");
	}
	if (what == MX_DET && m.r != m.c)
		printf("  no determinant: the matrix is not square\n");
	if ((what == MX_ALL || what == MX_DET || what == MX_INV) && m.r == m.c) {
		if (rm_det(R, &m, &det))
			printf("  det = %s%s\n", el_code(R, det), R->field ? "" : (R->deg
			    == 1 && r_isunit(R, det)) ? " (a unit)" : " (not a unit)");
		if (what == MX_DET)
			printf("  how: %s\n", R->field ? "Gaussian elimination mod p, "
			    "the product of the pivots" : "row reduction by Euclid's "
			    "algorithm on each column (only swaps and row subtractions, "
			    "which keep the determinant up to sign)");
	}
	if (what == MX_ALL || what == MX_RANK || what == MX_NULL) {
		struct rmat t;
		if (!R->field)
			printf("  rank: over %s, which is not a field, rank is not "
			    "well defined; reduce mod each prime factor\n", s->name);
		else {
			int rk;
			rm_init(&t, m.r, m.c);
			memcpy(t.a, m.a, sizeof(u64) * (size_t)m.r * m.c);
			rk = rm_rref(R, &t, NULL);
			printf("  rank %d over %s%s\n", rk, field_name(s), what ==
			    MX_RANK ? " (Gaussian elimination)" : "");
			if (what == MX_RANK) {
				printf("  row echelon form:\n");
				rm_print(R, &t, "    ");
			}
			rm_free(&t);
			if (what != MX_RANK) {
				int k = rm_nullspace(R, &m, &ns), bad = 0;
				printf("  nullspace: dimension %d%s\n", k, k ? ", basis "
				    "(rows):" : "");
				if (k)
					rm_print(R, &ns, "    ");
				for (i = 0; i < k; i++)
					for (j = 0; j < m.r; j++) {
						u64 acc = 0;
						int c;
						for (c = 0; c < m.c; c++)
							acc = r_add(R, acc, r_mul_alt(R, RM(&m, j, c),
							    RM(&ns, i, c)));
						bad += acc != 0;
					}
				if (k)
					printf("  check: M v = 0 for each basis vector: %s; "
					    "rank + nullity = %d columns\n", bad ? "FAILED" :
					    "holds", m.c);
				rm_free(&ns);
			}
		}
	}
	if ((what == MX_ALL || what == MX_INV) && m.r == m.c) {
		if (rm_inverse(R, &m, &inv)) {
			printf("  inverse:\n");
			rm_print(R, &inv, "    ");
			rm_mul(R, &prod, &m, &inv);
			printf("  check: M * M^-1 = %s\n", rm_is_identity(R, &prod) ?
			    "I" : "NOT I");
			rm_free(&prod);
			rm_free(&inv);
		} else
			printf("  no inverse: det is not a unit%s\n", R->field ?
			    " (it is 0)" : "");
	} else if (what == MX_INV)
		printf("  no inverse: the matrix is not square\n");
	if (what == MX_SOLVE && has_b) {
		struct rmat aug;
		int rk, *piv, ok = 1;
		if (!R->field) {
			if (m.r == m.c && rm_inverse(R, &m, &inv)) {
				rm_mul(R, &prod, &inv, &b);
				printf("x = (");
				for (i = 0; i < prod.r; i++)
					printf("%s%s", i ? ", " : "", el_code(R, prod.a[i]));
				printf(") over %s (the only solution: det is a unit)\n",
				    s->name);
				rm_free(&inv);
			} else {
				/* search, when small */
				u128 tot = 1;
				int j2;
				for (j2 = 0; j2 < m.c; j2++)
					tot *= (u128)R->n;
				if (tot > (1u << 20))
					rfail("solving over %s (not a field) when the matrix is "
					    "not invertible: solve mod each prime power", s->name);
				{
					u64 v[64], cnt = 0, xx, first[64];
					int r2, c2;
					for (xx = 0; xx < (u64)tot; xx++) {
						u64 t = xx;
						for (c2 = 0; c2 < m.c; c2++) {
							v[c2] = t % R->n;
							t /= R->n;
						}
						for (r2 = 0; r2 < m.r; r2++) {
							u64 acc = 0;
							for (c2 = 0; c2 < m.c; c2++)
								acc = r_add(R, acc, r_mul(R, RM(&m, r2, c2),
								    v[c2]));
							if (acc != b.a[r2])
								break;
						}
						if (r2 == m.r) {
							if (!cnt)
								memcpy(first, v, sizeof(u64) * (size_t)m.c);
							cnt++;
						}
					}
					if (!cnt)
						printf("no solution over %s (searched all %s vectors)\n",
						    s->name, u128s(tot));
					else {
						printf("x = (");
						for (c2 = 0; c2 < m.c; c2++)
							printf("%s%llu", c2 ? ", " : "", U(first[c2]));
						printf(") over %s, %s %llu solution%s (searched all %s "
						    "vectors; det is not a unit)\n", s->name, cnt > 1 ?
						    "one of" : "the only one of", U(cnt), cnt == 1 ?
						    "" : "s", u128s(tot));
					}
					rm_free(&m);
					rm_free(&b);
					return;
				}
			}
		} else {
			rm_init(&aug, m.r, m.c + 1);
			for (i = 0; i < m.r; i++) {
				for (j = 0; j < m.c; j++)
					RM(&aug, i, j) = RM(&m, i, j);
				RM(&aug, i, m.c) = b.a[i];
			}
			piv = xcalloc((size_t)m.r + 1, sizeof *piv);
			rk = rm_rref(R, &aug, piv);
			for (i = 0; i < rk; i++)
				if (piv[i] == m.c)
					ok = 0;
			if (!ok) {
				printf("no solution: the system is inconsistent over %s (b "
				    "is not in the column space)\n", field_name(s));
				free(piv);
				rm_free(&aug);
				return;
			}
			rm_init(&prod, m.c, 1);
			for (i = 0; i < rk; i++)
				prod.a[piv[i]] = RM(&aug, i, m.c);
			printf("x = (");
			for (i = 0; i < m.c; i++)
				printf("%s%s", i ? ", " : "", el_code(R, prod.a[i]));
			printf(")");
			{
				int k = rm_nullspace(R, &m, &ns);
				if (k) {
					printf(" + any combination of the %d nullspace vector%s:\n",
					    k, k == 1 ? "" : "s");
					rm_print(R, &ns, "    ");
				} else
					printf(", the only solution\n");
				rm_free(&ns);
			}
			free(piv);
			rm_free(&aug);
		}
		printf("  how: %s\n", R->field ? "Gaussian elimination on [M | b]" :
		    "x = M^-1 b");
		{
			struct rmat mx;
			rm_mul(R, &mx, &m, &prod);
			for (i = 0; i < m.r && mx.a[i] == b.a[i]; i++)
				;
			printf("  check: M x = b %s\n", i == m.r ? "holds" : "FAILS");
			rm_free(&mx);
		}
		rm_free(&prod);
		rm_free(&b);
	}
	rm_free(&m);
}
