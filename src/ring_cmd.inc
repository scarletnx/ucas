/* ring.c, part 5: the commands' entry point (parse.c calls ring_command) */

enum { C_RING = 1, C_ZMOD, C_GF, C_INVERSE, C_ORDER, C_PHI, C_LAMBDA,
	C_FACTORINT, C_LEGENDRE, C_JACOBI, C_SQRTMOD, C_QR, C_PRIMROOT, C_DLOG,
	C_CRT, C_CONGRUENCE, C_SOLVE, C_MAGIC, C_FACTOR, C_IRRED, C_PRIMITIVE,
	C_ROOTS, C_POLY, C_GCD, C_MATRIX, C_DET, C_RANK, C_NULL, C_MATINV,
	C_MATSOLVE, C_GF2M, C_NORM, C_LOG, C_SQRT };

static const struct {
	const char *w;
	int c;
	int shared;     /* also a word of other commands: needs "mod R" etc. */
} rcmds[] = {
	{"ring", C_RING, 0}, {"zmod", C_ZMOD, 0}, {"gf", C_GF, 0},
	{"GF", C_GF, 0}, {"inverse", C_INVERSE, 1}, {"inv", C_INVERSE, 1},
	{"invmod", C_INVERSE, 0}, {"modinv", C_INVERSE, 0},
	{"order", C_ORDER, 1}, {"ord", C_ORDER, 1},
	{"multorder", C_ORDER, 0}, {"phi", C_PHI, 0}, {"totient", C_PHI, 0},
	{"carmichael", C_LAMBDA, 0}, {"lambda", C_LAMBDA, 0},
	{"factorint", C_FACTORINT, 0}, {"legendre", C_LEGENDRE, 0},
	{"jacobi", C_JACOBI, 0}, {"sqrtmod", C_SQRTMOD, 0}, {"qr", C_QR, 0},
	{"primroot", C_PRIMROOT, 0}, {"primroots", C_PRIMROOT, 0},
	{"dlog", C_DLOG, 0}, {"crt", C_CRT, 0}, {"congruence", C_CONGRUENCE, 0},
	{"solve", C_SOLVE, 1}, {"magic", C_MAGIC, 0}, {"factor", C_FACTOR, 1},
	{"irreducible", C_IRRED, 0}, {"primitive", C_PRIMITIVE, 1},
	{"roots", C_ROOTS, 1}, {"poly", C_POLY, 1}, {"gcd", C_GCD, 1},
	{"matrix", C_MATRIX, 1}, {"det", C_DET, 1}, {"rank", C_RANK, 1},
	{"nullspace", C_NULL, 1}, {"kernel", C_NULL, 1}, {"matinv", C_MATINV, 0},
	{"matsolve", C_MATSOLVE, 0}, {"gf2matrix", C_GF2M, 0},
	{"gf2map", C_GF2M, 0}, {"norm", C_NORM, 1}, {"log", C_LOG, 1},
	{"sqrt", C_SQRT, 1}, {NULL, 0, 0}
};

static int
args_have_x(int a, int b)
{
	int i;

	for (i = a; i < b; i++)
		if (rt[i].k == RK_ID && (!strcmp(rt[i].s, "x") ||
		    named_index(rt[i].s) >= 0))
			return 1;
	return 0;
}

static void
ring_dispatch(int c, const char *text)
{
	static struct rspec s;
	int end, i;

	if (c == C_GF2M) {
		cmd_gf2matrix(text);
		return;
	}
	rtokenize(text);
	switch (c) {
	case C_RING:
	case C_ZMOD:
	case C_GF:
		if (c == C_ZMOD) {
			memset(&s, 0, sizeof s);
			acc_sym('(');
			ring_zn(&s.R, modulus_arg());
			acc_sym(')');
			s.kind = RS_ZN;
			spec_name(&s);
		} else if (!parse_ring(&s))
			rfail("%s: a ring such as Z/12, GF(2^8), GF(2^8, aes), "
			    "GF(2)[x]/(x^4 + 1), Z[i], Z[sqrt(2)], Q[x]/(x^2 - 2)",
			    c == C_GF ? "GF" : "ring");
		if (s.kind == RS_ZI || s.kind == RS_ZSQRT) {
			cmd_quadratic(&s);
			return;
		}
		if (s.kind == RS_QX) {
			cmd_qx(&s);
			return;
		}
		if (at_end()) {
			if (s.kind == RS_ZN && !s.gf)
				zn_info(s.R.n, SUB_ALL);
			else if (s.kind == RS_ZN) {
				printf("GF(%llu) = Z/%llu, the integers mod the prime %llu\n",
				    U(s.R.n), U(s.R.n), U(s.R.n));
				zn_info(s.R.n, SUB_ALL);
			} else
				ext_info(&s, SUB_ALL);
			return;
		}
		if ((i = sub_word()) >= 0) {
			if (s.kind == RS_ZN)
				zn_info(s.R.n, i);
			else
				ext_info(&s, i);
			return;
		}
		if (acc_id("log") || acc_id("dlog")) {
			acc_id("of");
			ring_log(&s);
			return;
		}
		if (acc_id("element") || acc_id("info") || acc_id("minpoly") ||
		    acc_id("about")) {
			if (s.kind == RS_ZN) {
				i128 v = iexpr_sum();
				zn_element(s.R.n, v);
			} else
				gf_element(&s, eexpr(&s.R, 0));
			return;
		}
		if (rlim - rti == 1 && (rt[rti].k == RK_NUM || (s.kind == RS_EXT &&
		    (at_id("x") || at_id("a"))))) {
			if (s.kind == RS_ZN)
				zn_element(s.R.n, (i128)(rt[rti].v % s.R.n));
			else
				gf_element(&s, eexpr(&s.R, 0));
			return;
		}
		ring_eval(&s);
		return;
	case C_PHI:
		cmd_phi(0);
		return;
	case C_LAMBDA:
		cmd_phi(1);
		return;
	case C_FACTORINT:
		cmd_factorint();
		return;
	case C_LEGENDRE:
	case C_JACOBI:
		cmd_symbol(c == C_LEGENDRE);
		return;
	case C_MAGIC:
		cmd_magic();
		return;
	case C_CRT:
	case C_CONGRUENCE:
		cmd_congruence();
		return;
	default:
		break;
	}
	/* the rest work in a ring named at the end */
	if (c == C_INVERSE || c == C_ORDER) {
		int k;
		if (pow2_modulus(0, &k, &end) && (k >= 64 || c == C_INVERSE) &&
		    !at_sym('[')) {
			i128 a;
			rlim = end;
			a = iexpr_sum();
			if (!at_end())
				rfail("unexpected %s", here());
			if (c == C_INVERSE)
				cmd_inverse_pow2(a, k);
			else
				rfail("order mod 2^%d: moduli below 2^64", k);
			return;
		}
	}
	if (!find_ring(&s, 0, &end, 1)) {
		if (c == C_IRRED || c == C_PRIMITIVE || c == C_FACTOR) {
			default_gf2(&s, c == C_FACTOR ? "factor" : c == C_IRRED ?
			    "irreducible" : "primitive");
			end = nrt;
		} else if (c == C_PRIMROOT || c == C_QR || c == C_SQRTMOD) {
			/* primroot N [A], qr N, sqrtmod A N */
			i128 a, n;
			if (c == C_SQRTMOD) {
				a = iexpr_unary();
				acc_sym(',');
				n = iexpr_unary();
				if (n < 2 || (n >> 64))
					rfail("sqrtmod A N with 2 <= N < 2^64");
				cmd_sqrtmod_int(a, (u64)n);
				return;
			}
			n = iexpr_unary();
			if (n < 1 || (n >> 64))
				rfail("a modulus from 1 to 2^64 - 1");
			if (c == C_QR) {
				cmd_qr((u64)n);
				return;
			}
			acc_sym(',');
			if (!at_end()) {
				a = iexpr_unary();
				cmd_primroot_int((u64)n, 1, (u64)(((a % n) + n) % n));
			} else
				cmd_primroot_int((u64)n, 0, 0);
			return;
		} else
			rfail("over which ring? (... mod n, over GF(q), in Z[i])");
	}
	rlim = end;
	switch (c) {
	case C_INVERSE:
		if (at_sym('[')) {
			cmd_matrix(&s, MX_INV);
			return;
		}
		cmd_inverse();
		return;
	case C_ORDER:
		if (args_have_x(rti, rlim) && s.kind == RS_ZN) {
			cmd_xorder(&s);
			return;
		}
		if (s.kind == RS_ZN) {
			i128 a = iexpr_sum();
			if (!at_end())
				rfail("unexpected %s", here());
			cmd_order_int(a, s.R.n);
			return;
		}
		if (s.kind == RS_EXT) {
			u64 a = eexpr(&s.R, 0), o = r_order(&s.R, a);
			if (!o)
				printf("%s is not a unit in %s: no order\n", el_code(&s.R, a),
				    s.name);
			else
				printf("ord(%s) = %llu in %s (it divides %llu)\n",
				    el_code(&s.R, a), U(o), s.name, U(s.R.qm1));
			return;
		}
		rfail("order: in Z/n or GF(q)");
		return;
	case C_SQRTMOD:
	case C_SQRT:
		if (s.kind == RS_ZN) {
			i128 a = iexpr_sum();
			if (!at_end())
				rfail("unexpected %s", here());
			cmd_sqrtmod_int(a, s.R.n);
			return;
		}
		if (s.kind == RS_EXT && s.R.field) {
			u64 a = eexpr(&s.R, 0), r;
			if (s.R.n == 2) {
				r = r_pow(&s.R, a, (s.R.qm1 + 1) / 2);
				printf("sqrt(%s) = %s in %s (a^(q/2): squaring is a bijection "
				    "in characteristic 2)\n", el_code(&s.R, a), el_str(&s.R, r),
				    s.name);
				printf("  check: %s^2 = %s\n", el_code(&s.R, r), el_code(&s.R,
				    r_mul_alt(&s.R, r, r)));
				return;
			}
			if (a == 0 || r_pow(&s.R, a, s.R.qm1 / 2) == 1) {
				/* search for small fields, else not done */
				u64 x;
				if (s.R.qm1 < (1u << 22)) {
					for (x = 0; x <= s.R.qm1; x++)
						if (r_mul(&s.R, x, x) == a)
							break;
					printf("sqrt(%s) = +-%s in %s\n", el_code(&s.R, a),
					    el_code(&s.R, x), s.name);
					printf("  how: Euler's criterion says a square (a^((q-1)/2) "
					    "= 1); found by search\n");
					printf("  check: %s^2 = %s\n", el_code(&s.R, x),
					    el_code(&s.R, r_mul_alt(&s.R, x, x)));
				} else
					printf("%s is a square in %s (a^((q-1)/2) = 1); the root "
					    "is not computed for fields this large\n",
					    el_code(&s.R, a), s.name);
			} else
				printf("%s is not a square in %s: a^((q-1)/2) = -1\n",
				    el_code(&s.R, a), s.name);
			return;
		}
		rfail("sqrt: in Z/n or GF(q)");
		return;
	case C_QR:
		if (s.kind != RS_ZN)
			rfail("qr mod n");
		if (!at_end()) {
			i128 a = iexpr_sum();
			cmd_sqrtmod_int(a, s.R.n);
			return;
		}
		cmd_qr(s.R.n);
		return;
	case C_PRIMROOT:
	case C_PRIMITIVE:
		if (c == C_PRIMITIVE && (args_have_x(rti, rlim) ||
		    s.kind != RS_ZN || !s.R.nprime ? args_have_x(rti, rlim) : 0)) {
			cmd_irreducible(&s, 1);
			return;
		}
		if (s.kind == RS_EXT) {
			ring_tables(&s.R);
			if (at_end())
				ext_info(&s, SUB_GEN);
			else {
				u64 a = eexpr(&s.R, 0), o = r_order(&s.R, a);
				printf("%s: %s %s a generator of %s* (order %llu of %llu)\n",
				    o == s.R.qm1 ? "yes" : "no", el_code(&s.R, a),
				    o == s.R.qm1 ? "is" : "is not", s.name, U(o), U(s.R.qm1));
			}
			return;
		}
		if (s.kind != RS_ZN)
			rfail("primitive roots: mod n or in GF(q)");
		if (at_end())
			cmd_primroot_int(s.R.n, 0, 0);
		else {
			i128 a = iexpr_sum();
			cmd_primroot_int(s.R.n, 1, (u64)(((a % (i128)s.R.n) +
			    (i128)s.R.n) % (i128)s.R.n));
		}
		return;
	case C_DLOG:
	case C_LOG:
		if (s.kind != RS_ZN && s.kind != RS_EXT)
			rfail("discrete logs in Z/n or GF(q)");
		acc_id("of");
		ring_log(&s);
		return;
	case C_SOLVE:
		if (at_sym('[')) {
			cmd_matrix(&s, MX_SOLVE);
			return;
		}
		rlim = nrt;
		rti = 0;
		cmd_congruence();
		return;
	case C_FACTOR:
		if (s.kind == RS_ZI) {
			QD = -1;
			{
				struct qi z = qi_sum();
				if (!at_end())
					rfail("unexpected %s", here());
				qi_factor_gauss(z);
			}
			return;
		}
		if (s.kind == RS_ZN && !args_have_x(rti, rlim) && !s.R.nprime)
			rfail("factor: a polynomial in x mod p (factorint N for integers)");
		cmd_factor(&s);
		return;
	case C_IRRED:
		if (s.kind == RS_QX) {
			cmd_qx(&s);
			return;
		}
		cmd_irreducible(&s, 0);
		return;
	case C_ROOTS:
		cmd_roots(&s);
		return;
	case C_POLY:
		cmd_poly(&s);
		return;
	case C_GCD:
		if (s.kind == RS_ZI || s.kind == RS_ZSQRT) {
			int a = rti;
			char buf[1024];
			(void)a;
			snprintf(buf, sizeof buf, "gcd");
			QD = s.d;
			{
				struct qi x, y, g, q, r1, r2;
				x = qi_sum();
				if (!acc_sym(',') && !acc_id("and"))
					rfail("gcd A, B in Z[i]");
				y = qi_sum();
				if (!at_end())
					rfail("unexpected %s", here());
				g = qi_gcd(x, y);
				printf("gcd(%s, %s) = %s in %s (up to a unit)\n", qi_str(x),
				    qi_str(y), qi_str(g), s.name);
				printf("  how: Euclid, the quotient rounded to the nearest "
				    "lattice point; norm %s\n", i128s(qi_norm(g)));
				printf("  check: it divides both: remainders %s and %s\n",
				    (qi_divmod(x, g, &q, &r1), qi_str(r1)),
				    (qi_divmod(y, g, &q, &r2), qi_str(r2)));
			}
			return;
		}
		cmd_pgcd(&s);
		return;
	case C_NORM:
		if (s.kind == RS_ZI || s.kind == RS_ZSQRT) {
			struct qi x;
			QD = s.d;
			x = qi_sum();
			if (!at_end())
				rfail("unexpected %s", here());
			printf("N(%s) = %s in %s\n", qi_str(x), i128s(qi_norm(x)), s.name);
			printf("  how: N(a + b*%s) = (a + b*%s)(a - b*%s) = a^2 - (%lld) "
			    "b^2\n", s.d == -1 ? "i" : "sqrt(d)", s.d == -1 ? "i" :
			    "sqrt(d)", s.d == -1 ? "i" : "sqrt(d)", (long long)s.d);
			return;
		}
		if (s.kind == RS_EXT && s.R.field) {
			u64 a = eexpr(&s.R, 0);
			printf("N(%s) = %s (to GF(%llu): a^((q-1)/(p-1)))\n",
			    el_code(&s.R, a), el_code(&s.R, r_pow(&s.R, a, s.R.qm1 /
			    (s.R.n - 1))), U(s.R.n));
			return;
		}
		rfail("norm: in Z[i], Z[sqrt(d)] or GF(q)");
		return;
	case C_MATRIX:
		cmd_matrix(&s, MX_ALL);
		return;
	case C_DET:
		cmd_matrix(&s, MX_DET);
		return;
	case C_RANK:
		cmd_matrix(&s, MX_RANK);
		return;
	case C_NULL:
		cmd_matrix(&s, MX_NULL);
		return;
	case C_MATINV:
		cmd_matrix(&s, MX_INV);
		return;
	case C_MATSOLVE:
		cmd_matrix(&s, MX_SOLVE);
		return;
	}
	rfail("not done");
}

int
ring_command(struct P *p)
{
	static char text[8192];
	char w[48];
	uint32_t pos = cmdp_pos(p);
	int k, c = 0, shared = 0, n;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (k = 0; rcmds[k].w; k++)
		if (!strcmp(rcmds[k].w, w)) {
			c = rcmds[k].c;
			shared = rcmds[k].shared;
			break;
		}
	if (!c || func_find(sym_cstr(w))) {
		cmdp_seek(p, pos);
		return 0;
	}
	text[0] = 0;
	n = 0;
	if (c == C_GF)
		n = snprintf(text, sizeof text, "GF");
	if (!cmdp_eos(p))
		cmdp_rest(p, text + n, sizeof text - (size_t)n, NULL);
	if (shared) {
		/* only with a ring: "order 2 mod 101", "factor x^4 + 1 over GF(2)" */
		static struct rspec s;
		int end, kk, ok;
		rtokenize(text);
		ok = find_ring(&s, 0, &end, 0) || pow2_modulus(0, &kk, &end);
		if (!ok && (c == C_IRRED || c == C_PRIMITIVE) && 0)
			ok = 1;
		if (!ok) {
			cmdp_seek(p, pos);
			return 0;
		}
	}
	notes_flush();
	ring_dispatch(c, text);
	return 1;
}

/* nlp.c: does this text (a command's arguments) name a ring at its end */
int
ring_has_ring(const char *text)
{
	static struct rspec s;
	int end, k, ok = 0;

	spec_depth++;
	if (setjmp(spec_jmp) == 0) {
		rtokenize(text);
		ok = find_ring(&s, 0, &end, 0) || pow2_modulus(0, &k, &end);
	}
	spec_depth--;
	notes_drop();
	return ok;
}
