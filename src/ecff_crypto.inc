/* ecff_cmd.c, part 2: cryptography commands and the test-vector table */

static int
hexval(int c)
{
	if (c >= '0' && c <= '9')
		return c - '0';
	if (c >= 'a' && c <= 'f')
		return c - 'a' + 10;
	if (c >= 'A' && c <= 'F')
		return c - 'A' + 10;
	return -1;
}

static size_t
hex2bytes(const char *s, unsigned char *b, size_t max)
{
	size_t n = 0;

	if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
		s += 2;
	while (hexval(s[0]) >= 0 && hexval(s[1]) >= 0 && n < max) {
		b[n++] = (unsigned char)(hexval(s[0]) * 16 + hexval(s[1]));
		s += 2;
	}
	return n;
}

static void
bytes2hex(char *o, const unsigned char *b, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++)
		sprintf(o + 2 * i, "%02x", b[i]);
	o[2 * n] = 0;
}

/* a hex byte string at sp */
static size_t
gethex(unsigned char *b, size_t max, const char *what)
{
	char tok[1200];
	size_t k = 0, n;

	skipws();
	while ((isxdigit((unsigned char)*sp) || *sp == 'x' || *sp == 'X') &&
	    k + 1 < sizeof tok)
		tok[k++] = *sp++;
	tok[k] = 0;
	n = hex2bytes(tok, b, max);
	if (n == 0 || (k > 2 && 2 * n + (tok[1] == 'x' ? 2 : 0) != k))
		fail("expected %s as hex bytes", what);
	return n;
}

/* msg TEXT (to the end of the line) or msghex HEX */
static size_t
getmsg(unsigned char *b, size_t max)
{
	skipws();
	if (takeword("msghex")) {
		skipws();
		if (*sp == 0)
			return 0;
		return gethex(b, max, "the message");
	}
	if (takeword("msg") || takeword("message")) {
		size_t n;
		while (*sp == ' ' || *sp == '\t')
			sp++;
		n = strlen(sp);
		while (n > 0 && (sp[n - 1] == ' ' || sp[n - 1] == '\t'))
			n--;
		if (n > max)
			n = max;
		memcpy(b, sp, n);
		sp += strlen(sp);
		return n;
	}
	fail("expected msg TEXT or msghex HEX");
	return 0;
}

static int
default_hash(void)
{
	int b = nn_bits(&cc.L.n);

	return b <= 256 ? 256 : b <= 384 ? 384 : 512;
}

static void
need_named_w(const char *cmd)
{
	if (!cc.named || cc.L.N->model != ECN_W)
		fail("%s: a named Weierstrass curve (secp256k1, P-256, P-384, "
		    "P-521, ...)", cmd);
}

static void
cmd_ecdsa(void)
{
	static unsigned char m[8192];
	size_t mn = 0;
	int sign, alg = 0, havekey = 0, havepub = 0, haver = 0, haves = 0,
	    havek = 0, havemsg = 0;
	nn d, r, s, k, kk;
	struct ecp_pt Q;

	if (takeword("sign"))
		sign = 1;
	else if (takeword("verify"))
		sign = 0;
	else
		fail("ecdsa sign CURVE key D msg TEXT | ecdsa verify CURVE pub "
		    "(x, y) r R s S msg TEXT");
	parse_curve();
	need_named_w("ecdsa");
	for (;;) {
		char w[16];
		skipws();
		if (!*sp)
			break;
		if (peekword("msg") || peekword("msghex") || peekword("message")) {
			mn = getmsg(m, sizeof m);
			havemsg = 1;
			break;
		}
		if (!getword(w, sizeof w))
			fail("unexpected \"%.20s\"", sp);
		if (!strcasecmp(w, "key") || !strcasecmp(w, "d")) {
			need_nn(&d, "the private key");
			havekey = 1;
		} else if (!strcasecmp(w, "pub")) {
			skipws();
			if (*sp == '(' || peekword("G"))
				parse_point(&Q, 1);
			else {
				unsigned char b[200];
				size_t n = gethex(b, sizeof b, "the public key");
				if (!sec1_decode(&cc.L, &Q, b, n))
					fail("the public key does not decode to a point "
					    "on the curve");
			}
			havepub = 1;
		} else if (!strcasecmp(w, "r")) {
			need_nn(&r, "r");
			haver = 1;
		} else if (!strcasecmp(w, "s")) {
			need_nn(&s, "s");
			haves = 1;
		} else if (!strcasecmp(w, "k") || !strcasecmp(w, "nonce")) {
			need_nn(&kk, "the nonce");
			havek = 1;
		} else if (!strcasecmp(w, "hash")) {
			char hn[16];
			getword(hn, sizeof hn);
			alg = !strcasecmp(hn, "sha256") ? 256 : !strcasecmp(hn,
			    "sha384") ? 384 : !strcasecmp(hn, "sha512") ? 512 : 0;
			if (!alg)
				fail("hash: sha256, sha384 or sha512");
		} else
			fail("unknown option %s", w);
	}
	if (!havemsg)
		fail("ecdsa: msg TEXT (or msghex HEX) at the end");
	if (!alg)
		alg = default_hash();
	if (sign) {
		struct ecp_pt P;
		char why[200];
		if (!havekey)
			fail("ecdsa sign: key D");
		if (!ecdsa_sign(&cc.L, alg, &d, m, mn, havek ? &kk : NULL, &r, &s,
		    &k))
			fail("ecdsa: this nonce gives r = 0 or s = 0");
		ecp_mul(CE, &P, &cc.L.G, &d);
		printf("signature (r, s) on %s with SHA-%d\n", cc.desc, alg);
		printf("  r: %s\n  s: %s\n", nn_hex(&r), nn_hex(&s));
		printf("  k: %s (%s)\n", nn_hex(&k), havek ? "given" :
		    "RFC 6979: HMAC-DRBG on the key and the message hash");
		printf("  public key: %s\n", pt(&P));
		printf("  how: r = x(k G) mod n, s = k^-1 (z + r d) mod n, z = the "
		    "leftmost bits of SHA-%d(msg)\n", alg);
		printf("  check: verifies with the public key: %s\n",
		    ecdsa_verify(&cc.L, alg, &P, m, mn, &r, &s) ? "yes" : "NO");
		printf("  check: public key valid: %s\n",
		    ec_validate_pub(&cc.L, &P, why, sizeof why) ? why : "NO");
		return;
	}
	if (!havepub || !haver || !haves)
		fail("ecdsa verify: pub, r and s");
	{
		char why[200];
		int ok = ecdsa_verify(&cc.L, alg, &Q, m, mn, &r, &s);
		int valid = ec_validate_pub(&cc.L, &Q, why, sizeof why);
		printf("%s\n", ok && valid ? "valid signature" :
		    "invalid signature");
		printf("  curve: %s\n  hash: SHA-%d\n", cc.desc, alg);
		printf("  public key: %s\n", valid ? why : why);
		printf("  how: w = s^-1, X = (z w) G + (r w) Q, valid iff x(X) = r "
		    "mod n\n");
	}
}

static void
cmd_ecdh(void)
{
	nn a, b;
	struct ecp_pt A, B, S1, S2, Q;
	char why[200];
	int havepub = 0;

	parse_curve();
	takeword("key") || takeword("a") || takeword("d");
	need_nn(&a, "a private key");
	skipws();
	if (takeword("pub")) {
		parse_point(&Q, 0);
		havepub = 1;
	} else {
		takeword("key") || takeword("b");
		need_nn(&b, "the second private key (or pub (x, y))");
	}
	if (!cc.named)
		fail("ecdh: a named curve (its generator G and order n)");
	ecp_mul(CE, &A, &cc.L.G, &a);
	if (havepub) {
		if (!ec_validate_pub(&cc.L, &Q, why, sizeof why))
			fail("ecdh: the peer key is not valid: %s", why);
		ecp_mul(CE, &S1, &Q, &a);
		printf("shared secret: x = %s\n", fe(S1.x));
		printf("  curve: %s\n  a G: %s\n  peer: %s\n  a peer: %s\n",
		    cc.desc, pt(&A), pt(&Q), pt(&S1));
		printf("  check: the peer key is valid: %s\n", why);
		return;
	}
	ecp_mul(CE, &B, &cc.L.G, &b);
	ecp_mul(CE, &S1, &B, &a);
	ecp_mul(CE, &S2, &A, &b);
	printf("shared secret: x = %s\n", fe(S1.x));
	printf("  curve: %s\n  A = a G: %s\n  B = b G: %s\n  a B: %s\n",
	    cc.desc, pt(&A), pt(&B), pt(&S1));
	printf("  check: a B = b A: %s\n", ecp_eq(CE, &S1, &S2) ? "yes" : "NO");
	printf("  check: A valid: %s\n", ec_validate_pub(&cc.L, &A, why,
	    sizeof why) ? why : "NO");
}

static void
cmd_xdh(int is448)
{
	unsigned char k[56], u[56], o[56];
	size_t L = is448 ? 56 : 32, n;
	char hk[200];

	memset(u, 0, sizeof u);
	n = gethex(k, L, "the scalar (bytes, little-endian)");
	if (n != L)
		fail("the scalar has %zu bytes", L);
	skipws();
	if (takeword("base") || !*sp)
		u[0] = is448 ? 5 : 9;
	else if ((n = gethex(u, L, "u (bytes, little-endian)")) != L)
		fail("u has %zu bytes", L);
	if (is448)
		x448(o, k, u);
	else
		x25519(o, k, u);
	bytes2hex(hk, o, L);
	printf("%s\n", hk);
	printf("  how: RFC 7748 Montgomery ladder on x only (A = %s, a24 = %s), "
	    "the scalar clamped\n", is448 ? "156326" : "486662",
	    is448 ? "39081" : "121665");
}

static void
cmd_ed25519(void)
{
	static unsigned char m[8192];
	unsigned char sk[32], pk[32], sig[64];
	size_t mn;
	char h1[200];

	if (takeword("pub")) {
		gethex(sk, 32, "the secret key");
		ed25519_pub(pk, sk);
		bytes2hex(h1, pk, 32);
		printf("%s\n  how: A = s B, s = the clamped low half of "
		    "SHA-512(secret)\n", h1);
		return;
	}
	if (takeword("sign")) {
		if (gethex(sk, 32, "the secret key") != 32)
			fail("the secret key has 32 bytes");
		mn = getmsg(m, sizeof m);
		ed25519_sign(sig, sk, m, mn);
		ed25519_pub(pk, sk);
		bytes2hex(h1, sig, 64);
		printf("%s\n", h1);
		bytes2hex(h1, pk, 32);
		printf("  public key: %s\n", h1);
		printf("  how: r = SHA-512(prefix || M) mod L, R = r B, S = r + "
		    "SHA-512(R || A || M) s mod L\n");
		printf("  check: verifies: %s\n", ed25519_verify(pk, m, mn, sig) ?
		    "yes" : "NO");
		return;
	}
	if (takeword("verify")) {
		int ok;
		if (gethex(pk, 32, "the public key") != 32)
			fail("the public key has 32 bytes");
		if (gethex(sig, 64, "the signature") != 64)
			fail("the signature has 64 bytes");
		mn = getmsg(m, sizeof m);
		ok = ed25519_verify(pk, m, mn, sig);
		printf("%s\n", ok ? "valid signature" : "invalid signature");
		printf("  how: [S] B = R + [SHA-512(R || A || M) mod L] A\n");
		return;
	}
	fail("ed25519 pub SK | sign SK msg TEXT | verify PK SIG msg TEXT");
}

static void
cmd_eccompress(void)
{
	struct ecp_pt P, Q;
	unsigned char b[200];
	char h[500];
	int n;

	parse_curve();
	if (!cc.named || cc.L.N->model != ECN_W)
		fail("eccompress: a named Weierstrass curve");
	parse_point(&P, 1);
	n = sec1_encode(&cc.L, &P, 1, b);
	bytes2hex(h, b, (size_t)n);
	printf("%s\n", h);
	printf("  compressed: %s\n", h);
	n = sec1_encode(&cc.L, &P, 0, b);
	bytes2hex(h, b, (size_t)n);
	printf("  uncompressed: %s\n", h);
	n = sec1_encode(&cc.L, &P, 1, b);
	printf("  check: decompresses to the point: %s\n",
	    sec1_decode(&cc.L, &Q, b, (size_t)n) && ecp_eq(CE, &P, &Q) ? "yes" :
	    "NO");
}

static void
cmd_ecdecompress(void)
{
	struct ecp_pt P;
	unsigned char b[200];
	size_t n;

	parse_curve();
	if (!cc.named || cc.L.N->model != ECN_W)
		fail("ecdecompress: a named Weierstrass curve");
	n = gethex(b, sizeof b, "a SEC 1 encoding");
	if (!sec1_decode(&cc.L, &P, b, n))
		fail("not the encoding of a point on %s", cc.desc);
	printf("%s\n", pt(&P));
	printf("  how: y = sqrt(x^3 + a x + b), the root with the parity in "
	    "the prefix\n");
	printf("  check: on the curve: %s\n", ecp_on_curve(CE, &P) ? "yes" :
	    "NO");
}

static void
cmd_ecvalidate(void)
{
	struct ecp_pt P;
	char why[200];
	int ok;

	parse_curve();
	if (!cc.named)
		fail("ecvalidate: a named curve");
	parse_point(&P, 0);
	ok = ec_validate_pub(&cc.L, &P, why, sizeof why);
	printf("%s\n", ok ? "valid public key" : "invalid public key");
	printf("  why: %s\n", why);
}

/* ---- the test vectors ---- */

struct ecvec {
	const char *src, *kind, *curve, *a, *b, *c, *want1, *want2;
};

static const struct ecvec vectors[] = {
	{"FIPS 180-4 example", "sha256", NULL, "abc", NULL, NULL,
	    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
	    NULL},
	{"FIPS 180-4 example", "sha384", NULL, "abc", NULL, NULL,
	    "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed"
	    "8086072ba1e7cc2358baeca134c825a7", NULL},
	{"FIPS 180-4 example", "sha512", NULL, "abc", NULL, NULL,
	    "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
	    "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
	    NULL},
	{"RFC 6979 A.2.5 (key)", "pub", "P-256",
	    "0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721",
	    NULL, NULL,
	    "0x60fed4ba255a9d31c961eb74c6356d68c049b8923b61fa6ce669622e60f29fb6",
	    "0x7903fe1008b8bc99a41ae9e95628bc64f2f1b20c2d7e9f5177a3c294d4462299"},
	{"RFC 6979 A.2.5 SHA-256 sample", "ecdsa256", "P-256",
	    "0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721",
	    "sample", "0xA6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60",
	    "0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716",
	    "0xF7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8"},
	{"RFC 6979 A.2.5 SHA-256 test", "ecdsa256", "P-256",
	    "0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721",
	    "test", "0xD16B6AE827F17175E040871A1C7EC3500192C4C92677336EC2537ACAEE0008E0",
	    "0xF1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367",
	    "0x019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083"},
	{"OpenSSL cross-check (RFC 6979 nonces)", "ecdsa256", "secp256k1",
	    "12345678901234567890123456789", "sample", NULL,
	    "0x9483af17f79b8153ec8a69f1a2f69dd140f6a72279536e82cfeec7340aeabf05",
	    "0xfbc92668f9da81777d2262a213ecd7efd9cce7c32d55e8b571772c1f6942d53b"},
	{"OpenSSL cross-check (RFC 6979 nonces)", "ecdsa384", "P-384",
	    "12345678901234567890123456789", "sample", NULL,
	    "0xbac146d3a4030ab85fa2163c583d54d00f23090ac7447e4ea1a7280ac55960bb"
	    "b4b7f71aff6a9e4cb8c22ddcd24dbc4f",
	    "0x97503b3c3d3e846685e1173f63bc43334d31ae6a1c189cc380fee6d5542fa95c"
	    "6495a143533a078d06f052b669296609"},
	{"OpenSSL cross-check (RFC 6979 nonces)", "ecdsa512", "P-521",
	    "12345678901234567890123456789", "sample", NULL,
	    "0x2b3e3062a8a64e416cc20e026ee43a1971c9ab5c361e83b3395aed93cbb90ac7"
	    "bed37b8c3f550a4a9fe79e81248e20ea3b21e16395cc7e16172dab7df4275c610a",
	    "0x6e7039db2535fb2d8416ca9fe80833c0d8690225c32bb50b8f6cef9e9f6bc1ea"
	    "bdaf6423d7e0f55b0503b2740953ada17aa99b456bf2aa937ce3922d1bb2407c92"},
	{"RFC 5903 8.1 (ECP-256)", "ecdh", "P-256",
	    "0xC88F01F510D9AC3F70A292DAA2316DE544E9AAB8AFE84049C62A9C57862D1433",
	    "0xC6EF9C5D78AE012A011164ACB397CE2088685D8F06BF9BE0B283AB46476BEE53",
	    NULL,
	    "0xD6840F6B42F6EDAFD13116E0E12565202FEF8E9ECE7DCE03812464D04B9442DE",
	    "0x522BDE0AF0D8585B8DEF9C183B5AE38F50235206A8674ECB5D98EDB20EB153A2"},
	{"RFC 7748 5.2 vector 1", "x25519", NULL,
	    "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4",
	    "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c",
	    NULL,
	    "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552",
	    NULL},
	{"RFC 7748 5.2 vector 2", "x25519", NULL,
	    "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d",
	    "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493",
	    NULL,
	    "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957",
	    NULL},
	{"RFC 7748 5.2 iterated, 1 and 1000", "x25519iter", NULL, NULL, NULL,
	    NULL,
	    "422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079",
	    "684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51"},
	{"RFC 7748 6.1 Diffie-Hellman", "x25519dh", NULL,
	    "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
	    "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb",
	    NULL,
	    "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a",
	    "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"},
	{"RFC 7748 5.2 X448 vector 1", "x448", NULL,
	    "3d262fddf9ec8e88495266fea19a34d28882acef045104d0d1aae121700a779c"
	    "984c24f8cdd78fbff44943eba368f54b29259a4f1c600ad3",
	    "06fce640fa3487bfda5f6cf2d5263f8aad88334cbd07437f020f08f9814dc031"
	    "ddbdc38c19c6da2583fa5429db94ada18aa7a7fb4ef8a086", NULL,
	    "ce3e4ff95a60dc6697da1db1d85e6afbdf79b50a2412d7546d5f239fe14fbaad"
	    "eb445fc66a01b0779d98223961111e21766282f73dd96b6f", NULL},
	{"RFC 7748 5.2 X448 iterated once", "x448iter", NULL, NULL, NULL, NULL,
	    "3f482c8a9f19b01e6c46ee9711d9dc14fd4bf67af30765c2ae2b846a4d23a8cd"
	    "0db897086239492caf350b51f833868b9bc2b3bca9cf4113", NULL},
	{"RFC 8032 7.1 test 1", "ed25519", NULL,
	    "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
	    "", NULL,
	    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
	    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
	    "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"},
	{"RFC 8032 7.1 test 2", "ed25519", NULL,
	    "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
	    "72", NULL,
	    "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
	    "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
	    "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"},
	{"RFC 8032 7.1 test 3", "ed25519", NULL,
	    "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
	    "af82", NULL,
	    "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
	    "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
	    "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"},
	{NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}
};

static int
run_vector(const struct ecvec *v, char *got, size_t gn)
{
	unsigned char a[128], b[128], o[128], o2[128];
	size_t an, bn;
	char h[300], h2[300];

	got[0] = 0;
	if (!strncmp(v->kind, "sha", 3)) {
		int alg = atoi(v->kind + 3);
		sha_hash(alg, v->a, strlen(v->a), o);
		bytes2hex(got, o, (size_t)sha_outlen(alg));
		return !strcmp(got, v->want1);
	}
	if (!strcmp(v->kind, "x25519") || !strcmp(v->kind, "x448")) {
		int is448 = v->kind[1] == '4';
		size_t L = is448 ? 56 : 32;
		hex2bytes(v->a, a, L);
		hex2bytes(v->b, b, L);
		if (is448)
			x448(o, a, b);
		else
			x25519(o, a, b);
		bytes2hex(got, o, L);
		return !strcmp(got, v->want1);
	}
	if (!strcmp(v->kind, "x25519iter") || !strcmp(v->kind, "x448iter")) {
		int is448 = v->kind[1] == '4', i, ok = 1;
		size_t L = is448 ? 56 : 32;
		unsigned char k[56], u[56], r[56];
		memset(k, 0, sizeof k);
		k[0] = is448 ? 5 : 9;
		memcpy(u, k, sizeof u);
		for (i = 1; i <= (v->want2 ? 1000 : 1); i++) {
			if (is448)
				x448(r, k, u);
			else
				x25519(r, k, u);
			memcpy(u, k, L);
			memcpy(k, r, L);
			if (i == 1) {
				bytes2hex(h, k, L);
				ok &= !strcmp(h, v->want1);
			}
		}
		if (v->want2) {
			bytes2hex(h2, k, L);
			ok &= !strcmp(h2, v->want2);
			snprintf(got, gn, "%s .. %s", h, h2);
		} else
			snprintf(got, gn, "%s", h);
		return ok;
	}
	if (!strcmp(v->kind, "x25519dh")) {
		unsigned char pa[32], pb[32], s1[32], s2[32], base[32];
		memset(base, 0, 32);
		base[0] = 9;
		hex2bytes(v->a, a, 32);
		hex2bytes(v->b, b, 32);
		x25519(pa, a, base);
		x25519(pb, b, base);
		x25519(s1, a, pb);
		x25519(s2, b, pa);
		bytes2hex(h, pa, 32);
		bytes2hex(h2, s1, 32);
		snprintf(got, gn, "%s, %s", h, h2);
		return !strcmp(h, v->want1) && !strcmp(h2, v->want2) &&
		    !memcmp(s1, s2, 32);
	}
	if (!strcmp(v->kind, "ed25519")) {
		unsigned char pk[32], sig[64];
		hex2bytes(v->a, a, 32);
		bn = hex2bytes(v->b, b, sizeof b);
		ed25519_pub(pk, a);
		ed25519_sign(sig, a, b, bn);
		bytes2hex(h, pk, 32);
		bytes2hex(h2, sig, 64);
		snprintf(got, gn, "%s, %s", h, h2);
		return !strcmp(h, v->want1) && !strcmp(h2, v->want2) &&
		    ed25519_verify(pk, b, bn, sig);
	}
	/* Weierstrass curves */
	{
		const struct ecnamed *N = ecnamed_find(v->curve);
		static struct ecload L;
		nn d, r, s, k, w1, w2;
		ecnamed_load(N, &L);
		L.E.K = &L.F;
		nn_parse(v->a, &d, NULL);
		nn_parse(v->want1, &w1, NULL);
		nn_parse(v->want2, &w2, NULL);
		if (!strcmp(v->kind, "pub")) {
			struct ecp_pt P;
			nn x, y;
			ecp_mul(&L.E, &P, &L.G, &d);
			mf_to_nn(&L.F, &x, P.x);
			mf_to_nn(&L.F, &y, P.y);
			snprintf(got, gn, "(%s, %s)", nn_hex(&x), nn_hex(&y));
			return nn_eq(&x, &w1) && nn_eq(&y, &w2);
		}
		if (!strncmp(v->kind, "ecdsa", 5)) {
			int alg = atoi(v->kind + 5), ok;
			struct ecp_pt P;
			nn kw;
			ecdsa_sign(&L, alg, &d, (const unsigned char *)v->b,
			    strlen(v->b), NULL, &r, &s, &k);
			ecp_mul(&L.E, &P, &L.G, &d);
			snprintf(got, gn, "r = %s, s = %s", nn_hex(&r), nn_hex(&s));
			ok = nn_eq(&r, &w1) && nn_eq(&s, &w2) && ecdsa_verify(&L, alg,
			    &P, (const unsigned char *)v->b, strlen(v->b), &r, &s);
			if (v->c) {
				nn_parse(v->c, &kw, NULL);
				ok &= nn_eq(&k, &kw);
			}
			return ok;
		}
		if (!strcmp(v->kind, "ecdh")) {
			struct ecp_pt A, B, S1, S2;
			nn x, y;
			nn_parse(v->b, &r, NULL);
			ecp_mul(&L.E, &A, &L.G, &d);
			ecp_mul(&L.E, &B, &L.G, &r);
			ecp_mul(&L.E, &S1, &B, &d);
			ecp_mul(&L.E, &S2, &A, &r);
			mf_to_nn(&L.F, &x, S1.x);
			mf_to_nn(&L.F, &y, S1.y);
			snprintf(got, gn, "(%s, %s)", nn_hex(&x), nn_hex(&y));
			return nn_eq(&x, &w1) && nn_eq(&y, &w2) &&
			    ecp_eq(&L.E, &S1, &S2);
		}
	}
	(void)an;
	(void)o2;
	return 0;
}

static void
cmd_ecvectors(void)
{
	int i, pass = 0, n = 0, ok[64], nok[16], edok;
	char got[64][600], why[16][200];
	static struct ecload L;

	for (i = 0; vectors[i].src; i++) {
		ok[i] = run_vector(&vectors[i], got[i], sizeof got[i]);
		n++;
		pass += ok[i];
	}
	for (i = 0; ecnamed_tab[i].name; i++) {
		ecnamed_load(&ecnamed_tab[i], &L);
		L.E.K = &L.F;
		nok[i] = ecnamed_check(&L, why[i], sizeof why[i]);
		n++;
		pass += nok[i];
	}
	edok = ed25519_check_base();
	n++;
	pass += edok;
	printf("%d of %d test vectors and parameter checks pass\n", pass, n);
	for (i = 0; vectors[i].src; i++) {
		printf("  check: %s (%s%s%s): %s\n", vectors[i].src,
		    vectors[i].kind, vectors[i].curve ? ", " : "",
		    vectors[i].curve ? vectors[i].curve : "", ok[i] ? "pass" :
		    "FAIL");
		if (!ok[i])
			printf("    got: %s\n", got[i]);
	}
	for (i = 0; ecnamed_tab[i].name; i++)
		printf("  check: %s parameters: %s\n", ecnamed_tab[i].name,
		    nok[i] ? why[i] : "FAIL");
	printf("  check: Ed25519 base point on -x^2 + y^2 = 1 + d x^2 y^2 "
	    "and L B = 0: %s\n", edok ? "pass" : "FAIL");
}
