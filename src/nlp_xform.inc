/*
 * English for the transforms, series and sums (xform.c, series.c),
 * included by nlp.c.  xn_raw() reads a sentence as written (the
 * mathematics in it is ucas text) and returns the ucas statement, or
 * NULL when the sentence is not about these:
 *
 *   the laplace transform of t^2 e^-t        laplace t^2 e^-t
 *   inverse laplace of 1/(s^2 + 1)           ilaplace 1/(s^2 + 1)
 *   solve y'' + y = 0 with y(0) = 1, y'(0) = 0 using laplace
 *                                            laplace solve ...
 *   the z transform of a^n u[n]              ztrans a^n u[n]
 *   inverse z transform of z/(z - 0.5)       iztrans z/(z - 0.5)
 *   solve y[n+1] = 2y[n] + 1, y[0] = 1 using the z transform   zsolve ...
 *   transfer function of y[n] = 0.9 y[n-1] + x[n]   (also frequency
 *   response, poles and zeros, impulse response, is ... stable)  tf ...
 *   fourier series of a square wave [on -1 to 1]   fourier series sign(x)
 *   fourier transform of e^(-x^2), inverse fourier transform of ...
 *   the dft of [1, 2, 3, 4], the fft of ..., the dtft of ...
 *   does the sum of 1/n^2 converge, is ... convergent   converge ...
 *   the sum of 1/n^2 from 1 to infinity, the infinite sum of ...
 *                                            sum(n, 1, inf, 1/n^2)
 *   the product of 1 - 1/n^2 for n from 2 to infinity   prod(...)
 *   radius (interval) of convergence of the sum of x^n/n!   radius ...
 *   the generating function of fibonacci     genfunc fibonacci
 *   the exponential generating function of 2^n   egf 2^n
 *   the power series of e^(2x)               powerseries e^(2x)
 *   the asymptotic expansion of the sum of 1/k   asymptotic sum(...)
 */

static char xn_buf[2048];

/* s at i starts phrase ph (spaces in ph match runs of spaces; '-' as ' ') */
static size_t
xn_at(const char *s, size_t i, const char *ph)
{
	size_t j = i, k = 0;

	if (i > 0 && (isalnum((unsigned char)s[i - 1]) || s[i - 1] == '_'))
		return 0;
	while (ph[k]) {
		if (ph[k] == ' ') {
			if (s[j] != ' ' && s[j] != '-')
				return 0;
			while (s[j] == ' ' || s[j] == '-')
				j++;
			k++;
			continue;
		}
		if (tolower((unsigned char)s[j]) != ph[k])
			return 0;
		j++;
		k++;
	}
	if (isalnum((unsigned char)s[j]) || s[j] == '_')
		return 0;
	/* a word is not the start of the mathematics: "a^n" is not "a" */
	if (k > 0 && isalpha((unsigned char)ph[k - 1]) && s[j] && s[j] != ' ' &&
	    s[j] != ',' && s[j] != '.' && s[j] != '?' && s[j] != ';')
		return 0;
	return j - i;
}

/* the first occurrence of ph in s from i: its offset, else -1 */
static long
xn_find(const char *s, const char *ph, size_t from)
{
	size_t i, n = strlen(s);

	for (i = from; i < n; i++)
		if (xn_at(s, i, ph))
			return (long)i;
	return -1;
}

static void
xn_trim(char *s)
{
	size_t n = strlen(s), i = 0;

	while (n > 0 && (isspace((unsigned char)s[n - 1]) || s[n - 1] == '?' ||
	    (s[n - 1] == '.' && (n < 2 || !isdigit((unsigned char)s[n - 2])))))
		s[--n] = 0;
	while (isspace((unsigned char)s[i]))
		i++;
	if (i)
		memmove(s, s + i, n - i + 1);
}

/* drop leading words of list[] from s, repeatedly */
static void
xn_drop(char *s, const char *const *list)
{
	int k, again = 1;

	while (again) {
		again = 0;
		xn_trim(s);
		for (k = 0; list[k]; k++) {
			size_t l = xn_at(s, 0, list[k]);
			if (l) {
				memmove(s, s + l, strlen(s + l) + 1);
				again = 1;
				break;
			}
		}
	}
}

static const char *const xn_fill[] = {"what is", "what's", "whats",
	"what are", "compute", "calculate", "find", "give me", "show me",
	"tell me", "evaluate", "determine", "get", "please", "can you",
	"could you", "would you", "i want", "i need", "now", "so", "ok", "okay",
	"hey", "and", "also", "the", "me", "work out", "derive", NULL};

/* cut s at the first of the phrases: returns the offset cut, else -1 */
static long
xn_cut(char *s, const char *const *stop)
{
	long best = -1;
	int k;

	for (k = 0; stop[k]; k++) {
		long at = xn_find(s, stop[k], 1);
		if (at >= 0 && (best < 0 || at < best))
			best = at;
	}
	if (best >= 0)
		s[best] = 0;
	return best;
}

/* the mathematics: "x squared" and a few words */
static void
xn_math(char *s)
{
	static const char *const art[] = {"of", "the", "for", "a", "an", NULL};
	char tmp[1024];
	size_t i, o = 0;

	xn_drop(s, art);
	for (i = 0; s[i] && o + 8 < sizeof tmp; i++) {
		size_t l;
		if ((l = xn_at(s, i, " squared")) || (s[i] == ' ' &&
		    (l = xn_at(s, i + 1, "squared")) && ++l)) {
			o += snprintf(tmp + o, sizeof tmp - o, "^2");
			i += l - 1;
			continue;
		}
		if ((l = xn_at(s, i, "cubed")) != 0) {
			while (o > 0 && tmp[o - 1] == ' ')
				o--;
			o += snprintf(tmp + o, sizeof tmp - o, "^3");
			i += l - 1;
			continue;
		}
		if ((l = xn_at(s, i, "infinity")) != 0) {
			o += snprintf(tmp + o, sizeof tmp - o, "inf");
			i += l - 1;
			continue;
		}
		tmp[o++] = s[i];
	}
	tmp[o] = 0;
	snprintf(s, 1024, "%s", tmp);
	xn_trim(s);
}

/* "from n = 1 to infinity", "for n from 0 to inf", ", n = 1..inf": the
 * index, bounds, and the offset where they start (-1 if none) */
static long
xn_bounds(char *s, char *idx, char *lo, char *hi)
{
	static const char *const intro[] = {"from", "for", "where", "with",
		"over", NULL};
	long at = -1;
	int k;

	idx[0] = lo[0] = hi[0] = 0;
	for (k = 0; intro[k]; k++) {
		long a = xn_find(s, intro[k], 1);
		if (a > 0 && (at < 0 || a < at)) {
			/* must be followed by bounds: ... to ... or .. */
			const char *r = s + a;
			if (strstr(r, " to ") || strstr(r, "..") || strstr(r, " until "))
				at = a;
		}
	}
	{
		/* ", n = 1 to inf" */
		char *c = strchr(s, ',');
		if (c && (at < 0 || c - s < at) && (strstr(c, " to ") ||
		    strstr(c, "..")) && strchr(c, '='))
			at = c - s;
	}
	if (at < 0)
		return -1;
	{
		char b[256], *p, *t;
		snprintf(b, sizeof b, "%s", s + at);
		p = b;
		if (*p == ',')
			p++;
		else
			while (*p && *p != ' ')
				p++;   /* the intro word */
		while (*p == ' ')
			p++;
		/* "n from 1", "n = 1", "1" */
		if (isalpha((unsigned char)p[0]) && !isalnum((unsigned char)p[1]) &&
		    !xn_at(p, 0, "from")) {
			idx[0] = p[0];
			idx[1] = 0;
			p++;
			while (*p == ' ')
				p++;
			if (*p == '=')
				p++;
			else if (xn_at(p, 0, "from"))
				p += 4;
			else if (xn_at(p, 0, "going from"))
				p += 10;
			while (*p == ' ')
				p++;
		} else if (xn_at(p, 0, "from"))
			p += 4;
		t = strstr(p, "..");
		if (t) {
			*t = 0;
			snprintf(lo, 64, "%s", p);
			snprintf(hi, 64, "%s", t + 2);
		} else {
			t = strstr(p, " up to ");
			if (!t)
				t = strstr(p, " until ");
			if (!t)
				t = strstr(p, " to ");
			if (!t)
				return -1;
			*t = 0;
			snprintf(lo, 64, "%s", p);
			t++;
			while (*t && *t != ' ')
				t++;
			if (!strncmp(t, " to ", 4))
				t += 3;
			snprintf(hi, 64, "%s", t);
		}
		xn_trim(lo);
		xn_trim(hi);
		xn_math(hi);
	}
	s[at] = 0;
	xn_trim(s);
	return at;
}

static int
xn_inf(const char *h)
{
	return !strcmp(h, "inf") || !strcmp(h, "oo") || !strcmp(h, "infinity") ||
	    !strcmp(h, "\xe2\x88\x9e");
}

/* the index of an expression: the first of n k i m j in it */
static void
xn_index(const char *e, char *idx)
{
	static const char *const names[] = {"n", "k", "i", "m", "j", NULL};
	int k;

	for (k = 0; names[k]; k++)
		if (xn_find(e, names[k], 0) >= 0) {
			snprintf(idx, 8, "%s", names[k]);
			return;
		}
	snprintf(idx, 8, "n");
}

/* "the sum of X [bounds]" -> sum(...) when infinite; else X alone */
static int
xn_sum(char *body, int isprod, int need_inf, char *out, size_t on)
{
	char idx[16], lo[64], hi[64];

	if (xn_bounds(body, idx, lo, hi) >= 0) {
		xn_math(body);
		if (!idx[0])
			xn_index(body, idx);
		if (!xn_inf(hi))
			return 0;   /* a finite sum: the closed-form path */
		snprintf(out, on, "%s(%s, %s, inf, %s)", isprod ? "prod" : "sum",
		    idx, lo, body);
		return 1;
	}
	if (need_inf)
		return 0;
	xn_math(body);
	snprintf(out, on, "%s", body);
	return 2;
}

static const char *
xn_raw(const char *sent)
{
	char s[1024], body[1024];
	size_t l;
	long at;

	snprintf(s, sizeof s, "%s", sent);
	xn_trim(s);
	xn_drop(s, xn_fill);
	/* solve ... using laplace / the z transform */
	if ((l = xn_at(s, 0, "solve")) != 0 || (l = xn_at(s, 0,
	    "use laplace to solve")) != 0 || (l = xn_at(s, 0,
	    "using laplace solve")) != 0 || (l = xn_at(s, 0,
	    "use the laplace transform to solve")) != 0) {
		static const char *const lap[] = {"using the laplace transform",
			"using laplace transforms", "using laplace", "by laplace",
			"with laplace", "via laplace", "with the laplace transform",
			"by the laplace transform", "by laplace transform", NULL};
		static const char *const zt[] = {"using the z transform",
			"using z transforms", "using the z transforms", "by z transform",
			"with the z transform", "via the z transform", "using z transform",
			"by the z transform", NULL};
		int laplace = xn_at(s, 0, "solve") == 0, k;
		snprintf(body, sizeof body, "%s", s + l);
		for (k = 0; lap[k]; k++)
			if ((at = xn_find(body, lap[k], 0)) >= 0) {
				size_t m = xn_at(body, (size_t)at, lap[k]);
				memmove(body + at, body + at + m, strlen(body + at + m) + 1);
				laplace = 1;
			}
		for (k = 0; !laplace && zt[k]; k++)
			if ((at = xn_find(body, zt[k], 0)) >= 0) {
				size_t m = xn_at(body, (size_t)at, zt[k]);
				memmove(body + at, body + at + m, strlen(body + at + m) + 1);
				xn_trim(body);
				snprintf(xn_buf, sizeof xn_buf, "zsolve %s", body);
				return xn_buf;
			}
		if (!laplace)
			return NULL;
		xn_trim(body);
		{
			/* "the equation", "the ode" */
			static const char *const eqw[] = {"the ode", "the equation",
				"the differential equation", NULL};
			xn_drop(body, eqw);
		}
		snprintf(xn_buf, sizeof xn_buf, "laplace solve %s", body);
		return xn_buf;
	}
	/* does ... converge, is ... convergent */
	if ((l = xn_at(s, 0, "does")) != 0 || (l = xn_at(s, 0, "is")) != 0 ||
	    (l = xn_at(s, 0, "test")) != 0 || (l = xn_at(s, 0, "check")) != 0) {
		static const char *const verb[] = {"converge", "converges",
			"convergent", "diverge", "diverges", "divergent",
			"for convergence", "absolutely convergent",
			"conditionally convergent", NULL};
		static const char *const lead[] = {"the", "infinite", "series",
			"sum", "of", "whether", NULL};
		snprintf(body, sizeof body, "%s", s + l);
		if (xn_cut(body, verb) >= 0) {
			char out[1024];
			int r;
			xn_drop(body, lead);
			r = xn_sum(body, 0, 0, out, sizeof out);
			if (!r)
				return NULL;
			snprintf(xn_buf, sizeof xn_buf, "converge %s", out);
			return xn_buf;
		}
		/* is the filter/system ... stable */
		if (xn_find(s, "stable", 0) >= 0 && (strchr(s, '[') ||
		    strchr(s, '\''))) {
			static const char *const sys[] = {"the filter", "the system",
				"filter", "system", NULL};
			snprintf(body, sizeof body, "%s", s + l);
			xn_cut(body, (const char *const[]){"stable", NULL});
			xn_drop(body, sys);
			snprintf(xn_buf, sizeof xn_buf, "tf %s", body);
			return xn_buf;
		}
		return NULL;
	}
	{
		static const struct {
			const char *ph;
			const char *cmd;
			int kind;   /* 0 plain, 1 sum, 2 prod, 3 fseries, 4 gf */
		} K[] = {
			{"inverse laplace transform", "ilaplace", 0},
			{"inverse laplace", "ilaplace", 0},
			{"laplace transform", "laplace", 0},
			{"laplace", "laplace", 0},
			{"inverse z transform", "iztrans", 0},
			{"inverse z", "iztrans", 0},
			{"z transform", "ztrans", 0},
			{"ztransform", "ztrans", 0},
			{"inverse fourier transform", "ifourier", 0},
			{"fourier series", "fourier series", 3},
			{"fourier transform", "fourier", 0},
			{"discrete time fourier transform", "dtft", 0},
			{"dtft", "dtft", 0},
			{"inverse discrete fourier transform", "idft", 0},
			{"inverse dft", "idft", 0},
			{"inverse fft", "ifft", 0},
			{"discrete fourier transform", "dft", 0},
			{"dft", "dft", 0},
			{"fft", "fft", 0},
			{"transfer function", "tf", 0},
			{"frequency response", "tf", 0},
			{"poles and zeros", "tf", 0},
			{"impulse response", "tf", 0},
			{"radius of convergence", "radius", 5},
			{"interval of convergence", "radius", 5},
			{"exponential generating function", "egf", 4},
			{"ordinary generating function", "genfunc", 4},
			{"generating function", "genfunc", 4},
			{"power series expansion", "powerseries", 0},
			{"power series", "powerseries", 0},
			{"asymptotic expansion", "asymptotic", 6},
			{"asymptotics", "asymptotic", 6},
			{"convergence", "converge", 7},
			{"infinite sum", "sum", 1},
			{"infinite series", "sum", 1},
			{"sum", "sum", 1},
			{"series", "sum", 1},
			{"infinite product", "prod", 2},
			{"product", "prod", 2},
			{NULL, NULL, 0}
		};
		static const char *const lead[] = {"of", "for", "the", "a", "an",
			NULL};
		int k;
		for (k = 0; K[k].ph; k++)
			if ((l = xn_at(s, 0, K[k].ph)) != 0)
				break;
		if (!K[k].ph)
			return NULL;
		snprintf(body, sizeof body, "%s", s + l);
		xn_trim(body);
		if (K[k].kind != 1 && K[k].kind != 2 && body[0] && !xn_at(body, 0,
		    "of") && !xn_at(body, 0, "for") && strcmp(K[k].cmd, "laplace"))
			return NULL;   /* "laplace t" is ucas, handled elsewhere */
		if (K[k].kind == 0 && !strcmp(K[k].cmd, "laplace") && !xn_at(body, 0,
		    "of") && !xn_at(body, 0, "transform"))
			return NULL;
		switch (K[k].kind) {
		case 1:
		case 2: {
			char out[1024];
			int infinite = xn_at(K[k].ph, 0, "infinite") != 0;
			if (!xn_at(body, 0, "of"))
				return NULL;
			xn_drop(body, lead);
			if (!xn_sum(body, K[k].kind == 2, !infinite, out, sizeof out))
				return NULL;
			if (infinite && !strstr(out, "inf")) {
				char idx[8];
				xn_index(out, idx);
				snprintf(xn_buf, sizeof xn_buf, "%s(%s, 1, inf, %s)",
				    K[k].kind == 2 ? "prod" : "sum", idx, out);
			} else
				snprintf(xn_buf, sizeof xn_buf, "%s", out);
			return xn_buf;
		}
		case 3: {
			/* a square wave ... [on a to b] */
			static const struct {
				const char *w, *e;
			} waves[] = {{"a square wave", "sign(x)"},
				{"square wave", "sign(x)"},
				{"a sawtooth wave", "x"}, {"sawtooth wave", "x"},
				{"a sawtooth", "x"}, {"sawtooth", "x"},
				{"a triangle wave", "abs(x)"}, {"triangle wave", "abs(x)"},
				{"a triangular wave", "abs(x)"}, {NULL, NULL}};
			char opt[256] = "";
			long o;
			xn_drop(body, lead);
			if ((o = xn_find(body, "on", 0)) > 0 || (o = xn_find(body,
			    "over", 0)) > 0 || (o = xn_find(body, "from", 0)) > 0) {
				snprintf(opt, sizeof opt, " %s", body + o);
				body[o] = 0;
				xn_trim(body);
				if (!strncmp(opt, " over ", 6))
					memcpy(opt, " on   ", 6);
			}
			for (o = 0; waves[o].w; o++)
				if (!strcmp(body, waves[o].w)) {
					snprintf(body, sizeof body, "%s", waves[o].e);
					break;
				}
			xn_math(body);
			snprintf(xn_buf, sizeof xn_buf, "fourier series %s%s", body, opt);
			return xn_buf;
		}
		case 4: {
			static const char *const tail[] = {" sequence", " numbers",
				NULL};
			int t2;
			xn_drop(body, lead);
			for (t2 = 0; tail[t2]; t2++) {
				size_t bl = strlen(body), tl = strlen(tail[t2]);
				if (bl > tl && !strcmp(body + bl - tl, tail[t2]) &&
				    strcmp(tail[t2], " numbers"))
					body[bl - tl] = 0;
			}
			xn_math(body);
			snprintf(xn_buf, sizeof xn_buf, "%s %s", K[k].cmd, body);
			return xn_buf;
		}
		case 5: {
			char out[1024];
			static const char *const ser[] = {"of", "the", "power",
				"series", "sum", "of", NULL};
			xn_drop(body, ser);
			if (!xn_sum(body, 0, 0, out, sizeof out))
				return NULL;
			snprintf(xn_buf, sizeof xn_buf, "radius %s", out);
			return xn_buf;
		}
		case 6: {
			/* the asymptotic expansion of the sum of 1/k [for k from 1 to n] */
			char idx[16], lo[64], hi[64];
			static const char *const ser[] = {"of", "the", "sum", "partial",
				"sums", "of", NULL};
			xn_drop(body, ser);
			if (xn_bounds(body, idx, lo, hi) < 0) {
				xn_math(body);
				xn_index(body, idx);
				snprintf(lo, sizeof lo, "1");
				snprintf(hi, sizeof hi, "%s", strcmp(idx, "n") ? "n" : "N");
			} else
				xn_math(body);
			if (!idx[0])
				xn_index(body, idx);
			snprintf(xn_buf, sizeof xn_buf, "asymptotic sum(%s, %s, %s, %s)",
			    idx, lo, hi, body);
			return xn_buf;
		}
		case 7: {
			char out[1024];
			static const char *const ser[] = {"of", "the", "infinite",
				"series", "sum", "of", NULL};
			xn_drop(body, ser);
			if (!xn_sum(body, 0, 0, out, sizeof out))
				return NULL;
			snprintf(xn_buf, sizeof xn_buf, "converge %s", out);
			return xn_buf;
		}
		}
		xn_drop(body, lead);
		if (!strcmp(K[k].cmd, "laplace") && xn_at(body, 0, "transform")) {
			body[0] = ' ';
			memmove(body, body + 9, strlen(body + 9) + 1);
			xn_drop(body, lead);
		}
		if (strcmp(K[k].cmd, "tf"))
			xn_math(body);
		if (!body[0])
			return NULL;
		snprintf(xn_buf, sizeof xn_buf, "%s %s", K[k].cmd, body);
		return xn_buf;
	}
}
