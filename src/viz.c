/*
 * viz: integrals and solids of revolution, drawn and measured.
 *
 *   area F [and G] [from a to b] [riemann [N] [left|right|mid|trapezoid|
 *       simpson]] [accumulate] [frames N] [fps F] [nofile] [to NAME]
 *   revolve F [and G] [from a to b] [about x|y|y = c|x = c]
 *       [disk|washer|shell] [still] [rotate] [frames N] [nofile] [to NAME]
 *   volume ... / surfacearea ...   revolve's numbers without the image
 *   arclength F from a to b        ∫ sqrt(1 + F'^2)
 *   centroid F [and G] from a to b the region's area and centroid
 *
 * area shades the region between F and the axis (or between F and G),
 * positive and negative parts in their own colours, and prints the
 * integral from the antiderivative when integ.c finds one and by
 * Gauss-Kronrod always.  riemann draws the sums' rectangles, trapezoids
 * or parabolas (without N an animation of n = 1, 2, 4, ... 64);
 * accumulate animates A(t) = ∫_a^t F with a sweeping line over F and
 * A's graph below.
 *
 * revolve turns the region between F and the axis (about a vertical
 * line: between F and the x-axis; with G: between F and G) around the
 * axis: the boundary of the region, revolved, is the solid's surface,
 * drawn through plot.c's z-buffer with lighting, while the region itself
 * sweeps from 0 to 2pi (then the solid turns, with rotate).  Volume by
 * washers or shells, exact when the integrand has an antiderivative,
 * numeric always, cross-checked by the other method on a grid; the
 * lateral surface area 2pi ∫ r sqrt(1 + F'^2), the region's centroid and
 * Pappus' theorem.
 *
 * Drawing is plot.c's (viz.h): the same canvas, palette, fonts, axes
 * and GIF delivery, so the images look like plot's.
 */
#include "ucas.h"
#include "calc.h"
#include "gif.h"
#include "viz.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum { VC_AREA, VC_REVOLVE, VC_VOLUME, VC_SURFACE, VC_ARCLEN, VC_CENTROID };
enum { RM_NONE, RM_LEFT, RM_RIGHT, RM_MID, RM_TRAP, RM_SIMP };
enum { AX_H, AX_V };		/* about y = c, about x = c */
enum { M_DEFAULT, M_WASHER, M_SHELL };

static const char *const cmdname[] = {"area", "revolve", "volume",
	"surfacearea", "arclength", "centroid"};
static const char *const rmname[] = {"", "left", "right", "midpoint",
	"trapezoid", "simpson"};

static struct {
	int cmd;
	struct func f, g;
	int ng;			/* G given */
	char fl[160], gl[160];	/* labels, ASCII */
	int bounds;		/* a and b given */
	double a, b;
	int axis;
	double c;
	int axis_given, method;
	int rkind, rn, accum, still, rotate, nofile, frames, fps;
	char name[64];
	char text[2048];
} R;

static const char *btxt(double v);
static nid minus_c(nid e, double c);

/* ---- the statement's text as tokens ---- */

struct vtk {
	int k, a, b, d;	/* 'w' word, 'n' number, 'D' "..", else the char */
};
static struct vtk T[400];
static int nt;

static void
vt_lex(const char *s)
{
	int i = 0, d = 0;

	nt = 0;
	while (s[i] && nt < 399) {
		struct vtk *t = &T[nt];
		unsigned char c = (unsigned char)s[i];
		if (isspace(c)) {
			i++;
			continue;
		}
		t->a = i;
		if (isalpha(c) || c == '_' || c >= 0x80) {
			t->k = 'w';
			while (isalnum((unsigned char)s[i]) || s[i] == '_' ||
			    (unsigned char)s[i] >= 0x80)
				i++;
			while (s[i] == '\'')
				i++;
		} else if (isdigit(c) || (c == '.' && isdigit((unsigned char)s[i + 1]))) {
			t->k = 'n';
			while (isdigit((unsigned char)s[i]) || (s[i] == '.' &&
			    s[i + 1] != '.'))
				i++;
			if ((s[i] == 'e' || s[i] == 'E') && (isdigit((unsigned char)
			    s[i + 1]) || ((s[i + 1] == '-' || s[i + 1] == '+') &&
			    isdigit((unsigned char)s[i + 2]))))
				for (i += 2; isdigit((unsigned char)s[i]); i++)
					;
		} else if (c == '.' && s[i + 1] == '.') {
			t->k = 'D';
			i += 2;
		} else {
			t->k = c;
			i++;
		}
		t->b = i;
		if (t->k == ')' || t->k == ']' || t->k == '}')
			d--;
		t->d = d;
		if (t->k == '(' || t->k == '[' || t->k == '{')
			d++;
		nt++;
	}
}

static int
tw(int i, const char *w)
{
	size_t n = strlen(w);

	return i >= 0 && i < nt && T[i].k == 'w' && (size_t)(T[i].b - T[i].a) ==
	    n && !strncmp(R.text + T[i].a, w, n);
}

static void
tk_str(int a, int b, char *buf, size_t n)
{
	size_t len;

	if (a >= b) {
		buf[0] = 0;
		return;
	}
	len = (size_t)(T[b - 1].b - T[a].a);
	if (len >= n)
		len = n - 1;
	memcpy(buf, R.text + T[a].a, len);
	buf[len] = 0;
}

static const char *const kw[] = {
	"from", "about", "around", "riemann", "left", "right", "mid",
	"midpoint", "middle", "trapezoid", "trapezoidal", "trapezium", "simpson",
	"disk", "disc", "disks", "discs", "washer", "washers", "shell", "shells",
	"accumulate", "accumulation", "nofile", "frames", "fps", "still",
	"rotate", "rotating", "spin", "to", "till", "until", "method", NULL
};

static int
kwat(int i)
{
	int k;

	if (i >= nt || T[i].d != 0 || T[i].k != 'w')
		return 0;
	for (k = 0; kw[k]; k++)
		if (tw(i, kw[k]))
			return 1;
	return tw(i + 1, "in") && i + 2 < nt;	/* x in 0..4 */
}

static double
const_of(int a, int b, const char *what)
{
	char buf[256];
	struct func tmp;
	nid e;

	tk_str(a, b, buf, sizeof buf);
	if (!buf[0])
		fail("%s: expected a number%s%s", cmdname[R.cmd], what ? " for " :
		    "", what ? what : "");
	memset(&tmp, 0, sizeof tmp);
	tmp.name = sym_cstr("_");
	e = calc_parse_text(buf, &tmp, TY_REAL);
	if (tmp.np || !is_const(e))
		fail("%s: expected a number, found '%s'", cmdname[R.cmd], buf);
	return TY_FP(N.ty[e]) ? N.kr[e] : (double)N.ki[e];
}

/* the end of a value from i: a keyword, "..", ',' or "and" at depth 0 */
static int
val_end(int i)
{
	int j = i;

	while (j < nt && !(T[j].d == 0 && (T[j].k == 'D' || T[j].k == ',' ||
	    (j > i && kwat(j)))) && T[j].d >= 0)
		j++;
	return j;
}

static void
range_from(int *i)
{
	int j = *i, e;

	if (j < nt && T[j].k == '[') {
		for (e = j + 1; e < nt && !(T[e].d == 1 && T[e].k == ','); e++)
			;
		R.a = const_of(j + 1, e, "a");
		for (j = e + 1; e < nt && !(T[e].d == 0 && T[e].k == ']'); e++)
			;
		R.b = const_of(j, e, "b");
		*i = e + 1;
	} else {
		e = val_end(j);
		R.a = const_of(j, e, "a");
		j = e;
		if (j < nt && (T[j].k == 'D' || tw(j, "to") || tw(j, "till") ||
		    tw(j, "until") || T[j].k == ','))
			j++;
		else
			fail("%s: from a to b", cmdname[R.cmd]);
		e = val_end(j);
		R.b = const_of(j, e, "b");
		*i = e;
	}
	R.bounds = 1;
	if (!(R.a == R.a) || !(R.b == R.b) || !isfinite(R.a) || !isfinite(R.b))
		fail("%s: finite bounds, please", cmdname[R.cmd]);
}

/* about x | y | the x-axis | y = c | x = c | the line y = 2 */
static int
axis_at(int i)
{
	char s[128], t[128];
	int e = i, k, m = 0;

	while (e < nt && !(T[e].d == 0 && kwat(e)))
		e++;
	tk_str(i, e, s, sizeof s);
	for (k = 0; s[k] && m + 1 < (int)sizeof t; k++)
		if (!isspace((unsigned char)s[k]))
			t[m++] = (char)tolower((unsigned char)s[k]);
	t[m] = 0;
	if (!strncmp(t, "the", 3))
		memmove(t, t + 3, strlen(t + 3) + 1);
	if (!strncmp(t, "line", 4))
		memmove(t, t + 4, strlen(t + 4) + 1);
	R.axis_given = 1;
	R.c = 0;
	if (!strcmp(t, "x") || !strcmp(t, "x-axis") || !strcmp(t, "xaxis") ||
	    !strcmp(t, "y=0"))
		R.axis = AX_H;
	else if (!strcmp(t, "y") || !strcmp(t, "y-axis") || !strcmp(t, "yaxis") ||
	    !strcmp(t, "x=0"))
		R.axis = AX_V;
	else if ((t[0] == 'x' || t[0] == 'y') && t[1] == '=') {
		int q;
		R.axis = t[0] == 'y' ? AX_H : AX_V;
		for (q = i; q < e && T[q].k != '='; q++)
			;
		R.c = const_of(q + 1, e, "the axis");
	} else
		fail("%s: about which axis? (about x, about y, about y = 2, about "
		    "x = -1; found '%s')", cmdname[R.cmd], s);
	return e;
}

static void
piece(int a, int b, int which)
{
	char buf[512];
	struct func *f;

	tk_str(a, b, buf, sizeof buf);
	if (!buf[0])
		fail("%s: expected a function", cmdname[R.cmd]);
	f = cmd_func_text(buf);
	if (which == 0) {
		R.f = *f;
		pr_ascii(R.fl, sizeof R.fl, f->anon ? node_str(f->body) : buf);
	} else {
		R.g = *f;
		R.ng = 1;
		pr_ascii(R.gl, sizeof R.gl, f->anon ? node_str(f->body) : buf);
	}
}

static void
fix_var(struct func *f, const char *label)
{
	if (f->np == 0) {
		f->p[0].name = sym_cstr("x");
		f->p[0].ty = TY_REAL;
		f->np = 1;
	}
	if (f->np > 1)
		fail("%s: %s has %d variables; one, please (fix the others first, "
		    "e.g. g(x) = ... with numbers)", cmdname[R.cmd], label, f->np);
}

static void
parse_req(void)
{
	int i = 0, s, np = 0, k;

	vt_lex(R.text);
	if (tw(0, "under") || tw(0, "of") || tw(0, "between") || tw(0, "for"))
		i++;
	/* the functions, up to the first option */
	for (s = i, k = i; ; k++)
		if (k >= nt || (T[k].d == 0 && (kwat(k) || T[k].k == ',' ||
		    tw(k, "and")))) {
			if (k > s) {
				if (np == 2)
					fail("%s: one or two functions", cmdname[R.cmd]);
				piece(s, k, np++);
			}
			if (k >= nt || kwat(k))
				break;
			s = k + 1;
		}
	if (np == 0)
		fail("%s what? e.g. %s x^2 from 0 to 4", cmdname[R.cmd],
		    cmdname[R.cmd]);
	i = k;
	while (i < nt) {
		char w[32];
		if (T[i].k == ',' || tw(i, "and") || tw(i, "with") || tw(i, "by") ||
		    tw(i, "using") || tw(i, "the") || tw(i, "method")) {
			i++;
			continue;
		}
		if (T[i].k != 'w') {
			tk_str(i, nt, w, sizeof w);
			fail("%s: I don't understand '%s'", cmdname[R.cmd], R.text +
			    T[i].a);
		}
		tk_str(i, i + 1, w, sizeof w);
		if (!strcmp(w, "from")) {
			i++;
			range_from(&i);
		} else if (tw(i + 1, "in")) {
			i += 2;
			range_from(&i);
		} else if (!strcmp(w, "about") || !strcmp(w, "around")) {
			i = axis_at(i + 1);
		} else if (!strcmp(w, "riemann")) {
			i++;
			if (R.rkind == RM_NONE)
				R.rkind = RM_MID;
			if (i < nt && T[i].k == 'n') {
				R.rn = (int)const_of(i, i + 1, "n");
				i++;
			}
		} else if (!strcmp(w, "left") || !strcmp(w, "right") ||
		    !strcmp(w, "mid") || !strcmp(w, "midpoint") ||
		    !strcmp(w, "middle") || !strcmp(w, "trapezoid") ||
		    !strcmp(w, "trapezoidal") || !strcmp(w, "trapezium") ||
		    !strcmp(w, "simpson")) {
			R.rkind = w[0] == 'l' ? RM_LEFT : w[0] == 'r' ? RM_RIGHT :
			    w[0] == 'm' ? RM_MID : w[0] == 't' ? RM_TRAP : RM_SIMP;
			i++;
		} else if (!strncmp(w, "dis", 3) || !strncmp(w, "washer", 6)) {
			R.method = M_WASHER;
			i++;
		} else if (!strncmp(w, "shell", 5)) {
			R.method = M_SHELL;
			i++;
		} else if (!strncmp(w, "accumulat", 9)) {
			R.accum = 1;
			i++;
		} else if (!strcmp(w, "nofile")) {
			R.nofile = 1;
			i++;
		} else if (!strcmp(w, "still")) {
			R.still = 1;
			i++;
		} else if (!strcmp(w, "rotate") || !strcmp(w, "rotating") ||
		    !strcmp(w, "spin")) {
			R.rotate = 1;
			i++;
		} else if (!strcmp(w, "frames") || !strcmp(w, "fps")) {
			int e = val_end(i + 1);
			double v = const_of(i + 1, e, w);
			if (!(v >= 1 && v <= 200))
				fail("%s: %s between 1 and 200", cmdname[R.cmd], w);
			if (w[0] == 'f' && w[1] == 'r')
				R.frames = (int)v;
			else
				R.fps = (int)v;
			i = e;
		} else if (!strcmp(w, "to") && i + 1 < nt && T[i + 1].k == 'w') {
			tk_str(i + 1, i + 2, R.name, sizeof R.name);
			i += 2;
		} else
			fail("%s: I don't understand '%s'", cmdname[R.cmd], R.text +
			    T[i].a);
	}
	fix_var(&R.f, R.fl);
	if (R.ng)
		fix_var(&R.g, R.gl);
	if (R.bounds && R.a > R.b) {
		double t = R.a;
		R.a = R.b;
		R.b = t;
		printf("  note: the bounds in order: from %s to %s\n", btxt(R.a),
		    btxt(R.b));
	}
	if (R.bounds && !(R.b > R.a))
		fail("%s: an empty interval [%s, %s]", cmdname[R.cmd], btxt(R.a),
		    btxt(R.b));
	if (R.rn < 0 || R.rn > 1000)
		fail("%s: riemann n between 1 and 1000", cmdname[R.cmd]);
}

/* ---- numbers ---- */

static const char *xname(void) { return sym_str(R.f.p[0].name); }

static nid
xvar(void)
{
	nid x = mk_var(0, R.f.p[0].name, R.f.p[0].ty);

	return TY_FP(N.ty[x]) ? x : mk_cast(TY_REAL, x);
}

static nid
body_of(const struct func *f)
{
	return mk_cast(TY_REAL, f->body);
}

static void
eval_n(const struct func *f, int n, const double *xs, double *ys,
    uint8_t *ok)
{
	pr_eval1(f, f->body, n, xs, ys, ok);
}

static int
eval_at(const struct func *f, double x, double *y)
{
	uint8_t ok;

	eval_n(f, 1, &x, y, &ok);
	return ok == 1;
}

/* a node in slot 0 at many points, through a temporary function */
static void
eval_node(nid e, int n, const double *xs, double *ys, uint8_t *ok)
{
	struct func t = R.f;

	t.where = 0;
	pr_eval1(&t, e, n, xs, ys, ok);
}

/*
 * v as a short exact form when it is one: a rational p/q (q <= 1000),
 * times pi or pi^2; "" if none.  Used only on values that come from an
 * antiderivative, so the form is a recognition of an exact result.
 */
static const char *
recog(double v)
{
	static char buf[4][64];
	static int r;
	static const int rt[] = {1, 2, 3, 5};
	char *b = buf[r++ & 3];
	int k, m;

	b[0] = 0;
	if (!isfinite(v))
		return b;
	if (v == 0) {
		snprintf(b, 64, "0");
		return b;
	}
	/* p/q, p/q pi, p/q pi^2, then the same times sqrt(2), sqrt(3), sqrt(5) */
	for (m = 0; m < 4; m++)
		for (k = 0; k < 3; k++) {
			double u = v / pow(M_PI, k) / sqrt(rt[m]);
			int64_t n = 0, d, dmax = m ? 100 : 1000;
			char num[48], f[16];
			if (fabs(u) > 1e9)
				continue;
			for (d = 1; d <= dmax; d++) {
				n = (int64_t)llround(u * d);
				if (fabs((double)n / d - u) <= 1e-10 * fmax(1, fabs(u)))
					break;
			}
			if (d > dmax)
				continue;
			snprintf(f, sizeof f, "%s%s%s", k == 1 ? "pi" : k == 2 ? "pi^2" :
			    "", k && m ? "*" : "", m ? (m == 1 ? "sqrt(2)" : m == 2 ?
			    "sqrt(3)" : "sqrt(5)") : "");
			if (!f[0])
				snprintf(num, sizeof num, "%lld", (long long)n);
			else if (n == 1 || n == -1)
				snprintf(num, sizeof num, "%s%s", n < 0 ? "-" : "", f);
			else
				snprintf(num, sizeof num, "%lld%s%s", (long long)n, k ? "" : "*",
				    f);
			if (d == 1)
				snprintf(b, 64, "%s", num);
			else
				snprintf(b, 64, "%s/%lld", num, (long long)d);
			return b;
		}
	return b;
}

/* an exact value in full: "21.3333333333333 = 64/3", "0" */
static const char *
full(double v)
{
	static char buf[4][96];
	static int r;
	char *b = buf[r++ & 3], num[40];
	const char *q = recog(v);

	if (v == 0)
		v = 0;
	snprintf(num, sizeof num, "%.15g", v);
	if (q[0] && strcmp(q, num))
		snprintf(b, 96, "%s = %s", num, q);
	else
		snprintf(b, 96, "%s", num);
	return b;
}

/* a bound as text: 2pi, pi/2, else %.4g */
static const char *
btxt(double v)
{
	static char buf[8][32];
	static int r;
	char *b = buf[r++ & 7];
	double u = v / M_PI;
	int d;

	for (d = 1; v != 0 && d <= 12; d++) {
		double nn = u * d;
		long n;
		if (fabs(nn - floor(nn + 0.5)) > 1e-9 || fabs(nn) > 1000)
			continue;
		n = (long)floor(nn + 0.5);
		if (n == 1 || n == -1)
			snprintf(b, 32, "%spi", n < 0 ? "-" : "");
		else
			snprintf(b, 32, "%ldpi", n);
		if (d > 1)
			snprintf(b + strlen(b), 32 - strlen(b), "/%d", d);
		return b;
	}
	snprintf(b, 32, "%s", pr_g4(v));
	return b;
}

/* "64/3 ≈ 21.33" or "21.33" */
static const char *
val_text(double v, int exact)
{
	static char buf[4][96];
	static int r;
	char *b = buf[r++ & 3];
	const char *q = exact ? recog(v) : "";

	if (q[0] && strcmp(q, pr_g4(v)))
		snprintf(b, 96, "%s = %s", q, pr_g4(v));
	else
		snprintf(b, 96, "%s", pr_g4(v));
	return b;
}

/* the integrand e (slot 0) defined at 400 points inside [a, b] */
static double last_max;		/* defined_on: the largest |e| seen */

static int
defined_on(nid e, double a, double b)
{
	double xs[401], ys[401];
	uint8_t ok[401];
	int i;

	last_max = 0;
	for (i = 0; i <= 400; i++)
		xs[i] = a + (b - a) * i / 400.0;
	eval_node(e, 401, xs, ys, ok);
	for (i = 0; i <= 400; i++) {
		if (ok[i] != 1)
			return 0;
		last_max = fmax(last_max, fabs(ys[i]));
	}
	return 1;
}

struct integ {
	double v, err;		/* numeric */
	int good, nev;
	int trim;		/* singular at an end: the ends trimmed */
	int improper;		/* F's limit at an end */
	int diverge;		/* F grows without bound at a (1) or b (2) */
	double tdiff;
	int exact;		/* from the antiderivative */
	double ve;
	nid F;
};

/* ∫ e from a to b, both ways; F is tried once per integrand */
static void
integrate(nid e, double a, double b, struct integ *I, int want_exact)
{
	memset(I, 0, sizeof *I);
	I->good = gk_integrate(&R.f, e, 0, a, b, &I->v, &I->err, &I->nev);
	if (!I->good) {
		/*
		 * sqrt(1 - x^2)' at 1: an end that is singular but integrable
		 * gives the same value with the ends trimmed by 1e-6 and 1e-9 of
		 * the interval; a divergent one or one inside does not
		 */
		double v6, v9, e6, e9, d6 = 1e-8 * (b - a), d9 = 1e-12 * (b - a);
		int n6, n9, g6 = gk_integrate(&R.f, e, 0, a + d6, b - d6, &v6, &e6,
		    &n6), g9 = gk_integrate(&R.f, e, 0, a + d9, b - d9, &v9, &e9,
		    &n9);
		if (g6 && g9 && fabs(v9 - v6) <= 1e-2 * fabs(v9) + 1e-12) {
			I->good = 1;
			I->trim = 1;
			I->v = v9;
			I->tdiff = fabs(v9 - v6);
			I->err = e9 + I->tdiff;
			I->nev = n6 + n9;
		}
	}
	if (!want_exact)
		return;
	I->F = antideriv(e, 0, NULL);
	notes_drop();
	if (I->F && (defined_on(e, a, b) || defined_on(e, a + 1e-9 * (b - a),
	    b - 1e-9 * (b - a)))) {
		double fe[2], x[2] = {a, b};
		int k, ok = 1;
		/* an end where F is undefined: its limit, or the integral diverges */
		for (k = 0; k < 2; k++) {
			int o, o6, o9;
			double s = k ? -1 : 1, f6, f9;
			fe[k] = eval_const_at(I->F, 0, x[k], &o);
			if (o)
				continue;
			f6 = eval_const_at(I->F, 0, x[k] + s * 1e-6 * (b - a), &o6);
			f9 = eval_const_at(I->F, 0, x[k] + s * 1e-12 * (b - a), &o9);
			if (o6 && o9 && fabs(f9 - f6) <= 1e-3 * (1 + fabs(f6))) {
				fe[k] = f9;
				I->improper = 1;
				continue;
			}
			ok = 0;
			if (o6 && o9 && fabs(f9) > fabs(f6)) {
				I->diverge = k + 1;
				I->good = 0;
			}
		}
		if (ok) {
			I->exact = 1;
			I->ve = fe[1] - fe[0];
			/* sin(2pi) is 2.4e-16: rounding, not a value */
			if (fabs(I->ve) < 1e-13 * fmax(fabs(fe[0]) + fabs(fe[1]), last_max *
			    fabs(b - a)))
				I->ve = 0;
		}
	}
}

static double
ival(const struct integ *I)
{
	return I->exact ? I->ve : I->v;
}

/* the two results and whether they agree */
static void
print_integ(const char *indent, const struct integ *I, double a, double b,
    double scale)
{
	if (I->F)
		printf("%santiderivative: %s\n", indent, node_str(I->F));
	if (I->exact)
		printf("%sexact: F(%s) - F(%s) = %s%s\n", indent, btxt(b), btxt(a),
		    full(I->ve), I->improper ? " (an improper integral: F's limit "
		    "at the end where it is undefined)" : "");
	else if (I->diverge)
		printf("%sdiverges: F = %s grows without bound towards %s = %s\n",
		    indent, node_str(I->F), xname(), btxt(I->diverge == 1 ? a : b));
	else if (I->F && !I->diverge)
		printf("%s(the antiderivative is not defined across the interval: "
		    "numeric only)\n", indent);
	else
		printf("%sno antiderivative found: numeric only\n", indent);
	if (I->good) {
		if (I->trim)
			printf("%s(singular at an end but integrable: the ends trimmed by "
			    "1e-8 and 1e-12 of the interval give values %.2g apart)\n",
			    indent, I->tdiff);
		printf("%snumeric (adaptive Gauss-Kronrod 7-15, %d evaluations): "
		    "%.15g, estimated error %.2g\n", indent, I->nev, I->v * scale,
		    I->err * fabs(scale));
		if (I->exact)
			printf("%sthe two agree to %.2g\n", indent, fabs(I->v - I->ve) *
			    fabs(scale));
	} else if (!I->diverge)
		printf("%snumeric: the integrand is undefined or infinite inside "
		    "the interval (singular or divergent)\n", indent);
}

/* ---- zeros and default bounds ---- */

#define NS 1201

/* sign changes of e on [a, b], refined by bisection; returns the count */
static int
zeros_of(nid e, double a, double b, double *z, int max)
{
	double xs[NS], ys[NS];
	uint8_t ok[NS];
	int i, n = 0;

	for (i = 0; i < NS; i++)
		xs[i] = a + (b - a) * i / (NS - 1);
	eval_node(e, NS, xs, ys, ok);
	for (i = 0; i + 1 < NS && n < max; i++) {
		double l = xs[i], r = xs[i + 1], fl = ys[i], m, fm;
		int k;
		uint8_t o;
		if (ok[i] != 1 || ok[i + 1] != 1)
			continue;
		if (fl == 0) {	/* on a sample: a crossing, not a touch */
			if (i > 0 && ok[i - 1] == 1 && ((ys[i - 1] < 0 && ys[i + 1] > 0) ||
			    (ys[i - 1] > 0 && ys[i + 1] < 0)) && (n == 0 ||
			    fabs(z[n - 1] - l) > 1e-12 * (1 + fabs(l))))
				z[n++] = l;
			continue;
		}
		if (!((fl < 0 && ys[i + 1] > 0) || (fl > 0 && ys[i + 1] < 0)))
			continue;
		for (k = 0; k < 80; k++) {
			m = (l + r) / 2;
			eval_node(e, 1, &m, &fm, &o);
			if (o != 1)
				break;
			if ((fm < 0) == (fl < 0)) {
				l = m;
				fl = fm;
			} else
				r = m;
		}
		m = (l + r) / 2;
		if (fabs(m - floor(m + 0.5)) < 1e-12)	/* 1, not 0.99999999 */
			m = floor(m + 0.5);
		z[n++] = m;
	}
	return n;
}

static nid
diff_top(void)
{
	return R.ng ? SUB(body_of(&R.f), body_of(&R.g)) : body_of(&R.f);
}

/* the longest stretch of [lo, hi] where f (and g) are defined */
static void
clip_defined(double *lo, double *hi)
{
	double xs[NS], y1[NS], y2[NS];
	uint8_t o1[NS], o2[NS];
	int i, best = 0, bs = 0, run = 0, rs = 0;

	for (i = 0; i < NS; i++)
		xs[i] = *lo + (*hi - *lo) * i / (NS - 1);
	eval_n(&R.f, NS, xs, y1, o1);
	if (R.ng)
		eval_n(&R.g, NS, xs, y2, o2);
	for (i = 0; i < NS; i++) {
		if (o1[i] == 1 && (!R.ng || o2[i] == 1)) {
			if (!run)
				rs = i;
			run++;
			if (run > best) {
				best = run;
				bs = rs;
			}
		} else
			run = 0;
	}
	if (best < 2)
		fail("%s: %s is not defined on [%s, %s]", cmdname[R.cmd], R.fl,
		    pr_g4(*lo), pr_g4(*hi));
	if (best < NS) {
		*lo = xs[bs];
		*hi = xs[bs + best - 1];
	}
}

/* from a to b, or a default with a note: the crossings, else plot's */
static void
default_bounds(int crossings)
{
	char why[300], why2[300];
	double lo, hi, l2, h2, z[64];
	int nz;

	if (R.bounds) {
		printf("  %s in [%s, %s]: given\n", xname(), pr_gpi(R.a),
		    pr_gpi(R.b));
		return;
	}
	pr_auto_range(&R.f, &lo, &hi, why, sizeof why);
	if (R.ng && !is_const(R.g.body)) {
		pr_auto_range(&R.g, &l2, &h2, why2, sizeof why2);
		lo = fmin(lo, l2);
		hi = fmax(hi, h2);
	}
	if (crossings && R.ng) {
		nz = zeros_of(diff_top(), lo, hi, z, 64);
		if (nz >= 2) {
			R.a = z[0];
			R.b = z[nz - 1];
			R.bounds = 1;
			printf("  %s in [%s, %s]: between the outermost crossings of %s "
			    "and %s\n", xname(), pr_gpi(R.a), pr_gpi(R.b), R.fl, R.gl);
			printf("  note: no bounds given: the region the curves enclose "
			    "(say from a to b for another)\n");
			return;
		}
	}
	clip_defined(&lo, &hi);
	if (!R.ng && R.cmd != VC_ARCLEN && !(R.cmd != VC_AREA && R.cmd !=
	    VC_CENTROID && R.axis == AX_V)) {
		/* one lobe: between two crossings of the axis, the one nearest 0 */
		double c = R.cmd == VC_AREA || R.cmd == VC_CENTROID ? 0 : R.c;
		int k, best = -1;
		nz = zeros_of(minus_c(body_of(&R.f), c), lo, hi, z, 64);
		for (k = 0; k + 1 < nz; k++) {
			double m = (z[k] + z[k + 1]) / 2;
			if (best < 0 || fabs(m) - 1e-9 * m < fabs((z[best] +
			    z[best + 1]) / 2) - 1e-9 * (z[best] + z[best + 1]) / 2)
				best = k;
		}
		if (best >= 0) {
			R.a = z[best];
			R.b = z[best + 1];
			R.bounds = 1;
			printf("  %s in [%s, %s]: between two crossings of %s, the lobe "
			    "nearest 0 (in plot's bounds [%s, %s])\n", xname(),
			    pr_gpi(R.a), pr_gpi(R.b), c == 0 ? "the x-axis" : "the axis",
			    btxt(lo), btxt(hi));
			printf("  note: no bounds given: one lobe of the region (say "
			    "from a to b for another)\n");
			return;
		}
	}
	if (R.cmd != VC_AREA && R.cmd != VC_CENTROID && R.cmd != VC_ARCLEN &&
	    R.axis == AX_V && lo < R.c && hi > R.c) {
		/* about a vertical axis: one side of it, the right one */
		printf("  note: plot's bounds [%s, %s] straddle %s: the side %s >= "
		    "%s (say from a to b for the other)\n", btxt(lo), btxt(hi),
		    R.c == 0 ? "the y-axis" : "the axis", xname(), pr_g4(R.c));
		lo = R.c;
	}
	R.a = lo;
	R.b = hi;
	R.bounds = 1;
	printf("  %s in [%s, %s]: plot's bounds (%s)%s\n", xname(), pr_gpi(lo),
	    pr_gpi(hi), why, R.ng ? ", for both" : "");
	printf("  note: no bounds given: plot's (say from a to b for others)\n");
}

/* ---- GIF ---- */

static int
delay_cs(int k, int fps)
{
	return (int)floor(100.0 * (k + 1) / fps + 0.5) -
	    (int)floor(100.0 * k / fps + 0.5);
}

static void
gif_finish(struct gif *g, int W, int H, int nfr, double secs)
{
	gif_end(g);
	if (nfr <= 1)
		printf("  image: %dx%d, %zu bytes\n", W, H, g->out.n);
	else
		printf("  animation: %dx%d, %d frames (%.1f s), %zu bytes\n", W, H,
		    nfr, secs, g->out.n);
	gif_deliver(g, R.name, R.nofile);
	free(g->out.p);
	free(g->prev);
}

/* ---- 2-D: the chart ---- */

static double
PX(const struct pr_ax *A, double v)
{
	return A->L + (v - A->lo[0]) / (A->hi[0] - A->lo[0]) * (A->R - A->L);
}

static double
PY(const struct pr_ax *A, double v)
{
	return A->B - (v - A->lo[1]) / (A->hi[1] - A->lo[1]) * (A->B - A->T);
}

static double
UX(const struct pr_ax *A, double px)
{
	return A->lo[0] + (px - A->L) / (A->R - A->L) * (A->hi[0] - A->lo[0]);
}

/* a curve over the chart's x range, anti-aliased in series col */
static void
curve(const struct pr_ax *A, const struct func *f, int col, double x0,
    double x1, double w)
{
	double xs[NS], ys[NS];
	uint8_t ok[NS];
	int i;

	for (i = 0; i < NS; i++)
		xs[i] = x0 + (x1 - x0) * i / (NS - 1);
	eval_n(f, NS, xs, ys, ok);
	for (i = 0; i + 1 < NS; i++)
		if (ok[i] == 1 && ok[i + 1] == 1)
			pr_seg(PX(A, xs[i]), PY(A, ys[i]), PX(A, xs[i + 1]),
			    PY(A, ys[i + 1]), w);
	pr_ink_ser(col);
}

static void
dashed_v(double x, double y0, double y1, int c)
{
	int y, a = (int)floor(fmin(y0, y1)), b = (int)ceil(fmax(y0, y1));

	for (y = a; y <= b; y++)
		if (((y - a) / 4) % 2 == 0)
			pr_put((int)floor(x + 0.5), y, c);
}

/* the y range of f (and g) over x0..x1, with 0 in it */
static void
y_range(double x0, double x1, double *lo, double *hi, const double *extra,
    int nextra)
{
	double *v = xmalloc((2 * NS + nextra + 1) * sizeof *v), xs[NS], ys[NS];
	double vmin, vmax, m;
	uint8_t ok[NS];
	int i, n = 0;

	for (i = 0; i < NS; i++)
		xs[i] = x0 + (x1 - x0) * i / (NS - 1);
	eval_n(&R.f, NS, xs, ys, ok);
	for (i = 0; i < NS; i++)
		if (ok[i] == 1)
			v[n++] = ys[i];
	if (R.ng) {
		eval_n(&R.g, NS, xs, ys, ok);
		for (i = 0; i < NS; i++)
			if (ok[i] == 1)
				v[n++] = ys[i];
	}
	for (i = 0; i < nextra; i++)
		v[n++] = extra[i];
	v[n++] = 0;
	pr_vrange(v, n, lo, hi, &vmin, &vmax);
	free(v);
	m = (*hi - *lo) * 0.04;
	if (*lo > -m && *lo <= 0)
		*lo = -m;	/* the axis a little above the frame */
}

/* the Riemann approximant at x for n pieces on [a, b] */
static double
r_piece(double x, int n, int kind, const double *fv, double *xl, double *xr)
{
	double h = (R.b - R.a) / n;
	int k = (int)floor((x - R.a) / h);

	if (k < 0)
		k = 0;
	if (k >= n)
		k = n - 1;
	*xl = R.a + k * h;
	*xr = *xl + h;
	switch (kind) {
	case RM_LEFT:
		return fv[k];
	case RM_RIGHT:
		return fv[k + 1];
	case RM_MID:
		return fv[n + 1 + k];
	case RM_TRAP:
		return fv[k] + (fv[k + 1] - fv[k]) * (x - *xl) / h;
	default: {	/* simpson: the parabola through pairs */
		int p = k & ~1;
		double x0 = R.a + p * h, t = (x - x0) / h;
		double y0 = fv[p], y1 = fv[p + 1], y2 = fv[p + 2];
		*xl = x0;
		*xr = x0 + 2 * h;
		return y0 * (t - 1) * (t - 2) / 2 - y1 * t * (t - 2) +
		    y2 * t * (t - 1) / 2;
	}
	}
}

/* f at the n + 1 nodes, then the n midpoints; 0 if any is undefined */
static int
r_values(int n, double *fv)
{
	double *xs = xmalloc((2 * n + 1) * sizeof *xs), h = (R.b - R.a) / n;
	uint8_t *ok = xmalloc(2 * n + 1);
	int i, good = 1;

	for (i = 0; i <= n; i++)
		xs[i] = R.a + i * h;
	for (i = 0; i < n; i++)
		xs[n + 1 + i] = R.a + (i + 0.5) * h;
	eval_n(&R.f, 2 * n + 1, xs, fv, ok);
	for (i = 0; i < 2 * n + 1; i++)
		if (ok[i] != 1 && !(i <= n && (R.rkind == RM_MID ||
		    (R.rkind == RM_LEFT && i == n) || (R.rkind == RM_RIGHT && i == 0))))
			good = 0;
	free(xs);
	free(ok);
	return good;
}

static double
r_sum(int n, int kind, const double *fv)
{
	double h = (R.b - R.a) / n, s = 0;
	int i;

	for (i = 0; i < n; i++)
		switch (kind) {
		case RM_LEFT:
			s += fv[i];
			break;
		case RM_RIGHT:
			s += fv[i + 1];
			break;
		case RM_MID:
			s += fv[n + 1 + i];
			break;
		case RM_TRAP:
			s += (fv[i] + fv[i + 1]) / 2;
			break;
		default:
			break;
		}
	if (kind == RM_SIMP) {
		for (i = 0; i + 2 <= n; i += 2)
			s += (fv[i] + 4 * fv[i + 1] + fv[i + 2]) / 3;
	}
	{	/* -1.7e-16 for a sum of +1 and -1: rounding, 0 */
		double m = 0;
		for (i = 0; i < 2 * n + 1; i++)
			if (isfinite(fv[i]))
				m = fmax(m, fabs(fv[i]));
		if (fabs(s) < 1e-13 * m * n)
			s = 0;
	}
	return s * h;
}

struct achart {
	double tcut;		/* shade up to here (accumulate), else b */
	int rn;			/* Riemann pieces drawn, 0 none */
	const double *fv;
	double rsum;
	double pos, neg;	/* for the legend */
	int sweep;
	double run;		/* the running area (accumulate) */
};

static void
area_chart(struct pr_ax *A, int x0, int y0, int w, int h, const char *title,
    const struct achart *ac)
{
	char lab[6][80];
	int col[6], sty[6], nl = 0, px, pa, pb, i;
	double y0px = PY(A, 0), cut = ac->tcut;
	int pos_c = pr_ser(0, ac->rn ? 3 : 2), neg_c = pr_ser(3, ac->rn ? 3 : 2);

	if (R.ng) {
		pos_c = pr_ser(0, 2);
		neg_c = pr_ser(1, 2);
	}
	pr_layout(A, x0, y0, w, h, 0);
	y0px = PY(A, 0);
	pr_clip(A->L + 1, A->T + 1, A->R - 1, A->B - 1);
	pr_axes(A, xname(), "y", 0);
	/* the region, column by column */
	pa = (int)ceil(PX(A, R.a) - 0.5);
	pb = (int)floor(PX(A, cut) - 0.5);
	if (pb >= pa) {
		int n = pb - pa + 1;
		double *xs = xmalloc(n * sizeof *xs), *fy = xmalloc(n * sizeof *fy),
		    *gy = xmalloc(n * sizeof *gy);
		uint8_t *o1 = xmalloc(n), *o2 = xmalloc(n);
		for (i = 0; i < n; i++) {
			xs[i] = UX(A, pa + i + 0.5);
			if (xs[i] < R.a)
				xs[i] = R.a;
			if (xs[i] > cut)
				xs[i] = cut;
		}
		eval_n(&R.f, n, xs, fy, o1);
		if (R.ng)
			eval_n(&R.g, n, xs, gy, o2);
		for (i = 0; i < n; i++) {
			double top, bot;
			int ya, yb, y;
			if (o1[i] != 1 || (R.ng && o2[i] != 1))
				continue;
			top = PY(A, fy[i]);
			bot = R.ng ? PY(A, gy[i]) : y0px;
			ya = (int)floor(fmin(top, bot) + 0.5);
			yb = (int)floor(fmax(top, bot) - 0.5);
			for (y = ya; y <= yb; y++)
				if (y > A->T && y < A->B)
					pr_put(pa + i, y, top < bot ? pos_c : neg_c);
		}
		free(xs);
		free(fy);
		free(gy);
		free(o1);
		free(o2);
	}
	/* Riemann pieces: filled, outlined */
	if (ac->rn) {
		int n = ac->rn;
		double xl, xr, hv, prev = NAN;
		pa = (int)ceil(PX(A, R.a) - 0.5);
		pb = (int)floor(PX(A, R.b) - 0.5);
		for (px = pa; px <= pb; px++) {
			double x = UX(A, px + 0.5);
			int ya, yb, y;
			if (x < R.a || x > R.b)
				continue;
			hv = r_piece(x, n, R.rkind, ac->fv, &xl, &xr);
			if (!isfinite(hv))
				continue;
			ya = (int)floor(fmin(PY(A, hv), y0px) + 0.5);
			yb = (int)floor(fmax(PY(A, hv), y0px) - 0.5);
			for (y = ya; y <= yb; y++)
				if (y > A->T && y < A->B)
					pr_put(px, y, hv >= 0 ? pr_ser(2, 2) : pr_ser(4, 2));
		}
		/* the tops and the sides */
		for (px = pa; px <= pb + 1; px++) {
			double x = UX(A, px), y;
			if (x < R.a)
				x = R.a;
			if (x > R.b)
				x = R.b;
			hv = r_piece(x - 1e-12 * (R.b - R.a), n, R.rkind, ac->fv, &xl,
			    &xr);
			y = PY(A, hv);
			if (isfinite(prev) && isfinite(y))
				pr_seg(px - 1, prev, px, y, 1.1);
			prev = y;
		}
		for (i = 0; i <= n; i++) {
			int step = R.rkind == RM_SIMP ? 2 : 1;
			double x = R.a + (R.b - R.a) * i / n, hl, hr;
			if (i % step || (n > 24 && i > 0 && i < n))
				continue;	/* many: the tops show the pieces */
			hl = i > 0 ? r_piece(x - 1e-9 * (R.b - R.a), n, R.rkind, ac->fv,
			    &xl, &xr) : 0;
			hr = i < n ? r_piece(x + 1e-9 * (R.b - R.a), n, R.rkind, ac->fv,
			    &xl, &xr) : 0;
			if (!isfinite(hl))
				hl = 0;
			if (!isfinite(hr))
				hr = 0;
			if (fabs(hl) < fabs(hr) ? 0 : 0) {}
			pr_seg(PX(A, x), y0px, PX(A, x), PY(A, fabs(hl) > fabs(hr) ? hl :
			    hr), 1.1);
		}
		pr_ink_ser(2);
		if (R.rkind == RM_MID || R.rkind == RM_LEFT || R.rkind == RM_RIGHT) {
			double hh = (R.b - R.a) / n;
			if (n <= 16)
				for (i = 0; i < n; i++) {
					double x = R.a + hh * (i + (R.rkind == RM_LEFT ? 0 :
					    R.rkind == RM_RIGHT ? 1 : 0.5));
					double v = R.rkind == RM_LEFT ? ac->fv[i] : R.rkind ==
					    RM_RIGHT ? ac->fv[i + 1] : ac->fv[n + 1 + i];
					pr_disc(PX(A, x), PY(A, v), n <= 16 ? 2.4 : 1.5);
				}
			pr_ink_solid(pr_ink(PR_TEXT));
		}
	}
	pr_axes(A, xname(), "y", 0);
	pr_clip(A->L + 1, A->T + 1, A->R - 1, A->B - 1);
	/* the curves */
	curve(A, &R.f, 0, A->lo[0], A->hi[0], 2.2);
	if (R.ng)
		curve(A, &R.g, 1, A->lo[0], A->hi[0], 2.2);
	/* a and b */
	{
		double fa, fb, ga = 0, gb = 0;
		int oa = eval_at(&R.f, R.a, &fa), ob = eval_at(&R.f, R.b, &fb);
		if (R.ng) {
			eval_at(&R.g, R.a, &ga);
			eval_at(&R.g, R.b, &gb);
		}
		dashed_v(PX(A, R.a), A->T + 1, A->B - 1, pr_ink(PR_AXIS));
		dashed_v(PX(A, R.b), A->T + 1, A->B - 1, pr_ink(PR_AXIS));
		if (oa)
			pr_seg(PX(A, R.a), PY(A, ga), PX(A, R.a), PY(A, fa), 1.6);
		if (ob)
			pr_seg(PX(A, R.b), PY(A, gb), PX(A, R.b), PY(A, fb), 1.6);
		pr_ink_solid(pr_ink(PR_TEXT));
		{
			char t[64];
			snprintf(t, sizeof t, "a=%s", btxt(R.a));
			pr_text_bg((int)PX(A, R.a) - pr_textw(t) - 3, A->B - 14, t,
			    pr_ink(PR_TEXT), pr_ink(PR_LEGBG));
			snprintf(t, sizeof t, "b=%s", btxt(R.b));
			pr_text_bg((int)PX(A, R.b) + 4, A->B - 14, t, pr_ink(PR_TEXT),
			    pr_ink(PR_LEGBG));
		}
	}
	/* the sweep */
	if (ac->sweep) {
		double ft;
		pr_seg(PX(A, cut), A->T + 1, PX(A, cut), A->B - 1, 1.4);
		pr_ink_solid(pr_ink(PR_MARK));
		if (eval_at(&R.f, cut, &ft)) {
			pr_disc(PX(A, cut), PY(A, ft), 3.2);
			pr_ink_solid(pr_ink(PR_MARK));
		}
	}
	/* the legend */
	snprintf(lab[nl], 80, "%s", R.fl);
	col[nl] = 0;
	sty[nl++] = 0;
	if (isnan(ac->pos)) {
		if (R.ng) {
			snprintf(lab[nl], 80, "%s", R.gl);
			col[nl] = 1;
			sty[nl++] = 0;
		}
	} else if (R.ng) {
		snprintf(lab[nl], 80, "%s", R.gl);
		col[nl] = 1;
		sty[nl++] = 0;
		if (ac->pos != 0) {
			snprintf(lab[nl], 80, "%s above: %s", R.fl, pr_g4(ac->pos));
			col[nl] = 0;
			sty[nl++] = 2;
		}
		if (ac->neg != 0) {
			snprintf(lab[nl], 80, "%s above: %s", R.gl, pr_g4(-ac->neg));
			col[nl] = 1;
			sty[nl++] = 2;
		}
	} else if (ac->sweep) {
		/* the running area */
		snprintf(lab[nl], 80, "A(%s) = %s", pr_g4(cut), pr_g4(ac->run));
		col[nl] = ac->run < 0 ? 3 : 0;
		sty[nl++] = 2;
	} else if (!ac->rn) {
		if (ac->pos != 0 || ac->neg == 0) {
			snprintf(lab[nl], 80, "above the axis: %s", pr_g4(ac->pos));
			col[nl] = 0;
			sty[nl++] = 2;
		}
		if (ac->neg != 0) {
			snprintf(lab[nl], 80, "below the axis: %s", pr_g4(ac->neg));
			col[nl] = 3;
			sty[nl++] = 2;
		}
	} else {
		snprintf(lab[nl], 80, "%s sum, n = %d: %s", rmname[R.rkind], ac->rn,
		    pr_g4(ac->rsum));
		col[nl] = 2;
		sty[nl++] = 2;
	}
	pr_legend(nl, (const char (*)[80])lab, col, sty, A);
	pr_clip(0, 0, 100000, 100000);
	pr_title(x0 + w / 2, y0 + 8, title, w - 20);
}

/* ---- area ---- */

static void
area_cmd(void)
{
	struct integ I, P;
	nid e;
	double z[64], pos = 0, neg = 0, abs_area, exact_v;
	double pts[66];
	int nz, k, npts, allexact = 1, W = 640, H = 440;
	struct pr_ax A;
	struct gif g;
	unsigned char pal[256][3];
	int r0, nr, np;
	char title[400];
	struct achart ac;

	if (R.ng)
		printf("area between %s and %s\n", R.fl, R.gl);
	else
		printf("area under %s\n", R.fl);
	default_bounds(1);
	e = diff_top();
	printf("  ∫ %s d%s from %s to %s\n", R.ng ? node_str(SUB(R.f.body,
	    R.g.body)) : node_str(R.f.body), xname(), pr_gpi(R.a), pr_gpi(R.b));
	integrate(e, R.a, R.b, &I, 1);
	print_integ("  ", &I, R.a, R.b, 1);
	exact_v = ival(&I);
	if (!I.good && !I.exact) {
		if (!I.diverge)
			printf("  the integral does not exist as a number: %s is singular "
			    "or undefined inside [%s, %s]\n", R.ng ? "f - g" : R.fl,
			    btxt(R.a), btxt(R.b));
		if (R.rkind || R.accum) {
			printf("  note: no Riemann sums or accumulation of a divergent "
			    "integral\n");
			R.rkind = RM_NONE;
			R.accum = 0;
		}
		exact_v = NAN;
	}
	/* the parts above and below */
	nz = zeros_of(e, R.a, R.b, z, 64);
	npts = 0;
	pts[npts++] = R.a;
	for (k = 0; k < nz; k++)
		if (z[k] > R.a + 1e-12 * (R.b - R.a) && z[k] < R.b - 1e-12 *
		    (R.b - R.a))
			pts[npts++] = z[k];
	pts[npts++] = R.b;
	if (npts > 2) {
		printf("  %s at %s = ", R.ng ? "crossings inside" : "zeros inside",
		    xname());
		for (k = 1; k + 1 < npts; k++)
			printf("%s%s", k > 1 ? ", " : "", pr_gpi(pts[k]));
		printf("\n");
	}
	for (k = 0; k + 1 < npts; k++) {
		double v;
		int ex;
		if (npts == 2) {
			v = ival(&I);
			ex = I.exact;
		} else if (I.exact) {
			int o1, o2;
			v = eval_const_at(I.F, 0, pts[k + 1], &o1) - eval_const_at(I.F, 0,
			    pts[k], &o2);
			ex = 1;
		} else {
			integrate(e, pts[k], pts[k + 1], &P, 0);
			v = P.v;
			ex = 0;
		}
		allexact &= ex;
		if (v >= 0)
			pos += v;
		else
			neg += v;
		if (npts > 2 && exact_v == exact_v)
			printf("    on [%s, %s]: %s\n", btxt(pts[k]), btxt(pts[k + 1]),
			    val_text(v, ex));
	}
	abs_area = pos - neg;
	if (I.diverge)
		printf("  no area: the integral diverges at %s = %s\n", xname(),
		    btxt(I.diverge == 1 ? R.a : R.b));
	if (!(exact_v == exact_v)) {
		pos = neg = abs_area = NAN;
	} else if (R.ng) {
		if (pos != 0 && neg != 0)
			printf("  %s above %s: %s; %s above %s: %s\n", R.fl, R.gl,
			    val_text(pos, allexact), R.gl, R.fl, val_text(-neg, allexact));
		else
			printf("  %s above %s everywhere\n", pos != 0 ? R.fl : R.gl,
			    pos != 0 ? R.gl : R.fl);
		printf("  area between the curves (∫ |f - g|): %s; signed integral "
		    "∫ (f - g): %s\n", val_text(abs_area, allexact),
		    val_text(exact_v, I.exact));
	} else if (neg != 0 && pos != 0) {
		printf("  above the axis %s, below %s: signed integral %s, total "
		    "area (∫ |f|) %s\n", val_text(pos, allexact), val_text(neg,
		    allexact), val_text(exact_v, I.exact), val_text(abs_area,
		    allexact));
	} else
		printf("  %s the axis everywhere: the area is %s\n", neg < 0 ?
		    "below" : "above", val_text(fabs(exact_v), I.exact));
	if (R.rkind && R.ng) {
		printf("  note: Riemann sums for one function; leaving them out\n");
		R.rkind = RM_NONE;
	}
	if (R.rkind && R.accum) {
		printf("  note: riemann or accumulate, one at a time: riemann\n");
		R.accum = 0;
	}
	if (R.accum && R.ng) {
		printf("  note: accumulate is for one function; leaving it out\n");
		R.accum = 0;
	}
	if (!R.name[0])
		snprintf(R.name, sizeof R.name, "area");
	/* the chart's range: [a, b] and some around it */
	{
		double w = R.b - R.a, ylo, yhi;
		A.lo[0] = R.a - w * (R.accum ? 0.07 : 0.12);
		A.hi[0] = R.b + w * (R.accum ? 0.07 : 0.12);
		{	/* not into nothing: clip to where f is defined */
			double l = A.lo[0], h = A.hi[0], fa;
			if (!eval_at(&R.f, l, &fa))
				l = R.a;
			if (!eval_at(&R.f, h, &fa))
				h = R.b;
			A.lo[0] = l;
			A.hi[0] = h;
		}
		y_range(A.lo[0], A.hi[0], &ylo, &yhi, NULL, 0);
		A.lo[1] = ylo;
		A.hi[1] = yhi;
	}
	pr_begin(W, H);
	np = pr_palette(pal, &r0, &nr);
	(void)r0;
	(void)nr;
	memset(&ac, 0, sizeof ac);
	ac.tcut = R.b;
	ac.pos = pos;
	ac.neg = neg;
	if (R.rkind) {
		int ns[16], nn = 0, f, nmax;
		double *fv;
		if (R.rn) {
			if (R.rkind == RM_SIMP && R.rn % 2) {
				printf("  note: Simpson's rule takes an even n: %d\n", R.rn +
				    1);
				R.rn++;
			}
			ns[nn++] = R.rn;
		} else {
			for (k = R.rkind == RM_SIMP ? 2 : 1; k <= 64; k *= 2)
				ns[nn++] = k;
		}
		nmax = ns[nn - 1];
		fv = xmalloc((2 * nmax + 2) * sizeof *fv);
		printf("  %s sums (exact %s):\n", rmname[R.rkind], val_text(exact_v,
		    I.exact));
		gif_begin(&g, W, H, (const unsigned char (*)[3])pal, np);
		for (f = 0; f < nn; f++) {
			double s;
			if (!r_values(ns[f], fv))
				fail("area: %s is undefined at a sample point of the %s "
				    "sum", R.fl, rmname[R.rkind]);
			s = r_sum(ns[f], R.rkind, fv);
			printf("    n = %d: %.10g, error %.3g\n", ns[f], s, s - exact_v);
			ac.rn = ns[f];
			ac.fv = fv;
			ac.rsum = s;
			pr_clear();
			snprintf(title, sizeof title, "%s sum of %s, n = %d: %s  "
			    "(integral %s, error %s)", rmname[R.rkind], R.fl, ns[f],
			    pr_g4(s), val_text(exact_v, I.exact), pr_g4(s - exact_v));
			area_chart(&A, 0, 0, W, H, title, &ac);
			gif_frame(&g, pr_pixels(), nn > 1 ? (f + 1 == nn ? 250 : 90) : 0);
		}
		free(fv);
		pr_end();
		gif_finish(&g, W, H, nn, nn > 1 ? 0.9 * (nn - 1) + 2.5 : 0);
		return;
	}
	if (R.accum) {
		/* A(t) = ∫_a^t f, on a fine grid (Simpson by pieces) */
		int nf = R.frames ? R.frames : 48, fps = R.fps ? R.fps : 12, f, i;
		int M = 2000, H2 = 640;
		double *ax = xmalloc((M + 1) * sizeof *ax), *ay = xmalloc((M + 1) *
		    sizeof *ay), *mx = xmalloc(M * sizeof *mx), *my = xmalloc(M *
		    sizeof *my), *fy = xmalloc((M + 1) * sizeof *fy);
		uint8_t *o1 = xmalloc(M + 1), *o2 = xmalloc(M);
		struct pr_ax B;
		nid Fx = 0;
		for (i = 0; i <= M; i++)
			ax[i] = R.a + (R.b - R.a) * i / M;
		for (i = 0; i < M; i++)
			mx[i] = (ax[i] + ax[i + 1]) / 2;
		eval_n(&R.f, M + 1, ax, fy, o1);
		eval_n(&R.f, M, mx, my, o2);
		ay[0] = 0;
		for (i = 0; i < M; i++) {
			double h = ax[i + 1] - ax[i];
			double piece = o1[i] == 1 && o1[i + 1] == 1 && o2[i] == 1 ?
			    h * (fy[i] + 4 * my[i] + fy[i + 1]) / 6 : 0;
			ay[i + 1] = ay[i] + piece;
		}
		if (I.exact) {
			nid map[NSLOT];
			memset(map, 0, sizeof map);
			map[0] = KR(R.a);
			Fx = tidy(SUB(I.F, subst(I.F, map, NSLOT)));
			printf("  accumulation: A(%s) = ∫ from %s to %s = %s\n", xname(),
			    btxt(R.a), xname(), node_str(Fx));
		} else
			printf("  accumulation: A(%s) = ∫ from %s to %s, numeric (Simpson "
			    "on %d pieces)\n", xname(), btxt(R.a), xname(), M);
		{
			double v[3], vmin, vmax, lo, hi;
			int q;
			v[0] = 0;
			lo = hi = 0;
			for (q = 0; q <= M; q++) {
				if (ay[q] < lo)
					lo = ay[q];
				if (ay[q] > hi)
					hi = ay[q];
			}
			v[1] = lo;
			v[2] = hi;
			pr_vrange(v, 3, &B.lo[1], &B.hi[1], &vmin, &vmax);
			B.lo[0] = A.lo[0];
			B.hi[0] = A.hi[0];
			B.lo[1] -= (B.hi[1] - B.lo[1]) * 0.02;
		}
		pr_end();
		pr_begin(W, H2);
		gif_begin(&g, W, H2, (const unsigned char (*)[3])pal, np);
		for (f = 0; f < nf; f++) {
			double t = R.a + (R.b - R.a) * f / (nf - 1), At;
			int it = (int)floor((t - R.a) / (R.b - R.a) * M + 0.5), q;
			char t2[200];
			if (it > M)
				it = M;
			At = ay[it];
			pr_clear();
			ac.tcut = t;
			ac.sweep = 1;
			ac.run = At;
			snprintf(title, sizeof title, "the area under %s from %s to %s = "
			    "%s", R.fl, btxt(R.a), pr_g4(t), pr_g4(At));
			area_chart(&A, 0, 0, W, H2 / 2 + 10, title, &ac);
			/* A's graph below */
			pr_layout(&B, 0, H2 / 2, W, H2 / 2, 0);
			pr_clip(B.L + 1, B.T + 1, B.R - 1, B.B - 1);
			pr_axes(&B, xname(), "A", 0);
			for (q = 0; q < M; q++)
				pr_seg(PX(&B, ax[q]), PY(&B, ay[q]), PX(&B, ax[q + 1]),
				    PY(&B, ay[q + 1]), 1.2);
			{
				int tt[4];
				tt[0] = tt[1] = pr_ser(2, 2);
				tt[2] = tt[3] = pr_ser(2, 3);
				pr_ink_tints(tt);
			}
			for (q = 0; q < it; q++)
				pr_seg(PX(&B, ax[q]), PY(&B, ay[q]), PX(&B, ax[q + 1]),
				    PY(&B, ay[q + 1]), 2.4);
			pr_ink_ser(2);
			pr_seg(PX(&B, t), B.T + 1, PX(&B, t), B.B - 1, 1.4);
			pr_ink_solid(pr_ink(PR_MARK));
			pr_disc(PX(&B, t), PY(&B, At), 3.2);
			pr_ink_solid(pr_ink(PR_MARK));
			pr_clip(0, 0, 100000, 100000);
			snprintf(t2, sizeof t2, "A(%s) = integral of %s from %s to %s",
			    xname(), R.fl, btxt(R.a), xname());
			pr_title(W / 2, H2 / 2 + 8, t2, W - 20);
			gif_frame(&g, pr_pixels(), f + 1 == nf ? 250 : delay_cs(f, fps));
		}
		free(ax);
		free(ay);
		free(mx);
		free(my);
		free(fy);
		free(o1);
		free(o2);
		pr_end();
		(void)Fx;
		gif_finish(&g, W, H2, nf, (double)(nf - 1) / fps + 2.5);
		return;
	}
	pr_clear();
	if (!(exact_v == exact_v))
		snprintf(title, sizeof title, "%s from %s to %s: singular, no value",
		    R.ng ? "f - g" : R.fl, btxt(R.a), btxt(R.b));
	else if (R.ng)
		snprintf(title, sizeof title, "between %s and %s from %s to %s: area "
		    "%s", R.fl, R.gl, btxt(R.a), btxt(R.b), val_text(abs_area,
		    allexact));
	else
		snprintf(title, sizeof title, "integral of %s from %s to %s = %s",
		    R.fl, btxt(R.a), btxt(R.b), val_text(exact_v, I.exact));
	gif_begin(&g, W, H, (const unsigned char (*)[3])pal, np);
	area_chart(&A, 0, 0, W, H, title, &ac);
	gif_frame(&g, pr_pixels(), 0);
	pr_end();
	gif_finish(&g, W, H, 1, 0);
}

/* ---- solids of revolution: the numbers ---- */

/*
 * The region: between top and bottom over [a, b].  One function about a
 * horizontal axis: between F and the axis; about a vertical one: between
 * F and the x-axis; two: between them.
 */
static struct func Gf;		/* the bottom curve as a function */
static int gconst;

static void
region_setup(void)
{
	if (R.ng) {
		Gf = R.g;
		gconst = 0;
		return;
	}
	Gf = R.f;
	Gf.body = KR(R.axis == AX_H ? R.c : 0);
	Gf.where = 0;
	gconst = 1;
}

static const char *
axis_text(void)
{
	static char b[64];

	if (R.axis == AX_H && R.c == 0)
		return "the x-axis";
	if (R.axis == AX_V && R.c == 0)
		return "the y-axis";
	snprintf(b, sizeof b, "the line %s = %s", R.axis == AX_H ? "y" : "x",
	    pr_g4(R.c));
	return b;
}

/* sign of e on [a, b] from samples: 1, -1, or 0 (changes or undefined) */
static int
sign_on(nid e)
{
	double xs[NS], ys[NS];
	uint8_t ok[NS];
	int i, p = 0, m = 0;

	for (i = 0; i < NS; i++)
		xs[i] = R.a + (R.b - R.a) * i / (NS - 1);
	eval_node(e, NS, xs, ys, ok);
	for (i = 0; i < NS; i++) {
		double tol = 1e-12 * (1 + fabs(ys[i]));
		if (ok[i] != 1)
			continue;
		if (ys[i] > tol)
			p = 1;
		if (ys[i] < -tol)
			m = 1;
	}
	return p && !m ? 1 : m && !p ? -1 : !p && !m ? 1 : 0;
}

/* ∫ over the region's slices perpendicular to the axis, on grids */
static double
vol_grid_other(void)
{
	int NX = 4000, NY = 1200, i, j;
	double *xs = xmalloc(NX * sizeof *xs), *ft = xmalloc(NX * sizeof *ft),
	    *gt = xmalloc(NX * sizeof *gt), V = 0, dx = (R.b - R.a) / NX;
	uint8_t *o1 = xmalloc(NX), *o2 = xmalloc(NX);
	double lo = INFINITY, hi = -INFINITY;

	for (i = 0; i < NX; i++)
		xs[i] = R.a + (i + 0.5) * dx;
	eval_n(&R.f, NX, xs, ft, o1);
	eval_n(&Gf, NX, xs, gt, o2);
	for (i = 0; i < NX; i++) {
		if (o1[i] != 1 || o2[i] != 1) {
			o1[i] = 0;
			continue;
		}
		if (ft[i] > gt[i]) {
			double t = ft[i];
			ft[i] = gt[i];
			gt[i] = t;
		}
		/* ft is now the lower value, gt the upper */
		lo = fmin(lo, ft[i]);
		hi = fmax(hi, gt[i]);
	}
	if (R.axis == AX_H) {
		/* shells: radius r, length of the x where r is in the washer */
		double Rm = 0, dr;
		for (i = 0; i < NX; i++)
			if (o1[i])
				Rm = fmax(Rm, fmax(fabs(ft[i] - R.c), fabs(gt[i] - R.c)));
		dr = Rm / NY;
		for (j = 0; j < NY; j++) {
			double r = (j + 0.5) * dr, L = 0;
			for (i = 0; i < NX; i++) {
				double u = ft[i] - R.c, v = gt[i] - R.c, rin, rout;
				if (!o1[i])
					continue;
				if (u <= 0 && v >= 0) {
					rin = 0;
					rout = fmax(-u, v);
				} else {
					rin = fmin(fabs(u), fabs(v));
					rout = fmax(fabs(u), fabs(v));
				}
				if (r >= rin && r <= rout)
					L += dx;
			}
			V += 2 * M_PI * r * L * dr;
		}
	} else {
		/* washers in y: at height y, the radii |x - c| of the region */
		double dy = (hi - lo) / NY;
		double *ri = xmalloc(2 * NX * sizeof *ri);
		for (j = 0; j < NY; j++) {
			double y = lo + (j + 0.5) * dy, area = 0, cur_lo = 0, cur_hi = -1;
			int n = 0, k;
			for (i = 0; i < NX; i++) {
				double x0, x1, r0, r1;
				if (!o1[i] || y < ft[i] || y > gt[i])
					continue;
				x0 = xs[i] - dx / 2 - R.c;
				x1 = xs[i] + dx / 2 - R.c;
				if (x0 < 0 && x1 > 0) {
					r0 = 0;
					r1 = fmax(-x0, x1);
				} else {
					r0 = fmin(fabs(x0), fabs(x1));
					r1 = fmax(fabs(x0), fabs(x1));
				}
				ri[2 * n] = r0;
				ri[2 * n + 1] = r1;
				n++;
			}
			/* the union of the radius intervals: sort by start */
			for (k = 1; k < n; k++) {
				double s0 = ri[2 * k], s1 = ri[2 * k + 1];
				int q = k - 1;
				while (q >= 0 && ri[2 * q] > s0) {
					ri[2 * q + 2] = ri[2 * q];
					ri[2 * q + 3] = ri[2 * q + 1];
					q--;
				}
				ri[2 * q + 2] = s0;
				ri[2 * q + 3] = s1;
			}
			for (k = 0; k < n; k++) {
				if (ri[2 * k] > cur_hi) {
					if (cur_hi > cur_lo)
						area += M_PI * (cur_hi * cur_hi - cur_lo * cur_lo);
					cur_lo = ri[2 * k];
					cur_hi = ri[2 * k + 1];
				} else if (ri[2 * k + 1] > cur_hi)
					cur_hi = ri[2 * k + 1];
			}
			if (cur_hi > cur_lo)
				area += M_PI * (cur_hi * cur_hi - cur_lo * cur_lo);
			V += area * dy;
		}
		free(ri);
	}
	free(xs);
	free(ft);
	free(gt);
	free(o1);
	free(o2);
	return V;
}

struct solid {
	double V;		/* the volume reported */
	int exact;
	double Vo;		/* the other method, on a grid */
	double S[2];		/* lateral surface of F (and G) */
	int Sok[2], Sexact[2];
	double A, xc, yc;	/* the region's area and centroid */
	int straddle;
};

/* the derivative of f's body, 0 if none */
static nid
deriv_of(const struct func *f)
{
	jmp_buf saved;
	nid d = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0)
		d = diff_node(f->body, 0);
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return d;
}

/* 2pi ∫ r sqrt(1 + f'^2): the surface swept by the curve f */
static void
surface_of(const struct func *f, const char *label, double *S, int *ok,
    int *ex, int print)
{
	nid d = deriv_of(f), r, e;
	struct integ I;

	*ok = *ex = 0;
	if (!d) {
		if (print)
			printf("    %s: no derivative, no surface area\n", label);
		return;
	}
	d = mk_cast(TY_REAL, d);
	r = R.axis == AX_H ? SUB(body_of(f), KR(R.c)) : SUB(xvar(), KR(R.c));
	if (sign_on(r) < 0)
		r = mk_un(O_NEG, r);
	else if (sign_on(r) == 0)
		r = mk_un(O_ABS, r);
	e = MUL(r, mk_un(O_SQRT, ADD(KR(1), MUL(d, d))));
	integrate(e, R.a, R.b, &I, 1);
	*ok = I.good || I.exact;
	*ex = I.exact;
	*S = 2 * M_PI * ival(&I);
	if (print) {
		printf("    %s: S = 2pi ∫ |%s| sqrt(1 + (%s)^2) d%s\n", label,
		    node_str(R.axis == AX_H ? minus_c(f->body, R.c) : minus_c(xvar(),
		    R.c)), node_str(d), xname());
		if (I.exact)
			printf("      exact: %s\n", full(*S));
		if (I.good)
			printf("      numeric: %.15g, estimated error %.2g\n", 2 * M_PI *
			    I.v, 2 * M_PI * I.err);
		else
			printf("      numeric: singular or divergent\n");
	}
}

/* e - c, printed as e + 1 for c = -1 */
static nid
minus_c(nid e, double c)
{
	return c == 0 ? e : c < 0 ? ADD(e, KR(-c)) : SUB(e, KR(c));
}

/*
 * e^2 in a form the antiderivative rules take: sqrt(u)^2 = u (where e is
 * defined: dom), sin^2 u = (1 - cos 2u)/2, cos^2 u = (1 + cos 2u)/2,
 * exp(u)^2 = exp(2u)
 */
static nid
sq(nid e, int dom)
{
	int op = N.op[e];
	nid a = N.a[e];

	if (op == O_CAST && TY_FP(N.ty[e]) && TY_FP(N.ty[a]))
		return sq(a, dom);
	if (op == O_SQRT && dom)
		return mk_cast(TY_REAL, a);
	if (op == O_ABS || op == O_NEG)
		return sq(a, dom);
	if (op == O_SIN || op == O_COS)
		return MUL(KR(0.5), mk_bin(op == O_SIN ? O_SUB : O_ADD, KR(1),
		    mk_un(O_COS, MUL(KR(2), mk_cast(TY_REAL, a)))));
	if (op == O_EXP)
		return mk_un(O_EXP, MUL(KR(2), mk_cast(TY_REAL, a)));
	if (op == O_ADD || op == O_SUB) {	/* (a + b)^2, term by term */
		nid b = N.b[e];
		return mk_bin(op, ADD(sq(a, dom), sq(b, dom)), MUL(KR(2), MUL(a, b)));
	}
	return MUL(e, e);
}

/* the region's area and centroid; with pappus, V = 2pi d A checked */
static void
region_centroid(struct solid *so, int print, int pappus)
{
	nid fb = body_of(&R.f), gb = body_of(&Gf), h = SUB(fb, gb), x = xvar();
	struct integ Ia, Ix, Iy;
	int sh = sign_on(h), ex;
	nid ha = sh > 0 ? h : sh < 0 ? mk_un(O_NEG, h) : mk_un(O_ABS, h);
	double d;

	integrate(ha, R.a, R.b, &Ia, 1);
	integrate(MUL(x, ha), R.a, R.b, &Ix, 1);
	integrate(MUL(MUL(KR(0.5), ADD(fb, gb)), ha), R.a, R.b, &Iy, 1);
	so->A = ival(&Ia);
	ex = Ia.exact && Ix.exact && Iy.exact;
	if (!(so->A > 0)) {
		if (print)
			printf("  region: no area\n");
		return;
	}
	so->xc = ival(&Ix) / so->A;
	so->yc = ival(&Iy) / so->A;
	if (!print)
		return;
	printf("  region: area %s, centroid (%s, %s)%s\n", val_text(so->A,
	    Ia.exact), val_text(so->xc, ex), val_text(so->yc, ex), ex ?
	    " (exact)" : "");
	if (!pappus)
		return;
	d = R.axis == AX_H ? fabs(so->yc - R.c) : fabs(so->xc - R.c);
	if (!so->straddle)
		printf("  Pappus: V = 2pi d A, d = %s from the centroid to the axis: "
		    "%s%s\n", val_text(d, ex), pr_g4(2 * M_PI * d * so->A),
		    fabs(2 * M_PI * d * so->A - so->V) <= 1e-6 * fabs(so->V) + 1e-12 ?
		    " (agrees)" : " (DIFFERS)");
	else
		printf("  Pappus does not apply: the region crosses the axis\n");
}

static void
solid_numbers(struct solid *so, int print, int what)
{
	nid fb = body_of(&R.f), gb = body_of(&Gf), e, ef;
	struct integ I;
	const char *method;
	int sf, sg, use_shell, s1, s2;

	memset(so, 0, sizeof *so);
	if (R.axis == AX_H) {
		sf = sign_on(SUB(fb, KR(R.c)));
		sg = sign_on(SUB(gb, KR(R.c)));
		so->straddle = sf * sg < 0 || sf == 0 || sg == 0;
		/* one curve: disks of radius |f - c| whatever its sign */
		if (gconst && !R.ng)
			so->straddle = sf == 0 ? 2 : 0;
	} else {
		so->straddle = (R.a < R.c && R.b > R.c);
	}
	use_shell = R.axis == AX_V;
	if (R.method == M_SHELL && R.axis == AX_H) {
		if (print)
			printf("  note: shells about a horizontal axis integrate in y "
			    "(x as a function of y): disks/washers in %s instead, shells "
			    "as the check\n", xname());
	}
	if (R.method == M_WASHER && R.axis == AX_V) {
		if (print)
			printf("  note: washers about a vertical axis integrate in y: "
			    "shells in %s instead, washers in y as the check\n", xname());
	}
	if (what == VC_SURFACE)
		goto surface;
	if (!use_shell) {
		/* washers: pi ∫ (R^2 - r^2), R, r the distances to the axis */
		nid u = minus_c(fb, R.c), v = minus_c(gb, R.c);
		int dom = defined_on(fb, R.a, R.b) && defined_on(gb, R.a, R.b);
		method = gconst && R.ng == 0 ? "disks" : "washers";
		if (so->straddle != 1) {
			/* the outer is the one further from the axis */
			s1 = sign_on(SUB(mk_un(O_ABS, u), mk_un(O_ABS, v)));
			e = s1 >= 0 ? SUB(sq(u, dom), sq(v, dom)) : SUB(sq(v, dom),
			    sq(u, dom));
			if (gconst && !R.ng)
				e = sq(u, dom);
			ef = e;
			if (s1 == 0) {
				ef = mk_un(O_ABS, e);
			}
		} else {
			/* the region crosses the axis: R = max, r = 0 */
			ef = e = mk_bin(O_MAX, MUL(u, u), MUL(v, v));
		}
		if (print) {
			if (gconst && !R.ng)
				printf("  method: %s, V = pi ∫ (%s)^2 d%s from %s to %s\n",
				    method, node_str(minus_c(R.f.body, R.c)), xname(),
				    btxt(R.a), btxt(R.b));
			else
				printf("  method: %s, V = pi ∫ (R^2 - r^2) d%s, R and r the "
				    "outer and inner distances to the axis\n", method,
				    xname());
			if (so->straddle == 1)
				printf("  note: the region crosses %s: each slice is a disk "
				    "out to the further curve (the two sides overlap)\n",
				    axis_text());
		}
	} else {
		/* shells: 2pi ∫ |x - c| |f - g| */
		nid rr = SUB(xvar(), KR(R.c)), hh = SUB(fb, gb);
		method = "shells";
		s1 = R.a >= R.c ? 1 : R.b <= R.c ? -1 : 0;
		s2 = sign_on(hh);
		rr = s1 > 0 ? rr : s1 < 0 ? mk_un(O_NEG, rr) : mk_un(O_ABS, rr);
		hh = s2 > 0 ? hh : s2 < 0 ? mk_un(O_NEG, hh) : mk_un(O_ABS, hh);
		e = ef = MUL(rr, hh);
		if (print) {
			printf("  method: shells, V = 2pi ∫ |%s| |%s| d%s from %s to %s\n",
			    node_str(minus_c(xvar(), R.c)), gconst && !R.ng ?
			    node_str(R.f.body) : node_str(SUB(R.f.body, R.g.body)),
			    xname(), btxt(R.a), btxt(R.b));
			if (so->straddle)
				printf("  note: [%s, %s] straddles %s: the sides of it turn "
				    "into the same solid; where they overlap the shell integral "
				    "counts twice, so the union is measured too\n",
				    btxt(R.a), btxt(R.b), axis_text());
		}
	}
	integrate(ef, R.a, R.b, &I, 1);
	{
		double k = use_shell ? 2 * M_PI : M_PI;
		so->exact = I.exact;
		so->V = k * ival(&I);
		if (print) {
			if (I.F)
				printf("    antiderivative of the integrand: %s\n",
				    node_str(I.F));
			if (I.exact)
				printf("    exact: V = %s\n", full(so->V));
			if (I.good)
				printf("    numeric (Gauss-Kronrod): %.15g, estimated error "
				    "%.2g%s\n", k * I.v, k * I.err, I.exact ? "" : "");
			else
				printf("    numeric: singular or divergent\n");
			if (I.exact && I.good)
				printf("    the two agree to %.2g\n", fabs(k * (I.v - I.ve)));
		}
	}
	so->Vo = vol_grid_other();
	if (use_shell && so->straddle) {
		if (fabs(so->Vo - so->V) <= 2e-3 * fabs(so->V)) {
			if (print)
				printf("  the two sides sweep separate parts (washers in y on a "
				    "grid: V ≈ %.6g): the shells add up\n", so->Vo);
		} else {
			so->V = so->Vo;
			so->exact = 0;
			if (print)
				printf("  the two sides overlap: the solid (their union), by "
				    "washers in y on a grid: V ≈ %.6g\n", so->V);
		}
	} else if (print) {
		double rel = fabs(so->Vo - so->V) / fmax(fabs(so->V), 1e-300);
		printf("  check: %s on a grid: V ≈ %.6g (relative difference %.1g%s)\n",
		    use_shell ? "washers in y" : "shells in the radius", so->Vo, rel,
		    rel < 2e-3 ? ", the grid's resolution" : ": MORE than the grid's "
		    "resolution");
	}
	if (print) {
		printf("  volume: %s%s\n", val_text(so->V, so->exact), so->exact ?
		    " (exact)" : " (numeric)");
	}
	if (what == VC_VOLUME || what == VC_REVOLVE)
		region_centroid(so, print, 1);
	if (what == VC_VOLUME)
		return;
surface:
	if (print)
		printf("  surface area (the curves swept around %s, without the end "
		    "disks):\n", axis_text());
	surface_of(&R.f, R.fl, &so->S[0], &so->Sok[0], &so->Sexact[0], print);
	if (R.ng)
		surface_of(&R.g, R.gl, &so->S[1], &so->Sok[1], &so->Sexact[1],
		    print);
	if (print && R.ng && so->Sok[0] && so->Sok[1])
		printf("    both: %s\n", pr_g4(so->S[0] + so->S[1]));
}

/* ---- solids of revolution: the picture ---- */

#define NU 96		/* points along each curve */
#define NTH 72		/* angle steps around */
#define NRAMP 4
#define RAMPN 48

static int ramp0;
static int VW, VH;		/* the canvas */
static double light_turn;	/* the view's turn since the start */

static void
ramps(unsigned char pal[256][3])
{
	static const double base[NRAMP][3] = {
		{38, 120, 190}, {120, 170, 215}, {235, 140, 45}, {120, 160, 120},
	};
	int r, k, c;

	for (r = 0; r < NRAMP; r++)
		for (k = 0; k < RAMPN; k++) {
			double t = (double)k / (RAMPN - 1);
			for (c = 0; c < 3; c++) {
				double v, b = base[r][c];
				if (t < 0.78)
					v = b * (0.28 + 0.72 * t / 0.78);
				else
					v = b + (250 - b) * (t - 0.78) / 0.22 * 0.55;
				pal[ramp0 + r * RAMPN + k][c] = (unsigned char)(v > 255 ? 255 :
				    v + 0.5);
			}
		}
}

struct bnd {			/* the boundary of the region, revolved */
	int n;
	double x[4 * NU + 8], y[4 * NU + 8];
	int part[4 * NU + 8];	/* 0 F, 1 G, 2 the ends; -1 breaks */
};

static void
boundary(struct bnd *B)
{
	double xs[NU], fy[NU], gy[NU];
	uint8_t o1[NU], o2[NU];
	int i;

	for (i = 0; i < NU; i++)
		xs[i] = R.a + (R.b - R.a) * i / (NU - 1);
	eval_n(&R.f, NU, xs, fy, o1);
	eval_n(&Gf, NU, xs, gy, o2);
	B->n = 0;
	/* F left to right, the right end, G right to left, the left end */
	for (i = 0; i < NU; i++) {
		B->x[B->n] = xs[i];
		B->y[B->n] = o1[i] == 1 ? fy[i] : NAN;
		B->part[B->n++] = 0;
	}
	for (i = NU - 1; i >= 0; i--) {
		B->x[B->n] = xs[i];
		B->y[B->n] = o2[i] == 1 ? gy[i] : NAN;
		B->part[B->n++] = 1;
	}
	B->x[B->n] = B->x[0];
	B->y[B->n] = B->y[0];
	B->part[B->n++] = 2;
}

struct view {
	struct pr_v3 V;
	double cx, cy, cz, s;	/* world centre and scale */
};

/* a point of the plane (x, y) turned by th about the axis, in world */
static void
turn(double x, double y, double th, double *X, double *Y, double *Z)
{
	if (R.axis == AX_H) {
		*X = x;
		*Y = R.c + (y - R.c) * cos(th);
		*Z = -(y - R.c) * sin(th);
	} else {
		*X = R.c + (x - R.c) * cos(th);
		*Y = y;
		*Z = -(x - R.c) * sin(th);
	}
}

static void
wproj(const struct view *v, double X, double Y, double Z, double *sx,
    double *sy, double *d)
{
	/* plot's projection takes Z as up; ours is Y */
	pr_proj(&v->V, (X - v->cx) * v->s, (Z - v->cz) * v->s, (Y - v->cy) *
	    v->s, sx, sy, d);
}

static void
view_fit(struct view *v, const double *bb, int x0, int y0, int x1, int y1,
    double az0, int turning)
{
	double a = INFINITY, b = -INFINITY, c = INFINITY, d = -INFINITY, sx, sy;
	double ext;
	int k, i;

	v->cx = (bb[0] + bb[1]) / 2;
	v->cy = (bb[2] + bb[3]) / 2;
	v->cz = (bb[4] + bb[5]) / 2;
	ext = fmax(bb[1] - bb[0], fmax(bb[3] - bb[2], bb[5] - bb[4])) / 2;
	v->s = ext > 0 ? 1 / ext : 1;
	v->V.S = 1;
	v->V.cx = v->V.cy = 0;
	for (k = 0; k < (turning ? 16 : 1); k++) {
		v->V.az = az0 + k * M_PI / 8;
		for (i = 0; i < 8; i++) {
			wproj(v, bb[i & 1], bb[2 + ((i >> 1) & 1)], bb[4 + (i >> 2)], &sx,
			    &sy, NULL);
			a = fmin(a, sx);
			b = fmax(b, sx);
			c = fmin(c, sy);
			d = fmax(d, sy);
		}
	}
	v->V.az = az0;
	v->V.S = fmin((x1 - x0) / (b - a), (y1 - y0) / (d - c));
	v->V.cx = (x0 + x1) / 2.0 - v->V.S * (a + b) / 2;
	v->V.cy = (y0 + y1) / 2.0 - v->V.S * (c + d) / 2;
}

/* a triangle with a shade per vertex (0..1), dithered, depth-tested */
static void
gtri(const double *x, const double *y, const double *z, const double *s,
    int ramp)
{
	static const unsigned char bayer[4][4] = {
		{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5},
	};
	unsigned char *px = pr_pixels();
	float *zb = pr_zbuf();
	int i, j, ia, ib, ja, jb;
	double area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);

	if (fabs(area) < 1e-9)
		return;
	ia = (int)floor(fmin(x[0], fmin(x[1], x[2])));
	ib = (int)ceil(fmax(x[0], fmax(x[1], x[2])));
	ja = (int)floor(fmin(y[0], fmin(y[1], y[2])));
	jb = (int)ceil(fmax(y[0], fmax(y[1], y[2])));
	if (ia < 0) ia = 0;
	if (ja < 0) ja = 0;
	if (ib >= VW) ib = VW - 1;
	if (jb >= VH) jb = VH - 1;
	for (j = ja; j <= jb; j++)
		for (i = ia; i <= ib; i++) {
			double qx = i + 0.5, qy = j + 0.5, w0, w1, w2, d, t;
			float *zp;
			int k;
			w0 = ((x[1] - qx) * (y[2] - qy) - (x[2] - qx) * (y[1] - qy)) / area;
			w1 = ((x[2] - qx) * (y[0] - qy) - (x[0] - qx) * (y[2] - qy)) / area;
			w2 = 1 - w0 - w1;
			if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9)
				continue;
			zp = &zb[(size_t)j * VW + i];
			d = w0 * z[0] + w1 * z[1] + w2 * z[2];
			if (d >= *zp)
				continue;
			*zp = (float)d;
			t = (w0 * s[0] + w1 * s[1] + w2 * s[2]) * (RAMPN - 1) +
			    (bayer[j & 3][i & 3] + 0.5) / 16.0 - 0.5;
			k = (int)floor(t + 0.5);
			if (k < 0)
				k = 0;
			if (k >= RAMPN)
				k = RAMPN - 1;
			px[(size_t)j * VW + i] = (unsigned char)(ramp0 + ramp * RAMPN + k);
		}
}

/* the unit tangent of the boundary at vertex i, within segment i..i+1's part */
static void
btan(const struct bnd *B, int i, int seg, double *tx, double *ty)
{
	double ax = 0, ay = 0, l;
	int p = B->part[seg + 1] == 2 || B->part[seg] != B->part[seg + 1] ? -1 :
	    B->part[seg];

	/* this segment, plus the neighbour on i's other side in the same part */
	ax = B->x[seg + 1] - B->x[seg];
	ay = B->y[seg + 1] - B->y[seg];
	l = hypot(ax, ay);
	if (l > 0)
		ax /= l, ay /= l;
	if (p >= 0) {
		int o = i == seg ? seg - 1 : seg + 2, a = i == seg ? o : i;
		int b = i == seg ? i : o;
		if (o >= 0 && o < B->n && B->part[o] == p && !isnan(B->y[o])) {
			double bx = B->x[b] - B->x[a], by = B->y[b] - B->y[a];
			double m = hypot(bx, by);
			if (m > 0) {
				ax += bx / m;
				ay += by / m;
			}
		}
	}
	l = hypot(ax, ay);
	*tx = l > 0 ? ax / l : 1;
	*ty = l > 0 ? ay / l : 0;
}

/* the surface for angles [0, th1], z-buffered, lit per vertex */
static void
solid_draw(const struct view *v, const struct bnd *B, double th1)
{
	static const double L0[3] = {-0.45, 0.62, -0.64};	/* towards the light */
	double L[3], ll = sqrt(L0[0] * L0[0] + L0[1] * L0[1] + L0[2] * L0[2]);
	int i, j, k, nj = (int)ceil(th1 / (2 * M_PI) * NTH - 1e-9);

	for (k = 0; k < 3; k++)
		L[k] = L0[k] / ll;
	{	/* the light turns with the eye (rotate) */
		double c = cos(light_turn), sn = sin(light_turn), lx = L[0], lz = L[2];
		L[0] = lx * c + lz * sn;
		L[2] = -lx * sn + lz * c;
	}
	for (i = 0; i + 1 < B->n; i++) {
		int part = B->part[i + 1] == 2 || B->part[i] != B->part[i + 1] ? 2 :
		    B->part[i];
		double t2[2][2];
		if (isnan(B->y[i]) || isnan(B->y[i + 1]))
			continue;
		if (part == 1 && gconst && R.axis == AX_H)
			continue;	/* on the axis: nothing */
		btan(B, i, i, &t2[0][0], &t2[0][1]);
		btan(B, i + 1, i, &t2[1][0], &t2[1][1]);
		for (j = 0; j < nj; j++) {
			double th[2], P[4][3], Nw[4][3], sx[4], sy[4], sd[4], sh[4];
			double tx[3], ty[3], tz[3], ts[3], cx, cy, cz, nx, ny, nz;
			int q, front, ramp;
			static const int vi[4] = {0, 1, 1, 0}, ti[4] = {0, 0, 1, 1};
			th[0] = 2 * M_PI * j / NTH;
			th[1] = 2 * M_PI * (j + 1) / NTH;
			if (th[1] > th1)
				th[1] = th1;
			if (th[1] <= th[0])
				continue;
			for (q = 0; q < 4; q++) {
				int a = i + vi[q];
				double t = th[ti[q]], n2x = -t2[vi[q]][1], n2y = t2[vi[q]][0];
				turn(B->x[a], B->y[a], t, &P[q][0], &P[q][1], &P[q][2]);
				/* the plane normal (-dy, dx), turned */
				if (R.axis == AX_H) {
					Nw[q][0] = n2x;
					Nw[q][1] = n2y * cos(t);
					Nw[q][2] = -n2y * sin(t);
				} else {
					Nw[q][0] = n2x * cos(t);
					Nw[q][1] = n2y;
					Nw[q][2] = -n2x * sin(t);
				}
			}
			/* which side faces the eye: the quad's centre and normal */
			cx = (P[0][0] + P[1][0] + P[2][0] + P[3][0]) / 4;
			cy = (P[0][1] + P[1][1] + P[2][1] + P[3][1]) / 4;
			cz = (P[0][2] + P[1][2] + P[2][2] + P[3][2]) / 4;
			nx = Nw[0][0] + Nw[1][0] + Nw[2][0] + Nw[3][0];
			ny = Nw[0][1] + Nw[1][1] + Nw[2][1] + Nw[3][1];
			nz = Nw[0][2] + Nw[1][2] + Nw[2][2] + Nw[3][2];
			if (part != 2) {	/* outwards: away from the axis */
				double rx = R.axis == AX_H ? 0 : cx - R.c;
				double ry = R.axis == AX_H ? cy - R.c : 0;
				if (nx * rx + ny * ry + nz * cz < 0) {
					nx = -nx, ny = -ny, nz = -nz;
					for (q = 0; q < 4; q++)
						for (k = 0; k < 3; k++)
							Nw[q][k] = -Nw[q][k];
				}
			}
			{
				double d0, d1, s0, s1, l = sqrt(nx * nx + ny * ny + nz * nz);
				if (!(l > 0))
					l = 1;
				wproj(v, cx, cy, cz, &s0, &s1, &d0);
				wproj(v, cx + nx / l * 1e-3 / v->s, cy + ny / l * 1e-3 / v->s,
				    cz + nz / l * 1e-3 / v->s, &s0, &s1, &d1);
				front = d1 < d0;
			}
			for (q = 0; q < 4; q++) {
				double lam = Nw[q][0] * L[0] + Nw[q][1] * L[1] + Nw[q][2] * L[2];
				if (!front)
					lam = -lam;
				if (part == 2)
					lam = fabs(lam);
				sh[q] = 0.16 + 0.84 * (lam > 0 ? lam : 0);
				wproj(v, P[q][0], P[q][1], P[q][2], &sx[q], &sy[q], &sd[q]);
			}
			ramp = part == 0 ? (front ? 0 : 1) : part == 1 ? 2 : 3;
			tx[0] = sx[0]; ty[0] = sy[0]; tz[0] = sd[0]; ts[0] = sh[0];
			tx[1] = sx[1]; ty[1] = sy[1]; tz[1] = sd[1]; ts[1] = sh[1];
			tx[2] = sx[2]; ty[2] = sy[2]; tz[2] = sd[2]; ts[2] = sh[2];
			gtri(tx, ty, tz, ts, ramp);
			tx[1] = sx[2]; ty[1] = sy[2]; tz[1] = sd[2]; ts[1] = sh[2];
			tx[2] = sx[3]; ty[2] = sy[3]; tz[2] = sd[3]; ts[2] = sh[3];
			gtri(tx, ty, tz, ts, ramp);
		}
	}
}

/* a world segment, depth-tested: visible parts into coverage set 0 */
static void
seg3(const struct view *v, double X0, double Y0, double Z0, double X1,
    double Y1, double Z1, double w, int hidden)
{
	double a, b, c, d, da, db;
	float *zb = pr_zbuf();
	int n, t, W = VW, H = VH;
	double px = NAN, py = NAN;

	wproj(v, X0, Y0, Z0, &a, &b, &da);
	wproj(v, X1, Y1, Z1, &c, &d, &db);
	n = (int)ceil(fmax(fabs(c - a), fabs(d - b)) / 1.5) + 1;
	for (t = 0; t <= n; t++) {
		double f = (double)t / n, x = a + (c - a) * f, y = b + (d - b) * f,
		    dep = da + (db - da) * f;
		int xi = (int)floor(x), yi = (int)floor(y), vis = 1;
		if (xi >= 0 && yi >= 0 && xi < W && yi < H)
			vis = dep <= zb[(size_t)yi * W + xi] + 0.02;
		if (vis != !hidden) {
			px = NAN;
			continue;
		}
		if (!isnan(px))
			pr_seg(px, py, x, y, w);
		px = x;
		py = y;
	}
}

/* the boundary at angle th, where it is seen */
static void
curve3(const struct view *v, const struct bnd *B, double th, int col,
    int xray)
{
	int i, hidden, t[4];

	for (hidden = xray; hidden >= 0; hidden--) {
		for (i = 0; i + 1 < B->n; i++) {
			double X0, Y0, Z0, X1, Y1, Z1;
			if (isnan(B->y[i]) || isnan(B->y[i + 1]))
				continue;
			if (gconst && R.axis == AX_H && B->part[i] == 1 &&
			    B->part[i + 1] == 1)
				continue;
			turn(B->x[i], B->y[i], th, &X0, &Y0, &Z0);
			turn(B->x[i + 1], B->y[i + 1], th, &X1, &Y1, &Z1);
			seg3(v, X0, Y0, Z0, X1, Y1, Z1, hidden ? 1.4 : 2.6, hidden);
		}
		if (hidden) {
			t[0] = t[1] = pr_ser(col, 1);
			t[2] = t[3] = pr_ser(col, 3);
			pr_ink_tints(t);
		} else
			pr_ink_ser(col);
	}
}

/* a flat triangle; with xray, where it is hidden every other pixel */
static void
rtri(const double *x, const double *y, const double *z, int col, int xray)
{
	unsigned char *px = pr_pixels();
	float *zb = pr_zbuf();
	int i, j, ia, ib, ja, jb;
	double area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);

	if (fabs(area) < 1e-9)
		return;
	ia = (int)floor(fmin(x[0], fmin(x[1], x[2])));
	ib = (int)ceil(fmax(x[0], fmax(x[1], x[2])));
	ja = (int)floor(fmin(y[0], fmin(y[1], y[2])));
	jb = (int)ceil(fmax(y[0], fmax(y[1], y[2])));
	if (ia < 0) ia = 0;
	if (ja < 0) ja = 0;
	if (ib >= VW) ib = VW - 1;
	if (jb >= VH) jb = VH - 1;
	for (j = ja; j <= jb; j++)
		for (i = ia; i <= ib; i++) {
			double qx = i + 0.5, qy = j + 0.5, w0, w1, w2, d;
			float *zp = &zb[(size_t)j * VW + i];
			w0 = ((x[1] - qx) * (y[2] - qy) - (x[2] - qx) * (y[1] - qy)) / area;
			w1 = ((x[2] - qx) * (y[0] - qy) - (x[0] - qx) * (y[2] - qy)) / area;
			w2 = 1 - w0 - w1;
			if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9)
				continue;
			d = w0 * z[0] + w1 * z[1] + w2 * z[2];
			if (d < *zp) {
				*zp = (float)d;
				px[(size_t)j * VW + i] = (unsigned char)col;
			} else if (xray && ((i + j) & 1) == 0)
				px[(size_t)j * VW + i] = (unsigned char)col;
		}
}

/* the region itself at angle th, flat, depth-tested (xray: seen through) */
static void
region3(const struct view *v, const struct bnd *B, double th, int c,
    int xray)
{
	int i, k;

	for (i = 0; i + 1 < NU; i++) {
		int j0 = 2 * NU - 1 - i, j1 = j0 - 1;
		double P[4][2], sx[4], sy[4], sd[4], tx[3], ty[3], tz[3];
		if (isnan(B->y[i]) || isnan(B->y[i + 1]) || isnan(B->y[j0]) ||
		    isnan(B->y[j1]))
			continue;
		P[0][0] = B->x[i]; P[0][1] = B->y[i];
		P[1][0] = B->x[i + 1]; P[1][1] = B->y[i + 1];
		P[2][0] = B->x[j1]; P[2][1] = B->y[j1];
		P[3][0] = B->x[j0]; P[3][1] = B->y[j0];
		for (k = 0; k < 4; k++) {
			double X, Y, Z;
			turn(P[k][0], P[k][1], th, &X, &Y, &Z);
			wproj(v, X, Y, Z, &sx[k], &sy[k], &sd[k]);
		}
		tx[0] = sx[0]; ty[0] = sy[0]; tz[0] = sd[0];
		tx[1] = sx[1]; ty[1] = sy[1]; tz[1] = sd[1];
		tx[2] = sx[2]; ty[2] = sy[2]; tz[2] = sd[2];
		rtri(tx, ty, tz, c, xray);
		tx[1] = sx[2]; ty[1] = sy[2]; tz[1] = sd[2];
		tx[2] = sx[3]; ty[2] = sy[3]; tz[2] = sd[3];
		rtri(tx, ty, tz, c, xray);
	}
}

struct rscene {
	double bb[6];		/* the solid's box */
	double lo, hi;		/* the axis drawn, along it */
	double Rm;		/* the largest radius */
	char title[300], foot[200];
};

static void
axes3(const struct view *v, const struct rscene *S)
{
	double tv[24], X0, Y0, Z0, X1, Y1, Z1, sx, sy;
	char lab[24][20];
	int n, k, t[4];

	/* the axis of revolution: dark where seen, faint behind */
	if (R.axis == AX_H) {
		X0 = S->lo, Y0 = R.c, X1 = S->hi, Y1 = R.c;
	} else {
		X0 = R.c, Y0 = S->lo, X1 = R.c, Y1 = S->hi;
	}
	Z0 = Z1 = 0;
	seg3(v, X0, Y0, Z0, X1, Y1, Z1, 1.0, 1);
	t[0] = t[1] = pr_ink(PR_LEGBORDER);
	t[2] = t[3] = pr_ink(PR_PANEGRID);
	pr_ink_tints(t);
	seg3(v, X0, Y0, Z0, X1, Y1, Z1, 1.4, 0);
	pr_ink_solid(pr_ink(PR_TEXT));
	/* the x-axis too, in the region's plane, when it is not the axis */
	if (!(R.axis == AX_H && R.c == 0)) {
		double xa = R.axis == AX_H ? S->lo : S->bb[0], xb = R.axis == AX_H ?
		    S->hi : S->bb[1];
		seg3(v, xa, 0, 0, xb, 0, 0, 1.0, 0);
		pr_ink_solid(pr_ink(PR_AXIS));
	}
	/* ticks along the axis, beside the solid */
	{	/* as many as the axis has room for on screen */
		double ax, ay, bx, by;
		wproj(v, X0, Y0, Z0, &ax, &ay, NULL);
		wproj(v, X1, Y1, Z1, &bx, &by, NULL);
		n = (int)(hypot(bx - ax, by - ay) / 55);
		n = pr_ticks(S->lo, S->hi, n < 2 ? 2 : n > 8 ? 8 : n, tv, lab);
	}
	{	/* a ruler beside the solid, parallel to the axis */
		double off = S->Rm * 1.1 + 1e-9, ex, ey;
		if (R.axis == AX_H)
			seg3(v, S->lo, R.c - off, 0, S->hi, R.c - off, 0, 1.0, 0);
		else
			seg3(v, R.c + off, S->lo, 0, R.c + off, S->hi, 0, 1.0, 0);
		pr_ink_solid(pr_ink(PR_AXIS));
		for (k = 0; k < n; k++) {
			if (R.axis == AX_H) {
				wproj(v, tv[k], R.c - off, 0, &sx, &sy, NULL);
				wproj(v, tv[k], R.c - off * 1.04, 0, &ex, &ey, NULL);
			} else {
				wproj(v, R.c + off, tv[k], 0, &sx, &sy, NULL);
				wproj(v, R.c + off * 1.04, tv[k], 0, &ex, &ey, NULL);
			}
			pr_line1((int)floor(sx + 0.5), (int)floor(sy + 0.5),
			    (int)floor(ex + 0.5), (int)floor(ey + 0.5), pr_ink(PR_AXIS));
			if (R.axis == AX_H)
				pr_text((int)sx - pr_textw(lab[k]) / 2, (int)sy + 6, lab[k],
				    pr_ink(PR_DIM));
			else
				pr_text((int)sx + 7, (int)sy - 5, lab[k], pr_ink(PR_DIM));
		}
	}
	{
		const char *an = R.axis == AX_H ? xname() : "y";
		if (R.axis == AX_H)
			wproj(v, S->hi, R.c, 0, &sx, &sy, NULL);
		else
			wproj(v, R.c, S->hi, 0, &sx, &sy, NULL);
		pr_text((int)sx + (R.axis == AX_H ? 6 : -3), (int)sy - (R.axis ==
		    AX_H ? 5 : 16), an, pr_ink(PR_TEXT));
	}
}

static void
revolve_frame(const struct view *v, const struct bnd *B,
    const struct rscene *S, double th, int region, const char *extra)
{
	pr_clear();
	solid_draw(v, B, th);
	if (region)
		region3(v, B, th, pr_ser(3, 2), th < 2 * M_PI - 1e-9);
	axes3(v, S);
	if (region)
		curve3(v, B, th, 3, th < 2 * M_PI - 1e-9);
	pr_clip(0, 0, VW - 1, VH - 1);
	pr_title(VW / 2, 8, S->title, VW - 20);
	pr_text(10, VH - 16, S->foot, pr_ink(PR_TEXT));
	if (extra)
		pr_text(VW - 10 - pr_textw(extra), VH - 16, extra, pr_ink(PR_DIM));
}

static void
revolve_cmd(int what)
{
	struct solid so;
	struct bnd *B;
	struct view v;
	struct rscene S;
	struct gif g;
	unsigned char pal[256][3];
	int np, nr, nfr = 0, k, K, fps;
	double secs = 0, az0 = R.axis == AX_H ? 0.4 : -0.35;

	if (!R.axis_given) {
		R.axis = AX_H;
		R.c = 0;
	}
	region_setup();
	if (what == VC_SURFACE)
		printf("surface of revolution: %s about %s\n", R.fl, axis_text());
	else if (R.ng)
		printf("solid of revolution: the region between %s and %s about %s\n",
		    R.fl, R.gl, axis_text());
	else
		printf("solid of revolution: the region between %s and %s about %s\n",
		    R.fl, R.axis == AX_H ? axis_text() : "the x-axis", axis_text());
	if (!R.axis_given)
		printf("  note: about the x-axis (say about y, about y = 2, about x "
		    "= -1 for another)\n");
	if (!R.ng && what != VC_SURFACE) {
		if (R.axis == AX_H && R.c != 0)
			printf("  note: the region between the curve and the axis; for "
			    "the region under it, say `%s %s and 0 about y = %s`\n",
			    cmdname[what], R.fl, pr_g4(R.c));
		if (R.axis == AX_V)
			printf("  note: the region between the curve and the x-axis "
			    "(with two curves, the region between them)\n");
	}
	default_bounds(R.ng);
	if (!R.ng && R.axis == AX_H) {
		double z[64];
		int nz = zeros_of(SUB(body_of(&R.f), KR(R.c)), R.a, R.b, z, 64);
		if (nz > 0)
			printf("  note: %s crosses %s at %s = %s%s: the parts on both "
			    "sides turn into the solid (radius |%s%s|)\n", R.fl,
			    axis_text(), xname(), pr_gpi(z[0]), nz > 1 ? ", ..." : "",
			    R.c == 0 ? "f" : "f - c", "");
	}
	solid_numbers(&so, 1, what);
	if (what != VC_REVOLVE)
		return;
	/* the picture */
	B = xmalloc(sizeof *B);
	boundary(B);
	{
		double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax =
		    -INFINITY, Rm = 0;
		int i;
		for (i = 0; i < B->n; i++) {
			if (isnan(B->y[i]))
				continue;
			xmin = fmin(xmin, B->x[i]);
			xmax = fmax(xmax, B->x[i]);
			ymin = fmin(ymin, B->y[i]);
			ymax = fmax(ymax, B->y[i]);
			Rm = fmax(Rm, R.axis == AX_H ? fabs(B->y[i] - R.c) :
			    fabs(B->x[i] - R.c));
		}
		if (!(Rm > 0))
			Rm = 1;
		S.Rm = Rm;
		if (R.axis == AX_H) {
			double m = (xmax - xmin) * 0.12 + Rm * 0.05;
			S.lo = xmin - m;
			S.hi = xmax + m;
			S.bb[0] = S.lo;
			S.bb[1] = S.hi;
			S.bb[2] = R.c - Rm * 1.15;
			S.bb[3] = R.c + Rm;
			S.bb[4] = -Rm;
			S.bb[5] = Rm;
		} else {
			double lo = fmin(ymin, 0), hi = fmax(ymax, 0), m = (hi - lo) * 0.12;
			S.lo = lo - m;
			S.hi = hi + m;
			S.bb[0] = R.c - Rm;
			S.bb[1] = R.c + Rm * 1.15;
			S.bb[2] = S.lo;
			S.bb[3] = S.hi;
			S.bb[4] = -Rm;
			S.bb[5] = Rm;
		}
	}
	if (R.ng)
		snprintf(S.title, sizeof S.title, "between %s and %s, %s in [%s, %s], "
		    "about %s", R.fl, R.gl, xname(), btxt(R.a), btxt(R.b),
		    axis_text());
	else
		snprintf(S.title, sizeof S.title, "%s, %s in [%s, %s], revolved about "
		    "%s", R.fl, xname(), btxt(R.a), btxt(R.b), axis_text());
	snprintf(S.foot, sizeof S.foot, "V = %s (%s)", val_text(so.V, so.exact),
	    R.axis == AX_V ? "shells" : gconst && !R.ng ? "disks" : "washers");
	VW = 600;
	VH = 500;
	memset(&v, 0, sizeof v);
	v.V.el = R.axis == AX_H ? 0.36 : 0.42;
	if (R.axis == AX_H) {
		/* from the narrow end: the curved surface faces the eye */
		double fa, fb2, ga, gb;
		int oa = eval_at(&R.f, R.a, &fa), ob = eval_at(&R.f, R.b, &fb2);
		eval_at(&Gf, R.a, &ga);
		eval_at(&Gf, R.b, &gb);
		if (oa && ob && fmax(fabs(fa - R.c), fabs(ga - R.c)) > fmax(fabs(fb2 -
		    R.c), fabs(gb - R.c)))
			az0 = -0.4;
	}
	view_fit(&v, S.bb, 24, 34, VW - 24, VH - 30, az0, R.rotate);
	if (!R.name[0])
		snprintf(R.name, sizeof R.name, "revolve");
	pr_begin(VW, VH);
	np = pr_palette(pal, &ramp0, &nr);
	if (nr < NRAMP * RAMPN)
		fail("revolve: the palette has no room");
	ramps(pal);
	gif_begin(&g, VW, VH, (const unsigned char (*)[3])pal, np);
	fps = R.fps ? R.fps : 12;
	K = R.frames ? R.frames : 36;
	light_turn = 0;
	if (R.still) {
		revolve_frame(&v, B, &S, 2 * M_PI, 0, NULL);
		gif_frame(&g, pr_pixels(), 0);
		nfr = 1;
	} else {
		/* the region sweeps around, leaving the solid */
		for (k = 0; k <= K; k++) {
			double th = 2 * M_PI * k / K;
			char ang[40];
			int d = k == 0 ? 80 : k == K ? 150 : delay_cs(k, fps);
			snprintf(ang, sizeof ang, "angle %d deg", (int)floor(th * 180 /
			    M_PI + 0.5));
			revolve_frame(&v, B, &S, th, 1, ang);
			gif_frame(&g, pr_pixels(), d);
			nfr++;
			secs += d / 100.0;
		}
		if (R.rotate) {
			for (k = 1; k <= K; k++) {
				int d = k == K ? 150 : delay_cs(k, fps);
				v.V.az = az0 + 2 * M_PI * k / K;
				light_turn = 2 * M_PI * k / K;
				revolve_frame(&v, B, &S, 2 * M_PI, 0, "turning");
				gif_frame(&g, pr_pixels(), d);
				nfr++;
				secs += d / 100.0;
			}
		}
	}
	free(B);
	pr_end();
	if (nfr > 1)
		printf("  the region sweeps from 0 to 2pi in %d steps%s\n", K,
		    R.rotate ? ", then the solid turns once" : "");
	gif_finish(&g, VW, VH, nfr, secs);
}

/* ---- arc length, centroid ---- */

static void
arclength_cmd(void)
{
	nid d = deriv_of(&R.f), e;
	struct integ I;

	printf("arc length of %s\n", R.fl);
	if (R.ng)
		fail("arclength: one curve at a time");
	default_bounds(0);
	if (!d)
		fail("arclength: no derivative of %s", R.fl);
	d = mk_cast(TY_REAL, d);
	e = mk_un(O_SQRT, ADD(KR(1), MUL(d, d)));
	printf("  L = ∫ sqrt(1 + (%s)^2) d%s from %s to %s\n", node_str(d),
	    xname(), btxt(R.a), btxt(R.b));
	integrate(e, R.a, R.b, &I, 1);
	print_integ("  ", &I, R.a, R.b, 1);
	{
		double x0 = R.a, x1 = R.b, y0, y1;
		if (eval_at(&R.f, x0, &y0) && eval_at(&R.f, x1, &y1))
			printf("  (the chord from end to end: %s; the arc is never "
			    "shorter)\n", pr_g4(hypot(x1 - x0, y1 - y0)));
	}
	printf("  arc length: %s\n", val_text(ival(&I), I.exact));
}

static void
centroid_cmd(void)
{
	struct solid so;

	R.axis = AX_V;	/* the region between F and the x-axis */
	R.c = 0;
	region_setup();
	if (R.ng)
		printf("centroid of the region between %s and %s\n", R.fl, R.gl);
	else
		printf("centroid of the region between %s and the x-axis\n", R.fl);
	default_bounds(R.ng);
	memset(&so, 0, sizeof so);
	region_centroid(&so, 1, 0);
	if (so.A > 0)
		printf("  Pappus: turned about the x-axis it sweeps 2pi*|ybar|*A = %s;"
		    " about the y-axis 2pi*|xbar|*A = %s (when the region stays on "
		    "one side)\n", pr_g4(2 * M_PI * fabs(so.yc) * so.A),
		    pr_g4(2 * M_PI * fabs(so.xc) * so.A));
}

/* ---- the command ---- */

int
viz_command(struct P *p)
{
	static const char *const words[] = {"area", "revolve", "volume",
		"surfacearea", "arclength", "centroid", NULL};
	char w[32], rest[64];
	uint32_t pos = cmdp_pos(p);
	jmp_buf saved;
	int k;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (k = 0; words[k] && strcmp(words[k], w); k++)
		;
	if (!words[k] || func_find(sym_cstr(w))) {
		cmdp_seek(p, pos);
		return 0;
	}
	memset(&R, 0, sizeof R);
	R.cmd = k;
	if (!cmdp_rest(p, R.text, sizeof R.text, NULL))
		fail("%s what? e.g. %s", w, k == VC_AREA ? "area x^2 from 0 to 4" :
		    k == VC_ARCLEN ? "arclength x^2 from 0 to 1" : k == VC_CENTROID ?
		    "centroid x^2 and x" : "revolve sqrt(x) from 0 to 4 about x");
	/* "area(x) = ..." is a definition */
	if (R.text[0] == '(') {
		const char *s = R.text;
		int d = 0;
		for (; *s; s++)
			if (*s == '(')
				d++;
			else if (*s == ')' && --d == 0)
				break;
		if (*s)
			for (s++; isspace((unsigned char)*s); s++)
				;
		if (*s == '=' && s[1] != '=') {
			cmdp_seek(p, pos);
			return 0;
		}
	}
	(void)rest;
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp)) {
		pr_end();
		memcpy(fail_jmp, saved, sizeof saved);
		longjmp(fail_jmp, 1);
	}
	parse_req();
	switch (k) {
	case VC_AREA:
		if (R.axis_given || R.method || R.still || R.rotate)
			fail("area: about, disk, washer, shell, still and rotate are "
			    "revolve's");
		area_cmd();
		break;
	case VC_REVOLVE:
	case VC_VOLUME:
	case VC_SURFACE:
		if (R.rkind || R.accum)
			fail("%s: riemann and accumulate are area's", w);
		revolve_cmd(k);
		break;
	case VC_ARCLEN:
		arclength_cmd();
		break;
	default:
		centroid_cmd();
		break;
	}
	memcpy(fail_jmp, saved, sizeof saved);
	return 1;
}
