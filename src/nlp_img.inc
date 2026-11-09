/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * nlp_img.inc (included by nlp.c): a request to run a shader, in
 * English, with what its image inputs should be.  The line before a code
 * block is rewritten as an animate exec statement with anim.c's options:
 *
 *   "run this on my images, for input 1 use pure black"
 *       -> animate exec of this images input 1 black
 *   "use a red image for the second input and show it at 128x128"
 *       -> animate exec of this input 2 red size 128x128
 *   "input 2 is a checkerboard, with an RG8 output"
 *       -> animate exec of this input 2 checker output RG8
 *   "use white for all inputs" -> animate exec of this inputs white
 *
 * Clauses (split at commas, semicolons and "and") pair the inputs they
 * name (input 2, the second input, all inputs) with what they say to
 * fill them with (a colour from tex.c's table, #rrggbb, noise, gradient,
 * checkerboard, a format); a format next to "output" is the output's.
 * Misspelt words are corrected towards this file's words, as the rest
 * of the translator does.
 */

static const char *const si_words[] = {
	"input", "inputs", "texture", "textures", "output", "outputs", "image",
	"images", "picture", "pictures", "photo", "photos", "gallery",
	"attached", "attachments", "checkerboard", "checker", "checkered",
	"chessboard", "gradient", "noise", "random", "static", "black",
	"white", "red", "green", "blue", "yellow", "cyan", "magenta",
	"purple", "orange", "gray", "grey", "pink", "brown", "violet",
	"maroon", "navy", "teal", "olive", "silver", "gold", "indigo",
	"crimson", "coral", "salmon", "turquoise", "beige", "khaki",
	"lavender", "chocolate", "transparent", "lime", "aqua", "fuchsia",
	"nearest", "linear", "bilinear", "trilinear", "clamp", "repeat",
	"mirror", "mirrored", "border", "sampler", "sampling", "filtering",
	"first", "second", "third", "fourth", "fifth", "sixth", "seventh",
	"eighth", "every", "both", "each", "resolution", "size", "pixels",
	"lightgray", "darkgray", "skyblue", "hotpink", "tomato", "orchid",
	"amber", "execute", "visualize", "visualise", "render", "shader",
	"illustrate", NULL
};

struct si_tok {
	char w[96];
	int sep;		/* a clause ends before this token */
	int used;
};

static int
si_ordinal(const char *w)
{
	static const char *const o[] = {"first", "second", "third", "fourth",
	    "fifth", "sixth", "seventh", "eighth", NULL};
	int k;

	for (k = 0; o[k]; k++)
		if (!strcmp(w, o[k]))
			return k + 1;
	if (isdigit((unsigned char)w[0]) && (strstr(w, "st") || strstr(w, "nd")
	    || strstr(w, "rd") || strstr(w, "th")) && atoi(w) > 0 &&
	    atoi(w) <= IMG_MAXIN)
		return atoi(w);
	return 0;
}

static int
si_is(const char *w, const char *const *l)
{
	return in_list(l, w);
}

/* a format word (not one that reads as English: "r", "float") */
static int
si_format(const char *w)
{
	int f;

	if (strlen(w) < 2 || (!isdigit((unsigned char)w[strlen(w) - 1]) &&
	    !strchr(w, '_') && w[strlen(w) - 1] != 'f' &&
	    w[strlen(w) - 1] != 'i' && strncmp(w, "srgb", 4)))
		return -1;
	if (!strcmp(w, "half") || !strcmp(w, "gray") || !strcmp(w, "grey"))
		return -1;
	f = tex_fmt_find(w);
	return f;
}

/* the line asks to run the shader (or says what its images are) */
static int
si_wanted(struct si_tok *t, int n, int animate)
{
	static const char *const verbs[] = {"run", "execute", "exec", "apply",
	    "show", "visualize", "visualise", "render", "draw", "illustrate",
	    "animate", "simulate", "preview", "demo", "demonstrate", "try",
	    "test", "evaluate", "shade", NULL};
	static const char *const english[] = {"use", "is", "are", "be", "make",
	    "pure", "solid", "plain", "first", "second", "third", "fourth",
	    "all", "every", "each", "both", "my", "these", "those", "pictures",
	    "images", "photos", "picture", "image", "attached", "as", "should",
	    "fill", "give", "pass", "feed", "take", "set", "colour", "color",
	    "colored", "coloured", NULL};
	static const char *const no[] = {"simplify", "simplified", "optimize",
	    "optimise", "optimized", "optimised", "peephole", "cheaper",
	    "prove", "simplification", "optimization", "superoptimize", NULL};
	int i, verb = 0, eng = 0;

	for (i = 0; i < n; i++) {
		if (si_is(t[i].w, no))
			return 0;
		verb |= si_is(t[i].w, verbs);
		eng |= si_is(t[i].w, english) || si_ordinal(t[i].w) > 0;
	}
	return animate ? eng : verb || eng;
}

/* "this input", "input 2", "the second input", "all inputs": the slot
 * (1..), 0 for all, -1 none; *at moves past it */
static int
si_ref(struct si_tok *t, int n, int *at)
{
	static const char *const in[] = {"input", "texture", "slot",
	    "sampler", "binding", "image", "picture", "tex", NULL};
	static const char *const ins[] = {"inputs", "textures", "slots",
	    "samplers", "bindings", "images", "pictures", NULL};
	int i = *at, k;

	if (si_is(t[i].w, in) && i + 1 < n) {
		const char *w = t[i + 1].w;
		int j = i + 1;
		if ((!strcmp(w, "number") || !strcmp(w, "no") || !strcmp(w, "#"))
		    && j + 1 < n)
			w = t[++j].w;
		if (w[0] == '#')
			w++;
		if (isdigit((unsigned char)w[0]) && strspn(w, "0123456789") ==
		    strlen(w) && (k = atoi(w)) >= 1 && k <= IMG_MAXIN &&
		    strcmp(t[i].w, "image") && strcmp(t[i].w, "picture")) {
			*at = j + 1;
			return k;
		}
		if (isdigit((unsigned char)w[0]) && strspn(w, "0123456789") ==
		    strlen(w) && (k = atoi(w)) >= 1 && k <= IMG_MAXIN &&
		    (i == 0 || strcmp(t[i - 1].w, "this"))) {
			*at = j + 1;	/* "image 2" */
			return k;
		}
	}
	if ((k = si_ordinal(t[i].w)) > 0 && i + 1 < n && (si_is(t[i + 1].w,
	    in) || !strcmp(t[i + 1].w, "one"))) {
		*at = i + 2;
		return k;
	}
	if ((si_is(t[i].w, (const char *const[]){"all", "every", "each",
	    "both", NULL})) && i + 1 < n) {
		int j = i + 1;
		while (j < n && (!strcmp(t[j].w, "of") || !strcmp(t[j].w, "the")))
			j++;
		if (j < n && (si_is(t[j].w, ins) || si_is(t[j].w, in))) {
			*at = j + 1;
			return 0;
		}
	}
	return -1;
}

/* what to fill an input with, at t[i]: 1 and spec's words added */
static int
si_spec(struct si_tok *t, int i, struct sbuf *spec)
{
	unsigned char c[4];
	const char *w = t[i].w;
	int f;

	if (!strcmp(w, "noise") || !strcmp(w, "random") || !strcmp(w, "static"))
		sb_puts(spec, " noise");
	else if (!strcmp(w, "gradient") || !strcmp(w, "ramp"))
		sb_puts(spec, " gradient");
	else if (!strncmp(w, "checker", 7) || !strcmp(w, "chessboard"))
		sb_puts(spec, " checker");
	else if ((f = si_format(w)) >= 0)
		sb_printf(spec, " %s", tex_fmts[f].name);
	else if (strcmp(w, "tan") && strcmp(w, "mint") && strcmp(w, "plum") &&
	    img_color(w, c)) {
		if (w[0] == '#' || !strncmp(w, "rgb", 3))
			sb_printf(spec, " #%02x%02x%02x%02x", c[0], c[1], c[2], c[3]);
		else
			sb_printf(spec, " %s", w);
	} else
		return 0;
	t[i].used = 1;
	return 1;
}

static void
si_tokens(const char *s, size_t n, struct si_tok *t, int *nt, int max)
{
	size_t i = 0;
	int sep = 0;

	*nt = 0;
	while (i < n && *nt < max) {
		unsigned char c = (unsigned char)s[i];
		if (strchr(",;!?", c) || (c == '.' && (i + 1 >= n ||
		    isspace((unsigned char)s[i + 1])))) {
			sep = 1;
			i++;
			continue;
		}
		if (isspace(c) || c == ':' || c == '"' || c == '\'' || c == '(' ||
		    c == ')') {
			i++;
			continue;
		}
		{
			struct si_tok *k = &t[(*nt)++];
			size_t m = 0;
			memset(k, 0, sizeof *k);
			k->sep = sep;
			sep = 0;
			if (!strncasecmp(s + i, "rgb", 3) && memchr(s + i, '(',
			    n - i) && memchr(s + i, ')', n - i) &&
			    (const char *)memchr(s + i, '(', n - i) < s + i + 5) {
				/* rgb(1, 0, 0) as one word */
				while (i < n && s[i] != ')' && m + 2 < sizeof k->w)
					if (s[i] == ' ')
						i++;
					else
						k->w[m++] = (char)tolower((unsigned char)s[i++]);
				if (i < n)
					k->w[m++] = s[i++];
				k->w[m] = 0;
				continue;
			}
			if (c == '=') {	/* "%x = 2" stays together below */
				k->w[0] = '=';
				k->w[1] = 0;
				i++;
				continue;
			}
			while (i < n && !isspace((unsigned char)s[i]) &&
			    !strchr(",;!?:=\"()", s[i]) && !(s[i] == '.' &&
			    (i + 1 >= n || isspace((unsigned char)s[i + 1]))) &&
			    m + 1 < sizeof k->w)
				k->w[m++] = (char)tolower((unsigned char)s[i++]);
			k->w[m] = 0;
			if (m == 0)
				i++, (*nt)--;
		}
	}
}

/* misspelt words towards si_words (and the colour names) */
static void
si_spell(struct si_tok *t, int n)
{
	int i, k;

	for (i = 0; i < n; i++) {
		const char *w = t[i].w, *best = NULL;
		int lim = spell_limit(strlen(w)), bd = lim + 1, d;
		unsigned char c[4];
		if (lim == 0 || si_is(w, si_words) || in_dict(w) || img_color(w, c) ||
		    !isalpha((unsigned char)w[0]))
			continue;
		for (k = 0; si_words[k]; k++)
			if ((d = osa(w, si_words[k], lim)) < bd) {
				bd = d;
				best = si_words[k];
			}
		if (best != NULL) {
			printf("  note: read \"%s\" as \"%s\"\n", w, best);
			snprintf(t[i].w, sizeof t[i].w, "%s", best);
		}
	}
}

/*
 * One line of an exec request: the animate statement in out, or 0 when
 * the line is not one
 */
static int
si_line(const char *s, size_t len, struct sbuf *out)
{
	struct si_tok t[160];
	int n, i, j, animate, k;
	struct sbuf opt = {0}, keep = {0};
	static const char *const units[] = {"fps", "seconds", "second", "s",
	    "lanes", "lane", NULL};
	static const char *const passw[] = {"nofile", "quad", "warp", "random",
	    NULL};

	si_tokens(s, len, t, &n, 160);
	if (n == 0)
		return 0;
	animate = !strcmp(t[0].w, "animate");
	if (!si_wanted(t, n, animate))
		return 0;
	si_spell(t, n);
	/* "black and white checkerboard": one pattern */
	for (i = 0; i + 2 < n; i++)
		if (!strcmp(t[i].w, "black") && !strcmp(t[i + 1].w, "and") &&
		    !strcmp(t[i + 2].w, "white")) {
			snprintf(t[i].w, sizeof t[i].w, "checker");
			t[i + 1].w[0] = t[i + 2].w[0] = 0;
			t[i + 1].used = t[i + 2].used = 1;
		}
	/* clauses */
	for (i = 0; i < n; i = j) {
		int refs[IMG_MAXIN + 1], nref = 0, outw = 0, gal = 0, nspec = 0;
		struct sbuf specs[8];
		memset(specs, 0, sizeof specs);
		for (j = i; j < n; j++) {
			if (j > i && (t[j].sep || !strcmp(t[j].w, "and") ||
			    !strcmp(t[j].w, "then") || !strcmp(t[j].w, "while") ||
			    !strcmp(t[j].w, "but") || !strcmp(t[j].w, "plus") ||
			    !strcmp(t[j].w, "with")))
				break;
		}
		for (k = i; k < j; k++)	/* "an RG8 output": the format is its */
			if (!strcmp(t[k].w, "output") || !strcmp(t[k].w, "target") ||
			    !strcmp(t[k].w, "framebuffer") ||
			    !strcmp(t[k].w, "attachment"))
				outw = 1;
		for (k = i; k < j; k++) {
			int at = k, r;
			const char *w = t[k].w;
			if (!w[0])
				continue;
			if ((r = si_ref(t, j, &at)) >= 0) {
				int q;
				for (q = k; q < at; q++)
					t[q].used = 1;
				if (nref <= IMG_MAXIN)
					refs[nref++] = r;
				k = at - 1;
				continue;
			}
			if (!strcmp(w, "output") || !strcmp(w, "target") ||
			    !strcmp(w, "framebuffer") || !strcmp(w, "attachment")) {
				t[k].used = 1;
				continue;
			}
			if ((si_is(w, (const char *const[]){"images", "pictures",
			    "photos", "pics", "gallery", "attachments", "photographs",
			    NULL})) || (!strcmp(w, "attached") && k + 1 < j &&
			    (!strcmp(t[k + 1].w, "image") ||
			    !strcmp(t[k + 1].w, "picture")))) {
				gal = 1;
				t[k].used = 1;
				continue;
			}
			/* a size: 128x128, 128 by 128 */
			{
				int a, b;
				char x;
				if (sscanf(w, "%d%c%d", &a, &x, &b) == 3 && (x == 'x' ||
				    x == '*') && a > 0 && b > 0) {
					sb_printf(&opt, " size %dx%d", a, b);
					t[k].used = 1;
					continue;
				}
				if (isdigit((unsigned char)w[0]) && k + 2 < j &&
				    (!strcmp(t[k + 1].w, "by") || !strcmp(t[k + 1].w, "x")) &&
				    isdigit((unsigned char)t[k + 2].w[0])) {
					sb_printf(&opt, " size %dx%d", atoi(w), atoi(t[k + 2].w));
					t[k].used = t[k + 1].used = t[k + 2].used = 1;
					k += 2;
					continue;
				}
			}
			if (si_is(w, (const char *const[]){"nearest", "bilinear",
			    "trilinear", "clamp", "repeat", "mirror", "mirrored",
			    "wrap", NULL}) || (!strcmp(w, "linear") && k + 1 < j &&
			    (!strcmp(t[k + 1].w, "filtering") || !strcmp(t[k + 1].w,
			    "sampling") || !strcmp(t[k + 1].w, "filter")))) {
				sb_printf(&opt, " sampler %s", w);
				t[k].used = 1;
				continue;
			}
			if (!strcmp(w, "file") && k + 1 < j) {
				if (nspec < 8)
					sb_printf(&specs[nspec++], " file %s", t[k + 1].w);
				t[k].used = t[k + 1].used = 1;
				k++;
				continue;
			}
			if (nspec < 8) {
				struct sbuf one = {0};
				if (si_spec(t, k, &one)) {
					if (si_format(w) >= 0 && outw)
						sb_printf(&opt, " output %s",
						    tex_fmts[si_format(w)].name);
					else if (si_format(w) >= 0 && nspec > 0 &&
					    si_format(specs[nspec - 1].p + 1) < 0)
						/* "a red R8 image": one input's words */
						sb_puts(&specs[nspec - 1], one.p);
					else
						sb_puts(&specs[nspec++], one.p);
				}
				free(one.p);
			}
		}
		if (gal)
			sb_puts(&opt, " images");
		/* a lone format with no input named: the output's */
		for (k = 0; k < nspec; k++) {
			int f = si_format(specs[k].p + 1);
			if (f >= 0 && !strchr(specs[k].p + 1, ' ') && nref == 0) {
				sb_printf(&opt, " output %s", tex_fmts[f].name);
				specs[k].n = 0;
			}
		}
		{
			const char *sp[8];
			int ns = 0;
			for (k = 0; k < nspec; k++)
				if (specs[k].n)
					sp[ns++] = specs[k].p;
			if (ns && nref == 0)
				refs[nref++] = 0;	/* "use white": every input */
			for (k = 0; k < nref && ns; k++) {
				int q;
				for (q = 0; q < ns; q++) {
					/* as many inputs as specs: in order; else each all */
					if (nref == ns && nref > 1 && q != k)
						continue;
					if (refs[k] == 0)
						sb_printf(&opt, " inputs%s", sp[q]);
					else
						sb_printf(&opt, " input %d%s", refs[k], sp[q]);
				}
			}
		}
		for (k = 0; k < 8; k++)
			free(specs[k].p);
	}
	/* animate's own words pass through: 30 fps, seed 7, to NAME, %x = 1 */
	for (i = animate; i < n; i++) {
		const char *w = t[i].w;
		if (t[i].used || !w[0])
			continue;
		if (animate && i <= 3 && (!strcmp(w, "exec") ||
		    !strcmp(w, "execute") || !strcmp(w, "of") || !strcmp(w, "this")))
			continue;
		if (isdigit((unsigned char)w[0]) && i + 1 < n && si_is(t[i + 1].w,
		    units)) {
			sb_printf(&keep, " %s %s", w, t[i + 1].w);
			i++;
		} else if ((si_is(w, units) || !strcmp(w, "seed") ||
		    !strcmp(w, "fps")) && i + 1 < n &&
		    isdigit((unsigned char)t[i + 1].w[0])) {
			sb_printf(&keep, " %s %s", w, t[i + 1].w);
			i++;
		} else if (w[0] == '%' && i + 2 < n && !strcmp(t[i + 1].w, "=")) {
			sb_printf(&keep, " %s = %s", w, t[i + 2].w);
			i += 2;
		} else if (!strcmp(w, "to") && i + 2 == n)
			sb_printf(&keep, " to %s", t[++i].w);
		else if (si_is(w, passw) || model_find(sym_cstr(w)) != NULL)
			sb_printf(&keep, " %s", w);
	}
	sb_printf(out, "animate exec of this%s%s", keep.p ? keep.p : "",
	    opt.p ? opt.p : "");
	free(opt.p);
	free(keep.p);
	return 1;
}

/*
 * The piece before a SPIR-V block: its exec request (the last line with
 * words) rewritten as an animate statement
 */
static void
shader_piece(struct sbuf *piece)
{
	char *p = piece->p, *ls = NULL, *le = NULL, *q;
	struct sbuf st = {0}, out = {0};

	if (p == NULL)
		return;
	for (q = p; *q; ) {
		char *e = strchr(q, '\n'), *a;
		if (e == NULL)
			e = q + strlen(q);
		for (a = q; a < e && isspace((unsigned char)*a); a++)
			;
		if (a < e && *a != '\x1e' && *a != '#') {
			ls = a;
			le = e;
		}
		q = *e ? e + 1 : e;
	}
	if (ls == NULL)
		return;
	if (!si_line(ls, (size_t)(le - ls), &st)) {
		free(st.p);
		return;
	}
	printf("  note: read as: %s\n", st.p);
	sb_printf(&out, "%.*s%s%s", (int)(ls - p), p, st.p, le);
	free(st.p);
	free(piece->p);
	*piece = out;
}
