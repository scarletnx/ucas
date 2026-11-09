/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Other bots as users (botuse.h): the allowlist, the rate limit, the
 * reply-chain cap, the "explain AI usage" phrases and the help text.
 */
#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "ucas.h"
#include "botuse.h"

struct bu_conf bu_conf;

static long
envnum(const char *name, long def, long lo, long hi)
{
	const char *s = getenv(name);
	char *e;
	long v;

	if (s == NULL || *s == 0)
		return def;
	v = strtol(s, &e, 10);
	if (*e != 0 || v < lo || v > hi) {
		fprintf(stderr, "ucas-discord: %s=%s: not a number from %ld to "
		    "%ld; using %ld\n", name, s, lo, hi, def);
		return def;
	}
	return v;
}

/*
 * "deny" (or "none", or empty): no bot is served; otherwise the user ids
 * of the bots that are, separated by commas or blanks.  There is no
 * "every bot" setting.  Returns 0 on a malformed list (then none).
 */
int
bu_conf_parse(struct bu_conf *c, const char *bots)
{
	const char *p = bots;

	c->nallow = 0;
	while (*p == ' ' || *p == '\t')
		p++;
	if (*p == 0 || !strcasecmp(p, "deny") || !strcasecmp(p, "none") ||
	    !strcasecmp(p, "off"))
		return 1;
	while (*p) {
		char *e;
		unsigned long long v;
		if (*p == ',' || *p == ' ' || *p == '\t') {
			p++;
			continue;
		}
		if (!isdigit((unsigned char)*p))
			goto bad;
		v = strtoull(p, &e, 10);
		if (v == 0 || (*e && *e != ',' && *e != ' ' && *e != '\t'))
			goto bad;
		if (c->nallow == BU_MAXALLOW)
			goto bad;
		c->allow[c->nallow++] = (uint64_t)v;
		p = e;
	}
	return 1;
bad:
	c->nallow = 0;
	return 0;
}

int
bu_conf_env(void)
{
	const char *b = getenv("UCAS_DISCORD_BOTS");
	int ok;

	bu_conf.rate = (int)envnum("UCAS_DISCORD_BOT_RATE", 6, 1, 600);
	bu_conf.burst = (int)envnum("UCAS_DISCORD_BOT_BURST", 3, 1, 100);
	bu_conf.chain = (int)envnum("UCAS_DISCORD_BOT_CHAIN", 5, 1, 1000);
	ok = bu_conf_parse(&bu_conf, b ? b : BU_DEFAULT_BOTS);
	if (!ok)
		fprintf(stderr, "ucas-discord: UCAS_DISCORD_BOTS=%s: not deny or "
		    "a list of user ids; no bot is served\n", b);
	bu_reset();
	return ok;
}

int
bu_allowed(uint64_t author)
{
	int i;

	for (i = 0; i < bu_conf.nallow; i++)
		if (bu_conf.allow[i] == author)
			return 1;
	return 0;
}

/* ---- the rate limit ---- */

#define BU_SLOTS 256

static struct bucket {
	uint64_t author;
	int64_t last;		/* ms */
	double tokens;
	int told;
} buckets[BU_SLOTS];
static int nbuckets;

static struct chain {
	uint64_t chan, author;
	int depth, told;
	uint64_t tick;
} chains[BU_SLOTS];
static int nchains;
static uint64_t chaintick;

void
bu_reset(void)
{
	nbuckets = nchains = 0;
}

int
bu_rate(uint64_t author, int64_t now_ms, int *notify)
{
	struct bucket *b = NULL;
	double per_ms = bu_conf.rate / 60000.0, wait;
	int i;

	*notify = 0;
	for (i = 0; i < nbuckets; i++)
		if (buckets[i].author == author)
			b = &buckets[i];
	if (b == NULL) {
		if (nbuckets < BU_SLOTS)
			b = &buckets[nbuckets++];
		else {	/* the longest idle one goes */
			b = &buckets[0];
			for (i = 1; i < nbuckets; i++)
				if (buckets[i].last < b->last)
					b = &buckets[i];
		}
		b->author = author;
		b->last = now_ms;
		b->tokens = bu_conf.burst;
		b->told = 0;
	}
	if (now_ms > b->last) {
		b->tokens += (double)(now_ms - b->last) * per_ms;
		if (b->tokens > bu_conf.burst)
			b->tokens = bu_conf.burst;
		b->last = now_ms;
	}
	if (b->tokens >= 1.0) {
		b->tokens -= 1.0;
		b->told = 0;
		return 0;
	}
	*notify = !b->told;
	b->told = 1;
	wait = ceil((1.0 - b->tokens) / per_ms / 1000.0);
	return wait < 1 ? 1 : (int)wait;
}

/* ---- reply chains ---- */

int
bu_chain(uint64_t chan, uint64_t author, int reply_to_self, int *notify)
{
	struct chain *c = NULL;
	int i;

	*notify = 0;
	for (i = 0; i < nchains; i++)
		if (chains[i].chan == chan && chains[i].author == author)
			c = &chains[i];
	if (c == NULL) {
		if (nchains < BU_SLOTS)
			c = &chains[nchains++];
		else {
			c = &chains[0];
			for (i = 1; i < nchains; i++)
				if (chains[i].tick < c->tick)
					c = &chains[i];
		}
		c->chan = chan;
		c->author = author;
		c->depth = 0;
		c->told = 0;
	}
	c->tick = ++chaintick;
	if (!reply_to_self) {
		c->depth = 0;
		c->told = 0;
	}
	if (c->depth >= bu_conf.chain) {
		*notify = !c->told;
		c->told = 1;
		return 0;
	}
	c->depth++;
	return 1;
}

void
bu_human(uint64_t chan)
{
	int i;

	for (i = nchains; i-- > 0;)
		if (chains[i].chan == chan)
			chains[i] = chains[--nchains];
}

/* ---- "explain AI usage" ---- */

/* the words of s: lowercase letters and digits; "a.i." is "ai" */
static int
words(const char *s, char w[][24], int max)
{
	int n = 0;
	size_t l = 0;

	for (;; s++) {
		int c = (unsigned char)*s;
		if (isalnum(c) && c < 0x80) {
			if (l + 1 < 24)
				w[n][l++] = (char)tolower(c);
			continue;
		}
		if (c == '\'' && l > 0)
			continue;	/* "AI's" */
		if (l > 0) {
			w[n][l] = 0;
			/* "a i" from "a.i." */
			if (n > 0 && l == 1 && w[n][0] == 'i' &&
			    !strcmp(w[n - 1], "a"))
				strcpy(w[n - 1], "ai");
			else if (n + 1 < max)
				n++;
			else
				return -1;	/* too long to be the phrase */
			l = 0;
		}
		if (c == 0)
			break;
	}
	return n;
}

/* the optimal string alignment distance (a swap of neighbours is one) */
static int
dist(const char *a, const char *b)
{
	int la = (int)strlen(a), lb = (int)strlen(b), i, j;
	int d[24][24];

	for (i = 0; i <= la; i++)
		d[i][0] = i;
	for (j = 0; j <= lb; j++)
		d[0][j] = j;
	for (i = 1; i <= la; i++)
		for (j = 1; j <= lb; j++) {
			int c = a[i - 1] != b[j - 1], v = d[i - 1][j - 1] + c;
			if (d[i - 1][j] + 1 < v)
				v = d[i - 1][j] + 1;
			if (d[i][j - 1] + 1 < v)
				v = d[i][j - 1] + 1;
			if (i > 1 && j > 1 && a[i - 1] == b[j - 2] &&
			    a[i - 2] == b[j - 1] && d[i - 2][j - 2] + 1 < v)
				v = d[i - 2][j - 2] + 1;
			d[i][j] = v;
		}
	return d[la][lb];
}

static int
word_ok(const char *got, const char *want)
{
	size_t n = strlen(want);

	if (!strcmp(got, want))
		return 1;
	if (n < 4)		/* ai, use, bot, how, for, you: exact */
		return 0;
	return dist(got, want) <= (n < 8 ? 1 : 2);
}

int
bu_is_ai_usage(const char *text)
{
	static const char *const phrases[] = {
		"explain ai usage", "explain ai use", "explain ai", "explain ais",
		"explain bot usage", "explain bot use", "explain bots usage",
		"explain usage for ai", "explain usage for bots",
		"how do bots use you", "how do ai use you", "how do ais use you",
		"how can bots use you", "how can ai use you",
		"how should bots use you", "how should ai use you",
		"ai usage", "ai use", "ais usage", "usage for ai", "usage for ais",
		"usage for bots", "usage for bot", "help for bots", "help for bot",
		"help for ai", "help for ais", "bot usage", "bots usage",
		"bot use", "ai help", "bot help", "help ai", "help bots",
		NULL
	};
	static const char *const fill[] = {"please", "pls", "plz", "wimsy",
	    "thanks", "thx", NULL};
	char w[10][24], p[10][24];
	int n, k, i, j, f, start = 0;

	if ((n = words(text, w, 10)) <= 0)
		return 0;
	/* "please" and the like at either end */
	for (f = 0; fill[f]; f++)
		if (n > start && !strcmp(w[n - 1], fill[f])) {
			n--;
			f = -1;
		}
	for (f = 0; fill[f] && start < n; f++)
		if (!strcmp(w[start], fill[f])) {
			start++;
			f = -1;
		}
	for (k = 0; phrases[k]; k++) {
		int m = words(phrases[k], p, 10), ok = m == n - start;
		for (i = 0, j = start; ok && i < m; i++, j++)
			ok = word_ok(w[j], p[i]);
		if (ok)
			return 1;
	}
	return 0;
}

int
bu_is_notice(const char *content)
{
	static const char *const pre[] = {BU_RATE_NOTICE, BU_CHAIN_NOTICE,
	    BU_HELP_HEAD, "mention me with a question or ucas input",
	    "**Wimsy for bots", NULL};
	int k;

	if (content == NULL)
		return 0;
	for (k = 0; pre[k]; k++)
		if (!strncmp(content, pre[k], strlen(pre[k])))
			return 1;
	return 0;
}

/* ---- the help text ---- */

static const char help_full[] =
#include "ai_usage.inc"
    ;
static const char help_short[] =
#include "ai_usage_short.inc"
    ;

char *
bu_help(int full, uint64_t self, const char *name, int cpu, int wall)
{
	const char *s = full ? help_full : help_short, *e;
	struct sbuf b = {0};

	sb_puts(&b, "");
	for (; *s; s = e) {
		const char *c = strchr(s, '}');
		e = s + 1;
		if (*s != '{' || c == NULL || c - s > 12) {
			sb_putc(&b, *s);
			continue;
		}
		e = c + 1;
		if (!strncmp(s, "{MENTION}", 9))
			sb_printf(&b, "<@%" PRIu64 ">", self);
		else if (!strncmp(s, "{ID}", 4))
			sb_printf(&b, "%" PRIu64, self);
		else if (!strncmp(s, "{NAME}", 6))
			sb_puts(&b, name);
		else if (!strncmp(s, "{RATE}", 6))
			sb_printf(&b, "%d", bu_conf.rate);
		else if (!strncmp(s, "{BURST}", 7))
			sb_printf(&b, "%d", bu_conf.burst);
		else if (!strncmp(s, "{CHAIN}", 7))
			sb_printf(&b, "%d", bu_conf.chain);
		else if (!strncmp(s, "{CPU}", 5))
			sb_printf(&b, "%d", cpu);
		else if (!strncmp(s, "{WALL}", 6))
			sb_printf(&b, "%d", wall);
		else {
			sb_putc(&b, *s);
			e = s + 1;
		}
	}
	return b.p;
}
