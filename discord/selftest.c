/*
 * ucas-discord --selftest: the `$` prefix, code fences, message
 * splitting and the JSON codecs used for events and replies.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "discord.h"

#include "text.h"

int dc_selftest(void);

static int nfail, ncheck;

#define CHECK(c) check((c), #c, __LINE__)

static void
check(int ok, const char *what, int line)
{
	ncheck++;
	if (!ok) {
		nfail++;
		printf("FAIL selftest.c:%d: %s\n", line, what);
	}
}

static void
prefix(const char *msg, int kind, const char *code, int echo)
{
	struct sbuf b = {0};
	int k, e;

	k = dc_command(msg, &b, &e);
	ncheck++;
	if (k != kind || (kind == DC_EVAL &&
	    (strcmp(sb_str(&b), code) != 0 || e != echo))) {
		nfail++;
		printf("FAIL prefix %s: kind %d code [%s] echo %d\n", msg, k,
		    sb_str(&b), e);
	}
	free(b.p);
}

static void
test_prefix(void)
{
	prefix("1 + 2", DC_IGNORE, NULL, 0);
	prefix(" $1+2", DC_IGNORE, NULL, 0);
	prefix("$", DC_IGNORE, NULL, 0);
	prefix("$   ", DC_IGNORE, NULL, 0);
	prefix("$1 + 2", DC_EVAL, "1 + 2", 0);
	prefix("$ range isqrt(x: u4)", DC_EVAL, "range isqrt(x: u4)", 0);
	prefix("$`show f`", DC_EVAL, "show f", 0);
	prefix("$reset", DC_RESET, NULL, 0);
	prefix("$ reset ", DC_RESET, NULL, 0);
	prefix("$resets", DC_EVAL, "resets", 0);
	prefix("$```\nf(x) = x\nshow f\n```", DC_EVAL, "f(x) = x\nshow f", 1);
	prefix("$```ucas\nf(x) = x\n```", DC_EVAL, "f(x) = x", 0);
	prefix("$ ```c\r\nf = 1\r\ng = 2\r\n``` ", DC_EVAL, "f = 1\ng = 2", 1);
	prefix("$```range f```", DC_EVAL, "range f", 0);
	prefix("$```\n\n```", DC_IGNORE, NULL, 0);
	prefix("$f(x) = x\n\n  where y = 1", DC_EVAL,
	    "f(x) = x\n\n  where y = 1", 1);
	/* nothing in the command set is special to the bridge */
	prefix("$someday_a_new_command f", DC_EVAL,
	    "someday_a_new_command f", 0);
}

static void
test_format(void)
{
	struct dc_reply r;
	struct sbuf big = {0}, joined = {0};
	char *s;
	int i, j;

	dc_format(&r, NULL, "3\n", 2);
	CHECK(r.n == 1 && !strcmp(r.msg[0], "```\n3\n```") && !r.file);
	dc_reply_free(&r);

	dc_format(&r, NULL, "", 0);
	CHECK(r.n == 1 && !strcmp(r.msg[0], "(no output)"));
	dc_reply_free(&r);

	dc_format(&r, "timed out", "", 0);
	CHECK(r.n == 1 && !strcmp(r.msg[0], "timed out"));
	dc_reply_free(&r);

	dc_format(&r, "timed out", "partial\n", 8);
	CHECK(r.n == 1 && !strcmp(r.msg[0], "timed out\n```\npartial\n```"));
	dc_reply_free(&r);

	/* a fence in the output cannot close the block */
	dc_format(&r, NULL, "a```b", 5);
	CHECK(r.n == 1 && strstr(r.msg[0] + 3, "```b") == NULL);
	dc_reply_free(&r);

	/* exactly at the limit: one message */
	s = xmalloc(DC_LIMIT);
	memset(s, 'x', DC_LIMIT - 8);
	dc_format(&r, NULL, s, DC_LIMIT - 8);
	CHECK(r.n == 1 && strlen(r.msg[0]) == DC_LIMIT);
	dc_reply_free(&r);
	dc_format(&r, NULL, s, DC_LIMIT - 7);
	CHECK(r.n == 2);
	dc_reply_free(&r);
	free(s);

	/* 250 lines: split at line breaks, nothing lost */
	for (i = 0; i < 250; i++)
		sb_printf(&big, "line %03d: %s\n", i, "0123456789");
	dc_format(&r, NULL, big.p, big.n);
	CHECK(r.n >= 2 && r.n <= DC_MAXMSG && r.file == NULL);
	for (i = 0; i < r.n; i++) {
		size_t l = strlen(r.msg[i]);
		CHECK(l <= DC_LIMIT);
		CHECK(!strncmp(r.msg[i], "```\n", 4) &&
		    !strcmp(r.msg[i] + l - 4, "\n```"));
		sb_printf(&joined, "%.*s\n", (int)(l - 8), r.msg[i] + 4);
	}
	CHECK(joined.n == big.n && !memcmp(joined.p, big.p, big.n));
	dc_reply_free(&r);

	/* long: one preview message and the whole output attached */
	for (i = 0; i < 3000; i++)
		sb_printf(&big, "more %d\n", i);
	dc_format(&r, "note", big.p, big.n);
	CHECK(r.n == 1 && strlen(r.msg[0]) <= DC_LIMIT);
	CHECK(!strncmp(r.msg[0], "note\n", 5));
	CHECK(strstr(r.msg[0], DC_FILENAME) != NULL);
	CHECK(r.file && r.filelen == big.n && !memcmp(r.file, big.p, big.n));
	dc_reply_free(&r);

	/* one very long line: cut on a UTF-8 boundary */
	sb_clear(&big);
	for (i = 0; i < 1500; i++)
		sb_puts(&big, "\xc3\xa9");	/* e acute, two bytes */
	dc_format(&r, NULL, big.p, big.n);
	CHECK(r.n == 2);
	for (i = 0; i < r.n; i++) {
		size_t l = strlen(r.msg[i]);
		for (j = 4; j < (int)l - 4; j++)
			if ((unsigned char)r.msg[i][j] == 0xc3)
				CHECK((unsigned char)r.msg[i][j + 1] == 0xa9);
		CHECK(((unsigned char)r.msg[i][4] & 0xc0) != 0x80);
	}
	dc_reply_free(&r);
	free(big.p);
	free(joined.p);
}

static void
test_json(void)
{
	static const char ev[] =
	    "{\"id\":\"1311111111111111111\",\"channel_id\":\"42\","
	    "\"guild_id\":\"7\",\"author\":{\"id\":\"99\",\"username\":\"ann\","
	    "\"bot\":false},\"content\":\"$```\\nf(x) = x \\u00b2\\n```\","
	    "\"type\":0,\"unknown_field\":[1,{\"x\":null}]}";
	struct discord_message m;
	struct discord_message_reference ref = {0};
	struct discord_allowed_mention am = {0};
	struct discord_create_message p = {0};
	char *none[1] = {NULL};
	struct strings parse = {0, none, 0};
	char buf[4096];
	size_t n;

	memset(&m, 0, sizeof m);
	discord_message_from_json(ev, sizeof ev - 1, &m);
	CHECK(m.id == 1311111111111111111ULL);
	CHECK(m.channel_id == 42 && m.guild_id == 7);
	CHECK(m.author && m.author->id == 99 && !m.author->bot);
	CHECK(m.content && !strcmp(m.content, "$```\nf(x) = x \xc2\xb2\n```"));
	discord_message_cleanup(&m);

	ref.message_id = 5;
	ref.channel_id = 42;
	am.parse = &parse;
	p.content = "```\n\"q\"\\\n```";
	p.message_reference = &ref;
	p.allowed_mentions = &am;
	n = discord_create_message_to_json(buf, sizeof buf, &p);
	buf[n] = 0;
	CHECK(strstr(buf, "\"content\":\"```\\n\\\"q\\\"\\\\\\n```\"") != NULL);
	CHECK(strstr(buf, "\"message_id\":\"5\"") != NULL);
	CHECK(strstr(buf, "\"parse\":[]") != NULL);
}

int
dc_selftest(void)
{
	test_prefix();
	test_format();
	test_json();
	printf("%s: %d checks, %d failed\n", nfail ? "FAIL" : "ok", ncheck,
	    nfail);
	return nfail != 0;
}
