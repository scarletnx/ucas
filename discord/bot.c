/*
 * ucas-discord: run `$`-prefixed Discord messages through ucas.
 *
 *   ucas-discord [--token-file PATH]   connect to the Gateway (Concord)
 *   ucas-discord --check-token         find and load the token, say where
 *                                       it came from (never print it)
 *   ucas-discord --stdin                read gateway events (JSON, one per
 *                                       line) and print the REST requests
 *                                       that would be sent
 *   ucas-discord --selftest             unit checks
 *
 * Both modes pass every message through handle_message(); only post()
 * differs.  Evaluation happens in the supervisor process (sup.c).
 */
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#ifdef __FreeBSD__
#include <sys/sysctl.h>
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include "discord.h"
#include "logconf.h"

#define JSMN_STRICT
#define JSMN_HEADER
#include "jsmn.h"
#include "jsmn-find.h"

#include "ucas.h"
#include "sup.h"
#include "text.h"

int dc_selftest(void);

struct pend {
	uint64_t id;
	u64snowflake chan, msg, guild, author;
	char *orig;      /* the user's code, for a follow-up question */
	char *answers;   /* "\x1eanswer ...\n" lines sent with it */
	int echo;
	int dropped;     /* it replaced an open question */
};

/*
 * Follow-up questions.  The engine runs in "defer" mode (see nlp.c): an
 * English request that needs an answer prints the question after a
 * "\x1eask" marker and stops.  The question is kept here per channel and
 * author; that author's next `$` message in the channel is the answer, and
 * the original message runs again with every answer so far queued.
 */
#define MAXASK 64
#define ASK_TTL (15 * 60)
struct ask {
	u64snowflake chan, author;
	char *orig;
	char *answers;
	int echo;
	int nopt;        /* a numbered choice; 0 for a free-text question */
	time_t when;
};

/*
 * Is this reply an answer, or has the user moved on to a new request?
 * A choice takes a short reply (a number, an option's name); a free-text
 * question takes anything that does not start like a new request.
 */
static int
looks_like_answer(const char *c, int nopt)
{
	static const char *const starts[] = {
		"approx", "approximate", "what", "whats", "how", "is", "are",
		"does", "do", "can", "show", "range", "domain", "wrap", "crit",
		"diff", "equiv", "gates", "find", "more", "fit", "error", "cost",
		"table", "let", "define", "integrate", "closed", "scheme",
		"continuous", "dsolve", "solve", "help", "alt", "parse", "use",
		"models", "funcs", "assume", "forget", "compute", "evaluate",
		"simplify", "plot", NULL
	};
	char w[24];
	size_t n = 0;
	int words = 0, k, inw = 0;
	const char *p;

	while (*c == ' ')
		c++;
	if (*c == 0)
		return 1;
	for (p = c; *p; p++) {
		if (*p == '=' || *p == '{')
			return 0;   /* a definition or a block */
		if (*p != ' ' && !inw)
			words++;
		inw = *p != ' ';
	}
	while (c[n] && c[n] != ' ' && c[n] != '?' && n + 1 < sizeof w) {
		w[n] = (char)tolower((unsigned char)c[n]);
		n++;
	}
	w[n] = 0;
	for (k = 0; starts[k]; k++)
		if (!strcmp(w, starts[k]))
			return 0;
	if (nopt > 0)
		return words <= 3;
	return 1;
}
static struct ask asks[MAXASK];
static int nasks;

static void
ask_drop(int i)
{
	free(asks[i].orig);
	free(asks[i].answers);
	asks[i] = asks[--nasks];
}

static struct ask *
ask_find(u64snowflake chan, u64snowflake author)
{
	time_t now = time(NULL);
	int i;

	for (i = nasks; i-- > 0;)
		if (now - asks[i].when > ASK_TTL)
			ask_drop(i);
	for (i = 0; i < nasks; i++)
		if (asks[i].chan == chan && asks[i].author == author)
			return &asks[i];
	return NULL;
}

static void
ask_drop_chan(u64snowflake chan)
{
	int i;

	for (i = nasks; i-- > 0;)
		if (asks[i].chan == chan)
			ask_drop(i);
}

/* user text must not carry engine directives */
static void
strip_directives(char *s)
{
	char *d = s;

	for (; *s; s++)
		if (*s != '\x1e')
			*d++ = *s;
	*d = 0;
}

static struct discord *client;	/* NULL in --stdin mode */
static struct sup_limits limits;
static int supfd = -1;
static u64snowflake self_id;
static char self_name[64] = "Wimsy", self_disc[16] = "5388";
static struct pend *pend;
static int npend, cappend;
static uint64_t nextid;

/* ---- outgoing ---- */

/*
 * Replies go out asynchronously.  Count them until Discord answers, so
 * --one-session does not shut the connection down before the reply has
 * been sent, and report the ones Discord refuses.
 */
static int outstanding;
static void maybe_end_session(void);
static void log_posted(u64snowflake chan, u64snowflake id, int code,
    const char *err);

static void
post_done(struct discord *c, struct discord_response *resp,
    const struct discord_message *ret)
{
	(void)c;
	(void)resp;
	log_posted(ret ? ret->channel_id : 0, ret ? ret->id : 0, 0, NULL);
	outstanding--;
	maybe_end_session();
}

static void
post_fail(struct discord *c, struct discord_response *resp)
{
	(void)c;
	fprintf(stderr, "ucas-discord: a reply was not sent: %s (code %d)\n",
	    discord_strerror(resp->code, NULL), (int)resp->code);
	log_posted(0, 0, (int)resp->code, discord_strerror(resp->code, NULL));
	outstanding--;
	maybe_end_session();
}

static void
post(u64snowflake chan, u64snowflake msg, u64snowflake guild,
    const char *content, const char *file, size_t filelen, const char *img,
    size_t imglen, const char *imgname, const char *imgtype)
{
	char *noparse[1] = {NULL};
	struct strings parse = {0, noparse, 0};
	struct discord_allowed_mention am = {0};
	struct discord_message_reference ref = {0};
	struct discord_attachment att[2];
	struct discord_attachments atts = {0};
	struct discord_create_message p = {0};
	char buf[16384];
	size_t n;

	/* a reply that pings nobody, not even the author */
	am.parse = &parse;
	am.replied_user = false;
	ref.message_id = msg;
	ref.channel_id = chan;
	ref.guild_id = guild;
	ref.fail_if_not_exists = false;
	p.content = (char *)content;
	p.allowed_mentions = &am;
	p.message_reference = &ref;
	memset(att, 0, sizeof att);
	if (img != NULL) {
		att[atts.size].id = atts.size;
		att[atts.size].content = (char *)img;
		att[atts.size].size = imglen;
		att[atts.size].filename = (char *)imgname;
		att[atts.size].content_type = (char *)imgtype;
		atts.size++;
	}
	if (file != NULL) {
		att[atts.size].id = atts.size;
		att[atts.size].content = (char *)file;
		att[atts.size].size = filelen;
		att[atts.size].filename = DC_FILENAME;
		att[atts.size].content_type = "text/plain; charset=utf-8";
		atts.size++;
	}
	if (atts.size) {
		atts.array = att;
		p.attachments = &atts;
	}
	if (client != NULL) {
		struct discord_ret_message ret = {0};
		ret.done = post_done;
		ret.fail = post_fail;
		outstanding++;
		discord_create_message(client, chan, &p, &ret);
		return;
	}
	n = discord_create_message_to_json(buf, sizeof buf, &p);
	printf("POST /channels/%" PRIu64 "/messages%s\n%.*s\n", chan,
	    atts.size ? " (multipart)" : "", (int)n, buf);
	if (img != NULL)
		printf("  files[0]: %s, %s, %zu bytes\n", imgname, imgtype, imglen);
	if (file != NULL)
		printf("  files[%d]: %s, %zu bytes\n", img != NULL, DC_FILENAME,
		    filelen);
}

/* ---- requests ---- */

/*
 * Session log (--log PATH): one JSON object per line, appended and flushed,
 * for reading afterwards how people use the bot.  "in" is a `$` message as
 * received (with what the bot made of it), "out" the engine's full output
 * for a request.  It holds user messages: keep it out of git (logs/).
 */
static FILE *slog;
static int one_session, sessions_done;

static void
jstr(FILE *f, const char *s, size_t n)
{
	size_t i;

	fputc('"', f);
	for (i = 0; i < n; i++) {
		unsigned char c = (unsigned char)s[i];
		if (c == '"' || c == '\\')
			fprintf(f, "\\%c", c);
		else if (c == '\n')
			fputs("\\n", f);
		else if (c == '\t')
			fputs("\\t", f);
		else if (c < 0x20 || c == 0x7f)
			fprintf(f, "\\u%04x", c);
		else
			fputc(c, f);
	}
	fputc('"', f);
}

static void
log_ts(FILE *f)
{
	char ts[32];
	time_t now = time(NULL);

	strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
	fprintf(f, "\"ts\":\"%s\"", ts);
}

static void
log_in(const struct discord_message *m, const char *as, uint64_t req)
{
	const char *name = m->author && m->author->username ?
	    m->author->username : "?";

	if (slog == NULL)
		return;
	fputc('{', slog);
	log_ts(slog);
	fprintf(slog, ",\"ev\":\"in\",\"as\":\"%s\",\"req\":%" PRIu64
	    ",\"chan\":\"%" PRIu64 "\",\"guild\":\"%" PRIu64 "\",\"msg\":\"%"
	    PRIu64 "\",\"author\":\"%" PRIu64 "\",\"name\":", as, req,
	    m->channel_id, m->guild_id, m->id, m->author ? m->author->id : 0);
	jstr(slog, name, strlen(name));
	fputs(",\"content\":", slog);
	jstr(slog, m->content ? m->content : "", m->content ?
	    strlen(m->content) : 0);
	fputs("}\n", slog);
	fflush(slog);
}

/* what Discord answered to a reply: the message id, or the error */
static void
log_posted(u64snowflake chan, u64snowflake id, int code, const char *err)
{
	if (slog == NULL)
		return;
	fputc('{', slog);
	log_ts(slog);
	if (err == NULL)
		fprintf(slog, ",\"ev\":\"posted\",\"chan\":\"%" PRIu64 "\","
		    "\"id\":\"%" PRIu64 "\"}\n", chan, id);
	else {
		fprintf(slog, ",\"ev\":\"post-failed\",\"code\":%d,\"error\":",
		    code);
		jstr(slog, err, strlen(err));
		fputs("}\n", slog);
	}
	fflush(slog);
}

static void
log_out(uint64_t req, int status, int question, const char *out, size_t n)
{
	if (slog == NULL)
		return;
	fputc('{', slog);
	log_ts(slog);
	fprintf(slog, ",\"ev\":\"out\",\"req\":%" PRIu64 ",\"status\":%d,"
	    "\"question\":%s,\"output\":", req, status, question ? "true" :
	    "false");
	jstr(slog, out ? out : "", out ? n : 0);
	fputs("}\n", slog);
	fflush(slog);
}


static int
send_req(uint32_t type, const struct discord_message *m, const char *text,
    int echo, const char *orig, const char *answers)
{
	struct sframe h;
	struct pend *pp;

	memset(&h, 0, sizeof h);
	h.type = type;
	h.id = ++nextid;
	h.chan = m->channel_id;
	h.arg = (uint32_t)echo;
	h.len = text ? (uint32_t)strlen(text) : 0;
	if (frame_write(supfd, &h, text) < 0)
		return -1;
	if (npend == cappend) {
		cappend = cappend ? cappend * 2 : 16;
		pend = xrealloc(pend, (size_t)cappend * sizeof *pend);
	}
	pp = &pend[npend++];
	pp->id = h.id;
	pp->chan = m->channel_id;
	pp->msg = m->id;
	pp->guild = m->guild_id;
	pp->author = m->author ? m->author->id : 0;
	pp->orig = orig ? strdup(orig) : NULL;
	pp->answers = answers ? strdup(answers) : NULL;
	pp->echo = echo;
	pp->dropped = 0;
	return 0;
}

/* returns DC_IGNORE, DC_EVAL or DC_RESET */
/*
 * "<@id> $expr" or "<@!id> $expr": without the Message Content intent,
 * Discord only delivers the text of server messages that mention the bot.
 */
static int
word_end(char c)
{
	return !(isalnum((unsigned char)c) || c == '_');
}

/*
 * Is the message for the bot?  It is when it mentions the bot anywhere:
 * <@id> or <@!id> (a real mention), or "@Name#1234" / "@Name" typed as
 * text.  The text without the mentions goes to out.
 */
static int
strip_mentions(const char *s, struct sbuf *out)
{
	char id1[40], id2[40], nm1[96], nm2[80];
	size_t l1, l2, l3, l4;
	int found = 0;

	sb_clear(out);
	sb_puts(out, "");
	if (s == NULL)
		return 0;
	l1 = (size_t)snprintf(id1, sizeof id1, "<@%" PRIu64 ">", self_id);
	l2 = (size_t)snprintf(id2, sizeof id2, "<@!%" PRIu64 ">", self_id);
	l3 = (size_t)snprintf(nm1, sizeof nm1, "@%s#%s", self_name, self_disc);
	l4 = (size_t)snprintf(nm2, sizeof nm2, "@%s", self_name);
	while (*s) {
		if (self_id && !strncmp(s, id1, l1)) {
			s += l1;
			found = 1;
		} else if (self_id && !strncmp(s, id2, l2)) {
			s += l2;
			found = 1;
		} else if (!strncasecmp(s, nm1, l3) && word_end(s[l3])) {
			s += l3;
			found = 1;
		} else if (!strncasecmp(s, nm2, l4) && word_end(s[l4])) {
			s += l4;
			found = 1;
		} else
			sb_putc(out, *s++);
	}
	return found;
}

static int
handle_message(const struct discord_message *m)
{
	struct sbuf code = {0}, req = {0}, ans = {0};
	const char *content;
	struct ask *q;
	int kind, echo, dropped = 0;

	if (m->author == NULL || m->author->bot || m->author->System ||
	    m->author->id == self_id || m->webhook_id != 0)
		return DC_IGNORE;
	{
		/*
		 * For the bot: a mention anywhere, or a Discord reply to one of
		 * its messages.  The asker's short reply to an open numbered
		 * question counts too, without a mention ("2", "ln").
		 */
		struct sbuf body = {0};
		int addressed = strip_mentions(m->content, &body);
		if (m->referenced_message && m->referenced_message->author &&
		    m->referenced_message->author->id == self_id)
			addressed = 1;
		if (!addressed) {
			q = ask_find(m->channel_id, m->author->id);
			if (q == NULL || q->nopt == 0 ||
			    !looks_like_answer(body.p, q->nopt)) {
				free(body.p);
				return DC_IGNORE;
			}
		}
		/* dc_command reads the "$" form: code fences, reset, ... */
		sb_clear(&req);
		sb_printf(&req, "$%s", body.p);
		free(body.p);
		{
			/* what addressing leaves behind: "hey <@id>, ..." */
			static const char *const greet[] = {"hey", "hi", "hello",
			    "yo", "ok", "okay", "so", "um", "hiya", "heya", NULL};
			const char *t = req.p + 1;
			int g, again = 1;
			while (again) {
				again = 0;
				while (*t == ' ' || *t == '\t' || *t == '\n' ||
				    *t == ',' || *t == ':' || *t == ';' || *t == '!' ||
				    *t == '-')
					t++;
				for (g = 0; greet[g]; g++) {
					size_t gl = strlen(greet[g]);
					if (!strncasecmp(t, greet[g], gl) &&
					    word_end(t[gl])) {
						t += gl;
						again = 1;
						break;
					}
				}
			}
			if (*t == 0 && ask_find(m->channel_id, m->author->id) == NULL) {
				post(m->channel_id, m->id, m->guild_id,
				    "mention me with a question or ucas input, e.g. "
				    "`@Wimsy what is the range of x squared for a signed "
				    "byte x?` or `@Wimsy range isqrt(x: u4)`; `@Wimsy help` "
				    "lists everything", NULL, 0, NULL, 0, NULL, NULL);
				free(req.p);
				return DC_IGNORE;
			}
			memmove(req.p + 1, t, strlen(t) + 1);
		}
		content = req.p;
		kind = dc_command(content, &code, &echo);
		req.p = NULL;
		req.n = req.cap = 0;
		free((char *)content);
	}
	if (kind == DC_RESET) {
		ask_drop_chan(m->channel_id);
		send_req(SF_RESET, m, NULL, 0, NULL, NULL);
		log_in(m, "reset", nextid);
		free(code.p);
		return kind;
	}
	if ((q = ask_find(m->channel_id, m->author->id)) != NULL &&
	    !looks_like_answer(kind == DC_EVAL ? code.p : "", q->nopt) &&
	    strcmp(kind == DC_EVAL ? code.p : "", "cancel") != 0) {
		/* a new request: the open question is dropped */
		ask_drop((int)(q - asks));
		log_in(m, "dropped-question", 0);
		q = NULL;
		dropped = 1;
	}
	if (q != NULL) {
		/* this author's next $ message answers the open question */
		const char *c = kind == DC_EVAL ? code.p : "";
		char *orig;
		if (!strcmp(c, "cancel")) {
			ask_drop((int)(q - asks));
			log_in(m, "cancel", 0);
			post(m->channel_id, m->id, m->guild_id,
			    "question cancelled", NULL, 0, NULL, 0, NULL, NULL);
			maybe_end_session();
			free(code.p);
			return DC_EVAL;
		}
		sb_puts(&ans, q->answers ? q->answers : "");
		sb_puts(&ans, "\x1e" "answer ");
		for (; *c; c++)
			sb_putc(&ans, *c == '\n' || *c == '\r' ? ' ' : *c);
		sb_putc(&ans, '\n');
		strip_directives(ans.p + (q->answers ? strlen(q->answers) : 0) + 1);
		sb_printf(&req, "\x1e" "defer\n%s%s", ans.p, q->orig);
		orig = q->orig;
		q->orig = NULL;
		echo = q->echo;
		ask_drop((int)(q - asks));
		send_req(SF_EVAL, m, req.p, echo, orig, ans.p);
		log_in(m, "answer", nextid);
		free(orig);
		free(code.p);
		free(req.p);
		free(ans.p);
		return DC_EVAL;
	}
	if (kind == DC_EVAL) {
		strip_directives(code.p);
		sb_printf(&req, "\x1e" "defer\n%s", code.p);
		send_req(SF_EVAL, m, req.p, echo, code.p, "");
		pend[npend - 1].dropped = dropped;
		log_in(m, "request", nextid);
	}
	free(code.p);
	free(req.p);
	return kind;
}

/* base64 to bytes, skipping newlines; returns the length */
static size_t
b64dec(const char *s, size_t n, char *out, size_t cap)
{
	unsigned v = 0;
	int bits = 0;
	size_t i, o = 0;

	for (i = 0; i < n; i++) {
		int c = (unsigned char)s[i], d;
		if (c >= 'A' && c <= 'Z')
			d = c - 'A';
		else if (c >= 'a' && c <= 'z')
			d = c - 'a' + 26;
		else if (c >= '0' && c <= '9')
			d = c - '0' + 52;
		else if (c == '+')
			d = 62;
		else if (c == '/')
			d = 63;
		else
			continue;
		v = (v << 6) | (unsigned)d;
		bits += 6;
		if (bits >= 8) {
			bits -= 8;
			if (o < cap)
				out[o++] = (char)(v >> bits & 0xff);
		}
	}
	return o;
}

static void
on_response(const struct sframe *h, const char *out)
{
	struct dc_reply r;
	struct sbuf txt = {0};
	char *img = NULL, imgname[64], imgtype[64];
	size_t imglen = 0, olen = h->len;
	struct pend pp;
	char sbuf[256];
	const char *head;
	int i;

	for (i = 0; i < npend; i++)
		if (pend[i].id == h->id)
			break;
	if (i == npend)
		return;
	pp = pend[i];
	pend[i] = pend[--npend];
	head = dc_status(h->status, h->arg, &limits, sbuf, sizeof sbuf);
	{
		/*
		 * "\x1efile NAME TYPE SIZE\n<base64>\x1eend\n": a file made by
		 * the engine (animate's GIF).  Decode it, attach it, and keep
		 * the block out of the text.
		 */
		const char *fb = out ? memmem(out, h->len, "\x1e" "file ", 6) : NULL;
		const char *fe = fb ? memmem(fb, (size_t)(out + h->len - fb),
		    "\x1e" "end\n", 5) : NULL;
		if (fb && fe) {
			char fname[64] = "file", ftype[64] = "application/octet-stream";
			const char *q = fb + 6, *nl = memchr(fb, '\n',
			    (size_t)(fe - fb));
			size_t cap;
			if (nl != NULL && sscanf(q, "%63s %63s", fname, ftype) >= 1) {
				cap = (size_t)(fe - nl) * 3 / 4 + 4;
				img = xmalloc(cap);
				imglen = b64dec(nl + 1, (size_t)(fe - nl - 1), img, cap);
				snprintf(imgname, sizeof imgname, "%s", fname);
				snprintf(imgtype, sizeof imgtype, "%s", ftype);
			}
			sb_printf(&txt, "%.*s%.*s", (int)(fb - out), out,
			    (int)(out + h->len - (fe + 5)), fe + 5);
			out = txt.p ? txt.p : "";
			olen = txt.n;
		} else if (fb) {
			/* cut by the output limit: drop the partial file */
			sb_printf(&txt, "%.*s  (the generated file did not fit in "
			    "the output limit of %zu bytes; ask for a smaller one)\n",
			    (int)(fb - out), out, (size_t)limits.max_out);
			out = txt.p;
			olen = txt.n;
		}
	}
	{
		/* a follow-up question: remember it, say how to answer */
		const char *mk = out ? memmem(out, olen, "\x1e" "ask ", 5) : NULL;
		struct sbuf o = {0};
		char hbuf[320];
		if (mk != NULL && pp.orig != NULL && pp.author != 0 &&
		    h->status == 0) {
			const char *eol = memchr(mk, '\n', (size_t)(out + olen - mk));
			int nopt = 0;
			struct ask *q;
			if (eol == NULL)
				eol = out + olen;
			if (!strncmp(mk + 5, "choice ", 7))
				nopt = atoi(mk + 12);
			sb_printf(&o, "%.*s%.*s", (int)(mk - out), out,
			    (int)(out + olen - (eol < out + olen ? eol + 1 :
			    eol)), eol < out + olen ? eol + 1 : eol);
			if ((q = ask_find(pp.chan, pp.author)) != NULL)
				ask_drop((int)(q - asks));
			if (nasks == MAXASK)
				ask_drop(0);
			q = &asks[nasks++];
			q->chan = pp.chan;
			q->author = pp.author;
			q->orig = pp.orig;
			q->answers = pp.answers ? pp.answers : strdup("");
			q->echo = pp.echo;
			q->nopt = nopt;
			q->when = time(NULL);
			pp.orig = pp.answers = NULL;
			if (nopt > 0)
				snprintf(hbuf, sizeof hbuf, "a question first: answer "
				    "1..%d (or the option's name), mention me alone for "
				    "the default, or say cancel", nopt);
			else
				snprintf(hbuf, sizeof hbuf, "a question first: answer "
				    "with a mention or a Discord reply (mention me alone "
				    "for the defaults), or say cancel");
			dc_format(&r, hbuf, o.p ? o.p : "", o.n);
			free(o.p);
		} else
			dc_format(&r, head == NULL && pp.dropped ? "(your open "
			    "question was dropped: this looked like a new request)" :
			    head, out, olen);
	}
	log_out(h->id, (int)h->status, out && memmem(out, olen, "\x1e" "ask ",
	    5) != NULL, out, olen);
	free(pp.orig);
	free(pp.answers);
	sessions_done++;
	maybe_end_session();
	for (i = 0; i < r.n; i++)
		post(pp.chan, pp.msg, pp.guild, r.msg[i],
		    i == 0 ? r.file : NULL, r.filelen, i == 0 ? img : NULL, imglen,
		    imgname, imgtype);
	dc_reply_free(&r);
	free(img);
	free(txt.p);
}

/* 1 handled, 0 the supervisor is gone */
static int
read_response(void)
{
	struct sframe h;
	char *out;

	if (frame_read(supfd, &h, &out) != 1)
		return 0;
	if (h.type == SF_RESP)
		on_response(&h, out);
	free(out);
	return 1;
}

/* ---- live mode (Concord) ---- */

static void
on_ready(struct discord *c, const struct discord_ready *ev)
{
	self_id = ev->user->id;
	if (ev->user->username)
		snprintf(self_name, sizeof self_name, "%s", ev->user->username);
	if (ev->user->discriminator)
		snprintf(self_disc, sizeof self_disc, "%s",
		    ev->user->discriminator);
	fprintf(stderr, "ucas-discord: connected as %s (%" PRIu64 ")\n",
	    ev->user->username ? ev->user->username : "?", self_id);
}

static void
on_message(struct discord *c, const struct discord_message *m)
{
	handle_message(m);
}

static int sup_dead;

static void
on_sup(struct io_poller *io, enum io_poller_events ev, void *data)
{
	if (!read_response()) {
		fprintf(stderr, "ucas-discord: evaluator process died\n");
		sup_dead = 1;
		io_poller_socket_del(io, supfd);
		discord_shutdown(client);
	}
}


/*
 * The token, in order of precedence:
 *   --token-file PATH
 *   DISCORD_TOKEN in the environment
 *   <directory of this executable>/secrets/token.txt
 *   <directory of this executable>/../secrets/token.txt  (in-tree build:
 *       the binary is in discord/, secrets/ is at the top of the tree)
 * The token is never printed; errors name the file only.
 */
static const char *token_file_opt;

static int
self_dir(char *dir, size_t n)
{
	char path[PATH_MAX];
	size_t len = sizeof path;

#ifdef __FreeBSD__
	int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
	if (sysctl(mib, 4, path, &len, NULL, 0) != 0)
		return 0;
#else
	ssize_t r = readlink("/proc/self/exe", path, sizeof path - 1);
	if (r <= 0)
		return 0;
	path[r] = 0;
	(void)len;
#endif
	snprintf(dir, n, "%s", dirname(path));
	return 1;
}

/* read a token file into a malloc'd string; NULL with a message on error */
static char *
read_token_file(const char *path, int quiet_missing)
{
	struct stat st;
	char buf[512];
	ssize_t r;
	size_t n;
	int fd;

	if ((fd = open(path, O_RDONLY)) < 0) {
		if (!(quiet_missing && errno == ENOENT))
			fprintf(stderr, "ucas-discord: %s: %s\n", path,
			    strerror(errno));
		return NULL;
	}
	if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
	    st.st_size >= (off_t)sizeof buf) {
		fprintf(stderr, "ucas-discord: %s: not a regular file of a "
		    "token's size\n", path);
		close(fd);
		return NULL;
	}
	if (st.st_mode & (S_IRWXG | S_IRWXO))
		fprintf(stderr, "ucas-discord: warning: %s is readable by "
		    "others; chmod 600 it\n", path);
	r = read(fd, buf, sizeof buf - 1);
	close(fd);
	if (r < 0) {
		fprintf(stderr, "ucas-discord: %s: %s\n", path, strerror(errno));
		return NULL;
	}
	n = (size_t)r;
	while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' ||
	    buf[n - 1] == ' ' || buf[n - 1] == '\t'))
		n--;
	buf[n] = 0;
	if (n == 0) {
		fprintf(stderr, "ucas-discord: %s is empty\n", path);
		return NULL;
	}
	{
		char *tok = strdup(buf);
		memset(buf, 0, sizeof buf);
		return tok;
	}
}

/* *from says where the token came from (a path or "DISCORD_TOKEN") */
static char *
load_token(char *from, size_t n)
{
	char dir[PATH_MAX], path[PATH_MAX + 32], *env, *tok;
	int i;

	if (token_file_opt != NULL) {
		snprintf(from, n, "%s", token_file_opt);
		return read_token_file(token_file_opt, 0);
	}
	if ((env = getenv("DISCORD_TOKEN")) != NULL && *env != 0) {
		tok = strdup(env);
		memset(env, 0, strlen(env));
		unsetenv("DISCORD_TOKEN");
		snprintf(from, n, "the DISCORD_TOKEN environment variable");
		return tok;
	}
	if (!self_dir(dir, sizeof dir)) {
		fprintf(stderr, "ucas-discord: cannot find this executable's "
		    "path; use --token-file\n");
		return NULL;
	}
	for (i = 0; i < 2; i++) {
		snprintf(path, sizeof path, "%s/%ssecrets/token.txt", dir,
		    i ? "../" : "");
		if ((tok = read_token_file(path, 1)) != NULL) {
			char real[PATH_MAX];
			snprintf(from, n, "%s", realpath(path, real) ? real : path);
			return tok;
		}
		if (access(path, F_OK) == 0)
			return NULL;   /* exists but unusable: already reported */
	}
	fprintf(stderr, "ucas-discord: no token: tried %s/secrets/token.txt "
	    "and %s/../secrets/token.txt; or use --token-file PATH or "
	    "DISCORD_TOKEN\n", dir, dir);
	return NULL;
}

static int
check_token(void)
{
	char from[PATH_MAX + 64], *tok, *p;
	size_t len, dots = 0;

	if ((tok = load_token(from, sizeof from)) == NULL)
		return 1;
	len = strlen(tok);
	for (p = tok; *p; p++)
		dots += *p == '.';
	printf("token from %s: %zu characters, %s\n", from, len,
	    dots == 2 ? "shaped like a bot token (three dot-separated parts)" :
	    "NOT shaped like a bot token (expected three dot-separated parts)");
	memset(tok, 0, len);
	free(tok);
	return dots == 2 ? 0 : 1;
}


/*
 * Concord's own log, to stderr.  Without a config file it is silent, so
 * a failed connection ended with no reason given.  UCAS_DISCORD_LOG sets
 * the level (trace debug info warn error fatal; default info).  Do not use
 * trace in production: it can include HTTP details.
 */
static volatile sig_atomic_t gw_close;   /* 4014, 4004 or 0 */

static void
concord_log_cb(log_Event *ev)
{
	static const char *const names[] = {"TRACE", "DEBUG", "INFO", "WARN",
	    "ERROR", "FATAL"};
	char ts[32];

	char msg[1024];

	vsnprintf(msg, sizeof msg, ev->fmt, ev->ap);
	/* the close codes that decide what the run loop does next */
	if (strstr(msg, "DISALLOWED_INTENTS") || strstr(msg, "code: 4014"))
		gw_close = 4014;
	else if (strstr(msg, "AUTHENTICATION_FAILED") ||
	    strstr(msg, "code: 4004"))
		gw_close = 4004;
	strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", ev->time);
	fprintf(ev->udata, "%s %-5s concord %s:%d: %s\n", ts,
	    names[ev->level], ev->file, ev->line, msg);
	fflush(ev->udata);
}

static int
concord_log_level(void)
{
	static const char *const names[] = {"trace", "debug", "info", "warn",
	    "error", "fatal"};
	const char *v = getenv("UCAS_DISCORD_LOG");
	int i;

	if (v == NULL || *v == 0)
		return LOG_INFO;
	for (i = 0; i < 6; i++)
		if (!strcasecmp(v, names[i]))
			return i;
	fprintf(stderr, "ucas-discord: UCAS_DISCORD_LOG=%s is not a level; "
	    "using info\n", v);
	return LOG_INFO;
}

/*
 * SIGINT/SIGTERM: the handler only writes a byte; the event loop reads it
 * and shuts the client down, and the run loop does not reconnect.
 */
static int stop_pipe[2] = {-1, -1};
static volatile sig_atomic_t stop_requested;

static void
on_signal(int sig)
{
	int saved = errno;

	(void)sig;
	stop_requested = 1;
	if (stop_pipe[1] >= 0)
		(void)write(stop_pipe[1], "x", 1);
	errno = saved;
}

static void
on_stop(struct io_poller *io, enum io_poller_events ev, void *data)
{
	char b[16];

	(void)ev;
	(void)data;
	while (read(stop_pipe[0], b, sizeof b) > 0)
		;
	io_poller_socket_del(io, stop_pipe[0]);
	discord_shutdown(client);
}

/* sleep up to secs, returning early on a stop signal */
static void
backoff_sleep(unsigned secs)
{
	struct timespec ts;

	ts.tv_sec = (time_t)secs;
	ts.tv_nsec = 0;
	while (!stop_requested && nanosleep(&ts, &ts) != 0 && errno == EINTR)
		;
}

/*
 * Connect, run, and keep running: the gateway is reconnected with
 * exponential backoff (2 s doubling to 5 min; back to 2 s after any
 * session that lasted over a minute).  Only a stop signal, a dead
 * evaluator or a rejected token end the bot.  If Discord refuses the
 * Message Content intent (close code 4014: it is not enabled for the
 * application), the bot says how to enable it and carries on without
 * it: it then sees the text of DMs and of messages that mention it.
 */
static void
maybe_end_session(void)
{
	if (!one_session || sessions_done == 0 || npend > 0 || nasks > 0 ||
	    outstanding > 0)
		return;
	fprintf(stderr, "ucas-discord: session complete (--one-session): "
	    "shutting down\n");
	stop_requested = 1;
	if (client != NULL)
		discord_shutdown(client);
}

static int
run_live(void)
{
	const struct discord_user *me;
	char from[PATH_MAX + 64], *tok;
	CCORDcode rc;
	uint64_t intents = DISCORD_GATEWAY_GUILD_MESSAGES |
	    DISCORD_GATEWAY_DIRECT_MESSAGES | DISCORD_GATEWAY_MESSAGE_CONTENT;
	unsigned delay = 2;
	time_t started;
	struct sigaction sa;
	int status = 0;

	if ((tok = load_token(from, sizeof from)) == NULL)
		return 1;
	(void)mlock(tok, strlen(tok) + 1);   /* kept for reconnects; no swap */
	fprintf(stderr, "ucas-discord: token from %s\n", from);
	if (pipe(stop_pipe) != 0) {
		perror("ucas-discord: pipe");
		return 1;
	}
	(void)fcntl(stop_pipe[0], F_SETFL, O_NONBLOCK);
	(void)fcntl(stop_pipe[1], F_SETFL, O_NONBLOCK);
	memset(&sa, 0, sizeof sa);
	sa.sa_handler = on_signal;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGINT, &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);
	sigaction(SIGHUP, &sa, NULL);
	ccord_global_init();
	while (!stop_requested && !sup_dead) {
		gw_close = 0;
		client = discord_init(tok);
		if (client == NULL) {
			fprintf(stderr, "ucas-discord: cannot initialise the "
			    "client; retrying in %u s\n", delay);
			backoff_sleep(delay);
			delay = delay < 150 ? delay * 2 : 300;
			continue;
		}
		logconf_add_callback(discord_get_logconf(client),
		    concord_log_cb, stderr, concord_log_level());
		if ((me = discord_get_self(client)) != NULL && me->id)
			self_id = me->id;
		discord_add_intents(client, intents);
		discord_set_on_ready(client, on_ready);
		discord_set_on_message_create(client, on_message);
		io_poller_socket_add(discord_get_io_poller(client), supfd,
		    IO_POLLER_IN, on_sup, NULL);
		io_poller_socket_add(discord_get_io_poller(client), stop_pipe[0],
		    IO_POLLER_IN, on_stop, NULL);
		started = time(NULL);
		rc = discord_run(client);
		discord_cleanup(client);
		client = NULL;
		if (stop_requested || sup_dead)
			break;
		if (gw_close == 4004) {
			fprintf(stderr, "ucas-discord: Discord rejected the token "
			    "(4004 authentication failed): replace %s and restart\n",
			    from);
			status = 1;
			break;
		}
		if (gw_close == 4014 &&
		    (intents & DISCORD_GATEWAY_MESSAGE_CONTENT)) {
			fprintf(stderr,
			    "ucas-discord: Discord refused the Message Content intent "
			    "(4014).\n"
			    "  Enable it: https://discord.com/developers/applications "
			    "-> this application\n"
			    "  -> Bot -> Privileged Gateway Intents -> Message Content "
			    "Intent, then restart.\n"
			    "  Continuing without it: the bot now sees DMs and server "
			    "messages that\n"
			    "  mention it, e.g. \"@bot $1 + 2\".\n");
			intents &= ~(uint64_t)DISCORD_GATEWAY_MESSAGE_CONTENT;
			delay = 2;
			continue;
		}
		if (time(NULL) - started > 60)
			delay = 2;
		fprintf(stderr, "ucas-discord: the gateway loop ended: %s (code "
		    "%d); reconnecting in %u s\n", discord_strerror(rc, NULL),
		    (int)rc, delay);
		backoff_sleep(delay);
		delay = delay < 150 ? delay * 2 : 300;
	}
	ccord_global_cleanup();
	memset(tok, 0, strlen(tok));
	free(tok);
	fprintf(stderr, "ucas-discord: stopped (%s)\n", stop_requested ?
	    "signal" : sup_dead ? "evaluator died" : "token rejected");
	return status;
}

/* ---- test mode: gateway events on stdin ---- */

static int
run_stdin(void)
{
	char *line = NULL;
	size_t cap = 0;
	ssize_t len;
	int ev = 0;

	self_id = 1000;		/* the bot's own user id in test events */
	while (!stop_requested && (len = getline(&line, &cap, stdin)) > 0) {
		jsmn_parser parser;
		jsmntok_t *toks = NULL;
		unsigned ntoks = 0, npairs = 0;
		jsmnf_loader loader;
		jsmnf_pair *pairs = NULL, *t, *d;
		struct discord_message m;
		const char *js = line;
		size_t jl = (size_t)len;
		int kind;

		while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
			line[--len] = 0;
		if (len == 0 || line[0] == '#')
			continue;
		ev++;
		jsmn_init(&parser);
		if (jsmn_parse_auto(&parser, line, (size_t)len, &toks,
		    &ntoks) <= 0) {
			printf("== event %d: bad JSON\n\n", ev);
			free(toks);
			continue;
		}
		jsmnf_init(&loader);
		if (jsmnf_load_auto(&loader, line, toks, parser.toknext, &pairs,
		    &npairs) <= 0) {
			printf("== event %d: bad JSON\n\n", ev);
			free(toks);
			free(pairs);
			continue;
		}
		/* a full dispatch {"op":0,"t":...,"d":{...}} or a bare message */
		t = jsmnf_find(pairs, line, "t", 1);
		d = jsmnf_find(pairs, line, "d", 1);
		if (t != NULL && d != NULL) {
			if (t->v.len != 14 ||
			    memcmp(line + t->v.pos, "MESSAGE_CREATE", 14) != 0) {
				printf("== event %d: %.*s (not handled)\n\n", ev,
				    (int)t->v.len, line + t->v.pos);
				free(toks);
				free(pairs);
				continue;
			}
			js = line + d->v.pos;
			jl = d->v.len;
		} else {
			jl = (size_t)len;
		}
		memset(&m, 0, sizeof m);
		discord_message_from_json(js, jl, &m);
		printf("== event %d: channel %" PRIu64 ", author %" PRIu64
		    "%s: %s\n", ev, m.channel_id, m.author ? m.author->id : 0,
		    m.author && m.author->bot ? " (bot)" : "",
		    m.content ? m.content : "(no content)");
		kind = handle_message(&m);
		if (kind == DC_IGNORE)
			printf("(ignored)\n");
		discord_message_cleanup(&m);
		free(toks);
		free(pairs);
		/* wait for the answer, so the output is in a fixed order */
		while (npend > 0)
			if (!read_response()) {
				printf("evaluator process died\n");
				return 1;
			}
		printf("\n");
		fflush(stdout);
	}
	free(line);
	return 0;
}

int
main(int argc, char **argv)
{
	int rc, st, mode = 0;
	pid_t sup;

	int i;
	const char *logpath = NULL;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--selftest"))
			return dc_selftest();
		else if (!strcmp(argv[i], "--check-token"))
			return check_token();
		else if (!strcmp(argv[i], "--stdin"))
			mode = 1;
		else if (!strcmp(argv[i], "--token-file") && i + 1 < argc)
			token_file_opt = argv[++i];
		else if (!strcmp(argv[i], "--log") && i + 1 < argc)
			logpath = argv[++i];
		else if (!strcmp(argv[i], "--one-session"))
			one_session = 1;
		else {
			fprintf(stderr, "usage: ucas-discord [--token-file PATH] "
			    "[--log PATH] [--one-session] [--stdin]\n"
			    "       ucas-discord --check-token | --selftest\n");
			return 2;
		}
	}
	if (logpath != NULL && (slog = fopen(logpath, "a")) == NULL) {
		perror(logpath);
		return 1;
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	signal(SIGPIPE, SIG_IGN);
	sup_limits_env(&limits);
	/* before any thread exists: the supervisor only ever forks */
	if ((sup = sup_start(&limits, &supfd)) < 0) {
		perror("ucas-discord: supervisor");
		return 1;
	}
	rc = mode ? run_stdin() : run_live();
	close(supfd);
	waitpid(sup, &st, 0);
	return rc;
}
