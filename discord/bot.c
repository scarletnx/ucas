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
	u64snowflake chan, msg, guild;
};

static struct discord *client;	/* NULL in --stdin mode */
static struct sup_limits limits;
static int supfd = -1;
static u64snowflake self_id;
static struct pend *pend;
static int npend, cappend;
static uint64_t nextid;

/* ---- outgoing ---- */

static void
post(u64snowflake chan, u64snowflake msg, u64snowflake guild,
    const char *content, const char *file, size_t filelen)
{
	char *noparse[1] = {NULL};
	struct strings parse = {0, noparse, 0};
	struct discord_allowed_mention am = {0};
	struct discord_message_reference ref = {0};
	struct discord_attachment att = {0};
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
	if (file != NULL) {
		att.content = (char *)file;
		att.size = filelen;
		att.filename = DC_FILENAME;
		att.content_type = "text/plain; charset=utf-8";
		atts.size = 1;
		atts.array = &att;
		p.attachments = &atts;
	}
	if (client != NULL) {
		discord_create_message(client, chan, &p, NULL);
		return;
	}
	n = discord_create_message_to_json(buf, sizeof buf, &p);
	printf("POST /channels/%" PRIu64 "/messages%s\n%.*s\n", chan,
	    file ? " (multipart)" : "", (int)n, buf);
	if (file != NULL)
		printf("  files[0]: %s, %zu bytes\n", DC_FILENAME, filelen);
}

/* ---- requests ---- */

static int
send_req(uint32_t type, const struct discord_message *m, const char *text,
    int echo)
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
	return 0;
}

/* returns DC_IGNORE, DC_EVAL or DC_RESET */
/*
 * "<@id> $expr" or "<@!id> $expr": without the Message Content intent,
 * Discord only delivers the text of server messages that mention the bot.
 */
static const char *
skip_mention(const char *s)
{
	char id[32];
	size_t n;

	if (s == NULL || self_id == 0)
		return s;
	n = (size_t)snprintf(id, sizeof id, "%" PRIu64 ">", self_id);
	if (!strncmp(s, "<@", 2)) {
		const char *p = s + 2;
		if (*p == '!')
			p++;
		if (!strncmp(p, id, n)) {
			p += n;
			while (*p == ' ' || *p == '\t' || *p == '\n')
				p++;
			return p;
		}
	}
	return s;
}

static int
handle_message(const struct discord_message *m)
{
	struct sbuf code = {0};
	int kind, echo;

	if (m->author == NULL || m->author->bot || m->author->System ||
	    m->author->id == self_id || m->webhook_id != 0)
		return DC_IGNORE;
	kind = dc_command(skip_mention(m->content), &code, &echo);
	if (kind == DC_EVAL)
		send_req(SF_EVAL, m, code.p, echo);
	else if (kind == DC_RESET)
		send_req(SF_RESET, m, NULL, 0);
	free(code.p);
	return kind;
}

static void
on_response(const struct sframe *h, const char *out)
{
	struct dc_reply r;
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
	dc_format(&r, head, out, h->len);
	for (i = 0; i < r.n; i++)
		post(pp.chan, pp.msg, pp.guild, r.msg[i],
		    i == 0 ? r.file : NULL, r.filelen);
	dc_reply_free(&r);
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
	while ((len = getline(&line, &cap, stdin)) > 0) {
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

	if (argc == 2 && !strcmp(argv[1], "--selftest"))
		return dc_selftest();
	if (argc == 2 && !strcmp(argv[1], "--check-token"))
		return check_token();
	if (argc == 2 && !strcmp(argv[1], "--stdin"))
		mode = 1;
	else if (argc == 3 && !strcmp(argv[1], "--token-file"))
		token_file_opt = argv[2];
	else if (argc != 1) {
		fprintf(stderr, "usage: ucas-discord [--token-file PATH] | "
		    "--check-token | --stdin | --selftest\n");
		return 2;
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
