/*
 * Evaluation supervisor and per-channel workers.
 *
 *   bot (threads, network)
 *    |  socketpair, struct sframe frames
 *   sup (single-threaded, forked before any thread exists; reaper)
 *    |  per channel: ctl socketpair (frames) + out pipe (raw output)
 *   worker W (holds the channel's engine state; Capsicum capability mode)
 *    |  fork per evaluation
 *   child C (rlimits; runs the input with stdout/stderr on the out pipe)
 *
 * State is committed by process succession: when C finishes it tells
 * W through a private pipe, W exits, and C carries on as the channel's
 * worker, so everything the input defined (functions, bases, models,
 * the find/more state, anything a future command adds) persists without
 * the bridge knowing what the commands are.  When C is killed (time,
 * CPU, memory) or crashes, W is untouched: the channel's state is rolled
 * back to before the failed message.
 */
#include <sys/types.h>
#include <sys/procctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#ifdef __FreeBSD__
#include <sys/capsicum.h>
#endif

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ucas.h"
#include "jsonmode.h"
#include "sup.h"

#define MAXFRAME (64u << 20)

/* ---- frames ---- */

static int
write_all(int fd, const void *p, size_t n)
{
	const char *s = p;
	ssize_t r;

	while (n > 0) {
		r = write(fd, s, n);
		if (r < 0 && errno == EINTR)
			continue;
		if (r <= 0)
			return -1;
		s += r;
		n -= (size_t)r;
	}
	return 0;
}

/* 1: all read, 0: EOF before the first byte, -1: error or short */
static int
read_all(int fd, void *p, size_t n)
{
	char *s = p;
	size_t got = 0;
	ssize_t r;

	while (got < n) {
		r = read(fd, s + got, n - got);
		if (r < 0 && errno == EINTR)
			continue;
		if (r < 0)
			return -1;
		if (r == 0)
			return got == 0 ? 0 : -1;
		got += (size_t)r;
	}
	return 1;
}

int
frame_write(int fd, const struct sframe *h, const void *p)
{
	if (write_all(fd, h, sizeof *h) < 0)
		return -1;
	if (h->len && write_all(fd, p, h->len) < 0)
		return -1;
	return 0;
}

int
frame_read(int fd, struct sframe *h, char **payload)
{
	int r;
	char *s;

	*payload = NULL;
	if ((r = read_all(fd, h, sizeof *h)) <= 0)
		return r;
	if (h->len > MAXFRAME)
		return -1;
	s = xmalloc((size_t)h->len + 1);
	if (h->len && read_all(fd, s, h->len) != 1) {
		free(s);
		return -1;
	}
	s[h->len] = 0;
	*payload = s;
	return 1;
}

/* ---- limits ---- */

static long
envnum(const char *name, long def, long lo, long hi)
{
	const char *s = getenv(name);
	char *e;
	long v;

	if (s == NULL || *s == 0)
		return def;
	v = strtol(s, &e, 10);
	if (*e != 0 || v < lo || v > hi)
		return def;
	return v;
}

void
sup_limits_env(struct sup_limits *l)
{
	l->cpu_s = (int)envnum("UCAS_DISCORD_CPU", 10, 1, 3600);
	l->wall_s = (int)envnum("UCAS_DISCORD_WALL", 20, 1, 3600);
	l->mem_mb = (int)envnum("UCAS_DISCORD_MEM", 1024, 64, 65536);
	l->max_chan = (int)envnum("UCAS_DISCORD_CHANNELS", 64, 1, 4096);
	l->max_queue = (int)envnum("UCAS_DISCORD_QUEUE", 4, 1, 100);
	l->max_out = (size_t)envnum("UCAS_DISCORD_MAXOUT", 8 << 20, 4096,
	    16 << 20);
}

/* ---- worker (W) and evaluation child (C) ---- */

static void
setlim(int res, rlim_t cur, rlim_t max)
{
	struct rlimit rl;

	rl.rlim_cur = cur;
	rl.rlim_max = max;
	(void)setrlimit(res, &rl);
}

static void __attribute__((noreturn))
worker_main(int ctl, int out, const struct sup_limits *l)
{
	struct sframe h, r;
	char *text;
	int cm[2], st, nul;
	pid_t pid;
	char c;

	/* fds: 0 /dev/null, 1 and 2 the out pipe, 3 ctl, nothing else */
	dup2(out, 1);
	dup2(out, 2);
	if ((nul = open("/dev/null", O_RDONLY)) >= 0 && nul != 0)
		dup2(nul, 0);
	if (ctl != 3) {
		dup2(ctl, 3);
		ctl = 3;
	}
	closefrom(4);
	/* line buffered: a killed evaluation still shows its last lines */
	setvbuf(stdout, NULL, _IOLBF, 1 << 16);

	setlim(RLIMIT_AS, (rlim_t)l->mem_mb << 20, (rlim_t)l->mem_mb << 20);
	setlim(RLIMIT_CORE, 0, 0);
	setlim(RLIMIT_FSIZE, 0, 0);
	/* soft limit only: lowered in each child, raised again in a worker */
	setlim(RLIMIT_CPU, RLIM_INFINITY, RLIM_INFINITY);

	/* the prelude (bases, the default cost model) belongs to every state */
	run_source("", "<init>", 0);
	fflush(stdout);
	fflush(stderr);

#ifdef __FreeBSD__
	/* no file system, no new sockets, no exec from here on */
	if (cap_enter() != 0 || (nul = open("/", O_RDONLY)) >= 0 ||
	    errno != ECAPMODE) {
		printf("ucas-discord: no capability mode; refusing to run\n");
		fflush(stdout);
		_exit(1);
	}
#endif

	for (;;) {
		if (frame_read(ctl, &h, &text) != 1)
			_exit(0);
		if (h.type != SF_EVAL || pipe(cm) != 0) {
			free(text);
			continue;
		}
		pid = fork();
		if (pid == 0) {
			/* C: evaluate, then succeed W as the channel's worker */
			close(cm[0]);
			setlim(RLIMIT_CPU, (rlim_t)l->cpu_s, RLIM_INFINITY);
			memset(&r, 0, sizeof r);
			r.type = SF_PID;
			r.id = (uint64_t)getpid();
			if (frame_write(ctl, &r, NULL) < 0)
				_exit(1);
			if (h.arg & SF_JSON)
				json_bot_serve(text);
			else
				run_text(text, "input", (int)(h.arg & 1));
			fflush(stdout);
			fflush(stderr);
			free(text);
			r.type = SF_DONE;
			if (frame_write(ctl, &r, NULL) < 0)
				_exit(1);
			c = 1;
			(void)write(cm[1], &c, 1);
			close(cm[1]);
			setlim(RLIMIT_CPU, RLIM_INFINITY, RLIM_INFINITY);
			continue;
		}
		free(text);
		close(cm[1]);
		if (pid < 0) {
			close(cm[0]);
			memset(&r, 0, sizeof r);
			r.type = SF_FAIL;
			r.status = 0x7f00;	/* "exited 127" */
			if (frame_write(ctl, &r, NULL) < 0)
				_exit(1);
			continue;
		}
		c = 0;
		while (read(cm[0], &c, 1) < 0 && errno == EINTR)
			;
		if (c == 1)
			_exit(0);	/* committed: C is the worker now */
		close(cm[0]);
		while (waitpid(pid, &st, 0) < 0 && errno == EINTR)
			;
		memset(&r, 0, sizeof r);
		r.type = SF_FAIL;
		r.status = (uint32_t)st;
		if (frame_write(ctl, &r, NULL) < 0)
			_exit(1);
	}
}

/* ---- supervisor ---- */

struct req {
	struct req *next;
	uint64_t id;
	uint32_t echo;
	char *text;
};

struct slot {
	uint64_t chan;
	int ctl, out;		/* ctl < 0: free */
	pid_t wpid, cpid;
	int busy, timedout;
	uint64_t id;
	long long deadline;	/* ms, CLOCK_MONOTONIC */
	struct sbuf ob;
	int trunc;
	struct req *qh, *qt;
	int nq;
	uint64_t lru;
};

static const struct sup_limits *lim;
static struct slot *slots;
static int nslots;
static int botfd = -1;
static uint64_t lrutick;

static long long
now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void
respond(uint64_t id, uint64_t chan, uint32_t status, uint32_t arg,
    const char *p, size_t n)
{
	struct sframe h;

	memset(&h, 0, sizeof h);
	h.type = SF_RESP;
	h.id = id;
	h.chan = chan;
	h.status = status;
	h.arg = arg;
	h.len = (uint32_t)n;
	if (frame_write(botfd, &h, p) < 0)
		_exit(0);	/* the bot is gone */
}

static void
drain(struct slot *s)
{
	char buf[65536];
	ssize_t r;
	size_t room;

	for (;;) {
		r = read(s->out, buf, sizeof buf);
		if (r < 0 && errno == EINTR)
			continue;
		if (r <= 0)
			return;
		room = lim->max_out > s->ob.n ? lim->max_out - s->ob.n : 0;
		if ((size_t)r > room) {
			s->trunc = 1;
			r = (ssize_t)room;
		}
		if (r > 0 && s->busy) {
			if (s->ob.n + (size_t)r + 1 > s->ob.cap) {
				s->ob.cap = s->ob.n + (size_t)r + 1 + 4096;
				s->ob.p = xrealloc(s->ob.p, s->ob.cap);
			}
			memcpy(s->ob.p + s->ob.n, buf, (size_t)r);
			s->ob.n += (size_t)r;
			s->ob.p[s->ob.n] = 0;
		}
	}
}

static void
slot_free(struct slot *s, uint32_t qstatus)
{
	struct req *q;

	if (s->busy)
		respond(s->id, s->chan, qstatus, 0, NULL, 0);
	while ((q = s->qh) != NULL) {
		s->qh = q->next;
		respond(q->id, s->chan, qstatus, 0, NULL, 0);
		free(q->text);
		free(q);
	}
	if (s->cpid > 0)
		kill(s->cpid, SIGKILL);
	if (s->wpid > 0)
		kill(s->wpid, SIGKILL);
	close(s->ctl);
	close(s->out);
	free(s->ob.p);
	memset(s, 0, sizeof *s);
	s->ctl = s->out = -1;
}

static struct slot *
slot_new(uint64_t chan)
{
	struct slot *s = NULL, *old = NULL;
	int i, sp[2], op[2];
	pid_t pid;

	for (i = 0; i < nslots; i++) {
		if (slots[i].ctl < 0) {
			s = &slots[i];
			break;
		}
		if (!slots[i].busy && slots[i].nq == 0 &&
		    (old == NULL || slots[i].lru < old->lru))
			old = &slots[i];
	}
	if (s == NULL) {
		if (old == NULL)
			return NULL;	/* every worker is busy */
		slot_free(old, SR_LOST);
		s = old;
	}
	if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0)
		return NULL;
	if (pipe(op) != 0) {
		close(sp[0]);
		close(sp[1]);
		return NULL;
	}
	pid = fork();
	if (pid < 0) {
		close(sp[0]);
		close(sp[1]);
		close(op[0]);
		close(op[1]);
		return NULL;
	}
	if (pid == 0)
		worker_main(sp[1], op[1], lim);	/* closes everything else */
	close(sp[1]);
	close(op[1]);
	fcntl(op[0], F_SETFL, fcntl(op[0], F_GETFL) | O_NONBLOCK);
	fcntl(op[0], F_SETFD, FD_CLOEXEC);
	fcntl(sp[0], F_SETFD, FD_CLOEXEC);
	s->chan = chan;
	s->ctl = sp[0];
	s->out = op[0];
	s->wpid = pid;
	return s;
}

static struct slot *
slot_find(uint64_t chan)
{
	int i;

	for (i = 0; i < nslots; i++)
		if (slots[i].ctl >= 0 && slots[i].chan == chan)
			return &slots[i];
	return NULL;
}

static void
dispatch(struct slot *s)
{
	struct req *q = s->qh;
	struct sframe h;
	int r;

	if (s->busy || q == NULL)
		return;
	s->qh = q->next;
	if (s->qh == NULL)
		s->qt = NULL;
	s->nq--;
	memset(&h, 0, sizeof h);
	h.type = SF_EVAL;
	h.arg = q->echo;
	h.len = (uint32_t)strlen(q->text);
	s->busy = 1;
	s->id = q->id;
	s->timedout = 0;
	s->cpid = 0;
	s->trunc = 0;
	sb_clear(&s->ob);
	s->deadline = now_ms() + (long long)lim->wall_s * 1000;
	r = frame_write(s->ctl, &h, q->text);
	free(q->text);
	free(q);
	if (r < 0)
		slot_free(s, SR_LOST);
}

static void
on_bot_frame(const struct sframe *h, char *text)
{
	struct slot *s = slot_find(h->chan);
	struct req *q;

	if (h->type == SF_RESET) {
		if (s != NULL)
			slot_free(s, SR_LOST);
		respond(h->id, h->chan, SR_RESET, 0, NULL, 0);
		free(text);
		return;
	}
	if (h->type != SF_EVAL) {
		free(text);
		return;
	}
	if (s == NULL && (s = slot_new(h->chan)) == NULL) {
		respond(h->id, h->chan, SR_BUSY, 0, NULL, 0);
		free(text);
		return;
	}
	if (s->nq >= lim->max_queue) {
		respond(h->id, h->chan, SR_BUSY, 0, NULL, 0);
		free(text);
		return;
	}
	s->lru = ++lrutick;
	q = xcalloc(1, sizeof *q);
	q->id = h->id;
	q->echo = h->arg;
	q->text = text;
	if (s->qt)
		s->qt->next = q;
	else
		s->qh = q;
	s->qt = q;
	s->nq++;
	dispatch(s);
}

static void
finish(struct slot *s, uint32_t status, uint32_t arg)
{
	drain(s);
	if (s->trunc)
		status |= SR_TRUNC;
	respond(s->id, s->chan, status, arg, s->ob.p, s->ob.n);
	s->busy = 0;
	s->cpid = 0;
	sb_clear(&s->ob);
	dispatch(s);
}

static void
on_worker_frame(struct slot *s)
{
	struct sframe h;
	char *p;
	int st;

	if (frame_read(s->ctl, &h, &p) != 1) {
		drain(s);
		slot_free(s, SR_LOST);
		return;
	}
	free(p);
	switch (h.type) {
	case SF_PID:
		if (s->busy)
			s->cpid = (pid_t)h.id;
		break;
	case SF_DONE:
		if (!s->busy)
			break;
		s->wpid = s->cpid;	/* the child is the worker now */
		finish(s, SR_OK, 0);
		break;
	case SF_FAIL:
		if (!s->busy) {
			/* the child died after committing: the old worker
			 * goes on serving, its pid is no longer known */
			s->wpid = 0;
			break;
		}
		st = (int)h.status;
		if (s->timedout)
			finish(s, SR_TIMEOUT, 0);
		else if (WIFSIGNALED(st) && WTERMSIG(st) == SIGXCPU)
			finish(s, SR_CPU, 0);
		else if (WIFEXITED(st) && WEXITSTATUS(st) == 2)
			finish(s, SR_MEM, 0);	/* xmalloc's exit(2) */
		else if (WIFSIGNALED(st))
			finish(s, SR_CRASH, (uint32_t)WTERMSIG(st));
		else
			finish(s, SR_CRASH, 1000u + (uint32_t)WEXITSTATUS(st));
		break;
	}
}

static void
reap(void)
{
	pid_t pid;
	int st, i;

	while ((pid = waitpid(-1, &st, WNOHANG)) > 0)
		for (i = 0; i < nslots; i++) {
			if (slots[i].wpid == pid)
				slots[i].wpid = 0;
			if (slots[i].cpid == pid)
				slots[i].cpid = 0;
		}
}

static void __attribute__((noreturn))
sup_main(void)
{
	struct pollfd *pf;
	struct slot **ps;
	struct sframe h;
	char *text;
	long long t, next;
	int i, n, tmo;

	signal(SIGPIPE, SIG_IGN);
	/* orphaned evaluation children are re-parented here and reaped */
#ifdef PROC_REAP_ACQUIRE
	(void)procctl(P_PID, getpid(), PROC_REAP_ACQUIRE, NULL);
#endif
	slots = xcalloc((size_t)lim->max_chan, sizeof *slots);
	nslots = lim->max_chan;
	for (i = 0; i < nslots; i++)
		slots[i].ctl = slots[i].out = -1;
	pf = xcalloc((size_t)(2 * nslots + 1), sizeof *pf);
	ps = xcalloc((size_t)(2 * nslots + 1), sizeof *ps);

	for (;;) {
		reap();
		n = 0;
		pf[n].fd = botfd;
		pf[n].events = POLLIN;
		ps[n++] = NULL;
		next = -1;
		t = now_ms();
		for (i = 0; i < nslots; i++) {
			struct slot *s = &slots[i];
			if (s->ctl < 0)
				continue;
			if (s->busy && !s->timedout && t >= s->deadline) {
				s->timedout = 1;
				if (s->cpid > 0)
					kill(s->cpid, SIGKILL);
				else {
					/* never started: give up on it */
					slot_free(s, SR_LOST);
					continue;
				}
			}
			if (s->busy && !s->timedout &&
			    (next < 0 || s->deadline < next))
				next = s->deadline;
			pf[n].fd = s->ctl;
			pf[n].events = POLLIN;
			ps[n++] = s;
			pf[n].fd = s->out;
			pf[n].events = POLLIN;
			ps[n++] = s;
		}
		tmo = next < 0 ? 1000 : (int)(next - t + 1);
		if (tmo > 1000)
			tmo = 1000;	/* also polls for zombies */
		if (poll(pf, (nfds_t)n, tmo) < 0) {
			if (errno == EINTR)
				continue;
			_exit(1);
		}
		/* output first, so a DONE frame sees all of it */
		for (i = 1; i < n; i++)
			if (pf[i].revents && ps[i]->out == pf[i].fd)
				drain(ps[i]);
		for (i = 1; i < n; i++)
			if (pf[i].revents && ps[i]->ctl == pf[i].fd)
				on_worker_frame(ps[i]);
		if (pf[0].revents) {
			if (frame_read(botfd, &h, &text) != 1) {
				for (i = 0; i < nslots; i++)
					if (slots[i].ctl >= 0) {
						slots[i].busy = 0;
						slots[i].nq = 0;
						slots[i].qh = NULL;
						slot_free(&slots[i], SR_LOST);
					}
				_exit(0);
			}
			on_bot_frame(&h, text);
		}
	}
}

pid_t
sup_start(const struct sup_limits *l, int *fd)
{
	static struct sup_limits copy;
	int sv[2], sz = 4 << 20;
	char *tok;
	pid_t pid;

	if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
		return -1;
	setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &sz, sizeof sz);
	setsockopt(sv[0], SOL_SOCKET, SO_RCVBUF, &sz, sizeof sz);
	setsockopt(sv[1], SOL_SOCKET, SO_SNDBUF, &sz, sizeof sz);
	setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &sz, sizeof sz);
	fflush(stdout);
	fflush(stderr);
	pid = fork();
	if (pid < 0) {
		close(sv[0]);
		close(sv[1]);
		return -1;
	}
	if (pid == 0) {
		/* the token must not reach the evaluation processes */
		if ((tok = getenv("DISCORD_TOKEN")) != NULL)
			memset(tok, 0, strlen(tok));
		unsetenv("DISCORD_TOKEN");
		close(sv[0]);
		copy = *l;
		lim = &copy;
		botfd = sv[1];
		sup_main();
	}
	close(sv[1]);
	fcntl(sv[0], F_SETFD, FD_CLOEXEC);
	*fd = sv[0];
	return pid;
}
