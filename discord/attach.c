/*
 * What a message points at besides its text: its attachments, and for a
 * Discord reply, the replied-to message's code blocks and attachments
 * ("@Wimsy animate exec of this" on a shader someone posted).  Each one
 * becomes a code block appended to the request:
 *
 *   binary SPIR-V       spirv-dis, as a ```spv block
 *   GLSL (.comp, .frag, .vert, .glsl ...)
 *                       glslangValidator -V, then spirv-dis
 *   SPIR-V assembly     as it is (```spv)
 *   other text          as it is
 *   images              a note: paste the code as text
 *
 * Downloads come only from Discord's CDN (file:// in --stdin tests),
 * capped in size and time; the tools run with CPU and memory limits.
 */
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/wait.h>

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include <curl/curl.h>

#include "discord.h"

#include "ucas.h"
#include "attach.h"

#define AT_MAXBYTES (256 * 1024)
#define AT_MAXFILES 4
#define AT_MAXTEXT (64 * 1024)	/* of tool output kept */

int attach_allow_file;		/* --stdin: file:// URLs (tests) */

struct buf {
	char *p;
	size_t n;
	int over;
};

static size_t
on_data(char *d, size_t sz, size_t nm, void *u)
{
	struct buf *b = u;
	size_t n = sz * nm;

	if (b->n + n > AT_MAXBYTES) {
		b->over = 1;
		return 0;
	}
	b->p = realloc(b->p, b->n + n + 1);
	if (b->p == NULL)
		return 0;
	memcpy(b->p + b->n, d, n);
	b->n += n;
	b->p[b->n] = 0;
	return n;
}

static int
url_ok(const char *u)
{
	if (!strncmp(u, "https://cdn.discordapp.com/", 27) ||
	    !strncmp(u, "https://media.discordapp.net/", 29))
		return 1;
	return attach_allow_file && !strncmp(u, "test:", 5);
}

/* the attachment's bytes, or NULL with why */
static char *
fetch(const char *url, size_t *len, const char **why)
{
	struct buf b = {0};
	CURL *c;
	CURLcode rc;
	long code = 0;

	if (!url_ok(url)) {
		*why = "not on Discord's CDN";
		return NULL;
	}
	if (attach_allow_file && !strncmp(url, "test:", 5)) {
		/* tests: a path relative to the working directory */
		static char fu[PATH_MAX + 8];
		char rp[PATH_MAX];
		if (realpath(url + 5, rp) == NULL) {
			*why = "no such test file";
			return NULL;
		}
		snprintf(fu, sizeof fu, "file://%s", rp);
		url = fu;
	}
	if ((c = curl_easy_init()) == NULL) {
		*why = "no curl";
		return NULL;
	}
	curl_easy_setopt(c, CURLOPT_URL, url);
	curl_easy_setopt(c, CURLOPT_PROTOCOLS_STR, attach_allow_file ?
	    "https,file" : "https");
	curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 0L);
	curl_easy_setopt(c, CURLOPT_TIMEOUT, 10L);
	curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_data);
	curl_easy_setopt(c, CURLOPT_WRITEDATA, &b);
	rc = curl_easy_perform(c);
	curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
	curl_easy_cleanup(c);
	if (b.over) {
		free(b.p);
		*why = "larger than 256 KiB";
		return NULL;
	}
	if (rc != CURLE_OK || (code != 0 && code != 200)) {
		free(b.p);
		*why = "the download failed";
		return NULL;
	}
	if (b.p == NULL)
		b.p = calloc(1, 1);
	*len = b.n;
	return b.p;
}

/*
 * Run argv with limits; its stdout (and stderr) into out.  Returns the
 * exit status, -1 if it could not run.
 */
static int
run_tool(char *const argv[], struct sbuf *out)
{
	int fd[2], st = 0;
	pid_t pid;
	char tmp[4096];
	ssize_t r;

	if (pipe(fd) < 0)
		return -1;
	if ((pid = fork()) < 0) {
		close(fd[0]);
		close(fd[1]);
		return -1;
	}
	if (pid == 0) {
		struct rlimit cpu = {5, 5}, mem = {512UL << 20, 512UL << 20},
		    fsz = {8UL << 20, 8UL << 20};
		int nul = open("/dev/null", O_RDONLY);
		setrlimit(RLIMIT_CPU, &cpu);
		setrlimit(RLIMIT_AS, &mem);
		setrlimit(RLIMIT_FSIZE, &fsz);
		if (nul >= 0)
			dup2(nul, 0);
		dup2(fd[1], 1);
		dup2(fd[1], 2);
		close(fd[0]);
		close(fd[1]);
		execvp(argv[0], argv);
		_exit(127);
	}
	close(fd[1]);
	while ((r = read(fd[0], tmp, sizeof tmp)) > 0)
		if (out->n < AT_MAXTEXT)
			sb_printf(out, "%.*s", (int)r, tmp);
	close(fd[0]);
	while (waitpid(pid, &st, 0) < 0)
		;
	return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

static int
write_tmp(char *path, const char *data, size_t n)
{
	int fd = mkstemps(path, (int)strlen(strrchr(path, '.')));

	if (fd < 0)
		return -1;
	if (write(fd, data, n) != (ssize_t)n) {
		close(fd);
		unlink(path);
		return -1;
	}
	close(fd);
	return 0;
}

/* binary SPIR-V to spirv-dis text; 0 on failure (why in out) */
static int
disassemble(const char *data, size_t n, struct sbuf *out)
{
	char path[] = "/tmp/ucas-att-XXXXXX.spv";
	char *argv[] = {"spirv-dis", "--no-color", path, NULL};
	int rc;

	if (write_tmp(path, data, n) < 0)
		return 0;
	rc = run_tool(argv, out);
	unlink(path);
	return rc == 0;
}

static const char *
ext_of(const char *name)
{
	const char *d = name ? strrchr(name, '.') : NULL;

	return d ? d + 1 : "";
}

static int
is_glsl_ext(const char *e)
{
	static const char *const g[] = {"glsl", "comp", "frag", "vert", "geom",
	    "tesc", "tese", "rgen", "rchit", "rmiss", "mesh", "task", NULL};
	int k;

	for (k = 0; g[k]; k++)
		if (!strcasecmp(e, g[k]))
			return 1;
	return 0;
}

/* a GLSL stage from the file name, else from what the code uses */
static const char *
glsl_stage(const char *e, const char *src)
{
	if (strcasecmp(e, "glsl") != 0)
		return e;
	if (strstr(src, "local_size_x") || strstr(src, "gl_GlobalInvocationID")
	    || strstr(src, "gl_LocalInvocationID") || strstr(src, "shared "))
		return "comp";
	if (strstr(src, "gl_Position"))
		return "vert";
	return "frag";
}

/* GLSL to SPIR-V to text; 0 on failure (the compiler's errors in out) */
static int
compile_glsl(const char *e, const char *data, size_t n, struct sbuf *out)
{
	char src[] = "/tmp/ucas-att-XXXXXX.glsl", spv[] = "/tmp/ucas-att-XXXXXX.spv";
	char stage[16];
	char *cc[] = {"glslangValidator", "-V", "-S", stage, "-o", spv, src,
	    NULL};
	char *dis[] = {"spirv-dis", "--no-color", spv, NULL};
	struct sbuf log = {0};
	int ok = 0;

	snprintf(stage, sizeof stage, "%s", glsl_stage(e, data));
	if (write_tmp(src, data, n) < 0)
		return 0;
	if (write_tmp(spv, "", 0) < 0) {
		unlink(src);
		return 0;
	}
	if (run_tool(cc, &log) == 0)
		ok = run_tool(dis, out) == 0;
	else {
		/* the error lines, not the banner */
		const char *l = log.p ? log.p : "";
		while (*l) {
			const char *nl = strchr(l, '\n');
			size_t len = nl ? (size_t)(nl - l) : strlen(l);
			if (!strncmp(l, "ERROR:", 6))
				sb_printf(out, "%.*s\n", (int)len, l);
			l += len + (nl != NULL);
		}
	}
	free(log.p);
	unlink(src);
	unlink(spv);
	return ok;
}

static int
is_text(const char *p, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++)
		if (p[i] == 0)
			return 0;
	return 1;
}

static void
add_block(struct sbuf *blocks, const char *lang, const char *s, size_t n)
{
	while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r'))
		n--;
	sb_printf(blocks, "\n```%s\n", lang);
	for (; n > 0; s++, n--)		/* sb_printf's buffer is bounded */
		sb_putc(blocks, *s);
	sb_puts(blocks, "\n```");
}

/* the SPIR-V magic number, either byte order, and a whole header */
static int
is_spirv(const char *d, size_t n)
{
	const unsigned char *u = (const unsigned char *)d;

	if (n < 20)
		return 0;
	return (u[0] == 0x03 && u[1] == 0x02 && u[2] == 0x23 && u[3] == 0x07) ||
	    (u[0] == 0x07 && u[1] == 0x23 && u[2] == 0x02 && u[3] == 0x03);
}

static void
one_attachment(const struct discord_attachment *a, struct sbuf *blocks,
    struct sbuf *notes)
{
	const char *name = a->filename ? a->filename : "attachment";
	const char *e = ext_of(name), *why = NULL;
	char *d;
	size_t n = 0;

	if (a->content_type && !strncmp(a->content_type, "image/", 6)) {
		sb_printf(notes, "I can't read the image %s: paste its code as "
		    "text in a ``` block, or attach the file\n", name);
		return;
	}
	if (a->size > AT_MAXBYTES) {
		sb_printf(notes, "%s is larger than 256 KiB: skipped\n", name);
		return;
	}
	if (a->url == NULL || (d = fetch(a->url, &n, &why)) == NULL) {
		sb_printf(notes, "could not read %s (%s)\n", name,
		    why ? why : "no URL");
		return;
	}
	if (is_spirv(d, n)) {
		struct sbuf t = {0};
		if (disassemble(d, n, &t)) {
			sb_printf(notes, "%s: binary SPIR-V, disassembled\n", name);
			add_block(blocks, "spv", t.p ? t.p : "", t.n);
		} else
			sb_printf(notes, "%s: SPIR-V that spirv-dis rejects: %.200s\n",
			    name, t.p ? t.p : "");
		free(t.p);
	} else if (!is_text(d, n))
		sb_printf(notes, "%s is binary and not SPIR-V: skipped\n", name);
	else if (is_glsl_ext(e)) {
		struct sbuf t = {0};
		if (compile_glsl(e, d, n, &t)) {
			sb_printf(notes, "%s: GLSL (%s), compiled to SPIR-V with "
			    "glslangValidator\n", name, glsl_stage(e, d));
			add_block(blocks, "spv", t.p ? t.p : "", t.n);
		} else
			sb_printf(notes, "%s: GLSL that does not compile:\n%.600s",
			    name, t.p ? t.p : "");
		free(t.p);
	} else if (!strcasecmp(e, "spvasm") || !strcasecmp(e, "spv") ||
	    strstr(d, "OpFunction") || strstr(d, "OpLabel"))
		add_block(blocks, "spv", d, n);
	else
		add_block(blocks, "", d, n);
	free(d);
}

/* the ``` blocks of a message's text, fences included */
static void
text_blocks(const char *s, struct sbuf *blocks)
{
	const char *a, *b;

	while (s && (a = strstr(s, "```")) != NULL &&
	    (b = strstr(a + 3, "```")) != NULL) {
		sb_printf(blocks, "\n%.*s", (int)(b + 3 - a), a);
		s = b + 3;
	}
}

int
attach_gather(const struct discord_message *m, uint64_t self,
    int have_block, struct sbuf *blocks, struct sbuf *notes)
{
	const struct discord_message *r = m->referenced_message;
	int k, nf = 0;
	size_t before = blocks->n;

	if (m->attachments)
		for (k = 0; k < m->attachments->size && nf < AT_MAXFILES; k++, nf++)
			one_attachment(&m->attachments->array[k], blocks, notes);
	/*
	 * A reply to someone's message: what it holds.  A reply to the bot
	 * is a follow-up, its output is not input.
	 */
	if (r != NULL && !(r->author && r->author->id == self)) {
		if (!have_block && blocks->n == before)
			text_blocks(r->content, blocks);
		if (r->attachments)
			for (k = 0; k < r->attachments->size && nf < AT_MAXFILES;
			    k++, nf++)
				one_attachment(&r->attachments->array[k], blocks,
				    notes);
	}
	return blocks->n > before;
}
