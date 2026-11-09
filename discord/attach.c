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
 *   C (.c, .h)          as it is, as a ```c block (the C frontend)
 *   other text          as it is (C by its content: ```c)
 *   images (PNG, JPEG, GIF, BMP, PPM)
 *                       a ```image NAME block of base64: the shader's
 *                       image inputs, in attachment order (animate exec)
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
#include "img.h"

#define AT_MAXBYTES (256 * 1024)	/* code */
#define AT_MAXIMAGE (8 << 20)		/* an image (decoded: capped by img.c) */
#define AT_MAXFILES 10
#define AT_MAXTEXT (64 * 1024)	/* of tool output kept */

int attach_allow_file;		/* --stdin: file:// URLs (tests) */

struct buf {
	char *p;
	size_t n, cap;
	int over;
};

static size_t
on_data(char *d, size_t sz, size_t nm, void *u)
{
	struct buf *b = u;
	size_t n = sz * nm;

	if (b->n + n > b->cap) {
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

/* the attachment's bytes (at most cap), or NULL with why */
static char *
fetch(const char *url, size_t cap, size_t *len, const char **why)
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
	b.cap = cap;
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
		*why = cap > AT_MAXBYTES ? "larger than 8 MiB" :
		    "larger than 256 KiB";
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

static int
is_c_ext(const char *e)
{
	static const char *const c[] = {"c", "h", "cc", "cpp", "cxx", "hpp",
	    NULL};
	int k;

	for (k = 0; c[k]; k++)
		if (!strcasecmp(e, c[k]))
			return 1;
	return 0;
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

static int
is_image_name(const struct discord_attachment *a, const char *e)
{
	static const char *const x[] = {"png", "jpg", "jpeg", "gif", "bmp",
	    "ppm", "pgm", "pnm", "webp", "qoi", NULL};
	int k;

	if (a->content_type && !strncmp(a->content_type, "image/", 6))
		return 1;
	for (k = 0; x[k]; k++)
		if (!strcasecmp(e, x[k]))
			return 1;
	return 0;
}

/* an image: a ```image block of base64, decoded by the engine */
static void
image_block(struct sbuf *blocks, const char *name, const unsigned char *d,
    size_t n)
{
	static const char t[] =
	    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	size_t i;
	int col = 0;
	char nm[96];

	/* the name on the fence line: no blanks or backticks */
	snprintf(nm, sizeof nm, "%s", name);
	for (i = 0; nm[i]; i++)
		if (isspace((unsigned char)nm[i]) || nm[i] == '`' ||
		    nm[i] == '\x1e')
			nm[i] = '_';
	sb_printf(blocks, "\n```image %s\n", nm);
	for (i = 0; i < n; i += 3) {
		unsigned v = (unsigned)d[i] << 16 | (i + 1 < n ? d[i + 1] << 8 : 0) |
		    (i + 2 < n ? d[i + 2] : 0);
		sb_putc(blocks, t[v >> 18 & 63]);
		sb_putc(blocks, t[v >> 12 & 63]);
		sb_putc(blocks, i + 1 < n ? t[v >> 6 & 63] : '=');
		sb_putc(blocks, i + 2 < n ? t[v & 63] : '=');
		if ((col += 4) >= 76) {
			sb_putc(blocks, '\n');
			col = 0;
		}
	}
	sb_puts(blocks, "\n```");
}

/* one attachment as a block; 1 when it was code (not an image) */
static int
one_attachment(const struct discord_attachment *a, struct sbuf *blocks,
    struct sbuf *notes)
{
	const char *name = a->filename ? a->filename : "attachment";
	const char *e = ext_of(name), *why = NULL;
	char *d;
	size_t n = 0;
	int img = is_image_name(a, e);

	if (a->size > (img ? AT_MAXIMAGE : AT_MAXBYTES)) {
		sb_printf(notes, "%s is larger than %s: skipped\n", name, img ?
		    "8 MiB" : "256 KiB");
		return 0;
	}
	if (a->url == NULL || (d = fetch(a->url, img ? AT_MAXIMAGE :
	    AT_MAXBYTES, &n, &why)) == NULL) {
		sb_printf(notes, "could not read %s (%s)\n", name,
		    why ? why : "no URL");
		return 0;
	}
	if (img || img_sniff((const unsigned char *)d, n) != NULL) {
		const char *f = img_sniff((const unsigned char *)d, n);
		if (f == NULL)
			sb_printf(notes, "%s is not an image file I can read (PNG, "
			    "JPEG, GIF, BMP, PPM): skipped\n", name);
		else if (!strcmp(f, "QOI") || !strcmp(f, "WebP"))
			sb_printf(notes, "%s: %s images are not read (PNG, JPEG, "
			    "GIF, BMP, PPM are): skipped\n", name, f);
		else
			image_block(blocks, name, (const unsigned char *)d, n);
		free(d);
		return 0;
	}
	if (n > AT_MAXBYTES) {
		sb_printf(notes, "%s is larger than 256 KiB: skipped\n", name);
		free(d);
		return 0;
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
	else if (is_c_ext(e) || (!is_glsl_ext(e) && strcasecmp(e, "spvasm") &&
	    code_lang(d, "", NULL) == CL_C)) {
		sb_printf(notes, "%s: C, for the C frontend\n", name);
		add_block(blocks, "c", d, n);
	} else if (is_glsl_ext(e) || glsl_like(d, "")) {
		struct sbuf t = {0}, why = {0};
		if (glsl_to_spvasm(d, &t, &why)) {
			sb_printf(notes, "%s: GLSL, compiled to SPIR-V with "
			    "glslangValidator\n%s", name, why.p ? why.p : "");
			add_block(blocks, "spv", t.p ? t.p : "", t.n);
		} else
			sb_printf(notes, "%s: GLSL that does not compile:\n%.600s",
			    name, why.p ? why.p : "");
		free(t.p);
		free(why.p);
	} else if (!strcasecmp(e, "spvasm") || !strcasecmp(e, "spv") ||
	    strstr(d, "OpFunction") || strstr(d, "OpLabel"))
		add_block(blocks, "spv", d, n);
	else
		add_block(blocks, "", d, n);
	free(d);
	return 1;
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

/*
 * The message's own ```glsl blocks (or untagged GLSL): compiled to
 * ```spv blocks in place, since the engine's sandbox cannot run the
 * compiler
 */
int
attach_convert(struct sbuf *code, struct sbuf *notes)
{
	struct sbuf out = {0};
	const char *p = code->p, *a, *b;
	int conv = 0;

	if (p == NULL)
		return 0;
	while ((a = strstr(p, "```")) != NULL &&
	    (b = strstr(a + 3, "```")) != NULL) {
		const char *body = a + 3, *nl;
		char lang[24] = "";
		size_t ln = 0;
		while (body < b && !isspace((unsigned char)*body) &&
		    ln + 1 < sizeof lang)
			lang[ln++] = (char)tolower((unsigned char)*body++);
		lang[ln] = 0;
		nl = memchr(body, '\n', (size_t)(b - body));
		sb_printf(&out, "%.*s", (int)(a - p), p);
		if (nl != NULL) {
			char *src = xmalloc((size_t)(b - nl));
			struct sbuf spv = {0}, why = {0};
			memcpy(src, nl + 1, (size_t)(b - nl - 1));
			src[b - nl - 1] = 0;
			if (glsl_like(src, lang) && glsl_to_spvasm(src, &spv, &why)) {
				sb_printf(notes, "the ```%s block: GLSL, compiled to SPIR-V "
				    "with glslangValidator\n%s", lang[0] ? lang : "",
				    why.p ? why.p : "");
				sb_printf(&out, "```spv\n%s```", spv.p ? spv.p : "");
				conv = 1;
			} else {
				if (glsl_like(src, lang))
					sb_printf(notes, "the GLSL block does not compile:\n"
					    "%.600s", why.p ? why.p : "");
				sb_printf(&out, "%.*s", (int)(b + 3 - a), a);
			}
			free(src);
			free(spv.p);
			free(why.p);
		} else
			sb_printf(&out, "%.*s", (int)(b + 3 - a), a);
		p = b + 3;
	}
	sb_puts(&out, p);
	free(code->p);
	*code = out;
	return conv;
}

int
attach_gather(const struct discord_message *m, uint64_t self,
    int have_block, struct sbuf *blocks, struct sbuf *notes)
{
	const struct discord_message *r = m->referenced_message;
	int k, nf = 0, code = have_block;
	size_t before = blocks->n;

	if (m->attachments)
		for (k = 0; k < m->attachments->size && nf < AT_MAXFILES; k++, nf++)
			code |= one_attachment(&m->attachments->array[k], blocks,
			    notes);
	/*
	 * A reply to someone's message: what it holds (the shader, or a
	 * gallery of pictures for it).  A reply to the bot is a follow-up,
	 * its output is not input.
	 */
	if (r != NULL && !(r->author && r->author->id == self)) {
		if (!code)
			text_blocks(r->content, blocks);
		if (r->attachments)
			for (k = 0; k < r->attachments->size && nf < AT_MAXFILES;
			    k++, nf++)
				one_attachment(&r->attachments->array[k], blocks,
				    notes);
	}
	return blocks->n > before;
}

int
attach_block(const struct discord_message *m, int k, struct sbuf *block,
    struct sbuf *notes)
{
	if (m->attachments == NULL || k < 0 || k >= m->attachments->size ||
	    k >= AT_MAXFILES)
		return 0;
	one_attachment(&m->attachments->array[k], block, notes);
	return block->n > 0;
}
