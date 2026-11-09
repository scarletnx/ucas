/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * What language a fenced code block (or an attachment) is in: SPIR-V
 * assembly, GLSL or C.  The tag decides when there is one (```glsl is
 * GLSL, ```c is C), except that a ```c block with GLSL's own marks
 * (#version, layout(, gl_*, uint, vec3) and none of C's is GLSL tagged
 * for its colours.  An untagged
 * block is read from its content, conservatively: marks only one of the
 * languages has (uint, vec3, uniform; #include, unsigned, uint32_t,
 * pointers).  With marks of both, or of neither, it is unclear: GLSL, as
 * before, and the caller says so in a note.
 */
#include <ctype.h>
#include <string.h>

#include "ucas.h"

/* the word w (whole) in s */
static int
has_word(const char *s, const char *w)
{
	size_t n = strlen(w);
	const char *p = s;

	while ((p = strstr(p, w)) != NULL) {
		if ((p == s || !(isalnum((unsigned char)p[-1]) || p[-1] == '_')) &&
		    !(isalnum((unsigned char)p[n]) || p[n] == '_'))
			return 1;
		p += n;
	}
	return 0;
}

/* GLSL's own: things C does not have */
static int
glsl_marks(const char *c)
{
	static const char *const w[] = {"uint", "vec2", "vec3", "vec4",
	    "uvec2", "uvec3", "uvec4", "ivec2", "ivec3", "ivec4", "bvec2",
	    "bvec3", "bvec4", "dvec2", "dvec3", "dvec4", "mat2", "mat3", "mat4",
	    "uniform", "sampler2D", "image2D", "texture", "imageLoad",
	    "imageStore", "barrier", "inout", "precision", "highp", "mediump",
	    "lowp", "discard", NULL};
	int k, n = 0;

	if (strstr(c, "#version") || strstr(c, "layout(") ||
	    strstr(c, "layout (") || strstr(c, "gl_"))
		n += 2;
	for (k = 0; w[k]; k++)
		n += has_word(c, w[k]);
	return n;
}

/* C's own: things GLSL does not have */
static int
c_marks(const char *c)
{
	static const char *const w[] = {"unsigned", "signed", "char", "long",
	    "short", "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
	    "int8_t", "int16_t", "int32_t", "int64_t", "uintptr_t", "static",
	    "extern", "typedef", "sizeof", "NULL", "goto", "enum", "union",
	    "restrict", "volatile", "register", "malloc", "printf", NULL};
	int k, n = 0;
	const char *p;

	if (strstr(c, "#include") || strstr(c, "->"))
		n += 2;
	for (k = 0; w[k]; k++)
		n += has_word(c, w[k]);
	/* a pointer declaration: "int *p", "float* a", "void *": a type
	 * word right before the star (an expression has an operand there) */
	for (p = c; (p = strchr(p, '*')) != NULL; p++) {
		static const char *const tw[] = {"int", "void", "float",
		    "double", "const", NULL};
		const char *a = p, *t;
		size_t l;
		while (a > c && a[-1] == ' ')
			a--;
		for (t = a; t > c && (isalnum((unsigned char)t[-1]) ||
		    t[-1] == '_'); t--)
			;
		l = (size_t)(a - t);
		for (k = 0; tw[k]; k++)
			if (l == strlen(tw[k]) && !strncmp(t, tw[k], l))
				n++;
	}
	return n;
}

static int
tag_in(const char *lang, const char *const *tags)
{
	int k;

	for (k = 0; tags[k]; k++)
		if (!strcmp(lang, tags[k]))
			return 1;
	return 0;
}

int
code_lang(const char *c, const char *lang, int *unclear)
{
	static const char *const gt[] = {"glsl", "comp", "frag", "vert", "geom",
	    "tesc", "tese", "hlsl", "shader", "vulkan", "rgen", "rchit",
	    "rmiss", "mesh", "task", NULL};
	static const char *const ct[] = {"c", "h", "cpp", "c++", "cc", "cxx",
	    "hpp", NULL};
	static const char *const st[] = {"spv", "spvasm", "spirv", NULL};
	int g, k;

	if (unclear)
		*unclear = 0;
	if (c == NULL)
		return CL_NONE;
	if (lang == NULL)
		lang = "";
	if (tag_in(lang, st) || strstr(c, "= Op") || strstr(c, "OpFunction") ||
	    strstr(c, "OpCapability"))
		return tag_in(lang, st) || !tag_in(lang, ct) ? CL_SPV : CL_C;
	if (!strchr(c, '{') && !strchr(c, ';'))
		return CL_NONE;
	if (tag_in(lang, gt))
		return CL_GLSL;
	g = glsl_marks(c);
	k = c_marks(c);
	if (tag_in(lang, ct)) {
		/* ```c is C, unless it is plainly GLSL tagged for colours */
		if (g > 0 && k == 0) {
			if (unclear)
				*unclear = 2;
			return CL_GLSL;
		}
		return CL_C;
	}
	if (lang[0])
		return CL_NONE;		/* ```python, ```text ... */
	if (g > 0 && k == 0)
		return CL_GLSL;
	if (k > 0 && g == 0)
		return CL_C;
	/* both, or neither: unclear; the more marks, else GLSL as before */
	if (!(strchr(c, '{') && strchr(c, ';')))
		return CL_NONE;
	if (!strstr(c, "int") && !strstr(c, "float") && !strstr(c, "void") &&
	    !strstr(c, "double") && !strstr(c, "bool") && g + k == 0)
		return CL_NONE;
	if (unclear)
		*unclear = 1;
	return k > g ? CL_C : CL_GLSL;
}
