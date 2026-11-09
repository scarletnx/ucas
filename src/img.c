/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Image files for animate exec's image inputs: PNG, JPEG (baseline and
 * progressive), GIF (the first frame), BMP and PPM/PGM, decoded by
 * stb_image (src/third_party, unmodified) from memory.  The files are
 * untrusted (chat attachments): the size of the file, the dimensions and
 * the decoded pixel count are checked before anything is decoded, and
 * stb_image's own dimension cap is set as well.
 */
#include "ucas.h"
#include "img.h"

#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_GIF
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#define STBI_NO_STDIO
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_MAX_DIMENSIONS IMG_MAXDIM
#include "third_party/stb_image.h"

const char *
img_sniff(const unsigned char *p, size_t n)
{
	if (n >= 8 && !memcmp(p, "\x89PNG\r\n\x1a\n", 8))
		return "PNG";
	if (n >= 3 && p[0] == 0xff && p[1] == 0xd8 && p[2] == 0xff)
		return "JPEG";
	if (n >= 6 && (!memcmp(p, "GIF87a", 6) || !memcmp(p, "GIF89a", 6)))
		return "GIF";
	if (n >= 2 && p[0] == 'B' && p[1] == 'M')
		return "BMP";
	if (n >= 2 && p[0] == 'P' && p[1] >= '1' && p[1] <= '6')
		return "PNM";
	if (n >= 4 && !memcmp(p, "qoif", 4))
		return "QOI";
	if (n >= 12 && !memcmp(p, "RIFF", 4) && !memcmp(p + 8, "WEBP", 4))
		return "WebP";
	return NULL;
}

int
img_decode(const unsigned char *p, size_t n, struct imgdec *out, char *err,
    size_t errn)
{
	const char *f = img_sniff(p, n);
	int w, h, c;
	unsigned char *px;

	memset(out, 0, sizeof *out);
	if (f == NULL) {
		snprintf(err, errn, "not an image file I can read (PNG, JPEG, GIF, "
		    "BMP, PPM/PGM)");
		return 0;
	}
	snprintf(out->format, sizeof out->format, "%s", f);
	if (!strcmp(f, "QOI") || !strcmp(f, "WebP")) {
		snprintf(err, errn, "%s images are not supported (PNG, JPEG, GIF, "
		    "BMP, PPM/PGM are)", f);
		return 0;
	}
	if (n > IMG_MAXFILE || n > 0x7fffffff) {
		snprintf(err, errn, "%s: the file is %zu bytes, over the %u MiB cap",
		    f, n, IMG_MAXFILE >> 20);
		return 0;
	}
	if (!stbi_info_from_memory(p, (int)n, &w, &h, &c)) {
		snprintf(err, errn, "%s: %s", f, stbi_failure_reason());
		return 0;
	}
	if (w <= 0 || h <= 0 || w > IMG_MAXDIM || h > IMG_MAXDIM ||
	    (uint64_t)w * (uint64_t)h > IMG_MAXPIX) {
		snprintf(err, errn, "%s: %dx%d is over the cap (%d per side, %u "
		    "Mpixels)", f, w, h, IMG_MAXDIM, IMG_MAXPIX >> 20);
		return 0;
	}
	px = stbi_load_from_memory(p, (int)n, &w, &h, &c, 4);
	if (px == NULL) {
		snprintf(err, errn, "%s: %s", f, stbi_failure_reason());
		return 0;
	}
	out->w = w;
	out->h = h;
	out->channels = c;
	out->rgba = px;
	return 1;
}

void
img_free(struct imgdec *im)
{
	stbi_image_free(im->rgba);
	im->rgba = NULL;
}
