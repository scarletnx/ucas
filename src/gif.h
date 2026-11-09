#ifndef UCAS_GIF_H
#define UCAS_GIF_H

#include "ucas.h"

struct gif {
	struct sbuf out;	/* the file, in memory */
	int w, h, bits, nframes;
	unsigned char *prev;
	size_t delay_at;	/* offset of the last frame's delay */
};

void gif_begin(struct gif *g, int w, int h, const unsigned char (*pal)[3],
    int ncolors);
/* a full frame of palette indices; only what changed is stored */
void gif_frame(struct gif *g, const unsigned char *pix, int delay_cs);
void gif_end(struct gif *g);

#endif
