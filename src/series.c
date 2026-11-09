/*
 * series.c: infinite sums and products (xform.h).
 */
#include "xform.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int
series_command(struct P *p, const char *w)
{
	(void)p;
	(void)w;
	return 0;
}

const char *
xf_ident(double v)
{
	(void)v;
	return NULL;
}
