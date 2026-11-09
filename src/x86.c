/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * x86 commands (parse.c calls x86_command first for every command line):
 *
 *   model NAME              an x86 µarch's costs (NAME: skl skx glc rpl
 *                           zen4 x86v2 x86v3 x86v4, or an alias)
 *   x86 models              the µarchs
 *   mca F [on M]            simulated cycles of F's x86 code on M
 *
 * Each x86 µarch is also an ordinary ucas cost model (`use zen4`, `cost F
 * model skl`, `find` under it), declared by x86_prelude from the tables.
 */
#include "x86.h"

#include <stdlib.h>
#include <string.h>

/* `on M`, `model M`, `for M`, `under M`: an x86 µarch; -1 if not given */
static int
opt_uarch(struct P *p)
{
	char w[48];
	uint32_t at = cmdp_pos(p);
	int ua;

	if (!cmdp_kw(p, "on") && !cmdp_kw(p, "model") && !cmdp_kw(p, "for") &&
	    !cmdp_kw(p, "under"))
		return -1;
	if (!cmdp_word(p, w, sizeof w)) {
		cmdp_seek(p, at);
		return -1;
	}
	if ((ua = uarch_find(w)) < 0)
		fail("unknown x86 µarch '%s' (skl skx glc rpl zen4 x86v2 x86v3 "
		    "x86v4)", w);
	return ua;
}

/* the active model when it is an x86 one, else Skylake (this host) */
static int
default_uarch(void)
{
	int ua = uarch_of_model(model_active());

	return ua >= 0 ? ua : UA_SKL;
}

static const char *const mca_stop[] = {"on", "model", "for", "under", NULL};

int
x86_command(struct P *p)
{
	uint32_t at = cmdp_pos(p);
	char w[64];
	int ua;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if (func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	if (!strcmp(w, "model")) {
		/* `model zen4` alone shows it; `model m { ... }` declares */
		char nm[48];
		if (cmdp_word(p, nm, sizeof nm) && cmdp_eos(p) &&
		    (ua = uarch_find(nm)) >= 0) {
			cmd_xmodel(ua);
			return 1;
		}
		cmdp_seek(p, at);
		return 0;
	}
	if (!strcmp(w, "x86")) {
		if (cmdp_kw(p, "models") || cmdp_kw(p, "uarchs")) {
			if (!cmdp_eos(p))
				fail("x86 models takes no arguments");
			cmd_xmodels();
			return 1;
		}
		fail("x86: models");
	}
	if (!strcmp(w, "mca")) {
		struct func *f;
		if (cmdp_eos(p))
			fail("mca F [on skl|skx|glc|rpl|zen4|x86v2|x86v3|x86v4]");
		f = cmdp_func(p, mca_stop);
		ua = opt_uarch(p);
		if (!cmdp_eos(p))
			fail("mca F [on MODEL]");
		notes_flush();
		cmd_mca(f, ua >= 0 ? ua : default_uarch());
		return 1;
	}
	cmdp_seek(p, at);
	return 0;
}
