/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Other bots as users: which bot accounts are served, their rate limit,
 * the cap on reply chains between Wimsy and a bot, and the "explain AI
 * usage" request with its help text.  No Concord types here, so it is
 * unit tested on its own (selftest.c).  See README.md, "Bots and
 * programs".
 */
#ifndef UCAS_DISCORD_BOTUSE_H
#define UCAS_DISCORD_BOTUSE_H

#include <stddef.h>
#include <stdint.h>

#define BU_MAXALLOW 32
#define BU_DEFAULT_BOTS "1562594665109393468,1566587484866093137"

struct bu_conf {
	int nallow;
	uint64_t allow[BU_MAXALLOW];	/* bot user ids served */
	int rate;		/* requests per minute, per bot */
	int burst;		/* requests at once */
	int chain;		/* reply-chain depth without a human */
};
extern struct bu_conf bu_conf;

/* UCAS_DISCORD_BOTS, UCAS_DISCORD_BOT_RATE, _BURST, _CHAIN; 0 if bad */
int bu_conf_env(void);
int bu_conf_parse(struct bu_conf *c, const char *bots);
int bu_allowed(uint64_t author);

/*
 * The rate limit (a token bucket per bot): 0 to go ahead, else the
 * seconds until the next request is taken; *notify is 1 the first time
 * in a run of refusals (the one time the bot is told).
 */
int bu_rate(uint64_t author, int64_t now_ms, int *notify);

/*
 * The reply chain: a bot's request that is a Discord reply to Wimsy
 * deepens its chain in that channel, any other starts one; a human's
 * message in the channel (bu_human) ends every chain there.  1 to go
 * ahead, 0 past the cap (*notify once per chain).
 */
int bu_chain(uint64_t chan, uint64_t author, int reply_to_self,
    int *notify);
void bu_human(uint64_t chan);
void bu_reset(void);	/* tests */

/* "explain AI usage" and its variants, misspellings tolerated */
int bu_is_ai_usage(const char *text);

/* notices: a bot's reply to one of these alone is not a request */
#define BU_RATE_NOTICE "rate limited: "
#define BU_CHAIN_NOTICE "reply-chain limit: "
#define BU_HELP_HEAD "-# bots and programs: mention me with "
int bu_is_notice(const char *content);

/*
 * The help text for bots (discord/ai_usage.md, the short form in
 * ai_usage_short.md), with {MENTION}, {ID}, {NAME}, {RATE}, {BURST},
 * {CHAIN}, {CPU}, {WALL} filled in.  malloc'd.
 */
char *bu_help(int full, uint64_t self, const char *name, int cpu,
    int wall);

#endif
