/*
 * The evaluation supervisor: a single-threaded process, forked before
 * the Discord client starts any threads, that owns one worker process
 * per channel and runs each evaluation in a short-lived, resource
 * limited child of that worker.  See README.md, "Discord bot".
 */
#ifndef UCAS_DISCORD_SUP_H
#define UCAS_DISCORD_SUP_H

#include <sys/types.h>

#include <stddef.h>
#include <stdint.h>

/* frame types */
enum {
	SF_EVAL = 1,	/* bot -> sup: run text in channel's state */
	SF_RESET,	/* bot -> sup: drop the channel's state */
	SF_RESP,	/* sup -> bot: status + output */
	SF_PID,		/* child -> sup: evaluation started, payload-less */
	SF_DONE,	/* child -> sup: evaluation finished normally */
	SF_FAIL		/* worker -> sup: child died; status = wait status */
};

/* SF_RESP status */
enum {
	SR_OK,		/* evaluation finished; state committed */
	SR_TIMEOUT,	/* wall-clock limit; state rolled back */
	SR_CPU,		/* CPU-time limit; state rolled back */
	SR_MEM,		/* memory limit; state rolled back */
	SR_CRASH,	/* other abnormal end; state rolled back */
	SR_LOST,	/* the channel's worker died; state lost */
	SR_RESET,	/* $reset done */
	SR_BUSY		/* too many requests queued for this channel */
};
/* SF_EVAL arg: bit 0 echo; SF_JSON: the text is a JSON frontend request */
#define SF_JSON 2

#define SR_MASK 0xff
#define SR_TRUNC 0x100	/* output was cut at the output limit */

struct sframe {
	uint32_t type;
	uint32_t len;		/* payload bytes that follow */
	uint32_t status;	/* SF_RESP: SR_*; SF_FAIL: wait status */
	uint32_t arg;		/* SF_EVAL: echo; SF_RESP: signal or exit */
	uint64_t id;		/* request id (bot side), pid (SF_PID) */
	uint64_t chan;		/* channel id */
};

struct sup_limits {
	int cpu_s;		/* CPU seconds per evaluation */
	int wall_s;		/* wall-clock seconds per evaluation */
	int mem_mb;		/* address-space limit of a worker */
	int max_chan;		/* live channel workers (LRU beyond) */
	int max_queue;		/* queued requests per channel */
	size_t max_out;		/* output bytes kept per evaluation */
};

void sup_limits_env(struct sup_limits *l);
pid_t sup_start(const struct sup_limits *l, int *fd);
int frame_write(int fd, const struct sframe *h, const void *p);
int frame_read(int fd, struct sframe *h, char **payload);

#endif
