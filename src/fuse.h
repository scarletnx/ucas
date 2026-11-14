/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * fuse.c: fusions declared by a cost model (several operations that one
 * instruction does) and the cheapest cover of an expression DAG by them;
 * irw.c: integer rewrites (signed -> unsigned division, compare and
 * select -> min/max, division by a variable through floats) that seed
 * find and simplify, each with its derivation and the conditions it
 * holds under.
 */
#ifndef UCAS_FUSE_H
#define UCAS_FUSE_H

#include "ucas.h"

#define FUSE_MAXP 4
#define FUSE_MAXT 4
#define MAXFUSE 96

enum { FP_INT, FP_FLOAT, FP_CONST, FP_ANY, FP_LANE };
#define FUSE_MAXBITS 32		/* an input's proved width: u1 .. u32, s1 .. s32 */

/*
 * fuse NAME(a: int, b: int, c: int) = a * b + c : imul
 * fuse NAME = lut 3 : logic          any tree of bitwise logic of up to
 *                                    3 inputs (LOP3)
 * fuse NAME(a: float) = -a : 0 in fadd fmul
 *                                    a modifier: costs this when every
 *                                    use of it is an operation of those
 *                                    classes
 * `contract` marks a fusion that rounds once where the operations round
 * twice (a*b + c as an fma): it applies only where contraction is
 * allowed (not to a SPIR-V result decorated NoContraction).
 * fuse NAME(a: u16, b: u16, c: int) = a * b + c : mul16
 *                                    an input of a proved width: uN, an
 *                                    integer proved inside [0, 2^N), sN
 *                                    inside [-2^(N-1), 2^(N-1)), from the
 *                                    ranges of the function being priced
 *                                    (fuse_domain) and the types
 * fuse NAME(a: float) = min(max(a, 0.0), 1.0) : 0 on fadd fmul
 *                                    a result modifier: the pattern folds
 *                                    into the instruction that makes its
 *                                    input, when that is an operation of
 *                                    those classes used only here
 * fuse NAME(a: int, b: int) = sg_ballot(a == b) : 3.875
 * fuse NAME(v: any, lane: lane, k: const) = sg_shuffle(v, lane xor k)
 *     : sg_shuffle                   subgroup operations (sg.c): an input
 *                                    in an operand is what the lane
 *                                    computes there, `lane: lane` the
 *                                    lane's index; any subgroup size
 */
struct fusion {
	uint32_t model, name;
	int np;
	uint32_t pname[FUSE_MAXP];
	uint8_t pkind[FUSE_MAXP];
	uint8_t pbits[FUSE_MAXP];	/* FP_INT of a proved width; 0: any */
	uint8_t psgn[FUSE_MAXP];	/* that width is signed (sN) */
	nid pat;		/* over O_VAR slots 0..np-1; 0 for a lut */
	int lut;		/* inputs of a lut fusion, else 0 */
	int nt;			/* cost: k + sum tmul[i] * class tcls[i] */
	int8_t tcls[FUSE_MAXT];
	float tmul[FUSE_MAXT];
	float k;
	uint64_t users;		/* a modifier: the classes that absorb it */
	uint64_t makers;	/* a result modifier: the classes it folds into */
	int contract;
};

void fuse_add(const struct model *m, const struct fusion *f);
void fuse_remove(const struct model *m, uint32_t name);
void fuse_inherit(const struct model *m, const struct model *parent);
void fuse_reset(const struct model *m);
int fuse_count(const struct model *m);
void fuse_text(struct sbuf *b, const struct fusion *f);
void fuse_print_models(const struct model *m);	/* `models`: its lines */
float fuse_price(const struct model *m, const struct fusion *f, int f64);

/*
 * A DAG to cover: nodes in topological order (operands first), each with
 * an operation (O_*; O_NONE for one priced as it is), its operands (-1:
 * none, or not part of the graph), its own cost and class.
 */
struct fgraph {
	int n;
	uint8_t *op, *fp, *f64, *isk, *uns, *isbool, *nocontract, *pinned;
	uint8_t *islane;	/* the lane's index (a variable named lane) */
	int64_t *kv;
	double *kr;
	int (*arg)[3];
	int *uses;
	float *own;
	int *cls;
	uint32_t *id;		/* the caller's id of each node */
	int64_t *lo, *hi;	/* an integer node's proved range, where */
	uint8_t *ranged;	/* ranged[i]; NULL: no ranges at all */
	uint64_t *dem;		/* the bits of each integer node that
				 * anything reads (fg_demanded); NULL: all */
	/* the cover */
	float *eff;		/* each node's share: its fusion's price at the
				 * root, 0 when absorbed, else its own cost */
	int *fz;		/* -1, or the fusion rooted (or absorbed) here */
	int *by;		/* absorbed into this root, or -1 */
	float *own0;		/* the own cost before a modifier priced it */
	const struct fusion *fl[MAXFUSE];
	int nfl;
};
void fg_init(struct fgraph *g, int n);
void fg_free(struct fgraph *g);
float fg_cover(const struct model *m, struct fgraph *g);
void fg_ranges(struct fgraph *g, const int64_t *tlo, const int64_t *thi);
void fg_demanded(struct fgraph *g, const int *width);
/* what the cover fused: one line each ("imad 2 x 3 = 6, saves 2") */
struct fuse_tally {
	int n;
	const struct fusion *f[MAXFUSE];
	double cnt[MAXFUSE], cost[MAXFUSE], saved[MAXFUSE];
};
void fg_tally(const struct fgraph *g, const uint8_t *in, double times,
    struct fuse_tally *t);
void fuse_tally_print(const struct fuse_tally *t, const char *ind);

/* ucas nodes: the nodes in mark (indexed by nid), pins counted as used
 * outside; eff (indexed by nid) gets each node's share; loops are priced
 * apart (own 0) unless loops_own */
float fuse_cover_nodes(const struct model *m, const uint8_t *mark,
    uint32_t lim, const nid *pins, int npins, int loops_own, float *eff,
    struct fgraph *keep);
float fuse_expr_cost(const struct model *m, nid root);
/*
 * The function whose nodes are priced next (NULL: none): its parameters'
 * ranges, narrowed by its where clause, are what a fusion input of a
 * proved width (uN, sN) is matched against.  Without it the types alone
 * bound the values.
 */
void fuse_domain(const struct func *f);
/* several expressions: what they share is covered once */
float fuse_roots_cost(const struct model *m, const nid *roots, int nroots);

#endif
