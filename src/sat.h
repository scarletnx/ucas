/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * A small SAT solver (CDCL) under an and-inverter graph, with word-level
 * circuits for bit-precise proofs: two expressions are equal for every
 * input when their miter (some output bit differs) is unsatisfiable.
 */
#ifndef UCAS_SAT_H
#define UCAS_SAT_H

#include <stdint.h>

/*
 * Literals: 2*var + sign.  Variable 0 is the constant: literal 0 is
 * false, literal 1 is true.
 */
typedef uint32_t lit;
#define L_FALSE 0u
#define L_TRUE 1u
#define L_NOT(a) ((a) ^ 1u)

/* 65: w_udivrem works in one bit more than its operands */
#define WMAX 65
struct word {
	int w;
	lit b[WMAX];	/* bit 0 first */
};

struct aig;

struct aig *aig_new(void);
void aig_free(struct aig *g);
lit aig_input(struct aig *g);
lit aig_and(struct aig *g, lit a, lit b);
lit aig_or(struct aig *g, lit a, lit b);
lit aig_xor(struct aig *g, lit a, lit b);
lit aig_mux(struct aig *g, lit s, lit t, lit e);	/* s ? t : e */
int aig_nodes(const struct aig *g);

/*
 * 1: some assignment makes lit true (a counterexample; aig_value reads
 * it), 0: never (proved), -1: gave up after `budget` conflicts.
 */
int aig_sat(struct aig *g, lit goal, long budget);
int aig_value(const struct aig *g, lit a);	/* after aig_sat == 1 */
extern long aig_conflicts;	/* conflicts in the last aig_sat */
extern double aig_seconds;	/* > 0: a CPU time limit for aig_sat too */
uint64_t aig_word_value(const struct aig *g, const struct word *x);

/* words */
void w_const(struct word *r, int w, uint64_t v);
void w_input(struct aig *g, struct word *r, int w);
void w_not(struct aig *g, struct word *r, const struct word *a);
void w_and(struct aig *g, struct word *r, const struct word *a,
    const struct word *b);
void w_or(struct aig *g, struct word *r, const struct word *a,
    const struct word *b);
void w_xor(struct aig *g, struct word *r, const struct word *a,
    const struct word *b);
void w_add(struct aig *g, struct word *r, const struct word *a,
    const struct word *b);
void w_sub(struct aig *g, struct word *r, const struct word *a,
    const struct word *b);
void w_neg(struct aig *g, struct word *r, const struct word *a);
void w_mul(struct aig *g, struct word *r, const struct word *a,
    const struct word *b);
void w_udivrem(struct aig *g, struct word *q, struct word *rem,
    const struct word *a, const struct word *b);	/* b == 0: q = r = 0 */
/* shifts by a word: the low 6 bits of the amount; >= width gives 0
 * (or the sign) */
void w_shl(struct aig *g, struct word *r, const struct word *a,
    const struct word *s);
void w_lshr(struct aig *g, struct word *r, const struct word *a,
    const struct word *s);
void w_ashr(struct aig *g, struct word *r, const struct word *a,
    const struct word *s);
lit w_eq(struct aig *g, const struct word *a, const struct word *b);
lit w_ult(struct aig *g, const struct word *a, const struct word *b);
lit w_slt(struct aig *g, const struct word *a, const struct word *b);
void w_mux(struct aig *g, struct word *r, lit s, const struct word *t,
    const struct word *e);
void w_resize(struct word *r, const struct word *a, int w, int sign);
void w_popcount(struct aig *g, struct word *r, const struct word *a);
void w_reverse(struct word *r, const struct word *a);
lit w_ne(struct aig *g, const struct word *a, const struct word *b);

int sat_selftest(void);	/* 0 when the solver and circuits check out */

#endif
