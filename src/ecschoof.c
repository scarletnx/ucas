/*
 * ecschoof.c - Schoof's algorithm and SEA (placeholder until they land).
 */
#include "ucas.h"
#include "ring.h"
#include "ecff.h"

int
ec_count_schoof(const struct ecp_curve *E, nn *N, int verbose)
{
	fail("schoof: not yet");
	return 0;
}

int
ec_count_sea(const struct ecp_curve *E, nn *N, int verbose)
{
	return ec_count_schoof(E, N, verbose);
}
