/* #if evaluates its operators (tests/cfppif.ucas): only an operand of &&
 * and || that decides nothing is skipped. */
#include <stdint.h>

#if 5 == 1
uint32_t wrong_branch(uint32_t x) { return x; }
#elif 2 * 3 == 6 && (1 || 1 / 0)
uint32_t right_branch(uint32_t x) { return x + 1; }
#endif
#if 4 > 3 && 3 < 2
uint32_t also_wrong(uint32_t x) { return x; }
#endif
uint32_t after(uint32_t x) { return x * 2; }
