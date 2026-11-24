/* Kernels of tests/cfsub.ucas. */
#include <stdint.h>

static const int16_t IMA_STEPS[89] = {
    7,    8,     9,     10,    11,    12,    13,    14,    16,    17,    19,    21,    23,    25,   28,
    31,   34,    37,    41,    45,    50,    55,    60,    66,    73,    80,    88,    97,    107,  118,
    130,  143,   157,   173,   190,   209,   230,   253,   279,   307,   337,   371,   408,   449,  494,
    544,  598,   658,   724,   796,   876,   963,   1060,  1166,  1282,  1411,  1552,  1707,  1878, 2066,
    2272, 2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,  5894,  6484,  7132,  7845, 8630,
    9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
};

/* i in [0, 88] (the table's bound), code in [0, 7] (assumed by the test):
 * d is a sub-value of i and code alone, 712 points */
int32_t step_diff(uint8_t i, uint8_t code, int32_t p) {
    int32_t s = IMA_STEPS[i], d = s >> 3;
    if (code & 1) d += s >> 2;
    if (code & 2) d += s >> 1;
    if (code & 4) d += s;
    return p + d;
}

