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

/* the same d twice in one clamp: one table, two reads */
int32_t step_sat(uint8_t i, uint8_t code, int32_t p) {
    int32_t s = IMA_STEPS[i], d = s >> 3;
    if (code & 1) d += s >> 2;
    if (code & 2) d += s >> 1;
    if (code & 4) d += s;
    return p + d < 32767 ? p + d : 32767;
}

/* ch_index in [0, 88] (the table's bound), nibble in [0, 15] and
 * ch_predictor in [0, 32767] (assumed by the test) */
int16_t ima_steps_t(uint8_t ch_index, uint8_t nibble, int16_t ch_predictor) {
    int32_t step = IMA_STEPS[ch_index], diff = step >> 3;
    if (nibble & 1) diff += step >> 2;
    if (nibble & 2) diff += step >> 1;
    if (nibble & 4) diff += step;
    int32_t predictor = ch_predictor;
    if (nibble & 8)
        predictor = predictor - diff > -0x7fff ? predictor - diff : -0x7fff;
    else
        predictor = predictor + diff < 0x7fff ? predictor + diff : 0x7fff;
    return predictor;
}

/* x - d never above 300, x + d never below -100: one clamp of x +- d */
int clamp2(_Bool neg, uint8_t x, uint8_t d) {
    int r;
    if (neg)
        r = x - d > -100 ? x - d : -100;
    else
        r = x + d < 300 ? x + d : 300;
    return r;
}

/* a < b and a >= b are both false when a or b is a NaN: the ?: on the second
 * is not decided by the first */
int nan_sel(float a, float b) {
    return a < b ? (a >= b ? 1 : 2) : (a >= b ? 3 : 4);
}
