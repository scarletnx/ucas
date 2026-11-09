#include <stdint.h>
/* a clamp the type already does, and a shift in range */
uint8_t lo8(uint32_t x) { uint32_t y = x & 255; return (uint8_t)(y > 255 ? 255 : y); }
uint32_t shr(uint32_t x, uint32_t n) { return n < 32 ? x >> n : 0; }
