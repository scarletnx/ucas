/* The C23 the frontend takes, one function per feature (README "C
 * frontend", "C23").  tests/cfc23.ucas lowers them. */
#include <stdint.h>
#include <stddef.h>

#if __STDC_VERSION__ < 202311L
#error the version of the standard is C23
#elifdef __STDC__
#define ELIFDEF 1
#endif
#ifdef NO_SUCH_MACRO
#define PICKED 0
#elifndef NO_SUCH_MACRO
#define PICKED 2
#else
#define PICKED 3
#endif

/* binary constants, digit separators, #elifdef and #elifndef */
uint32_t binary(uint32_t x) { return (x & 0b1111'0000u) | 0x1'00'00; }
int picked(void) { return ELIFDEF * 10 + PICKED; }
int32_t grouped(int32_t x) { return x / 1'000'000; }

/* _BitInt, its literals, its conversions: it is not promoted to int */
unsigned _BitInt(12) wrap12(unsigned _BitInt(12) x) { return x + 1uwb; }
int widen(_BitInt(7) x) { return x - 100; }
uint32_t bits(uint8_t x) { return (unsigned _BitInt(3))x; }

/* bool, true, false and nullptr, with no header; u8'c' is unsigned */
bool is_even(uint32_t x) { return (x & 1) == 0 ? true : false; }
int is_null(const uint8_t *p) { return p == nullptr; }
int high(uint8_t c) { return c == u8'\xff'; }

/* constexpr, static_assert, alignas and alignof */
constexpr uint32_t LIMIT = 0x40;
static_assert(LIMIT == 64, "the limit");
struct wide { alignas(16) uint8_t lead; uint32_t tail; };
static_assert(alignof(struct wide) == 16 && sizeof(struct wide) == 16);
uint32_t limited(uint32_t x) { constexpr uint32_t K = LIMIT / 2; return x % K; }
size_t wide_align(void) { return alignof(struct wide); }

/* typeof and typeof_unqual, thread_local, attributes */
static thread_local uint32_t counter;
uint32_t typed(const uint32_t x) {
    typeof_unqual(x) y = x;
    typeof(y) z = 3;
    y += z;
    return y;
}
[[nodiscard]] uint32_t attributed([[maybe_unused]] uint32_t unused, uint32_t x) {
    [[maybe_unused]] uint32_t twice = x << 1;
    switch (x & 1) {
    case 0:
        x += 1;
        [[fallthrough]];
    default:
        break;
    }
    return x;
}

/* an enumeration with a fixed type, and its constants */
typedef enum mode mode;
enum mode { OFF, ON };
enum level : uint8_t { LOW = 1, TOP = 250, };
uint32_t level_of(uint8_t x) { return x == TOP ? 7u : x == LOW ? 3u : 0u; }
uint32_t level_size(void) { return (uint32_t)sizeof(enum level); }

/* empty initialisers, and #embed */
int zero(int a) { int x = {}; return a + x; }
static const uint8_t text[] = {
#embed "embed.txt" limit(4)
    , 0
};
uint32_t embedded(uint32_t i) { return text[i & 3]; }
static const uint8_t empty[] = {
#embed "embed.txt" limit(0) if_empty(9)
};
uint32_t embed_empty(void) { return empty[0]; }
