/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * etc2blk.c: an ETC2 colour block (RGB8_ETC2) decoded, and the block for
 * one table in its red channel searched over every encoding of that
 * channel; ETC2's 8-bit alpha block (RGBA8_ETC2_EAC's first half) the
 * same way.  Khronos Data Format Specification 1.3, 21.1 and 22.1 to
 * 22.3.
 *
 * A colour block is 8 bytes read as a big-endian word w.  The texels'
 * 2-bit indices are bits 31..16 (high bit) and 15..0 (low bit), texel
 * (x, y) at bit y + 4x of each.  Bit 33 is the diff bit; R, G, B are
 * the 5-bit fields at bits 63, 55 and 47 (down), dR, dG, dB the 3-bit
 * ones below each, read as 0, 1, 2, 3, -4, -3, -2, -1.  With the diff
 * bit clear the block is individual; set, it is T when R + dR leaves
 * 0..31, else H when G + dG does, else planar when B + dB does, else
 * differential.
 *
 *   individual, differential: two sub-blocks, 2 x 4 side by side (flip,
 *     bit 32, clear) or 4 x 2 one over the other, each a base colour
 *     (4-bit fields, or a 5-bit one and the second as it plus a 3-bit
 *     difference) and a table of modifiers +a, +b, -a, -b (bits 39..37
 *     and 36..34); a texel is base + its index's modifier, clamped
 *   T: paint colours base 1, base 2 + d, base 2, base 2 - d (clamped)
 *   H: base 1 + d, base 1 - d, base 2 + d, base 2 - d, d's index's low
 *     bit being whether base 1 >= base 2 as 24-bit numbers
 *   planar: three colours O, H, V at texels (0, 0), (4, 0), (0, 4); a
 *     texel is clamp((x (H - O) + y (V - O) + 4 O + 2) >> 2)
 *
 * Fields of 4, 5, 6 and 7 bits widen to 8 by repeating their top bits.
 * A texel's channel reads c/255; in the sRGB formats (SRGB8_ETC2) red,
 * green and blue read the sRGB EOTF of c/255.
 */
#include "calc.h"
#include "texenc.h"

#include <math.h>
#include <string.h>

static const int etc1_mods[8][4] = {
	{2, 8, -2, -8}, {5, 17, -5, -17}, {9, 29, -9, -29}, {13, 42, -13, -42},
	{18, 60, -18, -60}, {24, 80, -24, -80}, {33, 106, -33, -106},
	{47, 183, -47, -183}};
static const int etc2_dist[8] = {3, 6, 11, 16, 23, 32, 41, 64};
static const int etc_delta[8] = {0, 1, 2, 3, -4, -3, -2, -1};

int
etc1_mod(int tab, int k)
{
	return etc1_mods[tab & 7][k & 3];
}

static int
clamp255(int v)
{
	return v < 0 ? 0 : v > 255 ? 255 : v;
}

static int
ext(int v, int bits)
{
	return (v << (8 - bits)) | (v >> (2 * bits - 8));
}

static int
floor4(int v)
{
	return v >= 0 ? v / 4 : -((-v + 3) / 4);
}

static uint64_t
word_of(const uint8_t q[8])
{
	uint64_t w = 0;
	int i;

	for (i = 0; i < 8; i++)
		w = w << 8 | q[i];
	return w;
}

static void
put_word(uint64_t w, uint8_t q[8])
{
	int i;

	for (i = 0; i < 8; i++)
		q[i] = (uint8_t)(w >> (56 - 8 * i));
}

int
etc2_mode(const uint8_t q[8])
{
	uint64_t w = word_of(q);
	int r = (int)(w >> 59 & 31) + etc_delta[w >> 56 & 7];
	int g = (int)(w >> 51 & 31) + etc_delta[w >> 48 & 7];
	int b = (int)(w >> 43 & 31) + etc_delta[w >> 40 & 7];

	if (!(w >> 33 & 1))
		return ETC2_INDIVIDUAL;
	if (r < 0 || r > 31)
		return ETC2_T;
	if (g < 0 || g > 31)
		return ETC2_H;
	if (b < 0 || b > 31)
		return ETC2_PLANAR;
	return ETC2_DIFFERENTIAL;
}

void
etc2_decode(const uint8_t q[8], uint8_t px[16][3])
{
	uint64_t w = word_of(q);
	int mode = etc2_mode(q), i, c, base[3][3], paint[4][3];

	switch (mode) {
	case ETC2_INDIVIDUAL: case ETC2_DIFFERENTIAL:
		for (c = 0; c < 3; c++) {
			int f = (int)(w >> (56 - 8 * c) & 255);
			if (mode == ETC2_INDIVIDUAL) {
				base[0][c] = ext(f >> 4, 4);
				base[1][c] = ext(f & 15, 4);
			} else {
				base[0][c] = ext(f >> 3, 5);
				base[1][c] = ext((f >> 3) + etc_delta[f & 7], 5);
			}
		}
		for (i = 0; i < 16; i++) {
			int x = i & 3, y = i >> 2, k = y + 4 * x;
			int idx = (int)((w >> (16 + k) & 1) << 1 | (w >> k & 1));
			int half = (w >> 32 & 1) ? y >= 2 : x >= 2;
			int m = etc1_mods[w >> (half ? 34 : 37) & 7][idx];
			for (c = 0; c < 3; c++)
				px[i][c] = (uint8_t)clamp255(base[half][c] + m);
		}
		return;
	case ETC2_T: case ETC2_H: {
		int d;
		if (mode == ETC2_T) {
			base[0][0] = (int)((w >> 59 & 3) << 2 | (w >> 56 & 3));
			base[0][1] = (int)(w >> 52 & 15);
			base[0][2] = (int)(w >> 48 & 15);
			base[1][0] = (int)(w >> 44 & 15);
			base[1][1] = (int)(w >> 40 & 15);
			base[1][2] = (int)(w >> 36 & 15);
		} else {
			base[0][0] = (int)(w >> 59 & 15);
			base[0][1] = (int)((w >> 56 & 7) << 1 | (w >> 52 & 1));
			base[0][2] = (int)((w >> 51 & 1) << 3 | (w >> 47 & 7));
			base[1][0] = (int)(w >> 43 & 15);
			base[1][1] = (int)(w >> 39 & 15);
			base[1][2] = (int)(w >> 35 & 15);
		}
		for (c = 0; c < 3; c++) {
			base[0][c] = ext(base[0][c], 4);
			base[1][c] = ext(base[1][c], 4);
		}
		if (mode == ETC2_T) {
			d = etc2_dist[(w >> 34 & 3) << 1 | (w >> 32 & 1)];
			for (c = 0; c < 3; c++) {
				paint[0][c] = base[0][c];
				paint[1][c] = clamp255(base[1][c] + d);
				paint[2][c] = base[1][c];
				paint[3][c] = clamp255(base[1][c] - d);
			}
		} else {
			int v1 = base[0][0] << 16 | base[0][1] << 8 | base[0][2];
			int v2 = base[1][0] << 16 | base[1][1] << 8 | base[1][2];
			d = etc2_dist[(w >> 34 & 1) << 2 | (w >> 32 & 1) << 1 |
			    (v1 >= v2)];
			for (c = 0; c < 3; c++) {
				paint[0][c] = clamp255(base[0][c] + d);
				paint[1][c] = clamp255(base[0][c] - d);
				paint[2][c] = clamp255(base[1][c] + d);
				paint[3][c] = clamp255(base[1][c] - d);
			}
		}
		for (i = 0; i < 16; i++) {
			int x = i & 3, y = i >> 2, k = y + 4 * x;
			int idx = (int)((w >> (16 + k) & 1) << 1 | (w >> k & 1));
			for (c = 0; c < 3; c++)
				px[i][c] = (uint8_t)paint[idx][c];
		}
		return;
	}
	default: {
		int o[3], h[3], v[3];
		o[0] = ext((int)(w >> 57 & 63), 6);
		o[1] = ext((int)((w >> 56 & 1) << 6 | (w >> 49 & 63)), 7);
		o[2] = ext((int)((w >> 48 & 1) << 5 | (w >> 43 & 3) << 3 |
		    (w >> 39 & 7)), 6);
		h[0] = ext((int)((w >> 34 & 31) << 1 | (w >> 32 & 1)), 6);
		h[1] = ext((int)(w >> 25 & 127), 7);
		h[2] = ext((int)(w >> 19 & 63), 6);
		v[0] = ext((int)(w >> 13 & 63), 6);
		v[1] = ext((int)(w >> 6 & 127), 7);
		v[2] = ext((int)(w & 63), 6);
		for (i = 0; i < 16; i++) {
			int x = i & 3, y = i >> 2;
			for (c = 0; c < 3; c++)
				px[i][c] = (uint8_t)clamp255(floor4(x * (h[c] - o[c]) +
				    y * (v[c] - o[c]) + 4 * o[c] + 2));
		}
	}
	}
}

/* ---- the red channel's best block ---- */

struct rsearch {
	const double *t;
	const uint8_t *care;
	double best, bestsq;
	int found, srgb;
};

/*
 * The sRGB EOTF of each 8-bit code (Khronos Data Format Specification
 * 13.3.1: c/255 over 12.92 up to 0.04045, else ((c/255 + 0.055) /
 * 1.055)^2.4), worked to 60 digits and rounded to the nearest f32: a
 * table, so no libm's pow decides a last bit
 */
static const float srgb8[256] = {
	0, 0.000303526991, 0.000607053982, 0.000910580973,
	0.00121410796, 0.00151763496, 0.00182116195, 0.00212468882,
	0.00242821593, 0.0027317428, 0.00303526991, 0.00334653584,
	0.00367650739, 0.00402471703, 0.00439144205, 0.00477695325,
	0.00518151652, 0.00560539169, 0.00604883302, 0.00651209056,
	0.00699541019, 0.00749903219, 0.00802319311, 0.00856812578,
	0.00913405884, 0.00972121768, 0.010329823, 0.0109600937,
	0.0116122449, 0.012286488, 0.0129830325, 0.0137020834,
	0.0144438436, 0.0152085144, 0.0159962941, 0.0168073755,
	0.0176419541, 0.01850022, 0.0193823613, 0.0202885624,
	0.0212190095, 0.0221738853, 0.0231533665, 0.0241576321,
	0.0251868591, 0.0262412224, 0.0273208916, 0.02842604,
	0.0295568351, 0.0307134446, 0.0318960324, 0.0331047662,
	0.0343398079, 0.0356013142, 0.0368894488, 0.0382043719,
	0.0395462364, 0.0409151986, 0.0423114114, 0.043735031,
	0.045186203, 0.0466650873, 0.0481718257, 0.0497065671,
	0.0512694567, 0.0528606474, 0.054480277, 0.0561284907,
	0.0578054301, 0.0595112368, 0.0612460524, 0.0630100146,
	0.064803265, 0.0666259378, 0.0684781671, 0.0703600943,
	0.0722718537, 0.0742135718, 0.0761853829, 0.078187421,
	0.0802198201, 0.0822827071, 0.0843762085, 0.0865004584,
	0.0886555836, 0.0908417106, 0.0930589661, 0.0953074694,
	0.097587347, 0.0998987257, 0.102241732, 0.104616486,
	0.107023105, 0.10946171, 0.111932427, 0.114435375,
	0.116970666, 0.119538426, 0.122138776, 0.124771819,
	0.127437681, 0.130136475, 0.13286832, 0.135633335,
	0.138431609, 0.141263291, 0.144128472, 0.147027269,
	0.149959788, 0.152926147, 0.155926466, 0.158960834,
	0.162029371, 0.165132195, 0.168269396, 0.171441108,
	0.174647406, 0.177888423, 0.18116425, 0.18447499,
	0.187820777, 0.191201687, 0.194617838, 0.198069319,
	0.20155625, 0.205078736, 0.208636865, 0.212230757,
	0.215860501, 0.219526201, 0.223227963, 0.226965874,
	0.230740055, 0.23455058, 0.238397568, 0.242281124,
	0.246201321, 0.25015828, 0.254152089, 0.258182853,
	0.262250662, 0.266355604, 0.270497799, 0.274677306,
	0.278894275, 0.283148736, 0.287440836, 0.291770637,
	0.296138257, 0.300543785, 0.304987311, 0.309468925,
	0.313988715, 0.318546772, 0.323143214, 0.327778101,
	0.332451522, 0.337163627, 0.341914415, 0.346704066,
	0.351532608, 0.356400132, 0.361306787, 0.366252601,
	0.371237695, 0.376262128, 0.38132602, 0.386429429,
	0.391572475, 0.396755219, 0.401977777, 0.407240212,
	0.412542611, 0.417885065, 0.423267663, 0.428690493,
	0.434153646, 0.439657182, 0.445201188, 0.450785786,
	0.456411034, 0.462076992, 0.467783809, 0.473531485,
	0.479320168, 0.48514995, 0.491020858, 0.496932983,
	0.502886474, 0.50888133, 0.514917672, 0.520995557,
	0.527115107, 0.533276379, 0.539479494, 0.545724452,
	0.55201143, 0.558340371, 0.564711511, 0.571124852,
	0.577580452, 0.584078431, 0.590618849, 0.597201765,
	0.603827357, 0.610495567, 0.617206573, 0.623960376,
	0.630757153, 0.637596846, 0.644479692, 0.651405632,
	0.658374846, 0.665387273, 0.672443151, 0.679542482,
	0.686685324, 0.693871737, 0.701101899, 0.708375752,
	0.715693474, 0.723055124, 0.730460763, 0.73791039,
	0.745404184, 0.752942204, 0.760524511, 0.768151164,
	0.775822222, 0.783537805, 0.791297913, 0.799102724,
	0.806952238, 0.814846575, 0.822785735, 0.830769897,
	0.838799, 0.846873224, 0.854992628, 0.863157213,
	0.871367097, 0.8796224, 0.887923121, 0.896269381,
	0.904661179, 0.913098633, 0.921581864, 0.930110872,
	0.938685715, 0.947306514, 0.955973327, 0.964686275,
	0.973445296, 0.982250571, 0.991102099, 1,
};

double
texenc_srgb8(int c)
{
	return srgb8[c & 255];
}

/* what a read of channel value c returns, sRGB or not */
static double
rd(int srgb, int c)
{
	return srgb ? (double)srgb8[c] : (double)(float)(c / 255.0);
}

/*
 * The block error of 16 texels each on its nearest of the palette its
 * sub-block has (pal[half][4]; half by split: 0 none, 1 x >= 2, 2 y >= 2),
 * the indices into sel; 0 when it is not better than the best so far
 * (the largest error, then the sum of squares; equal is not better, so
 * the first found stays)
 */
static int
r_try(struct rsearch *S, const int pal[2][4], int split, int *sel,
    double *mx_out, double *sq_out)
{
	double mx = 0, sq = 0;
	int i, k;

	for (i = 0; i < 16; i++) {
		int half = split == 1 ? (i & 3) >= 2 : split == 2 ? (i >> 2) >= 2 :
		    0;
		double be = INFINITY;
		sel[i] = 0;
		if (!S->care[i])
			continue;
		for (k = 0; k < 4; k++) {
			double e = fabs(rd(S->srgb, pal[half][k]) - S->t[i]);
			if (e < be) {
				be = e;
				sel[i] = k;
			}
		}
		if (be > mx)
			mx = be;
		if (mx > S->best)
			return 0;
		sq += be * be;
	}
	if (S->found && (mx > S->best || (mx == S->best && sq >= S->bestsq)))
		return 0;
	*mx_out = mx;
	*sq_out = sq;
	return 1;
}

static void
r_keep(struct rsearch *S, double mx, double sq)
{
	S->best = mx;
	S->bestsq = sq;
	S->found = 1;
}

/* the index bits of 16 2-bit selectors (texel x + 4y) */
static uint64_t
idx_bits(const int *sel)
{
	uint64_t w = 0;
	int i;

	for (i = 0; i < 16; i++) {
		int k = (i >> 2) + 4 * (i & 3);
		w |= (uint64_t)(sel[i] >> 1 & 1) << (16 + k) |
		    (uint64_t)(sel[i] & 1) << k;
	}
	return w;
}

double
etc2_encode_r(const double t[16], const uint8_t care[16], int srgb,
    uint8_t out[8], struct etc2_info *info)
{
	struct rsearch S = {t, care, INFINITY, INFINITY, 0, srgb};
	int sel[16], pal[2][4], flip, b0, b1, t0, t1, d, k, c;
	double mx, sq;
	uint64_t bw = 0;

	memset(info, 0, sizeof *info);
	/* individual: 4-bit bases, a table each half */
	for (flip = 0; flip < 2; flip++)
		for (b0 = 0; b0 < 16; b0++)
			for (t0 = 0; t0 < 8; t0++)
				for (b1 = 0; b1 < 16; b1++)
					for (t1 = 0; t1 < 8; t1++) {
						for (k = 0; k < 4; k++) {
							pal[0][k] = clamp255(ext(b0, 4) + etc1_mods[t0][k]);
							pal[1][k] = clamp255(ext(b1, 4) + etc1_mods[t1][k]);
						}
						if (!r_try(&S, (const int (*)[4])pal, flip ? 2 : 1, sel,
						    &mx, &sq))
							continue;
						r_keep(&S, mx, sq);
						bw = (uint64_t)b0 << 60 | (uint64_t)b1 << 56 |
						    (uint64_t)t0 << 37 | (uint64_t)t1 << 34 |
						    (uint64_t)flip << 32 | idx_bits(sel);
						info->mode = ETC2_INDIVIDUAL;
						info->flip = flip;
						info->v[0] = ext(b0, 4);
						info->v[1] = ext(b1, 4);
						info->v[2] = t0;
						info->v[3] = t1;
					}
	/* differential: a 5-bit base and the other within -4..3 of it; G
	 * and B 0 with no difference, so neither leaves its range */
	for (flip = 0; flip < 2; flip++)
		for (b0 = 0; b0 < 32; b0++)
			for (t0 = 0; t0 < 8; t0++)
				for (d = 0; d < 8; d++)
					for (t1 = 0; t1 < 8; t1++) {
						b1 = b0 + etc_delta[d];
						if (b1 < 0 || b1 > 31)
							continue;
						for (k = 0; k < 4; k++) {
							pal[0][k] = clamp255(ext(b0, 5) + etc1_mods[t0][k]);
							pal[1][k] = clamp255(ext(b1, 5) + etc1_mods[t1][k]);
						}
						if (!r_try(&S, (const int (*)[4])pal, flip ? 2 : 1, sel,
						    &mx, &sq))
							continue;
						r_keep(&S, mx, sq);
						bw = (uint64_t)b0 << 59 | (uint64_t)d << 56 |
						    (uint64_t)t0 << 37 | (uint64_t)t1 << 34 |
						    1ULL << 33 | (uint64_t)flip << 32 | idx_bits(sel);
						info->mode = ETC2_DIFFERENTIAL;
						info->flip = flip;
						info->v[0] = ext(b0, 5);
						info->v[1] = ext(b1, 5);
						info->v[2] = t0;
						info->v[3] = t1;
					}
	/* T: base 1, base 2 + d, base 2, base 2 - d */
	for (b0 = 0; b0 < 16; b0++)
		for (b1 = 0; b1 < 16; b1++)
			for (d = 0; d < 8; d++) {
				int a = b0 >> 2, lo = b0 & 3, e2 = ext(b1, 4);
				pal[0][0] = ext(b0, 4);
				pal[0][1] = clamp255(e2 + etc2_dist[d]);
				pal[0][2] = e2;
				pal[0][3] = clamp255(e2 - etc2_dist[d]);
				if (!r_try(&S, (const int (*)[4])pal, 0, sel, &mx, &sq))
					continue;
				r_keep(&S, mx, sq);
				/* R + dR out of 0..31 through the free bits 63..61 and
				 * 58: 28 + a + lo when a + lo >= 4, else a + lo - 4 */
				bw = (a + lo >= 4 ? 7ULL << 61 : 1ULL << 58) |
				    (uint64_t)a << 59 | (uint64_t)lo << 56 |
				    (uint64_t)b1 << 44 | (uint64_t)(d >> 1) << 34 | 1ULL << 33 |
				    (uint64_t)(d & 1) << 32 | idx_bits(sel);
				info->mode = ETC2_T;
				info->v[0] = ext(b0, 4);
				info->v[1] = e2;
				info->v[2] = etc2_dist[d];
			}
	/* H: base 1 +- d, base 2 +- d; d's low bit is whether base 1 >= base
	 * 2, so an unequal pair is written in the order that gives it, and
	 * an equal one decides it by base 2's blue */
	for (b0 = 0; b0 < 16; b0++)
		for (b1 = 0; b1 < 16; b1++)
			for (d = 0; d < 8; d++) {
				int ge = d & 1, blue2 = 0;
				if (b0 != b1 && (b0 > b1) != ge)
					continue;
				if (b0 == b1 && !ge)
					blue2 = 1;	/* base 2 the greater */
				pal[0][0] = clamp255(ext(b0, 4) + etc2_dist[d]);
				pal[0][1] = clamp255(ext(b0, 4) - etc2_dist[d]);
				pal[0][2] = clamp255(ext(b1, 4) + etc2_dist[d]);
				pal[0][3] = clamp255(ext(b1, 4) - etc2_dist[d]);
				if (!r_try(&S, (const int (*)[4])pal, 0, sel, &mx, &sq))
					continue;
				r_keep(&S, mx, sq);
				/* G1 = B1 = 0: R + dR is R in range; G + dG is 0 - 4
				 * (bit 50) */
				bw = (uint64_t)b0 << 59 | 1ULL << 50 | (uint64_t)b1 << 43 |
				    (uint64_t)blue2 << 35 | (uint64_t)(d >> 2 & 1) << 34 |
				    1ULL << 33 | (uint64_t)(d >> 1 & 1) << 32 |
				    idx_bits(sel);
				info->mode = ETC2_H;
				info->v[0] = ext(b0, 4);
				info->v[1] = ext(b1, 4);
				info->v[2] = etc2_dist[d];
			}
	/* planar: O, H, V, 6 bits each; no indices */
	for (b0 = 0; b0 < 64; b0++)
		for (b1 = 0; b1 < 64; b1++)
			for (c = 0; c < 64; c++) {
				int o = ext(b0, 6), h = ext(b1, 6), v = ext(c, 6), i, x;
				double e;
				mx = sq = 0;
				for (i = 0; i < 16 && mx <= S.best; i++) {
					if (!care[i])
						continue;
					x = clamp255(floor4((i & 3) * (h - o) + (i >> 2) *
					    (v - o) + 4 * o + 2));
					e = fabs(rd(srgb, x) - t[i]);
					if (e > mx)
						mx = e;
					sq += e * e;
				}
				if (i < 16 || (S.found && (mx > S.best || (mx == S.best &&
				    sq >= S.bestsq))))
					continue;
				r_keep(&S, mx, sq);
				{
					/* R + dR in range by bit 63 (dR from RO's low
					 * bits, GO 0); G + dG 0; B + dB 0 - 4 (bit 42) */
					int x0 = b0 >> 2, dr = etc_delta[(b0 & 3) << 1];
					bw = (uint64_t)(x0 + dr < 0) << 63 | (uint64_t)b0 << 57 |
					    1ULL << 42 | (uint64_t)(b1 >> 1) << 34 | 1ULL << 33 |
					    (uint64_t)(b1 & 1) << 32 | (uint64_t)c << 13;
				}
				info->mode = ETC2_PLANAR;
				info->v[0] = o;
				info->v[1] = h;
				info->v[2] = v;
			}
	put_word(bw, out);
	return S.best;
}

/* ---- ETC2's 8-bit alpha: EAC with base + m mod, clamped, over 255 ---- */

int
etc2_alpha_value(int base, int mul, int tab, int sel)
{
	return clamp255(base + eac_mod(tab, sel) * mul);
}

void
etc2_alpha_decode(const uint8_t q[8], int v[16])
{
	uint64_t w = word_of(q);
	int i;

	for (i = 0; i < 16; i++)
		v[i] = etc2_alpha_value(q[0], (int)(w >> 52 & 15),
		    (int)(w >> 48 & 15), (int)(w >> (45 - 3 * (4 * (i & 3) +
		    (i >> 2))) & 7));
}

double
etc2_alpha_encode(const double t[16], const uint8_t care[16],
    uint8_t out[8])
{
	double best = INFINITY, bestsq = INFINITY, pal[8];
	int b, mul, tab, i, k, sel[16], bsel[16], bb = 0, bm = 1, bt = 0;
	uint64_t w;

	memset(bsel, 0, sizeof bsel);
	/* the specification does not let an encoder write multiplier 0 */
	for (b = 0; b < 256; b++)
		for (mul = 1; mul < 16; mul++)
			for (tab = 0; tab < 16; tab++) {
				double mx = 0, sq = 0;
				for (k = 0; k < 8; k++)
					pal[k] = rd(0, etc2_alpha_value(b, mul, tab, k));
				for (i = 0; i < 16 && mx <= best; i++) {
					double be = INFINITY;
					sel[i] = 0;
					if (!care[i])
						continue;
					for (k = 0; k < 8; k++) {
						double e = fabs(pal[k] - t[i]);
						if (e < be) {
							be = e;
							sel[i] = k;
						}
					}
					if (be > mx)
						mx = be;
					sq += be * be;
				}
				if (i < 16)
					continue;
				if (mx < best || (mx == best && sq < bestsq)) {
					best = mx;
					bestsq = sq;
					bb = b;
					bm = mul;
					bt = tab;
					memcpy(bsel, sel, sizeof sel);
				}
			}
	w = (uint64_t)bb << 56 | (uint64_t)bm << 52 | (uint64_t)bt << 48;
	for (i = 0; i < 16; i++)
		w |= (uint64_t)bsel[i] << (45 - 3 * (4 * (i & 3) + (i >> 2)));
	put_word(w, out);
	return best;
}
