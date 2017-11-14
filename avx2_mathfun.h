/*
  AVX implementation of sin, cos, sincos, exp and log

  Based on "sse_mathfun.h", by Julien Pommier
  http://gruntthepeon.free.fr/ssemath/

  Copyright (C) 2012 Giovanni Garberoglio
  Interdisciplinary Laboratory for Computational Science (LISC)
  Fondazione Bruno Kessler and University of Trento
  via Sommarive, 18
  I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
  claim that you wrote the original software. If you use this software
  in a product, an acknowledgment in the product documentation would be
  appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
  misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef AVX2_MATHFUN_H__
#define AVX2_MATHFUN_H__

#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

/* declare some AVX constants -- why can't I figure a better way to do that? */
#define A2M_PS256_CONST(Name, Val)                                         \
    static const float a2m_ps256_##Name[8] __attribute__((aligned(32))) = { Val, Val, Val, Val, Val, Val, Val, Val }
#define A2M_PI32_CONST256(Name, Val)                                       \
    static const int a2m_pi32_256_##Name[8] __attribute__((aligned(32))) = { Val, Val, Val, Val, Val, Val, Val, Val }
#define A2M_PS256_CONST_TYPE(Name, Type, Val)                              \
    static const Type a2m_ps256_##Name[8] __attribute__((aligned(32))) = { Val, Val, Val, Val, Val, Val, Val, Val }

A2M_PS256_CONST(1, 1.0f);
A2M_PS256_CONST(0p5, 0.5f);

A2M_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
A2M_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

A2M_PI32_CONST256(0, 0);
A2M_PI32_CONST256(1, 1);
A2M_PI32_CONST256(inv1, ~1);
A2M_PI32_CONST256(2, 2);
A2M_PI32_CONST256(4, 4);

A2M_PS256_CONST(minus_cephes_DP1, -0.78515625);
A2M_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
A2M_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
A2M_PS256_CONST(sincof_p0, -1.9515295891E-4);
A2M_PS256_CONST(sincof_p1,  8.3321608736E-3);
A2M_PS256_CONST(sincof_p2, -1.6666654611E-1);
A2M_PS256_CONST(coscof_p0,  2.443315711809948E-005);
A2M_PS256_CONST(coscof_p1, -1.388731625493765E-003);
A2M_PS256_CONST(coscof_p2,  4.166664568298827E-002);
A2M_PS256_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI

__m256 a2m_logf(__m256 x);
__m256 a2m_expf(__m256 x);
__m256 a2m_sinf(__m256 x);
__m256 a2m_cosf(__m256 x);
void a2m_sincosf(__m256 x, __m256 *s, __m256 *c);
void a2m_cisf(__m256 x, __m256 p[2]);

__attribute__((always_inline)) static inline
void a2m_storeu(__m256 *p, __m256 v)
{
#ifndef __clang__
    // GCC somehow lowers _mm256_storeu_ps to two stores of 16 bytes per 32bit register.
    *(__m256_u*)p = (__m256_u)v;
#else
    _mm256_storeu_ps((float*)p, v);
#endif
}

// Convert a float x8 vector of real part and a float x8 vector of imaginary part
// to two float complex x4 vectors.
__attribute__((always_inline)) static inline
void a2m_pack_complexf(__m256 r, __m256 i, __m256 *lo, __m256 *hi)
{
    // input:
    // r = |r7|r6|r5|r4|r3|r2|r1|r0|
    // i = |i7|i6|i5|i4|i3|i2|i1|i0|
    // goal:
    // *lo = |i3|r3|i2|r2|i1|r1|i0|r0|
    // *hi = |i7|r7|i6|r6|i5|r4|i4|r4|
    // unpacklo and unpackhi almost does that except that it works on 128bit lane
    // instead of the whole 256bit vector.
    // Therefore we need to do cross lane shuffle first before using them.
    // The correct input vectors for them are
    // r' = |r7|r6|r3|r2|r5|r4|r1|r0|
    // i' = |i7|i6|i3|i2|i5|i4|i1|i0|
    // so we can achieve that with one 64bit shuffle on each vector.
    // The mask for the shuffle is [3, 1, 2, 0] or 0b11011000 or 0xd8
    r = (__m256)_mm256_permute4x64_pd((__m256d)r, 0xd8);
    i = (__m256)_mm256_permute4x64_pd((__m256d)i, 0xd8);
    *lo = _mm256_unpacklo_ps(r, i);
    *hi = _mm256_unpackhi_ps(r, i);
}

// sincos and cis are also given below as inlined function to help avoid
// going through memory for the return values.

/* since a2m_sinf and a2m_cosf are almost identical, a2m_sincosf could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
__attribute__((always_inline)) static inline void a2m_sincosf_(__m256 x, __m256 *s, __m256 *c)
{
    __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
    __m256i imm0, imm2, imm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(__m256*)a2m_ps256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(__m256*)a2m_ps256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(__m256*)a2m_ps256_cephes_FOPI);

    /* store the integer part of y in imm2 */
    imm2 = _mm256_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi32(imm2, *(__m256i*)a2m_pi32_256_1);
    imm2 = _mm256_and_si256(imm2, *(__m256i*)a2m_pi32_256_inv1);

    y = _mm256_cvtepi32_ps(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm256_and_si256(imm2, *(__m256i*)a2m_pi32_256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    __m256 swap_sign_bit_sin = (__m256)imm0;

    /* get the polynom selection mask for the sine*/
    imm2 = _mm256_and_si256(imm2, *(__m256i*)a2m_pi32_256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)a2m_pi32_256_0);
    __m256 poly_mask = (__m256)imm2;

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m256*)a2m_ps256_minus_cephes_DP1;
    xmm2 = *(__m256*)a2m_ps256_minus_cephes_DP2;
    xmm3 = *(__m256*)a2m_ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

    imm4 = _mm256_sub_epi32(imm4, *(__m256i*)a2m_pi32_256_2);
    imm4 = _mm256_andnot_si256(imm4, *(__m256i*)a2m_pi32_256_4);
    imm4 = _mm256_slli_epi32(imm4, 29);

    __m256 sign_bit_cos = (__m256)imm4;

    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    __m256 z = _mm256_mul_ps(x,x);
    y = *(__m256*)a2m_ps256_coscof_p0;

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(__m256*)a2m_ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(__m256*)a2m_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    __m256 tmp = _mm256_mul_ps(z, *(__m256*)a2m_ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, *(__m256*)a2m_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m256 y2 = *(__m256*)a2m_ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(__m256*)a2m_ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(__m256*)a2m_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    __m256 ysin2 = _mm256_and_ps(xmm3, y2);
    __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
    y2 = _mm256_sub_ps(y2,ysin2);
    y = _mm256_sub_ps(y, ysin1);

    xmm1 = _mm256_add_ps(ysin1,ysin2);
    xmm2 = _mm256_add_ps(y,y2);

    /* update the sign */
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

__attribute__((always_inline)) static inline void a2m_cisf_(__m256 x, __m256 p[2])
{
    __m256 s, c;
    a2m_sincosf_(x, &s, &c);
    a2m_pack_complexf(c, s, &p[0], &p[1]);
}

#ifdef __cplusplus
}
#endif

#endif
