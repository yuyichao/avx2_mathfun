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

#include "avx2_mathfun.h"

/* __m128 is ugly to write */
typedef __m256  v8sf; // vector of 8 float (avx)
typedef __m256i v8si; // vector of 8 int   (avx)

/* the smallest non denormalized float number */
A2M_PI32_CONST256(min_norm_pos, 0x00800000);
A2M_PI32_CONST256(inv_mant_mask, ~0x7f800000);

A2M_PI32_CONST256(0x7f, 0x7f);

A2M_PS256_CONST(cephes_SQRTHF, 0.707106781186547524);
A2M_PS256_CONST(cephes_log_p0, 7.0376836292E-2);
A2M_PS256_CONST(cephes_log_p1, - 1.1514610310E-1);
A2M_PS256_CONST(cephes_log_p2, 1.1676998740E-1);
A2M_PS256_CONST(cephes_log_p3, - 1.2420140846E-1);
A2M_PS256_CONST(cephes_log_p4, + 1.4249322787E-1);
A2M_PS256_CONST(cephes_log_p5, - 1.6668057665E-1);
A2M_PS256_CONST(cephes_log_p6, + 2.0000714765E-1);
A2M_PS256_CONST(cephes_log_p7, - 2.4999993993E-1);
A2M_PS256_CONST(cephes_log_p8, + 3.3333331174E-1);
A2M_PS256_CONST(cephes_log_q1, -2.12194440e-4);
A2M_PS256_CONST(cephes_log_q2, 0.693359375);

/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0 */
v8sf a2m_logf(v8sf x)
{
    v8si imm0;
    v8sf one = *(v8sf*)a2m_ps256_1;

    v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, *(v8sf*)a2m_pi256_min_norm_pos);  /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_srli_epi32((v8si)x, 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, *(v8sf*)a2m_pi256_inv_mant_mask);
    x = _mm256_or_ps(x, *(v8sf*)a2m_ps256_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi32(imm0, *(v8si*)a2m_pi256_0x7f);
    v8sf e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    /* part2:
       if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    v8sf mask = _mm256_cmp_ps(x, *(v8sf*)a2m_ps256_cephes_SQRTHF, _CMP_LT_OS);
    v8sf tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    v8sf z = _mm256_mul_ps(x,x);

    v8sf y = *(v8sf*)a2m_ps256_cephes_log_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_log_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_log_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_log_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_log_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_log_p5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_log_p6);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_log_p7);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    tmp = _mm256_mul_ps(e, *(v8sf*)a2m_ps256_cephes_log_q1);
    y = _mm256_add_ps(y, tmp);


    tmp = _mm256_mul_ps(z, *(v8sf*)a2m_ps256_0p5);
    y = _mm256_sub_ps(y, tmp);

    tmp = _mm256_mul_ps(e, *(v8sf*)a2m_ps256_cephes_log_q2);
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
    return x;
}

A2M_PS256_CONST(exp_hi, 88.3762626647949f);
A2M_PS256_CONST(exp_lo, -88.3762626647949f);

A2M_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
A2M_PS256_CONST(cephes_exp_C1, 0.693359375);
A2M_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

A2M_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
A2M_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
A2M_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
A2M_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
A2M_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
A2M_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

v8sf a2m_expf(v8sf x)
{
    v8sf tmp = _mm256_setzero_ps(), fx;
    v8si imm0;
    v8sf one = *(v8sf*)a2m_ps256_1;

    x = _mm256_min_ps(x, *(v8sf*)a2m_ps256_exp_hi);
    x = _mm256_max_ps(x, *(v8sf*)a2m_ps256_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm256_mul_ps(x, *(v8sf*)a2m_ps256_cephes_LOG2EF);
    fx = _mm256_add_ps(fx, *(v8sf*)a2m_ps256_0p5);

    /* how to perform a floorf with SSE: just below */
    // imm0 = _mm256_cvttps_epi32(fx);
    // tmp  = _mm256_cvtepi32_ps(imm0);

    tmp = _mm256_floor_ps(fx);

    /* if greater, substract 1 */
    v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);

    tmp = _mm256_mul_ps(fx, *(v8sf*)a2m_ps256_cephes_exp_C1);
    v8sf z = _mm256_mul_ps(fx, *(v8sf*)a2m_ps256_cephes_exp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);

    z = _mm256_mul_ps(x,x);

    v8sf y = *(v8sf*)a2m_ps256_cephes_exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm256_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm256_add_epi32(imm0, *(v8si*)a2m_pi256_0x7f);
    imm0 = _mm256_slli_epi32(imm0, 23);
    y = _mm256_mul_ps(y, (v8sf)imm0);
    return y;
}

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result. */
v8sf a2m_sinf(v8sf x) // any x
{
    v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
    v8si imm0, imm2;

    sign_bit = x;
    /* take the absolute value */
    x = _mm256_and_ps(x, *(v8sf*)a2m_pi256_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm256_and_ps(sign_bit, *(v8sf*)a2m_pi256_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(v8sf*)a2m_ps256_cephes_FOPI);

    /*
      Here we start a series of integer operations, which are in the
      realm of AVX2.
      If we don't have AVX, let's perform them using SSE2 directives
    */

    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    // another two AVX2 instruction
    imm2 = _mm256_add_epi32(imm2, *(v8si*)a2m_pi256_1);
    imm2 = _mm256_and_si256(imm2, *(v8si*)a2m_pi256_inv1);
    y = _mm256_cvtepi32_ps(imm2);

    /* get the swap sign flag */
    imm0 = _mm256_and_si256(imm2, *(v8si*)a2m_pi256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    /* get the polynom selection mask
       there is one polynom for 0 <= x <= Pi/4
       and another one for Pi/4<x<=Pi/2

       Both branches will be computed.
    */
    imm2 = _mm256_and_si256(imm2, *(v8si*)a2m_pi256_2);
    imm2 = _mm256_cmpeq_epi32(imm2,*(v8si*)a2m_pi256_0);

    v8sf swap_sign_bit = (v8sf)imm0;
    v8sf poly_mask = (v8sf)imm2;
    sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(v8sf*)a2m_ps256_minus_cephes_DP1;
    xmm2 = *(v8sf*)a2m_ps256_minus_cephes_DP2;
    xmm3 = *(v8sf*)a2m_ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(v8sf*)a2m_ps256_coscof_p0;
    v8sf z = _mm256_mul_ps(x,x);

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    v8sf tmp = _mm256_mul_ps(z, *(v8sf*)a2m_ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = *(v8sf*)a2m_ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf*)a2m_ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf*)a2m_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y,y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

/* almost the same as sin_ps */
v8sf a2m_cosf(v8sf x) // any x
{
    v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
    v8si imm0, imm2;

    /* take the absolute value */
    x = _mm256_and_ps(x, *(v8sf*)a2m_pi256_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm256_mul_ps(x, *(v8sf*)a2m_ps256_cephes_FOPI);

    /* store the integer part of y in mm0 */
    imm2 = _mm256_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm256_add_epi32(imm2, *(v8si*)a2m_pi256_1);
    imm2 = _mm256_and_si256(imm2, *(v8si*)a2m_pi256_inv1);
    y = _mm256_cvtepi32_ps(imm2);
    imm2 = _mm256_sub_epi32(imm2, *(v8si*)a2m_pi256_2);

    /* get the swap sign flag */
    imm0 = _mm256_andnot_si256(imm2, *(v8si*)a2m_pi256_4);
    imm0 = _mm256_slli_epi32(imm0, 29);
    /* get the polynom selection mask */
    imm2 = _mm256_and_si256(imm2, *(v8si*)a2m_pi256_2);
    imm2 = _mm256_cmpeq_epi32(imm2, *(v8si*)a2m_pi256_0);

    v8sf sign_bit = (v8sf)imm0;
    v8sf poly_mask = (v8sf)imm2;

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(v8sf*)a2m_ps256_minus_cephes_DP1;
    xmm2 = *(v8sf*)a2m_ps256_minus_cephes_DP2;
    xmm3 = *(v8sf*)a2m_ps256_minus_cephes_DP3;
    xmm1 = _mm256_mul_ps(y, xmm1);
    xmm2 = _mm256_mul_ps(y, xmm2);
    xmm3 = _mm256_mul_ps(y, xmm3);
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    y = *(v8sf*)a2m_ps256_coscof_p0;
    v8sf z = _mm256_mul_ps(x,x);

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_coscof_p1);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_coscof_p2);
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, z);
    v8sf tmp = _mm256_mul_ps(z, *(v8sf*)a2m_ps256_0p5);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, *(v8sf*)a2m_ps256_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v8sf y2 = *(v8sf*)a2m_ps256_sincof_p0;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf*)a2m_ps256_sincof_p1);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, *(v8sf*)a2m_ps256_sincof_p2);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_mul_ps(y2, x);
    y2 = _mm256_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
    y = _mm256_andnot_ps(xmm3, y);
    y = _mm256_add_ps(y,y2);
    /* update the sign */
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

void a2m_sincosf(v8sf x, v8sf *ps, v8sf *pc)
{
    v8sf s, c;
    a2m_sincosf_(x, &s, &c);
    a2m_storeu(ps, s);
    a2m_storeu(pc, c);
}

void a2m_cisf(v8sf x, v8sf p[2])
{
    v8sf v2[2];
    a2m_cisf_(x, v2);
    a2m_storeu(&p[0], v2[0]);
    a2m_storeu(&p[1], v2[1]);
}
