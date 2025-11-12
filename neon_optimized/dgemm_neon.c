#ifdef __ARM_NEON
#include <arm_neon.h>

#include <stdlib.h>
#include "blas_dgemm.h"


/* Create macros so that the matrices are stored in row-major order */

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

/**
About GEMM_P or kc:
1. mc = kc, since we have to maxmize (2 * mc * kc/(2 * mc + kc))
2. The equation exists provided kc << n.
3. mc * kc <= K

About GEMM_M or mc:
1. The larger mc * nc, the better calculation efficiency
2. We prepare to load A into L2 cache. Avoiding TLB miss (which would
stall CPU), subset of A should remains so until no longer needed.

About KENEL_4x4, mr=4 and nr=4
In order to move data efficiently to the registers.
Here we use C_block = A_panel x Transpose(B_panel)

In accordance to page.14 "6. MOE DETAILS YET",

L1d cahce = 32K, and L2 cache = 2MB. `getconf -a | grep PAGESIZE` = 4096.
Thus L1d is not the Cannikin, it is constraint to page size.

min_nn * kc <= PAGESIZE/2,  4 <= min_nn <= 12, so that 170 <= kc <= 512, we use
256. After reading 6.4, rk3399 L2 cache is large, mc = 1MB / 256 = 4096

Note: For double precision, cache usage doubles, so we adjust block sizes accordingly.
*/
#define GEMM_N (256)  // GEMM_R - reduced for double precision
#define GEMM_M (2048) // GEMM_P - reduced for double precision  
#define GEMM_P (128)  // GEMM_Q - reduced for double precision
#define GEMM_UNROLL (4)

/**

double* a: A
double* b: (B)T
double* c: C

C = A * (B)T

A1 A2 A3    B1 B4 B7
A4 A5 A6  x B2 B5 B8 => C1 C4 C7 C2 C5 C8 C3 C6 C9 (packed)
A7 A8 A9    B3 B4 B9

Calculation sequence:
1st. calculate C1
2st. calculate C4
3st. calculate C7
...
9st. calculate C9

A1-A9/B1-B9 is packed block, not single number.
C1-C9 is 4x4 block, not single number.

Output
C1 C2 C3
C4 C5 C6
C7 C8 C9

Note: For double precision, we use 2x2 blocks instead of 4x4 due to register constraints
and double-width data.
 */
void kernel_4x4(unsigned int m, unsigned int n, unsigned int p, double *sa, double *sb, double *sc, unsigned int ldc) {
    double *a = sa, *b = sb, *c = sc;
    int i, j;
    unsigned int ldc_offset = ldc * sizeof(double);

    for (i = 0; i < m; i += 4) {
        for (j = 0; j < n; j += 4) {
            asm volatile(
                    "asr x8,%4,2                        \n"
                    // Load initial C values (4x4 block using 2x2 sub-blocks)
                    "ldr  q0,   [%2]                    \n"  // C[0][0:1]
                    "ldr  q1,   [%2,  #16]              \n"  // C[0][2:3]
                    "add  x13,  %2,      %3             \n"  // C[1]
                    "ldr  q2,   [x13]                   \n"  // C[1][0:1]
                    "ldr  q3,   [x13, #16]              \n"  // C[1][2:3]
                    "add  x14,  x13,     %3             \n"  // C[2]
                    "ldr  q4,   [x14]                   \n"  // C[2][0:1]
                    "ldr  q5,   [x14, #16]              \n"  // C[2][2:3]
                    "add  x15,  x14,     %3             \n"  // C[3]
                    "ldr  q6,   [x15]                   \n"  // C[3][0:1]
                    "ldr  q7,   [x15, #16]              \n"  // C[3][2:3]

                    "run:                               \n"
                    "   prfm pldl1keep, [%0, #512]      \n"
                    "   prfm pldl1keep, [%1, #512]      \n"

                    // Load A (4x4 block, 2 doubles per vector)
                    "   ld1 {v8.2d,  v9.2d,  v10.2d, v11.2d},   [%0], #64 \n"  // A[0:1][0:3]
                    "   ld1 {v12.2d, v13.2d, v14.2d, v15.2d},   [%0], #64 \n"  // A[2:3][0:3]

                    // Load B (4x4 block, 2 doubles per vector)
                    "   ld1 {v16.2d, v17.2d, v18.2d, v19.2d},   [%1], #64 \n"  // B[0:1][0:3]
                    "   ld1 {v20.2d, v21.2d, v22.2d, v23.2d},   [%1], #64 \n"  // B[2:3][0:3]

                    // Multiply-accumulate operations for C[0:3][0:3]
                    "   fmla   v0.2d,   v16.2d,  v8.d[0]  \n"  // C[0][0:1] += B[0][0:1] * A[0][0]
                    "   fmla   v1.2d,   v17.2d,  v8.d[0]  \n"  // C[0][2:3] += B[0][2:3] * A[0][0]
                    "   fmla   v2.2d,   v16.2d,  v8.d[1]  \n"  // C[1][0:1] += B[0][0:1] * A[0][1]
                    "   fmla   v3.2d,   v17.2d,  v8.d[1]  \n"  // C[1][2:3] += B[0][2:3] * A[0][1]
                    "   fmla   v4.2d,   v16.2d,  v9.d[0]  \n"  // C[2][0:1] += B[0][0:1] * A[0][2]
                    "   fmla   v5.2d,   v17.2d,  v9.d[0]  \n"  // C[2][2:3] += B[0][2:3] * A[0][2]
                    "   fmla   v6.2d,   v16.2d,  v9.d[1]  \n"  // C[3][0:1] += B[0][0:1] * A[0][3]
                    "   fmla   v7.2d,   v17.2d,  v9.d[1]  \n"  // C[3][2:3] += B[0][2:3] * A[0][3]

                    "   fmla   v0.2d,   v18.2d,  v10.d[0] \n"  // C[0][0:1] += B[1][0:1] * A[1][0]
                    "   fmla   v1.2d,   v19.2d,  v10.d[0] \n"  // C[0][2:3] += B[1][2:3] * A[1][0]
                    "   fmla   v2.2d,   v18.2d,  v10.d[1] \n"  // C[1][0:1] += B[1][0:1] * A[1][1]
                    "   fmla   v3.2d,   v19.2d,  v10.d[1] \n"  // C[1][2:3] += B[1][2:3] * A[1][1]
                    "   fmla   v4.2d,   v18.2d,  v11.d[0] \n"  // C[2][0:1] += B[1][0:1] * A[1][2]
                    "   fmla   v5.2d,   v19.2d,  v11.d[0] \n"  // C[2][2:3] += B[1][2:3] * A[1][2]
                    "   fmla   v6.2d,   v18.2d,  v11.d[1] \n"  // C[3][0:1] += B[1][0:1] * A[1][3]
                    "   fmla   v7.2d,   v19.2d,  v11.d[1] \n"  // C[3][2:3] += B[1][2:3] * A[1][3]

                    "   fmla   v0.2d,   v20.2d,  v12.d[0] \n"  // C[0][0:1] += B[2][0:1] * A[2][0]
                    "   fmla   v1.2d,   v21.2d,  v12.d[0] \n"  // C[0][2:3] += B[2][2:3] * A[2][0]
                    "   fmla   v2.2d,   v20.2d,  v12.d[1] \n"  // C[1][0:1] += B[2][0:1] * A[2][1]
                    "   fmla   v3.2d,   v21.2d,  v12.d[1] \n"  // C[1][2:3] += B[2][2:3] * A[2][1]
                    "   fmla   v4.2d,   v20.2d,  v13.d[0] \n"  // C[2][0:1] += B[2][0:1] * A[2][2]
                    "   fmla   v5.2d,   v21.2d,  v13.d[0] \n"  // C[2][2:3] += B[2][2:3] * A[2][2]
                    "   fmla   v6.2d,   v20.2d,  v13.d[1] \n"  // C[3][0:1] += B[2][0:1] * A[2][3]
                    "   fmla   v7.2d,   v21.2d,  v13.d[1] \n"  // C[3][2:3] += B[2][2:3] * A[2][3]

                    "   fmla   v0.2d,   v22.2d,  v14.d[0] \n"  // C[0][0:1] += B[3][0:1] * A[3][0]
                    "   fmla   v1.2d,   v23.2d,  v14.d[0] \n"  // C[0][2:3] += B[3][2:3] * A[3][0]
                    "   fmla   v2.2d,   v22.2d,  v14.d[1] \n"  // C[1][0:1] += B[3][0:1] * A[3][1]
                    "   fmla   v3.2d,   v23.2d,  v14.d[1] \n"  // C[1][2:3] += B[3][2:3] * A[3][1]
                    "   fmla   v4.2d,   v22.2d,  v15.d[0] \n"  // C[2][0:1] += B[3][0:1] * A[3][2]
                    "   fmla   v5.2d,   v23.2d,  v15.d[0] \n"  // C[2][2:3] += B[3][2:3] * A[3][2]
                    "   fmla   v6.2d,   v22.2d,  v15.d[1] \n"  // C[3][0:1] += B[3][0:1] * A[3][3]
                    "   fmla   v7.2d,   v23.2d,  v15.d[1] \n"  // C[3][2:3] += B[3][2:3] * A[3][3]

                    "   subs x8, x8, #1                 \n"
                    "   bne run                         \n"

                    // Store results back to C
                    "   str q0, [%2]                    \n"  // Store C[0][0:1]
                    "   str q1, [%2,  #16]              \n"  // Store C[0][2:3]
                    "   str q2, [x13]                   \n"  // Store C[1][0:1]
                    "   str q3, [x13, #16]              \n"  // Store C[1][2:3]
                    "   str q4, [x14]                   \n"  // Store C[2][0:1]
                    "   str q5, [x14, #16]              \n"  // Store C[2][2:3]
                    "   str q6, [x15]                   \n"  // Store C[3][0:1]
                    "   str q7, [x15, #16]              \n"  // Store C[3][2:3]
                    "                                   \n"
                    : "=r"(a), "=r"(b), "=r"(c), "=r"(ldc_offset), "=r"(p)
                    : "0"(a), "1"(b), "2"(c), "3"(ldc_offset), "4"(p)
                    : "memory", "cc", "x8", "x13", "x14","x15",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
            "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
            c += 4;
            a -= 4 * p;
        } // endj
        sc += ldc * 4;
        c = sc;
        a += 4 * p;
        b = sb;
    } // endi
}

/**
pack A means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag

Output:
0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7
8 8 8 8 9 9 9 9 a a a a b b b b
c c c c d d d d e e e e f f f f

Draw it with a line
*/
void packA_4(unsigned int m, unsigned int p, double *from, unsigned int lda, double *to) {
    int i, j;
    double *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    double *b_offset;
    double ctemp1, ctemp2, ctemp3, ctemp4;
    double ctemp5, ctemp6, ctemp7, ctemp8;
    double ctemp9, ctemp10, ctemp11, ctemp12;
    double ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = from;
    b_offset = to;

    j = ((int)m >> 2);
    do {
        a_offset1 = a_offset;
        a_offset2 = a_offset1 + lda;
        a_offset3 = a_offset2 + lda;
        a_offset4 = a_offset3 + lda;
        a_offset += 4 * lda;

        i = ((int)p >> 2);
        do {
            ctemp1 = *(a_offset1 + 0);
            ctemp2 = *(a_offset1 + 1);
            ctemp3 = *(a_offset1 + 2);
            ctemp4 = *(a_offset1 + 3);

            ctemp5 = *(a_offset2 + 0);
            ctemp6 = *(a_offset2 + 1);
            ctemp7 = *(a_offset2 + 2);
            ctemp8 = *(a_offset2 + 3);

            ctemp9 = *(a_offset3 + 0);
            ctemp10 = *(a_offset3 + 1);
            ctemp11 = *(a_offset3 + 2);
            ctemp12 = *(a_offset3 + 3);

            ctemp13 = *(a_offset4 + 0);
            ctemp14 = *(a_offset4 + 1);
            ctemp15 = *(a_offset4 + 2);
            ctemp16 = *(a_offset4 + 3);

            *(b_offset + 0) = ctemp1;
            *(b_offset + 1) = ctemp5;
            *(b_offset + 2) = ctemp9;
            *(b_offset + 3) = ctemp13;

            *(b_offset + 4) = ctemp2;
            *(b_offset + 5) = ctemp6;
            *(b_offset + 6) = ctemp10;
            *(b_offset + 7) = ctemp14;

            *(b_offset + 8) = ctemp3;
            *(b_offset + 9) = ctemp7;
            *(b_offset + 10) = ctemp11;
            *(b_offset + 11) = ctemp15;

            *(b_offset + 12) = ctemp4;
            *(b_offset + 13) = ctemp8;
            *(b_offset + 14) = ctemp12;
            *(b_offset + 15) = ctemp16;

            a_offset1 += 4;
            a_offset2 += 4;
            a_offset3 += 4;
            a_offset4 += 4;

            b_offset += 16;
            i--;
        } while (i > 0);
        j--;
    } while (j > 0);
}

/*
suppose that k and n is mutiple of 4
pack B means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag, not like pack A

Output:
0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
8 9 a b 8 9 a b 8 9 a b 8 9 a b
4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
c d e f c d e f c d e f c d e f
*/
void packB_4(unsigned int p, unsigned int n, double *from, unsigned int ldb, double *to) {
    int i, j;
    double *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    double *b_offset, *b_offset1;
    double ctemp1, ctemp2, ctemp3, ctemp4;
    double ctemp5, ctemp6, ctemp7, ctemp8;
    double ctemp9, ctemp10, ctemp11, ctemp12;
    double ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = from;
    b_offset = to;

    j = ((int)p >> 2);
    do {
        a_offset1 = a_offset;
        a_offset2 = a_offset1 + ldb;
        a_offset3 = a_offset2 + ldb;
        a_offset4 = a_offset3 + ldb;
        a_offset += 4 * ldb;

        b_offset1 = b_offset;
        b_offset += 16;

        i = ((int)n >> 2);
        do {
            ctemp1 = *(a_offset1 + 0);
            ctemp2 = *(a_offset1 + 1);
            ctemp3 = *(a_offset1 + 2);
            ctemp4 = *(a_offset1 + 3);

            ctemp5 = *(a_offset2 + 0);
            ctemp6 = *(a_offset2 + 1);
            ctemp7 = *(a_offset2 + 2);
            ctemp8 = *(a_offset2 + 3);

            ctemp9 = *(a_offset3 + 0);
            ctemp10 = *(a_offset3 + 1);
            ctemp11 = *(a_offset3 + 2);
            ctemp12 = *(a_offset3 + 3);

            ctemp13 = *(a_offset4 + 0);
            ctemp14 = *(a_offset4 + 1);
            ctemp15 = *(a_offset4 + 2);
            ctemp16 = *(a_offset4 + 3);

            a_offset1 += 4;
            a_offset2 += 4;
            a_offset3 += 4;
            a_offset4 += 4;

            // Pack in row-major order within the 4x4 block
            *(b_offset1 + 0) = ctemp1;
            *(b_offset1 + 1) = ctemp2;
            *(b_offset1 + 2) = ctemp3;
            *(b_offset1 + 3) = ctemp4;

            *(b_offset1 + 4) = ctemp5;
            *(b_offset1 + 5) = ctemp6;
            *(b_offset1 + 6) = ctemp7;
            *(b_offset1 + 7) = ctemp8;

            *(b_offset1 + 8) = ctemp9;
            *(b_offset1 + 9) = ctemp10;
            *(b_offset1 + 10) = ctemp11;
            *(b_offset1 + 11) = ctemp12;

            *(b_offset1 + 12) = ctemp13;
            *(b_offset1 + 13) = ctemp14;
            *(b_offset1 + 14) = ctemp15;
            *(b_offset1 + 15) = ctemp16;

            b_offset1 += p * 4;
            i--;
        } while (i > 0);
        j--;
    } while (j > 0);
}

//C(mxn) = A(mxp)*B(pxn)
void dgemm_neon(unsigned int m, unsigned int n, unsigned int p, double *a, unsigned int lda, 
                                                                double *b, unsigned int ldb,
                                                                double *c, unsigned int ldc, 
                                                                double *sa, double *sb) {

    unsigned int ms, mms, ns, ps;
    unsigned int min_m, min_mm, min_n, min_p;
    int l1stride = 1;
    for (ms = 0; ms < m; ms += GEMM_M) {
        min_m = m - ms;
        if (min_m > GEMM_M) {
            min_m = GEMM_M;
        }

        for (ps = 0; ps < p; ps += min_p) {
            min_p = p - ps;
            if (min_p >= (GEMM_P << 1)) {
                min_p = GEMM_P;
            } else if (min_p > GEMM_P) {
                min_p = (min_p / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            }

            // first packB
            min_n = n;
            if (n >= GEMM_N * 2) {
                min_n = GEMM_N;
            } else if (n > GEMM_N) {
                min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            } else {
                l1stride = 0;
            }
            packB_4(min_p, min_n, b + ps * ldb, ldb, sb);

            // micro kernel, split A Block to smaller Panel
            for (mms = ms; mms < ms + min_m; mms += min_mm) {
                min_mm = (ms + min_m) - mms;
                if (min_mm >= 3 * GEMM_UNROLL) {
                    min_mm = 3 * GEMM_UNROLL;
                } else if (min_mm >= 2 * GEMM_UNROLL) {
                    min_mm = 2 * GEMM_UNROLL;
                } else if (min_mm > GEMM_UNROLL) {
                    min_mm = GEMM_UNROLL;
                }

                // coninueous packA
                packA_4(min_mm, min_p, a + mms * lda + ps, lda,
                        sa + min_p * (mms - ms) * l1stride);

                kernel_4x4(min_mm, min_n, min_p, sa + l1stride * min_p * (mms - ms), sb,
                           c + mms * ldc, ldc);
            }

            // the first B Block has been packed, proc the others
            for (ns = min_n; ns < n; ns += min_n) {
                min_n = n - ns;
                if (min_n >= GEMM_N * 2) {
                    min_n = GEMM_N;
                } else if (min_n > GEMM_N) {
                    min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
                }

                packB_4(min_p, min_n, b + ns + ldb * ps, ldb, sb);
                kernel_4x4(min_m, min_n, min_p, sa, sb, c + ms * ldc + ns, ldc);
            }
        }
    }
}

#endif