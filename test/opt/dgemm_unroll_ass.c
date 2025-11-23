
#define DGEMM_IMPLEMENTATION
#define M_BLAS_KERNEL_BLOCK_ROWS 4
#define M_BLAS_KERNEL_BLOCK_COLS 4

#define A(i, j) a[(i) * lda + j]
#define B(i, j) b[(i) * ldb + j]
#define C(i, j) c[(i) * ldc + j]

/* 
 * 使用内联汇编优化的版本 - 针对O0编译
 * 
 * 重要说明：
 * - 使用的是ARM标准浮点指令（VFP），不是NEON SIMD
 * - FMADD是ARMv8标准指令，不需要NEON扩展
 * - 所有ARMv8处理器都支持（包括不支持NEON的精简版）
 * 
 * 优化策略：
 * 1. 使用ARM标量FMA指令（FMADD）- 融合乘加，单条指令完成 a*b+c
 * 2. 显式管理寄存器分配 - 16个累加器固定在d16-d31
 * 3. 使用post-increment寻址 - 减少地址计算
 * 4. 使用配对load/store（LDP/STP）- 减少内存访问次数
 * 
 * 这是最后的优化手段，直接绕过编译器的限制
 */

#ifdef __aarch64__
// ARM64 内联汇编优化版本（使用标准浮点指令，非NEON）

static void addDot4x4_asm(unsigned int p, double *a, unsigned int lda,  
                          double *b, unsigned int ldb, 
                          double *c, unsigned int ldc)
{
    double *a_0p = &A(0, 0);
    double *a_1p = &A(1, 0);
    double *a_2p = &A(2, 0);
    double *a_3p = &A(3, 0);
    
    // 使用内联汇编来强制使用FMA指令和寄存器
    __asm__ __volatile__(
        // 初始化累加器为0（使用浮点寄存器 d16-d31）
        "movi    d16, #0  \n"  // c_00
        "movi    d17, #0  \n"  // c_01
        "movi    d18, #0  \n"  // c_02
        "movi    d19, #0  \n"  // c_03
        "movi    d20, #0  \n"  // c_10
        "movi    d21, #0  \n"  // c_11
        "movi    d22, #0  \n"  // c_12
        "movi    d23, #0  \n"  // c_13
        "movi    d24, #0  \n"  // c_20
        "movi    d25, #0  \n"  // c_21
        "movi    d26, #0  \n"  // c_22
        "movi    d27, #0  \n"  // c_23
        "movi    d28, #0  \n"  // c_30
        "movi    d29, #0  \n"  // c_31
        "movi    d30, #0  \n"  // c_32
        "movi    d31, #0  \n"  // c_33
        
        // 设置循环计数器
        "mov     w12, %w[p]   \n"  // k = p
        "cbz     w12, 2f      \n"  // if p == 0, skip loop
        
        "1:  \n"  // 循环开始
        
        // 加载A矩阵的4个元素（使用post-increment）
        "ldr     d0, [%[a0]], #8  \n"  // a_0p_reg = *a_0p++
        "ldr     d1, [%[a1]], #8  \n"  // a_1p_reg = *a_1p++
        "ldr     d2, [%[a2]], #8  \n"  // a_2p_reg = *a_2p++
        "ldr     d3, [%[a3]], #8  \n"  // a_3p_reg = *a_3p++
        
        // 加载B矩阵的4个元素
        "ldr     d4, [%[b], #0]   \n"  // b[0]
        "ldr     d5, [%[b], #8]   \n"  // b[1]
        "ldr     d6, [%[b], #16]  \n"  // b[2]
        "ldr     d7, [%[b], #24]  \n"  // b[3]
        
        // 使用FMADD指令（融合乘加）- 第0列
        "fmadd   d16, d0, d4, d16  \n"  // c_00 += a_0p * b[0]
        "fmadd   d20, d1, d4, d20  \n"  // c_10 += a_1p * b[0]
        "fmadd   d24, d2, d4, d24  \n"  // c_20 += a_2p * b[0]
        "fmadd   d28, d3, d4, d28  \n"  // c_30 += a_3p * b[0]
        
        // 第1列
        "fmadd   d17, d0, d5, d17  \n"  // c_01 += a_0p * b[1]
        "fmadd   d21, d1, d5, d21  \n"  // c_11 += a_1p * b[1]
        "fmadd   d25, d2, d5, d25  \n"  // c_21 += a_2p * b[1]
        "fmadd   d29, d3, d5, d29  \n"  // c_31 += a_3p * b[1]
        
        // 第2列
        "fmadd   d18, d0, d6, d18  \n"  // c_02 += a_0p * b[2]
        "fmadd   d22, d1, d6, d22  \n"  // c_12 += a_1p * b[2]
        "fmadd   d26, d2, d6, d26  \n"  // c_22 += a_2p * b[2]
        "fmadd   d30, d3, d6, d30  \n"  // c_32 += a_3p * b[2]
        
        // 第3列
        "fmadd   d19, d0, d7, d19  \n"  // c_03 += a_0p * b[3]
        "fmadd   d23, d1, d7, d23  \n"  // c_13 += a_1p * b[3]
        "fmadd   d27, d2, d7, d27  \n"  // c_23 += a_2p * b[3]
        "fmadd   d31, d3, d7, d31  \n"  // c_33 += a_3p * b[3]
        
        // 移动B指针到下一行
        "add     %[b], %[b], %[ldb_bytes]  \n"
        
        // 循环控制
        "subs    w12, w12, #1  \n"
        "b.ne    1b            \n"
        
        "2:  \n"  // 循环结束
        
        // 将结果写回C矩阵
        "ldp     d0, d1, [%[c], #0]   \n"  // 加载C[0][0-1]
        "fadd    d0, d0, d16           \n"
        "fadd    d1, d1, d17           \n"
        "stp     d0, d1, [%[c], #0]   \n"
        
        "ldp     d0, d1, [%[c], #16]  \n"  // 加载C[0][2-3]
        "fadd    d0, d0, d18           \n"
        "fadd    d1, d1, d19           \n"
        "stp     d0, d1, [%[c], #16]  \n"
        
        "add     %[c], %[c], %[ldc_bytes]  \n"  // 移到C的下一行
        
        "ldp     d0, d1, [%[c], #0]   \n"  // C[1][0-1]
        "fadd    d0, d0, d20           \n"
        "fadd    d1, d1, d21           \n"
        "stp     d0, d1, [%[c], #0]   \n"
        
        "ldp     d0, d1, [%[c], #16]  \n"  // C[1][2-3]
        "fadd    d0, d0, d22           \n"
        "fadd    d1, d1, d23           \n"
        "stp     d0, d1, [%[c], #16]  \n"
        
        "add     %[c], %[c], %[ldc_bytes]  \n"
        
        "ldp     d0, d1, [%[c], #0]   \n"  // C[2][0-1]
        "fadd    d0, d0, d24           \n"
        "fadd    d1, d1, d25           \n"
        "stp     d0, d1, [%[c], #0]   \n"
        
        "ldp     d0, d1, [%[c], #16]  \n"  // C[2][2-3]
        "fadd    d0, d0, d26           \n"
        "fadd    d1, d1, d27           \n"
        "stp     d0, d1, [%[c], #16]  \n"
        
        "add     %[c], %[c], %[ldc_bytes]  \n"
        
        "ldp     d0, d1, [%[c], #0]   \n"  // C[3][0-1]
        "fadd    d0, d0, d28           \n"
        "fadd    d1, d1, d29           \n"
        "stp     d0, d1, [%[c], #0]   \n"
        
        "ldp     d0, d1, [%[c], #16]  \n"  // C[3][2-3]
        "fadd    d0, d0, d30           \n"
        "fadd    d1, d1, d31           \n"
        "stp     d0, d1, [%[c], #16]  \n"
        
        : [a0] "+r" (a_0p), [a1] "+r" (a_1p), [a2] "+r" (a_2p), [a3] "+r" (a_3p),
          [b] "+r" (b), [c] "+r" (c)
        : [p] "r" (p), 
          [ldb_bytes] "r" ((long)(ldb * sizeof(double))),
          [ldc_bytes] "r" ((long)(ldc * sizeof(double)))
        : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
          "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
          "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
          "w12", "memory"
    );
}

#else
// 非ARM64平台，使用原始C代码
static void addDot4x4_asm(unsigned int p, double *a, unsigned int lda,  
                          double *b, unsigned int ldb, 
                          double *c, unsigned int ldc)
{
    register unsigned int k;
    register double
        c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,  
        c_10_reg,   c_11_reg,   c_12_reg,   c_13_reg,  
        c_20_reg,   c_21_reg,   c_22_reg,   c_23_reg,  
        c_30_reg,   c_31_reg,   c_32_reg,   c_33_reg,

        a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
        b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;
    
    double *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr;

    a_0p_pntr = &A(0, 0);
    a_1p_pntr = &A(1, 0);
    a_2p_pntr = &A(2, 0);
    a_3p_pntr = &A(3, 0);

    c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
    c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
    c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
    c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

    for(k = 0; k < p; ++k)
    {
        a_0p_reg = *a_0p_pntr++;
        a_1p_reg = *a_1p_pntr++;
        a_2p_reg = *a_2p_pntr++;
        a_3p_reg = *a_3p_pntr++;

        b_p0_reg = B(k, 0);
        b_p1_reg = B(k, 1);
        b_p2_reg = B(k, 2);
        b_p3_reg = B(k, 3);

        c_00_reg += a_0p_reg * b_p0_reg;
        c_10_reg += a_1p_reg * b_p0_reg;
        c_20_reg += a_2p_reg * b_p0_reg;
        c_30_reg += a_3p_reg * b_p0_reg;

        c_01_reg += a_0p_reg * b_p1_reg;
        c_11_reg += a_1p_reg * b_p1_reg;
        c_21_reg += a_2p_reg * b_p1_reg;
        c_31_reg += a_3p_reg * b_p1_reg;

        c_02_reg += a_0p_reg * b_p2_reg;
        c_12_reg += a_1p_reg * b_p2_reg;
        c_22_reg += a_2p_reg * b_p2_reg;
        c_32_reg += a_3p_reg * b_p2_reg;

        c_03_reg += a_0p_reg * b_p3_reg;
        c_13_reg += a_1p_reg * b_p3_reg;
        c_23_reg += a_2p_reg * b_p3_reg;
        c_33_reg += a_3p_reg * b_p3_reg;
    }

    C( 0, 0 ) += c_00_reg;   C( 0, 1 ) += c_01_reg;   C( 0, 2 ) += c_02_reg;   C( 0, 3 ) += c_03_reg;
    C( 1, 0 ) += c_10_reg;   C( 1, 1 ) += c_11_reg;   C( 1, 2 ) += c_12_reg;   C( 1, 3 ) += c_13_reg;
    C( 2, 0 ) += c_20_reg;   C( 2, 1 ) += c_21_reg;   C( 2, 2 ) += c_22_reg;   C( 2, 3 ) += c_23_reg;
    C( 3, 0 ) += c_30_reg;   C( 3, 1 ) += c_31_reg;   C( 3, 2 ) += c_32_reg;   C( 3, 3 ) += c_33_reg;
}
#endif

void dgemm_unroll_ass(unsigned int m, unsigned int n, unsigned int p, 
                      double *a, unsigned int lda,
                      double *b, unsigned int ldb,
                      double *c, unsigned int ldc)
{
    unsigned int i, j;
    
    for(i = 0; i < m; i += M_BLAS_KERNEL_BLOCK_ROWS)
    {
        for(j = 0; j < n; j += M_BLAS_KERNEL_BLOCK_COLS)
        {
            addDot4x4_asm(p, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}
