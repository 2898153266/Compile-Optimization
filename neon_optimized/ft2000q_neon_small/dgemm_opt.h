/*
 * DGEMM 优化版本函数声明
 * 独立头文件，无需其他依赖
 */

#ifndef DGEMM_OPT_H
#define DGEMM_OPT_H

#include <stdlib.h>

// DGEMM函数指针类型定义
typedef void (*dgemm_func_ptr)(unsigned int m, unsigned int n, unsigned int p, 
                               double *a, unsigned int lda,
                               double *b, unsigned int ldb,
                               double *c, unsigned int ldc);

// dgemm_neon_small 原始函数（需要额外缓冲区）
void dgemm_neon_small(unsigned int m, unsigned int n, unsigned int p, 
                      double *a, unsigned int lda,
                      double *b, unsigned int ldb,
                      double *c, unsigned int ldc,
                      double *sa, double *sb);

// dgemm_neon_small 的包装函数（符合标准接口）
static inline void dgemm_neon_small_wrapper(unsigned int m, unsigned int n, unsigned int p, 
                                            double *a, unsigned int lda,
                                            double *b, unsigned int ldb,
                                            double *c, unsigned int ldc) {
    // 为 NEON 版本分配必要的打包缓冲区
    #define GEMM_M_WRAPPER (2048)
    #define GEMM_P_WRAPPER (128)
    
    size_t sa_size = GEMM_M_WRAPPER * GEMM_P_WRAPPER * sizeof(double);
    size_t sb_size = GEMM_P_WRAPPER * n * sizeof(double);
    
    double *sa = (double*)malloc(sa_size);
    double *sb = (double*)malloc(sb_size);
    
    if (sa && sb) {
        dgemm_neon_small(m, n, p, a, lda, b, ldb, c, ldc, sa, sb);
    }
    
    if (sa) free(sa);
    if (sb) free(sb);
}

#endif // DGEMM_OPT_H

