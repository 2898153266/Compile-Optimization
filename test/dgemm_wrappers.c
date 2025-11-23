/*
 * DGEMM 包装函数
 * 
 * 将 blas_dgemm.h 中的 unsigned int 接口包装为 int 接口
 * 用于 test_interface.h 中的统一接口
 */

#include <stdlib.h>
#include "blas_dgemm.h"

// ========== 包装函数实现 ==========

// 原始循环展开实现的包装
void dgemm_unroll_int(int m, int n, int p,
                      const double *a, unsigned int lda,
                      const double *b, unsigned int ldb,
                      double *c, unsigned int ldc) {
    dgemm_unroll((unsigned int)m, (unsigned int)n, (unsigned int)p,
                 (double*)a, lda, (double*)b, ldb, c, ldc);
}

// 内联汇编FMA优化实现的包装
void dgemm_unroll_ass_int(int m, int n, int p,
                          const double *a, unsigned int lda,
                          const double *b, unsigned int ldb,
                          double *c, unsigned int ldc) {
    dgemm_unroll_ass((unsigned int)m, (unsigned int)n, (unsigned int)p,
                     (double*)a, lda, (double*)b, ldb, c, ldc);
}
