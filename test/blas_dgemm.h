#ifndef BLAS_DGEMM_H
#define BLAS_DGEMM_H

// 宏定义
#define M_BLAS_KERNEL_BLOCK_ROWS 4
#define M_BLAS_KERNEL_BLOCK_COLS 4

// GEMM 块大小配置
#define GEMM_N (256)
#define GEMM_M (2048)
#define GEMM_P (128)
#define GEMM_UNROLL (4)

// ========== 原始实现 (src/) ==========

// dgemm_unroll - 原始循环展开实现
void dgemm_unroll(unsigned int m, unsigned int n, unsigned int p,
                  double *a, unsigned int lda,
                  double *b, unsigned int ldb,
                  double *c, unsigned int ldc);

// ========== 优化实现 (opt/) ==========

// dgemm_unroll_ass - 优化的内联汇编实现
void dgemm_unroll_ass(unsigned int m, unsigned int n, unsigned int p,
                      double *a, unsigned int lda,
                      double *b, unsigned int ldb,
                      double *c, unsigned int ldc);

#endif // BLAS_DGEMM_H
