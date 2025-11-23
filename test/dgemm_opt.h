#ifndef DGEMM_OPT_H
#define DGEMM_OPT_H

// 统一的DGEMM函数指针类型（标准接口）
typedef void (*dgemm_func_ptr)(
    int m, int n, int p,
    const double *a, unsigned int lda,
    const double *b, unsigned int ldb,
    double *c, unsigned int ldc
);

// ========== DGEMM 实现函数声明 ==========
// 注意：所有实现统一使用 -O2 编译选项

// 原始循环展开实现的包装 (src/dgemm_unroll.c)
void dgemm_unroll_int(int m, int n, int p,
                      const double *a, unsigned int lda,
                      const double *b, unsigned int ldb,
                      double *c, unsigned int ldc);

// 内联汇编FMA优化实现的包装 (opt/dgemm_unroll_ass.c)
void dgemm_unroll_ass_int(int m, int n, int p,
                          const double *a, unsigned int lda,
                          const double *b, unsigned int ldb,
                          double *c, unsigned int ldc);

#endif // DGEMM_OPT_H
