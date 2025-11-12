#ifndef M_DGEMM_BLAS_H
#define M_DGEMM_BLAS_H



#define M_BLAS_KERNEL_BLOCK_ROWS 4
#define M_BLAS_KERNEL_BLOCK_COLS 4



/******************************************* naive *******************************************/
//C(mxn) = A(mxp)*B(pxn)
void dgemm_naive(unsigned int m, unsigned int n, unsigned int p, double *a, unsigned int lda,
                                                                 double *b, unsigned int ldb,
                                                                 double *c, unsigned int ldc);


//C(mxn) = A(mxp)*BT(pxn)
void dgemm_naive_abt(unsigned int m, unsigned int n, unsigned int p, double *a, unsigned int lda,
                                                                     double *b, unsigned int ldb,
                                                                     double *c, unsigned int ldc);

//C(mxn) = A(mxm)*B(mxm)*AT(mxm)
void dgemm_naive_abat(unsigned int m, unsigned int p, double *a, unsigned int lda,
                                                      double *b, unsigned int ldb,
                                                      double *c, unsigned int ldc);


//C_block(mxn) = A(mxp)*B(pxn)
void dgemm_naive_block(unsigned int beg_row, unsigned int beg_col,
                       unsigned int m, unsigned int n, unsigned int p, double *a, unsigned int lda,
                                                                       double *b, unsigned int ldb,
                                                                       double *c, unsigned int ldc);


/******************************************* unroll *******************************************/
//C(mxn) = A(mxp)*B(pxn)
void dgemm_unroll(unsigned int m, unsigned int n, unsigned int p, double *a, unsigned int lda,
                                                                  double *b, unsigned int ldb,
                                                                  double *c, unsigned int ldc);

//C(mxn) = A(mxp)*BT(pxn)
void dgemm_unroll_abt(unsigned int m, unsigned int n, unsigned int p, double *a, unsigned int lda,
                                                                      double *b, unsigned int ldb,
                                                                      double *c, unsigned int ldc);

//C(mxm) = A(mxm)*B(mxm)*AT(mxm)
void dgemm_unroll_abat(unsigned int m, unsigned int p, double *a, unsigned int lda,
                                                       double *b, unsigned int ldb,
                                                       double *c, unsigned int ldc);

/******************************************* neon *******************************************/
#ifdef __ARM_NEON
//C(mxn) = A(mxp)*B(pxn)
void dgemm_neon(unsigned int m, unsigned int n, unsigned int p, double *a, unsigned int lda, 
                                                                double *b, unsigned int ldb,
                                                                double *c, unsigned int ldc, 
                                                                double *sa, double *sb);
#endif

#endif // M_DGEMM_BLAS_H
