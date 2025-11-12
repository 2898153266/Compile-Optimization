#ifdef __ARM_NEON
#include <arm_neon.h>

#include <stdlib.h>
#include <string.h>

/* 矩阵按行优先顺序存储的宏定义 */
#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

/**
 * ============================================================================
 * 针对小矩阵（24×24×24 及以下）的高度优化 DGEMM
 * ============================================================================
 * 
 * 设计原则（针对 O0 编译级别）：
 * 
 * 1. **极简化** - 去掉所有不必要的逻辑
 * 2. **无打包或简化打包** - 小矩阵打包开销大于收益
 * 3. **小计算块** - 2×2 或 2×4 足够
 * 4. **无预取** - 小数据量预取反而浪费指令
 * 5. **内联汇编最小化** - 减少寄存器操作
 * 6. **直接计算** - 能不分块就不分块
 * 
 * 目标：在 24×24×24 及以下规模超越原版
 * ============================================================================
 */

/**
 * ============================================================================
 * 极简 2×2 微内核 - 专为小矩阵设计
 * ============================================================================
 * 
 * 特点：
 * - 最小的寄存器占用（只用4个向量寄存器存C，4个存A/B）
 * - 无预取
 * - 最简单的指令序列
 * - 在 O0 下也能保持高效
 * ============================================================================
 */
static inline void kernel_2x2_tiny(unsigned int p, 
                                   const double *a, const double *b, 
                                   double *c, unsigned int ldc) {
    // 加载 C (2×2)
    float64x2_t c00 = vld1q_f64(&c[0]);        // C[0][0:1]
    float64x2_t c10 = vld1q_f64(&c[ldc]);      // C[1][0:1]
    
    // 主循环：遍历 K 维度
    for (unsigned int k = 0; k < p; k++) {
        // 加载 A (2×1)
        float64x2_t a_vec = vld1q_f64(&a[k * 2]);  // A[0:1][k]
        
        // 加载 B (1×2)
        float64x2_t b_vec = vld1q_f64(&b[k * 2]);  // B[k][0:1]
        
        // 计算：C += A * B
        c00 = vfmaq_laneq_f64(c00, b_vec, a_vec, 0);  // C[0] += B * A[0]
        c10 = vfmaq_laneq_f64(c10, b_vec, a_vec, 1);  // C[1] += B * A[1]
    }
    
    // 存储结果
    vst1q_f64(&c[0], c00);
    vst1q_f64(&c[ldc], c10);
}

/**
 * ============================================================================
 * 稍大一点的 2×4 微内核
 * ============================================================================
 * 
 * 用于处理 n 维度较大的情况（如 16, 20, 24）
 * 在小矩阵中 2×4 比 4×4 更高效（O0级别下）
 * ============================================================================
 */
static inline void kernel_2x4_tiny(unsigned int p,
                                   const double *a, const double *b,
                                   double *c, unsigned int ldc) {
    // 加载 C (2×4)
    float64x2_t c00 = vld1q_f64(&c[0]);
    float64x2_t c01 = vld1q_f64(&c[2]);
    float64x2_t c10 = vld1q_f64(&c[ldc]);
    float64x2_t c11 = vld1q_f64(&c[ldc + 2]);
    
    for (unsigned int k = 0; k < p; k++) {
        // 加载 A (2×1)
        float64x2_t a_vec = vld1q_f64(&a[k * 2]);
        
        // 加载 B (1×4)
        float64x2_t b0 = vld1q_f64(&b[k * 4]);
        float64x2_t b1 = vld1q_f64(&b[k * 4 + 2]);
        
        // 计算
        c00 = vfmaq_laneq_f64(c00, b0, a_vec, 0);
        c01 = vfmaq_laneq_f64(c01, b1, a_vec, 0);
        c10 = vfmaq_laneq_f64(c10, b0, a_vec, 1);
        c11 = vfmaq_laneq_f64(c11, b1, a_vec, 1);
    }
    
    // 存储
    vst1q_f64(&c[0], c00);
    vst1q_f64(&c[2], c01);
    vst1q_f64(&c[ldc], c10);
    vst1q_f64(&c[ldc + 2], c11);
}

/**
 * ============================================================================
 * 简化的转置打包 - 仅用于提高数据局部性
 * ============================================================================
 * 
 * 只在必要时使用（矩阵足够大时才值得）
 * 使用 intrinsics 而不是汇编，在 O0 下更可靠
 * ============================================================================
 */
static inline void pack_a_2x2(unsigned int m, unsigned int p,
                              const double *from, unsigned int lda,
                              double *to) {
    for (unsigned int i = 0; i < m; i += 2) {
        for (unsigned int j = 0; j < p; j++) {
            to[j * 2 + 0] = from[i * lda + j];
            to[j * 2 + 1] = from[(i + 1) * lda + j];
        }
        to += p * 2;
    }
}

static inline void pack_b_2x2(unsigned int p, unsigned int n,
                              const double *from, unsigned int ldb,
                              double *to) {
    for (unsigned int i = 0; i < p; i++) {
        for (unsigned int j = 0; j < n; j += 2) {
            to[i * 2 + 0] = from[i * ldb + j];
            to[i * 2 + 1] = from[i * ldb + j + 1];
        }
        to += 2;
    }
}

/**
 * ============================================================================
 * 针对小矩阵的主函数 - 极简版本
 * ============================================================================
 * 
 * 策略：
 * 1. 对于很小的矩阵（< 16），直接计算，不打包
 * 2. 对于中等小矩阵（16-32），简单打包 + 小内核
 * 3. 使用 2×2 或 2×4 微内核，避免寄存器压力
 * 4. 无复杂的分块逻辑
 * 
 * 针对 O0 编译优化：
 * - 避免复杂分支
 * - 使用简单的循环
 * - 尽量用 inline 函数而不是宏
 * ============================================================================
 */
void dgemm_neon_small(unsigned int m, unsigned int n, unsigned int p,
                      double *a, unsigned int lda,
                      double *b, unsigned int ldb,
                      double *c, unsigned int ldc,
                      double *sa, double *sb) {
    
    // 对于极小矩阵（≤ 16），直接计算不打包
    if (m <= 16 && n <= 16 && p <= 16) {
        // 直接在原始数据上计算
        unsigned int i, j;
        
        // 按 2×4 块处理
        for (i = 0; i + 1 < m; i += 2) {
            for (j = 0; j + 3 < n; j += 4) {
                kernel_2x4_tiny(p, a + i * lda, b + j, c + i * ldc + j, ldc);
            }
            
            // 处理 j 维度的余数（2×2 块）
            for (; j + 1 < n; j += 2) {
                kernel_2x2_tiny(p, a + i * lda, b + j, c + i * ldc + j, ldc);
            }
            
            // 处理单列余数
            if (j < n) {
                for (unsigned int k = 0; k < p; k++) {
                    c[i * ldc + j] += a[i * lda + k] * b[k * ldb + j];
                    c[(i + 1) * ldc + j] += a[(i + 1) * lda + k] * b[k * ldb + j];
                }
            }
        }
        
        // 处理 i 维度的余数（单行）
        if (i < m) {
            for (j = 0; j < n; j++) {
                for (unsigned int k = 0; k < p; k++) {
                    c[i * ldc + j] += a[i * lda + k] * b[k * ldb + j];
                }
            }
        }
        
        return;
    }
    
    // 对于稍大的小矩阵（16 < size ≤ 32），使用简单打包
    if (m <= 32 && n <= 32 && p <= 32) {
        // 打包 A：按行打包成 2 行一组
        unsigned int packed_m = (m + 1) & ~1;  // 向上取偶数
        pack_a_2x2(m, p, a, lda, sa);
        
        // 打包 B：按列打包成 2 列一组  
        unsigned int packed_n = (n + 1) & ~1;
        pack_b_2x2(p, n, b, ldb, sb);
        
        // 使用打包后的数据计算
        unsigned int i, j;
        for (i = 0; i + 1 < m; i += 2) {
            for (j = 0; j + 3 < n; j += 4) {
                kernel_2x4_tiny(p, sa + i * p, sb + j * p, c + i * ldc + j, ldc);
            }
            
            for (; j + 1 < n; j += 2) {
                kernel_2x2_tiny(p, sa + i * p, sb + j * p, c + i * ldc + j, ldc);
            }
            
            if (j < n) {
                for (unsigned int k = 0; k < p; k++) {
                    c[i * ldc + j] += a[i * lda + k] * b[k * ldb + j];
                    c[(i + 1) * ldc + j] += a[(i + 1) * lda + k] * b[k * ldb + j];
                }
            }
        }
        
        if (i < m) {
            for (j = 0; j < n; j++) {
                for (unsigned int k = 0; k < p; k++) {
                    c[i * ldc + j] += a[i * lda + k] * b[k * ldb + j];
                }
            }
        }
        
        return;
    }
    
    // 对于更大的矩阵（但仍然算"小"，如 64×64），使用改进的分块
    // 但保持分块逻辑简单
    const unsigned int BLOCK_M = 16;
    const unsigned int BLOCK_N = 16;
    const unsigned int BLOCK_K = 16;
    
    for (unsigned int ii = 0; ii < m; ii += BLOCK_M) {
        unsigned int im = min(BLOCK_M, m - ii);
        
        for (unsigned int jj = 0; jj < n; jj += BLOCK_N) {
            unsigned int jn = min(BLOCK_N, n - jj);
            
            for (unsigned int kk = 0; kk < p; kk += BLOCK_K) {
                unsigned int kp = min(BLOCK_K, p - kk);
                
                // 在小块上使用 2×2 或 2×4 内核
                unsigned int i, j;
                for (i = 0; i + 1 < im; i += 2) {
                    for (j = 0; j + 3 < jn; j += 4) {
                        unsigned int abs_i = ii + i;
                        unsigned int abs_j = jj + j;
                        kernel_2x4_tiny(kp,
                                       a + abs_i * lda + kk,
                                       b + kk * ldb + abs_j,
                                       c + abs_i * ldc + abs_j,
                                       ldc);
                    }
                    
                    for (; j + 1 < jn; j += 2) {
                        unsigned int abs_i = ii + i;
                        unsigned int abs_j = jj + j;
                        kernel_2x2_tiny(kp,
                                       a + abs_i * lda + kk,
                                       b + kk * ldb + abs_j,
                                       c + abs_i * ldc + abs_j,
                                       ldc);
                    }
                    
                    // 余数
                    for (; j < jn; j++) {
                        unsigned int abs_i = ii + i;
                        unsigned int abs_j = jj + j;
                        for (unsigned int k = 0; k < kp; k++) {
                            c[abs_i * ldc + abs_j] += 
                                a[abs_i * lda + kk + k] * b[(kk + k) * ldb + abs_j];
                            c[(abs_i + 1) * ldc + abs_j] += 
                                a[(abs_i + 1) * lda + kk + k] * b[(kk + k) * ldb + abs_j];
                        }
                    }
                }
                
                // i 维度余数
                for (; i < im; i++) {
                    unsigned int abs_i = ii + i;
                    for (j = 0; j < jn; j++) {
                        unsigned int abs_j = jj + j;
                        for (unsigned int k = 0; k < kp; k++) {
                            c[abs_i * ldc + abs_j] += 
                                a[abs_i * lda + kk + k] * b[(kk + k) * ldb + abs_j];
                        }
                    }
                }
            }
        }
    }
}

#endif

