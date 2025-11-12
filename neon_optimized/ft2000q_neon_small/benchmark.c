/*
 * DGEMM 优化版本性能测试程序
 * 测试25个优化版本在9种矩阵规模下的性能
 * 
 * 测试模式:
 * MODE 0 (默认): 完全对应op-lyb逻辑
 *   - 每次运行重新分配和初始化矩阵（模拟新进程启动）
 *   - 只计时 DGEMM 调用本身
 *   - 简单平均50次，不排除异常值
 *   ⭐ 与op-lyb测试结果可直接对比
 * 
 * MODE 1: 热缓存优化测试模式
 *   - 分配一次，重复使用（热缓存）
 *   - 只计时 DGEMM 调用
 *   - 排除前后10%异常值
 * 
 * MODE 2: 完整流程性能模式
 *   - 每次运行重新分配和初始化
 *   - 计时包括分配、初始化、DGEMM、释放全流程
 *   - 排除前后10%异常值
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 #include <math.h>
 #include "dgemm_opt.h"
 
 // ========== 测试配置 ==========
 #define NUM_RUNS 50           // 每个测试运行次数
 #define OUTLIER_PERCENT 0.1   // 排除的异常值比例（前后各10%，仅MODE 1和2）
 #define VERIFY_CORRECTNESS 0  // 是否验证结果正确性（0=否，1=是）
 #define EPSILON 1e-9          // double精度比较阈值
 
 // ⭐⭐⭐ 测试模式选择 ⭐⭐⭐
 // 0 = op-lyb完全一致模式（每次重新分配，只计时DGEMM，简单平均）✅ 默认
 // 1 = 热缓存优化模式（分配一次，只计时DGEMM，排除异常值）
 // 2 = 完整流程模式（每次重新分配，计时全部，排除异常值）
 #define TEST_MODE 0
 
 // 测试用例结构
 typedef struct {
     const char *name;
     int M, P, N;
 } TestCase;
 
 // 9个测试用例
 static const TestCase test_cases[] = {
     {"Small_PowerOfTwo_Square",      16,  16,  16},
     {"Small_NonPowerOfTwo_Square",   24,  24,  24},
     {"Small_NonSquare",              24,  32,  16},
     {"Medium_NonPowerOfTwo_Square",  96,  96,  96},
     {"Medium_PowerOfTwo_Square",    128, 128, 128},
     {"Medium_NonSquare",            120, 128,  96},
     {"Large_NonPowerOfTwo_Square",  240, 240, 240},
     {"Large_PowerOfTwo_Square",     256, 256, 256},
     {"Large_NonSquare",             256, 240, 248}
 };
 #define NUM_TEST_CASES (sizeof(test_cases) / sizeof(TestCase))
 
 // 优化函数信息
 typedef struct {
     const char *name;
     dgemm_func_ptr func;
 } OptFunc;
 
// 1个优化版本：dgemm_neon_small
static const OptFunc opt_funcs[] = {
    {"dgemm_neon_small",     dgemm_neon_small_wrapper}
};
#define NUM_OPT_FUNCS (sizeof(opt_funcs) / sizeof(OptFunc))
 
 // 获取当前时间（毫秒）
 static double get_time_ms(void) {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
 }
 
 // 比较函数（用于排序）
 static int compare_double(const void *a, const void *b) {
     double diff = *(const double*)a - *(const double*)b;
     if (diff < 0) return -1;
     if (diff > 0) return 1;
     return 0;
 }
 
 // 计算去除异常值后的平均值（用于MODE 1和2）
 static double calculate_trimmed_mean(double *times, int n) {
     qsort(times, n, sizeof(double), compare_double);
     
     int outliers = (int)(n * OUTLIER_PERCENT);
     int start = outliers;
     int end = n - outliers;
     int count = end - start;
     
     if (count <= 0) {
         count = n;
         start = 0;
         end = n;
     }
     
     double sum = 0.0;
     for (int i = start; i < end; i++) {
         sum += times[i];
     }
     
     return sum / count;
 }
 
 // 计算简单平均值（用于MODE 0，对应op-lyb）
 static double calculate_simple_mean(double *times, int n) {
     double sum = 0.0;
     for (int i = 0; i < n; i++) {
         sum += times[i];
     }
     return sum / n;
 }
 
 // 计算标准差
 static double calculate_stddev(double *times, int n, double mean) {
     double sum_sq = 0.0;
     for (int i = 0; i < n; i++) {
         double diff = times[i] - mean;
         sum_sq += diff * diff;
     }
     return sqrt(sum_sq / n);
 }
 
 // 初始化矩阵
 static void init_matrix(int rows, int cols, double *mat) {
     for (int i = 0; i < rows * cols; i++) {
         double val = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
         mat[i] = val;
     }
 }
 
 // 清零矩阵
 static void zero_matrix(int rows, int cols, double *mat) {
     memset(mat, 0, rows * cols * sizeof(double));
 }
 
 #if VERIFY_CORRECTNESS
 // 简单的参考实现（用于验证正确性）
 static void reference_dgemm(int m, int n, int p, 
                            const double *a, int lda,
                            const double *b, int ldb,
                            double *c, int ldc) {
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             double sum = 0.0;
             for (int k = 0; k < p; k++) {
                 sum += a[i * lda + k] * b[k * ldb + j];
             }
             c[i * ldc + j] = sum;
         }
     }
 }
 
 // 验证两个矩阵是否相等
 static int verify_matrix(int rows, int cols, const double *mat1, const double *mat2) {
     for (int i = 0; i < rows * cols; i++) {
         double diff = fabs(mat1[i] - mat2[i]);
         double rel_error = diff / (fabs(mat1[i]) + 1e-15);
         if (diff > EPSILON && rel_error > EPSILON) {
             return 0;
         }
     }
     return 1;
 }
 #endif
 
 // ========== MODE 0: op-lyb完全一致模式 ==========
 // 每次重新分配和初始化，只计时DGEMM，简单平均
 #if TEST_MODE == 0
 static int run_single_test(const TestCase *tc, const OptFunc *opt, double *avg_time, double *stddev) {
     int M = tc->M;
     int P = tc->P;
     int N = tc->N;
     unsigned int lda = P;
     unsigned int ldb = N;
     unsigned int ldc = N;
     
     double *times = (double*)malloc(NUM_RUNS * sizeof(double));
     if (!times) {
         fprintf(stderr, "错误: 无法分配计时数组\n");
         return -1;
     }
     
     // 模拟op-lyb的test.sh循环：每次启动新进程
     for (int run = 0; run < NUM_RUNS; run++) {
         // === 不计时：分配内存（模拟进程启动） ===
         double *A = (double*)malloc(M * P * sizeof(double));
         double *B = (double*)malloc(P * N * sizeof(double));
         double *C = (double*)malloc(M * N * sizeof(double));
         
         if (!A || !B || !C) {
             fprintf(stderr, "错误: 内存分配失败 (run %d)\n", run);
             free(A);
             free(B);
             free(C);
             free(times);
             return -1;
         }
         
         // === 不计时：初始化矩阵 ===
         init_matrix(M, P, A);
         init_matrix(P, N, B);
         zero_matrix(M, N, C);
         
         // === ⭐ 只计时这部分：DGEMM调用 ⭐ ===
         double start = get_time_ms();
         opt->func(M, N, P, A, lda, B, ldb, C, ldc);
         double end = get_time_ms();
         
         times[run] = end - start;
         
         // === 不计时：释放内存（模拟进程退出） ===
         free(A);
         free(B);
         free(C);
     }
     
     // === 简单平均，不排除异常值（对应op-lyb的test.sh逻辑） ===
     *avg_time = calculate_simple_mean(times, NUM_RUNS);
     *stddev = calculate_stddev(times, NUM_RUNS, *avg_time);
     
     free(times);
     return 0;
 }
 
 // ========== MODE 1: 热缓存优化测试模式 ==========
 // 分配一次，只计时DGEMM，排除异常值
 #elif TEST_MODE == 1
 static int run_single_test(const TestCase *tc, const OptFunc *opt, double *avg_time, double *stddev) {
     int M = tc->M;
     int P = tc->P;
     int N = tc->N;
     unsigned int lda = P;
     unsigned int ldb = N;
     unsigned int ldc = N;
     
     double *times = (double*)malloc(NUM_RUNS * sizeof(double));
     if (!times) {
         fprintf(stderr, "错误: 无法分配计时数组\n");
         return -1;
     }
     
     // 分配矩阵（只一次）
     double *A = (double*)malloc(M * P * sizeof(double));
     double *B = (double*)malloc(P * N * sizeof(double));
     double *C = (double*)malloc(M * N * sizeof(double));
     
     if (!A || !B || !C) {
         fprintf(stderr, "错误: 内存分配失败\n");
         free(A);
         free(B);
         free(C);
         free(times);
         return -1;
     }
     
     // 初始化矩阵（只一次）
     init_matrix(M, P, A);
     init_matrix(P, N, B);
     
 #if VERIFY_CORRECTNESS
     double *C_ref = (double*)malloc(M * N * sizeof(double));
     if (C_ref) {
         zero_matrix(M, N, C_ref);
         reference_dgemm(M, N, P, A, lda, B, ldb, C_ref, ldc);
     }
 #endif
     
     // 运行多次测试（只测DGEMM，热缓存）
     for (int run = 0; run < NUM_RUNS; run++) {
         zero_matrix(M, N, C);
         
         double start = get_time_ms();
         opt->func(M, N, P, A, lda, B, ldb, C, ldc);
         double end = get_time_ms();
         
         times[run] = end - start;
         
 #if VERIFY_CORRECTNESS
         if (run == 0 && C_ref) {
             if (!verify_matrix(M, N, C, C_ref)) {
                 fprintf(stderr, "\n警告: 结果不正确！\n");
                 free(C_ref);
                 free(times);
                 free(A);
                 free(B);
                 free(C);
                 return -1;
             }
         }
 #endif
     }
     
 #if VERIFY_CORRECTNESS
     if (C_ref) free(C_ref);
 #endif
     
     free(A);
     free(B);
     free(C);
     
     // 计算统计数据（排除异常值）
     *avg_time = calculate_trimmed_mean(times, NUM_RUNS);
     *stddev = calculate_stddev(times, NUM_RUNS, *avg_time);
     
     free(times);
     return 0;
 }
 
 // ========== MODE 2: 完整流程性能模式 ==========
 // 每次重新分配，计时全部，排除异常值
 #elif TEST_MODE == 2
 static int run_single_test(const TestCase *tc, const OptFunc *opt, double *avg_time, double *stddev) {
     int M = tc->M;
     int P = tc->P;
     int N = tc->N;
     unsigned int lda = P;
     unsigned int ldb = N;
     unsigned int ldc = N;
     
     double *times = (double*)malloc(NUM_RUNS * sizeof(double));
     if (!times) {
         fprintf(stderr, "错误: 无法分配计时数组\n");
         return -1;
     }
     
     // 每次运行都重新分配和初始化，测量完整流程
     for (int run = 0; run < NUM_RUNS; run++) {
         double start = get_time_ms();
         
         // 分配矩阵
         double *A = (double*)malloc(M * P * sizeof(double));
         double *B = (double*)malloc(P * N * sizeof(double));
         double *C = (double*)malloc(M * N * sizeof(double));
         
         if (!A || !B || !C) {
             fprintf(stderr, "错误: 内存分配失败 (run %d)\n", run);
             free(A);
             free(B);
             free(C);
             free(times);
             return -1;
         }
         
         // 初始化矩阵
         init_matrix(M, P, A);
         init_matrix(P, N, B);
         zero_matrix(M, N, C);
         
         // DGEMM计算
         opt->func(M, N, P, A, lda, B, ldb, C, ldc);
         
         double end = get_time_ms();
         times[run] = end - start;
         
         // 释放内存
         free(A);
         free(B);
         free(C);
     }
     
     // 计算统计数据（排除异常值）
     *avg_time = calculate_trimmed_mean(times, NUM_RUNS);
     *stddev = calculate_stddev(times, NUM_RUNS, *avg_time);
     
     free(times);
     return 0;
 }
 
 #else
 #error "Invalid TEST_MODE. Must be 0, 1, or 2."
 #endif
 
 // 打印表头
 static void print_header(void) {
     printf("========================================================================================================\n");
     printf("DGEMM 优化版本性能测试报告 (double精度浮点运算)\n");
     printf("========================================================================================================\n");
     printf("测试配置:\n");
     printf("  - 数据类型: double (64位浮点)\n");
     printf("  - 运行次数: %d次\n", NUM_RUNS);
     printf("  - 编译优化: -O0\n");
     printf("  - 平台: FT2000Q (ARMv8)\n");
     
 #if TEST_MODE == 0
     printf("  - 测试模式: MODE 0 - op-lyb完全一致模式 ⭐⭐⭐\n");
     printf("              每次运行重新分配和初始化矩阵（模拟新进程）\n");
     printf("              只计时DGEMM调用本身\n");
     printf("              简单平均%d次，不排除异常值\n", NUM_RUNS);
     printf("              ✅ 结果可与op-lyb直接对比\n");
 #elif TEST_MODE == 1
     printf("  - 测试模式: MODE 1 - 热缓存优化测试模式\n");
     printf("              分配一次，重复使用（热缓存）\n");
     printf("              只计时DGEMM调用\n");
     printf("              排除前后各%.0f%%异常值\n", OUTLIER_PERCENT * 100);
 #elif TEST_MODE == 2
     printf("  - 测试模式: MODE 2 - 完整流程性能模式\n");
     printf("              每次运行重新分配和初始化\n");
     printf("              计时包括分配、初始化、DGEMM、释放\n");
     printf("              排除前后各%.0f%%异常值\n", OUTLIER_PERCENT * 100);
 #endif
     
 #if VERIFY_CORRECTNESS
     printf("  - 正确性验证: 已启用 (epsilon=%.0e)\n", EPSILON);
 #else
     printf("  - 正确性验证: 已禁用\n");
 #endif
     printf("========================================================================================================\n\n");
 }
 
 // 打印测试结果表格
 static void print_results_table(double results[NUM_OPT_FUNCS][NUM_TEST_CASES]) {
     // 打印列标题
     printf("%-25s", "优化版本");
     for (int i = 0; i < NUM_TEST_CASES; i++) {
         printf(" | %6s", test_cases[i].name + strlen(test_cases[i].name) - 6);
     }
     printf("\n");
     
     // 打印分隔线
     printf("%-25s", "-------------------------");
     for (int i = 0; i < NUM_TEST_CASES; i++) {
         printf("-+---------");
     }
     printf("\n");
     
     // 打印每个优化版本的结果
     for (int opt = 0; opt < NUM_OPT_FUNCS; opt++) {
         printf("%-25s", opt_funcs[opt].name);
         for (int tc = 0; tc < NUM_TEST_CASES; tc++) {
             if (results[opt][tc] >= 0) {
                 printf(" | %7.2f", results[opt][tc]);
             } else {
                 printf(" | %7s", "FAIL");
             }
         }
         printf("\n");
     }
     printf("\n");
 }
 
 // 保存CSV格式结果
 static void save_csv_results(const char *filename, double results[NUM_OPT_FUNCS][NUM_TEST_CASES]) {
     FILE *fp = fopen(filename, "w");
     if (!fp) {
         fprintf(stderr, "警告: 无法创建CSV文件 %s\n", filename);
         return;
     }
     
     // CSV表头
     fprintf(fp, "优化版本");
     for (int i = 0; i < NUM_TEST_CASES; i++) {
         fprintf(fp, ",%s(%dx%dx%d)", 
                 test_cases[i].name, 
                 test_cases[i].M, 
                 test_cases[i].P, 
                 test_cases[i].N);
     }
     fprintf(fp, "\n");
     
     // 数据行
     for (int opt = 0; opt < NUM_OPT_FUNCS; opt++) {
         fprintf(fp, "%s", opt_funcs[opt].name);
         for (int tc = 0; tc < NUM_TEST_CASES; tc++) {
             if (results[opt][tc] >= 0) {
                 fprintf(fp, ",%.3f", results[opt][tc]);
             } else {
                 fprintf(fp, ",FAIL");
             }
         }
         fprintf(fp, "\n");
     }
     
     fclose(fp);
     printf("结果已保存到: %s\n", filename);
 }
 
 int main(void) {
     // 初始化随机数生成器
     srand((unsigned int)time(NULL));
     
     // 打印表头
     print_header();
     
     // 结果存储数组
     double results[NUM_OPT_FUNCS][NUM_TEST_CASES];
     
     // 总测试数
     int total_tests = NUM_OPT_FUNCS * NUM_TEST_CASES;
     int current_test = 0;
     
     // 遍历所有优化版本
     for (int opt = 0; opt < NUM_OPT_FUNCS; opt++) {
         printf("\n[%2d/%2d] 测试: %s\n", 
                opt + 1, NUM_OPT_FUNCS, opt_funcs[opt].name);
         printf("-----------------------------------------------------------\n");
         
         // 遍历所有测试用例
         for (int tc = 0; tc < NUM_TEST_CASES; tc++) {
             current_test++;
             printf("  [%3d/%3d] %s (%dx%dx%d) ... ", 
                    current_test, total_tests,
                    test_cases[tc].name,
                    test_cases[tc].M,
                    test_cases[tc].P,
                    test_cases[tc].N);
             fflush(stdout);
             
             double avg_time, stddev;
             int ret = run_single_test(&test_cases[tc], &opt_funcs[opt], 
                                      &avg_time, &stddev);
             
             if (ret == 0) {
                 results[opt][tc] = avg_time;
                 printf("平均: %7.2f ms, 标准差: %6.2f ms\n", avg_time, stddev);
             } else {
                 results[opt][tc] = -1.0;
                 printf("失败\n");
             }
         }
     }
     
     // 打印汇总表格
     printf("\n\n");
     printf("========================================================================================================\n");
     printf("性能测试结果汇总（平均时间，单位：毫秒）\n");
     printf("========================================================================================================\n");
     print_results_table(results);
     
     // 保存CSV结果
     save_csv_results("benchmark_results.csv", results);
     
     printf("\n所有测试完成！\n");
     
     return 0;
 }
 