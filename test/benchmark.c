/*
 * DGEMM 优化版本性能测试程序
 * 测试 unroll 版本在9种矩阵规模下的性能
 * 
 * 移植说明：
 * 1. 如果目标平台不支持clock_gettime，需要替换get_time_ms()函数
 * 2. 只依赖标准C库：stdio.h, stdlib.h, string.h
 * 3. 数学函数使用自实现版本，不依赖math.h
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dgemm_opt.h"

/* ========== 平台相关：时间获取函数 ========== */
/* 如果目标平台不支持以下头文件和函数，请替换整个时间获取部分 */

#ifdef _WIN32
    /* Windows平台 */
    #include <windows.h>
    static double get_time_ms(void) {
        LARGE_INTEGER frequency, counter;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&counter);
        return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
    }
#elif defined(__linux__) || defined(__unix__)
    /* Linux/Unix平台 - 使用clock_gettime */
    #define _POSIX_C_SOURCE 199309L
    #include <time.h>
    static double get_time_ms(void) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
    }
#else
    /* 其他平台 - 使用标准C的clock()函数（精度较低） */
    #include <time.h>
    static double get_time_ms(void) {
        return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    }
#endif

/* ========== 数学函数：平方根（用于标准差计算）========== */
/* 使用牛顿迭代法实现平方根，不依赖math.h */

/* 简单的绝对值函数 */
static double my_fabs(double x) {
    return (x < 0.0) ? -x : x;
}

/* 平方根实现（牛顿迭代法） */
static double my_sqrt(double x) {
    if (x <= 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    
    double guess = x / 2.0;
    double epsilon = 1e-10;
    int max_iter = 50;
    
    for (int i = 0; i < max_iter; i++) {
        double new_guess = (guess + x / guess) / 2.0;
        if (my_fabs(new_guess - guess) < epsilon)
            break;
        guess = new_guess;
    }
    return guess;
}

/* ========== 固定测试数据 ========== */
/* 预定义256×256矩阵，用于所有测试 */

#define MAX_DIM 256
static double fixed_matrix_A[MAX_DIM * MAX_DIM];
static double fixed_matrix_B[MAX_DIM * MAX_DIM];

/* 初始化固定矩阵（使用确定性公式生成，支持不同数值范围） */
static void init_fixed_matrices_range(double min_val, double max_val) {
    /* 使用简单的数学公式生成固定值，避免随机数 */
    for (int i = 0; i < MAX_DIM; i++) {
        for (int j = 0; j < MAX_DIM; j++) {
            int idx = i * MAX_DIM + j;
            /* 生成[0,1)范围的确定性值 */
            double val_a = (double)((i * 17 + j * 13) % 10000) / 10000.0;
            double val_b = (double)((i * 23 + j * 19) % 10000) / 10000.0;
            
            /* 映射到目标范围 */
            fixed_matrix_A[idx] = min_val + val_a * (max_val - min_val);
            fixed_matrix_B[idx] = min_val + val_b * (max_val - min_val);
        }
    }
}

/* ========== 测试配置 ========== */
#include "dgemm_opt.h"
 
 // ========== 测试配置 ==========
 #define NUM_RUNS 500           // 每个测试运行次数
 #define OUTLIER_PERCENT 0.1   // 排除的异常值比例（前后各10%，仅MODE 1和2）
 #define VERIFY_CORRECTNESS 0  // 是否验证结果正确性（0=否，1=是）
 #define EPSILON 1e-9          // double精度比较阈值
 
 //  测试模式选择 
 // 0 = op-lyb完全一致模式（每次重新分配，只计时DGEMM，简单平均）
 // 1 = 热缓存优化模式（分配一次，只计时DGEMM，排除异常值）
 // 2 = 完整流程模式（每次重新分配，计时全部，排除异常值）
 #define TEST_MODE 0
 
 // 数值范围配置
 typedef struct {
     const char *name;
     double min_val;
     double max_val;
 } ValueRange;
 
 // 4组数值范围
 static const ValueRange value_ranges[] = {
     {"Range_0_1",      0.0,    1.0},
     {"Range_1_1e3",    1.0,    1e3},
     {"Range_1e3_1e5",  1e3,    1e5},
     {"Range_1e5_1e7",  1e5,    1e7}
 };
 #define NUM_VALUE_RANGES (sizeof(value_ranges) / sizeof(ValueRange))
 
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
     {"Small_48x48",                  48,  48,  48},
     {"Medium_NonPowerOfTwo_Square",  96,  96,  96},
     {"Medium_PowerOfTwo_Square",    128, 128, 128},
     {"Medium_NonSquare",            120, 128,  96},
     {"Large_PowerOfTwo_Square",     256, 256, 256},
     {"Large_NonSquare",             256, 240, 248}
 };
 #define NUM_TEST_CASES (sizeof(test_cases) / sizeof(TestCase))
 
 // 优化函数信息
 typedef struct {
     const char *name;
     dgemm_func_ptr func;
 } OptFunc;
 
// 2个DGEMM实现（统一使用 -O2 编译）
static const OptFunc opt_funcs[] = {
    {"dgemm_unroll",      dgemm_unroll_int},      // 原始循环展开实现 (src/)
    {"dgemm_unroll_ass",  dgemm_unroll_ass_int}   // 内联汇编FMA优化 (opt/)
};
#define NUM_OPT_FUNCS (sizeof(opt_funcs) / sizeof(OptFunc))
 
 // 比较函数（用于排序）赖qsort）
 static void sort_double_array(double *arr, int n) {
     for (int i = 0; i < n - 1; i++) {
         for (int j = 0; j < n - i - 1; j++) {
             if (arr[j] > arr[j + 1]) {
                 double temp = arr[j];
                 arr[j] = arr[j + 1];
                 arr[j + 1] = temp;
             }
         }
     }
 }
 
 // 计算去除异常值后的平均值（用于MODE 1和2）
 static double calculate_trimmed_mean(double *times, int n) {
     sort_double_array(times, n);
     
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
    return my_sqrt(sum_sq / n);
} // 从固定数组初始化矩阵（复制子矩阵）
 static void init_matrix(int rows, int cols, double *mat, const double *fixed_src) {
     /* 从256×256固定数组中复制rows×cols的左上角子矩阵 */
     for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
             mat[i * cols + j] = fixed_src[i * MAX_DIM + j];
         }
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
         
         // === 不计时：初始化矩阵（从固定数组复制） ===
         init_matrix(M, P, A, fixed_matrix_A);
         init_matrix(P, N, B, fixed_matrix_B);
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
     
     // 初始化矩阵（只一次，从固定数组复制）
     init_matrix(M, P, A, fixed_matrix_A);
     init_matrix(P, N, B, fixed_matrix_B);
     
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
         
         // 初始化矩阵（从固定数组复制）
         init_matrix(M, P, A, fixed_matrix_A);
         init_matrix(P, N, B, fixed_matrix_B);
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
    printf("  - 测试版本: 2个（统一使用 -O2 编译）\n");
    printf("  - 数值范围: 4组（0-1, 1-1e3, 1e3-1e5, 1e5-1e7）\n");
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
     printf("========================================================================================================\n");
    printf("\n测试版本说明:\n");
    printf("  [1] dgemm_unroll    : 原始循环展开实现 (src/)\n");
    printf("  [2] dgemm_unroll_ass: 内联汇编FMA优化 (opt/)\n");
     printf("========================================================================================================\n\n");
 }
 
 // 打印测试结果表格（单个数值范围）
 static void print_results_table_range(double results[NUM_OPT_FUNCS][NUM_TEST_CASES]) {
     // 打印列标题
     printf("%-25s", "优化版本");
     for (int i = 0; i < NUM_TEST_CASES; i++) {
         char dim_str[32];
         snprintf(dim_str, sizeof(dim_str), "%dx%d", test_cases[i].M, test_cases[i].N);
         printf(" | %14s", dim_str);
     }
     printf("\n");
     
     // 打印分隔线
     printf("%-25s", "-------------------------");
     for (int i = 0; i < NUM_TEST_CASES; i++) {
         printf("-+---------------");
     }
     printf("\n");
     
     // 打印每个优化版本的结果
     for (int opt = 0; opt < NUM_OPT_FUNCS; opt++) {
         printf("%-25s", opt_funcs[opt].name);
         for (int tc = 0; tc < NUM_TEST_CASES; tc++) {
             if (results[opt][tc] >= 0) {
                 printf(" | %14.8f", results[opt][tc]);
             } else {
                 printf(" | %14s", "FAIL");
             }
         }
         printf("\n");
     }
     printf("\n");
 }
 

 
 /* ========== 对外接口函数 ========== */
 
 /**
  * 运行DGEMM性能测试（对外接口）
  * 
  * 返回值：0=成功，-1=失败
  */
 int run_dgemm_benchmark(void) {
     // 打印表头
     print_header();
     
     // 结果存储数组 [范围][优化版本][测试用例]
     double results[NUM_VALUE_RANGES][NUM_OPT_FUNCS][NUM_TEST_CASES];
     
     // 总测试数
     int total_tests = NUM_VALUE_RANGES * NUM_OPT_FUNCS * NUM_TEST_CASES;
     int current_test = 0;
     
     // 遍历所有数值范围
     for (int range = 0; range < NUM_VALUE_RANGES; range++) {
         printf("\n\n");
         printf("========================================================================================================\n");
         printf("数值范围: %s [%.0e, %.0e]\n", 
                value_ranges[range].name,
                value_ranges[range].min_val,
                value_ranges[range].max_val);
         printf("========================================================================================================\n");
         
         // 初始化当前范围的固定矩阵
         init_fixed_matrices_range(value_ranges[range].min_val, value_ranges[range].max_val);
         
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
                     results[range][opt][tc] = avg_time;
                     printf("平均: %12.8f ms, 标准差: %12.8f ms\n", avg_time, stddev);
                 } else {
                     results[range][opt][tc] = -1.0;
                     printf("失败\n");
                 }
             }
         }
         
         // 打印当前范围的汇总表格
         printf("\n\n");
         printf("========================================================================================================\n");
         printf("数值范围 %s 性能汇总（平均时间，单位：毫秒）\n", value_ranges[range].name);
         printf("========================================================================================================\n");
         print_results_table_range(results[range]);
     }
     
     printf("\n\n");
     printf("========================================================================================================\n");
     printf("所有测试完成！\n");
     printf("总测试数: %d (4范围 × %d实现 × %ld规模)\n", total_tests, NUM_OPT_FUNCS, NUM_TEST_CASES);
     printf("========================================================================================================\n");
     
     return 0;
 }

 /* ========== 主函数（仅用于独立运行） ========== */
 
 #ifndef NO_MAIN
 int main(void) {
     return run_dgemm_benchmark();
 }
 #endif
