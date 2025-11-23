/*
 * DGEMM 测试接口头文件
 * 
 * 提供给外部程序调用的测试接口函数
 * 可集成到更大的程序中使用
 */

#ifndef TEST_INTERFACE_H
#define TEST_INTERFACE_H

/**
 * 运行DGEMM性能测试
 * 
 * 功能：测试所有DGEMM实现在不同矩阵规模下的性能
 * 
 * 输出：结果直接通过printf输出到控制台
 * 
 * 返回值：
 *   0  - 测试成功
 *   -1 - 测试失败
 */
int run_dgemm_benchmark(void);

#endif /* TEST_INTERFACE_H */
