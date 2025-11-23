/*
 * 测试接口调用示例
 * 
 * 演示如何在主程序中调用 DGEMM 性能测试接口
 */

#include <stdio.h>
#include "test_interface.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main(void) {
#ifdef _WIN32
    // Windows平台：设置控制台为UTF-8编码
    SetConsoleOutputCP(65001);
#endif

    printf("========================================\n");
    printf("DGEMM 性能测试程序\n");
    printf("========================================\n\n");
    
    // 调用性能测试接口
    int ret = run_dgemm_benchmark();
    
    printf("\n========================================\n");
    if (ret == 0) {
        printf("测试成功完成！\n");
    } else {
        printf("测试失败，错误代码: %d\n", ret);
    }
    printf("========================================\n");
    
    return ret;
}
