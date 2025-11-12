#!/bin/bash
# 快速测试 dgemm_neon_fast

echo "=========================================="
echo "快速测试 dgemm_neon_fast"
echo "=========================================="

# 清理
echo "1. 清理旧文件..."
make clean

# 编译
echo ""
echo "2. 编译 (O0优化级别)..."
make

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 编译失败！"
    exit 1
fi

# 运行测试
echo ""
echo "3. 运行性能测试..."
echo "=========================================="
./benchmark_O0

echo ""
echo "=========================================="
echo "✅ 测试完成！"
echo "=========================================="
echo ""
echo "结果已保存到: benchmark_results.csv"

