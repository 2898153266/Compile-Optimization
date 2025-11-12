#!/bin/bash
# FT2000Q 多优化级别自动测试脚本
# 自动编译和测试 O0, O1, O2 三个优化级别

echo "=========================================="
echo "DGEMM 多优化级别性能测试"
echo "平台: FT2000Q (ARMv8)"
echo "=========================================="
echo ""

# 清理旧文件
echo "🧹 清理旧文件..."
make clean_all
echo ""

# 编译所有优化级别
echo "🔨 编译 O0, O1, O2 三个优化级别..."
make all_opt
if [ $? -ne 0 ]; then
    echo "❌ 编译失败！"
    exit 1
fi
echo ""

# 测试 O0
echo "=========================================="
echo "📊 测试 O0 优化级别"
echo "=========================================="
./benchmark_O0 | tee results_O0.txt
if [ -f benchmark_results.csv ]; then
    mv benchmark_results.csv benchmark_results_O0.csv
fi
echo ""

# 测试 O1
echo "=========================================="
echo "📊 测试 O1 优化级别"
echo "=========================================="
./benchmark_O1 | tee results_O1.txt
if [ -f benchmark_results.csv ]; then
    mv benchmark_results.csv benchmark_results_O1.csv
fi
echo ""

# 测试 O2
echo "=========================================="
echo "📊 测试 O2 优化级别"
echo "=========================================="
./benchmark_O2 | tee results_O2.txt
if [ -f benchmark_results.csv ]; then
    mv benchmark_results.csv benchmark_results_O2.csv
fi
echo ""

# 显示文件列表
echo "=========================================="
echo "✅ 所有测试完成！"
echo "=========================================="
echo ""
echo "生成的文件："
ls -lh benchmark_O* results_*.txt benchmark_results_*.csv 2>/dev/null
echo ""

# 简单的结果对比
echo "=========================================="
echo "📊 结果对比（查看详细数据请查看CSV文件）"
echo "=========================================="
echo ""

for csv in benchmark_results_O0.csv benchmark_results_O1.csv benchmark_results_O2.csv; do
    if [ -f "$csv" ]; then
        echo "--- $csv ---"
        cat "$csv"
        echo ""
    fi
done

echo "=========================================="
echo "💡 提示："
echo "  - 详细结果已保存到 benchmark_results_O*.csv"
echo "  - 完整日志已保存到 results_O*.txt"
echo "  - 使用 'cat benchmark_results_O0.csv' 查看单个结果"
echo "=========================================="

