#!/bin/bash
# FT2000Q 快速测试脚本 - O1和O2优化级别对比
# 专注于测试编译器优化级别的影响

set -e  # 遇到错误立即退出

echo "=========================================="
echo "FT2000Q - O1 vs O2 优化级别对比测试"
echo "=========================================="
echo ""

# 显示系统信息
echo "📋 系统信息："
uname -a
echo ""
gcc --version | head -1
echo ""

# 检查NEON支持
echo "🔍 检查NEON支持："
if gcc -march=armv8-a -dM -E - < /dev/null | grep -q __ARM_NEON; then
    echo "✅ NEON支持正常"
else
    echo "⚠️  NEON可能不支持，但会继续测试"
fi
echo ""

# 清理
echo "🧹 清理旧文件..."
make clean_all > /dev/null 2>&1
echo ""

# 编译O1
echo "=========================================="
echo "🔨 编译 O1 版本..."
echo "=========================================="
make build_O1
if [ $? -ne 0 ]; then
    echo "❌ O1编译失败！"
    exit 1
fi
echo "✅ O1 编译成功"
echo ""

# 编译O2
echo "=========================================="
echo "🔨 编译 O2 版本..."
echo "=========================================="
make build_O2
if [ $? -ne 0 ]; then
    echo "❌ O2编译失败！"
    exit 1
fi
echo "✅ O2 编译成功"
echo ""

# 查看编译结果
echo "📦 编译产物："
ls -lh benchmark_O1 benchmark_O2
echo ""

# 运行O1测试
echo "=========================================="
echo "🚀 运行 O1 性能测试..."
echo "=========================================="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
./benchmark_O1 | tee results_O1.txt
if [ -f benchmark_results.csv ]; then
    mv benchmark_results.csv benchmark_results_O1.csv
    echo "✅ O1 测试完成，结果已保存到 benchmark_results_O1.csv"
fi
echo ""

# 运行O2测试
echo "=========================================="
echo "🚀 运行 O2 性能测试..."
echo "=========================================="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
./benchmark_O2 | tee results_O2.txt
if [ -f benchmark_results.csv ]; then
    mv benchmark_results.csv benchmark_results_O2.csv
    echo "✅ O2 测试完成，结果已保存到 benchmark_results_O2.csv"
fi
echo ""

# 完成
echo "=========================================="
echo "✅ 测试全部完成！"
echo "=========================================="
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "📊 生成的结果文件："
ls -lh results_O*.txt benchmark_results_O*.csv
echo ""

# 显示简单对比
echo "=========================================="
echo "📈 O1 vs O2 快速对比"
echo "=========================================="
echo ""

if [ -f benchmark_results_O1.csv ] && [ -f benchmark_results_O2.csv ]; then
    echo "--- O1 优化级别结果 ---"
    cat benchmark_results_O1.csv
    echo ""
    echo "--- O2 优化级别结果 ---"
    cat benchmark_results_O2.csv
    echo ""
else
    echo "⚠️  CSV文件未找到"
fi

echo "=========================================="
echo "💡 后续步骤："
echo "=========================================="
echo "1. 查看详细结果："
echo "   cat results_O1.txt"
echo "   cat results_O2.txt"
echo ""
echo "2. 下载结果到本地："
echo "   scp user@server:$(pwd)/benchmark_results_O*.csv ./"
echo "   scp user@server:$(pwd)/results_O*.txt ./"
echo ""
echo "3. 使用Python比较（如果有）："
echo "   python3 compare_results.py benchmark_results_O1.csv benchmark_results_O2.csv"
echo ""
echo "4. 如需测试O0级别："
echo "   make build_O0"
echo "   ./benchmark_O0"
echo ""
echo "=========================================="

