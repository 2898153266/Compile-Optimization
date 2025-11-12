#!/bin/bash
# 快速修复O1性能问题并测试

echo "=========================================="
echo "dgemm_naive O1性能问题修复脚本"
echo "=========================================="
echo ""

# 检查是否有备份
if [ ! -f dgemm_naive.c.original ]; then
    echo "📦 备份原始文件..."
    cp dgemm_naive.c dgemm_naive.c.original
    echo "✅ 已备份到 dgemm_naive.c.original"
else
    echo "ℹ️  检测到已有备份文件"
fi
echo ""

# 提供选项
echo "请选择修复方案："
echo "1) 快速修复 - 移除register关键字（推荐，5秒）"
echo "2) 最优化版 - 使用局部累加器（最佳性能，5秒）"
echo "3) 查看原始文件（不修复）"
echo "4) 恢复原始文件"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🔧 应用快速修复方案..."
        cp dgemm_naive_fixed.c dgemm_naive.c
        echo "✅ 已应用：移除register + 添加restrict"
        FIXED=1
        ;;
    2)
        echo ""
        echo "🚀 应用最优化方案..."
        cp dgemm_naive_optimal.c dgemm_naive.c
        echo "✅ 已应用：局部累加器 + ijk循环"
        FIXED=1
        ;;
    3)
        echo ""
        echo "📄 原始文件内容："
        cat dgemm_naive.c.original
        echo ""
        exit 0
        ;;
    4)
        echo ""
        echo "⏮️  恢复原始文件..."
        if [ -f dgemm_naive.c.original ]; then
            cp dgemm_naive.c.original dgemm_naive.c
            echo "✅ 已恢复原始文件"
        else
            echo "❌ 未找到备份文件"
        fi
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

if [ "$FIXED" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "开始重新编译和测试"
    echo "=========================================="
    echo ""
    
    # 清理
    echo "🧹 清理旧文件..."
    make clean_all > /dev/null 2>&1
    echo ""
    
    # 编译O0（对照组）
    echo "=========================================="
    echo "🔨 编译 O0 版本（对照组）"
    echo "=========================================="
    make build_O0
    if [ $? -ne 0 ]; then
        echo "❌ O0编译失败"
        exit 1
    fi
    echo ""
    
    # 编译O1（修复后）
    echo "=========================================="
    echo "🔨 编译 O1 版本（修复后）"
    echo "=========================================="
    make build_O1
    if [ $? -ne 0 ]; then
        echo "❌ O1编译失败"
        exit 1
    fi
    echo ""
    
    # 测试O0
    echo "=========================================="
    echo "📊 测试 O0 版本"
    echo "=========================================="
    echo "开始时间: $(date '+%H:%M:%S')"
    ./benchmark_O0 > results_O0_compare.txt 2>&1
    if [ -f benchmark_results.csv ]; then
        cp benchmark_results.csv benchmark_results_O0_compare.csv
    fi
    echo "完成时间: $(date '+%H:%M:%S')"
    echo ""
    
    # 测试O1
    echo "=========================================="
    echo "📊 测试 O1 版本（修复后）"
    echo "=========================================="
    echo "开始时间: $(date '+%H:%M:%S')"
    ./benchmark_O1 > results_O1_fixed.txt 2>&1
    if [ -f benchmark_results.csv ]; then
        cp benchmark_results.csv benchmark_results_O1_fixed.csv
    fi
    echo "完成时间: $(date '+%H:%M:%S')"
    echo ""
    
    # 显示对比
    echo "=========================================="
    echo "📈 性能对比结果"
    echo "=========================================="
    echo ""
    
    if [ -f benchmark_results_O0_compare.csv ] && [ -f benchmark_results_O1_fixed.csv ]; then
        echo "--- O0 版本（对照）---"
        cat benchmark_results_O0_compare.csv
        echo ""
        echo "--- O1 版本（修复后）---"
        cat benchmark_results_O1_fixed.csv
        echo ""
        
        # 提取dgemm_naive的时间进行对比
        echo "=========================================="
        echo "🎯 dgemm_naive 性能变化"
        echo "=========================================="
        
        # 提取O0的naive时间（第2列）
        o0_times=$(cat benchmark_results_O0_compare.csv | tail -n +2 | awk -F',' '{print $2}')
        # 提取O1的naive时间
        o1_times=$(cat benchmark_results_O1_fixed.csv | tail -n +2 | awk -F',' '{print $2}')
        
        # 简单显示
        echo ""
        echo "测试用例 | O0时间 | O1时间 | 加速比"
        echo "---------|--------|--------|--------"
        
        # 读取测试用例名称
        test_names=$(cat benchmark_results_O0_compare.csv | tail -n +2 | awk -F',' '{print $1}')
        
        paste <(echo "$test_names") \
              <(echo "$o0_times") \
              <(echo "$o1_times") | \
        while IFS=$'\t' read -r name t0 t1; do
            if [ ! -z "$t0" ] && [ ! -z "$t1" ]; then
                speedup=$(echo "scale=2; $t0 / $t1" | bc 2>/dev/null || echo "N/A")
                printf "%-20s | %7s | %7s | %6s\n" "$name" "$t0" "$t1" "${speedup}x"
            fi
        done
        
    else
        echo "⚠️  未找到结果文件，请检查测试是否成功"
    fi
    
    echo ""
    echo "=========================================="
    echo "✅ 修复和测试完成"
    echo "=========================================="
    echo ""
    echo "📁 生成的文件："
    ls -lh dgemm_naive.c* benchmark_results_*compare.csv benchmark_results_*fixed.csv results_*.txt 2>/dev/null
    echo ""
    echo "💡 说明："
    echo "  - 原始文件：dgemm_naive.c.original"
    echo "  - 当前文件：dgemm_naive.c (已修复)"
    echo "  - O0结果：benchmark_results_O0_compare.csv"
    echo "  - O1结果：benchmark_results_O1_fixed.csv"
    echo ""
    echo "✨ 预期效果："
    echo "  - 如果修复成功，O1的naive应该比O0快或相当"
    echo "  - 加速比应该 >= 1.0x（而不是之前的 0.2x）"
    echo ""
fi

