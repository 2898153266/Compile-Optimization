#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGEMM 多优化级别结果对比分析工具
比较 O0, O1, O2 三个优化级别的性能差异
"""

import pandas as pd
import sys
import os

def load_csv(filename):
    """加载CSV结果文件"""
    if not os.path.exists(filename):
        print(f"❌ 文件不存在: {filename}")
        return None
    return pd.read_csv(filename, index_col=0)

def compare_optimization_levels():
    """比较不同优化级别的结果"""
    
    # 加载三个优化级别的结果
    results = {}
    for opt in ['O0', 'O1', 'O2']:
        filename = f'benchmark_results_{opt}.csv'
        df = load_csv(filename)
        if df is not None:
            results[opt] = df
    
    if len(results) == 0:
        print("❌ 没有找到任何结果文件！")
        print("请先运行: make test_all")
        return
    
    print("=" * 100)
    print("DGEMM 优化级别性能对比分析")
    print("=" * 100)
    print()
    
    # 获取所有测试用例名称
    test_cases = list(results[list(results.keys())[0]].columns)
    functions = list(results[list(results.keys())[0]].index)
    
    print(f"📊 找到 {len(results)} 个优化级别的结果")
    print(f"📊 测试函数: {', '.join(functions)}")
    print(f"📊 测试用例数: {len(test_cases)}")
    print()
    
    # 对每个函数进行分析
    for func in functions:
        print("=" * 100)
        print(f"🔍 函数: {func}")
        print("=" * 100)
        
        # 创建对比表格
        comparison = pd.DataFrame()
        for opt in sorted(results.keys()):
            comparison[opt] = results[opt].loc[func]
        
        # 显示原始时间（毫秒）
        print("\n⏱️  执行时间 (ms):")
        print(comparison.to_string())
        
        # 计算加速比（相对于O0）
        if 'O0' in comparison.columns:
            print("\n📈 相对 O0 的加速比:")
            speedup = pd.DataFrame()
            for opt in comparison.columns:
                speedup[opt] = comparison['O0'] / comparison[opt]
            print(speedup.to_string())
            
            # 统计信息
            print("\n📊 统计信息:")
            for opt in comparison.columns:
                if opt != 'O0':
                    avg_speedup = speedup[opt].mean()
                    max_speedup = speedup[opt].max()
                    min_speedup = speedup[opt].min()
                    print(f"  {opt}: 平均加速 {avg_speedup:.2f}x, "
                          f"最大 {max_speedup:.2f}x, 最小 {min_speedup:.2f}x")
        
        print()
    
    # 总体对比
    print("=" * 100)
    print("📊 总体性能对比（所有测试用例平均）")
    print("=" * 100)
    
    overall = pd.DataFrame()
    for opt in sorted(results.keys()):
        # 计算每个优化级别的平均时间
        overall[opt] = results[opt].mean(axis=1)
    
    print("\n⏱️  平均执行时间 (ms):")
    print(overall.to_string())
    
    if 'O0' in overall.columns:
        print("\n📈 平均加速比 (相对 O0):")
        speedup_overall = pd.DataFrame()
        for opt in overall.columns:
            speedup_overall[opt] = overall['O0'] / overall[opt]
        print(speedup_overall.to_string())
    
    print()
    print("=" * 100)

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("用法: python compare_results.py")
        print("比较 benchmark_results_O0.csv, O1.csv, O2.csv 的结果")
        return
    
    compare_optimization_levels()

if __name__ == '__main__':
    main()


