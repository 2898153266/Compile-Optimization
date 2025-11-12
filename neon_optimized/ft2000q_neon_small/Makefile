# Makefile for DGEMM Optimization Benchmark
# 针对 FT2000Q 平台，支持多种优化级别 (O0, O1, O2, O3)

CC = gcc

# 优化级别可通过命令行指定: make OPT_LEVEL=O1
# 默认使用 O0
OPT_LEVEL ?= O0

# 编译选项
CFLAGS = -$(OPT_LEVEL) -Wall -march=armv8-a -mtune=cortex-a72 -fopenmp
LDFLAGS = -lm -lrt -fopenmp

# 目标可执行文件（包含优化级别后缀）
TARGET = benchmark_$(OPT_LEVEL)

# 源文件
BENCHMARK_SRC = benchmark.c
OPT_SRCS = dgemm_neon_small.c 
           

# 所有源文件
ALL_SRCS = $(BENCHMARK_SRC) $(OPT_SRCS)

# 对象文件
OBJS = $(ALL_SRCS:.c=.o)

# 默认目标
all: $(TARGET)
	@echo "编译完成！"
	@echo "运行命令: ./$(TARGET)"

# 链接
$(TARGET): $(OBJS)
	@echo "链接 $@..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 编译规则
%.o: %.c
	@echo "编译 $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# 清理
clean:
	@echo "清理中间文件..."
	rm -f $(OBJS) $(TARGET) benchmark_results.csv

# 清理所有优化级别的可执行文件
clean_all:
	@echo "清理所有优化级别的文件..."
	rm -f *.o benchmark_O0 benchmark_O1 benchmark_O2 benchmark_O3 benchmark_results*.csv

# 运行测试
run: $(TARGET)
	@echo "开始性能测试 (优化级别: $(OPT_LEVEL))..."
	./$(TARGET)

# 编译所有优化级别
all_opt: build_O0 build_O1 build_O2
	@echo "所有优化级别编译完成！"
	@ls -lh benchmark_O*

# 单独编译各个优化级别
build_O0:
	@echo "编译 O0 版本..."
	@$(MAKE) OPT_LEVEL=O0 all

build_O1:
	@echo "编译 O1 版本..."
	@$(MAKE) OPT_LEVEL=O1 all

build_O2:
	@echo "编译 O2 版本..."
	@$(MAKE) OPT_LEVEL=O2 all

build_O3:
	@echo "编译 O3 版本..."
	@$(MAKE) OPT_LEVEL=O3 all

# 运行所有优化级别的测试
test_all: all_opt
	@echo "========================================"
	@echo "测试 O0 优化级别"
	@echo "========================================"
	./benchmark_O0 | tee results_O0.txt
	mv benchmark_results.csv benchmark_results_O0.csv
	@echo ""
	@echo "========================================"
	@echo "测试 O1 优化级别"
	@echo "========================================"
	./benchmark_O1 | tee results_O1.txt
	mv benchmark_results.csv benchmark_results_O1.csv
	@echo ""
	@echo "========================================"
	@echo "测试 O2 优化级别"
	@echo "========================================"
	./benchmark_O2 | tee results_O2.txt
	mv benchmark_results.csv benchmark_results_O2.csv
	@echo ""
	@echo "所有测试完成！结果保存在 benchmark_results_O*.csv"

# 显示编译配置
info:
	@echo "编译器: $(CC)"
	@echo "当前优化级别: $(OPT_LEVEL)"
	@echo "编译选项: $(CFLAGS)"
	@echo "链接选项: $(LDFLAGS)"
	@echo "目标文件: $(TARGET)"
	@echo "优化版本数: 1 (dgemm_neon_small)"
	@echo "测试用例数: 9"
	@echo "平台: FT2000Q (ARMv8)"
	@echo ""
	@echo "可用的优化级别: O0, O1, O2, O3"
	@echo "使用方法: make OPT_LEVEL=O1"

# 帮助信息
help:
	@echo "=========================================="
	@echo "DGEMM Benchmark Makefile - 使用说明"
	@echo "=========================================="
	@echo ""
	@echo "📌 基本用法："
	@echo "  make                    - 编译默认版本 (O0)"
	@echo "  make OPT_LEVEL=O1       - 编译指定优化级别"
	@echo "  make run                - 运行默认版本"
	@echo "  make clean              - 清理当前版本"
	@echo ""
	@echo "📌 多优化级别测试："
	@echo "  make all_opt            - 编译 O0, O1, O2 三个版本"
	@echo "  make test_all           - 编译并运行所有优化级别测试"
	@echo "  make clean_all          - 清理所有优化级别文件"
	@echo ""
	@echo "📌 单独编译："
	@echo "  make build_O0           - 只编译 O0 版本"
	@echo "  make build_O1           - 只编译 O1 版本"
	@echo "  make build_O2           - 只编译 O2 版本"
	@echo "  make build_O3           - 只编译 O3 版本"
	@echo ""
	@echo "📌 单独运行："
	@echo "  ./benchmark_O0          - 运行 O0 版本"
	@echo "  ./benchmark_O1          - 运行 O1 版本"
	@echo "  ./benchmark_O2          - 运行 O2 版本"
	@echo ""
	@echo "📌 其他："
	@echo "  make info               - 显示编译配置"
	@echo "  make help               - 显示此帮助信息"
	@echo ""
	@echo "=========================================="

.PHONY: all clean clean_all run info all_opt build_O0 build_O1 build_O2 build_O3 test_all help

