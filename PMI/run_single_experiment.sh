#!/bin/bash

# 单次实验运行脚本
# 方便随时运行单个实验，参数易于修改

# 创建日志目录
LOG_DIR="single_experiment_logs"
mkdir -p $LOG_DIR

# 获取当前时间作为实验标识
EXP_ID=$(date +"%Y%m%d_%H%M%S")
echo "Starting single experiment with ID: $EXP_ID"

# 创建结果目录
RESULTS_DIR="single_experiment_results_${EXP_ID}"
mkdir -p $RESULTS_DIR

# 记录实验信息
echo "Experiment ID: $EXP_ID" > "$RESULTS_DIR/experiment_info.txt"
echo "Start time: $(date)" >> "$RESULTS_DIR/experiment_info.txt"
echo "" >> "$RESULTS_DIR/experiment_info.txt"

# 实验配置函数
run_experiment() {
    local train_a_0=$1
    local train_a_1=$2
    local train_b_0=$3
    local train_b_1=$4
    local test_a_0=$5
    local test_a_1=$6
    local test_b_0=$7
    local test_b_1=$8
    local penalty=$9
    local noise=${10}
    local description=${11}
    
    # 构建参数字符串
    local params="--train_size_a_0 $train_a_0 --train_size_a_1 $train_a_1 --train_size_b_0 $train_b_0 --train_size_b_1 $train_b_1 --test_size_a_0 $test_a_0 --test_size_a_1 $test_a_1 --test_size_b_0 $test_b_0 --test_size_b_1 $test_b_1 --penalty $penalty --noise_level $noise"
    
    echo "=========================================="
    echo "Running experiment: $description"
    echo "Train sizes: A($train_a_0,$train_a_1) B($train_b_0,$train_b_1)"
    echo "Test sizes: A($test_a_0,$test_a_1) B($test_b_0,$test_b_1)"
    echo "Penalty: $penalty, Noise: $noise"
    echo "=========================================="
    
    # 创建实验日志文件
    local log_file="$LOG_DIR/exp_${EXP_ID}.log"
    
    # 记录开始时间
    local start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Experiment started at: $start_time" | tee -a "$log_file"
    echo "Description: $description" | tee -a "$log_file"
    echo "Parameters: $params" | tee -a "$log_file"
    
    # 运行实验
    if ./run_copy_parallel.sh $params 2>&1 | tee -a "$log_file"; then
        echo "Experiment completed successfully" | tee -a "$log_file"
        
        # 复制结果文件
        if [ -f "output_copy_final.txt" ]; then
            cp "output_copy_final.txt" "$RESULTS_DIR/result.txt"
            echo "Results saved to: $RESULTS_DIR/result.txt"
        fi
    else
        echo "Experiment failed" | tee -a "$log_file"
    fi
    
    # 记录结束时间
    local end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Experiment ended at: $end_time" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
}

# ==========================================
# 在这里修改你的实验参数
# ==========================================

# 示例1：基础实验（默认参数）
echo "Running basic experiment..."
run_experiment 30 30 30 30 20 40 40 20 100 0.0 "Basic experiment with default parameters"

# 示例2：如果你想测试不同的训练集大小，取消下面的注释
# echo "Testing different training sizes..."
# run_experiment 20 20 20 20 20 40 40 20 100 0.0 "Small training set"
# sleep 5
# run_experiment 50 50 50 50 20 40 40 20 100 0.0 "Large training set"

# 示例3：如果你想测试不同的惩罚参数，取消下面的注释
# echo "Testing different penalty values..."
# for penalty in 10 50 100 200; do
#     run_experiment 30 30 30 30 20 40 40 20 $penalty 0.0 "Penalty $penalty"
#     sleep 5
# done

# 示例4：如果你想测试不同的噪声水平，取消下面的注释
# echo "Testing different noise levels..."
# for noise in 0.0 0.05 0.1 0.15; do
#     run_experiment 30 30 30 30 20 40 40 20 100 $noise "Noise level $noise"
#     sleep 5
# done

# 示例5：如果你想测试不平衡数据集，取消下面的注释
# echo "Testing imbalanced dataset..."
# run_experiment 20 40 40 20 15 45 45 15 100 0.0 "Imbalanced dataset A(20/40) B(40/20)"

# 示例6：如果你想测试不同的测试集配置，取消下面的注释
# echo "Testing different test set configurations..."
# run_experiment 30 30 30 30 15 30 30 15 100 0.0 "Small test set"
# sleep 5
# run_experiment 30 30 30 30 30 60 60 30 100 0.0 "Large test set"

echo "=========================================="
echo "SINGLE EXPERIMENT COMPLETED"
echo "=========================================="
echo "Experiment ID: $EXP_ID"
echo "End time: $(date)"

echo "Experiment results saved to: $RESULTS_DIR/"
echo "Experiment log saved to: $LOG_DIR/exp_${EXP_ID}.log"

echo "Process complete!" 