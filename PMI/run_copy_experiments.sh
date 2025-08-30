#!/bin/bash

# 实验运行脚本
# 使用循环设置参数，方便随时运行单个实验

# 创建日志目录
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

# 获取当前时间作为实验标识
EXP_ID=$(date +"%Y%m%d_%H%M%S")
echo "Starting experiment with ID: $EXP_ID"

# 定义参数范围（使用循环）
# 训练集大小配置
TRAIN_CONFIGS=(
    "30 30 30 30"
    "300 300 300 300"
)

# 测试集大小配置
TEST_CONFIGS=(
    "20 40 40 20"
    "200 400 400 200"
)

# 惩罚参数
PENALTIES=(10 100 1000)

# 噪声水平
NOISE_LEVELS=(0 10)

# 创建结果目录
RESULTS_DIR="experiment_results_${EXP_ID}"
mkdir -p $RESULTS_DIR

# 记录实验信息
echo "Experiment ID: $EXP_ID" > "$RESULTS_DIR/experiment_info.txt"
echo "Start time: $(date)" >> "$RESULTS_DIR/experiment_info.txt"
echo "" >> "$RESULTS_DIR/experiment_info.txt"

# 运行计数器
SUCCESS_COUNT=0
FAILED_COUNT=0
TOTAL_PROCESSES_STARTED=0
TOTAL_PROCESSES_COMPLETED=0

# 实验配置函数
run_single_experiment() {
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
    local exp_num=${11}
    
    # 构建参数字符串
    local params="--train_size_a_0 $train_a_0 --train_size_a_1 $train_a_1 --train_size_b_0 $train_b_0 --train_size_b_1 $train_b_1 --test_size_a_0 $test_a_0 --test_size_a_1 $test_a_1 --test_size_b_0 $test_b_0 --test_size_b_1 $test_b_1 --penalty $penalty --noise_level $noise"
    
    echo "=========================================="
    echo "Running experiment $exp_num"
    echo "Train sizes: A($train_a_0,$train_a_1) B($train_b_0,$train_b_1)"
    echo "Test sizes: A($test_a_0,$test_a_1) B($test_b_0,$test_b_1)"
    echo "Penalty: $penalty, Noise: $noise"
    echo "=========================================="
    
    # 创建实验日志文件
    local log_file="$LOG_DIR/exp_${EXP_ID}_${exp_num}.log"
    
    # 记录开始时间
    local start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Experiment $exp_num started at: $start_time" | tee -a "$log_file"
    
    # 运行实验
    if ./run_copy_parallel.sh $params 2>&1 | tee -a "$log_file"; then
        echo "Experiment $exp_num completed successfully" | tee -a "$log_file"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # 复制结果文件
        if [ -f "output_copy_final.txt" ]; then
            cp "output_copy_final.txt" "$RESULTS_DIR/result_${exp_num}.txt"
            echo "Results saved to: $RESULTS_DIR/result_${exp_num}.txt"
        fi
        
        # 记录成功运行的进程信息
        local success_log_file=$(ls -t successful_processes_*.txt 2>/dev/null | head -1)
        if [ -n "$success_log_file" ]; then
            cp "$success_log_file" "$RESULTS_DIR/process_log_${exp_num}.txt"
            echo "Process log saved to: $RESULTS_DIR/process_log_${exp_num}.txt"
            
            # 统计成功完成的进程数量
            local completed_count=$(grep -c "COMPLETED_SUCCESSFULLY" "$success_log_file" 2>/dev/null || echo "0")
            local total_started=$(grep -c "RUN_ID:" "$success_log_file" 2>/dev/null || echo "0")
            echo "Process completion summary: $completed_count/$total_started processes completed successfully" | tee -a "$log_file"
            
            # 更新全局统计
            TOTAL_PROCESSES_STARTED=$((TOTAL_PROCESSES_STARTED + total_started))
            TOTAL_PROCESSES_COMPLETED=$((TOTAL_PROCESSES_COMPLETED + completed_count))
        fi
    else
        echo "Experiment $exp_num failed" | tee -a "$log_file"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    # 记录结束时间
    local end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Experiment $exp_num ended at: $end_time" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
}

# 实验编号
exp_counter=1

# 基础实验：不同训练集大小
echo "Running experiments with different training set sizes..."
for train_config in "${TRAIN_CONFIGS[@]}"; do
    read -r train_a_0 train_a_1 train_b_0 train_b_1 <<< "$train_config"
    run_single_experiment $train_a_0 $train_a_1 $train_b_0 $train_b_1 20 40 40 20 100 0.0 $exp_counter
    exp_counter=$((exp_counter + 1))
    sleep 10
done

# 不同惩罚参数实验
echo "Running experiments with different penalty values..."
for penalty in "${PENALTIES[@]}"; do
    run_single_experiment 30 30 30 30 20 40 40 20 $penalty 0.0 $exp_counter
    exp_counter=$((exp_counter + 1))
    sleep 10
done

# 不同噪声水平实验
echo "Running experiments with different noise levels..."
for noise in "${NOISE_LEVELS[@]}"; do
    run_single_experiment 30 30 30 30 20 40 40 20 100 $noise $exp_counter
    exp_counter=$((exp_counter + 1))
    sleep 10
done

# 不同测试集配置实验
echo "Running experiments with different test set configurations..."
for test_config in "${TEST_CONFIGS[@]}"; do
    read -r test_a_0 test_a_1 test_b_0 test_b_1 <<< "$test_config"
    run_single_experiment 30 30 30 30 $test_a_0 $test_a_1 $test_b_0 $test_b_1 100 0.0 $exp_counter
    exp_counter=$((exp_counter + 1))
    sleep 10
done

# 生成实验总结报告
echo "=========================================="
echo "EXPERIMENT COMPLETED"
echo "=========================================="
echo "Experiment ID: $EXP_ID"
echo "Total experiments: $((exp_counter - 1))"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILED_COUNT"
echo "Success rate: $(echo "scale=2; $SUCCESS_COUNT * 100 / $((exp_counter - 1))" | bc)%"
echo "Total processes started: $TOTAL_PROCESSES_STARTED"
echo "Total processes completed: $TOTAL_PROCESSES_COMPLETED"
echo "Process completion rate: $(echo "scale=2; $TOTAL_PROCESSES_COMPLETED * 100 / $TOTAL_PROCESSES_STARTED" | bc)%"
echo "End time: $(date)"

# 保存实验总结
{
    echo "=========================================="
    echo "EXPERIMENT SUMMARY"
    echo "=========================================="
    echo "Experiment ID: $EXP_ID"
    echo "Total experiments: $((exp_counter - 1))"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: $FAILED_COUNT"
    echo "Success rate: $(echo "scale=2; $SUCCESS_COUNT * 100 / $((exp_counter - 1))" | bc)%"
    echo "Total processes started: $TOTAL_PROCESSES_STARTED"
    echo "Total processes completed: $TOTAL_PROCESSES_COMPLETED"
    echo "Process completion rate: $(echo "scale=2; $TOTAL_PROCESSES_COMPLETED * 100 / $TOTAL_PROCESSES_STARTED" | bc)%"
    echo "End time: $(date)"
    echo ""
    echo "Parameter ranges used:"
    echo "  Training configurations: ${TRAIN_CONFIGS[*]}"
    echo "  Test configurations: ${TEST_CONFIGS[*]}"
    echo "  Penalties: ${PENALTIES[*]}"
    echo "  Noise levels: ${NOISE_LEVELS[*]}"
} > "$RESULTS_DIR/experiment_summary.txt"

echo "Experiment results saved to: $RESULTS_DIR/"
echo "Experiment summary saved to: $RESULTS_DIR/experiment_summary.txt"
echo "Individual logs saved to: $LOG_DIR/"

echo "Process complete!" 