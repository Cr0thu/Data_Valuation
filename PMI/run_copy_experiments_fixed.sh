#!/bin/bash

# 实验配置
EXP_ID=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="experiment_results_${EXP_ID}"
LOG_DIR="experiment_logs"

# 创建结果目录
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# 实验参数
TRAIN_CONFIGS=("30 30 30 30" "300 300 300 300")
TEST_CONFIGS=("20 40 40 20" "200 400 400 200")
PENALTIES=(10 100 1000)
NOISE_LEVELS=(0 10)

# 统计变量
SUCCESS_COUNT=0
FAILED_COUNT=0
TOTAL_PROCESSES_STARTED=0
TOTAL_PROCESSES_COMPLETED=0

# 检查bc命令是否可用
check_bc() {
    if ! command -v bc &> /dev/null; then
        echo "Warning: bc command not found. Installing bc..."
        if command -v apt-get &> /dev/null; then
            apt-get update && apt-get install -y bc
        elif command -v yum &> /dev/null; then
            yum install -y bc
        else
            echo "Error: Cannot install bc automatically. Please install bc manually."
            exit 1
        fi
    fi
}

# 检查系统资源
check_resources() {
    echo "Checking system resources..."
    
    # 检查可用内存
    local available_mem=$(free -g | awk '/^Mem:/{print $7}')
    echo "Available memory: ${available_mem}GB"
    
    # 检查GPU内存
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU memory status:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r gpu_id used total; do
            local available=$((total - used))
            echo "  GPU $gpu_id: ${used}MB used, ${available}MB available out of ${total}MB"
        done
    fi
}

# 运行单个实验
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
    
    local log_file="$LOG_DIR/experiment_${exp_num}.log"
    local result_file="$RESULTS_DIR/result_${exp_num}.txt"
    local process_log_file="$RESULTS_DIR/process_log_${exp_num}.txt"
    
    echo "Starting experiment $exp_num..." | tee -a "$log_file"
    echo "Parameters: train=($train_a_0,$train_a_1,$train_b_0,$train_b_1), test=($test_a_0,$test_a_1,$test_b_0,$test_b_1), penalty=$penalty, noise=$noise" | tee -a "$log_file"
    
    # 运行实验
    if ./run_copy_parallel.sh \
        --train_size_a_0 "$train_a_0" \
        --train_size_a_1 "$train_a_1" \
        --train_size_b_0 "$train_b_0" \
        --train_size_b_1 "$train_b_1" \
        --test_size_a_0 "$test_a_0" \
        --test_size_a_1 "$test_a_1" \
        --test_size_b_0 "$test_b_0" \
        --test_size_b_1 "$test_b_1" \
        --penalty "$penalty" \
        --noise_level "$noise" \
        > "$process_log_file" 2>&1; then
        
        echo "Experiment $exp_num completed successfully" | tee -a "$log_file"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # 复制结果文件
        if [ -f "output_copy_final.txt" ]; then
            cp "output_copy_final.txt" "$result_file"
            echo "Results saved to: $result_file" | tee -a "$log_file"
        fi
        
        # 统计成功完成的进程数量
        local completed_count=$(grep -c "COMPLETED_SUCCESSFULLY" "$process_log_file" 2>/dev/null || echo "0")
        local total_started=$(grep -c "RUN_ID:" "$process_log_file" 2>/dev/null || echo "0")
        echo "Process completion summary: $completed_count/$total_started processes completed successfully" | tee -a "$log_file"
        
        # 更新全局统计
        TOTAL_PROCESSES_STARTED=$((TOTAL_PROCESSES_STARTED + total_started))
        TOTAL_PROCESSES_COMPLETED=$((TOTAL_PROCESSES_COMPLETED + completed_count))
    else
        echo "Experiment $exp_num failed" | tee -a "$log_file"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    # 记录结束时间
    local end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Experiment $exp_num ended at: $end_time" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
}

# 主程序
main() {
    echo "Starting experiments at: $(date)"
    echo "Experiment ID: $EXP_ID"
    
    # 检查依赖
    check_bc
    check_resources
    
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
    
    if [ $((exp_counter - 1)) -gt 0 ]; then
        local success_rate=$(echo "scale=2; $SUCCESS_COUNT * 100 / $((exp_counter - 1))" | bc)
        echo "Success rate: ${success_rate}%"
    else
        echo "Success rate: 0%"
    fi
    
    echo "Total processes started: $TOTAL_PROCESSES_STARTED"
    echo "Total processes completed: $TOTAL_PROCESSES_COMPLETED"
    
    if [ $TOTAL_PROCESSES_STARTED -gt 0 ]; then
        local completion_rate=$(echo "scale=2; $TOTAL_PROCESSES_COMPLETED * 100 / $TOTAL_PROCESSES_STARTED" | bc)
        echo "Process completion rate: ${completion_rate}%"
    else
        echo "Process completion rate: 0%"
    fi
    
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
        
        if [ $((exp_counter - 1)) -gt 0 ]; then
            local success_rate=$(echo "scale=2; $SUCCESS_COUNT * 100 / $((exp_counter - 1))" | bc)
            echo "Success rate: ${success_rate}%"
        else
            echo "Success rate: 0%"
        fi
        
        echo "Total processes started: $TOTAL_PROCESSES_STARTED"
        echo "Total processes completed: $TOTAL_PROCESSES_COMPLETED"
        
        if [ $TOTAL_PROCESSES_STARTED -gt 0 ]; then
            local completion_rate=$(echo "scale=2; $TOTAL_PROCESSES_COMPLETED * 100 / $TOTAL_PROCESSES_STARTED" | bc)
            echo "Process completion rate: ${completion_rate}%"
        else
            echo "Process completion rate: 0%"
        fi
        
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
}

# 运行主程序
main "$@" 