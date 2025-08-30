#!/bin/bash

# Default values
TRAIN_SIZE_A_0=30
TRAIN_SIZE_A_1=30
TRAIN_SIZE_B_0=30
TRAIN_SIZE_B_1=30
TEST_SIZE_A_0=20
TEST_SIZE_A_1=40
TEST_SIZE_B_0=40
TEST_SIZE_B_1=20
PENALTY=100
NOISE_LEVEL=0.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_size_a_0)
            TRAIN_SIZE_A_0="$2"
            shift 2
            ;;
        --train_size_a_1)
            TRAIN_SIZE_A_1="$2"
            shift 2
            ;;
        --train_size_b_0)
            TRAIN_SIZE_B_0="$2"
            shift 2
            ;;
        --train_size_b_1)
            TRAIN_SIZE_B_1="$2"
            shift 2
            ;;
        --test_size_a_0)
            TEST_SIZE_A_0="$2"
            shift 2
            ;;
        --test_size_a_1)
            TEST_SIZE_A_1="$2"
            shift 2
            ;;
        --test_size_b_0)
            TEST_SIZE_B_0="$2"
            shift 2
            ;;
        --test_size_b_1)
            TEST_SIZE_B_1="$2"
            shift 2
            ;;
        --penalty)
            PENALTY="$2"
            shift 2
            ;;
        --noise_level)
            NOISE_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--train_size_a_0 VALUE] [--train_size_a_1 VALUE] [--train_size_b_0 VALUE] [--train_size_b_1 VALUE] [--test_size_a_0 VALUE] [--test_size_a_1 VALUE] [--test_size_b_0 VALUE] [--test_size_b_1 VALUE] [--penalty VALUE] [--noise_level VALUE]"
            exit 1
            ;;
    esac
done

echo "Using parameters:"
echo "  train_size_a_0: $TRAIN_SIZE_A_0"
echo "  train_size_a_1: $TRAIN_SIZE_A_1"
echo "  train_size_b_0: $TRAIN_SIZE_B_0"
echo "  train_size_b_1: $TRAIN_SIZE_B_1"
echo "  test_size_a_0: $TEST_SIZE_A_0"
echo "  test_size_a_1: $TEST_SIZE_A_1"
echo "  test_size_b_0: $TEST_SIZE_B_0"
echo "  test_size_b_1: $TEST_SIZE_B_1"
echo "  penalty: $PENALTY"
echo "  noise_level: $NOISE_LEVEL"

# Function to check system resources
check_resources() {
    echo "Checking system resources..."
    
    # Check available memory
    local available_mem=$(free -g | awk '/^Mem:/{print $7}')
    echo "Available memory: ${available_mem}GB"
    
    # Check GPU memory
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU memory status:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r gpu_id used total; do
            local available=$((total - used))
            echo "  GPU $gpu_id: ${used}MB used, ${available}MB available out of ${total}MB"
        done
    fi
}

# Function to wait for resources to be available
wait_for_resources() {
    local gpu_id=$1
    local max_wait=60  # Maximum wait time in seconds
    local wait_time=0
    
    while [ $wait_time -lt $max_wait ]; do
        if command -v nvidia-smi &> /dev/null; then
            local gpu_memory=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' -v gpu="$gpu_id" '$1==gpu {print $2}')
            local total_memory=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' -v gpu="$gpu_id" '$1==gpu {print $3}')
            local available_memory=$((total_memory - gpu_memory))
            
            # If more than 8GB available, proceed
            if [ $available_memory -gt 8192 ]; then
                echo "GPU $gpu_id has sufficient memory (${available_memory}MB available)"
                return 0
            fi
        fi
        
        echo "Waiting for GPU $gpu_id resources... (${wait_time}s elapsed)"
        sleep 5
        wait_time=$((wait_time + 5))
    done
    
    echo "Warning: Timeout waiting for GPU $gpu_id resources"
    return 1
}

# Trap Ctrl+C and cleanup
cleanup() {
    echo "Cleaning up..."
    # Kill all background Python processes
    pkill -9 python
    exit 1
}

trap cleanup SIGINT SIGTERM

# Common arguments for all runs
COMMON_ARGS="--train_size_a_0 $TRAIN_SIZE_A_0 --train_size_a_1 $TRAIN_SIZE_A_1 --train_size_b_0 $TRAIN_SIZE_B_0 --train_size_b_1 $TRAIN_SIZE_B_1 \
            --test_size_a_0 $TEST_SIZE_A_0 --test_size_a_1 $TEST_SIZE_A_1 --test_size_b_0 $TEST_SIZE_B_0 --test_size_b_1 $TEST_SIZE_B_1 \
            --penalty $PENALTY --noise_level $NOISE_LEVEL"

# Array to store process PIDs
declare -a PIDS=()
declare -a RUN_IDS=()
declare -a GPU_IDS=()

# File to record successful process IDs
SUCCESS_LOG_FILE="successful_processes_$(date +%Y%m%d_%H%M%S).txt"
echo "# Successful process IDs for experiment started at $(date)" > "$SUCCESS_LOG_FILE"

# Function to run a single instance with retry mechanism
run_instance() {
    local GPU_ID=$1
    local RUN_ID=$2
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Starting run $RUN_ID on GPU $GPU_ID (attempt $((retry_count + 1)))"
        
        # Wait for resources to be available
        if wait_for_resources $GPU_ID; then
            # Start the process
            CUDA_VISIBLE_DEVICES=$GPU_ID python PMI_bias_cifar_copy.py $COMMON_ARGS --gpu_id 0 --run_id $RUN_ID &
            local pid=$!
            
            # Store process information
            PIDS+=($pid)
            RUN_IDS+=($RUN_ID)
            GPU_IDS+=($GPU_ID)
            
            echo "PID for run $RUN_ID: $pid"
            
            # Wait a moment to see if process starts successfully
            sleep 3
            
            # Check if process is still running
            if kill -0 $pid 2>/dev/null; then
                echo "Run $RUN_ID started successfully on GPU $GPU_ID"
                # Record successful process
                echo "RUN_ID: $RUN_ID, GPU_ID: $GPU_ID, PID: $pid" >> "$SUCCESS_LOG_FILE"
                return 0
            else
                echo "Run $RUN_ID failed to start on GPU $GPU_ID"
                # Remove from arrays
                unset PIDS[${#PIDS[@]}-1]
                unset RUN_IDS[${#RUN_IDS[@]}-1]
                unset GPU_IDS[${#GPU_IDS[@]}-1]
            fi
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "Retrying run $RUN_ID on GPU $GPU_ID in 10 seconds..."
            sleep 10
        fi
    done
    
    echo "Failed to start run $RUN_ID on GPU $GPU_ID after $max_retries attempts"
    return 1
}

# Function to monitor running processes
monitor_processes() {
    local failed_runs=()
    
    for i in "${!PIDS[@]}"; do
        local pid=${PIDS[$i]}
        local run_id=${RUN_IDS[$i]}
        local gpu_id=${GPU_IDS[$i]}
        
        if ! kill -0 $pid 2>/dev/null; then
            echo "Process for run $run_id (PID: $pid) on GPU $gpu_id has stopped"
            failed_runs+=($run_id)
        fi
    done
    
    if [ ${#failed_runs[@]} -gt 0 ]; then
        echo "Failed runs: ${failed_runs[*]}"
        return 1
    fi
    
    return 0
}

# Check system resources before starting
check_resources



echo "Starting all runs with improved resource management..."

# Start all 12 processes in parallel
echo "Starting all 12 processes in parallel..."
for GPU_ID in 0 1 2 3; do
    for i in {1..3}; do
        RUN_ID=$((GPU_ID * 3 + i))
        run_instance $GPU_ID $RUN_ID
        sleep 2  # Small delay between process starts
    done
done

# Wait for all processes to stabilize
echo "Waiting for all processes to stabilize..."
sleep 10

# Final status check
echo "Final status check..."
monitor_processes

echo "All processes launched. Total running: ${#PIDS[@]} out of 12"
echo "Running PIDs: ${PIDS[*]}"
echo "Running RUN_IDs: ${RUN_IDS[*]}"

# Wait for all background processes to complete
echo "Waiting for all processes to complete..."
wait

echo "All runs completed. Recording final successful processes..."

# Record final successful processes that completed
for i in "${!PIDS[@]}"; do
    local pid=${PIDS[$i]}
    local run_id=${RUN_IDS[$i]}
    local gpu_id=${GPU_IDS[$i]}
    
    # Check if process completed successfully (exit code 0)
    if wait $pid 2>/dev/null; then
        echo "RUN_ID: $run_id, GPU_ID: $gpu_id, PID: $pid - COMPLETED_SUCCESSFULLY" >> "$SUCCESS_LOG_FILE"
    else
        echo "RUN_ID: $run_id, GPU_ID: $gpu_id, PID: $pid - FAILED_OR_CRASHED" >> "$SUCCESS_LOG_FILE"
    fi
done

echo "Process completion status recorded in: $SUCCESS_LOG_FILE"
echo "All runs completed. Combining results..."

# Combine results from all runs
python - << EOF
import numpy as np
import glob
import time
import fcntl

# Extract all parameter values from shell variables
train_size_a_0 = $TRAIN_SIZE_A_0
train_size_a_1 = $TRAIN_SIZE_A_1
train_size_b_0 = $TRAIN_SIZE_B_0
train_size_b_1 = $TRAIN_SIZE_B_1
test_size_a_0 = $TEST_SIZE_A_0
test_size_a_1 = $TEST_SIZE_A_1
test_size_b_0 = $TEST_SIZE_B_0
test_size_b_1 = $TEST_SIZE_B_1
penalty = $PENALTY
noise = $NOISE_LEVEL

# Arrays to store results
all_scores = []
all_losses = []
all_accs = []

# Get all output files matching the pattern
output_files = glob.glob(f"output_copy_*_train_{train_size_a_0}_{train_size_a_1}_{train_size_b_0}_{train_size_b_1}_test_{test_size_a_0}_{test_size_a_1}_{test_size_b_0}_{test_size_b_1}.txt")

print(f"Found {len(output_files)} output files to process")

# Wait a bit to ensure all files are completely written
print("Waiting for files to be completely written...")
time.sleep(3)

# Read results from each file
for filename in sorted(output_files):
    try:
        with open(filename, 'r') as f:
            line = f.readline().strip()
            # Parse the line to extract score, loss, and accuracy values
            parts = line.split('&')
            score_mean = float(parts[2].split('\\\\pm')[0])
            loss_mean = float(parts[3].split('\\\\pm')[0])
            acc_mean = float(parts[4].split('\\\\pm')[0])
            
            all_scores.append(score_mean)
            all_losses.append(loss_mean)
            all_accs.append(acc_mean)
            print(f"Successfully processed {filename}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

print(f"Successfully processed {len(all_scores)} files")

if all_scores:
    # Calculate overall statistics
    score_mean = np.mean(all_scores)
    score_std = np.std(all_scores)
    loss_mean = np.mean(all_losses)
    loss_std = np.std(all_losses)
    acc_mean = np.mean(all_accs)
    acc_std = np.std(all_accs)

    # Write final results with file lock
    with open('output_copy_final.txt', 'w') as f:
        # 获取独占锁用于写入
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(f"train size: {train_size_a_0}, {train_size_a_1}, {train_size_b_0}, {train_size_b_1}; test size: {test_size_a_0}, {test_size_a_1}, {test_size_b_0}, {test_size_b_1}\\n")
            f.write(f"{penalty} & {noise} & Copy & {score_mean:.4f}\\\\pm{score_std:.4f} & {loss_mean:.4f}\\\\pm{loss_std:.4f} & {acc_mean:.4f}\\\\pm{acc_std:.4f} \\\\\\\\n")
            f.flush()  # 确保数据立即写入
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    print("Results have been combined and written to output_copy_final.txt")
else:
    print("No valid results found to combine")
EOF

echo "Process complete!" 