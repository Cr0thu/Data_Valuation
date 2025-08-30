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

# Function to run a single instance
run_instance() {
    local GPU_ID=$1
    local RUN_ID=$2
    echo "Starting run $RUN_ID on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python PMI_bias_cifar_copy.py $COMMON_ARGS --gpu_id 0 --run_id $RUN_ID &
    echo "PID for run $RUN_ID: $!"
}

# Make sure no previous Python processes are running
pkill -9 python
sleep 2

# Distribution of runs across GPUs:
# GPU 0: Runs 1, 2, 3
# GPU 1: Runs 4, 5, 6
# GPU 2: Runs 7, 8
# GPU 3: Runs 9, 10

echo "Starting all runs..."

# Launch runs on GPU 0
run_instance 0 1
run_instance 0 2
run_instance 0 3

# Launch runs on GPU 1
run_instance 1 4
run_instance 1 5
run_instance 1 6

# Launch runs on GPU 2
run_instance 2 7
run_instance 2 8

# Launch runs on GPU 3
run_instance 3 9
run_instance 3 10

echo "All processes launched. Waiting for completion..."

# Wait for all background processes to complete
wait

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