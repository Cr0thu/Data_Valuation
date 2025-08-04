#!/bin/bash

# Exit on error
set -e

# Log file
LOG_FILE="pmi_cifar_run.log"

echo "Starting PMI CIFAR experiment at $(date)" | tee -a "$LOG_FILE"

# Install required dependencies
echo "Installing required dependencies..." | tee -a "$LOG_FILE"
pip install jupyter nbconvert 2>&1 | tee -a "$LOG_FILE"

# Create necessary directories
echo "Creating output directories..." | tee -a "$LOG_FILE"
mkdir -p figure/imdb/detail

# Convert notebook to Python script
echo "Converting notebook to Python script..." | tee -a "$LOG_FILE"
jupyter nbconvert --to script PMI_LogisticRegression_bias_cifar.ipynb

# Run the script
echo "Running experiment..." | tee -a "$LOG_FILE"
python PMI_LogisticRegression_bias_cifar.py 2>&1 | tee -a "$LOG_FILE"

# Optional: Clean up
# echo "Cleaning up..." | tee -a "$LOG_FILE"
# rm PMI_LogisticRegression_bias_cifar.py

echo "Experiment completed at $(date)" | tee -a "$LOG_FILE" 