#!/bin/bash

# Define the parameter values
train_test_sizes=("100 100")
penalty=100
T_values=(2000)
K=20
N_dims=128
lmi_only=1

# Specific combinations for fix and rand
fix_rand_combinations=(
    "0 0"
)

# Iterate over all combinations of parameters
for train_test_size in "${train_test_sizes[@]}"; do
    for T in "${T_values[@]}"; do
        for fix_rand in "${fix_rand_combinations[@]}"; do
            # Extract fix and rand values
            fix=${fix_rand% *}
            rand=${fix_rand#* }
            echo "Current combination: fix=$fix, rand=$rand"
            # Run the Python script with the current set of parameters
            echo "Running: python pmi-rank-mnist.py --train_size ${train_test_size% *} --test_size ${train_test_size#* } --penalty $penalty --T $T --K $K --N_dims $N_dims $( [ "$fix" -eq 1 ] && echo "--fix" ) $( [ "$rand" -eq 1 ] && echo "--rand" ) $( [ "$lmi_only" -eq 1 ] && echo "--lmi_only" )"
            python pmi-rank-mnist.py --train_size ${train_test_size% *} --test_size ${train_test_size#* } --penalty $penalty --T $T --K $K --N_dims $N_dims $( [ "$fix" -eq 1 ] && echo "--fix" ) $( [ "$rand" -eq 1 ] && echo "--rand" ) $( [ "$lmi_only" -eq 1 ] && echo "--lmi_only" )
        done
    done
done