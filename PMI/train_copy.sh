#!/bin/bash

test_sizes=("20,40,40,20"
            "20,60,60,20"
            #  "20,80,80,20"
            #  "10,20,20,10"
            #  "10,30,30,10"
            #  "10,40,40,10"
             )

train_sizes=("30,30,30,30"
            # "40,40,40,40"
            # "40,20,20,40"
            )

penalties=(0.01)

for train_size in "${train_sizes[@]}"; do
    IFS=',' read -r train_size_a_0 train_size_a_1 train_size_b_0 train_size_b_1 <<< "$train_size"
    for test_size in "${test_sizes[@]}"; do
        IFS=',' read -r test_size_a_0 test_size_a_1 test_size_b_0 test_size_b_1 <<< "$test_size"
        for penalty in "${penalties[@]}"; do
            python PMI_bias_cifar_copy.py \
                --train_size_a_0 "$train_size_a_0" \
                --train_size_a_1 "$train_size_a_1" \
                --train_size_b_0 "$train_size_b_0" \
                --train_size_b_1 "$train_size_b_1" \
                --test_size_a_0 "$test_size_a_0" \
                --test_size_a_1 "$test_size_a_1" \
                --test_size_b_0 "$test_size_b_0" \
                --test_size_b_1 "$test_size_b_1" \
                --penalty "$penalty"
        done
    done
done
