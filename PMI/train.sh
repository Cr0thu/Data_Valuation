#!/bin/bash

test_sizes=("20,60,60,20"
            "20,40,40,20"
            "60,180,180,60"
            "60,120,120,60"
            "40,80,80,40"
            "40,120,120,40"
            )

train_sizes=("30,30,30,30"
             "40,40,40,40"
             "80,80,80,80"
             "120,120,120,120"
             "400,400,400,400"
            )

penalties=(10000 100000)

for train_size in "${train_sizes[@]}"; do
    IFS=',' read -r train_size_a_0 train_size_a_1 train_size_b_0 train_size_b_1 <<< "$train_size"
    for test_size in "${test_sizes[@]}"; do
        IFS=',' read -r test_size_a_0 test_size_a_1 test_size_b_0 test_size_b_1 <<< "$test_size"
        for penalty in "${penalties[@]}"; do
            python PMI_bias_cifar.py \
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
