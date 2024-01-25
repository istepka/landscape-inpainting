#!/bin/bash

epochs=20
batch_size=32
learning_rate=0.001
model="UNet"

# python train.py --epochs 2 --batch_size $batch_size --learning_rate $learning_rate --experiment_name "test" --model $model
optimizers=("adam" "sgd" "rmsprop" "adagrad")
loss=("mse" "l1" "cross_entropy" "poisson" "kldiv")

echo "Starting benchmarking"

echo "Running benchmark for optimizers"
for i in "${optimizers[@]}"; do
    echo "Running benchmark for optimizer $i"
    python train.py --optimizer $i --epochs $epochs --batch_size $batch_size --learning_rate $learning_rate --experiment_name "$model optimizer_$i" --model $model
done

echo "Running benchmark for loss functions"
for i in "${loss[@]}"; do
    echo "Running benchmark for loss function $i"
    python train.py --loss $i --epochs $epochs --batch_size $batch_size --learning_rate $learning_rate --experiment_name "$model loss_$i" --model $model
done

echo "Done benchmarking"
