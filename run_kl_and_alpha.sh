#!/bin/bash

# Script to run VCL on permuted and split MNIST with different alpha values
# This script runs experiments without coresets

# Create a log directory for experiment outputs
mkdir -p logs

# Alpha values to test
alpha_values=(0.1 0.5 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 5 10 20 50)

# Run permuted MNIST experiments
echo "====== Starting Permuted MNIST Experiments ======"
for alpha in "${alpha_values[@]}"; do
    echo "Running Permuted MNIST with alpha = $alpha"
    python vcl_new.py --experiment permuted --num_tasks 10 --alpha $alpha > "logs/permuted_mnist_alpha_${alpha}.log" 2>&1
    echo "Completed Permuted MNIST with alpha = $alpha"
done

# # Run split MNIST experiments
# echo "====== Starting Split MNIST Experiments ======"
# for alpha in "${alpha_values[@]}"; do
#     echo "Running Split MNIST with alpha = $alpha"
#     python vcl_new.py --experiment split --num_tasks 10 --alpha $alpha > "logs/split_mnist_alpha_${alpha}.log" 2>&1
#     echo "Completed Split MNIST with alpha = $alpha"
# done

echo "All experiments completed!"
echo "Results are saved in the 'results' directory"
echo "Logs are saved in the 'logs' directory"