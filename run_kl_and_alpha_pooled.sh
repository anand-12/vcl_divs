#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

alpha_values=(0.25 0.35 0.5 0.65 2 5 10)

parallel -j3 --halt now,fail=1 \
  'echo "→ running α={}" &&
   python vcl_alpha.py --experiment permuted --num_tasks 10 --alpha {} \
     > logs/permuted_mnist_alpha_{}.log 2>&1' ::: "${alpha_values[@]}"