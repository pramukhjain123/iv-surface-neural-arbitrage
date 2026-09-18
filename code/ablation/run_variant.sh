#!/bin/bash
# Usage: run_variant.sh <variant>
V=$1
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for f in $(seq 0 17); do
  python3 train_ablation.py "$V" "$f" 1 18000 >> "logs/${V}.log" 2>&1
  echo "$(date +%H:%M:%S) ${V} fold ${f} done" >> logs/progress.log
done
echo "$(date +%H:%M:%S) ${V} ALL DONE" >> logs/progress.log
