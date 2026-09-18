#!/bin/bash
V=$1; S=$2
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for f in $(seq 0 17); do
  python3 train_ablation.py "$V" "$f" "$S" 18000 >> "logs/${V}_s${S}.log" 2>&1
  echo "$(date +%H:%M:%S) ${V} seed${S} fold ${f} done" >> logs/progress_seeds.log
done
echo "$(date +%H:%M:%S) ${V} seed${S} ALL DONE" >> logs/progress_seeds.log
