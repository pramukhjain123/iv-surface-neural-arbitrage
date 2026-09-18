#!/bin/bash
# Usage: run_sweep.sh <fold>  -- 6 lambdas x 3 seeds, 4500 steps. Resumable:
# skips any (lambda, seed) whose checkpoint already sits at step 4500.
F=$1
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for LAM in 0.0 0.01 0.1 1.0 10.0 100.0; do
  for S in 1 2 3; do
    TAG=$(python3 -c "print(f'{float('$LAM'):g}')")
    D="runs/soft_cal_only_lam${TAG}_fold${F}_seed${S}/ckpt.npz"
    if python3 -c "
import numpy as np,sys
try: sys.exit(0 if int(np.load('$D')['step'])==4500 else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
      echo "$(date +%H:%M:%S) fold${F} lam${LAM} seed${S} SKIP (done)" >> logs/progress_sweep.log
      continue
    fi
    python3 train_ablation.py soft_cal_only "$F" "$S" 4500 "$LAM" >> "logs/sweep_fold${F}.log" 2>&1
    echo "$(date +%H:%M:%S) fold${F} lam${LAM} seed${S} done" >> logs/progress_sweep.log
  done
done
echo "$(date +%H:%M:%S) fold${F} SWEEP DONE" >> logs/progress_sweep.log
