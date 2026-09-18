"""Parallel driver for the ablation re-run on the author's machine.

Runs the job queue with a pool of worker processes, each pinned to one thread
(the model is tiny; threading buys nothing, so N single-threaded processes give
near-linear throughput). Resumable: any job whose checkpoint already sits at the
target step count is skipped, so an interrupted run can simply be relaunched.

Usage:  python run_all_windows.py [n_workers]
"""
import os, sys, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("IVS_DATA_ROOT", HERE.parent))
OUT_ROOT = Path(os.environ.get("IVS_OUT_ROOT", HERE))
STEPS = 18000
N_FOLDS = 18
LOG = OUT_ROOT / "logs" / "winrun_progress.log"

# arm A validates the environment against the published v3; D and E are the
# matched pair the paper's claim rests on, so those get three seeds.
JOBS = []
for f in range(N_FOLDS):
    JOBS.append(("hard", f, 1))
for variant in ("hard_cal_only", "soft_cal_only"):
    for seed in (1, 2, 3):
        for f in range(N_FOLDS):
            JOBS.append((variant, f, seed))


def done_already(variant, fold, seed):
    ck = OUT_ROOT / "runs" / f"{variant}_fold{fold}_seed{seed}" / "ckpt.npz"
    try:
        return int(np.load(ck)["step"]) >= STEPS
    except Exception:
        return False


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def run(job):
    variant, fold, seed = job
    if done_already(variant, fold, seed):
        log(f"SKIP {variant} fold{fold} seed{seed}")
        return 0
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "train_ablation.py"),
                        variant, str(fold), str(seed), str(STEPS)],
                       cwd=str(HERE), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"FAIL {variant} fold{fold} seed{seed}: {r.stderr.strip().splitlines()[-1:]}")
    else:
        log(f"done {variant} fold{fold} seed{seed} ({time.time()-t0:.0f}s)")
    return r.returncode


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    todo = [j for j in JOBS if not done_already(*j)]
    log(f"=== START {len(todo)} jobs pending of {len(JOBS)} total, {n_workers} workers ===")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        rcs = list(ex.map(run, JOBS))
    log(f"=== ALL DONE in {(time.time()-t0)/60:.1f} min, failures={sum(1 for r in rcs if r)} ===")
