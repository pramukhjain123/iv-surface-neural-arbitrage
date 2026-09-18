"""Driver: fill in seeds 2 and 3 for the 15 folds that currently only have
seed=1 (folds 2, 9, 17 already have all three seeds from the earlier
seed-variance check). Extends that check from 3 folds to all 18, so every
paper table can report mean +/- sd across 3 seeds instead of a single run.

Calls the project's own, unmodified 07b_nn_train_v3_fold.py exactly as the
published pipeline does (same steps=18000, same lambda=0.01). Resumable:
any (fold, seed) whose checkpoint is already at step>=18000 is skipped, so
this can be safely re-launched if interrupted.

Nothing under paper/ is touched by this script.
"""
import os, sys, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = Path(r"E:\Research\iv_surface")
SRC = ROOT / "src"
DEPS = ROOT / ".deps"
PY = r"C:\Users\pramu\AppData\Local\Programs\Python\Python313\python.exe"
STEPS = 18000
HERE = Path(__file__).resolve().parent
LOG = HERE / "progress.log"

# fold 0 seed 2 was kicked off manually before this driver started; still
# needs seed 3. Folds 2, 9, 17 already have seeds 1-3. Everything else needs
# seeds 2 and 3.
JOBS = [(0, 3)]
for f in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]:
    for s in (2, 3):
        JOBS.append((f, s))


def done_already(fold, seed):
    ck = ROOT / "06_nn" / "runs_v3_allfolds" / f"fold{fold}_seed{seed}" / "ckpt.npz"
    try:
        return int(np.load(ck)["step"]) >= STEPS
    except Exception:
        return False


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def run(job):
    fold, seed = job
    if done_already(fold, seed):
        log(f"SKIP fold{fold} seed{seed} (already at target)")
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DEPS)
    env["OMP_NUM_THREADS"] = "1"
    env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    t0 = time.time()
    r = subprocess.run(
        [PY, str(SRC / "07b_nn_train_v3_fold.py"), str(fold), str(seed), str(STEPS), str(STEPS)],
        cwd=str(SRC), env=env, capture_output=True, text=True,
    )
    dt = time.time() - t0
    if r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-3:] if r.stderr.strip() else ["<no stderr>"]
        log(f"FAIL fold{fold} seed{seed} ({dt:.0f}s): {tail}")
    else:
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        log(f"done fold{fold} seed{seed} ({dt:.0f}s) {last}")
    return r.returncode


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    todo = [j for j in JOBS if not done_already(*j)]
    log(f"=== START {len(todo)} jobs pending of {len(JOBS)} total, {n_workers} workers ===")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        rcs = list(ex.map(run, JOBS))
    log(f"=== ALL DONE in {(time.time()-t0)/60:.1f} min, failures={sum(1 for r in rcs if r)} ===")
