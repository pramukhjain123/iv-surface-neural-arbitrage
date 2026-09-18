"""
Phase 4 quick-fix #1: seed-variance check.

Question: is the fold-to-fold spread in NN Task A/B RMSE (5.71-8.32 vol points,
per phase4_nn_eval_by_fold.csv) mostly a real fold effect (different train/test
windows are genuinely harder/easier) or mostly training noise (a different random
seed on the SAME fold would give just as much spread)?

Design: folds {2, 9, 17} (early / mid / late in the walk-forward sequence) were
each retrained from scratch with seed=2 and seed=3, in addition to the existing
seed=1 run from 07b/07c. That gives 3 seeds x 3 folds = 9 (fold, seed) RMSE values
per task. We compare:
  - within-fold, across-seed std (noise floor)
  - across-fold (using seed=1 only, matching the original by-fold table) std
If across-fold std >> within-fold-across-seed std, the fold-to-fold spread is a
real fold effect, not noise.

Usage: python3 08a_seed_variance.py
Outputs:
  results/phase4_seed_variance.csv          -- per (fold, seed, task) RMSE/MAE
  results/phase4_seed_variance_summary.csv  -- within-fold vs across-fold std comparison
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
nn_train = import_module("07b_nn_train_v3_fold")
nn_eval = import_module("07c_nn_eval_v3_allfolds")

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

FOLDS = [2, 9, 17]
SEEDS = [1, 2, 3]


def main():
    folds_df = pd.read_csv(ROOT / "05_splits" / "folds.csv").set_index("fold")
    con = duckdb.connect()
    rows = []

    for fold_id in FOLDS:
        frow = folds_df.loc[fold_id]
        train_start, train_end = str(frow["train_start"]), str(frow["train_end"])
        test_start, test_end = str(frow["test_start"]), str(frow["test_end"])

        train_ho_meta, test_meta = nn_eval.load_meta(con, train_start, train_end, test_start, test_end)
        train_ho = train_ho_meta.assign(w=train_ho_meta["iv"] ** 2 * train_ho_meta["tau"])
        test = test_meta.assign(w=test_meta["iv"] ** 2 * test_meta["tau"])

        for seed in SEEDS:
            params, norm_stats, step = nn_eval.load_run(fold_id, seed)
            k_mean, k_std = [float(x) for x in norm_stats]
            forward_batch, w_scalar = nn_train.make_forward(k_mean, k_std)

            for task_name, df in [("A", train_ho), ("B", test)]:
                if len(df) == 0:
                    continue
                w_pred = np.array(forward_batch(params, jnp.array(df["k"].values), jnp.array(df["tau"].values)))
                iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(df["tau"].values, 1e-8))
                err_volpts = (iv_pred - df["iv"].values) * 100.0
                rmse = float(np.sqrt(np.mean(err_volpts ** 2)))
                mae = float(np.mean(np.abs(err_volpts)))
                rows.append(dict(fold=fold_id, seed=seed, step=step, task=task_name,
                                  n=len(df), rmse_volpts=rmse, mae_volpts=mae))
                print(f"fold={fold_id} seed={seed} task={task_name} n={len(df)} "
                      f"rmse={rmse:.3f} mae={mae:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "phase4_seed_variance.csv", index=False)

    # summary: within-fold across-seed std vs across-fold (seed=1) std, per task
    summary_rows = []
    for task in ["A", "B"]:
        sub = res[res["task"] == task]
        within = sub.groupby("fold")["rmse_volpts"].std(ddof=1)
        within_mean_std = float(within.mean())
        across_fold_seed1 = sub[sub["seed"] == 1]["rmse_volpts"]
        across_fold_std = float(across_fold_seed1.std(ddof=1)) if len(across_fold_seed1) > 1 else float("nan")
        summary_rows.append(dict(
            task=task,
            within_fold_across_seed_std_mean=within_mean_std,
            across_fold_seed1_std=across_fold_std,
            ratio_acrossfold_to_withinfold=across_fold_std / within_mean_std if within_mean_std > 0 else float("nan"),
            per_fold_within_seed_std=dict(within.round(4)),
        ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "results" / "phase4_seed_variance_summary.csv", index=False)

    print("\n=== per-fold, per-task RMSE across seeds ===")
    print(res.pivot_table(index=["fold", "task"], columns="seed", values="rmse_volpts").to_string())
    print("\n=== summary: within-fold (across-seed) vs across-fold (seed=1) std of RMSE ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
