"""Extend ablation/08a_seed_variance.py's seed-variance check from 3 folds
{2, 9, 17} to ALL 18 folds x 3 seeds, now that seeds 2 and 3 have been
trained everywhere (seed_full/run_missing_seeds.py). Evaluates Task A/B
RMSE/MAE per (fold, seed) with the exact same methodology as
src/07c_nn_eval_v3_allfolds.py (subset="all"; fold 16 is the disclosed
non-blind fold reused from Phase 3 model selection, per the paper's own
Section 6.x disclosure -- flagged here, excluded from headline means the
same way Table tab:accuracy excludes it).

NOTE: this reproduces the *base* Task A/B RMSE methodology (matching
results/phase4_nn_eval_by_fold.csv, seed=1 column). The paper's final
published Table tab:accuracy numbers (mean 6.68 / 5.10) additionally apply
a "matched 20% re-score" adjustment described in Sec. 6.x that this script
does not reproduce (that adjustment's source data isn't in a file this
script reads) -- so treat the comparison in SEED_VARIANCE_FULL.md as
approximate: same pipeline stage as phase4_nn_eval_by_fold.csv, not a
recomputation of the exact published headline numbers.

Nothing under paper/ is touched by this script.

Usage: python3 aggregate_full_seed_variance.py
Outputs (under E:\\Research\\iv_surface\\results\\):
  phase4_seed_variance_full.csv          -- per (fold, seed, task) RMSE/MAE, subset=all
  phase4_seed_variance_full_summary.csv  -- within-fold-across-seed vs across-fold std, per task
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from importlib import import_module
nn_train = import_module("07b_nn_train_v3_fold")
nn_eval = import_module("07c_nn_eval_v3_allfolds")

ROOT = Path(r"E:\Research\iv_surface")
N_FOLDS = 18
SEEDS = [1, 2, 3]
CONTAMINATED_FOLD = 16


def main():
    folds_df = pd.read_csv(ROOT / "05_splits" / "folds.csv").set_index("fold")
    con = duckdb.connect()
    rows = []
    t_start = time.time()

    for fold_id in range(N_FOLDS):
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
                                  n=len(df), rmse_volpts=rmse, mae_volpts=mae,
                                  contaminated=(fold_id == CONTAMINATED_FOLD)))
        print(f"fold={fold_id} done ({time.time()-t_start:.0f}s elapsed)")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "phase4_seed_variance_full.csv", index=False)

    summary_rows = []
    headline_rows = []
    for task in ["A", "B"]:
        sub_all = res[res["task"] == task]
        sub_blind = sub_all[~sub_all["contaminated"]]

        # within-fold, across-seed std (noise floor), averaged over all 18 folds
        within = sub_all.groupby("fold")["rmse_volpts"].std(ddof=1)
        within_mean_std = float(within.mean())

        # across-fold spread using seed=1 only, matching the original by-fold table
        across_fold_seed1 = sub_all[sub_all["seed"] == 1]["rmse_volpts"]
        across_fold_std = float(across_fold_seed1.std(ddof=1))

        # headline: mean RMSE across the 17 blind folds, computed separately per seed
        # (matches Table tab:accuracy's own aggregation: mean-across-folds, not pooled)
        per_seed_headline = sub_blind.groupby("seed")["rmse_volpts"].mean()
        headline_mean = float(per_seed_headline.mean())
        headline_std = float(per_seed_headline.std(ddof=1))

        summary_rows.append(dict(
            task=task,
            within_fold_across_seed_std_mean=within_mean_std,
            across_fold_seed1_std=across_fold_std,
            ratio_acrossfold_to_withinfold=across_fold_std / within_mean_std if within_mean_std > 0 else float("nan"),
            headline_mean_across_17blind_folds_seed1=float(per_seed_headline.loc[1]),
            headline_mean_across_3seeds=headline_mean,
            headline_std_across_3seeds=headline_std,
        ))
        for seed, val in per_seed_headline.items():
            headline_rows.append(dict(task=task, seed=seed, mean_rmse_17blind_folds=val))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "results" / "phase4_seed_variance_full_summary.csv", index=False)
    headline = pd.DataFrame(headline_rows)
    headline.to_csv(ROOT / "results" / "phase4_seed_variance_full_headline.csv", index=False)

    print(f"\ntotal wall time: {time.time()-t_start:.1f}s")
    print("\n=== per-fold, per-task RMSE across seeds (subset=all) ===")
    print(res.pivot_table(index=["fold", "task"], columns="seed", values="rmse_volpts").to_string())
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== per-seed headline (mean RMSE across 17 blind folds) ===")
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
