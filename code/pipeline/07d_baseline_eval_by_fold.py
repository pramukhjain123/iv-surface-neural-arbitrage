"""
Phase 4: per-fold baseline metrics, computed by filtering the EXISTING
results/baseline_predictions_long.parquet (Phase 2's per-slice SVI/SSVI/ThinPlate/
CubicSpline fits, scored on the same 20% held-out strikes as the NN's Task A/B
splits) by each fold's train/test window dates from 05_splits/folds.csv.

No retraining needed: Phase 2's baselines are fit fresh per (date, expiry) slice
with no dependency on any walk-forward training window, so the existing table
already spans the full 2020-01-01 to 2026-01-30 sample.

Two subsets per fold, per method, both drawn from the SAME held-out-strike set:
  - "train_window" (Task-A-comparable): held-out-strike rows whose date falls in
    the fold's [train_start, train_end) window -- directly comparable to the NN's
    Task A (held-out strikes within training-window dates).
  - "test_window": held-out-strike rows whose date falls in the fold's
    [test_start, test_end) window. NOTE: this is NOT a like-for-like Task B
    comparison -- these baselines are refit fresh on each of those days' own
    other 80% of strikes, so they carry no cross-time memory the way a Task B
    test for the NN does (Phase 2/3 correctly noted "Task B does not apply" to
    per-day-refit baselines in that sense). It IS, however, the correct held-out
    row set for a matched-pairs Diebold-Mariano test against the NN's Task B
    predictions restricted to the same held-out strikes (see 07e).

Phase 4 quick-fix #2 (this revision): CubicSpline's per-fold RMSE/MAE is
dominated by a small number of catastrophic blow-up days (e.g. fold 13
test_window RMSE=480.6), which silently swamps the mean-based metrics and can
be misread as CubicSpline being globally unreliable rather than "usually fine,
occasionally explodes." To make that visible instead of hidden inside a single
RMSE number, this version adds two robust-location columns alongside RMSE/MAE:
  - median_abs_err_volpts: median of |err| in vol points (insensitive to the
    handful of blow-up rows).
  - trimmed_mean_abs_err_volpts: mean of |err| after dropping the top and
    bottom 10% of |err| values (a compromise between the mean, which blow-ups
    dominate, and the median, which ignores tail behavior entirely).
A large gap between RMSE and the median/trimmed-mean for a given
(fold, method, window) is itself the diagnostic: it says "this method's
typical-day accuracy is fine, its worst-day accuracy is not," which is exactly
CubicSpline's known failure mode and is quite different from SVI/SSVI/ThinPlate
where RMSE and the robust stats stay close together.

Usage: python3 07d_baseline_eval_by_fold.py
Output: results/phase4_baseline_eval_by_fold.csv
"""
import time
from pathlib import Path

import duckdb
import pandas as pd

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT


def main():
    con = duckdb.connect()
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    predictions = ROOT / "results" / "baseline_predictions_long_fixed.parquet"
    if not predictions.exists():
        predictions = ROOT / "results" / "baseline_predictions_long.parquet"

    rows = []
    t0 = time.time()
    for _, frow in folds.iterrows():
        fold_id = int(frow["fold"])
        train_start, train_end = str(frow["train_start"]), str(frow["train_end"])
        test_start, test_end = str(frow["test_start"]), str(frow["test_end"])

        for window_name, wstart, wend in [("train_window", train_start, train_end),
                                            ("test_window", test_start, test_end)]:
            q_simple = f"""
            WITH base AS (
                SELECT method, abs(err * 100.0) AS abserr
                FROM '{predictions}'
                WHERE date >= '{wstart}' AND date < '{wend}'
            ),
            qtiles AS (
                SELECT method,
                       quantile_cont(abserr, 0.10) AS q10,
                       quantile_cont(abserr, 0.90) AS q90
                FROM base
                GROUP BY method
            )
            SELECT b.method,
                   count(*) AS n,
                   sqrt(avg(abserr*abserr)) AS rmse_volpts,
                   avg(abserr) AS mae_volpts,
                   median(abserr) AS median_abs_err_volpts,
                   avg(abserr) FILTER (WHERE abserr >= q.q10 AND abserr <= q.q90) AS trimmed_mean_abs_err_volpts
            FROM base b JOIN qtiles q ON b.method = q.method
            GROUP BY b.method
            """
            agg = con.execute(q_simple).df()
            for _, r in agg.iterrows():
                rows.append(dict(fold=fold_id, method=r["method"], window=window_name,
                                  n=int(r["n"]), rmse_volpts=float(r["rmse_volpts"]),
                                  mae_volpts=float(r["mae_volpts"]),
                                  median_abs_err_volpts=float(r["median_abs_err_volpts"]),
                                  trimmed_mean_abs_err_volpts=float(r["trimmed_mean_abs_err_volpts"]),
                                  contaminated=(fold_id == 16)))
        print(f"fold {fold_id} done")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "phase4_baseline_eval_by_fold.csv", index=False)
    print(f"\nwall time: {time.time()-t0:.1f}s")
    print(res.to_string(index=False))

    print("\n=== mean across blind folds (excl. fold 16), by method/window: RMSE vs robust stats ===")
    blind = res[~res["contaminated"]]
    summary = blind.groupby(["method", "window"])[
        ["rmse_volpts", "mae_volpts", "median_abs_err_volpts", "trimmed_mean_abs_err_volpts"]
    ].mean().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
