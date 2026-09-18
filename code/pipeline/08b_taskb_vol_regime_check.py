"""
Phase 4 quick-fix #3: check whether the "NN Task B RMSE < Task A RMSE" pattern
(observed in all 17 blind folds, per phase4_nn_eval_by_fold.csv) is confounded
by later test windows simply being calmer markets than the training-window days,
rather than reflecting genuine extrapolation ability.

Reuses the volatility proxy already built in 07g_robustness_cuts.py: for each
date, mean IV of ATM (|k|<=0.03), short/medium-maturity (tau<=90/365) options.

For each of the 17 blind folds, compute:
  - mean vol proxy over the fold's train_window dates (Task A dates)
  - mean vol proxy over the fold's test_window dates (Task B dates)
  - NN Task A RMSE and Task B RMSE (from phase4_nn_eval_by_fold.csv, subset=all)
Then check two things:
  1. Does vol_test tend to be lower than vol_train (the confound's precondition)?
  2. Across folds, is the RMSE gap (rmseA - rmseB) correlated with the vol gap
     (vol_train - vol_test)? If the RMSE gap is roughly constant/positive
     regardless of the vol gap's sign and size, the pattern is not primarily a
     volatility-regime artifact.

Usage: python3 08b_taskb_vol_regime_check.py
Output: results/phase4_taskb_vol_regime_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

CONTAMINATED_FOLD = 16


def main():
    con = duckdb.connect()
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    nn_eval = pd.read_csv(ROOT / "results" / "phase4_nn_eval_by_fold.csv")
    nn_all = nn_eval[nn_eval["subset"] == "all"]

    proxy = con.execute(f"""
        SELECT date, avg(iv) AS atm_iv_proxy
        FROM '{ROOT}/04_iv/panel.parquet'
        WHERE abs(k) <= 0.03 AND tau <= {90/365}
        GROUP BY date
    """).df()
    proxy["date"] = pd.to_datetime(proxy["date"])

    rows = []
    for _, frow in folds.iterrows():
        fold_id = int(frow["fold"])
        if fold_id == CONTAMINATED_FOLD:
            continue
        train_start, train_end = pd.Timestamp(frow["train_start"]), pd.Timestamp(frow["train_end"])
        test_start, test_end = pd.Timestamp(frow["test_start"]), pd.Timestamp(frow["test_end"])

        vol_train = proxy.loc[(proxy["date"] >= train_start) & (proxy["date"] < train_end), "atm_iv_proxy"]
        vol_test = proxy.loc[(proxy["date"] >= test_start) & (proxy["date"] < test_end), "atm_iv_proxy"]
        if len(vol_train) == 0 or len(vol_test) == 0:
            continue

        rmse_a_row = nn_all[(nn_all["fold"] == fold_id) & (nn_all["task"] == "A")]
        rmse_b_row = nn_all[(nn_all["fold"] == fold_id) & (nn_all["task"] == "B")]
        if len(rmse_a_row) == 0 or len(rmse_b_row) == 0:
            continue
        rmse_a = float(rmse_a_row["rmse_volpts"].iloc[0])
        rmse_b = float(rmse_b_row["rmse_volpts"].iloc[0])

        rows.append(dict(
            fold=fold_id,
            vol_train_mean=float(vol_train.mean()),
            vol_test_mean=float(vol_test.mean()),
            vol_gap_train_minus_test=float(vol_train.mean() - vol_test.mean()),
            nn_rmse_taskA=rmse_a,
            nn_rmse_taskB=rmse_b,
            rmse_gap_A_minus_B=rmse_a - rmse_b,
        ))

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "phase4_taskb_vol_regime_check.csv", index=False)

    n_test_calmer = int((res["vol_gap_train_minus_test"] > 0).sum())
    corr = res["vol_gap_train_minus_test"].corr(res["rmse_gap_A_minus_B"])
    corr_rmseA_volA = res["nn_rmse_taskA"].corr(res["vol_train_mean"])
    corr_rmseB_volB = res["nn_rmse_taskB"].corr(res["vol_test_mean"])

    print(res.to_string(index=False))
    print(f"\nfolds where test window is calmer than train window: {n_test_calmer}/{len(res)}")
    print(f"corr(vol_gap_train_minus_test, rmse_gap_A_minus_B) = {corr:.3f}")
    print(f"  (if the RMSE gap were purely a vol-regime artifact, this should be strongly")
    print(f"   POSITIVE: bigger vol_gap -> bigger rmse_gap. A weak/near-zero correlation")
    print(f"   means the accuracy gap persists independent of the vol difference.)")
    print(f"corr(nn_rmse_taskA, vol_train_mean) across folds = {corr_rmseA_volA:.3f}")
    print(f"corr(nn_rmse_taskB, vol_test_mean) across folds  = {corr_rmseB_volB:.3f}")
    print(f"\nmean vol_train={res['vol_train_mean'].mean():.4f}  mean vol_test={res['vol_test_mean'].mean():.4f}")


if __name__ == "__main__":
    main()
