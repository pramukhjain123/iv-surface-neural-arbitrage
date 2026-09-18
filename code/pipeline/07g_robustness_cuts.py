"""
Phase 4 robustness cuts, computed on the blind-pooled Task B (test-window) held-out
points across the 17 non-contaminated folds -- the same scope used for the headline
Diebold-Mariano comparison in 07f.

Cut (i) -- highest-volatility decile of days excluded.
  No VIX series exists in this project (confirmed absent in the Phase 0 inventory),
  so a volatility proxy is constructed directly from the panel: for each day, the
  mean IV of ATM (|k|<=0.03, matching the bucket-decomposition ATM definition),
  short-or-medium-maturity (tau<=90/365) options, i.e. a same-day ATM-IV level
  analogous to how a VIX-style index is built from near-term ATM options. Days are
  ranked into deciles by this proxy; the top decile (highest general vol level) is
  excluded, and RMSE/MAE are recomputed on the remaining 90% of days, for the NN and
  each baseline, to check that headline conclusions are not being driven by a few
  extreme-vol days.

Cut (ii) -- monthly vs weekly expiry split.
  No explicit expiry-cycle flag exists in the panel (Phase 0 inventory: instrument
  is generically "OPT"), so the monthly/weekly split is derived directly from the
  expiry calendar: within each (expiry's) calendar month, the LATEST expiry date
  present in the data is the monthly contract; every earlier expiry date in that
  month is a weekly contract. This is robust to NSE's several changes of weekly
  expiry weekday over 2020-2026 (no fixed "Thursday" assumption is made).

Usage: python3 07g_robustness_cuts.py
Outputs:
  results/phase4_robustness_volcut.csv
  results/phase4_robustness_expirycut.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
nn_train = import_module("07b_nn_train_v3_fold")

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

SEED = 1
N_LAYERS = len(nn_train.K_LAYER_SIZES) - 1
CONTAMINATED_FOLD = 16
METHODS = ["SVI", "SSVI", "ThinPlate", "CubicSpline"]
BLIND_FOLDS = [f for f in range(18) if f != CONTAMINATED_FOLD]


def load_run(fold_id, seed):
    run_dir = ROOT / "06_nn" / "runs_v3_allfolds" / f"fold{fold_id}_seed{seed}"
    params, m, v, key, step = nn_train.load_ckpt(run_dir / "ckpt.npz", N_LAYERS)
    ck = np.load(run_dir / "ckpt.npz")
    return params, ck["norm_stats"]


def nn_predict(params, norm_stats, k, tau):
    k_mean, k_std = [float(x) for x in norm_stats]
    forward_batch, _ = nn_train.make_forward(k_mean, k_std)
    w_pred = np.array(forward_batch(params, jnp.array(k), jnp.array(tau)))
    return np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(tau, 1e-8))


def rmse_mae(err_volpts):
    return float(np.sqrt(np.mean(err_volpts ** 2))), float(np.mean(np.abs(err_volpts)))


def main():
    con = duckdb.connect()
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    t0 = time.time()

    # ---- Build the same held-out-strike, blind-Task-B row set as 07f (all points, not just held-out this time,
    #      to give a full-Task-B picture -- for NN this is ALL test-window rows; for baselines we still only
    #      have held-out-strike predictions, so the two robustness cuts are computed on the NN-Task-B-all-rows
    #      set and, separately, on the baseline held-out-only set, and reported side by side per subset. ----
    nn_rows = []
    for fold_id in BLIND_FOLDS:
        frow = folds[folds["fold"] == fold_id].iloc[0]
        test_start, test_end = str(frow["test_start"]), str(frow["test_end"])
        q = f"""
        SELECT date, expiry, strike, k, tau, iv
        FROM '{ROOT}/04_iv/panel.parquet'
        WHERE date >= '{test_start}' AND date < '{test_end}'
        """
        rows = con.execute(q).df()
        if len(rows) == 0:
            continue
        params, norm_stats = load_run(fold_id, SEED)
        iv_pred = nn_predict(params, norm_stats, rows["k"].values, rows["tau"].values)
        rows["err_volpts"] = (iv_pred - rows["iv"].values) * 100.0
        nn_rows.append(rows)
        print(f"fold {fold_id}: n={len(rows)}")
    nn_all = pd.concat(nn_rows, ignore_index=True)

    blind_test_conditions = " OR ".join(
        f"(date >= '{str(r['test_start'])}' AND date < '{str(r['test_end'])}')"
        for _, r in folds.iterrows() if int(r["fold"]) != CONTAMINATED_FOLD
    )
    bl_all = con.execute(f"""
        SELECT method, date, expiry, strike, k, tau, err * 100.0 AS err_volpts
        FROM '{ROOT}/results/baseline_predictions_long.parquet'
        WHERE {blind_test_conditions}
    """).df()

    # ---- daily ATM-IV volatility proxy, from the full panel (not just held-out rows) ----
    proxy = con.execute(f"""
        SELECT date, avg(iv) AS atm_iv_proxy
        FROM '{ROOT}/04_iv/panel.parquet'
        WHERE abs(k) <= 0.03 AND tau <= {90/365}
        GROUP BY date
    """).df()
    proxy["decile"] = pd.qcut(proxy["atm_iv_proxy"], 10, labels=False, duplicates="drop")
    high_vol_dates = set(proxy.loc[proxy["decile"] == proxy["decile"].max(), "date"])
    print(f"\nvol proxy: {len(proxy)} days, top decile = {len(high_vol_dates)} days, "
          f"proxy range [{proxy['atm_iv_proxy'].min():.4f}, {proxy['atm_iv_proxy'].max():.4f}]")

    volcut_rows = []
    for label, keep_mask_fn in [("all_days", lambda d: np.ones(len(d), dtype=bool)),
                                  ("excl_top_vol_decile", lambda d: ~d["date"].isin(high_vol_dates))]:
        mask = keep_mask_fn(nn_all)
        rmse, mae = rmse_mae(nn_all.loc[mask, "err_volpts"].values)
        volcut_rows.append(dict(method="NN_v3", subset=label, n=int(mask.sum()), rmse_volpts=rmse, mae_volpts=mae))
        for method in METHODS:
            sub = bl_all[bl_all["method"] == method]
            m2 = keep_mask_fn(sub)
            rmse, mae = rmse_mae(sub.loc[m2, "err_volpts"].values)
            volcut_rows.append(dict(method=method, subset=label, n=int(m2.sum()), rmse_volpts=rmse, mae_volpts=mae))
    volcut_df = pd.DataFrame(volcut_rows)
    volcut_df.to_csv(ROOT / "results" / "phase4_robustness_volcut.csv", index=False)
    print("\n=== volatility-decile cut ===")
    print(volcut_df.to_string(index=False))

    # ---- monthly vs weekly expiry split ----
    expiry_all = con.execute(f"""
        SELECT DISTINCT expiry FROM '{ROOT}/04_iv/panel.parquet'
    """).df()
    expiry_all["month"] = pd.to_datetime(expiry_all["expiry"]).values.astype("datetime64[M]")
    max_per_month = expiry_all.groupby("month")["expiry"].transform("max")
    expiry_all["is_monthly"] = expiry_all["expiry"] == max_per_month
    monthly_expiries = set(expiry_all.loc[expiry_all["is_monthly"], "expiry"])
    print(f"\nexpiry calendar: {len(expiry_all)} distinct expiries, {len(monthly_expiries)} monthly, "
          f"{len(expiry_all) - len(monthly_expiries)} weekly")

    expirycut_rows = []
    for label, is_monthly_flag in [("monthly", True), ("weekly", False)]:
        mask = nn_all["expiry"].isin(monthly_expiries) == is_monthly_flag
        rmse, mae = rmse_mae(nn_all.loc[mask, "err_volpts"].values)
        expirycut_rows.append(dict(method="NN_v3", subset=label, n=int(mask.sum()), rmse_volpts=rmse, mae_volpts=mae))
        for method in METHODS:
            sub = bl_all[bl_all["method"] == method]
            m2 = sub["expiry"].isin(monthly_expiries) == is_monthly_flag
            rmse, mae = rmse_mae(sub.loc[m2, "err_volpts"].values)
            expirycut_rows.append(dict(method=method, subset=label, n=int(m2.sum()), rmse_volpts=rmse, mae_volpts=mae))
    expirycut_df = pd.DataFrame(expirycut_rows)
    expirycut_df.to_csv(ROOT / "results" / "phase4_robustness_expirycut.csv", index=False)
    print("\n=== monthly vs weekly expiry cut ===")
    print(expirycut_df.to_string(index=False))

    print(f"\ntotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
