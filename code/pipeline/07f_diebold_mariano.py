"""
Phase 4: Diebold-Mariano tests, NN v3 vs each of SVI/SSVI/ThinPlate/CubicSpline, on
EXACTLY matched held-out points (same (date, expiry, strike) rows for both sides).

Matching: results/baseline_predictions_long.parquet already scores each baseline on
the fixed 20% held-out strikes (05_splits/held_out_strikes.parquet). For a fair
comparison, the NN is evaluated on THOSE SAME held-out rows (not all Task B rows --
Task B includes visible strikes too, which the baselines never see held out), using
whichever fold's checkpoint owns that row's date as a Task-B (never-trained-on) date.

Loss: squared error in vol points (IV decimal x 100), matching the RMSE headline
metric. Per-row differentials d_i = sqerr_NN,i - sqerr_baseline,i are aggregated to a
per-date mean (so each calendar day, regardless of how many held-out strikes it has,
gets one observation) -- this is both the classical DM unit of analysis and a
practical fix for the strong same-day cross-sectional correlation between multiple
strikes' errors. The daily differential series is tested with a Diebold-Mariano
statistic using a Newey-West long-run variance (automatic bandwidth, Newey-West 1994
plug-in rule) to account for serial correlation across days.

Two scopes, exactly as elsewhere in Phase 4:
  - "blind_pooled_17folds": the 17 non-contaminated folds' test windows, which tile
    contiguously and disjointly from 2021-10-01 to 2026-01-31 (fold 16 excluded).
  - "fold16_contaminated": fold 16's test window alone, reported separately since its
    configuration was chosen partly by looking at this same window in Phase 3.

Usage: python3 07f_diebold_mariano.py
Output: results/phase4_diebold_mariano.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from scipy import stats

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

SEED = 1
CONTAMINATED_FOLD = 16
METHODS = ["SVI", "SSVI", "ThinPlate", "CubicSpline"]


def load_run(fold_id, seed):
    run_dir = ROOT / "06_nn" / "runs_v3_allfolds" / f"fold{fold_id}_seed{seed}"
    ckpt_path = run_dir / "ckpt.npz"
    ck = np.load(ckpt_path)
    params = [(ck[f"w{i}"], ck[f"b{i}"]) for i in range(3)]
    return params, ck["norm_stats"]


def nn_predict(params, norm_stats, k, tau):
    k_mean, k_std = [float(x) for x in norm_stats]
    x = ((np.asarray(k) - k_mean) / k_std)[:, None]
    for weight, bias in params[:-1]:
        x = np.logaddexp(0, x @ weight + bias)
    weight, bias = params[-1]
    raw = x @ weight + bias
    basis = np.column_stack([tau, np.sqrt(tau), np.log1p(tau)])
    w_pred = np.logaddexp(0, raw[:, 0]) + np.sum(np.logaddexp(0, raw[:, 1:]) * basis, axis=1)
    return np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(tau, 1e-8))


def nw_bandwidth(n):
    # Newey-West (1994) plug-in automatic bandwidth rule, simplified fixed variant
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def diebold_mariano(d):
    """d: 1-D array of per-period loss differentials (A - B). Returns (DM stat, 2-sided p)."""
    n = len(d)
    dbar = np.mean(d)
    L = nw_bandwidth(n)
    gamma0 = np.var(d, ddof=0)
    nw_var = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gamma_k = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        nw_var += 2 * w * gamma_k
    se = np.sqrt(max(nw_var, 1e-12) / n)
    dm_stat = dbar / se
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value, dbar, n, L


def main():
    con = duckdb.connect()
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    predictions = ROOT / "results" / "baseline_predictions_long_fixed.parquet"
    if not predictions.exists():
        predictions = ROOT / "results" / "baseline_predictions_long.parquet"
    t0 = time.time()

    rows_out = []
    matched_nn_rows = []
    per_scope_daily = {}

    for scope_name, scope_folds in [("blind_pooled_17folds", [f for f in range(18) if f != CONTAMINATED_FOLD]),
                                      ("fold16_contaminated", [CONTAMINATED_FOLD])]:
        nn_frames = []
        for fold_id in scope_folds:
            frow = folds[folds["fold"] == fold_id].iloc[0]
            test_start, test_end = str(frow["test_start"]), str(frow["test_end"])
            q = f"""
            SELECT p.date, p.expiry, p.strike, p.k, p.tau, p.iv
            FROM '{ROOT}/04_iv/panel.parquet' p
            JOIN '{ROOT}/05_splits/held_out_strikes.parquet' h
              ON p.date=h.date AND p.expiry=h.expiry AND p.strike=h.strike
            WHERE p.date >= '{test_start}' AND p.date < '{test_end}' AND h.is_heldout = true
            """
            ho = con.execute(q).df()
            if len(ho) == 0:
                continue
            params, norm_stats = load_run(fold_id, SEED)
            iv_pred = nn_predict(params, norm_stats, ho["k"].values, ho["tau"].values)
            sqerr_nn = ((iv_pred - ho["iv"].values) * 100.0) ** 2
            d = ho[["date", "expiry", "strike"]].copy()
            d["sqerr_nn"] = sqerr_nn
            nn_frames.append(d)
            err = (iv_pred - ho["iv"].values) * 100.0
            matched_nn_rows.append(dict(scope=scope_name, fold=fold_id, n=len(err),
                                        rmse_volpts=float(np.sqrt(np.mean(err**2))),
                                        mae_volpts=float(np.mean(np.abs(err))),
                                        contaminated=(fold_id == CONTAMINATED_FOLD)))
            print(f"[{scope_name}] fold {fold_id}: n_heldout_in_testwindow={len(d)}")
        nn_all = pd.concat(nn_frames, ignore_index=True)

        for method in METHODS:
            q2 = f"""
            SELECT date, expiry, strike, (err * 100.0) * (err * 100.0) AS sqerr_bl
            FROM '{predictions}'
            WHERE method = '{method}'
            """
            bl = con.execute(q2).df()
            bl["date"] = pd.to_datetime(bl["date"])
            bl["expiry"] = pd.to_datetime(bl["expiry"])
            merged = nn_all.merge(bl, on=["date", "expiry", "strike"], how="inner")
            merged["d"] = merged["sqerr_nn"] - merged["sqerr_bl"]
            daily = merged.groupby("date")["d"].mean().reset_index().sort_values("date")
            per_scope_daily[(scope_name, method)] = daily
            dm_stat, p_value, dbar, n_days, L = diebold_mariano(daily["d"].values)
            nz = daily.loc[daily["d"] != 0, "d"].to_numpy()
            wilcoxon = stats.wilcoxon(nz, alternative="two-sided", method="approx")
            n_nn_better = int(np.sum(nz < 0))
            sign_p = stats.binomtest(n_nn_better, len(nz), 0.5, alternative="two-sided").pvalue
            better = "NN" if dbar < 0 else method
            rows_out.append(dict(scope=scope_name, method=method, n_matched_rows=len(merged),
                                  n_days=n_days, nw_lag=L, mean_loss_diff_nn_minus_baseline=dbar,
                                  median_daily_loss_diff=float(np.median(nz)),
                                  nn_better_days=n_nn_better,
                                  baseline_better_days=int(len(nz) - n_nn_better),
                                  sign_test_p=float(sign_p),
                                  wilcoxon_stat=float(wilcoxon.statistic),
                                  wilcoxon_p=float(wilcoxon.pvalue),
                                  dm_stat=dm_stat, p_value=p_value, lower_sq_err=better))
            print(f"[{scope_name}] NN vs {method}: n_rows={len(merged)} n_days={n_days} "
                  f"dbar={dbar:.4f} DM={dm_stat:.3f} p={p_value:.4f} (lower sq.err: {better})")

    res = pd.DataFrame(rows_out)
    res.to_csv(ROOT / "results" / "phase4_diebold_mariano.csv", index=False)
    pd.DataFrame(matched_nn_rows).to_csv(
        ROOT / "results" / "phase5_nn_matched_taskb_by_fold.csv", index=False)
    print(f"\ntotal wall time: {time.time()-t0:.1f}s")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
