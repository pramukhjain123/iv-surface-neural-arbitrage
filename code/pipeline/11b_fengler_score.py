"""
11b_fengler_score.py -- score the Fengler baseline on exactly the conventions
already used for SVI / SSVI / ThinPlate / CubicSpline, so the new row drops
into the existing tables without special pleading.

Produces
  results/phase6_fengler_taskA_summary.csv   pooled Task A, in-range vs out-of-range
                                             (matches results/baseline_comparison_summary.csv)
  results/phase6_fengler_by_fold.csv         per-fold train/test window RMSE
                                             (matches results/phase4_baseline_eval_by_fold.csv)
  results/phase6_fengler_arbitrage.csv       butterfly + calendar violation rates
                                             (matches results/phase4_baseline_arb_summary_*.csv)

ARBITRAGE SCORING -- TWO MEASUREMENTS, DELIBERATELY
---------------------------------------------------
price-space : the quantity Fengler actually constrains. On the dense grid we
              check that the fitted normalised call curve is convex and has
              slope in [-1, 0], and that consecutive maturities are ordered.
              These should be identically zero; a non-zero count would mean the
              QP returned an infeasible point, so this doubles as a solver audit.
w-space     : the SAME Durrleman test the other four baselines are given, after
              converting the fitted price curve to implied volatility on the grid
              and finite-differencing. This is the comparable number, but it is a
              numerical proxy: near the no-arbitrage boundary the price-to-vol map
              is arbitrarily stiff, so grid points where the constrained price sits
              on the intrinsic bound have no finite implied volatility at all and
              are reported as undefined rather than silently dropped or counted
              as violations.

Usage: python3 11b_fengler_score.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

import importlib.util
_spec = importlib.util.spec_from_file_location("fengler", Path(__file__).parent / "11a_fengler.py")
FG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FG)

ROOT = FG.ROOT
D4, D5, RES = ROOT / "04_iv", ROOT / "05_splits", ROOT / "results"
K_PAD_FRAC = 0.2
K_GRID_N = 61
FD_H = 1e-3


def durrleman_g(k, w, wk, wkk):
    w_safe = np.maximum(w, 1e-8)
    return (1 - k * wk / (2 * w_safe)) ** 2 - (wk ** 2 / 4.0) * (1.0 / w_safe + 0.25) + wkk / 2.0


# ------------------------------------------------------------------ accuracy
def score_accuracy():
    parts = sorted((RES / "fengler_chunks").glob("fengler_pred_*_*.csv"))
    parts = [q for q in parts if q.stem in
             ("fengler_pred_0_500", "fengler_pred_500_1000", "fengler_pred_1000_1500")]
    if not parts:
        raise SystemExit("no Fengler prediction chunks found")
    pred = pd.concat([pd.read_csv(q, parse_dates=["date", "expiry"]) for q in parts],
                     ignore_index=True)
    pred = pred.drop_duplicates(["date", "expiry", "strike"])
    print(f"combined {len(parts)} chunks -> {len(pred):,} predictions")
    vr = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D5 / 'visible_k_range.parquet'}')").df()
    vr["date"] = pd.to_datetime(vr["date"]); vr["expiry"] = pd.to_datetime(vr["expiry"])
    pred = pred.merge(vr, on=["date", "expiry"], how="left")
    pred["k"] = np.log(pred["x"])
    pred["in_range"] = (pred["k"] >= pred["k_vis_min"]) & (pred["k"] <= pred["k_vis_max"])

    ok = pred["iv_pred"].notna()
    print(f"Fengler predictions: {len(pred):,}")
    print(f"  implied-vol recoverable: {ok.sum():,} ({100*ok.mean():.2f}%)")
    print(f"  fitted price on the no-arbitrage boundary (no finite IV): "
          f"{(~ok).sum():,} ({100*(~ok).mean():.2f}%)")
    print(f"    of those, out-of-range points: {int((~ok & ~pred['in_range']).sum()):,} "
          f"({100*(~ok & ~pred['in_range']).sum()/max((~ok).sum(),1):.1f}% of failures)")

    rows = []
    for rng, sub in [("in_range", pred[pred["in_range"]]), ("out_of_range", pred[~pred["in_range"]])]:
        s = sub[sub["iv_pred"].notna()]
        err = (s["iv_pred"] - s["iv_actual"]).to_numpy()
        rows.append({"method": "Fengler", "range": rng, "n_total": len(sub), "n_scored": len(s),
                     "coverage": len(s) / max(len(sub), 1),
                     "rmse_vol_pts": float(np.sqrt(np.mean(err ** 2)) * 100),
                     "mae_vol_pts": float(np.mean(np.abs(err)) * 100),
                     "median_abs_err_vol_pts": float(np.median(np.abs(err)) * 100)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "phase6_fengler_taskA_summary.csv", index=False)
    print("\nTask A, pooled over all 1,496 days:")
    print(out.to_string(index=False))

    folds = pd.read_csv(D5 / "folds.csv", parse_dates=["train_start", "train_end",
                                                       "test_start", "test_end"])
    frows = []
    for f in folds.itertuples():
        for win, lo, hi in [("train_window", f.train_start, f.train_end),
                            ("test_window", f.test_start, f.test_end)]:
            base = pred[(pred["date"] >= lo) & (pred["date"] < hi)]
            # 07d_baseline_eval_by_fold.py scores ALL held-out strikes, in-range
            # and out-of-range together, so the row that goes into Table
            # "accuracy" must use that convention or it is not comparable. The
            # in-range-only variant is kept alongside it, not instead of it.
            for scope, s0 in [("all_heldout", base), ("in_range_only", base[base["in_range"]])]:
                s = s0[s0["iv_pred"].notna()]
                if len(s) == 0:
                    continue
                err = (s["iv_pred"] - s["iv_actual"]).to_numpy()
                frows.append({"fold": f.fold, "method": "Fengler", "window": win,
                              "scope": scope, "n": len(s),
                              "coverage": len(s) / max(len(s0), 1),
                              "rmse_volpts": float(np.sqrt(np.mean(err ** 2)) * 100),
                              "mae_volpts": float(np.mean(np.abs(err)) * 100),
                              "median_abs_err_volpts": float(np.median(np.abs(err)) * 100),
                              "trimmed_mean_abs_err_volpts": float(
                                  np.mean(np.sort(np.abs(err))[
                                      int(0.1 * len(err)):max(int(0.9 * len(err)), 1)]) * 100)})
    fdf = pd.DataFrame(frows)
    fdf.to_csv(RES / "phase6_fengler_by_fold.csv", index=False)
    print("\nPer-fold means over the 17 blind folds (fold 16 excluded, as in Phase 4):")
    for scope in ("all_heldout", "in_range_only"):
        sub = fdf[(fdf["scope"] == scope) & (fdf["fold"] != 16)]
        tr = sub[sub["window"] == "train_window"]
        te = sub[sub["window"] == "test_window"]
        print(f"  {scope:15s} train {tr['rmse_volpts'].mean():8.3f}   "
              f"test {te['rmse_volpts'].mean():8.3f}   "
              f"test MAE {te['mae_volpts'].mean():7.3f}   "
              f"test median|err| {te['median_abs_err_volpts'].mean():7.3f}   "
              f"test trim-mean {te['trimmed_mean_abs_err_volpts'].mean():7.3f}   "
              f"coverage {te['coverage'].mean():.4f}")
    return pred


# ----------------------------------------------------------------- arbitrage
def score_arbitrage(start=0, end=10**9, tag="", stride=1):
    panel = FG.load()
    vr = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D5 / 'visible_k_range.parquet'}')").df()
    vr["date"] = pd.to_datetime(vr["date"]); vr["expiry"] = pd.to_datetime(vr["expiry"])
    vr_by_day = {d: g for d, g in vr.groupby("date")}

    all_days = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    # A contiguous block of days would tie the arbitrage measurement to one
    # volatility regime, so the default run strides evenly across the whole
    # 6.1 years instead.
    daysel = set(all_days.iloc[start:end:stride])
    print(f"scoring {len(daysel)} days (index {start}:{end}:{stride})")
    but_rows, cal_rows = [], []

    for d, gday in panel.groupby("date", sort=True):
        if d not in daysel:
            continue
        fits, _, _ = FG.fit_day(gday)
        if not fits:
            continue
        vrd = vr_by_day.get(pd.Timestamp(d))
        if vrd is None:
            continue

        # ---- butterfly, per slice
        for exp, (xk, g, gam, h, tau) in fits.items():
            r = vrd[vrd["expiry"] == pd.Timestamp(exp)]
            if len(r) == 0:
                continue
            kmin, kmax = float(r["k_vis_min"].iloc[0]), float(r["k_vis_max"].iloc[0])
            pad = (kmax - kmin) * K_PAD_FRAC
            kgrid = np.linspace(kmin - pad, kmax + pad, K_GRID_N)
            in_obs = (kgrid >= kmin) & (kgrid <= kmax)
            xg = np.exp(kgrid)

            # price space: what Fengler actually constrains
            c = FG.spline_eval(xk, g, gam, h, xg)
            dc = np.gradient(c, xg)
            d2c = np.gradient(dc, xg)
            n_convex_viol = int(np.sum(d2c < -1e-8))
            # exact piecewise-quadratic second derivative of the fitted spline,
            # i.e. the risk-neutral density up to a positive factor. This is the
            # butterfly condition itself, not a numerical proxy for it.
            gam_at = np.interp(xg, xk, gam)
            n_density_neg = int(np.sum(gam_at < -1e-12))
            n_slope_viol = int(np.sum((dc > 1e-8) | (dc < -1.0 - 1e-8)))

            # w space: the same Durrleman test the other baselines get
            iv = FG.invert_norm_call_vec(c, xg, tau)
            fin = np.isfinite(iv)
            n_und = int(np.sum(~fin))
            if fin.sum() >= 5:
                cp = FG.spline_eval(xk, g, gam, h, np.exp(kgrid + FD_H))
                cm = FG.spline_eval(xk, g, gam, h, np.exp(kgrid - FD_H))
                ivp = FG.invert_norm_call_vec(cp, np.exp(kgrid + FD_H), tau)
                ivm = FG.invert_norm_call_vec(cm, np.exp(kgrid - FD_H), tau)
                w = iv ** 2 * tau; wp = ivp ** 2 * tau; wm = ivm ** 2 * tau
                wk = (wp - wm) / (2 * FD_H)
                wkk = (wp - 2 * w + wm) / (FD_H ** 2)
                gg = durrleman_g(kgrid, w, wk, wkk)
                good = np.isfinite(gg)
                n_viol = int(np.sum((gg < 0) & good))
                n_viol_obs = int(np.sum((gg < 0) & good & in_obs))
            else:
                n_viol = n_viol_obs = 0
                good = fin
            but_rows.append(dict(date=d, expiry=exp, method="Fengler", n_grid=K_GRID_N,
                                 n_price_convex_viol=n_convex_viol,
                                 n_price_slope_viol=n_slope_viol,
                                 n_density_negative=n_density_neg,
                                 n_iv_undefined=n_und, n_scored=int(np.sum(good)),
                                 n_viol=n_viol, n_viol_observed=n_viol_obs,
                                 n_grid_observed=int(np.sum(in_obs))))

        # ---- calendar, consecutive expiries on the day's shared k grid
        if len(fits) >= 2:
            kmin_d, kmax_d = float(vrd["k_vis_min"].min()), float(vrd["k_vis_max"].max())
            pad_d = (kmax_d - kmin_d) * K_PAD_FRAC
            kgd = np.linspace(kmin_d - pad_d, kmax_d + pad_d, K_GRID_N)
            xgd = np.exp(kgd)
            ordered = sorted(fits.items(), key=lambda kv: kv[1][4])   # by tau ascending
            curves = []
            for exp, (xk, g, gam, h, tau) in ordered:
                c = FG.spline_eval(xk, g, gam, h, xgd)
                iv = FG.invert_norm_call_vec(c, xgd, tau)
                curves.append((exp, tau, c, iv ** 2 * tau, xk))
            for i in range(len(curves) - 1):
                e0, t0, c0, w0, xk0 = curves[i]
                e1, t1, c1, w1, xk1 = curves[i + 1]
                both = np.isfinite(w0) & np.isfinite(w1)
                # the ordering constraint is only imposed where BOTH maturities
                # have quotes; outside that the shorter curve is bounded by
                # nothing and the longer one is a tangent extrapolation, so
                # crossings there are an extrapolation artefact, not a failure
                # of the constrained fit. Reported separately, exactly as the
                # butterfly test separates the observed range from the padding.
                lo_c = max(xk0[0], xk1[0]); hi_c = min(xk0[-1], xk1[-1])
                supp = (xgd >= lo_c) & (xgd <= hi_c)
                cal_rows.append(dict(date=d, method="Fengler", expiry_earlier=e0, expiry_later=e1,
                                     n_grid=K_GRID_N,
                                     n_grid_common=int(supp.sum()),
                                     n_price_viol=int(np.sum(c1 < c0 - 1e-10)),
                                     n_price_viol_common=int(np.sum((c1 < c0 - 1e-10) & supp)),
                                     n_scored=int(both.sum()),
                                     n_viol=int(np.sum((w1 < w0 - 1e-10) & both)),
                                     n_viol_common=int(np.sum((w1 < w0 - 1e-10) & both & supp))))

    bdf = pd.DataFrame(but_rows); cdf = pd.DataFrame(cal_rows)
    bdf.to_csv(RES / f"phase6_fengler_butterfly{tag}.csv", index=False)
    cdf.to_csv(RES / f"phase6_fengler_calendar{tag}.csv", index=False)

    summ = pd.DataFrame([{
        "method": "Fengler",
        "n_slices": len(bdf),
        "price_convexity_violations": int(bdf["n_price_convex_viol"].sum()),
        "price_slope_violations": int(bdf["n_price_slope_viol"].sum()),
        "negative_density_points": int(bdf["n_density_negative"].sum()),
        "frac_slices_with_any_w_viol": float((bdf["n_viol"] > 0).mean()),
        "frac_slices_with_w_viol_in_observed_range": float((bdf["n_viol_observed"] > 0).mean()),
        "mean_frac_grid_points_viol": float((bdf["n_viol"] / bdf["n_grid"]).mean()),
        "mean_frac_grid_iv_undefined": float((bdf["n_iv_undefined"] / bdf["n_grid"]).mean()),
        "n_expiry_pairs": len(cdf),
        "calendar_price_viol_common_support": int(cdf["n_price_viol_common"].sum()) if len(cdf) else 0,
        "calendar_price_viol_incl_extrapolation": int(cdf["n_price_viol"].sum()) if len(cdf) else 0,
        "frac_pairs_w_calendar_viol_common_support":
            float((cdf["n_viol_common"] > 0).mean()) if len(cdf) else 0.0,
        "frac_pairs_w_calendar_viol_incl_extrapolation":
            float((cdf["n_viol"] > 0).mean()) if len(cdf) else 0.0,
    }])
    summ.to_csv(RES / f"phase6_fengler_arbitrage{tag}.csv", index=False)
    print("\nArbitrage measurement:")
    print(summ.T.to_string(header=False))


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "accuracy"):
        score_accuracy()
    if mode in ("all", "arbitrage"):
        a = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        b = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
        st = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        tg = f"_{a}_{b}_{st}" if len(sys.argv) > 2 else ""
        score_arbitrage(a, b, tg, st)
