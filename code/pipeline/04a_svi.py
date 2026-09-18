"""
04a_svi.py -- Phase 2.2, baseline 1/4: raw SVI per (date, expiry) slice.

w(k) = a + b(rho*(k-m) + sqrt((k-m)^2 + sigma^2))   [total variance, w=iv^2*tau]

Fit on the 80% "visible" strikes (held_out_strikes.parquet marks the fixed
20% Task-A test set), evaluated on the 20% held-out strikes -- same split
every baseline uses, for a fair comparison.

Constrained (SLSQP) to Gatheral & Jacquier (2014)'s sufficient no-butterfly
condition b*sigma*(1+|rho|) <= 2, plus b>=0, |rho|<1, sigma>0,
a + b*sigma*sqrt(1-rho^2) >= 0 (variance floor). Multi-start (3 starts) to
reduce sensitivity to local minima, matching the plan's "bad SVI fits are
a common source of unfair ML-wins results" warning. A slice needs >=5
visible points to fit 5 params; we require >=6 for a little slack.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.optimize import minimize

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT
D4, D5, RES = (ROOT / "iv_surface" / d for d in ("04_iv", "05_splits", "results"))
RES.mkdir(parents=True, exist_ok=True)
OUT_DIR = RES / "svi_fixed_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def svi_w(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def fit_one(k, w, rng):
    n = len(k)
    best = None
    bounds = [(-1.0, 5.0), (1e-5, 10.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 5.0)]
    cons = [{"type": "ineq", "fun": lambda p: 2.0 - p[1] * p[4] * (1 + abs(p[2]))},
            {"type": "ineq", "fun": lambda p: p[0] + p[1] * p[4] * np.sqrt(max(1 - p[2] ** 2, 0)) + 1e-6}]
    x0s = [
        [w.min(), 0.15, -0.4, 0.0, 0.15],
        [w.mean(), 0.35, -0.3, k.mean(), 0.25],
    ]
    for x0 in x0s:
        try:
            res = minimize(lambda p: np.sum((svi_w(k, *p) - w) ** 2), x0=x0,
                            bounds=bounds, constraints=cons, method="SLSQP",
                            options={"maxiter": 200, "ftol": 1e-12})
            if res.success and np.isfinite(res.fun):
                feasible = (
                    2.0 - res.x[1] * res.x[4] * (1 + abs(res.x[2])) >= -1e-7
                    and res.x[0] + res.x[1] * res.x[4]
                    * np.sqrt(max(1 - res.x[2] ** 2, 0)) >= -1e-7
                )
                if feasible and (best is None or res.fun < best.fun):
                    best = res
        except Exception:
            continue
    return best

def main(start, end):
    panel = duckdb.connect().execute(
        f"SELECT date, expiry, strike, k, tau, iv FROM read_parquet('{D4 / 'panel.parquet'}')").df()
    panel["w"] = panel["iv"] ** 2 * panel["tau"]
    held = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D5 / 'held_out_strikes.parquet'}')").df()
    panel = panel.merge(held, on=["date", "expiry", "strike"], how="left")

    slices = panel.drop_duplicates(["date", "expiry"])[["date", "expiry"]].reset_index(drop=True)
    slices = slices.iloc[start:end]

    rng = np.random.default_rng(0)
    rows = []
    n_fit, n_skip, n_infeasible = 0, 0, 0
    for r in slices.itertuples():
        g = panel[(panel["date"] == r.date) & (panel["expiry"] == r.expiry)]
        vis = g[~g["is_heldout"]]
        ho = g[g["is_heldout"]]
        if len(vis) < 6 or len(ho) == 0:
            n_skip += 1
            continue
        best = fit_one(vis["k"].to_numpy(), vis["w"].to_numpy(), rng)
        if best is None:
            n_skip += 1
            continue
        a, b, rho, m, sigma = best.x
        feasible = b * sigma * (1 + abs(rho)) <= 2.0 + 1e-6
        n_fit += 1
        n_infeasible += (0 if feasible else 1)
        w_pred = svi_w(ho["k"].to_numpy(), a, b, rho, m, sigma)
        iv_pred = np.sqrt(np.maximum(w_pred, 1e-10) / ho["tau"].to_numpy())
        for strike, iv_p, iv_a in zip(ho["strike"], iv_pred, ho["iv"]):
            rows.append({"method": "SVI", "date": r.date, "expiry": r.expiry,
                         "strike": strike, "iv_actual": iv_a, "iv_pred": iv_p,
                         "feasible": feasible})
    out = pd.DataFrame(rows)
    tag = f"{start}_{end}"
    out.to_csv(OUT_DIR / f"svi_chunk_{tag}.csv", index=False)
    print(f"chunk {tag}: slices fit={n_fit} skipped={n_skip} infeasible={n_infeasible} pred_rows={len(out)}")

if __name__ == "__main__":
    start, end = int(sys.argv[1]), int(sys.argv[2])
    main(start, end)
