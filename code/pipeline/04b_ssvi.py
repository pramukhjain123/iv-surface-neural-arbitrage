"""
04b_ssvi.py -- Phase 2.2, baseline 2/4: SSVI, fitted per day across all of
that day's expiries jointly.

    w(k,theta) = (theta/2) * (1 + rho*phi(theta)*k + sqrt((phi(theta)*k+rho)^2 + (1-rho^2)))
    phi(theta) = eta * theta^(-gamma)      [power-law SSVI, Gatheral & Jacquier 2014]

theta(tau) (the ATM total variance term) is calibrated in a first stage
directly from data: for each expiry that day, take the visible point
closest to k=0 as the empirical ATM total variance, then apply isotonic
regression (pooled-adjacent-violators) across the day's expiries sorted
by tau to enforce the calendar-arbitrage-free condition theta
non-decreasing in tau. (rho, eta, gamma) are then fit by least squares
across ALL of that day's visible points, constrained (SLSQP) to
Gatheral-Jacquier's sufficient no-butterfly condition
theta*phi(theta)*(1+|rho|) <= 4 evaluated at every theta(tau) used that
day, plus 0<gamma<1, eta>0, |rho|<1.

Because Task A only ever holds out strikes (never whole expiries), every
held-out point's tau matches an already-calibrated theta(tau) -- no
across-tau interpolation of theta is needed for scoring.
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

def isotonic_increasing(y):
    """Simple pooled-adjacent-violators for a non-decreasing fit."""
    y = np.array(y, dtype=float)
    n = len(y)
    level = y.copy()
    weight = np.ones(n)
    i = 0
    while i < len(level) - 1:
        if level[i] > level[i + 1]:
            new_val = (level[i] * weight[i] + level[i + 1] * weight[i + 1]) / (weight[i] + weight[i + 1])
            level[i] = new_val
            weight[i] += weight[i + 1]
            level = np.delete(level, i + 1)
            weight = np.delete(weight, i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    # expand back using weights (counts)
    expanded = []
    for lv, w in zip(level, weight):
        expanded += [lv] * int(round(w))
    return np.array(expanded)

def ssvi_w(k, theta, rho, eta, gamma):
    phi = eta * np.power(theta, -gamma)
    return (theta / 2.0) * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho ** 2)))

def main(start, end):
    panel = duckdb.connect().execute(
        f"SELECT date, expiry, strike, k, tau, iv FROM read_parquet('{D4 / 'panel.parquet'}')").df()
    panel["w"] = panel["iv"] ** 2 * panel["tau"]
    held = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D5 / 'held_out_strikes.parquet'}')").df()
    panel = panel.merge(held, on=["date", "expiry", "strike"], how="left")
    days = panel["date"].drop_duplicates().sort_values().reset_index(drop=True).iloc[start:end]

    rows = []
    n_fit, n_skip, n_infeasible = 0, 0, 0
    for d in days:
        g = panel[panel["date"] == d]
        vis = g[~g["is_heldout"]]
        ho = g[g["is_heldout"]]
        if len(vis) < 10 or len(ho) == 0:
            n_skip += 1
            continue
        exp_tau = vis.groupby("expiry")["tau"].first().sort_values()
        thetas_raw = []
        for exp, tau in exp_tau.items():
            sub = vis[vis["expiry"] == exp]
            idx = sub["k"].abs().idxmin()
            thetas_raw.append(sub.loc[idx, "w"])
        thetas = isotonic_increasing(thetas_raw)
        theta_map = dict(zip(exp_tau.index, thetas))
        vis = vis.copy()
        vis["theta"] = vis["expiry"].map(theta_map)

        theta_vals = np.array(list(theta_map.values()))
        cons = [{"type": "ineq", "fun": lambda p, tv=theta_vals: 4.0 - np.max(
            tv * p[1] * np.power(tv, -p[2]) * (1 + abs(p[0])))}]
        bounds = [(-0.999, 0.999), (1e-4, 5.0), (1e-4, 0.999)]
        best = None
        for x0 in [[-0.5, 0.5, 0.5], [0.0, 1.0, 0.3]]:
            try:
                res = minimize(lambda p: np.sum((ssvi_w(vis["k"].to_numpy(), vis["theta"].to_numpy(),
                                                          p[0], p[1], p[2]) - vis["w"].to_numpy()) ** 2),
                                x0=x0, bounds=bounds, constraints=cons, method="SLSQP",
                                options={"maxiter": 200, "ftol": 1e-12})
                if res.success and np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                continue
        if best is None:
            n_skip += 1
            continue
        rho, eta, gamma = best.x
        feasible = np.max(theta_vals * eta * np.power(theta_vals, -gamma) * (1 + abs(rho))) <= 4.0 + 1e-6
        n_fit += 1
        n_infeasible += (0 if feasible else 1)

        ho = ho.copy()
        ho["theta"] = ho["expiry"].map(theta_map)
        ho = ho.dropna(subset=["theta"])
        w_pred = ssvi_w(ho["k"].to_numpy(), ho["theta"].to_numpy(), rho, eta, gamma)
        iv_pred = np.sqrt(np.maximum(w_pred, 1e-10) / ho["tau"].to_numpy())
        for expiry, strike, iv_p, iv_a in zip(ho["expiry"], ho["strike"], iv_pred, ho["iv"]):
            rows.append({"method": "SSVI", "date": d, "expiry": expiry,
                         "strike": strike, "iv_actual": iv_a, "iv_pred": iv_p,
                         "feasible": feasible})
    out = pd.DataFrame(rows)
    tag = f"{start}_{end}"
    out.to_csv(RES / f"ssvi_chunk_{tag}.csv", index=False)
    print(f"chunk {tag}: days_fit={n_fit} skipped={n_skip} infeasible={n_infeasible} pred_rows={len(out)}")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
