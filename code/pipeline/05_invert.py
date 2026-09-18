"""
05_invert.py -- Phase 1.5. Invert Black-76 implied vol on the filtered OTM
NIFTY panel. Vectorized Newton-Raphson (numpy, whole panel at once) with a
Brent's-method fallback (per-row, scipy) for any rows Newton fails to
converge on -- this matches the plan's "Brent's method... vol bracket
[0.001, 5.0]" spirit while actually finishing in the device-bridge's
per-call time budget (a pure per-row Brent loop measured ~2ms/row, i.e.
~11 minutes for the full panel -- too slow to run here in one shot).
Discards rows that still fail to converge after both passes, and reports
the count.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import norm
from scipy.optimize import brentq

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT
D3 = ROOT / "iv_surface" / "03_filtered"
D4 = ROOT / "iv_surface" / "04_iv"
D4.mkdir(parents=True, exist_ok=True)

def black76_price_vec(F, K, T, sigma, r, is_call):
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)

def vega_vec(F, K, T, sigma, r):
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    return F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)

def newton_iv(price, F, K, T, r, is_call, n_iter=100, tol=1e-6):
    sigma = np.full_like(price, 0.25)
    for _ in range(n_iter):
        model = black76_price_vec(F, K, T, sigma, r, is_call)
        veg = vega_vec(F, K, T, sigma, r)
        veg = np.where(veg < 1e-8, 1e-8, veg)
        step = (model - price) / veg
        step = np.clip(step, -1.0, 1.0)
        sigma = sigma - step
        sigma = np.clip(sigma, 1e-4, 5.0)
    resid = black76_price_vec(F, K, T, sigma, r, is_call) - price
    converged = np.abs(resid) < tol
    sigma = np.where(converged, sigma, np.nan)
    return sigma, converged

def brentq_iv(price, F, K, T, r, is_call):
    intrinsic = max(0.0, (F - K) if is_call else (K - F)) * np.exp(-r * T)
    if price <= intrinsic + 1e-9:
        return np.nan
    try:
        return brentq(lambda s: black76_price_vec(np.array([F]), np.array([K]), np.array([T]),
                                                    np.array([s]), np.array([r]), np.array([is_call]))[0] - price,
                      1e-4, 5.0, xtol=1e-8, maxiter=200)
    except ValueError:
        return np.nan

def main():
    df = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D3 / 'nifty_filtered.parquet'}')").df()
    is_call = (df["opt_type"] == "CE").to_numpy()
    F = df["fwd"].to_numpy()
    K = df["strike"].to_numpy()
    T = df["tau"].to_numpy()
    r = df["r"].to_numpy()
    P = df["settle_price"].to_numpy()

    sigma, converged = newton_iv(P, F, K, T, r, is_call)
    print(f"Newton converged: {converged.sum()}/{len(df)} ({100*converged.mean():.2f}%)")

    fail_idx = np.where(~converged)[0]
    print(f"Falling back to Brent's method on {len(fail_idx)} rows...")
    n_recovered = 0
    for i in fail_idx:
        v = brentq_iv(P[i], F[i], K[i], T[i], r[i], is_call[i])
        if np.isfinite(v):
            sigma[i] = v
            n_recovered += 1
    print(f"Brent recovered: {n_recovered}/{len(fail_idx)}")

    df["iv"] = sigma
    n_final_nan = df["iv"].isna().sum()
    print(f"Final non-convergent (discarded): {n_final_nan} ({100*n_final_nan/len(df):.2f}%)")

    kept = df[df["iv"].notna()].copy()
    con = duckdb.connect()
    con.register("kept_tbl", kept)
    con.execute(f"COPY kept_tbl TO '{D4 / 'panel.parquet'}' (FORMAT PARQUET)")
    print(f"Wrote {len(kept)} rows to 04_iv/panel.parquet")
    print(kept["iv"].describe())

if __name__ == "__main__":
    main()
