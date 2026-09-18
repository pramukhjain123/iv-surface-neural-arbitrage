"""
11c_forward_multistrike.py -- multi-strike regression forward, and a full
sensitivity of the paper's conclusions to the forward construction.

BACKGROUND
----------
Phase 1.2 (02_attach_forward.py) builds the forward for each (date, expiry) in
preference order: the matched FUTIDX/IDF settlement price where one exists,
otherwise a SINGLE-STRIKE put-call-parity estimate taken at the strike where
|C - P| is smallest, preferring strikes where both legs traded. The review
raised single-strike parity as a possible source of error, and the Phase-5
diagnostic (09d) already showed the proposed mechanism does not hold -- F6
rejection is *highest* for futures-implied forwards (36.3%) and lowest for
illiquid parity forwards (13.0%). What remained open, and was recorded in the
revision note as future work, is the multi-strike sensitivity itself.

ESTIMATOR
---------
Put-call parity across strikes is exactly linear:

    C(K) - P(K) = exp(-r*tau) * (F - K)

so regressing the call-put spread on strike over all liquid pairs in a slice
gives both the forward and, as a by-product, the implied discount factor:

    C - P = alpha + beta*K,   beta = -exp(-r*tau),   F = -alpha/beta

Two estimators are reported:
  ols     -- unconstrained least squares on (alpha, beta); F = -alpha/beta.
             beta is not imposed, so the fitted discount factor is a free
             diagnostic: if the parity relation is clean, exp(r*tau)*(-beta)
             should sit at 1.
  fixed   -- beta pinned at its known value -exp(-r*tau), leaving
             F = mean_i [ K_i + (C_i - P_i)*exp(r*tau) ], i.e. the natural
             multi-strike generalisation of the single-strike estimator
             actually used in Phase 1.

Pairs are restricted to strikes where BOTH legs have volume>0 and oi>0 and
|log(K/F_baseline)| <= ATM_BAND, because far-from-the-money pairs put one leg
deep in the money where the settlement price is stale and the spread carries
almost no forward information. A slice needs at least MIN_PAIRS such strikes.

MODES
-----
  estimate  -- fit both estimators for every slice, write the comparison table
  rebuild   -- re-run filters F1-F7 and the Black-76 inversion on the ORIGINAL
               option panel but with the multi-strike forward substituted,
               producing 04_iv/panel_fwdmulti.parquet. This is a genuine
               end-to-end rebuild of the derived panel; it is deliberately NOT
               wired into the walk-forward evaluation, because refitting all
               baselines and all 18 network folds on an alternative panel is a
               different paper. The point is to show whether the panel and its
               arbitrage properties move at all.
  compare   -- summarise the two panels side by side

Usage:
    python3 11c_forward_multistrike.py estimate
    python3 11c_forward_multistrike.py rebuild
    python3 11c_forward_multistrike.py compare
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from scipy.stats import norm
from scipy.optimize import brentq


def _find_project_root():
    candidates = [
        Path(r"E:\Research") / "iv_surface",
        Path.home() / "mnt" / "Research" / "iv_surface",
        Path.home() / "mnt" / "iv_surface",
        Path(__file__).resolve().parent.parent,
    ]
    for c in candidates:
        if (c / "04_iv" / "panel.parquet").exists():
            return c
    raise SystemExit("could not locate iv_surface project root")


ROOT = _find_project_root()
D2, D3, D4, RES = (ROOT / d for d in ("02_parsed", "03_filtered", "04_iv", "results"))

ATM_BAND = 0.10          # |log(K/F_baseline)| band for parity pairs
MIN_PAIRS = 4
TICK = 0.05


# ---------------------------------------------------------------- estimation
def estimate_forwards():
    src = D2 / "nifty_with_forward.parquet"
    df = duckdb.connect().execute(f"SELECT * FROM read_parquet('{src}')").df()
    print(f"loaded {len(df):,} option rows, "
          f"{df.drop_duplicates(['date','expiry']).shape[0]:,} slices")

    liq = df[(df["volume"] > 0) & (df["oi"] > 0)].copy()
    wide = liq.pivot_table(index=["date", "expiry", "strike"], columns="opt_type",
                           values="settle_price", aggfunc="first").reset_index()
    for c in ("CE", "PE"):
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide.dropna(subset=["CE", "PE"])

    meta = df.drop_duplicates(["date", "expiry"])[["date", "expiry", "tau", "r", "fwd", "fwd_source"]]
    wide = wide.merge(meta, on=["date", "expiry"], how="left")
    wide = wide[wide["fwd"].notna() & (wide["fwd"] > 0)]
    wide["k_base"] = np.log(wide["strike"] / wide["fwd"])
    wide = wide[wide["k_base"].abs() <= ATM_BAND]

    wide["spread"] = wide["CE"] - wide["PE"]
    wide["disc"] = np.exp(-wide["r"] * wide["tau"])

    recs = []
    for (d, e), g in wide.groupby(["date", "expiry"], sort=False):
        n = len(g)
        if n < MIN_PAIRS:
            continue
        K = g["strike"].to_numpy(float)
        y = g["spread"].to_numpy(float)
        disc = float(g["disc"].iloc[0])
        # unconstrained OLS
        A = np.column_stack([np.ones(n), K])
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        alpha, beta = coef
        f_ols = -alpha / beta if beta < -1e-12 else np.nan
        resid = y - A @ coef
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        # discount-factor diagnostic
        disc_implied = -beta
        # beta pinned at its known value
        f_fixed = float(np.mean(K + y / disc))
        f_fixed_sd = float(np.std(K + y / disc, ddof=1)) if n > 1 else np.nan

        recs.append({"date": d, "expiry": e, "n_pairs": n,
                     "fwd_base": float(g["fwd"].iloc[0]),
                     "fwd_source": g["fwd_source"].iloc[0],
                     "fwd_ols": f_ols, "fwd_fixed": f_fixed,
                     "fwd_fixed_sd": f_fixed_sd,
                     "disc_known": disc, "disc_implied": float(disc_implied),
                     "parity_rmse": rmse, "tau": float(g["tau"].iloc[0])})

    out = pd.DataFrame(recs)
    out["rel_diff_fixed"] = (out["fwd_fixed"] - out["fwd_base"]) / out["fwd_base"]
    out["rel_diff_ols"] = (out["fwd_ols"] - out["fwd_base"]) / out["fwd_base"]
    out.to_csv(RES / "phase6_forward_multistrike.csv", index=False)

    print(f"\nslices with >= {MIN_PAIRS} liquid parity pairs in |k|<={ATM_BAND}: {len(out):,}")
    print("\nrelative difference of multi-strike vs baseline forward, by baseline source:")
    summ = out.groupby("fwd_source")["rel_diff_fixed"].describe(
        percentiles=[0.05, 0.5, 0.95])[["count", "mean", "5%", "50%", "95%"]]
    print((summ * np.array([1, 1e4, 1e4, 1e4, 1e4])).round(3).to_string()
          + "     (mean/pcts in basis points)")
    print("\nmedian |relative difference| overall: "
          f"{np.nanmedian(np.abs(out['rel_diff_fixed']))*1e4:.3f} bp")
    print("implied discount factor / known discount factor: "
          f"median {np.nanmedian(out['disc_implied']/out['disc_known']):.6f}")
    return out


# ------------------------------------------------------------------- rebuild
def black76_vec(F, K, T, sigma, r, is_call):
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    dfac = np.exp(-r * T)
    call = dfac * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = dfac * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def vega_vec(F, K, T, sigma, r):
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    return F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)


def newton_iv(price, F, K, T, r, is_call, n_iter=100, tol=1e-6):
    sigma = np.full_like(price, 0.25)
    for _ in range(n_iter):
        model = black76_vec(F, K, T, sigma, r, is_call)
        veg = np.maximum(vega_vec(F, K, T, sigma, r), 1e-8)
        sigma = np.clip(sigma - np.clip((model - price) / veg, -1.0, 1.0), 1e-4, 5.0)
    resid = black76_vec(F, K, T, sigma, r, is_call) - price
    conv = np.abs(resid) < tol
    return np.where(conv, sigma, np.nan), conv


def apply_filters(df, fwd_col):
    """Filters F1-F7 exactly as 04_filters.py, on an arbitrary forward column."""
    wf = []

    def row(name):
        wf.append({"filter": name, "rows_remaining": len(df),
                   "slices_remaining": df.drop_duplicates(["date", "expiry"]).shape[0]})

    df = df[df[fwd_col].notna() & (df[fwd_col] > 0)]
    df = df[(df["tau"] >= 7 / 365) & (df["tau"] <= 1.0)]
    row("F1_tau_range")
    df = df[(df["volume"] >= 10) & (df["oi"] > 0)]
    row("F2_liquidity_vol_ge_10")
    df = df.copy()
    df["k"] = np.log(df["strike"] / df[fwd_col])
    df = df[(df["k"] >= -0.35) & (df["k"] <= 0.25)]
    row("F3_moneyness_band")
    is_call = df["opt_type"] == "CE"
    intrinsic = np.where(is_call,
                         np.maximum(df[fwd_col] - df["strike"], 0.0),
                         np.maximum(df["strike"] - df[fwd_col], 0.0)) * np.exp(-df["r"] * df["tau"])
    df = df[df["settle_price"] > intrinsic + TICK]
    row("F4_above_intrinsic")
    is_call = df["opt_type"] == "CE"
    otm_call = is_call & (df["k"] > 0)
    otm_put = (~is_call) & (df["k"] < 0)
    df = df[otm_call | otm_put]
    row("F5_otm_only")

    df = df.sort_values(["date", "expiry", "strike"]).copy()
    g = df.groupby(["date", "expiry"], sort=False)
    Kp, Kn = g["strike"].shift(1), g["strike"].shift(-1)
    Pp, Pn = g["settle_price"].shift(1), g["settle_price"].shift(-1)
    has_nb = Kp.notna() & Kn.notna()
    lam = (Kn - df["strike"]) / (Kn - Kp)
    interp = lam * Pp + (1 - lam) * Pn
    conv_v = has_nb & (df["settle_price"] > interp + TICK)
    mono_c = has_nb & (df["opt_type"] == "CE") & (df["settle_price"] > Pp + TICK)
    mono_p = has_nb & (df["opt_type"] == "PE") & (df["settle_price"] < Pp - TICK)
    viol = conv_v | mono_c | mono_p
    n_viol = int(viol.sum())
    n_eligible = int(has_nb.sum())
    df = df[~viol]
    wf.append({"filter": f"F6_static_arbitrage (dropped {n_viol})",
               "rows_remaining": len(df),
               "slices_remaining": df.drop_duplicates(["date", "expiry"]).shape[0]})
    counts = df.groupby(["date", "expiry"])["strike"].transform("count")
    df = df[counts >= 8]
    row("F7_min_8_strikes_per_slice")
    return df, pd.DataFrame(wf), n_viol, n_eligible


def rebuild():
    fw = pd.read_csv(RES / "phase6_forward_multistrike.csv", parse_dates=["date", "expiry"])
    src = D2 / "nifty_with_forward.parquet"
    df = duckdb.connect().execute(f"SELECT * FROM read_parquet('{src}')").df()
    df = df[df["instrument"] == "OPT"].copy()
    df["date"] = pd.to_datetime(df["date"]); df["expiry"] = pd.to_datetime(df["expiry"])

    df = df.merge(fw[["date", "expiry", "fwd_fixed"]], on=["date", "expiry"], how="left")
    have = df["fwd_fixed"].notna()
    print(f"multi-strike forward available for {have.mean()*100:.1f}% of option rows; "
          f"remaining rows keep the Phase-1 forward")
    df["fwd_multi"] = df["fwd_fixed"].where(have, df["fwd"])

    for tag, col in [("baseline", "fwd"), ("multi", "fwd_multi")]:
        sub, wf, n_viol, n_elig = apply_filters(df.copy(), col)
        is_call = (sub["opt_type"] == "CE").to_numpy()
        sig, conv = newton_iv(sub["settle_price"].to_numpy(), sub[col].to_numpy(),
                              sub["strike"].to_numpy(), sub["tau"].to_numpy(),
                              sub["r"].to_numpy(), is_call)
        sub = sub.assign(iv=sig)
        kept = sub[sub["iv"].notna()].copy()
        kept = kept.rename(columns={col: "fwd_used"})
        print(f"\n--- {tag} forward ---")
        print(wf.to_string(index=False))
        print(f"F6 rejection rate among eligible interior points: "
              f"{100*n_viol/max(n_elig,1):.3f}%  ({n_viol}/{n_elig})")
        print(f"IV inversion: kept {len(kept):,} rows, "
              f"{kept.drop_duplicates(['date','expiry']).shape[0]:,} slices")
        if tag == "multi":
            con = duckdb.connect(); con.register("t", kept)
            con.execute(f"COPY t TO '{D4 / 'panel_fwdmulti.parquet'}' (FORMAT PARQUET)")
            print(f"wrote {D4 / 'panel_fwdmulti.parquet'}")


def compare():
    base = duckdb.connect().execute(
        f"SELECT date, expiry, strike, k, tau, iv FROM read_parquet('{D4 / 'panel.parquet'}')").df()
    alt = duckdb.connect().execute(
        f"SELECT date, expiry, strike, k, tau, iv FROM read_parquet('{D4 / 'panel_fwdmulti.parquet'}')").df()
    print(f"baseline panel: {len(base):,} rows, "
          f"{base.drop_duplicates(['date','expiry']).shape[0]:,} slices")
    print(f"multi-strike  : {len(alt):,} rows, "
          f"{alt.drop_duplicates(['date','expiry']).shape[0]:,} slices")
    m = base.merge(alt, on=["date", "expiry", "strike"], suffixes=("_b", "_m"))
    print(f"\ncommon quotes: {len(m):,}")
    dk = (m["k_m"] - m["k_b"]) * 1e4
    div = (m["iv_m"] - m["iv_b"]) * 100
    print(f"log-moneyness shift  : median {np.median(dk):+.3f} bp, "
          f"p5 {np.percentile(dk,5):+.3f}, p95 {np.percentile(dk,95):+.3f}")
    print(f"implied-vol shift    : median {np.median(div):+.4f} vol pts, "
          f"p5 {np.percentile(div,5):+.4f}, p95 {np.percentile(div,95):+.4f}")
    print(f"mean |IV shift|      : {np.mean(np.abs(div)):.4f} vol pts")
    pd.DataFrame({"metric": ["rows_base", "rows_multi", "slices_base", "slices_multi",
                             "common_quotes", "median_k_shift_bp", "median_iv_shift_volpts",
                             "mean_abs_iv_shift_volpts", "p95_abs_iv_shift_volpts"],
                  "value": [len(base), len(alt),
                            base.drop_duplicates(['date','expiry']).shape[0],
                            alt.drop_duplicates(['date','expiry']).shape[0],
                            len(m), float(np.median(dk)), float(np.median(div)),
                            float(np.mean(np.abs(div))),
                            float(np.percentile(np.abs(div), 95))]}
                 ).to_csv(RES / "phase6_forward_panel_compare.csv", index=False)


if __name__ == "__main__":
    mode = sys.argv[1]
    {"estimate": estimate_forwards, "rebuild": rebuild, "compare": compare}[mode]()
