"""
04_filters.py -- Phase 1.4. Apply filters F1-F7 to the forward-attached
NIFTY option panel, in the plan's own order, recording the drop count at
each step (the filter waterfall table for the paper).

F1  tau in [7/365, 1.0]
F2  volume>0 and oi>0
F3  log-moneyness k=ln(K/F) in [-0.35, 0.25]
F4  settle_price > intrinsic + one tick (tick = 0.05, NSE's NIFTY-options
    tick size)
F5  OTM only (calls for k>0, puts for k<0) -- avoids double counting a
    strike from both legs
F6  static-arbitrage check on the surviving OTM price curve: within each
    (date, expiry) slice, sorted by strike, a price must not exceed the
    linear interpolation of its two strike-neighbours (butterfly /
    convexity) and must be monotone in the direction implied by moneyness
    (increasing away from the forward). Violating points dropped, counted
    and reported -- this is itself a finding per the plan, not swept away.
F7  >= 8 surviving strikes per (date, expiry) slice, else the whole slice
    is dropped (can't fit a smile to 7 points).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT
D2 = ROOT / "iv_surface" / "02_parsed"
D3 = ROOT / "iv_surface" / "03_filtered"
D3.mkdir(parents=True, exist_ok=True)

TICK = 0.05

def load():
    df = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D2 / 'nifty_with_forward.parquet'}')").df()
    if "r" not in df.columns:
        rf = pd.read_csv(D2 / "riskfree_daily.csv", parse_dates=["date"])
        df = df.merge(rf, on="date", how="left")
    return df

def waterfall_row(name, df, wf):
    wf.append({"filter": name, "rows_remaining": len(df),
               "slices_remaining": df.drop_duplicates(["date", "expiry"]).shape[0]})
    return wf

def main():
    df = load()
    wf = []
    wf = waterfall_row("F0_start (options with forward)", df, wf)

    df = df[df["fwd"].notna()]
    df = df[(df["tau"] >= 7/365) & (df["tau"] <= 1.0)]
    wf = waterfall_row("F1_tau_range", df, wf)

    df = df[(df["volume"] >= 10) & (df["oi"] > 0)]  # tightened from >0: see phase1_notes.md finding on stale thin-volume strikes
    wf = waterfall_row("F2_liquidity_vol_ge_10", df, wf)

    df["k"] = np.log(df["strike"] / df["fwd"])
    df = df[(df["k"] >= -0.35) & (df["k"] <= 0.25)]
    wf = waterfall_row("F3_moneyness_band", df, wf)

    is_call = df["opt_type"] == "CE"
    intrinsic = np.where(is_call,
                          np.maximum(df["fwd"] - df["strike"], 0.0),
                          np.maximum(df["strike"] - df["fwd"], 0.0)) * np.exp(-df["r"] * df["tau"])
    df = df[df["settle_price"] > intrinsic + TICK]
    wf = waterfall_row("F4_above_intrinsic", df, wf)

    otm_call = is_call & (df["k"] > 0)
    otm_put = (~is_call) & (df["k"] < 0)
    df = df[otm_call.reindex(df.index, fill_value=False) | otm_put.reindex(df.index, fill_value=False)]
    wf = waterfall_row("F5_otm_only", df, wf)

    # F6: static-arbitrage (convexity + monotonicity) on the OTM price curve
    df = df.sort_values(["date", "expiry", "strike"]).copy()
    g = df.groupby(["date", "expiry"], sort=False)
    df["_K_prev"] = g["strike"].shift(1)
    df["_K_next"] = g["strike"].shift(-1)
    df["_P_prev"] = g["settle_price"].shift(1)
    df["_P_next"] = g["settle_price"].shift(-1)
    has_neighbors = df["_K_prev"].notna() & df["_K_next"].notna()
    lam = (df["_K_next"] - df["strike"]) / (df["_K_next"] - df["_K_prev"])
    interp = lam * df["_P_prev"] + (1 - lam) * df["_P_next"]
    convex_violation = has_neighbors & (df["settle_price"] > interp + TICK)
    # monotonicity: OTM call price should fall as strike rises; OTM put price should fall as strike falls (rise as strike rises)
    mono_violation_call = has_neighbors & (df["opt_type"] == "CE") & (df["settle_price"] > df["_P_prev"] + TICK)
    mono_violation_put = has_neighbors & (df["opt_type"] == "PE") & (df["settle_price"] < df["_P_prev"] - TICK)
    violation = convex_violation | mono_violation_call | mono_violation_put
    n_viol = int(violation.sum())
    df = df[~violation].drop(columns=["_K_prev", "_K_next", "_P_prev", "_P_next"])
    wf.append({"filter": f"F6_static_arbitrage (violations dropped: {n_viol})",
               "rows_remaining": len(df),
               "slices_remaining": df.drop_duplicates(["date", "expiry"]).shape[0]})

    counts = df.groupby(["date", "expiry"])["strike"].transform("count")
    df = df[counts >= 8]
    wf = waterfall_row("F7_min_8_strikes_per_slice", df, wf)

    wf_df = pd.DataFrame(wf)
    wf_df["dropped_this_step"] = wf_df["rows_remaining"].shift(1) - wf_df["rows_remaining"]
    wf_df.to_csv(D3 / "filter_waterfall.csv", index=False)
    print(wf_df.to_string())

    con = duckdb.connect()
    con.register("df_tbl", df)
    con.execute(f"COPY df_tbl TO '{D3 / 'nifty_filtered.parquet'}' (FORMAT PARQUET)")
    print(f"\nfinal filtered rows: {len(df)}, slices: {df.drop_duplicates(['date','expiry']).shape[0]}")

if __name__ == "__main__":
    main()
