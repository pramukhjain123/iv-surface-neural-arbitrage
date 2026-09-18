"""
02_attach_forward.py -- Phase 1.2. Attach a matched-futures forward price
to every NIFTY option row, keyed on (date, expiry).

Preference order:
  1. Direct match: a FUTIDX/IDF settlement price exists for that exact
     (date, expiry) -- true for the handful of monthly expiries that also
     trade as futures.
  2. Put-call parity fallback: at the strike within that (date, expiry)
     slice where |C - P| is smallest (closest to at-the-money), solve
       F = K + (C - P) * exp(r * tau)
     Restricted first to strikes where BOTH legs have volume>0 and OI>0
     (a stale, untraded settle price on either leg makes the straddle
     unreliable); if no such liquid pair exists for that slice, falls
     back to the unrestricted argmin so every slice still gets a value,
     flagged separately (`fwd_source='put_call_parity_illiquid'`) so
     Phase 1.4 filtering can drop or scrutinize those explicitly.

Rows where neither is possible get fwd = NaN and are dropped downstream.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT
D = ROOT / "iv_surface" / "02_parsed"

def pick_atm(wide, rf, tau_lookup, label):
    """wide must already be restricted to the desired liquidity tier."""
    w = wide.dropna(subset=["CE", "PE"]).copy()
    if w.empty:
        return pd.DataFrame(columns=["date", "expiry", f"F_{label}", f"{label}_strike"])
    w["absdiff"] = (w["CE"] - w["PE"]).abs()
    idx = w.groupby(["date", "expiry"])["absdiff"].idxmin()
    atm = w.loc[idx].copy()
    atm = atm.merge(rf, on="date", how="left")
    atm = atm.merge(tau_lookup, on=["date", "expiry"], how="left")
    atm[f"F_{label}"] = atm["strike"] + (atm["CE"] - atm["PE"]) * np.exp(atm["r"] * atm["tau"])
    return atm.rename(columns={"strike": f"{label}_strike"})[
        ["date", "expiry", f"F_{label}", f"{label}_strike"]]

def main():
    df = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D / 'nifty_parsed.parquet'}')").df()
    rf = pd.read_csv(D / "riskfree_daily.csv", parse_dates=["date"])
    df = df.merge(rf, on="date", how="left")

    fut = (df[df["instrument"] == "FUT"]
           .dropna(subset=["settle_price"])
           .groupby(["date", "expiry"], as_index=False)["settle_price"].first()
           .rename(columns={"settle_price": "F_direct"}))

    opt = df[df["instrument"] == "OPT"].copy()
    tau_lookup = opt[["date", "expiry", "tau"]].drop_duplicates(["date", "expiry"])

    liquid = opt[(opt["volume"] > 0) & (opt["oi"] > 0)]
    wide_liquid = liquid.pivot_table(index=["date", "expiry", "strike"], columns="opt_type",
                                      values="settle_price", aggfunc="first").reset_index()
    for c in ("CE", "PE"):
        if c not in wide_liquid.columns:
            wide_liquid[c] = np.nan
    atm_liquid = pick_atm(wide_liquid, rf, tau_lookup, "pcp")

    wide_any = opt.pivot_table(index=["date", "expiry", "strike"], columns="opt_type",
                                values="settle_price", aggfunc="first").reset_index()
    for c in ("CE", "PE"):
        if c not in wide_any.columns:
            wide_any[c] = np.nan
    atm_any = pick_atm(wide_any, rf, tau_lookup, "pcp_illiq")

    fwd = fut.merge(atm_liquid, on=["date", "expiry"], how="outer")
    fwd = fwd.merge(atm_any, on=["date", "expiry"], how="outer")

    fwd["fwd"] = fwd["F_direct"]
    fwd["fwd_source"] = np.where(fwd["F_direct"].notna(), "futures", None)

    need = fwd["fwd"].isna() & fwd["F_pcp"].notna()
    fwd.loc[need, "fwd"] = fwd.loc[need, "F_pcp"]
    fwd.loc[need, "fwd_source"] = "put_call_parity"

    need2 = fwd["fwd"].isna() & fwd["F_pcp_illiq"].notna()
    fwd.loc[need2, "fwd"] = fwd.loc[need2, "F_pcp_illiq"]
    fwd.loc[need2, "fwd_source"] = "put_call_parity_illiquid"

    fwd["fwd_source"] = fwd["fwd_source"].fillna("none")

    out = opt.merge(fwd[["date", "expiry", "fwd", "fwd_source"]],
                     on=["date", "expiry"], how="left")

    n_total = len(out)
    print(f"total option rows: {n_total}")
    for src in ["futures", "put_call_parity", "put_call_parity_illiquid", "none"]:
        n = (out["fwd_source"] == src).sum()
        print(f"  {src}: {n} ({100*n/n_total:.1f}%)")

    n_dexp = fwd.shape[0]
    print(f"\ndistinct (date,expiry) slices: {n_dexp}")
    print(fwd["fwd_source"].value_counts())

    con = duckdb.connect()
    con.register("out_tbl", out)
    con.execute(f"COPY out_tbl TO '{D / 'nifty_with_forward.parquet'}' (FORMAT PARQUET)")

if __name__ == "__main__":
    main()
