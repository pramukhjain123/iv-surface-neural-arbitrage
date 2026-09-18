"""
01_parse.py -- Phase 1.1. Parse raw NSE F&O daily bhavcopy zips into the
unified schema for NIFTY only: date, expiry, opt_type, strike, settle_price,
oi, volume, tau, instrument (FUT/OPT).

Handles TWO raw formats found in E:\Research\data\code downloaded data:
  - "OLD" (up to 2024-07-05): INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR,
    OPTION_TYP, SETTLE_PR, CONTRACTS, OPEN_INT, TIMESTAMP; dates as
    "DD-Mon-YYYY"; instrument codes FUTIDX/OPTIDX (index), FUTSTK/OPTSTK
    (single name).
  - "NEW" UDiFF format (from 2024-07-08, confirmed clean cutover, no
    overlap): TckrSymb, FinInstrmTp, XpryDt, StrkPric, OptnTp, SttlmPric,
    TtlTradgVol, OpnIntrst, TradDt; dates as ISO "YYYY-MM-DD"; instrument
    codes IDF/IDO (index), STF/STO (single name).
Both map to the same unified schema below.

Portable Windows/bridge root resolution, same pattern as 00_inventory.py.
Run one half-year at a time (`python3 01_parse.py 2020 H1` / `H2`) to stay
inside the device-bridge's per-call time limit; each half-year's output is
independent and never overwrites raw input.
"""
import sys
from pathlib import Path
import zipfile
import pandas as pd

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

RAW = ROOT / "data" / "code downloaded data"
OUT = ROOT / "iv_surface" / "02_parsed"
OUT.mkdir(parents=True, exist_ok=True)

INSTR_FUT = {"FUTIDX", "IDF"}
INSTR_OPT = {"OPTIDX", "IDO"}
OPT_MAP = {"CA": "CE", "PA": "PE", "C": "CE", "P": "PE", "CE": "CE", "PE": "PE"}

def normalize_old(df):
    df = df[df["SYMBOL"] == "NIFTY"]
    df = df[df["INSTRUMENT"].isin(INSTR_FUT | INSTR_OPT)].copy()
    if df.empty:
        return df
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y"),
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y"),
        "instrument": df["INSTRUMENT"].map(lambda x: "FUT" if x in INSTR_FUT else "OPT"),
        "opt_type": df["OPTION_TYP"].map(OPT_MAP).fillna(df["OPTION_TYP"]),
        "strike": df["STRIKE_PR"].astype(float),
        "settle_price": df["SETTLE_PR"].astype(float),
        "oi": df["OPEN_INT"].astype(float),
        "volume": df["CONTRACTS"].astype(float),
    })
    return out

def normalize_new(df):
    df = df[df["TckrSymb"] == "NIFTY"]
    df = df[df["FinInstrmTp"].isin(INSTR_FUT | INSTR_OPT)].copy()
    if df.empty:
        return df
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"], format="%Y-%m-%d"),
        "expiry": pd.to_datetime(df["XpryDt"], format="%Y-%m-%d"),
        "instrument": df["FinInstrmTp"].map(lambda x: "FUT" if x in INSTR_FUT else "OPT"),
        "opt_type": df["OptnTp"].map(OPT_MAP).fillna(df["OptnTp"]),
        "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
        "settle_price": pd.to_numeric(df["SttlmPric"], errors="coerce"),
        "oi": pd.to_numeric(df["OpnIntrst"], errors="coerce"),
        "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
    })
    return out

def parse_months(month_dirs):
    rows_all = []
    raw_row_count = 0
    n_old, n_new = 0, 0
    for mdir in month_dirs:
        for zpath in sorted(mdir.glob("*.zip")):
            try:
                zf = zipfile.ZipFile(zpath)
                name = zf.namelist()[0]
                df = pd.read_csv(zf.open(name))
            except Exception as e:
                print(f"  SKIP {zpath.name}: {e}")
                continue
            df.columns = [c.strip() for c in df.columns]
            raw_row_count += len(df)
            if "TradDt" in df.columns:
                out = normalize_new(df)
                n_new += 1
            else:
                out = normalize_old(df)
                n_old += 1
            if len(out):
                out["tau"] = (out["expiry"] - out["date"]).dt.days / 365.0
                rows_all.append(out)
    print(f"    files: {n_old} old-format, {n_new} new-format")
    if not rows_all:
        return pd.DataFrame(), raw_row_count
    panel = pd.concat(rows_all, ignore_index=True)
    return panel, raw_row_count

if __name__ == "__main__":
    year = int(sys.argv[1])
    half = sys.argv[2] if len(sys.argv) > 2 else "ALL"
    months = {"H1": range(1, 7), "H2": range(7, 13), "ALL": range(1, 13)}[half]
    month_dirs = [RAW / f"{m:02d}-{year}" for m in months]
    month_dirs = [d for d in month_dirs if d.exists()]
    panel, raw_count = parse_months(month_dirs)
    tag = f"{year}_{half}"
    panel.to_csv(OUT / f"panel_{tag}.csv", index=False)
    print(f"{tag} raw_rows_scanned={raw_count} nifty_rows_kept={len(panel)}")
    with open(OUT / "raw_row_counts.log", "a") as f:
        f.write(f"{tag},{raw_count},{len(panel)}\n")
