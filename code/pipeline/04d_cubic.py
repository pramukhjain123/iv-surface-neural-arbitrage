"""
04d_cubic.py -- Phase 2.2, baseline 4/4a: cubic spline in k, per
(date, expiry) slice (the "linear in tau across slices" cross-expiry
extension isn't exercised by Task A, since held-out points share their
slice's own tau -- documented in phase2_notes.md).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.interpolate import CubicSpline

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT
D4, D5, RES = (ROOT / "iv_surface" / d for d in ("04_iv", "05_splits", "results"))

def main(start, end):
    panel = duckdb.connect().execute(
        f"SELECT date, expiry, strike, k, tau, iv FROM read_parquet('{D4 / 'panel.parquet'}')").df()
    held = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D5 / 'held_out_strikes.parquet'}')").df()
    panel = panel.merge(held, on=["date", "expiry", "strike"], how="left")
    slices = panel.drop_duplicates(["date", "expiry"])[["date", "expiry"]].reset_index(drop=True).iloc[start:end]

    rows = []
    n_fit, n_skip = 0, 0
    for r in slices.itertuples():
        g = panel[(panel["date"] == r.date) & (panel["expiry"] == r.expiry)].sort_values("k")
        vis = g[~g["is_heldout"]]
        ho = g[g["is_heldout"]]
        if len(vis) < 4 or len(ho) == 0:
            n_skip += 1
            continue
        cs = CubicSpline(vis["k"].to_numpy(), vis["iv"].to_numpy(), extrapolate=True)
        iv_pred = cs(ho["k"].to_numpy())
        n_fit += 1
        for strike, iv_p, iv_a in zip(ho["strike"], iv_pred, ho["iv"]):
            rows.append({"method": "CubicSpline", "date": r.date, "expiry": r.expiry,
                         "strike": strike, "iv_actual": iv_a, "iv_pred": iv_p})
    out = pd.DataFrame(rows)
    tag = f"{start}_{end}"
    out.to_csv(RES / f"cubic_chunk_{tag}.csv", index=False)
    print(f"chunk {tag}: fit={n_fit} skipped={n_skip} pred_rows={len(out)}")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
