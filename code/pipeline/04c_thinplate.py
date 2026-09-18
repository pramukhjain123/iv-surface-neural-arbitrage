"""
04c_thinplate.py -- Phase 2.2, baseline 3/4: thin-plate spline on (k, tau),
fit per DAY, pooling all of that day's expiries together (a genuine
surface fit, not per-slice). Evaluated on the fixed Task-A held-out
strikes; because we only ever hold out strikes (never whole expiries),
every held-out point's tau always matches some visible point's tau on the
same day, so no extrapolation occurs in the tau dimension -- only k can
fall outside the visible range (see 05_splits/visible_k_range.parquet).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.interpolate import RBFInterpolator

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
    days = panel["date"].drop_duplicates().sort_values().reset_index(drop=True).iloc[start:end]

    rows = []
    n_fit, n_skip = 0, 0
    for d in days:
        g = panel[panel["date"] == d]
        vis = g[~g["is_heldout"]]
        ho = g[g["is_heldout"]]
        if len(vis) < 10 or len(ho) == 0:
            n_skip += 1
            continue
        X = vis[["k", "tau"]].to_numpy()
        y = vis["iv"].to_numpy()
        try:
            rbf = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=1e-6)
        except Exception:
            n_skip += 1
            continue
        Xh = ho[["k", "tau"]].to_numpy()
        iv_pred = rbf(Xh)
        n_fit += 1
        for expiry, strike, iv_p, iv_a in zip(ho["expiry"], ho["strike"], iv_pred, ho["iv"]):
            rows.append({"method": "ThinPlate", "date": d, "expiry": expiry,
                         "strike": strike, "iv_actual": iv_a, "iv_pred": iv_p})
    out = pd.DataFrame(rows)
    tag = f"{start}_{end}"
    out.to_csv(RES / f"thinplate_chunk_{tag}.csv", index=False)
    print(f"chunk {tag}: days_fit={n_fit} skipped={n_skip} pred_rows={len(out)}")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
