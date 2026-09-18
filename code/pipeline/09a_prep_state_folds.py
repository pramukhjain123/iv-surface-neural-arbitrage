"""Prepare walk-forward data for the state-conditioned hard-constraint network.

The daily state is estimated only from that day's visible 80% strike sample.  For
each expiry we take the visible quote closest to k=0, regress its ATM total
variance on maturity, and retain the fitted 30-day level and slope.  These two
numbers are constant within a date and therefore do not alter the proof that the
surface is non-decreasing in tau for a fixed market state.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(r"E:\Research\iv_surface")
TAU_REF = 30.0 / 365.0


def add_state(frame):
    visible = frame[~frame["is_heldout"]].copy()
    visible["w"] = visible["iv"] ** 2 * visible["tau"]
    atm = visible.loc[visible.groupby(["date", "expiry"])["k"].apply(lambda s: s.abs().idxmin())]

    rows = []
    for date, day in atm.groupby("date"):
        x = day["tau"].to_numpy(float)
        y = day["w"].to_numpy(float)
        if len(np.unique(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 0.0, float(y[0])
        rows.append({"date": date, "state_atm30": intercept + slope * TAU_REF,
                     "state_term_slope": slope})
    state = pd.DataFrame(rows)
    return frame.merge(state, on="date", how="inner")


def pack(frame):
    return {
        "k": frame["k"].to_numpy(float),
        "tau": frame["tau"].to_numpy(float),
        "iv": frame["iv"].to_numpy(float),
        "w": (frame["iv"] ** 2 * frame["tau"]).to_numpy(float),
        "state_atm30": frame["state_atm30"].to_numpy(float),
        "state_term_slope": frame["state_term_slope"].to_numpy(float),
    }


def main(fold_id):
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    row = folds.loc[folds["fold"] == fold_id].iloc[0]
    con = duckdb.connect()
    panel = con.execute(f"""
        SELECT p.date, p.expiry, p.strike, p.k, p.tau, p.iv,
               coalesce(h.is_heldout, false) AS is_heldout
        FROM '{ROOT}/04_iv/panel.parquet' p
        LEFT JOIN '{ROOT}/05_splits/held_out_strikes.parquet' h
          USING (date, expiry, strike)
        WHERE (p.date >= '{row.train_start}' AND p.date < '{row.train_end}')
           OR (p.date >= '{row.test_start}' AND p.date < '{row.test_end}')
    """).df()
    train = add_state(panel[(panel["date"] >= pd.Timestamp(row.train_start)) &
                            (panel["date"] < pd.Timestamp(row.train_end))])
    test = add_state(panel[(panel["date"] >= pd.Timestamp(row.test_start)) &
                           (panel["date"] < pd.Timestamp(row.test_end))])

    sets = {
        "train_vis": train[~train["is_heldout"]],
        "train_ho": train[train["is_heldout"]],
        "test_ho": test[test["is_heldout"]],
        "test_all": test,
    }
    out = {}
    for name, frame in sets.items():
        for key, value in pack(frame).items():
            out[f"{name}_{key}"] = value
    path = ROOT / "06_nn" / f"fold{fold_id}_state_data.npz"
    np.savez(path, **out)
    print(f"fold {fold_id}: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))


if __name__ == "__main__":
    main(int(sys.argv[1]))
