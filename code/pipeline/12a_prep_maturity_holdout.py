"""
12a_prep_maturity_holdout.py -- Task C data prep: hide an interior maturity
band from training so that maturity-direction interpolation can be measured
rather than asserted.

MOTIVATION
----------
After the Phase-5 revision the manuscript's surviving claim is narrow and
specific: the architecture guarantees calendar monotonicity AND "supplies a
differentiable surface at arbitrary maturities", whereas the implemented SSVI
obtains ATM total variance only at that day's listed expiries and needs a
separate interpolation rule between them. Nothing in the paper measured that.
Every reported number -- Task A and Task B alike -- is scored at maturities the
model saw. A reviewer can fairly say the one remaining advantage is asserted.

DESIGN
------
Hold out a contiguous, INTERIOR band of maturities, BAND_LO..BAND_HI days to
expiry, from the training data entirely. The band is interior in the sense that
the overwhelming majority of days still carry at least one expiry shorter than
BAND_LO and one longer than BAND_HI, so recovering the band is genuine
interpolation in tau and not extrapolation off the end of the term structure.
Days that are not bracketed on both sides are recorded but excluded from the
headline score, and the count is reported.

[30, 60] days is chosen because (a) it is the one-to-two-month region a
practitioner actually needs to interpolate, (b) 1,771 slices fall in it, and
(c) 97.7% of the days holding such a slice are bracketed on both sides.

TWO SCORING SETS
----------------
c1_train_window : band slices whose DATE lies in the fold's training window.
                  The network has seen those dates (at other maturities) and
                  the daily-refit classical models see the same day's non-band
                  quotes. Information sets match, so this isolates maturity
                  interpolation and is the headline comparison.
c2_test_window  : band slices in the fold's test window. The static network
                  must additionally extrapolate in date, exactly the Task B
                  asymmetry the manuscript already documents. Reported as
                  secondary.

Usage: python3 12a_prep_maturity_holdout.py <fold_id>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb


def _find_project_root():
    for c in [Path(r"E:\Research") / "iv_surface",
              Path.home() / "mnt" / "Research" / "iv_surface",
              Path.home() / "mnt" / "iv_surface",
              Path(__file__).resolve().parent.parent]:
        if (c / "04_iv" / "panel.parquet").exists():
            return c
    raise SystemExit("could not locate iv_surface project root")


ROOT = _find_project_root()
BAND_LO_D, BAND_HI_D = 30, 60


def main():
    fold_id = int(sys.argv[1])
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    row = folds[folds["fold"] == fold_id]
    if len(row) != 1:
        raise SystemExit(f"fold {fold_id} not found")
    row = row.iloc[0]
    TRAIN_START, TRAIN_END = str(row["train_start"]), str(row["train_end"])
    TEST_START, TEST_END = str(row["test_start"]), str(row["test_end"])

    con = duckdb.connect()
    q = f"""
    SELECT p.date, p.expiry, p.strike, p.k, p.tau, p.iv,
           CASE WHEN h.is_heldout THEN 1 ELSE 0 END AS is_heldout
    FROM '{ROOT}/04_iv/panel.parquet' p
    LEFT JOIN '{ROOT}/05_splits/held_out_strikes.parquet' h
      ON p.date = h.date AND p.expiry = h.expiry AND p.strike = h.strike
    """
    panel = con.execute(q).df()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["dte"] = (panel["tau"] * 365).round().astype(int)
    panel["in_band"] = (panel["dte"] >= BAND_LO_D) & (panel["dte"] <= BAND_HI_D)

    # a day is "bracketed" if, among its NON-band slices, some expiry is shorter
    # than the band and some is longer -- i.e. the band really is interior
    nb = panel[~panel["in_band"]].drop_duplicates(["date", "expiry"])
    short = nb[nb["dte"] < BAND_LO_D].groupby("date").size()
    long_ = nb[nb["dte"] > BAND_HI_D].groupby("date").size()
    bracketed = set(short.index) & set(long_.index)
    panel["bracketed"] = panel["date"].isin(bracketed)

    in_train = (panel["date"] >= TRAIN_START) & (panel["date"] < TRAIN_END)
    in_test = (panel["date"] >= TEST_START) & (panel["date"] < TEST_END)

    # training data: train window, visible strikes, band REMOVED
    train_vis = panel[in_train & (panel["is_heldout"] == 0) & (~panel["in_band"])]
    # scoring sets: every quote of every band slice
    c1 = panel[in_train & panel["in_band"]]
    c2 = panel[in_test & panel["in_band"]]

    def pack(df, name, out, with_ids=False):
        out[f"{name}_k"] = df["k"].to_numpy(np.float64)
        out[f"{name}_tau"] = df["tau"].to_numpy(np.float64)
        out[f"{name}_iv"] = df["iv"].to_numpy(np.float64)
        out[f"{name}_w"] = (df["iv"].to_numpy(np.float64) ** 2) * df["tau"].to_numpy(np.float64)
        if with_ids:
            out[f"{name}_date"] = df["date"].values.astype("datetime64[ns]").astype(np.int64)
            out[f"{name}_expiry"] = pd.to_datetime(df["expiry"]).values.astype("datetime64[ns]").astype(np.int64)
            out[f"{name}_strike"] = df["strike"].to_numpy(np.float64)
            out[f"{name}_bracketed"] = df["bracketed"].to_numpy(bool)
            out[f"{name}_heldout"] = df["is_heldout"].to_numpy(np.int64)

    out = {}
    pack(train_vis, "train_vis", out)
    pack(c1, "c1", out, with_ids=True)
    pack(c2, "c2", out, with_ids=True)

    out_path = ROOT / "06_nn" / f"fold{fold_id}_matband_data.npz"
    np.savez(out_path, **out)

    print(f"fold {fold_id}: band = [{BAND_LO_D},{BAND_HI_D}] days")
    print(f"  train_vis rows (band removed): {len(train_vis):,}")
    print(f"  c1 (train-window band) rows: {len(c1):,} "
          f"({c1['bracketed'].mean()*100:.1f}% bracketed), "
          f"slices: {c1.drop_duplicates(['date','expiry']).shape[0]}")
    print(f"  c2 (test-window band) rows: {len(c2):,} "
          f"({c2['bracketed'].mean()*100 if len(c2) else 0:.1f}% bracketed), "
          f"slices: {c2.drop_duplicates(['date','expiry']).shape[0]}")
    if len(train_vis):
        print(f"  train tau range: {train_vis['tau'].min()*365:.0f}d .. {train_vis['tau'].max()*365:.0f}d "
              f"(gap {BAND_LO_D}-{BAND_HI_D}d)")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
