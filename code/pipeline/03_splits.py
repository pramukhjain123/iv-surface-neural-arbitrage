"""
03_splits.py -- Phase 2.1. Expanding-origin walk-forward folds, purged at
the boundary:
    fold k: train [t0, t0+(18+3k)m)  val [+.., +3m)  test [+.., +3m)
stepping 3 months per fold, t0 = panel's first date. Also pre-selects a
fixed-seed random 20% of strikes per (date,expiry) slice as the Task A
(interpolation) held-out set; Task B (extrapolation) is simply "this
(date,expiry) slice's date falls in fold k's test window" -- no extra
selection needed, so it isn't materialized as a separate column here.

Output:
  - folds.csv: one row per fold (train/val/test start/end dates)
  - slice_fold_membership.parquet: (date, expiry, fold, role) rows -- a
    slice can appear multiple times if it's "train" in fold k and "test"
    in a later fold k' (expanding window), which is intentional.
  - held_out_strikes.parquet: (date, expiry, strike, is_heldout) for the
    Task A 20% split, seed=42, fixed once and reused by every method so
    the comparison is apples-to-apples.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT
D4 = ROOT / "iv_surface" / "04_iv"
D5 = ROOT / "iv_surface" / "05_splits"
D5.mkdir(parents=True, exist_ok=True)

def build_folds(t0, t_end):
    folds = []
    k = 0
    while True:
        train_start = t0
        train_end = t0 + pd.DateOffset(months=18 + 3 * k)
        val_end = train_end + pd.DateOffset(months=3)
        test_end = val_end + pd.DateOffset(months=3)
        if train_end >= t_end:
            break
        partial = test_end > t_end
        folds.append({
            "fold": k,
            "train_start": train_start, "train_end": train_end,
            "val_start": train_end, "val_end": val_end,
            "test_start": val_end, "test_end": min(test_end, t_end + pd.Timedelta(days=1)),
            "partial_test": partial,
        })
        k += 1
        if val_end >= t_end:
            break
    df = pd.DataFrame(folds)
    df = df[df["test_start"] < df["test_end"]].reset_index(drop=True)
    return df

def main():
    panel = duckdb.connect().execute(
        f"SELECT DISTINCT date, expiry FROM read_parquet('{D4 / 'panel.parquet'}')").df()
    panel["date"] = pd.to_datetime(panel["date"])
    t0, t_end = panel["date"].min(), panel["date"].max()
    print(f"sample: {t0.date()} to {t_end.date()}")

    folds = build_folds(t0, t_end)
    folds.to_csv(D5 / "folds.csv", index=False)
    print(folds.to_string())

    rows = []
    for f in folds.itertuples():
        d = panel["date"]
        tr = panel[(d >= f.train_start) & (d < f.train_end)].copy(); tr["fold"] = f.fold; tr["role"] = "train"
        va = panel[(d >= f.val_start) & (d < f.val_end)].copy(); va["fold"] = f.fold; va["role"] = "val"
        te = panel[(d >= f.test_start) & (d < f.test_end)].copy(); te["fold"] = f.fold; te["role"] = "test"
        rows += [tr, va, te]
    membership = pd.concat(rows, ignore_index=True)

    con = duckdb.connect()
    con.register("mem", membership)
    con.execute(f"COPY mem TO '{D5 / 'slice_fold_membership.parquet'}' (FORMAT PARQUET)")
    print(f"\nslice-fold membership rows: {len(membership)} "
          f"(a slice can appear in >1 fold's train set as the window expands)")

    # Task A: fixed 20% held-out strikes per (date,expiry) slice, seed=42
    full = duckdb.connect().execute(
        f"SELECT date, expiry, strike FROM read_parquet('{D4 / 'panel.parquet'}')").df()
    rng = np.random.default_rng(42)
    full["u"] = rng.random(len(full))
    frac = full.groupby(["date", "expiry"])["u"].transform(
        lambda s: s.rank(method="first") / len(s))
    full["is_heldout"] = frac <= 0.20
    held = full[["date", "expiry", "strike", "is_heldout"]]
    con2 = duckdb.connect()
    con2.register("held", held)
    con2.execute(f"COPY held TO '{D5 / 'held_out_strikes.parquet'}' (FORMAT PARQUET)")
    print(f"Task A held-out: {held['is_heldout'].sum()} / {len(held)} strikes "
          f"({100*held['is_heldout'].mean():.1f}%)")

if __name__ == "__main__":
    main()
