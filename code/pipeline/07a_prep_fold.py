"""
Phase 4 data prep: extract any of the 18 walk-forward folds (train visible / train
held-out / test) into a compact .npz for fast repeated loading by the NN training script.

Generalizes 06a_prep_fold16.py (which hardcoded fold 16) to take a fold id and read its
train/test window from 05_splits/folds.csv, so the same script covers all 18 folds.

Usage: python3 07a_prep_fold.py <fold_id>
"""
import sys
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT


def main():
    fold_id = int(sys.argv[1])

    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    row = folds[folds["fold"] == fold_id]
    if len(row) != 1:
        raise SystemExit(f"fold {fold_id} not found in folds.csv (have {folds['fold'].tolist()})")
    row = row.iloc[0]
    TRAIN_START, TRAIN_END = str(row["train_start"]), str(row["train_end"])
    TEST_START, TEST_END = str(row["test_start"]), str(row["test_end"])

    con = duckdb.connect()

    q = f"""
    SELECT p.date, p.expiry, p.strike, p.opt_type, p.k, p.tau, p.iv,
           CASE WHEN h.is_heldout THEN 1 ELSE 0 END AS is_heldout
    FROM '{ROOT}/04_iv/panel.parquet' p
    LEFT JOIN '{ROOT}/05_splits/held_out_strikes.parquet' h
      ON p.date = h.date AND p.expiry = h.expiry AND p.strike = h.strike
    WHERE p.date >= '{TRAIN_START}' AND p.date < '{TRAIN_END}'
    """
    train = con.execute(q).df()
    train_vis = train[train["is_heldout"] == 0].reset_index(drop=True)
    train_ho = train[train["is_heldout"] == 1].reset_index(drop=True)

    q2 = f"""
    SELECT date, expiry, strike, opt_type, k, tau, iv
    FROM '{ROOT}/04_iv/panel.parquet'
    WHERE date >= '{TEST_START}' AND date < '{TEST_END}'
    """
    test = con.execute(q2).df()

    def pack(df):
        return dict(
            k=df["k"].to_numpy(dtype=np.float64),
            tau=df["tau"].to_numpy(dtype=np.float64),
            iv=df["iv"].to_numpy(dtype=np.float64),
            w=(df["iv"].to_numpy(dtype=np.float64) ** 2) * df["tau"].to_numpy(dtype=np.float64),
        )

    out = {}
    for name, df in [("train_vis", train_vis), ("train_ho", train_ho), ("test", test)]:
        d = pack(df)
        for k, v in d.items():
            out[f"{name}_{k}"] = v

    out_path = ROOT / "06_nn" / f"fold{fold_id}_data.npz"
    np.savez(out_path, **out)

    print(f"fold {fold_id}: train [{TRAIN_START}, {TRAIN_END}) test [{TEST_START}, {TEST_END})")
    print("train_vis rows:", len(train_vis))
    print("train_ho rows:", len(train_ho))
    print("test rows:", len(test))
    if len(train_vis) > 0:
        print("k range (train_vis):", train_vis["k"].min(), train_vis["k"].max())
        print("tau range (train_vis):", train_vis["tau"].min(), train_vis["tau"].max())
    print("saved:", out_path)


if __name__ == "__main__":
    main()
