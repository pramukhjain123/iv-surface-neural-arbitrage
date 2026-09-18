"""
Phase 3 data prep: extract fold 16 (train visible / train held-out / test) into a
compact .npz for fast repeated loading by the NN training script.

Fold 16: train 2020-01-01..2025-07-01, val 2025-07-01..2025-10-01, test 2025-10-01..2026-01-01
(the last full, non-partial fold; val window is not used by the NN itself here -- Task A/B
mirror the Phase 2 baseline evaluation, which uses train-window held-out strikes and the
test window respectively).
"""
import duckdb
import numpy as np
from pathlib import Path

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

con = duckdb.connect()

TRAIN_START, TRAIN_END = "2020-01-01", "2025-07-01"
TEST_START, TEST_END = "2025-10-01", "2026-01-01"

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

np.savez(ROOT / "06_nn" / "fold16_data.npz", **out)

print("train_vis rows:", len(train_vis))
print("train_ho rows:", len(train_ho))
print("test rows:", len(test))
print("k range (train_vis):", train_vis["k"].min(), train_vis["k"].max())
print("tau range (train_vis):", train_vis["tau"].min(), train_vis["tau"].max())
print("w range (train_vis): ", out["train_vis_w"].min(), out["train_vis_w"].max())
