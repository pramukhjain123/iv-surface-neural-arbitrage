"""
00_inventory.py -- Phase 0 data inventory for the arbitrage-constrained
NIFTY IV surface paper. Scoped to the folders actually relevant to this
project (raw NSE F&O bhavcopy + the combined panels derived from it),
skipping sibling research projects that happen to live under E:\Research.

Portable: runs natively on Windows (E:\Research\...) or through the
Claude device bridge, where the same drive is mounted at
~/mnt/Research. Requires duckdb (`pip install duckdb`) -- pyarrow was
impractical to install in the bridge environment (very slow network
download there); duckdb reads the parquet files directly and was much
faster in practice, so the pipeline should standardize on it (or
polars) over pandas+pyarrow.
"""
from pathlib import Path
import pandas as pd
import duckdb

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

DATA_ROOT = ROOT / "data"
NORMALIZED = ROOT / "pramukh" / "data" / "combined_bhavcopy_normalized.parquet"
OUT = ROOT / "iv_surface" / "00_inventory"
OUT.mkdir(parents=True, exist_ok=True)

EXTS = {".csv", ".parquet", ".zip", ".txt", ".xlsx"}
RELEVANT_TOP = [DATA_ROOT / "code downloaded data"] + [
    DATA_ROOT / f for f in [
        "combined_bhavcopy.csv", "combined_bhavcopy_2026_clean.csv",
        "combined_bhavcopy_2026_clean.parquet", "combined_bhavcopy_v2.csv",
        "clean_options_fast.parquet", "bigfile.parquet",
    ] if (DATA_ROOT / f).exists()
]

def display_path(p: Path) -> str:
    # Always record the canonical Windows-style path in the deliverable,
    # regardless of which root we actually walked.
    return r"E:\Research" + str(p).replace(str(ROOT), "").replace("/", "\\")

rows = []
for base in RELEVANT_TOP:
    if base.is_file():
        rows.append({"path": display_path(base), "suffix": base.suffix.lower(),
                     "mb": round(base.stat().st_size / 1e6, 2)})
    else:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                rows.append({"path": display_path(p), "suffix": p.suffix.lower(),
                             "mb": round(p.stat().st_size / 1e6, 2)})

if NORMALIZED.exists():
    rows.append({"path": display_path(NORMALIZED), "suffix": ".parquet",
                 "mb": round(NORMALIZED.stat().st_size / 1e6, 2)})

inv = pd.DataFrame(rows).sort_values("mb", ascending=False)
inv.to_csv(OUT / "file_inventory.csv", index=False)
print(inv.groupby("suffix")["mb"].agg(["count", "sum"]))
print(f"\n{len(inv)} files, {inv.mb.sum():.0f} MB total")

# --- date range / instrument-code check on the combined panel ---
con = duckdb.connect()
f = str(DATA_ROOT / "combined_bhavcopy_2026_clean.parquet")
print(con.execute(f"SELECT MIN(Date), MAX(Date), COUNT(DISTINCT Date) FROM read_parquet('{f}')").fetchall())
print(con.execute(f"""
    SELECT Instrument, COUNT(*), MIN(Date), MAX(Date)
    FROM read_parquet('{f}') WHERE Symbol='NIFTY' GROUP BY 1
""").df())
