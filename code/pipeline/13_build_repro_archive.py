"""
13_build_repro_archive.py -- assemble the reproducibility package the Data
Availability Statement points at.

WHAT CAN AND CANNOT BE SHARED
-----------------------------
The raw input is NSE F&O bhavcopy, which lives outside this repository
(config.yaml: data_root) and is not redistributable. So the archive ships the
DERIVED panel -- the frozen 307,458-row (date, expiry, strike) table of implied
volatilities the paper is actually estimated on -- together with every script
that turns raw bhavcopy into it. A reader with their own bhavcopy licence can
rerun the pipeline from stage 0; a reader without one can still reproduce every
table and figure in the paper from the derived panel, which is the part that
matters for verification.

That is why there is no run_all.py that starts from raw data: it would fail for
almost everyone who downloads this. Instead the README documents two entry
points, one from raw data and one from the derived panel.
"""
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import importlib.util

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fengler", _HERE / "11a_fengler.py")
FG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FG)
ROOT = FG.ROOT
STAGE = ROOT / "repro" / "iv_surface_replication"

COLUMN_DOC = {
    "date": ("date", "Trading date (NSE settlement date) of the quote."),
    "expiry": ("date", "Option expiry date."),
    "instrument": ("string", "NSE instrument code, OPTIDX or IDO after the 2024-07-08 rename."),
    "opt_type": ("string", "CE (call) or PE (put). The panel is OTM-only: CE for k>0, PE for k<0."),
    "strike": ("float", "Strike price in index points."),
    "settle_price": ("float", "Daily settlement price from the raw bhavcopy zip (SETTLE_PR)."),
    "oi": ("float", "Open interest at close."),
    "volume": ("float", "Contracts traded that day. Panel requires >= 10 (filter F2)."),
    "tau": ("float", "Time to expiry in years, act/365. Panel requires 7/365 <= tau <= 1 (F1)."),
    "r": ("float", "Risk-free rate used for discounting, from 03_riskfree.py."),
    "fwd": ("float", "Forward price for that (date, expiry). See fwd_source."),
    "fwd_source": ("string", "How fwd was obtained: 'futures' (matched futures settlement), "
                             "'put_call_parity' (single-strike parity, both legs liquid), or "
                             "'put_call_parity_illiquid' (parity without the liquidity screen)."),
    "k": ("float", "Log forward moneyness log(strike/fwd). Panel requires -0.35 <= k <= 0.25 (F3)."),
    "iv": ("float", "Black-76 implied volatility, decimal (0.20 = 20%). Newton with a Brent fallback."),
}


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_data_dictionary():
    panel = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{ROOT / '04_iv' / 'panel.parquet'}')").df()
    rows = []
    for col in panel.columns:
        dtype, desc = COLUMN_DOC.get(col, (str(panel[col].dtype), ""))
        s = panel[col]
        rows.append({
            "column": col, "type": dtype, "description": desc,
            "n_missing": int(s.isna().sum()),
            "min": (str(s.min()) if not np.issubdtype(s.dtype, np.number) else float(s.min())),
            "max": (str(s.max()) if not np.issubdtype(s.dtype, np.number) else float(s.max())),
        })
    dd = pd.DataFrame(rows)
    return dd, panel


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE, ignore_errors=True)
    for sub in ("code", "data", "results"):
        (STAGE / sub).mkdir(parents=True, exist_ok=True)

    # code
    for f in sorted((ROOT / "src").glob("*.py")):
        shutil.copy2(f, STAGE / "code" / f.name)
    for f in ("config.yaml", "requirements.txt"):
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, STAGE / f)

    # derived data
    shutil.copy2(ROOT / "04_iv" / "panel.parquet", STAGE / "data" / "panel.parquet")
    for f in ("folds.csv", "held_out_strikes.parquet", "slice_fold_membership.parquet",
              "visible_k_range.parquet"):
        src = ROOT / "05_splits" / f
        if src.exists():
            shutil.copy2(src, STAGE / "data" / f)
    if (ROOT / "03_filtered" / "filter_waterfall.csv").exists():
        shutil.copy2(ROOT / "03_filtered" / "filter_waterfall.csv",
                     STAGE / "data" / "filter_waterfall.csv")

    dd, panel = build_data_dictionary()
    dd.to_csv(STAGE / "data" / "data_dictionary.csv", index=False)

    # result tables (CSV only -- the large intermediate parquet predictions stay local)
    n_res = 0
    for f in sorted((ROOT / "results").glob("*.csv")):
        if f.stat().st_size > 25 * 1024 * 1024 or "_chunk_" in f.name:
            continue
        shutil.copy2(f, STAGE / "results" / f.name)
        n_res += 1
    for f in sorted((ROOT / "results").glob("*.md")):
        shutil.copy2(f, STAGE / "results" / f.name)

    n_slices = panel.drop_duplicates(["date", "expiry"]).shape[0]
    readme = f"""# Replication package

Arbitrage-Constrained Neural Interpolation of the Implied Volatility Surface --
NIFTY 50 index options.

## What is here

    code/       every pipeline, model and evaluation script, in run order
    data/       the derived panel, the walk-forward fold definitions, the fixed
                Task A held-out-strike split, and a data dictionary
    results/    every results table and phase note the paper draws on
    config.yaml, requirements.txt

## What is NOT here, and why

The raw input is NSE F&O bhavcopy. It is licensed data and is not
redistributable, so it is not included. Everything downstream of it is.

`data/panel.parquet` is the frozen derived panel the paper is estimated on:
{len(panel):,} quotes across {n_slices:,} (date, expiry) slices,
{panel['date'].min().date()} to {panel['date'].max().date()}. Every table and
figure in the paper can be reproduced from this file alone.

## Reproducing

Two entry points.

**From the derived panel (no licence needed).** Start at stage 2. The fold
definitions and the held-out-strike split in `data/` are the exact ones used, so
results are reproducible to the seed:

    python code/04a_svi.py 0 3700          # classical baselines, chunked
    python code/04b_ssvi.py 0 1500
    python code/04c_thinplate.py
    python code/04d_cubic.py 0 3700
    python code/11a_fengler.py predict 0 1500      # Fengler constrained smoother
    python code/07a_prep_fold.py <fold>            # 0..17
    python code/07b_nn_train_v3_fold.py <fold> 1 18000 18000
    python code/07c_nn_eval_v3_allfolds.py
    python code/12a_prep_maturity_holdout.py <fold>
    python code/12b_train_v3_matband.py <fold> 1 18000 18000
    python code/12c_eval_maturity_holdout.py

**From raw bhavcopy (licence needed).** Point `config.yaml: paths.raw_bhavcopy`
at your own copy and run `code/00_inventory.py` through `code/05_invert.py` in
numeric order; that regenerates `data/panel.parquet`.

Scripts take explicit chunk arguments (`<start> <end>`) because they were run
inside a per-call time budget; the chunk boundaries do not affect results.

## Determinism

Seeds are fixed (`config.yaml: seed`, and `seed=42` for the held-out-strike
split). Network training is seeded per run; `results/phase4_seed_variance.csv`
reports the across-seed spread. The Fengler smoothing parameter is selected by
nested cross-validation on visible strikes only
(`code/11a_fengler.py lambda_cv`), never on the held-out set.

## Environment

Python 3.13 with `requirements.txt`, plus `jax` (0.4.38, CPU) for the network
and `quadprog` for the Fengler QP.

## File integrity

See `MANIFEST.sha256`.
"""
    (STAGE / "README.md").write_text(readme, encoding="utf-8")

    lines = []
    for f in sorted(STAGE.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.sha256":
            lines.append(f"{sha256(f)}  {f.relative_to(STAGE).as_posix()}")
    (STAGE / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zpath = ROOT / "repro" / "iv_surface_replication.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                z.write(f, f"iv_surface_replication/{f.relative_to(STAGE).as_posix()}")

    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"staged  : {STAGE}")
    print(f"files   : {sum(1 for f in STAGE.rglob('*') if f.is_file())}  "
          f"({total/1e6:.1f} MB uncompressed)")
    print(f"code    : {len(list((STAGE/'code').glob('*.py')))} scripts")
    print(f"results : {n_res} CSV tables + phase notes")
    print(f"archive : {zpath}  ({zpath.stat().st_size/1e6:.1f} MB)")
    print("\nNEXT STEP (needs your account): upload the zip to Zenodo, mint a DOI,")
    print("and put that DOI in the Data Availability Statement in paper/main.tex.")


if __name__ == "__main__":
    main()
