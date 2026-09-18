"""
15_build_github_release.py -- assemble a minimal GitHub-ready release folder:
code, derived data, results, and the single final compiled paper. Nothing else.

This extends 13_build_repro_archive.py (which built the Zenodo replication
package before the Sep 2026 revision) to also include:
  - the isolating hard-vs-soft ablation (ablation/*.py, ablation/results/*.csv,
    ablation/ABLATION_RESULTS.md)
  - the full 18-fold x 3-seed seed-variance check (seed_full/*.py,
    seed_full/SEED_VARIANCE_FULL.md)
  - the Tier-1 figure-generation scripts and their input data
    (gen_*.py at the project root, tier1_writing_*.{csv,npz})
  - every figure actually used in the paper
  - the final compiled paper PDF only (no LaTeX source, no template files)

WHAT CAN AND CANNOT BE SHARED
-----------------------------
The raw input is NSE F&O bhavcopy, which lives outside this repository
(config.yaml: data_root) and is not redistributable. So the release ships the
DERIVED panel -- the frozen 307,458-row (date, expiry, strike) table of implied
volatilities the paper is actually estimated on -- together with every script
that turns raw bhavcopy into it, and every script downstream of it. A reader
with their own bhavcopy licence can rerun the pipeline from stage 0; a reader
without one can still reproduce every table and figure in the paper from the
derived panel.

Trained model checkpoints (06_nn/) are NOT included -- they are large binary
weights, not source data, and training is deterministic given the fixed seeds,
so anyone who wants them can regenerate them by rerunning the training scripts.

This release is intentionally minimal: code/, data/, results/, paper/ (the
single final PDF), plus a README and requirements.txt. No config.yaml,
.gitignore, or MANIFEST.sha256 -- and no LaTeX source in paper/, just the
compiled PDF.
"""
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

ROOT = Path(r"E:\Research\iv_surface")
STAGE = ROOT / "github_release"

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
    return pd.DataFrame(rows), panel


def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE, ignore_errors=True)
    for sub in ("code/pipeline", "code/ablation", "code/seed_variance", "code/figures",
                "data/figure_inputs", "results/ablation", "results/figures", "paper"):
        (STAGE / sub).mkdir(parents=True, exist_ok=True)

    # ---- code: main pipeline (00_.. through 14_, in src/) ----
    for f in sorted((ROOT / "src").glob("*.py")):
        shutil.copy2(f, STAGE / "code" / "pipeline" / f.name)

    # ---- code: isolating hard-vs-soft ablation ----
    for f in sorted((ROOT / "ablation").glob("*.py")) + sorted((ROOT / "ablation").glob("*.sh")):
        shutil.copy2(f, STAGE / "code" / "ablation" / f.name)

    # ---- code: full 18-fold x 3-seed seed-variance check ----
    for f in sorted((ROOT / "seed_full").glob("*.py")):
        shutil.copy2(f, STAGE / "code" / "seed_variance" / f.name)

    # ---- code: Tier-1 figure generation (project root gen_*.py) ----
    for f in sorted(ROOT.glob("gen_*.py")):
        shutil.copy2(f, STAGE / "code" / "figures" / f.name)

    # ---- top-level project files: requirements.txt only ----
    copy_if_exists(ROOT / "requirements.txt", STAGE / "requirements.txt")

    # ---- derived data: the frozen panel + splits ----
    shutil.copy2(ROOT / "04_iv" / "panel.parquet", STAGE / "data" / "panel.parquet")
    for f in ("folds.csv", "held_out_strikes.parquet", "slice_fold_membership.parquet",
              "visible_k_range.parquet"):
        copy_if_exists(ROOT / "05_splits" / f, STAGE / "data" / f)
    copy_if_exists(ROOT / "03_filtered" / "filter_waterfall.csv", STAGE / "data" / "filter_waterfall.csv")

    dd, panel = build_data_dictionary()
    dd.to_csv(STAGE / "data" / "data_dictionary.csv", index=False)

    # ---- data: inputs the Tier-1 figure scripts consume ----
    for name in ("tier1_writing_bucket_export.csv", "tier1_writing_gk_data.csv",
                 "tier1_writing_gk_obs.csv", "tier1_writing_smile_data.csv",
                 "tier1_writing_surface_data.npz"):
        copy_if_exists(ROOT / name, STAGE / "data" / "figure_inputs" / name)

    # ---- results: every main results table + phase note (CSV/MD, <25MB, no chunks) ----
    n_res = 0
    for f in sorted((ROOT / "results").glob("*.csv")):
        if f.stat().st_size > 25 * 1024 * 1024 or "_chunk_" in f.name:
            continue
        shutil.copy2(f, STAGE / "results" / f.name)
        n_res += 1
    for f in sorted((ROOT / "results").glob("*.md")):
        shutil.copy2(f, STAGE / "results" / f.name)

    # ---- results: ablation experiment outputs ----
    copy_if_exists(ROOT / "ablation" / "ABLATION_RESULTS.md", STAGE / "results" / "ablation" / "ABLATION_RESULTS.md")
    n_abl = 0
    for f in sorted((ROOT / "ablation" / "results").glob("*.csv")):
        shutil.copy2(f, STAGE / "results" / "ablation" / f.name)
        n_abl += 1

    # ---- results: full seed-variance report ----
    copy_if_exists(ROOT / "seed_full" / "SEED_VARIANCE_FULL.md", STAGE / "results" / "SEED_VARIANCE_FULL.md")

    # ---- results: every figure actually used in the paper ----
    fig_names = ["phase2_baseline_comparison.png", "phase3_v3_tradeoff.png"]
    n_fig = 0
    for name in fig_names:
        if copy_if_exists(ROOT / "results" / name, STAGE / "results" / "figures" / name):
            n_fig += 1
    for f in sorted((ROOT / "tier1_writing" / "figures").glob("*.png")):
        shutil.copy2(f, STAGE / "results" / "figures" / f.name)
        n_fig += 1

    # ---- paper: the single final compiled PDF, nothing else ----
    copy_if_exists(ROOT / "paper" / "main.pdf", STAGE / "paper" / "paper.pdf")

    n_slices = panel.drop_duplicates(["date", "expiry"]).shape[0]
    readme = f"""# Arbitrage-Constrained Neural Interpolation of the Implied Volatility Surface

Code, derived data, and results for the working paper *"Arbitrage-Constrained
Neural Interpolation of the Implied Volatility Surface: Evidence from NSE
Index Options"* (IEEE Access, in preparation), evaluated on 6.1 years of
NIFTY 50 index options. The final paper is included as a single PDF.

## What is here

    code/
      pipeline/       stages 00-14: raw-data parsing through the neural
                       models, classical baselines, Fengler's constrained
                       smoother, the maturity-holdout (Task C) experiment,
                       and the repro/verification scripts, in run order
      ablation/        isolating hard-vs-soft-constraint ablation (arms A-E)
      seed_variance/   the full 18-fold x 3-seed seed-variance check
      figures/         generators for the paper's 3D surface / smile-overlay /
                       g(k)-violation / error-heatmap figures
    data/
      panel.parquet            the frozen derived panel (see below)
      folds.csv, held_out_strikes.parquet, slice_fold_membership.parquet,
      visible_k_range.parquet  the exact walk-forward fold and Task A
                                held-out-strike definitions used
      filter_waterfall.csv     raw-panel -> frozen-panel filter counts
      data_dictionary.csv      column-by-column description of panel.parquet
      figure_inputs/           the small extracted CSV/NPZ inputs the
                                code/figures/ scripts consume (no retraining
                                needed to reproduce the figures)
    results/
      *.csv, *.md              every results table and phase note the paper
                                draws on
      ablation/                the isolating-ablation results (matches
                                Results \u00a7"Isolating the constraint from the
                                decomposition") and ABLATION_RESULTS.md, its
                                full write-up
      SEED_VARIANCE_FULL.md    the full-scale seed-variance write-up
      figures/                 every figure used in the paper, already
                                rendered
    paper/
      paper.pdf                the final compiled paper (only, no LaTeX source)
    requirements.txt

## What is NOT here, and why

- **Raw NSE bhavcopy.** Licensed data, not redistributable. Everything
  downstream of it is included.
- **Trained model checkpoints** (the `06_nn/` run directories). These are
  large binary weights, not source data. Training is deterministic given the
  fixed seeds, so rerunning `code/pipeline/07b_nn_train_v3_fold.py` (etc.)
  regenerates them exactly.
- **The paper's LaTeX source.** This release ships only the final compiled
  PDF (`paper/paper.pdf`), not `main.tex`/`sections/`/`refs.bib` or the IEEE
  Access template files, to keep the repository to code + data + results +
  the one final paper.

`data/panel.parquet` is the frozen derived panel the paper is estimated on:
{len(panel):,} quotes across {n_slices:,} (date, expiry) slices,
{panel['date'].min().date()} to {panel['date'].max().date()}. Every table and
figure in the paper can be reproduced from this file alone.

## Reproducing

Four entry points, depending on what you want to reproduce.

**1. Main pipeline results, from the derived panel (no data licence needed).**
Start at stage 2 -- the fold definitions and the held-out-strike split in
`data/` are the exact ones used, so results are reproducible to the seed:

    python code/pipeline/04a_svi.py 0 3700          # classical baselines, chunked
    python code/pipeline/04b_ssvi.py 0 1500
    python code/pipeline/04c_thinplate.py
    python code/pipeline/04d_cubic.py 0 3700
    python code/pipeline/11a_fengler.py predict 0 1500     # Fengler constrained smoother
    python code/pipeline/07a_prep_fold.py <fold>           # 0..17
    python code/pipeline/07b_nn_train_v3_fold.py <fold> 1 18000 18000
    python code/pipeline/07c_nn_eval_v3_allfolds.py
    python code/pipeline/12a_prep_maturity_holdout.py <fold>
    python code/pipeline/12b_train_v3_matband.py <fold> 1 18000 18000
    python code/pipeline/12c_eval_maturity_holdout.py

**2. The isolating ablation (hard vs. soft constraint, arms A-E).** Reuses the
per-fold data files stage 07 above produces:

    python code/ablation/train_ablation.py hard 0 1 18000            # arm A
    python code/ablation/train_ablation.py hard_cal_only 0 2 18000   # arm D, seed 2
    python code/ablation/train_ablation.py soft_cal_only 0 3 18000   # arm E, seed 3
    python code/ablation/run_all_windows.py 8       # all 126 (arm, fold, seed) jobs, resumable
    python code/ablation/eval_ablation.py           # scores whatever checkpoints exist
    python code/ablation/aggregate_ablation.py

**3. The full 18-fold x 3-seed seed-variance check.** Reuses the same stage 07
per-fold checkpoints, trained with 2 more seeds each:

    python code/seed_variance/run_missing_seeds.py
    python code/seed_variance/aggregate_full_seed_variance.py

**4. The paper's figures**, from the small extracted inputs in
`data/figure_inputs/` (no retraining needed):

    python code/figures/gen_surface_data.py
    python code/figures/gen_smile_data.py
    python code/figures/gen_gk_data.py

**From raw bhavcopy (data licence needed).** Given your own copy of NSE F&O
bhavcopy, run `code/pipeline/00_inventory.py` through `code/pipeline/05_invert.py`
in numeric order; that regenerates `data/panel.parquet`.

Pipeline scripts take explicit chunk arguments (`<start> <end>`) because they
were run inside a per-call time budget; the chunk boundaries do not affect
results.

## Determinism

Seeds are fixed throughout (seed=42 for the held-out-strike split). Network
training is seeded per run; `results/phase4_seed_variance_full.csv` and
`results/SEED_VARIANCE_FULL.md` report the across-seed spread at full scale
(3 seeds x all 18 folds). The Fengler smoothing parameter is selected by
nested cross-validation on visible strikes only
(`code/pipeline/11a_fengler.py lambda_cv`), never on the held-out set.

## Environment

Python 3.13 with `requirements.txt`, plus `jax` (0.4.38, CPU) for the neural
models and the ablation, and `quadprog` for the Fengler QP.

## Citation

If you use this code or data, please cite the paper (details to be added on
acceptance/DOI assignment).

## License

No license file is included yet -- add one (e.g. MIT for code, CC-BY for the
derived data table) before making this repository public, so others know how
they may reuse it.
"""
    (STAGE / "README.md").write_text(readme, encoding="utf-8")

    zpath = ROOT / "github_release.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                z.write(f, f"iv-surface-neural-arbitrage/{f.relative_to(STAGE).as_posix()}")

    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    n_code = sum(1 for _ in (STAGE / "code").rglob("*.py")) + sum(1 for _ in (STAGE / "code").rglob("*.sh"))
    print(f"staged     : {STAGE}")
    print(f"files      : {sum(1 for f in STAGE.rglob('*') if f.is_file())}  ({total/1e6:.1f} MB uncompressed)")
    print(f"code       : {n_code} scripts")
    print(f"results    : {n_res} main CSV/MD + {n_abl} ablation CSV + {n_fig} figures")
    print(f"paper      : {'paper/paper.pdf present' if (STAGE / 'paper' / 'paper.pdf').exists() else 'MISSING main.pdf!'}")
    print(f"archive    : {zpath}  ({zpath.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
