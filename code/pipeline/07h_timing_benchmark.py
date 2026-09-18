"""
Phase 4: wall-clock timing benchmark for fit and inference, NN vs the four Phase 2
baselines. The NN's fit (training) times are read directly from each fold's
06_nn/runs_v3_allfolds/fold{id}_seed1/log.csv (summed across resumed training calls,
detected by elapsed_s resets); its inference/eval time is from
results/phase4_nn_eval_by_fold.csv's nn_eval_time_s column.

The Phase 2 baseline scripts (04a-04d) were never instrumented for timing, so this
script benchmarks each one fresh on a fixed sample (a small slice/day range near the
start of the panel) to get a representative per-unit (per slice, or per day for the
two surface-level methods) fit+predict wall-clock cost, using each script's own
main(start, end) entry point but redirected to a scratch output path so nothing in
results/ is touched.

Usage: python3 07h_timing_benchmark.py
Output: results/phase4_timing.csv
"""
import sys
import time
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

sys.path.insert(0, str(ROOT / "src"))
N_BENCH_UNITS = 100  # slices (SVI/SSVI) or days (ThinPlate/CubicSpline) to time


def nn_train_time_by_fold():
    rows = []
    for fold_id in range(18):
        log_path = ROOT / "06_nn" / "runs_v3_allfolds" / f"fold{fold_id}_seed1" / "log.csv"
        df = pd.read_csv(log_path)
        elapsed = df["elapsed_s"].values
        total = 0.0
        for i in range(1, len(elapsed)):
            if elapsed[i] < elapsed[i - 1]:
                total += elapsed[i - 1]
        total += elapsed[-1]
        rows.append(dict(fold=fold_id, train_wall_s=total,
                          note="reused Phase-3 checkpoint" if fold_id == 16 else "trained in Phase 4"))
    return pd.DataFrame(rows)


def bench_baseline(module_name, start, end, res_dir_override):
    mod = importlib.import_module(module_name)
    orig_res = mod.RES
    mod.RES = res_dir_override
    t0 = time.time()
    mod.main(start, end)
    elapsed = time.time() - t0
    mod.RES = orig_res
    return elapsed


def main():
    scratch = Path("/tmp/phase4_timing_scratch")
    scratch.mkdir(exist_ok=True)

    rows = []

    nn_train_df = nn_train_time_by_fold()
    nn_train_df.to_csv(ROOT / "results" / "phase4_nn_train_time_by_fold.csv", index=False)
    fresh = nn_train_df[nn_train_df["fold"] != 16]
    rows.append(dict(method="NN_v3", stage="fit_18000steps_per_fold_mean_s",
                      value=fresh["train_wall_s"].mean(), n_units=1))
    rows.append(dict(method="NN_v3", stage="fit_18000steps_total_17folds_s",
                      value=fresh["train_wall_s"].sum(), n_units=17))

    eval_df = pd.read_csv(ROOT / "results" / "phase4_nn_eval_by_fold.csv")
    grid_df = pd.read_csv(ROOT / "results" / "phase4_nn_violation_grid_by_fold.csv")
    rows.append(dict(method="NN_v3", stage="predict_taskA_taskB_plus_grid_per_fold_mean_s",
                      value=grid_df["nn_eval_time_s"].mean(), n_units=1))

    for module_name, label in [("04a_svi", "SVI"), ("04b_ssvi", "SSVI")]:
        t = bench_baseline(module_name, 0, N_BENCH_UNITS, scratch)
        rows.append(dict(method=label, stage=f"fit_predict_{N_BENCH_UNITS}_slices_s", value=t, n_units=N_BENCH_UNITS))
        rows.append(dict(method=label, stage="fit_predict_per_slice_mean_s", value=t / N_BENCH_UNITS, n_units=1))
        print(f"{label}: {N_BENCH_UNITS} slices in {t:.2f}s ({t/N_BENCH_UNITS*1000:.2f} ms/slice)")

    for module_name, label in [("04c_thinplate", "ThinPlate"), ("04d_cubic", "CubicSpline")]:
        n_days = 30
        t = bench_baseline(module_name, 0, n_days, scratch)
        rows.append(dict(method=label, stage=f"fit_predict_{n_days}_days_s", value=t, n_units=n_days))
        rows.append(dict(method=label, stage="fit_predict_per_day_mean_s", value=t / n_days, n_units=1))
        print(f"{label}: {n_days} days in {t:.2f}s ({t/n_days*1000:.2f} ms/day)")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "phase4_timing.csv", index=False)
    print("\n=== timing summary ===")
    print(res.to_string(index=False))

    for f in scratch.glob("*.csv"):
        f.unlink()


if __name__ == "__main__":
    main()
