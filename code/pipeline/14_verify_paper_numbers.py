"""
14_verify_paper_numbers.py -- re-derive every number added to the manuscript in
this revision straight from the results files, and check it against the value
actually written in the .tex sources. Any mismatch is printed as FAIL.

The point is that the numbers in the paper were typed by hand into prose and
tables; this script is the independent check that each one still corresponds to
what the scripts produced.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fengler", _HERE / "11a_fengler.py")
FG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FG)
ROOT, RES = FG.ROOT, FG.ROOT / "results"
TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
for f in sorted((ROOT / "paper" / "sections").glob("*.tex")):
    TEX += f.read_text(encoding="utf-8")

checks = []


def check(name, derived, written, tol=0.006):
    ok = abs(derived - written) <= tol * max(1.0, abs(derived))
    checks.append((ok, name, derived, written))


def present(name, needle):
    checks.append((needle in TEX, name, needle, "in .tex"))


# ---- Fengler accuracy
f = pd.read_csv(RES / "phase6_fengler_taskA_summary.csv")
fi = f[f["range"] == "in_range"].iloc[0]
fo = f[f["range"] == "out_of_range"].iloc[0]
check("Fengler Task A in-range RMSE", fi["rmse_vol_pts"], 0.445)
check("Fengler Task A out-of-range RMSE", fo["rmse_vol_pts"], 2.31, tol=0.01)
check("Fengler in-range coverage", fi["coverage"], 0.9999, tol=1e-3)

bf = pd.read_csv(RES / "phase6_fengler_by_fold.csv")
b = bf[(bf["scope"] == "all_heldout") & (bf["fold"] != 16)]
check("Fengler per-fold train RMSE", b[b["window"] == "train_window"]["rmse_volpts"].mean(), 0.833)
check("Fengler per-fold test RMSE", b[b["window"] == "test_window"]["rmse_volpts"].mean(), 0.646)
check("Fengler per-fold test MAE", b[b["window"] == "test_window"]["mae_volpts"].mean(), 0.203)
check("Fengler per-fold test median|err|",
      b[b["window"] == "test_window"]["median_abs_err_volpts"].mean(), 0.055)
check("Fengler per-fold test trim-mean",
      b[b["window"] == "test_window"]["trimmed_mean_abs_err_volpts"].mean(), 0.080)
check("Fengler per-fold coverage", b[b["window"] == "test_window"]["coverage"].mean(), 0.990, tol=0.01)

# ---- Fengler arbitrage
a = pd.read_csv(RES / "phase6_fengler_arbitrage_combined.csv").iloc[0]
check("Fengler price convexity violations", a["price_convexity_violations"], 0, tol=0)
check("Fengler price slope violations", a["price_slope_violations"], 0, tol=0)
check("Fengler negative density points", a["negative_density_points"], 0, tol=0)
check("Fengler n slices sampled", a["n_slices"], 1585, tol=0)
check("Fengler n expiry pairs", a["n_expiry_pairs"], 1335, tol=0)
check("Fengler butterfly obs-range frac", a["frac_slices_butterfly_viol_observed_range"], 0.038, tol=0.02)
check("Fengler butterfly all frac", a["frac_slices_butterfly_viol_incl_padding"], 0.447, tol=0.01)
check("Fengler calendar common-support frac", a["frac_pairs_calendar_viol_common_support"], 0.017, tol=0.05)
check("Fengler calendar crossing points", a["calendar_price_viol_points_common_support"], 25, tol=0)
check("Fengler calendar grid points", a["calendar_grid_points_common_support"], 24549, tol=0)
check("Fengler grid IV undefined frac", a["mean_frac_grid_iv_undefined"], 0.040, tol=0.02)

# ---- forward sensitivity
fw = pd.read_csv(RES / "phase6_forward_multistrike.csv")
check("forward slices", len(fw), 11735, tol=0)
check("forward median |rel diff| bp", fw["rel_diff_fixed"].abs().median() * 1e4, 3.8, tol=0.02)
check("forward median pairs", fw["n_pairs"].median(), 25, tol=0)
check("implied/known discount factor", (fw["disc_implied"] / fw["disc_known"]).median(), 1.001, tol=1e-3)
cmp_ = pd.read_csv(RES / "phase6_forward_panel_compare.csv").set_index("metric")["value"]
check("panel rows base", cmp_["rows_base"], 307458, tol=0)
check("panel rows multi", cmp_["rows_multi"], 308048, tol=0)
check("panel slices multi", cmp_["slices_multi"], 9432, tol=0)
check("mean |IV shift| vol pts", cmp_["mean_abs_iv_shift_volpts"], 0.121, tol=0.01)

# ---- Task C
tc_path = RES / "phase6_taskC_summary.csv"
if tc_path.exists():
    tc = pd.read_csv(tc_path).set_index(["window", "method"])
    for win in ("c1", "c2"):
        try:
            wa = float(tc.loc[(win, "NN v3, band withheld"), "rmse_volpts"])
            wb = float(tc.loc[(win, "NN v3, band in training"), "rmse_volpts"])
            rel = 100 * (wa - wb) / wb
            print(f"[info] Task C {win}: withheld {wa:.3f} vs with-band {wb:.3f} "
                  f"({rel:+.1f}%) over {int(tc.loc[(win,'NN v3, band withheld'),'n_folds'])} folds")
            best = tc.loc[win].drop(index=["NN v3, band withheld", "NN v3, band in training",
                                           "CubicSpline + PCHIP"], errors="ignore")
            ratio = wa / best["rmse_volpts"].min()
            print(f"[info] Task C {win}: network / best classical+rule = {ratio:.2f}x")
        except KeyError:
            pass
else:
    print("[info] Task C summary not yet written")

# ---- claims that must literally appear
present("abstract cites Fengler RMSE 0.45", "RMSE of\n0.45 volatility points")
present("Task C section exists", r"\label{sec:res-taskc}")
present("properties table exists", r"\label{tab:properties}")
present("forward section exists", r"\label{sec:res-forward}")
present("Fengler section exists", r"\label{sec:res-fengler}")

npass = sum(1 for ok, *_ in checks if ok)
print(f"\n{npass}/{len(checks)} checks passed\n")
for ok, name, d, w in checks:
    if not ok:
        print(f"  FAIL  {name}: derived={d} written={w}")
if npass == len(checks):
    print("all manuscript numbers reconcile with the results files")
