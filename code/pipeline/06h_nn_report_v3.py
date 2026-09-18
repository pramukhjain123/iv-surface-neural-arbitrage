"""
Phase 3 v3 reporting: aggregate the v3 (hard-calendar-constrained) sweep into
mean+-std tables per lambda, and build the corrected fit-error vs
butterfly-arbitrage trade-off figure. Calendar-arbitrage is uniformly 0% at
every lambda by construction, so unlike the v1 report, there is no calendar
axis to plot -- this is a pure butterfly-vs-fit sweep, with v1's numbers
overlaid for direct before/after comparison.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

res = pd.read_csv(ROOT / "results" / "nn_v3_eval_summary.csv")
grid = pd.read_csv(ROOT / "results" / "nn_v3_violation_grid.csv")
baseline = pd.read_csv(ROOT / "results" / "baseline_comparison_summary.csv")

# v1 numbers for comparison (already aggregated in Phase 3 v1 report)
v1_agg = pd.read_csv(ROOT / "results" / "nn_eval_agg.csv")
v1_viol = pd.read_csv(ROOT / "results" / "nn_violation_agg.csv")

# --- per-(lambda, task, subset) mean+-std across seeds ---
agg = (
    res.groupby(["lam", "task", "subset"])
    .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
         mae_mean=("mae", "mean"), mae_std=("mae", "std"),
         n_seeds=("seed", "nunique"))
    .reset_index()
)
agg.to_csv(ROOT / "results" / "nn_v3_eval_agg.csv", index=False)

viol_agg = (
    grid.groupby("lam")
    .agg(butterfly_mean=("butterfly_viol_rate", "mean"), butterfly_std=("butterfly_viol_rate", "std"),
         calendar_mean=("calendar_viol_rate", "mean"), calendar_std=("calendar_viol_rate", "std"),
         min_dwdtau=("min_dwdtau", "min"))
    .reset_index()
)
viol_agg.to_csv(ROOT / "results" / "nn_v3_violation_agg.csv", index=False)

print("=== v3 RMSE by lambda, task, subset (mean +- std over 3 seeds) ===")
print(agg.to_string())
print()
print("=== v3 Violation rates by lambda (mean +- std over 3 seeds) ===")
print(viol_agg.to_string())

# --- headline before/after comparison table (Task A, all; Task B, all) ---
taskA_v3 = agg[(agg.task == "A") & (agg.subset == "all")].sort_values("lam").reset_index(drop=True)
taskB_v3 = agg[(agg.task == "B") & (agg.subset == "all")].sort_values("lam").reset_index(drop=True)
taskA_v1 = v1_agg[(v1_agg.task == "A") & (v1_agg.subset == "all")].sort_values("lam").reset_index(drop=True)
taskB_v1 = v1_agg[(v1_agg.task == "B") & (v1_agg.subset == "all")].sort_values("lam").reset_index(drop=True)
v = viol_agg.sort_values("lam").reset_index(drop=True)
v1v = v1_viol.sort_values("lam").reset_index(drop=True)

compare = pd.DataFrame({
    "lam": v["lam"],
    "v1_calendar_viol": v1v["calendar_mean"],
    "v3_calendar_viol": v["calendar_mean"],
    "v1_butterfly_viol": v1v["butterfly_mean"],
    "v3_butterfly_viol": v["butterfly_mean"],
    "v1_taskA_rmse": taskA_v1["rmse_mean"],
    "v3_taskA_rmse": taskA_v3["rmse_mean"],
    "v1_taskB_rmse": taskB_v1["rmse_mean"],
    "v3_taskB_rmse": taskB_v3["rmse_mean"],
})
compare.to_csv(ROOT / "results" / "nn_v1_vs_v3_comparison.csv", index=False)
print()
print("=== v1 vs v3 comparison ===")
print(compare.to_string())

# --- corrected trade-off figure ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.errorbar(v["lam"].clip(lower=0.003), taskA_v3["rmse_mean"], yerr=taskA_v3["rmse_std"],
            fmt="o-", color="#1f77b4", label="Task A (held-out strikes, train window)", capsize=3)
ax.errorbar(v["lam"].clip(lower=0.003), taskB_v3["rmse_mean"], yerr=taskB_v3["rmse_std"],
            fmt="s-", color="#ff7f0e", label="Task B (all strikes, unseen test window)", capsize=3)
ax.set_xscale("log")
ax.set_xlabel("$\\lambda$ (butterfly-penalty weight; $\\lambda$=0 shown at 0.003 for log scale)")
ax.set_ylabel("RMSE (IV, decimal)")
ax.set_title("v3 (hard calendar constraint): RMSE vs $\\lambda$\ncalendar-arbitrage violation = 0.0 at every $\\lambda$ (by construction)")
ax.legend(fontsize=8)

ax = axes[1]
ax.errorbar(v["butterfly_mean"], taskA_v3["rmse_mean"], xerr=v["butterfly_std"],
            yerr=taskA_v3["rmse_std"], fmt="o-", color="#1f77b4", label="v3 (Task A, fold 16)", capsize=3)
for lam, x, y in zip(v["lam"], v["butterfly_mean"], taskA_v3["rmse_mean"]):
    ax.annotate(f"$\\lambda$={lam:g}", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axhline(baseline.set_index("method").loc["SVI", "rmse"], ls="--", color="gray", lw=1, label="SVI baseline (Task A, in-range)")
ax.axhline(baseline.set_index("method").loc["ThinPlate", "rmse"], ls=":", color="gray", lw=1, label="ThinPlate baseline (Task A, in-range)")
ax.set_xlabel("post-hoc butterfly-arbitrage violation rate (dense grid)")
ax.set_ylabel("Task A RMSE (IV, decimal)")
ax.set_title("Fit error vs butterfly-arbitrage violation rate\n(each point = one $\\lambda$, mean over 3 seeds)")
ax.legend(fontsize=8)

plt.tight_layout()
out_path = ROOT / "results" / "phase3_v3_tradeoff.png"
plt.savefig(out_path, dpi=140)
print(f"saved {out_path}")
