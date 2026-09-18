"""
Phase 3 reporting: aggregate the (lambda, seed) sweep into mean+-std tables per lambda,
and build the headline fit-error vs arbitrage-freedom trade-off figure, with the Phase 2
baseline points overlaid for context.
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

res = pd.read_csv(ROOT / "results" / "nn_eval_summary.csv")
grid = pd.read_csv(ROOT / "results" / "nn_violation_grid.csv")
baseline = pd.read_csv(ROOT / "results" / "baseline_comparison_summary.csv")

# --- per-(lambda, task, subset) mean+-std across seeds ---
agg = (
    res.groupby(["lam", "task", "subset"])
    .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
         mae_mean=("mae", "mean"), mae_std=("mae", "std"),
         n_seeds=("seed", "nunique"))
    .reset_index()
)
agg.to_csv(ROOT / "results" / "nn_eval_agg.csv", index=False)

viol_agg = (
    grid.groupby("lam")
    .agg(butterfly_mean=("butterfly_viol_rate", "mean"), butterfly_std=("butterfly_viol_rate", "std"),
         calendar_mean=("calendar_viol_rate", "mean"), calendar_std=("calendar_viol_rate", "std"))
    .reset_index()
)
viol_agg.to_csv(ROOT / "results" / "nn_violation_agg.csv", index=False)

print("=== RMSE by lambda, task, subset (mean +- std over 3 seeds) ===")
print(agg.to_string())
print()
print("=== Violation rates by lambda (mean +- std over 3 seeds) ===")
print(viol_agg.to_string())

# --- headline trade-off figure ---
taskA_all = agg[(agg.task == "A") & (agg.subset == "all")].sort_values("lam")
taskB_all = agg[(agg.task == "B") & (agg.subset == "all")].sort_values("lam")
v = viol_agg.sort_values("lam")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.errorbar(v["calendar_mean"], taskA_all["rmse_mean"], xerr=v["calendar_std"],
            yerr=taskA_all["rmse_std"], fmt="o-", color="#1f77b4", label="NN (Task A, fold 16)", capsize=3)
for lam, x, y in zip(v["lam"], v["calendar_mean"], taskA_all["rmse_mean"]):
    ax.annotate(f"$\\lambda$={lam:g}", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axhline(baseline.set_index("method").loc["SVI", "rmse"], ls="--", color="gray", lw=1, label="SVI baseline (Task A, in-range)")
ax.axhline(baseline.set_index("method").loc["ThinPlate", "rmse"], ls=":", color="gray", lw=1, label="ThinPlate baseline (Task A, in-range)")
ax.set_xlabel("post-hoc calendar-arbitrage violation rate (dense grid)")
ax.set_ylabel("Task A RMSE (IV, decimal)")
ax.set_title("Fit error vs calendar-arbitrage violation rate\n(each point = one $\\lambda$, mean over 3 seeds)")
ax.legend(fontsize=8)

ax = axes[1]
ax.errorbar(v["lam"].clip(lower=0.003), taskA_all["rmse_mean"], yerr=taskA_all["rmse_std"],
            fmt="o-", label="Task A (held-out strikes, train window)", capsize=3)
ax.errorbar(v["lam"].clip(lower=0.003), taskB_all["rmse_mean"], yerr=taskB_all["rmse_std"],
            fmt="s-", label="Task B (all strikes, unseen test window)", capsize=3)
ax.set_xscale("log")
ax.set_xlabel("$\\lambda$ (arbitrage-penalty weight; $\\lambda$=0 shown at 0.003 for log scale)")
ax.set_ylabel("RMSE (IV, decimal)")
ax.set_title("RMSE vs $\\lambda$")
ax.legend(fontsize=8)

plt.tight_layout()
out_path = ROOT / "results" / "phase3_tradeoff.png"
plt.savefig(out_path, dpi=140)
print(f"saved {out_path}")
