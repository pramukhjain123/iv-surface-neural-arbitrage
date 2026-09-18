"""Aggregate the ablation into the tables used in the results write-up.

Headline aggregates use the 17 blind folds (fold 16 excluded), matching the
paper's own convention for fold-16 contamination. Fold 16 is reported separately.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABEL = {"hard": "A. v3 hard (published repl.)",
         "soft_broad": "B. v3-soft, broad colloc.",
         "soft_targeted": "C. v3-soft, targeted colloc.",
         "hard_cal_only": "D. v3 hard, no bfly penalty",
         "soft_cal_only": "E. v3-soft, no bfly penalty"}
ORDER = ["hard", "soft_broad", "soft_targeted", "hard_cal_only", "soft_cal_only"]


def main():
    ev = pd.read_csv(ROOT / "results" / "ablation_eval_by_fold.csv")
    gr = pd.read_csv(ROOT / "results" / "ablation_violation_grid.csv")

    blind_ev = ev[~ev["contaminated"]]
    blind_gr = gr[~gr["contaminated"]]

    # ---- arbitrage table -------------------------------------------------
    arb = (blind_gr.groupby("variant")
           .agg(n_folds=("fold", "nunique"),
                calendar_viol_pct=("calendar_viol_rate", lambda s: 100 * s.mean()),
                folds_with_cal_viol=("calendar_viol_rate", lambda s: int((s > 0).sum())),
                worst_fold_cal_pct=("calendar_viol_rate", lambda s: 100 * s.max()),
                min_dwdtau=("min_dwdtau", "min"),
                butterfly_viol_pct=("butterfly_viol_rate", lambda s: 100 * s.mean()),
                cal_viol_pct_ext=("calendar_viol_rate_ext", lambda s: 100 * s.mean()),
                folds_with_cal_viol_ext=("calendar_viol_rate_ext", lambda s: int((s > 0).sum())),
                min_dwdtau_ext=("min_dwdtau_ext", "min"))
           .reindex(ORDER).reset_index())
    arb["label"] = arb["variant"].map(LABEL)

    # ---- accuracy table --------------------------------------------------
    acc = (blind_ev[blind_ev["subset"] == "all"]
           .pivot_table(index="variant", columns="task",
                        values=["rmse_volpts", "mae_volpts"], aggfunc="mean")
           .reindex(ORDER))
    acc.columns = [f"{a}_{b}" for a, b in acc.columns]
    acc = acc.reset_index()
    acc["label"] = acc["variant"].map(LABEL)

    acc_ir = (blind_ev[blind_ev["subset"] == "in_range"]
              .pivot_table(index="variant", columns="task",
                           values="rmse_volpts", aggfunc="mean")
              .reindex(ORDER).reset_index())
    acc_ir.columns = ["variant"] + [f"taskA_inrange" if c == "A" else "taskB_inrange"
                                    for c in acc_ir.columns[1:]]

    # ---- per-fold calendar detail ---------------------------------------
    per_fold = (gr.pivot_table(index="fold", columns="variant",
                               values="calendar_viol_rate")
                .reindex(columns=ORDER) * 100).round(3)

    arb.to_csv(ROOT / "results" / "ablation_summary_arbitrage.csv", index=False)
    acc.to_csv(ROOT / "results" / "ablation_summary_accuracy.csv", index=False)
    per_fold.to_csv(ROOT / "results" / "ablation_calendar_by_fold.csv")

    pd.set_option("display.width", 200)
    print("=== ARBITRAGE (mean over 17 blind folds; fold 16 excluded) ===")
    print(arb[["label", "n_folds", "calendar_viol_pct", "folds_with_cal_viol",
               "worst_fold_cal_pct", "min_dwdtau", "butterfly_viol_pct"]].to_string(index=False))
    print("\n=== EXTENDED tau GRID (tau out to 2x tau_max_train) ===")
    print(arb[["label", "cal_viol_pct_ext", "folds_with_cal_viol_ext",
               "min_dwdtau_ext"]].to_string(index=False))
    print("\n=== ACCURACY, vol points (mean over 17 blind folds) ===")
    print(acc[["label", "rmse_volpts_A", "rmse_volpts_B",
               "mae_volpts_A", "mae_volpts_B"]].to_string(index=False))
    print("\n=== in-range RMSE ===")
    print(acc_ir.to_string(index=False))
    print("\n=== calendar violation %, per fold ===")
    print(per_fold.to_string())
    print("\n=== fold 16 (contaminated, reported separately) ===")
    print(gr[gr["contaminated"]][["variant", "calendar_viol_rate", "min_dwdtau",
                                  "butterfly_viol_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
