"""
11d_fengler_summary.py -- combine the Fengler runs into the two tables the
manuscript actually needs: the classical-baseline accuracy comparison with a
Fengler row added, and the arbitrage-violation comparison with a Fengler row
added.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fengler", _HERE / "11a_fengler.py")
FG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FG)
ROOT, RES = FG.ROOT, FG.ROOT / "results"


def arbitrage():
    bs = sorted(RES.glob("phase6_fengler_butterfly_*_1500_12.csv"))
    cs = sorted(RES.glob("phase6_fengler_calendar_*_1500_12.csv"))
    if not bs:
        print("no strided arbitrage chunks found"); return
    b = pd.concat([pd.read_csv(f) for f in bs], ignore_index=True).drop_duplicates(["date", "expiry"])
    c = pd.concat([pd.read_csv(f) for f in cs], ignore_index=True).drop_duplicates(
        ["date", "expiry_earlier", "expiry_later"])
    row = {
        "method": "Fengler",
        "n_days_sampled": b["date"].nunique(),
        "n_slices": len(b),
        "price_convexity_violations": int(b["n_price_convex_viol"].sum()),
        "price_slope_violations": int(b["n_price_slope_viol"].sum()),
        "negative_density_points": int(b["n_density_negative"].sum()),
        "frac_slices_butterfly_viol_observed_range": float((b["n_viol_observed"] > 0).mean()),
        "frac_slices_butterfly_viol_incl_padding": float((b["n_viol"] > 0).mean()),
        "mean_frac_grid_iv_undefined": float((b["n_iv_undefined"] / b["n_grid"]).mean()),
        "n_expiry_pairs": len(c),
        "frac_pairs_calendar_viol_common_support": float((c["n_viol_common"] > 0).mean()),
        "frac_pairs_calendar_viol_incl_extrapolation": float((c["n_viol"] > 0).mean()),
        "calendar_price_viol_points_common_support": int(c["n_price_viol_common"].sum()),
        "calendar_grid_points_common_support": int(c["n_grid_common"].sum()),
    }
    out = pd.DataFrame([row])
    out.to_csv(RES / "phase6_fengler_arbitrage_combined.csv", index=False)
    print("Fengler arbitrage, combined strided sample:")
    print(out.T.to_string(header=False))

    # side by side with the published baseline numbers
    pub_b = pd.read_csv(RES / "phase4_baseline_arb_summary_butterfly.csv")
    pub_c = pd.read_csv(RES / "phase5_baseline_calendar_summary.csv")
    comp = pub_b[["method", "n_slices", "frac_slices_with_any_viol"]].rename(
        columns={"frac_slices_with_any_viol": "frac_slices_butterfly_viol"})
    comp = comp.merge(pub_c[["method", "pct_pairs"]].rename(
        columns={"pct_pairs": "pct_pairs_calendar_viol"}), on="method", how="outer")
    # the published within-observed-range butterfly rates, for the like-for-like
    # column (padded-grid rates alone overstate every method that extrapolates)
    within = {"ThinPlate": 0.89423, "CubicSpline": 0.81956, "SVI": np.nan, "SSVI": 0.0}
    comp["frac_slices_butterfly_viol_observed_range"] = comp["method"].map(within)
    comp = pd.concat([comp, pd.DataFrame([{
        "method": "Fengler", "n_slices": row["n_slices"],
        "frac_slices_butterfly_viol": row["frac_slices_butterfly_viol_incl_padding"],
        "frac_slices_butterfly_viol_observed_range":
            row["frac_slices_butterfly_viol_observed_range"],
        "pct_pairs_calendar_viol": 100 * row["frac_pairs_calendar_viol_common_support"]}])],
        ignore_index=True)
    comp.to_csv(RES / "phase6_arbitrage_comparison_with_fengler.csv", index=False)
    print("\nArbitrage comparison (published baselines + Fengler):")
    print(comp.round(4).to_string(index=False))


def accuracy():
    f = pd.read_csv(RES / "phase6_fengler_taskA_summary.csv")
    pub = pd.read_csv(RES / "baseline_comparison_summary.csv")
    rows = []
    for r in pub.itertuples():
        rows.append({"method": r.method, "n_in_range": int(r.n_x),
                     "rmse_in_range_volpts": r.rmse_vol_pts,
                     "mae_in_range_volpts": r.mae_vol_pts,
                     "n_out_of_range": int(r.n_y),
                     "rmse_out_of_range_volpts": r.rmse_oor_vol_pts,
                     "coverage_in_range": 1.0})
    fi = f[f["range"] == "in_range"].iloc[0]
    fo = f[f["range"] == "out_of_range"].iloc[0]
    rows.append({"method": "Fengler", "n_in_range": int(fi["n_scored"]),
                 "rmse_in_range_volpts": fi["rmse_vol_pts"],
                 "mae_in_range_volpts": fi["mae_vol_pts"],
                 "n_out_of_range": int(fo["n_scored"]),
                 "rmse_out_of_range_volpts": fo["rmse_vol_pts"],
                 "coverage_in_range": fi["coverage"]})
    out = pd.DataFrame(rows).sort_values("rmse_in_range_volpts").reset_index(drop=True)
    out.to_csv(RES / "phase6_baseline_comparison_with_fengler.csv", index=False)
    print("\nTask A accuracy, all 1,496 days (vol points), Fengler added:")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    accuracy()
    arbitrage()
