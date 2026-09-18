"""Deduplicate regenerated arbitrage chunks and rebuild reviewer-facing summaries."""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(r"E:\Research\iv_surface")


def main():
    arb_dir = ROOT / "results" / "arb_chunks_v2"
    butterflies = pd.concat([pd.read_csv(p) for p in arb_dir.glob("butterfly_chunk_*.csv")],
                            ignore_index=True)
    calendars = pd.concat([pd.read_csv(p) for p in arb_dir.glob("calendar_chunk_*.csv")],
                          ignore_index=True)
    butterflies = butterflies.drop_duplicates(["date", "expiry", "method"])
    calendars = calendars.drop_duplicates(
        ["date", "method", "expiry_earlier", "expiry_later"])
    butterflies.to_csv(ROOT / "results" / "phase5_baseline_butterfly_full.csv", index=False)
    calendars.to_csv(ROOT / "results" / "phase5_baseline_calendar_full.csv", index=False)

    rows = []
    for method, group in butterflies.groupby("method"):
        rows.append({
            "method": method, "n_slices": len(group),
            "pct_slices_any": 100*(group["n_viol"] > 0).mean(),
            "mean_pct_grid_any": 100*(group["n_viol"]/group["n_grid"]).mean(),
            "pct_slices_observed": 100*(group["n_viol_observed"] > 0).mean(),
            "mean_pct_grid_observed": 100*(group["n_viol_observed"]/group["n_grid_observed"]).mean(),
            "pct_slices_padded": 100*(group["n_viol_padded"] > 0).mean(),
            "mean_pct_grid_padded": 100*(group["n_viol_padded"]/group["n_grid_padded"]).mean(),
        })
    pd.DataFrame(rows).to_csv(ROOT / "results" / "phase5_baseline_arb_by_domain.csv", index=False)

    cal = calendars.groupby("method").apply(lambda g: pd.Series({
        "n_pairs": len(g), "pct_pairs": 100*(g["n_viol"] > 0).mean(),
        "mean_pct_grid": 100*(g["n_viol"]/g["n_grid"]).mean()}), include_groups=False).reset_index()
    cal.to_csv(ROOT / "results" / "phase5_baseline_calendar_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(cal.to_string(index=False))

    svi_files = sorted((ROOT / "results" / "svi_fixed_chunks").glob("svi_chunk_*.csv"))
    if svi_files:
        svi = pd.concat([pd.read_csv(p) for p in svi_files], ignore_index=True).drop_duplicates(
            ["date", "expiry", "strike", "method"])
        svi["date"] = pd.to_datetime(svi["date"])
        svi["expiry"] = pd.to_datetime(svi["expiry"])
        svi["err"] = svi["iv_pred"] - svi["iv_actual"]
        con = duckdb.connect()
        other = con.execute(f"""SELECT * FROM '{ROOT}/results/baseline_predictions_long.parquet'
                                 WHERE method <> 'SVI'""").df()
        other["date"] = pd.to_datetime(other["date"])
        other["expiry"] = pd.to_datetime(other["expiry"])
        combined = pd.concat([other, svi], ignore_index=True)
        con.register("combined", combined)
        con.execute(f"COPY combined TO '{ROOT}/results/baseline_predictions_long_fixed.parquet' (FORMAT PARQUET)")
        print(f"fixed SVI prediction rows={len(svi)}; combined rows={len(combined)}")


if __name__ == "__main__":
    main()
