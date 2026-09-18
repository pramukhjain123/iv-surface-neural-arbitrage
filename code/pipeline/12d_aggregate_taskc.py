"""
12d_aggregate_taskc.py -- aggregate Task C across folds from the per-fold raw
prediction files.

Aggregating from the raw files rather than from 12c's own summary is
deliberate: 12c writes its cross-fold summary at the end of whatever fold list
it was given, so invoking it one fold at a time (which is how the long run is
chunked) would leave that file holding only the last fold. The raw per-fold
files are complete and additive, so they are the source of truth.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fengler", _HERE / "11a_fengler.py")
FG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FG)
RES = FG.ROOT / "results"

LABEL = {
    "NN_v3_matband": "NN v3, band withheld",
    "NN_v3_trained_with_band": "NN v3, band in training",
    "Fengler+PCHIP": "Fengler + PCHIP",
    "SSVI+PCHIP": "SSVI + PCHIP",
    "SVI+PCHIP": "SVI + PCHIP",
    "CubicSpline+PCHIP": "CubicSpline + PCHIP",
    "ThinPlate": "ThinPlate (native 2-D)",
}
ORDER = ["NN v3, band withheld", "NN v3, band in training", "Fengler + PCHIP",
         "SSVI + PCHIP", "SVI + PCHIP", "ThinPlate (native 2-D)",
         "CubicSpline + PCHIP"]


def main():
    files = sorted(RES.glob("phase6_taskC_fold*_raw*.csv"))
    if not files:
        raise SystemExit("no Task C raw files found")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["fold", "window", "date", "expiry", "k", "method"])
    df["label"] = df["method"].map(LABEL).fillna(df["method"])

    # fold 16 is excluded from headline aggregates throughout the paper
    blind = df[df["fold"] != 16]

    rows = []
    for (win, lab), g in blind.groupby(["window", "label"]):
        per_fold = []
        for f, gf in g.groupby("fold"):
            s = gf.dropna(subset=["iv_pred"])
            if len(s) == 0:
                continue
            err = (s["iv_pred"] - s["iv_actual"]).to_numpy()
            per_fold.append({"rmse": np.sqrt(np.mean(err ** 2)) * 100,
                             "mae": np.mean(np.abs(err)) * 100,
                             "med": np.median(np.abs(err)) * 100,
                             "cov": len(s) / len(gf)})
        if not per_fold:
            continue
        pf = pd.DataFrame(per_fold)
        rows.append({"window": win, "method": lab, "n_folds": len(pf),
                     "rmse_volpts": pf["rmse"].mean(),
                     "mae_volpts": pf["mae"].mean(),
                     "median_abs_err_volpts": pf["med"].mean(),
                     "coverage": pf["cov"].mean(),
                     "n_quotes": int(g["iv_pred"].notna().sum())})
    out = pd.DataFrame(rows)
    out["ord"] = out["method"].apply(lambda m: ORDER.index(m) if m in ORDER else 99)
    out = out.sort_values(["window", "ord"]).drop(columns="ord").reset_index(drop=True)
    out.to_csv(RES / "phase6_taskC_summary.csv", index=False)

    print(f"Task C, mean over folds (fold 16 excluded), from {len(files)} fold files")
    print(f"folds present: {sorted(df['fold'].unique())}\n")
    for win, name in [("c1", "c1  train-window band slices (matched information sets)"),
                      ("c2", "c2  test-window band slices (static net also extrapolates in date)")]:
        sub = out[out["window"] == win]
        if len(sub) == 0:
            continue
        print(name)
        print(sub[["method", "n_folds", "rmse_volpts", "mae_volpts",
                   "median_abs_err_volpts", "coverage", "n_quotes"]]
              .round(4).to_string(index=False))
        print()

    # the control: what did withholding the band actually cost the network?
    for win in ("c1", "c2"):
        a = out[(out["window"] == win) & (out["method"] == "NN v3, band withheld")]
        b = out[(out["window"] == win) & (out["method"] == "NN v3, band in training")]
        if len(a) and len(b):
            ra, rb = float(a["rmse_volpts"].iloc[0]), float(b["rmse_volpts"].iloc[0])
            print(f"{win}: withholding the [30,60]d band changes the network's RMSE on that "
                  f"band from {rb:.3f} to {ra:.3f} vol pts ({100*(ra-rb)/rb:+.1f}%)")


if __name__ == "__main__":
    main()
