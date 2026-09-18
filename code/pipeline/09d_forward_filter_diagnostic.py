"""Quantify the F6 rejection rate by forward-estimation source."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
filters = import_module("04_filters")

ROOT = Path(r"E:\Research\iv_surface")


def main():
    df = filters.load()
    df = df[df["fwd"].notna()]
    df = df[(df["tau"] >= 7/365) & (df["tau"] <= 1.0)]
    df = df[(df["volume"] >= 10) & (df["oi"] > 0)]
    df = df.assign(k=np.log(df["strike"] / df["fwd"]))
    df = df[(df["k"] >= -0.35) & (df["k"] <= 0.25)]
    is_call = df["opt_type"] == "CE"
    intrinsic = np.where(is_call, np.maximum(df["fwd"]-df["strike"], 0),
                         np.maximum(df["strike"]-df["fwd"], 0)) * np.exp(-df["r"]*df["tau"])
    df = df[df["settle_price"] > intrinsic + filters.TICK]
    is_call = df["opt_type"] == "CE"
    df = df[(is_call & (df["k"] > 0)) | (~is_call & (df["k"] < 0))]
    df = df.sort_values(["date", "expiry", "strike"]).copy()
    grouped = df.groupby(["date", "expiry"], sort=False)
    kp, kn = grouped["strike"].shift(1), grouped["strike"].shift(-1)
    pp, pn = grouped["settle_price"].shift(1), grouped["settle_price"].shift(-1)
    neighbors = kp.notna() & kn.notna()
    weight = (kn-df["strike"]) / (kn-kp)
    convex = neighbors & (df["settle_price"] > weight*pp + (1-weight)*pn + filters.TICK)
    mono_call = neighbors & (df["opt_type"] == "CE") & (df["settle_price"] > pp + filters.TICK)
    mono_put = neighbors & (df["opt_type"] == "PE") & (df["settle_price"] < pp - filters.TICK)
    df["f6_violation"] = convex | mono_call | mono_put
    out = (df.groupby("fwd_source").agg(
        rows_before_f6=("strike", "size"), rows_dropped_f6=("f6_violation", "sum"),
        slices_before_f6=("expiry", "size")).reset_index())
    # Replace the temporary row count with an exact distinct-slice count.
    slices = df.groupby("fwd_source").apply(
        lambda x: x[["date", "expiry"]].drop_duplicates().shape[0], include_groups=False)
    out["slices_before_f6"] = out["fwd_source"].map(slices)
    out["drop_rate_pct"] = 100*out["rows_dropped_f6"]/out["rows_before_f6"]
    out.to_csv(ROOT / "results" / "phase5_f6_by_forward_source.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
