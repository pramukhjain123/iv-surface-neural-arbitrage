import time
from pathlib import Path
import numpy as np, pandas as pd, duckdb, jax, jax.numpy as jnp
import train_ablation as TA

ROOT = TA.DATA_ROOT
OUT = TA.OUT_ROOT
SEED = 1
N_LAYERS = len(TA.K_LAYER_SIZES) - 1

folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
con = duckdb.connect()
rows = []

for variant in ["hard", "hard_cal_only"]:
    constrain = variant.startswith("hard")
    for _, frow in folds.iterrows():
        fold_id = int(frow["fold"])
        ckpt = OUT / "runs" / f"{variant}_fold{fold_id}_seed{SEED}" / "ckpt.npz"
        if not ckpt.exists():
            continue
        params, norm_stats, step = TA.load_ckpt(ckpt, N_LAYERS)
        k_mean, k_std = [float(x) for x in norm_stats]
        forward_batch, w_scalar = TA.make_forward(k_mean, k_std, constrain)

        data = np.load(ROOT / "06_nn" / f"fold{fold_id}_data.npz")
        k_min = float(np.min(data["train_vis_k"])); k_max = float(np.max(data["train_vis_k"]))

        q = f"""
        SELECT p.k, p.tau, p.iv,
               CASE WHEN p.k >= vr.k_vis_min AND p.k <= vr.k_vis_max THEN true ELSE false END AS in_range_slice
        FROM '{ROOT}/04_iv/panel.parquet' p
        JOIN '{ROOT}/05_splits/held_out_strikes.parquet' h
          ON p.date=h.date AND p.expiry=h.expiry AND p.strike=h.strike
        JOIN '{ROOT}/05_splits/visible_k_range.parquet' vr
          ON p.date=vr.date AND p.expiry=vr.expiry
        WHERE p.date >= '{frow["train_start"]}' AND p.date < '{frow["train_end"]}' AND h.is_heldout = true
        ORDER BY p.date, p.expiry, p.strike
        """
        train_ho_meta = con.execute(q).df()
        q2 = f"""
        SELECT date, k, tau, iv FROM '{ROOT}/04_iv/panel.parquet'
        WHERE date >= '{frow["test_start"]}' AND date < '{frow["test_end"]}'
        ORDER BY date, expiry, strike
        """
        test_meta = con.execute(q2).df()

        for task_name, meta in [("A", train_ho_meta), ("B", test_meta)]:
            if len(meta) == 0:
                continue
            w_pred = np.array(forward_batch(params, jnp.array(meta["k"].values), jnp.array(meta["tau"].values)))
            iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(meta["tau"].values, 1e-8))
            err = (iv_pred - meta["iv"].values) * 100.0
            if task_name == "A":
                in_range = meta["in_range_slice"].values
            else:
                in_range = (meta["k"].values >= k_min) & (meta["k"].values <= k_max)
            for ir_flag, label in [(True, "in_range"), (False, "out_of_range"), (None, "all")]:
                mask = np.ones(len(err), bool) if ir_flag is None else (in_range == ir_flag)
                if mask.sum() == 0:
                    continue
                rows.append(dict(variant=variant, fold=fold_id, task=task_name, subset=label,
                                  n=int(mask.sum()), rmse=float(np.sqrt(np.mean(err[mask]**2))),
                                  contaminated=(fold_id==16)))
        print(f"done {variant} fold {fold_id}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "results" / "quick_check_hard_rmse.csv", index=False)
print("\n=== Task B (test), subset=all, blind folds only (excl fold16) ===")
b_all = df[(df.task=="B") & (df.subset=="all") & (~df.contaminated)]
print(b_all.groupby("variant")["rmse"].agg(["mean","std","count"]))
print("\n=== Task B (test), subset=in_range, blind folds only ===")
b_ir = df[(df.task=="B") & (df.subset=="in_range") & (~df.contaminated)]
print(b_ir.groupby("variant")["rmse"].agg(["mean","std","count"]))
