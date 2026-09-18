"""
Evaluation for the isolating ablation. Mirrors 07c_nn_eval_v3_allfolds.py exactly:
same Task A / Task B definitions, same vol-point convention, same dense-grid
arbitrage scoring (121 k-points x 41 tau-points, butterfly via Durrleman g,
calendar via dw/dtau), so numbers are directly comparable to the paper's v3 row.

Usage: python3 eval_ablation.py
Outputs: results/ablation_eval_by_fold.csv, results/ablation_violation_grid.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import jax
import jax.numpy as jnp

import train_ablation as TA

ROOT = TA.DATA_ROOT
OUT = TA.OUT_ROOT
SEED = 1
N_LAYERS = len(TA.K_LAYER_SIZES) - 1
K_PAD_FRAC = TA.K_PAD_FRAC


def load_meta(con, train_start, train_end, test_start, test_end):
    q = f"""
    SELECT p.k, p.tau, p.iv,
           CASE WHEN p.k >= vr.k_vis_min AND p.k <= vr.k_vis_max THEN true ELSE false END AS in_range_slice
    FROM '{ROOT}/04_iv/panel.parquet' p
    JOIN '{ROOT}/05_splits/held_out_strikes.parquet' h
      ON p.date=h.date AND p.expiry=h.expiry AND p.strike=h.strike
    JOIN '{ROOT}/05_splits/visible_k_range.parquet' vr
      ON p.date=vr.date AND p.expiry=vr.expiry
    WHERE p.date >= '{train_start}' AND p.date < '{train_end}' AND h.is_heldout = true
    ORDER BY p.date, p.expiry, p.strike
    """
    train_ho_meta = con.execute(q).df()
    q2 = f"""
    SELECT date, k, tau, iv FROM '{ROOT}/04_iv/panel.parquet'
    WHERE date >= '{test_start}' AND date < '{test_end}'
    ORDER BY date, expiry, strike
    """
    return train_ho_meta, con.execute(q2).df()


def main():
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    con = duckdb.connect()
    rows, grid_rows = [], []

    for variant in TA.VARIANTS:
        constrain = variant.startswith("hard")
        for _, frow in folds.iterrows():
            fold_id = int(frow["fold"])
            ckpt = OUT / "runs" / f"{variant}_fold{fold_id}_seed{SEED}" / "ckpt.npz"
            if not ckpt.exists():
                print(f"[skip] {variant} fold {fold_id}: no checkpoint")
                continue
            params, norm_stats, step = TA.load_ckpt(ckpt, N_LAYERS)
            k_mean, k_std = [float(x) for x in norm_stats]
            forward_batch, w_scalar = TA.make_forward(k_mean, k_std, constrain)
            dwdk_fn = jax.grad(w_scalar, argnums=1)
            d2wdk2_fn = jax.grad(dwdk_fn, argnums=1)
            dwdtau_fn = jax.grad(w_scalar, argnums=2)

            data = np.load(ROOT / "06_nn" / f"fold{fold_id}_data.npz")
            k_min = float(np.min(data["train_vis_k"])); k_max = float(np.max(data["train_vis_k"]))
            tau_min = float(np.min(data["train_vis_tau"])); tau_max = float(np.max(data["train_vis_tau"]))
            k_pad = (k_max - k_min) * K_PAD_FRAC

            train_ho_meta, test_meta = load_meta(
                con, str(frow["train_start"]), str(frow["train_end"]),
                str(frow["test_start"]), str(frow["test_end"]))
            train_ho = train_ho_meta.assign(w=train_ho_meta["iv"] ** 2 * train_ho_meta["tau"])
            test = test_meta.assign(w=test_meta["iv"] ** 2 * test_meta["tau"])

            for task_name, df, meta in [("A", train_ho, train_ho_meta), ("B", test, test_meta)]:
                if len(df) == 0:
                    continue
                w_pred = np.array(forward_batch(params, jnp.array(df["k"].values),
                                                jnp.array(df["tau"].values)))
                iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(df["tau"].values, 1e-8))
                err = (iv_pred - df["iv"].values) * 100.0
                if task_name == "A":
                    in_range = meta["in_range_slice"].values
                else:
                    in_range = (df["k"].values >= k_min) & (df["k"].values <= k_max)
                for ir_flag, label in [(True, "in_range"), (False, "out_of_range"), (None, "all")]:
                    mask = np.ones(len(err), bool) if ir_flag is None else (in_range == ir_flag)
                    if mask.sum() == 0:
                        continue
                    rows.append(dict(variant=variant, fold=fold_id, step=step, task=task_name,
                                     subset=label, n=int(mask.sum()),
                                     rmse_volpts=float(np.sqrt(np.mean(err[mask] ** 2))),
                                     mae_volpts=float(np.mean(np.abs(err[mask]))),
                                     contaminated=(fold_id == 16)))

            gk = np.linspace(k_min - k_pad, k_max + k_pad, 121)
            gtau = np.linspace(tau_min, tau_max, 41)
            GK, GT = np.meshgrid(gk, gtau, indexing="ij")
            gkf, gtf = jnp.array(GK.ravel()), jnp.array(GT.ravel())
            w_b = np.array(jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, gkf, gtf))
            wk_b = np.array(jax.vmap(dwdk_fn, in_axes=(None, 0, 0))(params, gkf, gtf))
            wkk_b = np.array(jax.vmap(d2wdk2_fn, in_axes=(None, 0, 0))(params, gkf, gtf))
            wtau_b = np.array(jax.vmap(dwdtau_fn, in_axes=(None, 0, 0))(params, gkf, gtf))
            gkn = np.array(GK.ravel())
            w_safe = np.maximum(w_b, 1e-6)
            g = (1 - gkn * wk_b / (2 * w_safe)) ** 2 - (wk_b ** 2 / 4) * (1 / w_safe + 0.25) + wkk_b / 2
            # --- extended grid: tau beyond the training range ---------------
            # The hard constraint holds at EVERY (k, tau) by construction. A soft
            # penalty only shapes the surface where collocation points were drawn,
            # i.e. tau in [tau_min, tau_max]. Scoring tau out to 2x tau_max tests
            # whether "no measured violations" is a guarantee or an artifact of
            # where we looked.
            gtau_ext = np.linspace(tau_min, 2.0 * tau_max, 41)
            GKe, GTe = np.meshgrid(gk, gtau_ext, indexing="ij")
            gkfe, gtfe = jnp.array(GKe.ravel()), jnp.array(GTe.ravel())
            wtau_ext = np.array(jax.vmap(dwdtau_fn, in_axes=(None, 0, 0))(params, gkfe, gtfe))

            grid_rows.append(dict(
                variant=variant, fold=fold_id, step=step,
                butterfly_viol_rate=float(np.mean(g < 0)),
                calendar_viol_rate=float(np.mean(wtau_b < 0)),
                min_dwdtau=float(wtau_b.min()),
                calendar_viol_rate_ext=float(np.mean(wtau_ext < 0)),
                min_dwdtau_ext=float(wtau_ext.min()),
                frac_w_neg=float(np.mean(w_b < 0)),
                contaminated=(fold_id == 16)))
            print(f"{variant} fold={fold_id} step={step} "
                  f"cal_viol={grid_rows[-1]['calendar_viol_rate']:.4f} "
                  f"bfly_viol={grid_rows[-1]['butterfly_viol_rate']:.4f}", flush=True)

    (OUT / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "results" / "ablation_eval_by_fold.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT / "results" / "ablation_violation_grid.csv", index=False)
    print("\nwrote results/ablation_eval_by_fold.csv and results/ablation_violation_grid.csv")


if __name__ == "__main__":
    main()
