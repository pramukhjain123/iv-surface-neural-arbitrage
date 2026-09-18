"""
Phase 4 evaluation: v3 (hard-calendar-constrained, lambda=0.01 frozen) NN evaluated
across all 18 walk-forward folds. Same Task A / Task B methodology as 06g_nn_eval_v3.py,
generalized to loop over folds and read each fold's train/test window from
05_splits/folds.csv instead of hardcoding fold 16's dates.

Metrics reported in vol points (IV decimal x 100), per the project's Phase 2 convention.

Fold 16 reuses the exact Phase 3 checkpoint (identical arch/data/lambda/step count) --
this is disclosed because fold 16's Task B window was used repeatedly during Phase 3
model/lambda selection, so its numbers here are NOT a blind test in the same sense as
the other 17 folds. See phase4_notes.md for full disclosure.

Usage: python3 07c_nn_eval_v3_allfolds.py
Outputs:
  results/phase4_nn_eval_by_fold.csv       -- Task A/B RMSE/MAE (vol points) per fold
  results/phase4_nn_violation_grid_by_fold.csv -- dense-grid arbitrage violation rates per fold
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
nn_train = import_module("07b_nn_train_v3_fold")

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

SEED = 1
N_LAYERS = len(nn_train.K_LAYER_SIZES) - 1
K_PAD_FRAC = nn_train.K_PAD_FRAC
N_FOLDS = 18


def load_run(fold_id, seed):
    run_dir = ROOT / "06_nn" / "runs_v3_allfolds" / f"fold{fold_id}_seed{seed}"
    ckpt_path = run_dir / "ckpt.npz"
    params, m, v, key, step = nn_train.load_ckpt(ckpt_path, N_LAYERS)
    ck = np.load(ckpt_path)
    norm_stats = ck["norm_stats"]
    return params, norm_stats, step


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
    SELECT date, k, tau, iv
    FROM '{ROOT}/04_iv/panel.parquet'
    WHERE date >= '{test_start}' AND date < '{test_end}'
    ORDER BY date, expiry, strike
    """
    test_meta = con.execute(q2).df()
    return train_ho_meta, test_meta


def main():
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    con = duckdb.connect()

    rows = []
    grid_rows = []
    t_start_all = time.time()

    for _, frow in folds.iterrows():
        fold_id = int(frow["fold"])
        train_start, train_end = str(frow["train_start"]), str(frow["train_end"])
        test_start, test_end = str(frow["test_start"]), str(frow["test_end"])

        data = np.load(ROOT / "06_nn" / f"fold{fold_id}_data.npz")
        k_min_train = float(np.min(data["train_vis_k"]))
        k_max_train = float(np.max(data["train_vis_k"]))
        tau_min_train = float(np.min(data["train_vis_tau"]))
        tau_max_train = float(np.max(data["train_vis_tau"]))
        k_pad = (k_max_train - k_min_train) * K_PAD_FRAC

        train_ho_meta, test_meta = load_meta(con, train_start, train_end, test_start, test_end)
        train_ho = train_ho_meta.assign(w=train_ho_meta["iv"] ** 2 * train_ho_meta["tau"])
        test = test_meta.assign(w=test_meta["iv"] ** 2 * test_meta["tau"])

        params, norm_stats, step = load_run(fold_id, SEED)
        k_mean, k_std = [float(x) for x in norm_stats]
        forward_batch, w_scalar = nn_train.make_forward(k_mean, k_std)
        dwdk_fn = jax.grad(w_scalar, argnums=1)
        d2wdk2_fn = jax.grad(dwdk_fn, argnums=1)
        dwdtau_fn = jax.grad(w_scalar, argnums=2)

        t0 = time.time()
        for task_name, df, meta in [("A", train_ho, train_ho_meta), ("B", test, test_meta)]:
            if len(df) == 0:
                continue
            w_pred = np.array(forward_batch(params, jnp.array(df["k"].values), jnp.array(df["tau"].values)))
            iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(df["tau"].values, 1e-8))
            err_decimal = iv_pred - df["iv"].values
            err_volpts = err_decimal * 100.0
            if task_name == "A":
                in_range = meta["in_range_slice"].values
            else:
                in_range = (df["k"].values >= k_min_train) & (df["k"].values <= k_max_train)
            for ir_flag, label in [(True, "in_range"), (False, "out_of_range"), (None, "all")]:
                if ir_flag is None:
                    mask = np.ones(len(err_volpts), dtype=bool)
                else:
                    mask = in_range == ir_flag
                if mask.sum() == 0:
                    continue
                rmse = float(np.sqrt(np.mean(err_volpts[mask] ** 2)))
                mae = float(np.mean(np.abs(err_volpts[mask])))
                rows.append(dict(fold=fold_id, seed=SEED, step=step, task=task_name,
                                  subset=label, n=int(mask.sum()),
                                  rmse_volpts=rmse, mae_volpts=mae,
                                  contaminated=(fold_id == 16)))
        eval_time_s = time.time() - t0

        gk = np.linspace(k_min_train - k_pad, k_max_train + k_pad, 121)
        gtau = np.linspace(tau_min_train, tau_max_train, 41)
        GK, GT = np.meshgrid(gk, gtau, indexing="ij")
        gk_flat, gtau_flat = GK.ravel(), GT.ravel()

        w_b = np.array(jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, jnp.array(gk_flat), jnp.array(gtau_flat)))
        wk_b = np.array(jax.vmap(dwdk_fn, in_axes=(None, 0, 0))(params, jnp.array(gk_flat), jnp.array(gtau_flat)))
        wkk_b = np.array(jax.vmap(d2wdk2_fn, in_axes=(None, 0, 0))(params, jnp.array(gk_flat), jnp.array(gtau_flat)))
        wtau_b = np.array(jax.vmap(dwdtau_fn, in_axes=(None, 0, 0))(params, jnp.array(gk_flat), jnp.array(gtau_flat)))
        w_safe = np.maximum(w_b, 1e-6)
        g = (1 - gk_flat * wk_b / (2 * w_safe)) ** 2 - (wk_b ** 2 / 4) * (1 / w_safe + 0.25) + wkk_b / 2
        butterfly_viol_rate = float(np.mean(g < 0))
        calendar_viol_rate = float(np.mean(wtau_b < 0))
        grid_rows.append(dict(fold=fold_id, seed=SEED, step=step,
                               butterfly_viol_rate=butterfly_viol_rate,
                               calendar_viol_rate=calendar_viol_rate,
                               min_dwdtau=float(wtau_b.min()),
                               mean_w_neg=float(np.mean(w_b < 0)),
                               nn_eval_time_s=eval_time_s,
                               contaminated=(fold_id == 16)))
        print(f"fold={fold_id} step={step}: n_taskA={len(train_ho)} n_taskB={len(test)} "
              f"butterfly_viol={butterfly_viol_rate:.4f} calendar_viol={calendar_viol_rate:.4f} "
              f"({time.time()-t0:.1f}s)")

    res = pd.DataFrame(rows)
    grid = pd.DataFrame(grid_rows)
    res.to_csv(ROOT / "results" / "phase4_nn_eval_by_fold.csv", index=False)
    grid.to_csv(ROOT / "results" / "phase4_nn_violation_grid_by_fold.csv", index=False)
    print(f"\ntotal wall time: {time.time()-t_start_all:.1f}s")
    print("\n=== Task A/B RMSE/MAE (vol points) by fold, subset=all ===")
    print(res[res["subset"] == "all"].to_string(index=False))
    print("\n=== violation grid by fold ===")
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
