"""
Phase 3 evaluation: for each (lambda, seed) trained NN checkpoint, compute
  - Task A: RMSE on held-out strikes within the training window (fold 16),
    split in-range/out-of-range vs that slice's own visible-k range (same
    definition used for the Phase 2 baselines, for direct comparability).
  - Task B: RMSE on ALL strikes in the entirely-unseen test window, split
    in-range/out-of-range vs the GLOBAL training k-range (a different, and
    for a pooled global surface, more meaningful notion of extrapolation
    than the per-slice one -- see phase3_notes.md for why Task B is not
    directly comparable to the baselines' Task B in the same way Task A is).
  - Post-hoc arbitrage violation rates (butterfly, calendar) on a dense
    (k, tau) grid spanning the training domain plus the same padding used
    for collocation sampling during training.

Writes results/nn_eval_summary.csv and results/nn_violation_grid.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
nn_train = import_module("06b_nn_train")

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
SEEDS = [1, 2, 3]
N_LAYERS = len(nn_train.LAYER_SIZES) - 1
K_PAD_FRAC = nn_train.K_PAD_FRAC


def load_run(lam, seed):
    run_dir = ROOT / "06_nn" / "runs" / f"lam{lam}_seed{seed}"
    ckpt_path = run_dir / "ckpt.npz"
    params, m, v, key, step = nn_train.load_ckpt(ckpt_path, N_LAYERS)
    ck = np.load(ckpt_path)
    norm_stats = ck["norm_stats"]
    return params, norm_stats, step


def main():
    data = np.load(ROOT / "06_nn" / "fold16_data.npz")

    # Single source of truth for Task A / Task B rows: one duckdb query each,
    # carrying k/tau/iv (and the Task A in-range flag) together so there is no
    # risk of a row-order mismatch between separately-built arrays.
    train_ho_meta, test_meta = load_meta()
    train_ho = train_ho_meta.assign(w=train_ho_meta["iv"] ** 2 * train_ho_meta["tau"])
    test = test_meta.assign(w=test_meta["iv"] ** 2 * test_meta["tau"])

    k_min_train = float(np.min(data["train_vis_k"]))
    k_max_train = float(np.max(data["train_vis_k"]))
    tau_min_train = float(np.min(data["train_vis_tau"]))
    tau_max_train = float(np.max(data["train_vis_tau"]))
    k_pad = (k_max_train - k_min_train) * K_PAD_FRAC

    rows = []
    grid_rows = []

    # dense grid for violation checks: within the padded collocation domain
    gk = np.linspace(k_min_train - k_pad, k_max_train + k_pad, 121)
    gtau = np.linspace(tau_min_train, tau_max_train, 41)
    GK, GT = np.meshgrid(gk, gtau, indexing="ij")
    gk_flat, gtau_flat = GK.ravel(), GT.ravel()

    for lam in LAMBDAS:
        for seed in SEEDS:
            params, norm_stats, step = load_run(lam, seed)
            k_mean, k_std, tau_mean, tau_std = [float(x) for x in norm_stats]
            forward_batch, w_scalar = nn_train.make_forward(k_mean, k_std, tau_mean, tau_std)
            dwdk_fn = jax.grad(w_scalar, argnums=1)
            d2wdk2_fn = jax.grad(dwdk_fn, argnums=1)
            dwdtau_fn = jax.grad(w_scalar, argnums=2)

            # --- predictions on Task A / Task B ---
            for task_name, df, meta in [("A", train_ho, train_ho_meta), ("B", test, test_meta)]:
                w_pred = np.array(forward_batch(params, jnp.array(df["k"].values), jnp.array(df["tau"].values)))
                iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(df["tau"].values, 1e-8))
                err = iv_pred - df["iv"].values
                if task_name == "A":
                    in_range = meta["in_range_slice"].values
                else:
                    in_range = (df["k"].values >= k_min_train) & (df["k"].values <= k_max_train)
                for ir_flag, label in [(True, "in_range"), (False, "out_of_range"), (None, "all")]:
                    if ir_flag is None:
                        mask = np.ones(len(err), dtype=bool)
                    else:
                        mask = in_range == ir_flag
                    if mask.sum() == 0:
                        continue
                    rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
                    mae = float(np.mean(np.abs(err[mask])))
                    rows.append(dict(lam=lam, seed=seed, step=step, task=task_name,
                                      subset=label, n=int(mask.sum()), rmse=rmse, mae=mae))

            # --- post-hoc violation rates on dense grid ---
            w_b = np.array(jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, jnp.array(gk_flat), jnp.array(gtau_flat)))
            wk_b = np.array(jax.vmap(dwdk_fn, in_axes=(None, 0, 0))(params, jnp.array(gk_flat), jnp.array(gtau_flat)))
            wkk_b = np.array(jax.vmap(d2wdk2_fn, in_axes=(None, 0, 0))(params, jnp.array(gk_flat), jnp.array(gtau_flat)))
            wtau_b = np.array(jax.vmap(dwdtau_fn, in_axes=(None, 0, 0))(params, jnp.array(gk_flat), jnp.array(gtau_flat)))
            w_safe = np.maximum(w_b, 1e-6)
            g = (1 - gk_flat * wk_b / (2 * w_safe)) ** 2 - (wk_b ** 2 / 4) * (1 / w_safe + 0.25) + wkk_b / 2
            butterfly_viol_rate = float(np.mean(g < 0))
            calendar_viol_rate = float(np.mean(wtau_b < 0))
            grid_rows.append(dict(lam=lam, seed=seed, step=step,
                                   butterfly_viol_rate=butterfly_viol_rate,
                                   calendar_viol_rate=calendar_viol_rate,
                                   mean_w_neg=float(np.mean(w_b < 0))))
            print(f"lam={lam} seed={seed} step={step}: done")

    res = pd.DataFrame(rows)
    grid = pd.DataFrame(grid_rows)
    res.to_csv(ROOT / "results" / "nn_eval_summary.csv", index=False)
    grid.to_csv(ROOT / "results" / "nn_violation_grid.csv", index=False)
    print(res.to_string())
    print(grid.to_string())


def load_meta():
    import duckdb
    con = duckdb.connect()
    TRAIN_START, TRAIN_END = "2020-01-01", "2025-07-01"
    TEST_START, TEST_END = "2025-10-01", "2026-01-01"
    q = f"""
    SELECT p.k, p.tau, p.iv,
           CASE WHEN p.k >= vr.k_vis_min AND p.k <= vr.k_vis_max THEN true ELSE false END AS in_range_slice
    FROM '{ROOT}/04_iv/panel.parquet' p
    JOIN '{ROOT}/05_splits/held_out_strikes.parquet' h
      ON p.date=h.date AND p.expiry=h.expiry AND p.strike=h.strike
    JOIN '{ROOT}/05_splits/visible_k_range.parquet' vr
      ON p.date=vr.date AND p.expiry=vr.expiry
    WHERE p.date >= '{TRAIN_START}' AND p.date < '{TRAIN_END}' AND h.is_heldout = true
    ORDER BY p.date, p.expiry, p.strike
    """
    train_ho_meta = con.execute(q).df()
    q2 = f"""
    SELECT k, tau, iv
    FROM '{ROOT}/04_iv/panel.parquet'
    WHERE date >= '{TEST_START}' AND date < '{TEST_END}'
    ORDER BY date, expiry, strike
    """
    test_meta = con.execute(q2).df()
    return train_ho_meta, test_meta


if __name__ == "__main__":
    main()
