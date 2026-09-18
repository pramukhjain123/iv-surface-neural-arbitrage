import numpy as np, pandas as pd, duckdb, jax, jax.numpy as jnp
import train_ablation as TA

ROOT = TA.DATA_ROOT
OUT = TA.OUT_ROOT
N_LAYERS = len(TA.K_LAYER_SIZES) - 1
K_PAD_FRAC = TA.K_PAD_FRAC

folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
con = duckdb.connect()
acc_rows, grid_rows = [], []

for variant in ["hard_cal_only", "soft_cal_only"]:
    constrain = variant.startswith("hard")
    for seed in (1, 2, 3):
        for _, frow in folds.iterrows():
            fold_id = int(frow["fold"])
            ckpt = OUT / "runs" / f"{variant}_fold{fold_id}_seed{seed}" / "ckpt.npz"
            if not ckpt.exists():
                print(f"[MISSING] {variant} seed{seed} fold{fold_id}")
                continue
            params, norm_stats, step = TA.load_ckpt(ckpt, N_LAYERS)
            k_mean, k_std = [float(x) for x in norm_stats]
            forward_batch, w_scalar = TA.make_forward(k_mean, k_std, constrain)
            dwdtau_fn = jax.grad(w_scalar, argnums=2)
            dwdk_fn = jax.grad(w_scalar, argnums=1)
            d2wdk2_fn = jax.grad(dwdk_fn, argnums=1)

            data = np.load(ROOT / "06_nn" / f"fold{fold_id}_data.npz")
            k_min = float(np.min(data["train_vis_k"])); k_max = float(np.max(data["train_vis_k"]))
            tau_min = float(np.min(data["train_vis_tau"])); tau_max = float(np.max(data["train_vis_tau"]))
            k_pad = (k_max - k_min) * K_PAD_FRAC

            q2 = f"""
            SELECT date, k, tau, iv FROM '{ROOT}/04_iv/panel.parquet'
            WHERE date >= '{frow["test_start"]}' AND date < '{frow["test_end"]}'
            ORDER BY date, expiry, strike
            """
            test_meta = con.execute(q2).df()
            if len(test_meta) > 0:
                w_pred = np.array(forward_batch(params, jnp.array(test_meta["k"].values), jnp.array(test_meta["tau"].values)))
                iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(test_meta["tau"].values, 1e-8))
                err = (iv_pred - test_meta["iv"].values) * 100.0
                acc_rows.append(dict(variant=variant, seed=seed, fold=fold_id,
                                      rmse=float(np.sqrt(np.mean(err**2))), contaminated=(fold_id==16)))

            gk = np.linspace(k_min - k_pad, k_max + k_pad, 121)
            gtau = np.linspace(tau_min, tau_max, 41)
            GK, GT = np.meshgrid(gk, gtau, indexing="ij")
            gkf, gtf = jnp.array(GK.ravel()), jnp.array(GT.ravel())
            w_b = np.array(jax.vmap(w_scalar, in_axes=(None,0,0))(params, gkf, gtf))
            wk_b = np.array(jax.vmap(dwdk_fn, in_axes=(None,0,0))(params, gkf, gtf))
            wkk_b = np.array(jax.vmap(d2wdk2_fn, in_axes=(None,0,0))(params, gkf, gtf))
            wtau_b = np.array(jax.vmap(dwdtau_fn, in_axes=(None,0,0))(params, gkf, gtf))
            gkn = np.array(GK.ravel())
            w_safe = np.maximum(w_b, 1e-6)
            g = (1 - gkn*wk_b/(2*w_safe))**2 - (wk_b**2/4)*(1/w_safe + 0.25) + wkk_b/2
            grid_rows.append(dict(variant=variant, seed=seed, fold=fold_id,
                                   calendar_viol_rate=float(np.mean(wtau_b < 0)),
                                   butterfly_viol_rate=float(np.mean(g < 0)),
                                   min_dwdtau=float(wtau_b.min()),
                                   contaminated=(fold_id==16)))
        print(f"finished {variant} seed{seed}", flush=True)

acc = pd.DataFrame(acc_rows); grid = pd.DataFrame(grid_rows)
acc.to_csv(OUT / "results" / "de_comparison_rmse.csv", index=False)
grid.to_csv(OUT / "results" / "de_comparison_grid.csv", index=False)

print("\n=== Per-seed mean Task-B RMSE (blind folds), then mean+-sd across 3 seeds ===")
acc_b = acc[~acc.contaminated]
per_seed_rmse = acc_b.groupby(["variant","seed"])["rmse"].mean().reset_index()
print(per_seed_rmse.pivot(index="seed", columns="variant", values="rmse"))
print(per_seed_rmse.groupby("variant")["rmse"].agg(["mean","std"]))

print("\n=== Per-seed mean calendar_viol_rate (blind folds), then mean+-sd across 3 seeds ===")
grid_b = grid[~grid.contaminated]
per_seed_cal = grid_b.groupby(["variant","seed"])["calendar_viol_rate"].mean().reset_index()
print(per_seed_cal.pivot(index="seed", columns="variant", values="calendar_viol_rate"))
print(per_seed_cal.groupby("variant")["calendar_viol_rate"].agg(["mean","std"]))

print("\n=== Per-seed min(dw/dtau) worst case across folds ===")
per_seed_min = grid_b.groupby(["variant","seed"])["min_dwdtau"].min().reset_index()
print(per_seed_min.pivot(index="seed", columns="variant", values="min_dwdtau"))
