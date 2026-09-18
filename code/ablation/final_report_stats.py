import numpy as np, pandas as pd, duckdb, jax, jax.numpy as jnp
import train_ablation as TA

ROOT = TA.DATA_ROOT
OUT = TA.OUT_ROOT
N_LAYERS = len(TA.K_LAYER_SIZES) - 1
K_PAD_FRAC = TA.K_PAD_FRAC

folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
con = duckdb.connect()
acc_rows, grid_rows = [], []

def load_meta(train_start, train_end, test_start, test_end):
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
    train_ho = con.execute(q).df()
    q2 = f"""
    SELECT date, k, tau, iv FROM '{ROOT}/04_iv/panel.parquet'
    WHERE date >= '{test_start}' AND date < '{test_end}'
    ORDER BY date, expiry, strike
    """
    return train_ho, con.execute(q2).df()

JOBS = [("hard", [1]), ("hard_cal_only", [1,2,3]), ("soft_cal_only", [1,2,3])]

for variant, seeds in JOBS:
    constrain = variant.startswith("hard")
    for seed in seeds:
        for _, frow in folds.iterrows():
            fold_id = int(frow["fold"])
            ckpt = OUT / "runs" / f"{variant}_fold{fold_id}_seed{seed}" / "ckpt.npz"
            if not ckpt.exists():
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

            train_ho_meta, test_meta = load_meta(str(frow["train_start"]), str(frow["train_end"]),
                                                  str(frow["test_start"]), str(frow["test_end"]))
            for task_name, meta in [("A", train_ho_meta), ("B", test_meta)]:
                if len(meta) == 0: continue
                w_pred = np.array(forward_batch(params, jnp.array(meta["k"].values), jnp.array(meta["tau"].values)))
                iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(meta["tau"].values, 1e-8))
                err = (iv_pred - meta["iv"].values) * 100.0
                acc_rows.append(dict(variant=variant, seed=seed, fold=fold_id, task=task_name,
                                      rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.mean(np.abs(err))),
                                      contaminated=(fold_id==16)))

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
                                   frac_w_neg=float(np.mean(w_b < 0)),
                                   contaminated=(fold_id==16)))
        print(f"finished {variant} seed{seed}", flush=True)

acc = pd.DataFrame(acc_rows); grid = pd.DataFrame(grid_rows)
acc.to_csv(OUT / "results" / "final_report_rmse.csv", index=False)
grid.to_csv(OUT / "results" / "final_report_grid.csv", index=False)

print("\n=== TASK A/B RMSE+MAE, blind folds, per-seed mean then agg across seeds ===")
accb = acc[~acc.contaminated]
per_seed = accb.groupby(["variant","seed","task"])[["rmse","mae"]].mean().reset_index()
for task in ["A","B"]:
    print(f"-- Task {task} --")
    sub = per_seed[per_seed.task==task]
    print(sub.pivot(index="seed", columns="variant", values=["rmse","mae"]))
    print(sub.groupby("variant")[["rmse","mae"]].agg(["mean","std"]))

print("\n=== GRID: calendar/butterfly viol rate + min dwdtau + w<0, per-seed mean then agg ===")
gridb = grid[~grid.contaminated]
per_seed_g = gridb.groupby(["variant","seed"])[["calendar_viol_rate","butterfly_viol_rate","frac_w_neg"]].mean().reset_index()
per_seed_min = gridb.groupby(["variant","seed"])["min_dwdtau"].min().reset_index()
print(per_seed_g.pivot(index="seed", columns="variant", values=["calendar_viol_rate","butterfly_viol_rate","frac_w_neg"]))
print(per_seed_g.groupby("variant")[["calendar_viol_rate","butterfly_viol_rate","frac_w_neg"]].agg(["mean","std"]))
print(per_seed_min.pivot(index="seed", columns="variant", values="min_dwdtau"))

print("\n=== Per-fold calendar viol rate, seed1 only (for per-fold table) ===")
pf = gridb[gridb.seed==1].pivot(index="fold", columns="variant", values="calendar_viol_rate")
print(pf)

print("\n=== hard arm (A) full 18-fold check (incl fold16) ===")
acc_a_all = acc[(acc.variant=="hard") & (acc.task=="B")]
print("count(incl fold16)=", len(acc_a_all), "mean_incl=", acc_a_all["rmse"].mean())
grid_a_all = grid[grid.variant=="hard"]
print("cal_viol max=", grid_a_all["calendar_viol_rate"].max(), "bfly_viol max=", grid_a_all["butterfly_viol_rate"].max())
