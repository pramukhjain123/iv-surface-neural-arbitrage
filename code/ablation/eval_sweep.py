"""Score the lambda sweep: calendar violations vs fit, per (lambda, fold, seed)."""
import re, glob
from pathlib import Path
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import train_ablation as TA

ROOT = Path(__file__).resolve().parent
N_LAYERS = len(TA.K_LAYER_SIZES) - 1
rows = []
for run in sorted(glob.glob(str(ROOT / "runs" / "soft_cal_only_lam*"))):
    m = re.search(r"lam([\d.]+)_fold(\d+)_seed(\d+)$", run)
    lam, fold_id, seed = float(m.group(1)), int(m.group(2)), int(m.group(3))
    ckpt = Path(run) / "ckpt.npz"
    if not ckpt.exists() or int(np.load(ckpt)["step"]) != 4500:
        continue
    params, norm_stats, step = TA.load_ckpt(ckpt, N_LAYERS)
    k_mean, k_std = [float(x) for x in norm_stats]
    fwd, w_scalar = TA.make_forward(k_mean, k_std, False)
    dwdtau_fn = jax.grad(w_scalar, argnums=2)

    data = np.load(ROOT / "06_nn" / f"fold{fold_id}_data.npz")
    k_min, k_max = float(np.min(data["train_vis_k"])), float(np.max(data["train_vis_k"]))
    tau_min, tau_max = float(np.min(data["train_vis_tau"])), float(np.max(data["train_vis_tau"]))
    k_pad = (k_max - k_min) * TA.K_PAD_FRAC
    gk = np.linspace(k_min - k_pad, k_max + k_pad, 121)
    gtau = np.linspace(tau_min, tau_max, 41)
    GK, GT = np.meshgrid(gk, gtau, indexing="ij")
    gkf, gtf = jnp.array(GK.ravel()), jnp.array(GT.ravel())
    wtau = np.array(jax.vmap(dwdtau_fn, in_axes=(None, 0, 0))(params, gkf, gtf))
    w_b = np.array(jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, gkf, gtf))

    # in-sample fit on the fold's visible training points (w-space RMSE, vol pts)
    kk, tt, ww = data["train_vis_k"], data["train_vis_tau"], data["train_vis_w"]
    idx = np.random.default_rng(0).choice(len(kk), size=min(50000, len(kk)), replace=False)
    pred = np.array(fwd(params, jnp.array(kk[idx]), jnp.array(tt[idx])))
    iv_p = np.sqrt(np.maximum(pred, 1e-12) / np.maximum(tt[idx], 1e-8))
    iv_t = np.sqrt(np.maximum(ww[idx], 1e-12) / np.maximum(tt[idx], 1e-8))
    rmse = float(np.sqrt(np.mean(((iv_p - iv_t) * 100) ** 2)))

    rows.append(dict(lam=lam, fold=fold_id, seed=seed,
                     calendar_viol_rate=float(np.mean(wtau < 0)),
                     min_dwdtau=float(wtau.min()),
                     frac_w_neg=float(np.mean(w_b < 0)),
                     train_rmse_volpts=rmse))

df = pd.DataFrame(rows).sort_values(["lam", "fold", "seed"])
df.to_csv(ROOT / "results" / "ablation_lambda_sweep.csv", index=False)

print("=== lambda sweep: soft_cal_only, 4500 steps, 3 seeds x 2 folds ===\n")
agg = df.groupby("lam").agg(
    runs=("seed", "size"),
    runs_violating=("calendar_viol_rate", lambda s: int((s > 0).sum())),
    mean_viol_pct=("calendar_viol_rate", lambda s: 100 * s.mean()),
    worst_viol_pct=("calendar_viol_rate", lambda s: 100 * s.max()),
    min_dwdtau=("min_dwdtau", "min"),
    mean_w_neg_pct=("frac_w_neg", lambda s: 100 * s.mean()),
    mean_train_rmse=("train_rmse_volpts", "mean"))
print(agg.round(4).to_string())
print("\n=== per fold ===")
print(df.groupby(["lam", "fold"]).agg(
    viol=("calendar_viol_rate", lambda s: int((s > 0).sum())),
    mean_pct=("calendar_viol_rate", lambda s: 100 * s.mean()),
    min_dwdtau=("min_dwdtau", "min"),
    rmse=("train_rmse_volpts", "mean")).round(4).to_string())
