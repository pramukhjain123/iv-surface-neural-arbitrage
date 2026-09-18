"""Score arm E (soft_cal_only) across seeds -> results/ablation_seed_grid.csv.

Same dense-grid calendar scoring as eval_ablation.py (121 k x 41 tau).
"""
import re, glob
from pathlib import Path
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import train_ablation as TA

ROOT = Path(__file__).resolve().parent
N_LAYERS = len(TA.K_LAYER_SIZES) - 1
VARIANT = "soft_cal_only"

rows = []
for run in sorted(glob.glob(str(ROOT / "runs" / f"{VARIANT}_fold*_seed*"))):
    m = re.search(r"fold(\d+)_seed(\d+)$", run)
    fold_id, seed = int(m.group(1)), int(m.group(2))
    ckpt = Path(run) / "ckpt.npz"
    if not ckpt.exists():
        continue
    params, norm_stats, step = TA.load_ckpt(ckpt, N_LAYERS)
    if step < 18000:
        continue
    k_mean, k_std = [float(x) for x in norm_stats]
    _, w_scalar = TA.make_forward(k_mean, k_std, False)
    dwdtau_fn = jax.grad(w_scalar, argnums=2)

    data = np.load(ROOT / "06_nn" / f"fold{fold_id}_data.npz")
    k_min, k_max = float(np.min(data["train_vis_k"])), float(np.max(data["train_vis_k"]))
    tau_min, tau_max = float(np.min(data["train_vis_tau"])), float(np.max(data["train_vis_tau"]))
    k_pad = (k_max - k_min) * TA.K_PAD_FRAC
    gk = np.linspace(k_min - k_pad, k_max + k_pad, 121)
    gtau = np.linspace(tau_min, tau_max, 41)
    GK, GT = np.meshgrid(gk, gtau, indexing="ij")
    wtau = np.array(jax.vmap(dwdtau_fn, in_axes=(None, 0, 0))(
        params, jnp.array(GK.ravel()), jnp.array(GT.ravel())))
    rows.append(dict(variant=VARIANT, fold=fold_id, seed=seed, step=step,
                     calendar_viol_rate=float(np.mean(wtau < 0)),
                     min_dwdtau=float(wtau.min()),
                     contaminated=(fold_id == 16)))
    print(f"seed={seed} fold={fold_id} cal_viol={rows[-1]['calendar_viol_rate']:.5f} "
          f"min_dwdtau={rows[-1]['min_dwdtau']:+.5f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "ablation_seed_grid.csv", index=False)
print(f"\nwrote {len(df)} rows; seeds={sorted(df.seed.unique())}")
b = df[~df.contaminated]
print(b.groupby("seed").agg(folds=("fold","nunique"),
                            violating=("calendar_viol_rate", lambda s:int((s>0).sum())),
                            mean_pct=("calendar_viol_rate", lambda s: 100*s.mean()),
                            min_dwdtau=("min_dwdtau","min")).to_string())
