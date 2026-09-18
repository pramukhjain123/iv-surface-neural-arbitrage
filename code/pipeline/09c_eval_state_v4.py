"""Evaluate the state-conditioned network on matched held-out strikes."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
model = import_module("09b_train_state_v4")
ROOT = Path(r"E:\Research\iv_surface")


def main():
    rows, arb = [], []
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    for fold_id in folds["fold"]:
        data = np.load(ROOT / "06_nn" / f"fold{fold_id}_state_data.npz")
        params, _, _, _, _, mean, std = model.load(
            ROOT / "06_nn" / "runs_v4_state" / f"fold{fold_id}_seed1" / "ckpt.npz")
        forward, scalar = model.make_forward(mean, std)
        for task, prefix in [("A", "train_ho"), ("B", "test_ho")]:
            args = map(jnp.asarray, [data[f"{prefix}_k"], data[f"{prefix}_tau"],
                data[f"{prefix}_state_atm30"], data[f"{prefix}_state_term_slope"]])
            pred_w = np.asarray(forward(params, *args))
            pred_iv = np.sqrt(np.maximum(pred_w, 1e-12)/data[f"{prefix}_tau"])
            err = (pred_iv-data[f"{prefix}_iv"])*100
            rows.append({"fold": fold_id, "task": task, "n": len(err),
                         "rmse_volpts": np.sqrt(np.mean(err**2)),
                         "mae_volpts": np.mean(np.abs(err)), "contaminated": fold_id == 16})
        dwdk = jax.grad(scalar, argnums=1)
        d2wdk2 = jax.grad(dwdk, argnums=1)
        dwdtau = jax.grad(scalar, argnums=2)
        k0, k1 = data["train_vis_k"].min(), data["train_vis_k"].max()
        pad = (k1-k0)*0.2
        kg = np.linspace(k0-pad, k1+pad, 121)
        tg = np.linspace(data["train_vis_tau"].min(), data["train_vis_tau"].max(), 41)
        states = np.unique(np.column_stack([data["test_all_state_atm30"],
                                             data["test_all_state_term_slope"]]), axis=0)
        states = states[np.linspace(0, len(states)-1, min(len(states), 25)).astype(int)]
        K,T,S = np.meshgrid(kg,tg,np.arange(len(states)),indexing="ij")
        kk,tt,ss = K.ravel(),T.ravel(),S.ravel()
        ll,sl = states[ss,0],states[ss,1]
        axes=(None,0,0,0,0)
        args=(params,jnp.asarray(kk),jnp.asarray(tt),jnp.asarray(ll),jnp.asarray(sl))
        wb=np.asarray(jax.vmap(scalar,in_axes=axes)(*args))
        wk=np.asarray(jax.vmap(dwdk,in_axes=axes)(*args))
        wkk=np.asarray(jax.vmap(d2wdk2,in_axes=axes)(*args))
        wt=np.asarray(jax.vmap(dwdtau,in_axes=axes)(*args))
        ws=np.maximum(wb,1e-6)
        g=(1-kk*wk/(2*ws))**2-(wk**2/4)*(1/ws+0.25)+wkk/2
        arb.append({"fold":fold_id,"n_grid":len(g),"butterfly_viol_rate":np.mean(g<0),
                    "calendar_viol_rate":np.mean(wt<0),"min_dwdtau":wt.min(),
                    "contaminated":fold_id==16})
        print(f"fold={fold_id} A={rows[-2]['rmse_volpts']:.3f} B={rows[-1]['rmse_volpts']:.3f} "
              f"bfly={arb[-1]['butterfly_viol_rate']:.5f}")
    pd.DataFrame(rows).to_csv(ROOT/"results"/"phase5_state_nn_eval_by_fold.csv",index=False)
    pd.DataFrame(arb).to_csv(ROOT/"results"/"phase5_state_nn_arbitrage_by_fold.csv",index=False)


if __name__ == "__main__":
    main()
