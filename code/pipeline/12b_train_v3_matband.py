"""
12b_train_v3_matband.py -- train the v3 hard-calendar-constrained network for
the Task C maturity-holdout experiment.

This is 07b_nn_train_v3_fold.py with exactly two changes:
  * it reads fold{f}_matband_data.npz, whose training set has the [30,60]-day
    maturity band removed, instead of fold{f}_data.npz;
  * it checkpoints into 06_nn/runs_v3_matband/ so the published Phase-4 runs are
    untouched.

Everything else -- architecture, LAMBDA=0.01, batch sizes, learning rate, the
collocation scheme, the 18,000-step budget -- is imported directly from
07b_nn_train_v3_fold rather than copied, so the two cannot silently drift apart
and the comparison stays like-for-like.

Note the collocation range for the butterfly penalty is still drawn over the
FULL tau span of the reduced training set, i.e. it spans the removed band. That
is deliberate: the no-arbitrage penalty is a statement about the surface, not
about where quotes happen to exist, and withholding collocation points inside
the gap would hand the experiment its own answer.

Usage: python3 12b_train_v3_matband.py <fold_id> <seed> <n_steps> [<total_target>]
"""
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("nn07b", _HERE / "07b_nn_train_v3_fold.py")
NN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NN)


def _find_project_root():
    for c in [Path(r"E:\Research") / "iv_surface",
              Path.home() / "mnt" / "Research" / "iv_surface",
              Path.home() / "mnt" / "iv_surface",
              _HERE.parent]:
        if (c / "04_iv" / "panel.parquet").exists():
            return c
    raise SystemExit("could not locate iv_surface project root")


ROOT = _find_project_root()


def main():
    fold_id = int(sys.argv[1])
    seed = int(sys.argv[2])
    n_steps_call = int(sys.argv[3])
    total_target = int(sys.argv[4]) if len(sys.argv) > 4 else n_steps_call

    run_dir = ROOT / "06_nn" / "runs_v3_matband" / f"fold{fold_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.npz"
    log_path = run_dir / "log.csv"

    data = np.load(ROOT / "06_nn" / f"fold{fold_id}_matband_data.npz")
    k_tr = jnp.array(data["train_vis_k"])
    tau_tr = jnp.array(data["train_vis_tau"])
    w_tr = jnp.array(data["train_vis_w"])
    n_train = k_tr.shape[0]

    k_mean, k_std = float(np.mean(data["train_vis_k"])), float(np.std(data["train_vis_k"]))
    k_min, k_max = float(np.min(data["train_vis_k"])), float(np.max(data["train_vis_k"]))
    tau_min, tau_max = float(np.min(data["train_vis_tau"])), float(np.max(data["train_vis_tau"]))
    k_pad = (k_max - k_min) * NN.K_PAD_FRAC
    colloc_k_lo, colloc_k_hi = k_min - k_pad, k_max + k_pad
    colloc_tau_lo, colloc_tau_hi = tau_min, tau_max
    norm_stats = [k_mean, k_std]

    forward_batch, w_scalar = NN.make_forward(k_mean, k_std)
    dwdk_fn = jax.grad(w_scalar, argnums=1)
    d2wdk2_fn = jax.grad(dwdk_fn, argnums=1)

    def penalty_terms(params, ck, ctau):
        w_b = jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, ck, ctau)
        wk_b = jax.vmap(dwdk_fn, in_axes=(None, 0, 0))(params, ck, ctau)
        wkk_b = jax.vmap(d2wdk2_fn, in_axes=(None, 0, 0))(params, ck, ctau)
        w_safe = jnp.maximum(w_b, 1e-6)
        g = (1 - ck * wk_b / (2 * w_safe)) ** 2 - (wk_b ** 2 / 4) * (1 / w_safe + 0.25) + wkk_b / 2
        return jnp.mean(jax.nn.relu(-g) ** 2)

    def loss_fn(params, bk, btau, bw, ck, ctau, lam):
        pred = forward_batch(params, bk, btau)
        data_mse = jnp.mean((pred - bw) ** 2)
        pen = penalty_terms(params, ck, ctau)
        return data_mse + lam * pen, (data_mse, pen)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jax.jit
    def train_step(params, m, v, t, key, lam):
        key, k1, k2, k3 = jax.random.split(key, 4)
        idx = jax.random.randint(k1, (NN.DATA_BATCH,), 0, n_train)
        bk, btau, bw = k_tr[idx], tau_tr[idx], w_tr[idx]
        ck = jax.random.uniform(k2, (NN.COLLOC_BATCH,), minval=colloc_k_lo, maxval=colloc_k_hi)
        ctau = jax.random.uniform(k3, (NN.COLLOC_BATCH,), minval=colloc_tau_lo, maxval=colloc_tau_hi)
        (loss, (data_mse, pen)), grads = grad_fn(params, bk, btau, bw, ck, ctau, lam)
        params, m, v = NN.adam_step(params, grads, m, v, t)
        return params, m, v, key, loss, data_mse, pen

    n_layers = len(NN.K_LAYER_SIZES) - 1
    if ckpt_path.exists():
        params, m, v, key, step0 = NN.load_ckpt(ckpt_path, n_layers)
        print(f"resumed from step {step0}")
    else:
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        params = NN.init_params(subkey, NN.K_LAYER_SIZES)
        m, v = NN.adam_init(params)
        step0 = 0
        with open(log_path, "w") as f:
            f.write("step,loss,data_mse,penalty,elapsed_s\n")

    t0 = time.time()
    n_do = min(n_steps_call, max(total_target - step0, 0))
    with open(log_path, "a") as f:
        for i in range(n_do):
            t = step0 + i + 1
            params, m, v, key, loss, data_mse, pen = train_step(params, m, v, t, key, NN.LAMBDA)
            if t % 100 == 0 or i == n_do - 1:
                f.write(f"{t},{float(loss):.8f},{float(data_mse):.8f},{float(pen):.8f},{time.time()-t0:.2f}\n")
                f.flush()
            if t % NN.CKPT_EVERY == 0:
                NN.save_ckpt(ckpt_path, params, m, v, key, t, norm_stats)

    step_final = step0 + n_do
    NN.save_ckpt(ckpt_path, params, m, v, key, step_final, norm_stats)
    print(f"[matband] fold={fold_id} seed={seed}: {n_do} steps ({step0}->{step_final}) "
          f"in {time.time()-t0:.1f}s  DONE={step_final >= total_target}")


if __name__ == "__main__":
    main()
