"""
Phase 4: v3 (hard-calendar-constrained) training, generalized across the 18 walk-forward
folds with lambda FROZEN at 0.01 (the Phase 3 recommendation) -- no more lambda sweep.

Same architecture and butterfly-penalty formulation as 06f_nn_train_v3.py:
  w(k, tau) = softplus(base(k)) + sum_i softplus(c_i(k)) * phi_i(tau)
with phi_i fixed monotonic basis functions of tau alone, guaranteeing dw/dtau >= 0 by
construction. Butterfly arbitrage is a soft L2 penalty as before, weighted by lambda=0.01.

Usage: python3 07b_nn_train_v3_fold.py <fold_id> <seed> <n_steps_this_call> [<total_target_steps>]
Resumable/checkpointed like 06f: checkpoints in 06_nn/runs_v3_allfolds/fold{fold_id}_seed{seed}/ckpt.npz.
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

LAMBDA = 0.01  # frozen, per Phase 3's recommendation -- no sweep in Phase 4

K_LAYER_SIZES = [1, 32, 32, 4]     # input: normalized k. output: [base_raw, c1_raw, c2_raw, c3_raw]
N_BASIS = 3
DATA_BATCH = 8192
COLLOC_BATCH = 2048
LR = 1e-3
B1, B2, EPS = 0.9, 0.999, 1e-8
K_PAD_FRAC = 0.2
CKPT_EVERY = 150


def init_params(key, sizes):
    params = []
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / sizes[i])
        b = jnp.zeros(sizes[i + 1])
        params.append((w, b))
    return params


def basis(tau):
    phi1 = tau
    phi2 = jnp.sqrt(tau)
    phi3 = jnp.log1p(tau)
    return jnp.stack([phi1, phi2, phi3], axis=-1)


def make_forward(k_mean, k_std):
    def k_net(params, k):
        kn = (k - k_mean) / k_std
        x = jnp.atleast_1d(kn)
        for w, b in params[:-1]:
            x = jax.nn.softplus(x @ w + b)
        w, b = params[-1]
        raw = x @ w + b
        return raw

    def w_scalar(params, k, tau):
        raw = k_net(params, k)
        base = jax.nn.softplus(raw[0])
        coeffs = jax.nn.softplus(raw[1:1 + N_BASIS])
        phis = basis(tau)
        return base + jnp.dot(coeffs, phis)

    def forward_batch(params, k, tau):
        return jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, k, tau)

    return forward_batch, w_scalar


def adam_init(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return m, v


def adam_step(params, grads, m, v, t, lr=LR):
    m = jax.tree_util.tree_map(lambda m_, g: B1 * m_ + (1 - B1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: B2 * v_ + (1 - B2) * g * g, v, grads)
    mhat = jax.tree_util.tree_map(lambda m_: m_ / (1 - B1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v_: v_ / (1 - B2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + EPS), params, mhat, vhat
    )
    return new_params, m, v


def flatten_params(params):
    flat = {}
    for i, (w, b) in enumerate(params):
        flat[f"w{i}"] = np.array(w)
        flat[f"b{i}"] = np.array(b)
    return flat


def unflatten_params(flat, n_layers):
    return [(jnp.array(flat[f"w{i}"]), jnp.array(flat[f"b{i}"])) for i in range(n_layers)]


def save_ckpt(ckpt_path, params, m, v, key, step, norm_stats):
    flat = flatten_params(params)
    m_flat = {f"m_{kk}": vv for kk, vv in flatten_params(m).items()}
    v_flat = {f"v_{kk}": vv for kk, vv in flatten_params(v).items()}
    tmp_path = ckpt_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, step=step, key=np.array(key), norm_stats=np.array(norm_stats),
              **flat, **m_flat, **v_flat)
    tmp_path.replace(ckpt_path)


def load_ckpt(ckpt_path, n_layers):
    ck = np.load(ckpt_path)
    flat = {kk: ck[kk] for kk in ck.files if kk not in ("step", "key", "norm_stats")}
    p_flat = {kk: vv for kk, vv in flat.items() if kk.startswith(("w", "b")) and not kk.startswith(("m_", "v_"))}
    m_flat = {kk[2:]: vv for kk, vv in flat.items() if kk.startswith("m_")}
    v_flat = {kk[2:]: vv for kk, vv in flat.items() if kk.startswith("v_")}
    params = unflatten_params(p_flat, n_layers)
    m = unflatten_params(m_flat, n_layers)
    v = unflatten_params(v_flat, n_layers)
    step0 = int(ck["step"])
    key = jnp.array(ck["key"], dtype=jnp.uint32)
    return params, m, v, key, step0


def main():
    fold_id = int(sys.argv[1])
    seed = int(sys.argv[2])
    n_steps_call = int(sys.argv[3])
    total_target = int(sys.argv[4]) if len(sys.argv) > 4 else n_steps_call

    run_dir = ROOT / "06_nn" / "runs_v3_allfolds" / f"fold{fold_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.npz"
    log_path = run_dir / "log.csv"

    data = np.load(ROOT / "06_nn" / f"fold{fold_id}_data.npz")
    k_tr = jnp.array(data["train_vis_k"])
    tau_tr = jnp.array(data["train_vis_tau"])
    w_tr = jnp.array(data["train_vis_w"])
    n_train = k_tr.shape[0]

    k_mean, k_std = float(np.mean(data["train_vis_k"])), float(np.std(data["train_vis_k"]))
    k_min, k_max = float(np.min(data["train_vis_k"])), float(np.max(data["train_vis_k"]))
    tau_min, tau_max = float(np.min(data["train_vis_tau"])), float(np.max(data["train_vis_tau"]))
    k_pad = (k_max - k_min) * K_PAD_FRAC
    colloc_k_lo, colloc_k_hi = k_min - k_pad, k_max + k_pad
    colloc_tau_lo, colloc_tau_hi = tau_min, tau_max
    norm_stats = [k_mean, k_std]

    forward_batch, w_scalar = make_forward(k_mean, k_std)
    dwdk_fn = jax.grad(w_scalar, argnums=1)
    d2wdk2_fn = jax.grad(dwdk_fn, argnums=1)

    def penalty_terms(params, ck, ctau):
        w_b = jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, ck, ctau)
        wk_b = jax.vmap(dwdk_fn, in_axes=(None, 0, 0))(params, ck, ctau)
        wkk_b = jax.vmap(d2wdk2_fn, in_axes=(None, 0, 0))(params, ck, ctau)
        w_safe = jnp.maximum(w_b, 1e-6)
        g = (1 - ck * wk_b / (2 * w_safe)) ** 2 - (wk_b ** 2 / 4) * (1 / w_safe + 0.25) + wkk_b / 2
        butterfly_viol = jax.nn.relu(-g)
        return jnp.mean(butterfly_viol ** 2)

    def loss_fn(params, bk, btau, bw, ck, ctau, lam):
        pred = forward_batch(params, bk, btau)
        data_mse = jnp.mean((pred - bw) ** 2)
        pen = penalty_terms(params, ck, ctau)
        return data_mse + lam * pen, (data_mse, pen)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jax.jit
    def train_step(params, m, v, t, key, lam):
        key, k1, k2, k3 = jax.random.split(key, 4)
        idx = jax.random.randint(k1, (DATA_BATCH,), 0, n_train)
        bk, btau, bw = k_tr[idx], tau_tr[idx], w_tr[idx]
        ck = jax.random.uniform(k2, (COLLOC_BATCH,), minval=colloc_k_lo, maxval=colloc_k_hi)
        ctau = jax.random.uniform(k3, (COLLOC_BATCH,), minval=colloc_tau_lo, maxval=colloc_tau_hi)
        (loss, (data_mse, pen)), grads = grad_fn(params, bk, btau, bw, ck, ctau, lam)
        params, m, v = adam_step(params, grads, m, v, t)
        return params, m, v, key, loss, data_mse, pen

    n_layers = len(K_LAYER_SIZES) - 1

    if ckpt_path.exists():
        params, m, v, key, step0 = load_ckpt(ckpt_path, n_layers)
        print(f"resumed from step {step0}")
    else:
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        params = init_params(subkey, K_LAYER_SIZES)
        m, v = adam_init(params)
        step0 = 0
        with open(log_path, "w") as f:
            f.write("step,loss,data_mse,penalty,elapsed_s\n")

    t0 = time.time()
    n_do = min(n_steps_call, max(total_target - step0, 0))
    losses = []
    with open(log_path, "a") as f:
        for i in range(n_do):
            t = step0 + i + 1
            params, m, v, key, loss, data_mse, pen = train_step(params, m, v, t, key, LAMBDA)
            if t % 100 == 0 or i == n_do - 1:
                f.write(f"{t},{float(loss):.8f},{float(data_mse):.8f},{float(pen):.8f},{time.time()-t0:.2f}\n")
                f.flush()
                losses.append((t, float(loss), float(data_mse), float(pen)))
            if t % CKPT_EVERY == 0:
                save_ckpt(ckpt_path, params, m, v, key, t, norm_stats)

    step_final = step0 + n_do
    save_ckpt(ckpt_path, params, m, v, key, step_final, norm_stats)

    print(f"fold={fold_id} seed={seed}: trained {n_do} steps ({step0}->{step_final}) in {time.time()-t0:.1f}s")
    if losses:
        print("recent losses (step, loss, data_mse, penalty):")
        for row in losses[-3:]:
            print(f"  {row[0]:6d}  loss={row[1]:.6f}  data_mse={row[2]:.6f}  pen={row[3]:.6f}")
    print(f"DONE={step_final >= total_target}")


if __name__ == "__main__":
    main()
