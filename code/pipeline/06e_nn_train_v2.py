"""
Phase 3, v2: fixes the calendar-arbitrage penalty problem diagnosed in phase3_notes.md.

Two changes from 06b_nn_train.py (v1):
  1. Calendar penalty is now L1 (mean(relu(-dw/dtau))) instead of squared L2. The
     squared penalty gave a near-zero gradient for the small-but-widespread
     violations that actually occurred (a small negative slope, squared, is tiny
     relative to the data loss), so lambda had almost no reliable effect. L1 gives
     a constant-magnitude gradient regardless of violation size.
  2. Collocation sampling is now split: half sampled broadly (as in v1, k padded
     20% beyond observed range x full observed tau range), half sampled from the
     specific region diagnosed as the failure mode -- near-ATM (|k|<0.1) and
     long-tenor (tau > 0.3*tau_max), where real training data is extremely sparse
     (19 points in the whole 5.5y fold) and the surface previously collapsed
     toward zero, violating monotonicity in tau.

Butterfly penalty is left as squared L2 (v1 already handles it cleanly).

Usage: python3 06e_nn_train_v2.py <lambda> <seed> <n_steps_this_call> [<total_target_steps>]
Resumable the same way as v1: checkpoints saved every CKPT_EVERY steps to
06_nn/runs_v2/lam{lambda}_seed{seed}/ckpt.npz.
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

LAYER_SIZES = [2, 64, 64, 64, 1]
DATA_BATCH = 8192
COLLOC_BATCH = 2048       # split half broad / half targeted
LR = 1e-3
B1, B2, EPS = 0.9, 0.999, 1e-8
K_PAD_FRAC = 0.2
CKPT_EVERY = 150
TARGETED_K_HALFWIDTH = 0.1     # |k| < 0.1 for the targeted half
TARGETED_TAU_FRAC = 0.3        # tau > TARGETED_TAU_FRAC * tau_max for the targeted half
CALENDAR_MARGIN = 1e-3         # require dw/dtau >= MARGIN (not just >=0) during training --
                                # a bare >=0 target lets the optimizer "solve" the penalty by
                                # shrinking |dw/dtau| to a hair's-width negative value near
                                # zero (satisfies the L1 loss almost fully without flipping
                                # sign), especially in the sparse long-tenor region where the
                                # data loss does not resist it. A strictly positive margin
                                # closes that loophole.


def init_params(key, sizes):
    params = []
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / sizes[i])
        b = jnp.zeros(sizes[i + 1])
        params.append((w, b))
    return params


def make_forward(k_mean, k_std, tau_mean, tau_std):
    def forward_batch(params, k, tau):
        kn = (k - k_mean) / k_std
        taun = (tau - tau_mean) / tau_std
        x = jnp.stack([kn, taun], axis=-1)
        for w, b in params[:-1]:
            x = jax.nn.softplus(x @ w + b)
        w, b = params[-1]
        out = jax.nn.softplus(x @ w + b)
        return out[..., 0]

    def w_scalar(params, k, tau):
        kn = (k - k_mean) / k_std
        taun = (tau - tau_mean) / tau_std
        x = jnp.array([kn, taun])
        for w, b in params[:-1]:
            x = jax.nn.softplus(x @ w + b)
        w, b = params[-1]
        out = jax.nn.softplus(x @ w + b)
        return out[0]

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
    lam = float(sys.argv[1])
    seed = int(sys.argv[2])
    n_steps_call = int(sys.argv[3])
    total_target = int(sys.argv[4]) if len(sys.argv) > 4 else n_steps_call

    run_dir = ROOT / "06_nn" / "runs_v2" / f"lam{lam}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.npz"
    log_path = run_dir / "log.csv"

    data = np.load(ROOT / "06_nn" / "fold16_data.npz")
    k_tr = jnp.array(data["train_vis_k"])
    tau_tr = jnp.array(data["train_vis_tau"])
    w_tr = jnp.array(data["train_vis_w"])
    n_train = k_tr.shape[0]

    k_mean, k_std = float(np.mean(data["train_vis_k"])), float(np.std(data["train_vis_k"]))
    tau_mean, tau_std = float(np.mean(data["train_vis_tau"])), float(np.std(data["train_vis_tau"]))
    k_min, k_max = float(np.min(data["train_vis_k"])), float(np.max(data["train_vis_k"]))
    tau_min, tau_max = float(np.min(data["train_vis_tau"])), float(np.max(data["train_vis_tau"]))
    k_pad = (k_max - k_min) * K_PAD_FRAC
    colloc_k_lo, colloc_k_hi = k_min - k_pad, k_max + k_pad
    colloc_tau_lo, colloc_tau_hi = tau_min, tau_max
    targeted_tau_lo = TARGETED_TAU_FRAC * tau_max
    norm_stats = [k_mean, k_std, tau_mean, tau_std]

    half = COLLOC_BATCH // 2

    forward_batch, w_scalar = make_forward(k_mean, k_std, tau_mean, tau_std)
    dwdk_fn = jax.grad(w_scalar, argnums=1)
    d2wdk2_fn = jax.grad(dwdk_fn, argnums=1)
    dwdtau_fn = jax.grad(w_scalar, argnums=2)

    def penalty_terms(params, ck, ctau):
        w_b = jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, ck, ctau)
        wk_b = jax.vmap(dwdk_fn, in_axes=(None, 0, 0))(params, ck, ctau)
        wkk_b = jax.vmap(d2wdk2_fn, in_axes=(None, 0, 0))(params, ck, ctau)
        wtau_b = jax.vmap(dwdtau_fn, in_axes=(None, 0, 0))(params, ck, ctau)
        w_safe = jnp.maximum(w_b, 1e-6)
        g = (1 - ck * wk_b / (2 * w_safe)) ** 2 - (wk_b ** 2 / 4) * (1 / w_safe + 0.25) + wkk_b / 2
        butterfly_viol = jax.nn.relu(-g)
        # calendar: L1 hinge with a strictly-positive margin (see CALENDAR_MARGIN) --
        # gives a constant-magnitude gradient regardless of violation size (unlike
        # squared L2), AND requires dw/dtau to clear a positive threshold rather than
        # letting the optimizer "solve" the penalty by flattening the slope to a
        # hair's-width negative value near zero.
        calendar_viol = jax.nn.relu(CALENDAR_MARGIN - wtau_b)
        return jnp.mean(butterfly_viol ** 2) + jnp.mean(calendar_viol)

    def loss_fn(params, bk, btau, bw, ck, ctau, lam):
        pred = forward_batch(params, bk, btau)
        data_mse = jnp.mean((pred - bw) ** 2)
        pen = penalty_terms(params, ck, ctau)
        return data_mse + lam * pen, (data_mse, pen)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jax.jit
    def train_step(params, m, v, t, key, lam):
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        idx = jax.random.randint(k1, (DATA_BATCH,), 0, n_train)
        bk, btau, bw = k_tr[idx], tau_tr[idx], w_tr[idx]

        # broad half: same as v1
        ck_a = jax.random.uniform(k2, (half,), minval=colloc_k_lo, maxval=colloc_k_hi)
        ctau_a = jax.random.uniform(k3, (half,), minval=colloc_tau_lo, maxval=colloc_tau_hi)
        # targeted half: near-ATM, long-tenor -- the sparse region where the
        # surface previously collapsed toward zero and broke calendar monotonicity
        ck_b = jax.random.uniform(k4, (half,), minval=-TARGETED_K_HALFWIDTH, maxval=TARGETED_K_HALFWIDTH)
        ctau_b = jax.random.uniform(k5, (half,), minval=targeted_tau_lo, maxval=colloc_tau_hi)

        ck = jnp.concatenate([ck_a, ck_b])
        ctau = jnp.concatenate([ctau_a, ctau_b])

        (loss, (data_mse, pen)), grads = grad_fn(params, bk, btau, bw, ck, ctau, lam)
        params, m, v = adam_step(params, grads, m, v, t)
        return params, m, v, key, loss, data_mse, pen

    n_layers = len(LAYER_SIZES) - 1

    if ckpt_path.exists():
        params, m, v, key, step0 = load_ckpt(ckpt_path, n_layers)
        print(f"resumed from step {step0}")
    else:
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        params = init_params(subkey, LAYER_SIZES)
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
            params, m, v, key, loss, data_mse, pen = train_step(params, m, v, t, key, lam)
            if t % 100 == 0 or i == n_do - 1:
                f.write(f"{t},{float(loss):.8f},{float(data_mse):.8f},{float(pen):.8f},{time.time()-t0:.2f}\n")
                f.flush()
                losses.append((t, float(loss), float(data_mse), float(pen)))
            if t % CKPT_EVERY == 0:
                save_ckpt(ckpt_path, params, m, v, key, t, norm_stats)

    step_final = step0 + n_do
    save_ckpt(ckpt_path, params, m, v, key, step_final, norm_stats)

    print(f"lam={lam} seed={seed}: trained {n_do} steps ({step0}->{step_final}) in {time.time()-t0:.1f}s")
    if losses:
        print("recent losses (step, loss, data_mse, penalty):")
        for row in losses[-5:]:
            print(f"  {row[0]:6d}  loss={row[1]:.6f}  data_mse={row[2]:.6f}  pen={row[3]:.6f}")
    print(f"DONE={step_final >= total_target}")


if __name__ == "__main__":
    main()
