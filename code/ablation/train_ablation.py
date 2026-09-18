"""
Isolating ablation for the v3 hard-calendar constraint.

The published v1->v3 comparison changes TWO things at once: the constraint
mechanism (soft penalty -> hard architectural constraint) AND the functional
form (generic MLP in (k,tau) -> strike-network x monotone maturity basis).
This script holds the functional form FIXED at v3's decomposition and varies
ONLY the constraint mechanism, so the constraint can be attributed cleanly.

    w(k, tau) = softplus(base_raw(k)) + sum_i g(c_i_raw(k)) * phi_i(tau)
    phi = [tau, sqrt(tau), log1p(tau)]        (all increasing in tau)

    variant "hard"          : g = softplus  -> coefficients >= 0 -> dw/dtau >= 0
                              by construction. Butterfly penalty only.
                              (exact replication of 07b_nn_train_v3_fold.py)

    variant "soft_broad"    : g = identity  -> coefficients unconstrained.
                              Calendar imposed by soft penalty instead.
                              Collocation identical to "hard" (broad only), so
                              the ONLY difference vs "hard" is the constraint.

    variant "soft_targeted" : as soft_broad, but with 06e_nn_train_v2.py's
                              split collocation (half broad, half targeted at
                              the near-ATM / long-tenor sparse region that was
                              diagnosed as the failure mode). This is the
                              strongest soft-penalty design in the project, so
                              the soft arm fails (if it fails) at its best.

Soft calendar penalty replicates 06e_nn_train_v2.py exactly: L1 hinge with a
strictly-positive margin, relu(CALENDAR_MARGIN - dw/dtau), which was the
project's own fix for the squared-penalty vanishing-gradient problem.

Everything else -- layer sizes, basis, batch sizes, LR, Adam constants,
lambda, step count, seed, fold data -- is shared across all three arms.

Usage: python3 train_ablation.py <variant> <fold_id> <seed> <total_steps> [lambda]

If <lambda> is given it overrides LAMBDA and the run directory is tagged with it,
so sweep runs never collide with the frozen-lambda=0.01 runs.
"""
import sys
import time
from pathlib import Path

import os

import numpy as np
import jax
import jax.numpy as jnp

# DATA_ROOT holds 06_nn/ 05_splits/ 04_iv/ ; OUT_ROOT receives runs/ and results/.
# Both default to this file's folder (the cloud layout) and are overridable so the
# same script runs unmodified against the project tree on the author's machine.
_HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("IVS_DATA_ROOT", _HERE))
OUT_ROOT = Path(os.environ.get("IVS_OUT_ROOT", _HERE))
ROOT = DATA_ROOT  # inputs

# ---- shared hyperparameters (identical to 07b_nn_train_v3_fold.py) ----
LAMBDA = 0.01
K_LAYER_SIZES = [1, 32, 32, 4]
N_BASIS = 3
DATA_BATCH = 8192
COLLOC_BATCH = 2048
LR = 1e-3
B1, B2, EPS = 0.9, 0.999, 1e-8
K_PAD_FRAC = 0.2
CKPT_EVERY = 1000

# ---- soft-calendar-penalty constants (identical to 06e_nn_train_v2.py) ----
CALENDAR_MARGIN = 1e-3
TARGETED_K_HALFWIDTH = 0.1
TARGETED_TAU_FRAC = 0.3

# "hard"/"soft_broad"/"soft_targeted" carry the butterfly penalty, as v3 does.
# The "*_cal_only" pair drops the butterfly penalty entirely. Reason: with
# unconstrained coefficients w loses its positivity floor, and the Durrleman g
# contains 1/w terms, so the butterfly penalty can explode for reasons that have
# nothing to do with calendar monotonicity. The cal_only pair removes that
# confound and isolates the calendar question alone, with a matched hard control.
VARIANTS = ("hard", "soft_broad", "soft_targeted", "soft_cal_only", "hard_cal_only")


def init_params(key, sizes):
    params = []
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / sizes[i])
        b = jnp.zeros(sizes[i + 1])
        params.append((w, b))
    return params


def basis(tau):
    return jnp.stack([tau, jnp.sqrt(tau), jnp.log1p(tau)], axis=-1)


def make_forward(k_mean, k_std, constrain_coeffs):
    """constrain_coeffs=True reproduces v3 (softplus, non-negative coefficients).
       constrain_coeffs=False is the ablation (raw, unconstrained coefficients)."""
    def k_net(params, k):
        kn = (k - k_mean) / k_std
        x = jnp.atleast_1d(kn)
        for w, b in params[:-1]:
            x = jax.nn.softplus(x @ w + b)
        w, b = params[-1]
        return x @ w + b

    def w_scalar(params, k, tau):
        raw = k_net(params, k)
        base = jax.nn.softplus(raw[0])          # positivity of w: kept in BOTH arms
        c_raw = raw[1:1 + N_BASIS]
        coeffs = jax.nn.softplus(c_raw) if constrain_coeffs else c_raw
        return base + jnp.dot(coeffs, basis(tau))

    def forward_batch(params, k, tau):
        return jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, k, tau)

    return forward_batch, w_scalar


def adam_init(params):
    return (jax.tree_util.tree_map(jnp.zeros_like, params),
            jax.tree_util.tree_map(jnp.zeros_like, params))


def adam_step(params, grads, m, v, t, lr=LR):
    m = jax.tree_util.tree_map(lambda m_, g: B1 * m_ + (1 - B1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: B2 * v_ + (1 - B2) * g * g, v, grads)
    mhat = jax.tree_util.tree_map(lambda m_: m_ / (1 - B1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v_: v_ / (1 - B2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + EPS), params, mhat, vhat)
    return params, m, v


def flatten_params(params):
    flat = {}
    for i, (w, b) in enumerate(params):
        flat[f"w{i}"] = np.array(w)
        flat[f"b{i}"] = np.array(b)
    return flat


def unflatten_params(flat, n_layers):
    return [(jnp.array(flat[f"w{i}"]), jnp.array(flat[f"b{i}"])) for i in range(n_layers)]


def save_ckpt(path, params, step, norm_stats, variant):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, step=step, norm_stats=np.array(norm_stats),
             variant=np.array(variant), **flatten_params(params))
    tmp.replace(path)


def load_ckpt(path, n_layers):
    ck = np.load(path, allow_pickle=False)
    flat = {k: ck[k] for k in ck.files if k.startswith(("w", "b"))}
    return unflatten_params(flat, n_layers), ck["norm_stats"], int(ck["step"])


def main():
    variant = sys.argv[1]
    assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
    fold_id, seed, total_steps = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    lam_override = float(sys.argv[5]) if len(sys.argv) > 5 else None
    lam_value = LAMBDA if lam_override is None else lam_override

    constrain = variant.startswith("hard")
    use_calendar_penalty = not constrain
    use_butterfly_penalty = not variant.endswith("cal_only")
    targeted_colloc = (variant == "soft_targeted")

    tag = f"{variant}" if lam_override is None else f"{variant}_lam{lam_override:g}"
    run_dir = OUT_ROOT / "runs" / f"{tag}_fold{fold_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = run_dir / "ckpt.npz", run_dir / "log.csv"

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
    targeted_tau_lo = TARGETED_TAU_FRAC * tau_max
    norm_stats = [k_mean, k_std]

    forward_batch, w_scalar = make_forward(k_mean, k_std, constrain)
    dwdk_fn = jax.grad(w_scalar, argnums=1)
    d2wdk2_fn = jax.grad(dwdk_fn, argnums=1)
    dwdtau_fn = jax.grad(w_scalar, argnums=2)

    def penalty_terms(params, ck, ctau):
        pen = 0.0
        if use_butterfly_penalty:
            w_b = jax.vmap(w_scalar, in_axes=(None, 0, 0))(params, ck, ctau)
            wk_b = jax.vmap(dwdk_fn, in_axes=(None, 0, 0))(params, ck, ctau)
            wkk_b = jax.vmap(d2wdk2_fn, in_axes=(None, 0, 0))(params, ck, ctau)
            w_safe = jnp.maximum(w_b, 1e-6)
            g = (1 - ck * wk_b / (2 * w_safe)) ** 2 - (wk_b ** 2 / 4) * (1 / w_safe + 0.25) + wkk_b / 2
            pen = pen + jnp.mean(jax.nn.relu(-g) ** 2)
        if use_calendar_penalty:
            wtau_b = jax.vmap(dwdtau_fn, in_axes=(None, 0, 0))(params, ck, ctau)
            pen = pen + jnp.mean(jax.nn.relu(CALENDAR_MARGIN - wtau_b))
        return pen

    def loss_fn(params, bk, btau, bw, ck, ctau, lam):
        pred = forward_batch(params, bk, btau)
        data_mse = jnp.mean((pred - bw) ** 2)
        pen = penalty_terms(params, ck, ctau)
        return data_mse + lam * pen, (data_mse, pen)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    half = COLLOC_BATCH // 2

    @jax.jit
    def train_step_broad(params, m, v, t, key, lam):
        # 4-way split, identical to 07b_nn_train_v3_fold.py, so the "hard" arm
        # reproduces the published v3 draw-for-draw and every broad-collocation
        # arm sees the same minibatches and collocation points.
        key, k1, k2, k3 = jax.random.split(key, 4)
        idx = jax.random.randint(k1, (DATA_BATCH,), 0, n_train)
        bk, btau, bw = k_tr[idx], tau_tr[idx], w_tr[idx]
        ck = jax.random.uniform(k2, (COLLOC_BATCH,), minval=colloc_k_lo, maxval=colloc_k_hi)
        ctau = jax.random.uniform(k3, (COLLOC_BATCH,), minval=colloc_tau_lo, maxval=colloc_tau_hi)
        (loss, (data_mse, pen)), grads = grad_fn(params, bk, btau, bw, ck, ctau, lam)
        params, m, v = adam_step(params, grads, m, v, t)
        return params, m, v, key, loss, data_mse, pen

    @jax.jit
    def train_step_targeted(params, m, v, t, key, lam):
        # 6-way split, identical to 06e_nn_train_v2.py's split collocation.
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        idx = jax.random.randint(k1, (DATA_BATCH,), 0, n_train)
        bk, btau, bw = k_tr[idx], tau_tr[idx], w_tr[idx]
        ck_a = jax.random.uniform(k2, (half,), minval=colloc_k_lo, maxval=colloc_k_hi)
        ctau_a = jax.random.uniform(k3, (half,), minval=colloc_tau_lo, maxval=colloc_tau_hi)
        ck_b = jax.random.uniform(k4, (half,), minval=-TARGETED_K_HALFWIDTH,
                                  maxval=TARGETED_K_HALFWIDTH)
        ctau_b = jax.random.uniform(k5, (half,), minval=targeted_tau_lo, maxval=colloc_tau_hi)
        ck = jnp.concatenate([ck_a, ck_b])
        ctau = jnp.concatenate([ctau_a, ctau_b])
        (loss, (data_mse, pen)), grads = grad_fn(params, bk, btau, bw, ck, ctau, lam)
        params, m, v = adam_step(params, grads, m, v, t)
        return params, m, v, key, loss, data_mse, pen

    train_step = train_step_targeted if targeted_colloc else train_step_broad

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    params = init_params(subkey, K_LAYER_SIZES)
    m, v = adam_init(params)

    t0 = time.time()
    with open(log_path, "w") as f:
        f.write("step,loss,data_mse,penalty,elapsed_s\n")
        for i in range(total_steps):
            t = i + 1
            params, m, v, key, loss, data_mse, pen = train_step(params, m, v, t, key, lam_value)
            if t % 500 == 0 or t == total_steps:
                f.write(f"{t},{float(loss):.8f},{float(data_mse):.8f},{float(pen):.8f},"
                        f"{time.time()-t0:.2f}\n")
                f.flush()
            if t % CKPT_EVERY == 0:
                save_ckpt(ckpt_path, params, t, norm_stats, variant)

    save_ckpt(ckpt_path, params, total_steps, norm_stats, variant)
    print(f"{variant} lam={lam_value:g} fold={fold_id} seed={seed}: {total_steps} steps in "
          f"{time.time()-t0:.1f}s  final_loss={float(loss):.6f} "
          f"data_mse={float(data_mse):.6f} pen={float(pen):.6f}", flush=True)


if __name__ == "__main__":
    main()
