"""Train the state-conditioned hard-calendar-constrained network with JAX."""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(r"E:\Research\iv_surface")
LAMBDA = 0.01
SIZES = [3, 32, 32, 4]
DATA_BATCH, COLLOC_BATCH = 8192, 2048
LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8
K_PAD_FRAC, CKPT_EVERY = 0.2, 150


def init_params(key):
    params = []
    for fan_in, fan_out in zip(SIZES[:-1], SIZES[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (fan_in, fan_out))*jnp.sqrt(2/fan_in),
                       jnp.zeros(fan_out)))
    return params


def make_forward(mean, std):
    mean, std = jnp.asarray(mean), jnp.asarray(std)

    def scalar(params, k, tau, level, slope):
        x = (jnp.stack([k, level, slope])-mean)/std
        for weight, bias in params[:-1]:
            x = jax.nn.softplus(x @ weight + bias)
        weight, bias = params[-1]
        raw = x @ weight + bias
        phi = jnp.stack([tau, jnp.sqrt(tau), jnp.log1p(tau)])
        return jax.nn.softplus(raw[0]) + jnp.dot(jax.nn.softplus(raw[1:]), phi)
    return jax.vmap(scalar, in_axes=(None, 0, 0, 0, 0)), scalar


def flatten(prefix, tree):
    out = {}
    for i, (weight, bias) in enumerate(tree):
        out[f"{prefix}w{i}"] = np.asarray(weight)
        out[f"{prefix}b{i}"] = np.asarray(bias)
    return out


def unflatten(ck, prefix):
    return [(jnp.asarray(ck[f"{prefix}w{i}"]), jnp.asarray(ck[f"{prefix}b{i}"]))
            for i in range(len(SIZES)-1)]


def save(path, params, m, v, key, step, mean, std):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, step=step, key=np.asarray(key), norm_mean=mean, norm_std=std,
             **flatten("", params), **flatten("m_", m), **flatten("v_", v))
    # Windows indexing/antivirus can hold the destination for a few
    # milliseconds. Retrying preserves the atomic checkpoint update.
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def load(path):
    ck = np.load(path)
    return (unflatten(ck, ""), unflatten(ck, "m_"), unflatten(ck, "v_"),
            jnp.asarray(ck["key"], dtype=jnp.uint32), int(ck["step"]),
            ck["norm_mean"], ck["norm_std"])


def adam(params, grads, m, v, step):
    m = jax.tree_util.tree_map(lambda x, g: B1*x+(1-B1)*g, m, grads)
    v = jax.tree_util.tree_map(lambda x, g: B2*x+(1-B2)*g*g, v, grads)
    mh = jax.tree_util.tree_map(lambda x: x/(1-B1**step), m)
    vh = jax.tree_util.tree_map(lambda x: x/(1-B2**step), v)
    params = jax.tree_util.tree_map(lambda p, a, b: p-LR*a/(jnp.sqrt(b)+EPS), params, mh, vh)
    return params, m, v


def main(fold_id, seed, target):
    data = np.load(ROOT / "06_nn" / f"fold{fold_id}_state_data.npz")
    features = np.column_stack([data["train_vis_k"], data["train_vis_state_atm30"],
                                data["train_vis_state_term_slope"]])
    mean, std = features.mean(0), np.maximum(features.std(0), 1e-8)
    k, tau, w, level, slope = map(jnp.asarray, [data["train_vis_k"], data["train_vis_tau"],
        data["train_vis_w"], data["train_vis_state_atm30"], data["train_vis_state_term_slope"]])
    n = len(k)
    klo, khi, tlo, thi = float(k.min()), float(k.max()), float(tau.min()), float(tau.max())
    pad = (khi-klo)*K_PAD_FRAC
    forward, scalar = make_forward(mean, std)
    dwdk = jax.grad(scalar, argnums=1)
    d2wdk2 = jax.grad(dwdk, argnums=1)

    def objective(params, idx, sidx, ck, ct):
        pred = forward(params, k[idx], tau[idx], level[idx], slope[idx])
        wb = forward(params, ck, ct, level[sidx], slope[sidx])
        wk = jax.vmap(dwdk, in_axes=(None,0,0,0,0))(params, ck, ct, level[sidx], slope[sidx])
        wkk = jax.vmap(d2wdk2, in_axes=(None,0,0,0,0))(params, ck, ct, level[sidx], slope[sidx])
        wsafe = jnp.maximum(wb, 1e-6)
        g = (1-ck*wk/(2*wsafe))**2 - (wk**2/4)*(1/wsafe+0.25) + wkk/2
        mse = jnp.mean((pred-w[idx])**2)
        penalty = jnp.mean(jax.nn.relu(-g)**2)
        return mse+LAMBDA*penalty, (mse, penalty)
    vg = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def train_step(params, m, v, step, key):
        key, a, b, c, d = jax.random.split(key, 5)
        idx = jax.random.randint(a, (DATA_BATCH,), 0, n)
        sidx = jax.random.randint(b, (COLLOC_BATCH,), 0, n)
        ck = jax.random.uniform(c, (COLLOC_BATCH,), minval=klo-pad, maxval=khi+pad)
        ct = jax.random.uniform(d, (COLLOC_BATCH,), minval=tlo, maxval=thi)
        (value, aux), grads = vg(params, idx, sidx, ck, ct)
        params, m, v = adam(params, grads, m, v, step)
        return params, m, v, key, value, aux

    run = ROOT / "06_nn" / "runs_v4_state" / f"fold{fold_id}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    path = run / "ckpt.npz"
    if path.exists():
        params, m, v, key, start, mean, std = load(path)
    else:
        key = jax.random.PRNGKey(seed)
        key, sub = jax.random.split(key)
        params = init_params(sub)
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        start = 0
    t0, last = time.time(), None
    for step in range(start+1, target+1):
        params, m, v, key, value, aux = train_step(params, m, v, step, key)
        last = (value, aux)
        if step % CKPT_EVERY == 0:
            save(path, params, m, v, key, step, mean, std)
    save(path, params, m, v, key, target, mean, std)
    print(f"fold={fold_id} steps={start}->{target} elapsed={time.time()-t0:.1f}s "
          f"loss={float(last[0]):.7f} data={float(last[1][0]):.7f} pen={float(last[1][1]):.7f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
