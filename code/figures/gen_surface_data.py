import numpy as np, json
from pathlib import Path
import jax, jax.numpy as jnp

ROOT = Path(r"E:\Research\iv_surface")

# ---- v3 (unconditional) ----
K_LAYER_SIZES = [1,32,32,4]; N_BASIS=3
def basis(tau):
    return jnp.stack([tau, jnp.sqrt(tau), jnp.log1p(tau)], axis=-1)
def make_forward_v3(k_mean,k_std):
    def k_net(params,k):
        kn=(k-k_mean)/k_std
        x=jnp.atleast_1d(kn)
        for w,b in params[:-1]:
            x=jax.nn.softplus(x@w+b)
        w,b=params[-1]
        return x@w+b
    def w_scalar(params,k,tau):
        raw=k_net(params,k)
        base=jax.nn.softplus(raw[0])
        coeffs=jax.nn.softplus(raw[1:1+N_BASIS])
        return base+jnp.dot(coeffs, basis(tau))
    return jax.vmap(w_scalar, in_axes=(None,0,0)), w_scalar

def load_v3(path, n_layers):
    ck = np.load(path)
    params = [(jnp.asarray(ck[f"w{i}"]), jnp.asarray(ck[f"b{i}"])) for i in range(n_layers)]
    return params, ck["norm_stats"]

fold_id, seed = 0, 1
ck_v3_path = ROOT/"06_nn"/"runs_v3_allfolds"/f"fold{fold_id}_seed{seed}"/"ckpt.npz"
ck = np.load(ck_v3_path)
print("v3 keys:", list(ck.keys()))
params_v3, norm_stats = load_v3(ck_v3_path, 3)
k_mean, k_std = [float(x) for x in norm_stats]
fwd_v3, _ = make_forward_v3(k_mean, k_std)

data = np.load(ROOT/"06_nn"/f"fold{fold_id}_data.npz")
k_min, k_max = float(data["train_vis_k"].min()), float(data["train_vis_k"].max())
tau_min, tau_max = float(data["train_vis_tau"].min()), float(data["train_vis_tau"].max())
print("k range", k_min, k_max, "tau range", tau_min, tau_max)

gk = np.linspace(k_min, k_max, 60)
gt = np.linspace(tau_min, tau_max, 60)
GK, GT = np.meshgrid(gk, gt, indexing="ij")
w_v3 = np.array(fwd_v3(params_v3, jnp.array(GK.ravel()), jnp.array(GT.ravel()))).reshape(GK.shape)
iv_v3 = np.sqrt(np.maximum(w_v3,1e-12)/np.maximum(GT,1e-8))

# ---- v4 (state-conditioned) ----
SIZES=[3,32,32,4]
def make_forward_v4(mean,std):
    mean,std=jnp.asarray(mean),jnp.asarray(std)
    def scalar(params,k,tau,level,slope):
        x=(jnp.stack([k,level,slope])-mean)/std
        for w,b in params[:-1]:
            x=jax.nn.softplus(x@w+b)
        w,b=params[-1]
        raw=x@w+b
        phi=jnp.stack([tau, jnp.sqrt(tau), jnp.log1p(tau)])
        return jax.nn.softplus(raw[0])+jnp.dot(jax.nn.softplus(raw[1:]), phi)
    return jax.vmap(scalar, in_axes=(None,0,0,0,0)), scalar

ck4_path = ROOT/"06_nn"/"runs_v4_state"/f"fold{fold_id}_seed{seed}"/"ckpt.npz"
ck4 = np.load(ck4_path)
print("v4 keys:", list(ck4.keys()))
n_layers=3
params_v4 = [(jnp.asarray(ck4[f"w{i}"]), jnp.asarray(ck4[f"b{i}"])) for i in range(n_layers)]
mean4, std4 = ck4["norm_mean"], ck4["norm_std"]
fwd_v4, _ = make_forward_v4(mean4, std4)

sdata = np.load(ROOT/"06_nn"/f"fold{fold_id}_state_data.npz")
level_med = float(np.median(sdata["train_vis_state_atm30"]))
slope_med = float(np.median(sdata["train_vis_state_term_slope"]))
print("median level,slope:", level_med, slope_med)

levels = np.full(GK.size, level_med); slopes = np.full(GK.size, slope_med)
w_v4 = np.array(fwd_v4(params_v4, jnp.array(GK.ravel()), jnp.array(GT.ravel()), jnp.array(levels), jnp.array(slopes))).reshape(GK.shape)
iv_v4 = np.sqrt(np.maximum(w_v4,1e-12)/np.maximum(GT,1e-8))

np.savez(ROOT/"tier1_writing_surface_data.npz", GK=GK, GT=GT, iv_v3=iv_v3, iv_v4=iv_v4,
         level_med=level_med, slope_med=slope_med, fold_id=fold_id, seed=seed)
print("saved tier1_writing_surface_data.npz")
