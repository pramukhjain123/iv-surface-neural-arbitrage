import numpy as np, pandas as pd
from pathlib import Path
import jax, jax.numpy as jnp

ROOT = Path(r"E:\Research\iv_surface")
K_LAYER_SIZES=[1,32,32,4]; N_BASIS=3

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

def load_v3(path):
    ck = np.load(path)
    params = [(jnp.asarray(ck[f"w{i}"]), jnp.asarray(ck[f"b{i}"])) for i in range(3)]
    return params, ck["norm_stats"]

folds = pd.read_csv(ROOT/"05_splits"/"folds.csv")
folds["test_start"] = pd.to_datetime(folds["test_start"])
folds["test_end"] = pd.to_datetime(folds["test_end"])

def fold_for_date(date):
    d = pd.Timestamp(date)
    row = folds[(folds.test_start<=d)&(d<folds.test_end)]
    return int(row.iloc[0]["fold"]) if len(row) else None

picks = [("2021-12-23","2021-12-30"), ("2024-01-18","2024-01-25"), ("2025-06-19","2025-06-26")]

bl = pd.read_parquet(ROOT/"results"/"baseline_predictions_long.parquet")
bl["date"] = pd.to_datetime(bl["date"]); bl["expiry"] = pd.to_datetime(bl["expiry"])

out_rows = []
for date_s, exp_s in picks:
    date, exp = pd.Timestamp(date_s), pd.Timestamp(exp_s)
    fold_id = fold_for_date(date)
    print(date_s, "-> fold", fold_id)
    sub = bl[(bl.date==date)&(bl.expiry==exp)][["method","strike","iv_actual","iv_pred","k","tau"]].copy()
    sub["date"] = date_s; sub["expiry"] = exp_s
    out_rows.append(sub)

    # Fengler: search its chunk files for this slice
    found = False
    for chunk in (ROOT/"results"/"fengler_chunks").glob("fengler_pred_*.csv"):
        fdf = pd.read_csv(chunk, parse_dates=["date","expiry"])
        m = fdf[(fdf.date==date)&(fdf.expiry==exp)]
        if len(m):
            m2 = m[["strike","iv_actual","iv_pred","tau"]].copy()
            # join to get k from baseline slice on same strike (same convention)
            m2 = m2.merge(sub[["strike","k"]].drop_duplicates(), on="strike", how="left")
            m2["method"]="Fengler"; m2["date"]=date_s; m2["expiry"]=exp_s
            out_rows.append(m2[["method","strike","iv_actual","iv_pred","k","tau","date","expiry"]])
            found = True
            break
    if not found:
        print("  Fengler: no match found for this slice")

    # NN v3, correct per-fold checkpoint
    ck_path = ROOT/"06_nn"/"runs_v3_allfolds"/f"fold{fold_id}_seed1"/"ckpt.npz"
    params, norm_stats = load_v3(ck_path)
    k_mean,k_std = [float(x) for x in norm_stats]
    fwd,_ = make_forward_v3(k_mean,k_std)
    ks = jnp.array(sub["k"].values); taus = jnp.array(sub["tau"].values)
    w_pred = np.array(fwd(params, ks, taus))
    iv_pred_nn = np.sqrt(np.maximum(w_pred,1e-12)/np.maximum(sub["tau"].values,1e-8))
    nn_rows = sub[["strike","iv_actual","k","tau"]].copy()
    nn_rows["iv_pred"] = iv_pred_nn
    nn_rows["method"]="NN v3"; nn_rows["date"]=date_s; nn_rows["expiry"]=exp_s
    out_rows.append(nn_rows[["method","strike","iv_actual","iv_pred","k","tau","date","expiry"]])

result = pd.concat(out_rows, ignore_index=True)
result.to_csv(ROOT/"tier1_writing_smile_data.csv", index=False)
print(result.groupby(["date","method"]).size())
print("saved tier1_writing_smile_data.csv")
