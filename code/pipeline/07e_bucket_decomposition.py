"""
Phase 4: error decomposition by moneyness bucket and maturity bucket, for both the
NN (v3, lambda=0.01) and the four Phase 2 baselines (SVI/SSVI/ThinPlate/CubicSpline).

Design choice (why not naively pool every fold): fold test windows are disjoint and
tile the sample contiguously (2021-10-01 -> 2026-01-31 across folds 0-17, confirmed
from 05_splits/folds.csv), so pooling Task B / test-window rows across the 17 blind
folds (all except fold 16, held out separately below) gives each blind test date
exactly one weight -- a valid full-sample decomposition.
Training windows are NOT disjoint (expanding-origin: every fold's train_start is the
same 2020-01-01, only train_end grows), so pooling Task A / train-window rows across
folds would count early dates up to 18x. Instead, Task A is decomposed using ONLY
fold 17 -- the largest-training-window blind fold (train up to 2025-10-01) -- as a
single representative fold; the full per-fold Task A RMSE table already exists in
results/phase4_nn_eval_by_fold.csv for how this varies fold-to-fold.

Fold 16 (Phase-3-contaminated) is reported separately throughout, never pooled into
the "blind" aggregates.

Buckets:
  moneyness (by k = log-moneyness): deep_OTM_put (k<=-0.10), OTM_put (-0.10,-0.03],
    ATM (-0.03,0.03], OTM_call (0.03,0.10], deep_OTM_call (k>0.10)
  maturity (by tau, years): short (<=30d), medium (30-90d], long (>90d)

Usage: python3 07e_bucket_decomposition.py
Outputs:
  results/phase4_bucket_decomposition_nn.csv
  results/phase4_bucket_decomposition_baseline.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
nn_train = import_module("07b_nn_train_v3_fold")

WINDOWS_ROOT = Path(r"E:\Research") / "iv_surface"
BRIDGE_ROOT = Path.home() / "mnt" / "Research" / "iv_surface"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT

SEED = 1
N_LAYERS = len(nn_train.K_LAYER_SIZES) - 1
REPRESENTATIVE_TASK_A_FOLD = 17
CONTAMINATED_FOLD = 16
BLIND_TESTB_FOLDS = [f for f in range(18) if f != CONTAMINATED_FOLD]

MONEYNESS_EDGES = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
MONEYNESS_LABELS = ["deep_OTM_put(k<=-0.10)", "OTM_put(-0.10,-0.03]",
                     "ATM(-0.03,0.03]", "OTM_call(0.03,0.10]", "deep_OTM_call(k>0.10)"]
MATURITY_EDGES = [0, 30 / 365, 90 / 365, np.inf]
MATURITY_LABELS = ["short(<=30d)", "medium(30-90d)", "long(>90d)"]


def bucket_labels(k, tau):
    mny = pd.cut(k, bins=MONEYNESS_EDGES, labels=MONEYNESS_LABELS, right=True)
    mat = pd.cut(tau, bins=MATURITY_EDGES, labels=MATURITY_LABELS, right=True)
    return mny, mat


def load_run(fold_id, seed):
    run_dir = ROOT / "06_nn" / "runs_v3_allfolds" / f"fold{fold_id}_seed{seed}"
    ckpt_path = run_dir / "ckpt.npz"
    params, m, v, key, step = nn_train.load_ckpt(ckpt_path, N_LAYERS)
    ck = np.load(ckpt_path)
    norm_stats = ck["norm_stats"]
    return params, norm_stats, step


def load_meta(con, train_start, train_end, test_start, test_end):
    q = f"""
    SELECT p.k, p.tau, p.iv,
           CASE WHEN p.k >= vr.k_vis_min AND p.k <= vr.k_vis_max THEN true ELSE false END AS in_range_slice
    FROM '{ROOT}/04_iv/panel.parquet' p
    JOIN '{ROOT}/05_splits/held_out_strikes.parquet' h
      ON p.date=h.date AND p.expiry=h.expiry AND p.strike=h.strike
    JOIN '{ROOT}/05_splits/visible_k_range.parquet' vr
      ON p.date=vr.date AND p.expiry=vr.expiry
    WHERE p.date >= '{train_start}' AND p.date < '{train_end}' AND h.is_heldout = true
    ORDER BY p.date, p.expiry, p.strike
    """
    train_ho_meta = con.execute(q).df()
    q2 = f"""
    SELECT date, k, tau, iv
    FROM '{ROOT}/04_iv/panel.parquet'
    WHERE date >= '{test_start}' AND date < '{test_end}'
    ORDER BY date, expiry, strike
    """
    test_meta = con.execute(q2).df()
    return train_ho_meta, test_meta


def nn_predict(params, norm_stats, k, tau):
    k_mean, k_std = [float(x) for x in norm_stats]
    forward_batch, w_scalar = nn_train.make_forward(k_mean, k_std)
    w_pred = np.array(forward_batch(params, jnp.array(k), jnp.array(tau)))
    iv_pred = np.sqrt(np.maximum(w_pred, 1e-12) / np.maximum(tau, 1e-8))
    return iv_pred


def agg_bucket(df, err_col, group_cols, extra=None):
    g = df.groupby(group_cols, observed=True)[err_col]
    out = g.agg(n="count",
                rmse_volpts=lambda s: float(np.sqrt(np.mean(s.values ** 2))),
                mae_volpts=lambda s: float(np.mean(np.abs(s.values)))).reset_index()
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def main():
    folds = pd.read_csv(ROOT / "05_splits" / "folds.csv")
    con = duckdb.connect()
    t0 = time.time()

    # ---------- NN Task B: pooled over blind folds + fold 16 separately ----------
    nn_taskB_rows = []
    for _, frow in folds.iterrows():
        fold_id = int(frow["fold"])
        train_start, train_end = str(frow["train_start"]), str(frow["train_end"])
        test_start, test_end = str(frow["test_start"]), str(frow["test_end"])
        _, test_meta = load_meta(con, train_start, train_end, test_start, test_end)
        if len(test_meta) == 0:
            continue
        params, norm_stats, step = load_run(fold_id, SEED)
        iv_pred = nn_predict(params, norm_stats, test_meta["k"].values, test_meta["tau"].values)
        err_volpts = (iv_pred - test_meta["iv"].values) * 100.0
        mny, mat = bucket_labels(test_meta["k"].values, test_meta["tau"].values)
        d = pd.DataFrame(dict(err_volpts=err_volpts, moneyness=mny, maturity=mat))
        d["fold"] = fold_id
        d["blind"] = fold_id != CONTAMINATED_FOLD
        nn_taskB_rows.append(d)
        print(f"NN Task B fold {fold_id}: n={len(d)} done")
    nn_taskB_all = pd.concat(nn_taskB_rows, ignore_index=True)

    blind_b = agg_bucket(nn_taskB_all[nn_taskB_all["blind"]], "err_volpts",
                          ["moneyness", "maturity"], extra=dict(method="NN_v3", task="B", scope="blind_pooled_17folds"))
    contam_b = agg_bucket(nn_taskB_all[~nn_taskB_all["blind"]], "err_volpts",
                           ["moneyness", "maturity"], extra=dict(method="NN_v3", task="B", scope="fold16_contaminated"))

    # ---------- NN Task A: representative fold 17 only ----------
    frow17 = folds[folds["fold"] == REPRESENTATIVE_TASK_A_FOLD].iloc[0]
    train_ho_meta17, _ = load_meta(con, str(frow17["train_start"]), str(frow17["train_end"]),
                                    str(frow17["test_start"]), str(frow17["test_end"]))
    params17, norm_stats17, step17 = load_run(REPRESENTATIVE_TASK_A_FOLD, SEED)
    iv_pred17 = nn_predict(params17, norm_stats17, train_ho_meta17["k"].values, train_ho_meta17["tau"].values)
    err17_volpts = (iv_pred17 - train_ho_meta17["iv"].values) * 100.0
    mny17, mat17 = bucket_labels(train_ho_meta17["k"].values, train_ho_meta17["tau"].values)
    d17 = pd.DataFrame(dict(err_volpts=err17_volpts, moneyness=mny17, maturity=mat17))
    taskA_fold17 = agg_bucket(d17, "err_volpts", ["moneyness", "maturity"],
                               extra=dict(method="NN_v3", task="A", scope=f"fold{REPRESENTATIVE_TASK_A_FOLD}_representative"))

    nn_out = pd.concat([blind_b, contam_b, taskA_fold17], ignore_index=True)
    nn_out.to_csv(ROOT / "results" / "phase4_bucket_decomposition_nn.csv", index=False)
    print(f"\nNN bucket decomposition done in {time.time()-t0:.1f}s")
    print(nn_out.to_string(index=False))

    # ---------- Baselines: same scope structure, from baseline_predictions_long ----------
    t1 = time.time()
    frow16 = folds[folds["fold"] == CONTAMINATED_FOLD].iloc[0]

    blind_test_conditions = " OR ".join(
        f"(date >= '{str(r['test_start'])}' AND date < '{str(r['test_end'])}')"
        for _, r in folds.iterrows() if int(r["fold"]) != CONTAMINATED_FOLD
    )
    q_blind_b = f"""
    SELECT method, k, tau, err * 100.0 AS err_volpts
    FROM '{ROOT}/results/baseline_predictions_long.parquet'
    WHERE {blind_test_conditions}
    """
    blind_bl = con.execute(q_blind_b).df()
    mny, mat = bucket_labels(blind_bl["k"].values, blind_bl["tau"].values)
    blind_bl["moneyness"], blind_bl["maturity"] = mny, mat
    blind_bl_agg = agg_bucket(blind_bl, "err_volpts", ["method", "moneyness", "maturity"],
                               extra=dict(task="B", scope="blind_pooled_17folds"))

    q_contam_b = f"""
    SELECT method, k, tau, err * 100.0 AS err_volpts
    FROM '{ROOT}/results/baseline_predictions_long.parquet'
    WHERE date >= '{str(frow16["test_start"])}' AND date < '{str(frow16["test_end"])}'
    """
    contam_bl = con.execute(q_contam_b).df()
    mny, mat = bucket_labels(contam_bl["k"].values, contam_bl["tau"].values)
    contam_bl["moneyness"], contam_bl["maturity"] = mny, mat
    contam_bl_agg = agg_bucket(contam_bl, "err_volpts", ["method", "moneyness", "maturity"],
                                extra=dict(task="B", scope="fold16_contaminated"))

    q_taskA17 = f"""
    SELECT method, k, tau, err * 100.0 AS err_volpts
    FROM '{ROOT}/results/baseline_predictions_long.parquet'
    WHERE date >= '{str(frow17["train_start"])}' AND date < '{str(frow17["train_end"])}'
    """
    taskA17_bl = con.execute(q_taskA17).df()
    mny, mat = bucket_labels(taskA17_bl["k"].values, taskA17_bl["tau"].values)
    taskA17_bl["moneyness"], taskA17_bl["maturity"] = mny, mat
    taskA17_bl_agg = agg_bucket(taskA17_bl, "err_volpts", ["method", "moneyness", "maturity"],
                                 extra=dict(task="A", scope=f"fold{REPRESENTATIVE_TASK_A_FOLD}_representative"))

    bl_out = pd.concat([blind_bl_agg, contam_bl_agg, taskA17_bl_agg], ignore_index=True)
    bl_out.to_csv(ROOT / "results" / "phase4_bucket_decomposition_baseline.csv", index=False)
    print(f"\nbaseline bucket decomposition done in {time.time()-t1:.1f}s")
    print(bl_out.to_string(index=False))


if __name__ == "__main__":
    main()
