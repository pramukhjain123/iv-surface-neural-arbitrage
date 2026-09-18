"""
12c_eval_maturity_holdout.py -- Task C: score maturity-direction interpolation.

THE QUESTION
------------
After the Phase-5 revision the network's surviving claim is that it "supplies a
differentiable surface at arbitrary maturities", where the per-slice and
per-day classical methods obtain their smile only at that day's listed
expiries and need a separate rule in between. Task A and Task B both score at
maturities the model saw, so neither tests this. Task C removes the
[30,60]-day band from training and asks every method to reconstruct it.

FAIRNESS -- THE POINT THAT DECIDES THIS EXPERIMENT
--------------------------------------------------
The manuscript's own discussion already concedes that "a shape-preserving
interpolation of theta(tau) could give SSVI similar functionality". Running
the network against classical models that were simply not given an
interpolation rule would therefore prove nothing: it would measure the absence
of a rule, not the value of the architecture. So every classical baseline here
is equipped with the best interpolation rule its own structure admits, and the
rule is monotone (PCHIP) so that it cannot itself introduce calendar
violations:

  SSVI + PCHIP theta   fit SSVI to the day's non-band quotes, then interpolate
                       the ATM total-variance term structure theta(tau) to the
                       target maturity with a shape-preserving monotone cubic.
                       This is exactly the construction the discussion names.
  SVI + PCHIP w        per-expiry SVI fits, then interpolate total variance
                       across tau at each fixed k with the same monotone cubic.
  Fengler + PCHIP w    same, on the constrained-spline curves.
  CubicSpline + PCHIP  same, on the unconstrained per-slice smile fits.
  ThinPlate            needs no rule: it is already a 2-D smoother in (k, tau)
                       and interpolates the gap natively.
  NN v3 (matband)      evaluates its own closed form at the target tau.

SCORING
-------
Only "bracketed" days count -- days that still have an expiry shorter than 30
days and one longer than 60 days after the band is removed -- so this is
interpolation in tau for every method, never extrapolation off the end of the
term structure. Two windows are reported: c1, band slices dated inside the
fold's training window, where the network has seen the date and the daily
methods see the same day's non-band quotes, so the information sets match; and
c2, band slices in the test window, where the static network must additionally
extrapolate in date.

Usage: python3 12c_eval_maturity_holdout.py <fold_id> [<fold_id> ...]
"""
import sys
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

# SLSQP emits bound-clipping warnings on essentially every SSVI fit; the Phase-2
# pipeline produced the same ones. They are noise here and printing them per fit
# dominates the runtime.
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
from scipy.interpolate import CubicSpline, RBFInterpolator, PchipInterpolator
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, _HERE / fn)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


svi_mod = _load("svi04a", "04a_svi.py")
ssvi_mod = _load("ssvi04b", "04b_ssvi.py")
FG = _load("fengler11a", "11a_fengler.py")

ROOT = FG.ROOT
BAND_LO_D, BAND_HI_D = 30, 60
N_BASIS = 3


# --------------------------------------------------------------- NN forward
def nn_predict_w(ckpt_path, k, tau):
    """Evaluate the trained v3 surface. Same closed form as the training code:
    w(k,tau) = softplus(base(k)) + sum_i softplus(c_i(k)) * phi_i(tau)."""
    ck = np.load(ckpt_path)
    n_layers = len([f for f in ck.files if f.startswith("w") and not f.startswith(("w_",))])
    params = [(ck[f"w{i}"], ck[f"b{i}"]) for i in range(n_layers)]
    k_mean, k_std = ck["norm_stats"]

    def softplus(z):
        return np.logaddexp(0.0, z)

    x = ((np.asarray(k, dtype=float) - k_mean) / k_std).reshape(-1, 1)
    for W, b in params[:-1]:
        x = softplus(x @ W + b)
    W, b = params[-1]
    raw = x @ W + b
    base = softplus(raw[:, 0])
    coeffs = softplus(raw[:, 1:1 + N_BASIS])
    tau = np.asarray(tau, dtype=float)
    phis = np.stack([tau, np.sqrt(tau), np.log1p(tau)], axis=-1)
    return base + np.sum(coeffs * phis, axis=-1)


# ------------------------------------------------------- classical baselines
def fit_day_nonband(vis_day, rng):
    """Fit every classical model on one day's NON-band visible quotes."""
    out = {"svi": {}, "cubic": {}, "fengler": {}, "ssvi": None, "thinplate": None,
           "taus": {}}
    expiries = vis_day.groupby("expiry")["tau"].first().sort_values()
    for exp, tau in expiries.items():
        s = vis_day[vis_day["expiry"] == exp].sort_values("k")
        out["taus"][exp] = tau
        if len(s) >= 6:
            best = svi_mod.fit_one(s["k"].to_numpy(), s["w"].to_numpy(), rng)
            if best is not None:
                out["svi"][exp] = best.x
        if len(s) >= 4:
            try:
                out["cubic"][exp] = CubicSpline(s["k"].to_numpy(), s["iv"].to_numpy(),
                                                extrapolate=True)
            except Exception:
                pass

    # Fengler, with its own calendar pass over the non-band expiries
    try:
        fits, _, _ = FG.fit_day(vis_day)
        out["fengler"] = fits
    except Exception:
        out["fengler"] = {}

    if len(vis_day) >= 10:
        exp_tau = expiries
        thetas_raw = []
        for exp, tau in exp_tau.items():
            sub = vis_day[vis_day["expiry"] == exp]
            thetas_raw.append(sub.loc[sub["k"].abs().idxmin(), "w"])
        thetas = ssvi_mod.isotonic_increasing(thetas_raw)
        theta_map = dict(zip(exp_tau.index, thetas))
        vday = vis_day.copy()
        vday["theta"] = vday["expiry"].map(theta_map)
        tv = np.array(list(theta_map.values()))
        cons = [{"type": "ineq", "fun": lambda p, tv=tv: 4.0 - np.max(
            tv * p[1] * np.power(tv, -p[2]) * (1 + abs(p[0])))}]
        best = None
        for x0 in [[-0.5, 0.5, 0.5], [0.0, 1.0, 0.3]]:
            try:
                res = minimize(lambda p: np.sum((ssvi_mod.ssvi_w(
                    vday["k"].to_numpy(), vday["theta"].to_numpy(), p[0], p[1], p[2])
                    - vday["w"].to_numpy()) ** 2),
                    x0=x0, bounds=[(-0.999, 0.999), (1e-4, 5.0), (1e-4, 0.999)],
                    constraints=cons, method="SLSQP", options={"maxiter": 200, "ftol": 1e-12})
                if res.success and np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                    best = res
            except Exception:
                continue
        if best is not None:
            out["ssvi"] = (theta_map, best.x)
        try:
            out["thinplate"] = RBFInterpolator(vis_day[["k", "tau"]].to_numpy(),
                                               vis_day["iv"].to_numpy(),
                                               kernel="thin_plate_spline", smoothing=1e-6)
        except Exception:
            pass
    return out


def pchip_across_tau(taus, values, tau_target):
    """Monotone shape-preserving interpolation in tau. Needs >=2 points and
    tau_target strictly inside the range (bracketed days guarantee this)."""
    taus = np.asarray(taus, float)
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    taus, values = taus[ok], values[ok]
    if len(taus) < 2:
        return np.nan
    order = np.argsort(taus)
    taus, values = taus[order], values[order]
    uniq, idx = np.unique(taus, return_index=True)
    if len(uniq) < 2 or not (uniq[0] <= tau_target <= uniq[-1]):
        return np.nan
    try:
        return float(PchipInterpolator(uniq, values[idx])(tau_target))
    except Exception:
        return np.nan


def predict_band_slice(fitd, kq, tau_target):
    """Every method's implied vol at the band slice's own strikes."""
    preds = {}
    taus_all = fitd["taus"]

    # SVI + PCHIP on total variance at fixed k
    if len(fitd["svi"]) >= 2:
        vals = []
        for kk in kq:
            ts, ws = [], []
            for exp, x in fitd["svi"].items():
                ts.append(taus_all[exp]); ws.append(float(svi_mod.svi_w(kk, *x)))
            vals.append(pchip_across_tau(ts, ws, tau_target))
        preds["SVI+PCHIP"] = _w_to_iv(np.array(vals), tau_target)

    # SSVI + PCHIP on theta(tau) -- the construction the discussion names
    if fitd["ssvi"] is not None:
        theta_map, (rho, eta, gam) = fitd["ssvi"]
        ts = [taus_all[e] for e in theta_map]
        th = [theta_map[e] for e in theta_map]
        theta_t = pchip_across_tau(ts, th, tau_target)
        if np.isfinite(theta_t) and theta_t > 0:
            w = ssvi_mod.ssvi_w(np.asarray(kq), np.full(len(kq), theta_t), rho, eta, gam)
            preds["SSVI+PCHIP"] = _w_to_iv(np.asarray(w), tau_target)

    # Cubic spline + PCHIP
    if len(fitd["cubic"]) >= 2:
        vals = []
        for kk in kq:
            ts, ws = [], []
            for exp, cs in fitd["cubic"].items():
                t = taus_all[exp]
                ts.append(t); ws.append(float(cs(kk)) ** 2 * t)
            vals.append(pchip_across_tau(ts, ws, tau_target))
        preds["CubicSpline+PCHIP"] = _w_to_iv(np.array(vals), tau_target)

    # Fengler + PCHIP
    if len(fitd["fengler"]) >= 2:
        vals = []
        xq = np.exp(np.asarray(kq))
        curves = []
        for exp, (xk, g, gam_, h, tau_e) in fitd["fengler"].items():
            c = FG.spline_eval(xk, g, gam_, h, xq)
            iv = FG.invert_norm_call_vec(c, xq, tau_e)
            curves.append((tau_e, iv ** 2 * tau_e))
        for j in range(len(kq)):
            ts = [t for t, _ in curves]
            ws = [wv[j] for _, wv in curves]
            vals.append(pchip_across_tau(ts, ws, tau_target))
        preds["Fengler+PCHIP"] = _w_to_iv(np.array(vals), tau_target)

    # ThinPlate: native 2-D, no rule needed
    if fitd["thinplate"] is not None:
        pts = np.stack([np.asarray(kq), np.full(len(kq), tau_target)], axis=1)
        preds["ThinPlate"] = np.asarray(fitd["thinplate"](pts), dtype=float)

    return preds


def _w_to_iv(w, tau):
    w = np.asarray(w, dtype=float)
    return np.where(w > 0, np.sqrt(np.maximum(w, 1e-12) / tau), np.nan)


# ------------------------------------------------------------------ driver
def run_fold(fold_id, day_start=0, day_end=10**9, max_days=None):
    npz = np.load(ROOT / "06_nn" / f"fold{fold_id}_matband_data.npz")
    ckpt = ROOT / "06_nn" / "runs_v3_matband" / f"fold{fold_id}_seed1" / "ckpt.npz"
    if not ckpt.exists():
        print(f"fold {fold_id}: no matband checkpoint yet, skipping")
        return None
    # The control that makes this experiment interpretable: the PUBLISHED fold
    # checkpoint, trained WITH the band present, evaluated on the very same band
    # quotes. The gap between the two networks is the cost of the maturity hole
    # and nothing else -- it nets out the static model's overall accuracy level,
    # which is a separate (and already reported) matter. Comparing the held-out
    # network only against the classical baselines would confound the two.
    ckpt_full = ROOT / "06_nn" / "runs_v3_allfolds" / f"fold{fold_id}_seed1" / "ckpt.npz"

    # FG.load() already merges the held-out-strike flag and, crucially, builds
    # the normalised forward moneyness x and call price c that the Fengler
    # comparator needs -- rebuilding those here by hand would risk the two
    # drifting apart.
    panel = FG.load().copy()
    panel["date"] = pd.to_datetime(panel["date"]); panel["expiry"] = pd.to_datetime(panel["expiry"])
    panel["is_heldout"] = panel["is_heldout"].fillna(False).astype(bool)
    panel["k"] = np.log(panel["x"])
    panel["w"] = panel["iv"] ** 2 * panel["tau"]
    panel["dte"] = (panel["tau"] * 365).round().astype(int)
    panel["in_band"] = (panel["dte"] >= BAND_LO_D) & (panel["dte"] <= BAND_HI_D)

    rng = np.random.default_rng(0)
    rows = []

    for which in ("c1", "c2"):
        dates = pd.to_datetime(npz[f"{which}_date"])
        if len(dates) == 0:
            continue
        df = pd.DataFrame({
            "date": dates,
            "expiry": pd.to_datetime(npz[f"{which}_expiry"]),
            "strike": npz[f"{which}_strike"], "k": npz[f"{which}_k"],
            "tau": npz[f"{which}_tau"], "iv": npz[f"{which}_iv"],
            "bracketed": npz[f"{which}_bracketed"]})
        df = df[df["bracketed"]]
        if len(df) == 0:
            continue
        uniq_days = np.sort(df["date"].unique())[day_start:day_end]
        if max_days is not None and len(uniq_days) > max_days:
            # even coverage of the window rather than its first N days, so the
            # score is not tied to one stretch of the volatility history
            uniq_days = uniq_days[np.linspace(0, len(uniq_days) - 1, max_days).astype(int)]
        df = df[df["date"].isin(set(uniq_days))]
        if len(df) == 0:
            continue

        # network predicts everything at once -- no interpolation rule
        w_nn = nn_predict_w(ckpt, df["k"].to_numpy(), df["tau"].to_numpy())
        df["NN_v3_matband"] = np.sqrt(np.maximum(w_nn, 1e-12) / df["tau"].to_numpy())
        if ckpt_full.exists():
            w_full = nn_predict_w(ckpt_full, df["k"].to_numpy(), df["tau"].to_numpy())
            df["NN_v3_trained_with_band"] = np.sqrt(np.maximum(w_full, 1e-12) / df["tau"].to_numpy())

        for d, gsl in df.groupby("date"):
            day = panel[panel["date"] == d]
            vis_nb = day[(~day["in_band"]) & (~day["is_heldout"])].copy()
            if vis_nb["expiry"].nunique() < 2 or len(vis_nb) < 10:
                continue
            disc_ok = True
            try:
                fitd = fit_day_nonband(vis_nb, rng)
            except Exception:
                disc_ok = False
            if not disc_ok:
                continue
            for exp, gb in gsl.groupby("expiry"):
                tau_t = float(gb["tau"].iloc[0])
                kq = gb["k"].to_numpy()
                preds = predict_band_slice(fitd, kq, tau_t)
                preds["NN_v3_matband"] = gb["NN_v3_matband"].to_numpy()
                if "NN_v3_trained_with_band" in gb.columns:
                    preds["NN_v3_trained_with_band"] = gb["NN_v3_trained_with_band"].to_numpy()
                for method, pv in preds.items():
                    pv = np.asarray(pv, dtype=float)
                    for kk, ia, ip in zip(kq, gb["iv"].to_numpy(), pv):
                        rows.append({"fold": fold_id, "window": which, "date": d,
                                     "expiry": exp, "tau": tau_t, "k": kk,
                                     "method": method, "iv_actual": ia, "iv_pred": ip})

    if not rows:
        return None
    out = pd.DataFrame(rows)
    suffix = "" if (day_start == 0 and day_end >= 10**9 and max_days is None) else \
        f"_d{day_start}_{day_end}" + (f"_m{max_days}" if max_days else "")
    out.to_csv(ROOT / "results" / f"phase6_taskC_fold{fold_id}_raw{suffix}.csv", index=False)
    s = out.dropna(subset=["iv_pred"])
    summ = (s.assign(se=(s["iv_pred"] - s["iv_actual"]) ** 2,
                     ae=(s["iv_pred"] - s["iv_actual"]).abs())
              .groupby(["fold", "window", "method"])
              .agg(n=("se", "size"), rmse_volpts=("se", lambda x: np.sqrt(x.mean()) * 100),
                   mae_volpts=("ae", lambda x: x.mean() * 100))
              .reset_index())
    cov = (out.groupby(["fold", "window", "method"])["iv_pred"]
              .apply(lambda x: x.notna().mean()).rename("coverage").reset_index())
    summ = summ.merge(cov, on=["fold", "window", "method"])
    print(f"\n=== fold {fold_id} ===")
    print(summ.to_string(index=False))
    return summ


if __name__ == "__main__":
    args = sys.argv[1:]
    ds, de = 0, 10**9
    md = None
    if "--days" in args:
        i = args.index("--days")
        ds, de = int(args[i + 1]), int(args[i + 2])
        args = args[:i] + args[i + 3:]
    if "--maxdays" in args:
        i = args.index("--maxdays")
        md = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    folds = [int(a) for a in args] or list(range(18))
    parts = [r for r in (run_fold(f, ds, de, md) for f in folds) if r is not None]
    if parts:
        allsum = pd.concat(parts, ignore_index=True)
        allsum.to_csv(ROOT / "results" / "phase6_taskC_by_fold.csv", index=False)
        print("\n=== pooled across folds ===")
        print(allsum.groupby(["window", "method"])
              .agg(folds=("fold", "nunique"), mean_rmse_volpts=("rmse_volpts", "mean"),
                   mean_mae_volpts=("mae_volpts", "mean"), mean_coverage=("coverage", "mean"))
              .round(4).to_string())
