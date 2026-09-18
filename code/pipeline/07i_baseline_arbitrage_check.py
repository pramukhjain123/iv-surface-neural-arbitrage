"""
Phase 4 follow-up: measure butterfly/calendar arbitrage violation rates for the
four classical baselines (SVI, SSVI, ThinPlate, CubicSpline), on the same footing
as the NN's dense-grid check in 07c_nn_eval_v3_allfolds.py (0% violations across
all 18 folds).

Why this matters: the paper's whole framing is "the NN trades some accuracy for a
guaranteed arbitrage-free surface." That framing has never actually been tested on
the baseline side. SVI and SSVI are already constrained during fitting to
Gatheral-Jacquier sufficient no-butterfly conditions (SSVI additionally enforces
calendar via isotonic theta(tau)), so they may already be close to arbitrage-free
in practice -- which would weaken the NN's differentiation from precisely the two
baselines that also beat it on accuracy. ThinPlate and CubicSpline have no
arbitrage handling at all, so they are the more likely place to find real
violations. This script measures all four directly instead of assuming.

Method (uniform across all four baselines, refit per day since no fitted
parameters were saved from Phase 2 -- only point predictions were):
  - Butterfly (fixed tau, vary k): for each (date, expiry) slice, evaluate each
    method's own w(k) = iv(k)^2 * tau function and its first/second k-derivatives
    on a dense grid over [k_vis_min - 0.2*range, k_vis_max + 0.2*range] (same pad
    convention as 07c), and check the Durrleman condition
        g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2 >= 0
    SVI/SSVI: closed-form derivatives (same algebra as the raw-SVI Durrleman check).
    CubicSpline: scipy's CubicSpline.derivative() gives exact piecewise-polynomial
    derivatives of iv(k); w' and w'' obtained by the product/chain rule.
    ThinPlate: RBFInterpolator has no closed-form derivative, so w', w'' are
    computed by central finite differences in k at the slice's own tau.
  - Calendar (fixed k, vary tau): for each day with >=2 expiries, evaluate each
    method's fitted w(k) function for every expiry that day on a SHARED dense k
    grid (that day's union of visible k ranges, padded), then check that total
    variance is non-decreasing across consecutive expiries (sorted by tau) at
    every shared k point. This is evaluated at the day's actual traded expiries
    (discrete), matching how the panel itself is structured, rather than
    synthesizing a continuous tau interpolant that none of these baselines
    actually define.

Usage: python3 07i_baseline_arbitrage_check.py <day_start_idx> <day_end_idx>
Output: results/arb_chunks/baseline_arb_chunk_<start>_<end>.csv (one row per
        (date, method) with butterfly grid-point violation counts, plus one row
        per (date, method, expiry_pair) for calendar violations).
Then concatenate all chunks with 07j_concat_baseline_arb.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from scipy.interpolate import CubicSpline, RBFInterpolator
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
svi_mod = import_module("04a_svi")
ssvi_mod = import_module("04b_ssvi")

WINDOWS_ROOT = Path(r"E:\Research")
BRIDGE_ROOT = Path.home() / "mnt" / "Research"
ROOT = WINDOWS_ROOT if WINDOWS_ROOT.exists() else BRIDGE_ROOT
D4, D5, RES = (ROOT / "iv_surface" / d for d in ("04_iv", "05_splits", "results"))
ARB_DIR = RES / "arb_chunks_v2"
ARB_DIR.mkdir(parents=True, exist_ok=True)

K_PAD_FRAC = 0.2
K_GRID_N = 61
FD_H = 1e-3  # finite-difference step in k for ThinPlate derivatives


def durrleman_g(k, w, wk, wkk):
    w_safe = np.maximum(w, 1e-8)
    return (1 - k * wk / (2 * w_safe)) ** 2 - (wk ** 2 / 4.0) * (1.0 / w_safe + 0.25) + wkk / 2.0


def svi_w_derivs(k, a, b, rho, m, sigma):
    d = k - m
    s = np.sqrt(d ** 2 + sigma ** 2)
    w = a + b * (rho * d + s)
    wk = b * (rho + d / s)
    wkk = b * sigma ** 2 / s ** 3
    return w, wk, wkk


def ssvi_w_derivs(k, theta, rho, eta, gamma):
    phi = eta * theta ** (-gamma)
    u = phi * k + rho
    S = np.sqrt(u ** 2 + (1 - rho ** 2))
    w = (theta / 2.0) * (1 + rho * phi * k + S)
    wk = (theta / 2.0) * (rho * phi + phi * u / S)
    wkk = (theta / 2.0) * (phi ** 2) * (1 - rho ** 2) / S ** 3
    return w, wk, wkk


def cubic_w_derivs(cs, k, tau):
    iv = cs(k)
    iv1 = cs.derivative(1)(k)
    iv2 = cs.derivative(2)(k)
    w = iv ** 2 * tau
    wk = 2 * iv * iv1 * tau
    wkk = 2 * tau * (iv1 ** 2 + iv * iv2)
    return w, wk, wkk


def thinplate_w_derivs(rbf, k, tau):
    def w_at(kk):
        pts = np.stack([kk, np.full_like(kk, tau)], axis=1)
        iv = rbf(pts)
        return iv ** 2 * tau

    w = w_at(k)
    w_plus = w_at(k + FD_H)
    w_minus = w_at(k - FD_H)
    wk = (w_plus - w_minus) / (2 * FD_H)
    wkk = (w_plus - 2 * w + w_minus) / (FD_H ** 2)
    return w, wk, wkk


def fit_cubic(vis_k, vis_iv):
    return CubicSpline(vis_k, vis_iv, extrapolate=True)


def fit_thinplate(vis_kt, vis_iv):
    return RBFInterpolator(vis_kt, vis_iv, kernel="thin_plate_spline", smoothing=1e-6)


def main(start, end):
    panel = duckdb.connect().execute(
        f"SELECT date, expiry, strike, k, tau, iv FROM read_parquet('{D4 / 'panel.parquet'}')").df()
    panel["w"] = panel["iv"] ** 2 * panel["tau"]
    held = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D5 / 'held_out_strikes.parquet'}')").df()
    panel = panel.merge(held, on=["date", "expiry", "strike"], how="left")
    vis_range = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D5 / 'visible_k_range.parquet'}')").df()

    days = panel["date"].drop_duplicates().sort_values().reset_index(drop=True).iloc[start:end]

    butterfly_rows = []
    calendar_rows = []
    rng = np.random.default_rng(0)

    for d in days:
        gday = panel[panel["date"] == d]
        vis_day = gday[~gday["is_heldout"]]
        if len(vis_day) < 10:
            continue

        # fit per-expiry models (SVI, CubicSpline) and collect for calendar check
        expiries = vis_day.groupby("expiry")["tau"].first().sort_values()
        per_expiry_svi = {}
        per_expiry_cubic = {}
        for exp, tau in expiries.items():
            vis_s = vis_day[vis_day["expiry"] == exp].sort_values("k")
            if len(vis_s) >= 6:
                best = svi_mod.fit_one(vis_s["k"].to_numpy(), vis_s["w"].to_numpy(), rng)
                if best is not None:
                    per_expiry_svi[exp] = (tau, best.x)
            if len(vis_s) >= 4:
                try:
                    cs = fit_cubic(vis_s["k"].to_numpy(), vis_s["iv"].to_numpy())
                    per_expiry_cubic[exp] = (tau, cs)
                except Exception:
                    pass

        # SSVI: one fit for the whole day
        ssvi_fit = None
        if len(vis_day) >= 10:
            exp_tau = vis_day.groupby("expiry")["tau"].first().sort_values()
            thetas_raw = []
            for exp, tau in exp_tau.items():
                sub = vis_day[vis_day["expiry"] == exp]
                idx = sub["k"].abs().idxmin()
                thetas_raw.append(sub.loc[idx, "w"])
            thetas = ssvi_mod.isotonic_increasing(thetas_raw)
            theta_map = dict(zip(exp_tau.index, thetas))
            vday = vis_day.copy()
            vday["theta"] = vday["expiry"].map(theta_map)
            theta_vals = np.array(list(theta_map.values()))
            cons = [{"type": "ineq", "fun": lambda p, tv=theta_vals: 4.0 - np.max(
                tv * p[1] * np.power(tv, -p[2]) * (1 + abs(p[0])))}]
            bounds = [(-0.999, 0.999), (1e-4, 5.0), (1e-4, 0.999)]
            best = None
            for x0 in [[-0.5, 0.5, 0.5], [0.0, 1.0, 0.3]]:
                try:
                    res = minimize(lambda p: np.sum((ssvi_mod.ssvi_w(vday["k"].to_numpy(), vday["theta"].to_numpy(),
                                                                       p[0], p[1], p[2]) - vday["w"].to_numpy()) ** 2),
                                    x0=x0, bounds=bounds, constraints=cons, method="SLSQP",
                                    options={"maxiter": 200, "ftol": 1e-12})
                    if res.success and np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                        best = res
                except Exception:
                    continue
            if best is not None:
                ssvi_fit = (theta_map, best.x)

        # ThinPlate: one fit for the whole day
        thinplate_fit = None
        if len(vis_day) >= 10:
            try:
                thinplate_fit = fit_thinplate(vis_day[["k", "tau"]].to_numpy(), vis_day["iv"].to_numpy())
            except Exception:
                pass

        # --- butterfly check, per (expiry, method) ---
        for exp, tau in expiries.items():
            vr = vis_range[(vis_range["date"] == d) & (vis_range["expiry"] == exp)]
            if len(vr) == 0:
                continue
            kmin, kmax = float(vr["k_vis_min"].iloc[0]), float(vr["k_vis_max"].iloc[0])
            pad = (kmax - kmin) * K_PAD_FRAC
            kgrid = np.linspace(kmin - pad, kmax + pad, K_GRID_N)
            in_observed = (kgrid >= kmin) & (kgrid <= kmax)

            def butterfly_record(method, w, g):
                outside = ~in_observed
                return dict(
                    date=d, expiry=exp, method=method, n_grid=K_GRID_N,
                    n_viol=int(np.sum(g < 0)), n_w_neg=int(np.sum(w < 0)),
                    n_grid_observed=int(np.sum(in_observed)),
                    n_viol_observed=int(np.sum((g < 0) & in_observed)),
                    n_grid_padded=int(np.sum(outside)),
                    n_viol_padded=int(np.sum((g < 0) & outside)),
                )

            if exp in per_expiry_svi:
                _, x = per_expiry_svi[exp]
                w, wk, wkk = svi_w_derivs(kgrid, *x)
                g = durrleman_g(kgrid, w, wk, wkk)
                butterfly_rows.append(butterfly_record("SVI", w, g))

            if exp in per_expiry_cubic:
                _, cs = per_expiry_cubic[exp]
                w, wk, wkk = cubic_w_derivs(cs, kgrid, tau)
                g = durrleman_g(kgrid, w, wk, wkk)
                butterfly_rows.append(butterfly_record("CubicSpline", w, g))

            if ssvi_fit is not None:
                theta_map, (rho, eta, gamma) = ssvi_fit
                if exp in theta_map:
                    theta_e = theta_map[exp]
                    w, wk, wkk = ssvi_w_derivs(kgrid, theta_e, rho, eta, gamma)
                    g = durrleman_g(kgrid, w, wk, wkk)
                    butterfly_rows.append(butterfly_record("SSVI", w, g))

            if thinplate_fit is not None:
                w, wk, wkk = thinplate_w_derivs(thinplate_fit, kgrid, tau)
                g = durrleman_g(kgrid, w, wk, wkk)
                butterfly_rows.append(butterfly_record("ThinPlate", w, g))

        # --- calendar check, per method, across consecutive expiries that day ---
        if len(expiries) >= 2:
            vr_day = vis_range[vis_range["date"] == d]
            if len(vr_day) > 0:
                kmin_day = float(vr_day["k_vis_min"].min())
                kmax_day = float(vr_day["k_vis_max"].max())
                pad_day = (kmax_day - kmin_day) * K_PAD_FRAC
                kgrid_day = np.linspace(kmin_day - pad_day, kmax_day + pad_day, K_GRID_N)
                exp_sorted = list(expiries.items())  # already sorted by tau

                def method_w_curve(method, exp, tau):
                    if method == "SVI" and exp in per_expiry_svi:
                        _, x = per_expiry_svi[exp]
                        return svi_w_derivs(kgrid_day, *x)[0]
                    if method == "CubicSpline" and exp in per_expiry_cubic:
                        _, cs = per_expiry_cubic[exp]
                        return cubic_w_derivs(cs, kgrid_day, tau)[0]
                    if method == "SSVI" and ssvi_fit is not None:
                        theta_map, (rho, eta, gamma) = ssvi_fit
                        if exp in theta_map:
                            return ssvi_w_derivs(kgrid_day, theta_map[exp], rho, eta, gamma)[0]
                    if method == "ThinPlate" and thinplate_fit is not None:
                        return thinplate_w_derivs(thinplate_fit, kgrid_day, tau)[0]
                    return None

                for method in ["SVI", "SSVI", "ThinPlate", "CubicSpline"]:
                    curves = [(exp, tau, method_w_curve(method, exp, tau)) for exp, tau in exp_sorted]
                    curves = [(e, t, w) for e, t, w in curves if w is not None]
                    for i in range(len(curves) - 1)   :
                        e0, t0, w0 = curves[i]
                        e1, t1, w1 = curves[i + 1]
                        n_viol = int(np.sum(w1 < w0 - 1e-10))
                        calendar_rows.append(dict(date=d, method=method, expiry_earlier=e0, expiry_later=e1,
                                                   n_grid=K_GRID_N, n_viol=n_viol))

    but_df = pd.DataFrame(butterfly_rows)
    cal_df = pd.DataFrame(calendar_rows)
    tag = f"{start}_{end}"
    but_df.to_csv(ARB_DIR / f"butterfly_chunk_{tag}.csv", index=False)
    cal_df.to_csv(ARB_DIR / f"calendar_chunk_{tag}.csv", index=False)
    print(f"chunk {tag}: days={len(days)} butterfly_rows={len(but_df)} calendar_rows={len(cal_df)}")
    if len(but_df):
        print(but_df.groupby("method")[["n_viol"]].apply(lambda g: (g["n_viol"] > 0).mean()).rename("frac_slices_with_viol"))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
