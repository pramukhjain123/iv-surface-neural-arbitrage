"""
11a_fengler.py -- Fengler (2009) arbitrage-free smoothing of the call-price
surface, added as a fifth classical baseline on exactly the Phase-2/Phase-4
protocol (Task A: fit the visible 80% of strikes per (date,expiry) slice,
predict the held-out 20%).

Reference: M. R. Fengler, "Arbitrage-free smoothing of the implied volatility
surface", Quantitative Finance 9(4), 2009, 417-428.

WHY THIS EXISTS
---------------
The manuscript's related-work section cites Fengler as the established
constrained-nonparametric comparator and, until now, declared its absence a
material baseline limitation. The two unconstrained smoothers actually in the
paper (thin-plate, cubic spline) are *not* substitutes for it: they smooth in
implied-volatility space with no shape constraints at all, which is precisely
why they violate the butterfly condition on 89.4% / 82.0% of slices. Fengler's
method is the honest nonparametric benchmark because it is arbitrage-free by
construction, like the network.

COORDINATES
-----------
Fengler smooths the *call price* surface. We work in normalised coordinates

    x  = K / F_tau                      (forward moneyness; k = log(x))
    c  = C / (exp(-r*tau) * F_tau)      (undiscounted normalised call price)

so that c(x) = N(d1) - x*N(d2), with

    c convex and decreasing in x,   -1 <= dc/dx <= 0,
    max(0, 1-x) <= c <= 1,
    c(x, tau) non-decreasing in tau at fixed x.

Two reasons for this choice rather than raw (K, C):
  1. x is an affine rescaling of K within a slice, so convexity in x is
     exactly convexity in K -- the butterfly condition is unchanged.
  2. fixed x is fixed k = log(K/F_tau), which is the calendar convention the
     rest of the paper uses (Section on the calendar test). The maturity pass
     below therefore enforces the same calendar condition the network
     guarantees, rather than a different one in absolute-strike space.

The panel is OTM-only, so out-of-the-money puts are converted to calls by
put-call parity before fitting:  c = p_norm + 1 - x.

THE QUADRATIC PROGRAM
---------------------
Reinsch representation of a natural cubic smoothing spline on the visible
grid x_1 < ... < x_n, with values g_i = c(x_i) and interior second
derivatives gamma_2..gamma_{n-1} (gamma_1 = gamma_n = 0):

    minimise    1/2 (g-y)' W (g-y)  +  lambda * gamma' R gamma
    subject to  Q' g = R gamma                     (spline consistency)
                gamma >= 0                         (convexity / butterfly)
                (g_2-g_1)/h_1 - (h_1/6) gamma_2 >= -1      (left slope bound)
                (g_n-g_{n-1})/h_{n-1} + (h_{n-1}/6) gamma_{n-1} <= 0
                                                   (right slope bound)
                g_i >= g_{i+1}                     (monotone decreasing)
                max(0, 1-x_i) <= g_i <= 1          (price bounds)
                g <= ub                            (calendar, see below)

Q (n x n-2) and R (n-2 x n-2) are the usual tridiagonal spline matrices.
Convexity plus the two endpoint slope bounds imply -1 <= c' <= 0 on the whole
interval, so the full Fengler constraint set is enforced, not just at nodes.

Weights: W is diagonal with w_i = 1/max(vega_i, VEGA_FLOOR)^2, vega_i being
the normalised Black-76 vega phi(d1)*sqrt(tau) at the *observed* implied vol
of that visible quote. This makes the price-space objective approximately a
least-squares fit in implied-volatility space, which is the space every other
baseline in the paper is scored in; an unweighted price fit would silently
concentrate all accuracy at the money. Weights are normalised to mean 1 so
that lambda has a stable meaning across slices.

CALENDAR PASS
-------------
Fengler's algorithm sweeps maturities and carries the neighbouring fitted
curve in as a bound. We follow the paper's direction: sort that day's expiries
by tau DESCENDING, fit the longest first with no calendar constraint, then fit
each shorter maturity subject to g <= (fitted next-longer curve evaluated at
this slice's x nodes). The bound is applied only at x nodes lying inside the
longer slice's own observed x-range (a small margin allowed), because outside
it the longer curve is an extrapolation and would impose an artificial bound;
those nodes are left calendar-unconstrained and the fact is counted and
reported in the run log rather than hidden.

EXTRAPOLATION
-------------
The dense-grid arbitrage scoring pads the visible k-range by 20% on each side
(same convention as 07c/07i). Fengler's spline is defined only on
[x_1, x_n], so outside it we continue with the endpoint tangent line, clipped
to the price bounds. For a convex function the endpoint tangent preserves
convexity, so the padded region stays butterfly-consistent by construction;
this is a documented choice, not a property of Fengler's method.

Usage:
    python3 11a_fengler.py predict <day_start> <day_end>
    python3 11a_fengler.py lambda_sweep <day_start> <day_end>
Outputs:
    results/fengler_chunks/fengler_pred_<start>_<end>.csv
    results/fengler_chunks/fengler_lamsweep_<start>_<end>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import quadprog
from scipy.stats import norm
from scipy.optimize import brentq

def _find_project_root():
    """Locate the iv_surface project root across Windows and the two bridge
    mount layouts (E:\\Research mounted as .../mnt/Research, or the project
    folder itself mounted as .../mnt/iv_surface)."""
    candidates = [
        Path(r"E:\Research") / "iv_surface",
        Path.home() / "mnt" / "Research" / "iv_surface",
        Path.home() / "mnt" / "iv_surface",
        Path(__file__).resolve().parent.parent,
    ]
    for c in candidates:
        if (c / "04_iv" / "panel.parquet").exists():
            return c
    raise SystemExit("could not locate iv_surface project root")


ROOT = _find_project_root()
D4, D5, RES = (ROOT / d for d in ("04_iv", "05_splits", "results"))
OUT_DIR = RES / "fengler_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_DEFAULT = 1e-8   # selected by nested CV on visible strikes only; see run_lambda_cv
VEGA_FLOOR = 0.02
CAL_MARGIN = 0.02          # allowed x-overshoot when carrying the calendar bound
PRICE_EPS = 1e-9
MIN_VIS = 6                # minimum visible strikes to attempt a slice
K_PAD_FRAC = 0.2           # same padding convention as the dense-grid arbitrage scoring


# ----------------------------------------------------------------------------
# normalised Black-76 in (x, c) coordinates:  c = N(d1) - x N(d2)
# ----------------------------------------------------------------------------
def norm_call(x, sigma, tau):
    s = np.maximum(sigma, 1e-8) * np.sqrt(tau)
    d1 = (-np.log(np.maximum(x, 1e-12)) + 0.5 * s * s) / s
    d2 = d1 - s
    return norm.cdf(d1) - x * norm.cdf(d2)


def norm_vega(x, sigma, tau):
    """d c / d sigma in normalised units."""
    s = np.maximum(sigma, 1e-8) * np.sqrt(tau)
    d1 = (-np.log(np.maximum(x, 1e-12)) + 0.5 * s * s) / s
    return norm.pdf(d1) * np.sqrt(tau)


def invert_norm_call(c, x, tau):
    """Implied vol from a normalised call price. NaN if outside no-arb bounds."""
    lo = max(0.0, 1.0 - x)
    if not np.isfinite(c) or c <= lo + PRICE_EPS or c >= 1.0 - PRICE_EPS:
        return np.nan
    try:
        return brentq(lambda s: norm_call(x, s, tau) - c, 1e-4, 5.0, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def invert_norm_call_vec(c, x, tau, n_iter=60, tol=1e-9):
    """
    Vectorised implied vol from normalised call prices. Newton on vega with a
    bisection safeguard; returns NaN where the price is outside the strict
    no-arbitrage interval (max(0,1-x), 1), which is exactly where no finite
    implied volatility exists. Used for the dense-grid arbitrage scoring, where
    a per-point Brent solve would mean ~1.7M scalar root-finds.
    """
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    lo_b = np.maximum(0.0, 1.0 - x)
    ok = np.isfinite(c) & (c > lo_b + PRICE_EPS) & (c < 1.0 - PRICE_EPS)

    sig = np.full_like(c, 0.25)
    lo = np.full_like(c, 1e-6)
    hi = np.full_like(c, 5.0)
    for _ in range(n_iter):
        f = norm_call(x, sig, tau) - c
        lo = np.where(f < 0, sig, lo)
        hi = np.where(f > 0, sig, hi)
        v = norm_vega(x, sig, tau)
        step = np.where(v > 1e-12, f / np.maximum(v, 1e-12), 0.0)
        cand = sig - step
        bad = ~np.isfinite(cand) | (cand <= lo) | (cand >= hi)
        sig = np.where(bad, 0.5 * (lo + hi), cand)
    resid = np.abs(norm_call(x, sig, tau) - c)
    return np.where(ok & (resid < tol * 10), sig, np.nan)


# ----------------------------------------------------------------------------
# spline machinery
# ----------------------------------------------------------------------------
def spline_matrices(x):
    """Reinsch Q (n x m) and R (m x m), m = n-2, for knots x."""
    n = len(x)
    m = n - 2
    h = np.diff(x)
    Q = np.zeros((n, m))
    R = np.zeros((m, m))
    for j in range(m):
        i = j + 1                      # interior node, 0-based
        Q[i - 1, j] = 1.0 / h[i - 1]
        Q[i, j] = -1.0 / h[i - 1] - 1.0 / h[i]
        Q[i + 1, j] = 1.0 / h[i]
        R[j, j] = (h[i - 1] + h[i]) / 3.0
        if j + 1 < m:
            R[j, j + 1] = h[i] / 6.0
            R[j + 1, j] = h[i] / 6.0
    return Q, R, h


def spline_eval(x_knots, g, gamma_full, h, xq):
    """Evaluate the fitted cubic spline; linear (tangent) extrapolation outside."""
    xq = np.atleast_1d(np.asarray(xq, dtype=float))
    n = len(x_knots)
    out = np.empty_like(xq)

    idx = np.clip(np.searchsorted(x_knots, xq, side="right") - 1, 0, n - 2)
    inside = (xq >= x_knots[0]) & (xq <= x_knots[-1])

    i = idx
    hi = h[i]
    dx = xq - x_knots[i]
    b = (g[i + 1] - g[i]) / hi - hi * (2.0 * gamma_full[i] + gamma_full[i + 1]) / 6.0
    out = (g[i] + dx * b + 0.5 * dx ** 2 * gamma_full[i]
           + (dx ** 3) * (gamma_full[i + 1] - gamma_full[i]) / (6.0 * hi))

    # tangent extrapolation
    left = xq < x_knots[0]
    if np.any(left):
        b0 = (g[1] - g[0]) / h[0] - h[0] * (2.0 * gamma_full[0] + gamma_full[1]) / 6.0
        out[left] = g[0] + (xq[left] - x_knots[0]) * b0
    right = xq > x_knots[-1]
    if np.any(right):
        hn = h[-1]
        bn = (g[-1] - g[-2]) / hn + hn * (gamma_full[-2] + 2.0 * gamma_full[-1]) / 6.0
        out[right] = g[-1] + (xq[right] - x_knots[-1]) * bn

    lo = np.maximum(0.0, 1.0 - xq)
    return np.clip(out, lo, 1.0)


def spline_eval_matrix(x_knots, h, Gamma, xq):
    """
    Build B such that (spline values at xq) = B @ g, exactly matching
    spline_eval's piecewise-cubic interior and tangent extrapolation.

    Why this exists: the calendar condition has to hold at every point of the
    day's shared grid, not merely at the knots of whichever slice is being
    fitted. Each maturity has its own visible strikes, so knot sets differ
    between maturities and imposing g <= (previous curve) only at this slice's
    own knots leaves the two splines free to cross in between -- which is
    precisely what an initial version of this script did, and it showed up as
    thousands of dense-grid calendar crossings despite a feasible QP. Fengler
    avoids this by discretising every maturity on one common grid. Since the
    spline value at ANY point is a linear function of g (gamma = Gamma @ g is
    itself linear), the ordering can instead be imposed directly as linear
    inequality rows at as many shared points as we like, which keeps each
    slice's own knots while making the constraint dense.

    Note the clipping to the price bounds that spline_eval applies is not
    reproduced here: the unclipped value is the linear one, and constraining it
    is strictly tighter, so the resulting curve is still admissible.
    """
    n = len(x_knots)
    xq = np.atleast_1d(np.asarray(xq, dtype=float))
    E = np.eye(n)
    B = np.zeros((len(xq), n))

    idx = np.clip(np.searchsorted(x_knots, xq, side="right") - 1, 0, n - 2)
    for r, (xv, i) in enumerate(zip(xq, idx)):
        if xv < x_knots[0]:
            b0 = (E[1] - E[0]) / h[0] - h[0] * (2.0 * Gamma[0] + Gamma[1]) / 6.0
            B[r] = E[0] + (xv - x_knots[0]) * b0
        elif xv > x_knots[-1]:
            hn = h[-1]
            bn = (E[n - 1] - E[n - 2]) / hn + hn * (Gamma[n - 2] + 2.0 * Gamma[n - 1]) / 6.0
            B[r] = E[n - 1] + (xv - x_knots[-1]) * bn
        else:
            hi = h[i]
            dx = xv - x_knots[i]
            bi = (E[i + 1] - E[i]) / hi - hi * (2.0 * Gamma[i] + Gamma[i + 1]) / 6.0
            B[r] = (E[i] + dx * bi + 0.5 * dx ** 2 * Gamma[i]
                    + (dx ** 3) * (Gamma[i + 1] - Gamma[i]) / (6.0 * hi))
    return B


def gamma_matrix(x):
    """Gamma with gamma_full = Gamma @ g (zero rows at the natural-spline ends)."""
    n = len(x)
    Q, R, h = spline_matrices(x)
    Rinv_Qt = np.linalg.solve(R, Q.T)
    Gamma = np.zeros((n, n))
    Gamma[1:n - 1, :] = Rinv_Qt
    return Gamma, Rinv_Qt, Q, R, h


def fit_fengler_slice(x, y, weights, lam=LAMBDA_DEFAULT, upper_bound=None,
                      dense_cal=None):
    """
    Solve the Fengler QP on one slice.

    The Reinsch system is stated in the module docstring with both g and the
    interior second derivatives gamma as free variables, coupled by the
    equality Q'g = R gamma. Carrying that equality explicitly makes the KKT
    system very badly conditioned whenever adjacent strikes are close together
    (the entries of Q are 1/h and those of R are h/6, so a 50-point strike gap
    on a 12,000 forward puts ~5 orders of magnitude between the two blocks; a
    first-order splitting method simply does not converge on it). Since R is
    tridiagonal, diagonally dominant and positive definite it is invertible, so
    we eliminate gamma analytically:

        gamma = R^{-1} Q' g

    leaving a dense but small and well-conditioned QP in g alone,

        minimise    1/2 g' (W + 2*lam*Q R^{-1} Q') g  -  (W y)' g
        subject to  R^{-1} Q' g >= 0                       (convexity)
                    the two endpoint slope bounds
                    g_i - g_{i+1} >= 0                     (monotone)
                    max(0, 1-x_i) <= g_i <= min(1, ub_i)

    which is solved exactly by the Goldfarb-Idnani dual active-set method
    (quadprog). W is positive definite and lam > 0, so the Hessian is positive
    definite and the solution is unique.

    Returns (g, gamma_full, h) or None on failure.
    """
    n = len(x)
    if n < 4:
        return None
    m = n - 2
    Q, R, h = spline_matrices(x)

    W = np.asarray(weights, dtype=float)
    try:
        Rinv_Qt = np.linalg.solve(R, Q.T)          # (m x n), gamma = Rinv_Qt @ g
    except np.linalg.LinAlgError:
        return None

    G = np.diag(W) + 2.0 * lam * (Q @ Rinv_Qt)
    G = 0.5 * (G + G.T)
    G += np.eye(n) * (1e-12 * max(1.0, float(np.max(np.abs(np.diag(G))))))
    a = W * y

    C_rows, b_vals = [], []

    # convexity: gamma >= 0
    C_rows.append(Rinv_Qt)
    b_vals.append(np.zeros(m))

    # left slope >= -1 : (g_2-g_1)/h_1 - (h_1/6) gamma_2 >= -1
    row = np.zeros(n); row[0] = -1.0 / h[0]; row[1] = 1.0 / h[0]
    row = row - (h[0] / 6.0) * Rinv_Qt[0]
    C_rows.append(row.reshape(1, -1)); b_vals.append(np.array([-1.0]))

    # right slope <= 0 -> negate for >= form
    row = np.zeros(n); row[n - 2] = -1.0 / h[-1]; row[n - 1] = 1.0 / h[-1]
    row = row + (h[-1] / 6.0) * Rinv_Qt[m - 1]
    C_rows.append(-row.reshape(1, -1)); b_vals.append(np.array([0.0]))

    # monotone decreasing
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = 1.0; D[i, i + 1] = -1.0
    C_rows.append(D); b_vals.append(np.zeros(n - 1))

    # price bounds
    lb = np.maximum(0.0, 1.0 - x)
    ub = np.ones(n)
    if upper_bound is not None:
        ub = np.minimum(ub, upper_bound)
    ub = np.maximum(ub, lb + 1e-12)
    C_rows.append(np.eye(n)); b_vals.append(lb)
    C_rows.append(-np.eye(n)); b_vals.append(-ub)

    # dense calendar ordering: B g <= bound  ->  -B g >= -bound
    if dense_cal is not None:
        B_cal, bound_cal = dense_cal
        keep = np.isfinite(bound_cal)
        if np.any(keep):
            C_rows.append(-B_cal[keep]); b_vals.append(-bound_cal[keep])

    Cmat = np.vstack(C_rows)
    bvec = np.concatenate(b_vals)

    try:
        sol = quadprog.solve_qp(G, a, Cmat.T, bvec, 0)
        g = sol[0]
    except Exception:
        return None
    if not np.all(np.isfinite(g)):
        return None

    gamma_full = np.concatenate([[0.0], np.maximum(Rinv_Qt @ g, 0.0), [0.0]])
    return g, gamma_full, h


# ----------------------------------------------------------------------------
# data prep
# ----------------------------------------------------------------------------
def load():
    panel = duckdb.connect().execute(
        f"SELECT date, expiry, strike, k, tau, iv, opt_type, settle_price, fwd, r "
        f"FROM read_parquet('{D4 / 'panel.parquet'}')").df()
    held = duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{D5 / 'held_out_strikes.parquet'}')").df()
    panel = panel.merge(held, on=["date", "expiry", "strike"], how="left")
    panel["x"] = panel["strike"] / panel["fwd"]
    df = np.exp(-panel["r"] * panel["tau"])
    pnorm = panel["settle_price"] / (df * panel["fwd"])
    panel["c"] = np.where(panel["opt_type"] == "CE", pnorm, pnorm + 1.0 - panel["x"])
    return panel[["date", "expiry", "strike", "tau", "iv", "is_heldout", "x", "c"]]


CAL_GRID_N = 81


def day_calendar_grid(gday):
    """Shared per-day evaluation grid for the calendar constraint, spanning the
    day's union visible k-range padded by K_PAD_FRAC -- the same span the
    dense-grid arbitrage scoring uses, so the constraint is enforced exactly
    where the paper measures it."""
    k = np.log(gday["x"].to_numpy(float))
    kmin, kmax = float(np.min(k)), float(np.max(k))
    pad = (kmax - kmin) * K_PAD_FRAC
    return np.exp(np.linspace(kmin - pad, kmax + pad, CAL_GRID_N))


def fit_day(gday, lam=LAMBDA_DEFAULT):
    """
    Fit every expiry of one day, longest maturity first, carrying the calendar
    bound backwards onto a shared dense grid (see spline_eval_matrix for why the
    grid has to be shared rather than per-slice).

    Returns {expiry: (x_knots, g, gamma_full, h, tau)} plus counters:
    how many slices took the dense calendar constraint, and how many had to fall
    back because the dense constraint set made the QP infeasible.
    """
    expiries = gday.groupby("expiry")["tau"].first().sort_values(ascending=False)
    fits = {}
    prev = None
    n_dense = n_fallback_node = n_fallback_none = n_uncon = 0

    xcal_day = day_calendar_grid(gday)

    for exp, tau in expiries.items():
        vis = gday[(gday["expiry"] == exp) & (~gday["is_heldout"])].sort_values("x")
        vis = vis.drop_duplicates("x")
        if len(vis) < MIN_VIS:
            continue
        x = vis["x"].to_numpy(float)
        y = vis["c"].to_numpy(float)
        iv_obs = vis["iv"].to_numpy(float)

        veg = norm_vega(x, iv_obs, tau)
        wts = 1.0 / np.maximum(veg, VEGA_FLOOR) ** 2
        wts = wts / np.mean(wts)

        out = None
        if prev is None:
            out = fit_fengler_slice(x, y, wts, lam=lam)
        else:
            px, pg, pgam, ph = prev
            # Order the two curves only where BOTH are supported by quotes.
            # Beyond the longer maturity's own strike range its curve is a
            # tangent extrapolation, and using that as a hard bound is not a
            # statement the data supports -- it also made the QP infeasible on
            # 12% of slices, because the extrapolated long curve can fall below
            # the short maturity's own observed prices. Points outside the
            # common support are left unconstrained and counted.
            lo_c = max(px[0], x[0])
            hi_c = min(px[-1], x[-1])
            xpts = np.unique(np.concatenate([xcal_day, x]))
            xpts = xpts[(xpts >= lo_c) & (xpts <= hi_c)]
            if len(xpts) == 0:
                out = fit_fengler_slice(x, y, wts, lam=lam)
                if out is not None:
                    n_fallback_none += 1
                    g, gamma_full, h = out
                    fits[exp] = (x, g, gamma_full, h, tau)
                    prev = (x, g, gamma_full, h)
                continue
            n_uncon += int(np.sum((x < lo_c) | (x > hi_c)))
            bound = spline_eval(px, pg, pgam, ph, xpts)
            Gamma, _, _, _, h_cur = gamma_matrix(x)
            B = spline_eval_matrix(x, h_cur, Gamma, xpts)
            out = fit_fengler_slice(x, y, wts, lam=lam, dense_cal=(B, bound))
            if out is not None:
                n_dense += 1
            else:
                # dense set infeasible: fall back to knot-level ordering, then none
                ub = np.full(len(x), np.inf)
                inside = (x >= px[0] - CAL_MARGIN) & (x <= px[-1] + CAL_MARGIN)
                if np.any(inside):
                    ub[inside] = spline_eval(px, pg, pgam, ph, x[inside])
                out = fit_fengler_slice(x, y, wts, lam=lam, upper_bound=ub)
                if out is not None:
                    n_fallback_node += 1
                else:
                    out = fit_fengler_slice(x, y, wts, lam=lam)
                    if out is not None:
                        n_fallback_none += 1

        if out is None:
            continue
        g, gamma_full, h = out
        fits[exp] = (x, g, gamma_full, h, tau)
        prev = (x, g, gamma_full, h)

    return fits, n_dense, (n_fallback_node, n_fallback_none, n_uncon)


# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------
def run_predict(start, end, lam=LAMBDA_DEFAULT, tag_extra=""):
    panel = load()
    days = panel["date"].drop_duplicates().sort_values().reset_index(drop=True).iloc[start:end]
    daysel = set(days)
    rows = []
    n_slices = n_failed = 0
    tot_applied = tot_skipped = 0

    for d, gday in panel.groupby("date", sort=True):
        if d not in daysel:
            continue
        fits, na, ns = fit_day(gday, lam=lam)
        tot_applied += na; tot_skipped += (ns[0] + ns[1])
        for exp, (xk, g, gam, h, tau) in fits.items():
            n_slices += 1
            ho = gday[(gday["expiry"] == exp) & (gday["is_heldout"])]
            if len(ho) == 0:
                continue
            xq = ho["x"].to_numpy(float)
            cq = spline_eval(xk, g, gam, h, xq)
            for strike, xv, cv, iv_a in zip(ho["strike"], xq, cq, ho["iv"]):
                iv_p = invert_norm_call(cv, xv, tau)
                if not np.isfinite(iv_p):
                    n_failed += 1
                rows.append({"method": "Fengler", "date": d, "expiry": exp,
                             "strike": strike, "iv_actual": iv_a, "iv_pred": iv_p,
                             "x": xv, "c_pred": cv, "c_lo": max(0.0, 1.0 - xv),
                             "tau": tau})

    out = pd.DataFrame(rows)
    tag = f"{start}_{end}{tag_extra}"
    out.to_csv(OUT_DIR / f"fengler_pred_{tag}.csv", index=False)
    ok = out["iv_pred"].notna()
    rmse = np.sqrt(np.mean((out.loc[ok, "iv_pred"] - out.loc[ok, "iv_actual"]) ** 2)) * 100 if ok.any() else float("nan")
    print(f"chunk {tag}: days={len(days)} slices_fit={n_slices} pred_rows={len(out)} "
          f"inversion_failed={n_failed} RMSE={rmse:.4f} vol pts | "
          f"calendar: dense-constrained slices={tot_applied} fell_back={tot_skipped}")


def run_lambda_cv(n_pilot=120, seed=7):
    """
    Select the smoothing parameter WITHOUT touching the Task A held-out strikes.

    Every other baseline in the paper either has no free smoothing parameter
    (SVI, SSVI) or uses a fixed one (thin-plate, smoothing=1e-6). Fengler's
    lambda has to come from somewhere, and tuning it on the held-out 20% would
    give this baseline an advantage no other method gets. So lambda is chosen by
    a NESTED split: inside each slice's *visible* 80%, a further random 20% of
    those visible strikes is withheld, the spline is fit on the rest at each
    candidate lambda, and the withheld visible strikes are predicted. The Task A
    held-out strikes play no part. One global lambda is then fixed for the whole
    panel from a pilot sample of days spread evenly across the 6.1 years.
    """
    panel = load()
    all_days = panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    idx = np.linspace(0, len(all_days) - 1, n_pilot).astype(int)
    days = all_days.iloc[idx]
    rng = np.random.default_rng(seed)

    lams = [0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    acc = {lam: [] for lam in lams}
    nfail = {lam: 0 for lam in lams}
    ntot = 0

    for d in days:
        gday = panel[panel["date"] == d]
        for exp, tau in gday.groupby("expiry")["tau"].first().sort_values(ascending=False).items():
            vis = gday[(gday["expiry"] == exp) & (~gday["is_heldout"])].sort_values("x").drop_duplicates("x")
            if len(vis) < MIN_VIS + 3:
                continue
            nv = len(vis)
            inner_ho = rng.random(nv) < 0.20
            if inner_ho.sum() == 0 or (~inner_ho).sum() < MIN_VIS:
                continue
            # inner held-out points must be interior, else we are scoring extrapolation
            interior = np.zeros(nv, dtype=bool)
            interior[1:-1] = True
            inner_ho = inner_ho & interior
            if inner_ho.sum() == 0:
                continue

            sub = vis[~inner_ho]
            tst = vis[inner_ho]
            x = sub["x"].to_numpy(float); y = sub["c"].to_numpy(float)
            iv_obs = sub["iv"].to_numpy(float)
            veg = norm_vega(x, iv_obs, tau)
            wts = 1.0 / np.maximum(veg, VEGA_FLOOR) ** 2
            wts = wts / np.mean(wts)
            xq = tst["x"].to_numpy(float); iva = tst["iv"].to_numpy(float)
            ntot += len(xq)

            for lam in lams:
                out = fit_fengler_slice(x, y, wts, lam=lam)
                if out is None:
                    nfail[lam] += len(xq)
                    continue
                g, gam, h = out
                cq = spline_eval(x, g, gam, h, xq)
                for xv, cv, ia in zip(xq, cq, iva):
                    ip = invert_norm_call(cv, xv, tau)
                    if np.isfinite(ip):
                        acc[lam].append((ip - ia) ** 2)
                    else:
                        nfail[lam] += 1

    recs = []
    for lam in lams:
        e = acc[lam]
        rmse = float(np.sqrt(np.mean(e)) * 100) if e else float("nan")
        recs.append({"lambda": lam, "n_scored": len(e), "n_fail": nfail[lam],
                     "n_inner_heldout": ntot,
                     "fail_rate": nfail[lam] / max(ntot, 1), "rmse_volpts": rmse})
        print(f"lambda={lam:g}: scored={len(e)} fail={nfail[lam]}/{ntot} "
              f"({100*nfail[lam]/max(ntot,1):.2f}%) inner-CV RMSE={rmse:.4f}")
    df = pd.DataFrame(recs)
    df.to_csv(OUT_DIR / "fengler_lambda_cv.csv", index=False)
    best = df.loc[df["rmse_volpts"].idxmin()]
    print(f"\nselected lambda = {best['lambda']:g} (inner-CV RMSE {best['rmse_volpts']:.4f} vol pts, "
          f"{n_pilot} pilot days)")


def run_lambda_sweep(start, end):
    panel = load()
    days = panel["date"].drop_duplicates().sort_values().reset_index(drop=True).iloc[start:end]
    recs = []
    for lam in [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        errs = []
        nfail = ntot = 0
        for d in days:
            gday = panel[panel["date"] == d]
            fits, _, _ = fit_day(gday, lam=lam)
            for exp, (xk, g, gam, h, tau) in fits.items():
                ho = gday[(gday["expiry"] == exp) & (gday["is_heldout"])]
                if len(ho) == 0:
                    continue
                xq = ho["x"].to_numpy(float)
                cq = spline_eval(xk, g, gam, h, xq)
                for xv, cv, iv_a in zip(xq, cq, ho["iv"]):
                    ntot += 1
                    iv_p = invert_norm_call(cv, xv, tau)
                    if np.isfinite(iv_p):
                        errs.append((iv_p - iv_a) ** 2)
                    else:
                        nfail += 1
        rmse = float(np.sqrt(np.mean(errs)) * 100) if errs else float("nan")
        recs.append({"lambda": lam, "n": len(errs), "n_fail": nfail, "n_tot": ntot,
                     "fail_rate": nfail / max(ntot, 1), "rmse_volpts": rmse})
        print(f"lambda={lam:g}: n={len(errs)} fail={nfail}/{ntot} ({100*nfail/max(ntot,1):.1f}%) RMSE={rmse:.4f}")
    pd.DataFrame(recs).to_csv(OUT_DIR / f"fengler_lamsweep_{start}_{end}.csv", index=False)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "predict":
        run_predict(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "lambda_cv":
        run_lambda_cv()
    elif mode == "lambda_sweep":
        run_lambda_sweep(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(f"unknown mode {mode}")
