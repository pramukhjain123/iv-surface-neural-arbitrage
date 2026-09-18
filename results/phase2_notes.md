# Phase 2 — Splits and Baselines

## 2.1 Walk-forward folds
Expanding-origin, 18m/3m/3m (train/val/test), stepping 3 months, from the panel's
first date (2020-01-01) through its last (2026-01-30): **18 folds** (`05_splits/folds.csv`),
fold 17 partial (only ~1 month of test data left at the sample end).

Task A (interpolation): a **fixed** random 20% of strikes per (date,expiry) slice held
out, seed=42 (`05_splits/held_out_strikes.parquet`) -- every baseline (and later, the
network) is scored on the exact same held-out points, so the comparison is apples-to-apples.
Task B (extrapolation in time): not separately materialized -- it's just "does this
(date,expiry) fall in fold k's test window," already captured by `slice_fold_membership.parquet`.
Task B only really differentiates methods that use historical data to fit (the neural
network, Phase 3); none of the four classical baselines below use any information outside
their own single day, so Task B does not apply to them -- noted, not silently skipped.

## 2.2 Four baselines, fit on the 80% visible strikes, scored on the 20% held-out
| method | fit unit | information used | n predicted |
|---|---|---|---|
| SVI (raw, 5-param) | per (date,expiry) slice | that slice only | 57,727 |
| SSVI (power-law, 3-param + calibrated theta(tau)) | per day | all of that day's expiries pooled | 57,727 |
| Thin-plate spline (RBF) | per day | all of that day's expiries pooled | 57,719 |
| Cubic spline in k | per (date,expiry) slice | that slice only | 57,727 |

SVI and SSVI are both constrained (SLSQP) to Gatheral & Jacquier (2014)'s sufficient
no-butterfly condition; SSVI's theta(tau) term is additionally isotonic-regressed to be
non-decreasing (no calendar arbitrage) before (rho,eta,gamma) are fit. **100% of SVI and
SSVI fits were feasible** (satisfied the no-arbitrage condition) -- no infeasible slices
anywhere in the sample, across all ~9,438 slices / 1,496 days.

## An important scoring fix: interpolation vs extrapolation
The first pass scored cubic-spline naively and got RMSE = 118 vol points -- nonsense.
Root cause: ~7% of the fixed 20% held-out strikes fall *outside* the visible-strike range
for their slice (e.g. the single highest or lowest listed strike happened to be drawn into
the held-out set), which is extrapolation, not interpolation -- and naive cubic-spline
extrapolation is known to be wild. Since Task A is explicitly named "interpolation," the
right fix is to score it as such: `05_splits/visible_k_range.parquet` records each slice's
visible k-range, and every table below reports **in-range ("true interpolation") and
out-of-range ("extrapolation") separately**, rather than one blended number that would
either hide the blow-up or unfairly penalize every method for a cubic-spline-specific
failure mode.

## Results (`results/baseline_comparison_summary.csv`, `phase2_baseline_comparison.png`)
| method | RMSE, in-range (vol pts) | MAE, in-range | RMSE, out-of-range (vol pts) |
|---|---|---|---|
| ThinPlate | **0.42** | 0.13 | 2.43 |
| CubicSpline | 1.35 | 0.19 | **446.07** |
| SSVI | 2.16 | 1.46 | 4.57 |
| SVI | 2.39 | 0.98 | 4.73 |

Two findings worth being upfront about, because they shape how Phase 3/4 should be framed:

1. **The unconstrained, fully flexible methods (ThinPlate, CubicSpline) interpolate the
   raw market data far more accurately than the two arbitrage-constrained parametric
   methods (SVI, SSVI).** This is expected, not a bug: a 5-parameter (SVI) or 3-parameter
   (SSVI) functional form is a much tighter constraint on the fitted shape than a
   near-unregularized RBF/spline that can bend through almost every training point. The
   apples-to-apples comparison is SSVI vs ThinPlate (both pool the whole day's expiries) --
   SSVI is ~5x worse in RMSE than ThinPlate for that reason alone, purely from the
   arbitrage-consistent functional form being restrictive. **This is exactly the fit-error-
   vs-arbitrage-freedom trade-off the plan's Phase 3 ablation is designed to quantify** --
   none of these four baselines get both low fit error AND a guarantee of no arbitrage at
   the same time; that combination is the paper's actual contribution to test for.
2. **Cubic spline's extrapolation failure (446 vol points RMSE on ~7% of held-out points)
   is a real, large effect, not a rounding issue** -- naive per-slice cubic splines have no
   mechanism to behave sensibly past their fitted strike range, unlike SVI's asymptotically
   linear total-variance wings (which degrade far more gracefully: 4.7 vol points RMSE on
   the same out-of-range subset) or ThinPlate's smoother RBF extrapolation (2.4 vol points).
   Worth a sentence in the paper's baseline description -- and a reason SVI's wing behavior
   is genuinely useful, not just "traditional."

## What wasn't done in this pass (flagging rather than skipping silently)
- SSVI's theta(tau) is a first-stage empirical estimate (closest-to-zero-k point's total
  variance per expiry, then isotonic-regressed), not jointly optimized with (rho,eta,gamma)
  -- a common two-stage simplification, but a joint fit would likely improve SSVI's numbers
  somewhat and narrow its gap to SVI.
- Per-fold breakdown (does baseline quality drift across the 18 walk-forward folds) wasn't
  computed yet -- the four baselines don't use training-window history so this is lower
  priority than it will be once the network is in the picture, but it's cheap to add before
  Phase 4 if useful for the robustness cuts the plan asks for.
- Long-format results table with all four methods' every prediction:
  `results/baseline_predictions_long.parquet` (230,900 rows: method, date, expiry, strike,
  k, tau, iv_actual, iv_pred, in_range, feasible).
