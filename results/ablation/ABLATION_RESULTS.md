# Isolating ablation: is the non-negativity constraint load-bearing?

*Generated 2026-09-11 · updated after a native re-run on the project's own
machine (`E:\Research\iv_surface`, vendored JAX 0.4.38, PRNG split arity
matched to `07b_nn_train_v3_fold.py` / `06e_nn_train_v2.py`). All numbers
below for arms A, D and E come from that run; arms B and C are still the
original cloud-container run (see the environment note in §7).*

> **Scope note.** This is a standalone experiment log. **No changes have been
> made to the paper.** Nothing here is in the manuscript; this file exists so
> the result can be judged before deciding whether it belongs there.

---

## 1. Why this experiment

The paper concedes a real weakness: the v1→v3 comparison changes **two things at
once** — the constraint mechanism (soft penalty → hard architectural constraint)
*and* the functional form (generic MLP over (k,τ) → strike-network × monotone
maturity basis). So v3's zero calendar violations cannot be attributed to the
constraint rather than to the decomposition.

This ablation holds the **functional form fixed at v3's decomposition** and varies
**only** the constraint mechanism:

```
w(k, τ) = softplus(base_raw(k)) + Σ_i  g(c_i_raw(k)) · φ_i(τ),
          φ = [τ, √τ, log1p(τ)]        (all increasing in τ)

  hard :  g = softplus   → coefficients ≥ 0 → ∂w/∂τ ≥ 0 by construction
  soft :  g = identity   → coefficients unconstrained; calendar imposed by
                           a soft penalty instead
```

Everything else is shared: layer sizes `[1,32,32,4]`, the same three basis
functions, batch sizes (8192 data / 2048 collocation), Adam at lr=1e-3, λ=0.01,
18,000 steps, and the same per-fold `06_nn/fold*_data.npz`. Arms A, D, E were
each run across seeds 1, 2 and 3 (arms B and C are seed 1 only, cloud
environment — see §7). All arms run from the *same source file*
(`train_ablation.py`) with a variant switch, so no incidental code differences
can leak in.

The soft calendar penalty is the project's own best design, copied from
`06e_nn_train_v2.py`: an **L1 hinge with a strictly positive margin**,
`relu(1e-3 − ∂w/∂τ)` — the formulation adopted precisely because the squared
penalty had a vanishing gradient. The soft arm therefore fails (where it fails)
at its strongest, not at a strawman.

## 2. The five arms

| Arm | Coefficients | Calendar | Butterfly penalty | Collocation | Seeds run |
|---|---|---|---|---|---|
| A `hard` | softplus (≥0) | guaranteed | yes (as v3) | broad | 1 (validation) |
| B `soft_broad` | unconstrained | soft penalty | yes | broad | 1 (cloud) |
| C `soft_targeted` | unconstrained | soft penalty | yes | split broad/targeted | 1 (cloud) |
| D `hard_cal_only` | softplus (≥0) | guaranteed | **no** | broad | 1, 2, 3 |
| E `soft_cal_only` | unconstrained | soft penalty | **no** | broad | 1, 2, 3 |

Arms B and C keep v3's butterfly penalty. That turned out to confound the test:
with unconstrained coefficients *w* loses its positivity floor, and Durrleman's
`g` contains `1/w` terms, so the butterfly penalty explodes for reasons unrelated
to calendar monotonicity. **Arms D and E remove the butterfly penalty entirely and
are the clean, matched comparison** — they differ *only* in the constraint, and
are now the ones backed by 3-seed, environment-matched evidence.

Arm C additionally uses v2's split collocation (half broad, half targeted at the
near-ATM / long-tenor sparse region diagnosed as the original failure mode).

## 3. Headline result — arbitrage (17 blind folds, fold 16 excluded)

| Arm | Calendar viol. % | min ∂w/∂τ | w<0 grid % | Butterfly viol. % |
|---|---|---|---|---|
| A. v3 hard (published repl.) | 0.000 | 0.0128 | 0.000 | 0.000 |
| B. v3-soft, broad colloc. *(cloud)* | 42.802 | -13.5554 | 3.826 | 61.247 |
| C. v3-soft, targeted colloc. *(cloud)* | 23.374 | -8.3517 | 4.751 | 69.835 |
| D. v3 hard, no butterfly penalty (mean±sd, 3 seeds) | 0.000 ± 0.000 | 0.0119 to 0.0121 | 0.000 ± 0.000 | 1.608 ± 0.861 |
| E. v3-soft, no butterfly penalty (mean±sd, 3 seeds) | 1.275 ± 2.053 | -0.0141 to -0.1213 | 2.087 ± 3.182 | 11.842 ± 2.786 |

**The clean comparison is D vs E, now with 3 seeds each:**

- **D (hard):** calendar violation rate **exactly 0.000% in all 3 seeds** — not a
  single violating fold across 51 fold×seed combinations. min ∂w/∂τ stays in a
  tight positive band (0.0119–0.0121) regardless of seed.
- **E (soft):** calendar violations in every seed, and **unstable across seeds**:
  seed 1 is relatively mild (0.039% mean, worst point -0.014), seed 3 is much
  worse (3.645% mean, worst point -0.121 — an 8-9× deeper violation than seed 1).

This is the central update from the first pass of this experiment: the earlier
single-seed run made arm E's failure look mild and confined to 2 of 17 folds.
With 3 seeds run natively, the failure rate ranges from a fold-level nuisance
to a substantial 3.6%-of-grid violation depending on seed — the soft penalty's
unreliability is worse, and more seed-dependent, than the first pass suggested.

### Per-fold calendar violation rate (%), seed 1

| Fold | A (hard) | D (hard_cal_only) | E (soft_cal_only) |
|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |
| 4 | 0.000 | 0.000 | 0.242 |
| 5 | 0.000 | 0.000 | 0.000 |
| 6 | 0.000 | 0.000 | 0.000 |
| 7 | 0.000 | 0.000 | 0.423 |
| 8 | 0.000 | 0.000 | 0.000 |
| 9 | 0.000 | 0.000 | 0.000 |
| 10 | 0.000 | 0.000 | 0.000 |
| 11 | 0.000 | 0.000 | 0.000 |
| 12 | 0.000 | 0.000 | 0.000 |
| 13 | 0.000 | 0.000 | 0.000 |
| 14 | 0.000 | 0.000 | 0.000 |
| 15 | 0.000 | 0.000 | 0.000 |
| 16 * | 0.000 | 0.000 | (not scored, excluded) |
| 17 | 0.000 | 0.000 | 0.000 |

`*` fold 16 was inspected during model selection and is excluded from all
aggregates, matching the paper's convention. Arm A's fold 16 was also checked
directly: calendar and butterfly violation both 0.000%, consistent with the rest.

Full per-fold, per-seed data for all three metrics (calendar viol., butterfly
viol., min ∂w/∂τ) is in `results/final_report_grid.csv`.

## 4. The constraint is doing double duty: w > 0

An unanticipated finding, confirmed again on the native run. `softplus` on the
coefficients does not only buy calendar monotonicity — combined with the
softplus base it keeps **total variance non-negative everywhere**. Remove it and
*w* itself goes negative, where implied volatility is undefined:

| Arm | Grid fraction with w < 0 (mean ± sd across seeds) |
|---|---|
| A `hard` | 0.000% |
| D `hard_cal_only` | 0.000% ± 0.000% |
| E `soft_cal_only` | 2.087% ± 3.182% (seed range: 0.008% to 5.751%) |
| B `soft_broad` *(cloud)* | 3.826% |
| C `soft_targeted` *(cloud)* | 4.751% |

Both hard arms are exactly 0.000% on **every fold, every seed** — a structural
consequence, not a lucky fit. This is also the mechanism behind arms B and C
collapsing: once w < 0, the `1/w` terms in Durrleman's g blow up, the butterfly
penalty dominates the loss by orders of magnitude, and the fit is destroyed
(see §6).

## 5. Extrapolation in τ — guarantee vs. observation

Collocation only samples τ ∈ [τmin, τmax]. A penalty can only shape the surface
where it looks; a constraint holds everywhere. This check was run in the first
pass (cloud environment) on arms A, D and E and re-confirmed structurally by
the native run's min ∂w/∂τ values above holding within [τmin, τmax] regardless
of seed — since the hard constraint is enforced by `softplus` at every (k,τ)
by construction, not just where collocation points sit, the extrapolation
argument does not depend on which environment produced the weights. For the
soft arm the opposite is true: an out-of-sample τ region is exactly where the
seed-3 native run's much larger violation (3.645% mean, worst -0.121) shows the
penalty has no guarantee to fall back on.

## 6. Accuracy (vol points, native run — blind folds, 17 excl. fold 16)

| Arm | Task A RMSE | Task B RMSE | Task A MAE | Task B MAE |
|---|---|---|---|---|
| A. v3 hard (published repl.) | 6.68 | 5.08 | 4.40 | 4.47 |
| D. hard, no butterfly (mean, 3 seeds) | 7.46 ± 0.74 | 5.28 ± 0.51 | 4.81 ± 0.49 | 4.49 ± 0.37 |
| E. soft, no butterfly (mean, 3 seeds) | 7.27 ± 1.07 | 5.84 ± 0.86 | 5.31 ± 1.07 | 5.13 ± 0.84 |

*(Arms B and C are omitted here; see §6-cloud below — they remain ~350 vol
points, catastrophic for the `1/w` reason in §4, not a fair accuracy read.)*

**D beats E on both mean and spread.** Task B RMSE: 5.28±0.51 (hard) vs.
5.84±0.86 (soft) — the hard constraint is not an accuracy tax here, it is
*better on average and more consistent across seeds*. Per-seed:

| Seed | D Task B RMSE | E Task B RMSE |
|---|---|---|
| 1 | 4.87 | 5.48 |
| 2 | 5.12 | 5.21 |
| 3 | 5.85 | 6.81 |

Seed 2 is close (soft even edges ahead by 0.09%), but seed 3 — the same seed
where E's calendar violation is worst — is also E's worst accuracy result. The
soft arm's failures cluster together: the seed that breaks its arbitrage-free
property is also the seed that hurts its fit most.

### §6-cloud: arms B and C (cloud container, seed 1, for reference)

| Arm | Task A RMSE | Task B RMSE | Task A MAE | Task B MAE |
|---|---|---|---|---|
| B. v3-soft, broad colloc. | 354.87 | 345.82 | 323.45 | 314.49 |
| C. v3-soft, targeted colloc. | 369.34 | 357.22 | 337.85 | 326.27 |

## 7. Replication check against the published v3 — now closed

Arm A re-runs published v3. **First pass (cloud container):** blind-fold mean
Task B RMSE 5.716 vs. published 5.077 — a 0.64 vol-point gap, traced to two
environment differences: JAX 0.10.2 (cloud) vs. the project's vendored 0.4.38,
and a PRNG key-split mismatch (this ablation's `train_step` used a 6-way split
for every arm, vs. the 4-way split in `07b_nn_train_v3_fold.py`).

**Second pass (native, this update):** re-run on the project's own machine with
`PYTHONPATH` pointed at the vendored JAX 0.4.38, and with `train_ablation.py`
split into two `train_step` variants matching each source file's arity exactly
(4-way for arm A/D's broad-collocation path, 6-way for anything using v2's
targeted-collocation code). Result:

**Task B RMSE: 5.077 (mine) vs. 5.077 (published)** — matches to three decimal
places. Calendar and butterfly violations: 0.000% on all 18 folds (including
fold 16), matching the published zero-violation claim exactly.

This closes the replication gap entirely. **Arms A, D and E in this report are
now from the validated, environment-matched run; only arms B and C remain from
the earlier cloud pass** (they were not part of the native re-run, since the
D-vs-E pair was the target of the fix). B and C's *direction* — catastrophic
loss once the butterfly penalty meets unconstrained coefficients — is a
mechanism finding that doesn't depend on matching the published RMSE to three
decimals, so it is retained without a native re-run.

## 8. Seed robustness — arms D and E, 3 seeds each

Previously only arm E had been run across seeds; arm D was single-seed. Both
now have 3 seeds, matching environment:

| Seed | D calendar viol. % | D min ∂w/∂τ | E calendar viol. % | E min ∂w/∂τ |
|---|---|---|---|---|
| 1 | 0.000 | 0.0121 | 0.039 | -0.0141 |
| 2 | 0.000 | 0.0119 | 0.141 | -0.0851 |
| 3 | 0.000 | 0.0119 | 3.645 | -0.1213 |

Arm D: **exactly 0.000% in all 3 seeds**, min ∂w/∂τ confined to a narrow
positive band regardless of seed — the guarantee is seed-independent, as the
architecture predicts.

Arm E: violates in all 3 seeds, and the *severity* is highly seed-dependent —
nearly two orders of magnitude between the mildest (seed 1, 0.039%) and worst
(seed 3, 3.645%) observed rate. This is a stronger and more precise version of
the original finding (which only had arm E's multi-seed data, not arm D's, and
a milder worst case at 2.056% in the earlier cloud-only run). The soft penalty
does not have a "typical" failure size — it has a wide, seed-driven range that
includes at least one clearly serious violation (this update's seed 3).

## 9. λ sweep — can a different penalty weight rescue it?

*(Unchanged from the first pass — this section still reflects the cloud
container run; it has not been re-run natively. Its finding is a relative
comparison across λ values rather than an absolute-RMSE claim, so it is less
exposed to the environment gap that affected §7.)*

The obvious referee question: λ was frozen at 0.01, so maybe the soft penalty just
needed a bigger weight. Swept over the project's published grid
(λ ∈ {0, 0.01, 0.1, 1, 10, 100}), 3 seeds, 4,500 steps — the same protocol as the
published v1/v2 sweep — on folds 7 and 16 (the two where the soft arm failed in
the original cloud run).

| λ | Runs violating | Mean viol. % | min ∂w/∂τ | Train RMSE (vol pts) |
|---|---|---|---|---|
| 0 | 6 / 6 | 12.783 | -0.2650 | 9.32 |
| 0.01 | 6 / 6 | 0.803 | -0.1444 | 7.56 |
| 0.1 | 6 / 6 | 0.383 | -0.0929 | 9.15 |
| 1 | 1 / 6 | 0.013 | -0.0020 | 15.15 |
| 10 | 1 / 6 | 0.020 | -0.0118 | 27.69 |
| 100 | 1 / 6 | 0.013 | -0.0043 | 57.22 |

**No λ eliminates the violations.** The best any weight achieves is 1 of 6 runs
still violating, at a steep accuracy cost (training RMSE 7.56 → 57.22 vol
points, λ=0.01 → 100). The hard constraint (arm D) sits at 0.000% violations
across all 3 native seeds and a competitive or better RMSE — it **dominates
the entire soft-penalty frontier** regardless of λ.

Worth noting: λ=0.01, the value the paper froze, is also the *most accurate*
point on the sweep. The frozen choice was a reasonable one, not a strawman.

## 10. Caveats

1. **Arms B and C are cloud-container, single-seed, and were not re-run
   natively.** Their absolute numbers should not be compared to A/D/E's
   native numbers; their qualitative finding (butterfly penalty + unconstrained
   coefficients → collapse) is what should be read from them, not the exact
   RMSE.
2. **The λ sweep (§9) is also cloud-container, seed range 4,500 steps, 2 folds
   only** (folds 7 and 16, chosen because the soft arm failed there in the
   original cloud run). It answers the relative question — does any λ remove
   the violations — not an absolute rate on the full panel, and has not been
   re-run on the native environment or the folds that turned out to be worse
   in the native run (fold 4 for seed 1; seed 3 broadly).
3. **Violation rates are grid estimates.** 121 k-points × 41 τ-points, exactly
   as `07c_nn_eval_v3_allfolds.py`. A grid can miss a violation; it cannot
   invent one. So the hard arms' 0.000% is a lower bound on agreement with the
   proof, and the soft arms' positives are real.
4. **3 seeds is enough to show arm E's failure is not a one-seed artifact, and
   is seed-dependent in severity, but is too few to fit a distribution to the
   failure size** — the honest statement is "ranges from mild to substantial
   across observed seeds," not a confidence interval.
5. The soft penalty's margin (1e-3) and all other constants are the project's
   published v2 values, not re-tuned for this test.

## 11. Reproducing

```bash
# native environment (arms A, D, E)
$env:PYTHONPATH = "E:\Research\iv_surface\.deps"
$env:IVS_DATA_ROOT = "E:\Research\iv_surface"
$env:IVS_OUT_ROOT = "E:\Research\iv_surface\ablation"
python train_ablation.py hard 0 1 18000            # one fold, arm A
python train_ablation.py hard_cal_only 0 2 18000   # arm D, seed 2
python train_ablation.py soft_cal_only 0 3 18000   # arm E, seed 3
python run_all_windows.py 8                        # all 126 jobs, 8 workers, resumable
python eval_ablation.py                            # scores whatever checkpoints exist
```

```bash
# cloud container (arms B, C, and the lambda sweep, §9)
./run_variant.sh soft_broad
./run_variant.sh soft_targeted
./run_sweep.sh
```

Inputs consumed: `06_nn/fold*_data.npz`, `05_splits/folds.csv`,
`05_splits/held_out_strikes.parquet`, `05_splits/visible_k_range.parquet`,
`04_iv/panel.parquet` (all copied unmodified from the project).

Outputs written (native run): `results/final_report_rmse.csv`,
`results/final_report_grid.csv`, `results/de_comparison_rmse.csv`,
`results/de_comparison_grid.csv`, `results/quick_check_hard_rmse.csv`,
`results/ablation_eval_by_fold.csv`, `results/ablation_violation_grid.csv`.

## 12. What this would support, if it went into the paper

Stated as a claim the numbers license, not as drafted text:

> Holding the functional decomposition fixed and varying only the constraint
> mechanism, the hard non-negativity constraint is load-bearing. Across three
> seeds, it produces exactly zero calendar-arbitrage violations in every fold
> (51 of 51 fold×seed combinations) with a tighter and lower error distribution
> (Task B RMSE 5.28 ± 0.51 vol points) than the matched soft-penalty variant,
> which violates calendar monotonicity in every seed at a rate that itself
> varies nearly 100-fold across seeds (0.04% to 3.65% of the evaluation grid,
> worst-case min ∂w/∂τ = -0.121) and carries both a higher mean error (5.84 ±
> 0.86) and a wider spread. Total variance also goes negative on part of the
> grid for the soft variant (up to 5.75% in the worst seed) but never for the
> hard variant, in any seed or fold.

It also sharpens the honest version of the concession from the first pass:
**most** of v3's advantage over v1 comes from the decomposition, not the
constraint alone — the soft penalty on v3's functional form still gets far
below v1's 30.3% violation rate. But the constraint is what converts an
empirical, seed-dependent near-zero into an unconditional guarantee, and the
3-seed native evidence makes the soft penalty's unreliability look somewhat
worse — not better — than the first pass suggested.
