# Phase 4 — Full Walk-Forward Evaluation, v3 (Hard-Calendar-Constrained), λ=0.01 Frozen

## 0. Why this phase looks different from the plan, and what to trust

Phase 3's model/hyperparameter selection (choosing v3 over v1/v2, and λ=0.01 within
v3's sweep) was done by repeatedly evaluating Task A **and** Task B on fold 16 across
36+ (architecture, λ, seed) configurations. The project plan's own pre-registration
rule ("Test data | Never touched until Phase 5 is complete... Fill the cells once.
Whatever they say is the paper.") was violated by that process, discovered only after
Phase 3 was written up.

Rather than discard that work or pretend the discipline held, the fix adopted here
(explicit user decision, see below) is:

- **Freeze the configuration**: v3 architecture, λ=0.01, no further tuning of any
  kind from this point on.
- **Evaluate all 18 walk-forward folds** (0–17) with that one frozen configuration,
  one seed (seed=1) per fold.
- **Report fold 16 separately, everywhere, as contaminated.** Folds 0–15 and 17 (17
  folds total) are genuinely blind — their Task A/B windows were never inspected
  during model selection — and are pooled together as "blind" results. Fold 16's
  numbers are shown alongside for reference but excluded from every pooled/blind
  aggregate and from the headline Diebold-Mariano and robustness results.

This is a real compromise, not a full re-run of Phase 3's model-selection process
with proper sample-splitting discipline (e.g., a separate validation fold group for
selection and a fully held-out group for final reporting) — that would have required
redoing Phase 3 from scratch. It is disclosed here, as with every other scope
reduction in this project, rather than hidden.

**Scope reduction**: one seed per fold (not the plan's originally-implied multiple
seeds), because 18 folds × even 3 seeds would triple total compute. The per-fold
walk-forward axis was prioritized over the seed-variance axis, since Phase 3 already
characterized seed-to-seed variance at fold 16 (see `phase3_notes.md` Finding 2). A
natural follow-up — 2–3 extra seeds on a few representative folds (early/middle/late
in the walk-forward sequence) to bound seed variance without tripling total compute —
was discussed but not executed in this session.

## 1. Training

All 18 folds trained to 18,000 Adam steps at λ=0.01 (frozen), identical architecture
and optimizer settings to Phase 3's v3 (`src/07b_nn_train_v3_fold.py`, generalizing
`06f_nn_train_v3.py`). Fold 16 **reuses its exact Phase 3 checkpoint** (same fold,
same architecture, same λ, same 18,000 steps) rather than retraining, since retraining
would be functionally identical and wasteful — verified by comparing checkpoint
architecture (layer shapes, `norm_stats`) and step count before reuse
(`06_nn/runs_v3_allfolds/fold16_seed1/REUSED_FROM_PHASE3.txt`).

Mean training wall-clock: **120.1s per fold** (17,000 steps average across folds,
range 112–133s; `results/phase4_nn_train_time_by_fold.csv`), ≈2,041s (34 min) total
for the 17 freshly-trained folds.

## 2. Arbitrage-freedom holds at scale, not just at fold 16

Post-hoc dense-grid (121×41, k-padded 20% beyond the training range) violation rates,
every fold:

| metric | value, every one of 18 folds |
|---|---|
| butterfly violation rate | 0.0000 |
| calendar violation rate | 0.0000 |

The hard-calendar-constraint architecture (`w(k,τ) = softplus(base(k)) + Σᵢ
softplus(cᵢ(k))·φᵢ(τ)`) guarantees ∂w/∂τ ≥ 0 by construction, and this holds exactly
across all 18 independently-trained folds — Phase 3's fold-16-only finding was not a
fluke. Butterfly arbitrage (still a soft λ-weighted penalty) is also driven to exactly
0% on the dense grid at λ=0.01 in every fold, consistent with Phase 3's Finding 2
(butterfly is easy to fix cheaply; calendar was the hard part, now solved
structurally). Full grid: `results/phase4_nn_violation_grid_by_fold.csv`.

## 3. Task A / Task B accuracy, NN vs baselines

All errors in **vol points** (IV decimal × 100, Phase 2's convention).

### NN v3, RMSE across the 17 blind folds

| | mean | median | min | max |
|---|---|---|---|---|
| Task A (held-out strikes, training window) | 6.68 | 6.47 | 5.71 | 8.32 |
| Task B (entire held-out future window) | 5.08 | 4.76 | 3.11 | 7.54 |

Fold 16 (contaminated, shown separately): Task A RMSE 5.72, Task B RMSE 5.71 — inside
the range of the blind folds, i.e. the contamination does not appear to have produced
an unusually favorable number for the NN, but it is excluded from all headline
aggregates on principle regardless. Full per-fold table:
`results/phase4_nn_eval_by_fold.csv`.

Two things carry over unchanged from Phase 3: Task B RMSE is consistently *lower*
than Task A RMSE in every fold (the pooled global surface extrapolates across time
somewhat better than it fills in held-out strikes within its own training window —
the opposite of what one might naively expect, and a pattern that now holds across 17
independent folds, not just one), and both remain far above the classical baselines
below.

### Baseline RMSE, mean across the 17 blind folds (via existing per-slice/per-day
fits, filtered by fold window — no retraining needed, since Phase 2's baselines carry
no dependence on any walk-forward training window)

| method | train-window-style (Task A-comparable) | test-window-style |
|---|---|---|
| ThinPlate | 0.947 | 0.633 |
| SSVI | 2.791 | 2.179 |
| SVI | 3.156 | 2.124 |
| CubicSpline | 47.59 | 51.09 |

Per-fold detail: `results/phase4_baseline_eval_by_fold.csv`. CubicSpline's large mean
is driven by a small number of catastrophic per-fold blow-ups (one fold's test-window
RMSE hit 480 vol points) — a known instability of unconstrained cubic splines on
sparse/irregular strike grids, present already in Phase 2 and not something Phase 4
attempted to fix. ThinPlate, SSVI, and SVI all beat the NN by roughly 2–10x on both
axes, confirming Finding 4 from Phase 3 (fixed-architecture pooled NN vs
per-slice/per-day-refit baselines is not a like-for-like contest) generalizes across
the full walk-forward sample, not just fold 16.

## 4. Error decomposition by moneyness and maturity bucket

Buckets: moneyness by k (deep OTM put ≤ −0.10, OTM put, ATM ±0.03, OTM call, deep OTM
call > 0.10); maturity by τ (short ≤30d, medium 30–90d, long >90d).

Because fold test windows are disjoint and tile the sample contiguously (confirmed
from `05_splits/folds.csv`: 2021-10-01 → 2026-01-31 across folds 0–17), **Task B**
bucket decomposition pools all 17 blind folds' test windows with each date counted
exactly once — a valid full-sample view. Training windows are *not* disjoint
(expanding-origin design, every fold shares `train_start = 2020-01-01`), so pooling
**Task A** across folds would count early dates up to 18x; instead Task A is
decomposed using only fold 17 (the largest blind training window) as a single
representative fold.

Full tables: `results/phase4_bucket_decomposition_nn.csv` and
`..._baseline.csv`. Headline pattern (both NN and baselines): errors are worst in the
deep-OTM buckets and best ATM, as expected; the NN's degradation in deep-OTM /
long-maturity cells is generally milder in relative terms than the baselines'
occasional blow-ups there (e.g. CubicSpline's `deep_OTM_call × medium(30-90d)` cell
alone reaches RMSE ≈900 vol points in the fold-17 sample), though the NN's *absolute*
level remains higher than ThinPlate/SSVI/SVI in every bucket.

## 5. Diebold-Mariano tests, NN v3 vs each baseline

Matched-pairs test on the fixed 20% held-out strikes (identical rows scored by both
sides), loss = squared error in vol points, aggregated to a per-date mean (correcting
for same-day cross-sectional correlation across strikes) and tested with a
Diebold-Mariano statistic using a Newey-West long-run-variance estimate (automatic
bandwidth, Newey-West 1994 plug-in rule) to account for day-to-day serial
correlation. `results/phase4_diebold_mariano.csv`.

| scope | vs. | n days | mean loss diff (NN − baseline) | DM stat | p-value | lower sq. error |
|---|---|---|---|---|---|---|
| blind, 17 folds | SVI | 1004 | +22.65 | 14.57 | <0.0001 | SVI |
| blind, 17 folds | SSVI | 1004 | +22.78 | 17.12 | <0.0001 | SSVI |
| blind, 17 folds | ThinPlate | 1004 | +27.23 | 20.07 | <0.0001 | ThinPlate |
| blind, 17 folds | CubicSpline | 1004 | −15,427 | −1.07 | 0.286 | (NN, but see caveat) |
| fold 16 (contaminated) | SVI/SSVI/ThinPlate | 61 | +28 to +32 | 16–18 | <0.0001 | baseline, all three |
| fold 16 (contaminated) | CubicSpline | 61 | −305 | −1.77 | 0.076 | (NN, but see caveat) |

SVI, SSVI, and ThinPlate all beat the NN with overwhelming statistical significance,
on both the blind pool and the (separately-reported) contaminated fold — this is not
close, and matches the raw RMSE gap in §3.

**The CubicSpline row is not a real NN win and should not be read as one.** The
enormous negative mean loss difference is driven entirely by CubicSpline's rare
catastrophic blow-ups (§3); a handful of days with squared errors in the tens of
thousands of vol-points² swamp the day-level mean and produce a technically
non-significant DM statistic in the "NN's favor" direction purely because
CubicSpline's variance, not its typical accuracy, dominates. CubicSpline's *typical*
(median) day is far more accurate than the NN, consistent with its RMSE excluding
those blow-up days. This asymmetric-tail behavior should be discussed explicitly in
the paper rather than reported as "NN ties or beats CubicSpline."

## 6. Robustness cuts

Both computed on the blind-pooled (17-fold) Task B sample, same scope as §5.

**(i) Highest-volatility decile of days excluded.** No VIX-style series exists in
this project (confirmed absent in the Phase 0 data inventory); a proxy was
constructed as the mean ATM (|k|≤0.03), short/medium-maturity (τ≤90/365) IV per day,
analogous to how a VIX-style index is built from near-term ATM options. Excluding the
top decile (150 of 1,496 days) moves NN RMSE from 5.14 → 5.08 vol points and each
baseline by a similarly small amount (`results/phase4_robustness_volcut.csv`) — the
headline comparison is **not** an artifact of a handful of extreme-volatility days.

**(ii) Monthly vs weekly expiry split.** No explicit expiry-cycle flag exists in the
panel; the split was derived directly from the expiry calendar (within each calendar
month, the latest expiry date present is the monthly contract, every earlier one that
month is weekly — robust to NSE's several changes of weekly-expiry weekday over
2020–2026, no fixed "Thursday" assumption made). 329 distinct expiries: 78 monthly,
251 weekly. NN RMSE is somewhat higher on weekly expiries (5.38) than monthly (4.93);
ThinPlate/SSVI/SVI show smaller and less consistent differences
(`results/phase4_robustness_expirycut.csv`). CubicSpline's instability is far worse
on monthly expiries (RMSE 181) than weekly (22) in this cut, consistent with monthly
contracts having sparser same-day strike grids at the far tenors where CubicSpline is
most prone to blowing up.

## 7. Wall-clock timing

| method | fit | predict |
|---|---|---|
| NN v3 | 120.1s / fold (18,000 steps, one global surface for the whole 18–66 month training window) | 0.83s / fold (Task A + Task B + dense violation grid combined) |
| SVI | 84.9 ms / (date, expiry) slice | included above |
| SSVI | 37.7 ms / day (pooled across a day's expiries) | included above |
| ThinPlate | 16.1 ms / day | included above |
| CubicSpline | 10.2 ms / day | included above |

(`results/phase4_timing.csv`; baselines benchmarked fresh since Phase 2's scripts
were never instrumented for timing, using a 100-slice / 30-day sample near the start
of the panel.) The NN's fit cost buys one persistent global surface amortized over an
entire multi-year training window and near-instant subsequent inference, versus the
baselines' cheap-but-disposable per-slice/per-day fits that must be redone for every
new day of data — a real practical trade-off distinct from, and arguably more
favorable to the NN than, the raw accuracy comparison in §3 and §5.

## 8. Quick-fix addendum: seed variance, CubicSpline robust stats, Task B/volatility-regime check

Three follow-up checks, run after an initial "is this ready to publish" self-review
flagged them as gaps. None of them change the headline conclusions above; all three
are disclosed here because they either add an honest uncertainty band or rule out
an alternative explanation that a reviewer would otherwise be right to ask about.

**8.1 Seed variance.** Folds 2, 9, 17 (early / mid / late in the walk-forward
sequence) were each retrained from scratch with two additional seeds (seed=2,
seed=3), giving 3 seeds × 3 folds = 9 additional (fold, seed) training runs on top
of the existing seed=1 runs. Within-fold, across-seed RMSE standard deviation
averaged 0.34 vol points on Task A and 0.54 vol points on Task B (`results/phase4_seed_variance.csv`,
`results/phase4_seed_variance_summary.csv`). That is a real, non-trivial noise
floor: any single-seed RMSE number in this document should be read with roughly
±0.3–0.5 vol points of seed-driven uncertainty. Comparing that noise floor to the
across-fold spread computed on the very same 3 folds' seed=1 runs gives a mixed
signal — the across-fold std is larger for Task A (0.99, i.e. fold effects dominate
seed noise there) but *smaller* than the seed noise for Task B (0.17). We do not
read too much into the Task B ratio: n=3 folds is a small and non-random sample for
estimating across-fold variance, so that comparison is more a caveat than a
conclusion. The actionable takeaway is the noise floor itself (±0.3–0.5 vol points),
not a precise fold-vs-seed variance decomposition.

**8.2 CubicSpline robust statistics.** `07d_baseline_eval_by_fold.py` was extended
to report the median and a 10%-trimmed mean of |error| (vol points) alongside
RMSE/MAE, per fold/method/window. Averaged over the 17 blind folds
(`results/phase4_baseline_eval_by_fold.csv`):

| method | window | RMSE | MAE | median &#124;err&#124; | trimmed mean &#124;err&#124; |
|---|---|---:|---:|---:|---:|
| ThinPlate | test | 0.633 | 0.181 | 0.051 | 0.073 |
| CubicSpline | test | 51.094 | 2.089 | 0.057 | 0.089 |
| SVI | test | 2.124 | 0.910 | 0.381 | 0.510 |
| SSVI | test | 2.179 | 1.469 | 1.006 | 1.189 |

CubicSpline's median error (0.057) is essentially tied with ThinPlate's (0.051) —
both far ahead of SVI and SSVI on a *typical* day — even though CubicSpline's RMSE
(51.1) is roughly 25–80× larger than every other baseline's. This is exactly the
"usually fine, occasionally explodes" signature described qualitatively earlier in
this document, now shown numerically rather than asserted: CubicSpline is not a
globally unreliable interpolator, it is a locally excellent one with a heavy right
tail (a handful of blow-up days per fold reaching RMSE in the hundreds). The
Diebold-Mariano result in §5 comparing NN vs CubicSpline should be read in that
light — it is a comparison of mean squared error, which the tail dominates, not a
comparison of typical-day accuracy.

**8.3 Task B < Task A: volatility-regime check.** The pattern "NN Task B RMSE is
lower than Task A RMSE in every one of the 17 blind folds" invites an obvious
alternative explanation: later test windows might simply be calmer markets than the
(longer, earlier-starting) training windows, rather than the NN genuinely
extrapolating well. Using the same ATM short/medium-maturity IV proxy built for the
volatility-decile robustness cut, we compared each fold's mean volatility proxy over
its train window against its test window (`results/phase4_taskb_vol_regime_check.csv`).

The confound's precondition does hold: the test window is calmer than the train
window in all 17/17 folds (mean proxy 0.190 train vs 0.141 test — training windows
are expanding and still carry the 2020–2021 high-volatility period, while test
windows drift later into calmer years). However, the confound's predicted
*signature* is absent: if the Task A/B accuracy gap were driven by this
volatility-regime difference, folds with a larger train-minus-test volatility gap
should show a *larger* RMSE gap (a positive correlation). Instead the correlation is
strongly **negative** (r = −0.82, n=17) — folds where training data is much more
volatile than the test window (e.g. folds 6, 7) show the *smallest* Task A/B RMSE
gap, in two cases (folds 6, 7) even reversing the pattern (Task A RMSE < Task B
RMSE). Task A RMSE does correlate strongly with train-window volatility (r=0.93,
consistent with a genuinely harder-to-fit training sample producing worse held-out
interpolation), but Task B RMSE correlates *negatively* with test-window volatility
(r=−0.53) — the opposite of what "calmer test windows are just easier" would
predict. We read this as evidence that the Task A/B gap is not primarily a
volatility-regime artifact, while stopping short of a causal claim about *why* Task
B outperforms Task A — with n=17 folds this is a correlational check, not a
controlled experiment, and it should be described that way in the paper.

Artifacts added in this addendum: `src/08a_seed_variance.py`,
`src/08b_taskb_vol_regime_check.py`, and a revision to `src/07d_baseline_eval_by_fold.py`
(added `median_abs_err_volpts` / `trimmed_mean_abs_err_volpts` columns).

## 9. Baseline arbitrage-violation measurement: does the guarantee actually differentiate the NN?

Section 2 measured the NN's own violation rate (0% butterfly and calendar across
all 18 folds). Until now, the equivalent number was never measured for the four
classical baselines -- the paper's whole framing ("the NN trades accuracy for a
guaranteed arbitrage-free surface") implicitly assumed the baselines are not
arbitrage-free, but that assumption was never tested. It matters because SVI and
SSVI are already constrained during fitting (§ code review of `04a_svi.py`,
`04b_ssvi.py`): SVI to Gatheral-Jacquier's sufficient no-butterfly condition, SSVI
to the same condition plus an isotonic-regression-enforced non-decreasing
theta(tau) for calendar. If those constraints hold up in practice, the NN's
differentiation would shrink to only the two baselines with no arbitrage handling
at all (ThinPlate, CubicSpline) -- and those two are not the ones that beat the NN
most convincingly on accuracy.

**Method.** No fitted parameters were saved from Phase 2 (only point predictions),
so all four baselines were refit per day/slice (`src/07i_baseline_arbitrage_check.py`)
and evaluated on a dense grid, mirroring the NN's own check: butterfly (fixed tau,
vary k) via the Durrleman condition using each method's own w(k)=iv(k)^2*tau and
its exact or finite-difference first/second k-derivatives (SVI/SSVI: closed form;
CubicSpline: scipy's exact spline derivatives; ThinPlate: central finite
differences); calendar (fixed k, vary tau) by comparing each method's fitted
w(k, tau_e) across a day's actual traded expiries on a shared k-grid, checking for
non-decreasing total variance between consecutive expiries. Run across the full
1496-day, ~9500-slice panel (`results/phase4_baseline_butterfly_full.csv`,
`results/phase4_baseline_calendar_full.csv`).

**Results.**

| method | butterfly: % slices with a violation | butterfly: mean % of grid points violating | calendar: % expiry-pairs with a violation | calendar: mean % of grid points violating |
|---|---:|---:|---:|---:|
| SVI | 4.3% | 0.4% | 58.8% | 22.3% |
| SSVI | 0.0% | 0.0% | 0.0% | 0.0% |
| ThinPlate | 90.9% | 14.0% | 35.6% | 4.3% |
| CubicSpline | 84.2% | 12.5% | 91.2% | 35.3% |

(n=9504 slices for butterfly, n=7993 consecutive same-day expiry pairs for
calendar; no method ever predicted a negative total variance.)

This is a more nuanced picture than the paper's original framing assumed, and it
cuts in different directions for different baselines:

**SSVI is empirically arbitrage-free, and it is also more accurate than the NN.**
Its theoretical construction (isotonic theta(tau) plus the SLSQP-enforced
Gatheral-Jacquier condition) holds up exactly in practice on this dataset --
0% violations of any kind. Against SSVI specifically, the NN's "we're arbitrage-free
and you're not" argument does not hold, and the paper should not claim it does. The
NN's real differentiation from SSVI has to be framed differently: SSVI requires a
fresh constrained joint optimization across every expiry, every single day
(37.7 ms/day, per §7, which is fast in isolation but is a per-day refit with no
persistent state), while the NN is a single global surface fit once and evaluated
in 0.83 s for an entire fold's Task A+B+violation-grid combined -- a genuine
architectural difference, but a computational/deployment one, not an
accuracy-vs-arbitrage-freedom trade-off.

**SVI is not arbitrage-free in practice, mainly on the calendar dimension.** Its
per-slice butterfly constraint mostly holds (95.7% of slices clean), and the small
remainder is a real, non-numerical issue: spot-checking the worst offending slices
found Durrleman-condition violations as large as g=-0.16, not floating-point noise,
traced to `04a_svi.py`'s `fit_one` occasionally accepting an SLSQP result whose
`res.success` was `False` (the acceptance logic keeps the lowest-SSE attempt
regardless of convergence flag) -- so the "feasible" column already in the
Phase 2 predictions is an optimistic diagnostic, not a hard guarantee, in a small
minority of slices. Far more importantly, SVI has no cross-expiry constraint at
all, and 58.8% of same-day expiry pairs show calendar-arbitrage violations, with
substantial magnitude (mean 22.3% of grid points on an average violating pair).
Against SVI, the NN's arbitrage-freedom claim is real and substantial.

**ThinPlate and CubicSpline are not arbitrage-free, decisively.** Both violate
butterfly on the large majority of slices (84-91%) and calendar on a large
fraction of expiry pairs (36-91%). This matters most for ThinPlate specifically,
since ThinPlate is the single most accurate baseline in this study (§3): the
honest trade-off the paper can defend is "if you want ThinPlate's accuracy, you
accept frequent arbitrage violations; if you want a static-arbitrage guarantee,
the NN's accuracy cost buys that guarantee, where ThinPlate's does not." That is
a real, measured trade-off, not an assumed one.

**Revised positioning.** The paper's contribution should now be stated precisely
rather than as a blanket claim: the NN is empirically arbitrage-free where SVI,
ThinPlate, and CubicSpline are not (differing degrees, detailed above), and where
SSVI also is, but only by paying a per-day joint-optimization refitting cost that
the NN's persistent global surface does not pay. This is a more defensible and
more interesting claim than "the NN is the only arbitrage-free method here" would
have been, and it survived a check that could easily have gone the other way.

Artifact added: `src/07i_baseline_arbitrage_check.py`. Outputs:
`results/phase4_baseline_butterfly_full.csv`, `results/phase4_baseline_calendar_full.csv`,
`results/phase4_baseline_arb_summary_{butterfly,calendar}.csv`.

## Bottom line

The v3 hard-calendar-constraint architecture's core claim from Phase 3 — arbitrage-free by construction, essentially for free — is now confirmed at full walk-forward scale: 0% butterfly and calendar violations across all 18 independently-trained folds, not just the one fold examined in Phase 3. That result stands on its own and is the paper's most defensible contribution (see `phase3_notes.md`'s positioning section for how this is framed against Zheng/Yu/Li 2022 and related prior art). Measured against the four classical baselines (§9), that guarantee is a real differentiator against SVI, ThinPlate, and CubicSpline (all show meaningful butterfly and/or calendar violations on this dataset), but not against SSVI, which is empirically arbitrage-free too -- there the NN's advantage is architectural (one persistent global fit vs. a fresh per-day joint optimization), not a guarantee SSVI lacks.

The accuracy comparison is unchanged in substance from Phase 3: on both Task A and Task B, and now confirmed across 17 genuinely blind folds with proper Diebold-Mariano significance testing, the pooled NN surface is reliably and significantly less accurate than three of the four per-slice/per-day-refit classical baselines (SVI, SSVI, ThinPlate), by a wide and statistically decisive margin. The apparent exception (CubicSpline) is an artifact of that baseline's tail instability, not a genuine NN advantage, and should be reported as such — see §8.2, where CubicSpline's median error is shown to be competitive with ThinPlate's on a typical day, so the RMSE gap is a tail effect, not evidence that CubicSpline is broadly unreliable. This is not a favorable finding for a "the NN wins" narrative, and the paper should continue to frame its contribution around arbitrage-freedom-by-construction and methodological rigor (walk-forward Task A/B design, DM testing, bucket decomposition, robustness cuts) rather than raw predictive accuracy.

The one honest procedural blemish — fold 16's Task B window having been examined repeatedly during Phase 3's model selection — is disclosed throughout rather than concealed, and every headline number in this document excludes fold 16 from pooled/blind aggregates.

## Artifacts

- `src/07a_prep_fold.py` — generalized per-fold data prep (all 18 folds).
- `src/07b_nn_train_v3_fold.py` — generalized, λ=0.01-frozen v3 training, resumable/checkpointed. `06_nn/runs_v3_allfolds/fold{0..17}_seed1/{ckpt.npz,log.csv}`.
- `src/07c_nn_eval_v3_allfolds.py` — Task A/B RMSE/MAE (vol points) and violation rates, all 18 folds. → `results/phase4_nn_eval_by_fold.csv`, `results/phase4_nn_violation_grid_by_fold.csv`.
- `src/07d_baseline_eval_by_fold.py` — per-fold baseline metrics from existing Phase 2 predictions, no retraining. → `results/phase4_baseline_eval_by_fold.csv`.
- `src/07e_bucket_decomposition.py` — moneyness × maturity error decomposition, NN and baselines. → `results/phase4_bucket_decomposition_{nn,baseline}.csv`.
- `src/07f_diebold_mariano.py` — matched-pairs DM tests, NN vs each baseline, Newey-West HAC. → `results/phase4_diebold_mariano.csv`.
- `src/07g_robustness_cuts.py` — volatility-decile and monthly/weekly-expiry cuts. → `results/phase4_robustness_{volcut,expirycut}.csv`.
- `src/07h_timing_benchmark.py` — fit/predict wall-clock, NN and baselines. → `results/phase4_timing.csv`, `results/phase4_nn_train_time_by_fold.csv`.
- `src/08a_seed_variance.py` — seed-variance check (2 extra seeds × 3 folds). → `results/phase4_seed_variance.csv`, `results/phase4_seed_variance_summary.csv`.
- `src/08b_taskb_vol_regime_check.py` — Task B/volatility-regime confound check. → `results/phase4_taskb_vol_regime_check.csv`.
- `src/07i_baseline_arbitrage_check.py` — butterfly/calendar violation rates for all four baselines, full 1496-day panel. → `results/phase4_baseline_butterfly_full.csv`, `results/phase4_baseline_calendar_full.csv`, `results/phase4_baseline_arb_summary_{butterfly,calendar}.csv`.
