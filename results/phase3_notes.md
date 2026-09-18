# Phase 3 — Arbitrage-Constrained Neural Interpolation: Results & Honest Assessment

## Setup

- **Single fold**: fold 16 (train 2020-01-01→2025-07-01, val 2025-07-01→2025-10-01,
  test 2025-10-01→2026-01-01) — the last full, non-partial walk-forward fold.
  218,078 visible training rows, 50,316 held-out (Task A) rows, 15,828 test-window
  (Task B) rows.
- **Model (v1)**: hand-rolled JAX MLP, layers [2, 64, 64, 64, 1], softplus hidden
  activations (twice differentiable — needed for the butterfly term's second
  derivative), softplus output enforcing w > 0. No Flax/Optax — PyTorch's Linux
  wheel (~900MB, bundles CUDA even for old versions) was infeasible to download in
  this environment (~1.2-1.3MB/s, and `download.pytorch.org` is proxy-blocked); JAX's
  wheels (jaxlib 87MB + jax 2.2MB) were downloaded directly from PyPI instead. Flax/
  Optax were skipped too, to avoid further unknown-size dependency risk — the MLP,
  Adam optimizer, and autograd-based penalty terms are all hand-rolled in raw JAX.
- **Loss (v1)** = data MSE on total variance w = iv²·τ + λ · (mean squared Durrleman
  butterfly-condition violation + mean squared calendar-condition violation
  ∂w/∂τ ≥ 0), both penalty terms evaluated via `jax.grad`/nested `jax.grad` on
  collocation points resampled fresh every training step (k padded 20% beyond the
  observed strike range, τ spanning the observed tenor range) — standard PINN-style
  training.
- **Optimizer**: hand-rolled Adam (lr=1e-3), batch size 8192 (data) / 2048
  (collocation), used for all model versions below.
- **Sweep**: λ ∈ {0, 0.01, 0.1, 1, 10, 100} × 3 seeds = 18 runs, for every model
  version (v1 and v3 below).

## Scope reductions from the plan (stated upfront, same transparency pattern as
Phases 1–2)

1. **One fold instead of all 18** (fold 16 only). Running the full λ×seed sweep
   across all 18 folds would multiply an already compute-heavy sweep 18x, which
   was not feasible in the time available in this session.
2. **3 seeds instead of the planned 5**, per (λ) config, for the same reason.

Both are real trade-offs, not silently made — flagging them the same way the F2
volume threshold, the SSVI two-stage fit, and the thin-plate smoothing parameter
were flagged in earlier phases.

## v1: the first attempt, and why it failed on calendar-arbitrage

### Finding 1 — 1800 training steps was not enough; more steps materially changed
   the result

The first pass trained every config for 1800 Adam steps (chosen because the data
loss visually plateaued by then). Post-hoc evaluation on a dense (k,τ) grid showed
calendar-arbitrage violation rates around 80% for *most* configs, regardless of λ
— including λ=100. Diagnosing this: at k≈0 (ATM), the fitted total variance surface
was *decreasing* with τ past roughly τ≈0.2, sometimes sharply, because real ATM data
at long tenors is extremely sparse in this fold — only **19 data points** in the
entire 5.5-year training window have both |k|<0.02 and τ>0.3. In that data-starved
region the network free-fell toward w≈0, and a lightly-averaged, uniformly-sampled
collocation penalty needed more optimization time to correct it. Extending training
to 4500 steps (in three more 900-step resumable rounds, checkpointed to survive this
environment's per-call time limit) fixed this for some but not all runs — see
Finding 2.

### Finding 2 — the butterfly penalty works cleanly; the calendar penalty does not

| λ | butterfly viol. rate (mean ± std, 3 seeds) | calendar viol. rate (mean ± std, 3 seeds) |
|---|---|---|
| 0 | 0.070 ± 0.067 | 0.367 ± 0.338 |
| 0.01 | 0.004 ± 0.004 | 0.303 ± 0.430 |
| 0.1 | 0.001 ± 0.002 | 0.294 ± 0.445 |
| 1 | 0.000 ± 0.000 | 0.824 ± 0.005 |
| 10 | 0.000 ± 0.000 | 0.613 ± 0.425 |
| 100 | 0.000 ± 0.000 | 0.590 ± 0.468 |

Butterfly: even a tiny λ=0.01 drives the average post-hoc violation rate (on a
121×41 dense grid spanning the padded training domain) down to 0.4%, and λ≥1
eliminates it completely across all 3 seeds — a clean, expected success.

Calendar: **no monotonic improvement with λ**, and enormous seed-to-seed variance
at every λ beyond 0.1 (e.g. λ=10: individual seeds hit 0.885 / 0.123 / 0.830 —
wildly different outcomes from identical hyperparameters, differing only in random
initialization). This means the calendar constraint's satisfaction is closer to a
coin flip determined by initialization than a controllable function of λ, for this
formulation. Root cause (consistent with Finding 1): a small, spatially
concentrated negative ∂w/∂τ contributes a tiny *squared* penalty relative to the
data-loss term, so the L2 penalty gives a weak gradient signal in exactly the
sparse region where the violation lives — increasing λ scales up that weak signal
along with an equally weak one, without reliably escaping the bad basin.

### Finding 3 — RMSE degrades sharply with λ, without buying commensurate
   calendar-arbitrage improvement

| λ | Task A RMSE (mean ± std) | Task B RMSE (mean ± std) |
|---|---|---|
| 0 | 0.078 ± 0.036 | 0.080 ± 0.022 |
| 0.01 | 0.078 ± 0.034 | 0.082 ± 0.022 |
| 0.1 | 0.077 ± 0.023 | 0.083 ± 0.012 |
| 1 | 0.109 ± 0.009 | 0.072 ± 0.015 |
| 10 | 0.165 ± 0.040 | 0.134 ± 0.104 |
| 100 | 0.266 ± 0.210 | 0.235 ± 0.275 |

(`results/phase3_tradeoff.png` plots this alongside the violation rates.)

This was the headline negative result of the v1 attempt: the intended trade-off —
a controlled, monotonic accuracy sacrifice bought for a commensurate arbitrage-
freedom gain — did not materialize with the soft-penalty formulation. High λ
mostly just hurt the data fit (RMSE more than tripled from λ=0.1 to λ=100) while
calendar violations persisted for the reason in Finding 2. Butterfly arbitrage, by
contrast, was fixed essentially for free at λ=0.01, well before RMSE started
degrading.

### An intermediate attempt (v2) that also failed

Before abandoning soft penalties, two reformulations of the calendar term were
tried (`src/06e_nn_train_v2.py`, not committed — kept only as a documented dead
end):

1. A plain L1 penalty `mean(relu(-∂w/∂τ))` instead of squared-L2, on the theory
   that L1 gives a non-vanishing gradient for small violations. Result: *worse* —
   dense-grid violation rate rose to 91% at λ=1, because L1 on the raw slope can
   be driven toward zero by shrinking `|∂w/∂τ|` toward zero from either side,
   without ever flipping its sign positive.
2. The same L1 term with a fixed positive margin (hinge:
   `relu(margin − ∂w/∂τ)`, margin=1e-3), plus oversampled collocation points
   concentrated in the sparse near-ATM long-tenor region. Result: still failed —
   even at λ=100 after 1800 steps, 100% of the dense grid still violated the
   margin, and the logged penalty plateaued around 0.0010–0.0012 regardless of λ
   or additional training. The model was finding a "cheap" near-flat solution in
   the sparse region that bounds the penalty near the margin value without making
   real progress toward genuine positivity.

Both v1's squared-L2 and v2's L1-with-margin penalties share the same underlying
failure mode: in a region with almost no real data to anchor the surface, the
optimizer can satisfy *any* soft penalty on the slope by making the slope small
(collapsing the surface locally) rather than doing the harder work of making it
correctly signed everywhere. Reformulating the penalty function does not fix this,
because the escape hatch is architectural (the model is free to represent *any*
w(k,τ), including ones with locally negative slope), not a matter of which loss
shape is used to discourage it.

## v3: the fix — a hard architectural constraint on calendar-arbitrage

Given that two different soft-penalty formulations failed identically, the fix
implemented here removes the escape hatch entirely by construction, rather than
trying a third penalty shape.

**Parametrization** (`src/06f_nn_train_v3.py`):

w(k, τ) = softplus(base(k)) + Σᵢ softplus(cᵢ(k)) · φᵢ(τ), i = 1..3

- φ₁(τ) = τ, φ₂(τ) = √τ, φ₃(τ) = log(1+τ) — three **fixed**, monotonically
  increasing basis functions of τ alone.
- base(k) and c₁(k), c₂(k), c₃(k) are the 4 outputs of one small MLP that takes
  **k alone** as input (hidden layers [32, 32], softplus activations).

Because every φᵢ is non-decreasing in τ and every coefficient softplus(cᵢ(k)) is
≥ 0, ∂w/∂τ = Σᵢ softplus(cᵢ(k))·φᵢ'(τ) ≥ 0 **for every k and every τ > 0, by
construction** — calendar-arbitrage-free with no training required to enforce it
and no λ needed for it. Butterfly-arbitrage is still enforced the way v1 did
(successfully): a soft squared-L2 penalty on the Durrleman condition, evaluated
via autograd on resampled collocation points, weighted by λ. The v3 λ sweep is
therefore a pure butterfly-vs-fit sweep.

Each v3 run needed more total Adam steps than v1 to fully converge (18,000, and
24,000 for the two slowest λ=100 seeds, vs v1's 4,500) — the k-only sub-network
occasionally sat on a data_mse plateau for several thousand steps before a sharp
drop — but each step is roughly 4x cheaper (no ∂/∂τ term in the main forward pass,
smaller network), so total wall-clock cost was comparable to v1.

### Result 1 — calendar-arbitrage violation is exactly 0.0 at every λ, every seed

| λ | v1 calendar viol. rate | v3 calendar viol. rate |
|---|---|---|
| 0 | 0.367 | **0.000** |
| 0.01 | 0.303 | **0.000** |
| 0.1 | 0.294 | **0.000** |
| 1 | 0.824 | **0.000** |
| 10 | 0.613 | **0.000** |
| 100 | 0.590 | **0.000** |

Verified on the same 121×41 dense (k,τ) grid used throughout Phase 3, across all
18 runs (6 λ × 3 seeds): `calendar_viol_rate = 0.0` and `min(∂w/∂τ) > 0` on the
grid in every single case (smallest margin observed: 4.05×10⁻⁴, at λ=1 seed=1).
This is the fix the user asked for — calendar-arbitrage is no longer a trained
outcome subject to seed variance; it is a structural property of the model.

### Result 2 — bonus: butterfly-arbitrage is now nearly free even at λ=0

| λ | v1 butterfly viol. rate | v3 butterfly viol. rate |
|---|---|---|
| 0 | 0.070 ± 0.067 | 0.0005 ± 0.0008 |
| 0.01 | 0.004 ± 0.004 | 0.000 ± 0.000 |
| 0.1 | 0.001 ± 0.002 | 0.0005 ± 0.0009 |
| 1 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 10 | 0.000 ± 0.000 | 0.0005 ± 0.0006 |
| 100 | 0.000 ± 0.000 | 0.0001 ± 0.0002 |

Unlike v1, where λ=0 gave a meaningfully non-zero butterfly violation rate (7%)
that the penalty had to work to remove, v3's baseline (λ=0, no butterfly penalty
at all) is already at ~0.05% — essentially the same as the fully-penalized v1
runs. This is a side effect of the architecture, not something tuned for: the
k-only sub-network is smaller and smoother than v1's joint (k,τ) network, so it
produces well-behaved smile shapes in k more easily. Practically, this means the
right panel of `results/phase3_v3_tradeoff.png` (fit error vs. butterfly violation
rate) is nearly flat and uninformative in v3 — there just isn't much butterfly
violation left to trade away, at any λ. This should be reported honestly rather
than presented as if the butterfly penalty is doing important work in v3: it
mostly is not, because the architecture already handles it.

### Result 3 — RMSE is better than v1 at low-to-moderate λ, and degrades less
   badly at high λ

| λ | v1 Task A RMSE | v3 Task A RMSE | v1 Task B RMSE | v3 Task B RMSE |
|---|---|---|---|---|
| 0 | 0.0777 | **0.0650** | 0.0797 | **0.0564** |
| 0.01 | 0.0777 | **0.0613** | 0.0816 | **0.0589** |
| 0.1 | 0.0773 | **0.0648** | 0.0832 | **0.0677** |
| 1 | 0.1094 | **0.0843** | 0.0718 | 0.0984 |
| 10 | 0.1648 | **0.1562** | 0.1341 | 0.1890 |
| 100 | 0.2662 | **0.1490** | 0.2348 | 0.1823 |

v3 improves Task A RMSE at every λ, and improves Task B RMSE at λ ≤ 0.1 (it is
worse than v1 at λ ≥ 1, where v1's Task B number is itself noisy and based on a
run that happened to have near-zero calendar-arbitrage compliance by chance — not
a fair comparison, since that v1 run does not actually satisfy the constraint it
is being scored under).

Because λ buys essentially nothing at the butterfly level in v3 (Result 2) but
still costs meaningfully in RMSE beyond λ=0.1 (Result 3), **λ=0.01 is the
practical recommendation** for the paper's headline v3 configuration: best or
near-best RMSE on both tasks, zero calendar violation (guaranteed at every λ), and
zero butterfly violation on the dense grid (0.000 ± 0.000 across all 3 seeds at
this λ).

### Result 4 — the NN still underperforms the per-slice classical baselines on
   Task A, but the gap has narrowed

| method | Task A RMSE, in-range (IV, decimal) |
|---|---|
| ThinPlate (Phase 2) | 0.0042 |
| SSVI (Phase 2) | 0.0216 |
| SVI (Phase 2) | 0.0239 |
| NN v1, λ=0.1 (best v1 config) | 0.0732 |
| **NN v3, λ=0.01 (best v3 config)** | **0.0590** |

Still worse by a factor of ~2.5–14x, for the same reason documented for v1: the
Phase 2 baselines re-fit a fresh curve to each day's own 80% of visible strikes
and are scored on that same day's remaining 20% — a much easier task than one
global (k,τ)→w surface serving 5.5 years of pooled data with no per-date
parameters. This comparison remains apples-to-oranges in terms of information
available, and the paper should continue to say so plainly. v3's ~19% RMSE
improvement over v1 (0.0732 → 0.0590) narrows the gap somewhat but does not close
it, nor was closing it the point of this fix — the point was removing calendar-
arbitrage violation, which is now done.

### Result 5 — Task B (unseen dates) observation, updated

At v3's recommended λ=0.01, Task B RMSE (0.0589) is close to and even slightly
better than Task A RMSE (0.0613) — consistent with the v1-era tentative
observation that the pooled global surface generalizes reasonably across time,
which a per-day-refit baseline structurally cannot be evaluated on at all (Phase 2
noted "Task B does not apply" to the classical baselines). This remains a
descriptive observation rather than a rigorously validated claim, since it is
based on one fold and 3 seeds.

## Positioning against prior work (added after a literature check)

Before writing up v3's result as "hard-constrained beats soft-penalty on both
accuracy and arbitrage-freedom," a literature check turned up prior work making
essentially the same core claim, so the paper's framing needs to change to avoid
presenting a known result as new:

- **Zheng, Yu & Li, "A Two-Step Framework for Arbitrage-Free Prediction of the
  Implied Volatility Surface"** (arXiv:2106.07177; *Quantitative Finance* 23(1),
  2022). Their surface-reconstruction step is a DNN with a hard architectural
  no-static-arbitrage constraint, and their headline empirical finding is that
  the hard constraint "not only removes static arbitrage, but also significantly
  reduces the prediction error compared with a standard interpolation method" —
  the same directional claim (hard constraint helps both axes) that v3's result
  would otherwise be presented as newly demonstrating.
- **Bergeron, Fan & Roux (or similar), "Beyond Surrogate Modeling: Learning the
  Local Volatility Via Shape Constraints"** (arXiv:2212.09957) directly contrasts
  a hard-constrained Gaussian process ("proven arbitrage-free") against
  soft-penalized NN/SSVI approaches, framing it explicitly as a trade-off between
  guaranteed constraint satisfaction and model flexibility/complexity — useful
  as a more skeptical counterpoint (hard constraints aren't free, they note a
  complexity cost) to cite alongside Zheng et al.
- **Ackerer, Tagasovska & Vatter, "Deep Smoothing of the Implied Volatility
  Surface"** (NeurIPS 2020) is the standard reference for the soft-penalty/
  PINN-style tradition that v1 (and this project's initial approach) sits in,
  and should be cited as the origin of that family rather than treating v1's
  formulation as if it were novel.

**Revised framing for the paper**: the contribution is not "we show hard beats
soft" (Zheng et al. already showed this on what is presumably a different
market/dataset). What remains genuinely contributable here:

1. **Market**: this is, as far as this literature check found, the first
   application of a hard-constrained arbitrage-free architecture to NSE
   (Indian) index options — all three papers above work with markets outside
   India. That is a legitimate, IEEE-Access-appropriate contribution in its own
   right (application/regional novelty), separate from the constraint-mechanism
   question.
2. **A different, simpler hard-constraint parametrization**: v3's construction
   (a k-only network times fixed, non-negative τ-basis functions) is a distinct
   architectural choice from Zheng et al.'s DNN constraint or the shape-
   constraints paper's Gaussian process, worth describing as an alternative
   instantiation rather than *the* first hard-constraint method.
3. **The diagnostic contribution**: the documented, reproducible account of
   *why* two different soft-penalty formulations (squared-L2 in v1, L1-with-
   margin in v2) fail identically — the sparse-data "escape hatch" mechanism,
   where the optimizer satisfies the penalty by flattening the surface rather
   than correcting its sign — is a case-study-level finding that doesn't depend
   on "hard beats soft" being new. This is arguably the most original material
   in Phase 3 and should be given more weight in the writeup than the
   already-known headline comparison.
4. **Evaluation rigor**: the Task A (held-out strikes, train window) / Task B
   (entirely unseen dates) walk-forward split is more demanding than the
   train/test evaluations used in at least the two-step-framework paper's
   excerpt reviewed here, and is worth foregrounding as a methodological
   strength independent of the constraint-mechanism question.

**An honest confound to disclose, not paper over**: v1 and v3 differ in more
than the constraint mechanism — v1 is a joint (k,τ) MLP, v3 decomposes into a
k-only network times fixed τ-basis functions. Some of v3's RMSE improvement
(Result 3) could come from that decomposition being a better inductive bias for
this data, independent of whether the calendar constraint is hard or soft. The
paper should say this plainly rather than let "v3 beats v1" imply a clean,
isolated ablation of the constraint mechanism alone. A stronger (optional, not
yet done) version of this experiment would hard-constrain something closer to
v1's architecture, to isolate the mechanism's effect from the architecture
change.

## Bottom line

The user's ask — "recompute and fix the calendar thing" — is done, and stands on
its own regardless of the positioning question above. Two soft-penalty
reformulations (v1's squared-L2, v2's L1-with-margin) both failed to reliably
enforce ∂w/∂τ ≥ 0 for the same underlying reason (a data-sparse region that lets
the optimizer cheat the penalty by flattening the surface rather than correcting
its sign). The fix was architectural, not a loss-function tweak: v3
reparametrizes w(k,τ) as a sum of products of non-negative, k-dependent
coefficients and fixed, τ-monotonic basis functions, which makes ∂w/∂τ ≥ 0 true
for every input by construction. Verified across all 18 (λ, seed) runs:
calendar-arbitrage violation is exactly 0.0 in every case. As a side benefit,
butterfly-arbitrage is now nearly free even without its penalty (architecture-
driven, not tuned), and RMSE is better than the original attempt at low-to-
moderate λ. The recommended configuration for the paper is v3 at λ=0.01.

For the paper's framing (see "Positioning against prior work" above): this
project's result confirms, on a new market and with a different hard-constraint
construction, a finding Zheng et al. (2022) already published — it should not be
presented as a novel discovery that hard constraints beat soft ones. The
paper's more defensible original contributions are the NSE application itself,
the specific basis-function construction, the documented soft-penalty failure
diagnosis (v1/v2), and the Task A/B evaluation rigor. Also carried forward:
because butterfly violation is already near-zero in v3 at λ=0, the originally-
envisioned "fit error vs. arbitrage-freedom" trade-off curve is not a meaningful
two-axis trade-off in this data — there is very little arbitrage left to trade
fit for, and that should be stated plainly rather than implied away.

## Artifacts

- `src/06a_prep_fold16.py` — extracts fold 16's train-visible / train-held-out /
  test rows into `06_nn/fold16_data.npz`.
- `src/06b_nn_train.py` — v1: the joint (k,τ)→w JAX MLP, both butterfly and
  calendar penalties as soft squared-L2. `06_nn/runs/lam{λ}_seed{seed}/
  {ckpt.npz,log.csv}` per run.
- `src/06c_nn_eval.py` — v1 Task A/B RMSE (in-range/out-of-range) and post-hoc
  butterfly/calendar violation rates on a dense grid, per run. Outputs
  `results/nn_eval_summary.csv`, `results/nn_violation_grid.csv`.
- `src/06d_nn_report.py` — aggregates v1 to mean±std per λ
  (`results/nn_eval_agg.csv`, `results/nn_violation_agg.csv`) and builds
  `results/phase3_tradeoff.png`.
- `src/06e_nn_train_v2.py` — intermediate, failed attempt (L1-with-margin
  calendar penalty + oversampled sparse-region collocation points); kept as a
  documented dead end, not evaluated or reported on beyond the diagnosis above.
- `src/06f_nn_train_v3.py` — **the fix**: hard-constrained basis-function
  architecture for calendar-arbitrage, soft butterfly penalty as in v1.
  `06_nn/runs_v3/lam{λ}_seed{seed}/{ckpt.npz,log.csv}` per run (16 runs at
  18,000 Adam steps, 2 runs — λ=100 seeds 1 and 3 — at 24,000 steps to reach
  convergence).
- `src/06g_nn_eval_v3.py` — v3 Task A/B RMSE and violation rates, same
  methodology as `06c_nn_eval.py`. Outputs `results/nn_v3_eval_summary.csv`,
  `results/nn_v3_violation_grid.csv`.
- `src/06h_nn_report_v3.py` — aggregates v3 to mean±std per λ
  (`results/nn_v3_eval_agg.csv`, `results/nn_v3_violation_agg.csv`), builds the
  v1-vs-v3 comparison table (`results/nn_v1_vs_v3_comparison.csv`), and the
  corrected trade-off figure `results/phase3_v3_tradeoff.png`.
