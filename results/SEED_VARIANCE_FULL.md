# Full seed-variance check (3 seeds x all 18 folds) — v3

**Standalone results file. Nothing under `paper/` has been touched.**
Source data: `results/phase4_seed_variance_full.csv`,
`results/phase4_seed_variance_full_summary.csv`,
`results/phase4_seed_variance_full_headline.csv`.
Training driver: `seed_full/run_missing_seeds.py`. Aggregation script:
`seed_full/aggregate_full_seed_variance.py`.

## What this answers

Reviewer objection #7 (Tier-2 critique): the paper's headline v3 numbers
rested on a single seed per fold, plus a 3-seed check on only 3 of 18 folds
(the original `ablation/08a_seed_variance.py` / `phase4_seed_variance.csv`,
folds {2, 9, 17}). That check reported a noise floor of roughly ±0.34
(Task A) to ±0.54 (Task B) vol points, but a reviewer could reasonably ask
whether that 3-fold sample generalizes, and whether the headline
mean-across-folds numbers move with seed.

This extends the same check to **all 18 folds**, so every fold now has
seeds {1, 2, 3} trained (54 checkpoints total; 24 already existed, 30 were
trained for this check — see `seed_full/progress.log` for the run log, 0
failures).

## Headline result

Table `tab:accuracy`'s own methodology: mean RMSE across the **17 blind
folds** (fold 16 excluded — it reused a Phase-3 model-selection checkpoint
and is disclosed as non-blind in the paper). Computed separately per seed,
then mean ± sd across the 3 seeds:

| Task | seed 1 (as published*) | seed 2 | seed 3 | mean ± sd across 3 seeds |
|---|---|---|---|---|
| A (training-window hold-out) | 6.680 | 6.681 | 7.237 | **6.87 ± 0.32** vol pts |
| B (test-window, out-of-sample) | 5.077 | 4.835 | 5.706 | **5.21 ± 0.45** vol pts |

\* seed 1's Task A number (6.680) matches the paper's published 6.68
exactly. Seed 1's Task B number (5.077) is close to but not identical to
the paper's published 5.10 — the paper applies a further "matched 20%
re-score" adjustment (Sec. 6.x) on top of this base Task A/B methodology
that this script does not reproduce (its source data isn't in a file this
script reads). Treat the comparison as approximate: same pipeline stage as
`results/phase4_nn_eval_by_fold.csv`, not a recomputation of the exact
published post-adjustment figure.

## Noise floor, confirmed at full scale

| Task | within-fold, across-seed std (mean over 18 folds) | across-fold std (seed 1 only) | ratio (fold effect / seed noise) |
|---|---|---|---|
| A | 0.353 | 0.776 | 2.20x |
| B | 0.508 | 1.232 | 2.43x |

This matches the original 3-fold check's disclosed ±0.34–0.54 almost
exactly (0.353 and 0.508 here), now backed by all 18 folds instead of 3.
Fold-to-fold variation is real and roughly 2.2–2.4x larger than seed noise
— folds are genuinely different in difficulty, not just noisy — but seed
noise is not negligible either: it's large enough that a single-seed
headline number can move by several tenths of a vol point purely from
initialization.

## A pattern worth flagging

Seed 3 is consistently the "worst" seed on Task A for folds 9–17 (roughly
0.6–1.1 vol points above seeds 1/2 on those folds specifically), while
being close to seeds 1/2 on folds 0–8. This looks like an unlucky
initialization interacting with the later, larger training windows rather
than a systematic problem — worth a quick look if this gets reported, but
it is exactly the kind of single-run artifact that mean ± sd across seeds
is meant to catch and average out.

## What this does NOT do

- Does not change any number in `paper/`. Nothing under `main.tex`,
  `refs.bib`, or any `sections/*.tex` file was modified.
- Does not reproduce the paper's "matched 20% re-score" Task B adjustment.
- Does not extend the calendar/butterfly arbitrage-violation tables to
  3 seeds (only Task A/B RMSE was in scope for this check) — that would be
  a small additional script reusing the same checkpoints if wanted.
