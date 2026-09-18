# -*- coding: utf-8 -*-
import io

path = "results/phase4_notes.md"
with io.open(path, "r", encoding="utf-8") as f:
    content = f.read()

ADDENDUM = """## 8. Quick-fix addendum: seed variance, CubicSpline robust stats, Task B/volatility-regime check

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

"""

marker = "## Bottom line"
assert marker in content, "marker not found"
content = content.replace(marker, ADDENDUM + marker, 1)

# Update the Bottom line CubicSpline sentence to point at the new robust-stats finding.
old_sentence = (
    "The apparent exception (CubicSpline) is an artifact of that baseline's tail "
    "instability, not a genuine NN advantage, and should be reported as such."
)
new_sentence = (
    "The apparent exception (CubicSpline) is an artifact of that baseline's tail "
    "instability, not a genuine NN advantage, and should be reported as such — see "
    "§8.2, where CubicSpline's median error is shown to be competitive with "
    "ThinPlate's on a typical day, so the RMSE gap is a tail effect, not evidence "
    "that CubicSpline is broadly unreliable."
)
assert old_sentence in content, "bottom line sentence not found"
content = content.replace(old_sentence, new_sentence, 1)

# Append new artifacts to the Artifacts list.
artifacts_addition = (
    "- `src/08a_seed_variance.py` — seed-variance check (2 extra seeds × 3 folds). "
    "→ `results/phase4_seed_variance.csv`, `results/phase4_seed_variance_summary.csv`.\n"
    "- `src/08b_taskb_vol_regime_check.py` — Task B/volatility-regime confound check. "
    "→ `results/phase4_taskb_vol_regime_check.csv`.\n"
)
content = content.rstrip("\n") + "\n" + artifacts_addition

with io.open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("done, new length:", len(content))
