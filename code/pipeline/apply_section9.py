# -*- coding: utf-8 -*-
import io

path = "results/phase4_notes.md"
with io.open(path, "r", encoding="utf-8") as f:
    content = f.read()

SECTION9 = """## 9. Baseline arbitrage-violation measurement: does the guarantee actually differentiate the NN?

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

"""

marker = "## Bottom line"
assert marker in content, "marker not found"
content = content.replace(marker, SECTION9 + marker, 1)

# Revise the first Bottom-line paragraph to add the nuance from section 9.
old_para = (
    "The v3 hard-calendar-constraint architecture's core claim from Phase 3 — "
    "arbitrage-free by construction, essentially for free — is now confirmed at "
    "full walk-forward scale: 0% butterfly and calendar violations across all 18 "
    "independently-trained folds, not just the one fold examined in Phase 3. That "
    "result stands on its own and is the paper's most defensible contribution "
    "(see `phase3_notes.md`'s positioning section for how this is framed against "
    "Zheng/Yu/Li 2022 and related prior art)."
)
new_para = old_para + (
    " Measured against the four classical baselines (§9), that guarantee is a real "
    "differentiator against SVI, ThinPlate, and CubicSpline (all show meaningful "
    "butterfly and/or calendar violations on this dataset), but not against SSVI, "
    "which is empirically arbitrage-free too -- there the NN's advantage is "
    "architectural (one persistent global fit vs. a fresh per-day joint "
    "optimization), not a guarantee SSVI lacks."
)
assert old_para in content, "bottom line first paragraph not found"
content = content.replace(old_para, new_para, 1)

# Append new artifact to the Artifacts list.
artifacts_addition = (
    "- `src/07i_baseline_arbitrage_check.py` — butterfly/calendar violation rates "
    "for all four baselines, full 1496-day panel. → `results/phase4_baseline_butterfly_full.csv`, "
    "`results/phase4_baseline_calendar_full.csv`, `results/phase4_baseline_arb_summary_{butterfly,calendar}.csv`.\n"
)
content = content.rstrip("\n") + "\n" + artifacts_addition

with io.open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("done, new length:", len(content))
