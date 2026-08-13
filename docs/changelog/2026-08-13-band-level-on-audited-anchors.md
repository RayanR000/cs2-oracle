# The band over-covers by 6-9pp on four audited anchors, and the served width is 1.53x its own calibration

**Date:** 2026-08-13
**Run:** `31657639707`, commit `d51fa32`, `horizons=matrix`, control (no arm flags),
`replay_anchors=2026-04-22,2026-05-16,2026-06-16,2026-07-06`
**Follows:** `2026-08-12-july-09-anchor-is-a-feed-substitution.md`, which retired the old anchor set
**Status:** ✅ the first band read on four anchors that all pass the collection audit. The tilt is
confirmed a third time. The **level** is larger than published and now has a candidate mechanism.

## The audit worked, and CI agrees with the local archive

All four anchors passed, and the item counts CI read match the local audit exactly — 25,000 /
24,965 / 25,953 / 26,076. The replay loop runs under `-e` with `set -o pipefail`, so a refusal would
have killed the step rather than printing a warning nobody reads.

The downstream symptom agrees too. Tied shares are **362 / 598 / 445 / 404 of ~1,060** (33.7 / 56.4 /
41.6 / 39.3%), with no 2.5% outlier of the kind 2026-07-09 produced.

One unplanned gain: **2026-07-06 is an up-majority date at h=3** (33.89% realised down-rate). The
retired set was down-majority at 4 of 4 anchors at every horizon, so every DA figure it produced was
scored against a one-sided composition.

## The level

| h | cov% published (old 4) | predicted from the 2 clean | **measured (audited 4)** | vs 80% |
|---|---:|---:|---:|---:|
| 3 | 77.80 | 85.78 | **87.58** | **+7.58** |
| 7 | 84.94 | 90.20 | **88.97** | **+8.97** |
| 14 | 84.76 | 84.14 | **86.09** | **+6.09** |
| 30 | 87.07 | 86.24 | **86.19** | **+6.19** |

**Over-coverage at 4 of 4 horizons and in 15 of 16 cells** (the exception is 2026-07-06 at h=30,
77.94%). The prediction made from the two surviving anchors of the retired set lands within
**1.8 / 1.2 / 2.0 / 0.05pp** — so dropping the two mis-collected anchors was not a matter of taste,
and the published 77.80% at h=3 was the artifact it was diagnosed as.

Median half-widths are unchanged from the retired set (9.80 / 14.75 / 21.22 / 31.32% against
9.80 / 14.56 / 20.87 / 31.32%). Only *which* dates and items are scored moved, which is what makes
these two reads comparable at all.

## The tilt, a third time

Coverage still ramps across quintiles of the served width: **+7.20 / +11.09 / +11.97 / +11.03pp**
mean, positive in **15 of 16 cells** (again 2026-07-06 at h=3, −0.04). Bigger at 7d/14d/30d than the
retired set's +9.94 / +8.10 / +9.13. The fitted elasticities reproduce yet again — **0.431 / 0.368 /
0.335 / 0.317** against 0.429 / 0.369 / 0.350 / 0.313 on the OOF residuals.

So `sigma` is the right *shape* variable and three attempts to exploit it have all failed on the
level. Nothing here reopens `SIGMA_EXPONENT` or `LEARNED_SCALE`; both stay off.

## 🔑 The new lead: 1.53x, and it is flat in horizon

`q_hat` is calibrated to a median half-width of **6.41 / 9.53 / 13.66 / 20.61%** of the mid on its own
OOF records. The same `q_hat`, served, produces **9.80 / 14.75 / 21.22 / 31.32%**:

| h | calibration halfw% | served halfw% | **ratio** |
|---|---:|---:|---:|
| 3 | 6.41 | 9.80 | **1.529** |
| 7 | 9.53 | 14.75 | **1.548** |
| 14 | 13.66 | 21.22 | **1.553** |
| 30 | 20.61 | 31.32 | **1.520** |

**1.52-1.55x at all four horizons** — a constant, which is what a level defect looks like and what a
horizon-dependent one does not. It is also *bigger than the known cause*: served `sigma` was measured
at **1.28-1.29x** the calibration median (`2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md`),
and under `beta = 1` the width ratio should equal the `sigma` ratio. **1.53 / 1.29 = 1.19 is
unaccounted for**, and the two candidates are the denominator (the served mid is not the calibration
mid) and the cohort (the served ≥$1 panel of ~1,030 items on 4 dates against 155-176K pooled OOF
rows).

⚠️ **This is one comparison of two medians from a log, not a paired read.** It is a lead, not a
result: the calibration figure is pooled over the whole OOF cohort and the served figure is a median
over four dates, so a cohort difference alone could produce it. The next step is to compute both on
the *same* items — which needs no retrain, only the served rows joined to their own OOF records.

## Not to be misread

DA is **43.41 / 46.72 / 48.04 / 55.10%** against realised down-rates of **54.51 / 70.03 / 78.38 /
76.39%** (`backend/AGENTS.md` invariant 4: never quote one without the other). Four dates cannot
support a skill claim in either direction — `da-excess-needs-50-dates` puts 56-89% of a pooled gap
like this in market composition. **Nothing in this entry is a statement about directional accuracy.**
