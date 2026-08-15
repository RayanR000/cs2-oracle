# The CV leg and the serving leg of every arm are measured on disjoint calendar periods

**Date:** 2026-08-13
**Status:** ⚠️ **Diagnosis only — nothing fixed, nothing dispatched.** No code changed.
**Instrument:** code read of `models/forecaster.py::_compute_cv_splits` plus the fold table already
stored in `backend/models/saved_models/meta.json` (`cv_results[h].per_fold`), artifact trained
2026-08-09.
**Opens:** a fifth explanation for C1's CV→serving gap, after
`2026-08-13-cohort-geometry-is-not-c1s-gap.md` declared the named four spent. It is **not** one of
those four — it is not a cohort or a path difference, it is a **calendar** difference.

## The observation is not new; the cause and the consequence are

`docs/research/2026-08-10-training-cost-levers.md` already records it, in a footnote, as a bound on
what the numbers mean:

> *"For 7d, 14d and 30d the last CV fold ended 2026-02-09 on the 2026-08-09 artifact — only 3d
> reached July. `q_hat`, rank IC and PT for three of four horizons were calibrated on data ending
> before the 2026-03-22 consensus break and every July cutover. Not a cost issue; flagged because
> it bounds what those numbers mean."*

What that entry did not do is connect it to the arm-reading protocol. Every C1, N1 and
feature-contribution read of the last two weeks is a comparison of a **CV** rank IC against a
**served** rank IC on anchors `2026-04-18 … 2026-06-08`. For three of four horizons those two legs
share **no dates at all**.

| h | folds | last validation window | overlap with the six serving anchors |
|---|---:|---|---|
| 3 | 9 | 2026-06-13 → 2026-07-21 | none (anchors end 06-08, fold starts 06-13) |
| 7 | 8 | **2026-01-11 → 2026-02-09** | **none — 68 days short** |
| 14 | 8 | **2026-01-11 → 2026-02-09** | **none** |
| 30 | 8 | **2026-01-11 → 2026-02-09** | **none** |

The CV leg for 7/14/30d stops **before** `2026-03-22`, which this repo's own
`2026-08-11`-era work characterises as a synthetic market-wide consensus break, and before every
July feed cutover.

## The cause is grid alignment, and it is a knife edge

`models/forecaster.py:3646`:

```python
for end in range(min_train, len(sorted_dates) - val_window + 1, step):
```

with `CV_MIN_TRAIN_DAYS = 200` (`:594`), `VALIDATION_WINDOW_DAYS = 30` (`:346`) and
`CV_STEP_DAYS = 150` (`:604`). The grid is **anchored to the start** of `sorted_dates` and strides
forward, so the last fold lands wherever the stride lands — up to `step - 1 = 149` days short of
the frame's end. Nothing pulls a fold to the present.

`sorted_dates` is the distinct dates surviving `dropna(subset=[target_return_{h}d])`, so it is
~`horizon` days shorter at longer horizons — about **4 dates** between h=3 and h=7 on a dense
frame.

Solving the fold counts back (⚠️ **arithmetic, not measured** — `len(sorted_dates)` is not
serialised):

- h=3 admits a 9th fold, so `200 + 8·150 = 1400 ≤ len₃ − 29`, i.e. **`len₃ ≥ 1429`**
- h=7 does not, so **`len₇ ≤ 1428`**

The three long horizons therefore miss their 9th fold by **single-digit dates**, and the penalty
for missing it is a full `CV_STEP_DAYS` — 150 days — of the most recent validation coverage. That
is the whole of the gap in the table above.

This also explains why h=3 looks different from its neighbours in every read: nothing about h=3 is
better, it simply won the rounding.

## Why this is worth a fifth explanation

`2026-08-13-serving-transforms-do-not-explain-the-cv-gap.md` established that with all three
`predict()` transforms removed, **CV still says +0.0684 / +0.0759 / +0.0588 / +0.0408 at 4 of 4
while serving says +0.040 / +0.006 / −0.026 / −0.080 on the mean**, and called the residual "the
finding". The four explanations examined and spent — label denominator, anchor set, serving
transforms, cohort geometry — are all about *which rows* or *which code path*. None is about
*which months*.

An effect that is real in 2023–2025 and absent in 2026 Q2 would produce **exactly** this pattern,
including the detail that h=3 (the only horizon whose CV reaches 2026-06/07) is also the only
horizon where C1's serving leg passed its bar at 4/6.

⚠️ **This is a hypothesis with a named instrument, not a result.** It does not un-refute C1. It
says the refutation was read against a CV leg that cannot see the period the serving leg is drawn
from, and that this was never controlled for.

## Two instruments, in cost order

**(a) Free, no retrain, decides the hypothesis.** `rank_ic` is already stored per fold
(`cv_results[h].per_fold[i].rank_ic`, alongside `val_start` / `val_end` — see the shipped
`meta.json`). Pull `per_fold` from the C1 pair that has already run — control `31663300447`, arm
`31663312585`, commit `ee76a9c` — and compute the arm−control `rank_ic` edge **per fold**. If the
+0.04–0.08 pooled gain is carried by pre-2026 folds and collapses on the 2026-01/02 fold, the CV
leg and the serving leg are measuring different regimes and the gap is explained. If the edge is
flat across the fold sequence, this explanation is spent too and should be written up as such.

**(b) The fix, ~5 lines.** Anchor the fold grid to the **end** of `sorted_dates` rather than the
start, so the newest fold always abuts the frame's end and the grid walks backwards from there.
Fold count is unchanged or +1 per horizon (~13s each in the conformal CV, which is 50.4% of a
retrain — so budget it as ~1% of training, not free). ⚠️ **Two things it would move that are not
accuracy:** `q_hat` is calibrated on those same out-of-fold residuals, so the served band width
changes; and the PT sample changes, so `invariant_4_signal` must be re-read, not carried over.
Neither is a reason not to do it — a `q_hat` fitted on a window ending 2026-02-09 and served in
2026-08 has the same defect the rest of this entry describes.

## What must not be concluded from this

- **Not** that the CV numbers are wrong. They are correct for the period they cover.
- **Not** that C1 is back. Instrument (a) has not been run.
- **Not** that this is the only remaining explanation. It is the first one proposed that is about
  time rather than rows, and it is cheap enough to settle before proposing a sixth.
