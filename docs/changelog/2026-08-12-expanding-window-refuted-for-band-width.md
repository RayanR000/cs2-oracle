# The expanding window is not why the band over-covers — the calibration window's regime is

**Date:** 2026-08-12
**Runs:** `31611508808` (control, `CV_MAX_TRAIN_ROWS=300000`, `cv_diagnostic_classifier=q50`,
all four horizons, same commit)
**Status:** ❌ hypothesis REFUTED. Nothing about the served band changed.

## What was being tested

The served band over-covers at **87.2 / 91.8 / 90.6 / 89.0%** against an 80% target (19,917
prod outcomes), needing the half-width to shrink to **0.72 / 0.61 / 0.73 / 0.77**. Four causes
were already excluded — the calibration centre, quiet forecast dates, the calibration
denominator, and `sigma`'s level. The leading remaining hypothesis was the **expanding window
itself**: `q_hat` is fitted on OOF residuals from fold models trained on 82K–300K rows while
the shipped model trains on the full 1.2M budget, so split conformal's exchangeability
assumption fails and a pooled `q_hat` is conservative *by construction*.

The hypothesis was not invented for this investigation. `CV_MAX_TRAIN_ROWS`' own comment
(shipped 2026-08-09 as `6b6fc81`, a cost cap) predicted it:

> The fold model then fits on less data than the served one, so `q_hat` comes out LARGER and
> the band wider — over-coverage, which is the safe direction. Verify against
> `NOMINAL_COVERAGE` before lowering it.

The band does now over-cover. This is that verification.

## The measurement

`_cv_evaluate_horizon` now calibrates each fold's own records and reports `fold_q_hat`;
`_train_horizon_inline` reduces the series to one `Expanding-window audit` line and a
`q_hat_trend` block in `cv_results`, which reaches `meta.json` via `per_fold`.

| h | pooled `q_hat` | fold series |
|---|---:|---|
| 3 | 95.9 | 127.0 → 61.5 → 91.2 → 96.0 → 81.3 → 101.8 → 126.5 → 93.2 → 84.8 |
| 7 | 143.0 | 239.4 → 90.4 → 152.0 → 152.9 → 107.2 → 140.4 → 173.2 → 119.6 |
| 14 | 207.4 | 400.1 → 152.3 → 213.1 → 223.1 → 148.3 → 192.8 → 236.5 → 148.9 |
| 30 | 315.1 | 774.3 → 310.6 → 265.4 → 438.4 → 194.6 → 266.8 → 356.7 → 200.6 |

Pooled values land within 1.5% of the 2026-08-12 control (`31564943172`: 94.72 / 141.77 /
204.34 / 312.05), so the run is comparable and HP reuse held.

## Why it is refuted

**1. The pooled `q_hat` is not inflated — it sits BELOW the well-trained folds' own p80.**
`CV_MAX_TRAIN_ROWS` binds from fold 4 on, so folds 4–9 all train on exactly 300,000 rows.
Against those folds' own 80th percentile the pooled value is **0.94 / 0.91 / 0.92 / 0.84×**.
If the pooled fit were inheriting weakness from the small-`n` early folds this ratio would
exceed 1. It is below 1 at every horizon.

**2. Dropping the weakest-trained fold moves `q_hat` the WRONG way.** Fold 1 is the only fold
materially below the cap (74–87K rows) and is 2.07–2.65× fold 2 — the single most
hypothesis-consistent data point in the run. But it carries only 8.8–10.0% of the OOF rows, and
excluding it *widens* the calibration to **1.06 / 1.07 / 1.08 / 1.13×** pooled, against a
needed 0.61–0.77. Same failure shape as the served-basis arm: the remedy has the wrong sign.

**3. At identical training size, `q_hat` still varies 1.56–2.25×.** Among folds 4–9, where
`n_train` is pinned at 300,000, the spread is 81.3–126.5 (h=3), 107.2–173.2 (h=7), 148.3–236.5
(h=14) and 194.6–438.4 (h=30). Training size cannot explain any of that, and it is the same
order as the whole effect being chased.

The `Expanding-window audit` line's Spearman is **−0.05 / −0.14 / −0.36 / −0.48**, which looks
mildly supportive and is not: with the cap binding there are only **4 distinct `n_train`
values** across 8–9 folds, and the rho is almost entirely fold 1's leverage. That is why the
log line and `q_hat_trend` also carry `n_train_distinct` and `cv_max_train_rows` — a rho read
without them would have been published as weak confirmation of a refuted hypothesis.

## What the folds actually track

The variation is the **calibration window's volatility regime**, and it is synchronised across
horizons. Ranking folds 4–8 (identical `n_train`) within each horizon, narrowest to widest:

| val window | h=3 | h=7 | h=14 | h=30 |
|---|---:|---:|---:|---:|
| 2024-05 | 3 | 4 | 4 | 5 |
| 2024-10 | 1 | 1 | 1 | 1 |
| 2025-03 | 4 | 3 | 3 | 3 |
| 2025-08 | 5 | 5 | 5 | 4 |
| 2026-01 | 2 | 2 | 2 | 2 |

Pairwise Spearman between horizons is **0.70–1.00**, and h=7/h=14 agree perfectly. One market
factor sets the cross-sectional dispersion in a window, and every horizon's `q_hat` reads it.

**This is the same fact as the 58.2–99.2% per-date coverage.** `q_hat` is a regime-dependent
quantity being served as a constant, and the pooled value lands where 2023–2026 averages out —
too wide for the calm regime production is currently serving into.

## The one estimator that moves the right way, and its limit

The most recent fold's own `q_hat` is **0.88 / 0.84 / 0.72 / 0.64×** pooled — the right
direction at 4/4, delivering **41% / 42% / 105% / 158%** of the needed shrink. No previously
tested candidate managed the right sign at all.

⚠️ **A trailing 2–3 fold window is WORSE than pooled** (1.13–1.21×), because the volatile
2025-08 window dominates the p80 of any set containing it. So "calibrate on the late folds",
the remedy the expanding-window hypothesis implied, does not work either — only the single most
recent fold does, and that is an 18–23K-row estimate against a 1.56–2.25× fold-to-fold spread.
Reading one fold is exactly the high-variance mistake this project has made before.

## What this closes

Five causes are now excluded: the calibration centre, quiet forecast dates, the calibration
denominator, `sigma`'s level, and the expanding window / training-size gap. Together they rule
out the whole class of *"calibrate on different rows"* remedies — including the part of C5
scoped as a cost change and re-scoped as a correctness one, which is now neither.

What is left is not a choice of calibration set. A **scalar** `q_hat` cannot track a
several-fold swing in cross-sectional dispersion, so the remedy class is a **conditional**
`q_hat` — time-varying, or a function of a realised-volatility state — and that is the same
change the per-date coverage problem needs. It has not been designed, costed or measured.

## Reproducing

`CV_MAX_TRAIN_ROWS` is now a `model-diagnostics.yml` input, so the training-size axis can be
swept directly rather than inferred from fold geometry. It is worth doing only to size the
sensitivity, not to fix coverage — points 1 and 2 above do not depend on the cap's value.
⚠️ Raising it scales the conformal-CV phase roughly linearly off a base that was 50.4% of an
872s retrain, so 1.2M will not fit the 30-minute working cap. Sweep 600000 first.

The arm label carries `+cvrows<N>` and `meta.json` carries
`cv_results[h].q_hat_trend.cv_max_train_rows`, so no run is ambiguous about which cap it used.
