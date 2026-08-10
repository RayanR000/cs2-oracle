# Two serving-path calibration defects: the band is recentred after calibration, and the served confidence is uncalibrated

**Date:** 2026-08-10
**Found by:** reading the `predict` path against `_calibrate_conformal` during the model audit
recorded in `docs/changelog/2026-08-10-constant-call-is-hindsight-picked.md`.
**Status:** ⚠️ **diagnosis only, nothing fixed.** Both are code-path findings with supporting
production numbers; neither has a controlled measurement yet. Tracked as **F1** and **F2** in
`docs/research/2026-08-10-next-steps.md`.

## 1. `q_hat` is calibrated around the q50 mid, then the mid is moved three times

`_calibrate_conformal` fits `q_hat` on `residual_pct = actual_ret − mid_ret`
(`_conformal_records`, `forecaster.py:5432-5468`), where `mid_ret` is the **q50 model's** output. So
the band is calibrated to contain the outcome 80% of the time *when centred on the q50 mid*.

`predict` then moves that centre three times, each step documented as "preserving interval
half-widths":

| order | step | `forecaster.py` | effect on the centre |
|---:|---|---|---|
| 1 | `_blend_returns_with_prior` | `:6075` | pulls 15% toward yesterday's forecast (`FORECAST_BLEND_WEIGHT`) |
| 2 | per-tier bias correction | `:6083` | additive shift, fallback path only |
| 3 | `_recenter_on_direction` | `:6101` | `mid → +|mid|`, `−|mid|`, or **exactly 0** |

**Step 3 is the large one and it is unconditional in production.** A "down" call turns a `+2.0%` mid
into `−2.0%` — a 4pp translation of a band whose half-width is often of the same order. A "flat"
call sets the centre to exactly zero regardless of what the q50 said. The half-widths are preserved,
so the band is the right *width* around the wrong *centre*, and the coverage guarantee does not
survive the move.

The code says so, in a way that reads as reassurance: *"Applied last, after all return-space
corrections, preserving interval half-widths."* Preserving half-widths is precisely what breaks
coverage here — the calibration is a statement about a centre, not a width.

**Supporting production numbers** (run `31409508960`, `>=$1` cohort, `IntCov` against
`NOMINAL_COVERAGE = 0.80`):

| cohort | 3d | 7d | 14d | 30d |
|---|---:|---:|---:|---:|
| `lgbm-v3` | 54.6% | 57.3% | 61.8% | **34.6%** |
| `lgbm-v3-global-only` | 87.8% | 91.8% | 91.6% | — |
| `lgbm-v3-regime` | 48.2% | 78.2% | 79.8% | — |

⚠️ **Do not treat these as the measurement.** Each cell spans 1–5 forecast dates (see §4 of the
companion entry), the `lgbm-v3` rows include a 2025-12-01 backdated batch, and the three cohorts are
different artifacts on different dates — the `global-only` row over-covering while `lgbm-v3`
under-covers is not something the recentring mechanism alone explains. What the table establishes is
that served coverage is nowhere near 80% and is not even consistently on one side of it. The
mechanism above is a hypothesis that predicts under-coverage; it is not confirmed by this table.

**The fix is an ordering fix, not a new model.** Calibrate `q_hat` on residuals of the *recentred*
mid — i.e. apply `_recenter_on_direction` inside the CV records path before
`_calibrate_conformal` — so the calibration and the serving centre are the same object. That
requires the CV to have classifier predictions available, which is exactly what
`CV_DIAGNOSTIC_CLASSIFIER=1` now supplies in `model-diagnostics.yml`. Cost: no extra fit.

**Why this matters beyond coverage.** The band is the only thing that makes a published price or
percentage honest. Until it holds, the `low`/`high` legs should not be presented as an 80% interval,
and `docs/product.md` should not describe them as one.

## 2. The served confidence label and the calibrated confidence thresholds are two different things

`meta.json` stores, per horizon:

| h | `high_range` | `high_change` | `high_accuracy` | q50 `mean_dir_acc` |
|---|---:|---:|---:|---:|
| 3 | 0.4885 | 0.0 | **39.9%** | 39.9% |
| 7 | 0.7311 | 0.0 | **41.8%** | 42.0% |
| 14 | 1.1230 | 0.0 | **42.6%** | 43.1% |
| 30 | 1.7670 | 0.0 | **47.1%** | 48.4% |

**Two separate problems.**

**(a) `range_pct` carries no information about whether the call is right.** `_calibrate_confidence`
targets `target_accuracy = 0.80`. It never gets there at any horizon, so it takes the fallback
branch at `:6761`, which maximises **coverage** rather than accuracy. The result is a threshold near
the wide end of the distribution and a "high confidence" subset whose accuracy is
indistinguishable from the overall mean — at 30d it is 1.3pp *below* it. `high_change = 0.0` at every
horizon is the same failure showing again: the change filter is only accepted if accuracy stays
≥ 80%, and it never does. **A narrow conformal band does not predict a correct direction call**, and
the stored numbers have been saying so.

**(b) None of that is what production serves.** When a classifier is loaded — which is the
production path, `clf_{3,7,14,30}d.txt` all ship — `predict` sets confidence from the classifier's
own max probability instead:

```python
confidence = ("high" if float(dir_conf_arr[i]) >= self.DIRECTION_CONFIDENCE_HIGH
              else "low")          # forecaster.py:6120, DIRECTION_CONFIDENCE_HIGH = 0.5
```

`_compute_confidence` — the only consumer of `confidence_thresholds` — runs solely in the
no-classifier `else` branch. So the served confidence label is a **bare 0.5 cut on a 3-class argmax
probability, never calibrated against outcomes and never measured**, while the calibrated thresholds
that *were* fitted describe a path production does not take. The `conf_gap_pp` values in the
backtest (−5.2 to −38.7pp) are measuring the served label; the `high_accuracy` in `meta.json` is
measuring the fallback. They have been read as the same quantity.

**Minimum fix:** calibrate the served label on the classifier's probability against realised hits,
the way `_calibrate_confidence` already does for `range_pct`, and store it under a distinct key so
the two paths can never be confused again. **Cheaper interim fix:** stop publishing `confidence`
until it is calibrated — an uncalibrated confidence tag is worse than none, because consumers
weight on it.

## Why these were not caught

Both sit in the seam between two components that are individually well tested. `models/conformal.py`
is pure and correct — its own docstring notes it replaced p10/p90 GBMs whose empirical coverage was
39–48%, which is the same failure mode now appearing downstream of it. The recentring is tested for
the property it claims (`_recenter_on_direction` preserves half-widths; the served price agrees with
the served direction) and that property holds. Nothing tested the *composition*: that the band
served is the band calibrated.

That suggests the test worth adding is an end-to-end one — pooled served coverage on a held-out
window against `NOMINAL_COVERAGE` — rather than more unit tests on either side. The audit note in
`2026-08-10-served-classifier-scored.md` §4 already flagged the existing check as unfalsifiable:
`q_hat` measured on the pooled OOF it was fitted on is ≥80% by construction.

## Verification

Code paths read at source and cited by line. `meta.json` values are the shipped artifact
(`trained_at 2026-08-09 21:55:20 UTC`). `IntCov` figures are from run `31409508960`'s own log lines,
quoted with their date counts. **No code changed and no claim here is measured** — the coverage table
is observational and confounded, as stated inline.
