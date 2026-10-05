# Break-aware lookbacks: the Part B gate reads null at every horizon

**Date:** 2026-10-05
**Script:** `backend/scripts/archive/ab_test_break_aware_lookbacks.py` (new)
**Status:** measured, **inconclusive**. `BREAK_AWARE_LOOKBACKS` stays off. No modelling or serving
change.

`specs/2026-09-30-composition-break-calendar-design.md` shipped the flag built but off and set
a gate: paired arms through the production trainer, judged on band width at matched coverage
and interval score, flipped only on a SUPPORTED primary at h=14 **and** h=30. This is that gate.

## Design

- **Frame:** production's `build_training_data(days_back=1460, backfilled_only=True,
  max_feature_rows=1_200_000, min_median_price=1.0)`, built twice, once with the flag off and
  once on. The mask runs before the cross-sectional rank transform, so the on-arm cannot be
  derived from the off-frame. Both frames carry the same 639,664 `(item_id, date)` rows, the
  same 28 features and identical labels (asserted).
- **Calendar:** `2026-01-01, 03-22, 04-16, 07-09..07-12`, the same seven days today's CI
  retrain logged. With the flag on, 2,237,930 cells are NaN. After 03-22, `return_90d/120d/180d`,
  `price_dist_ma100/200` and `macd_histogram_rel` are 100% NaN and `return_60d` is 85% NaN, as
  the spec's "Known cost" predicted.
- **Folds:** 21-day validation windows every 21 days from 2026-02-01, each trained on every
  earlier row purged by `_purge_overlapping_train_rows` (horizon + 13). Fit rows are sampled to
  400k with a per-fold seed both arms share. Eval rows are those priced at or above
  `MIN_SERVED_PRICE_USD`.
- **Trainer:** `_train_ensemble_member` at `_boost_rounds(h, cv=True)`, the horizon's production
  objective (h=30 is L2 regression), the ab_test_* family's tree params, and NaN passed through
  as under `FEATURE_NATIVE_NAN=1`.
- **Band:** centre ± q × climatology scale. The scale is built from the fold's fit labels, so
  it is identical for both arms. Each arm is scored at its own q80 (matched coverage).
- **Statistics:** fold-paired log matched width (`paired_fold_deltas`), and row-paired interval
  score at α = 0.2 resampled by fold (`paired_metric_difference`).

## MDE first

Seed 42 against seed 7 on the control arm. The spec's stop line is ~2%.

| h | folds | width MDE | interval-score MDE |
|---|---|---|---|
| 3 | 10 | 0.14% | 0.05% |
| 7 | 10 | 0.26% | 0.09% |
| 14 | 10 | 0.17% | 0.05% |
| 30 | 8 | 0.84% | 0.67% |

It passes, but it is a floor, not the design's power. Two seeds on identical rows hardly move.
At h=7 even the seed-only interval excludes zero (+0.03 to +0.54%), so the 10-fold normal
interval is optimistic. The A/B's own half-widths below are the honest MDE.

## Result

Flag on minus flag off. Negative width is narrower (better); negative interval score is better.

| h | width Δ | 95% CI | half-width | interval score Δ | CI |
|---|---|---|---|---|---|
| 3 | −0.28% | [−2.42, +1.86] | 2.14% | +0.33% | [−0.70, +1.25] |
| 7 | −0.70% | [−4.21, +2.80] | 3.51% | +0.61% | [−1.29, +2.18] |
| 14 | −0.88% | [−5.21, +3.44] | 4.32% | +0.18% | [−1.76, +1.49] |
| 30 | −4.43% | [−15.39, +6.54] | 10.97% | −0.98% | [−6.53, +2.58] |

Every interval spans zero, on both metrics, at every horizon. The two metrics also disagree in
sign at h=3, 7 and 14.

Per-fold width Δ at the two gating horizons, by validation start:

| fold | 02-01 | 02-22 | 03-15 | 04-05 | 04-26 | 05-17 | 06-07 | 06-28 | 07-19 | 08-09 |
|---|---|---|---|---|---|---|---|---|---|---|
| h=14 | +2.6 | +0.1 | +5.1 | +6.2 | +1.8 | −5.2 | −0.3 | −13.3 | −11.6 | +5.8 |
| h=30 | +4.1 | — | −2.0 | +11.3 | +0.7 | −3.0 | +10.7 | −30.3 | −26.8 | — |

The mean is carried by the two folds whose windows straddle the July breaks (06-28, 07-19).
The folds straddling 03-22 and 04-16 (03-15, 04-05) go the other way. There is no consistent
"masking helps across a break" pattern to build on, and two folds are not a mechanism.

## Verdict

**Inconclusive. Do not flip the default.** The gate needs SUPPORTED at h=14 and h=30; both are
null. Because the achieved half-widths (4.3% at h=14, 11% at h=30) sit above the spec's ~2% bar,
this is not a refutation either: the design cannot rule out a few-percent gain. The secondary
served replay (`replay_serving.py`) was not run, since it only confirms a supported primary.

## Caveats

- The local archive stops at 2026-09-08 (August has 26 of 31 days), so the frame lags the data
  repo by four weeks. That does not bias a paired read, but it does cap the post-July folds.
- Tree params are the ab_test_* family's fixed set, not production's cached Optuna params.
  This is the family convention, and it is the same for both arms.
- The climatology scale is static: no `CLIMATOLOGY_REACTIVE` multiplier and no feedback factor.
  Both scale every row of both arms alike, so they cancel at matched coverage.

## Re-read

Only worth doing with more folds past 2026-07-12, from a fresh data-repo archive, and only if
the h=30 point estimate holds. The masks on the 90d+ and 100/200d features lift at about
2027-01, after which the flag is near-identical to off anyway. It goes onto the next-steps list
as an open idea, not a scheduled read.

## Reproduce

From `backend/` (reads prod Postgres for events, read-only):

```
venv/bin/python -m scripts.archive.ab_test_break_aware_lookbacks --cache-dir <dir> --build-cache-only
venv/bin/python -m scripts.archive.ab_test_break_aware_lookbacks --cache-dir <dir> --mde --horizon 14 --out <dir>/mde_h14.json
venv/bin/python -m scripts.archive.ab_test_break_aware_lookbacks --cache-dir <dir> --out <dir>/ab.json
```

The frame build takes about 40s. Each horizon's A/B takes 20–60s.
