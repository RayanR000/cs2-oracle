# The anomaly head is the first GBM here to beat a featureless null

**Date:** 2026-09-09
**Change:** New `scripts/anomaly_gbm_ab.py` and `tests/test_anomaly_gbm_ab.py`
(12 tests). New `tests/test_anomaly_label.py` (7 tests). Direction-aware
`fold_tally` added to `scripts/exceedance_meta_ab.py` (+2 tests). **Production
change: the anomaly label's threshold in `forecaster.py:5369` is rebuilt from the
strictly-prior `return_{h}d`.** `ANOMALY_GBM: "1"` was already live in
`price-forecast.yml` and stays.
**Context:** `ANOMALY_GBM=1` was already built end to end — head at
`forecaster.py:7711`, trained at `:6487`, persisted `:11278`, reloaded `:11843`,
label `target_anomaly_{h}d` at `:5375`, and already in the public API schema as
`anomaly_p` (`api/schemas.py:113`, `api/routes/items.py:696,751`). It had never
been measured and had no tests, so the field ships today as a permanent null.

## The verdict

**It works.** Three arms — the production head on the 28 allowlisted
`price_technicals` columns, a per-item anomaly rate learned on the train fold,
and a single pooled constant — paired per fold across 25-26 walk-forward folds,
served cohort (>=$1). Deltas are GBM minus null.

| h | held-out AUC vs null | held-out log loss vs null | GBM AUC (abs) |
|---|---|---|---|
| 3d  | **+0.134\*** (better 25/26) | **−0.0118\*** (better 22/26) | 0.634 |
| 7d  | **+0.140\*** (better 26/26) | **−0.0117\*** (better 18/26) | 0.640 |
| 14d | **+0.129\*** (better 23/25) | **−0.0104\*** (better 21/25) | 0.629 |
| 30d | **+0.129\*** (better 22/25) | +0.0000 (tie) | 0.629 |

Ranking improves at every horizon with the interval clear of zero, and
calibration improves at three of four (h=30 is a dead tie, not a loss). Base
rates are 6.4% / 7.7% / 9.5% / 14.9%.

This is worth stating plainly because it is the exception on this panel. The
band lost to climatology (`2026-08-20`), the centre lost to last-price
(`centre-shrinkage-lambda-is-zero`), the magnitude booster lost to a pooled
constant (`magnitude-target-beats-climatology`), and the composite centre never
beat last-price. **A GBM here has been decorative every previous time it was
measured against a featureless rule.** This one is not.

## Why the two nulls collapse on held-out items

`item_rate` and `global_rate` produce identical held-out numbers at every
horizon, and that is correct rather than a bug: held-out items are by
construction absent from the training fold, so there is no per-item rate to
apply and every row falls back to the pooled constant. The per-item null is
only meaningful on the **trained** cohort, and there the GBM also wins —
+0.130/+0.158/+0.131/+0.110 AUC at 3/7/14/30d, all intervals clear of zero.

Worth noting for the wider pattern: on the trained cohort `item_rate` is
*worse* than `global_rate` on log loss at every horizon (e.g. 0.2717 vs 0.2514
at h=3). That is the third independent reproduction of
`magnitude-target-beats-climatology`'s finding that a pooled constant beats
per-item climatology — per-item rates overfit item noise.

## The label was leaky, it is fixed, and the result survives the fix

`prepare_targets` built the threshold as `2 * rolling(60).std()` of
`target_return_{h}d.shift(1)`. Every trailing target in that window resolves h
days after its own row, so at h>1 the threshold was normalised by returns
overlapping the prediction window and neighbouring rows shared it. That could
not explain the paired deltas — all three arms see the same label — but it did
mean the label was not knowable at serve time.

`forecaster.py:5369` now builds it from `return_{h}d`, the BACKWARD h-day return
already observed at the row's own date, shifted one row and rolled identically.
Same units (both percent, winsorized at ±500%), same k, same window. Re-running
every horizon on the rebuilt label:

| h | held-out AUC vs null | held-out log loss | base rate |
|---|---|---|---|
| 3d  | **+0.139\*** (better 26/26) | **−0.0154\*** (better 24/26) | 6.8% |
| 7d  | **+0.153\*** (better 26/26) | **−0.0187\*** (better 20/26) | 8.7% |
| 14d | **+0.128\*** (better 25/25) | **−0.0135\*** (better 20/25) | 11.4% |
| 30d | **+0.129\*** (better 23/25) | +0.0053 (null) | 17.6% |

Unchanged to slightly stronger at 3d and 7d. **So the overlap was never what the
head was reading**, and the label is now honest as well. `ANOMALY_GBM: "1"` was
already live in `price-forecast.yml`, so this fix lands before the head's first
real build rather than after it.

New `tests/test_anomaly_label.py` (7 tests) pins the property directly: tampering
with a later row's `return_{h}d` must not move an earlier label, and a frame
without the backward return emits no label rather than falling back to the
leaky definition.

## What is still open at h=30

Log loss at 30d is null on both labels — the head ranks (AUC +0.129) but does not
calibrate better than a pooled constant there. `anomaly_p` is served as a
probability, so a consumer treating the 30d value as calibrated is reading more
than the measurement supports. Either restrict the served field to 3/7/14d or
calibrate the 30d head against the pooled rate before quoting it.

## Reproduce

```
python -m scripts.anomaly_gbm_ab --horizon 7 \
  --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \
  --frame-cache /tmp/exc_meta_frame.parquet --out /tmp/anom_h7.json
```

The harness sets `ANOMALY_GBM=1` itself before any `prepare_targets` call —
without the gate the label is never computed and every horizon skips silently.

## A reporting fix that came out of this

`paired_fold_deltas` returns `wins` = folds where the delta is positive, which
is the good direction for AUC and the bad one for log loss. The h=3 calibration
result above printed as "wins 4/26" when it was in fact better in 22 of 26
folds. `fold_tally` now counts in each metric's own direction. The earlier
`2026-09-09-exceedance-meta-refuted.md` entry is unaffected — its log-loss
claims were read from the signed means and intervals, not the fold counts.
