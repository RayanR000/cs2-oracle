# Vol-scaled directional-classifier labels — A/B sweep results (2026-07-27)

## Background

The directional classifier (`_fit_direction_classifier` in
`backend/models/forecaster.py`) labels up/flat/down using a flat
±`DIRECTION_FLAT_TOLERANCE_PCT` (0.5%) band around zero return, with a
mover-weight applied to up/down samples during training.

The hypothesis under test: replacing the fixed ±0.5% band with a
per-row, volatility-scaled band (`k * sigma_30d * sqrt(h)`, clamped to
`[DIRECTION_THRESHOLD_FLOOR_PCT, DIRECTION_THRESHOLD_CAP_PCT]`) would
better separate genuine moves from noise on low/high-vol items and
improve directional accuracy, especially on "movers" (non-flat actuals).

An earlier, uncontrolled run of this idea showed ~69% overall directional
accuracy and looked promising. That number turned out to be a harness
artifact (see below), which is why a proper control cell was added
before drawing any conclusion.

## Method

`backend/scripts/ab_test_direction_labels.py` runs an 8-fold walk-forward
sweep, always scored against the same **fixed ±0.5% yardstick**
(`_direction_classes` with the legacy constant), regardless of what band
was used to *train* the classifier being scored. This isolates "does
vol-scaled training produce a classifier that is more accurate under the
band we actually serve with" from confounds in fold selection.

Cells swept:
- **Control**: fixed ±0.5% band for both training and scoring, mover_weight
  (`mw`) = 3.0 (i.e. the pre-branch, currently-shipped configuration).
- **Vol-scaled**: `K_GRID = [0.25, 0.5, 1.0]` × `MOVER_WEIGHT_GRID = [3.0,
  5.0, 8.0]`, trained with the vol-scaled band, scored against the same
  fixed ±0.5% yardstick as the control.

Adoption rule (pre-registered): a vol-scaled cell must beat the control
by **>= 2 percentage points** (on overall or movers-only accuracy) at a
given horizon to be adopted for that horizon.

## Results (8 folds, fixed ±0.5% yardstick, overall% / movers%)

| Horizon | Control (fixed band, mw=3.0) | Best vol-scaled cell | Delta (control - best) |
|---|---|---|---|
| 3d | 67.9 / 56.0 | 67.5 / 55.4 (k=0.25, mw=8.0) | +0.5 / +0.6 |
| 7d | 68.6 / 54.8 | 67.7 / 53.5 (k=0.5, mw=8.0) | +0.9 / +1.3 |
| 14d | 67.5 / 52.0 | 67.1 / 51.5 (k=0.5, mw=8.0) | +0.4 / +0.5 |
| 30d | not measured (DART; run too slow, terminated) | — | — |

Raw sweep output: `docs/research/2026-07-27-direction-label-sweep-raw.txt`.

## Conclusion: NOT adopted

The fixed-band control **beats every vol-scaled cell** on 3d/7d/14d,
on both overall and movers-only accuracy. The adoption rule (>=2pp
improvement over control) **failed at every horizon** — the best
vol-scaled cells were 0.4–1.3pp *worse* than control, not better.

**Vol-scaling is not adopted. Production remains fixed-band** (±0.5%,
mover_weight=3.0), i.e. identical to pre-branch behavior. See Task 1 in
this branch: both call sites of `_fit_direction_classifier` in
`backend/models/forecaster.py` now pass `sigma_train=None,
sigma_val=None` explicitly.

## Why the earlier ~69% figure was misleading

The original, uncontrolled measurement of vol-scaled labels showed ~69%
directional accuracy and no comparison point. Once a proper control was
run in the *same* harness (200 rich items, cross-validation, recent
folds), the control also scored ~68% — i.e. the ~69% was largely a
property of the harness (item selection, fold recency, CV setup), not of
vol-scaling. Without the control cell, this would have been
misattributed to the labeling change.

## Other observations

- `k=1.0` was catastrophic at 3d (~47% accuracy) — an overly wide
  vol-scaled band swallows real moves into the "flat" class and destroys
  the classifier's discriminative signal at short horizons. This is a
  useful negative data point for anyone revisiting this idea.
- The sweep is slow: 14d and 30d horizons train with DART boosting at
  500 rounds, which multiplies wall-clock time across 8 folds × 9 grid
  cells. The 30d cell was terminated before completion for this reason
  and is not reported above. A future rerun of this tooling should cap
  the training lookback window (or reduce DART rounds / fold count for
  exploratory sweeps) before attempting 30d.

## Disposition

- **Tooling retained**: `backend/scripts/ab_test_direction_labels.py`
  (control cell + 8-fold sweep + configurable grids) is kept as reusable
  infrastructure for future direction-label experiments.
- **Production unchanged**: vol-scaling is wired but disabled (both call
  sites pass `sigma_train=None, sigma_val=None`), so live training
  behavior is identical to before this branch.
- **Supporting code kept**: `_direction_threshold`, threshold-aware
  `_direction_classes`/`_direction_sample_weights`, the `label_vol_30d`
  column, and the `DIRECTION_*_MAP`/threshold constants are all retained
  since the sweep tooling depends on them and they are no-ops in
  production when `sigma=None`.
