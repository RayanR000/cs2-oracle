# Post-CSMarketAPI served performance scoring (2026-09-22)

Follow-up to [2026-09-20-csmarketapi-backfill-and-retrain](2026-09-20-csmarketapi-backfill-and-retrain.md).
The 09-20 changelog reported training-time metrics (rank IC on held-out folds).
This entry measures **served** performance on resolved production outcomes.

## Comparison: pre-CSMarketAPI era vs post-CSMarketAPI era

Pre-CSM: forecasts 08-11 to 08-27 (single-source training era).
Post-CSM: forecasts 09-07 to 09-18 (multi-source era, after 09-06 retrain).

### Return-basis rank IC (Spearman of predicted vs actual return)

| Horizon | Pre-CSM (n dates) | Post-CSM (n dates) | Delta |
|---------|-------------------|---------------------|-------|
| h=3  | −0.033 ± 0.055 (10) | +0.022 ± 0.017 (9) | +0.055 |
| h=7  | +0.023 ± 0.026 (12) | +0.138 ± 0.031 (8) | +0.115 |
| h=14 | −0.007 ± 0.020 (13) | +0.142 (1)          | +0.149 |
| h=30 | +0.032 ± 0.040 (9)  | (unresolved)        | —     |

### Band width (relative to midpoint) and coverage

| Horizon | Pre-CSM width | Post-CSM width | Narrowing | Pre cov | Post cov |
|---------|--------------|----------------|-----------|---------|----------|
| h=3  | 31.9% | 18.0% | −44% | 90.7% | 92.1% |
| h=7  | 47.8% | 27.4% | −43% | 92.9% | 93.5% |
| h=14 | 63.2% | 42.7% | −32% | 93.0% | 94.4% |

~40% narrower bands at equal or better coverage across all resolved horizons.

## Caveats

1. **Combined effect.** The 09-06 retrain shipped CSMarketAPI data alongside iflow
   promotion, source weights, and adaptive conformal. The improvement is the combined
   effect of the optimization PR (ad2d43e), not CSMarketAPI alone.

2. **Small post-CSM sample.** h=14 has 1 resolved date; h=30 has zero. The h=3 and
   h=7 numbers (8-9 dates) are more trustworthy but still narrow.

3. **09-21 retrain unresolved.** The Sunday full retrain on the expanded archive has
   zero resolved outcomes yet. Earliest: h=3 resolves 09-24. The numbers above
   measure the post-CSM *era* (09-06 retrain), not the 09-21 retrain specifically.

4. **Over-coverage.** All horizons cover 90-94% against a target of 80%. The bands
   could narrow further without breaching the calibration target. This is the
   conformal recalibration headroom noted in the q_hat dispersion study (09-15).

## Verdict

The multi-source era is a material improvement on both axes: rank IC went from
near-zero to positive, and bands narrowed ~40% at matched coverage. The 09-20
training-time gains (especially the 30d rank IC jump from 0.03 to 0.10) are
consistent with the served-panel read at shorter horizons. Full confirmation at
h=14 and h=30 requires waiting for resolution (early to mid October).

## Correction (2026-09-30)

Caveat 1 above says the 09-06 retrain "shipped CSMarketAPI data alongside iflow promotion,
source weights, and adaptive conformal". **Source weights and adaptive conformal did not ship.**
`models/source_weight_model.py` never had an importer; `conformal.calibrate_adaptive` /
`update_adaptive` and `served_recalibration.stratum_factors_from_panel` had no callers in the
trainer, the predict path, any workflow or any script; `models/data_quality.py` was imported
only by its own test (spec items 12-14 of `specs/2026-09-19-optimization-ml-promotion-design.md`
were written but never wired). The combined-effect caveat therefore covers CSMarketAPI data plus
iflow promotion only. The dead code was deleted 2026-09-30 (see
`2026-09-30-delete-uncalled-code.md`). The paragraph above is left as written, per the
changelog's append-only rule.
