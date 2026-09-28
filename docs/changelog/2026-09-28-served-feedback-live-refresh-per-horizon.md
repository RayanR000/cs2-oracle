# Served-coverage feedback: live refresh is per-horizon

**Date:** 2026-09-28

## The defect

`predict` live-reads the served-coverage factor from the panel so a horizon that crosses
`MIN_FEEDBACK_DATES` can narrow on a predict-only run. The read only fired when
`served_coverage_factor` was **empty**. Once a retrain baked any horizon, every other horizon
stayed at 1.0 until the next retrain, however many resolved dates it had.

The 2026-09-28 full retrain (run 36473131841) baked `{3: 0.5385, 7: 0.6111}`. Under the old check,
14d (7 of 8 dates today) and 30d (first resolution 10-07) could not activate on predict-only runs.

## The fix

`ItemForecaster._refresh_served_coverage_factor` requests only the horizons missing from the
dict and merges the result. A baked factor is never overridden, and when every horizon is baked
the panel is not read. The estimator, gate, geometry floor and 0.5–2.0 clamp are unchanged.
Only `train()` writes `meta.json`, so a live factor is recomputed on each predict-only run and is
baked only at the next retrain. The `band_multiplier` fixed-point refit
(`2026-09-23-feedback-factor-refit-composes.md`) is what makes that daily re-read safe.

## Evidence the factor works where it is live

On the durable ops mirror, ≥$1 items, h=3:

| forecast dates | multiplier | coverage | median width |
|---|---|---|---|
| 09-07 … 09-15 | 1.0 | 91.8–96.4% | ~14% |
| 09-20 … 09-24 (5 dates) | 0.54 | 75.8 / 79.1 / 82.7 / 84.9 / 86.0% | 9.3–9.6% |

It is a level correction. Mean coverage lands near 80%, but per-date coverage still swings, and
the 2026-09-22 ACI read found no date-level information to condition that swing on.

Tests: `tests/test_served_recalibration_wiring.py` (6 new).
