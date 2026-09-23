# Interval score in the backtest; MAPIE oracle for `calibrate`

**Date:** 2026-09-23

## Interval (Winkler) score

`backtest/scoring.py::interval_score` returns the band width plus `2/α` times the
distance by which the outcome fell outside the band (Gneiting & Raftery 2007). It is a
proper score, so a band cannot improve it by being uniformly narrower or wider than the
true 10/90 quantiles. Coverage and width reported separately do allow that trade.

- `_derive_verdict` adds `interval_score_pct`, a reporting-only key. It is computed on the
  **same calibrated basis as `in_interval`**: the band is rebased by `base / quote`, and
  band and outcome are expressed as returns off `base`, in pp. `_verdict_for_storage`
  drops the key, so the `forecast_outcomes` schema is unchanged.
- `score_cohort` reports `interval_score_pct` as the mean over rows that carry one, and
  `None` when no row does (`walkforward_records`, and stored rows read back). It lands in
  the `prediction_accuracy.metrics` JSON beside `interval_coverage`.
- Only compare it within a band-geometry era (`served-panel-and-geometry-eras`). Like
  coverage, it is a property of the band shape that was served.

## MAPIE oracle for `conformal.calibrate`

`tests/test_conformal_mapie_oracle.py` checks `calibrate` against MAPIE 1.x
`SplitConformalRegressor`, using a prefit constant-zero estimator. MAPIE returns exactly
the ⌈(n+1)(1−α)⌉-th smallest score. Ours reads that same level through `np.quantile`'s
linear interpolation, so it lands **between that order statistic and the next one up**.
That is the conservative side, and it is negligible at production n. The test pins that
relationship. A mutation check that dropped the finite-sample correction failed all 5
cases.

## CI caveat

`mapie` and `scoringrules` are not in the `dev` extra, so both cross-checks
`importorskip` in CI. They run locally, where both are installed in `backend/venv`. They
could not be added to `dev`, because `uv lock` already fails on main: `requests==2.31.0`
conflicts with `evidently>=0.5` in the `mlops` extra. Fix that pin first, then add
`mapie>=1.0` and `scoringrules>=0.11` to `dev`.
