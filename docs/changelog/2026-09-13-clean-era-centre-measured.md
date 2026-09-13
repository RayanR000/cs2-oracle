# Clean-era centre measured: CONVERGED stands, with one bounded surprise (2026-09-13)

Scores the preregistration `docs/research/2026-09-13-clean-era-centre-preregistration.md`:
walk-forward q50 GBM centre vs last-price (level) and vs `−return_1d` (ranking) on the
2025 voted composite, H+13 embargo, date-block bootstrap. Instrument:
`backend/scripts/clean_era_centre_ab.py` (+ `tests/test_clean_era_centre_ab.py`, 22 tests).

## What ran (as pre-registered, no deviations)

- Frame: 2025 NULL + buff_iflow rows through `prices_relation` + universe filter, voted
  via production's `_apply_multi_source_voting` (711,110 rows, 2,524 items ≥$1/≥180d),
  `engineer_features` + cross-sectional, **price_technicals allowlist only (28 cols, the
  served set; no corr prune, no metadata)**. No production code touched.
- Folds: 21-day val windows every 21 days from 2025-04-15; train purged by production's
  `_purge_overlapping_train_rows`; trainer is production's `_train_ensemble_member` at
  `_boost_rounds(h, cv=True)`. Arms differ by features alone (placebo permutes all 28).
- Primary read: tied cohort (`p == S`, wedge-free). h=14: 10 folds / 177 dates; h=30:
  8 folds / 133 dates.

## Result

| h | cohort | model IC | naive IC | paired | placebo | MAE skill |
|---|---|---|---|---|---|---|
| 14 | tied | **+0.094 [+0.067, +0.123]** | −0.000 [−0.023, +0.023] | **+0.094 [+0.055, +0.133]** | −0.002, null | −0.015 (6/10 folds neg) |
| 14 | all | +0.127 | +0.134 | −0.006, null | null | — |
| 30 | tied | +0.030 [+0.001, +0.059] | +0.024 [−0.002, +0.050] | +0.006, null | null | **−0.162 (7/8 folds neg)** |
| 30 | all | +0.060 | +0.091 | **−0.031 [−0.055, −0.006]** | null | — |

## Bars

- **PASS fails at both horizons**: MAE skill vs last-price is negative (−0.015 / −0.162).
  The level half of CONVERGED (`lambda* = 0`) survives on the clean panel too.
- **KILL fires at h=30**: paired model−naive is null on tied and significantly negative
  on the full cohort — the centre adds nothing over `−return_1d` there.
- h=14 reads UNDERPOWERED by the letter (naive null on tied) while model and paired CIs
  sit entirely positive. The bars did not foresee this cell; it is reported, not promoted
  (below). **No serving change, no re-litigation. CONVERGED stands.**

## Integrity notes

- Placebo null at every cell (4 horizons-cohorts × ranking): the harness is sound; the
  h=14 ranking is in feature values, not capacity.
- The 08-11 wedge reproduces exactly: naive's entire signal is the wedge (h=14 all
  +0.134 → tied −0.000; h=30 all +0.091 → tied +0.024). The model keeps ~3/4 of its
  signal on tied — it reads structure plus some wedge, naive reads only wedge.
- Gain leaders are multi-week reversal features, stable across folds: h=14 `return_60d`
  10/10, `return_3d` 10/10, `trend_divergence_30_60` 8/10; h=30 `price_cv_60d` 8/8,
  `return_60d` 7/8, `return_90d` 7/8. Not a single-fold fluke (10 folds, Apr–Dec).

## The one observation that narrows "empty" (not a result)

On tied rows at h=14 the centre ranks the composite at +0.094 over 177 dates with a
null placebo — ~3× its 2026 self, via features naive cannot see. So "the centre is
information-free" over-generalises: on clean labels it **ranks without level accuracy**
at 14d. Caveats that keep this an observation: one contiguous regime year (date
bootstrap overstates independence — the 14–70-episodes binding constraint applies to
2025 too), and h=30 shows no such edge. Any follow-up (does h=14 tied ranking survive
on served 2026 dates?) needs a **new** preregistration — this one's PASS failed, and
failed bars do not license their own revision.

## Reproduce

```
venv/bin/python -m scripts.clean_era_centre_ab --frame-cache /tmp/cec_frame.parquet --build-cache-only
venv/bin/python -m scripts.clean_era_centre_ab --frame-cache /tmp/cec_frame.parquet --horizon 14 --out /tmp/cec_h14.json
venv/bin/python -m scripts.clean_era_centre_ab --frame-cache /tmp/cec_frame.parquet --horizon 30 --out /tmp/cec_h30.json
venv/bin/python -m pytest tests/test_clean_era_centre_ab.py -q
```
