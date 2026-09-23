# The served-coverage refit now scores rows on the base band

**Date:** 2026-09-23
**Status:** FIXED in code; needs migration 0028 + `scripts/backfill_band_multiplier.py --apply`
on prod **before the 2026-09-28 `mode=full` retrain**.

## The defect

`served_recalibration.factors_from_panel` fits the factor as P80 of
`r = |actual - mid| / stored half`, and `train()` **assigns** the result as the new
multiplier. That is only right while every stored band was served at 1.0. Once the factor is
live, the stored half is already `m x` the base half, so `r` is inflated by `1/m`, and a
refit pools rows in two units:

- mixed panel (rows at 1.0 and at 0.54) -> the factor lands between them and the band
  **re-widens**;
- panel served entirely at the calibrated factor -> P80(r) ~ 1.0 and the band **snaps back to
  full width**, then narrows again the week after: an oscillation, not a convergence.

Projected on the real h=3 panel (9 resolved dates 09-07..09-15, factor 0.5385), resampling
those base-band scores for the dates served narrowed:

| panel at refit | old code | fixed |
|---|---|---|
| + 4 dates at 0.5385 (the 09-28 retrain) | 0.667 | 0.539 |
| + 9 dates | 0.754 | 0.539 |
| + 20 dates | 0.848 | 0.542 |
| 20 dates, all at 0.5385 | 1.013 | 0.546 |

The multiplier's own activation is a band-geometry change, the same class the three
`*_SERVING_START` floors handle, but a recurring one, so a floor would restart the 8-date clock
every week.

## Also found: it went live at 2026-09-17, not at the 09-21 retrain

`predict()` live-reads the panel when the artifact's factor dict is empty, so the gate crossing
(8 dates) activated it on a **predict-only** run before any retrain baked it in. Reconstructed
from the Price Forecast logs and confirmed on prod against h=7 (factor 1.0) as control:

| h=3 fd | final write | factor | stored multiplier (post-blend) | observed width ratio |
|---|---|---|---|---|
| <=09-16 | - | 1.0 | 1.0 | 1.0 |
| 09-17 | run 35306325544 (manual, live read) | 0.5251 | 0.5963 | 0.594 |
| 09-18 | run 35480138859 (09-20, re-served) | 0.5385 | 0.5381 | 0.535 |
| 09-20.. | run 35549467741 (full) + predict-only | 0.5385 | ~0.5385 | ~0.54 |

The stored multiplier is post-blend: `predict` mixes 15% of the prior row's band in, and the
prior is the latest row dated before the run's **wall-clock** day, so a same-day redispatch
blends with that date's own earlier write.

## The fix

- `item_forecasts.band_multiplier` (migration 0028, nullable): `predict` records the
  post-blend multiplier per row (`ItemForecaster._blended_band_multiplier`, reading the prior
  row's value through `_fetch_prior_forecasts`), written behind the existing missing-column
  guard.
- `_load_panel` LEFT JOINs it via `forecast_id`; `factors_from_panel` multiplies `r` by it, so
  the refit is a fixed point.
- NULL resolution (`resolve_band_multiplier`): before `FEEDBACK_FIRST_SERVED_DATE`
  (2026-09-17) NULL is provably 1.0; on/after it the row is dropped. A missing column
  (migration not applied) takes the same path, so it degrades to "fit on pre-09-17 rows only"
  rather than to a failed read (which would return `{}` -> 1.0 -> full width).
- `scripts/backfill_band_multiplier.py` fills 09-17..09-27 from the table above (dry-run by
  default; NULL rows only; refuses dates past 09-27, the last the 09-21 artifact can serve).

## Not fixed here

- **Archive day 2026-09-19 is missing** from `prices-2026-09.parquet` (the 09-20 Aggregator run
  was green but wrote no day). h=3 forecasts 09-16/17/18 are unscoreable until it is backfilled.
- `_sanitize_forecasts` can clip a band edge after the multiplier is applied; those rows' `r`
  is then slightly off the base band. Rare at >= $1.
- Same-day redispatches blend a forecast with its own earlier write (the `_now()` prior
  lookup is wall-clock). Recorded correctly by the multiplier; the blend itself is unchanged.
