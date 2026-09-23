# `forecast_date` is the day the band was anchored on, not the wall clock

**Date:** 2026-08-15
**Scope:** the serving-write label, and the one API consumer that pinned to it. A correctness
fix for decoupling **(A)** of the serve-vs-score freshness wedge — **not** a coverage fix.
**Related:** `docs/changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md` (the ops/data-
freshness thread this opens from), `docs/plans/2026-08-11-serving-anchor-freshness.md`
(a *different*, closed wedge — do not conflate), `.claude/rules/backtest-scoring.md`.

## What changed

`predict()` anchors the band on the newest day in the loaded frame — `df["date"].max()`
(`forecaster.py`) — but the row was stamped `forecast_date = date.today()`
(`forecast_prices.py`). The dump lands ~22:00 UTC, so the newest archived day is usually
`today-1`; the label was therefore a frame ahead of the price the band was quoted against. The
scorer resolves **both** `base_price` (`backtest_accuracy.py:1259`) **and**
`target_date = forecast_date + horizon` (`:1237`) at the stored `forecast_date`, so the mislabel
shifted both outcome legs and left a latent horizon skew (the model predicts `anchor + h`, the
scorer resolved `today + h`).

- `predict()` now carries `anchor_date = df["date"].max().date()` on every result row.
- `_write_forecasts_to_db` stamps `forecast_date` from that anchor day. `FORECAST_DATE_OVERRIDE`
  still wins (a replay stamps a chosen date deliberately); a legacy frame without the column
  falls back to `today` rather than a NULL label.
- `api/routes/items.py::_build_trending` filtered `forecast_date == today`, which now goes empty
  after every normal run. Replaced with a bounded freshness window
  `forecast_date >= today - MAX_ARCHIVE_LAG_DAYS` (7); the existing distinct-on still selects each
  item's newest forecast within it. The other forecast reads already order by
  `desc(forecast_date)` and were unaffected.

The `item_forecasts` PK is `(item_id, forecast_date, horizon_days)` and the write is
`ON CONFLICT DO UPDATE`, so a missed-dump day that repeats an anchor day overwrites the prior
row — the same same-day semantics config suffixes already had, and the correct behaviour (it is
a re-forecast of the same anchor).

## This is a correctness fix, not a coverage fix

Measured 2026-08-15, archive-only (independent of stored prices): a pure one-day anchor shift
moves the resolved base by **median 0.05%, p90 4.71%** (~23% of item-days shift ≥2% on a single
day). That cannot produce the prod dollar-basis wedge of **median 5.7% / p90 38%** — decoupling
**(B)**, the serving frame lagging *multiple* days and/or source/backfill divergence, dominates
that, and is untouched here. (B) is an ops/data-freshness change and needs the prod DB +
`ingested_at` snapshot history to size; it could not be reproduced in the local checkout, whose
`item_forecasts` is a fixture incoherent with the archive.

**Do not** read this as, or replace it with, `SERVE_OUTLIER_GATED_ANCHOR` — that attacks the
raw-vs-smoothed wedge *within one frame* and is closed.

## Tests

- `tests/test_forecast_date_is_anchor_day.py` — predict exposes the anchor day; the writer stamps
  it; the override still wins; the legacy-frame fallback is `today` (4 tests).
- `tests/test_trending_ranking.py` — the freshness guard is a window, not an exact-today pin.
- Regression: `test_forecaster`, `test_clean_anchor_gate` (incl. the write path),
  `test_minimal_model_shape`, `test_serving_anchor_span`, `test_opportunity_selection` and
  neighbours pass. The full suite was not run end-to-end (it trains models).
