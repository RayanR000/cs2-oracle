# `anomaly_p` was outside the missing-column guard

**Date:** 2026-09-12
**Status:** fixed

## What was wrong

`scripts/forecast_prices.py::_write_forecasts_to_db` probes `item_forecasts` for
the disclosure columns it is about to name, and narrows the SQL payload to the
ones that exist. The probe set was:

```python
("anchor_clean", "anchor_wedge_pct", "exceed_p")
```

`anomaly_p` was in the payload but not in the probe. This repo's prod schema runs
behind its migrations — that is the stated reason the guard exists at all — so on
a database predating migration `0025_add_forecast_anomaly_p` the INSERT names a
column the table lacks, every batch raises, the three retries all fail, and the
daily forecast run goes down. For a disclosure field. That is exactly the failure
mode the `exceed_p` guard was written to prevent, reproduced one column over.

The bug was latent only because the anomaly head had never produced a non-NULL
value in prod: `ANOMALY_GBM=1` has been live in `price-forecast.yml` since
`4949714`, but the last artifact to actually carry anomaly heads is recent, so
the column had not yet been exercised on a prod write.

## Fix

Add `anomaly_p` to the probe set, and correct the warning text — it claimed the
run was proceeding "WITHOUT the clean-anchor disclosure" regardless of which
column was actually absent.

`tests/test_forecast_anomaly_p_persistence.py` (5 tests) covers both legs: the
SQL payload must drop `anomaly_p` when the column is absent and keep it when
present, and the Parquet mirror — which has no schema to violate — keeps it
either way. Verified by mutation: reverting the one-line fix fails
`test_db_payload_drops_anomaly_p_when_the_column_is_absent` and nothing else.

## Serving state, confirmed

Everything else on the path was already built and needed no change:

- `ItemForecaster.ANOMALY_SERVED_HORIZONS = (3, 7, 14)`, enforced in
  `anomaly_probability` — 30d returns None because that head ranks (AUC +0.129)
  without calibrating (log loss ties the featureless null). See
  `2026-09-09-anomaly-head-beats-its-null.md`.
- `api/schemas.py::PredictionOut.anomaly_p` and both read paths in
  `api/routes/items.py` already surface it, with the withholding documented.
- Migration `0025` and `database.py::ItemForecast.anomaly_p` exist.

Checked against the local artifact (`trained_at` 2026-09-12, `anomaly_gbm: true`,
four `anomaly_clf_*.txt` present): heads load at 3/7/14/30, and
`anomaly_probability` returns real spreads at the served horizons — median
0.098 / 0.129 / 0.155 at 3/7/14d over 200 rows — and `None` at 30d.

So shipping is now a retrain away, not a code change away: the next Price
Forecast run on an artifact carrying anomaly heads populates the field.
