# Volume features shelved — the archive's volume column is identically 0

**Date:** 2026-08-06
**Change:** all **thirteen** volume-derived columns are shelved from
`feature_cols`. `MODEL_ARTIFACT_VERSION` 2 → 3.

Eleven of them were in the 2026-08-05 artifact's 47-column `feature_cols`. The
other two — `volume_mean_7d` and `volume_std_60d` — were not, because the >0.95
correlation prune dropped them in favour of their 30d partners. **Shelving those
partners removes what they correlated against, so they survive the prune and
re-enter production.** Shelving only the eleven left exactly those two behind,
and they are among the most misleading of the set: raw levels, served as `0`
against a real training level. The final count is prune-dependent, so it is
data-dependent too; on the test mock it goes 37 → 35.

## What was found

Volume in `price-archive/prices-*.parquet` has been **identically 0 since
2026-05**, and it is stored as `0`, never `NULL`:

| month | rows | non-zero volume | NULLs |
|---|---|---|---|
| 2026-08 | 725,056 | **0** | 0 |
| 2026-07 | 6,172,930 | **0** | 0 |
| 2026-06 | 779,351 | **0** | — |
| 2026-05 | 778,383 | **0** | — |
| 2026-04 | 1,631,398 | 1,181,424 | — |
| 2026-03 | 934,391 | 885,635 | 48,756 zeros |
| 2026-01/02 | 305,123 | 305,123 | — |

Zero-not-NULL defeats every guard in the feature code:

* `has_volume` tests `df["volume"].notna().any()`, and `0` is not NaN, so it
  stays **True** and the "no volume" branch never runs.
* `volume_missing` is therefore **0** — the flag built to tell the model "volume
  is absent" reports that it is present.
* The raw level features are handed a real-looking `0`, **not** median-filled,
  against training medians of `volume_lag_1d` **98.0**, `volume_lag_7d` 99.0,
  `volume_mean_30d` 115.7, `volume_std_30d` 43.9, `volume_mean_60d` 124.2.
* Only `volume_log_change_1d/_7d` and `volume_zscore_30d` go NaN → median-filled
  (0.0, −0.0048, −0.246).
* `volume_price_conf_1d/_7d` collapse to `0.0`, which equals their training
  median — the only two that were harmless.

So these features carry real signal on training rows predating 2026-05 and are
dead-or-misleading on **every** served row. That is a train/serve gap on 100% of
items, every day, and it never appeared as a per-item outlier precisely because
it hits all items at once.

## Why this and not an admission rule

This came out of re-measuring the serving down-bias after the `is_backfilled`
fix (`065613b`) cut the served universe 8,691 → 5,542. The two fixes previously
on the table were both refuted:

* A **minimum-history admission rule** is now a no-op: the retained 5,542 items
  have min 119 / median 714 distinct days of history, and **0** fall below a
  90-day bar. The corrected `is_backfilled` gate already is a stricter version
  of that rule.
* **Withholding up/down on missing lookback**, keyed on the original 14-feature
  block, fires on **0 of 5,542** items.

And excluding the 3,149 low-history items did **not** fix the skew: 7d serving
went 52.8% → 48.2% "down" against 34.8% on interior rows — a **+13.4pp**
residual on the cohort actually served.

## What is NOT claimed

The accuracy effect is **unverified**. No offline rig can certify it: every ≥$1
cohort currently reports `NO HEADLINE` at 1–2 distinct forecast dates against
`MIN_FORECAST_DATES = 20`, so this ships on mechanism, not on a measured
directional-accuracy delta. The mechanism is direct (feature values and
medians read off the artifact and the archive); the accuracy consequence is
inferred.

A second, independent cause of the same skew was measured and is **left open**:
`_compute_price_features` joins lags by calendar date, so a day missing from the
archive exactly `lag` before the anchor NaNs `return_{lag}d` for every item
simultaneously. On the 2026-08-04 anchor, 08-03 is absent, so `return_1d`,
`log_return_1d`, `price_lag_1d`, `autocorr_1d` and `volume_price_conf_1d` are
median-filled for **5,542/5,542** items. The affected lag set rotates daily with
the archive's gap calendar.

## Implementation

* The eleven names join `SHELVED_FEATURES`. Shelved, **not deleted** — the
  columns are still engineered because `supply_to_volume_ratio` reads
  `volume_mean_30d` and `item_volume_vs_market_30d` reads raw `volume`.
* `has_volume` is deliberately left testing `notna()`. Making it treat all-zero
  as absent would only swap a served `0` for a median-filled `~98`, which is no
  more truthful.
* Two extractions, so the behaviour is testable without a training run:
  `_compute_volume_features` and `_select_feature_cols`.
* `MODEL_ARTIFACT_VERSION` → 3. This is a *set* change, which the constant's own
  rule says not to bump for — but a v2 artifact carries its own 47-column
  `feature_cols` and the columns are still engineered, so `predict()` would keep
  serving the dead features until the 14-day age trigger fired. The bump forces
  the retrain that realises the fix.

Reinstate these features only with a repaired volume feed; the Steam
listing-page route is the candidate.

## Tests

`backend/tests/test_volume_features_shelved.py` — seven tests: the shelving
itself, that `volume_*` resolves to `price_technicals` (so the allowlist cannot
be relied on instead), that selection excludes them while keeping price
technicals, that the columns are still computed for both real and all-zero
volume, and that the artifact version forces a retrain.

The one that earns its keep is
`test_no_volume_feature_survives_the_real_selection_and_prune`: it runs the
actual `build_training_data` path and asserts no surviving `volume` feature. A
name-list assertion structurally cannot catch the prune interaction above —
shelving a feature changes which *other* features the prune keeps.

`test_build_training_data_feature_count`'s lower bound moves 45 → 30; that mock
feeds real Poisson volume, so it exercises the `has_volume=True` path and lands
at 37 features.
