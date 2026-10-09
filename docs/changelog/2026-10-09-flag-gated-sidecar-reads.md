# 2026-10-09 — Sidecars are read only when their feature flag is on

## Cost

Predict feature engineering takes 3 min 54 s of the 7.5-min Price Forecast run (`37876868155`,
day 10-09): 6 chunks × 2 passes, ~20 s per `engineer_features` call, ~14 s of it before
"Engineering price features" logs. That gap is `_attach_sidecars`, which re-reads and merges
every sidecar on every call. The volume panel was already gated; the other three were not:

| Sidecar | Rows | Only consumer | Flag (default off) |
|---|---|---|---|
| `bid-panel.parquet` | 0.63M | `_compute_bid_features` | `BID_FEATURES` |
| `stattrak-panel.parquet` | 4.5M | `_compute_stattrak_feature` | `STATTRAK_FEATURE` |
| `supply-history.parquet` | 15.4M | `_compute_supply_churn_features` | `SUPPLY_CHURN_FEATURES` |

The raw columns (`buff_bid`, `st_premium`, `buff_listing_count`) are never selectable:
`_select_feature_cols` excludes them by name. All three files ship in `cs2-oracle-data`, so CI
paid for them.

## Change

`ItemForecaster._sidecar_needed` decides per file; `_attach_sidecars` skips a file whose flag is
off. Turning a flag on reads its file exactly as before. Training and predict both go through
`engineer_features`, so both get it.

No sidecar has duplicate `(item_id, date)` keys (checked on the local archive), so the left
merges never added rows and skipping them is column-only.

## Measured (local archive, 1,000 random items, 240 days, `engineer_features`)

- 7.3 s → 3.3 s per call.
- Same 132,868 rows; the 160 shared columns are identical (`assert_frame_equal`). The only
  dropped columns are the three raw sidecar literals.
- Full suite 170 s → 82 s locally, since tests that build features no longer read the real
  15.4M-row supply history.

New test: `test_sidecar_attach.py::test_flag_gated_sidecars_are_not_read_when_their_flag_is_off`.
The two bid-join tests now set `BID_FEATURES=1`.

## Expected

About 2–3 min off each daily Price Forecast. Read the next run's `market pass` / `feature pass`
timestamps to confirm.

## Not done

Review §2's other half, skipping pass A of `_engineer_features_chunked` and passing
`skip_unused_groups`, is safe only while `regime_models` is empty and the tier-lead / xs-rank
flags are off. It needs its own served-column equivalence check.
