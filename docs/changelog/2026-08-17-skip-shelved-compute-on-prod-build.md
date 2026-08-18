# Skip the shelved-and-unused feature compute on the production frame build

**2026-08-17**

## What changed

`engineer_features(skip_unused_groups=True)` — the production training build — no
longer *computes* two shelved blocks that live inside `_compute_price_features`
(the one group the allowlist never skips):

1. **The volume pipeline** — the `volume-panel.parquet` read + merge in
   `_attach_sidecars`, and `_compute_volume_features`' shifts + five
   `groupby.rolling` aggregations. All 13 outputs are in `SHELVED_FEATURES`; their
   only live consumers are `supply_to_volume_ratio` (`supply_depth`) and
   `item_volume_vs_market_30d` (`cross_sectional`), **both of which the production
   allowlist (`price_technicals`) already skips**. So on the prod build nothing
   read them, yet they were built every frame construction.
2. **The 2026-07-26 price primitives** — `vol_semidev_{down,up}_30d`,
   `vol_skew_30d`, `rsi_divergence_7d`, `rsi_price_divergence_7d`,
   `macd_hist_slope_7d`. All shelved (cleared no A/B gate,
   `2026-07-31-price-primitives-shelved.md`), no reinstate flag, read only by
   `ab_test_price_primitives.py` off the full frame.

## Why this was safe to skip (and where it is NOT skipped)

Shelving already kept these out of `feature_cols`; this change removes the wasted
*compute*, not anything the trainer fits on. The gate is byte-identical for every
caller that still reads the columns:

- `need_volume = _volume_features_enabled() or "supply_depth" not in skip or "cross_sectional" not in skip`.
  So `VOLUME_FEATURES=1` (the documented "reinstate with a repaired feed" path)
  and any allowlist that admits a consuming group still build volume.
- `need_primitives = not skip_unused_groups`. The A/B harnesses and the chunked
  predict path both call `engineer_features` with `skip_unused_groups=False`, so
  they are unchanged.

## Cost

Synthetic (not a prod timing): on a 1,000,000-row frame, skipping both compute
blocks cut `_compute_price_features` from 7.21s to 6.40s (**−0.81s, −11%**), plus
the avoided `volume-panel.parquet` read + full-frame merge. This trims the **cold
frame build** only — it is amortized by the engineered-data parquet cache and is
not the dominant training cost (the conformal CV is). It is a waste-removal, not a
speed lever with an accuracy trade-off.

## Tests

- `tests/test_shelved_compute_skipped_in_prod.py` (new): prod build omits both
  blocks; the A/B/predict path (`skip_unused_groups=False`) still builds them;
  `VOLUME_FEATURES=1` reinstates volume; admitting `supply_depth` reinstates
  volume.
- `tests/test_volume_features_shelved.py::test_no_volume_feature_survives_the_real_selection_and_prune`
  updated to the stronger contract (compute skipped, not merely dropped from
  `feature_cols`). The prune-interaction guard it carried is unit-tested against
  `_select_feature_cols` in `test_feature_selection_excludes_shelved_volume_columns`.
