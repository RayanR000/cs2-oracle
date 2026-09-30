# Delete uncalled spec-12..14 code and the dead SHRINK_K_GBM / vol-rank training branches

2026-09-30. Items 15-17 of `research/2026-09-28-next-steps.md`.

## Deleted

Re-verified with no callers across the whole repo, workflows, scripts and docs. There are no
notebooks, and no saved artifact references any of it.

- `models/source_weight_model.py` (no importers, no tests).
- `models/data_quality.py` and `tests/test_data_quality.py` (imported only by that test).
- `conformal.calibrate_adaptive` / `update_adaptive` and `tests/test_adaptive_conformal.py`, plus
  `served_recalibration.stratum_factors_from_panel`. Per-sigma-decile q_hats are the refuted
  conditioning family (`research/2026-09-22-adaptive-conformal-preregistration.md`, which already
  says they "have no callers").
- A correction note was appended to `2026-09-22-post-csmarketapi-served-performance.md`: source
  weights and adaptive conformal never shipped.

## SHRINK_K_GBM / VOLATILITY_RANK_GBM: flag stays, training branches go

`SHRINK_K_GBM` is set (`"0"`) in `price-forecast.yml`, and both flags are persisted in the model
artifact's `meta.json` (`shrink_k_gbm`, `vol_rank_gbm`, `vol_rank_norm`) and read back on load.
Under the workspace rule (never remove a flag without a retrain in the same PR) the env reads, the
meta write/read, the serve-side `_*_served()` accessors, the serve-time vol-rank multiplier and
model save/load are kept. Removed: the three branches reachable only with a flag on (vol-rank
training in the auxiliary-head block, shrink-K training in the climatology calibration, vol-rank
modulation in the calibration scale). Setting either flag to 1 is now a no-op for training.

The helpers (`_fit_vol_rank_model`, `_compute_per_item_optimal_k`, `_fit_shrink_k_model`,
`predict_shrink_k`, `_build_climatology_table_adaptive`) stay because the archived replays
`scripts/archive/shrink_k_vol_rank_ab.py` and `shrink_k_stability.py` call them. The flags and the
remaining plumbing go in the next retrain PR.

## Stale reproduce paths

Paths to the seven scripts moved into `scripts/archive/` were fixed in `docs/architecture/`,
`docs/research/` (preregistrations: path-only, each with an erratum note), code comments and
`tests/test_mde_plumbing.py`. Changelogs, specs and plans are left as written (append-only).
`run_forecast_local.sh` still checks the old `scripts/centre_vs_lastprice.py` path and so skips the
informational centre gate; left alone because fixing the path would start running it.
