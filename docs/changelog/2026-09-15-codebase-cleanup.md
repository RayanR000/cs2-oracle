# Codebase cleanup (2026-09-15)

Implements `docs/specs/2026-09-15-codebase-cleanup-spec.md` (tiers 1–5, 7–8
fully; 6.3–6.7 fully; three deliberate deviations below).

## What changed

- **Correctness.** `GET /items/{id}/price-history` applied `skip` twice
  (returned records 20–24 for `skip=10&limit=5`); `/opportunities` disclosed
  the raw DB `direction` as `current_trend` while `DIRECTION_DISCLOSED=False`.
  Both fixed — opportunities now routes through `served_direction()` with
  range-based reason copy. Pinned by new tests in
  `test_direction_withheld.py`.
- **Deleted:** `models/tft/` (+5 test files, incl. the spec-unlisted
  `test_tft_integration.py`/`test_tft_model.py`, which only tested the deleted
  module), `backtest/papertrade.py`, `backtest/longshort.py`,
  `api/routes/portfolio.py`, `collectors/{google_trends,export_finbert_onnx,
  supply_scraper,data_validation}.py` (+ tests), the TFT/WACI/momentum-fallback
  forecaster wiring, `MOMENTUM_FALLBACK_HORIZONS` + the predict-time
  conditional + alias (tests repointed at `direction.recenter_on_momentum`),
  WACI in `conformal.py` (+ `scripts/measure_waci.py`, `test_measure_waci.py`),
  dead `pipeline.py` loaders, the social-sentiment endpoint, the constant-zero
  `supply_skinport_qty_log` feature (ORM column kept), and
  `forecast_market_factor_diagnostics` (+ its test).
- **Archived to `scripts/archive/`:** 43 concluded experiment scripts + 7
  backfill/ingest helpers the suite imports (`ingest_supply_history`,
  `ingest_volume_panel`, `backfill_supply_metadata`,
  `backfill_steam_listing_history` + `backfill_ssr_history` (imported by the
  former at module level), `ingest_fx_history`, `purge_phantom_items`).
  Test/staying-script/workflow imports repointed to `scripts.archive.*`;
  `centre_vs_lastprice.py` and `backtest_walkforward_report.py` got one-extra-
  parent path fixes (still executed live — CI gate and `run_task`
  respectively). **Deleted outright:** `discover_steam_items.py` (broken
  import), `build_market_catalog.py`, `repair_catalog_gaps.py`, `migrate.sh`;
  `discover-new-items.yml` now exits with an explanation instead of a
  file-not-found.
- **DRY:** `QualityVariantOut`/`GroupedMarketItemOut`/`parse_item_name` live
  in `schemas.py` only; parquet/DB trend paths share `_trend_indicators()`;
  duplicate `score_sentiment` removed.
- **Quality:** parquet→DB fallbacks log at debug; `directional_accuracy()`
  vectorized via `direction_classes()` (equivalence spot-checked over 200
  random frames); startup moved to `lifespan`; session token dropped from the
  OAuth redirect URL (cookie already carries it); `Settings.check_secret_key()`
  refuses prod boot with the default key; dead `volatility` schema fields
  removed; `/ab-test` responses carry an unpowered-results disclaimer;
  `csmarketapi_backfill` no longer spends quota on the frozen player-counts
  panel; VADER docstrings → FinBERT.

## Deliberate deviations from the spec

1. **3.7 skipped.** `_build_trend_explanation` keeps its bullish/bearish
   branches: they are the republication path (flipping `DIRECTION_DISCLOSED`
   republishes with no code change, per `serving_policy.served_direction`'s
   contract) and are pinned by `test_trend_explanation_copy.py`. Simplifying
   to neutral-only would break that design for zero runtime effect — the
   branches are unreachable while the flag is False.
2. **6.1/6.2 deferred.** Splitting `_train_horizon_inline` (~1k lines) and
   `predict()` (~600 lines) is behavior-preserving only if done perfectly,
   and a mistake silently changes the trained model or served bands — failure
   modes the suite cannot catch (full `train()` is far too slow to test).
   The file already decomposes via 60+ helpers (direction, conformal,
   climatology); the remaining orchestration split should ride along with a
   change that gives test coverage, not land alone.
3. **6.8 skipped.** `market_summary` paginates (`skip`/`limit`) from a cached
   whole-list build; a DB-level `LIMIT` would silently truncate later pages.
   The 600s cache already amortizes the ~3.5s build.
4. **Tier 4 vs the suite (user-approved).** 26 of the 59 scripts are imported
   by tests — several pin live invariants (`STEAM_FEE_MULTIPLIER`, rarity
   coalescing, universe guards). Per agreement: archive (not delete) the
   imported ones and repoint imports, rather than delete scripts + tests.
5. **6.6 placement.** The prod-default-key guard lives in
   `Settings.check_secret_key()`, called from `lifespan` — not in
   `model_post_init`. `backend/.env` declares `ENVIRONMENT=production` with
   no key, so a construction-time raise broke every import (the whole suite).
   Boot-time refusal keeps the guarantee where it matters (server startup)
   without breaking tests and offline scripts.
