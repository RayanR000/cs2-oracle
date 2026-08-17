# Clean Steam spot persisted as `aggregator_steam_spot`

**2026-08-17.** Added a new archive source `aggregator_steam_spot` = Steam's
`last_24h` written **without** the trailing-window fallback, as the prerequisite Steam
leg for a future cross-venue (Steam–Buff) basis feature.

## Why

`docs/research/2026-08-16-cross-venue-basis-steam-buff.md` measured a real signal — the
Steam–Buff basis deviation predicts Steam catch-up (corr −0.12, n≈880k) and confirms Buff
leads Steam — but it was measured on the **source-null Steam backfill** (clean daily Steam,
2022–2025), which does not exist for the 2026+ serving regime. The only serve-time Steam
series is `aggregator_sync`, which is `last_24h` **falling back to 7d/30d/90d means on
exactly the illiquid items** where a basis matters, so building the feature on it would
fabricate basis moves. There was no point-in-time Steam price in the archive at all
(`models/item_parser.py:61-63`). This persists one.

## What changed

- **`collectors/pipeline.py`** steam branch: emit `("aggregator_steam_spot", p24)` — `p24`
  only, no fallback (empty on illiquid item-days, which is the point). Archive-only; the DB
  `SOURCE_LABELS` path is untouched (the forecaster trains on Parquet).
- **`models/item_parser.py`**: `STEAM_SPOT_SOURCES = frozenset({"aggregator_steam_spot"})`,
  re-exported on `ItemForecaster`.
- **`models/forecaster.py`**: added to the `_apply_multi_source_voting` exclusion
  (`BID_SOURCES | TRAILING_WINDOW_SOURCES | STEAM_SPOT_SOURCES`) so it never casts a second
  Steam ballot against `aggregator_sync`; `VOTED_CACHE_VERSION` 6 → 7 (a voting source-set
  change is invisible to the cache key otherwise — item-universe rule).
- **`tests/test_steam_spot_source.py`**: 7 tests (named; excluded from vote; does not
  double-count vs `aggregator_sync`; spot-only day dropped not zeroed; NULL-safe; cache
  bumped). Full voting suite (33 tests) + pipeline/aggregator suite (25) green.

## Scope / caveats

- **No backfill.** Past `last_24h` was folded into `aggregator_sync` and cannot be
  separated, so the clean series starts empty and accumulates only from the first CI
  aggregator run after this merge (`price-archive` is CI-written). ~2–4 weeks before there is
  enough clean spot to build and A/B the basis sidecar.
- **Feature not built yet.** This is the ingest prerequisite only. The basis feature
  (`basis_dev` sidecar + flag-gated `_compute_basis_features`, mirroring the bid/stattrak
  pattern) is a later step, and per `AGENTS.md` this is a range forecaster where
  relative/directional features have been CV-positive / serving-negative — so the feature may
  still not ship. This step is additive and low-risk regardless: it only persists data.
