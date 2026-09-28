---
paths:
  - "backend/models/{item_parser,forecaster}.py"
  - "backend/scripts/ab_test_*.py"
  - "backend/scripts/walkforward_backtest.py"
  - "backend/api/**"
---

# The item universe and the voted price

Three rules define which item-days exist. All live in `models/item_parser.py` as SQL
predicates, re-exported from `models/forecaster.py`, because each has already cost a
separate fix per loader.

- **Any loader that globs the archive must apply `archive_universe_sql_filter()`.** Every
  rule is NULL-safe: a bare `NOT IN`/`NOT LIKE` over the pre-2026 `source IS NULL` series
  evaluates to NULL and silently drops 13 years of prices. Every `ab_test_*` harness and
  `walkforward_backtest.py` route through it (2026-08-08); `ab_test_recency_weights.py` is the
  one exemption and only because it reads a pre-built frame. Pass `source_column=None` for a
  relation with no `source` column — safe only because every bid source is a 2026 feed.
  **Frame caches fingerprint `forecaster.py`'s bytes, not `item_parser.py`'s**, so five
  harnesses hash the universe predicate into their key explicitly; a new one must too.
- **A bid, and Steam's trailing-window means, must never vote in the consensus price.**
  `aggregator_buff163_buy` is BUFF's `highest_order`, a bid; `aggregator_steam_7d/30d/90d` are
  MA(7)/MA(30)/MA(90) trailing sale prices, the wrong *time* basis rather than the wrong side
  of the book. Both `BID_SOURCES` and `TRAILING_WINDOW_SOURCES` are dropped in
  `_apply_multi_source_voting` *before* the group is read, so neither counts toward the median
  nor the ≥3-source gate that enables the 2σ mask. `STEAM_SPOT_SOURCES`
  (`aggregator_steam_spot`) joins the drop for a different reason — it would be a second Steam
  ballot alongside `aggregator_sync`, double-counting the venue. The filter is
  `~df["source"].isin(BID_SOURCES | TRAILING_WINDOW_SOURCES | STEAM_SPOT_SOURCES)`
  (`forecaster.py::_apply_multi_source_voting`, near the `~df["source"].isin(excluded)` line) — NULL-safe by construction,
  which is what keeps the pre-2026 `source IS NULL` series voting. An item-day whose only
  source was excluded returns **no row** rather than falling back to it. **The 2σ guard is not
  a defence for the bid**: it ran on 99.3% of bid item-days and kept the bid four times out of
  five, because the ask panel's own dispersion is wider than the bid–ask wedge. See
  `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md` and
  `docs/changelog/2026-08-09-trailing-window-sources-excluded.md`.
- **A `historical_fallback:` re-stamp is a stale price under a fresh date, not an
  observation.** When a day's collection misses an item, `collectors/pipeline.py:244-252`
  re-writes a quote up to 7 days old under *today's* `day`, prefixing the original source
  (`historical_fallback:<source>`). The production voted path already excludes it inline
  (`forecaster.py`'s DB and DuckDB reads), but every archive-globbing loader that bypasses
  the voted path kept it, so a stale print entered features and labels under a fresh date.
  The rule is `historical_fallback_sql_filter` / `HISTORICAL_FALLBACK_PREFIX`, NULL-safe like
  the bid filter and applied by `archive_universe_sql_filter` only when a `source` column is
  present. Worth **12,655 rows over 2026-07-11..16**, on the cohort that failed to match that
  day. This changed no voted cache (production already filtered it), so **no
  `VOTED_CACHE_VERSION` bump**. See `docs/changelog/2026-08-19-historical-fallback-universe-filter.md`.
- **A name that prices several assets is not in the universe.** Every Doppler and Gamma
  Doppler `market_hash_name` collapses its phases into one series, and the quoted headline is
  the *cheapest* phase 95.5% of the time — so the series steps when the cheapest phase
  changes, which is a fabricated return. The rule is `PHASE_COLLAPSED_SLUG_PATTERNS`,
  `phase_collapsed_sql_filter`, `is_phase_collapsed`, light enough for `api/` to import. It is
  applied at `_fetch_voted_price_history`, both `walkforward_backtest` loaders,
  `opportunities.py::_load_items` and — since 2026-08-08 — every `ab_test_*` harness. Two
  names match the word and must stay: `Sticker | Doppler Poison Frog (Foil)` and its Sticker
  Slab twin, which is what the `sticker` exemption is for. Worth **6 items of the 926-item
  ≥$1 cohort** — a correctness fix, not an accuracy lever. See
  `docs/changelog/2026-08-08-phase-collapsed-names-dropped.md`.
- **A key that is not a `market_hash_name` is a second copy of an item already in the
  universe.** `items.item_id` is the archive's `item_slug` verbatim, and two writers keyed
  rows on something else: `migrate_historical_data.py:345` wrote `slugify(name)` (3,145
  keys) and a since-deleted `real_data_collector.py` wrote `f"steam_{...}"` (4). The
  aggregator matches its price lookup on `name`, so both copies collect every day's price.
  The rule is `PHANTOM_SLUG_PATTERN`, `phantom_slug_sql_filter`, `is_phantom_slug`, and the
  two forms need **separate arms** — the `steam_` keys hold `_` and `|`, so the slug regex
  misses them. Worth **3,149 keys / 786,408 rows (3.6%)**, and all 3,149 pair 1:1 onto a
  correctly-keyed row at 99.93% same-cent agreement, so it is a **de-duplication, not a
  universe reduction**. The cost of leaving them was validation, not storage: a split that
  partitions by item can seat the same series on both sides of a fold. `is_phantom_slug` is
  also what `scripts/purge_phantom_items.py` deletes on — one definition, so the database
  and the readers cannot disagree about which items exist. The prod `items` rows were
  **purged 2026-08-09** — 3,149 items and 100,600 child rows — so the daily leak is
  stopped; the durable archive's 786,408 historical rows are inert but unpurged. See
  `docs/changelog/2026-08-08-phantom-slug-keys-dropped.md`.
- **`fetch_price_history` caches the voted frame** to `data/voted_<key>.parquet`
  (gitignored), keyed on the `prices-*.parquet` fingerprint + cutoff date + backfill slug
  set. **Bump `ItemForecaster.VOTED_CACHE_VERSION` if you change `_fetch_voted_price_history`,
  `_apply_multi_source_voting`, or the *contents* of `BID_SOURCES` / `TRAILING_WINDOW_SOURCES`**
  — the key cannot see code changes. Both frozensets are defined in `models/item_parser.py`,
  and `.github/workflows/price-forecast.yml:131` hashes only `forecaster.py` into the CI cache
  key — so editing a source name into or out of either set is invisible to CI unless the
  version constant (re-exported on `forecaster.py`, which the key does hash) moves too.
  `VOTED_CACHE=0` disables. Now at **v10**: v2 was the `BID_SOURCES` exclusion, which changed
  the consensus level, so any surviving v1 frame holds a displaced price series and would have
  trained the next model on it silently; v3 dropped the phase-collapsed names; v4 dropped the
  phantom slug keys; v5 added `n_ask_sources`; v6 excluded `TRAILING_WINDOW_SOURCES` (Steam's
  trailing-window means); v7 excluded `STEAM_SPOT_SOURCES`; v8 added orderbook features; v10
  moved the archive vote into DuckDB; v11 kept exact 2σ ties. A source-set change and a universe
  change both count as voting changes.
- **The vote has two implementations and they must change together.** The archive read
  (`_fetch_voted_price_history`) votes in SQL via `_multi_source_voting_sql`, one query per
  item-hash chunk (`VOTED_CHUNKS`, default 8) under a `VOTED_DUCKDB_MEMORY_LIMIT` cap (default
  2GB). **Do not rely on DuckDB spilling**: the CI runner did not spill, and a whole-archive
  query hit the cap there (run `36470822456`), which reproduces locally only with
  `temp_directory=''`. The DB path and the voting tests use the pandas
  `_apply_multi_source_voting`, the reference; `tests/test_sql_voting.py` holds them equal.
  The 2σ cut keeps ties within `VOTE_TIE_RTOL` (1e-9) in **both**: cent-rounded prints sit
  exactly on 2σ, where the std's last ulp depends on summation order, and without the slack
  the SQL vote was **non-deterministic across thread/chunk counts**. Against the old pandas
  frame that changed 17 of 6,584,167 rows, all sub-$1.
  `docs/changelog/2026-09-28-voting-moved-into-duckdb.md`. ⚠️ The CI key's literal prefix still reads `voted-v6`
  and was not bumped with v8 — it is now 2 versions behind, harmless only because the constant lives in the hashed file.
- **The TRAIN universe is derived from the archive; the SERVE universe is `is_backfilled` in
  the DB.** `ItemForecaster._resolve_backfilled_slugs(universe=…)` routes them:
  `universe="train"` calls `_archive_universe_slugs(exclude_iflow=True)` — distinct slugs with a
  `day < 2026-01-01` row from a source other than `buff_iflow`, read through `prices_relation` +
  `archive_universe_sql_filter` — and `universe="serve"` reads `SELECT item_id FROM items WHERE
  is_backfilled = 1`, falling back to the same archive derivation. **`Item.is_trainable` /
  `trainable_item_clause()` (`database.py:73,130`, migration `0023`) are NOT the authority for
  the train universe** and no loader reads them: the managed Postgres lacked the column when the
  split was built (migration `0023` added it later), so the DB read threw and fell back to
  loading all 41,885 slugs, which OOMed a cold retrain. The
  archive derivation makes the cohort a pure function of the data the model trains on. The
  voted-frame cache key includes `universe`, because two backfill sets give different frames
  from an identical archive. See
  `docs/changelog/2026-08-18-train-universe-derived-from-archive.md`.
