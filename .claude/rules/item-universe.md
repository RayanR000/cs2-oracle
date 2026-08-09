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

- **Any loader that globs the archive must apply `archive_universe_sql_filter()`.** Both
  rules are NULL-safe: a bare `NOT IN` over the pre-2026 `source IS NULL` series evaluates to
  NULL and silently drops 13 years of prices. Every `ab_test_*` harness and
  `walkforward_backtest.py` route through it (2026-08-08); `ab_test_recency_weights.py` is the
  one exemption and only because it reads a pre-built frame. Pass `source_column=None` for a
  relation with no `source` column — safe only because every bid source is a 2026 feed.
  **Frame caches fingerprint `forecaster.py`'s bytes, not `item_parser.py`'s**, so five
  harnesses hash the universe predicate into their key explicitly; a new one must too.
- **A bid must never vote in the consensus price.** `aggregator_buff163_buy` is BUFF's
  `highest_order`, and `BID_SOURCES` is dropped in `_apply_multi_source_voting` *before* the
  group is read, so it counts toward neither the median nor the ≥3-source gate that enables
  the 2σ mask. The filter is `~df["source"].isin(BID_SOURCES)` — NULL-safe by construction,
  which is what keeps the pre-2026 `source IS NULL` series voting. An item-day whose only
  source was the bid returns **no row** rather than falling back to it. **The 2σ guard is not
  a defence here**: it ran on 99.3% of bid item-days and kept the bid four times out of five,
  because the ask panel's own dispersion is wider than the bid–ask wedge. See
  `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md`.
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
  set. **Bump `ItemForecaster.VOTED_CACHE_VERSION` if you change `_fetch_voted_price_history`
  or `_apply_multi_source_voting`** — the key cannot see code changes. `VOTED_CACHE=0` disables.
  Now at **v4**: v2 was the `BID_SOURCES` exclusion, which changed the consensus level, so any
  surviving v1 frame holds a displaced price series and would have trained the next model on
  it silently; v3 dropped the phase-collapsed names; v4 dropped the phantom slug keys. A
  source-set change and a universe change both count as voting changes.
