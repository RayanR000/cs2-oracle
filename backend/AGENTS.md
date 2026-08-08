# CS2 Oracle — Backend

Run everything from this directory through `venv/bin/python`. Note that `.env` here
points at production Supabase — see the root `AGENTS.md` gotcha before running scripts.

## Gotchas

- **Training data comes from Parquet, not the DB.** `fetch_price_history(backfilled_only=True)`
  reads `price-archive/*.parquet` via DuckDB. The DB supplies only the `is_backfilled`
  flag and events metadata.
- **Read prices through `db/archive.py::prices_relation`, never a raw glob.** A plain
  `SELECT * FROM read_parquet('prices-*.parquet')` used to return **four** columns and no
  error — DuckDB narrows a multi-file read to the first file's schema and
  `prices-2013.parquet` predates `source`, so `source`/`min_price`/`max_price` silently
  vanished. `scripts/normalize_price_schema.py` gave every file one schema
  (`item_slug, day, source, mean_price, volume, ingested_at`, `day` as `DATE`), but the
  reader still projects an explicit column list and NULLs what is absent, so it works against
  an unmigrated archive too. `source IS NULL` still selects the pre-2026 series — the
  migration materialised a typed NULL rather than stamping a label, because
  `init_local_db.py` derives `is_backfilled` from that equivalence. A materialised column is
  NULLed to `COLUMN_TYPES[c]`, not to VARCHAR: that shortcut was only ever right while
  `source` was the one column that could be missing.
- **`ingested_at` is arrival; `day` is what the row describes.** Added 2026-08-08 and
  **NULL for every row written before it** — the archive had no arrival timestamp anywhere
  and one cannot be reconstructed backwards. A NULL means "arrival unknown" and must never be
  read as "arrived on `day`", which is exactly what 13 years of backfilled rows would falsely
  claim. `append_to_parquet.py` stamps the run's wall clock, not `--date`, so re-exporting an
  old day records an old `day` with a present-day arrival — the truth. A re-append keeps the
  **first** arrival, so a corrected price does not date the whole month forward. **The
  migration has not been run**: only CI writes the canonical archive, so the column is absent
  from every stored file until `aggregator-update.yml` runs with `normalize_schema = true`.
- **A bid must never vote in the consensus price.** `aggregator_buff163_buy` is BUFF's
  `highest_order`, and `BID_SOURCES` is dropped in `_apply_multi_source_voting` *before* the
  group is read, so it counts toward neither the median nor the ≥3-source gate that enables
  the 2σ mask. The filter is `~df["source"].isin(BID_SOURCES)` — NULL-safe by construction,
  which is what keeps the pre-2026 `source IS NULL` series voting. An item-day whose only
  source was the bid returns **no row** rather than falling back to it. **The 2σ guard is not
  a defence here**: it ran on 99.3% of bid item-days and kept the bid four times out of five,
  because the ask panel's own dispersion is wider than the bid–ask wedge. See
  `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md`.
- **Any loader that globs the archive must apply `archive_universe_sql_filter()`.**
  `models/item_parser.py` holds the two universe rules — `BID_SOURCES` (re-exported from
  `forecaster.py`, so existing imports are unaffected) and the phase-collapsed names — as SQL
  predicates, because each of them has already cost a separate fix per loader. Both are
  NULL-safe: a bare `NOT IN` over the pre-2026 `source IS NULL` series evaluates to NULL and
  silently drops 13 years of prices. Every `ab_test_*` harness and `walkforward_backtest.py`
  now route through it (2026-08-08); `ab_test_recency_weights.py` is the one exemption and
  only because it reads a pre-built frame. Pass `source_column=None` for a relation with no
  `source` column — safe only because every bid source is a 2026 feed. **Frame caches
  fingerprint `forecaster.py`'s bytes, not `item_parser.py`'s**, so five harnesses hash the
  universe predicate into their key explicitly; a new one must too.
- **A name that prices several assets is not in the universe.** Every Doppler and Gamma
  Doppler `market_hash_name` collapses its phases into one series, and the quoted headline is
  the *cheapest* phase 95.5% of the time — so the series steps when the cheapest phase
  changes, which is a fabricated return. `models/item_parser.py` holds the rule
  (`PHASE_COLLAPSED_SLUG_PATTERNS`, `phase_collapsed_sql_filter`, `is_phase_collapsed`),
  light enough for `api/` to import and re-exported from `models/forecaster.py` beside
  `BID_SOURCES`. It is applied at `_fetch_voted_price_history`, both
  `walkforward_backtest` loaders, `opportunities.py::_load_items` and — since 2026-08-08 —
  every `ab_test_*` harness. Two names match the word and must stay:
  `Sticker | Doppler Poison Frog (Foil)` and its Sticker Slab twin, which is what the
  `sticker` exemption is for. Worth **6 items of the 926-item ≥$1 cohort** — a correctness
  fix, not an accuracy lever. See
  `docs/changelog/2026-08-08-phase-collapsed-names-dropped.md`.
- **The embargo is `horizon + 13`, not `horizon`.** `models/forecaster.py::embargo_days`
  derives the 13 at call time from `LAG_TOLERANCE_DAYS` (3) + `SMOOTH_WINDOW` (3) +
  `MAX_WINDOW_SPAN_DAYS` (7): the label at `d + horizon` is a **resolved anchor**, not a
  point observation, so its support runs 13 days past its nominal date and a bare-`horizon`
  purge left that carry inside the validation window. Never pass a bare horizon to
  `_compute_cv_splits(purge_days=…)` or re-derive the band locally. At h=30 the embargo (43d)
  **exceeds `VALIDATION_WINDOW_DAYS`** — that is the correct cost, not a bug. It also tracks
  the env-overridable `FALLBACK_MAX_AGE_DAYS`, so fold geometry is not a constant.
- **`walkforward_backtest.py` embargoes by default** since 2026-08-08, and writes
  **`model_version = "lgbm-v4-embargoed"`** when it does. That bump is the point: appending
  purged rows to the un-purged `lgbm-v3-clustered` series would surface as a model regression
  on the dashboard trend and in `backtest-triage`, with nothing stored to say otherwise. The
  persisted `metrics` also carry `purge` and `embargo_days`, so a row is self-describing.
  `--no-purge` reproduces the old split — and the old version string — for a like-for-like
  read against a pre-flip run, and must never be published from. The discontinuity was
  **measured on 2026-08-08** by running both arms and pairing them: the un-purged split inflates
  DA by **+10.15pp at 30d** and **+5.44pp at 14d** (both intervals exclude zero); h=3 and h=7 are
  unresolved, not clean. See `docs/changelog/2026-08-08-embargo-discontinuity-measured.md`.
- **An A/B verdict is a fold-clustered interval, never a win count.** Ten harnesses decided
  on a fold win-count, a ±0.5pp pooled-delta threshold or a bare `a > b` until 2026-08-08 —
  none is a test, against a measured item-level MDE of 2.21–3.69pp. Use
  `backtest/paired_mde.py`: `paired_da_difference` for a hit rate,
  `paired_metric_difference(value_key=…)` for anything else (pass
  `higher_is_better=False` for a loss), and `paired_arm_contrasts` / `format_paired` /
  `verdict` for the reporting. Records come from `backtest/walkforward_records.py`:
  `paired_records` at row grain — always prefer it — and `fold_level_records` only where the
  rows are genuinely gone, as in the three pinball harnesses that shard folds across
  processes. `unresolved` (fewer than 2 shared clusters, or a non-finite bound) is **not**
  `null`. Strip `records` with `without_records` before printing; under `--arm` sharding no
  shard can contrast in-process, and `merge_price_primitives_ab.py` pairs the shards at fold
  grain off `per_fold`. Only `NoPairedRows` is catchable — a missing `cluster_key` is a wiring
  bug and propagates, because swallowing it restores the 2026-08-07 under-dispersion bug
  behind a confident wrong message. **Every stored A/B result predates this**, and no harness
  has been re-run.
- **Never quote a directional accuracy on its own.** The published headline is a
  Pesaran–Timmermann test (`backtest/directional_test.py`), computed per forecast date with a
  Newey–West t-stat over dates and a `|t| > 3.0` hurdle; `score_cohort` stores it as `pt_*` and
  `scripts/backtest_accuracy.py::_headline_line` logs it. DA is quotable only beside
  `constant_call_accuracy` and `realised_down_rate` — an always-down call scored **29.4% on
  2025-12-01 and 76.9% on 2026-07-17** at 7d, so a fixed hit rate is skill on one date and
  incompetence on the next. Note a **constant call has per-date excess identically zero**, so it
  resolves as `degenerate`, never as skill. `baseline_directional_accuracy` is the always-*flat*
  call, not the constant-call baseline, despite the name. See
  `docs/changelog/2026-08-07-pesaran-timmermann-headline.md`.
- **`price_tier` has six bands, and tier 4 changed meaning.** The cut at $1000 landed
  2026-08-07 because the bid–ask spread is 10.8% at $50–500 against 5.2% at $1000+ — the two
  most different liquidity populations in the market. **A stored row with `price_tier == 4`
  written before that date means `≥$100`, not `$100–1000`.** `score_by_tier` also emits the
  `FLOOR_SWEEP` sentinels `-1`/`-2`/`-3` for the `≥$1`/`≥$5`/`≥$20` headline sweep; only `-1`
  (`HEADLINE_TIER`) is the published headline, and `price_tier = NULL` is pooled across those
  liquidity populations and is **not a quotable number**. Adding a floor that is not a
  `price_tier` cut silently rounds down — `floor_records` filters in tier space.
- **`actionable_*` is populated only at h ∈ {14, 30}.** `backtest/actionable.py` conditions
  DA on `|r̂| > round trip + tier spread`, so it answers "does this call imply a trade at
  all" — at CSFloat that bar is 7.2% at tier 5 and 23.1% at tier 1. Read
  **`actionable_scope`** first: `out_of_scope` (wrong horizon), `no_prediction_leg` (records
  without `predicted_mid`) and an `actionable_n` of 0 are three different states, and only
  the last one is a result. It uses the **raw sign**, not `direction_from_return`, so a
  carried-forward price (`r_act == 0`) scores as a miss. `SPREAD_BY_TIER` in
  `backtest/friction.py` is a **nearest-band approximation** — the spread was measured at
  bands that are not the tier cuts — so never cite it as measured per tier.
- **Operational tables live in `price-archive/ops/*.parquet`.** API routes read Parquet
  first with a DB fallback. See `db/parquet.py`.
- **The ops mirror carries `item_slug`; the DB table does not.** Prices key on the slug
  and ops keys on the Postgres surrogate `item_id`, so the archive could not be joined to
  itself without Supabase. `item_forecasts` and `forecast_outcomes` denormalise the slug
  onto the **Parquet side only** (`_with_item_slug` in `scripts/backtest_accuracy.py`) —
  `to_write` goes to `bulk_insert_mappings`, which errors on an unmapped key. **Every
  mirror writer must supply the column**: `append_table` dedups on the key and replaces
  the whole row, so an omission blanks the slug on everything it touches. That includes
  `_refresh_verdict_columns`, which is why it reads the id→slug map too.
- **Nested values are stored as JSON text, store-wide.** `db/parquet.py::_jsonify_nested`
  serialises dict/list columns before they reach DuckDB, because DuckDB infers a nested
  column's SQL type *from the batch's contents* — the same code produced `STRUCT`,
  `MAP(VARCHAR, DOUBLE)` and `VARCHAR` for `prediction_accuracy.metrics` on different
  days, and the frozen on-disk struct rejected all but one. Never hand raw dicts to
  `append_table`. See `tests/test_parquet_nested_columns.py`.
- **`fetch_price_history` caches the voted frame** to `data/voted_<key>.parquet`
  (gitignored), keyed on the `prices-*.parquet` fingerprint + cutoff date + backfill slug
  set. **Bump `ItemForecaster.VOTED_CACHE_VERSION` if you change `_fetch_voted_price_history`
  or `_apply_multi_source_voting`** — the key cannot see code changes. `VOTED_CACHE=0` disables.
  Now at **v3**: v2 was the `BID_SOURCES` exclusion, which changed the consensus level, so any
  surviving v1 frame holds a displaced price series and would have trained the next model on
  it silently; v3 dropped the phase-collapsed names. A source-set change and a universe change
  both count as voting changes.
- **Training subsamples the pool, and the draw is not free.** `TRAIN_FEATURE_ROWS` (default
  `DEFAULT_TRAIN_FEATURE_ROWS = 100_000` in `scripts/forecast_prices.py`) caps the feature
  rows, so a retrain sees a small fraction of the item universe — **99 items of 5,542, while
  `predict()` scores 8,691**. Raising it is a measured cost increase (weekly, not daily: the
  workflow retrains Mondays only), and the fresh-model gate cannot detect the difference.
  `_stratified_item_subsample` uses a hardcoded `seed=42`, and **changing only that seed
  moves `mean_classifier_acc_ge1` by sd 1.5–3.1pp** (8 seeds, 2026-08-07) — so never read an
  effect off a single retrain, and never compare a universe-changing arm against a
  one-seed baseline. **`TRAIN_MIN_MEDIAN_PRICE`** (default `None`) restricts the universe by
  median price *before* the subsample; at `1.0` with a budget ≥ 1.0M it covers the whole
  926-item ≥ $1 cohort with no subsample at all, which is the only setting that removes the
  draw rather than shrinking it. Raise `max_rows` alongside it — the 700K default starts
  binding at ~1M feature rows. See `docs/changelog/2026-08-07-training-item-universe.md`.
- **`TRAIN_PER_ITEM_ROWS` changes what the per-horizon cap spends the budget on.** Default
  off, where `_build_production_split` caps an oversized slice with a uniform
  `train_set.sample(n=max_rows)` and an item's share of the sample is its share of the
  rows. Set it and the cap draws an **equal quota per item** instead
  (`_per_item_row_sample`), which is the axis the paired harness measured: 71K rows/fold
  spread over 728 items beat 728K rows/fold at **+5.72pp vs +3.50pp at 30d**. Only
  `train_set` is thinned — never `val_set`. It is safe **only after feature engineering**:
  thinning earlier computes lags over a punctured series, and thinning before
  `prepare_targets` voids the label of any row whose `date + horizon` partner was dropped.
  It does **not** reduce feature-engineering time, which is what a wide
  `TRAIN_FEATURE_ROWS` actually buys.
- **`MIN_SERVED_PRICE_USD = 1.0` is a convention, not a derivation.** `api/serving_policy.py`
  sets the floor deliberately equal to the lower bound of `HEADLINE_MIN_TIER` so the
  population the product shows is the population the headline accuracy number describes.
  `tests/test_serving_policy.py` fails if the two diverge. Sub-$1 items are ~72% of the
  forecast universe and one cent there is a 20% move.
- **Training is fully sequential.** Horizons, quantiles, and ensemble members train one at
  a time; LightGBM's OpenMP threads supply the CPU parallelism. Ensemble members get
  `n_jobs = max(1, cpu_count // 2)`; the Optuna search params still use `n_jobs: -1`.
- **The backtest resolves BOTH legs through `backtest/price_resolution.py::resolve_anchors`.**
  `item_forecasts.current_price` is stored on the outcome for reference and is **never
  scored on** — using it as the base leg is the bug that let one cohort score 61.76% and
  33.74% on different days. Resolved outcomes are **frozen**: `base_price` / `actual_price`
  / `resolved_at` are final, and `--reresolve` is the only thing that can move them
  (`--rescore` recomputes verdicts from the frozen actuals without reading the archive).
- **Backtest maturity is bounded by archive coverage, not `date.today()`.** A forecast is
  evaluable only once `prices-*.parquet` covers its target date; the cutoff is
  `min(today, archive_max_day())`. The archive always lags the calendar, and admitting
  that lag window puts guaranteed misses into the cohort — it tripped the 10%
  unresolvable gate at 38.5% and reported nothing.
- **`scripts/walkforward_backtest.py` does NOT use `fetch_price_history`.** Its own
  `_load_all_prices` skips multi-source voting, the `historical_fallback:` source filter,
  the dead-item filter, and `backfilled_only`. It does apply the universe rules, in SQL,
  through `archive_universe_sql_filter`. Lags are *not* corrupted — `engineer_features`
  collapses to one row per item-day — but it collapses the archive's 1.37× duplicate
  item-days with a plain **mean**, where production serves an **outlier-voted median**
  (sources >2σ from the median are rejected). The fresh-model gate therefore scores a
  different price consensus over a different item universe than production trains on.
  Not directly comparable to production DA.
- **Social sentiment features are permanently zero in production.** `reddit-sentiment.yml`
  was deleted: old.reddit.com returns `403 Blocked` from runner IPs, and prod
  `social_mentions` holds 0 rows all-time. The features also rank outside the top 20 of
  122 at every horizon (`docs/changelog/2026-07-22-social-feature-audit.md`), so don't
  rebuild the collector. `collectors/social_sentiment.py` is kept for local/authenticated
  runs and scores with **FinBERT ONNX INT8** — the "VADER" comments in `models/forecaster.py`
  and `database.py` are stale.
- **Model size/speed levers are documented.** See `docs/architecture/model-optimization.md`
  for the options that retain ≥90% quality.
