# CS2 Oracle — Backend

Run everything from this directory through `venv/bin/python`. Note that `.env` here
points at production Supabase — see the root `AGENTS.md` gotcha before running scripts.

**Training data comes from Parquet, not the DB.** `fetch_price_history(backfilled_only=True)`
reads `price-archive/*.parquet` via DuckDB. The DB supplies only the `is_backfilled` flag and
events metadata.

## Invariants

These five hold everywhere. The reasoning behind each is in the rule file named beside it.

1. **Read prices through `db/archive.py::prices_relation`, never a raw glob.** A bare
   `read_parquet('prices-*.parquet')` silently returns the first file's schema and drops
   columns without erroring. → `archive-reads`
2. **Any loader that globs the archive must apply `archive_universe_sql_filter()`.** It
   carries all four universe rules — the excluded bid sources, the re-stamped
   `historical_fallback:` prints, the phase-collapsed names, and the phantom duplicate keys —
   and each is NULL-safe by construction. A bare `NOT IN` drops 13 years of prices.
   → `item-universe`
3. **Never pass a bare horizon to a purge.** The embargo is `horizon + 13`, derived at call
   time by `models/forecaster.py::embargo_days`. → `labels-and-embargo`
4. **Never quote a directional accuracy on its own.** The headline is a Pesaran–Timmermann
   test; DA is quotable only beside `constant_call_accuracy` and `realised_down_rate`.
   Of those two, **`realised_down_rate` is the runnable baseline** — `constant_call_accuracy`
   picks its direction with hindsight, so **never difference model DA against it**.
   → `backtest-scoring`
5. **Shadow predictions never drive serving; champions change only by reviewed code.**
   `forecast_candidates` / `rank_score` must not appear in `item_forecasts`, public
   schemas, or the band path, and no report, scorer, or database row may mutate
   `CENTRE_CHAMPIONS` — promotion is a separate PR after a
   `PASS_FOR_MANUAL_PROMOTION` report.

## Rules

Subsystem detail lives in `.claude/rules/` and loads automatically when you read a matching
file. It does **not** load when you are writing a new file from scratch, and it is not
re-injected after `/compact` — **read the rule yourself before starting work in its area.**

| Rule | Covers | Loads on |
|---|---|---|
| `archive-reads` | archive schema, `ingested_at`, ops mirror, nested columns | `db/`, `collectors/`, the Parquet scripts |
| `item-universe` | universe filter, bid exclusion, phase collapse, voted cache | `models/item_parser.py`, `forecaster.py`, `api/`, harnesses |
| `labels-and-embargo` | frozen price runs, the H+13 embargo, walkforward's loader | `models/staleness.py`, `forecaster.py`, forecast/walkforward scripts |
| `ab-statistics` | paired intervals, `walkforward_records`, why win counts are not tests | `backtest/paired_mde.py`, `ab_test_*.py` |
| `backtest-scoring` | resolution, maturity, PT test, tiers, staleness bands, actionable | `backtest/`, `backtest_accuracy.py` |
| `training-budget` | `TRAIN_FEATURE_ROWS`, the seed, `TRAIN_PER_ITEM_ROWS` | `models/forecaster.py`, `forecast_prices.py` |
| `serving-policy` | `MIN_SERVED_PRICE_USD` and the headline tier it tracks | `api/` |

## Gotchas

- **Social sentiment features are permanently zero in production.** `reddit-sentiment.yml`
  was deleted: old.reddit.com returns `403 Blocked` from runner IPs, and prod
  `social_mentions` holds 0 rows all-time. The features also rank outside the top 20 of
  123 at every horizon (`docs/changelog/2026-07-22-social-feature-audit.md`), so don't
  rebuild the collector. `collectors/social_sentiment.py` is kept for local/authenticated
  runs and scores with **FinBERT ONNX INT8** — the "VADER" comments in `models/forecaster.py`
  and `database.py` are stale.
- **Every A/B result stored in this repo predates the 2026-08-08 statistics fix and the
  2026-08-13 trainer fix.** Ten of the thirteen harnesses have not been re-run, and until
  2026-08-13 all thirteen early-stopped on the rows they scored using a trainer production
  abandoned on 2026-08-08. The defects are stacked — embargo, `paired_mde` clustering, two label
  changes, the ≥$1 universe, the trainer — so re-running one harness settles one harness. Don't
  cite a stored verdict without checking. → `ab-statistics`
- **The harness family was repinned off `STEAMCOMMUNITY` (2026-08-13); the invariant it broke
  still stands.** `source = 'STEAMCOMMUNITY'` matches **0 rows** in the archive — the values are
  `NULL` (pre-2026, ends 2025-12-31) and thirteen `aggregator_*` feeds. It used to sit inside an
  `item_slug IN (...)` subquery in `supply_side` / `regime` / `ensemble`, which selected zero
  items, and four more harnesses degenerated silently to a NULL-only frame. All are now repinned
  to `source IS NULL` and carry a non-zero `RuntimeError` guard (verified: 871 / 5,536 / 5,536
  items). **Still run the universe query and assert a non-zero row count before citing or
  re-running any `ab_test_*.py`** — the guard is the invariant, not a one-time fix. The six that
  skipped `_apply_feature_allowlist` (measuring a 138–180-column model where production serves 28)
  now apply it. → `changelog/2026-08-13-the-null-verdicts-were-not-all-tested.md`
- **Neither copy of `ops/forecast_outcomes.parquet` is the full scored panel — query prod
  Postgres read-only for any panel figure.** The **durable** archive that CI writes is fresh
  and its cells are complete, but shallow: 70,409 rows over 10 dates, with 2025-12-01 and
  2026-07-17 absent entirely. Your **local working copy** is the dangerous one — deep but stale
  since 2026-08-02, missing 36% of the ≥$1 rows, and its surviving rows in a deficient cell are
  **selected on "the verdict changed"** by a later verdict-refresh run, so their rates are not
  the population's. That cost nine published figures on 2026-08-11, three of which reversed
  sign. **The publish leg is not broken** — don't go hunting for a dead one. Per-cell test and
  all three stores' numbers are in the `archive-reads` rule, which is scoped to `db/` and
  `collectors/` and so will **not** load just because you are doing backtest analysis.
