# CS2 Oracle — Backend

Run everything from this directory through `venv/bin/python`. Note that `.env` here
points at production Supabase — see the root `AGENTS.md` gotcha before running scripts.

**Training data comes from Parquet, not the DB.** `fetch_price_history(backfilled_only=True)`
reads `price-archive/*.parquet` via DuckDB. The DB supplies only the `is_backfilled` flag and
events metadata.

## Invariants

These four hold everywhere. The reasoning behind each is in the rule file named beside it.

1. **Read prices through `db/archive.py::prices_relation`, never a raw glob.** A bare
   `read_parquet('prices-*.parquet')` silently returns the first file's schema and drops
   columns without erroring. → `archive-reads`
2. **Any loader that globs the archive must apply `archive_universe_sql_filter()`.** It
   carries all three universe rules — the excluded bid sources, the phase-collapsed names,
   and the phantom duplicate keys — and each is NULL-safe by construction. A bare `NOT IN`
   drops 13 years of prices. → `item-universe`
3. **Never pass a bare horizon to a purge.** The embargo is `horizon + 13`, derived at call
   time by `models/forecaster.py::embargo_days`. → `labels-and-embargo`
4. **Never quote a directional accuracy on its own.** The headline is a Pesaran–Timmermann
   test; DA is quotable only beside `constant_call_accuracy` and `realised_down_rate`.
   → `backtest-scoring`

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
  122 at every horizon (`docs/changelog/2026-07-22-social-feature-audit.md`), so don't
  rebuild the collector. `collectors/social_sentiment.py` is kept for local/authenticated
  runs and scores with **FinBERT ONNX INT8** — the "VADER" comments in `models/forecaster.py`
  and `database.py` are stale.
- **Every A/B result stored in this repo predates the 2026-08-08 statistics fix.** Ten of the
  thirteen harnesses have not been re-run. Don't cite a stored verdict without checking.
