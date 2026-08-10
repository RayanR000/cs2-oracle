# Repo cleanup: 107 dead files, 67MB of tracked artifacts, and a test that hit production

**Date:** 2026-08-10

Housekeeping pass over the whole repo. Two findings were more than housekeeping and are
written up first.

## The test suite was connecting to production on every collection

`backend/tests/test_price_history.py` contained **no test functions and no test classes**. It
was a top-level script: line 25 was a module-level `db = SessionLocal()`, and line 31 opened a
loop over five items hitting `steamcommunity.com/market/pricehistory/` with
`time.sleep(10)` between requests.

There is no `conftest.py`, `pytest.ini`, `pyproject.toml` or `setup.cfg` anywhere in the repo,
so nothing excluded it. `pytest tests/ -q` — the command `AGENTS.md` documents as the safe
one — imported it on collection, which opened a session against **prod Supabase**
(`backend/.env`) and made five live Steam calls, every single run.

This also explains a symptom the docs had already recorded but attributed to nothing in
particular. `AGENTS.md` said the suite "spends ~45s collecting before the first test runs" and
told readers to prefer targeted runs because of it. That was four `time.sleep(10)` calls plus
the Steam round-trips.

| | Before | After |
|---|---|---|
| `pytest tests/ --collect-only -q` | ~45s | **1.43s** |
| Prod DB connections per collection | 1 | 0 |
| Live Steam requests per collection | 5 | 0 |

Deleted. The `AGENTS.md` note recommending targeted runs on cost grounds is now obsolete and
was removed with it.

## Two `.gitignore` rules were one character away from working

Both were path-exact in a way that let a near-miss through, and both had already cost
something:

- `*.db` did not match `cs2_market.db.backup` — 9.1MB tracked.
- `backend/models/saved_models_*/` keys on the `_` separator, so `saved_models.bak/` slipped
  past — 58MB, 73 boosters, tracked since 2026-07-17.

The second is the *third* instance of this exact failure mode. The rule itself exists only
because a snapshot directory escaped the four `saved_models/*.ext` rules above it, and its
in-file comment records that commit `8783c67` committed 1.8G that way, including a 1.7GB
parquet over GitHub's 100MB cap that silently blocked every push for five days.

Added `*.db.backup` and `backend/models/saved_models.*/`, each with a comment naming what got
through and when, matching the existing convention in that file.

## Deleted

107 files. Everything below was verified to have zero importers, zero `.github/workflows/`
references and zero test references before removal.

**Tracked build artifacts (67MB)** — `models/saved_models.bak/` (73 files),
`cs2_market.db.backup`, `catboost_info/` (CatBoost was removed 2026-07-13;
`docs/architecture/model.md` already described this directory as "a stale directory remains"),
and `optuna_3d_result.json` / `optuna_horizons_result.json`, each written by its own script and
read by nothing.

**Provably broken** — `scripts/build_chart_points.py` wrote to `chart_points`, a table dropped
by migration `0012_drop_chart_points`. `scripts/export_daily_snapshot.py` claimed in its own
docstring to be "used by the aggregator GitHub Actions workflow"; that workflow has run
`append_to_parquet.py` for some time. `scripts/restore_aggregator_history.py` had zero
references anywhere in the repo, docs included.

**20 spent one-off scripts (~290KB)** — finished experiments, completed backfills and probes
for features that were subsequently refuted. Their findings live in `docs/changelog/`, which is
the durable record; git history keeps the code recoverable. Includes
`scripts/test_social_signal.py`, whose deletion removes the pytest-collection hazard
`AGENTS.md` previously documented a workaround for.

**`backend/analytics/`** — a package containing one 20-byte file, `# Analytics package`, and
nothing else.

Full suite after: **1,773 passed, 0 failed.**

## Kept, deliberately

- `api/routes/auth.py`, `portfolio.py`, `ab_test.py` — orphaned by the frontend deletion and
  untested, but the frontend is being rebuilt and `auth.py` is the only Steam OpenID
  implementation that exists.
- `collectors/export_finbert_onnx.py` — a run-once script with no importers, but it is the only
  way to regenerate the cache that `collectors/social_sentiment.py` reads, and `backend/AGENTS.md`
  keeps that collector on purpose for local authenticated runs.
- `scripts/build_market_catalog.py`, `repair_catalog_gaps.py`, `collectors/csmarketapi_backfill.py`
  — no automation references them, but `docs/references/catalog-build.md` and `backfill.md` are
  live runbooks for them.
- `backend/runtime/price_history_cache/` (2.3GB) — a download cache for
  `import_price_history_source.py`. Kept because that import is **staged and not yet promoted**
  (`2026-08-09-price-history-import-staged.md`); its 1.96M rows still sit in `archive-staging/`.
- All 119 changelog entries, including the frontend ones and their dangling links. An
  append-only record is supposed to point at things that no longer exist.

## Disk

8.1GB → ~4.4GB. `.git` alone was 2.0GB against a HEAD of a few MB: the 1.7GB
`engineered_data.parquet` from unreachable commit `8783c67` was still reachable through the
reflog. Expiring the reflog and repacking reclaimed it. The `saved_models_pre2026_20260725_182444/`
snapshot (1.8GB, superseded by later retrains) and the loose `train_*.log` files, `__pycache__`
directories and stray root `.pytest_cache` went with it. Nothing tracked or pushed was
rewritten.

## Documentation

Corrected the factual drift the audit turned up. The load-bearing ones:

- `README.md` still described a Next.js dashboard in five places, including the architecture
  diagram and a `Node 20+` requirement, three days after the frontend was deleted.
- `README.md` advertised the trailing-window voting defect as unfixed — "the fix has not
  shipped" — when `TRAILING_WINDOW_SOURCES` shipped 2026-08-09 (`backend/models/item_parser.py:67`).
- `docs/design.md` and `docs/product.md` describe the deleted frontend and carried **no marker
  saying so**; `design.md` actively told readers to reference `frontend/app/globals.css` as the
  source of truth. Both now open with a banner. Both also claimed q10/q50/q90 quantile models —
  `forecaster.py:261` is `QUANTILES = [0.5]`, with the band from split conformal.
- `docs/architecture/pipeline.md` and `docs/operations.md` called `price-archive/` a symlink to a
  checkout. It is a plain local directory that runs behind the durable archive — a
  production-safety point `AGENTS.md` makes and these two contradicted.
- `docs/research/2026-08-09-next-steps.md`, billed as the live action list, listed O4 as open in
  its status block and as done 100 lines later, and instructed readers to dispatch with a
  `FORCE_RETRAIN` flag the workflow does not expose.
- Stale counts throughout: 5 workflows → 6, 112 changelog entries → 119, 926-item cohort → 916,
  1,385 tests → 1,773. `AGENTS.md` described two of `docs/`'s five subdirectories; it now names
  all five.
