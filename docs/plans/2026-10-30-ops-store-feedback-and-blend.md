# Phase 2 PR 4: feedback factors and prior-day blend read the store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The served-coverage feedback panel, the prior-day forecast blend and the (off-by-default)
tier bias fit read the monthly ops partitions through `db/ops_store.py`, with Postgres kept only
as the read-twice side.

**Architecture:** Each reader's SQL becomes a plain string run through a `sql -> rows` reader, so
the same query runs on `ops_store.reader(...)` (the value used) and on Postgres (the read-twice
value, via `ops_store.compare`). Same idiom as PR 1 (`scripts/backtest_accuracy.py`
`_store_read` / `_pg_read` / `_read_twice_matured`).

**Tech Stack:** Python, DuckDB, SQLAlchemy, pytest (+ xdist).

**Spec:** `docs/specs/2026-10-13-ops-readers-switch-design.md` (§ PR 4, § Errors, § Tests, § Docs).

## Why now, and why before PR 2

Postgres went read-only after the 10-30 02:36 UTC forecast write (forecast date 10-29 is in both
stores; partitions' 10-29 ids are ordinary Postgres ids). From the **11-01 02:30 UTC** run the
prior-day blend reads a Postgres whose newest forecast is a day behind, and the feedback panel
already lacks every outcome scored since read-only. Both change served numbers. Merge before
10-31 evening ET. The spec's "remaining PRs go in order without the 3-night wait" applies; the
order change (4 before 2, 3) was chosen by the user on 2026-10-30.

## Global Constraints

- No reader falls back to Postgres silently once switched (spec § Errors).
- A read-twice difference never fails a run (`ops_store.compare` never raises).
- Feedback factors keep today's behaviour on a failed panel read: `{}` → factor 1.0, with the
  store and the error named in the log.
- Integers and ISO dates are inlined into SQL only after `int(...)` / `date.fromisoformat(...)`;
  DuckDB has no SQLAlchemy expanding bind.
- Tests never touch prod: conftest `_hermetic_ops_dir`, in-memory SQLite, the `ops_mirror`
  fixture (`tests/conftest.py`) for dual-store tests.
- Gate before every push: `cd backend && pytest tests -m "not slow" -q -n 4`.
- No Claude/Codex attribution in commits or the PR.

## Decision for review: the blend's missing-store behaviour

Spec § Errors says a switched reader fails loudly on a missing table. For the blend, a crash
means **no forecasts that night**, while today's failed prior read warns and skips the blend.
This plan keeps the skip but makes it loud: on `ops_store.MissingTable` the blend logs an error
and prints `::error title=Prior-day blend::...` naming `OPS_DIR`, then skips. Every other
exception keeps today's warning. Override in review if you want a hard failure.

## Review Focus

1. **Forecasts written on a read-only night** (ids ≥ `READ_ONLY_ID_BASE`) must be blended from
   and joined to their outcomes like any other row → Task 2 test
   `test_prior_fetch_uses_partition_only_forecasts`, Task 1 test
   `test_panel_joins_partition_only_forecast_multiplier`.
2. **A `pg_deleted` forecast or outcome** must not enter the blend or the panel → Task 1 and
   Task 2 each carry a `pg_deleted` test.
3. **Replay anchor:** `_now()` in the past must not blend forecasts dated on/after the anchor
   from the store → Task 2 `test_prior_fetch_respects_the_replay_anchor`.
4. **Postgres dead (25006 / DiskFull) during read twice** must leave the served value untouched
   and roll the session back → covered by `compare`'s existing contract; Task 1 test
   `test_factors_survive_a_failing_postgres_read`.
5. **Empty partitions in local/test runs** must not change factor behaviour (`{}`) → Task 1
   `test_missing_store_returns_no_factor_and_names_the_store`.

---

### Task 1: feedback panel reads the store (`served_recalibration`)

**Files:**
- Modify: `backend/db/ops_store.py` (add `pg_reader`)
- Modify: `backend/models/served_recalibration.py:243-345` (`_load_panel`, `served_coverage_factors`)
- Modify: `backend/tests/test_served_recalibration_panel_schema.py`,
  `backend/tests/test_served_recalibration_wiring.py`
- Test: `backend/tests/test_served_recalibration_store.py` (new)

**Interfaces:**
- Produces: `ops_store.pg_reader(session) -> Callable[[str], list]`
  (`lambda sql: session.execute(text(sql)).fetchall()`); used by Tasks 2–3.
- Produces: `_load_panel(read: Callable[[str], list], horizons, *, since: str | None = None) -> pd.DataFrame`
  — first parameter is now a reader, not a session.
- Unchanged signature: `served_coverage_factors(session, horizons, *, min_dates, alpha, since)`.

- [ ] **Step 1: Failing tests** in `test_served_recalibration_store.py`, using the `session`
  fixture pattern of `test_served_recalibration_panel_schema.py` plus `ops_mirror`:
  - `test_store_panel_equals_postgres_panel`: seed 3 forecasts + outcomes (mixed multipliers,
    one NULL pre-activation); `pd.testing.assert_frame_equal(_load_panel(store, [7]), _load_panel(ops_store.pg_reader(session), [7]))`.
  - `test_panel_joins_partition_only_forecast_multiplier`: write a forecast with
    `id = READ_ONLY_ID_BASE + 1`, `band_multiplier=0.5385` and its outcome only via
    `ops_partitions.upsert` (no mirror); panel row has `band_multiplier == approx(0.5385)`.
  - `test_panel_hides_pg_deleted_outcomes`: an outcome with `pg_deleted=True` is absent.
  - `test_missing_store_returns_no_factor_and_names_the_store`: empty `OPS_DIR`,
    `served_coverage_factors(session, [7], since="2026-08-01") == {}`, and `caplog` contains
    `"ops partitions"` and `"MissingTable"`.
  - `test_factors_survive_a_failing_postgres_read`: seed enough dates to activate a factor
    (`min_dates=1`), patch `ops_store.pg_reader` to raise; the factor equals the one computed
    with Postgres working, and `session.rollback` was called.
  - `test_read_twice_logs_factor_per_horizon`: `caplog` has `"read twice served-coverage factor per horizon"`.
- [ ] **Step 2:** `pytest tests/test_served_recalibration_store.py -q` → FAIL.
- [ ] **Step 3: Implement.**
  - `_load_panel(read, ...)`: build the existing SQL as a string; `IN (3, 7)` from
    `int(h)`; `since` inlined as `'YYYY-MM-DD'` after `date.fromisoformat`. Drop the
    `inspect(...).get_columns` probe: the view always projects `band_multiplier` (NULL when a
    month lacks it), and prod Postgres has migration 0028. Rows → `pd.DataFrame(rows, columns=cols)`.
    Rewrite the docstring: it reads whatever reader it is given; the "stale/selected" remark was
    about the retired single-file mirror, not the partitions.
  - `served_coverage_factors`: store reader =
    `ops_store.reader(("forecast_outcomes", "item_forecasts", "items"))`; on exception log
    `served-coverage panel read failed from the ops partitions ({type(e).__name__}: {e}); no q_hat correction applied.`
    and return `{}`. After computing `factors`, call
    `ops_store.compare("served-coverage factor per horizon", lambda: factors_from_panel(_load_panel(ops_store.pg_reader(session), horizons, since=since), horizons, min_dates=min_dates, alpha=alpha, since=since), factors, tol=1e-9, db=session)`.
    Keys of both dicts are horizons; a horizon present in one only is a difference (by design).
- [ ] **Step 4:** Update existing tests: panel-schema tests use `ops_mirror` and call
  `_load_panel(ops_store.reader(), [7])`; the "column missing" probe test is deleted (the probe
  is gone). `test_default_since_floors_the_panel_to_the_latest_cutover`'s `_capture` takes
  `(read, horizons, *, since=None)`. Run `pytest tests/test_served_recalibration*.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: the served-coverage feedback panel reads the ops partitions`.

### Task 2: prior-day blend reads the store (`forecaster._fetch_prior_forecasts`)

**Files:**
- Modify: `backend/models/forecaster.py:9446-9540` (`_fetch_prior_forecasts`; delete
  `_item_forecasts_has_band_multiplier`, its only caller goes)
- Modify: `backend/tests/test_band_multiplier_persistence.py`, `backend/tests/test_forecaster.py`
  (`TestForecastBlending` prior-fetch tests)
- Test: `backend/tests/test_prior_blend_store.py` (new)

**Interfaces:**
- Consumes: `ops_store.pg_reader` (Task 1), `ops_store.reader`, `ops_store.compare`,
  `ops_store.MissingTable`.
- Unchanged: `_fetch_prior_forecasts(self, item_ids, horizon) -> dict[str, np.ndarray]`.
  The slug→id map (`serve_universe.served_items(self.db, ...)`) stays on Postgres until PR 5.

- [ ] **Step 1: Failing tests** in `test_prior_blend_store.py` (forecaster from
  `ItemForecaster.__new__`, `db` = the SQLite `session`, `_now` fixed to 2026-09-25, items 1/2
  slugs "a"/"b" in SQLite, forecasts written via `ops_partitions.upsert`):
  - `test_prior_fetch_reads_the_partitions`: forecasts only in partitions → `mask` true,
    `mid_ret` from the partition row (current 10, mid 12 → 20.0).
  - `test_prior_fetch_uses_partition_only_forecasts`: the latest row has
    `id = READ_ONLY_ID_BASE + 5`; it wins over an older ordinary-id row.
  - `test_prior_fetch_hides_pg_deleted`: the latest row is `pg_deleted=True` → the older row is used.
  - `test_prior_fetch_respects_the_replay_anchor`: a row dated 2026-09-25 (== anchor) is ignored.
  - `test_missing_store_skips_the_blend_loudly`: empty `OPS_DIR` → `mask` all false and
    `capsys` stdout contains `"::error title=Prior-day blend::"`.
  - `test_read_twice_counts_rows_and_sums_price_mid`: `caplog` has
    `"read twice prior-day forecasts h=3"` with 0 differences when SQLite holds the same rows (`ops_mirror`).
- [ ] **Step 2:** `pytest tests/test_prior_blend_store.py -q` → FAIL.
- [ ] **Step 3: Implement.** The query (`item_id, price_low, price_mid, price_high,
  current_price, forecast_date, band_multiplier ... WHERE horizon_days = {int(horizon)} AND forecast_date < '{today.isoformat()}'`)
  runs on `ops_store.reader(("item_forecasts",))`. `except ops_store.MissingTable` → `logger.error`
  + `print("::error title=Prior-day blend::{e}; blend skipped", flush=True)` → return `result`;
  other exceptions keep today's warning. Read twice:
  `ops_store.compare(f"prior-day forecasts h={horizon}", lambda: {...aggregate from pg...}, {"rows": len(rows), "sum price_mid": sum of non-null price_mid}, tol=0.01, db=self.db)`
  where the Postgres side runs `SELECT count(*) AS rows, sum(price_mid) AS s ... ` with the same
  WHERE (an aggregate, so Postgres is not asked for ~700k rows twice). `tol=0.01` absorbs float
  summation order on a sum near 1e7 and still flags any row-count difference.
- [ ] **Step 4:** Rewrite the existing MagicMock prior-fetch tests to seed the partitions instead
  of `db.execute.side_effect` (the slug map still comes from `db.execute`); delete
  `test_prior_fetch_selects_a_null_placeholder_when_the_column_is_missing` (no probe; a month
  lacking the column reads NULL, covered by `test_ops_store`). Run
  `pytest tests/test_prior_blend_store.py tests/test_band_multiplier_persistence.py tests/test_forecaster.py -q -k "prior or blend"` → PASS.
- [ ] **Step 5: Commit** `feat: the prior-day blend reads the ops partitions`.

### Task 3: tier bias fit reads the store (`update_bias_corrections_from_outcomes`)

**Files:**
- Modify: `backend/models/forecaster.py:2010-2050`
- Test: `backend/tests/test_bias_corrections_store.py` (new)

**Interfaces:** consumes `ops_store.reader`, `ops_store.pg_reader`. No read twice (spec: off by
default since #113).

- [ ] **Step 1: Failing test** `test_bias_fit_is_identical_from_both_stores`: seed ≥ 100
  `lgbm-v3` outcomes over enough dates for `_has_date_coverage` in SQLite with `ops_mirror`;
  run the fit once as switched and once with `ops_store.reader` patched to
  `ops_store.pg_reader(session)`; the resulting threshold attributes are equal.
  Plus `test_bias_fit_reads_partition_only_outcomes`: rows only in partitions still produce thresholds.
- [ ] **Step 2:** run → FAIL (the partition-only test).
- [ ] **Step 3: Implement:** same SQL string on `ops_store.reader(("forecast_outcomes",))`; on
  failure keep today's `logger.warning("  Could not query outcomes for bias update: ...")`
  but name the ops partitions. Update the comment block's "frozen outcomes" note only where it says Postgres.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: Commit** `feat: the tier bias fit reads the ops partitions`.

### Task 4: docs, gate, proof, PR

**Files:**
- Create: `docs/changelog/2026-10-30-feedback-and-blend-read-ops-partitions.md`
- Modify: `.claude/rules/archive-reads.md:44-56` (switched readers: backtest, feedback panel,
  prior-day blend, tier bias fit); `docs/specs/2026-10-13-ops-readers-switch-design.md` (PR 4
  **Shipped** line; note the 4-before-2 order and the blend's loud-skip decision);
  `backend/models/served_recalibration.py` module docstring if it names Postgres as the source.

- [ ] **Step 1:** Write the docs above.
- [ ] **Step 2: Gate:** `cd backend && ruff check . && ruff format --check . && pytest tests -m "not slow" -q -n 4` → all pass; record the count.
- [ ] **Step 3: Proof script** (scratchpad, not committed): for a run date of 2026-10-30, print
  per horizon the factor and the prior-day `rows` / `sum price_mid` from both stores, with the
  partitions read from a fresh `origin/main` of cs2-oracle-data and both panels restricted to
  outcomes with `resolved_at < '2026-10-29'` (the last night Postgres took writes). The user runs
  it with `!` (prod reads are blocked for the agent). Expected: equal to 1e-9 / exactly equal.
- [ ] **Step 4:** Commit docs; push; `gh pr create` with the proof table in the description.
  Report the PR and stop (do not wait on CI).
