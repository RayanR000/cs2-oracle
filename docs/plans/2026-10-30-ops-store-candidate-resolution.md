# Phase 2 PR 2: candidate resolution reads and writes the store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shadow candidate outcomes resolve on a read-only Postgres. The plan step reads
candidates, their outcomes, production outcomes and the slug map from the ops partitions. The
write step lands the rows in Postgres when it can, and straight in the partitions when it cannot.

**Architecture:** `backtest/candidate_resolution.py` splits into a pure
`plan_candidate_outcomes(read, ...)`, where `read` is a `sql -> rows` reader, and a write step.
The ORM read-modify-write goes away. Same reader idiom as PR 1 and PR 4 (`ops_store.reader`,
`ops_store.pg_reader`, `ops_store.compare`).

**Tech Stack:** Python, DuckDB, SQLAlchemy, pytest (+ xdist).

**Spec:** `docs/specs/2026-10-13-ops-readers-switch-design.md` § PR 2 and its 2026-10-30
amendment, § Errors, § Tests.

**Base:** stacked on `feat/ops-store-feedback-blend` (#125), which adds `ops_store.pg_reader`.
Open the PR against that branch; it retargets to `main` when #125 merges.

## Global Constraints

- Frozen means frozen: a candidate that has an outcome in the partitions is skipped unless
  `reresolve` is set (module docstring).
- A production/candidate leg mismatch raises `RuntimeError`, and the backtest keeps letting
  that propagate (`_resolve_candidates`).
- A candidate with an id ≥ `ops_partitions.READ_ONLY_ID_BASE` is never written to Postgres,
  because `forecast_candidate_outcomes.candidate_id` is a foreign key.
- On a read-only (25006) or out-of-disk (53100) rejection, roll back, call `report_read_only`,
  and write the partitions. Never skip the write silently.
- Integers and dates are inlined into SQL only after `int()` / `date.isoformat()`.
- Tests are hermetic (`_hermetic_ops_dir`, in-memory SQLite). Gate:
  `cd backend && pytest tests -m "not slow" -q -n 4`.
- No Claude/Codex attribution.

## Review Focus

1. **A read-only night with mixed ids.** Ordinary and partition-only candidates mature in the
   same run. All rows must land in the partitions, and none may be inserted into Postgres →
   Task 3 `test_read_only_lands_every_row_in_the_partitions`.
2. **Postgres writable, partition-only candidates present** (a night after read-only ends).
   Ordinary ids go to Postgres, then are read back into the partitions; high ids go straight to
   the partitions → Task 3 `test_partition_only_candidates_skip_postgres`.
3. **DiskFull rather than 25006** → Task 1 tests and Task 3
   `test_disk_full_is_treated_as_unwritable`.
4. **`pg_deleted` candidate or production outcome** stays invisible to the plan → Task 2
   `test_plan_ignores_pg_deleted_rows`.
5. **Re-running the same night** must not duplicate rows or move frozen legs → Task 3
   `test_rerun_is_idempotent` (partitions key on `candidate_id`).

---

### Task 1: one test for "Postgres cannot take writes"

**Files:** Modify `backend/db/read_only.py`, `backend/scripts/ops_parity.py:208-220`. Test
`backend/tests/test_read_only_detection.py` (new, or extend the existing read-only test module if
`grep -l is_read_only_error tests/` finds one).

**Interfaces — produces:**
- `DISK_FULL_SQLSTATE = "53100"`
- `is_disk_full_error(exc) -> bool`: walks the cause chain like `is_read_only_error`. Matches
  pgcode 53100 or `"No space left on device"`.
- `is_unwritable_error(exc) -> bool`: `is_read_only_error(exc) or is_disk_full_error(exc)`.

- [ ] Step 1: tests:
  - `test_disk_full_pgcode_matches`: an exception whose `.orig.pgcode == "53100"`.
  - `test_disk_full_message_matches_through_the_cause_chain`: an `OperationalError` wrapping
    `"No space left on device"`.
  - `test_unrelated_error_is_writable`.
  - `test_read_only_is_unwritable`.
- [ ] Step 2: run → FAIL (import error).
- [ ] Step 3: implement. `ops_parity._is_disk_full` becomes `read_only.is_disk_full_error` and
  its local constant is deleted.
- [ ] Step 4: `pytest tests/test_read_only_detection.py tests/test_ops_parity*.py -q` → PASS.
- [ ] Step 5: commit `feat: one check for a Postgres that cannot take writes (read-only or out of disk)`.

### Task 2: the plan reads the store (`plan_candidate_outcomes`)

**Files:** Modify `backend/backtest/candidate_resolution.py`. Tests
`backend/tests/test_candidate_resolution.py` and `test_candidate_outcome_freeze.py`: seed SQLite
as today, call `tests._ops_mirror.mirror_session_to_ops(session)` before resolving. New test
`backend/tests/test_candidate_resolution_store.py`.

**Interfaces — produces:**
- `plan_candidate_outcomes(read, *, today, archive_dir, reresolve=False) -> tuple[list[dict], dict]`
  - Each row dict holds the partition columns `candidate_id, base_price, actual_price,
    resolved_at, resolution_version, absolute_error, percentage_error, in_interval`, plus
    `_part` (the candidate's forecast_date).
  - `resolved_at` is the production outcome's `resolved_at` when the legs came from production,
    otherwise `None`. The write step stamps `None` with the write time. Keeping the plan free
    of a clock is what makes the dry-run proof comparable.
  - `stats` keeps today's keys (`resolved`, `reresolved`, `skipped_frozen`, `skipped_immature`,
    `unresolvable`).
- `CANDIDATE_TABLES = ("forecast_candidates", "forecast_candidate_outcomes", "forecast_outcomes", "items")`
- Reads are four SQL strings over those views:
  - candidates: all columns `_derived_metrics` needs, plus `id, item_id, forecast_date, horizon_days`.
  - existing outcomes: `candidate_id`.
  - production legs: `item_id, forecast_date, horizon_days, base_price, actual_price, resolved_at`
    where `item_id IN (...)`, chunked by `CHUNK`.
  - slug map: `id, item_id FROM items`.

- [ ] Step 1: tests in `test_candidate_resolution_store.py`:
  - `test_plan_is_identical_from_both_stores`: seed a shared-production candidate and a ranking
    candidate in SQLite, mirror them, then compare
    `plan(ops_store.reader(CANDIDATE_TABLES), ...)` with `plan(ops_store.pg_reader(session), ...)`.
    Rows are compared after sorting on `candidate_id`, with dates normalised as in PR 4's
    `_sorted`; stats are compared too.
  - `test_plan_resolves_partition_only_candidates`: a candidate with
    `id = READ_ONLY_ID_BASE + 3`, written only to the partitions, with a production outcome in
    the partitions → one row with `candidate_id` equal to that id and the production legs.
  - `test_plan_skips_candidates_frozen_in_the_partitions`: an outcome present only in the
    partitions → `skipped_frozen == 1`, no row.
  - `test_plan_ignores_pg_deleted_rows`: a `pg_deleted` production outcome → the plan does not
    use its legs (falls to the archive stub or counts as unresolvable).
  - `test_plan_mismatch_raises`: production legs in the partitions that differ from the stubbed
    archive under `reresolve=True` → `RuntimeError`.
- [ ] Step 2: run → FAIL (no `plan_candidate_outcomes`).
- [ ] Step 3: implement. Port the body of `resolve_candidate_outcomes` with every
  `db.query(...)` replaced by a `read(...)` on the strings above, and `_create_or_update`
  replaced by appending a row dict (`_planned_row(candidate, base, actual, resolved_at)`). The
  archive branch and the mismatch check stay as they are.
- [ ] Step 4: run `test_candidate_resolution_store.py` → PASS. The old tests still fail until
  Task 3 wires the write; that is expected.
- [ ] Step 5: commit `feat: candidate resolution plans from the ops partitions`.

### Task 3: the write step and the read-only path

**Files:** Modify `backend/backtest/candidate_resolution.py`
(`resolve_candidate_outcomes`, `_rescore_candidate_outcomes`, new `_write_candidate_outcomes`;
`_create_or_update` deleted). Tests: `test_candidate_resolution_store.py`, the two existing
modules, `test_ops_dual_write_candidate_outcomes.py`.

**Interfaces:**
- Consumes `plan_candidate_outcomes` and `CANDIDATE_TABLES` (Task 2), and
  `read_only.is_unwritable_error` (Task 1).
- Produces `_write_candidate_outcomes(db, rows: list[dict]) -> str`, returning `"postgres"` or
  `"partitions"`:
  1. Stamp `resolved_at = utcnow_naive()` where it is `None`.
  2. Rows with `candidate_id < READ_ONLY_ID_BASE`: upsert into Postgres. Query the existing
     `ForecastCandidateOutcome` rows by `candidate_id` in `CHUNK`s, update or add, commit.
     Then `_candidate_outcome_partitions(db, ids)` (read back; Postgres wins).
  3. Rows with higher ids: `ops_partitions.upsert("forecast_candidate_outcomes", frame)` directly.
  4. On `is_unwritable_error`: `db.rollback()`, call `report_read_only(f"{n:,} candidate outcome writes")`,
     upsert every row directly, and return `"partitions"`.
- `resolve_candidate_outcomes(db, *, today, archive_dir=None, reresolve=False, rescore=False) -> dict`
  keeps its signature and return value. It plans from `ops_store.reader(CANDIDATE_TABLES)`,
  writes, then runs read twice:
  `ops_store.compare("candidate outcomes per forecast date", <Postgres count per candidate forecast_date>, <partitions count per date>, db=db)`.
  Both counts come from SQL joining outcomes to candidates, grouped by `forecast_date`, keyed
  on the ISO date.
- `_rescore_candidate_outcomes(db)` reads outcomes joined to candidates from the store. It
  recomputes `_derived_metrics`, keeps the rows whose derived values changed, writes them
  through `_write_candidate_outcomes`, and returns `{"rescored": n}`.

- [ ] Step 1: tests in `test_candidate_resolution_store.py`. For the read-only tests, a session
  whose `commit` raises an error carrying pgcode `25006` (or `53100`):
  - `test_read_only_lands_every_row_in_the_partitions`: one ordinary and one partition-only
    candidate, both mature → both `candidate_id`s are in the partitions, Postgres holds 0
    candidate outcomes, and stdout contains `"::warning title=Postgres read-only::"`.
  - `test_disk_full_is_treated_as_unwritable`: the same with 53100.
  - `test_partition_only_candidates_skip_postgres`: a writable session → Postgres holds only the
    ordinary candidate's outcome; the partitions hold both.
  - `test_rerun_is_idempotent`: resolve twice → each `candidate_id` appears once in the
    partitions, and the second run reports `skipped_frozen == 2`.
  - `test_read_twice_logs_outcomes_per_date`: `caplog` contains
    `"read twice candidate outcomes per forecast date"`.
  - `test_rescore_reads_the_partitions`: a candidate outcome only in the partitions, with a
    stale `absolute_error` → `rescore=True` fixes it in the partitions.
- [ ] Step 2: run → FAIL.
- [ ] Step 3: implement as above.
- [ ] Step 4: update the existing tests.
  - `test_resolution_writes_partitions_for_touched_candidates` keeps its assertion, because the
    Postgres path still calls `_candidate_outcome_partitions(db, ids)`.
  - Freeze, reresolve and rescore tests seed, mirror, then assert on Postgres for ordinary ids,
    as they do today.
  - Run `pytest tests/test_candidate_resolution*.py tests/test_candidate_outcome_freeze.py tests/test_ops_dual_write_candidate_outcomes.py tests/test_candidate_scoring.py -q` → PASS.
- [ ] Step 5: commit `feat: candidate outcomes land in the partitions when Postgres cannot take writes`.

### Task 4: docs, gate, dry-run proof, PR

- [ ] Step 1: write `docs/changelog/2026-10-30-candidate-resolution-reads-ops-partitions.md`.
  Mark PR 2 **Shipped** in the spec. In `.claude/rules/archive-reads.md`, add candidate
  resolution to the list of switched readers.
- [ ] Step 2: run the gate (`pytest ... -n 4`, `ruff check`, `ruff format --check` on the
  changed files) and record the counts.
- [ ] Step 3: write the dry-run proof script in the plan workspace, not committed. It runs
  `plan_candidate_outcomes` with `today = 2026-10-28` and `reresolve=False` from both stores:
  - **Partitions:** `cs2-oracle-data` origin/main, extracted.
  - **Postgres:** the four tables pulled as plain filtered scans into DuckDB, as in PR 4's
    proof, because Postgres cannot spill a join.
  - On both sides, candidate outcomes and production outcomes with `resolved_at >= 2026-10-28`
    are hidden, so that night is recomputed.
  - Price archive: the same extract's `price-archive`.
  - It prints the row count, a per-date count, the number of differing rows and the stats per
    side. Expected: 0 differing rows and equal stats.
  - The user runs it with `!` from `cs2-oracle/backend`.
- [ ] Step 4: push, then `gh pr create --base feat/ops-store-feedback-blend` with the proof in
  the description. Report and stop.
