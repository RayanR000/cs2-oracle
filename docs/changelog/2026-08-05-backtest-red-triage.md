# Backtest red — triage, 2026-08-05

Status: **problem 1 fixed** (2026-08-05, see "Resolution" below). Problem 2 is a
calendar problem with nothing to fix. The units bug under "Also visible" is
fixed too.

Two separate problems. The first stopped the workflow. The second means the
number the workflow produces is not yet quotable even once it runs.

---

## 1. The Parquet append kills the run (blocking)

### Timeline

| when (UTC) | trigger | failure |
|---|---|---|
| 08-03 11:21 | schedule | `100.0% of mature forecasts could not be resolved (cap 10.0%)` |
| 08-04 00:25 | workflow_run | `17,607 of 98,119 (17.9%) could not be resolved` |
| 08-04 10:37 | schedule | same |
| 08-05 00:07 | workflow_run | **success** |
| 08-05 10:32 | schedule | `Conversion Error: Could not convert string 'None' to DOUBLE` |
| 08-05 21:30 | workflow_run | `Type VARCHAR … can't be cast to … STRUCT(…)` |

The resolution/cap problem that caused the 08-03 and 08-04 failures is
**fixed** — 08-05 00:07 succeeded. A different failure then took over. Do not
read the unbroken run of red as one bug.

The 08-05 21:30 run was chained off a manual `price-forecast` dispatch, but the
10:32 scheduled run had already failed the same way, so the dispatch is not the
cause.

### Root cause

Both 08-05 failures are the same code path:

- `backend/scripts/backtest_accuracy.py:114` — `_upsert_accuracy` calls
  `append_table("prediction_accuracy", rows, [...])`
- `backend/db/parquet.py:194` → `_append_parquet`
- `backend/db/parquet.py:123` — the `COPY … UNION ALL` that raises

`rows[i]["metrics"]` is a **raw Python dict**. On disk, `prediction_accuracy.parquet`
holds `metrics` as a **STRUCT with a frozen field list**. The `UNION ALL` has to
reconcile the incoming column with that STRUCT, and it cannot when either the
field set or the value types change.

`_append_parquet` *does* have schema-drift handling — `union_cols` and `_project`
at `parquet.py:102-111` NULL out columns a side lacks. **That logic operates on
top-level columns only.** `metrics` is one column; drift *inside* its STRUCT is
invisible to it. That is the gap.

Two manifestations observed:

1. **New keys.** `d7ffaae` (2026-08-05 16:55 -0400) added four fields to
   `score_cohort` — `directional_accuracy_moved`, `directional_accuracy_unchanged`,
   `n_unchanged`, `unchanged_pct` (`backend/backtest/scoring.py:209-212`). None
   are in the on-disk STRUCT, whose field list is visible in full in the 21:30
   error text.
2. **`None` in a DOUBLE field.** Data-dependent, which is why 00:07 passed and
   10:32 failed on identical code. Candidates, all legitimately nullable:
   - `skill_vs_baseline` — `None` when `baseline_mae == 0` (`scoring.py:217`)
   - `directional_accuracy_moved` / `_unchanged` — `_dir_acc` returns `None` for
     an empty partition (`scoring.py:173-176`), which is the deliberate
     empty-partition rule, not a bug in `score_cohort`

Note the incoming value arrives as **VARCHAR holding a Python `repr`** (single
quotes in the error text), not as a struct — so pandas is stringifying the dict
on the way in. Worth confirming during the fix; it may mean the existing STRUCT
on disk was written by a different path than the one writing now.

### Consequence beyond the red X

`db.commit()` at `backtest_accuracy.py:110` runs **before** `append_table` at
`:114`. So on every one of these failures the **DB is updated and the Parquet
mirror is not**, and the run then exits 1. The API reads Parquet first with DB
fallback only on `None`/exception, so a present-but-stale mirror wins over a
current DB. Each failure widens that divergence.

### Resolution (2026-08-05)

The root cause was **worse than "new keys"**, and the triage's guess about the
VARCHAR was right for a reason worth recording: DuckDB infers the SQL type of a
pandas object column **from the batch's contents**. Measured on the real
`score_by_tier` output, the same code path produces three different types —
`STRUCT` when every dict in the batch has identical keys and value types,
`MAP(VARCHAR, DOUBLE)` when a nested dict's key set differs between rows (which
`mape_by_tier` guarantees, since a cohort carries only the tiers it has), and
`VARCHAR` holding a Python `repr` when it can reconcile neither. That
data-dependence *is* why 00:07 passed and 10:32 failed on identical code.

Chosen fix: **nested values are stored as JSON text**, as a store-wide invariant
in `db/parquet.py` rather than a `prediction_accuracy` special case. Three ops
tables carried a nested column — `prediction_accuracy.metrics`,
`collection_runs.source_breakdown`, `accuracy_alerts.details` — so all three had
the same latent bug; `source_breakdown` would have fired the day a new price
source appeared. One stable scalar type, no frozen field list, and a new metric
now costs nothing.

- `_jsonify_nested` serialises dict/list columns on the write paths
  (`_append_parquet`, and `replace_rows`, which `--reresolve` takes).
- Existing nested files are **migrated in place**, in the same single rewrite,
  via `to_json`. Deliberately not a migration script: `price-archive/` is
  gitignored, so the served mirror exists only where the job runs and no
  operator can be relied on to migrate it before the next unattended run.
  Verified against a copy of the production file — 84 rows preserved, inner
  values intact including dead legacy fields like `window_start`, and still
  queryable as `metrics->>'$.mae'`.
- `_upsert_accuracy` now appends to Parquet **before** `db.commit()`, the
  discipline `_flush_verdict_refresh` and the `--reresolve` path already
  documented. The mirror can no longer be left behind a committed DB.
- The API parses the JSON back to an object, so the HTTP contract is unchanged
  and the Parquet and DB-fallback legs still agree.

**One further bug found while verifying, not in the original triage.** Appending
the same batch twice grew the file 91 → 92: the dedup anti-join compared keys
with `=`, and `NULL = NULL` is NULL rather than TRUE, so any row with a NULL in
a dedup key never matched its replacement. That is the **all-tiers aggregate**
row (`price_tier IS NULL`) — the row every accuracy endpoint serves by default —
which would have gained a duplicate per run, leaving "the latest row" arbitrary
among them. It was latent only because `price_tier` was absent from the file
entirely, which dropped it from `usable_keys`; this fix is what would have
exposed it. Now `IS NOT DISTINCT FROM`. The DB side was never affected
(`filter_by(price_tier=None)` emits `IS NULL`), so this was one more way for the
two stores to disagree.

Regression tests: `backend/tests/test_parquet_nested_columns.py` (includes the
two cases this doc asked for — a key absent from the existing file, and a `None`
in a numeric field — plus heterogeneous nested key sets, in-place migration, and
the NULL-key dedup), `backend/tests/test_accuracy_mirror_write_order.py`.

---

## 2. The headline is unquotable even when the run succeeds

From the 21:30 log, every horizon:

```
[3d / lgbm-v3] >=$1: NO HEADLINE — 2,092 samples span only 2 forecast date(s),
below the 20 required.
```

`MIN_FORECAST_DATES = 20` (`backend/backtest/scoring.py:38`). Current cohorts
span **1–2 distinct forecast dates**. Directional outcomes are clustered by
forecast date — every item forecast on the same day rides the same market move —
so these cohorts cannot separate model skill from which way the market went on
those two days. The gate is correct to refuse them.

The figures still printed, and still current:

| horizon | ≥$1 DirAcc | dates | status |
|---|---|---|---|
| 3d | 45.1% | 2 | NO HEADLINE |
| 7d | 49.2% | 2 | NO HEADLINE |
| 14d | 45.2% | 2 | NO HEADLINE |
| 30d | 38.7% | 1 | NO HEADLINE |

### Why this matters for the cohort-parity work

`docs/specs/2026-08-05-cv-cohort-parity-design.md` reports a residual
CV-vs-production gap of 3.6–19.1pp, widening with horizon. The **CV side of that
comparison is sound** — out-of-fold, and confirmed by retrain (run
`31048504009`). The **production side is these NO HEADLINE numbers.** So the
residual gap, and particularly its horizon slope, is not yet established. It
rests on 1–2 market days.

This is a calendar problem, not a code problem: it needs ~20 distinct forecast
dates to accumulate. Nothing to fix, but the gap figure should not be quoted as
settled until then, and the spec's table should be read with that caveat.

---

## Also visible, unrelated to the above

The `directional_accuracy_ci_*` units bug already listed as a non-goal in the
cohort-parity spec is plainly visible in the 21:30 error dump — the same dict
carries `'directional_accuracy': 49.57` (a percent) next to
`'directional_accuracy_ci_lower': 0.4854` (a fraction).

**Fixed 2026-08-05.** All four directional bounds are now percent, rescaled in
`score_cohort` at the one place the fractions are produced. `_score_groups` had
been compensating with a `* 100` in the log line only, so the console was right
and the *stored* row — the one that gets audited — was not; that `* 100` is gone.
`mae_ci_*` is deliberately left in dollars, the same units as `mae`. Rows written
before this date hold the same numbers as fractions. No frontend or production
Python consumer reads these fields, so nothing downstream needed changing.
Pinned by `backend/tests/test_directional_ci_units.py`, whose central test is
that the interval brackets its own point estimate.

Failure notification **is working** — verified, not assumed. Both 08-05 failures
filed issues automatically, labelled `automation`:

- #4 — `[Backtest Accuracy] Failed - 2026-08-05`, 21:31 (the run analysed here)
- #3 — `[Backtest Accuracy] Failed - 2026-08-05`, 10:35

These are the only two issues in the repo, so the earlier 08-03 / 08-04 failures
did *not* notify; the notify fix appears to have landed between 08-04 and 08-05.
Any older note claiming notifications are swallowed is out of date.
