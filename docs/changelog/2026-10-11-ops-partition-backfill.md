# Ops partitions seeded from the Supabase backup; Postgres wins every shared row

**Date:** 2026-10-11

Phase 1, step 1 of `docs/specs/2026-10-11-postgres-to-parquet-history-design.md`.

## What changed

- The 2026-10-11 backup is a private release on `cs2-oracle-data`,
  `supabase-backup-2026-10-11`: 17 tables plus `_rowcounts.json` and `SHA256SUMS`. `users`
  is left out (it may hold PII, and nothing reads it). A fresh download passed
  `shasum -c` on all 19 files. This is the backup's first copy off the Mac.
- `scripts/backfill_ops_partitions.py` writes `ops/{item_forecasts,forecast_outcomes,
  forecast_candidates,forecast_candidate_outcomes}/YYYY-MM.parquet` and `ops/items.parquet`
  (ZSTD, sorted) from the backup merged with the single-file mirror. It **reads** the
  single-file mirror and never writes it, so every current reader sees what it saw before.
- `ops-partition-backfill.yml` runs it once by manual dispatch, dry run by default. It
  refuses if any target already exists.

## The merge rule, and why it is not "mirror wins"

The spec said the mirror lacked history but was otherwise a copy. Measured against the
backup, it is not:

| Disagreement | Rows | Which is the record |
|---|---|---|
| Forecast **band** differs (08-05, 08-24, 08-26, 09-08), up to 48x, mirror usually created hours later | ~28k | Postgres. Every scored outcome for the first three dates matches the Postgres band and none matches the mirror's; 09-08 splits 6,238 / 4,933. |
| Outcome `in_interval` (07-18 to 08-07) | 9,847 | Postgres. Rewritten there alone by the 2026-08-11 21:10 verdict refresh after e4b57e9 moved `in_interval` to return space; the mirror keeps the retired dollar-basis verdict. |
| `exceed_p`, `band_multiplier`, price columns | many | Either. float8 round-trip, <=4.7e-15 relative. |

The frozen instruments that read forecasts or verdicts (PID, ranking transfer, centre
promotion, direction reopen, date dependence) read Postgres. `outside_baseline` reads the
mirror but only outcome price columns, which agree on every shared row to 4.7e-15. So
**Postgres wins every shared row, whole-row**, and the mirror contributes only what Postgres
lacks: `item_slug`, `model_config`, and rows Postgres no longer has.

`outside_baseline` keeps reading the untouched single-file mirror. Its panel from the
partitions differs on 6,590 rows by float noise and gains 18,070 rows the mirror never had,
so under the spec's Phase 3 rule it switches after its ~10-25 read, not before.

## Rows Postgres deleted: kept, flagged

25,192 forecasts (08-04, 08-05) and 9,447 outcomes (07-29 and 07-31 h=7, 08-04 h=3) exist
in the mirror and not in Postgres. Every one belongs to an `item_id` that is no longer in
`items`, so they look like an item purge that cascaded. No changelog records it. They are
kept with `pg_deleted = true`: a deletion can be applied later, a dropped row cannot be
recovered. 3,149 of the deleted forecasts get their `id` back from a mirror outcome; 22,043
have none. Rows dated after the backup (10-10) are never flagged.

`item_slug` is filled from `items` on rows that only the backup had. The `pg_deleted` rows
have no slug, because their items are gone.

## Seeded counts (dry run on the 10-11 durable mirror)

| Table | Rows | Months |
|---|---|---|
| item_forecasts | 1,358,045 | 2025-12, 2026-07 through 2026-10 |
| forecast_outcomes | 964,511 | 2025-12, 2026-07 through 2026-10 |
| forecast_candidates | 974,336 | 2026-09, 2026-10 |
| forecast_candidate_outcomes | 470,560 | 2026-09, 2026-10 |
| items | 32,281 | — |

The CI run reads the mirror as of its dispatch, so its counts include any day written since.

## Not done here

- Nothing writes the partitions after the seed. Dual-write (step 2) has to start from them
  and catch up on whatever the single-file mirror gained in between.
- No release of closed months yet, and no local pull.
