# Candidate resolution reads and writes the ops partitions

**Date:** 2026-10-30
**PR:** Phase 2, PR 2 of 6 — `docs/specs/2026-10-13-ops-readers-switch-design.md` (amended 2026-10-30)

Shadow candidate outcomes now resolve on a read-only Postgres. Since the 10-30 forecast run,
the shadow candidates land only in the partitions, with ids from `READ_ONLY_ID_BASE`. Candidate
resolution still read and wrote Postgres through the ORM, so it resolved nothing, and its first
insert failed every night. The backtest swallowed that failure.

How it works now:

- **Plan.** `plan_candidate_outcomes(read, ...)` reads candidates, their outcomes, the
  production outcomes it pairs against and the slug map through a `sql -> rows` reader. It
  returns the rows to write and the counts. It writes nothing and reads no clock: a row
  resolved from the archive carries `resolved_at = None`, stamped at write time. The
  frozen-outcome, maturity and production/candidate mismatch rules are unchanged.
- **Write.** `_write_candidate_outcomes` upserts candidates with ordinary ids into Postgres,
  then reads them back into the partitions as before (Postgres wins). Partition-only candidates
  go straight to the partitions, because Postgres's `candidate_id` is a foreign key. On a
  read-only (25006) or out-of-disk (53100) rejection it rolls back, raises the
  `Postgres read-only` warning, and writes every row straight to the partitions.
- **`--rescore`** reads frozen outcomes from the partitions and writes only the rows whose
  derived metrics changed, through the same write.
- **Read twice:** candidate outcomes per candidate forecast date, from both stores.

`db/read_only.py` gains `is_disk_full_error` and `is_unwritable_error`. `scripts/ops_parity.py`
now uses the shared check instead of its own.

**Why it matters:** the champion–challenger read needs 20 shared dates, and it reads these
outcomes. The 10-30 and later candidates were not lost. They resolve on the first night this
runs, from production legs in the partitions or from the archive.

Not changed: the scoring reports, the freshness check (PR 3), the catalog (PR 5), the API (PR 6).
