# Closed ops months are released once, and checked every night

**Date:** 2026-10-12

The second and third copies from `docs/specs/2026-10-11-postgres-to-parquet-history-design.md`.

## What changed

- `scripts/ops_releases.py publish` is the last step of Backtest Accuracy, after the archive
  push. Each month under `ops/<table>/YYYY-MM.parquet` closes at **calendar end + 43 days**
  (h=30 plus the 13-day embargo carry; September 2026 closes 11-12).
  - **A closed month with no release:** it creates `ops-YYYY-MM` on `cs2-oracle-data`,
    holding `<table>-YYYY-MM.parquet` for each table plus `SHA256SUMS`.
  - **A closed month with a release:** it compares the tree with that release's
    `SHA256SUMS`. A file rewritten, added or removed fails the run. It never replaces an
    asset.
- `scripts/ops_releases.py pull --dest ~/personal-projects/cs2-analyzer/ops-releases` runs on
  the Mac. It downloads every `ops-*` release it lacks and re-verifies every one it holds.
  Run it by hand for now.

## Why calendar end

The spec first counted from a month's last forecast date, which put August's close at 10-09.
A row backfilled later into 08-28..08-31 would then land in a month already released and
fail every run after it. Calendar end costs a few days and removes that case.

## Still to verify

- **The token:** that `CS2_DATA_REPO_TOKEN` can create releases. The first night with a
  closed partition answers this. If it can't, the step fails loudly and the archive push
  before it is unaffected.

## Immutable releases: on

Immutable releases were enabled on `cs2-oracle-data` on 2026-10-12. The API now returns
`{"enabled":true}`. A release published from then on cannot have its assets edited or
deleted, so the backup no longer depends on the token behaving. `supabase-backup-2026-10-11`
was published before the switch and stays mutable.
