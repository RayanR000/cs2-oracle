# Data organisation — next steps (2026-08-07)

Follow-on from `docs/changelog/2026-08-07-archive-schema-and-keys.md`, which
landed items 1 and 2 of a six-item review. Items 3–6 remain, plus two things the
landed work created or exposed.

## Blocking

### 1. Ship items 1 and 2 to production

Everything so far is on the local `price-archive/`, an unlinked working copy.
Prod still has the four-column glob read and un-joinable ops tables.

1. Commit. Two files had uncommitted edits before this work and need
   `git add -p` to split: `backend/models/forecaster.py`,
   `backend/scripts/forecast_prices.py`. Everything else is cleanly separable.
2. Push. The branch is 12 commits ahead of `origin/main`, 0 behind. A
   `workflow_dispatch` runs the workflow file **at the ref you pick**, so the
   new inputs do not exist until the branch is pushed — and CI-affecting fixes
   have twice sat unpushed here while runs failed on the old SHA.
3. Dispatch *Aggregator Market Update* with `normalize_schema = true` and
   `backfill_ops_slug = true`. Both idempotent, both ordered before the daily
   append. Leave them off afterwards.
4. Verify on the data repo, not locally: `SELECT *` over `prices-*.parquet`
   returns five columns, and `ops/forecast_outcomes.parquet` has `item_slug`.

The code is safe to ship ahead of the data migration — `prices_relation` reads a
migrated and an unmigrated archive identically — so a partial rollout is fine.

### 2. Decide what the 31,422 orphaned outcomes mean

`ops/forecast_outcomes.parquet` has 31,422 of 104,642 rows whose `item_id` has
no row in `items` (30%). They keep a NULL slug after the backfill.

This is pre-existing and was surfaced, not caused, by the slug work. It matters
because **those rows are in the scored cohort** — the published directional
accuracy is computed partly over forecasts whose item cannot be identified.

Answer first, act second:
- Are they the phantom-item purge (`scripts/purge_phantom_items.py`,
  `slug-keyed-items-are-duplicates`)? If so the outcomes are probably valid and
  only the mapping is gone.
- Do they skew the headline? Compare accuracy on the mapped vs unmapped halves.
  If the two agree, this is bookkeeping. If they diverge, it is a scoring bug.

Cheap to answer now that the slug column exists: `WHERE item_slug IS NULL`.

## Remaining review items

| # | Action | Effort | Buys |
|---|---|---|---|
| 3 | Move `price-archive/raw/` (605 MB, 84 JSON files) out of the archive; delete `exchange-rates-2026.parquet`, `player-counts/`, `2026/` | 15 min | −605 MB and three fewer ambiguities |
| 4 | Route the remaining ~25 scripts through `db/archive.py` instead of their own `ARCHIVE_DIR` | ~1h | One definition; kills two CWD-relative defaults that resolve differently depending on where they are run |
| 5 | Publish the three local-only ingests (`event-calendar`, `exchange-rates-history`, `item-metadata-bymykel`) or move them to `price-archive/derived/` | ~1h | The local/prod boundary becomes visible in `ls` |
| 6 | Document rarity precedence across `items`, `item-metadata.parquet`, `item-metadata-bymykel.parquet`; refresh the `data.md` file tree | 30 min | — |

Notes on the ones with a wrinkle:

- **3** is safe but irreversible for `raw/` — it is scratch from the 17mafo
  merge, already consumed. Confirm nothing re-reads it before deleting rather
  than moving.
- **4** is mechanical except `scripts/compact_price_archive.py:271` and
  `scripts/purge_phantom_items.py:292`, whose `Path("../price-archive")`
  defaults are load-bearing for how they are invoked in CI. Change those two
  last, with the workflow lines.
- **5** is the one with a real decision in it: publishing means the daily
  workflow grows three steps and the archive grows ~0.75 MB; `derived/` means
  admitting these stay local and no model trained on them can be served. The
  event calendar is already known to be a clock and null once stripped, so
  `derived/` may be the honest answer for that one.

## Explicitly not doing

- **Adding `item_slug` to the Postgres `forecast_outcomes` table.** The DB has
  `items` to join against; a denormalised copy there is a second thing to keep
  true, for no read that needs it.
- **Dropping the `db/archive.py` projection** now that the data is uniform. It
  is what makes the code safe against an unmigrated archive, which every fresh
  clone of the data repo is. `test_plain_glob_silently_drops_source` fails if
  DuckDB ever stops narrowing — revisit then, not before.
- **`scripts/test_social_signal.py`.** It breaks `pytest -q` collection from
  `backend/` (imports `thefuzz`, absent from `requirements.txt`). Pre-existing,
  unrelated, and `pytest tests/` is clean. Rename it or add the dep — not part
  of this thread.
