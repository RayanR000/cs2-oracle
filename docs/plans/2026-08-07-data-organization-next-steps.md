# Data organisation — next steps (2026-08-07)

> ## Status — 2026-09-15: items 1 ✅, 2 ✅, 4 ✅, 5 ✅ (published), 6 ✅ resolved; 3 🟡 partly done (local only — canonical still carries all four).
>
> **This is the most outstanding plan in `docs/plans/`.** Verified against the
> canonical `RayanR000/cs2-oracle-data` repo, not the local copy.
>
> | # | Status |
> |---|---|
> | 1 — ship to production | ✅ **done**. Commits `b55c15b`, `c0f0349` are ancestors of `origin/main`; dispatch `31246426269` ran `Normalize the price schema: success`. **Both legs verified on the canonical repo:** `prices-2026-08.parquet` carries `item_slug, day, source, mean_price, volume, ingested_at`, and `ops/forecast_outcomes.parquet` has **22 columns including `item_slug`**. |
> | 2 — the orphaned outcomes | ✅ **RESOLVED 2026-08-22: bookkeeping, not a scoring bug.** Re-measured on canonical: **109,191 rows, 34% NULL slug** (the file was rewritten again; 48,241/77% is stale). NULL-slug is perfectly determined by *which resolution run wrote the row* — all slugged rows trace to a single run (`evaluated_at 2026-08-09`, 5,542 items); 5,542 item_ids appear both mapped and unmapped, i.e. the same real item slugged on some rows and NULL on others. Scoring keys on `item_id`, not slug: `backtest_accuracy.py:208-209` computes `direction_correct`/`in_interval` per row from predicted-vs-actual, and slug is only ever attached as a projection (`_with_item_slug`/`id_to_slug.get`). The raw slug-vs-no-slug accuracy gap (cover 0.77 vs 0.92) is a horizon/run composition artifact, not identifiability. Phantom-purge worry refuted for the bulk. **Residual (cosmetic only):** 3,149 item_ids are only-ever-unmapped and can't be *named* when joining to price history until the one-off `backfill_ops_item_slug.py --apply` runs against the **canonical** archive (needs prod `items`; never done end-to-end). |
> | 3 — move `raw/`, delete the three ambiguities | 🟡 **partly done 2026-08-22 (local checkouts only).** Deleted `raw/` (empty — a re-downloadable `merge_17mafo_gap.py` cache, not the claimed 605 MB), `2026/` (empty superseded date-layout), and the `player-counts/` **dir** of 48-byte daily CSVs — all verified unread by any runtime loader (the price loader uses a non-recursive `directory.glob("prices-*.parquet")`, `db/archive.py:35,117`). ⚠️ **`exchange-rates-2026.parquet` is NOT scratch — do not delete it.** It is live daily aggregator output (`ingest_fx_history.py:37-38`; written by `pipeline.py::fetch_exchange_rates` → `append_to_parquet.py`, committed as "prices + exchange-rates" in `aggregator-update.yml:227`) and regenerates every run. Guard held: the 16 `player-counts-YYYY.parquet` panel files were left intact. **Canonical still carries all four** — these deletions were to the local checkouts only; the canonical `cs2-oracle-data` is CI-write-only (orphan commit + force-push), so a real removal must land in CI's committed tree. |
> | 4 — route scripts through `db/archive.py` | ✅ **done 2026-09-15.** All seven CWD-relative `Path("../price-archive")` / `"../price-archive"` defaults now resolve via `db/archive.py::ARCHIVE_ROOT` (compact, purge, normalize, backfill_ops_slug, import_price_history_source, merge_17mafo_gap, promote_iflow_staging incl. its staging dir). Verified: `compact_price_archive.py` dry-run from the repo root hits the real archive; 179 tests green. Remaining `Path(__file__)`-anchored sites were already CWD-safe. |
> | 5 — publish or `derive/` the three local-only ingests | 🟡 **effectively resolved the other way.** All three are now **published** to the canonical repo (`event-calendar.parquet`, `exchange-rates-history.parquet`, `item-metadata-bymykel.parquet` + codes JSON), so the "publish vs `derived/`" decision was taken by publishing. No `price-archive/derived/` exists. Item 5's stated cost — three extra workflow steps — is already paid. |
> | 6 — document rarity precedence, refresh the `data.md` tree | ✅ **done 2026-09-15.** Tree in `docs/architecture/data.md` now lists volume-*, event-calendar/news, exchange-rates-history, item-metadata-bymykel, supply-history, the three sidecar panels, the tier snapshot CSV, and `ops/anchor_audit/`; new § Rarity precedence documents the read chain (item-metadata.parquet → items DB) and the ByMykel fill path (50.6% → 99.9%). |

Follow-on from `docs/changelog/2026-08-07-archive-schema-and-keys.md`, which
landed items 1 and 2 of a six-item review. Items 3–6 remain, plus two things the
landed work created or exposed.

## Blocking

### 1. Ship items 1 and 2 to production — ✅ **DONE 2026-08-08**

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

### 2. Decide what the 31,422 orphaned outcomes mean — ✅ **RESOLVED 2026-08-22: bookkeeping, not a scoring bug**

> **Answered 2026-08-22 on the canonical repo (109,191 rows, 34% NULL slug — the 48,241/77%
> reading is stale; the file was rewritten again).** The `WHERE item_slug IS NULL` comparison
> was run. Verdict: **bookkeeping.** NULL-slug is perfectly determined by *which resolution run
> wrote the row* — every slugged row traces to a single run (`evaluated_at 2026-08-09`, 5,542
> items), and those same 5,542 item_ids also appear unmapped in other runs, so the item is real
> and identifiable via `item_id`; the slug column just wasn't backfilled on the older rows.
> Scoring keys on `item_id`, never on slug (`backtest_accuracy.py:208-209` computes
> `direction_correct`/`in_interval` per row from predicted-vs-actual; slug is attached only as a
> projection via `_with_item_slug`/`id_to_slug.get`), so NULL slug cannot corrupt the headline.
> The raw slug-vs-no-slug accuracy gap (coverage 0.77 vs 0.92, pct_error 11.8 vs 2.7) is a
> horizon/run **composition** artifact — NULL rows skew to h=3 and earlier/global-only runs —
> not identifiability. Phantom-purge hypothesis refuted for the bulk. **Residual (cosmetic
> only):** 3,149 item_ids are only-ever-unmapped and cannot be *named* when joining to price
> history until the one-off `backfill_ops_item_slug.py --apply` is run against the **canonical**
> archive (needs prod `items`; never done end-to-end).

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
| 3 | 🟡 partly done 2026-08-22 (local only). Deleted `raw/` (was empty, not 605 MB), `player-counts/` dir, `2026/`. **Do NOT delete `exchange-rates-2026.parquet` — it is live daily aggregator output, not scratch.** Remaining: replay the same deletion in CI's committed tree so the canonical repo drops them. | 15 min | three fewer ambiguities (the −605 MB estimate was stale; `raw/` was already empty) |
| 4 | Route the remaining ~25 scripts through `db/archive.py` instead of their own `ARCHIVE_DIR` | ~1h | One definition; kills two CWD-relative defaults that resolve differently depending on where they are run |
| 5 | Publish the three local-only ingests (`event-calendar`, `exchange-rates-history`, `item-metadata-bymykel`) or move them to `price-archive/derived/` | ~1h | The local/prod boundary becomes visible in `ls` |
| 6 | Document rarity precedence across `items`, `item-metadata.parquet`, `item-metadata-bymykel.parquet`; refresh the `data.md` file tree | 30 min | — |

Notes on the ones with a wrinkle:

- **3** — done for `raw/`/`player-counts/`/`2026/` (confirmed unread; `raw/` was
  already an empty, re-downloadable `merge_17mafo_gap.py` cache). `exchange-rates-2026.parquet`
  was in the original delete list by mistake — it is live daily aggregator output; leave it.
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
