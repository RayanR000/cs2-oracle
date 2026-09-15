# Data-org items 4 + 6 and training-cost O2 closed (2026-09-15)

Three leftover docs items, all local-only, no retrain, no serving change.

## Item 4 — archive-dir defaults routed through `db/archive.py`

Seven scripts defaulted to a CWD-relative `../price-archive`, which resolves to
the real archive from `backend/` and to nowhere from the repo root. All seven
now default to `db/archive.py::ARCHIVE_ROOT` (absolute, repo-anchored):

- `compact_price_archive.py`, `purge_phantom_items.py`, `normalize_price_schema.py`,
  `backfill_ops_item_slug.py`, `import_price_history_source.py` (`--archive-dir`)
- `merge_17mafo_gap.py` (`--out-dir`)
- `promote_iflow_staging.py` (`--out-dir` + `--staging-dir`, the latter via
  `ARCHIVE_ROOT.parent / "buff-iflow-staging" / "price-archive"`)

CI already passes explicit `--archive-dir` on the three load-bearing invocations
(`aggregator-update.yml:127,139,151`), so no workflow change was needed. Verified:
`compact_price_archive.py` dry-run from the repo root now hits the real archive
(previously: "Archive dir does not exist"); 179 tests across the eight covering
modules green; `ruff check` clean. Remaining `Path(__file__)`-anchored sites were
already CWD-safe and left alone.

## Item 6 — `data.md` tree refreshed + rarity precedence

`docs/architecture/data.md` tree now lists `volume-*`, `event-calendar/news`,
`exchange-rates-history`, `item-metadata-bymykel`, `supply-history`, the three
sidecar panels (`volume`/`bid`/`stattrak`), the tier-snapshot CSV, and
`ops/anchor_audit/`. New § Rarity precedence: read chain is
`item-metadata.parquet` → `items` DB (`_fetch_supply_metadata`); ByMykel is the
build-time fill (coverage 50.6% → 99.9%), not a serve-time read.

## Training-cost O2 — last stale cell fixed

`model-optimization.md` row-budget table still labelled the 99-item / 100K-row
config "(default)". It is now marked pre-2026-08-08 historical; the shipped
default (`DEFAULT_TRAIN_FEATURE_ROWS = 1_200_000` + `$1` floor) was already in
the same table. `model.md` needed no change (already corrected in `895005a`).

## Follow-up the same day: resolver dispatched, freshness already on

- **Freshness Check needed nothing** — it is `active` with green scheduled runs;
  the workplan's "disabled_manually" premise predates the workflow re-enable.
- **Backtest Accuracy dispatched** (`34984152746`, green, ~10 min): resolved
  nothing new. 09-06 is structurally unscoreable at every horizon (base leg
  inside the 08-28..09-05 outage; resolution gate says collection gap). Floored
  panel h=3: 5 dates, h=7: 1. Gate now waits on 09-12 (h=3) + 09-08 (h=7),
  both maturing 09-15 — ETA ~09-16 via the daily chain. Workplan Item 1 amended
  with the finding; do not dispatch again for 09-06.
