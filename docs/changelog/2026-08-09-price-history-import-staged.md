# A historical price importer is staged, not promoted — and the spec's basis premise was wrong

**Date:** 2026-08-09
**Spec:** `docs/superpowers/specs/2026-08-08-price-history-source-import-design.md`
**Plan:** `docs/superpowers/plans/2026-08-08-price-history-source-import.md`
**Ledger:** `.superpowers/sdd/2026-08-08-price-history-source-import/progress.md`
**Commits:** `fb6d98c`, `a657db7`, `3355ae6`, `287b070`, `99636e9`, `68890d6`, `52f4b3b` (branch `feat/price-history-import`)
**Suite:** 1,657 pass, no regressions; 44 new tests in `tests/test_price_history_import.py`

**1,963,626 rows across 5,153 items sit in `archive-staging/`. Nothing has been promoted, and
four open defects say it should not be yet.**

## Why the source was pursued

`is_backfilled` is derived as "the item has any row with `day < '2026-01-01'`"
(`scripts/init_local_db.py:84-90`) and gates both training and predict. Items that entered the
archive when the aggregator feeds widened in 2026-03 are excluded for lacking *history*, not
for lacking prices. `LukeX404/cs2-prices-tracker` publishes dated daily Steam price files from
2025-02-17 and supplies exactly what the gate tests for.

**What the source is not.** Measured 2026-08-08 before any code was written: it adds **zero new
items**. `item_slug` *is* the raw `market_hash_name`; three join strategies returned identical
counts with 0 key-format failures. It also covers 95.8% of items already inside the gate, where
the archive already averages 305 of 318 days — for the existing cohort it is redundant. Its
entire value is that **18,843 of its names have no pre-2026 archive rows** and would flip.

Unfiltered that is not a usable universe: the gate would go 5,536 → 24,379 and the ≥$1 cohort
918 → 15,511, ≈5.28M feature rows against a 1.2M `TRAIN_FEATURE_ROWS` budget
(`2026-08-08-training-price-floor-shipped.md`). Hence a quality gate.

## What shipped

| Path | Role |
|---|---|
| `backend/collectors/price_history_sources/cs2_prices_tracker.py` | this source only — `day_url`, `parse_day`, `SOURCE`, `STEAM_FEE_MULTIPLIER` |
| `backend/collectors/price_history_sources/__init__.py` | the `ADAPTERS` dict |
| `backend/collectors/price_history_import.py` | source-agnostic: stall detection, quality gate, canonical frame, archive write |
| `backend/scripts/import_price_history_source.py` | CLI, resumable fetch, `report_promotion_gate` |
| `.gitignore` | `archive-staging/` |

An adapter supplies how to fetch a day and how to parse it into `(item_slug, day, price)` and
nothing else. A second backfill source is expected; the seam is a plain dict and a function
signature, not a plugin framework.

### Decisions worth recording

**Source label `tracker_steam_24h`, and it votes.** `last_24h` is Steam's 24-hour average sale
price — the same quantity as `aggregator_sync`'s primary field. It is not a bid, so it is
deliberately **not** added to `BID_SOURCES` (`models/item_parser.py:34`), and it is not a
trailing-window mean like `aggregator_steam_7d/30d/90d`, which are the feeds already documented
as voting against point-in-time asks. The label is distinct so the rows stay attributable and a
read-time filter can drop them without a rewrite.

**The quality gate is two conditions and both are required**
(`price_history_import.py::apply_gap_gate`): no interior gap wider than **7 days**
(`MAX_GAP_DAYS = MAX_WINDOW_SPAN_DAYS`, the archive's own constant) **and** at least **180
distinct days**. The second is not redundant. Max-gap alone passes *trivially* for a near-empty
item, because two consecutive observations have a max interior gap of 1. Measured on the 90-day
block: **904 items (611 at ≥$1) clear the gap bar on fewer than 10 of 90 days, median 2 days
present.** Each would have flipped `is_backfilled` on two rows — the exact failure
`init_local_db.py:72-77` records happening once before, when the flag read 8,691 of 8,691.

**Gap gate at import, price floor at training** — the archive must not bake in a cohort
decision. That held until the fee finding below forced a *data-validity* floor; see there.

**`volume` is a typed NULL, never 0** (`Int64` with `pd.NA`). A real zero never occurs: a day
with no sale produces an absent row. Fabricated zeros are what kept `has_volume` reading True
and `volume_missing` reporting "present"
(`2026-08-08-volume-null-not-zero-and-skinport-sales.md`). `ingested_at` is the run's wall
clock — arrival, not `day`.

## The spec's basis premise was wrong, and the promotion gate is what caught it

The spec asserted the source needed no basis conversion because both sides are Steam. **False.**
Measured against `aggregator_sync` on the overlap:

| | raw ratio |
|---|---|
| Whole overlap | **1.147–1.154** |
| Across seven price tiers | 1.107 (<$0.10) → 1.163 (≥$100) — **flat** |
| Median within-item CV | 0.046–0.063 |

Flat across four orders of magnitude is the signature of a **constant**, not a market wedge. The
source serves Steam's **buyer** price (fee included); the archive stores **net**. Dividing by
`STEAM_FEE_MULTIPLIER = 1.1607` collapses the ratio to **0.998 at ≥$1**.
`scripts/backfill_steam_listing_history.py:131` already carried that constant for exactly this
trap, and `docs/references/data-sources.md` already documents it. The first import ran without
it.

Before → after (`52f4b3b`, division moved into `parse_day`):

| | before | after |
|---|---|---|
| `overlap_ratio_median` | 1.155 | **1.0054** |
| IQR | 1.042 – 1.210 | **0.984 – 1.037** |

A steady-state control confirms it item-by-item at **0.9970** over 338,841 overlapping
item-days.

**A `$1` median floor was then added as a data-validity decision, not a cohort one.** The
constant is documented as **synthetic** and its original validation as circular
(`2026-08-07-cs2-forecasting-research-review.md` §4): it is flat to 4 dp where Steam's
cent-ceiling schedule must swing 1.150→1.168 over a single dollar. The correction lands at
**0.906 below $1** against 0.998 above, so sub-$1 rows are rows the source cannot be fee-corrected
into. `apply_gap_gate(min_median_price=…)` defaults to `None` in the library — so the module
still bakes in no cohort rule — and the CLI passes `1.0`. The floor judges an item by the
**median** of its observations, so a single spike cannot carry a cheap item over.

## Final staged result

| | |
|---|---:|
| Rows | **1,963,626** |
| Items | **5,153** |
| `new_gate_items` | **4,196** |
| `zero_volume_rows` | 0 |
| `duplicate_keys` | 0 |
| Rejected on gaps | 12,818 |
| Rejected on the 180-day floor | 2,041 |
| Rejected on the $1 floor | 7,024 |

The fetch covered 408 days (287 fetched, 116 pre-seeded, 5 absent upstream — the known March
2025 holes) with no stall detected.

`new_gate_items` **4,196 is below the spec's projected 9,000–12,000**. That is a projection
error, not a code defect: the pass-rate was extrapolated from a 90-day block, and over 408 days
far more items carry a >7-day gap somewhere. Recorded so the spec's number is not cited again.

## A second defect the review caught: `append_monthly` does not preserve first arrival

`db/parquet.py:250-262` selects `_new` unconditionally and keeps an existing row only
`WHERE NOT EXISTS` a match, so a re-append takes the **new** row's `ingested_at`. The
first-arrival logic exists only in `scripts/append_to_parquet.py:267-270`
(`groupby(...).transform("min")`), the CI daily writer.

**`.claude/rules/archive-reads.md` states "A re-append keeps the **first** arrival" as an
archive-wide property. It is not — it is a property of one writer.** The importer restores it
itself in `_preserve_first_arrival` (`287b070`, ruled on by a human mid-task and recorded in
`e2fc41b`), with two tests: `test_a_reappend_keeps_the_first_arrival_not_the_latest` and
`test_a_row_with_no_prior_arrival_takes_the_new_timestamp`. Anything else built on
`append_monthly` inherits the same surprise. **The rule file needs correcting and this entry
does not do it** — `.claude/rules/` is outside this pass's write scope.

## Not done, deliberately

- **Not promoted.** Promotion goes through the `cs2-oracle-data` repo; `archive-staging/` is
  gitignored and the local `price-archive/` is untouched.
- **No training knob moved.** `TRAIN_FEATURE_ROWS` and `TRAIN_MIN_MEDIAN_PRICE` are unchanged,
  no model was retrained, and no accuracy number is claimed. Whether to spend a wider pool is a
  separate decision with its own measurement.
- **The `is_backfilled` derivation is unchanged.** The flag would move as a consequence of rows
  existing. Whether the derivation should distinguish source series is a real question and is
  out of scope. Note the gate's *meaning* would drift: it is documented as "carries the
  CSMarketAPI historical series" and would also come to mean "or ~13 months from the tracker".
- **No second source.** The adapter seam exists for one and is not built out.
- **Licensing.** The upstream repo has **no LICENSE** — all rights reserved. Private training
  use only. The imported rows must never be redistributed or surfaced as a public data product.

## Four blockers, from the whole-branch review

1. **The end seam is unmeasured.** The source stops 2026-03-31 and leaves the vote on 04-01;
   consensus steps **−0.78% median, 80% one-directional, on one date**. `report_promotion_gate`
   reports the start seam and is blind to this.
2. **`report_promotion_gate`'s "consensus" is not production's consensus.** It medians every
   existing source, so it includes `aggregator_buff163_buy` (a bid) and `historical_fallback:%`,
   and it omits `archive_universe_sql_filter()` — backend invariant 2. The prod-equivalent
   overlap is **1.020, IQR 0.973–1.092**, so the gate *understated* disagreement while its
   docstring claims the opposite ("This is Steam-to-Steam, unlike the earlier … figure, which
   overstated the disagreement").
3. **`_preserve_first_arrival` silently NULLs `ingested_at` for any non-`RangeIndex` frame.**
   The merge returns a fresh index and the assignment is index-aligned. Reproduced through the
   public entry point. It does not fire today because the only caller passes a fresh frame, but
   the module is advertised for reuse and a chunked write trips it.
4. **Silent under-import.** A corrupt download is cached and hashed without validation,
   `parse_day(None, …)` returns `[]`, nothing counts days that produced no records, and `main()`
   returns 0 regardless of rows written. There is no analogue of `run_task.py`'s zero-row guard —
   the guard the root `AGENTS.md` names as the only thing between a dead collector and a green
   badge.

## Still open

- The 2026-03-22 consensus break, found while validating this import's seam and **not caused by
  it**: `docs/changelog/2026-08-09-march-22-consensus-break.md`. Promotion does not fix it and
  excluding the imported items does not either.
- Deferred minors are in the ledger, including: `MAX_GAP_DAYS` resolving through an
  env-overridable chain, so gate strictness is env-dependent for a one-shot import that fixes
  archive contents; `_preserve_first_arrival` full-scanning the archive per call;
  `report_promotion_gate` reusing one filtered relation for both the overlap and the
  `new_gate_items` check, which would understate overlap after a partial promotion.

## Related

- `docs/changelog/2026-08-08-volume-null-not-zero-and-skinport-sales.md` — why `volume` is NULL
- `docs/changelog/2026-08-06-steam-listing-backfill-and-phantom-items.md` — the 1.1607 trap, hit
  and fixed once before
- `docs/changelog/2026-08-07-cs2-forecasting-research-review.md` §4 — the constant is synthetic
- `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md` — why a new source's basis
  matters at all
- `docs/references/data-inventory.md` §5 — source coverage; updated with a pointer to the
  03-22 break
