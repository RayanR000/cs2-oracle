# Importing a historical price source, starting with cs2-prices-tracker

**Date:** 2026-08-08
**Status:** design, approved for planning
**Related:** `docs/references/data-sources.md`, `docs/research/2026-08-07-next-steps.md` (R13),
`docs/changelog/2026-08-08-volume-null-not-zero-and-skinport-sales.md`

## Why

`is_backfilled = 1` is derived as "the item has any row with `day < '2026-01-01'`"
(`scripts/init_local_db.py:84-90`). It is the gate on both training and predict. Items that
entered the archive when the aggregator feeds widened in **2026-03** have no pre-2026 rows, so
they are excluded — not because they lack prices today, but because they lack history.

`LukeX404/cs2-prices-tracker` commits dated daily Steam price files from **2025-02-17**. Its
data is exactly what the gate tests for.

### What was measured, 2026-08-08

Sample A — 31 days, every 10th day 2025-02-17 → 2025-12-31, all 31 files byte-distinct:

- **24,238** items priced on at least one pre-2026 day; **15,545** at median ≥ $1.
- Chao1 on the sample puts the true union at ~24,836, so this is ~3% conservative, entirely in
  the thin-coverage tail.

Sample B — 90 contiguous days, 2025-06-01 → 2025-08-29, all 90 present:

| proxy days (of 31) | items | median days/90 | median max gap | max gap ≤ 7 |
|---|---:|---:|---:|---:|
| 31 | 8,616 | 90 | 1 | 100% |
| 20–30 | 6,084 | 78 | 4 | 83% |
| 10–19 | 3,939 | 35 | 11 | 21% |
| 3–9 | 3,353 | 12 | 21 | 10% |

There is a cliff between 20 and 19. **15,539** items have max gap ≤ 7 over the block; **7,464**
of those are ≥ $1.

Join against `price-archive/prices-*.parquet`:

- **0** repo names are absent from the archive. `item_slug` *is* the raw `market_hash_name`;
  three join strategies return identical counts, 0 key-format failures. The repo adds **no new
  items**.
- **18,843** repo names have zero archive rows before 2026-01-01 — currently `is_backfilled = 0`
  and would flip. 18,842 of them already carry 2026 rows.
- Unfiltered, the gate would go **5,536 → 24,379** and the ≥ $1 training cohort **918 → 15,511**
  (≈5.28M feature rows against a 1.2M `TRAIN_FEATURE_ROWS` budget).
- The repo covers **95.8%** of existing gate items pre-2026, where the archive already averages
  305 of 318 days — so for the current gate it is redundant. Its entire value is the flippers.

Bridging — items priced on all of 2026-01-15, 2026-02-15 and 2026-03-05: **14,450**. Of the
max-gap ≤ 7 ∩ ≥ $1 set, **5,768 of 7,464 (77.3%)** bridge. Bridge rate *falls* as price rises
(77.3% at ≥$1, 72.1% at ≥$5, 63.6% at ≥$20) because `last_24h` requires a sale that day and
expensive items do not trade daily.

## What this is not

Not an accuracy claim. Every breadth change measured so far has come back null or unresolved,
including the +3.50pp that motivated the ≥$1 training floor, which re-derived to +1.642pp
[−0.809, +4.505]. What the import buys is a **pool worth selecting from**: today's 926-item
cohort is not a choice, it is everything that qualifies.

Not an item-onboarding path either — it adds zero items. Onboarding still requires the Steam
listing pages, which remain blocked on egress.

## Design

### Components

```
scripts/import_price_history_source.py     entry point, CLI, orchestration
  collectors/price_history_sources/
    __init__.py                            adapter registry
    cs2_prices_tracker.py                  THIS source: fetch + parse only
  collectors/price_history_import.py       generic: gap gate, schema, write
```

The split exists because a second backfill source is expected. An adapter supplies two things —
how to fetch a day, and how to parse it into `(item_slug, day, price)` — and nothing else.
Everything downstream is shared. No plugin framework, no registry beyond a dict; the seam is a
function signature.

### Data flow

1. **Fetch.** Day files from `raw.githubusercontent.com`, one request per date over
   **2025-02-17 → 2026-03-31**, into a scratch directory. Resumable: a present, non-empty,
   parseable file is skipped. Not a `git clone` — the full history is 3.1 GB and we need ~2.3 GB
   of it, without the packfile.
2. **Stall detection.** MD5 every fetched file. Byte-identical consecutive files mean the
   upstream scraper stalled (documented from mid-July 2026 onward). A duplicate group inside the
   requested range is a hard error, not a warning — a frozen price silently becomes a frozen
   price *run* in the label path.
3. **Parse.** `{market_hash_name: {steam: {last_24h, ...}}}` → rows. Only `last_24h`; a null is
   an absent row, never a zero.
4. **Quality gate — two conditions, both required.** Per item:
   (a) the largest gap between *consecutive observations inside the imported range* must not
   exceed **7 days** (`MAX_WINDOW_SPAN_DAYS`), and (b) the item must carry at least
   **180 distinct days** inside the range.

   Condition (b) is not redundant. Max-gap alone passes *trivially* for a near-empty item: two
   consecutive observations have a max interior gap of 1 and sail through, then flip
   `is_backfilled` on two rows. Measured on the 90-day block, **904 items (611 at ≥$1) clear
   the max-gap bar on fewer than 10 of 90 days, median 2 days present** — so max-gap is
   necessary and not sufficient. Together the two conditions mean "densely observed over at
   least half a year"; a genuinely dense item carries 300+ of the range's 408 days, so (b)
   never binds on real data.

   The gate measures interior gaps only — an item that starts late or ends early is judged on
   the span it does cover, not penalised for its edges. An item whose imported rows fall
   entirely in 2026 is written but flips nothing, since the gate derivation tests
   `day < '2026-01-01'`. Rejected items are counted and logged, not written.
5. **Write.** Through `db/parquet.py::append_monthly`, columns
   `item_slug, day, source, mean_price, volume, ingested_at`, into the **staging** archive dir
   (`--out-dir`, default `../archive-staging`, mirroring `merge_hf_dataset.py`'s `--out-dir`
   convention; the file layout underneath is `price-archive/prices-YYYY.parquet` as usual).

### Decisions and their reasons

**Source label `tracker_steam_24h`, and it votes.** The repo's `last_24h` is Steam's 24-hour
average sale price — the same quantity as `aggregator_sync`'s primary field, not a bid
(`aggregator_buff163_buy`) and not a trailing mean (`aggregator_steam_7d/30d/90d`). It is a
legitimate ask-side observation, so it is **not** added to `BID_SOURCES` and is left in the
consensus. The label is distinct so the rows stay attributable and a read-time filter can
remove them without a rewrite.

**Range ends 2026-03-31, not 2025-12-31.** Extending past the gate boundary is the seam
mitigation: the flippers' own archive coverage begins ~2026-03, so the overlap carries both
bases and `_apply_multi_source_voting` absorbs the transition. Ending at 2025-12-31 would put a
hard basis change at a single date and fabricate a return there for ~18,000 items — the defect
class of `aggregator_buff163_buy` and the MA feeds.

**Gap gate at import, price floor at training.** A punctured series yields no usable features
and would still flip `is_backfilled`, re-breaking the flag the way `init_local_db.py:72-77`
records it being broken before (it read 8,691/8,691). Price, by contrast, is a cohort decision
that belongs to `TRAIN_MIN_MEDIAN_PRICE` — the archive must not bake it in.

**`volume` is NULL, never 0.** The repo has no volume field. A real zero never occurs; a day
with no sale produces an absent row. Writing 0 is what defeated `has_volume` and
`volume_missing` and got eleven features shelved.

**`ingested_at` is the run's wall clock.** Arrival, not `day`.

### Staging and promotion

The import writes to a staging archive root, never to `price-archive/` and never to
`cs2-oracle-data`. Promotion is a separate, explicitly authorised step.

**Expected post-gate size, to be confirmed at import.** The 18,843 flip count and the
5,536 → 24,379 gate figure above are *unfiltered*. The max-gap condition passes 15,539 of 24,178
items (64%) on the 90-day block, and the 180-day floor removes a further ~13%, so the filtered
flip count is projected at **~10,500** and the gate at **~16,000**. These are extrapolations, not
measurements — the import computes both directly and they are the projection referenced in the
promotion gate below. Note the earlier headline figures (7,464 items at ≥$1, 5,768 bridging) were
computed before the 180-day floor and are ~8% optimistic at ≥$1.

Promotion gate — all must hold:

1. Row and item counts match the projection above within 10%.
2. Zero rows with `volume = 0`; every imported row has `volume IS NULL`.
3. No duplicate `(item_slug, day, source)`.
4. Recount `is_backfilled` under the derivation and record the new gate size.
5. **Overlap agreement**: per-item ratio of `tracker_steam_24h` to the archive's voted consensus
   over 2026-03, reported as median and IQR. This is the real seam measurement. The earlier
   ±25% IQR figure compared the repo against a 7-source consensus including non-Steam venues and
   overstates the disagreement; this is Steam-to-Steam and must be re-derived before promotion.

A failure at (5) does not necessarily block promotion, but it must be recorded and it sizes the
discontinuity that every downstream label inherits.

### Testing

`tests/test_price_history_import.py`, following `tests/test_supply_depth.py`:

- Parser: schema, `last_24h` only, null → absent row, no fabricated zeros.
- Gap gate: an 8-day gap rejects, a 7-day gap passes, boundary at exactly 7.
- Stall detection: byte-identical consecutive files raise.
- Date bounds: rows outside the requested range are refused.
- Write path: `volume IS NULL`, `source == "tracker_steam_24h"`, `ingested_at` populated,
  written into a temp-dir archive and read back through `db/archive.py::prices_relation`.

## Licensing

The repo has **no LICENSE and no README** — all rights reserved. `cs2-oracle-data` is private
and the import stays there; the public `cs2-oracle` repo carries the import *code* only, never
the data. Do not redistribute and do not surface the imported rows as a public data product.

## What this deliberately does not do

- **Does not retrain or change any training knob.** `TRAIN_FEATURE_ROWS` and
  `TRAIN_MIN_MEDIAN_PRICE` are untouched. Whether to spend the enlarged pool is a separate
  decision with its own measurement.
- **Does not change the `is_backfilled` derivation.** The flag will move as a consequence of new
  rows existing. Whether the derivation should distinguish source series is a real question and
  is out of scope here.
- **Does not repair the `volume` column.** Nothing here is a volume source.
- **Does not raise `MIN_SERVED_PRICE_USD`.** The FLOOR_SWEEP cannot currently decide it: the
  served model has resolved on 2 forecast dates against a `MIN_FORECAST_DATES` bar of 20, so
  every sweep cell stores `pt_verdict = "insufficient_dates"`.
- **Does not import a second source.** The adapter seam exists for one; it is not built out.

## Risks

- **Gate meaning drifts.** `is_backfilled` is documented as "carries the CSMarketAPI historical
  series". After this it also means "or ~13 months from the tracker". Shallower histories enter a
  cohort that averaged 1,075 days. This is the intended effect, but it should be stated in
  `database.py` beside the column.
- **Sample-based projections.** Item and row counts come from a 31-day sample and a 90-day
  block; February–May 2025 coverage is unverified and March 2025 has 5 known missing days. Treat
  row counts as ±15%, item counts as firm.
- **Every stored A/B and accuracy number computed after promotion sits downstream of a widened
  universe** and is not comparable to the series before it.
