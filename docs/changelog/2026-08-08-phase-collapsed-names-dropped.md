# One name, 181 assets — the Doppler names leave the item universe, and it is worth 0.65% of the served cohort

**Date:** 2026-08-08
**Plan:** `docs/research/2026-08-07-next-steps.md` step 4
**Change:** `backend/models/item_parser.py` (new `PHASE_COLLAPSED_SLUG_PATTERNS`,
`PHASE_COLLAPSED_EXEMPT_PATTERNS`, `phase_collapsed_sql_filter`, `is_phase_collapsed`),
`backend/models/forecaster.py` (re-export, filter in `_fetch_voted_price_history`,
`VOTED_CACHE_VERSION` 2 → 3), `backend/scripts/walkforward_backtest.py`
(`_load_all_prices` and `_load_parquet_items`), `backend/api/routes/opportunities.py::_load_items`,
`backend/tests/test_phase_collapsed_universe.py` (new, 19 cases).
**Suite:** 1,317 pass in 79s (`venv/bin/python -m pytest tests/ -q` from `backend/`), 0 failures.

**This is a correctness fix on a 0.65% slice, not an accuracy lever.** The measured cut is an
order of magnitude below the fold-clustered item-level MDE floor of **2.21–3.69pp**
(`2026-08-07-training-item-universe.md`). No A/B was run, and none should be — the harness
cannot resolve an effect this size, so any number it returned would be noise wearing a sign.

## The defect: a `market_hash_name` cannot name a Doppler

`market_hash_name` encodes weapon + finish + wear bucket + StatTrak/Souvenir and nothing else.
A Doppler or Gamma Doppler name therefore collapses every phase — Ruby, Sapphire, Black Pearl,
Emerald, Phase 1–4 — into one series. Measured 2026-08-07 against the free BUFF dump
(46,041 items), `docs/research/2026-08-07-cs2-forecasting-research.md` §25, which is §22 D9
quantified:

| | |
|---|---:|
| Base names → distinct `paint_index` assets | **29 → 181** |
| Median max/min phase ratio **within one name** | **3.25×** |
| p90 / max | 6.08× / **23.5×** |
| Share with >2× internal dispersion | **87.3%** |
| Quoted headline **is** the cheapest phase | **95.5%** |

The last row is the defect. If the headline tracks the cheapest phase, the series steps
whenever *which* phase is cheapest changes — a level shift with no asset repricing. That step
is a return the model is trained to predict and a label the backtest scores, and it was
fabricated by composition. Worst case: `★ StatTrak™ M9 Bayonet | Doppler (Minimal Wear)` spans
**$1,261 → $29,685** under one name.

Step 8 (the hedonic index) would have inherited the artifact, which is why the plan ordered
this before it.

## Dropped, not split

The dump carries a `doppler` sub-object with ask *and* bid per phase on 110 names, so splitting
is technically available. It was declined: splitting means ingesting that sub-object daily as
its own price series, with its own archive schema, its own backfill and no history before the
ingest starts. That is a data-collection project, not a filter, and the filter is what step 4
asked for. The names are still in the price archive — nothing was deleted — so the split
remains available if the sub-object is ever ingested.

The rule lives in `backend/models/item_parser.py`, not in `forecaster.py`, because
`api/routes/opportunities.py` needs the predicate and must not import LightGBM to get it.
`models/forecaster.py` re-exports all four names, so archive readers keep taking their universe
rules from one module exactly as they already do for `BID_SOURCES`.

```python
PHASE_COLLAPSED_SLUG_PATTERNS = ("doppler",)
PHASE_COLLAPSED_EXEMPT_PATTERNS = ("sticker",)
```

`phase_collapsed_sql_filter(column)` emits `(col IS NULL OR NOT (…) OR (…exempt…))`. The
`IS NULL` leg is load-bearing for the same reason it was in the bid filter: a bare `NOT LIKE`
over a NULL evaluates to NULL and silently drops the row, and `source IS NULL` selects 13 years
of this archive. A NULL-unsafe filter on the *key* would delete rows nobody asked to delete.

## Matched on the name, and the name has two false positives

Detection is by name rather than by the dump's `doppler` sub-object because the archive read has
to work offline and 13 years back, while the sub-object lives in a live feed.

Measured today against the local `price-archive/` (bounds **20,756,038 rows / 41,725 slugs /
2013-08-14 → 2026-08-04**):

| | value | share of archive |
|---|---:|---:|
| Slugs matching `doppler` | **129** | 0.309% of slugs |
| Rows | **47,081** | 0.227% of rows |
| — Gamma Doppler | 52 | |
| — plain Doppler | 75 | |
| — **false positives** | **2** | 396 rows |

The two false positives are `Sticker | Doppler Poison Frog (Foil)` and
`Sticker Slab | Doppler Poison Frog (Foil)` — ordinary single assets that borrow the name and
collapse nothing. They are what the `sticker` exemption is for. **This corrected the first
version of the filter**, which asserted that nothing but a phase name carries the word; it was
wrong, and it would have dropped two clean series.

Six of the 127 real matches are lowercase-hyphenated duplicate slug keys for names already
present (`gut-knife-doppler-factory-new` beside `★ Gut Knife | Doppler (Factory New)`), so
**121 distinct real assets** leave the universe.

## The cut on the served cohort is six items

On the ≥$1 cohort (`MIN_SERVED_PRICE_USD = 1.0`) the drop is **6 items of 926 (0.65%) and
7,560 item-days of 994,432 (0.76%)**. *(The item-day denominator differs slightly from the
993,464 in `docs/architecture/model.md` — different measurement dates against a growing
archive. Neither figure is a correction of the other.)*

| Item | median price |
|---|---:|
| ★ Huntsman Knife \| Gamma Doppler (FN) | $479.38 |
| ★ Falchion Knife \| Gamma Doppler (FN) | $429.25 |
| ★ Paracord Knife \| Doppler (FN) | $329.12 |
| ★ Navaja Knife \| Doppler (FN) | $216.20 |
| ★ Gut Knife \| Doppler (FN) | $212.36 |
| ★ Shadow Daggers \| Gamma Doppler (FN) | $208.68 |

**The other ~121 are absent from the cohort because they are not in the gated pool at all, not
because of the price floor.** Every Doppler slug clears $1 by two to three orders of magnitude.
Most first appear on 2026-03-22 with ~128 days of history, short of the backfill gate. So the
size of this drop is bounded by the pool gate, and it will grow if that gate ever admits them —
which is the argument for landing the filter now rather than when it is expensive.

Currently served: **12 distinct slugs** in `price-archive/ops/item_forecasts.parquet` (the six
real items plus their six duplicate keys), **48 of 34,764 rows on a serving day — 0.138%**.

## Four readers, because there is no single read

1. **`models/forecaster.py::_fetch_voted_price_history`** — the one archive read behind *both*
   `train()` and `predict()`. Filtering here removes the names from the training universe and
   stops the product forecasting them, in one place.
2. **`ItemForecaster.VOTED_CACHE_VERSION` 2 → 3.** Mandatory, not hygienic. The cache key covers
   the `prices-*.parquet` fingerprint, the cutoff date and the backfill slug set; it cannot see
   the query. A surviving v2 frame still carries the names and would have trained the next model
   on them with nothing in the logs to say so.
3. **`scripts/walkforward_backtest.py`, both loaders.** The published gate never calls
   `fetch_price_history` — it globs the archive itself, which is why the bid exclusion needed a
   second fix in this same file. `_load_all_prices` filters the prices; `_load_parquet_items`
   filters item *selection* too, so the `max_items` budget is not spent on items the price
   loader then returns nothing for.
4. **`api/routes/opportunities.py::_load_items`.** These queries take each item's **latest**
   forecast with **no date bound**, so an item that stops being forecast keeps its final row on
   the ranked surfaces permanently. The 12 slugs are $200–500 knives; they rank. All four
   endpoints (`/`, `/undervalued`, `/overheated`, `/momentum`) already skip a forecast whose
   item is missing from the map, which makes `_load_items` the one choke point.

## Verification

`backend/tests/test_phase_collapsed_universe.py`, **19 cases**:

- the predicate on real archive keys, in **both spellings** — `★ Gut Knife | Doppler (Factory
  New)` and the duplicate key `gut-knife-doppler-factory-new`
- both sticker exemptions, as the measured false positives rather than as invented examples
- NULL-safety of the SQL filter, asserted against DuckDB rather than by reading the string
- Python/SQL agreement, so the cheap predicate cannot drift from the one production runs
- the training read, both gate loaders and the opportunities surface, each end to end against a
  temp-dir Parquet archive
- two guards: that `PHASE_COLLAPSED_SLUG_PATTERNS` is non-empty (an emptied constant would make
  every other case pass for the wrong reason — the bid exclusion needed the same guard), and
  that `VOTED_CACHE_VERSION >= 3`

**Full suite: 1,317 pass in 79s**, 0 failures.

**Smoke check against the real local `price-archive/`:**
`_fetch_voted_price_history(days_back=120)` returns **41,444 items / 3,111,549 rows**, with
**zero phase-collapsed survivors** and both Poison Frog stickers retained.

## What was deliberately not done

- **No A/B.** See the header. 0.65% of items against a 2.21–3.69pp floor.
- **Stored forecasts and outcomes were not purged.** The rows for these items stay in
  `item_forecasts` and `forecast_outcomes` and keep being scored until they mature out. They
  drain from the scored cohort on their own within 30 days.
- **`backtest/price_resolution.py::load_voted_prices` was deliberately left unfiltered.**
  Filtering it would make the pending outcomes above unresolvable, pushing them into the
  archive-gap/unresolvable gate — turning a 0.76% label-quality improvement into a gate risk.
  This is the one production consumer of the consensus that the bid exclusion *did* have to
  touch and this one does not.
- **Nothing was deleted from the archive.** The bid-style property holds: the rows stay
  recoverable, the exclusion is a read-time filter, and no rewrite of the canonical archive is
  needed.
- **The names were not split by phase.** See § Dropped, not split.

## Still open

- **The ten-plus `scripts/ab_test_*.py` harnesses carry private archive globs and still see the
  Doppler names**, so their item universe now differs from production's — on top of the
  `BID_SOURCES` filter they also lack and the missing purge/embargo. Step 5 of the next-steps
  list already owns those loaders; the phase filter should land in the same pass. Until then an
  A/B on those harnesses is not measuring production's universe.
- **Whether this moves any accuracy number is unmeasured, and probably unmeasurable.** No claim
  is made either way. Nothing here was scored before and after.

## Related

- `docs/research/2026-08-07-cs2-forecasting-research.md` §25 and §22 D9 — where the phase
  collapse was measured
- `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md` — the same shape of fix, and
  the source of both the NULL-safety trap and the "the gate loader needs its own filter" lesson
- `docs/changelog/2026-08-07-training-item-universe.md` — the 2.21–3.69pp MDE floor and the
  926-item ≥$1 cohort
- `docs/changelog/2026-08-07-archive-schema-and-keys.md` — `db/archive.py::prices_relation`,
  the reader all of these filters are applied on top of

## Docs touched

- `docs/architecture/model.md` — § Training window now states the item-universe rule; § Known
  limitations records the drop's size and the unfiltered A/B harnesses; the
  `VOTED_CACHE_VERSION` note names v3; `models/item_parser.py` added to Files Reference
- `docs/architecture/data.md` — § Reading the price archive records that the universe rule is a
  read-time filter over `prices_relation`, with the two exempt stickers named
- `docs/research/2026-08-07-next-steps.md` and `backend/AGENTS.md` were updated when the change
  landed, outside this pass
