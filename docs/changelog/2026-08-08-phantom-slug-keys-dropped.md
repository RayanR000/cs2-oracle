# Phantom slug keys leave the item universe

**2026-08-08.** The archive carries two copies of 3,149 items under two spellings
of the same key. The duplication was documented on 2026-08-06 and the purge
script written then, but nothing had been applied and no reader excluded them.
They are now out of the universe at read time. The production `items` rows that
generate them daily are still there — that part needs prod access.

## What they are

`items.item_id` is the archive's `item_slug`, copied verbatim
(`append_to_parquet.py:33`, `export_daily_snapshot.py:24`). Two writers keyed
rows on something other than the `market_hash_name` every other inserter uses:

| Form | Keys | Writer |
|---|---|---|
| `sealed-graffiti-popdog-battle-green` | 3,145 | `migrate_historical_data.py:345` — `"item_id": slugify(mn)` |
| `steam_sticker_\|_sico_\|_rio_2022` | 4 | since-deleted `real_data_collector.py:340` |

The aggregator matches its external price lookup on `name`, not `item_id`
(`collectors/pipeline.py:124-128`), so two rows sharing a name both receive every
day's price. Fixing the writers changes nothing: the bad `items` rows already
exist and are re-read every run.

## Measured, 2026-08-08, whole archive

- **3,149 keys / 786,408 rows — 3.6% of 21.8M.**
- **All 3,149 pair 1:1 onto a correctly-keyed row. Zero unpaired.** So this is a
  de-duplication, not a universe reduction — nothing is dropped that is not also
  present under its real name. The earlier "14 unpaired" reading was an artifact
  of a `★ → star` substitution the original `slugify` does not do.
- On 2026-08-07, matching on day *and* source: **32,623 of 32,645 pairs identical
  to the cent (99.93%)**; the 22 exceptions all differ by under 0.5%.
- Per-day cost: ~32,700 rows, ~7.6% of the day's item count.

## Why it mattered

Not storage. **Validation.** A CV or A/B split that partitions by item can seat
`ak-47-redline-field-tested` in train and `AK-47 | Redline (Field-Tested)` in
test. Those are the same price series to the cent, so the fold reports a score
for an item it has already memorised. It also silently double-weights those
3,149 items during training relative to every other item.

The universe filter carried the bid exclusion and the phase-collapse rule but
not this one, so **every loader was exposed** — `walkforward_backtest.py`, which
publishes the Backtest Accuracy number, included.

## The change

- `models/item_parser.py` — `is_phantom_slug`, `phantom_slug_sql_filter`, both
  NULL-safe, and `archive_universe_sql_filter` now carries three rules rather
  than two. Invariant 2 in `backend/AGENTS.md` still holds: one call, every rule.
- `models/forecaster.py` — applied in `_fetch_voted_price_history`, the one read
  behind both `train()` and `predict()`. **`VOTED_CACHE_VERSION` 3 → 4**: the key
  fingerprints the archive and the window but cannot see the query, so a
  surviving v3 frame would train the next model on the duplicated universe.
- `scripts/purge_phantom_items.py` — now imports the predicate instead of
  redefining it. What the script deletes from production and what a training read
  drops have to be the same set.
- `tests/test_bid_source_voting.py` — the ladder fixture keyed on `"ak"`, which
  is itself phantom-shaped. It now uses a real `market_hash_name`. Left as-is,
  `test_published_gate_loader_excludes_the_bid` would have passed vacuously on an
  empty frame.

Verified against the real archive: 21,842,207 → 21,055,799 rows, 41,775 → 38,626
items, every dropped key confirmed to have a surviving twin.

## Not done

- **The prod `items` rows still exist**, so the daily run keeps writing ~32,700
  phantom rows. Readers ignore them; the archive keeps growing them.
  `scripts/purge_phantom_items.py --target db --apply`, run from `backend/`,
  is the fix and is irreversible. **The dry run was taken 2026-08-08 and every
  safety condition holds** — re-run it before applying, but the expected shape is:

  | | |
  |---|---|
  | `items` rows scanned | 35,058 |
  | phantoms with an identified keeper | **3,149** |
  | phantoms unresolved (left alone) | **0** |
  | child rows to delete | 97,520 — `item_forecasts` 75,576, `forecast_outcomes` 9,447, `price_history` 9,348, `supply_snapshots` 3,149 |

  The 4 rows the script flags as `slugify(name) != item_id` are the four
  `steam_` keys; that cross-check only applies to the slug arm, and each one
  matches its keeper exactly under the `steam_` transform.

  Checked separately against the ops mirror: **0 orphans** — every phantom's
  keeper already holds its own `item_forecasts` and `forecast_outcomes` rows, so
  deleting the phantom's copies de-duplicates rather than removing an item from
  the product.
- **The durable archive still holds the 786,408 historical rows.** Purging them
  means running `--target archive --apply` against a checkout of
  `RayanR000/cs2-oracle-data` and force-pushing, since CI carries that repo
  forward run to run rather than rebuilding it.

Neither is required for correctness now that the readers filter, but until the
first one lands, treat any statistic computed directly off recent archive days —
without the universe filter — as ~7.6% contaminated.
