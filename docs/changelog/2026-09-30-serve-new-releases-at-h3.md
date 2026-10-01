# New releases are discovered and served, at h=3 only

2026-09-30. Item 23 of `research/2026-09-28-next-steps.md`.

## The gap

Two separate things kept new releases out of the product, and neither was detection.

- **Nothing added items to the catalog.** `discover-new-items.yml` is a stub that exits 1
  (its script was deleted 09-15), and the monthly CSMarketAPI backfill that was meant to add
  items is gone. The 2026-07-08 release had 0 of 529 items in `items`.
- **The serve universe was `is_backfilled = 1`.** Only the backfill could set that flag, so a
  release added to `items` would still never be forecast.

The prices were never the problem. The aggregator already writes every dump item to the
archive (`collectors/pipeline.py`, the raw CSGOTrader append), so the July release had 43 days
of 7-source history by 09-08 without being in the catalog.

## Detection is accurate

Measured on the 2026-09-30 dumps:

- **Recall:** 529/529 of the 2026-07-08 release are in the dump union, 180 of them priced in
  all 7 sources.
- **Stability:** across the 10 Aggregator runs 09-20..09-29 the per-source name counts moved by
  +2 in total (most sources not at all), so a name appearing later is a release, not a flicker.
- **Noise:** about 60 Steam-only names carry every price null (`AWP | Doodle Lore`,
  `M4A4 | Emperor`). Steam's own market search returns 0 results for them. Requiring a positive
  price in >= 2 sources excludes them.

## Serving young items is safe at h=3 only

A model trained with production's flags on archive data bounded at 2026-09-15
(`REPLAY_ANCHOR`, so no outcome was in training) was replayed at 09-15 and 09-22 with the July
cohort added to the serve universe. Production's feedback factors (0.5385 / 0.6111 / 0.7059)
were applied. Cohort >= $1, target 80%:

| h | established 09-15 / 09-22 | July release 09-15 / 09-22 | half-width est. / July |
|---|---|---|---|
| 3 | 84.9% / 86.6% (n 892 / 881) | **83.1% / 80.7%** (n 195 / 192) | 5.4% / 6.2% |
| 7 | 84.5% / 88.6% | **74.7% / 73.8%** | 7.7% / 8.9% |
| 14 | 86.3% (09-15 only) | **70.1%** | 10.9% / 12.4% |

The model already widens young bands by ~15%, but not enough past h=3, and the misses there
are two-sided. One release is one market episode and the two anchors share items, so these are
a safety screen, not a calibration: pooled h=3 is 81.9% (317/387, naive Wilson [77.8, 85.5]).

A per-cohort width would be a Mondrian cut (`research/2026-09-28-next-steps.md` §7), so the
longer horizons are **withheld** instead.

## What shipped

- **`models/serve_universe.py`** is the one definition of the served universe: established
  (`is_backfilled = 1`, all horizons) plus young (`release_date` set, not backfilled, at least
  `YOUNG_MIN_HISTORY_DAYS = 60` of history, `YOUNG_SERVED_HORIZONS = {3}`). The predict
  universe, the prior-row blend map and the forecast writer all read it. They used to carry
  three copies of `WHERE is_backfilled = 1`.
- **`predict` drops every non-h=3 horizon from young items**, and their shadow candidates are
  dropped too. The champion–challenger panel judges the established universe, and a cohort
  joining mid-window would change what its 20 shared dates mean.
- **Panels that size or judge the established band exclude young rows.** These are the
  feedback-factor refit (`served_recalibration._load_panel`) and the PID prereg's served panel
  (`measure_conformal_pid.load_prod_panel`). The prereg's population is the established panel.
  Young items first serve inside its window (forecast dates 09-07..10-18), and the join keeps
  that population as frozen.
- **`collectors/new_item_discovery.py`** runs inside the Aggregator after prices are saved, and
  a failure there cannot cost the collection. A name is inserted when it is priced in >= 2
  sources, is not in the catalog, is not phase-collapsed or a phantom key, and is not in
  `data/discovery_baseline_2026-09-30.txt.gz`. The baseline holds every name priced anywhere
  that day, plus never-priced names that ByMykel dates before 2026-07 or that carry a pre-2026
  event year, which leaves 411 never-priced names discoverable (Cologne 2026, Doodle Lore).
  Without the baseline, the first run would insert the ~15K old names the catalog leaves out on
  purpose. More than 1,500 fresh names in one run is treated as the dumps changing shape:
  nothing is inserted and it logs an error.
- **`release_date` is the first day the archive priced the item**, which is what the 60 days
  count from (NULL on every prod row before this). The July release predates the code, so its
  true first days ship as `data/discovery_seed_2026-07-08_release.tsv` (418 items).

Young rows are `is_backfilled = 0`, so they never join the train universe (archive-derived,
pre-2026).

## End-to-end check

On a scratch SQLite catalog holding prod's 5,542 established items:
- **Discovery** on the real 2026-09-30 dumps inserted 348 items, all from the seed (the other
  70 seed items are priced in < 2 sources today and wait for a second). A rerun inserted 0.
- **The real `forecast_prices.py --predict-only`** at `REPLAY_ANCHOR=2026-09-22` served 317
  young items. It wrote 22,461 forecasts: production's 22,144 plus 317 young h=3 rows, with no
  young row at h=7/14/30. Young h=3 half-width was 7.0% against 5.9% established, and it wrote
  22,144 shadow candidates with 0 young.

## Not changed

The published headline (`prediction_accuracy`) does score young h=3 rows. At ~320 of ~5,850
h=3 rows and ~82% coverage, they move the headline by well under 1pp. Split them out there
before quoting a coverage figure that has to describe the established band alone.

## Next

Re-read young coverage at h=7/h=14 around **120 days** of history, about mid-November for the
July release, from the served panel (h=3) plus a replay (h>=7, which is not served). Lift the
withholding only on that read.

## Also fixed

`measure_conformal_pid.load_prod_panel`'s prior-row read selected `predicted_price_*` from
`item_forecasts`, whose columns are `price_low/mid/high` (verified against prod). The single
~10-23 read would have raised before scoring. It was fixed in its own commit, with a schema test.
