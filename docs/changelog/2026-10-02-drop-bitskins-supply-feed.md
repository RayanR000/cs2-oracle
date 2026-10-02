# Drop the Bitskins supply feed

2026-10-02. Bitskins' public `/market/insell/730` export has been empty since late August. The
feed is removed from `collectors/supply_depth.py`, leaving three scalar feeds (Skinport, Waxpeer,
market.csgo.com) and the lis-skins ladder.

## Evidence

- **The archive's last Bitskins day is 2026-08-23.** `supply-2026-08.parquet` holds 17 Bitskins
  days (185,640 rows). The local copy of `supply-2026-09.parquet` (data repo as of 09-20) holds 0.
- **Every Aggregator run from 09-27 to 10-02 logged it failing**, e.g. run `36954043078`:
  `bitskins: 2 raw items but 0 parsed -- the payload shape has probably changed`. The run stays
  green because one dead feed is allowed by design (`collect()` docstring). That is why it went
  unnoticed for ~40 days.
- **The shape did not change; the list is empty.** Fetched 2026-10-02 19:41 UTC, the response
  is `{"list": [], "created_at": 1790970004}`. `created_at` and `last-modified` were both one
  minute old, so Bitskins still regenerates the file and Cloudflare serves the same empty file
  to every client (browser and `python-requests` user agents alike). The "2 raw items" in the
  error is `fetch_feed` counting the dict's two keys, not listings.
- No other public CS2 listing endpoint was found. API v2 market search needs an account key.

## Why drop rather than wait

- **Its value was always small.** On 2026-08-06 it added +0.73pp of ≥$1 coverage over the other
  scalar feeds and +0.43pp over the full union with the lis-skins ladder
  (`changelog/2026-08-06-supply-depth-collector.md`). The scalar feeds are one feature, not
  several (Spearman 0.65–0.82).
- **No feature reads it.** The order-book features filter to `source == 'lis_skins'`
  (`models/forecaster.py`), and `SKINPORT_VOLUME` / `ORDERBOOK_FEATURES` are default off. So
  nothing trained or served changes, and its loss is not a composition break for any served
  feature.
- **A permanently failing feed hides real failures.** "One feed failed" was true every day, so a
  second dead feed would have looked the same.

The August rows stay in the archive. The schema is long-format by `source`, so dropping a venue is
not a migration. If Bitskins republishes, restoring it means bringing back the parser and one
`Feed` entry (both in this commit's parent).

## Changed

- `collectors/supply_depth.py`: `parse_bitskins` and its `SCALAR_FEEDS` entry removed; the
  module docstring no longer counts four scalar feeds.
- `tests/test_supply_depth.py`: the Bitskins parser test is removed; the dead-feed test uses
  Waxpeer; the row-count test reads `len(SCALAR_FEEDS)` instead of a literal.
- `docs/references/data-sources.md`: the Bitskins row is marked dropped.
