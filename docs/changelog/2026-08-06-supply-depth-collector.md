# Supply depth is collected daily now — and nothing about it is measured

**Date:** 2026-08-06
**Plan:** `docs/research/lis-skins-snapshot-plan.md`
**Context:** `docs/changelog/2026-08-06-free-bulk-supply-depth-feeds-exist.md`,
`docs/changelog/2026-08-06-retroactive-supply-feeds.md`
**Commits:** none — the whole change sits uncommitted in the working tree on
`docs/refresh-2026-08-05` (tip `4cf95fc`). The workflow edit is **not committed and
not pushed**, so CI has never executed it.

## Read this first: no lift is claimed, and the gate has not been cleared

This collector exists to accumulate a series. It does not improve the model and no
measurement here says it will. Every prior reason for scepticism is intact:

- Trade volume correlates with forward returns at **|r| < 0.002 across 4.47M rows**
  (`docs/research/volume-data.md`).
- The listing-count **level** is a ~0pp liquidity signal. Only the change/velocity
  variant was ever argued predictive, at a calibrated **+1–2pp**
  (`docs/changelog/2026-07-16-drop-supply-depth.md`).
- `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` (`backend/models/forecaster.py:269`)
  still excludes `supply_*` from training. **The served model is byte-identical on this
  change.** Nothing was retrained, nothing was re-scored, no accuracy number moved.

The pre-registered test rule is unchanged and is *not* satisfied: run
`backend/scripts/compute_mde.py` first, then a **permutation** A/B on the
change/velocity features, never a plain A/B. If the MDE exceeds ~2pp the harness cannot
resolve a +1–2pp effect and the accumulated data buys nothing measurable. **An MDE
measurement was commissioned and its result is not available as of this entry.** Do not
read "the collector shipped" as "the gate cleared".

## What landed

| Path | Role |
|---|---|
| `backend/collectors/supply_depth.py` | the collector; 4 scalar feeds + the lis-skins ladder |
| `backend/scripts/run_supply_depth.py` | entry point (`--no-ladder`, `--dry-run`, `--date`, `--archive-dir`) |
| `backend/tests/test_supply_depth.py` | 20 tests |
| `backend/scripts/probe_supply_feeds.py` | the pre-build verification probe (read-only, 590 lines) |
| `backend/scripts/run_task.py:54` | `supply_rows` registered in `ROW_COUNT_FIELDS` |
| `backend/requirements.txt:7-12` | `brotli>=1.1.0` pinned, with the 406 trap written next to it |
| `backend/models/forecaster.py:1609-1681` | `_fetch_supply_snapshots` reads the archive, not Postgres |
| `.github/workflows/aggregator-update.yml:129-153` | the daily step |

Output is one row per `(item_slug, snapshot_day, source)` in
`price-archive/supply-YYYY-MM.parquet` — long format, keyed by `source`, ladder columns
NULL for the scalar feeds. Long rather than one column per marketplace so that adding or
dropping a venue is not a schema migration; `skinport_quantity` in the old
`supply_snapshots` table is the fossil showing what the wide shape cost, a column added
for a second market that stayed 100% NULL for its whole life.

`item_slug` is `market_hash_name`, so the table joins to `prices-*.parquet` on name with
no DB lookup. Nothing in the collector imports `database` —
`backend/scripts/run_supply_depth.py:13-16` says why, and it is the `.env`-points-at-prod
gotcha.

## The step is placed between the archive append and the archive publish

`.github/workflows/aggregator-update.yml:129-153`, `continue-on-error: true`, sitting
after *Append to Parquet archive* and before *Publish updated archive (flat history)*.
That ordering is the point: the day's supply rows land in the same orphan commit as the
day's prices. The previous supply table failed on exactly this — written to Postgres and
a local Parquet, never published to the data repo, therefore invisible to everything that
reads the archive.

`continue-on-error` because Price Forecast and Backtest Accuracy chain off this
workflow's success, and an unproven experiment must not be able to take down the daily
forecast. It suppresses only a *total* outage: `collect()` already exits non-zero solely
when every feed fails, and per-feed counts go to the log either way.

The step passes `AGGREGATOR_SNAPSHOT_DATE: ${{ env.SNAPSHOT_DATE }}`, pinning the same
day label the price rows got.

## First production pull: 84,408 rows, 4 of 5 feeds, one feed blocked

Measured directly from `price-archive/supply-2026-08.parquet` (1.6 MB on disk, 1 distinct
`snapshot_day` = 2026-08-06, **30,330** distinct `item_slug`):

| Feed | Items | Median `listing_count` | Max | Elapsed |
|---|---:|---:|---:|---:|
| `market_csgo` | 27,570 | 13 | 16,178 | 0.6 s |
| `lis_skins` | 23,879 | 14 | 56,917 | 364.5 s |
| `waxpeer` | 22,039 | 12 | 58,537 | 0.6 s |
| `bitskins` | 10,920 | 3 | 33,277 | 0.5 s |
| `skinport` | — | — | — | 0.3 s → **HTTP 403** |

Counts, medians and maxima are from the stored Parquet; elapsed times are the run's
`per_feed_seconds` summary and are not re-derivable from the file.

The run's own summary reported 30,311 distinct items against the file's 30,330. The cause
is that **day 1 merges two pulls, not one**: a scalar-feeds-only test run preceded the full
run by ~20 minutes, and the writer's dedup key is `(item_slug, snapshot_day, source)`, so
items present in the earlier pull but absent from the later one survive rather than being
overwritten. `market_csgo` is the only source carrying two distinct `collected_at` values
(27,563 items in the first pull, 27,517 in the second, 27,570 in their union) — the ~50
item difference is live-feed churn between the two reads. Every other source shows a single
`collected_at` because the later pull's item set covered the earlier one's.

This is day-1-only and self-corrects: a normal daily run is a single pull. It is recorded
because the union semantics are a deliberate property of the dedup key — re-running a day
adds any items the first attempt missed rather than replacing the day wholesale — and
because it means `collected_at` is not unique per `(snapshot_day, source)`. Anything that
assumes one timestamp per source-day must group, not assume.

Coverage against `price-archive/prices-2026-08.parquet` (41,381 distinct `item_slug`; the
≥$1 cohort is 27,502 items by `avg(mean_price) >= 1.0`, matching
`api/serving_policy.py`'s `MIN_SERVED_PRICE_USD = 1.0`):

- **29,527 items / 71.4% of the universe**
- **19,543 items / 71.1% of the ≥$1 cohort**

Marginal contribution of each feed to the ≥$1 cohort — union of all four minus the union
without it, which is the number that decides whether a feed earns its daily request:

| Feed | Items only it supplies | pp of ≥$1 cohort |
|---|---:|---:|
| `lis_skins` | 1,428 | +5.19 |
| `market_csgo` | 1,067 | +3.88 |
| `bitskins` | 200 | +0.73 |
| `waxpeer` | 117 | +0.43 |

Note the ≥$1 denominators differ between this entry (27,502) and the two 08-06 research
entries (25,459 and 26,428) because each computed the cohort with a different aggregation.
**Percentages across the three are not comparable; the raw matched counts are.**

## `volume` on market.csgo.com is a listing count, not trade volume

This mattered enough to be the probe's load-bearing question, because in this repo
`volume` otherwise means completed-sale count — the quantity refuted at |r| < 0.002. If
this field were trade volume the feed would be worthless here. Four independent checks,
run by `backend/scripts/probe_supply_feeds.py`, say it is inventory:

1. **Shape.** Heavy right tail: probe pull median 13, p99 813, max 17,028. The stored
   snapshot, a separate live pull hours later, gives median 13, p99 799, max 16,178 — the
   same shape, and the drift between pulls is what live feeds do.
2. **Cross-feed rank correlation.** Spearman 0.56 against Waxpeer's `count`, which is
   documented as a listing count.
3. **Against a genuine sale count.** For all 9 canary items tested, market.csgo.com's
   `volume` exceeds CSFloat `/api/v1/history/<name>/graph`'s daily completed-sale count,
   by a multiple that **widens as liquidity falls** — 1.4x on Kilowatt Case to 15x on
   Glock Fade. That is the inventory ≈ trade rate × dwell time signature; a trade count
   would not behave that way.
4. The site exposes a per-physical-listing full export, consistent with (1)–(3).

The reasoning is duplicated in the parser docstring
(`backend/collectors/supply_depth.py:311-324`) so a future reader hitting the field name
does not have to re-derive it.

## Skinport has a second block, and it is not the brotli trap

`docs/references/data-sources.md` records Skinport's HTTP 406 as caused by a missing
`Accept-Encoding: br` — the misdiagnosis that had the API written off as
"Cloudflare-dead". That finding is correct and `brotli>=1.1.0` is now pinned, with
`_probe_brotli()` (`backend/collectors/supply_depth.py:155-170`) failing loudly rather
than letting the feed vanish behind a header bug.

It is not the only Skinport failure mode. **Skinport's WAF returns HTTP 403 with an HTML
challenge page to Cloudflare-owned egress IPs (AS13335)**, and no header change fixes it.
Measured 2026-08-06 with brotli decoding correctly and across every User-Agent tried. The
probe reports this as a distinct `blocked_waf` status specifically so it cannot be folded
back into "the 406 trap" or, worse, into a fabricated zero.

Consequence: **the Skinport coverage figures in the two 08-06 research entries
(24,878 items / 15,875 of the ≥$1 cohort) are unverified from this session's egress** —
they were measured from a residential IP and could not be reproduced here. The collector
keeps the feed configured; it contributed 0 rows on 08-06.

## Waxpeer's coverage does not reproduce, and it is near-redundant

`2026-08-06-free-bulk-supply-depth-feeds-exist.md:44` reports Waxpeer matching 16,081
items of the ≥$1 cohort. This pull matched **15,016** — 1,065 fewer items, on the raw
count, which is the figure that survives the cohort-denominator change. Against the 27,502
denominator used here that is 54.6%, not the 63.2% in that entry; both effects are in play
and the raw counts are the comparison to make.

Its marginal value is small either way. Restricted to the scalar feeds actually available
(Skinport blocked), `market_csgo` + `bitskins` covers 17,914 of the ≥$1 cohort (65.1%) and
adding Waxpeer takes it to 18,115 (65.9%) — **+201 items, +0.73pp**. Against the full
four-feed union including the lis-skins ladder it is +117 items, +0.43pp.

Kept anyway, on two grounds that are cost arguments and not signal arguments: it costs
0.6 s, and it is a different venue class (P2P bot inventory) from a marketplace-held
book, so the redundancy could be regime-dependent in a way one day cannot show.

## The day label comes from the aggregator's resolver, not the wall clock

`collect()` calls `collectors/snapshot_date.py::resolve_snapshot_date`
(`backend/collectors/supply_depth.py:600-607`). These feeds are live snapshots with no
dump boundary of their own, so on their own terms "today" would be defensible — but a
supply row is only useful joined to the price row from the same run, and the price side
labels its day from the CSGOTrader dump boundary at ~21:40 UTC
(`DUMP_PUBLISHED_HOUR_UTC = 22`).

Caught live rather than in review: at **00:29 UTC on 08-07** the resolver returned
**2026-08-06**, matching the price rows' label. The wall clock would have stamped supply
08-07 against prices 08-06, and the feature join would have missed every item — silently,
because a missed join reads as "no supply data for this item" rather than as an error.
`test_snapshot_day_matches_the_aggregator_not_the_wall_clock` pins it.

## `sell_listings` is a MAX across marketplaces, not a sum

`backend/models/forecaster.py:1654-1660` groups the archive rows by
`(item_slug, snapshot_day)` and takes `max(listing_count)`.

A sum is wrong twice. The feeds overlap heavily — Spearman 0.65–0.82 between them, and
`market_csgo` is close to a superset of Waxpeer — so a sum double-counts the same
inventory. Worse, a sum makes the series lurch whenever a feed drops out: losing one
marketplace would read as a real supply crash. That is fatal specifically because the
variant argued to be predictive is the **change**, so a feed outage would present as a
move. A max degrades gracefully; today's Skinport 403 lowers the level slightly instead of
manufacturing a crash.

## The dead `supply_snapshots` table is out of the read path

`_fetch_supply_snapshots` used to query Postgres `supply_snapshots`. That table holds one
stale day (2026-07-15, 35,037 rows) from the Steam burst scraper, which cannot run from CI
at all — Steam 429s runner IPs — and it was never published to the data repo. It now reads
`price-archive/supply-*.parquet`.

`skinport_quantity` is retained and hard-set to 0 (`forecaster.py:1668`) so
`_add_supply_depth_features` keeps its column contract. Dropping it would change the
feature set as a side effect of a data-source change, which is not what this change is.

The local `price-archive/ops/supply_snapshots.parquet` was deleted; the 35,037 rows in
prod Postgres were left in place. `docs/architecture/data.md` now marks the table
deprecated.

## Depth is measured from a trimmed anchor, and both open questions are instrumented

- **Anchor.** `docs/research/lis-skins-snapshot-plan.md:52-54` recorded 358 items whose
  lowest ask is >10x the archive price, worst case 1,154x. `depth_5pct` / `depth_10pct`
  are therefore counted from `p05_ask`, not `min_ask` (`DEPTH_ANCHOR_Q = 0.05`,
  `supply_depth.py:115`). `min_ask`, `p05_ask`, `p25_ask` and `median_ask` are all stored
  so the anchor is re-choosable downstream without re-collecting.
  `test_depth_anchor_resists_a_single_mispriced_listing` asserts the property.
- **`created_at`: listing age or last-reprice time?** A single snapshot cannot tell. The
  discriminator is whether `created_at` moves for a listing `id` that persists across
  consecutive days, so `listing_id_digest` — an order-independent SHA-1 of the item's
  listing-id set — is stored for exactly that diagnostic
  (`supply_depth.py:441-442`). Day-1 reading: median `age_median_days` across lis-skins
  items is **6.2 days**. That number is uninterpretable until the question is settled,
  which needs at least a second consecutive day.
- **The raw ladder is not stored.** 173 MB/day of individual listings to support an
  untested feature set would dominate an 88 MB archive holding 20.7M price rows.
  Per-item aggregates only.

## Guards against this repo's specific failure modes

Silent-zero-success is the failure this codebase keeps repeating — the Steam supply
scraper reported green while storing nothing for 16 days. The collector is written against
that:

- Every fetch either returns rows or raises. An empty payload is a `SupplyFeedError`, not
  an empty result (`_get_json`, `supply_depth.py:193-194`).
- A feed that answers with raw items but parses to zero rows raises, on the grounds that
  the payload shape changed (`fetch_feed`, `supply_depth.py:497-501`).
- Unparseable counts are dropped, never zero-filled — `listing_count = 0` is a real
  observation and coercing a parse failure into it would fabricate exactly the signal the
  collector exists to measure (`_scalar_rows` docstring,
  `test_unparseable_count_is_dropped_not_zero_filled`).
- `collect()` raises when *every* feed fails, which is indistinguishable from a
  network-level block and must never exit 0.
- `supply_rows` is registered in `run_task.ROW_COUNT_FIELDS`, and
  `test_collect_reports_row_count_field_for_the_guard` asserts the registration rather
  than trusting it.
- Writes are idempotent per `(item_slug, snapshot_day, source)`, so a partial run is
  re-runnable rather than needing a repair step.

## Verification

- `backend/tests/test_supply_depth.py` — 20 tests, reported passing by the implementing
  run. Not re-executed for this entry (this record is written read-only).
- `backend/tests/test_forecaster.py` — 154 tests, reported passing after the
  `_fetch_supply_snapshots` rewrite.
- Coverage, per-feed item counts, medians, maxima and the marginal-contribution table
  above were recomputed for this entry with DuckDB against
  `price-archive/supply-2026-08.parquet` and `price-archive/prices-2026-08.parquet`.
- The four-way `volume` check, the Skinport WAF diagnosis and the cross-feed Spearman
  figures come from `backend/scripts/probe_supply_feeds.py` and are not re-derivable from
  the stored file.
- **Production: one manual run from a residential IP, 2026-08-06.** Zero CI runs.

## Still open

- **Never verified from a GitHub runner — the top follow-up.** The feeds are non-Steam
  hosts, so the runner-IP 429 that killed `supply-scraper.yml` should not apply, but that
  is an inference, not a measurement. The workflow change is uncommitted and unpushed; a
  collector that is silently empty in CI is precisely the shape this repo keeps hitting,
  and `continue-on-error` means it would fail quietly.
- **Skinport unresolved**, pending an egress IP that is not Cloudflare-owned. Its prior
  coverage numbers stay unverified until then.
- **lis-skins costs 364.5 s/day**, by far the heaviest step in the aggregator workflow.
  `--no-ladder` drops it, at the cost of the ask-ladder shape and listing age — the only
  quantities here that are not already refuted.
- **`created_at` semantics**, answerable from the second consecutive day via
  `listing_id_digest`.
- **Whether any of this is measurable at all**, blocked on `compute_mde.py`.
- Skinport `/v1/sales/history` (90-day retroactive volume windows) is not collected; it is
  trade volume and falls under the |r| < 0.002 audit.

## Docs touched

- `docs/references/data-sources.md` — status rows for market.csgo.com, lis-skins, Waxpeer
  and Bitskins moved off "Not integrated"; Waxpeer and Bitskins rows added (they had none);
  the Skinport section now records the AS13335 WAF 403 as a second, distinct block.
- `docs/architecture/data.md` — `supply-YYYY-MM.parquet` added to the archive tree and the
  storage table.

## Related

- `docs/research/lis-skins-snapshot-plan.md` — the plan this implements
- `docs/changelog/2026-08-06-free-bulk-supply-depth-feeds-exist.md` — the availability
  refutation
- `docs/changelog/2026-08-06-retroactive-supply-feeds.md` — the retroactive-window finding
- `docs/changelog/2026-07-16-drop-supply-depth.md` — the original decision; its accuracy
  leg still stands
- `docs/changelog/2026-07-15-supply-scraper.md` — the collector this replaces
- `docs/research/volume-data.md` — the trade-volume audit that still stands
