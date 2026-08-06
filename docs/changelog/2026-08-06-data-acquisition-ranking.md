# Data inventory audited, and the acquisition candidates ranked against the reopen bar

**Date:** 2026-08-06
**Change:** no code change. Audit and prioritisation only.
**New doc:** `docs/references/data-inventory.md` — the full coverage measurement this
record draws on. Numbers are not repeated here; read that first.

A full inventory of `price-archive/` was taken to answer two questions: what data exists
and how much of the market it covers, and which of the many uncollected feeds in
`data-sources.md` are actually worth building a collector for.

The second question has a bar already set, by the stop banner in
`docs/research/accuracy-opportunities.md`:

> the bar is a data source that is genuinely new (not inferable from price history) *and*
> has multi-year history so it can be trained and A/B'd over the archive window.

This record ranks the candidates against that bar. **No lift is claimed for any of them.**
The measurement floor (1.15pp at 3d, 2.76–7.13pp at 7d/14d/30d) is unchanged, and
`compute_mde.py` still gates any experiment.

## What the audit found that changes the picture

Three findings bear directly on the ranking. Full detail in `data-inventory.md`.

**1. Breadth is solved; depth is bimodal.** The archive holds ~38,600 real items against a
~40,000-item catalogue — ~96%. But **every item with ≥180 days of history is in the 5,542
pre-2026 cohort**; the other 36,183 were first seen in 2026 and cap at ~136 days. There is
no middle tier. Median history is 128 days overall and 1,507 days inside the deep cohort.

**2. The served cohort is inverted against the archive.** 63.9% of items with recent data
are ≥$1, but only **16.4%** of the 8,691 items actually forecast are. The gate selects
against the cohort that `classifier_accuracy_ge1` scores.

**3. One column carries information.** Over the last 90 days `mean_price` is the only
non-degenerate field: `volume` is identically zero (111 days), `min_price`/`max_price` are
100% NULL outside 2026-03/04, supply is one day, bid/ask is 25 days deep. The
multi-source era is **25 days deep**, not the 4.5 months the buff163/youpin/csfloat date
ranges suggest — those carry a 90-day hole.

## Ranking

### Tier 1 — clears the reopen bar

**1. Finish the Steam listing-page backfill.** `backend/scripts/backfill_steam_listing_history.py`,
262 of 29,933 targets collected. Daily median price plus real traded volume back to 2013,
no cookie, ~4 items per request.

Its value is **not** a feature — traded volume is refuted at |r| < 0.002 and that audit
stands. It is the only source that attacks finding (1) and finding (2) at once, because
it *lowers* rows/item instead of raising it: 5,542 → ~31,590 items, 1,341 → 923 rows/item,
`target_items` at the 700K budget **521 → 758 (+45%)**. Treat 5.7× breadth as an upper
bound; the sample is biased toward items that resolved.

Blocked on egress, not code. The residential IP is soft-blocked with no decay over 5 h,
and both datacenter VPN exits were served a stripped shell on request #1 — Steam filters
by **ASN class**, so switching VPN region cannot help. A cellular hotspot is the one
untested residential-classified route. Run at `--delay 8`, and verify
`"wallet_currency": 1` before trusting any new egress — a non-US exit silently writes EUR
against a USD-calibrated fee divisor.

**2. lis-skins full export.** 2,297,323 listings, 25,660 items, one 44 s call, free. The
only feed that makes supply depth computable **retroactively**: `created_at` is 100%
populated, so listing age, ask-ladder shape and supply inflow rate exist on the first
pull. That directly retires the accumulation half of the 2026-07-16 drop decision, which
declined supply depth partly because the change/velocity variant needed 30+ days of
forward collection.

Open first: `docs/research/lis-skins-snapshot-plan.md` carries an unresolved `created_at`
ambiguity (listing creation vs. relist) that decides whether the age features mean
anything, plus the collector constraints (never run from `backend/`, verify from a GitHub
runner, return row counts for the zero-row guard).

**3. CSFloat `/api/v1/history/<name>/graph`.** Free, no auth, daily completed-sale average
and count **from 2020-04**. A second independent multi-year daily transaction series,
per-item. `data-sources.md` records it as "viable and unblocked; simply not integrated" —
it is under-rated there. It is the cheapest way to get a second opinion on the 2013–2025
window, which the archive currently covers with a single unattributed series (45% of all
rows have `source IS NULL`).

### Tier 2 — cheap one-shots that fix identified gaps

**4. `ByMykel/CSGO-API`** (crates.json + skins.json, ~14 MB, two calls). Supplies item age
via `first_sale_date`, collection, crate, float caps and StatTrak/Souvenir flags. Item age
is not inferable from a truncated price history. Worth weighting: **rarity is the one
metadata family that ever measured causal here** (+10–12pp within the model), and
`item-metadata.parquet` has rarity NULL on 4,296 of its 8,691 rows — this fills that too.

**5. Steam `ISteamNews/GetNewsForApp`** — 500 entries back to 2022-03-01, keyless, 0.29 s.
A free CS2 event calendar for `event_correlation_analysis.py`, which currently reads a
Postgres table that has been empty by design since 2026-07-19.

**6. Historical CNY/USD rates.** `aggregator_buff163` and `aggregator_youpin` are the two
largest live sources by item count and both are CNY-denominated markets, against 7 days of
FX in `exchange-rates-2026.parquet`. Free from ECB / Frankfurter.

### Tier 3 — repairs, no new source needed

- **Volume has been identically zero for 111 days** and `price_resolution.py:249` still
  reads it. Either restore it — Skinport `/v1/sales/history` is free, 1.9 s, 36,004 items,
  89% of the ≥$1 cohort — or stop reading it.
- **`aggregator_steam_17mafo` has been dead 25 days** with nothing flagging it. Same shape
  as every other entry in the collectors-fail-silently family.
- **No 30-day labels exist for 2026.** Every h=30 metric rests on 5,461 usable rows from
  the single backdated 2025-12-01 batch.
- **The ~3,145 slug-keyed duplicate items** (686K phantom rows) inflate every item count
  in the archive. `scripts/purge_phantom_items.py` exists for this.
- **Four missing archive days** (2026-07-27, 07-30, 08-02, 08-03), all cron drift.

### Explicitly not worth pursuing

| Candidate | Why not |
|---|---|
| Trade volume as a predictive feature | |r| < 0.002 across 4.47M rows. Audit stands |
| `atalantus/buff-price-history-archive` | Depth-only: rows/item 1,341 → 1,641, `target_items` 521 → **426** |
| CSMarketAPI | Quota permanently burned across all 5 free keys |
| Any paid feed ($9.99–$179/mo) | Skinport + lis-skins + market.csgo.com cover the same ground free |
| Wayback Machine for listing history | Zero captures across all six endpoints probed |

## The constraint that bounds all of it

**More data does not move accuracy on its own.** The training row budget
(`TRAIN_FEATURE_ROWS`, 700K) binds, and raising it costs 4.5× and was declined. More rows
for the *same* 5,542 items makes things worse — that is exactly why the BUFF archive was
rejected. The argument for the Steam listing backfill is specifically that it buys
**breadth**: a training universe that matches what `predict()` scores, rather than twelve
years of history on 14% of the catalogue, 84% of it sub-dollar.

And the measurement floor still applies. With an MDE of 1.15pp (3d) to 7.13pp (30d), a new
feature group has to be genuinely large to be provable at all. Of everything above,
**lis-skins listing age / ask-ladder shape is the only candidate with a mechanism argument
for clearing that floor** — and it is untested, not supported. Run `compute_mde.py` before
building anything, and use a permutation A/B, never a plain one.

## Docs touched

- `docs/references/data-inventory.md` — **new**; the coverage audit
- `docs/references/data-sources.md` — the 2026-07-16 "Volume Data Status" section was
  measured against an 11.09M-row archive and reports the volume series running to
  2026-03-29; corrected to point at the inventory, which measures 20.76M rows and volume
  identically zero since 2026-04-16. Also fixed a stale `snapshots-YYYY.parquet` reference
  in the HF merge-script description.

## Related

- `docs/research/accuracy-opportunities.md` — the stop banner and the reopen bar
- `docs/changelog/2026-08-06-retroactive-supply-feeds.md` — the lis-skins and Skinport
  measurements
- `docs/changelog/2026-08-06-steam-listing-backfill-and-phantom-items.md`
- `docs/changelog/2026-08-06-price-archive-compaction.md`
- `docs/research/lis-skins-snapshot-plan.md` — read before building the lis-skins collector
