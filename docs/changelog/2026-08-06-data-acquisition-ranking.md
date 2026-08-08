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

> **Updated 2026-08-06 (later the same day).** Two of the arguments below have since been
> measured — the breadth-beats-depth premise and the ByMykel item-age claim. See
> `2026-08-06-breadth-beats-depth-item-age-does-not.md`. Breadth at a fixed budget is
> confirmed but **saturates by ~350 items**, and the inline notes below are corrected
> accordingly.
>
> **Updated again, same day.** A market-relative re-run of the metadata arms **reversed the
> item-age refutation** this banner previously carried. The ByMykel recommendation is now
> *ingest the whole 9-column bundle*, not "the static columns, skip age". Tier-2 item 4
> carries the corrected numbers.

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

**Measured 2026-08-06, and smaller than this section implies.** Breadth at a fixed
200K-row budget is worth **+1.18pp at 14d and +0.73pp at 30d** (150 → 350 items, held-out
≥$1 CV), null at 3d, inconsistent at 7d — but **350 → 700 items adds only +0.10pp**, so the
gain saturates well before the backfill's 5.7×. Production's ~1,343 rows/item is the
150-item arm's density and the backfill would move it to ~923, between the 150- and
350-item arms; interpolating gives roughly **+0.5 to +0.8pp at 14d/30d and nothing at
3d/7d**. Real, but below the 1pp bar used elsewhere in this project. The
breadth-of-*coverage* argument — a training universe that matches what `predict()` scores —
is unaffected and is now the stronger half of the case.
See `2026-08-06-breadth-beats-depth-item-age-does-not.md`.

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

**Measured 2026-08-06 and this recommendation is withdrawn.** Coverage is as good as
claimed — 98% of the deep ≥$1 cohort, median 1,058 days, 92.5% spanning the fold split —
but the feature is not there. A cross-market basis is null on held-out items at 3d, 7d and
14d and **significantly harmful at 30d** (−1.42pp), against an MDE of 0.32–0.95pp that
could have resolved a 1pp effect; the one apparent 7d gain is the sale count, not the
spread. Two further corrections to this section: the budget is **500 requests per day**,
not the 3.7 req/s implied here, so a catalogue backfill is ~52 days and not ~2 hours; and
`avg_price` is float-composition noise at the median 4 sales/day (within-item CV 1.347,
falling to 0.098 only at ≥50 sales/day). The "second opinion on 2013–2025" argument also
fails on its own terms — the series starts 2020-04.
See `2026-08-06-csfloat-basis-refuted.md`.

### Tier 2 — cheap one-shots that fix identified gaps

> **Built 2026-08-06 (later the same day).** The ingest now exists as
> `backend/scripts/ingest_bymykel_metadata.py`, and rebuilding the metadata table from
> scratch reproduced the harness controls exactly while measuring the bundle *higher*
> than the scratchpad build did — raw-label treatment **+2.68 → +3.72pp at 30d**, and
> significantly positive at **all four** horizons under market-relative labels. 3d and
> 14d were measured under raw labels for the first time and are **null**. The "two
> calls" and "a name parser is required" claims below are both wrong in detail. See
> `2026-08-06-bymykel-metadata-ingest.md`. **Then refuted on the production path the
> same evening** — the in-model permutation test scores the bundle at −0.24 / −0.69 /
> −0.06 / +0.79pp, i.e. the model does not use it. Tier 2 item 4 is **closed**. See
> `2026-08-06-bymykel-metadata-refuted.md`.

**4. `ByMykel/CSGO-API`** (crates.json + skins.json, ~14 MB, two calls). Supplies item age
via `first_sale_date`, collection, crate, float caps and StatTrak/Souvenir flags. Item age
is not inferable from a truncated price history. Worth weighting: **rarity is the one
metadata family that ever measured causal here** (+10–12pp within the model), and
`item-metadata.parquet` has rarity NULL on 4,296 of its 8,691 rows — this fills that too.

**Measured 2026-08-06; three corrections, one of them since reversed.** (a) ~~Item age is
not worth ingesting for accuracy.~~ **Withdrawn.** Under raw labels, age's marginal
contribution over the static columns is +0.73pp at 7d and **−0.69pp at 30d** —
sign-inconsistent — and `item_age_days` (observation date − first sale date) carries a
calendar term. But under **market-relative labels** the full 9-column bundle beats the
7-column static subset at **all four** horizons (+0.34 vs +0.13 at 3d, +0.75 vs +0.54 at
7d, +0.99 vs −0.29 at 14d, +1.85 vs +1.29 at 30d), and `age_only` reads significantly
positive at 7d/14d/30d (+0.81 / +0.67 / +0.56, CIs excluding zero). **Ingest the whole
bundle.** The residual uncertainty is narrow but real: age's marginal over the static subset
is still sign-inconsistent (−0.02 / +0.27 / +0.96 / −0.73) and no arm isolates age *inside*
the bundle, so no per-column attribution exists.
(b) The **static** columns (rarity, crate, collection, float caps, StatTrak/Souvenir) *are*
a small real effect: **+0.70pp at 7d, +1.92pp at 30d**, held-out ≥$1 CV, placebo at ~0 —
**+0.54 / +1.29pp** once the label is demeaned by the market factor, so about two thirds of
it is not the market term. Null at 3d and 14d in that regime.
(c) It is **not a two-call one-shot**: 10 further dumps from the same repo are required
(item age reaches 72.5% of the ≥$1 served cohort with them, **45.7% from skins + crates
alone**), joining needs a new `market_hash_name` parser (`models/steam_types.py` cannot do
it), item age is ambiguous for 11.1% of the served cohort, and one file carries three date
formats.

All of (a) and (b) are **held-out-item CV numbers on an 870-item deep ≥$1 universe, not
production DA**. The market-relative figures score *idiosyncratic* direction and are not
comparable to the raw-label ones or to production; the label flag defaults off and was
itself refuted for production the same day
(`2026-08-06-market-relative-labels-refuted.md`). See
`2026-08-06-breadth-beats-depth-item-age-does-not.md`.

**5. Steam `ISteamNews/GetNewsForApp`** — 500 entries back to 2022-03-01, keyless, 0.29 s.
A free CS2 event calendar for `event_correlation_analysis.py`, which currently reads a
Postgres table that has been empty by design since 2026-07-19.

**Built 2026-08-06, and the depth claim above is wrong.** `count=500` returns one
*page*, not the feed. Paging with `enddate` reaches **1,752 unique items back to
2012-03-16** (439 official Valve posts) in four requests and ~2 s — full coverage of
the archive window, so this clears the multi-year half of the reopen bar. Ingested by
`scripts/ingest_steam_news.py` into `price-archive/event-calendar.parquet`, alongside
item 6 below. A case-release flag was attempted and **did not validate** from either
news text (33% recall) or ByMykel `first_sale_date` (disagrees with the news date on
14/14 overlapping cases, median 38 days), so none ships. No lift claimed; the MDE gate
has to be re-derived for a date-level design before anything here is measurable.
See `2026-08-06-date-level-exogenous-ingest.md`.

**6. Historical CNY/USD rates.** `aggregator_buff163` and `aggregator_youpin` are the two
largest live sources by item count and both are CNY-denominated markets, against 7 days of
FX in `exchange-rates-2026.parquet`. Free from ECB / Frankfurter.

**Built 2026-08-06, and demoted to a control column on measurement.** 3,479 daily
quotes from 2012-12-31 in one 1.1 s call, now in
`price-archive/exchange-rates-history.parquet`. The mechanism holds — arbitrage
against BUFF is weak, so a currency move passes through to the USD series rather than
being absorbed — but CNY/USD realises a **0.2307% daily sd (3.66% annualised)**, one
of the lowest-volatility major pairs there is, so it is noise against skin moves at
7d/14d/30d. Ingested because it costs one call, not because it is expected to carry a
result. See `2026-08-06-date-level-exogenous-ingest.md`.

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
| `atalantus/buff-price-history-archive` | Depth-only: rows/item 1,341 → 1,641, `target_items` 521 → **426**. Now measured, not just arithmetic: the unbudgeted 700-item arm carried 5.8× the rows and scored **worse** than the same items at 200K rows at 7d/14d/30d |
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

**This paragraph is the one claim here that has since been measured directly, and it holds
in both directions.** Removing the row cap from a 700-item arm — 5.8× the rows, ~1,518
rows/item — scored −0.83 / −0.59 / −0.15pp at 7d/14d/30d against the same items at 200K
rows, and +0.27pp at 3d. What pays is item diversity per row, not row count. But the
breadth payoff itself saturates by ~350 items, so "buys breadth" is worth sub-1pp here
rather than the multi-pp this section leaves open.

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
- `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md` — the measurement of
  this record's breadth premise and its ByMykel item-age claim
- `docs/changelog/2026-08-06-steam-listing-backfill-and-phantom-items.md`
- `docs/changelog/2026-08-06-price-archive-compaction.md`
- `docs/research/lis-skins-snapshot-plan.md` — read before building the lis-skins collector
