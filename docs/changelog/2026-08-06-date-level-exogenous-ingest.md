# The first date-level data in the archive — an event calendar and FX history

**Date:** 2026-08-06
**Change:** two new one-shot ingests. No collector, no workflow step, no model change.
**New tables:** `price-archive/event-calendar.parquet`,
`price-archive/exchange-rates-history.parquet`
**New scripts:** `backend/scripts/ingest_steam_news.py`,
`backend/scripts/ingest_fx_history.py`
**Tests:** `backend/tests/test_event_calendar.py` (27)

## Read this first: no lift is claimed, and the gate has not been run

`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` (`backend/models/forecaster.py:269`)
excludes everything here. **The served model is byte-identical on this change.**
Nothing was retrained, nothing re-scored, no accuracy number moved.

The MDE gate is **not** satisfied. `scripts/compute_mde.py` has to be run before any
result off these tables means anything, and then a **permutation** A/B, never a plain
one.

> **Corrected 2026-08-07.** This section originally argued the per-item floor
> (1.15–7.13pp) was "the wrong denominator entirely" because a date-level column's
> effective N is the count of distinct dates. That framing was wrong on its premise:
> `backtest/paired_mde.py` already resampled at a grain coarser than rows. The real
> defect was that it clustered on `forecast_date`, which is **not** independent — every
> date in a fold's validation window is scored by one fitted model — so its intervals
> were too *narrow*. Fixed 2026-08-07; see
> `2026-08-07-training-item-universe.md` for the fix's blast radius and the three A/Bs
> it invalidates.
>
> The operative floor for a date-level experiment is therefore the **fold-clustered
> 2.21–3.69pp** measured on the breadth A/B at 25–26 folds, not 1.15pp and not the
> 0.008pp a seed-only placebo returns at h=3. A seed placebo perturbs only the RNG, so
> its two arms are near-identical models with almost no contrast to bound; it measures
> reseeding noise, not the floor for an arm that changes the training data. **Nothing
> in these two tables can be resolved below roughly 2pp**, which is the practical
> answer to whether the date-level axis is testable here: not unless the effect is
> large.

## Why date-level, when the item-level candidates are not exhausted

Because they are. ByMykel metadata, the CSFloat basis, the six price primitives,
float composition, Reddit sentiment, trade volume — every feed ingested and refuted
in this project is keyed on the item. Two findings explain the whole streak:

- Demeaning labels by the market factor drops accuracy below a constant call at
  every horizon (`2026-08-06-market-relative-labels-refuted.md`).
- A constant always-down call beats the model on every stored date.

The variance left to explain is a date-level common factor. A per-item feature has
approximately zero loading on a date-level common shock, so no amount of better item
attributes can reach it. Nothing in `price-archive/` was keyed on the date alone
before this change. These two tables are the first.

That is an argument for the *axis*, not a prediction about the result.

## Two corrections to `data-sources.md` / the acquisition ranking

**1. `ISteamNews` is not 500 entries deep, and does not start 2022-03-01.**
`2026-08-06-retroactive-supply-feeds.md:127-130` and the Tier-2 item 5 entry in
`2026-08-06-data-acquisition-ranking.md:163-166` both record the feed as "500 entries
back to 2022-03-01". That is the *unpaginated first page*. Passing `enddate` walks
backwards: **four pages, ~2 s, 1,752 unique items back to 2012-03-16**, of which 439
are official Valve announcements. It covers the archive window (2013-08-14 →
2026-08-04) end to end, which is what the reopen bar asks for.

**2. The FX argument is real in mechanism and small in magnitude.** BUFF163 and
YouPin are CNY-denominated and reach the archive already converted to USD, and
arbitrage against Western venues is weak — the 10–30% BUFF discount to Steam is
persistent, because withdrawal rails are the binding friction for non-Chinese users.
Weak arbitrage means the currency move passes through instead of being absorbed. But
measured over this table's own 3,478 daily returns:

| | CNY/USD |
|---|---|
| daily sd | **0.2307%** (annualised 3.66%) |
| last 504 trading days | 0.1711% (annualised 2.72%) |
| largest 1-day move | 1.99% |
| median \|30-day move\| | 0.77% (p95 3.30%) |

USD/CNY is a PBoC-managed float and one of the lowest-volatility major pairs there
is. Against skin moves this is noise at 7d/14d/30d. The one place it could bite is
3d direction on sticky, illiquid items, where a common 0.23% shift is a real
fraction of a typical daily move and can flip labels sitting near zero — and 3d is
where the gate's MDE is tightest. FX ships as a **control column, not a bet**.

## What is deliberately absent: a case-release flag

Both candidate sources were measured and **neither validated**, so no
`is_case_release_day` column exists.

- **Regex over announcement text** finds 14 of the 42 known cases (**33% recall**)
  and misfires both ways — `Removed Gallery Case` is a drop-pool *removal* scored as
  a release, and `Fixed a case where...` is prose.
- **ByMykel `first_sale_date`** has 42/42 coverage but disagrees with the first news
  mention on **all 14** overlapping cases. Median gap **38 days**, range −76 to
  +1878. Zero agree within ±3 days.

| Case | ByMykel | 1st news mention | Δ days |
|---|---|---|---:|
| Danger Zone | 2018-11-29 | 2018-12-07 | 8 |
| Revolution | 2023-01-31 | 2023-02-10 | 10 |
| Spectrum | 2017-03-01 | 2017-03-16 | 15 |
| Kilowatt | 2024-01-16 | 2024-02-07 | 22 |
| Recoil | 2022-05-30 | 2022-07-04 | 35 |
| Fever | 2025-02-04 | 2025-03-31 | 55 |
| Dreams & Nightmares | 2021-11-17 | 2022-01-21 | 65 |
| Gallery | 2024-07-01 | 2024-10-02 | 93 |
| Shattered Web | 2019-06-12 | 2019-11-19 | 160 |
| Revolver | 2015-11-30 | 2015-09-15 | −76 |

The two fields are not the same quantity and neither is verifiably the day a case
became purchasable. Crate dates therefore ship under their source's own field name
(`crate_case_first_sales`), not as a release flag. **The archive itself can arbitrate
this** — first appearance of a case's items in `prices-*.parquet` is the market-entry
date in the terms the model actually sees — and that is left open below.

## Schemas

`event-calendar.parquet` — 4,967 rows, one per day, 2013-01-01 → 2026-08-07, 65.7 KB.

| Family | Columns |
|---|---|
| Valve posts | `valve_announcements`, `_7d`, `_30d`, `days_since_valve_announcement` |
| Press | `press_articles`, `_7d`, `_30d` |
| Cases | `crate_case_first_sales`, `_365d`, `days_since_crate_case_first_sale` |
| Capsules | `crate_capsule_first_sales`, `_30d`, `days_since_crate_capsule_first_sale` |
| Collections | `collection_releases`, `days_since_collection_release` |

Valve posts (`feed_type == 1`) and syndicated press (`feed_type == 0`) are counted
separately on purpose: a games-press article is a reaction to the market as often as
a cause of it, and merging them would put a reactive series in the same column as a
causal one. Cross-posted announcements are deduped on `(day, title)`, not just `gid`
— Steam serves one announcement under several gids, which would otherwise inflate
the count on exactly the days the column matters.

`exchange-rates-history.parquet` — 18,251 rows, long `day, currency, rate, is_filled`,
CNY/EUR/GBP/RUB, 144.8 KB. Written as a **separate file** because the daily
aggregator owns `exchange-rates-2026.parquet` (`collectors/pipeline.py:373-384`) and
a second writer on that file would collide with the daily chain.

Both join the archive's 4,735 distinct days at **100%**, and the columns are
non-degenerate on those days — 462 distinct values for
`days_since_valve_announcement`, 543 for `days_since_crate_case_first_sale`. That is
worth stating against the inventory's finding that `volume` is identically zero for
111 days and `min_price`/`max_price` are 100% NULL outside 2026-03/04.

## The RUB trap, and why the fill is capped

The ECB **stopped quoting RUB on 2022-03-01**. An uncapped forward-fill carried one
frozen number across the next four years — 2,621 of RUB's 4,966 days — and a
constant reads to a model as a perfectly stable feature, not as missing data. Fill is
capped at `MAX_FILL_DAYS = 7`, enough for any real ECB closure, so a dead series
simply stops (RUB 4,966 → 3,353 days) instead of becoming a fabricated one. Any
currency added here needs the same per-currency quote-range check.

This is the same family as the supply scraper's 16 green days storing nothing and
the event-correlation script reading an empty table: the failure is not a crash, it
is a plausible wrong number.

## Causality

Every column is backward-looking as of its own `day`. Trailing windows include the
current day; `days_since_*` reads only events on or before `day`, and is NULL rather
than 0 before the first event, since zero-filling would assert an event on the
calendar's first day. There is no `days_until_*` column — a case release is knowable
in advance only if it was announced in advance, and this table does not track
announcements separately from the events they describe.

`test_truncating_the_future_changes_nothing` is the invariant: building the calendar
over a short span must be byte-identical to building it long and slicing. A date-level
table is where lookahead is easiest to introduce and hardest to see — a centred window
or a back-fill still produces a plausible-looking series.

## Still open

- **The MDE gate**, re-derived for a date-level design. Nothing here is measurable
  until that is known, and it may come back saying nothing here is measurable.
- **Arbitrating the case-release date from the archive** — first appearance of a
  case's items in `prices-*.parquet`. This is the one source that defines the date in
  the terms the model sees, and it would settle the 38-day disagreement above.
- **Whether the FX pass-through is present at all.** The mechanism assumes csgotrader
  converts CNY at a live daily rate. If it uses a fixed rate, there is nothing to
  remove. Regress a market-wide daily return on the CNY return — note this cannot be
  done by comparing a Chinese source against a Western one, because BUFF is the
  global price reference and the term propagates into Western quotes too.
- **Neither ingest is wired into the daily chain.** Both are one-shots, like
  `ingest_bymykel_metadata.py`. The event calendar goes stale as new updates ship; FX
  goes stale daily. Re-run before any measurement.

## Related

- `docs/changelog/2026-08-06-data-acquisition-ranking.md` — Tier 2 items 5 and 6, the
  two candidates this implements; its ISteamNews depth claim is corrected above
- `docs/changelog/2026-08-06-bymykel-metadata-refuted.md` — the most recent item-level
  refutation, and the reason for changing axis
- `docs/changelog/2026-08-06-market-relative-labels-refuted.md` — why removing the
  *whole* market factor fails where removing an identified slice of it might not
- `docs/research/accuracy-opportunities.md` — the stop banner and the reopen bar
