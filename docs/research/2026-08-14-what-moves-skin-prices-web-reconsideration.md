# What moves skin prices — a web cross-check of our null streak

**Date:** 2026-08-14
**Method:** four parallel web-research passes (marketplace/analyst writeups, trading
communities, and academic literature on illiquid-asset return predictability), each briefed on
exactly how we test — directional (up/down) forecasting at 3/7/14/30d, LightGBM, ≥$1 served
cohort, daily *voted* prices, market-factor demeaning. The question was not "does X affect
prices" but "is our **null** on X a true null, or an artifact of the wrong target / cohort /
construct / data."
**Bears on:** the whole feature-nulls catalog; `research/2026-08-13-next-steps.md` item 5 (C2 /
lambdarank), which this **independently converges on and reframes**.
**Status:** research only. No code changed. Magnitudes quoted from trade blogs are *asserted, not
measured* — treated as hypotheses, not evidence.

## Bottom line

The nulls are correct for everything that is a **fixed cross-sectional attribute** or a
**market-wide shock** — which is most of what we've tested. But three families of factor are
**not truly null; we tested them wrong or lack the data.** The strongest, most consistent finding
across all four passes: **we are optimizing the wrong target.** Every factor that shows structure
in our own data is *relative / cross-sectional* (the market factor, expensive→cheap lead-lag, the
C1 rank transform, cross-venue basis, reversal-from-own-mean); everything that is null is *per-item
time-series direction*. Per-item directional forecasting is structurally unable to capture a
cross-sectional signal — so the pivot is to a **cross-sectional ranker (lambdarank) scored on Rank
IC**, not another item feature.

## Why a factor shows null — three causes, three responses

| Cause | Meaning | Examples | Response |
|---|---|---|---|
| **A. Static cross-sectional attribute** | Fixed per item; sets the price *level*, can't vary in time → nothing to predict day-to-day; removed by demeaning | rarity tier, StatTrak/Souvenir, weapon identity, float/pattern | **True null. Stop.** |
| **B. Market-wide shock = the market factor** | Moves ~all items together; absorbed by the date effect by construction | speculation booms/crashes, CNY FX, crypto/risk-appetite, Steam trade-hold policy, gambling crackdowns | **True null. Stop** (it *is* the factor we already capture). |
| **C. Real signal, mis-tested** | Wrong target, construct, moment, functional form, cohort, or missing data | lead-lag, C1 rank, cross-venue basis, supply depth, category×event, hype/pro-usage | **Re-test differently** (below). |

## The reconsidered verdict, per factor

| Factor | Old result | Reconsidered | Why |
|---|---|---|---|
| Rarity tier | null | **True null (A)** | static level; redundant with market factor |
| StatTrak / Souvenir | null | **True null (A)** | scarcity premium is a level; any drift is multi-year, sub-MDE |
| Weapon identity | null | **True null (A)** | fixed attribute; confirmed internally |
| Float / pattern / blue-gem | untested | **Can't test (data)** | we have only item-level aggregates, no per-instance series; and effect is still cross-sectional |
| Speculation / store-of-value | null | **True null (B)** | this *is* the market factor (the Oct-25 crash moved everything together) |
| CNY FX / crypto | ~null, tiny | **True null (B)** | re-prices all USD quotes equally → common factor; use only for level/timing |
| Steam policy / trade-holds | null | **True null (B)** | market-wide liquidity shock |
| Naive volume / liquidity | null | **True null (B/A)** | volume is *coincident* with moves, not leading; liquidity is a static sort |
| Technical indicators | null | **True null (correct)** | low SNR on voted daily prices; parametric momentum detectors |
| **Cross-market basis** | null | **🔑 never actually tested (C-construct)** | we blend venues into one price → no spread to mean-revert against |
| **Expensive→cheap lead-lag** | positive | **🔑 real edge, under-exploited (C-target)** | textbook thin-market info diffusion; belongs in a ranker, not a direction model |
| **C1 rank transform** | CV+, serving− | **🔑 scored on the wrong ruler (C-target)** | a ranking gain *shouldn't* show in per-item direction |
| **−return_1d reversal** | beats model | **Artifact, not edge (C-construct)** | bid-ask bounce on illiquid voted prices; won't survive the 15% fee |
| **Supply depth (listings)** | untested/null | **Wrong moment (C)** | mechanism is volatility/spread (2nd moment) + time-varying; test Δlistings vs \|return\| |
| **Supply shocks (vaulting)** | null | **Wrong cohort + form (C)** | event-driven regime on *cases/collectibles*, not daily ≥$1 skins |
| **Game patches (trade-ups)** | null | **🔑 wrong functional form (C)** | category×event; signed opposite across tiers → cancels under a market-wide flag |
| **Majors / stickers** | null | **Correct null for our cohort** | signal is in capsules/souvenirs, a different universe |
| **Hype / pro-player usage** | untested | **Missing data (C)** | most credible short-horizon item catalyst; no feed exists |

## The signals we've actually been hiding (bucket C, detailed)

**1. Wrong TARGET — the big one.** Lead-lag (z=9.1, Granger R²≈9%, survives demeaning) and the C1
rank gain are *relative* effects. The learning-to-rank literature is explicit that cross-sectional
ranking beats per-asset directional classification on exactly this structure, and that
**LambdaRank optimizes Rank IC directly** — the quantity C1 already moved in CV. This resolves our
open puzzle: C1's CV rank gain *not* transferring to served directional accuracy is *expected* — we
scored a ranker on a classifier's ruler. → **within-date cross-sectional rank of forward return as
the target; `lambdarank`; score on Rank IC + top-minus-bottom decile spread + a Pesaran-Timmermann
test on the long-short portfolio.** This is next-steps item 5, now reframed from "try a different
model" to "measure the signal on the ruler it lives on."

**2. Wrong CONSTRUCT — cross-market basis was never tested.** Our served price is a *blended*
`aggregator_sync` construct, so there is no Steam−Buff spread to mean-revert against. Chinese venues
(Buff163/YouPin) lead price discovery; Western venues follow over days — a classic relative-value
signal at daily frequency. We already ingest the per-venue components (Buff bid, Skinport is live).
→ **keep venues un-blended and test `spread = P_buff − P_steam` as a convergence predictor.**

**3. Wrong FUNCTIONAL FORM — events are category×date, not a daily clock.** The largest datable
event in the data, the **Oct 22 2025 trade-up update**, moved categories in *opposite directions*
(knives/gloves −~70%, Coverts +10–20×). A market-wide daily flag and market-factor demeaning both
*cancel* that by construction — which is exactly why our "event calendar stripped of its clock"
went null. → **`tier × event-window` interactions**, and container/collection retirements as
*regime shifts* on the affected series, not daily dummies. Cheapest test: regress the Oct-22-2025
window as a `tier × post` interaction on the existing archive — a pure natural-experiment check.

**4. Wrong MOMENT — supply depth predicts volatility, not direction.** The mechanism the market
describes is spread/liquidity (a 2nd-moment effect), and depth is genuinely *time-varying* (unlike
rarity). We tested it against direction. → **Δlisting-count vs \|return\| and realized volatility.**
We already collect the sidecar data (~71% of the ≥$1 cohort).

**5. Missing DATA.** Pro-player/streamer/tournament usage is the single most credible short-horizon
*item-specific* catalyst (blogs claim 30–50% intraday pumps), and we have no feed. The tractable
first cut needs no social scraping: a **Major/tournament calendar mapped to pro-loadout and
sticker/souvenir items** as an event-window feature. Also missing: per-instance float/pattern,
live per-item volume, and a real-time attention/mention feed.

## Ranked re-test plan

1. **Pivot to a cross-sectional ranker (lambdarank), scored on Rank IC + decile long-short.**
   Highest leverage, uses data we have, resolves the C1 transfer puzzle, and is where lead-lag
   structurally belongs. Fits the <30-min cap. Placebo = shuffled-date arm first. *(= item 5,
   reframed.)*
2. **Oct-22-2025 `tier × post` natural experiment** on the existing archive — cheap check that a
   category×event interaction carries signal the market factor can't. If positive, generalize to a
   patch/retirement date table.
3. **Cross-market basis** — reconstruct per-venue daily closes (components already ingested); test
   spread mean-reversion / convergence.
4. **Supply depth as a 2nd-moment signal** — Δlistings vs realized volatility, on the sidecars we
   already collect.
5. **Idiosyncratic mean-reversion after large item-specific moves** — price-only, testable now, but
   ⚠️ treat as suspect: our `−return_1d` win looks like bid-ask-bounce quote noise, so test *net of
   the 35%→5% spread and 15% fee* before believing it.
6. **Data to acquire** (in value order): Major/tournament→pro-loadout calendar; un-blended per-venue
   closes; container/collection retirement date table; (longer term) a pro-usage feed.

## Stop testing (true nulls A/B — do not spend more here)

Rarity, StatTrak/Souvenir, weapon identity, float (no data and cross-sectional anyway), speculation
/ store-of-value, CNY FX, crypto, Steam trade-hold policy, gambling-crackdown sentiment, naive
lagged volume/liquidity, and technical indicators on voted daily prices.

## Caveats

- Blog-asserted magnitudes (30–50% pumps, "ST appreciates 15–25% faster") are **never measured**;
  they are hypotheses to test, not evidence.
- Several bucket-C signals live in **cheaper / lower-liquidity / novelty items, cases, or stickers**
  — cohorts our ≥$1-and-liquid universe and daily aggregation partially filter out. Even with the
  data, effects may not survive into the served cohort.
- Any relative-value edge (lead-lag, basis) must still clear the **15% Steam fee and 35%→5%
  spread** to be *tradeable* — so pair the ranker with a microstructure cost/tradeability filter
  (spread, staleness, listing count as conditioners) before claiming a live edge.

## The one-line update to the roadmap

The external evidence and our own internal findings now agree: **stop hunting item features; change
the target.** The measurable signal in this asset class is cross-sectional relative value, and the
current per-item directional design cannot express it. That makes **item 5 (lambdarank) the top
priority, not the last** — reframed as a target change, with basis reconstruction and the Oct-2025
category×event experiment as the two cheapest independent confirmations.
