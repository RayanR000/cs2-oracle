# The listing-count floor: the effect is real, the threshold is imported wrong

**Date:** 2026-08-16
**Ran:** read-only over `price-archive/supply-2026-08.parquet` (cohort cost) and a join of
`supply-history.parquet` × `volume-panel.parquet` (payoff).
**Bears on:** `references/cs2-market-domain.md` §8/§10 item 1 — the last surviving item of the
three that doc proposed. The other two died in
`research/2026-08-16-wash-trade-screen-and-volume-spike-exceedance.md`.
**Status:** research only. No code changed.

## Bottom line

Thin items really are wilder — **monotonically, and within price tier**, so it is not the
cheap-items-are-volatile confound. But the ≥30 threshold that published index methodologies use
**sits in the flat part of our payoff curve**: it costs 69% of the served cohort and moves
P(|r₇|>5%) by 0.011. The discriminating cut is >100 listings, which costs 90%.

**Recommendation: don't build the floor. Use listing count as a band-width conditioner instead** —
same information, no scope loss. That is also the shape the exceedance work already settled on
(width, not filter).

## 1. What a floor would cost

Served cohort = `aggregator_sync`, ≥$1, 2026-08-06 (the last local supply snapshot): **21,995
items**. Listing count = **max** across the four supply feeds (`lis_skins`, `market_csgo`,
`waxpeer`, `bitskins`), matching `forecaster.py`'s documented max rule.

**83.7%** of the cohort has a listing count at all (18,417); 3,578 have none. Among those that do:
median **18**, p25 **5**.

| floor | kept, missing = FAIL | kept, missing = PASS |
|---|---|---|
| ≥1 | 18,417 (83.7%) | 21,995 (100%) |
| ≥10 | 11,679 (53.1%) | 15,257 (69.4%) |
| **≥30** | **6,862 (31.2%)** | **10,440 (47.5%)** |
| ≥50 | 4,433 (20.2%) | 8,011 (36.4%) |
| ≥100 | 2,243 (10.2%) | 5,821 (26.5%) |

**The cut is broad, not a cheap-tail trim.** At ≥30 the kept share is 24.9% / 28.0% / 40.9% /
38.4% / 46.3% across the $1-5 → ≥$1k tiers — even the ≥$1k tier loses over half. Median price kept
$16.47 vs dropped $7.05, so it skews cheap, but 58% of what it drops is above $5.

**Why the imported number doesn't transfer:** `skintrackers.com`'s ≥30–50 is a **Steam** listing
count on an index of a few hundred items. Ours is a max over four **non-Steam** venues with
structurally smaller books. The threshold is not portable; only the *idea* is.

## 2. What a floor would buy

The one window with both a deep listing-count history and a price series:
`supply-history.parquet` (BUFF counts, real only from **2023-01-25**; before that the column is
zeros) joined to `volume-panel.parquet` (Steam sale median).

**797,218 item-days, 2,912 items, 2023-01-25 → 2024-02-15, 14 distinct months.** (Episode count
first, per the wash-screen method note. 14 months and 2,912 items is genuinely better powered than
the spike arm — but it is still **one regime**, and it straddles the 2023-09-27 CS2 launch.)

| BUFF listings | item-days | items | med px | med \|r₇\| | P(\|r₇\|>5%) | P(\|r₇\|>10%) |
|---|---|---|---|---|---|---|
| 1–5 | 823 | 86 | 10.47 | 0.0779 | **0.6841** | 0.4070 |
| 6–15 | 9,034 | 257 | 7.57 | 0.0735 | 0.6370 | 0.3638 |
| 16–30 | 24,993 | 492 | 6.45 | 0.0670 | 0.6106 | 0.3289 |
| 31–50 | 35,720 | 718 | 7.02 | 0.0651 | 0.6003 | 0.3200 |
| 51–100 | 88,260 | 1,133 | 6.48 | 0.0636 | 0.5909 | 0.3132 |
| >100 | 385,760 | 1,978 | 5.99 | 0.0499 | **0.4990** | 0.2298 |

Perfectly monotone, and **it is not the price confound** — within every price tier the same
gradient holds:

**P(|r₇|>5%) by tier × listing bucket**

| tier | 1–5 | 6–15 | 16–30 | 31–50 | 51–100 | >100 |
|---|---|---|---|---|---|---|
| $1-5 | 0.747 | 0.679 | 0.662 | 0.643 | 0.617 | **0.518** |
| $5-20 | 0.626 | 0.616 | 0.580 | 0.574 | 0.569 | **0.486** |
| $20-100 | 0.706 | 0.603 | 0.555 | 0.558 | 0.564 | **0.465** |
| ≥$100 | 0.857 | 0.663 | 0.694 | 0.688 | 0.627 | **0.523** |

## 3. The two curves don't line up

Read the payoff table as a step function and the problem is obvious:

- **16–30 → 31–50 → 51–100** is *flat*: 0.611 → 0.600 → 0.591. A floor placed anywhere in there
  buys ~0.01 of exceedance rate.
- The only real break is **>100**: 0.591 → 0.499.

So a ≥30 floor pays **69% of the cohort** for **0.011**. A >100 floor buys a genuine 0.09 drop and
costs **90% of the cohort** — 2,243 items, which is not a product.

**This is a continuous variable being asked to behave like a gate.** The information is real and
monotone; discretising it at any threshold is where the value gets destroyed.

## 4. What to do instead

Feed listing count to the **band width**, not to the cohort filter. The gradient above is exactly
a conditional-σ statement — thin items need wider intervals, not exclusion — and it is the same
conclusion the exceedance thread reached: this class of signal is band-width, never a gate or a
trade. Concretely, `log1p(listing_count)` as a conditioner in the same place the served-outcome
`q_hat` multiplier already lives.

Two things to check before that is worth building:

1. **Coverage is the real ruler, not exceedance rate.** The question is whether low-listing rows
   are the ones *under*-covered today. That needs served outcomes and cannot be answered from this
   local archive (`cs2_market.db` here is a fixture).
2. **The 16.3% with no listing count at all** need a policy, and "missing = wide" is a different
   default from "missing = median". Note the missingness is itself informative — it runs 26.2% in
   $1-5 and 0.8% at ≥$1k.

## Caveats

- Cost and payoff come from **different windows and different venues**: the cost table is 2026-08
  across four Western venues; the payoff table is 2023-01 → 2024-02 on BUFF counts vs Steam
  prices. The *shape* is what transfers, not the bucket edges.
- 14 months is one regime spanning the CS2 launch. Forward windows overlap.
- The >100 bucket is 48% of all item-days, so the panel is dominated by liquid items and the thin
  buckets are comparatively small (823 item-days across 86 items at 1–5).
