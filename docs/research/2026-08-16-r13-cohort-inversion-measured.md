# R13 measured: the inversion is real and worse — but its stated premise is wrong

> **Status (as of 2026-08-21): ✅ the decision this doc asked for WAS TAKEN, as recommended.**
> `MIN_SERVED_PRICE_USD` was **not** raised; the two-tier framing shipped instead — served
> forecasts now carry a tradeability label, so the sub-$1 tier is served *and* honestly marked
> (`changelog/2026-08-17-tradeability-label-on-served-forecasts.md`, commit `cda16ab`).
> The over-coverage half of the finding has since been attacked on the band side rather than the
> cohort side: the `sigma` denominator was replaced by a featureless per-item climatology scale
> on 2026-08-20 (`docs/research/2026-08-19-climatology-vs-gbm-band.md`), which is both narrower
> and better calibrated on served replay.


**Date:** 2026-08-16
**Ran:** read-only over `price-archive/ops/item_forecasts.parquet` (158,200 rows) and
`ops/forecast_outcomes.parquet` (114,489 rows; 92,321 on `lgbm-v3*`).
**Bears on:** R13 in `research/2026-08-07-next-steps.md:782`, which
`research/2026-08-16-research-docs-review.md` §1 found had fallen out of tracking on 2026-08-09.
**Status:** research only. No code changed. **No cohort decision taken — see §4, this one is not
mine to make.**

## Bottom line

Three findings, and the third overturns the item:

1. **The inversion is real and has gotten worse.** The served cohort is **16.1% ≥$1** today, down
   from the 16.4% R13 recorded. Median served price is **$0.09**.
2. **The gate never moved.** `backend/database.py:105` still serves on `is_backfilled == 1`;
   `MIN_SERVED_PRICE_USD` does not exist in the backend.
3. **🔑 R13's premise — that the sub-$1 cohort is bad to serve — is not supported.** Sub-$1 is
   the *best-calibrated tier we have*. The problem with a $0.09 forecast is not that it is wrong.
   It is that it is right and useless.

## 1. The inversion, measured on what was actually served

`item_forecasts.parquet`, latest served date **2026-08-05**, `lgbm-v3-regime`, 8,691 distinct items:

| | items | share |
|---|---:|---:|
| ≥$1 | 1,398 | **16.1%** |
| <$1 | 7,293 | **83.9%** |

R13 recorded 1,423 / 7,268 = 16.4% on 2026-08-07. **It has drifted the wrong way**, and the cohort
grew from 5,542 to 8,691 items on 2026-07-29 while the ≥$1 share fell 19.0% → 16.1% — the
expansion was almost entirely sub-dollar.

Served price distribution: p25 **$0.03**, median **$0.09**, p75 $0.475, p90 $2.10. **Half the
product is under a dime.**

## 2. What that means against our own friction numbers

From `references/cs2-market-domain.md` and `backtest/friction.py`, for the tier that is 84% of what
we serve:

- median relative bid-ask spread **35.5%** (n = 22,449)
- Steam's fee is **piecewise and rounding-dominated** at these prices — effective rate ~16% at
  $0.50 rising to ~66% at $0.03
- a median served item is **$0.09**

No forecast of any accuracy is actionable there. This is the real content of R13, and it survives.

## 3. But the quality premise is refuted

R13 argues the gate "excludes most of the items that have tradeable signal and includes mostly
ones that don't." That is a *tradeability* claim. It has been read ever since as a *quality* claim.
The quality claim is false.

**Band coverage by price tier** (92,321 resolved `lgbm-v3*` outcomes, 2025-12-01 → 2026-08-07,
nominal **80%**):

| tier | h=3 | h=7 | h=14 | h=30 |
|---|---|---|---|---|
| **<$1** | **0.832** | **0.822** | **0.824** | 0.882 |
| $1-5 | 0.841 | 0.855 | 0.858 | 0.883 |
| $5-20 | 0.857 | 0.873 | 0.863 | 0.889 |
| $20-100 | 0.859 | 0.880 | 0.811 | 0.899 |
| ≥$100 | 0.891 | 0.940 | 0.800 | 1.000 |

n at h=3: 22,105 / 4,145 / 1,563 / 617 / 64.

**Sub-$1 is the closest tier to nominal at every horizon.** Median |pct_error| is also no worse
(8.70 / 7.41 / 10.04 / 15.48 vs 8.91 / 8.12 / 10.65 / 26.42 for $1-5).

**Corollary worth its own line:** the **over-coverage** problem — the live C5 thread — is
concentrated in the **≥$1** tiers, which run 84–94% against an 80% nominal. That is exactly the
cohort every published metric in this repo is computed on. The tier we serve most of is the tier
that is calibrated best, and the tier we *report* on is the one that over-covers.

## 4. So what is R13 actually asking?

Not "fix a defect." It is asking: **should the product serve 8,691 well-calibrated forecasts that
mostly cannot be traded, or 1,398 that can?**

That is a product-scope decision with no measurement that settles it, and raising
`MIN_SERVED_PRICE_USD` would cut the served catalogue by **84%**. It is not a data-plumbing fix and
should stop being filed as one. **Deliberately not taken here.**

What can be said:

- Raising the floor **will not improve calibration** — it removes the best-calibrated tier and
  leaves the over-covering one. Anyone expecting an accuracy win from this will not get one.
- The honest framing is a **two-tier product**: serve everything, label the sub-$1 tier
  *not tradeable* using the spread and effective-fee numbers we already have. That costs no
  catalogue and states the truth.
- If the floor is ever raised, the ≥$1 band needs the over-coverage work **first**, or the
  surviving product is the worse-calibrated half.

## Caveats

- `ops/*.parquet` are archive mirrors of prod tables; local copies run behind. Latest served date
  here is **2026-08-05**, latest resolved outcome **2026-08-07**.
- Coverage is `in_interval` as stored, i.e. the published dollar-basis column. The calibrated-basis
  gap discussed in `README.md` applies here too; both columns should be read together, and this
  table is the published one.
- Tier is cut on `base_price` (the anchor), not the resolved price.
- The ≥$100 h=30 row is n=13. Ignore it.
