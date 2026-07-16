# Multi-source outlier voting

**Date:** 2026-07-16

**Files changed:**
- `models/forecaster.py` — `fetch_price_history()` now loads per-source data; added `_apply_multi_source_voting()` static method

## What

Added outlier rejection when consolidating multi-source prices into daily prices. For each `(item, day)` with ≥3 sources:
1. Compute median and std of source prices
2. Reject sources >2 std from median
3. Use median of remaining as consensus price

Previously, sources were blindly averaged (Parquet path: `mean_price`; DB path: `AVG(price)`). This let a single stale/spoofed source contaminate the daily price used for features AND targets.

## Impact

**A/B test (walk-forward CV, 7d horizon, full 1.4M-row dataset):**
- Naive mean: **67.64%** directional accuracy
- Voted: **67.64%** directional accuracy
- Delta: **0.00pp**

**Why:** 99.6% of training rows are from single-source STEAMCOMMUNITY backfill. The 0.4% that are multi-source (aggregator data) have an average change of only $0.19. Too few rows and too small a per-row change to affect a LightGBM model trained on 1M+ rows.

**Inference impact is essential:**
- 100% of items with live aggregator data have ≥3 sources
- 96.4% see >0.5% price correction on latest price
- 74.4% see >5% correction
- Max correction: $34.28

Without voting, every forecast's `current_price` (used to convert return % → dollar amounts) can be contaminated by outlier sources.

## Recommendation

**KEEP.** 0pp training impact but critical for inference quality. The documented +2-4pp estimate was made before the STEAMCOMMUNITY backfill (which provides 99.6% of training data). The voting is retained because it prevents corrupted current_prices from propagating into forecasts.

The highest-ROI remaining improvement was **listing count / supply depth** (#6) as of this writing. 🛑 **Subsequently DROPPED on 2026-07-16** — see `docs/research/accuracy-opportunities.md` §1 DECISION. Top remaining work is now regime-switching models.
