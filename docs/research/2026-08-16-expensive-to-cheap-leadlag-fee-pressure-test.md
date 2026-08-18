# Expensive→cheap lead-lag: net-of-fee pressure test (spike)

**Date:** 2026-08-16
**Type:** Spike. Deliverable: a finding.
**Question:** The expensive→cheap lead-lag is the only feature that survived removing the
market factor (docs: lag-1 corr +0.213, z=9.1, post-demean R²=4.5%). Does it clear trading
costs as a long-only Steam strategy?

## Verdict

**No — real structure, untradeable on Steam.** The signal is a ~1-day-ahead tilt worth
~0.15pp per trade; a round-trip on cheap items costs ~45% (15% fee + ~30% bid-ask spread).
Net is −15% (fee only) to −35% (fee+spread) at every holding period. Even at the docs'
stronger documented strength the predictable daily move (~0.32%) is 50–100× too small to
clear one round-trip. Confirms the corpus's prior suspicion (the reversal cousin was flagged
"won't survive the 15% fee").

## Method

- Built daily mean log-return indices per price tier (sub-$1 / cheap $1–10 / mid $10–100 /
  exp >$100) from the full archive (4,718 days, 2013→2026), one price per (item,day) = median
  across sources ex-bid/ex-window.
- Confirmed direction: cheap_{t+1} regressed on exp_t + cheap_t (own AR1). (My tier index
  reproduces the sign but weaker than the headline — the >$100 bucket is polluted by noisy
  ultra-rare quotes; the cost conclusion is invariant to this.)
- Cost-aware backtest: buy cheap basket on days expensive index rose, hold H∈{1,3,5,10,20},
  sell. Charge 15% Steam fee on sale + ~30% cheap-tier spread (docs: 35% sub-$1, 5% at
  $1000+).

## Results

| H (days) | gross edge vs all-days | net (fee only) | net (fee + 30% spread) |
|---|---|---|---|
| 1  | +0.11pp | −15.0% | −34.6% |
| 3  | +0.16pp | −15.0% | −34.6% |
| 5  | +0.15pp | −15.1% | −34.7% |
| 10 | +0.18pp | −15.4% | −34.9% |
| 20 | +0.07pp | −16.0% | −35.4% |

Best-case (docs' post-demean R²=4.5%, cheap daily std ~1.5%): predictable 1-day move ~0.32%
(1 σ). Break-even needs >+17.6% (fee) or >~+30% (fee+spread).

## Why it can't be rescued

The edge is a **single-day** effect that decays; extending the hold does not accumulate
more edge, it just accumulates the cheap tier's drift while still paying the one-time
round-trip cost. And the signal lives precisely in the cheap, illiquid, widest-spread items
— the most expensive to trade. There is no holding period, threshold, or tier where the
0.15pp tilt clears a ~45% round-trip.

Corollary: the lead-lag remains valuable as **structure** (confidence weighting, a relative
ranker scored on rank-IC, or nowcasting the cheap tier from the liquid tier) — just not as a
directional trade net of Steam costs.
