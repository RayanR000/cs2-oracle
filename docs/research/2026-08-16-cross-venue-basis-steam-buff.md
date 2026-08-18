# Cross-venue basis (Steam–Buff): signal + net-of-fee pressure test (spike)

**Date:** 2026-08-16
**Type:** Spike. Deliverable: a finding.
**Question:** The Steam–Buff basis was flagged as the last untested driver with a real
mechanism ("we blend venues into one price, so no spread to mean-revert against"). Does the
basis predict returns, does Buff actually lead Steam, and does it clear the 15% fee?

## Verdict

**The strongest, cleanest predictive signal in the project — and it settles the folklore —
but it is not a Steam trade.** A basis deviation (Steam cheap vs its own Buff-norm) predicts
Steam catching up with corr ≈ −0.12 at n≈880k, and the basis closes mainly by *Steam*
moving toward Buff — the first direct measurement that **Buff leads, Steam follows** (the
docs had this as "well-attested direction, folklore lag"). But the Steam-side catch-up tops
out at ~2% (typical) to ~7% (extreme dislocations), always below the 15% fee. Real value is
as a **fee-free forecast / nowcast feature**, not a directional trade.

## Data

- **Panel:** 909k rows, 2,028 items, **2022-04 → 2025-12** (daily). Far deeper than the
  ~25-day live multi-source window.
- **Steam** = source-null historical backfill (CSMarketAPI daily). **Buff** = iflow staging
  (`buff_iflow`, already USD). Inner join on (item, day); basis = log(Steam/Buff), clipped |·|<2.
- Steam runs **+7.2%** over Buff on average, but wide: 5–95% basis range [−93%, +60%].
  Per-item persistent gap (between-item σ of mean basis = 0.37) is the static wallet-lock/fee
  level difference — *not* tradeable; the tradeable part is within-item deviation (σ ≈ 0.17).
- Deviation = basis − trailing-30d item mean (no lookahead).

## Results

### Prediction (real, and directionally informative)
| H (days) | corr(dev, Steam_fwd) | corr(dev, Buff_fwd) |
|---|---|---|
| 7  | **−0.122** | +0.085 |
| 14 | −0.114 | +0.085 |
| 30 | −0.085 | +0.085 |

Negative Steam corr = basis mean-reverts by Steam moving. Buff corr is weaker and same-sign
⇒ **the gap closes primarily via Steam adjusting toward Buff. Buff leads.**

### Tradeability (fails the fee at every dislocation size)
Long the Steam-underpriced items (can't short on Steam); forward *Steam* catch-up:

| dev bucket (Steam below Buff-norm) | n | Steam +14d | Steam +30d | Steam +60d | net-of-fee (30d) |
|---|---|---|---|---|---|
| −5…−10% | 125k | +1.3% | +1.4% | +3.1% | −13.8% |
| −10…−18% | 88k | +1.9% | +0.7% | −0.1% | −14.4% |
| −18…−33% | 24k | +2.2% | −1.5% | −6.3% | −16.3% |
| −33…−55% | 4.9k | +1.6% | +0.8% | −1.8% | −14.3% |
| −55…−86% | 1.5k | +2.8% | +6.4% | +7.3% | **−9.6%** (best) |

Break-even needs a Steam move > +17.6%. Even the most extreme, thinnest, widest-spread
bucket peaks at +7.3% — ~10pp short, before its (large) spread. The strong side is the
*short* (Steam-expensive falls −3 to −5%), which Steam disallows.

## Why untradeable but still valuable

The 15% Steam fee is a fixed % on every round-trip regardless of item price, and no basis
dislocation predicts a Steam move that large. But the deviation carries **corr ≈ −0.12 with
forward Steam direction at fee-free cost** — an order of magnitude better than the
expensive→cheap lead-lag (0.04). Use it to:
- **Nowcast the Steam tier** and sharpen point forecasts / confidence bands (no fee to clear).
- Feed `basis_dev` as a forecast feature — this is the concrete payoff of the long-standing
  **un-blend recommendation**: the live `aggregator_sync` blend destroys exactly this feature.

## Go / no-go

- **Trade: no-go** (same fee wall as every other signal).
- **Feature: go.** Persist un-blended per-venue daily closes (Buff + Steam) and add
  `basis_dev` to the forecaster. It is the best fee-free directional signal measured.
- **Bonus:** this panel (2022–2025 Steam×Buff) also settles the "Buff leads Steam" question
  the corpus had left as folklore.
