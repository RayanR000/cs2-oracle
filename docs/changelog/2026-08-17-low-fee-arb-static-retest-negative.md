# Static cross-venue arbitrage has no capturable edge even at a 5% fee — do not build a spread scanner

**Date:** 2026-08-17

Re-tested whether the "untradeable net of the ~15% fee" cross-venue call
(`research/2026-08-16-cross-venue-basis-steam-buff.md`) flips at a low (~5%) round-trip fee. For
the **static** (single-snapshot) interpretation, it does **not**.

## Result

On 2026-08-08 spot, median price ratios to Buff163 across the served universe: **CSFloat 0.98×,
YouPin 1.00×, CSMoney 1.03×, Skinport 1.12×** — the tradeable low-fee venues are already priced
within ~2–5% of each other, which a ~5% round-trip eats. Steam sits **~1.25–1.32×** but is
**uncapturable** (locked funds, no Buff→Steam item transfer). CSGOtrader ~1.53× is a different
basis, not a clean venue.

## Two traps confirmed (why naive scans look like free money and aren't)

1. The raw **Steam − Buff** gap is the permanent Steam-listing premium, not arbitrage.
2. A naive **raw-feed spread scan returns garbage** — 62% "median spread" and $148k quotes from
   unconverted currency and illiquid scrape outliers (venue p99 ratios 3–6×). Any spread surface
   MUST sit on the cleaning pipeline (voted price, FX conversion, outlier rejection, liquidity
   gate).

## Verdict

**Do not build a raw-feed spread scanner.** The static-arb interpretation has no free edge; the
low-fee venues are efficiently priced. The model's durable value stays the long-term scarcity
screener plus the calibrated volatility band, not short-term arb.

**Not closed:** the **temporal** basis-catchup signal (corr −0.12, Buff leads Steam;
`research/2026-08-16-cross-venue-basis-steam-buff.md`) was **not** cleanly retested here — a
one-day proxy picks up the permanent premium, not the detrended deviation. It needs the real
persisted-basis feature (`2026-08-17-steam-spot-persisted-for-basis.md`) in a walkforward.
