# Supply churn (Δlisting count) as a band-width / volatility signal (spike)

**Date:** 2026-08-17
**Type:** Spike. Deliverable: a finding.
**Question:** The docs flag supply depth as *mis-tested* — "predicts volatility / 2nd moment,
not direction," and the proper test (Δlistings vs |return|) was never run. Does the change in
BUFF listing count predict forward price *volatility*, and does it add anything over the
item's own trailing realized volatility (already a feature)?

## Verdict

**The best-positioned signal tested so far: a genuine, fee-free, leading volatility
indicator — but it only fires on the minority of days when supply actually moves.** Over the
full panel it is ~redundant with trailing vol (+0.0025 R²). But *conditional on a supply
move*, |Δlisting| magnitude predicts a 1.5× spread in forward |return| that trailing vol does
**not** capture — and on the largest supply shocks the current feature under-widens the band
by ~4×. It targets the 2nd moment (what a range forecaster actually sells), so it neither
fights the "no per-item directional signal" wall nor the 15% fee. Worth building as a
band-width feature.

## Data

- **Panel:** iflow BUFF `buff_sell_num` (listing count) joined to the iflow BUFF price,
  item_slug-keyed, **2022-04 → 2026-05**. 9.56M item-day observations, 19,610 items.
- Δlisting = `Δlog(buff_sell_num)`; forward vol = `|log(p_{t+H}/p_t)|`; trailing vol =
  30-day mean `|daily return|`, shifted (no lookahead).

## Results

### It predicts volatility, and it is a 2nd-moment effect (not direction)
| feature | corr with forward \|7d return\| |
|---|---|
| \|Δlistings\| | **+0.111** |
| signed Δlistings | −0.008 (≈0 → not directional) |
| log listing *level* | −0.001 (illiquidity alone doesn't predict vol here) |
| trailing realized vol (baseline) | +0.298 |

Signed Δlisting and listing level are both null; only the *magnitude* of the change carries
signal — confirming the docs' "2nd moment, not direction."

### Incremental over trailing vol: tiny on the full panel, real on supply-move days
Full-panel R²: trailing-vol-only 0.0886 → +|Δlistings| 0.0911 (**+0.0025**). Redundant,
because most item-days have no listing change.

But conditional on nonzero supply moves, forward |7d return| by |Δlisting| octile:

| octile (|Δlisting|) | forward \|7d ret\| | trailing vol |
|---|---|---|
| 0 (smallest) | 9.6% | 3.5% |
| 3 | 11.2% | 3.9% |
| 6 | 14.1% | 4.4% |
| 7 (largest) | **14.3%** | **3.6%** |

Forward vol ramps **9.6% → 14.3% (1.49×)** while trailing vol stays flat. The top octile is
the payoff: **14.3% realized forward move against a 3.6% trailing-vol estimate — a ~4×
under-width** the current model would serve. That is precisely the case a leading supply
signal should catch and trailing vol cannot.

## Why this one is different from the earlier nulls

The souvenir, lead-lag, and basis probes were all *directional* and died on (a) no per-item
directional signal and (b) the 15% fee. This is a *volatility* signal for a *range*
forecaster: no trade, no fee, and it targets band width — the actual deliverable. It also adds
information exactly where the incumbent feature is weakest (vol jumps), rather than duplicating
it.

## Go / no-go

**Conditional go — build as a band-width feature and A/B on interval width/coverage, not
direction.** Caveats:
- Its aggregate R² lift is small; the value is concentrated on supply-shock days, so it must
  be scored on **conditional coverage** (does the band widen correctly when supply moves?),
  not pooled R². This fits the existing sigma-tilt / conditional-`q_hat` work, which is where
  band width is the open problem.
- **Serve-time data exists** (unlike the basis): the live `supply_depth` collector writes
  `listing_count` to `supply-*.parquet` daily (lis_skins / market_csgo / waxpeer / bitskins).
  A sidecar builder + a `Δlisting` feature can be built now — no ingest wait.
- Coverage is partial (~71% of the ≥$1 cohort per the data inventory), and Δlisting is
  zero-inflated, so emit a presence indicator alongside it (bid/stattrak pattern).

## Reproduction

Throwaway script: scratchpad `supply_vol.py`. Panels: `buff-iflow-staging/price-archive/`
(`iflow-liquidity-*`, `prices-buff_iflow-*`). Nothing written to the pipeline.
