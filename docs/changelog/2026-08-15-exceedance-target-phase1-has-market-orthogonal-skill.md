# Phase 1: the exceedance target has real, market-orthogonal, date-stable skill — but it is not a trade

**Date:** 2026-08-15. Offline reduced-feature screen, read-only, no prod write, no model shipped.
**Scope:** `docs/research/2026-08-15-p-exceed-cost-target-scope.md`. **Script:**
`scratchpad/phase1_exceedance.py`.

## What was tested

The Phase 1 go/no-go for `P(r_h > c)`, `c = actionable_threshold(tier, csfloat)`, one-sided upside.
A **reduced-feature** binary LightGBM (price/return/vol/tier/rarity only — the volatility-clustering
hypothesis, not the 33-feature production model), on the 2026 broad panel (4.2M item-days, 38K items,
Jan–Aug), end-anchored 5-fold CV with the `horizon+13` embargo, invariant-compliant reads
(`prices_relation` + universe filter + BID/TRAILING source exclusion). Scored Brier-skill vs base
rate, AUC, a shuffled-label placebo, a **market-relative (demeaned)** variant, and a top-decile
net-of-cost economic proxy.

## Result — GO on the signal, NO on tradability

| variant | h | base% | AUC | per-fold AUC | AUC placebo | top-decile net% |
|---|---|---|---|---|---|---|
| absolute | 3 | 4.6 | 0.683 | 0.65 0.75 0.77 0.77 0.77 | 0.500 | **−5.9** |
| absolute | 7 | 7.3 | 0.644 | 0.51 0.72 0.76 0.78 0.73 | 0.500 | **−1.7** |
| absolute | 14 | 9.0 | 0.645 | 0.60 0.74 0.74 0.74 | 0.503 | **−5.4** |
| absolute | 30 | 9.9 | 0.543 | 0.54 0.69 0.74 | 0.500 | **−12.7** |
| **relative** | 3 | 3.9 | **0.682** | 0.67 0.75 0.78 0.77 0.82 | 0.500 | −22.1 |
| **relative** | 7 | 5.8 | **0.636** | 0.52 0.72 0.76 0.77 0.77 | 0.500 | −27.4 |
| relative | 14 | 7.1 | 0.590 | 0.64 0.74 0.72 0.77 | 0.502 | −52.6 |
| relative | 30 | 8.2 | 0.454 | 0.61 0.68 0.71 | 0.501 | −31.4 |

**Three findings, in order of importance:**

1. **The skill is real and — uniquely for this project — market-orthogonal.** AUC 0.64–0.68 at
   h=3–14, and it **barely changes when forward returns are cross-sectionally demeaned**
   (0.683→0.682 at h=3, 0.644→0.636 at h=7). Every prior cross-sectional arm (C1, lambdarank,
   market-relative direction labels) *collapsed* under demeaning because it rode the common factor.
   This does not: it is **idiosyncratic volatility-clustering** — the model identifies which items
   are about to move a lot, a per-item property the market factor cannot explain.
2. **It is not a few-episode artifact.** The failure mode that sank every CV-positive arm is skill
   concentrated in a handful of forecast dates (`2026-08-03-accuracy-is-clustered-by-forecast-date`).
   Here **every fold at every horizon clears 0.5** (weakest folds are only the earliest, most
   data-starved blocks). The signal is stable across all five date-blocks and both variants.
3. **It is not a profitable trade.** Even the top-probability decile has **negative** mean
   net-of-cost return at every horizon (−1.7% to −12.7% absolute; worse demeaned). Predicting that an
   item will *move* is not predicting a *profitable direction*: base exceedance is 4–10%, the top
   decile lifts that to 17–25% (a genuine 2.5–3.6× lift), but 75–83% of even the best picks still
   fail to clear cost, and the downside realisations dominate the net. This is consistent with
   `2026-08-07-next-steps` (median predicted |return| ≪ the 7–37% threshold) — the model now *ranks*
   exceedance well, but the achievable probability level does not cross break-even.

## What this changes

- **It clears Phase 1's go/no-go for the *signal*.** For the first time there is a per-item,
  placebo-clean, date-stable, market-orthogonal predictive skill in this feature set. It does **not**
  contradict the range-forecaster positioning (`2026-08-15-cs2-oracle-is-a-range-forecaster.md`) or
  invariant 4: it is a *magnitude* skill, not a directional one, and it is not tradable net of cost.
- **The highest-value use is probably the range product, not an annotation.** A per-item,
  market-orthogonal *magnitude* predictor is exactly a band-width / confidence input — and the three
  width scales tried before were all `sigma`-based within-date reweightings
  (`2026-08-14-next-steps` "do not run a fourth band-width scale"), none a learned per-item
  exceedance probability. This is a *different* lever than the ones ruled out, and it targets the
  actual deliverable (calibrated range) rather than re-slicing sign.
- **A served annotation** (`p_exceed_cost` on the forecast, "this item is unusually likely to make a
  cost-clearing move") is honest as *information*, but must be labelled as magnitude-not-direction and
  never as a buy signal, given finding 3.

## Confirmation on the production pipeline (2026-08-16) — holds and strengthens

Re-ran with the **production 33-feature `engineer_features` allowlist** (from the shipped
`meta.json`), the **production voted consensus price**, and the **production chain-linked
`build_market_index` / `market_factor_for_horizon` demean** (`scratchpad/phase1_confirm_prod_features.py`,
2.29M item-days). The reduced-feature screen was not inflating anything:

| variant | h | AUC (screen → prod) | per-fold AUC (prod) | AUC placebo |
|---|---|---|---|---|
| market-relative | 3 | 0.682 → **0.701** | 0.60 0.74 0.76 0.75 0.74 | 0.500 |
| market-relative | 7 | 0.636 → **0.663** | 0.53 0.72 0.74 0.71 0.70 | 0.500 |
| market-relative | 14 | 0.590 → **0.638** | 0.53 0.71 0.67 0.65 | 0.500 |
| market-relative | 30 | 0.454 → **0.575** | 0.54 0.60 0.62 | 0.500 |

The production market-index demean gives a **higher** market-relative AUC than the crude
cross-sectional mean at every horizon — the market-orthogonality is real, not an artifact of a weak
demean.

**Tradability confirmed negative, on robust statistics.** After applying the production ±500% return
winsorization, the top-probability decile's **median** net-of-cost return is −17% to −24% at every
horizon, and only **16–22%** of top-decile picks are net-positive (the exceedance rate, by
definition). (A first pass reported a +100–345% *mean* net — an outlier artifact of a mean over
capped heavy tails; the median and the win-fraction are the honest reads and both say the same thing
the screen did: predicting magnitude is not a profitable direction call.)

## Remaining caveats

1. **Offline, not served.** The `MIN_FORECAST_DATES = 20` served-validation bind is unchanged; this
   is CV skill on ~5 date-blocks (Apr–Aug 2026), not durable served outcomes.
2. **h=30 is weak** (AUC 0.58 demeaned) — the signal is a short-horizon volatility-clustering
   phenomenon.

## Next step

The signal is confirmed; the open decision is **how to use it, not whether it exists**: a per-item
band-width / confidence input to the conformal range (the actual deliverable, and a new non-`sigma`
width lever) versus a magnitude-not-direction served annotation. Either way it is not a buy signal
(finding 3) and cannot be *served-validated* until durable ≥$1 dates accumulate past
`MIN_FORECAST_DATES`. The band-width route is the higher-value one to prototype next.
</content>
