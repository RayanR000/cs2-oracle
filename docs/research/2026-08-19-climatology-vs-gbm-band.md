# The GBM band loses to a featureless climatology at matched coverage

**Date:** 2026-08-19
**Type:** research — the null the programme never had (deep-model-review §9.1 / §12.12)
**Script:** `backend/scripts/climatology_vs_gbm.py` (read-only, no retrain, no serving code)

## Question

The product is almost entirely a **scale** estimate — median `|r_hat|` moves the served
interval by <5% of its own width — so the gate on all further modelling work is: at matched
coverage, is the GBM's band narrower than a per-item **climatological** band that uses no
features and no booster? If climatology is not wider, the GBM's scale is decorative.

## Method

- **Data:** the trained artifact's `engineered_data.parquet` (1,328,302 OOF rows, 5,542 items,
  2025-05-22 … 2026-08-04). Forward h-day returns by exact calendar-date lookup, matching
  `prepare_targets`.
- **GBM band:** half-width `clip(price_std_60d/price, floor, cap)` — the exact
  `conformal.sigma_from_columns` scale the served band multiplies by `q_hat`.
- **Climatology band:** per item, the 80th percentile of its OWN trailing `|h-day return|`,
  taken over the **calibration** period only (causal), shrunk toward the price tier's pooled
  quantile by `n_i/(n_i+K)`. Unseen items fall back to the tier pool. No features, no booster.
- **Matched coverage:** both bands centred at 0. For per-row half-width `w_i`, the multiplier
  giving exactly 80% test coverage is `λ* = quantile(|r_i|/w_i, 0.80)` and the mean half-width
  is `λ*·mean(w_i)`. Both methods are compared at **identical 80% coverage**; only width
  differs. `λ*` is set on the held-out rows for both, so neither gets a calibration-level edge.
- **Split:** earliest 70% of dates fit the climatology; latest 30% (104–114 dates) are the test.
- **Uncertainty:** bootstrap over test **dates** (returns correlate within a date).

## Result

Width ratio = climatology width / GBM width at matched 80% coverage. **<1 means climatology is
narrower — i.e. better.**

| h | GBM width % | clim width % | ratio (clim/GBM) | 90% CI |
|---|---|---|---|---|
| 3 | 23.66 | 12.88 | **0.544** | [0.532, 0.562] |
| 7 | 30.66 | 17.81 | **0.581** | [0.569, 0.598] |
| 14 | 38.75 | 26.13 | **0.674** | [0.641, 0.715] |
| 30 | 50.23 | 47.46 | **0.945** | [0.905, 0.997] |

**The featureless climatology is 46% / 42% / 33% narrower at h=3/7/14 for the same coverage,
and matches-to-beats the GBM at h=30.** The GBM's `price_std_60d`-based scale is not decorative —
it is *worse* than directly estimating each item's h-day return dispersion.

Robust to the one free knob (shrinkage `K`), extreme horizons:

| K | h=3 | h=30 |
|---|---|---|
| 5 | 0.542 | 0.990 [0.952, 1.036] |
| 20 | 0.544 | 0.945 [0.906, 0.997] |
| 50 | 0.547 | 0.895 [0.863, 0.946] |
| 100 | 0.556 | 0.865 [0.830, 0.907] |

h=3 is insensitive; h=30 improves with more pooling (per-item h-day observations are sparse at
30d, so shrinkage toward the tier pool helps), never worse than the GBM.

## Why this is directionally safe, not a CV-positive/serving-negative trap

- Climatology is fit on the calibration window and evaluated **out of sample in time**; it is
  variance reduction, not signal extraction, so it carries no date-factor confound.
- Coverage is matched to exactly 80% on the test rows for **both** methods, so this is not a
  coverage-for-width trade — it is purely how well each allocates width across items. A scale
  aligned to each item's own dispersion needs less global inflation to cover the tail, which is
  the entire mechanism.
- A **static** per-item climatology beats the **daily-updating** GBM sigma, so the GBM's daily
  re-estimation is not buying width-allocation quality either.

## Production confirmation (2026-08-19)

Re-run `--source archive` on the **full durable archive** (2024-01-01…2026-08-08, **285 test
dates**, 30,548 items, ≥$1 cohort), voted through production's own
`ItemForecaster._apply_multi_source_voting`, with `price_std_60d` engineered exactly as
`engineer_features` does — no dependence on the local artifact. Using production's **persisted
sigma clip bounds** (`--clip artifact`, cap 0.648) so a loose cohort cap cannot inflate the GBM:

| h | GBM width % | clim width % | ratio | 90% CI |
|---|---|---|---|---|
| 3 | 15.82 | 8.97 | **0.567** | [0.552, 0.582] |
| 7 | 24.32 | 13.87 | **0.570** | [0.557, 0.585] |
| 14 | 33.05 | 18.26 | **0.552** | [0.536, 0.569] |
| 30 | 44.61 | 23.73 | **0.532** | [0.516, 0.550] |

**Confirmed and stronger:** climatology is 43–47% narrower at *every* horizon on 285 test dates
(well above the ~50-date floor coverage claims need). The 30d tie in the artifact run was a
small-sample artifact — the large prod-faithful panel shows climatology wins at 30d too. With
bounds derived from the broad cohort instead (looser cap 2.42) the ratios are 0.41–0.43, i.e. the
prod-clip figures above are the conservative read.

## Gate #1 + #2 (served cohort, out-of-sample coverage) — 2026-08-19

Before making climatology the *default* scale, two cheap gates (`--served-cohort`, a 3-way
temporal split so both width AND the conformal level are out of sample: climatology fit on the
earliest 60% of dates, λ fit on the next 20%, coverage+width read on the last 20%). Restricted to
the trainable/backfilled **served** cohort (2,430 items, 190 eval dates), prod clip bounds:

| h | GBM cov (OOS) | clim cov (OOS) | GBM w% | clim w% | ratio | 90% CI |
|---|---|---|---|---|---|---|
| 3 | 0.781 | 0.794 | 10.49 | 7.55 | **0.719** | [0.710, 0.728] |
| 7 | 0.783 | 0.795 | 15.18 | 10.90 | **0.718** | [0.708, 0.731] |
| 14 | 0.783 | 0.795 | 20.85 | 14.98 | **0.718** | [0.708, 0.729] |
| 30 | 0.790 | 0.802 | 28.62 | 21.31 | **0.745** | [0.734, 0.755] |

- **Gate #1 (width) PASS:** 26–28% narrower on the served cohort — smaller than the 44–47% on the
  broad universe, but still large with tight CIs below 1 at every horizon.
- **Gate #2 (coverage level) PASS:** climatology's out-of-sample marginal coverage (0.79–0.80) is
  as close to, or closer to, 0.80 than the GBM's (0.78–0.79) — the tighter band does **not** cost
  coverage.
- **Caveat:** this is the archive/voted basis, not the production serving basis. The prod
  over-coverage (87–91%) is a separate anchor-denominator issue that affects **both** scales
  equally, so it does not change the relative verdict, but the *absolute* served coverage is
  unresolved here — the served-panel A/B settles it before a default flip.

## Caveats

- The centre is excluded (both centred at 0), justified by the <5%-of-width framing.
- The **served** GBM band over-covers (87–91% vs 80%), so in production it is even wider than
  this matched-80% version — the real gap to climatology is likely **larger**, not smaller.
- The archive cohort (30k ≥$1 items) is broader than prod's ~942 served items; the ranking is
  robust across cohort and clip choices, but exact served-panel widths would differ.

## Implication

The band's scale should come from a **per-item climatological h-day dispersion, pooled/shrunk by
tier** (deep-model-review §9.1/§9.2), not from daily volatility × a global `q_hat`. This is the
highest-value modelling change surfaced so far, and unlike every refuted width arm it wins out
of sample. Next: build it as a candidate scale behind a flag and A/B on width-at-matched-coverage
on the served panel.
