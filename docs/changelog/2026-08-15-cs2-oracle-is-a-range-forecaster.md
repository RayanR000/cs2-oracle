# CS2 Oracle is a range forecaster, not a directional predictor

**Date:** 2026-08-15
**Scope:** what kind of model this project is. No code changed; this records a classification that
a year of accuracy work has already settled, so the next reader does not re-open it.
**Related:** `2026-08-14-lambdarank-serving-transfer-measured.md`,
`2026-08-14-recency-decay-does-not-transfer-to-served-coverage.md`,
`2026-08-11-in-interval-basis.md`, `.claude/rules/backtest-scoring.md`.

The deliverable, per item per horizon, is a **calibrated price range and its center**. It is not an
up/down call. Directional accuracy is not a shippable product claim and must not be presented as
one.

## Direction is structurally unavailable at this data scale

Not a tuning failure — a sample-size fact about the serving regime.

- Once the market-wide factor is removed, there is no idiosyncratic per-item signal in the features.
- The binding constraint is the number of **independent market episodes**: roughly **14–70
  non-overlapping h-windows in 2026**. Thousands of item-rows are pseudo-replicates of the same few
  moves. More features, more items, or more model capacity cannot buy episodes; only 2026+ calendar
  time can.
- Every relative / cross-sectional arm tried is CV-positive and serving-negative: the C1 rank
  transform; lambdarank (transfers at 1 of 4 horizons, net-negative after cost, h=14 untrainable
  under the >10K-rows-per-query cap); market/basket direction lead-lag; market own-momentum. The
  Oct-22-2025 tier × post event study is null. Recency decay is CV-positive and placebo-clean at
  30d but does not transfer to served coverage.
- Per the repo invariant, DA is never quotable alone — only beside `realised_down_rate` and the
  Pesaran–Timmermann test.

## The band is the part that works, and it is calibrated-to-slightly-wide

Measured 2026-08-15 by paired `scripts/replay_serving.py` at audited anchor **2026-06-16**
(n ≈ 1071, ≥$1 cohort, local artifact trained 2026-08-09):

| horizon | served coverage | median half-width |
|---|---|---|
| 3d | 85.90% | 8.07% |
| 7d | 88.70% | 11.65% |
| 14d | 86.46% | 17.34% |

Against an 80% nominal target, i.e. it **over**-covers on a coherent price basis.

Caveats: one anchor is one date; the artifact post-dates the replay anchor, so the mid is leaky and
the accuracy columns are not readable there.

## There are two coverage numbers and they must be quoted together

- **`interval_coverage`** — the **calibrated** figure. It rebases the band by
  `base_price / current_price` before the predicate, which is the basis `q_hat` was fitted in. Reads
  ~80–86%.
- **`interval_coverage_dollar_basis`** — the **published / user-facing** figure. No rebasing. Reads
  much lower.

Their gap *is* the serving-anchor wedge. This is pre-existing, documented behaviour
(`2026-08-11-in-interval-basis.md`, `.claude/rules/backtest-scoring.md`) and is not re-derived here.

⚠️ Do **not** describe `in_interval` as "broken" — it is the calibrated figure by design. The
failure mode is quoting either number alone.

## Prod-confirmed: the honest dollar-basis coverage is well below nominal, and the shortfall tracks anchor divergence

Read-only query 2026-08-15, ≥$1, `forecast_date >= 2026-07-01`. The live serving model switched
`lgbm-v3-regime` → `lgbm-v3` around 2026-08-11.

For `lgbm-v3`, dollar-basis coverage is **64.0 / 51.5 / 71.8%** at h=3/7/14, against its
`in_interval` of **79 / 83 / 86%**.

Split by drift `|base_price / current_price − 1|`:

- fresh rows (<2%) cover **~90%+**
- stale rows (≥2%) collapse to **~19–65%**

On stale rows the realised outcome tracks `base_price` (the resolver basis) far closer than
`current_price` (the serving anchor). Also verified against the fresh durable `cs2-oracle-data`
archive, so this is not a stale-local-mirror artifact.

**The remaining open thread is therefore an ops / data-freshness question — is production's serving
price frame running behind the archive the scorer later reads? — not a modelling one.**

## A trivial baseline matches the heavy model on the range product (measured, not decided)

A persistence median plus an empirical-quantile conformal band — fits in seconds, no features, no
LightGBM — ties or beats the served band at equal-or-narrower width on clean cash days.

⚠️ This is recorded as a **measured finding and an open question**. **No decision has been made to
replace the production model.** Known weak spot of the baseline: the penny / sub-$1 tier, which
would need a volatility-scaled width.

## Do not propose these again

- **A fourth band-width scale.** Three (`sigma`, `sigma ** beta`, learned) all calibrate to exactly
  80% on their own records and land elsewhere when served. The width variable is not the lever.
- **The "calibrate `q_hat` on a different *set* of rows" class.** Closed.
- **`SERVE_OUTLIER_GATED_ANCHOR=1` as a coverage fix.** Measured 2026-08-15: it moves coverage
  **+2.65 / +0.22 / +1.06pp away from nominal** at h=3/7/14 with width unchanged. Its case remains
  dollar-error only (deviating-cohort served-quote error 2.86% → 0.39% at h=3).
- **`scripts/replay_serving.py` for the published-coverage question.** It structurally cannot
  reproduce the production anchor wedge: it resolves the serving anchor and the outcome from the
  same archive read, so it is coherent-basis by construction. Wrong instrument.

## Docs updated in this pass

- `docs/README.md` — index line and a range-forecaster banner.
- `docs/product.md` — the positioning section, which led on the directional claim.
- `docs/architecture/model.md` — the header banner.

## Not sourced here

The persistence-baseline comparison, the market/basket lead-lag read, the Oct-22-2025 tier × post
event study and the cross-venue basis read have **no standalone changelog entry** as of this date;
they are recorded above as measured but are not independently citable from `docs/`.
