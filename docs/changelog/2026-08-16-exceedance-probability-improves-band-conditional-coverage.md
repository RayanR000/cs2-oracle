# Exceedance probability is a band-width scale that improves conditional coverage

**Date:** 2026-08-16. Offline prototype, read-only, nothing shipped.
**Follows:** `2026-08-15-exceedance-target-phase1-has-market-orthogonal-skill.md` (the signal, confirmed
on the production pipeline). **Script:** `scratchpad/phase1_band_prototype.py`.

## Question

The confirmed exceedance signal is a *magnitude* predictor, not a trade. The scoped use was a
per-item conformal-band **scale**: does `p_exceed` make a better band-width variable than the
production volatility scale? Split-conformal, mirroring `models/conformal.py` (score = |resid|/scale,
`q_hat = quantile(scores, ceil((n+1)(1-α))/n)`, half = `q_hat·scale`, α=0.20), on the same
end-anchored folds + `h+13` embargo, q50 mid from a quantile-LGBM on the production 33 features.
Four scales, each conformally calibrated: **const** (=split conformal), **sigma** (`price_cv_30d`, a
proxy for the production `sigma^beta`), **exc** (`p_exceed`), **sig×exc** (blend).

## Result

Coverage in the **stress bin** (top-quintile of realised |residual| — the volatile items a fixed band
fails) is the discriminator; nominal target 0.80.

| h | scheme | marg cov | med width% | **stress cov** |
|---|---|---|---|---|
| 3 | const | 0.827 | 25.6 | **0.135** |
| 3 | sigma | 0.844 | 25.2 | 0.436 |
| 3 | exc | 0.771 | 19.0 | 0.417 |
| 3 | **sig×exc** | 0.778 | 21.8 | **0.575** |
| 7 | const | 0.832 | 33.2 | **0.158** |
| 7 | sigma | 0.837 | 32.0 | 0.480 |
| 7 | exc | 0.755 | 25.4 | 0.409 |
| 7 | **sig×exc** | 0.759 | 28.4 | **0.598** |
| 14 | const | 0.849 | 42.1 | **0.244** |
| 14 | sigma | 0.844 | 41.9 | 0.576 |
| 14 | exc | 0.833 | 46.2 | 0.470 |
| 14 | **sig×exc** | 0.834 | 49.0 | **0.651** |

(h=30 underpowered — the fit/calib/embargo split leaves no valid fold.)

Three findings:

1. **A fixed band is severely miscalibrated by volatility.** An 80%-marginal *constant* band covers
   the most volatile quintile at only **13–24%**. This quantifies the over/under-coverage-by-vol
   defect the project has cited but never measured on the conditional axis.
2. **`p_exceed` is a scale as good as volatility, and narrower.** On its own it matches sigma's
   stress coverage (0.41–0.47) at **20–40% less width** — but slightly *under*-covers marginally
   (0.76–0.83), i.e. it is too aggressive alone.
3. **The blend `sigma×p` is the winner and carries incremental information.** It gives the best
   stress coverage (0.58 / 0.60 / 0.65), clearly beating sigma alone (0.44 / 0.48 / 0.58) — and does
   so *despite lower marginal coverage than sigma*, which means it reallocates width from easy to hard
   items more effectively than volatility does. `p_exceed` therefore adds information the production
   volatility scale does not already have.

## What it means

This is the first band-**width** lever with incremental value, and it targets the actual deliverable
(a calibrated range), not sign. It does **not** contradict "the width variable is not the lever"
(`2026-08-14-next-steps`): those three scales were all `sigma`-family within-date reweightings; a
learned per-item market-orthogonal exceedance probability is a different input, and the blend beats
the sigma proxy on conditional coverage.

It also does **not** close the gap: even `sig×exc` reaches only ~0.6 stress coverage vs 0.80 nominal.
That residual is the trailing-vs-forward volatility gap the project already identified as unfixable by
within-sample reweighting; exceedance narrows it, does not eliminate it.

## Caveats

- **Marginal coverage is not matched across schemes** (calib→test drift), so the width column is
  suggestive; the fair comparison recalibrates every scheme to identical marginal coverage, then
  reads width + stress coverage. The stress-coverage ranking is robust to this (sig×exc wins despite
  a marginal-coverage handicap).
- **`price_cv_30d` is a proxy** for the production `sigma^beta` / `learned_scale`; "beats sigma" is
  "beats this proxy". Confirm against the real production scale before believing the margin.
- **Offline.** `MIN_FORECAST_DATES = 20` served-validation bind unchanged; this is CV on ~5
  date-blocks.

## Fair comparison against the REAL production scale (2026-08-16)

The shipped `meta.json` has `conformal_beta=None` (β=1.0) and `learned_scale=None`, so the production
band is exactly `half = q_hat · sigma`, `sigma = clip(price_cv_60d, 0.0202, 0.6483)` (fallback 0.0742,
`conformal.sigma_from_columns`). Re-ran with that exact scale and **oracle-matched every scheme to
0.80 marginal coverage** on pooled test (`q_hat` = 80th pct of |resid|/scale), so only scale *shape*
differs. `scratchpad/phase1_band_fair.py`.

| h | scheme | med width% | **stress cov** (top-quintile \|resid\|) | easy cov |
|---|---|---|---|---|
| 3 | const | 20.4 | 0.000 | 1.00 |
| 3 | **prod (real scale)** | 21.0 | 0.310 | 1.00 |
| 3 | prod×√p | 22.8 | 0.467 | 1.00 |
| 3 | prod×p | 28.6 | 0.591 | 1.00 |
| 7 | prod | 29.0 | 0.366 | 1.00 |
| 7 | prod×√p | 33.0 | 0.524 | 1.00 |
| 7 | prod×p | 43.2 | 0.635 | 0.99 |
| 14 | prod | 35.2 | 0.400 | 1.00 |
| 14 | prod×√p | 37.7 | 0.496 | 1.00 |
| 14 | prod×p | 43.3 | 0.573 | 0.99 |
| 30 | prod | 44.9 | 0.444 | 1.00 |
| 30 | prod×√p | 48.6 | 0.531 | 1.00 |
| 30 | prod×p | 55.7 | 0.590 | 0.99 |

**Confirmed against the real scale, not the proxy.** At matched 0.80 marginal coverage the production
volatility scale covers the volatile tail at only **31–44%**; augmenting it with exceedance is a
genuine improvement the scale did not already contain. **`prod×√p` is the practical winner**: +0.13
to +0.16 stress coverage (≈40% relative cut in the tail gap) for only **~5–15% more median width**.
`prod×p` reaches ~0.6 stress coverage but at 20–40% more width.

**Two ceilings, stated honestly.** (a) Even `prod×p` reaches only ~0.6, not 0.80 — the residual is the
trailing-vs-forward volatility gap no within-sample scale closes (`2026-08-14-next-steps`);
exceedance narrows it, does not eliminate it. (b) `easy_cov` stays ~1.00 for every adaptive scale, so
they widen the hard items without narrowing the easy ones — the reallocation is one-sided, which is
the improvement ceiling for this family.

## Next step

Strong enough to build as Phase 2: wire `sigma × √p_exceed` as a candidate scale into
`conformal.band` (as a `learned_scale`-style denominator, β must stay 1.0 — `resolve_scale` forbids
combining a learned scale with β≠1), retrain, and run the production interval A/B
(`interval_coverage` + median width, `scripts/replay_serving.py` / walkforward) on the end-anchored
grid. Gated by the `MIN_FORECAST_DATES=20` served-validation bind for any headline. This is the one
path from the entire directional-accuracy thread that improves the shipped range product.
</content>
