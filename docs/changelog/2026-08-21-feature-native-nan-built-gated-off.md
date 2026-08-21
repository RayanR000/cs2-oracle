# 2026-08-21 — Native-NaN feature handling built, gated off (`FEATURE_NATIVE_NAN`)

Addresses deep-model-review §10.4: NaN features are median-imputed at train, val,
CV, and predict, filling a missing feature with the **training cross-sectional
median**. For the long-return features that median is strongly positive
(`return_180d` ~6.33, `120d` ~4.25, `90d` ~2.77, `60d` ~1.46 in the 2026-08-20
artifact), so a newly-eligible / short-history item is fed a coherent multi-month
uptrend it never had.

## Change

`FEATURE_NATIVE_NAN=1` (off by default) makes `_impute_features` a pass-through so
NaN reaches LightGBM's native handling instead of being filled. Routed through a
single helper at all six served/OOF sites (global + regime train/val, CV fold
train/val, exceedance-head serve, batch predict). The two refuted-arm rank-IC
diagnostics (`_fold_q50_scores`, lambdarank) keep their local imputation — they
touch neither serving nor the band.

**Matched pair, serving follows the artifact.** A booster trained on imputed
frames never learned a NaN default-direction, so `meta.json.feature_native_nan`
is persisted and `_feature_native_nan_served()` reads it (falling back to the env
only when the artifact is silent), exactly like `naive_init_score`. Off = the fill
is byte-identical to the previous `X.fillna(medians)`.

Tests: `tests/test_feature_native_nan.py` (7).

## Magnitude probe (no retrain) — corrects the review's direction

Ran the **same** 2026-08-20 booster through the serving path twice — median-impute
vs NaN-passthrough — at the latest anchor. Items whose centre moves are exactly
the short-history ones. On the **≥$1 scored cohort** (948 items):

| h | % items affected | median Δ (impute−nan) | mean Δ | %up |
|---|---|---|---|---|
| 3 | 0.1% | −0.31% | −0.31% | 0% |
| 7 | 1.2% | +0.07% | +0.15% | 64% |
| 14 | 2.3% | −0.73% | −0.63% | 18% |
| 30 | 4.3% | −1.04% | −2.31% | 20% |

So the effect is **real but modest on the scored universe** (~1–2% median centre
shift, rising to a ~7–10% tail for individual items — e.g. P250 Visions FN
10.62→9.69, AWP Crakow! FT 26.82→25.24 at h30), affecting 1–4% of items, growing
with horizon.

**Direction is DOWNWARD**, opposite the review's "bull prior" framing. The *feature*
fill is bullish, but the booster learned mean-reversion on long returns, so a
fabricated +633% 180d return yields a *lower* forward forecast. On the all-served
(penny-heavy) cohort the sign flips upward, but those are sub-$1 rounding artifacts
(1¢↔2¢), not economically meaningful.

## Paired retrain + replay (2026-08-21) — no regression, marginal upside

Retrained both arms locally reusing the 2026-08-20 artifact's HP (only the flag
differs), replayed at two clean anchors on the ≥$1 cohort. Both boosters are
genuinely distinct (DA/rankIC differ). Control → arm:

- **Coverage:** moves ≤0.5pp at every anchor·horizon (e.g. 06-01 h14 60.70→61.08,
  h30 84.03→84.51). **Width:** ≤0.1% (28.86→28.66% at h30). Effectively a no-op —
  the band is climatology-driven, so imputation barely touches it.
- **DA:** mixed ±1–3.5pp, within single-anchor noise (1033–1077 items, 1 date).
- **Centre accuracy (dollar error):** the one consistent signal — a small
  improvement at long horizons on the volatile anchor: 06-01 h14 9.67→9.52, h30
  9.50→9.41 (pooled model %). This is where the fabricated long-return fill did the
  most damage. Effect is SMALLER than the fixed-booster probe, because the retrained
  booster learns proper NaN directions.

## Status

BUILT, gated off, byte-identical when off. Paired retrain shows **zero regression**
on coverage/width/DA and a marginal center-accuracy gain at h14/h30. It is a clean
correctness fix (stops fabricating a bullish long-return history for the growing
new-item universe) that does not hurt and slightly helps, but the measured upside is
within noise on a 2-anchor local read. **Ship decision is low-risk either way**;
recommendation is to ship (correctness + removes a new-item footgun) via the paired
model-diagnostics dispatch for a wider-panel confirmation before flipping the flag on
`price-forecast.yml`.
