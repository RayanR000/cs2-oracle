# Volume in the learned band scale is net-negative — the served-band question closes

*2026-08-17. Closes the volume thread opened by the band-quality A/B pass
(`2026-08-17-volume-band-quality-passes-on-refold.md`).*

The h=14 band-quality A/B pass tightened the **retired q10/q90 quantile band**, not the served
band. Production serves `q_hat * scale`, `scale` defaulting to `sigma = price_std_60d/price`
(`models/conformal.py`); the only route for volume into that band is the learned scale
(`LEARNED_SCALE=1`, `models/scale_model.py`), which the 2026-08-12 refutation tested **without
live volume**. So one cell was untested: does volume as a NEW scale feature flatten conditional
band miscalibration?

## Route-2 offline probe (`SCALE_PROBE=1` in `scripts/ab_test_volume_features.py`)
On the baseline model's h=14 OOF residuals (60,220 rows, 16 folds, iflow era), cross-fit four
denominators and score **level-matched conditional coverage** (marginal pinned to 80%, so a
uniformly narrower band cannot fake a win — `conformal.py:255-260`), stratified on sigma (the
documented tilt axis) and on a continuous volume feature (`volume_mean_60d`):

| Denominator | sigma-strata err | volume-strata err |
|---|---|---|
| `sigma` | 8.00pp | 1.60pp |
| `scale_sigma_only` | **1.32pp** | 1.48pp |
| `scale_volume` | 1.82pp | **1.15pp** |
| `scale_placebo` (shuffled vol) | 1.51pp | 1.46pp |

## Verdict: net-negative, park it
Volume carries a **real, placebo-clean** signal on its own axis (1.15 < 1.48 sigma-only < 1.46
shuffle). But it **degrades the dominant sigma-tilt axis** (1.82 vs 1.32 sigma-only) — the served
band's big defect. Net trade: a 0.50pp regression on the primary calibration axis for a 0.33pp
gain on a secondary one. Not shippable, and this is offline OOF residuals, where the sigma
exponent already showed this project's serve-vs-CV flip.

This confirms the standing prior (`AGENTS.md`; `2026-08-12-learned-band-scale-measured.md`): **the
width variable is not the lever.** Volume is not the exception. The volume thread is closed on the
served band: *a real signal on the retired quantile band and a minor secondary axis, net-negative
where it would actually be served.* Do not build the production integration.

Probe wiring (`_stratified_coverage`, `_run_scale_probe`, `SCALE_PROBE` capture) is behind a flag
defaulting off; the harness's default behavior is unchanged.
