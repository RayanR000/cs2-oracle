# Served-outcome feedback calibration — built, dormant until the data gate

**Date:** 2026-08-16. Ships **no-op**; the served band is byte-identical to before until a
horizon crosses `MIN_FORECAST_DATES = 20` served dates (0–4 exist today).
**Design:** `docs/specs/2026-08-16-served-outcome-feedback-calibration-design.md`.
**Motivates:** the served band over-covers (prod panel 87–92% vs 80%), traced this session to a
scalar `q_hat` pooled over a volatility-regime-mixed calibration window (fold `q_hat` varies
1.4–1.85× at identical 300K training rows) served on individual dates. The offline conditioning
remedies are exhausted; this is the one that targets **served** coverage directly.

## What it does

`models/served_recalibration.py` re-runs split conformal on the served `forecast_outcomes` panel
(prod Postgres, ≥$1 cohort): per horizon, the factor = `P80` of the ratio `r` to the relevant HALF
about the served mid (`(actual-mid)/(high-mid)` above, `(mid-actual)/(mid-low)` below), which is the
scale that lands 80% coverage on the panel. It multiplies `q_hat` at serve time via
`ItemForecaster.served_qhat_multiplier`. Over-covering ⇒ factor < 1 (narrow); under ⇒ > 1 (widen).

- **Gated** on `>= 20` distinct served dates per horizon; below the gate the horizon is absent and
  the multiplier is 1.0. So it ships dormant and **self-activates** as dates accumulate.
- **Clamped** to `[0.5, 2.0]`; a contaminated panel cannot wreck the band, and a clamp hit logs
  WARNING as a data-quality signal.
- **Computed once per retrain** after the CV `q_hat`s are set (`train()`), persisted in `meta.json`
  as `served_coverage_factor`, read back with a `{}` default (missing ⇒ all 1.0).
- **Orthogonal** to `band_beta` and `band_scale`: it scales `q_hat`, not the scale's units, so it
  composes with either without a matched-pair conflict, and is a no-op on every prior artifact.

## Why this is not the refuted served-number-in-calibration

Prior attempts to carry an elasticity/width/coverage **shape** from served outcomes into OOF
calibration failed because "OOF residuals and served outcomes are different populations"
(`2026-08-12-served-sigma-profile.md`). This does not mix populations: it runs the same estimator on
the served population and applies a single scalar per horizon end-to-end. It replaces the calibration
population once served is sufficient, rather than transferring a shape between two.

## Honest limits

- **Application is approximate.** The factor is measured on the final served band but applied to
  `q_hat`, upstream of the blend/bias/recentre/momentum transforms, which do not perfectly commute
  with a width scale. One step moves coverage *toward* 80%; the feedback loop re-measures and
  re-corrects each window, and the clamp bounds any transient.
- **Marginal only.** The calm/volatile coverage spread is unchanged — it re-centres on 80%, so
  volatile dates land ~70% and calm ~85%. This is the accepted cost of removing the ~10pp
  conservative inflation (the target chosen deliberately over keeping the cushion).
- **Assumes the miss is structural** (CV-vs-serving geometry, not model-specific), so it transfers
  across a retrain. Revisit if served coverage does not converge to 80% after activation.

## Status

Built and tested (`tests/test_served_recalibration.py`, `tests/test_served_recalibration_wiring.py`:
estimator recovery, per-horizon gate, clamp, degenerate-row/cohort handling, accessor guards, meta
round-trip, no-op-when-absent; real-train regression green). No env flag — a no-op-until-data
mechanism has nothing to gate and the self-activation is the intended behaviour. No served coverage
claim until a horizon crosses the gate.
</content>
