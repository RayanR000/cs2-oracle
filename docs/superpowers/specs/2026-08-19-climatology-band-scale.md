# Spec: climatological band scale (`CLIMATOLOGY_SCALE=1`)

**Status:** proposed
**Motivation:** `docs/research/2026-08-19-climatology-vs-gbm-band.md` — a per-item
climatological h-day dispersion produces a band **43–47% narrower than the GBM's
`price_std_60d`-based scale at matched 80% coverage, at every horizon**, confirmed prod-faithful
on 285 test dates. This ships that scale behind a flag and A/Bs it on the served panel.

## Goal

Add a third conformal **scale** source alongside `sigma**beta` (`SIGMA_EXPONENT`) and the
learned scale (`LEARNED_SCALE`): a per-item climatological dispersion of the item's own trailing
h-day returns, shrunk toward its price tier. Same product (a calibrated range), ~half the width,
near-zero training cost. Off by default; read on width-at-matched-coverage.

## The scale

For item `i` at date `d`, horizon `h`:

```
raw_i   = dispersion({ h-day return of i anchored at d' : d' + h <= d })   # causal, resolved
pool_t  = same dispersion pooled over the item's price tier t
w       = n_i / (n_i + K)
scale_i = w * raw_i + (1 - w) * pool_t          # James–Stein-style shrink, K≈20
```

- **Dispersion** = the trailing scale of h-day returns. Use `std` (or MAD for robustness) over a
  trailing window (default: all resolved history in the training window; a bounded EWMA is a
  later refinement). The validated prototype used the 80th pct of `|r_h|`; `std` composes more
  cleanly with the conformal q_hat that re-normalizes globally. Pick one and pin it.
- **Causality/embargo:** only h-day returns whose resolution `d'+h <= d` may enter `raw_i`, the
  same discipline `prepare_targets` and the embargo enforce — otherwise the scale peeks at the
  label. This is the single correctness-critical detail.
- q_hat is then calibrated against `scale_i` exactly as `LEARNED_SCALE` calibrates against its
  learned scale (`conformal.calibrate(residuals, scale=scale_i)`), so the band is
  `mid ± q_hat·scale_i` and marginal coverage stays conformal-guaranteed.

## Integration (mirror `LEARNED_SCALE`, `models/forecaster.py`)

1. **Flag:** `climatology_scale_enabled()` reading `CLIMATOLOGY_SCALE`, next to
   `_sigma_exponent_enabled` (:1252) / `scale_model.enabled()`. **Mutually exclusive** with
   `SIGMA_EXPONENT` and `LEARNED_SCALE` — raise if more than one is set (extend the existing
   `resolve_scale` guard rather than adding a new one).
2. **Builder:** `_fit_climatology_scale(horizon, records_df/train_df)` parallel to
   `_fit_learned_scale` (:7397). Produces the per-item scale table + tier pool + `K` + a fallback,
   computed causally. No LightGBM — it is a groupby, so it costs ~nothing (contrast the learned
   scale's 26.6s).
3. **Serve:** an accessor parallel to `_learned_scale_for` (:6971) returning the per-row scale for
   predict, from the persisted table with tier-pool fallback for unseen items. Feed it through
   `conformal.resolve_scale(learned=...)` (the existing "serve an array as-is" path) or add a
   sibling `climatology=` arg — prefer reusing `learned=` to avoid touching `conformal.py`'s
   matched-pair invariants.
4. **Calibrate:** `calibrate` / `calibrate_signed` already take a scale array; pass the clim scale.
   Keep the signed band (shipped 2026-08-19) — the centre stays q50, only the scale changes.
5. **Persist (`meta.json`):** `conformal_scale: "climatology"`, the per-item scale table
   (item→scale per h; ~942×4, small), tier pool, `K`, fallback. Predict reconstructs the scale
   from these. Same matched-pair rule as `conformal_beta`/learned scale: **never difference a
   q_hat across the scale flag — compare COVERAGE and WIDTH only.**

## A/B (the decision, not a CV number)

- Add a `climatology_scale` arm to `.github/workflows/model-diagnostics.yml`, alongside
  `sigma_exponent` / `learned_scale` / `exceedance_scale` (mutually-exclusive group).
- Read **`BAND COVERAGE` and WIDTH in `scripts/replay_serving.py`** on the SERVED panel, arm vs
  control on the same commit/folds. Gate: ship only if it wins width-at-matched-coverage on the
  served dates (the review's bar; do not ship on CV alone — that is the LEARNED_SCALE trap).
- Note the served band over-covers (87–91%): report width at the coverage each arm achieves and
  level-matched, not raw width.

## Tests

- Scale builder: causality (a future h-day return never enters an earlier row's scale),
  shrinkage endpoints (`n=0` → pure tier pool; `n→∞` → pure item), tier fallback for unseen items.
- Conformal: q_hat calibrated against the clim scale yields ~80% marginal coverage on held-out
  rows; `band` is byte-consistent with `resolve_scale`.
- Serving: a served item absent from the train table falls back to its tier pool, never NaN.
- Mutual-exclusion raise when combined with `SIGMA_EXPONENT` / `LEARNED_SCALE`.

## Risks / traps

- **Leakage** via non-causal dispersion — the one thing that would make it CV-positive and
  serving-negative. Enforced by the `d'+h <= d` rule and a test.
- **Static table drift:** the persisted table is as-of the train cutoff and updates only on
  retrain. The prototype was static per item and still won, so acceptable v1; a predict-time
  archive read (daily update) is a later refinement, not a blocker.
- **Cohort/clip:** validated on the ≥$1 archive cohort; the served cohort (~942 items) is
  narrower. The A/B on the served panel is what settles the production number.

## Effort

~1 day implementation (builder + wiring + persistence + tests), then one paired arm/control
dispatch. No new dependencies; no `conformal.py` surface change if the `learned=` path is reused.
