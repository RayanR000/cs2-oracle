# Served-outcome feedback calibration — design

**Date:** 2026-08-16. **Status:** approved design, pre-implementation.
**Problem source:** `docs/changelog/2026-08-16-exceedance-band-scale-refuted-at-serving.md` and the
Phase-1 debugging that followed it (this session): the served conformal band **over-covers**
(prod panel 87.2 / 91.8 / 90.6 / 89.0% against an 80% target; A/B control @2026-04-28 covered
94–95% at 3/7/14d). The cause is not a fixable level bug — a scalar `q_hat` pooled over a
calibration window that mixes volatility regimes (fold `q_hat` varies **1.4–1.85× at identical
300K training rows**) is served on individual dates, over-covering on calm ones and under-covering
on volatile ones (@2026-05-19 h=14 = 78.6%). The offline conditioning remedies are exhausted
(temporal date-level: refuted, placebos null; cross-sectional sigma: refuted at serving; a fourth
per-item width scale, exceedance: refuted this session).

The one remedy that directly targets **served** coverage and is not refuted is to calibrate against
the served outcomes themselves. This design builds that as a **dormant** mechanism: it does nothing
until enough served dates accumulate (`MIN_FORECAST_DATES = 20`, `scoring.py:145`; ~0–4 dates exist
per horizon today), then self-activates and pulls served coverage toward an honest 80%.

## Goal and non-goal

- **Goal:** a per-horizon multiplicative correction on `q_hat`, computed from realized served
  interval coverage, that makes the **published dollar** band cover at the 80% nominal. Ships
  no-op (byte-identical to today) and activates automatically at the data gate.
- **Non-goal:** any conditional (per-date, per-sigma, per-item) band. This corrects the marginal
  **level** only. The calm/volatile coverage spread is left intact and re-centers on 80% — an
  accepted consequence (volatile dates land ~70%, calm ~85%), chosen over keeping the ~10pp
  conservative inflation. No directional/DA claim is involved.

## Why this is not the refuted "carry a served number back into calibration"

`.claude/rules/training-budget.md` and `2026-08-12-served-sigma-profile.md` refute carrying an
**elasticity / width / coverage shape** from served outcomes into the OOF calibration — it "went
wrong twice in one day in opposite directions" because "OOF residuals and served outcomes are
different populations." This design does the opposite of mixing them: it runs the **same split-
conformal estimator on the served population**, replacing OOF as the calibration set once served is
sufficient. No shape is transferred between populations; the correction is a single dimensionless
scalar per horizon measured end-to-end on the served band's own hits and misses.

## The estimator

Split conformal, re-run on the served panel. The served band is **asymmetric** — `predict()`
recentres on the direction call and can substitute a momentum mid, so `mid` is not the midpoint of
`[low, high]`. The correction therefore scales each **half** about the mid, and `r_i` is the ratio to
the half the realised price fell on:

```
r_i = (actual_i - mid_i) / (high_i - mid_i)      if actual_i >= mid_i     # upper half
    = (mid_i - actual_i) / (mid_i - low_i)        if actual_i <  mid_i     # lower half
factor_h = quantile(r_i, ceil((n+1)(1-ALPHA)) / n)      # ALPHA = 0.20, the same conformal level
```

Scaling both half-widths about the mid by `s` covers row `i` iff `s >= r_i`, so `factor_h =
P80(r_i)` is exactly the `s` that lands 80% coverage on the panel. This is the split-conformal
nonconformity score with an asymmetric, mid-anchored scale — the analogue of `_conformal_records`
measuring residuals about the served centre.

- `factor_h` is the multiplier that would have made the served band cover exactly 80% on the panel.
  Over-covering (bands too wide) ⇒ the realized `r_i` are small ⇒ `factor_h < 1` (narrow);
  under-covering ⇒ `factor_h > 1` (widen). It is **dimensionless** (a ratio within a horizon's own
  price units), so it multiplies `q_hat` directly.
- **Serve-time application is approximate, and the feedback loop absorbs the gap.** The factor is
  measured on the *final* served band, but applied by scaling `q_hat`, which sits **upstream** of the
  blend / bias / recentre / momentum transforms — these do not perfectly commute with a width scale
  (the prior-day blend mixes at a fixed weight; bias thresholds are level-based). So one application
  moves served coverage **toward** 80%, not exactly onto it. This is acceptable because the mechanism
  is a **feedback loop**: the next `MIN_FORECAST_DATES` window re-measures the residual miss and
  re-corrects, and the clamp bounds any single step. Exactness would require moving the correction
  below every transform (a larger change to the serve path) and buys nothing a second iteration does
  not.
- **Basis:** dollar / published. `predicted_price_low/mid/high` and `actual_price` are the columns
  the served product is built from, so `factor_h` makes the **user-visible** band honest and
  absorbs every miss source (weak-fold-model inflation, the anchor-denominator incoherence, the
  sigma mix) into one number, rather than only the calibrated-basis slice.
- **Cohort:** the ≥ $1 served cohort (`price_tier >= HEADLINE_MIN_TIER`), matching the served
  headline that `scoring.py` reports and the population the 80% claim is about.
- **Rows with a degenerate `halfwidth_i`** (≤ 0, or NaN mid/actual) are dropped, mirroring
  `_conformal_records`.

### Gate

`factor_h` is emitted **only** when the panel holds `>= MIN_FORECAST_DATES (20)` distinct
`forecast_date` values for that horizon (per-horizon; h=30 accumulates slowest). Below the gate the
horizon is absent from the returned map and the serve-time accessor falls back to `1.0`. This is the
same count `scoring.py` gates its published coverage on, imported rather than re-spelled.

### Safety clamp

`factor_h` is clamped to `[FACTOR_MIN, FACTOR_MAX] = [0.5, 2.0]`. A pathological or contaminated
panel (a bad cutover, a resolver regression) cannot then halve or double the band; a clamp hit is
logged at WARNING as a data-quality signal, mirroring the `beta` clamp philosophy
(`conformal.fit_beta`). The clamp bounds are module constants, not magic numbers at the call site.

## Module and seams

- **New module `backend/models/served_recalibration.py`.** One public function,
  `served_coverage_factors(session, horizons, *, min_dates=MIN_FORECAST_DATES, alpha=ALPHA) ->
  Dict[int, float]`. It runs the read + the estimator + the gate + the clamp and returns the
  per-horizon factor map (horizons below the gate are absent). Pure enough to unit-test on a
  synthetic panel: the DB read is one function it calls, injectable for tests. Kept out of
  `forecaster.py` because it reads the served panel (Postgres) and is calibration logic with its own
  clear boundary — what it does (measure served coverage → a q_hat multiplier), how it is used (one
  call in the train flow), and what it depends on (the `forecast_outcomes` table) are all isolable.
- **Read path.** `SessionLocal` + raw SQL over `forecast_outcomes` (the scored panel per
  `backend/AGENTS.md:70` — the ops parquet mirrors are stale/selected and must not be used), pulling
  `forecast_date, horizon_days, price_tier, predicted_price_low, predicted_price_mid,
  predicted_price_high, actual_price`. This is the same table `backtest_accuracy.py`
  (`_records_from_frozen_outcomes`, `:631`) reads; the query here is narrower and read-only.
- **Instance state.** `self.served_coverage_factor: Dict[int, float] = {}` on `ItemForecaster`,
  initialised beside `self.conformal_beta` (`forecaster.py:6687`).
- **Compute site.** In the per-horizon train loop, right after
  `q_hat = self._calibrate_conformal(horizon, records_df, tdf)` (`forecaster.py:5682`) and before
  `save_models`. The factors are fetched **once per train run** (not per horizon) and each horizon's
  is stored; a horizon absent from the map leaves `served_coverage_factor` without that key. The
  panel read is gated on the process having a DB session (CI train runs carry `DATABASE_URL`); with
  no session or an empty panel the map is `{}` and the mechanism is a no-op.
- **Persistence.** `save_models` writes `"served_coverage_factor": {str(h): v for h, v in ...}` beside
  `conformal_beta` (`forecaster.py:9445`); `load_models` reads it back with `.get` default `{}`
  (missing key ⇒ empty ⇒ all-`1.0`), mirroring the `conformal_beta` load (`forecaster.py:9586`).
- **Serve-time accessor.** `served_qhat_multiplier(horizon) -> float`: returns
  `self.served_coverage_factor.get(horizon, 1.0)`, re-clamped and `1.0` if non-finite (the `band_beta`
  guard pattern, `forecaster.py:6750`). At the band call (`forecaster.py:7997`):
  `q_hat_eff = q_hat * self.served_qhat_multiplier(horizon)` passed into `conformal.band`.
- **Confidence thresholds are NOT re-fit.** `_calibrate_confidence` thresholds `range_pct` off the
  uncorrected `q_hat`; those thresholds feed only the no-classifier fallback path
  (`_compute_confidence`), which is not the served headline path. Applying the factor to the served
  band half-width only, and leaving confidence as-is, is the documented, deliberate scope — the
  alternative (re-derive `range_pct` from the corrected `q_hat`) buys nothing on the served path and
  couples two changes.

## Key assumption

The correction sizes the **new** artifact's `q_hat` using the **previous** artifact's realized
coverage. Valid because the over-coverage is **structural** — a property of the CV-calibration-vs-
serving gap and the pooled-scalar-over-mixed-regimes geometry, not of a particular booster fit — so
it transfers across a retrain. If a future model change altered the band geometry enough to break
this, the next panel would re-measure and re-correct within one `MIN_FORECAST_DATES` window; the
clamp bounds the transient. This assumption is stated in the changelog and is the thing to revisit if
served coverage does not converge to 80% after activation.

## Interaction with existing band knobs

`served_qhat_multiplier` is orthogonal to `band_beta` (exponent) and `band_scale`
(learned/exceedance denominator): it scales `q_hat`, which multiplies whatever scale/exponent those
produce. It composes with any of them without a matched-pair conflict, because it does not change the
scale's **units** — only the overall width. It is a no-op (`1.0`) on every artifact that predates it
and on every horizon below the gate, so it is safe beside the shipped β = 1.0 / learned = None band.

## Testing

TDD, unit-level (the mechanism is data-blocked in production, so real-panel validation waits for
accumulation; correctness is proven on synthetic panels):

1. **Estimator recovers the target multiplier.** A synthetic panel whose bands cover, say, 92%
   yields `factor < 1` such that applying it to the half-widths lands 80% coverage on that panel
   (re-score after scaling; assert ≈ 0.80). A 65%-covering panel yields `factor > 1` landing 80%.
2. **Gate.** A panel with 19 distinct dates for a horizon returns no factor for it; 20 returns one.
   Per-horizon: a horizon at 25 dates emits while another at 10 does not, in the same call.
3. **Clamp.** A panel driving `factor` outside `[0.5, 2.0]` is clamped and logs WARNING.
4. **Degenerate rows** (`halfwidth <= 0`, NaN mid/actual) are dropped, not propagated as NaN.
5. **Cohort.** Sub-$1 rows do not affect the factor.
6. **Serve-time application.** `served_qhat_multiplier` returns the stored factor, `1.0` for an
   absent horizon and for a non-finite stored value; the band half-width scales by it.
7. **Meta round-trip.** `save_models` → `load_models` restores `served_coverage_factor`; an artifact
   written before this key loads as `{}` (all `1.0`), serving byte-identical.
8. **No-op integration.** With an empty/absent panel, a retrain's `q_hat` and served band are
   unchanged from the pre-feature artifact.

## Rollout / disposition

Ships **dormant** and **gated off by data**, not by a flag — there is no `EXCEEDANCE_SCALE`-style
env switch, because a no-op-until-enough-data mechanism has nothing to gate and an always-on
feedback that self-activates is the intended behaviour. A dated changelog records the design, the
dormancy, and the honest-80% target with its calm/volatile-spread consequence. First real read comes
when a horizon crosses 20 served dates; until then the unit tests and a one-shot offline print of the
would-be factors (below the gate, for a sanity read) are the only evidence, and no served coverage
claim is published before the gate opens.
</content>
