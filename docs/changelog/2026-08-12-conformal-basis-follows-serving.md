# The conformal band's calibration basis — diagnosed, built, and NOT confirmed

C5's correctness half. `2026-08-11-in-interval-basis.md` closed by naming over-coverage as "the
new open question … C5's problem and now well posed for the first time". This entry establishes
what the over-coverage is, builds the coherent calibration, and then **fails its own
pre-registered check** — so the arm ships **off** as `CONFORMAL_SERVED_BASIS=1`.

Built `ef455ab`, gated off `4b0bcb1`.

> ## ❌ REFUTED as a remedy, on a clean paired read. The arm makes the band WIDER.
> The pre-registration below said `q_hat` would **shrink** at every horizon. It does the
> opposite, at all four. Arm `31564924194` against control `31564943172`, same commit
> `97146b7`, same fold counts, **identical `n` per horizon**:
>
> | h | control (`basis=raw_anchor`) | arm (`basis=served`) | Δ | n |
> |---|---|---|---|---|
> | 3 | 94.7172 | 103.7049 | **+9.49%** | 175,739 |
> | 7 | 141.7735 | 148.7987 | **+4.96%** | 157,805 |
> | 14 | 204.3413 | 205.8030 | **+0.72%** | 157,338 |
> | 30 | 312.0487 | 316.9594 | **+1.57%** | 155,617 |
>
> The over-coverage needs `q_hat` **25–39% smaller**. This moves it 0.7–9.5% larger. The
> earlier confounded read (`31563209228`) was directionally right and the 8-vs-9-fold
> confound was not what produced it.
>
> **And the mechanism is now evidence, not a hypothesis.** With `r̂ ≈ 0` the raw-basis
> residual would be the *more* dispersed of the two — `R/k − 1` against `R − 1`, with
> `k = p[d]/S[d]` scattered about 1 — so the arm should have shrunk `q_hat`. It widened it.
> The only way the raw-basis residual comes out **smaller** is if `r̂` already contains the
> `k` term and cancels part of it: the booster is trained on the raw-basis label, and
> `return_1d` is built from the same raw quote, so it can read the anchor deviation and does.
> Against a smoothed-basis outcome that component becomes **added** error instead of
> cancelled error.
>
> This is the same free-factor structure `2026-08-11-smoothed-anchor-label-measured.md`
> refuted in the label, now measured in the conformal residual — and it is the first
> *direct* evidence that the served model is fitting `p[d]/S[d]` rather than the market.
>
> **What this closes:** the calibration basis is **not** the cause of the over-coverage.
> Fixing the incoherence is correct on its own terms and makes the symptom worse, so the arm
> stays off. Everything below about the measurement, the cause being a real defect, and the
> rejection of the quiet-dates alternative stands; **"this change fixes it" does not**, and
> neither does the causal story in the section named for it.

## The number

Production `interval_coverage`, on the fixed `in_interval` basis, read from prod Postgres —
19,917 scored outcomes, `base_price >= 1`, `model_version LIKE 'lgbm-v3%'`, 2026-07-19 excluded
as the dead-band serving config:

| h | n | coverage | target | half-width must shrink to |
|---|---|---|---|---|
| 3 | 7,376 | 87.2% | 80% | 0.722 |
| 7 | 8,405 | 91.8% | 80% | 0.614 |
| 14 | 3,146 | 90.6% | 80% | 0.730 |
| 30 | 990 | 89.0% | 80% | 0.767 |

The shrink factor is the 80th percentile of the served nonconformity score `s = |r_act − r_mid| / h`,
measured with `_derive_verdict`'s own rebased predicate — the multiplier that would put coverage on
nominal exactly. An interval covering 90% at a nominal 80% is uninformative at the width it charges
for, which is the failure this entry is about; the run log's `IntCov` 86.6 / 91.5 / 88.8 / 88.8%
is the same measurement on the run's own cohort filter.

## The cause

`prepare_targets` (`forecaster.py:3992`) builds the training label as

    (P[d+h] − p_raw[d]) / p_raw[d]

but `predict` quotes every item from `_smoothed_anchor_prices`' span-bounded median `S[d]`, and
`backtest/price_resolution.resolve_anchors` resolves `base_price` with the same statistic (median
of the last `SMOOTH_WINDOW = 3` observations). So the residual production is judged on is

    P[d+h]/S[d] − 1 − r̂        while q_hat was fitted on        P[d+h]/p_raw[d] − 1 − r̂

With `k = p_raw[d]/S[d]`, the second is `(1+r)/k − 1` against the first's `r`. Wherever `k ≠ 1` —
which is most rows; `anchor_is_tied` is a minority cohort — the calibration residual is the more
dispersed of the two, so `q_hat` comes out too wide for the basis it is served in.

**This is the denominator half of the incoherence
`2026-08-11-conformal-centre-follows-serving.md` fixed for the centre.** That entry made `q_hat`
cover the mid production serves; this one makes it cover the *return* production is scored on.

Reproduced in isolation (`tests/test_conformal_basis.py::TestTheMechanism`, and at n=200,000 in
the scratch sweep behind it): with no anchor deviation the calibration lands on **exactly 80.0%**,
and it rises monotonically in the deviation — **90.2%** at a 5% deviation spread against a 9%
return spread, **95.1%** at 7.5%. Production's own observable wedge between the served quote and
the resolved base is median **5.2–7.3%** at 3/7/14d. The mechanism reproduces the observed failure
at the observed input scale.

## The alternative that was tested and rejected

A band calibrated on a multi-year window will over-cover a quiet stretch, and the scored dates are
recent. That would make shrinking `q_hat` an over-correction, so it had to be settled first.

The calibration set is the pooled OOF of a 9-fold expanding-window CV whose validation windows span
**2023-02-26 → 2026-07-21** (`cv_results` in `meta.json`; `training_window_days = 1460`). Measured
against the cross-sectional dispersion of the ≥$1 cohort **on that window's own dates**, the scored
forecast dates carry a median `rel_cal` (date p80 |ret| ÷ calibration-window median p80 |ret|) of:

| h | dates | median `rel_cal` | implied coverage from dispersion alone |
|---|---|---|---|
| 3 | 7 | 1.006 | 79.7% |
| 7 | 6 | 0.959 | 81.9% |
| 14 | 3 | 1.074 | 76.7% |
| 30 | 1 | 0.967 | 81.5% |

Ordinary. Quietness explains **~none** of the 7–12pp. Two details that make this sharper than the
table: at **h=14 the dates are more dispersed than the calibration basis** (1.05–1.07), which
predicts *under*-coverage exactly where 90.6% is observed; and dropping exact-zero returns from
both sides — the staleness objection — *strengthens* it, moving median `rel_cal` at h=3 from 1.006
to 1.028. Only two out-of-fold cells are genuinely calm (2026-08-01 h=7 at 0.80, 2026-08-05 h=3 at
0.82).

Two caveats kept: 2026-07-17 and 07-18 fall inside fold 9's validation window, so their `rel_cal ≈ 1`
is near-tautological and carries no evidence; and the pre-2026 single-source Steam backfill vs the
2026 voted multi-source consensus is a real measurement-regime break inside the calibration window,
worth −5% to +27% on p80 depending on horizon, so **`rel_cal` at h=7 is not defensible beyond ±0.15**.
It does not threaten the sign at h=3 or h=14.

## What changed

- `calibration_target_col(h)` — a named column, `target_return_{h}d_cal`, emitted by
  `prepare_targets` alongside the label and always on the smoothed anchor regardless of the
  `LABEL_SMOOTHED_ANCHOR` arm. Winsorized at ±500% and voided on exactly the rows the label is
  voided on: a void means the return is fabricated, which is a property of the series, not of the
  denominator.
- `_conformal_records(..., residual_actual_ret=)` measures `residual_pct` from it. **`actual_ret`
  still drives `hit` and `change_pct`**, which are `_calibrate_confidence`'s inputs — moving those
  under cover of this change would alter a second thing silently.
- `_calibration_returns` resolves the column, records `conformal_basis[h]`, and **WARNs** when it
  falls back to the label. The fallback is reachable on any frame built before this change and it
  silently restores the over-covering band.
- `meta.json` carries `conformal_basis` per horizon (`"served"` / `"raw_anchor"`), and the
  calibration log line prints it. Same contract as `conformal_centre`: `.get`, not strict, and an
  artifact that does not say is not the same as one that says `raw_anchor`.

`tests/test_conformal_basis.py`, 10 tests: the mechanism with its zero-deviation control, the tied
cohort as the negative control for the two columns agreeing exactly, the deviating cohort for them
differing on every row, void propagation, and the WARN-on-fallback. Full suite 2,024 passed.

## What this does NOT fix, and it is the larger half

**Coverage is not stable across dates, and no scalar `q_hat` can make it so.** Per (horizon, date)
on the same panel it runs **58.2% to 99.2%**, and the shrink factor a date needs runs **0.41 to
1.55** — a 3.8× spread at h=3 alone:

| h | date | n | coverage | shrink to |
|---|---|---|---|---|
| 3 | 2026-08-06 | 1,115 | 94.8% | 0.411 |
| 3 | 2026-08-05 | 1,113 | 94.9% | 0.438 |
| 3 | 2026-07-17 | 1,093 | **58.2%** | **1.545** |
| 7 | 2026-08-01 | 1,115 | 99.2% | 0.332 |
| 7 | 2026-07-17 | 1,093 | 67.2% | 1.231 |

The band's only width input is `sigma = price_std_60d / price`, a per-**item** trailing volatility.
It has no date term at all, while realised cross-sectional dispersion is a date-level quantity.
This is the band's version of the structure `da-is-dominated-by-the-market-date` describes.

ACI — the other half of C5 as scoped in `2026-08-09-next-steps.md` — is the candidate, and it
adapts α from *realised* coverage, which is measured in the serving basis and would therefore
absorb a residual basis error too. **It cannot be validated on this panel.** ACI needs a coverage
feedback series and production has served 7 forecast dates at h=3, 6 at h=7, 3 at h=14 and **1 at
h=30**; there is no history to fit or check an adaptation rate γ against. That is a calendar
constraint, not a code one, and it is the same wall `MIN_FORECAST_DATES = 20` describes.

Also unaddressed and separately visible: the 2026-08-11 daily run reordered **242 of ~990** 30d
bands (24%) whose lower leg had gone non-positive, with `_sanitize_forecasts`' own warning naming
"an oversized `q_hat`" as the usual cause. And the pooled fit inherits fold 1's tail directly —
that 2023 window carries p80 |ret| of **0.432 at h=30**, ~2.4× the calibration median, which is a
plausible second reason the 30d band is uniformly wide. Neither is chased here.

## Verification status

**The prediction, as recorded before the read:** `q_hat` shrinks at every horizon and the
calibration line reports `basis=served`.

**The result:** half of it held. `basis=served` at 4/4 horizons, so the wiring works. `q_hat` did
not shrink — see the banner. The arm is now gated off and the entry is filed as a diagnosis plus
an instrument, not as a fix.

**Settled by the paired dispatch** (`conformal_served_basis` input added to
`model-diagnostics.yml` in `97146b7`): the arm widens `q_hat` at 4/4 horizons. The basis is not
the cause. See the banner.

**What is still open, and the order it should be taken in.**

1. **What actually makes the served band 25–39% too wide.** Three candidates are now excluded:
   the calibration centre (F1, ≤1.1pp), quiet forecast dates (median `rel_cal` ≈ 1.0), and the
   calibration denominator (this entry, wrong sign). The strongest untested one is **`sigma`
   itself**: `q_hat` multiplies it, so a serving-time `sigma` systematically larger than the
   CV-time `sigma` for the same items widens the band by exactly that ratio and nothing in the
   calibration would notice. `sigma = price_std_60d / price`, and `price` is the **raw** quote in
   the CV frame and the **smoothed anchor** at serving — the same axis as this entry, one level
   down. Cheap to check: the served median half-width is 8.64% at h=3 against `q_hat = 94.72`,
   implying a served median `sigma` of 0.0912; compare with the CV frame's.
2. **The prediction is fitting `p[d]/S[d]`.** The banner's read is indirect evidence. Direct
   version: correlate `fold_p50` with `p[d]/S[d]` on the OOF rows. If it is large, it reaches
   well past the band — it is the same defect the label arm was refuted for, sitting in the
   served mid.
3. **Conditional coverage** (58.2–99.2% per date) is the larger half and is calendar-blocked
   regardless of 1 and 2.

**Nothing about the served band has changed.** The daily path is on the default, which is the
pre-2026-08-12 behaviour, and no retrain has promoted an artifact from this work.
