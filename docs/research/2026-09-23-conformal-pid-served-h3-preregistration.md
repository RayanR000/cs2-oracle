# Pre-registration: conformal PID vs the batch served-feedback factor (served panel, h=3)

**Date:** 2026-09-23, written **before any PID number is computed.**
**Status:** FROZEN 2026-09-23 on commit. Nothing may be scored before this commit, and
nothing below may change after it except under a dated *Amendments* entry written before
the read.

**Seen before freezing (disclosed):** (a) the h=3 per-date coverage of 09-07..15 at m=1
(0.918–0.964, from the ACI prereg); (b) the incumbent's live refit on the fixed code,
computed during the PR #64 deploy check on 2026-09-23: h=3 0.5385, h=7 0.6002. No per-date
coverage under any online arm, no QI/Q/placebo number, and no post-09-15 outcome has been
looked at.
**Follows:** `research/2026-09-22-adaptive-conformal-preregistration.md` (ACI refuted
offline; h=3 near-pass: date err 6.34→4.12pp, placebo 10.60, failed only a guard that S0
also fails). That result's stated reopen condition is exactly this doc: a **new** prereg,
**h=3 only**, guard **relative to the incumbent**, scored on the **served panel going
forward**. The offline archive is spent for this question and is not used.
**Method source:** Angelopoulos, Candès & Tibshirani, "Conformal PID Control for Time
Series Prediction" (NeurIPS 2023), reference code `github.com/aangelopoulos/conformal-time-series`
(MIT), `core/methods.py::quantile_integrator_log_scorecaster`. Port, don't import: the
repo uses `np.infty` (removed in NumPy 2) and assumes one score per time step.

## Why now: the incumbent is about to misbehave

`served_recalibration.factors_from_panel` refits the h=3 multiplier as the P80 of `r`
on the stored band and **replaces** the old multiplier with it. From forecast_date
2026-09-20 the stored h=3 band is already scaled by the first activation, so the next
`mode=full` retrain (Mon 2026-09-28) pools rows served at two multipliers. The projected
refit was 0.5385 → 0.667, so the band would re-widen, then snap back toward 1.0, then
oscillate. **Fixed and shipped 2026-09-23** (PR #64, `dc53b97`); see the prerequisite below.

The PID update is **incremental**. It moves the multiplier by the miss rate of the band
that was actually served, so it never pools rows across multipliers and composes by
construction. The candidate therefore addresses the ACI near-pass and the open
composition defect together. That is the reason to run it, not a claim that it wins.

## Prerequisite: per-row served multiplier (needed whatever the result)

Every arm is scored by counterfactual replay on served rows, which requires knowing the
multiplier applied to each row. PR #64 (`dc53b97`, merged 2026-09-23) adds
`item_forecasts.band_multiplier` (migration 0028, applied to prod the same day), the
post-blend multiplier per served row, and fixes the composition defect in B
(`r_base = r_stored × m`). Code from `dc53b97` on records it at write time.

**Frozen multiplier history (h=3):** NULL, read as 1.0, before 09-17. The backfill wrote
0.5963 on 09-17, 0.5381 on 09-18, 0.5384 on 09-20, and 0.5385 on 09-21 and 09-22. There
are no rows for 09-19 (that Price Forecast run failed). Every other horizon is 1.0. These
are 110,720 rows (5 dates × 4 horizons × 5,536), verified in prod after `--apply`, and
derived from run logs in the backfill script's docstring. The factor first went live at
fd 09-17 through a predict-only live read, not on 09-20.

## Replay (why served-panel arms can be paired)

For a served row, `r_base = r_stored × m_row` is the nonconformity score against the
un-multiplied band. `band_signed` scales `q_lo/q_hi` by the multiplier, so the row would
have been covered under any multiplier `m` iff `r_base ≤ m`. That gives exact paired
counterfactuals for every arm on the same rows, with no prod change. The one
approximation is the post-band `blend` transform. Each stored band is
`0.85·m·B_today + 0.15·m_prior·B_prior`, so a single recorded multiplier is exact only
when the base width `B` did not move since the prior row. The error is at most about
`0.15·m_prior·|B_prior/B_today − 1|` per row, and it is largest on the first predict after
a retrain. The validity check below is what bounds it.

**Replay validity check (void if it fails; SUPERSEDED by Amendment A1):** on the 09-17+ dates, the replay's coverage at
the recorded multiplier must reproduce observed served coverage to within 0.5pp per date.

## Population

`forecast_outcomes`, h=3, ≥$1 (`HEADLINE_MIN_TIER`), well-formed bands, forecast_date ≥
2026-09-06 (current `_geometry_floor()`). A date counts once its outcomes resolve. Dates
that are unscoreable because of archive gaps (09-16/17/18 until 09-19 is backfilled) are
excluded; they are not imputed.

**Fixed window (no optional stopping):** forecast_dates 2026-09-07 through 2026-10-18.
Read once, after all of them resolve (~2026-10-23). A new geometry cutover inside the
window truncates it at the cutover. It does not extend it.

## Arms (all replayed on the same rows)

Every online arm starts at `m = 1.0` on the floor date, which is what prod actually did.
`α = 0.20` throughout.
Each one updates on serve date `d` using every forecast date whose h=3 outcomes have
resolved by `d`. Each date contributes one cross-sectional miss fraction `err_t`, so a
date is a step and a row is not.

- **F1**: fixed `m = 1.0` (no feedback). Reference only.
- **B, the incumbent**: the batch rule as prod runs it, with the composition bug fixed.
  It first activates on the first serve date with ≥ `MIN_FEEDBACK_DATES` (8) resolved
  post-floor dates (the predict-only live read), then refits on each Monday `mode=full`
  date as the P80 of `r_base` over post-floor resolved rows, with the [0.5, 2.0] clamp.
  This is the bar to beat. Beating the *buggy* rule proves nothing.
- **Q, the P term**: the quantile tracker, `m ← m + η·(err_t − α)`, with α = 0.20. It is
  ACI in multiplier space and is reported only.
- **QI, the candidate**: `m ← m + η·(err_t − α) + r_I(Σ(err − α), t)`, with the log
  saturation integrator from the reference code. No scorecaster (D term); that needs a
  second prereg.
- **P_QI, the placebo**: QI fed the same `err_t` values in a permuted order (seed 0).

Hyperparameters are **fixed a priori**. There is no tuning block, because the window is
too short to spend a third of it tuning.

- **η = 0.5**, fixed in multiplier units, with no proportional scaling. A 10pp miss-rate
  gap moves `m` by 0.05 per resolved date, which roughly matches the γ=0.2 step that won
  the ACI tuning block. The reference repo's `proportional_lr` scales its step by the range
  of recent scores. That has no analogue for a cross-sectional miss fraction, so it is off.
- **Csat = 1**, copied from the reference config
  (`tests/configs/AMZN.yaml`, market-returns experiment, repo commit `b729c3f5ff`).
- **KI = 0.3**, derived rather than copied. The reference `KI: 200` is in raw
  price-score units and does not transfer to a multiplier near 1. The integrator is
  `r_I = KI·tan(S_t·log(t+1) / (Csat·(t+1)))`, with `S_t = Σ(err − α)`, as in
  `saturation_fn_log`. A persistent 10pp miss over 20 dates gives an argument of about
  0.30 and `tan` of about 0.31, so KI = 0.3 adds roughly +0.09 to `m`. The integrator then
  matters at the same order as a couple of P steps, without dominating them. If
  `tan`'s argument reaches ±π/2, the clamp below binds.

Clamp all online arms to [0.5, 2.0] too, so a clamp is never what separates two arms.
The instrument ports the update from `core/methods.py` and is unit-tested on synthetic
regime switches before it touches the served panel.

## Metrics (per scoring date, h=3)

- **Primary:** the mean over dates of `|cov_t − 0.80|` (date-dimension error), in pp.
- **P10 date coverage:** the volatile-date tail.
- Marginal coverage `M1`, mean multiplier (width proxy), and number of clamp hits.
- The per-sigma-decile coverage profile, pooled over the window (deciles defined by Amendment A2).

Inference is date-level. Resample dates in a paired bootstrap (10,000 draws, seed 0) for
QI − B on the primary metric. Report `n_dates` and the CI on every figure.

## Bar (fixed now)

QI **passes** only if all of the following hold:

1. `M1` is within 80 ± 3pp.
2. Primary error is ≤ B − 1.0pp, the paired bootstrap 95% CI excludes 0, **and** it is
   below P_QI.
3. P10 date coverage is ≥ B + 3pp.
4. Mean multiplier is ≤ 1.05 × B's.
5. No sigma decile (Amendment A2) covers more than 5pp below B's coverage in the same decile. This is
   relative to the incumbent, which corrects the absolute-70% flaw in the ACI prereg.

**Void, not null:**
- Fewer than 25 scoreable dates in the window.
- The replay validity check fails.
- B's own primary error is below 3pp, meaning the window has no time variation to track.
  The 09-07..15 dates all sat at 0.92–0.96, so this is a live risk.
- P_QI also clears (2) against B.

**Unresolved, not null:** (2) fails only because the CI spans 0. Record the achieved MDE
and do not call it refuted (`ab-family-was-never-powered`).

## What each outcome buys

| Outcome | Action |
|---|---|
| PASS | Wire QI behind a new env flag (default off) that replaces the batch refit at h=3 only. Ship the flag and its first activation in the same PR (`meta-flag-removal-breaks-predict`). h=7 gets its own prereg; feedback lag killed ACI there. |
| Q passes (1)–(5) and QI does not | The integrator adds nothing. Close PID and keep B. |
| Null | Close PID for the band and add it to `band-width-levers-refuted`. The factor-history fix stays either way. |
| Void / unresolved | Record it. Rerun only under a new window declared before its first date resolves. |

An offline or replay pass is still a replay. This repo's pattern is CV-positive and
serving-negative (`AGENTS.md`), and the flag's first live weeks are the actual confirm.

## Amendments

**2026-09-23, written while building the instrument and before any read of served data.**
No served outcome, per-date coverage or arm number was computed for any of these.
Everything was checked on synthetic panels only.

- **A1: the replay validity check is replaced.** As written it cannot fail. The feedback
  factor's predicate is the dollar band against `actual_price`, and `r_base ≤ m_row` is
  algebraically the same statement as the stored band containing the actual, so "replay
  reproduces observed coverage" holds by construction and bounds nothing. It is replaced by
  a **blend-sensitivity check**. Every row's `r_base` is moved by its blend bound
  `δ = 0.15 · m_prior · |B_prior / B_today − 1| / m_row`, where `B = half-width / mid / m`
  comes from the item's latest prior `item_forecasts` row. The whole evaluation is re-run
  at `r_base·(1 + δ)` and at `r_base·(1 − δ)`. **Void** if QI's verdict differs under
  either one.
- **A2: "sigma decile" means the base relative half-width decile.** The served panel
  stores no sigma. The served scale is `(high − low) / 2 / mid / m_row`, the band's own
  width with the multiplier removed. Decile edges are pooled over the window, and the same
  edges are used for every arm.
- **A3: resolution timing.** A forecast date's outcomes inform serve dates from
  `target + 1 day` onward (`RESOLUTION_LAG_DAYS = 1`), which is the measured resolution lag.
  `resolved_at` is not used, because re-resolution runs rewrite it after the fact.
- **A4: the coverage predicate is the factor's own.** Coverage is the dollar-band test
  (`r_base ≤ m`), the one `factors_from_panel` calibrates. The rebased `in_interval` that
  the published headline uses is not substituted for it.
- **A5: clamp semantics.** Each online arm's *served* value is clipped to [0.5, 2.0]. Its
  internal state evolves unclipped, as in the reference. Clamp hits are counted per arm.
- **A6: the Q row of the outcome table** means Q judged against the same bar (1)–(5),
  including its own paired bootstrap against B.

**Instrument:** `backend/scripts/measure_conformal_pid.py`. It refuses to read prod
before 2026-10-23. Its 21 synthetic tests are in
`backend/tests/test_measure_conformal_pid.py`, and a mutation check caught 11 of 11
deliberate logic breaks.

