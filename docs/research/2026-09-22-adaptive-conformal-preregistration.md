# Pre-registration: time-adaptive conformal (ACI) vs pooled and rolling q_hat

**Date:** 2026-09-22, written **before any ACI number is computed.**
**Status:** PROPOSED. Instrument not yet built.
**Follows:** `research/2026-08-09-model-and-data-research.md` §ACI ("ACI is the right
drift answer"), `research/2026-08-19-deep-model-review.md` item 14 (ACI/DtACI as the
upgrade to the batch served-feedback scalar, h=3/7 only),
`changelog/2026-09-09-waci-measured-void-on-placebo.md` (WACI left the **date**
dimension untouched: S0 8.4–11.4pp → WACI 8.1–10.6pp).

## Why this is not another conditioning lever

Every refuted band lever (`band-width-levers-refuted`, Mondrian, WACI, vol-rank,
listing count, bagging) conditions the band on an **item** attribute. The root cause
of the over-coverage is different: split conformal assumes exchangeability, and the
panel breaks it across **time** (volatility regimes; per-fold `q_hat` 1.4–1.85x at
identical training rows). ACI (Gibbs & Candès 2021) adjusts the miss-rate target online
from realised coverage, which is the mechanism designed for that failure. The date
dimension is the one axis no prior arm moved.

`conformal.calibrate_adaptive` / `update_adaptive` are **not** ACI and are not used
here: they are per-sigma-decile q_hats (the refuted conditioning family) with an
absolute `[0.5, 2.0]` clamp that is dimensionally wrong for q_hat. They have no
callers.

## Why offline first: the served panel cannot answer this yet

Durable `forecast_outcomes`, ≥$1, counted 2026-09-22:

| h | dates ≥ 2026-09-06 (K=320 floor) | per-date coverage range |
|---|---|---|
| 3 | 9 (09-07..09-15) | 0.918–0.964 |
| 7 | 8 (09-07..09-14) | 0.917–0.948 |
| 14 | 1 | 0.903 |

Too few dates for an online method with burn-in, and **no under-covering date is in
the sample**, so the volatile-date hypothesis cannot be tested there. Earlier dates
belong to other band-geometry eras and must not be pooled
(`served-panel-and-geometry-eras`).

## Harness

Extend `scripts/archive/measure_qhat_bagging_mondrian.py`, or add a sibling that
imports it. Reuse `CalibPanel`, `test_dates_for_panel`, `fit_s0`, `_cond_err`,
`_sigma_stratum_err`, `_level_match_c` and `evaluate` unchanged. Model-free
realised-return residuals, as validated there (ratio to shipped q_hat
1.020/0.958/0.887/0.785). Read-only: no DB, no artifact.

The test dates are split by time: the first third is the **tuning block**, the last
two thirds the **scoring block**. All verdicts are read on the scoring block only.

### Arms (paired, same test dates)

- **S0_pooled** is the production shape: `fit_s0` as-is, with the H+13 embargo.
- **R_roll** is the recency control and the analog of the shipped served-feedback
  factor. It is a pooled q_hat over the last `W = 20` dates whose outcomes have
  resolved by the test date (forecast date ≤ day − h).
- **A_aci** is Gibbs–Candès ACI on top of S0. Serve `q_t` = S0's calibration quantile
  at level `1 − α_t`. After each date's outcomes resolve (h days later), update
  `α ← α + γ·(α* − err)`, where `α* = 0.20` and `err` is that date's cross-sectional
  miss fraction. Clip `α_t` to `[0.005, 0.60]` and initialise it at 0.20. Updates run
  on every date, so there is no stride.
  - `γ` is chosen from `{0.005, 0.01, 0.02, 0.05}` by the lowest level-matched date
    error on the tuning block, then frozen for the scoring block.
- **P_aci** is the placebo: A_aci with the same `γ`, fed the same `err` values in a
  permuted date order (seed fixed). This destroys the temporal information and keeps
  the machinery.

No further arms (for example DtACI or a γ ensemble) without a second pre-registration.

### Horizons

h=3 and h=7 are **primary**. h=14 is reported only. h=30 is excluded, because
feedback lags by a month there (deep review).

## Metrics (scoring block, per horizon)

- **Primary:** level-matched date-dimension error `M2lm_cond_err_pp`, the mean
  absolute gap between per-date coverage and 80%. Level matching removes the
  marginal offset, so an arm cannot win by being uniformly narrower (the raw-M2 trap
  that voided two earlier reads).
- **P10 date coverage:** the 10th percentile of per-date level-matched coverage. This
  is the volatile-date tail.
- Raw marginal `M1`; level-matched mean width; the level-matched sigma-decile
  profile.

## Bar (fixed now)

A_aci **passes a horizon** only if all of the following hold:

1. Raw M1 is within 80 ± 3pp.
2. `M2lm_cond_err` is below S0 by **≥ 1.0pp**, **and** below R_roll, **and** below
   P_aci.
3. P10 date coverage is **≥ S0 + 3pp**.
4. Level-matched mean width is **≤ 1.05 × S0**.
5. No level-matched sigma decile covers below 70% (the WACI trade guard).

**Void rather than null** if any of these hold:
- The scoring block has fewer than 100 test dates at a horizon.
- S0's own `M2lm_cond_err` is below 3pp, meaning the panel lacks the defect.
- P_aci also clears (2) against S0, meaning the metric is rewarding machinery rather
  than timing.

## What each outcome buys

| Outcome | Action |
|---|---|
| PASS at h=3 **and** h=7 | Wire ACI behind a new env flag (default off) on top of the served-feedback factor. Then run a served confirm once ≥ 20 post-floor dates exist per horizon, about 09-29 at h=3 and 10-03 at h=7 if the chain stays green. No ship before the served confirm. |
| R_roll passes (1)–(5) against S0 and A_aci does not | The batch feedback already shipped captures the recency gain. Close ACI. |
| Anything else | ACI is closed for the band. Record it in `band-width-levers-refuted`. |

This repo's recurring pattern is **CV-positive, serving-negative** (`AGENTS.md`). An
offline pass is a reason to run the served confirm, not evidence of a served gain.
