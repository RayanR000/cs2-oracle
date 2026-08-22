# Pre-registration: does dividing out `sigma`'s date-level level fix the band's coverage?

> **Status (as of 2026-08-21): SCORED 2026-08-13 in two parts — NOT SHIPPED.**
> (1) `changelog/2026-08-13-date-level-rescaling-passes-its-offline-gate.md` — passes the offline
> gate at 3/7/30d and **voids its own instrument at 14d**.
> (2) `changelog/2026-08-13-the-rescaling-is-a-level-fix-not-a-spread-fix.md` — on a clean audited
> anchor set the **pre-registered mechanism claim (S) FAILS**; the effect is a *level* fix, not a
> spread fix. The follow-up low-`L[t]` read also failed
> (`changelog/2026-08-13-the-low-level-read-fails-its-axis.md`).
> **Superseded:** `sigma` is no longer the band denominator — the featureless climatology scale
> shipped 2026-08-20 (`docs/research/2026-08-19-climatology-vs-gbm-band.md`).


**Date:** 2026-08-13 (written and committed **before** the read)
**Instrument:** offline first — `backend/scripts/attribute_band_level.py` extended, read-only. A
paired `model-diagnostics.yml` dispatch **only if the offline legs pass**.
**Question opened by:** `docs/changelog/2026-08-13-the-band-is-sized-on-trailing-volatility.md`
**Bars, placebo, basis and void conditions are fixed here. Nothing below may be revised after a
number is seen.**

## The mechanism, stated as one testable claim

`sigma = price_std_60d / price` is a **trailing** window whose cross-sectional level swings by
quarter — median 0.066 / 0.115 / 0.066 / 0.115 / 0.069 for 2025Q3→2026Q3 — while the forward return
it is supposed to scale does not swing with it. Measured on each item's own two-year history: served
`sigma` **1.488 / 1.502 / 1.448 / 1.699×** the item's norm, realised `|resid|` **1.123 / 0.984 /
1.027 / 1.106×**, so the conformal score `|resid| / sigma` is **0.755 / 0.655 / 0.709 / 0.651×** the
pooled score and a pooled-`p80` `q_hat` over-covers.

**The claim:** a *part* of `sigma`'s date-level level is uninformative about forward dispersion, and
dividing it out moves served coverage toward 80% and flattens coverage across dates.

## The dose-response already in hand, and a correction to yesterday's note

⚠️ **Yesterday's entry said the audited anchor set is "concentrated in one volatility regime". That
is wrong** — it rested on the three anchors the panel instrument could measure, because
`prepare_targets` voids every `2026-07-06` label as spanning a collector cutover. On the served side
the set spans a **2.2× range** of trailing-vol level, and served coverage tracks it:

| anchor | `sigma` vs own history | cov% h=3 | h=7 | h=14 | h=30 |
|---|---:|---:|---:|---:|---:|
| 2026-04-22 | **1.888×** | 89.85 | 94.04 | 95.90 | 93.76 |
| 2026-05-16 | 1.528× | 85.58 | 91.52 | 82.38 | 88.12 |
| 2026-06-16 | 0.994× | 85.99 | 89.45 | 85.99 | 84.92 |
| 2026-07-06 | 0.857× | 88.90 | 80.86 | 80.08 | 77.94 |

Spearman rank correlation with the `sigma` ratio: **0.2 / 1.0 / 0.8 / 1.0** at 3/7/14/30d
(n = 4, so this is a pattern to test against, not a result). The across-anchor coverage spread is
**4.27 / 13.18 / 15.82 / 15.82pp**. **This is the quantity the arm has to shrink**, and it is why the
read is worth running despite the neighbouring refutation.

## The arm

`sigma_tilde[i,t] = sigma[i,t] / L[t] ** gamma`, where `L[t]` is the **cross-sectional median of
`sigma` over the ≥$1 training universe on date `t`**, and `q_hat` is calibrated on `sigma_tilde`.

- `gamma = 0` is production exactly. `gamma = 1` removes the whole date-level level.
- `gamma` is **fitted, not chosen**: `gamma = 1 − b`, where `b` is the OLS slope of
  `log median_i |resid[i,t]|` on `log L[t]` across dates. If forward dispersion tracks trailing level
  one-for-one, `b = 1`, `gamma = 0`, and production is already right.
- **Pre-registered point prediction: `b` ≈ 0.20–0.30, so `gamma` ≈ 0.70–0.80.** From
  `log(1.10) / log(1.50) = 0.235` on the same-items ratios above. A fitted `b` outside [0, 1] voids
  the arm (see below).

**Why this is not the refuted class.** `2026-08-12-the-band-is-tilted-in-sigma.md` refuted
conditioning `q_hat` on a **date-level state** (a 60-day trailing `q_hat`, three volatility states),
with placebos at ≤0.11pp. Two differences, and both are arguments rather than results — stated as
such **before** the read:

1. That comparison was scored **level-matched to 80% marginal coverage**, which removes exactly the
   quantity this arm targets before the comparison begins.
2. It moved `q_hat`; this rescales `sigma` itself, so the **within-date ordering of items is
   untouched** and the tilt work (elasticity 0.43/0.37/0.34/0.32, positive ramp at 15 of 16 cells) is
   orthogonal to it. `SIGMA_EXPONENT` and this arm are composable in principle and **must not be
   tested together**.

If the placebo below fires, this arm joins that refuted class and the two arguments are withdrawn.

## Predictions, before the run

**Basis declaration, required by `2026-08-12-sigma-exponent-paired-read.md`.** Every coverage bar
below is on the **SERVED** band via `scripts/replay_serving.py`'s `BAND COVERAGE`. Widths are reported
on **both** bases (calibration-set and served) and no bar is placed on a width. `q_hat` is **not
comparable across this flag** — it absorbs `L ** gamma` — so it is never differenced, only reported.

**(V) Instrument validity.** Recomputing the control's per-anchor coverage from the panel's own scores
must reproduce the served table above to **≤ 3pp mean absolute error** at each horizon. The panel is a
stand-in (own clip bounds, own cohort, realised return as the residual); if it cannot reproduce the
control it cannot referee the arm, and the read stops here.

**(P) Placebo — the leg that decides whether this is the refuted class.** Re-run with `L[t]` **shuffled
across dates** (each date receives another date's level, one permutation seed fixed in advance at
20260813, 200 permutations). A genuine date-level channel must collapse under the shuffle.
**Bar: |Δ coverage| ≤ 1.0pp at every horizon under permutation, against the arm's own effect.** If the
shuffled arm moves coverage as much as the real one, the effect is a rescaling artifact — `q_hat`
re-absorbing a constant — and **the arm is refuted regardless of every other leg.**

**(A) Primary — the level.** Served marginal coverage must be **closer to 80% than the control at ≥3
of 4 horizons**, with no horizon moving further than **2pp** away. Control: 87.58 / 88.97 / 86.09 /
86.19%. ⚠️ **Overshoot is a failure, not a partial success** — `SIGMA_EXPONENT` fell 8pp and turned
77.80% into 69.72%, and this arm has the same shape of risk.

**(S) Secondary — the spread, which is the actual claim.** The across-anchor coverage spread must fall
by **≥ 30%** at ≥3 of 4 horizons from 4.27 / 13.18 / 15.82 / 15.82pp. A level fix that leaves the
spread intact has not removed a date-level term; it has moved a constant.

**(D) Dose-response.** After the arm, the Spearman correlation between an anchor's `sigma` ratio and
its served coverage must fall in magnitude at ≥3 of 4 horizons from 0.2 / 1.0 / 0.8 / 1.0.

**(N) Neutrality.** The within-date `sigma` ramp (`BAND COVERAGE BY SIGMA`) must be unchanged within
**±3pp**. This arm is a date-level rescaling and **must not** touch the item-level tilt; if it does,
`L[t]` is not what is being divided out.

## The anchor set — fixed now, six anchors, regime-spanning

All six pass `audit_anchor_feed`, all carry 735–944 items of the ≥$1 cohort, and all resolve h=30
without crossing the 2026-03-22 consensus break:

| anchor | `sigma` vs own history | regime |
|---|---:|---|
| 2026-02-14 | 0.811× | low |
| 2026-03-10 | 0.698× | low |
| 2026-04-06 | 1.491× | high |
| 2026-04-22 | 1.888× | high |
| 2026-06-16 | 0.994× | norm |
| 2026-07-06 | 0.857× | low |

**Six, not four**, because bars (S) and (D) are about variation *across* dates and four points cannot
carry them. ⚠️ **The replay costs ~4 minutes per anchor**, so six anchors × four horizons will not fit
one 30-minute job — the dispatch must be **two jobs of three anchors** or horizons must be split.
That cost is why (V) and (P) run **offline first**: a dispatch that fails its placebo is 50 minutes of
runner time spent on a refutation the panel could have delivered in 90 seconds.

## Void conditions

- Fitted `b` outside **[0, 1]**, or its bootstrap CI containing both 0 and 1 — the arm has no
  defensible `gamma` and nothing is dispatched.
- (V) fails: the panel cannot reproduce the control's served coverage to 3pp.
- Any anchor is refused by `audit_anchor_feed` at run time (the archive moves; the CI copy is ahead of
  the local one).
- `L[t]` computed on a different item universe in calibration and serving. Both must use the ≥$1
  training universe, asserted in code, **not** the served cohort — a cross-sectional median over two
  different item sets is a basis difference, which is the trap that cost two cancelled dispatches on
  2026-08-12.
- Both `SIGMA_EXPONENT` and this flag set at once.

## Cost

Offline legs: ~3 minutes for both panels, no dispatch, no retrain. Dispatch, only on a pass: two
diagnostics jobs, ~25 minutes each, within the cap individually. The rescaling itself is arithmetic —
like `SIGMA_EXPONENT`, it adds no model fit.

## What a pass would and would not mean

A pass means the band's **level** is fixable by an arithmetic change to `sigma`, at zero training
cost, with the item-level tilt left for a separate decision. It would **not** license shipping: the
level has been observed to be period-dependent (`SIGMA_EXPONENT`'s control covered 74–77% on an
earlier period), so a shipping decision needs the walk-forward panel across regimes, not six anchors.
