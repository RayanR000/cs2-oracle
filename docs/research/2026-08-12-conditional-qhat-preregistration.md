# Pre-registered read: can a conditional `q_hat` fix the band's coverage?

> ## ❌ SCORED 2026-08-12 — VOID on the placebo clause, and the answer is on the other axis.
> `changelog/2026-08-12-the-band-is-tilted-in-sigma.md`.
> `S2` on the cross-sectional MAD and `S3` both cleared the primary bar — **and so did `P2`, the
> same `S2` with its state shuffled across dates** (M2 better at 3/4, M1 in band 4/4). The clause
> below fires: the read is void and **M2 is the wrong statistic.** It falls whenever *marginal*
> coverage moves from 83% toward 80%, information or not, so a uniformly lower band scores as a
> conditional fix. `P1` had no power either — "random dates from all history" *is* the pooled set,
> and it reproduced `S0` to ±0.02pp.
>
> **Level-matched to 80% marginal first** (post-hoc, and labelled as such), the picture inverts:
> **every date-conditional scheme is worse than pooled**, the placebos move ≤0.11pp so nothing is
> being drowned out, and **a time-varying `q_hat` is refuted** — the same wall
> `2026-08-11-market-factor-is-not-forecastable-from-its-own-history.md` hit. What carries the
> whole defect is `S3`'s axis: coverage ramps **62→95%** (h=3) to **58→98%** (h=30) across `sigma`
> deciles, monotone in all ten, and a fitted **β = 0.408 / 0.401 / 0.363 / 0.327** cuts the
> stratum error from **8.0–9.7pp to 0.6–1.8pp** at 4/4. Not the clip.
>
> ⚠️ **The magnitude is unresolved.** This contradicts the **0.798 / 0.692 / 1.034 / 1.150** in
> `2026-08-12-conformal-basis-follows-serving.md`, which implied `sigma` as `half_pct / q_hat` from
> ~20K prod rows and called the tilt second-order. The committed interpretation below holds: a
> confirm dispatch on **real OOF residuals**, and no implementation before it.

**Written 2026-08-12 before any scheme was scored.** It is the direct continuation of
`changelog/2026-08-12-expanding-window-refuted-for-band-width.md`, which closed the whole
*"calibrate on different rows"* remedy class — the centre, quiet dates, the denominator,
`sigma`'s level and the expanding window are all excluded — and named what is left:

> A **scalar** `q_hat` cannot track a several-fold swing in cross-sectional dispersion, so the
> remedy class is a **conditional** `q_hat` — time-varying, or a function of a realised-volatility
> state. It has not been designed, costed or measured.

This document designs and costs it, and fixes the bar before the measurement.

## The two defects being aimed at

Both are on record and both are the same fact:

1. **Marginal over-coverage.** The served band covers **87.2 / 91.8 / 90.6 / 89.0%** against an
   80% target on 19,917 prod outcomes; the half-width has to shrink to **0.72 / 0.61 / 0.73 /
   0.77×**.
2. **Conditional over- and under-coverage.** Per (horizon, date) the same band runs
   **58.2%–99.2%**, and `sigma` carries no date term. Per-fold `q_hat` varies **1.56–2.25×** at
   identical `n_train`, synchronised across horizons at pairwise Spearman **0.70–1.00**.

A scheme that fixes (1) and leaves (2) is not the remedy this is scoped as.

## The instrument, and why this is not a CI dispatch

The conformal score is `|actual − r̂| / sigma`. This model's predicted `|return|` is **median
0.95% / p90 9.01%** against half-widths of 10–31%
(`2026-08-11-conformal-centre-follows-serving.md`), so **`|actual| / sigma` is the score to first
order** and needs no model at all. That makes the design space explorable offline over hundreds of
dates instead of one ~30-minute dispatch per candidate — which is the stated bottleneck
(`2026-08-10-next-steps.md`, "the velocity problem is the experiment loop").

**The instrument was validated before this document was written**, on the local voted panel
(2024-07-09 → 2026-07-09, 731 dates, the 946-item ≥$1 cohort, 597,992 item-days), using
production's own `prepare_targets` and `conformal.sigma_from_columns`:

| h | pooled p80 of `|actual|/sigma` | shipped `q_hat` (run `31564943172`) | ratio |
|---|---:|---:|---:|
| 3 | 96.59 | 94.72 | **1.020** |
| 7 | 135.82 | 141.77 | **0.958** |
| 14 | 181.31 | 204.34 | **0.887** |
| 30 | 245.04 | 312.05 | **0.785** |

The ratio falls monotonically in `h`, which is what dropping `r̂` predicts: the omitted term grows
with the horizon. 3d and 7d are tight; **30d is declared provisional in advance** at 0.785, and
14d is borderline.

## Schemes, fixed in advance

`S0` is production. Every other scheme forms `q_hat[d]` using only anchors on or before
`d − h − 13` — the `embargo_days` band, so nothing in the calibration set can have resolved after
the forecast is made.

| id | scheme |
|---|---|
| `S0` | **pooled** (control): one scalar, the finite-sample-corrected p80 over the whole calibration set. |
| `S1` | **trailing**: `q_hat[d]` = p80 of realised scores over anchors in `[d − h − 13 − W, d − h − 13]`, `W = 60` days. Falls back to `S0` when the window holds < 2,000 rows. |
| `S2` | **state**: `log q_hat[d]` regressed on one state variable, fitted on embargoed history only. |
| `S3` | **sigma exponent**: score becomes `|r| / sigma^β`, `β` fitted on embargoed history. Orthogonal to `S1`/`S2` — it conditions on the *item*, not the date. |
| `S4` | `S1` ∘ `S3`. |

`S3` is in the design because the exponent is already measured wrong: `d log|resid| / d log sigma`
is **0.798 / 0.692** at h=3/7 against the **1.000** split conformal's normalisation assumes.

State variables for `S2`, all computable at serve time from date `d` with no forward look:

- `V1` cross-sectional median `sigma` on `d`
- `V2` cross-sectional MAD of `return_1d` on `d`
- `V3` trailing 20-day sd of the market factor (mean cross-sectional `return_1d`)

## The bar, fixed in advance

Walk-forward over test dates, per horizon. Per-date coverage `cov[d]` is the fraction of rows
anchored on `d` with `score ≤ q_hat[d]`.

- **M1 — marginal coverage.** Pooled over test rows. Target 80%.
- **M2 — conditional-coverage error.** `mean_d |cov[d] − 80%|`. This is the primary statistic.
- **M3 — fraction of dates inside 80 ± 10pp.** Reported, not a bar.

**Primary:** the winning scheme reduces **M2** against `S0` at **≥ 3 of 4 horizons**, *and* holds
**M1** inside **80 ± 3pp** at **≥ 3 of 4 horizons**. Both legs, because shrinking M2 while
breaking marginal coverage is not a fix — it is a different miscalibration.

**Placebo, and it is required.** A per-date `q_hat` is noisier than a scalar, and per-date coverage
is measured on ~900 rows, so M2 can move for reasons that carry no information. The placebo re-runs
the winning scheme with its date ordering destroyed — the state variable shuffled across dates
(`S2`), or the trailing window drawn from random non-adjacent embargoed dates (`S1`). **The placebo
must fail the primary bar.** If it passes, the read is **void** and the finding is that M2 is the
wrong statistic, not that the scheme works. This is the `ab-fold-count-floor` rule applied at the
date level: read placebo before treatment.

**Void conditions:**

- Fewer than 100 walk-forward test dates at a horizon → that horizon is unreadable, not null.
- Two or more horizons void → the whole read is void.
- `S0`'s own per-date coverage range must be **wide** on this panel. If it comes out near 80% at
  every date, the offline panel does not contain the defect and nothing here is measuring it.

## Declared confounds

- **`r̂ = 0`.** The instrument is model-free, so it orders *dates* by the dispersion of realised
  returns rather than of residuals. The two coincide only to the extent `r̂` is small, and the
  validation ratio says that holds at 3d/7d and degrades to 0.785 by 30d. **Any pass must be
  confirmed on real OOF residuals in one dispatch before it is implemented** — this read decides
  whether that dispatch is worth spending, not whether to ship.
- **One window.** 731 dates over 2024-07-09 → 2026-07-09, including the Oct-2025 crash, against
  the 2023–2026 span the CV folds cover. A scheme that works here is not thereby shown to work on
  the fold geometry production calibrates on.
- **The local archive copy runs behind the durable one**, and this panel is a cached voted frame
  with a 2026-07-09 cutoff. It is the right basis for a design question and the wrong one for a
  published production figure.
- **Overlapping windows.** Adjacent anchors share `h − 1` days of outcome, so `cov[d]` is
  autocorrelated and M2 has no naive binomial standard error. Every scheme is scored on the
  **same dates**, so the comparison is paired and the autocorrelation is common to both arms;
  no absolute interval is claimed for M2.
- **Label voiding is production's.** `prepare_targets` supplies the frozen-run, snapshot and
  collection-shift rules and the ±500% winsorization, so voided rows are absent from every arm
  identically.

## Committed interpretation

- **Primary passes, placebo null** → the conditional `q_hat` is designed, costed and measured, and
  the next step is a confirm dispatch on real OOF residuals — **not** a ship. The cost of the
  scheme itself is a quantile over an array already in memory.
- **Primary fails at ≥ 2 horizons** → the date axis fails too. Coverage is then a **`sigma`-model**
  problem (`S3`) rather than a calibration-set, regime or date problem, and the whole
  time-varying class joins the five already excluded. Record and stop.
- **`S3` passes where `S1`/`S2` fail** → the remedy is the normalisation exponent, which is a
  one-line change in `models/conformal.py` and reachable without any date-level state.
- **Placebo passes** → void. Report the statistic's failure, not a scheme.
