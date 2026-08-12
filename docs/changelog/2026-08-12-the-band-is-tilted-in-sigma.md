# The band is not miscalibrated in time — it is tilted in `sigma`

> ## ✅ CONFIRMED the same day on real OOF residuals — run `31619383780`.
> `changelog/2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md`. The elasticity is
> **0.429 / 0.369 / 0.350 / 0.313** on ~157K real conformal records, against the
> **0.408 / 0.401 / 0.363 / 0.327** the model-free instrument below predicted — accurate to
> **0.014–0.032**, which retires this entry's largest declared confound. The
> **0.798 / 0.692 / 1.034 / 1.150** on record is refuted at 4/4, and hardest at the two horizons it
> called fine: 14d and 30d are where the tilt is worst (level-matched deciles **57→96** and
> **52→97**, err 9.49 / 10.07pp). Not the clip — excluding all 1.8–1.9% of clipped rows moves the
> exponent *further* from 1.0.
>
> ⚠️ **The one thing this entry got optimistic is the remedy's reach.** Held out — `β` fitted on
> every fold but the last, scored on the last — the conditional error falls **7.26→1.78pp (−76%)**
> at 3d and **7.05→2.65pp (−62%)** at 7d, but only **6.01→4.44pp (−26%)** at 14d and
> **6.09→5.39pp (−12%)** at 30d. A single global exponent is a fix at the short horizons and a
> partial one at the long ones. Nothing shipped.

**Date:** 2026-08-12
**Pre-registration:** `docs/research/2026-08-12-conditional-qhat-preregistration.md`
**Instrument:** `backend/scripts/measure_conditional_qhat.py` (new, offline, read-only)
**Status:** the pre-registered read is ❌ **VOID on its own placebo clause.** A post-hoc
level-matched re-read is decisive and points at a different axis. **Nothing shipped** — the daily
path is unchanged and no artifact was promoted.

## What was being tested

`2026-08-12-expanding-window-refuted-for-band-width.md` closed the whole *"calibrate on different
rows"* remedy class — the centre, quiet dates, the denominator, `sigma`'s level and the expanding
window are all excluded — and named what was left: a **conditional** `q_hat`, time-varying or a
function of a realised-volatility state, *"undesigned, uncosted, unmeasured."* This is that design,
that cost, and that measurement.

## The instrument, and why this was not a dispatch

The conformal score is `|actual − r̂| / sigma`, and this model's predicted `|return|` is median
0.95% / p90 9.01% against half-widths of 10–31%, so **`|actual| / sigma` is the score to first
order and needs no model at all.** That turns a question that costs one ~30-minute dispatch per
candidate into a question that costs seconds per candidate.

Validated *before* the pre-registration was written, on the local voted panel (731 dates,
2024-07-09 → 2026-07-09, the 946-item ≥$1 cohort, 597,992 item-days), using production's own
`prepare_targets` and `conformal.sigma_from_columns`:

| h | pooled p80 of \|actual\|/sigma | shipped `q_hat` (`31564943172`) | ratio |
|---|---:|---:|---:|
| 3 | 96.59 | 94.72 | **1.020** |
| 7 | 135.82 | 141.77 | **0.958** |
| 14 | 181.31 | 204.34 | **0.887** |
| 30 | 245.04 | 312.05 | **0.785** |

The ratio falls monotonically in `h`, exactly as dropping `r̂` predicts. And the panel contains the
defect being chased: the walk-forward pooled control over-covers (81.9 / 82.8 / 83.0 / 81.7%
against 80%) with per-date coverage running **16–98%**, against production's 87.2–91.8% and
58.2–99.2%. Nothing here is a synthetic generator.

Every `q_hat` in the run comes from `conformal.calibrate`, including the finite-sample level
`ceil((n+1)(1−α))/n`. Every label-voiding rule is `prepare_targets`'. Nothing is reimplemented.

## Why the pre-registered read is void

Five schemes were pre-registered: `S0` pooled (production), `S1` a 60-day trailing window, `S2` a
regression of `log q_hat` on one of three date-level state variables, `S3` a fitted `sigma`
exponent, `S4` = `S1`∘`S3`. Every one forms `q_hat[d]` from anchors on or before `d − embargo_days(h)`.

Two arms cleared the pre-registered bar — and **so did the placebo**:

| arm | M2 better than `S0` | M1 in 80±3pp | verdict |
|---|---:|---:|---|
| `S2` on cross-sectional MAD of `return_1d` | 3/4 | 4/4 | PASS |
| `S3` `sigma` exponent | 3/4 | 4/4 | PASS |
| **`P2` — the same `S2`, state SHUFFLED across dates** | **3/4** | **4/4** | **PASS** |

The pre-registration says what to do here: *"If the placebo passes, the read is void and the
finding is that M2 is the wrong statistic, not that the scheme works."*

**And the reason is worth more than the read was.** `M2 = mean_d |cov[d] − 80%|` falls whenever
marginal coverage moves from 83% toward 80%, for any reason at all — including a uniform
downscaling that carries no conditional information whatsoever. A shuffled state variable still
produces a *lower* band on average, so it still scores as a conditional-coverage fix. **M2
conflates a level correction with a conditional one, and it cannot be used to referee this
question.** `P1` exposes a second flaw: drawing the trailing window from random embargoed dates
reproduces `S0` to ±0.02pp, because "random dates from all history" *is* the pooled set. That
placebo has no power by construction.

## The level-matched re-read

Each arm's `q_hat[d]` is multiplied by the single scalar `c` that puts its marginal coverage at
**exactly 80%**, and only then is conditional coverage compared. `c` is fitted on the test rows —
in-sample for the level, deliberately, and applied identically to the control and both placebos, so
the comparison stays paired and the only thing left that can separate the arms is conditional
information. **This is post-hoc and is labelled as such everywhere it appears.**

Δ against pooled `S0`, in pp, at 3/7/14/30d — negative is better:

| arm | ΔM2\* on **dates** | ΔM2\* on **`sigma` deciles** |
|---|---|---|
| `S1` trailing 60d | +1.45 +0.87 +0.49 +1.89 | +0.37 +0.76 +0.94 +0.56 |
| `S2` xs median `sigma` | +0.31 +0.92 +0.69 −0.51 | −1.47 −2.03 −2.37 −2.77 |
| `S2` xs MAD `return_1d` | +0.02 −0.40 −0.37 +0.36 | +0.25 +0.23 +0.34 +0.26 |
| `S2` 20d market vol | +0.31 +0.27 +0.69 +0.81 | +0.42 +0.66 +0.65 +0.26 |
| `S3` `sigma` exponent | +0.25 +0.38 −0.01 −1.44 | **−8.05 −6.83 −6.18 −8.72** |
| `S4` = `S1`∘`S3` | +0.56 +0.31 +0.15 +0.62 | −7.88 −7.47 −6.88 −8.89 |
| `P1` placebo | +0.00 +0.02 +0.00 +0.00 | +0.00 −0.00 +0.00 +0.00 |
| `P2` placebo | −0.03 −0.04 +0.04 +0.11 | +0.05 +0.09 +0.18 +0.12 |

**1. The date axis is empty.** Once the level is matched, no date-conditional scheme beats pooled:
the trailing window is worse at 4/4, and the three state variables are worse at 3/4, 2/4 and 4/4.
The placebos move ≤0.11pp, so this statistic has almost no noise floor — the schemes are not being
drowned out, there is nothing there to find. **A time-varying `q_hat` is refuted**, and it joins
the five causes already excluded.

This is the same wall N2 hit. `2026-08-11-market-factor-is-not-forecastable-from-its-own-history.md`
found the market factor unforecastable from its own history, and
`2026-08-11-mean-reversion-does-not-replicate.md` closed that leg in both directions. A date-level
`q_hat` needs to forecast the date's realised cross-sectional dispersion, which is the same
quantity, and it fails the same way.

**2. The `sigma` axis is the whole defect.** Level-matched to 80% marginal, production's `β = 1`
normalisation leaves coverage a monotone ramp across `sigma` deciles:

| h | coverage by `sigma` decile, `β = 1` (production) | `β` fitted |
|---|---|---|
| 3 | 62 68 72 76 80 83 86 88 91 **95** | 82 80 79 79 80 80 80 80 79 81 |
| 7 | 62 69 73 77 80 83 85 87 90 **96** | 83 82 81 80 80 80 79 78 77 79 |
| 14 | 61 69 73 77 80 83 84 86 90 **97** | 83 83 82 81 81 80 78 77 76 79 |
| 30 | 58 67 72 76 79 83 85 89 93 **98** | 81 81 80 80 80 80 78 79 79 83 |

A **33–40pp tilt**, monotone in all ten deciles at all four horizons. The stratum error falls from
**8.62 / 8.02 / 8.00 / 9.73pp** to **0.57 / 1.20 / 1.82 / 1.01pp** — an 85–93% reduction, at 4/4,
with both placebos at 0.00. The fitted exponent is **0.408 / 0.401 / 0.363 / 0.327**.

**The mechanism, stated without a model.** Across `sigma` deciles at h=3, median `sigma` spans
**0.028 → 0.308 (11.0×)** while the median `|residual|` it is meant to scale spans only
**1.97% → 5.07% (2.6×)**; at h=30, 11.3× against 2.25×. So `median |resid| / sigma` falls
**0.711 → 0.164** and **2.054 → 0.406**. `sigma`'s cross-sectional range is roughly **4× too wide
for the dispersion it normalises**, and dividing by `sigma¹` over-corrects by that factor.

**It is not the clip.** Only **1.19% / 1.23%** of rows sit at the `sigma` floor and **1.00% /
1.03%** at the cap (h=3 / h=30), the tilt is monotone across all ten deciles rather than
concentrated at the ends, and dropping every clipped row moves the whole-panel exponent from
**0.389 → 0.395** (h=3) and **0.348 → 0.369** (h=30).

## This contradicts the elasticity already on record, and the disagreement is not resolved

`2026-08-12-conformal-basis-follows-serving.md` measured the same elasticity as **0.798 / 0.692 /
1.034 / 1.150** and concluded *"at h=14 and h=30 it is fine, or slightly under-corrected… it is not
worth 25–39% of width."* This read gets **0.408 / 0.401 / 0.363 / 0.327** — below 1 at **4 of 4**,
including the two horizons that read as fine.

Neither number is obviously the right one and the difference has three candidate causes, in
descending order of how much they would matter:

- **`sigma` was implied, not computed.** That read reconstructed it as `half_pct / q_hat` from
  ~20K prod outcomes served by artifacts whose `q_hat` differ by up to ~3%. This one computes it
  from `price_std_60d / price` on ~500K item-days. The caveat is stated in that entry.
- **`r̂ = 0` here.** A booster's residual could scale with `sigma` more steeply than the raw return
  does, which would push the true exponent up toward that read's numbers.
- **Different populations.** Prod outcomes on ~10 forecast dates against 588/576/555/507
  walk-forward dates.

What both reads agree on is the **sign**, at every horizon in this one and at the two short
horizons in that one. What is unresolved is the **magnitude**, and the magnitude is the whole
question of how much width the fix is worth. That read also dismissed the tilt partly because
`p80(s)` was *below 1 in 19 of 20 strata* — which is the uniform over-coverage, not the tilt, and
is exactly what level-matching removes before looking.

## What this changes

- **The remedy class named on 2026-08-12 is the wrong one.** "Conditional `q_hat`" was scoped as
  time-varying / regime-following. That half is refuted. The conditional axis that carries the
  defect is the **item's `sigma`**, and it needs no date-level state, no feedback series and no ACI.
- **`C5`'s calendar block is irrelevant to it.** ACI was parked because production has 7/6/3/1
  forecast dates. The `sigma` axis is cross-sectional — ~500K rows per horizon are already
  available offline, and ~50K OOF rows per fold in any CV run.
- **The fix needs no new math in `models/conformal.py`.** `calibrate` and `band` already take the
  denominator as an argument, so `sigma ** beta` expresses it; `beta` is a fitted constant the
  artifact would persist beside `sigma_bounds`, the way `q_hat` already is.
- **A blind spot in the test suite is closed.**
  `test_calibrate_gives_conditional_coverage_across_volatility_strata` builds residuals at
  `scale = sigma * 10.0` — elasticity **exactly 1.0 by construction** — so it passes whatever the
  archive does. `test_sigma_normalization_fails_when_the_elasticity_is_not_one` now pins the defect,
  asserts marginal coverage is *unaffected* by it (which is why nothing caught it), and shows the
  exponent restores the deciles to within 10pp.

## What is NOT claimed

- Not a ship, and not a size. The pre-registration committed to a confirm dispatch on real OOF
  residuals before implementation, the primary bar came back void, and the deciding statistic is
  post-hoc. Two of three reads of this exponent now disagree about its value.
- Not a fix for per-date coverage. `S3` leaves the per-date spread where it was (level-matched
  sd 11.5pp against `S0`'s 10.9pp at h=3). The date-level spread is real, unexplained by `sigma`,
  and — on this evidence — not reducible from any observable tried.
- One window, one panel: 2024-07-09 → 2026-07-09, the local voted cache, which runs behind the
  durable archive. Right basis for a design question, wrong one for a published production figure.

## The next step, costed

**One `model-diagnostics.yml` dispatch, control arm, all four horizons**, reporting
`d log|resid| / d log sigma` and level-matched coverage by `sigma` decile from the **real OOF
conformal records** — the population `q_hat` is actually fitted on. That resolves the 0.33–0.41 vs
0.69–1.15 disagreement, which is the only thing standing between this and a one-constant change.
It is a report-only addition to a run that already exists (`31611508808`'s shape), so it costs a
dispatch and no new model time.

Do **not** implement the exponent before that read. Three of the five causes excluded in this
investigation were refuted by a *sign*, not a size, and this one currently has two candidate sizes
that differ by 3×.

## Reproducing

```
cd backend && venv/bin/python -m scripts.measure_conditional_qhat --horizons 3,7,14,30
```

Read-only: it reads a voted panel and writes a CSV. `--voted` overrides the panel; the default
picks the cache with the **longest date span**, not the newest — the cache key carries a cutoff
date, so a replay anchor leaves a 283-date file beside a 731-date one and picking by mtime silently
halves the panel with no symptom but a smaller anchor-date count. The provenance line is logged;
absolute `q_hat` levels from this script are not the artifact's, because it derives `sigma_bounds`
from its own panel. Read ratios.
