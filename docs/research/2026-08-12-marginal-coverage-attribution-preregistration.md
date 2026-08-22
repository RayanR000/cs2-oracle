# Pre-registration: does the `sigma`-mix shift explain the marginal over-coverage?

> **Status (as of 2026-08-21): SCORED 2026-08-12 — every pre-registered leg PASSED; verdict
> PARTIAL at 4 of 4.** `changelog/2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md`:
> the served `sigma`-mix shift explains ~40–60% of the marginal over-coverage, and the 30d
> counter-example was an artifact.
> **Forward pointer:** the remedy line that followed is dead — `sigma` is no longer the served
> band denominator. The featureless climatology scale shipped 2026-08-20 and beats `sigma` at
> matched coverage (`docs/research/2026-08-19-climatology-vs-gbm-band.md`), and
> `CLIMATOLOGY_REACTIVE` — the regime-reactive follow-up — was **shelved** after a prod A/B
> (`changelog/2026-08-20-climatology-reactive-band-scale.md`).


**Date:** 2026-08-12 (written and committed **before** the read)
**Instrument:** `backend/scripts/attribute_marginal_coverage.py` (new, offline, read-only)
**Question opened by:** `docs/changelog/2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md` §"One lead
it does open, stated as a lead"

## The lead, stated as a testable mechanism

The band's conditional coverage is a monotone increasing function of `sigma` — measured
**62→93 / 60→95 / 57→96 / 52→97%** across `sigma` deciles, level-matched, on real OOF residuals.
`q_hat` is fitted so that this curve integrates to 80% over the **calibration** `sigma`
distribution. If serving sees a **different** `sigma` distribution, the same curve integrates to
something else — with no other defect required.

Served `sigma` is on record at **1.28 / 1.34 / 1.33 / 0.93×** the calibration median
(`2026-08-12-conformal-basis-follows-serving.md`). Production's marginal coverage is
**87.2 / 91.8 / 90.6 / 89.0%** against 80% — an excess of **7.2 / 11.8 / 10.6 / 9.0pp**.

So: **how many pp of that excess does a `sigma`-mix shift of the observed size actually buy?**

## Predictions, before the run

**Instrument validity (V).** Coverage is modelled as `c(sigma) = F_ε(log q_hat − a + (1−β)·log sigma)`
from the log-log fit `log|resid| = a + β·log sigma + ε`. It is valid only if it reproduces the
**empirical** decile coverage profile it is fitted on. **Bar: mean absolute error ≤ 2pp per decile
at ≥3 of 4 horizons.** If it fails, only the in-support empirical-bin leg is readable and the
extrapolated shift figures are void.

**Placebo (P).** Re-run the whole shift calculation with `β` forced to **1.0** — the exponent split
conformal assumes. Then `c(sigma)` is flat and a `sigma`-mix shift **must** move predicted marginal
coverage by **0.00pp at every `k`**. Anything else means the calculation has a path that is not the
tilt, and the read is void.

**Primary (A).** Attribution fraction `A(h) = (predicted marginal at published k − 80) / (observed
marginal − 80)`, per horizon.

- `A ≥ 0.50` at ≥3 of 4 → the shift is a **material** cause and the remedy list changes.
- `0.20 ≤ A < 0.50` → **partial**; it is a contributing term, not the cause.
- `A < 0.20` at ≥3 of 4 → **refuted as a primary cause**, and the marginal defect stays
  unattributed after a seventh cause.

**Stated in advance: 30d cannot pass.** `k = 0.93 < 1` predicts *under*-coverage while production
over-covers at 89.0%, so `A(30) ≤ 0`. That is a property of the mechanism, not a result of this run,
and it is why the bar is 3 of 4 rather than 4 of 4. A mechanism that predicts the wrong sign at one
of four horizons cannot be the sole cause whatever the other three do.

**Inverse (K).** The shift `k*` that would be needed to produce the *whole* observed excess, solved
by bisection. Compared against the **natural range of per-date `sigma`-median ratios** over the
panel's own dates. If `k*` exceeds the maximum ratio any single date in ~2 years achieves, the
mechanism cannot produce the excess **even in principle**, independently of `A`.

**Footprint (F).** The mechanism claims marginal coverage on a date is a function of that date's
`sigma` mix. So predicted-from-`sigma`-mix coverage per date should correlate **positively** with
realised coverage per date on the panel. **Bar: positive at ≥3 of 4 horizons.** A null here is
evidence against the channel that does not depend on the size of any shift.

## What this run fixes about the input

The published **1.28 / 1.34 / 1.33 / 0.93×** implies served `sigma` as `half_pct / q_hat` — the same
reconstruction whose elasticity was refuted at 4 of 4 the same day. This run measures the served
side **directly**: `price_std_60d / price` through `conformal.sigma_from_columns` on a voted panel
covering **2025-10-26 → 2026-08-08**, which spans production's actual forecast anchors, with one
common clip applied to both sides. Both the published ratio and the directly measured one are
reported, and `A` is reported against **both**.

⚠️ `sigma` is a property of an anchor date, **not of a horizon** — it carries no `h` term. The four
published ratios can therefore only differ because the served row *sets* differ per horizon, and
production has **7 / 6 / 3 / 1** forecast dates at 3/7/14/30d. If the 30d ratio of 0.93 rests on a
single date, the sign objection above is a one-date sample and must be reported as such.

## Void conditions

1. Placebo `P` moves by more than 0.05pp at any `k` → void.
2. Validity `V` fails at ≥2 horizons → the extrapolated legs are void; the empirical-bin leg stands.
3. Fewer than 100 calibration dates at any horizon → that horizon is not reported.

## Cost

Seconds, offline, read-only, no dispatch. Reads voted panels and writes a CSV.
