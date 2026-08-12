# The marginal over-coverage is ~40–60% a served `sigma`-mix shift, and the 30d counter-example was an artifact

**Date:** 2026-08-12
**Pre-registration:** `docs/research/2026-08-12-marginal-coverage-attribution-preregistration.md`
(committed as `4dc7514`, **before** the run)
**Instrument:** `backend/scripts/attribute_marginal_coverage.py` (new, offline, read-only)
**Tests:** `backend/tests/test_marginal_coverage_attribution.py` (6 tests)
**Follows:** `2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md` §"One lead it does open"
**Status:** ✅ every pre-registered leg passes. Verdict **PARTIAL** at 4 of 4. ⚠️ **Refutes** the
`0.93×` served `sigma` at 30d on record, and with it the sign objection that had this lead marked
"at best a partial mechanism". **Nothing shipped** — the daily path is untouched.

## What was being settled

Six causes had been examined and the band's marginal over-coverage — **87.2 / 91.8 / 90.6 / 89.0%**
against 80% — belonged to none of them. The confirmed `sigma` tilt opened one lead, explicitly
labelled as a lead: coverage is a monotone increasing function of `sigma`, `q_hat` makes that curve
integrate to 80% over the **calibration** `sigma` distribution, and served `sigma` was on record at
**1.28 / 1.34 / 1.33 / 0.93×** the calibration median. If serving sees a different `sigma`
distribution, the same curve integrates to something else — **no second defect required**.

That entry declined to size it and named the reason it might not survive: at 30d the ratio is
`0.93 < 1`, which predicts *under*-coverage where production over-covers.

## The mechanism, as arithmetic

`log|resid| = a + β·log sigma + ε` fitted on the panel, so a row is covered iff
`ε ≤ log q_hat − a + (1−β)·log sigma`. The noise CDF is the **empirical** one, not a normal — these
residuals are heavy-tailed and the whole question sits at the 80th percentile. With `β < 1` the
threshold **rises** with `sigma`, which is the tilt; predicted marginal coverage under a served
distribution is then `mean c(sigma_served)`.

| leg | result | bar |
|---|---|---|
| **V** validity — does `c` reproduce the deciles it is fitted on? | MAE **0.89 / 1.03 / 1.43 / 1.23pp** | ≤2pp at ≥3/4 → **PASS 4/4** |
| **P** placebo — `β` forced to 1.0, the exponent conformal assumes | **0.0000pp** at every `k`, 4/4 | ≤0.05pp → **PASS** |
| **F** footprint — does the `sigma` mix predict realised per-date coverage? | corr **+0.495 / +0.505 / +0.551 / +0.614** | >0 at ≥3/4 → **PASS 4/4** |

**The placebo is the load-bearing one.** Forcing `β = 1` flattens `c` and a `sigma`-mix shift then
moves marginal coverage by **exactly zero** at every `k` from 0.5 to 5.0. So this channel exists
*only* because the elasticity is not 1, and the instrument has no other path to marginal coverage.

## The size

`A = (predicted marginal − 80) / (observed marginal − 80)`.

| h | excess to explain | published `k` | `A` | **`k` measured directly** | predicted | **`A`** | same date, calibration items only |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | +7.2pp | 1.28 | +0.57 | **1.29×** | 84.89% | **+0.68** | 1.23× → 84.33% (+0.60) |
| 7 | +11.8pp | 1.34 | +0.40 | **1.29×** | 84.77% | **+0.40** | 1.23× → 84.21% (+0.36) |
| 14 | +10.6pp | 1.33 | +0.42 | **1.29×** | 84.66% | **+0.44** | 1.23× → 84.08% (+0.39) |
| 30 | +9.0pp | **0.93** | **−0.22** | **1.28×** | 84.57% | **+0.51** | 1.23× → 83.96% (+0.44) |

So the `sigma`-mix shift buys **+4.0 to +4.9pp** of marginal coverage, which is **36–68%** of the
excess depending on horizon and on whether the served cohort's composition is held fixed. The
verdict on the pre-registered scale is **PARTIAL** on both the published and the directly measured
shift: material at 2 of 4, partial at 2 of 4.

## The 30d counter-example does not survive direct measurement

**`sigma` carries no horizon term.** It is `price_std_60d / price` at the anchor, so one anchor date
has **one** `sigma` distribution for all four horizons. The four published ratios could therefore
only differ because the served row *sets* differ per horizon — and production has **7 / 6 / 3 / 1**
forecast dates at 3/7/14/30d, so `0.93×` rested on a **single date's** mix, reconstructed as
`half_pct / q_hat` from the same implied-`sigma` route whose elasticity was refuted at 4 of 4 the
same day.

Measured directly — `conformal.sigma_from_columns` on a voted panel spanning production's actual
anchors, one clip applied to both sides — the served ratio is **1.28–1.29× at all four horizons**,
and `1.23×` once the calibration panel's own item set is imposed. **The sign objection is
withdrawn**, and 30d moves from `A = −0.22` to `A = +0.51`.

### The control that makes the direct leg readable

The served side comes from a **different voted cache** (283 dates to 2026-08-08) than the
calibration side (731 dates to 2026-07-09), so an elevated served `sigma` could be a property of the
file. On the **257 shared dates and 913 shared items** the two caches agree at a median ratio of
**1.000**, and every disagreement is confined to **2025-10-26 → 2025-12-22** — the shorter cache's
opening, where the rolling std is over 60 **rows** and the window has not filled. The ratio ramps
`0.158 → 1.000` across exactly that stretch and is `1.000` for every shared date after it. The
served read is 283 rows deep and uncontaminated.

## What this says about the date axis, which was thought empty

`2026-08-12-the-band-is-tilted-in-sigma.md` refuted a **time-varying** `q_hat`: level-matched, no
date-conditional scheme beat pooled, with placebos at ≤0.11pp. That still stands and this does not
contradict it — but the `sigma` mix **is** observable at serve time and it does carry date-level
information about the *level*:

| h | realised per-date coverage | predicted from the `sigma` mix alone | R² |
|---|---|---|---:|
| 3 | 14.4–98.2% (sd 11.4pp) | 52.3–91.2% (sd 5.3pp) | 0.245 |
| 7 | 18.4–97.5% (sd 12.2pp) | 52.5–91.0% (sd 5.3pp) | 0.255 |
| 14 | 19.8–97.7% (sd 13.4pp) | 51.0–91.1% (sd 5.5pp) | 0.304 |
| 30 | 12.7–96.2% (sd 12.5pp) | 47.9–91.4% (sd 6.1pp) | 0.377 |

The distinction: those schemes tried to **forecast** a date's dispersion from date-level state and
failed, the same wall N2's own-history leg hit. This reads the date's **own realised `sigma` mix**,
which serving already has in hand. A quarter to well over a third of the per-date coverage variance
is the `sigma` mix — and, per the placebo, all of it flows through `β ≠ 1`.

## What can and cannot produce the whole excess

`k*` is the shift that would explain **all** of it, against the natural range of per-date
`sigma`-median ratios over the panel's own 670–724 dates:

| h | `k*` for the full excess | rows at the cap under `k*` | natural per-date ratio (p05 / p50 / p95 / **max**) |
|---|---:|---:|---|
| 3 | **1.60×** | 2.4% | 0.68 / 0.93 / 2.00 / **2.16** |
| 7 | **2.45×** | 5.7% | 0.68 / 0.93 / 1.98 / **2.15** |
| 14 | **2.16×** | 4.6% | 0.68 / 0.93 / 2.00 / **2.16** |
| 30 | **1.85×** | 3.5% | 0.67 / 0.93 / 1.99 / **2.16** |

At **7d, `k* = 2.45` exceeds the ratio of every single date in two years**, so the remainder there
cannot come from the `sigma` mix at any market state. At 3d and 30d `k*` sits inside the range, and
at 14d exactly at its maximum — reachable in principle, but nowhere near the `1.23–1.29×` actually
observed.

## The seventh cause is now named and bounded, not unknown

Marginal coverage is an identity:
`∫ P(|resid| ≤ q_hat·sigma | sigma) dF_served(sigma)`. This run varied `F_served` while holding the
conditional residual law at its calibration value. **The unexplained 32–64% is therefore, by
identity, a shift in that conditional law** — served residuals being smaller at a given `sigma` than
history's — and not a seventh unrelated defect. That term is measurable on resolved served outcomes,
which is a defined next read rather than another hypothesis.

## What this changes about shipping `β`

The exponent was justified on **conditional** coverage, and held out it reached only 3d/7d
(−76% / −62% against −26% / −12%). This adds a second, independent argument: normalising by
`sigma ** β` makes the score's distribution independent of `sigma`, so marginal coverage becomes
invariant to the served `sigma` mix — **the placebo column is that statement, computed.** That
closes **+4.0 to +4.9pp** of the marginal excess *and* ~25–38% of the per-date variance, at all four
horizons, including the two where the conditional fix was weak.

⚠️ **Stated as an inference, not a measurement.** It requires the fitted `β` to transfer to serving,
and the held-out read is precisely where a single global `β` was shown to drift at 14d/30d. The
marginal channel needs only average flatness, which is a weaker requirement than the conditional
one — but "weaker requirement" is an argument, and the confirm belongs in the same paired dispatch
that ships the exponent.

## What is NOT claimed

- **Not a ship and not a production figure.** One panel, `2024-07-09 → 2026-07-09` for calibration
  and to `2026-08-08` for serving, on the local voted cache, which runs behind the durable archive.
- **`r̂ = 0`.** The panel's `q_hat` are 1.020 / 0.958 / 0.887 / 0.785× the shipped ones, so absolute
  levels are not the artifact's. `A` is a ratio of two quantities computed on the same curve, which
  is why it is the reported statistic.
- **Not a constant bias.** Monthly median `sigma` on the served cache runs **0.056 → 0.135**
  (2026-03 to 2026-04) against a calibration median of **0.0717** — roughly **0.78× to 1.88×**. So
  this channel's contribution *drifts with market volatility and changes sign*; a single number for
  "the over-coverage" is a snapshot of one month.
- **The 87.2/91.8/90.6/89.0% denominators are taken from record**, not re-measured here.

## Reproducing

```
cd backend && venv/bin/python -m scripts.attribute_marginal_coverage --horizons 3,7,14,30
```

Read-only: reads two voted panels and writes a CSV. Seconds, no dispatch. It selects the
**longest-span** cache for calibration and the **newest-last-date** cache for serving — deliberately
opposite rules, because depth and recency are different requirements and in this repo they are
different files. Both provenance lines are logged; read ratios, not levels.
