# The exponent is paired-read: it fixes the tilt, and shipping it alone costs served coverage

**Date:** 2026-08-12
**Runs:** `31629626929` (control) / `31629638834` (arm, `sigma_exponent=true`), both on commit
`5d1bc04`, both `horizons=matrix`, both `replay_anchors=2026-04-15,2026-05-16,2026-06-16,2026-07-09`
**Reads:** `2026-08-12-sigma-exponent-implemented.md` §"What is NOT done"
**Status:** ✅ β reproduces and the implementation is coherent. ❌ the pre-registered width bar
**fails on the basis it was quoted against**. ⚠️ served marginal coverage falls ~8pp at 3d/7d/14d,
which is a regression at 3d. **`SIGMA_EXPONENT` stays off.**

## The read had to be instrumented first

Two arms were dispatched on `ece4922` and **cancelled at ~8 minutes**: no dispatch path printed a
half-width anywhere. `q_hat` may not be differenced across this flag — the arms are ~5.5× apart in
units — so coverage was the only comparable number either run would have produced, and the primary
bar (median half-width 0.87 / 0.86 / 0.84 / 0.77× vs control) was unreadable. `5d1bc04` adds the
width on both bases, report-only: `beta=` and `median half-width=X% of mid` on the calibration line,
and a `halfw%` column on `replay_serving.py`'s `BAND COVERAGE` table.

**That gap is the reason this entry exists as a separate read** rather than a line in the
implementation changelog. A bar that names a number without naming its basis cannot be failed, and
this one turns out to resolve differently on the two available bases.

## β reproduces, and the implementation is coherent

| h | β fitted | q_hat control | q_hat arm | prior OOF read (`31619383780`) |
|---|---:|---:|---:|---:|
| 3 | **0.4241** | 94.72 | 19.56 | 0.429 |
| 7 | **0.3672** | 141.77 | 24.66 | 0.369 |
| 14 | **0.3370** | 204.34 | 32.43 | 0.350 |
| 30 | **0.3266** | 312.05 | 49.13 | 0.313 |

Within 0.014–0.024 of the previous read at every horizon. Four checks on the wiring, all of which
pass:

- `meta.json` carries the matched (β, `q_hat`) pair for the horizon its job trained and `1.0` beside
  the restored `q_hat` for the other three. The pair is never split.
- The control's `q_hat` is byte-identical to the published 94.72 / 141.77 / 204.34 / 312.05, so the
  new log lines moved no calibration.
- The `Sigma-tilt audit` lines are **identical between the arms** at all four horizons — it stayed
  report-only, as designed.
- No pooled clamp fired.
- Backing σ out of the calibration widths through `(q_b/q_1)·σ^(β−1)` gives **0.0687** at 3d and
  **0.0678** at 30d — the documented ~0.07, recovered independently at both ends. This is what
  establishes that β reached the `calibrate` *and* the `band` call rather than one of them.

## The width bar fails on its own basis and over-delivers on the served one

| h | calibration set | **served (4 anchors)** | pre-registered |
|---|---:|---:|---:|
| 3 | 0.966 | **0.794** | 0.87 |
| 7 | 0.957 | **0.768** | 0.86 |
| 14 | 0.947 | **0.750** | 0.84 |
| 30 | 0.965 | **0.761** | 0.77 |

The 0.87 / 0.86 / 0.84 / 0.77× prediction sits *between* the two bases, and misses the calibration
set — the basis it was quoted on, since it came from a panel of calibration-style residuals — by
9–19pp of ratio.

**This is mechanical, not a defect.** The per-item ratio is `(q_b/q_1)·σ^(β−1)`, and with `β−1 < 0`
it decreases in σ. The median calibration row barely moves because both arms are level-matched to
80% on those same rows; the served cohort's σ runs **above** the calibration median, so the served
band narrows much further. That is the 1.28–1.29× served-σ shift of
`2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md`, now visible from the width side.

⚠️ **Consequence for future pre-registrations:** a width or coverage bar on this flag must name
**calibration set** or **served** before the dispatch. The two differ by ~0.2 of ratio here and they
support opposite verdicts on the same 0.87 number.

## The cost: served marginal coverage falls

Paired per anchor, four anchors, t on 3 df (critical 3.18):

| h | control | arm | Δ | t | \|dev from 80\| |
|---|---:|---:|---:|---:|---|
| 3 | 77.80 | 69.72 | **−8.07pp** | −3.91 ✓ | 2.20 → **10.28** |
| 7 | 84.94 | 76.68 | **−8.26pp** | −5.81 ✓ | 4.94 → 3.32 |
| 14 | 84.76 | 76.88 | **−7.88pp** | −3.80 ✓ | 4.76 → 3.12 |
| 30 | 87.07 | 84.89 | −2.19pp | −1.33 ✗ | 7.07 → 4.89 |

Misses grow on **both** sides in proportion, so this is the width and not the centre. At 7d/14d/30d
the arm moves the level toward target; **at 3d it takes an almost-exactly-calibrated 77.80% and
overshoots to 69.72%**, which is a regression and is the single reason this is not shippable as one
global flag.

Two cautions on reading the level at all. Per-anchor coverage swings **65.70–85.90%** at 3d in the
control alone, so four clustered anchors size it weakly — only the *paired* delta is powered. And
the control does **not** reproduce the 87.2 / 91.8 / 90.6 / 89.0% prod over-coverage except at 30d,
so the premise that the band over-covers is period-dependent and is not in evidence here at 3d.

## Two things that contradict the record

**1. `fold_beta` hits the clamp, on real data.** Exactly `0.2000` — the lower bound of `[0.2, 1.0]` —
on folds 2–3 at 7d, folds 2–4 at 14d, and folds 3–4 at 30d. The implementation treats a binding
clamp as "a data problem worth surfacing" on the stated grounds that no window of this archive has
produced one. That is now false for the early folds. It does not touch the served exponent, which is
fitted pooled and unclamped.

**2. β drifts ~2.5× between folds, with a break.** Folds 1–4 run **0.20–0.31** and folds 5–9 run
**0.33–0.61**, synchronised across horizons. The pooled fit (0.42 / 0.37 / 0.34 / 0.33) therefore
sits *below* the recent-period value that production's 14-day refit cadence would land on. This is
the drift measurement the 14d/30d dispute was said to turn on, and it is large.

**3. The held-out leg replicates 3d/7d-only, for the second time.** Conditional error
**−78% / −67% / −22% / −35%** at 3/7/14/30d, against `31619383780`'s −76% / −62% / −26% / −12%. Two
independent reads on real OOF residuals now stand against the walk-forward **−84% / −73%** at
14d/30d of `2026-08-12-the-sigma-scale-is-one-exponent-per-horizon.md`. The "unsettled until the
paired dispatch" label on that dispute resolves toward the held-out side: **the exponent's reach at
14d/30d is weaker than the walk-forward panel claimed.**

## Verdict

`SIGMA_EXPONENT` stays **off**. The exponent does what it claims on the axis it targets — pooled
level-matched decile error falls 7.87 → 0.81pp at 3d and 9.50 → 2.54pp at 30d — but it is a *tilt*
correction being asked to carry a *level* it was never calibrated for, and at 3d that trade is
clearly negative.

What would make it shippable is β **plus a re-level against the served σ mix**, which is the marginal
problem already on record as open in both directions. Do not ship one without the other, and do not
read this entry as a refutation of the tilt: the tilt is confirmed for the third time.

## Not measured here

- **Served coverage by σ decile.** The replay reports pooled coverage only, so the property the
  exponent actually targets is unverified *on the serving path* — every by-decile number in this
  entry is OOF. That is the natural next instrument.
- **Whether the coverage drop is concentrated in the low-σ deciles**, which is what the OOF profile
  predicts. Same blocker.
