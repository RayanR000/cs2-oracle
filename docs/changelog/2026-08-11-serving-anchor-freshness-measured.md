# Quoting from the fresher price: a serving win, not a forecasting one

Control run `31512455612` and arm run `31514765343`, both on `e2c2c80`, same four
non-overlapping anchors (2026-04-15, 05-16, 06-16, 07-09), 4 horizons each.
`SERVE_OUTLIER_GATED_ANCHOR` confirmed `0` and `1` in each run's own replay log — production
smooths every item; the arm smooths only the items that deviate >10% from their local median.

**Verdict: arm A passes its gate on dollar error, 14 of 16 cells above the placebo. It is a
quoting correction, and it buys no forecasting skill.** The published dollar figure lands
closer to what happens; the model's ability to rank items is untouched.

## The gate

Median absolute dollar error, `|mid − realised| / realised`, on the **deviating** cohort. The
tied cohort is inert here by construction — both arms serve an identical price there — and it
came back **exactly 0.00 different in all 16 cells**, which is the negative control working.

| h | anchors improved | best cell | worst cell |
|---|---|---|---|
| 3 | 4 / 4 | −2.31pp (05-16) | −0.03pp (07-09) |
| 7 | 3 / 4 | −1.00pp (05-16) | +0.09pp (07-09) |
| 14 | 4 / 4 | −0.79pp (05-16) | −0.11pp (07-09) |
| 30 | 4 / 4 | −0.37pp (06-16) | −0.03pp |

Gate was improvement at 3 of 4 anchors and 3 of 4 horizons. Cleared.

**The placebo matters, because most cells are small.** `MIN_SERVED_PRICE_USD` applies to
`current_price`, which the arm moves, so the two arms score slightly different item sets
(n 799→794, 463→457, 626→629, 1006→1005). The `naive` column is arm-invariant *per item* —
it is converted to dollars against the pinned anchor — so any movement in it is pure cohort
membership. That placebo runs 0.00–0.19pp. Reading each cell against its own placebo:
**14 of 16 improve by more than it**; 3d/07-09 improves *within* it (−0.03 vs 0.05) and
7d/07-09 is the one cell that worsens (+0.09).

## What it is not: the pinnedIC gain is the same arithmetic that killed the label arm

`pinnedIC` rose at **14 of 16 cells**, and at h=3 it flips sign at 4 of 4 anchors
(−0.094→+0.089, −0.138→+0.147, −0.046→+0.352, −0.049→−0.041). **Do not publish that as a
rank IC gain.**

The pin fixes the *denominator* — `D_fixed` is the shipped smoothed anchor under both arms —
but this arm moves the *numerator's level*:

```
control mid / D − 1 = r̂                    (D = S[d], the smoothed anchor)
arm     mid / D − 1 = (p[d]/S[d])(1 + r̂) − 1
```

On the deviating cohort `|p/S − 1| > 10%` by construction while `r̂` is a few percent, so the
arm's predicted return is dominated by `p/S` — and the outcome leg, `realised/S[d] − 1`,
carries `p/S` as a factor too, because a persistent price series reaches `d+h` from `p[d]`.
The correlation is largely `p/S` against itself. **This is the same free-factor mechanism as
`2026-08-11-smoothed-anchor-label-measured.md`, moved from the label to the prediction.**

**Corrects a claim from task 1 of the plan.** Pinning the denominator was described there as
making rank IC comparable across arms. It does not: it protects one side of the ratio. An arm
that changes the level of the served mid still moves both legs together. Dollar error, whose
denominator is the realised price, is the only metric here that no arm touches.

## The corroborating read, and the uncomfortable one

The decisive test of "skill or level?" is the model's **edge over its own no-change quote**
(`model − quote`; negative means the forecast beats republishing the served price). If the arm
improved discrimination, the edge would improve. It does not: **10 of 16 cells**, deltas
±0.35pp, indistinguishable from a coin flip.

And the control's edge is **positive in 12 of 16 cells** — on the deviating cohort, at three
of four anchors, *republishing the served price would have been closer in dollars than the
forecast was*. That sits beside the standing result that a one-line `−return_1d` baseline
beats the model on rank IC at all four horizons
(`no-idiosyncratic-signal`, `model-loses-to-minus-return-1d`). Neither arm of this experiment
touches it.

## What this says about the deviation

The gain concentrates at **h=3** (−2.31pp, −2.29pp at two anchors) and decays with horizon,
and the `quote` column moves with the `model` column almost one for one (−1.96, −2.47 at those
same cells). That is the signature of the smoothing **lagging a real level move**, not of it
suppressing a transient spike: where the median disagrees with the latest quote by >10%, the
latest quote is usually right about where the price now is, and the median is a stale average
of the way up. Hazard 2 of the plan — "a fresher quote is not always a better one" — is the
opposite failure, a frozen re-publication, and it cannot reach this cohort: a repeated quote
equals its own median and lands in the **tied** third, where both arms are identical.

## What was not measured

- **Coverage of the conformal band under the arm.** `q_hat` is fitted in return space, so a
  different base rescales every served dollar band. F1 (the band is calibrated around a mid it
  is not served around) is prior to this and still open — read nothing about the band here.
- **Whether the backtest's resolver should move with it.** `backtest/price_resolution.py`
  resolves `base_price` on the same `SMOOTH_WINDOW`, and serving and the backtest already
  diverge by a median 3.6% / p90 35% when they disagree. Arm A changes serving only, so it
  **widens** that divergence by construction. That is acceptable for a measurement and is the
  open question before the default moves.
- **Any interval.** Four anchors is a replication, not power. The sign consistency and the
  per-cell placebo are the whole read.

Artifacts: `diagnostics-{3,7,14,30}d` on both runs, 90-day retention. The arm remains **off by
default**; nothing about production's served price changed.
