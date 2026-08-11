# The serving transforms do not explain the CV→serving gap, 2026-08-11

`REPLAY_DISABLE` (`6784123`) turns off the transforms that run inside `predict` and nowhere else.
Sweep: 5 configurations × 2 anchors (2026-06-01, 2026-07-05), one artifact (local, trained
2026-08-09), all four horizons. Because every configuration serves the **same** artifact, the
anchor-basis question cancels and the deltas are attributable.

## The null control passed

`REPLAY_DISABLE=bias` reproduces the control **byte-for-byte** at both anchors, all four horizons,
every column. That is the intended result: `bias_corrections.json` holds `corrections: {}` *and*
`thresholds: {}`, so the tier bias is inert. A transform documented as one of four unscored
serving steps is in fact applying nothing, and the sweep's zero row is the evidence the
attribution machinery works rather than a wasted run.

## Rank IC is barely moved by any of them

Δ rank IC against the full serving path:

| h | blend off (06-01 / 07-05) | recenter off | all three off |
|---|---|---|---|
| 3 | +0.0091 / +0.0006 | −0.0052 / −0.0732 | −0.0030 / −0.0655 |
| 7 | +0.0053 / −0.0038 | +0.0729 / +0.0400 | +0.0713 / −0.0020 |
| 14 | −0.0088 / +0.0002 | +0.0284 / −0.0007 | +0.0160 / −0.0115 |
| 30 | +0.0084 / −0.0095 | −0.0247 / −0.0020 | −0.0329 / −0.0181 |

Means across the 8 cells: **blend +0.0002, recenter +0.0044, all three off −0.0057.**

**With every serving transform disabled, served rank IC is still −0.25 to +0.14** — nowhere near
the 0.09–0.18 CV reports for this model family. The mid is then nothing but the q50 boosters'
output converted to dollars, so the arithmetic between the boosters and the served number is not
where the signal goes.

The prior-day blend is negligible outright: |Δ| ≤ 0.0095 in all 8 cells.

## What `_recenter_on_direction` actually does

It has no cross-sectional information (mean Δ rank IC +0.0044) and an enormous effect on
directional accuracy:

| anchor | realised down% (3/7/14/30) | Δ DA from disabling recenter |
|---|---|---|
| 2026-06-01 | 34.8 / 25.2 / 22.3 / 47.6 — market **rose** | **+21.29 / +17.57 / +9.63 / +0.30** |
| 2026-07-05 | 37.8 / 58.3 / 81.3 / 83.0 — market **fell** | +15.19 / +0.39 / **−13.18 / −10.91** |

The pattern is a systematic **down bias**: recentring helps DA when the market falls and hurts it
when the market rises, by up to 21pp either way, while leaving the ranking untouched. That is the
signature of a directional bet, not of information. It is not a free bet either — at 2026-06-01,
where the market rose, it cost 9.6–21.3pp of DA at three of four horizons.

This sharpens `serving-transform-down-bias`, which found the "+13.4pp skew" to be a market-period
artifact: the transform's *effect* is real and large, its *sign* is whatever the market did, and
its contribution to ranking is nil.

## So where does the signal go?

Not the serving arithmetic — that is now measured rather than assumed. The remaining candidates,
none yet tested:

1. **The evaluation basis.** The replay divides by `predict`'s span-bounded median while CV scores
   fold predictions against a differently-resolved label. `2026-08-10-serving-replay.md` sized this
   at up to 0.6 rank IC at h=3 for the naive baseline alone.
2. **Date sampling.** Two anchors against hundreds of fold-dates, and DA is known to be dominated
   by the market date (`da-is-dominated-by-the-market-date`). The two anchors here have opposite
   market signs, which is a start but not a sample.
3. **Cohort and feature-frame differences** between the CV folds and the predict frame —
   `PREDICT_FETCH_DAYS = 730` against training's 1460 already moves the cohort 916 → 948.

## Cost

Ten local replays, roughly 50 minutes, no training. The knob is refused without `REPLAY_ANCHOR`,
so none of this can reach the served path.
