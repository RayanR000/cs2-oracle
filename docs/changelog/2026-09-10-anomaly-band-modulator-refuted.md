# Anomaly_p band modulator refuted: wider everywhere, never more equal

2026-09-10. New `backend/scripts/anomaly_band_modulator_ab.py` (+
`tests/test_anomaly_band_modulator_ab.py`, 18 tests). No production change —
the harness wires nothing.

## The idea

The anomaly GBM is the only head here that beat a featureless null
(`2026-09-09-anomaly-head-beats-its-null.md`: held-out AUC +0.13–0.14). Items
with high `anomaly_p` are about to make outsized moves, so multiply the served
climatology scale by `f(anomaly_p)`: per-decile q80 ratio on the fit split,
isotonised to monotone non-decreasing, normalised to mean 1.0 (level-preserving,
as with vol-rank), clipped to [0.25, 4.0]. A modifier of the climatology scale,
not a sixth denominator. Scored at matched 80% per arm per fold — width
(`log_width`) plus level-matched per-anomaly-decile coverage error
(`decile_err`) — with a shuffled-p placebo arm of equal capacity.

## Measured 2026-09-10: refuted at every horizon (102 paired folds)

Paired per-fold deltas, arm − control (negative = good on both metrics):

| h | width delta [95%] | narrower | decile_err delta [95%] | more equal |
|---|---|---|---|---|
| 3d | +0.0100 [+0.0060, +0.0139]\* (+1.0%) | 4/26 | −0.40 [−0.93, +0.13]pp | 18/26 |
| 7d | +0.0106 [+0.0040, +0.0172]\* (+1.1%) | 5/26 | +0.29 [−0.32, +0.89]pp | 12/26 |
| 14d | +0.0059 [+0.0019, +0.0099]\* (+0.6%) | 6/25 | **+0.84** [+0.21, +1.47]\*pp | 7/25 |
| 30d | +0.0170 [+0.0081, +0.0260]\* (+1.7%) | 6/25 | **+2.12** [+1.23, +3.01]\*pp | 4/25 |

The shuffled placebo is null on both metrics at every horizon (width ratios
0.9996–1.0000, decile ratios ~1.00), so the harness is honest and the failure
is the shape, not the metric. In-fold head AUC (~0.63 at 7d) confirms the head
itself still ranks — its predicted-p deciles just do not transfer into a
usable width shape. Likely mechanism: `f` is fitted on in-sample predicted p
(overconfident, too spread) and applied to out-of-sample p, so the bins
misalign; and per-decile q80 ratios are fit noise the isotonic step preserves
rather than removes.

## What this means

A head that ranks anomalies is not a band-width signal. Do not wire this
without a new mechanism — e.g. fitting `f` on out-of-fold predicted p so the
fit distribution matches serving, or conditioning on calibrated rather than
raw p. Either re-opens this as a new arm with its own paired read; a re-run
of this arm does not.
