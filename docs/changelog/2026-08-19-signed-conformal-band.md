# The served band is two signed conformal quantiles

**2026-08-19.** The conformal band is now built from a signed `(q_lo, q_hi)` pair instead of one
absolute `q_hat`, and `_recenter_on_direction` is removed from the serving path. This is the RANGE
stance: the band's asymmetry comes from the q50 residual, not from a directional classifier moving
the mid after calibration.

## Why

`conformal.calibrate` folded the residual with `np.abs`, so the served band was symmetric by
construction. On a residual whose median is not zero — which is what `DIRECTION_UPWEIGHT = 1.5`
produces, fitting the q50 near the 58–60th percentile (`P(actual < mid)` = 0.60/0.65/0.71) — the
absolute quantile is inflated by the fat tail and the band is wider than it needs to be. Two signed
quantiles (`quantile(res/scale, α/2)`, `quantile(res/scale, 1−α/2)`) centre on the residual's own
median and are narrower at the same nominal coverage.

The earlier belief that the classifier's `_recenter_on_direction` was the biased centre is refuted
by the arithmetic (2026-08-11): toggling it moves coverage ≤1.1pp because the shift is bounded by
2·|mid| (median 0.95%) against a 10–31% half-width. The bias is in the **q50**, and the signed pair
absorbs it post-hoc. Removing the classifier from serving is a separate, coverage-negligible change
made for stance coherence — the two must land together, or the classifier drags the asymmetric mid
after the band is built.

## What changed

- `conformal.calibrate_signed` / `conformal.band_signed` (new). `band_signed(mid, σ, −q_hat, q_hat)`
  is byte-identical to `band(mid, σ, q_hat)`, which is the fallback for any artifact without the pair.
- `_calibrate_conformal` fits and persists `(conformal_q_lo, conformal_q_hi)` from the SAME
  residuals/beta/scale as `q_hat` — a matched set, same rule as `conformal_beta`.
- `predict` serves `band_signed` via a new `band_offsets(horizon)` accessor (symmetric fallback when
  the pair is absent or non-finite); `_recenter_on_direction` removed from `predict`. The classifier
  still populates the `direction`/`confidence` fields; it no longer moves the price.
- `meta.json` carries `conformal_q_lo`/`conformal_q_hi` (loaded with `.get`, no `MODEL_ARTIFACT_VERSION`
  bump — same forward-compat pattern as `conformal_beta`).
- The `recenter` REPLAY_DISABLE token was retired (`REPLAY_DISABLABLE = {"blend", "bias"}`); a knob
  that guards nothing would read as a clean attribution control.
- The `:7131` `_recenter_on_direction` call stays — it is in `_conformal_records`, diagnostic-only
  (fires under `CV_DIAGNOSTIC_CLASSIFIER`, off by default), not a serving path.

## Evidence

Paired local replay (signed artifact vs the same artifact with the pair stripped), mean over three
feed-clean anchors (2026-05-01, 05-15, 06-01):

| h | cov signed / control | width Δ | closer to 80%? |
|---|---|---|---|
| 3 | 86.3 / 87.1 | −1.7% | ✓ |
| 7 | 86.2 / 88.0 | −1.9% | ✓ |
| 14 | 82.4 / 83.7 | −0.9% | ✓ |
| 30 | 81.9 / 84.6 | −5.7% | ✓ |

Narrower **and** closer to nominal at all four horizons. ⚠️ **Magnitude is not the 11–13% quoted in
the review** (here 1–6% width, 1–3pp coverage): three clustered May–June anchors on the stale LOCAL
archive, and per-anchor variance is real — 2026-06-01 alone under-shoots at h=7/14 (76%) because the
OOF-fitted skew pointed against that anchor's up-move, the same OOF→served non-transfer wall this
repo keeps hitting. Both arms still over-cover (82–88%); the signed pair narrows the marginal gap,
it does not close it. Directional, not publishable — the definitive read is a prod A/B.

## Follow-up

The durable fix is `DIRECTION_UPWEIGHT → 1.0` (unbias the q50 itself), so the band carries no skew
that has to transfer across regimes. See
`docs/specs/2026-08-19-signed-conformal-quantile-design.md`.
