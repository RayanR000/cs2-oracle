# 2026-09-09 — exceedance head gets an isotonic calibration layer

The `move_odds` signal (`exceed_p`, served at h=3/h=7) was a raw LightGBM binary
output with no probability-calibration layer: replay reliability read ECE <1.3pp
at h3/h7 but ~3.8pp at h30, and `forecaster.py` tracked no Brier and no
reliability curve — only the band got conformal treatment.

## What changed (`models/forecaster.py`, `api/schemas.py`)

- Per-horizon isotonic calibrator (Pool Adjacent Violators, pure numpy — sklearn
  is not a declared dependency), fitted on the CV path's OUT-OF-FOLD (p, y)
  pairs, the same cross-fit honesty the learned band scale needs. Persisted in
  `meta.json` as `exceedance_calibration` (map + Brier/ECE before/after report);
  served through `exceedance_probability()` on the disclosed path only.
- The BAND keeps the raw p: its q_hat is dimensionally tied to it (matched
  pair), so `band_scale()` explicitly asks for `calibrated=False`. No band
  width changes; no artifact version bump (absent key = serve raw, which is
  what old artifacts always served).
- Brier + fixed-width reliability table + ECE are now logged per horizon at
  train time and stored in `cv_results[h]["exceedance_calibration"]`, so the
  served number is quotable beside its calibration.
- Kill-switch: `EXCEEDANCE_CALIBRATE=0` fits nothing and serves raw (default on).
- The CV loop previously computed OOF `exceed_p` only under `EXCEEDANCE_SCALE`;
  production runs `EXCEEDANCE_HEAD=1` on a climatology band, so no OOF pairs
  existed at all. The gate is now either flag, and `_conformal_records` carries
  the `exceed_y` label both paths.

## Tests

`backend/tests/test_exceedance_calibration.py` (14 tests): PAV monotonicity /
bounds, in-sample Brier never loses to raw (identity is monotone), helper
arithmetic, disclosed-calibrated vs band-raw divergence, meta round-trip,
old-artifact-serves-raw, OOF-gate source guard, min-rows refusal, flag-off.

## To quote after the next full retrain

Read `cv_results[h]["exceedance_calibration"]` (`brier_raw`/`brier_cal`,
`ece_raw_pp`/`ece_cal_pp`, both reliability curves) from the training log —
that is the honest OOF read. Confirm on served outcomes once they mature; the
replay table in `scripts/replay_serving.py` is unchanged and still the
pre-flip check.
