# Pre-registration: exceedance-head reliability + isotonic on held-out dates only

**Date:** 2026-09-13, written and committed **before any probe number is computed.**
**Status:** PROPOSED — runnable now.
**Follows:** `changelog/2026-09-09-exceedance-isotonic-calibration.md` (the shipped
layer), `changelog/2026-09-13-mondrian-oof-confirm-closed.md` (the 3.4–3.6x
real-OOF q_hat spread + the low-sigma-bin level failure), and
`changelog/2026-09-10-anomaly-band-modulator-refuted.md` (the in-sample-p failure
mode this probe is designed not to repeat).

## Why this leg exists, and what it is not

The exceedance head is the one live magnitude signal on this panel, and unlike
the band its open question is **calibration**, not width: the head already ranks
(trained-cohort mean AUC 0.774 at h=7 on price technicals alone;
`2026-09-09-exceedance-meta-refuted.md`), and production already ships an
isotonic layer fitted on OOF (p, y) pairs
(`forecaster.py::_fit_exceedance_calibrator`). That layer has never been measured
against the two bars that matter: does it state better probabilities than the
raw head on rows neither saw, and does it beat the featureless constant that is
the shipping floor for any served probability.

This is **not** a band-width experiment and **not** a conditioning experiment.
No sigma, no climatology modifier, no Mondrian bin. Horizons are never pooled.
A null here changes nothing about the served `exceed_p` (it stays, raw or
calibrated per the result); a positive changes only the calibration path, never
the band.

## The two fixes applied in advance

1. **Pinned fold grid across arms.** Real-OOF per-fold q_hat spread is 3.4–3.6x
   (vs ±26% model-free), so any width-adjacent A/B smaller than that is
   unreadable without the grid pinned. This probe scores probabilities, not
   widths, but the same regime-shift moves log loss between folds far more than
   between arms — so all five arms share one fold enumeration, one
   `_stratified_sample` per fold (seed depends on the fold, never the arm), and
   identical val rows. Paired deltas on fold index only.
2. **Isotonic scored on held-out dates only.** The anomaly modulator fitted `f`
   on in-sample predicted p (overconfident, too spread) and applied it to
   out-of-sample p, so the bins misaligned and the arm widened everywhere. Here
   the map is fitted on a calibration slice that is date-disjoint from the
   inner train slice with the H+13 embargo at the inner boundary
   (`_purge_overlapping_train_rows`, same rule as the outer split), applied to
   the val window, and **scored only there**. Scoring the map on its own fit
   rows voids the run, not just the fold.

## Served panel (read-only diagnostic, no fit)

Prod Postgres, 2026-09-13: served `exceed_p` exists on **6 / 5 / 6 clean dates**
at h=3/7/14 (≈5.5k/4.6k/5.5k rows) and **0 dates** at h=30 — all well below
`MIN_FORECAST_DATES=20`. The probe therefore **reports** the served raw
reliability table + ECE + Brier at h=3/7/14 (same bins, same one-sided label,
same tier threshold as `replay_serving.py::_reliability_rows`) but **fits no
map on served rows**: 5–6 dates cannot support an honest fit, and anything
fitted there would be scored in-sample by construction. The verdict comes from
the offline OOF leg below; the served read is the baseline the next retrain's
`cv_results[h]["exceedance_calibration"]` will be checked against once the
panel matures.

## Offline OOF leg (the verdict)

Instrument: `backend/scripts/exceedance_calibration_ab.py` (new), modelled on
`anomaly_calibration_ab.py` with the exceedance production head
(`_fit_exceedance_classifier`), the exceedance one-sided label
(`target_exceed_{h}d`, unconditional — no flag needed), and the served clip
`P_CLIP=(1e-3, 1.0)` matching `exceedance_probability`.

Fold machinery, item split (`assign_items`: 150 held-out / 500 train / 150
trained-eval), row budget (`ROW_BUDGET`), purge (h+13), and the paired
t-interval are imported wholesale from the harness family, never re-derived.

Arms (all on the identical pinned grid):

| arm | fit | scored p |
|---|---|---|
| `gbm` | full train fold | raw head |
| `gbm_inner` | inner train slice only (calib slice held out) | raw head — the honest control for `gbm_cal` |
| `gbm_cal` | inner train slice | `gbm_inner` mapped through the isotonic fit on the calib slice |
| `item_rate` | train-fold per-item exceedance frequency (`min_obs=10`, fallback pooled) | featureless per-item rate |
| `global_rate` | train-fold pooled exceedance rate | the shipping floor for log loss |

Calibration slice: last `--calib-frac` (default 0.25) of the train fold's
DATES; inner train purged at the calib boundary with the H+13 embargo. Folds
too short to yield both slices score no `gbm_cal` (drop the fold, never fit
the map on head-seen rows). Map via production's own `_isotonic_fit`
(piecewise-linear interp, flat outside); minimum
`MIN_EXCEEDANCE_CALIBRATION_ROWS` pairs or no map.

Cohort: served ≥$1 only, matching every read on this panel. Primary cohort is
**held-out items** (generalisation is the claim); trained-eval reported.

Metrics per fold: AUC, log loss (primary — the head is served as a
probability), Brier, ECE + fixed-width (10-bin) reliability tables pooled over
held-out val rows per horizon (diagnostic, not verdict).

## Bars (held-out, per horizon at h=3/7/14)

- **LAYER WORKS:** `gbm_cal` vs `gbm_inner` held-out log-loss delta negative
  with the 95% paired CI clear of zero, **and** the AUC delta exactly 0.00000
  at both CI ends (the monotone signature — any reorder voids the fold set).
- **SHIPPING TEST:** `gbm_cal` vs `global_rate` held-out log-loss delta
  negative with the CI clear of zero. A calibrated head that cannot beat a
  constant at stating numbers is misleading, however well it ranks (the exact
  30d-anomaly shape).
- **CONFIRMED at a horizon:** both bars pass. **NOT CONFIRMED:** anything else.
  Horizons are independent — a pass at h=7 ships nothing at h=14.

Reference (not bars): `gbm` vs `global_rate` (head-vs-null sanity — must stay
positive on AUC, or the frame not the head is broken); `item_rate` vs
`global_rate` on held-out (expected collapse to ~null — fourth reproduction of
pooled-beats-per-item if so).

## Void conditions

- Fewer than 10 paired folds at a horizon (no interval worth quoting).
- Any fold/grid/embargo/universe/allowlist change after a number is seen.
- Any `gbm_cal` score computed on its own calibration rows (in-sample).
- Arms seeing different val rows in any fold (unpinning).
- Quoting the trained cohort as the read, or any pooled-across-horizons number.
- Served-panel map-fitting at ≤6 dates, for any reason.

## Cost and instrument

Offline walk-forward LightGBM only (≈2 fits × ~25 folds × 3 horizons), no
serving touch, no retrain, no dispatch. Frame via `build_frame` +
`--frame-cache` (same 1.46M-row frame as `exceedance_meta_ab`).
Reproduce:

```
python -m scripts.exceedance_calibration_ab --horizon 7 \
  --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \
  --frame-cache /tmp/exc_cal_frame.parquet --out /tmp/exc_cal_h7.json
python -m scripts.exceedance_calibration_ab --served-only  # maturity + raw served reliability, no fit
```
