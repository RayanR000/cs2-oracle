# Exceedance isotonic calibration measured: keep the layer, h=7 is load-bearing

**Date:** 2026-09-13
**Instrument:** `backend/scripts/exceedance_calibration_ab.py`
(+ `backend/tests/test_exceedance_calibration_ab.py`, 7 tests)
**Prereg:** `docs/research/2026-09-13-exceedance-calibration-preregistration.md`
(written before any number was computed)
**Status:** diagnostic only — no wiring, no serving change. `EXCEEDANCE_CALIBRATE`
is already default-on; this probe confirms that is correct rather than changing it.

Pre-registered design, as fixed in advance: all five arms share one fold
enumeration, one `_stratified_sample` per fold, identical val rows (paired
deltas on fold index only — the 3.4–3.6x real-OOF spread is the reason), and
the isotonic map is fitted on a calibration slice date-disjoint from the inner
train slice with the H+13 embargo at the inner boundary, scored on the val
window only. 24–26 paired folds per horizon, served ≥$1 cohort, production's
own `_fit_exceedance_classifier` / `_isotonic_fit` / `(1e-3, 1.0)` clip.

## Verdict (held-out, per horizon)

| h | LAYER: cal vs inner logloss | SHIPPING: cal vs global logloss | read |
|---|---|---|---|
| 3d | −0.00201 [−0.00456, +0.00054], ns | **−0.00561 [−0.00794, −0.00328], SIG** | NOT CONFIRMED |
| 7d | **−0.00947 [−0.01670, −0.00225], SIG** | **−0.00401 [−0.00788, −0.00015], SIG** | **CONFIRMED** |
| 14d | −0.00393 [−0.01139, +0.00353], ns | **−0.00846 [−0.01405, −0.00286], SIG** | NOT CONFIRMED |

CONFIRMED needed both bars. Only h=7 passes both — and there the shape is the
exact positive resolution of the 30d-anomaly question: raw `gbm` vs the pooled
constant is **ns** (−0.00490 [−0.01002, +0.00023]) while calibrated `gbm_cal`
is **SIG** (−0.00401, CI clear of zero). At h=7 the layer is what carries the
head over the shipping floor. At h=3/14 the raw head already beats the
constant (SIG at both), and isotonic adds a directionally-positive ns gain —
pooled held-out ECE still falls (0.21→0.15pp at h=3, 0.65→0.09pp at h=14), and
at no horizon on no metric is calibrated significantly worse than raw.

## What this buys

1. **Keep `EXCEEDANCE_CALIBRATE` on.** The shipped default survives its first
   honest measurement: monotone, never significantly harmful, load-bearing at
   h=7, ECE-improving everywhere. No flag change, no artifact bump needed.
2. **`item_rate` collapses onto `global_rate` on held-out exactly** (pooled
   Brier identical to 5dp at all three horizons) — fourth reproduction of
   pooled-beats-per-item, this time on the exceedance label. No per-item-rate
   follow-up.
3. **Served baseline recorded, no fit.** Served `exceed_p` sits on 6/5/6 clean
   dates at h=3/7/14 (0 at h=30) — below the 20-date gate, so nothing was
   fitted there by design. Raw served reliability: ECE 0.27/1.21/2.70pp with
   base rates 0.5–0.9% (calm-window regime); the bulk bin overstates at h=14
   (pred 0.032 vs realised 0.007, n=5,319). Immature — but if that overstatement
   persists past 20 dates, the OOF-fitted map is overstating into the calm
   served regime (the same calibration→serving shift that sank Mondrian's
   low-sigma bin), and the fix is a served refit, not a fancier map. Re-check
   with `--served-only` at maturity.

## Amendment to the prereg (read, not re-run)

The prereg's monotone signature bar ("AUC delta exactly 0.00000 at both CI
ends") was overstated: a stepwise isotonic map has flat segments that tie
previously-ordered pairs, and ties move ROC AUC by small amounts in either
direction (verified per-fold: the nonzero means come from 1–3 folds with flat-
segment ties, e.g. h=14 fold 1 +0.038 on a near-degenerate slice). Measured
AUC deltas are +0.00063 / +0.00000 / +0.00152, all ns — no reordering, layer
honest. The bar is corrected to "AUC delta ns and |mean| ≤ 0.005" for any
re-read; the verdicts above already apply it.

## Reproduce

```
python -m scripts.exceedance_calibration_ab --horizons 3,7,14 \
  --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \
  --frame-cache /tmp/exc_cal_frame.parquet --out /tmp/exc_cal_full.json
python -m scripts.exceedance_calibration_ab --served-only  # maturity + raw served reliability, no fit
```

Full paired tables + pooled reliability: `/tmp/exc_cal_full.json`
(this run) and `/tmp/exc_cal_full.log`.
