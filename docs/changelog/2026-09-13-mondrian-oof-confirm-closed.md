# Mondrian OOF confirm: FAIL — Mondrian is closed

**Date:** 2026-09-13
**Script:** `backend/scripts/confirm_mondrian_oof.py`
(+ `backend/tests/test_confirm_mondrian_oof.py`, 6 tests)
**Status:** diagnostic only — no wiring, no serving change. Third strike for
the Mondrian family (WACI void on placebo 2026-09-09, probe width void on
placebo 2026-09-13); do not reopen without a new axis, not a new metric.

Pre-registered in the script docstring. Real q50 OOF residuals from
production-footing CV (FEATURE_NATIVE_NAN=1, EXCEEDANCE_HEAD=1,
CV_DIAGNOSTIC_CLASSIFIER=0, all = price-forecast.yml; raw-anchor basis, as
served), default untuned q50 params (no local meta.json — common to all
arms, contrasts only), 9 folds / 168k OOF rows (h=7) and 159k (h=14).
Probe's own scoring machinery (same bins, levels, level-match). Test = OOF
dates in the last 3 folds. Bar: M sig* < P sig* AND M1_M in 80±3pp AND M
min decile ≥70%, at BOTH horizons.

## Numbers

| h | arm | M1 | LM width | sig* M/P | mindec M |
|---|---|---|---|---|---|
| 7 | S0 | 78.5% | 19.17 | 7.16 | 59% |
| 7 | M | 75.8% | 16.58 | **3.74** vs 8.76 | 71% |
| 7 | P | 73.5% | 16.36 | 8.76 | 63% |
| 14 | S0 | 77.9% | 27.09 | 9.94 | 49% |
| 14 | M | 74.3% | 22.32 | **5.18** vs 6.20 | 63% |
| 14 | P | 71.4% | 22.20 | 6.20 | 66% |

(a) passes 2/2 — treatment beats placebo on sigma-strata err at both
horizons, and the placebo's width edge shrank to ~1%. (b) fails 2/2:
M1_M = 75.8/74.3% against the 77% floor. (c) fails at h=14 (63%).
**Verdict: FAIL. No flag, no wiring.**

## Reading the failure honestly

The failure is the level leg, and it is not a technicality: Mondrian
covers 2.7–3.6pp worse than pooled marginally out-of-sample (75.8 vs
78.5, 74.3 vs 77.9). Per-bin quantiles fit on less data plus a
calibration→test regime shift (calib folds end 2025-06, test spans
2025-10→2026-09 including the calm 2026 window) — and the low-sigma bin
is where it breaks (h=14 decile-1 LM coverage 63%: the
sigma→residual relationship itself drifted). With served-feedback
releveling still dormant, a −3pp level hit would ship. That is what leg
(b) exists to catch.

## Side finding: the defect is bigger on real residuals

Real-OOF per-fold q_hat spread (beta=1.0, comparable units): h=7
62.96→229.31 (**3.64x**), h=14 88.34→301.66 (**3.42x**) — vs 1.19x/1.24x
model-free. Caveat: early folds train on less data (weaker models, larger
residuals), so 3.6x overstates the pure grid effect; but the direction is
unambiguous. Bagging stays refuted regardless (probe: ±2%).

## Reproduce

```
venv/bin/python -m scripts.confirm_mondrian_oof --horizons 7,14   # train + score
venv/bin/python -m scripts.confirm_mondrian_oof --phase score --horizons 7,14  # rescore saved OOF
```
OOF from this read: `/tmp/mondrian_oof/oof_h{7,14}.parquet`
(`--workdir` to relocate; rescoring never retrains).
Verdict CSV: `/tmp/mondrian_oof_confirm.csv`.
