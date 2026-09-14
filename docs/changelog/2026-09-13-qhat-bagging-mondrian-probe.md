# q_hat bagging vs Mondrian probe: variance confirmed, bagging null, Mondrian confounded-but-real

**Date:** 2026-09-13
**Script:** `backend/scripts/measure_qhat_bagging_mondrian.py`
(+ `backend/tests/test_measure_qhat_bagging_mondrian.py`, 8 tests)
**Status:** diagnostic only — no wiring, no serving change.

Pre-registered in the script docstring. Model-free realised-return scores
(`|actual|/sigma`, `r_hat = 0`), calibration grids mirroring
`_compute_cv_splits` (30-date val windows, 150-step, end-anchored, K=4
shifts), fixed shared test grid (stride 7, ~180 dates/horizon),
level-matched verdict (mean width at forced 80% marginal), shuffled-sigma
placebo, 70% top-decile guardrail.

## 1. The defect reproduces: 1.11x–1.39x across shifted grids, same data

Pooled q_hat per grid on full-history OOF windows (K=4 shifts):

| h | grid qs | max/min |
|---|---|---|
| 3 | 95.13 → 91.79 → 86.02 → 95.11 | 1.106 |
| 7 | 130.34 → 117.53 → 121.86 → 140.26 | 1.193 |
| 14 | 159.91 → 160.02 → 163.13 → 198.87 | 1.244 |
| 30 | 203.94 → 236.49 → 284.08 → 223.90 | 1.393 |

The swing grows with horizon. Same rows, different grid, up to 39% width
arbitrariness at h=30.

## 2. Bagged (mean/median of K grid quantiles): null

LM mean width vs S0: h=3 10.43/10.53 vs 10.28 (worse), h=7 13.28/13.31 vs
13.09 (worse), h=14 17.79/17.98 vs 18.23 (-2%), h=30 25.56/25.83 vs 25.72
(-1%/+0%). The mean of regime-dependent quantiles is still
regime-dependent; averaging only smooths sampling noise, which was never
the 26%. **Do not pay conformal-CV × K for this.**

## 3. Mondrian (per-sigma-decile empirical quantiles): narrower 4/4, but the placebo matches — third void of the same shape

LM width S0 → Mondrian: 10.28→8.26, 13.09→10.32, 18.23→13.56,
25.72→19.20 (-20–26%), guardrail (min LM decile ≥70%) passes 4/4
(77/77/72/75%). But the shuffled-sigma placebo is narrower still at 3/4
horizons (7.76/10.02/13.34/18.88). Binning machinery alone narrows width —
the 2026-08-12 S1/P1 and 2026-09-09 WACI signatures, third occurrence.
**Width at matched coverage cannot separate Mondrian from bagging-noise;
the width bar is VOID again.**

## 4. The unconfounded contrast is clean: treatment beats placebo 4/4 on the sigma axis

LM sigma-strata err, Mondrian vs placebo: h=3 1.40 vs 5.91, h=7 2.84 vs
7.07, h=14 3.90 vs 6.63, h=30 2.04 vs 5.68. Top-decile LM coverage:
treatment 84/83/81/82% vs placebo 69/67/68/69% — the placebo buys its
narrowness by abandoning the volatile decile, which is exactly what the
guardrail was written for (catches it 4/4). Real scale information is
present; the metric that saw it was treatment-vs-placebo sigma err, not
width.

## Verdict per the pre-registered bar

No arm passes (bagged fails width; Mondrian fails M1-in-band at h=14/30
— 76.0/75.9% — and the placebo voids width). So: **no OOF-confirm on
width grounds, no wiring.** The earned follow-up is a re-registered
Mondrian-vs-placebo confirm on **real OOF residuals** (this panel is
sigma-symmetric and model-free; production serves a climatology-signed
band, so only ratios transfer) with sigma-strata err as primary and the
70% guardrail attached — a contrast Mondrian would pass 4/4 here.

## Reproduce

```
venv/bin/python -m scripts.measure_qhat_bagging_mondrian --horizons 3,7,14,30 --out qhat_bagging_mondrian.csv
```
Per-horizon CSVs from this read: `/tmp/qhat_probe_h{3,7,14,30}.csv`.
