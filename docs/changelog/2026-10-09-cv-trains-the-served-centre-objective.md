# 2026-10-09 — CV trains the centre on the objective production serves

Performance review 2026-10-08, correctness finding 3.

## Defect

`CENTRE_OBJECTIVE` (`30:regression` by default, and set explicitly in
`price-forecast.yml`) chooses the served centre's loss per horizon. The 09-17 change
(`changelog/2026-09-17-per-horizon-objective-and-ranking-head.md`) wired it into three places:
Optuna, the HP-reuse path and the fresh-search path. It missed the fourth, the CV fold loop in
`_cv_evaluate_horizon`, which kept forcing `objective="quantile", alpha=q`.

So the 30d OOF residuals came from an MAE model while the served 30d centre was trained on
MSE. Every 30d artifact since the 09-17 change has been affected in three ways:

- `q_hat` and the signed `(q_lo, q_hi)`;
- the fold rank IC, DA and PT records;
- the published `cv_results[30]`.

The exceedance head and its calibrator train their own fold booster, so they are not affected.

The 09-17 note's line "same conformal calibration, no predict-path changes needed" was wrong
at 30d. h=3/7/14 were unaffected, because their map entry is quantile and CV already matched.

## Change

`ItemForecaster._centre_objective_params(horizon, q)` is now the single definition, and all
four sites read it. The CV loop drops any carried `alpha` and applies the served objective.
Pinned in `tests/test_cv_centre_objective.py`: CV fits regression and huber where the map says
so, unmentioned horizons keep quantile at the fold's alpha, and the helper's three outputs
are fixed.

## Measured before the retrain (offline, paired)

Both arms ran on one frame and one fold grid, with the 09-22 artifact's tuned 30d params and
its 28 features, through production's `_calibrate_conformal`. Only `CENTRE_OBJECTIVE_MAP[30]`
differed. Rows and scales are identical across the arms (checked). The local archive ends
2026-09-08, so the frame is 638,548 rows / 707 items, against production's ~958K / 926:
expect the size of the shift to carry over, not the exact values.

| 30d, 8 folds, 79,937 OOF rows | old (quantile CV) | fixed (regression CV) |
|---|---|---|
| `q_lo` / `q_hi` | −0.7323 / +1.2091 | −0.8695 / +1.1038 |
| width `q_hi − q_lo` | 1.9414 | 1.9733 (**+1.6%**) |
| signed midpoint `(q_lo + q_hi) / 2` | +0.238 | +0.117 |
| mean fold rank IC | 0.3375 | 0.3397 |
| OOF coverage of the **served** (regression) centre | **78.1%** | 80.0% |

- **The defect under-covered by ~1.9pp at 30d**, on OOF rows. The served MSE centre sits
  higher than the MAE one, so residuals are more negative, and a band calibrated on the MAE
  residuals was shifted up.
- **The fix shifts the band down more than it widens it.** It is +1.6% wider, and its signed
  midpoint moves −0.12 scale units.
- **Rank IC is unchanged.** The +0.041 rank-IC win the 09-17 change claimed for MSE at 30d does
  not appear in this frame's CV, but that was a different split, and this run is not a test of
  it.
- These are OOF figures, a different population from served outcomes
  (`.claude/rules/training-budget.md`), so this predicts no served coverage. The served read
  is the one below.

## What moves, and when

Only h=30. The change ships with the next `mode=full` retrain, **Mon 2026-10-12**; predict-only
runs reuse the saved artifact. The 30d `q_lo` / `q_hi` refit on regression-centre residuals,
so the 30d band width changes at the first forecast date that retrain serves (expected
**2026-10-11**; the 10-05 retrain served 10-04).

- **Split every h=30 coverage and width read at that date**, the same rule as the h=14 splits
  at 09-30 and 10-05 (next-steps items 7 and 8). Do not pool across it.
- The h=30 feedback factor has not woken yet (next-steps item 8). If it wakes on a panel
  that straddles the date, read it knowing the band changed underneath it.
- `cv_results[30]` rank IC, DA and PT are now scored on the served centre, so they are not
  comparable to earlier artifacts' 30d CV figures.

## Coverage check at the retrain

1. In the 10-12 Price Forecast log, compare the 30d calibration line (`q_lo`, `q_hi`) against
   the 10-05 artifact's `meta.json`. The objective is logged as `HORIZON 30d (objective:
   regression)`.
2. Served h=30 coverage on forecast dates ≥ 10-11 matures from ~11-10 onward. Read it split at
   10-11 against the 80% target.
