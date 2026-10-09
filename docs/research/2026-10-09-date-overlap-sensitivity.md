# Date-overlap sensitivity for the October reads — 2026-10-09

Written before either read it covers. It **adds** reported rows. It edits no frozen
preregistration, changes no bar, and changes no verdict.

Covers:
- the PID read (`research/2026-09-23-conformal-pid-served-h3-preregistration.md`, ~10-23);
- the h=7 naive-quantile band read (`research/2026-10-08-naive-quantile-band-h7-preregistration.md`,
  ~10-25);
- the h=3 outside-baseline re-read (next-steps item 22, reported only, ~10-23).

## The problem

Both preregistrations freeze iid date bootstraps:
- PID: `paired_bootstrap`, 10,000 draws, seed 0, 95%;
- h=7: `paired_date_diff`, 2,000 draws, seed 20261004, 90%.

Both resample forecast dates as if they were independent, and they are not. Two h-day
forecasts made k < h days apart score against h − k shared forward days, so a per-date series
is at least MA(h − 1) by construction. The 10-08 performance review flagged this, and its
worst-case simulation (not real data) put nominal 90% coverage at 36% for h=7.

## Measured on pre-window dates (no read was peeked at)

The read's own pipeline was used (`outside_baseline.load_panel` and `build_baselines`, the
naive arm, on the data repo's `origin/main` as of 2026-10-09 15:09Z). Only dates before each
window were used: h=3 < 09-07 (PID start), h=7 < 09-27 (naive-band start). The ACF is taken at
calendar lags, because forecast dates have gaps. Robust intervals: a circular moving-block
bootstrap (block = h) and a Bartlett HAC (lag h − 1) with t(n−1).

| Series | Dates | ACF lag 1 / 2 / 3 | MBB width ÷ iid | HAC width ÷ iid |
|---|---|---|---|---|
| h=7 IS served − naive, signed band (≥ 08-19) | 24 | +0.86 / +0.62 / +0.26 | 1.38 | 1.34 |
| h=7 IS served − naive, all | 40 | +0.94 / +0.69 / +0.53 | 1.65 | 1.71 |
| h=7 served \|coverage − 80\|, signed band | 24 | +0.73 / +0.36 / +0.13 | 1.11 | 1.17 |
| h=3 IS served − naive, all (08-04..08-24) | 15 | +0.84 / +0.49 / +0.52 | 1.34 | 1.43 |
| h=3 served \|coverage − 80\|, all | 15 | −0.14 / −0.26 / −0.06 | 0.79 | 0.77 |

- **The interval-score difference is strongly autocorrelated at both horizons.** The iid
  intervals are about 1.35–1.7× too narrow. A nominal 90% iid interval is closer to a 70–75%
  one: material, but far from the review's simulated 36%.
- **Per-date coverage error is less dependent.** At h=3 it is mildly negative. The PID
  statistic is a difference of two arms' coverage errors on the same dates, which is not
  measured here: its arms need the in-window panel.
- **A worked example of the failure mode.** On the 24 pre-window signed-band h=7 dates, the
  iid 90% CI for served − naive is [+0.0066, +0.0368], above zero ("naive better"). The HAC
  interval is [−0.0006, +0.0416], which spans zero. These are pre-factor dates, outside the
  read's window, and are no evidence about the read itself.

## Reading rule (fixed now)

At each read, run `backend/scripts/date_dependence_sensitivity.py` with the read's own
arguments, `pid` or `outside-baseline --horizon {3,7} --since … --until …`. It reproduces the
frozen iid interval to 1e-9 and refuses if it can't, because a mismatch means the series is
wrong. It refuses before 10-23 for PID, and before a window holds 20 dates for the
outside-baseline reads.

1. **The frozen verdict governs.** It is recorded exactly as its prereg says, in the changelog
   and in `experiment_log.csv`.
2. **The sensitivity row is reported beside it:** the ACF to lag h, and all three intervals with
   their width ratios.
3. **FRAGILE** means a robust interval (MBB or HAC) puts zero on a different side than the iid
   one. A fragile verdict is written as "<verdict>, fragile to date overlap", in the changelog
   and in the experiment-log notes.
4. **Downgrade only, never upgrade.** A robust interval that clears zero where the iid one does
   not changes nothing. It is not a pass.
5. **A fragile PASS (PID) or NAIVE BETTER (h=7) cannot carry a serving change on its own.** The
   follow-up has to state the fragility and use a dependence-robust interval in its own gate.
   For h=7 the frozen consequence is already only a design spec; that spec inherits this rule.

## Not done

- No frozen instrument was edited. `git diff 997e31c -- scripts/outside_baseline.py
  backtest/scoring.py` is empty.
- `scoring.block_bootstrap_ci`, and `promotion.py` through `candidate_scoring.paired_daily_interval`,
  still resample iid dates. Moving them to a
  robust interval changes published figures, so it is a separate, dated change.
- No `arch` dependency was added. The block bootstrap is about 15 lines in
  `backtest/date_dependence.py`, and HAC reuses `directional_test.hac_long_run_variance`.
