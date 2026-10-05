# Outside baseline: the old served band lost to naive and AutoETS; the current h=3 band ties

**Date:** 2026-10-04
**Script:** `backend/scripts/outside_baseline.py` (+ `backend/tests/test_outside_baseline.py`, 9 tests)
**Dependency:** `statsforecast` 2.1.1 in a new `baseline` extra (not `dev`: it is heavy and
CI tests only the naive math). `uv.lock` adds packages only, no version moves.
**Status:** measured. No modelling or serving change. Closes the statsforecast line that
next-steps item 22 left unstarted (`references/open-source-shortlist.md` #3).

## Question

On the served panel, does the served band beat what a reviewer would build in an
afternoon? This is the credibility check for `PORTFOLIO.md`, and the shortlist expected
it to say nothing about accuracy.

## Arms

All three are scored on the same rows, as ratios to `base_price`:

- **served**: the archived band, rebased to the model's own quote as in
  `scripts/archive/conditional_coverage_by_stratum.py`.
- **naive**: random walk. The band is the empirical 10th-90th percentile of the item's
  trailing h-day log returns over 365 days, needing at least 30 returns.
- **ets**: statsforecast `AutoETS` on daily log price (365 days, forward-filled, at least
  200 points), with `ConformalIntervals(h=30, n_windows=5)` at 80%.

The baselines see only prices at or before the forecast date (tested). They fit on the
scorer's own basis: `load_voted_prices`, the loader `backtest_accuracy.py` resolves actuals
through, smoothed to the trailing median of three observations as `resolve_anchors` does.
That loader applies no extra universe filter, so neither do the baselines. Rows where any arm
is missing are dropped from all three (142 of 123,863).

The panel is the 10-04 data-repo `forecast_outcomes.parquet`: the served cohort (>=$1),
`excluded_forecast_date` applied, 1,281 items. That is the whole served universe, so there is
no item sampling.

The metric is the interval score at alpha = 0.2 (`backtest.scoring.interval_score`), which
prices width and misses in one proper number. Lower is better. "served - arm" is the paired
per-date difference with a 90% date-bootstrap CI, so **positive means the served band is
worse**. "hw @served cov" is each baseline's median half-width after rescaling its log band
to the served coverage on the same rows. That scale is fitted in hindsight and reads as "how
wide at the same coverage", not as a forecast.

## Pooled: every forecast date (47)

| h | arm | rows | dates | cover | med hw | IS | served - arm [90% CI] | hw @served cov |
|---|---|---|---|---|---|---|---|---|
| 3 | served | 34,809 | 38 | 89.1% | 6.30% | 0.1959 | | |
| 3 | naive | | | 93.1% | 6.65% | 0.1757 | +0.020 [+0.011, +0.030] | 5.44% |
| 3 | ets | | | 82.8% | 4.78% | 0.1778 | +0.018 [+0.010, +0.025] | 6.45% |
| 7 | served | 37,045 | 40 | 92.6% | 11.11% | 0.2971 | | |
| 7 | naive | | | 93.7% | 9.91% | 0.2614 | +0.035 [+0.023, +0.048] | 9.38% |
| 7 | ets | | | 84.1% | 7.38% | 0.2573 | +0.039 [+0.025, +0.054] | 11.00% |
| 14 | served | 32,334 | 35 | 90.0% | 14.69% | 0.4281 | | |
| 14 | naive | | | 92.4% | 13.34% | 0.3772 | +0.050 [+0.028, +0.073] | 12.00% |
| 14 | ets | | | 85.0% | 10.91% | 0.3654 | +0.062 [+0.037, +0.088] | 13.30% |
| 30 | served | 19,533 | 21 | 92.4% | 27.67% | 0.7155 | | |
| 30 | naive | | | 91.0% | 19.57% | 0.6142 | +0.104 [+0.058, +0.149] | 21.40% |
| 30 | ets | | | 89.4% | 17.11% | 0.5508 | +0.167 [+0.121, +0.212] | 19.68% |

Pooled, both baselines beat the served band at every horizon, with every CI clear of zero.
At the served band's own coverage, naive is narrower at all four horizons.

**The pooled table spans band-geometry eras and is not quotable as a level** (next-steps
§7: "Pooling across band-geometry eras"). The baselines are one fixed method throughout,
while the served band changed at the 09-20 CSMarketAPI retrain and the h=3 feedback factor
(served from 09-21).

## Split by era

Served - naive interval score (naive arm only, same rows):

| Forecast dates | h=3 | h=7 | h=14 | h=30 |
|---|---|---|---|---|
| <=09-14 | +0.035 [+0.022, +0.048], 23 d | +0.041 [+0.025, +0.058], 29 d | +0.059 [+0.036, +0.083], 31 d | +0.104 [+0.058, +0.149], 21 d |
| 09-15..09-27 | -0.003 [-0.008, +0.003], 12 d | +0.019 [+0.012, +0.026], 11 d | -0.019 [-0.021, -0.016], 4 d | n/a |

`--since 2026-09-21`, both baselines:

| h | arm | rows | dates | cover | med hw | IS | served - arm [90% CI] |
|---|---|---|---|---|---|---|---|
| 3 | served | 8,719 | 10 | 84.0% | 4.85% | 0.1577 | |
| 3 | naive | | | 94.3% | 6.42% | 0.1625 | -0.005 [-0.007, -0.002] |
| 3 | ets | | | 81.0% | 4.03% | 0.1564 | +0.001 [-0.001, +0.004] |
| 7 | served | 5,254 | 6 | 94.7% | 11.57% | 0.2693 | |
| 7 | naive | | | 95.3% | 9.28% | 0.2393 | +0.030 [+0.028, +0.032] |
| 7 | ets | | | 82.5% | 5.86% | 0.2094 | +0.060 [+0.056, +0.063] |

h=14 and h=30 have no resolved dates after 09-21 yet.

## Verdict

- **The pre-09-20 served band lost to both off-the-shelf baselines at every horizon.** Its
  bands were wider on average, with a fat right tail of very wide bands, and it missed on
  the downside about twice as often as naive (h=3 on a 50-item check: 8.5% vs 3.9% below the
  band). A per-item empirical return quantile is itself a featureless climatology, so this
  is in line with the 08-20 result that featureless beats modelled. The served pipeline
  layered a biased centre and a pooled scale on top of it.
- **The current h=3 band ties.** Since 09-21 it is narrowly better than naive (CI clear of
  zero) and level with AutoETS, at 84% coverage against naive's over-covering 94%. That
  is 10 dates, below the 20-date bar, so it is directional only.
- **h=7 still loses** on 6 dates, but h=7's feedback factor only began serving on 09-28,
  so these are mostly pre-factor bands. h=14/h=30 have no post-change data.
- **Not a reason to change serving today.** Swapping the served band for a naive quantile
  would be a new band arm, and it gets a preregistration, not a pooled read. The
  re-read below decides whether that prereg is worth writing.

## Re-read

Run `--since 2026-09-28` once each horizon has 20 resolved dates in the feedback-factor
era: about 10-23 for h=3 and late October for h=7. h=14 and h=30 follow in November.
If h=7 still loses to naive at 20 dates, preregister a naive-quantile band arm.

## Reproduce

Extract `price-archive/ops/forecast_outcomes.parquet` and `price-archive/prices-2025.parquet`
plus `prices-2026-*.parquet` from the data repo's `origin/main` into `<dir>/ops/` and `<dir>/`.
Then, from `backend/`, after `pip install statsforecast==2.1.1` or the `baseline` extra:

```
venv/bin/python scripts/outside_baseline.py --archive-dir <dir>                      # pooled, ~15 min
venv/bin/python scripts/outside_baseline.py --archive-dir <dir> --since 2026-09-21   # ~3 min
```

The era-split table is the naive arm of:

```
venv/bin/python scripts/outside_baseline.py --archive-dir <dir> --no-ets --until 2026-09-14
venv/bin/python scripts/outside_baseline.py --archive-dir <dir> --no-ets --since 2026-09-15 --until 2026-09-27
```
