# Pre-registration: served h=7 band vs a naive empirical-quantile band (served panel)

**Date:** 2026-10-08, written **before any h=7 outcome in the window is scored.**
**Status:** FROZEN 2026-10-08 on commit. Nothing may be scored before this commit, and
nothing below may change after it except under a dated *Amendments* entry written before
the read.

**Seen before freezing (disclosed):**

- (a) Every number in `changelog/2026-10-04-outside-baseline-statsforecast.md`, including
  the h=7 read since 09-21: served − naive +0.030 [+0.028, +0.032] on 6 dates. By resolution
  arithmetic those were forecast dates 09-21..09-26, and prod records an h=7
  `band_multiplier` of 1.0 on every one of them. That changelog called them "mostly"
  pre-factor; in fact they were **all** pre-factor.
- (b) The per-date mean h=7 `band_multiplier` from prod `item_forecasts` over forecast dates
  09-24..10-07, read today to fix the window: 1.0 through 09-26, then 0.6694 (09-27), 0.6199,
  0.6124, 0.6113, 0.5940, 0.5914, 0.5910, 0.5827 (10-04), 0.5815, 0.5813, 0.5812 (10-07). No
  outcome or coverage figure was read with it.
- (c) The served-coverage factor log lines in `research/2026-09-28-next-steps.md` item 8.

No h=7 naive, ETS or served interval score, and no h=7 coverage figure, has been computed for
any forecast date ≥ 09-27. The daily Backtest Accuracy runs publish h=7 coverage, but none was
read for this prereg.

**Follows:** `changelog/2026-10-04-outside-baseline-statsforecast.md`, which says: "If h=7
still loses to naive at 20 dates, preregister a naive-quantile band arm." This doc is that
arm. It is written now rather than after the re-read so that the re-read **is** the
preregistered read. **This prereg governs the h=7 `outside_baseline.py` re-read.** No
separate unregistered h=7 re-read is to be run or quoted.

## Question

Under its live policy (climatology scale, signed conformal, weekly-refit served-coverage
factor), does the served h=7 band beat a featureless random-walk band built from the item's
own trailing return quantiles, on a proper interval score?

A PASS does not change serving. It authorises a design spec only (see *Outcomes*).

## Why h=7

h=3 already ties both baselines (10 dates since 09-21, reported only). h=14 and h=30 have no
resolved factor-era dates. h=7 is the only horizon with a measured loss that its feedback
factor may since have repaired. The factor narrows the band by ~40%, and the loss was a width
loss: the served band covered 94.7% at a 11.57% median half-width, against naive's 95.3% at
9.28%. So the prior is genuinely open in both directions.

## Window (fixed now, no optional stopping)

- **Start:** forecast date **2026-09-27**, the first h=7 date with a recorded
  `band_multiplier` < 1. The 10-04 changelog's suggested `--since 2026-09-28` and next-steps'
  "first forecast date 09-28" both predate this lookup and are superseded by it.
- **Length:** the **first 20 scoreable** h=7 forecast dates ≥ 09-27. A date is scoreable if
  it survives `load_panel` (≥ $1, well-formed band, `excluded_forecast_date` not set) and has
  resolved outcomes. Expect forecast dates 09-27..~10-16 and a read around **10-25**.
- **Truncation:** a band-geometry **code** change truncates the window at its first served
  forecast date. Examples are a new scale, a new conformal method, a flag flip on the band
  path, or a PID promotion. **Scheduled Monday refits of the served-coverage factor do not
  truncate it**: the weekly refit is part of the incumbent policy being tested, as in
  `2026-09-23-conformal-pid-served-h3-preregistration.md` (arm B). That is what lets this
  window pool across 09-27 and 10-04 without breaching next-steps §7's "pooling across
  band-geometry eras". The 10-04 refit is instead used as the era-split guard (bar 3).
- **Deadline:** if 20 scoreable dates do not exist before a truncation, or by
  **2026-11-30**, the read is **VOID**.

## Arms (same rows, as ratios to `base_price`)

All three are exactly as `backend/scripts/outside_baseline.py` builds them at commit
`997e31c` (#91), which is its latest change.

- **S, served:** the archived band, rebased to the model's own quote (`load_panel`).
- **N, naive (the candidate):** a zero-centred random-walk band. Its ends are the empirical
  10th and 90th percentiles of the item's trailing 7-day log returns over `WINDOW_DAYS` = 365,
  needing `MIN_RETURNS` = 30. The returns are taken on the scorer's basis (`load_voted_prices`,
  trailing three-observation median) and use only prices at or before the forecast date. No
  scale is fitted.
- **E, AutoETS + conformal:** reported only, never judged.

Rows where any arm is missing are dropped from all of them. This means the read requires
`statsforecast` 2.1.1 (the `baseline` extra). If E cannot run, rerun with `--no-ets` and record
in the read that E is missing; that doesn't void the read.

## Metric and inference (the script's constants, frozen)

- **Metric:** the interval score at α = 0.2 (`backtest.scoring.interval_score`), averaged per
  date. The **primary statistic** is the mean over dates of per-date (IS_S − IS_N). Positive
  means the served band is worse.
- **CI:** a date-level bootstrap with `n_boot` = 2000, `RNG_SEED` = 20261004, and the 5th and
  95th percentiles. This is a two-sided 90% interval, i.e. a one-sided 95% test of "N better".
- **Coverage:** marginal coverage of S and N over the window's rows.

**Constant check (VOID if any moved):** `ALPHA` 0.2, `WINDOW_DAYS` 365, `MIN_RETURNS` 30,
`RNG_SEED` 20261004, `paired_date_diff` defaults (2000 draws, 5/95 percentiles), and the
`load_panel` SQL filters. Diff `scripts/outside_baseline.py` against `997e31c` before reading.
If anything in those paths changed, the read is VOID unless an amendment dated before the read
explains it.

**Row-loss check (VOID if it fails):** in the window, the common h=7 rows must be ≥ 95% of the
h=7 panel rows that had a baseline attempt.

## Bar (fixed now)

N **passes** only if all of these hold:

1. **IS win:** the primary statistic is > 0 **and** the CI's lower bound is > 0.
2. **Coverage guard:** |cov_N − 80| ≤ |cov_S − 80| + 2pp. A narrower interval score bought
   with worse calibration does not pass. Note the interval score could favour N here only
   through width, because N over-covered (93–95%) on every read so far.
3. **Era-split guard:** the per-date mean (IS_S − IS_N) is > 0 in **both** sub-windows: forecast
   dates 09-27..10-03 (factor ~0.59–0.67) and 10-04 onward (factor ~0.58). Point estimates
   only, no CI. This stops a pass carried entirely by the 2–3 blended dates at the start.

## Outcomes

| Result | Condition | Action |
|---|---|---|
| **PASS** | 1, 2 and 3 hold | Write `specs/` design for a naive-quantile h=7 band: what it replaces, its own level calibration (N over-covers, so it needs one), and a shadow period before serving. **Serving is not changed by this read.** experiment_log `measured`. |
| **UNRESOLVED** | 1 holds, 2 or 3 fails | No arm. The loss is width the feedback factor has not removed, or it sits in the start-up dates. It feeds the 10-23 PID read's context, not a new band. experiment_log `inconclusive`. |
| **TIE** | the CI spans 0 | The served h=7 band stands. No re-read is scheduled. experiment_log `inconclusive`. |
| **SERVED BETTER** | the CI's upper bound is < 0 | Close the naive-band line at h=7. experiment_log `refuted`. |
| **VOID** | a window, constant or row-loss condition fails | Record why. A re-read needs a new prereg. experiment_log `void`. |

Report beside the verdict, never judged: n_dates, rows, both coverages and median
half-widths, served − E with its CI, each sub-window's estimate, and the hindsight
"hw @served cov" column (a scale fitted on the outcomes, not a forecast).

## Out of scope

- **h=3** is re-read with the same script around 10-23, reported only, not judged here.
- **h=14 and h=30** have too few factor-era dates. They would need their own prereg in
  November.
- **Replacing the band with N, or blending S and N.** A PASS argues for a design, and the
  design gets its own served test.
- **Any sigma-shaped scale or fourth denominator** (`AGENTS.md`). N is featureless per-item
  climatology, the same family that won on 08-20, not a modelled width.

## Reproduce (at the read)

Extract `price-archive/ops/forecast_outcomes.parquet`, `price-archive/prices-2025.parquet` and
`prices-2026-*.parquet` from the data repo's `origin/main` into `<dir>/ops/` and `<dir>/`. Set
`<D20>` to the 20th scoreable h=7 forecast date ≥ 09-27, taken from the panel's date list.
Then, from `backend/`:

```
git diff 997e31c -- scripts/outside_baseline.py backtest/scoring.py   # constant check
venv/bin/python scripts/outside_baseline.py --archive-dir <dir> --since 2026-09-27 --until <D20> --json-out <out>.json
venv/bin/python scripts/outside_baseline.py --archive-dir <dir> --no-ets --since 2026-09-27 --until 2026-10-03
venv/bin/python scripts/outside_baseline.py --archive-dir <dir> --no-ets --since 2026-10-04 --until <D20>
```

Read only the h=7 rows of each. The other horizons in the same output are reported only.

## Amendments

None.
