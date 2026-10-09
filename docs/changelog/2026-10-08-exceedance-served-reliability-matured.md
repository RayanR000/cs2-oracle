# Served `exceed_p` reaches maturity: overstated, but a level refit is the wrong fix

**2026-10-08.** Measurement only. No code, flag or served number changed.

The 09-13 note (`2026-09-13-exceedance-calibration-measured.md`) recorded the h=14 bulk bin at
0.032 predicted against 0.007 realised on 6 dates, and said that if it persisted past 20 dates
the fix is "a served refit, not a fancier map". It has matured, so this is that re-check.

## Reading

Read-only `python -m scripts.archive.exceedance_calibration_ab --served-only` against prod
`forecast_outcomes` joined to `item_forecasts`, ≥$1, `excluded_forecast_date` applied. The label is
the script's own: `actual_ret > actionable_threshold(tier, "csfloat")`.

| h | dates | rows | base rate | ECE | bulk bin [0, 0.1) predicted vs realised |
|---|---|---|---|---|---|
| 3 | 30 | 27,289 | 0.3% | 0.41pp | 0.006 vs 0.002 |
| 7 | 28 | 24,908 | 0.3% | 1.11pp | 0.013 vs 0.002 |
| 14 | 22 | 19,702 | 0.5% | 2.09pp | 0.024 vs 0.004 |
| 30 | 7 (immature) | 6,371 | 0.6% | 5.22pp | 0.039 vs 0.004 |

The h=14 overstatement persisted (0.032 vs 0.007 → 0.024 vs 0.004), and the 0.1–0.2 bin there
reads 0.131 vs 0.047 (n=321).

## The pooled panel hides three regimes

The panel above pools every served date, so it mixes models and market regimes. Split by forecast
date (era A before 09-06, B 09-06..09-19, C from 09-22, after the 09-20 retrain; 09-20 and 09-21
are omitted as the changeover), mean `exceed_p`
against the realised exceedance rate:

| h | era | dates | rows | events | realised | mean p | p / realised |
|---|---|---|---|---|---|---|---|
| 3 | A | 3 | 2,777 | 29 | 1.04% | 0.88% | 0.8 |
| 3 | B | 12 | 10,722 | 3 | 0.03% | 0.49% | 17.6 |
| 3 | C | 13 | 12,027 | 38 | 0.32% | 0.81% | 2.6 |
| 7 | A | 5 | 4,584 | 26 | 0.57% | 1.78% | 3.1 |
| 7 | B | 12 | 10,722 | 19 | 0.18% | 1.10% | 6.2 |
| 7 | C | 9 | 7,839 | 15 | 0.19% | 1.47% | 7.7 |
| 14 | A | 6 | 5,466 | 47 | 0.86% | 3.56% | 4.1 |
| 14 | B | 12 | 10,722 | 42 | 0.39% | 2.11% | 5.4 |
| 14 | C | 2 | 1,754 | 2 | 0.11% | 2.47% | 21.7 |

Three things follow.

1. **It is not an old-model artifact.** The post-retrain era (C) is still 2.6× over at h=3 and
   7.7× over at h=7.
2. **The realised base rate moves far more than the prediction does.** At h=3 it runs 1.04% →
   0.03% → 0.32% across eras (a 35× range, with era A on only 3 dates) while mean `exceed_p` stays
   between 0.5% and 0.9%. The head carries little date-level regime information, which is the same wall the band hit
   (`2026-08-12-the-band-is-tilted-in-sigma.md`). Era A at h=3 is **calibrated** (0.8×), so a
   scalar shrink fitted on the calm eras would then under-state the next volatile window.
3. **h=7 is robustly overstated.** Even era A, the most volatile h=7 window, is 3.1× over. With
   Poisson 95% intervals on the event counts (optimistic, since events cluster by date), the
   realised rate is 0.37–0.83% there against a mean p of 1.78%, so the ratio is at least ~2×.

Event counts are small (2 to 47 per cell). h=3 era B (3 events) and h=14 era C (2 events) are
shown but not leaned on. The cells are read as ratios with intervals, not as a fit.

## Decision

- **No served `exceed_p` refit.** The 09-13 note named it, but the era split shows a level
  correction would trade one regime's error for the other's, on a handful of events per window. It
  would need a regime signal the head does not have.
- **Open product question, not changed here:** `move_odds` is published at h=3 and h=7
  (`api/volatility_tags.py::CALIBRATED_MOVE_ODDS_HORIZONS`) on an absolute ECE bar of about 1.3pp.
  With a 0.2–0.3% base rate that bar passes a bulk bin that is several times too high (h=7 ECE is
  1.11pp). Suppressing h=7 (`(3,)`), or adding a relative-error test beside the ECE bar, is the
  lever. Suppressing h=14 and h=30 was right: h=14's ECE went from ~1.7pp in the replay to 2.09pp
  served.
- Re-read `--served-only` split by era when h=30 reaches 20 dates (November).
