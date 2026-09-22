# NAIVE_INIT_SCORE refuted on post-CSMarketAPI data (2026-09-22)

## What

Re-measured `NAIVE_INIT_SCORE=1` (boost the q50 from `-return_1d`) on the
post-CSMarketAPI training data (11.9M rows, 20 sources). The flag is worse at
all four horizons on the clean-anchor basis — the last untested ML lever is closed.

## Results (clean anchor rank IC, local paired retrain)

| Horizon | Control | Treatment | Δ |
|---------|---------|-----------|---|
| h=3  | 0.1251 | 0.0984 | **−0.027** |
| h=7  | 0.1366 | 0.1247 | **−0.012** |
| h=14 | 0.1419 | 0.1408 | **−0.001** |
| h=30 | 0.0926 | 0.0730 | **−0.020** |

The 30d treatment arm also triggered the "ML subtracts from its own best feature"
warning (rank IC 0.1258 vs naive 0.126). The control's warning was already gone
at all horizons — the multi-source data resolved the naive-beats-model gap that
motivated this instrument.

## Prior measurement

The 08-10 read (single-source era) was "10 of 16 coin flip" — mixed at 3/14d,
negative at 7/30d. The multi-source training data turned it from ambiguous to
clearly negative: the model no longer needs the offset because the expanded
label set already solved the noise problem `-return_1d` was compensating for.

## Decision

`NAIVE_INIT_SCORE` stays off. No further init_score variants to test — the
feature set is now mature on the post-CSMarketAPI data.
