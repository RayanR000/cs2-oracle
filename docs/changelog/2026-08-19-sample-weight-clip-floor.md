# Sample weights were degenerate — fix the clip floor and the NaN fill

`_compute_sample_weights` weights rows by the 30-day rolling std of daily returns — a
**fraction**, median ~0.0398 on the >=$1 cohort — then clipped it to `[0.1, p99]` and filled
no-history rows with `1.0`. Measured on the archive (n=313,753, deep-model-review §6b):

- the `0.1` floor sat **2.5x above the median**, so **89.2%** of rows received one identical
  weight;
- `.fillna(1.0)` gave the **1.5%** of rows with too little history (`<5` obs) the maximum
  weight after clipping — ~3.9x the majority.

Both are the inverse of the docstring's "movers up, flat/dead down", and the volatility
weighting production believes it ships was effectively never active. `DIRECTION_UPWEIGHT` and
the recency decay both multiply this degenerate base.

## Change (`models/forecaster.py`)

- Winsorize to `[p1, p99]` instead of a fixed `[0.1, p99]`, so the low tail is data-driven
  and symmetric with the existing high cap. Restores the spread across 98% of rows.
- Fill no-history rows with the **median** volatility (neutral) instead of `1.0`, so a row
  with unknown volatility is no longer maximally up-weighted.
- Docstring updated to name the fraction scale and the failure it caused.

## Effect

Verified on a synthetic flat/volatile/no-history frame (recency decay and `DIRECTION_UPWEIGHT`
neutralised): rows at the min weight drop from ~89% to ~1%, movers outweigh flat items
(~4.4x vs ~0.16x), and no-history rows sit near the median rather than the max. Regression
tests added in `tests/test_forecaster.py::TestSampleWeights`.

Takes effect on the next **retrain** (weights are computed at fit time; nothing cached). No
artifact-schema or `VOTED_CACHE_VERSION` change. A/B-able via the model-diagnostics harness,
but the current base is degenerate, so this only restores intended behaviour.
