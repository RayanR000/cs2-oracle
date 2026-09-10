# Served direction withheld at every horizon

**2026-09-10.** `trend_direction` now reports `"neutral"` on every API surface and
at every horizon. The `direction` column is still written by the forecaster and
still scored — this is a disclosure change, not a modelling one.

## Why

Measured on the first backtest panel after the leg-window resolver fix
(`2a0ace3`, run 34427386566 — the three preceding runs were red, so this is the
first clean directional read in a while). Pesaran-Timmermann on the `>=$1`
`lgbm-v3%` cohort:

| h | PT (≥$1) | t | p | dates | DA | constant-"down" |
|---|---|---|---|---|---|---|
| 3 | **−1.62pp** | −2.34 | **0.032** | 17 | 43.8% [39.5–47.9] | 55.0% |
| 7 | −0.66pp | −1.34 | 0.193 | 23 | 48.2% [44.3–52.2] | 58.1% |
| 14 | −0.54pp | −0.51 | 0.617 | 23 | 48.9% [45.9–51.8] | 65.2% |
| 30 | −1.31pp | −1.78 | 0.118 | 8 | 45.3% [36.0–56.5] | — |

Three things make this a withdrawal rather than a "wait for more dates":

1. **PT is negative at every horizon and every price floor.** Nothing in the
   panel is positive. The floor sweeps do not rescue it — h=7 gets *worse* with
   size (−0.66 at ≥$1 → −1.17 at ≥$5 → **−1.89pp p=0.037** at ≥$20), which is
   the opposite of the tick-quantisation story that would excuse a weak reading.
2. **DA is below a coin flip at every horizon** (43.8–48.9%), and a constant
   "down" call beats the model everywhere. The call is anti-informative, not
   merely uninformative — and consumers act on it, so shipping it is worse than
   shipping nothing. Same reasoning that withdrew `confidence` on 2026-08-12.
3. **The centre verdict was already converged.** `lambda* = 0.00` has now
   reproduced four times across three different labels and harnesses
   (`centre-shrinkage-lambda-is-zero`, `composite-centre-ranks-cannot-scale`,
   the powered 2025 test). The direction call is a sign read off that centre.

Withholding is also the asymmetric-safe move: h=3 and h=30 are still
`NO HEADLINE` (17 and 8 usable dates against `MIN_FORECAST_DATES=20`), and while
that bars *publishing a skill claim*, it does not bar *declining to publish a
call*. No date count is required to stop asserting something.

## What changed

- `api/serving_policy.py`: `DIRECTION_DISCLOSED = False` and
  `served_direction(raw, horizon=None)`, the single choke point. Deliberately
  **not** a per-horizon tuple like `ANOMALY_SERVED_HORIZONS`: no horizon earned
  disclosure, so there is no partial gate to drift out of sync with evidence.
- `api/routes/items.py`: all five `trend_direction=` sites route through it; the
  two `direction_map` literals are gone, as are the now-unreachable
  `factors.append("Forecast predicts upward movement")` branches, which would
  otherwise have leaked the withheld call through `factors`.
- `_build_trend_explanation` keeps its bullish/bearish branches on purpose —
  pinned by `test_trend_explanation_copy`, and live again the moment
  `DIRECTION_DISCLOSED` flips. It is simply never fed a non-neutral value now.
- `api/schemas.py`: both `trend_direction` fields document the withholding. The
  field stays `str` on the schema — the frontend consumes it, and removing a
  field is a breaking change.

## Reopening

Flip `DIRECTION_DISCLOSED` to `True` once a panel with `>= MIN_FORECAST_DATES`
usable dates shows **positive** PT at the horizon in question. If only some
horizons qualify, convert `served_direction` to a horizon tuple at that point.
