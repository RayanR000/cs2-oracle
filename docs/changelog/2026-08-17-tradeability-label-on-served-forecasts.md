# Tradeability label on served forecasts (`tradeable`, `est_roundtrip_cost_pct`)

**2026-08-17.** `PredictionOut` now carries `tradeable: bool` and
`est_roundtrip_cost_pct: float`, derived from the forecast's own `current_price`.
Sub-$1 items are served a forecast as before, now **flagged not economically
tradeable** rather than presented without context.

## Why

Product call (`docs/research/2026-08-16-next-steps.md` item 1): serve the full
catalogue but be honest that a $0.09 forecast cannot be traded. The sub-$1 tier
sits on a book with a **35.5% median spread** and a Steam fee climbing past 60%
at the cheapest prices (`references/cs2-market-domain.md` §1–4), so any
few-percent forecast edge is dead there.

## What the investigation corrected

The premise that the daily job *withholds* forecasts for sub-$1 items was
**false**. `predict()` (`models/forecaster.py:7775-7784`) fetches every
backfilled item and filters only on history depth (`PREDICT_MIN_HISTORY_DAYS`),
not price; the writer does not filter by price; `GET /{item_id}/prediction` has
no floor. Production already serves ~8,691 items (median $0.09). The
`MIN_SERVED_PRICE_USD = 1.0` floor governs **only** the ranked `/opportunities`
and `/trending` surfaces and the headline accuracy tier — never per-item
serving. So covering sub-$1 items needed **no serving change**; the only real
gap was the label. The `predict()`/floor changes originally scoped were dropped
as no-ops.

## What changed (schema-only, one derivation point)

- **`api/serving_policy.py`**: `tradeability(price) -> Tradeability(tradeable,
  est_roundtrip_cost_pct)`. Cost is `friction.actionable_threshold(price_tier)`
  (cheapest-venue round trip + tier spread) × 100. `tradeable` is the same $1
  line as the ranking/headline floor. Module docstring corrected to state the
  floor is a **ranking floor, not a serving floor**.
- **`api/schemas.py`**: two fields on `PredictionOut`, filled by a
  `model_validator(mode="after")` from `current_price` — so all three route
  construction sites (and any future one) are correct by construction. No route
  edits. `serving_policy` is imported inside the validator to avoid an
  import-time cycle.
- **Tests**: `tests/test_serving_policy.py::TestTradeability` (6) and
  `tests/test_prediction_tradeability.py` (3). Serving/schema-adjacent suites
  green (51).

## Scope / caveats

- **Not retrained, floor untouched.** `HEADLINE_MIN_TIER` and
  `MIN_SERVED_PRICE_USD` are unchanged; `/opportunities` and `/trending` stay
  ≥$1; no coverage number is quoted over the sub-$1 tier.
- **Sub-$1 bands are extrapolation.** `q_hat` is calibrated on ≥$1 residuals, so
  the 80% coverage claim does not transfer to sub-$1 — the `tradeable=False`
  flag is what keeps that honest.
- `tradeable` is the $1 cutoff, not a per-forecast edge-vs-cost test. The cost
  estimate is served alongside so a consumer can apply its own hurdle.
