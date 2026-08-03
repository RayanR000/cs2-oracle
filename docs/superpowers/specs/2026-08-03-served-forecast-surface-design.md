# Served Forecast Surface — Design

**Date:** 2026-08-03
**Status:** Approved, not yet implemented
**Scope:** `backend/api/routes/opportunities.py`, `backend/api/routes/items.py`, a
new `backend/api/serving_policy.py`, and `.github/workflows/price-forecast.yml`.
No change to `backend/models/forecaster.py`, to features, or to training.

## Problem

The goal is to raise the accuracy of the forecasts the product actually shows,
without spending training time. The forecasts themselves are not the immediate
problem — the selection and ranking applied on top of them is. Three defects, and
one measurement gap that blocks fixing anything else by experiment.

### 1. The `confidence` flag is anti-predictive

Directional accuracy split by served confidence, computed **within each forecast
date** so that both groups faced the same market:

| Forecast date | Horizon | Model | n_high | acc high | acc low | gap |
|---|---|---|---|---|---|---|
| 2025-12-01 | 7d | `lgbm-v3` | 1,717 | 25.0% | 48.4% | −23.4pp |
| 2025-12-01 | 3d | `lgbm-v3` | 54 | 29.6% | 46.6% | −17.0pp |
| 2025-12-01 | 14d | `lgbm-v3` | 77 | 3.9% | 42.7% | −38.8pp |
| 2025-12-01 | 30d | `lgbm-v3` | 65 | 4.6% | 42.8% | −38.2pp |
| 2026-07-17 | 7d | `lgbm-v3` | 109 | 21.1% | 55.3% | −34.2pp |
| 2026-07-17 | 14d | `lgbm-v3` | 84 | 23.8% | 38.4% | −14.6pp |
| 2026-07-18 | 7d | `lgbm-v3-global-only` | 64 | 7.8% | 25.5% | −17.7pp |
| 2026-07-19 | 7d | `lgbm-v3-regime` | 132 | 28.8% | 36.3% | −7.5pp |
| 2026-07-19 | 3d | `lgbm-v3-regime` | 428 | 31.1% | 31.7% | −0.6pp |

Source: `price-archive/ops/forecast_outcomes.parquet` joined to
`item_forecasts.parquet` on `(item_id, forecast_date, horizon_days)`; all cells
with `n_high >= 30`.

Nine cells, four dates, three model versions, all four horizons, every gap
negative. `_calibrate_confidence` (`forecaster.py:4099`) targets
`CONFIDENCE_TARGET_ACCURACY = 80.0`; realized high-confidence accuracy is 4–31%.

**Why this contrast is trustworthy while most others in this dataset are not:**
directional outcomes cluster by `forecast_date` — every item forecast on one day
shares one market-wide move — which is why `scoring.py:MIN_FORECAST_DATES` exists.
A high-vs-low comparison inside a single date is paired against that shared move,
so it survives the clustering that invalidates unpaired comparisons. Any claim of
the form "the model beats/loses to always-predicting-down" is *not* supportable
from this data and is deliberately absent from this document.

`opportunities.py` gates on this flag in four query filters (lines 124, 136, 172,
184) and branches on it when assigning `opportunity_type` (lines 95–97);
`items.py:114-124` sorts by it descending. The product surfaces the
worst-performing subset and ranks it first.

### 2. The surfaces have no price floor

`/undervalued` selects 7d + `direction == "up"` + `confidence == "high"` and orders
by `desc((price_mid - current_price) / current_price)`. On the newest forecast date
that is 1,680 candidates with a **median price of $0.50**. No `current_price`
threshold exists anywhere in `opportunities.py` beyond `> 0`.

`scoring.py:209-213` already states why percentage-ranking this universe fails:
sub-$1 items are 72% of it and *"one cent there is a 20% move, so its up/flat/down
label is dominated by tick quantisation."* Sorting by percentage move therefore
promotes rounding artifacts to the top of the list.

This also means the quoted headline accuracy and the displayed list describe
different populations: `HEADLINE_MIN_TIER = 1` computes the headline over ≥$1
items only, while the surfaces are ~84% below $1.

### 3. `flat` is a penny-item marker, not a forecast

On the newest forecast date at 7d, `flat` is emitted 2,337 times and every
instance is a $0.02–$0.04 item (`max_price = $0.03` for the 1,941 high-confidence
flats). Zero `flat` forecasts exist at ≥$1. Separately, `confidence == "medium"`
has **never** been emitted — 0 rows in the entire history — yet `items.py:115-117`
ranks it as a distinct tier.

### 4. Five forecast dates exist, ever

`item_forecasts.parquet` holds 123,436 rows across **5 distinct forecast dates**:
2025-12-01, 2026-07-17, 07-18, 07-19, 07-29 — four of them manual local runs.
`scoring.py:MIN_FORECAST_DATES = 20` is the repo's own stated threshold below
which "a cohort cannot separate model skill from the market's direction."

Two consequences. Any change to a serving rule is currently **undecidable** — the
same clustering that closed the model roadmap applies to the serving instrument,
except here it is cheap to fix, because predict-only days do not retrain.
And `items.py:108` filters `forecast_date == today`, so on the large majority of
days — 5 dates written across the eight months since 2025-12-01 — the trending
surface silently degrades to no forecast at all.

## Relation to the closed accuracy roadmap

`docs/research/accuracy-opportunities.md` is closed and stays closed. Nothing here
touches features, architecture, hyperparameters, or the `ab_test_*.py` harness whose
noise floor closed it. This is serving-layer correctness — the same category as
`docs/changelog/2026-08-01-deterministic-backtest.md`, which that closure explicitly
redirects effort toward.

## Design

### Component 1 — Match the served universe to the measured universe

New module `backend/api/serving_policy.py` holding the single serving threshold:

```python
MIN_SERVED_PRICE_USD = 1.0
```

with a comment recording its provenance (the tick-quantization argument in
`scoring.py:209-213`) and a helper returning the SQLAlchemy clause so the four
call sites cannot drift apart:

```python
def price_floor_clause(column):
    return column >= MIN_SERVED_PRICE_USD
```

A test asserts `price_tier(MIN_SERVED_PRICE_USD) == HEADLINE_MIN_TIER`, which fails
if either constant moves independently — the floor and the headline must keep
describing the same population.

Applied in `_latest_forecasts`, `/undervalued`, `/overheated`, and the `items.py`
forecast surfaces, against `ItemForecast.current_price`.

**Coverage cost**, 7d on the newest forecast date: 1,402 of 8,691 forecasts survive
the floor (16.1%). `/undervalued`'s candidate pool goes 1,680 → 599 from the floor
alone.

### Component 2 — Stop gating and ranking on `confidence`

- Remove `ItemForecast.confidence == "high"` from the four `opportunities.py`
  filters.
- Remove `confidence_order` from the `items.py:124` sort, leaving the predicted-
  return ordering. Delete the dead `"medium"` branch.
- `_reason_for_type` and `_build_trend_explanation` stop asserting confidence —
  the strings "with high confidence" and "Confidence is {confidence}" become
  false once the flag no longer gates anything, and they were never true.

Combined with Component 1, `/undervalued`'s pool becomes 893 and `/overheated`'s
509 on the newest date — larger than the 599 the floor alone leaves, because
dropping the gate returns more items than the floor removes.

**No replacement confidence signal is introduced.** The evidence that the current
flag is inverted is paired and consistent; there is no evidence supporting any
specific alternative, and installing an unvalidated selector is what produced
this defect. `_calibrate_confidence` and the stored column are left in place and
keep being scored, so Component 3 can diagnose them on real data later.

### Component 3 — Make the serving layer measurable

`price-forecast.yml` chains off `Aggregator Market Update` (daily, 23:00 UTC) and
runs `predict-only` on non-Mondays, so the schedule is already correct. What is
missing is any verification that a run actually *added* a forecast date — the same
silent-success shape as the collectors in `collectors-fail-silently`.

Add a post-run step that queries `item_forecasts` for its newest `forecast_date`
and fails when that is not today, so a run that persists nothing reports failure
instead of success. Place it after the forecast step, unconditional on that step's
outcome, so a partial write is still caught.

This is the component that unblocks everything else: at one date per day the
existing `block_bootstrap_ci` / `MIN_FORECAST_DATES` machinery reaches its own
threshold in ~3 weeks, at which point serving-rule changes become decidable by
experiment rather than argument.

## Testing

- **Component 1:** the tier-coupling assertion above; route tests asserting no
  sub-$1 item appears in `/opportunities`, `/undervalued`, `/overheated` given a
  fixture containing both.
- **Component 2:** route tests asserting a low-confidence item is eligible for
  `/undervalued`, and that ordering follows predicted return rather than
  confidence. A test asserting no response text claims a confidence level.
- **Component 3:** a test that the freshness check fails on a stale
  `item_forecasts` and passes on a fresh one.

## Risks

- **The floor is a judgement call, not a derivation.** $1 is chosen to match
  `HEADLINE_MIN_TIER`, not because $1 is where tick noise stops mattering; the
  tier table in `docs/superpowers/specs/2026-08-01-deterministic-backtest-design.md`
  puts the sharp break at $0.50 (27.6% actual-flat below it, ~1% above).
  Coupling it to the
  headline tier is the point — the number shown and the number quoted should
  describe one population — but it is a convention.
- **Component 2 is a deletion justified by evidence for the harm, not evidence for
  the alternative.** Ranking by predicted return magnitude is not established as
  good; it is established as not-inverted. It is what `opportunity_score` already
  reports today.
- **Losing 84% of surface coverage is visible to users.** Accepted deliberately:
  the removed items are the ones whose direction labels are quantization
  artifacts.
- **The confidence contrast rests on four dates.** Pairing within date defends it
  against market drift but not against the possibility that all four dates are
  unrepresentative. Component 3 is what would eventually settle that.

## Follow-ups, gated on Component 3

Not in scope now; each needs ≥20 forecast dates to decide:

1. Whether a predicted % change can be shown as a number at all.
2. Whether `flat` should be reachable for ≥$1 items, or the served label should be
   binary up/down.
3. Why `_calibrate_confidence` ships a rule realizing 4–31% against an 80% target,
   and whether a corrected version beats no selection at all.
4. Whether p10–p90 intervals are calibrated (one look suggests 35–62% coverage
   against an 80% target; two dates, not reportable).

## Related

- `docs/changelog/2026-08-01-deterministic-backtest.md` — the fix that made served
  accuracy trustworthy, and the three open data follow-ups
- `docs/research/accuracy-opportunities.md` — the closed model roadmap
- `backend/backtest/scoring.py` — `MIN_FORECAST_DATES`, `HEADLINE_MIN_TIER`, and
  the clustering argument this document leans on
