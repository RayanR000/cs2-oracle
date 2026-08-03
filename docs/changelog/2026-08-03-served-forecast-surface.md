# Served Forecast Surface: Confidence Gate Removed, Price Floor Added (2026-08-03)

## What changed

Serving layer only. No features, no architecture, no training changes; the
accuracy roadmap stays closed.

1. **The four `confidence == "high"` gates in `opportunities.py` are gone**
   (the `subq` and outer filters of `/undervalued` and `/overheated`), as is the
   `confidence` branch that assigned `opportunity_type`. Direction alone now
   decides the label, via `opportunity_type_for`.
2. **`items.py:_build_trending` no longer sorts by confidence.** The
   `confidence_order` `case` expression is deleted; the list orders by predicted
   return alone.
3. **`MIN_SERVED_PRICE_USD = 1.0` now floors every ranked surface** —
   `/opportunities`, `/undervalued`, `/overheated`, `/momentum`, and the forecast
   join behind the trending list. It lives in the new
   `backend/api/serving_policy.py` alongside `price_floor_clause` and
   `meets_price_floor`, so the five call sites cannot drift.
4. **Copy no longer asserts a confidence level.** `_build_trend_explanation` lost
   its `confidence` parameter and the "Confidence is {x}." sentence;
   `_reason_for_type` lost "with high confidence"; the item page's Trend metric
   card lost its `sub={`Confidence ${confidence}`}` badge.
5. **`price-forecast.yml` fails when a run persists nothing.** A new
   `Verify forecasts were persisted` step runs
   `backend/scripts/check_forecast_freshness.py`, which asserts both
   `item_forecasts` and its Parquet mirror carry a forecast dated today or later.

## Why

**The confidence flag is anti-predictive.** Directional accuracy split by served
confidence, computed within each forecast date so both groups faced the same
market, was negative in all nine `date x horizon x model` cells with
`n_high >= 30`: high confidence scored 4–31% against low confidence 25–55%, gaps
from −0.6pp to −38.8pp. `_calibrate_confidence` targets 80%. Four dates, three
model versions, all four horizons, every gap negative. The surface gated on that
flag in four queries and ranked by it, so it selected the worst-performing subset
and put it first.

The within-date pairing is what makes this contrast usable when most others in
this dataset are not: directional outcomes cluster by `forecast_date`, and a
high-vs-low comparison inside a single date is paired against that shared
market-wide move. No claim of the form "the model beats always-predicting-down"
is made here — that one is not supportable from this data.

**The served population was not the measured population.** The headline accuracy
figure is computed over `price_tier >= HEADLINE_MIN_TIER` (≥$1) only, while the
ranked surfaces had no floor beyond `current_price > 0` — 84% of what they ranked
was sub-$1, where one cent is a 20% move and the up/flat/down label is dominated
by tick quantisation. `/undervalued` offered 1,680 candidates at a median price of
$0.50, sorted by percentage move. The number quoted and the list displayed
described different populations.

## Coverage effect

Visible to users, accepted deliberately — the removed items are the ones whose
direction labels are quantisation artifacts. On 7d, forecast date 2026-07-29:

| Surface | Candidate pool before | After |
|---|---|---|
| `/undervalued` | 1,680 | 893 |
| `/overheated` | — | 509 |

1,402 of 8,691 forecasts clear the floor.

## What was deliberately not done

- **No replacement confidence signal.** This is a deletion justified by evidence
  for the harm, not evidence for the alternative. Ranking by predicted-return
  magnitude is not established as good; it is established as not-inverted, and it
  is what `opportunity_score` already reported.
- **`_calibrate_confidence`, the `confidence` column, and `conf_gap_pp` scoring
  are all retained.** The flag keeps being computed and scored so it stays
  diagnosable; only its use as a filter or sort key is removed.
- **`confidence` stays on every response schema.** `TrendAnalysisOut.confidence`
  and `PredictionOut.confidence` are consumed by the frontend; dropping a
  response field is a breaking change and was out of scope.
- **No floor on per-item detail endpoints.** A user who opens a $0.30 item still
  sees its forecast. The floor governs what the product *chooses* to show, not
  what it shows on request — so it was applied to the trending list's forecast
  join but not to `/items/{id}/trends` or the prediction detail endpoints.

## Verification

`cd backend && python3 -m pytest tests/ -q` → **395 passed, 0 failed** (186s),
against a pre-change baseline of 355 passed. The 40 new tests are 7 in
`test_serving_policy.py`, 13 in `test_opportunity_selection.py`, 3 in
`test_trending_ranking.py`, 7 in `test_trend_explanation_copy.py`, 10 in
`test_forecast_freshness.py`. No previously-passing test broke.

Route tests are unit-level by necessity: these endpoints use PostgreSQL
`DISTINCT ON`, which SQLite cannot compile, and the repo has no API fixtures. The
in-Python selection was extracted into pure functions testable with plain
objects, with source-level guards (`test_module_source_never_filters_on_confidence`,
`test_confidence_order_is_gone`) covering the SQL filters that cannot be
exercised.

`frontend`: `npm run build` compiles and typechecks clean.

The freshness script was run against real local stores both ways: exit 0 for
`--expected-date 2026-07-29` (DB at 2026-08-02, Parquet at 2026-07-29), and exit
1 with a per-store message for `2026-08-03`. Both legs read correctly and the
failure path fires.

## Still open

The four follow-ups from the design spec, all gated on reaching
`MIN_FORECAST_DATES = 20` — which the freshness check above is what makes
reachable, in roughly three weeks at one forecast date per day:

1. Whether a predicted % change can be shown as a number at all.
2. Whether `flat` should be reachable for ≥$1 items, or the served label should
   be binary up/down.
3. Why `_calibrate_confidence` ships a rule realizing 4–31% against an 80%
   target, and whether a corrected version beats no selection at all.
4. Whether p10–p90 intervals are calibrated (one look suggests 35–62% coverage
   against an 80% target; two dates, not reportable).

Also unresolved: the $1 floor is a convention, not a derivation. The sharp break
in the tier evidence is nearer $0.50 (27.6% actual-flat below it against ~1%
above). Matching `HEADLINE_MIN_TIER` is what earns the headline number its
meaning, and `test_serving_policy.py` fails if the two ever diverge.

## Related

- `docs/superpowers/specs/2026-08-03-served-forecast-surface-design.md` — the design
- `docs/superpowers/plans/2026-08-03-served-forecast-surface.md` — the plan
- `docs/changelog/2026-07-31-accuracy-work-closed.md` — the collector audit whose
  silent-success shape `check_forecast_freshness.py` is written against
- `docs/changelog/2026-08-01-deterministic-backtest.md` — the fix that made served
  accuracy trustworthy enough to argue from
