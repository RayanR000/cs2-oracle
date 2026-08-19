# Drop the replayed 2025-12-01 cohort; h=30 has no valid published figure

`2025-12-01` is a **replay**, not a live forecast: `item_forecasts.created_at` is 2026-07-17, and
its stored `current_price` matches the **2026-07-18** price frame to a median |log ratio| of
**0.016**, against **0.244** for its own nominal date — so its dollar coverage (0.19–0.35) is an
artifact of the rebasing, and its calibrated coverage looks fine only because the rebasing divides
the contamination out. Critically, it is the **only h=30 date in the panel**, so every published
h=30 accuracy / coverage / actionable figure rested on this one contaminated cohort.
Deep-model-review §3.

## Change

- `backtest/scoring.py`: `2025-12-01` added to `EXCLUDED_FORECAST_DATES` with a mechanism reason.
  `excluded_forecast_date` is applied in both published paths — `scripts/backtest_accuracy.py`
  and `scripts/papertrade_report.py` — so the cohort is dropped from every published figure (all
  horizons of that date, since the whole cohort's quote is contaminated).
- Effect: h=30 now has **zero clean forecast dates**, so it falls below `MIN_FORECAST_DATES` and
  reports `insufficient_dates` — the honest state. That is the banner at the data level.
- `.claude/rules/backtest-scoring.md`: a banner that any pre-2026-08-19 h=30 headline in the docs
  is void, and h=30 is not quotable until fresh dates accumulate.
- `tests/test_excluded_forecast_dates.py`: covers the new exclusion; the deliberate-size guard is
  now 2.

## Not done (separate follow-up)

Review §3 also calls for making a **stale anchor refuse to serve** rather than disclose
(`forecast_prices.py` `anchor_date → today` fallback; ~34% of served rows still carry a dirty
anchor). That is a serving-path behaviour change and is left as its own task.
