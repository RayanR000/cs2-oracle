---
paths:
  - "backend/api/**"
---

# Serving policy

**`MIN_SERVED_PRICE_USD = 1.0` is a convention, not a derivation.** `api/serving_policy.py`
sets the floor deliberately equal to the lower bound of `HEADLINE_MIN_TIER` so the
population the product shows is the population the headline accuracy number describes.
`tests/test_serving_policy.py` fails if the two diverge. Sub-$1 items are ~72% of the
forecast universe and one cent there is a 20% move.

⚠️ **It is a RANKING floor, not a serving floor**, despite the name. `predict` generates and
serves a forecast for every eligible item at any price (it filters only on history depth); the
floor gates the ranked `/opportunities` and `/trending` surfaces and the headline tier. Served
sub-$1 rows instead carry `tradeability(price)` → `(tradeable, est_roundtrip_cost_pct)`, the
round trip at the cheapest venue plus the item's tier spread
(`backtest/friction.py::actionable_threshold`), so the forecast is honest about the fee wall
rather than withheld. Do not describe sub-$1 items as unserved.

**`move_odds` is publishable at h=3 and h=7 only.** `api/volatility_tags.py` is the pure-function
layer behind `/volatility`: `swing_pct` = half-band / mid, a Stable/Moderate/Volatile
`stability_label` from the within-horizon swing tertiles, and `move_odds` = the served
`exceed_p`. `CALIBRATED_MOVE_ODDS_HORIZONS = (3, 7)` — replay ECE is <1.3pp at h3/h7, ~1.7pp at
h14 and ~3.8pp at h30, so `move_odds` is **suppressed (None)** elsewhere and a
`sort="move_odds"` request at another horizon is rejected rather than served. Same rule as
`swing_pct`: never fabricate a number the data does not support.

**The clean-anchor gate keeps the deviating cohort off RANKED surfaces only.**
`meets_anchor_gate` / `anchor_clean_clause` (same module) filter `/opportunities`;
per-item lookups still serve their forecast and carry `anchor_clean` /
`anchor_wedge_pct` on `PredictionOut`. The evidence is an ordering statistic — served
rank IC **+0.1321 / +0.1562 / +0.1747** at 3/7/14d where the anchor quote equals its own
local median, 4 CI anchors of 4, against **−0.2014** (0 of 4) at h=3 and no
distinguishable signal at the other horizons on the rest
(`docs/changelog/2026-08-11-clean-anchor-confirmed-in-ci.md`) — so it licenses gating a
ranking, not refusing a forecast asked for by name. Four things not to get wrong:

- **NULL passes.** It means "not recorded" — all 310,304 rows written before 2026-08-11,
  and any run against an un-migrated table. `anchor_clean_clause` spells the `IS NULL`
  out because SQL three-valued logic would otherwise drop them silently and empty the
  ranked surfaces the moment the column landed.
- **It is NOT `ANCHOR_OUTLIER_TOLERANCE`.** The gate is exact equality
  (`isclose(rtol=0, atol=1e-9)`), matching `replay_serving._tied_mask`, which is the
  split every published figure came from. The 10% outlier test calls ~95% of items clean
  and nothing was measured on it.
- **The flag is computed BEFORE `_serving_base_price`** (`_anchor_disclosure`). That
  method overwrites the raw quote with the served base, which under the shipped arm *is*
  the smoothed median — computed after, every item reads as clean.
- **Gating is not an accuracy change.** It stops publishing a ranking over a cohort with
  no measured ordering skill. The model is unchanged.

Migration `0022`; `scripts/forecast_prices.py` drops the columns from the DB payload
with a WARNING if the table lacks them, so an un-migrated schema degrades instead of
failing the daily run. The same guard now covers three columns — `anchor_clean`,
`anchor_wedge_pct` and `exceed_p` (migration `0024`, the served exceedance probability behind
the `/volatility` `move_odds` / `stability_label` tags).
