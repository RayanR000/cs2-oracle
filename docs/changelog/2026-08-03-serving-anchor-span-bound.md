# Serving's Base Price Now Obeys the Backtest's Staleness Bound (2026-08-03)

`item_forecasts.current_price` and the backtest's archive-resolved `base_price`
are both documented as "the median of the last 3 days", yet they diverge by a
**median 3.6%, 90th percentile 35%, 99th percentile 227%**. Only one of them had
a calendar bound.

## The asymmetry

| | selection | calendar bound |
|---|---|---|
| serving, `predict()` | `df.groupby("item_id").tail(3)` | **none** |
| backtest, `resolve_anchors` | last `SMOOTH_WINDOW` observations | `MAX_WINDOW_SPAN_DAYS` from the anchor |

`tail(3)` takes the last three *rows*, not the last three *days*. For a sparsely
observed item those rows can be months apart, so the "3-day median" became a
median of prices from unrelated periods, stamped as today's value.

Concretely, an item observed at $500 in January and $20 on 17 July anchored at
`median(500, 20) = 260` — a price it never traded at, on a day it was worth $20.
Every dollar figure served for that item was then built from $260.

This is the price-laundering shape `db5bddb` removed from the collector's
historical fallback, and the one `MAX_WINDOW_SPAN_DAYS` bounds on both scoring
legs. Serving was the remaining copy without it.

## The fix

`ItemForecaster._smoothed_anchor_prices(df, anchor)` — the median of the most
recent `SMOOTH_WINDOW` observations lying within `MAX_WINDOW_SPAN_DAYS` of the
anchor. Both constants are imported from `backtest.price_resolution`, which
derives its bound from `collectors.pipeline.FALLBACK_MAX_AGE_DAYS`, so the
codebase keeps **one** staleness convention rather than three.

`test_matches_the_backtest_resolver_on_the_same_observations` pins the parity
directly: given the same observations, serving and `resolve_anchors` must return
the same price.

### Serving degrades, it does not drop

The one deliberate divergence from the backtest. An unresolvable anchor there is
one fewer scored row; here it would be a missing product. An item with nothing
inside the window therefore falls back to its latest observation — no smoothing,
but never a manufactured price and never a silent disappearance.

## What this does and does not fix

**It does not improve directional accuracy.** Worth stating plainly, because the
change was originally scoped on the assumption that it would.

`direction_predicted` is the **directional classifier's** output
(`dir_class_arr`), not a threshold on the anchor, and `_recenter_on_direction`
then adjusts the served median to agree with that call — the price follows the
direction, not the reverse. So the anchor never enters the direction label. It
does enter `predicted_price_{low,mid,high}`, and through them `abs_error`,
`pct_error` and `in_interval`.

That also explains the earlier observation that `direction_predicted` agrees
with `sign(mid − current_price)` at 92% on 30d: the agreement is manufactured by
the recentering, not evidence that direction derives from the anchor.

So this is a **serving-correctness and price-error** fix. It is worth doing on
its own terms — a served price built from a months-stale median is wrong
regardless of what any metric says — but it should not be expected to move the
headline.

## What could not be verified locally

Whether the divergence is *concentrated in sparse items*, which is the
hypothesis the fix rests on, could not be confirmed here: the frozen cohort's
`item_id`s have **zero overlap** with the local `items` table, so `item_id` →
slug could not be resolved to join against the archive. The divergence's shape
(median 3.6%, p99 227%, max 4325%) is consistent with a pathological subset, and
the code asymmetry is direct and readable, but the correlation itself is
unmeasured. Confirming it needs a prod DB connection.

## Tests

`tests/test_serving_anchor_span.py`, 7 new:

- `test_observations_beyond_the_span_are_excluded` — the regression; unbounded
  `tail(3)` returns 100.0 where the bounded window returns 10.0
- `test_a_single_in_window_observation_is_used_unsmoothed`
- `test_item_with_nothing_in_window_falls_back_to_its_latest` — degrade, never
  drop
- `test_only_the_three_most_recent_in_window_observations_vote`
- `test_items_are_smoothed_independently`
- `test_dense_item_is_the_three_day_median` — the common case is unchanged
- `test_matches_the_backtest_resolver_on_the_same_observations`

Two existing tests — `test_predict_skips_items_with_insufficient_history` and
`test_predict_smooths_spike_outlier` — caught a real bug in the first draft: the
predict frame carries `datetime.date` while the archive path carries
`Timestamp`, and the helper compared them directly. Fixed by coercing inside the
helper rather than assuming a dtype. They were not modified.

`VOTED_CACHE_VERSION` is deliberately **not** bumped: the cache key covers
`_fetch_voted_price_history` / `_apply_multi_source_voting`, neither of which
changed. This edit is downstream of the voted frame.

Full suite: 336 pass.

## Files changed

- `backend/models/forecaster.py` — `_smoothed_anchor_prices`; `predict()` uses it
- `backend/tests/test_serving_anchor_span.py` — new

## Related

- `docs/changelog/2026-08-01-deterministic-backtest.md` — `MAX_WINDOW_SPAN_DAYS`
  and the two-estimator bug on the scoring legs
- `db5bddb` — the same shape removed from the collector's historical fallback
