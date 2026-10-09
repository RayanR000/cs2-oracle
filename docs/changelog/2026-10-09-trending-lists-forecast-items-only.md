# 2026-10-09 — `/items/trending` lists forecast items only

## What was wrong (measured on prod, read-only, 2026-10-09)

`_build_trending` outer-joined listed items to their latest fresh h=7 forecast and ordered by
`price_mid / current_price DESC`. Postgres sorts NULLs first under `DESC`, so the 1,805 of 2,376
listed items with no fresh forecast came before every real one. The prod top 10 was nine Sealed
Graffiti and a Music Kit, all with a NULL ratio. SQLite sorts NULLs last, which is why the tests
never caught it.

The route also priced each row from `price_history`, which the aggregator no longer writes. On
prod it has 7,139 rows and stops at 2026-07-11, so prices were three months stale, and a row with
no price was dropped after the `LIMIT`, which could return fewer rows than asked for.

## Change

- Inner join: an item with no fresh h=7 forecast is not trending. `desc(ratio).nulls_last()`
  still guards a NULL `price_mid`, and `Item.id` breaks ties so the order is deterministic.
- `latest_price` is the forecast's `current_price`, the quote the ratio was computed from. The
  price floor already holds it at ≥ $1, so no post-limit filter is needed.
- `_latest_prices` is no longer called by this route. It stays, with its tests.

After the change, the same read-only prod probe returns ten forecast items (FAMAS | Survivor Z
(FN), Dual Berettas | Oil Change (FN), …).

## Tests

`test_trending_ranking.py::TestTrendingRows` runs the route on a SQLite session with no
`price_history` rows. It checks that items without a forecast are excluded, that rows are ranked
by the predicted ratio, and that they are priced from the forecast. Both tests fail on the old
code.

## Not changed

`/trending` is a ranked surface but does not apply the clean-anchor gate that `/opportunities`
does (`.claude/rules/serving-policy.md`). Adding it is a serving-policy decision, left open.
