# Friction-conditioned tier scoring — design

**Implements:** step 3 of `docs/research/2026-08-07-next-steps.md` ("Score by price tier,
conditioned on `|predicted move| > round-trip cost`"), sourced from
`docs/research/2026-08-07-cs2-forecasting-research.md` §19–20 and §10 Tier 1 #2.

**Scope decision made up front:** this change reads nothing but columns already frozen on
`forecast_outcomes`. `backtest/scoring.py` stays pure and `--rescore` keeps working with no
archive access — that purity is the reason a scoring fix can land on historical rows at all.
Two fields the review asks for are archive-derived and therefore out of scope here: a
4-quartile staleness axis (needs `stale_run_days`, which step 6 owns) and a per-item BUFF
spread (needs a frozen `buff_spread_rel`; 25 days of bid history would leave it NULL on
almost every stored outcome anyway). Both are named as gaps below rather than approximated.

---

## 1. Six price bands

`price_tier()` gains one cut:

| tier | band |
|---|---|
| 0 | < $1 |
| 1 | $1–5 |
| 2 | $5–20 |
| 3 | $20–100 |
| 4 | **$100–1000** |
| 5 | **≥ $1000** |

**Why:** tier 4 currently merges the 10.8%-spread and 5.2%-spread cohorts — the two most
different liquidity populations in the market (§19, n = 22,449).

**This is a series discontinuity.** Rows stored with `price_tier = 4` before this change mean
`≥ $100`; after it they mean `$100–1000`. Tiers 0–3 keep their meaning, so
`HEADLINE_MIN_TIER = 1`, `MIN_SERVED_PRICE_USD = 1.0` and every `>= HEADLINE_MIN_TIER`
comparison in `models/forecaster.py` are unaffected. `api/serving_policy.py` does **not**
change: raising the serving floor is a decision the floor sweep (§4) informs, and is not part
of this change.

## 2. `backend/backtest/friction.py` — a new pure module

Holds the two friction constants and nothing else. No I/O, no imports from `scoring`.

```python
ROUND_TRIP_COST = {"csfloat": 0.020, "dmarket": 0.020, "skinport": 0.087, "steam": 0.161}
DEFAULT_VENUE = "csfloat"
```

The spread table is the awkward part and must stay honest about it. The measured bands do
**not** align with `price_tier`'s cuts, so each tier borrows its **nearest source band by
geometric midpoint in log price**, and the module records which one it borrowed:

| tier | our band | source band | median spread |
|---|---|---|---:|
| 0 | < $1 | < $1 | 35.5% |
| 1 | $1–5 | $1–10 | 21.1% |
| 2 | $5–20 | $10–50 | 17.3% |
| 3 | $20–100 | $10–50 | 17.3% |
| 4 | $100–1000 | $50–500 | 10.8% |
| 5 | ≥ $1000 | $1000+ | 5.2% |

`actionable_threshold(tier, venue=DEFAULT_VENUE) -> float` returns
`ROUND_TRIP_COST[venue] + SPREAD_BY_TIER[tier]`. At CSFloat that is 37.5% at tier 0 down to
**7.2% at tier 5** — the number that decides whether anything is actionable at all.

The table is a **stated approximation**, not a measurement at these cuts. A per-item
`buff_spread_rel` replaces it when the bid feed has depth; the module says so in its
docstring so nobody later reads these as measured at tier granularity.

## 3. ActionableDA

`ActionableDA(v,h) = P( sign(r_act) = sign(r̂) | |r̂| > RT_v + s_i )`.

**New record fields.** `predicted_mid` and `horizon_days`, added in all three record builders:
`scripts/backtest_accuracy.py::_records_from_frozen_outcomes`, the fresh-resolution path in
the same file, and `backtest/walkforward_records.py::fold_records` (which reconstructs `mid`
already and takes `horizon_days` as a new optional parameter). Both are frozen columns or
group keys — no archive read.

`horizon_days` travels **on the record** rather than as a new `score_cohort` parameter:
records are grouped by `(horizon, model_version)` so every record in a cohort shares it, and
eight test modules call `score_cohort(records)` positionally. Read with `.get()`, so a legacy
record without it degrades to out-of-scope rather than raising.

**Computation**, on the cohort's records:

- `r̂ = (predicted_mid − base_price) / base_price`
- `r_act = (actual_price − base_price) / base_price`
- actionable ⟺ `abs(r̂) > actionable_threshold(price_tier)`
- correct ⟺ `sign(r_act) == sign(r̂)`, using the **raw** sign — not
  `direction_from_return`. A ±0.5% flat band is meaningless on a move required to clear
  ≥7.2%, and a bit-identical carry-forward row has `r_act = 0`, matches no sign, and scores
  as a miss. A frozen price is not a correct call.
- `net_i = sign(r̂) · r_act − threshold`

**Emitted keys**, flat on every scored row:

| key | meaning |
|---|---|
| `actionable_scope` | `in_scope` at h ∈ {14, 30}, else `out_of_scope` |
| `actionable_venue` | `csfloat` |
| `actionable_n`, `actionable_share_pct` | `n_actionable`, and `n_actionable / n_total` |
| `actionable_da` | percent, `None` when `actionable_n == 0` |
| `actionable_e_net_pct` | `mean(net_i)` in percent, `None` when `actionable_n == 0` |
| `actionable_pt_*` | `pesaran_timmermann` on the actionable subset, prefixed |

**Scoped to h ∈ {14, 30}.** An actionable metric at h=3 is a category error. Outside the
scope every value is `None` and `actionable_scope` says why, so "we did not measure this"
never reads as "nothing was actionable" — the same rule the `pt_verdict` states already
follow.

Keys are flat and `None`-filled rather than a nested sub-dict, matching `pt_*`: nested values
serialise to JSON text store-wide (`db/parquet.py::_jsonify_nested`) and every added nesting
level is a parsing surface.

## 4. Floor sweep at $1 / $5 / $20

```python
FLOOR_SWEEP = {-1: 1.0, -2: 5.0, -3: 20.0}   # price_tier sentinel -> floor in USD
```

`HEADLINE_TIER` stays `-1` and keeps meaning `≥ $1`, so `/accuracy/headline`, the homepage
placard and `test_accuracy_headline_route.py` are untouched. `score_by_tier` emits one stored
row per floor, for the reason `HEADLINE_TIER` is stored at all: a headline that exists only in
a run's console output cannot be audited or recomputed.

`_headline_line` gains one log line printing DA, PT and `n` at each floor, which is what
answers "where does the headline stabilise".

## 5. The grid, and what is not in it

The review asks for 6 price bands × 4 staleness quartiles, never pooled.

- **Band axis:** the six tier rows.
- **Staleness axis: 2 buckets, not 4.** Every row already carries
  `directional_accuracy_unchanged` / `directional_accuracy_moved` / `unchanged_pct`, which is
  the `actual_price == base_price` split. That is a real staleness partition and it ships now.
  The 4-quartile version needs `stale_run_days` — a run-length over the archive, which is
  step 6's deliverable. **Gap, stated deliberately:** no quartile is invented here.
- **Pooled row:** `price_tier = NULL` stays stored, for API defaults and months of series
  continuity. Nothing quotes it — `_headline_line` reads `HEADLINE_TIER` — and the changelog
  records that it is retained for continuity only.

## 6. API surface

`PRICE_TIER_QUERY` in `api/routes/accuracy.py` widens from `ge=HEADLINE_TIER, le=4` to
`ge=-3, le=5`, with its description updated to name the two new floors. `/accuracy/headline`
is unchanged.

No frontend change follows: `frontend/lib/api.ts` types `price_tier` as an unconstrained
`number | null`, and `frontend/app/accuracy/page.tsx` renders no tier labels.

## 7. Tests

**New `tests/test_friction.py`:** venue constants; `actionable_threshold` arithmetic; the
nearest-band mapping rule reproduced from the source table; every `price_tier()` output has a
spread entry (so adding a tier without a spread fails loudly rather than `KeyError`-ing at
score time).

**Extensions to `tests/test_backtest_scoring.py`:**

- `price_tier(999.99) == 4`, `price_tier(1000) == 5`, and tiers 0–3 unmoved.
- Actionable gating: h=3 → `out_of_scope` with `None` values; h=14 → populated.
- A `|r̂|` just under and just over the tier threshold flips membership.
- `r_act == 0` (carry-forward) scores as a miss, not a hit.
- `actionable_e_net_pct` is negative when every actionable call is wrong.
- `actionable_n == 0` yields `None` for `actionable_da`, never `0.0`.
- The three floor sentinels are emitted, nest correctly (`n(-3) ≤ n(-2) ≤ n(-1)`), and are
  absent when no record reaches the floor.

**Existing assertions that change:** `tests/test_parquet_nested_columns.py:212` (hardcoded 7
rows per `(horizon, model)`) and `tests/test_backtest_scoring.py:1072` (the expected tier
set). `tests/test_serving_policy.py` must keep passing untouched — it is the guard that the
served population still equals the headline population.

## 8. Out of scope

- Moving `MIN_SERVED_PRICE_USD`. The sweep measures; policy follows later.
- `stale_run_days` and the 4-quartile staleness axis — step 6.
- Per-item `buff_spread_rel` and any new frozen column, migration or `--reresolve`.
- Re-scoring history. A `--rescore` populates the new keys on existing rows at no archive
  cost, but running it is an operational step, not part of this change.

## 9. Expected result

Failure, on `n_actionable` first — the review says so, and calls it the publishable internal
result. At CSFloat the model must predict a **>7.2% move at tier 5 and >23.1% at tier 1** for
a call to be actionable at all. Every live cohort also still spans 1–2 forecast dates, so
`actionable_pt_verdict` will read `insufficient_dates` exactly as `pt_verdict` does. The
deliverable is the instrument, not a number.
