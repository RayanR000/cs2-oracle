# iflow serve-universe expansion — design

**Date:** 2026-08-17
**Status:** design, pending review
**Author:** rane.ra@northeastern.edu (with Claude)

## Problem

The product serves a calibrated price range for ~942 items priced ≥ $1. That ceiling is a
**backfill limit, not a market limit**: the daily feed collects ~38,000 items, but only items
with deep archived history become forecastable, and today that set is ~5,536 (≥$1 subset ~942).

A staged BUFF/iflow dataset (`buff-iflow-staging/price-archive/prices-buff_iflow-*.parquet`,
already USD, 2022–2026, 19,610 items) holds deep history for **~10,700 additional ≥$1 items**.
Ingesting it would raise the served ≥$1 universe from ~942 to ~11,600 (~12×).

Validation done 2026-08-17 (`replay_serving.py`, isolated temp DB, iflow temp-copied into the
archive): the **existing** trained model served n=11,085 ≥$1 items at anchor 2026-04-01 with band
coverage 96.98 / 97.46 / 95.48% (h=3/7/14, target 80%). The expansion is forecastable with **no
retrain**, and the bands hold (they over-cover; they do not collapse).

## Goal / non-goals

**Goal:** make the ~11,000 iflow-backed ≥$1 items appear in the forecast/serve output, durably
across retrains, **without growing the training set past its current ~926 items.**

**Non-goals:**
- Not retraining on the new items. Prior A/B (2026-08-16) showed adding iflow data does not
  improve per-item accuracy (null); this change is pure **coverage/breadth**, not accuracy.
- Not fixing the marginal over-coverage (87–97% vs 80%). That is a separate, tracked problem; the
  new items inherit and slightly amplify it, which is acceptable (they overshoot, not miss, target).
- Not a new history-depth threshold. The predict path's existing ≥14-day floor is unchanged.

## Key mechanism (why this is small)

Train and predict currently share **one gate**, `items.is_backfilled = 1`, set by
`init_local_db.py` from "any archive row with `day < 2026-01-01`." The ONLY thing separating the
~926 training set from the wider served frame is the `$1` **median-price floor** applied on the
training path (`_filter_by_median_price`, `min_median_price=1.0`) and not on the predict path.

`is_backfilled` **widens for free** once iflow is in the archive (iflow rows are pre-2026). The
risk is that this same widening pulls the new ≥$1 items into training. So the fix is to give
**training its own narrow gate** and leave serving on the (now-wider) `is_backfilled`.

## Design

### 1. Ingest

Promote the 50 `prices-buff_iflow-*.parquet` files from `buff-iflow-staging/price-archive/` into
the durable archive repo `RayanR000/cs2-oracle-data` (`price-archive/`), where CI's
`prices-*.parquet` glob reads them. Keep the distinct `buff_iflow` filename/source so nothing
overwrites canonical monthly files. (Liquidity sidecars are out of scope — no features depend on
them; see [[buff-iflow-backfill-source]].)

`buff_iflow` needs **no** allow-list change — it is not in `BID_SOURCES`,
`TRAILING_WINDOW_SOURCES`, or `STEAM_SPOT_SOURCES`, so it votes as an ordinary ask.

### 2. New narrow training gate: `items.is_trainable`

Add a column `items.is_trainable` (Integer, default 0). Derivation (in `init_local_db.py`,
alongside the existing `is_backfilled` derivation, re-derived every run):

> `is_trainable = 1` iff the item has an archive row with `day < 2026-01-01` **from a source other
> than `buff_iflow`** (NULL sources — the pre-2026 series — count).

This makes the training universe **independent of the iflow ingest**: it is exactly the set that
is backfilled today, so `is_trainable ∩ median≥$1` reproduces the current ~926 cohort regardless
of how much iflow is added. It is a live SQL predicate, not a frozen snapshot (robust to future
non-iflow backfills, which SHOULD grow training).

### 3. Route the training path to `is_trainable`; leave serving on `is_backfilled`

`fetch_price_history(backfilled_only=True)` → `_resolve_backfilled_slugs` (`forecaster.py:1795`)
is called by **both** `predict()` and `build_training_data()`. Parameterize which gate it reads:

- Add `universe: str = "serve"` to `fetch_price_history` / `_resolve_backfilled_slugs`; `"serve"`
  → `WHERE is_backfilled = 1`, `"train"` → `WHERE is_trainable = 1`.
- `predict()` (`forecaster.py:7825`) passes `universe="serve"` (default).
- `build_training_data()` (`forecaster.py:~4917`) passes `universe="train"`.
- `_voted_cache_key` (`forecaster.py:5049`) already keys on the resolved slug set, so the two
  universes cache separately with no extra work.
- `forecast_prices.py:601` `slug_to_id` query (used to persist rows) widens to `is_backfilled = 1`
  — it already is; serving-side reads need no change because they already gate on
  `backfilled_item_clause()` = `is_backfilled == 1` (`database.py:105`), which now includes iflow.

### 4. Does `buff_iflow` vote in the training consensus?

Once ingested, `buff_iflow` extends **existing** training items' series (an extra ask vote per
item-day). This is exactly what the 2026-08-16 A/B tested → **null on accuracy**. Decision:
**let it vote** (simpler; measured-null impact). The training *item set* stays ~926 via
`is_trainable`; only the per-item price series gains iflow votes on overlapping days.

Alternative (rejected unless review disagrees): exclude `buff_iflow` from voting on the train
path for byte-identical reproducibility — adds a source-filter branch for no measured benefit.

### 5. Guardrail

Add a test asserting the training universe stays within a band of its current size
(`is_trainable ∩ median≥$1` ≈ 926, tolerance e.g. ±50), so a future change cannot silently widen
training. This is the invariant the whole design protects; it must be enforced, not documented.

## Data flow (after change)

```
archive (now incl. buff_iflow)
  ├─ init_local_db → is_backfilled=1  (any pre-2026 row; INCLUDES iflow)   → SERVE gate
  │                → is_trainable=1   (pre-2026 row, non-iflow source)     → TRAIN gate
  ├─ predict()          fetch_price_history(universe="serve")  → ~17k items, ≥14d → bands for all
  │                     serve routes gate on is_backfilled + $1 floor      → ~11.6k ≥$1 served
  └─ build_training_data fetch_price_history(universe="train") + $1 median → ~926 items (unchanged)
```

## Touch-points (files)

- `backend/database.py` — add `is_trainable` column + index; a `trainable_item_clause()` helper.
- `backend/scripts/init_local_db.py` — derive/re-derive `is_trainable` (source-aware SQL).
- `backend/models/forecaster.py` — `universe` param on `fetch_price_history` /
  `_resolve_backfilled_slugs`; `predict()` → `"serve"`, `build_training_data()` → `"train"`.
- `RayanR000/cs2-oracle-data` — add the 50 `prices-buff_iflow-*.parquet` files to `price-archive/`.
- `backend/tests/` — training-size guardrail test; a test that `is_trainable` excludes iflow-only
  items and `is_backfilled` includes them.

**Not touched:** serve routes (`opportunities.py`, `items.py`) — they already gate on
`is_backfilled`, which widens automatically; the model artifact — no retrain; the ≥$1 serving
floor; `PREDICT_MIN_HISTORY_DAYS`.

**One serve-route caveat:** `items.py` **trending** additionally requires `Item.icon_url IS NOT
NULL` (`items.py:131`). New iflow items are served via `/opportunities` and the per-item forecast
regardless, but will not appear on the trending surface until `icon_url` is populated for them.
Backfilling `icon_url` is a separate, optional follow-up (metadata only, not on the forecast path).

## Migration / rollout

1. Land the code (flag + gate routing + tests) — no behavior change until iflow is in the archive.
2. Run the DB migration to add `is_trainable` and back-derive it in prod (Supabase). Before iflow
   ingest, `is_trainable` = `is_backfilled` (no iflow rows yet), so training is unchanged.
3. Ingest iflow into the durable archive; re-run the `is_backfilled`/`is_trainable` derivation.
   Serving widens; training stays put.
4. Predict-only forecast run publishes ranges for the new items.

**Rollback:** remove iflow files from the archive and re-derive; `is_backfilled` contracts back to
~5,536 and the served set returns to today's. `is_trainable` is inert without iflow.

## Risks

- **Prod `is_backfilled` derivation.** `init_local_db.py` is the documented sole writer; confirm
  the prod flag is set by the same path (or add the derivation to whatever seeds prod) before
  step 2. If prod sets `is_backfilled` differently, the `is_trainable` derivation must mirror it.
- **Serving quality on new items is over-coverage (95–97%), not calibrated.** Acceptable per
  non-goals, but the product should not claim 80% bands on these items without the marginal-
  coverage fix. Surface the same caveat the existing cohort already carries.
- **Survivorship.** iflow deep-history items are liquidity-selected; a de-listed item still has a
  stale series. The ≥14-day-in-730 predict floor bounds this but does not remove it.
- **Archive size / CI time.** +9.8M rows in the archive grows every glob (predict feature build,
  backtests). Predict inference stays cheap; feature-engineering the wider serve frame is the cost
  to watch against the daily 30-min cap. Measure the predict-step wall time in step 4.

## Open questions for review

1. `is_trainable` (new narrow flag, this design) vs `is_served` (new wide flag) — confirm the
   narrow choice; it minimizes serve-route churn.
2. Let `buff_iflow` vote in the training consensus (recommended) vs exclude for reproducibility.
3. Training-size guardrail tolerance (±50?) and whether it lives in unit tests or a run-time assert.
