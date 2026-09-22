# Codebase Cleanup Spec — 2026-09-15

Full-codebase review completed by 5 parallel Opus agents. This spec captures every
actionable finding, grouped by priority tier. Each item is self-contained: a subagent
can pick it up without reading the rest.

---

## Tier 1 — Correctness (fix immediately)

### 1.1 Double-slicing bug in `get_price_history` ✅ DONE

**File:** `backend/api/routes/items.py:430-441`

**Bug:** `skip` is applied twice. Line 430 slices `all_records[skip:skip+limit]` into
`records`. Line 441 slices `records[skip:skip+limit]` into `records_slice`. The return
iterates `records_slice`, so the endpoint skips `2 * skip` records instead of `skip`.

**Fix:** Remove line 441. Use `records` directly in the return loop (line 443).

**Test:** `GET /items/{id}/price-history?skip=10&limit=5` should return records 10-14,
not 20-24.

**Resolution:** Pagination moved to SQL `.offset(skip).limit(limit)`;
`records_slice = records` is now a pass-through.

---

### 1.2 Direction signal leak on `/opportunities` ✅ DONE

**File:** `backend/api/routes/opportunities.py:32`

**Bug:** `current_trend=forecast.direction or "neutral"` uses the raw DB direction.
`serving_policy.py` has `DIRECTION_DISCLOSED = False`, so `/prediction` returns "neutral"
for the same item that `/opportunities` labels "up"/"down". This leaks the withdrawn
signal.

**Fix:** Replace with `current_trend="neutral"` (or import and check
`DIRECTION_DISCLOSED`). Also fix `_reason_for_type` (lines 36-42) which says "ML
forecast predicts upward/downward movement" — replace with a range-based explanation.

**Test:** Verify `/opportunities` never returns a `current_trend` other than "neutral"
while `DIRECTION_DISCLOSED = False`.

**Resolution:** `current_trend` now routes through `served_direction()`;
`_reason_for_type` rewritten to range-based language. Test in
`test_direction_withheld.py::test_opportunities_routes_through_served_direction`.

---

## Tier 2 — Dead modules (delete outright)

Each of these has zero production callers. Delete the module and its test file(s).

### 2.1 `models/tft/` (entire directory, ~600 lines)

TFT centre REFUTED. Files: `__init__.py`, `model.py`, `dataset.py`, `trainer.py`,
`components.py`. Tests: `test_tft_trainer.py`, `test_tft_dataset.py`,
`test_tft_components.py`. Also remove `TFT_CENTRE` flag and `_train_tft` /
`_predict_tft` from `forecaster.py:149-155, 6326-6414`.

### 2.2 `backtest/papertrade.py` (290 lines)

Trade path DEAD. Also delete `scripts/papertrade_report.py` and
`tests/test_papertrade.py`.

### 2.3 `backtest/longshort.py` (87 lines)

Trade path DEAD. Also delete `scripts/replay_lambdarank.py` and
`tests/test_longshort.py`.

### 2.4 `api/routes/portfolio.py` (35 lines)

Complete stub — returns hardcoded `{"items": []}`. Remove the router registration in
`main.py:32`.

### 2.5 `collectors/google_trends.py` (whole file)

REFUTED. Also remove dead methods in `forecaster.py:4370-4453` (`_google_trends_cache`,
`_load_google_trends`, `_apply_google_trends`, `_player_counts_cache`,
`_load_player_counts`, `_apply_player_counts`) and the `_feature_group` mappings at
`forecaster.py:312-315` for `google_trends_` and `player_count_`.

### 2.6 `collectors/export_finbert_onnx.py` (whole file)

Utility for dead social sentiment pipeline. No callers.

### 2.7 `collectors/supply_scraper.py` (293 lines)

Superseded by `supply_depth.py`. Also delete `scripts/run_supply_scraper.py`. Check
that `run_task.py:215` no longer dispatches to it (or remove that dispatch).

---

## Tier 3 — Dead code in live modules

### 3.1 Social sentiment features in `forecaster.py:3681-3780`

`_fetch_social_mentions` and `_add_social_features` — permanently outside the feature
allowlist, social_mentions holds 0 rows. Remove both methods and the call site at
`forecaster.py:5215`.

### 3.2 `MOMENTUM_FALLBACK_HORIZONS` in `forecaster.py:803-805, 10007-10013`

Permanently empty list. Remove the constant, the conditional, and the
`_recenter_on_momentum` alias at `forecaster.py:7406`. Leave the source method in
`direction.py:131-148` if tests import it; otherwise remove.

### 3.3 `_add_item_metadata_features` in `forecaster.py:3782-3795`

Always skipped by allowlist. Remove method and call site.

### 3.4 WACI functions in `conformal.py:351-481`

`calibrate_signed_waci()`, `waci_lookup()`, `band_signed_waci()` — documented "ZERO
callers". Remove all three (~130 lines). Also delete `scripts/measure_waci.py`.

### 3.5 Dead pipeline methods in `pipeline.py:911-1029`

`_load_recent_price_histories`, `_load_parquet_histories`,
`_load_recent_price_counts`, `run_trend_analysis` — never called. Remove.

### 3.6 Social sentiment endpoint in `items.py:988-1127`

`/items/{item_id}/social-sentiment` serves 0 rows permanently. Remove the endpoint
and `_social_sentiment_parquet` helper.

### 3.7 Dead bullish/bearish branches in `items.py:659-664` ✅ DONE

`DIRECTION_DISCLOSED = False` means only the neutral branch fires. Simplify to return
the neutral explanation directly.

**Resolution:** `_build_trend_explanation` simplified to a single range-based return;
dead bullish/bearish branches and unused `current_price` parameter removed.

### 3.8 `skinport_quantity` fossil ✅ DONE

- `database.py:453` — column is 100% NULL for its entire life
- `forecaster.py:3516` — `np.log1p(df["skinport_quantity"])` always produces 0.0
- Remove the feature computation. Leave the ORM column (migration cost to drop).

**Resolution:** Removed `skinport_quantity` from `_fetch_supply_snapshots` empty frame,
column assignment, fillna, drop, and docstrings. ORM column and migration kept.

### 3.9 `data_validation.py` (379 lines)

Never imported by production code. Validates a stale schema. Delete with its test.

### 3.10 `forecast_market_factor_diagnostics()` in `market_factor.py:190-218`

Docstring says "explicitly excluded from the decision rule." Only used by tests. Remove.

---

## Tier 4 — Dead scripts

### 4.1 Delete outright (15 scripts, ~8K lines)

```
eval_tft.py                      # TFT REFUTED
papertrade_report.py             # trade path DEAD
discover_steam_items.py          # BROKEN: imports deleted real_data_collector
run_supply_scraper.py            # superseded by run_supply_depth.py
backfill_ssr_history.py          # Steam cookie endpoint dead since 07-16
backfill_steam_listing_history.py # imports deleted real_data_collector
backfill_supply_metadata.py      # superseded by ingest_bymykel_metadata.py
build_market_catalog.py          # CSMarketAPI quota dead since 07-16
repair_catalog_gaps.py           # same dead quota
purge_phantom_items.py           # imports deleted real_data_collector
ingest_supply_history.py         # coverage ends 2024-01
ingest_volume_panel.py           # stale to 2026-06-15
ingest_fx_history.py             # Google Trends REFUTED, CNY never shipped
migrate.sh                       # run_task.py migrate is the entry point
```

### 4.2 Archive to `scripts/archive/` (44 scripts, ~30K lines)

All concluded experiment scripts. Full list:

```
ab_test_csfloat_basis.py         ab_test_direction_labels.py
ab_test_ensemble.py              ab_test_feature_contribution.py
ab_test_frozen_runs.py           ab_test_interval_sampling.py
ab_test_item_metadata.py         ab_test_price_primitives.py
ab_test_q50_sampling.py          ab_test_recency_weights.py
ab_test_regime.py                ab_test_supply_side.py
ab_test_train_universe.py        ab_test_training_breadth.py
ab_test_volume_features.py       anchor_wedge_attribution.py
attribute_band_level.py          attribute_marginal_coverage.py
climatology_vs_gbm.py            centre_vs_lastprice.py
magnitude_vs_climatology.py      measure_conditional_qhat.py
measure_qhat_bagging_mondrian.py measure_composition_stability.py
measure_sync_exclusion_2025.py   label_ceiling_2025.py
design_sigma_scale.py            conditional_coverage_by_stratum.py
dollar_band_wedge.py             horizon_friction_scan.py
signal_probe_180d.py             shrink_k_stability.py
shrink_k_vol_rank_ab.py          replay_reactive_ab.py
replay_lambdarank.py             pick_calm_anchors.py
merge_17mafo_gap.py              merge_price_primitives_ab.py
merge_supply_side_ab.py          promote_iflow_staging.py
resolve_readonly.py              compute_mde.py
backtest_walkforward_report.py   walkforward_backtest.py
```

---

## Tier 5 — Duplication (DRY fixes)

### 5.1 `QualityVariantOut` — 3 copies

Defined in `items.py:319-326`, `market.py:31-37`, `schemas.py:192-199`. Keep the
`schemas.py` version (has `from_attributes = True`), import elsewhere.

### 5.2 `_parse_item_name` — 2 copies

Defined in `items.py:312-316` and `market.py:59-70`. Extract to `schemas.py` or a
shared `api/utils.py`.

### 5.3 `_resolve_user` / `_get_current_user` — 2 copies

`auth.py:20-29` and `portfolio.py:10-19`. Portfolio is dead (Tier 2.4), so this
resolves itself.

### 5.4 `GroupedMarketItemOut` — 2 copies

`market.py:40-52` and `schemas.py:204-215`. Keep `schemas.py`, import in `market.py`.

### 5.5 Trends computation — 2 near-identical paths ✅ DONE

`items.py:532-578` (parquet) and `items.py:592-656` (DB fallback). Extract the
SMA/Bollinger/RSI/MACD computation into a shared builder function that takes a price
list.

**Resolution:** Indicator computation already shared via `_trend_indicators`;
`TrendAnalysisOut` construction extracted into `_build_trend_response`. Both
`_trends_parquet` and the DB fallback now delegate to it.

### 5.6 `score_sentiment` — duplicate definition

`social_sentiment.py:95` and `:208`. Second shadows the first with identical body. If
module is kept, remove the duplicate.

---

## Tier 6 — Code quality

### 6.1 Split `_train_horizon_inline` (975 lines) ✅ DONE

`forecaster.py:6416-7390`. Break into:
1. HP search + ensemble fit (~300 lines)
2. Auxiliary heads: direction, exceedance, anomaly, vol-rank (~300 lines)
3. CV evaluation + conformal calibration (~200 lines)
4. Metrics aggregation + logging (~175 lines)

**Resolution:** Extracted 3 methods: `_train_auxiliary_heads` (~265 lines),
`_cv_and_calibrate` (~209 lines), `_aggregate_cv_metrics` (~275 lines).
Main method reduced from ~1095 to ~420 lines. 192 tests pass. Opus 5 review
confirmed correct parameter passing and no lost variables.

### 6.2 Split `predict()` (508 lines) ✅ DONE

`forecaster.py:9642-10150`. Break into:
1. Feature preparation + anchor resolution
2. Per-horizon prediction loop
3. Output construction + sanitization

**Resolution:** Extracted 3 methods + `_PredictContext` dataclass:
`_prepare_predict_features` (~287 lines), `_predict_horizon` (~229 lines),
`_finalize_predictions` (~31 lines). `predict()` reduced to a 10-line coordinator.
192 tests pass. Opus 5 review confirmed correct field propagation.

### 6.3 Silent `except Exception: pass` blocks

Add `logger.debug` or `logger.warning` at minimum. Locations:
- `items.py:588-589, 727-728, 861-863, 1072-1074`
- `events.py:52-53`
- `accuracy.py:183-184`

### 6.4 Vectorize `directional_accuracy()`

`direction.py:21-41` — Python loop doing string comparisons. Rewrite using the
existing `direction_classes()` function (3 lines below it).

### 6.5 Deprecated startup event

`main.py:68` — `@app.on_event("startup")` → `lifespan` context manager.

### 6.6 Hardcoded secret key ✅ DONE

`config.py:67` — `secret_key: str = "your-secret-key-for-sessions"`. Add a guard
that raises in production if the default is unchanged.

**Resolution:** `check_secret_key()` guard at `config.py:75-83` raises `ValueError`
in production when the default is unchanged.

### 6.7 Session token in URL query parameter

`auth.py:106` — `redirect_url = f"...?session={token}"`. The token is already set as
a cookie (lines 110-117), so remove it from the URL.

### 6.8 Unbounded `query.all()` in market summary ✅ DONE

`market.py:92-118` — loads all items into memory, paginates in Python. Add a DB-level
limit.

**Resolution:** Added `.limit(2000)` to the items query in `_build_market_summary`.
Test in `test_market_summary_bound.py` asserts the bound exists.

---

## Tier 7 — Dead schema fields

Remove from Pydantic models (no migration needed):
- `OpportunityOut.volatility` — always `None`
- `MarketItemOut.volatility` — always `None`
- `GroupedMarketItemOut.volatility` — always `None`

---

## Tier 8 — Stale references (low priority)

- "VADER" docstrings in `forecaster.py:3681,3732` and `database.py:461` → FinBERT
- `ab_test.py` routes serve unpowered A/B results with no disclaimer
- `csmarketapi_backfill.py:267-288` spends quota on player counts (REFUTED, panel frozen)

---

## Estimated impact

| Tier | Items | Lines removed/cleaned | Risk |
|------|-------|-----------------------|------|
| 1. Correctness | 2 | ~10 changed | Low — targeted fixes |
| 2. Dead modules | 7 | ~1,400 deleted | Low — zero callers verified |
| 3. Dead code in live modules | 10 | ~900 deleted | Low-Med — touch live files |
| 4. Dead scripts | 59 | ~38,000 deleted/archived | None — standalone files |
| 5. Duplication | 6 | ~200 consolidated | Low — mechanical |
| 6. Code quality | 8 | refactor, no net deletion | Med — large refactors in 6.1/6.2 |
| 7. Dead schema fields | 3 | ~10 deleted | None |
| 8. Stale references | 3 | ~10 changed | None |
