# Open Code-Review Findings — line refs verified 2026-08-21

Live punch list, not a history. Originally produced by the 2026-07-21 review; every
finding below was re-checked against `main` on 2026-08-21 and every line reference
re-anchored. Ordered by severity within each section.

**2026-08-21 re-audit summary.** The security cluster (#1–#3) is open verbatim, and #4, #7,
#8, #10, #12, #13, #14 and all three DORMANT items still reproduce. Four findings closed
because the file they lived in was deleted: **#5** (`migrate_to_parquet.py` gone,
`append_to_parquet.py` now routes through the atomic `os.replace` helper), **#6**
(`export_historical_parquet.py` gone), **#9** (both scripts gone), **#11**
(`tests/test_price_history.py` gone). Those four are marked RESOLVED in place.

- **LIVE** — reachable in a code path that runs today. Triage these.
- **DORMANT** — the defect is still in the file, but the workflow or quota that
  invoked it is dead. Do not spend triage time here; fix only if the path is revived.
- **RESOLVED** — kept briefly with fix evidence so nobody re-reports them.

---

## 🔴 The security cluster — the one actionable theme

Three findings share a root cause: **the codebase treats untrusted input as trusted
text**. They are independent bugs but they are one workstream, and they are the
highest-value thing on this list. Fix them together.

### 1. SQL injection via f-string interpolation

**All sites live.** A parameterized path already exists and is unused by these callers:
`backend/db/parquet.py:522` — `def query(self, sql: str, params: Optional[dict] = None)`,
which forwards `params` to DuckDB. Every site below builds SQL by string formatting instead.

| File | Line(s) | Interpolated |
|---|---|---|
| `backend/api/routes/events.py` | 20 | `type_filter`, hand-escaped by doubling quotes |
| `backend/api/routes/accuracy.py` | 134 | `where` clause + `limit` |
| `backend/api/routes/accuracy.py` | 449 | `where` clause + `limit` |
| `backend/api/routes/items.py` | 482–484 | `item_id`, `horizon_days` — **no escaping at all** |
| `backend/api/routes/items.py` | 770–775 | `item_id`, `limit` |
| `backend/api/routes/items.py` | 794–803 | `item_id`, `limit` |

```python
# events.py:20 — manual quote-doubling is the entire defense
where.append(f"type = '{type_filter.replace(chr(39), chr(39)+chr(39))}'")
```

The `int`-typed FastAPI params (`item_id`, `limit`, `horizon_days`) are coerced by
Pydantic and are not directly exploitable today; `events.py:20` and the two
`accuracy.py` clause builders are the string-valued ones. The reason to fix all six is
that the safety currently rests on an incidental type annotation rather than on the
query layer. DuckDB connections are opened without `enable_external_access=false`, so a
successful injection reaches `LOAD` and the filesystem.

### 2. Hardcoded default secret key

**File:** `backend/config.py:67`

```python
secret_key: str = "your-secret-key-for-sessions"  # Should be changed in production
```

Session tokens are `itsdangerous.URLSafeTimedSerializer` signatures over this key
(`_make_session_token`, `api/routes/auth.py:14-16`). Any deployment that forgets `SECRET_KEY` gets forgeable
sessions with a publicly known key. This should fail closed — no default, or refuse to
boot when `is_production()` and the value is the placeholder.

### 3. Session token leaked into the redirect URL

**File:** `backend/api/routes/auth.py:105`

```python
redirect_url = f"{settings.frontend_url}/portfolio?session={token}"
```

The same token is also set as an `httponly` cookie at `:108-112`. The cookie is sufficient;
the query param is redundant and puts the token in browser history, server access logs,
`Referer` headers, and any frontend analytics. Deleting the query param is a one-line
change gated only on confirming the frontend doesn't read it.

---

## 🔴 LIVE — correctness

### 4. Pagination double-slice in `/price-history`

**File:** `backend/api/routes/items.py:398` and `:409`

```python
records = all_records[skip:skip + limit]     # :398 — already sliced
...
records_slice = records[skip:skip + limit]   # :409 — re-slices the slice
```

Any request with `skip > 0` returns wrong, misaligned history. `skip=20, limit=50` over
200 records: the first slice yields records 20–69, the second indexes 20 into *those*,
returning 30 records starting at global offset 40. Still **zero test coverage** — no test
in `backend/tests/` exercises `/price-history`.

### 5. Non-atomic Parquet writes in the migration scripts — ✅ RESOLVED 2026-08-21

`backend/db/parquet.py` writes to a writer-unique temp and renames — `os.replace` at
`:454` and `:475`, with `_tmp_path` at `:457` deliberately including pid + uuid so two
writers can't interleave into one temp file.

Both offending callers are gone: `scripts/migrate_to_parquet.py` was deleted, and
`scripts/append_to_parquet.py` no longer calls `DataFrame.to_parquet` at all — it goes
through DuckDB and its own `os.replace` at `:232`.

### 6. `export_historical_parquet.py` overwrites pre-2026 year files — ✅ RESOLVED 2026-08-21

`backend/scripts/export_historical_parquet.py` no longer exists, so the destructive branch
is gone with it. If a historical exporter is ever rebuilt, route it through the
`db/parquet.py` helpers from #5.

---

## 🟡 LIVE — medium

### 7. Duplicated auth logic with divergent failure modes

`backend/api/routes/auth.py:19` `_resolve_user` and
`backend/api/routes/portfolio.py:13` `_get_current_user` are byte-for-byte equivalent
token-verification helpers. Both return `None` on failure, but callers diverge:
`auth.py:33` returns `None` as a 200 body, `portfolio.py:29` raises 401. There is no
shared `Depends`, so the two will drift — and any hardening applied to one (e.g. the
secret-key fix above) has to be remembered twice. Neither file has any test coverage.

### 8. `data_validation.py` is dead code; the live path checks `price > 0`

`backend/collectors/data_validation.py` (`DataValidator`, `DataCleaner`) is imported in
exactly two places, neither of which is a collector:
`backend/collectors/__init__.py:1` (re-export only) and
`backend/tests/test_data_validation.py:6`. The actual ingest gate is
`backend/collectors/pipeline.py:175` — `if price is not None and price > 0`. A misparsed
`$50,000` or a stale-but-positive price flows straight into `price_history`. Either wire
the validators into `pipeline.py` or delete them; the current state gives false
confidence that validation exists.

### 9. Destructive scripts have no confirmation gate — ✅ RESOLVED 2026-08-21

Both scripts were deleted: `backend/scripts/migrate_historical_data.py` (the bulk-`DELETE`
one) and `backend/scripts/import_steam_items.py`. The rule they motivated still stands for
anything new — **require `--yes` for mutation, dry-run otherwise.**

### 10. Copy-pasted HTTP retry/session logic across 7 files

`RETRY_BACKOFF` / bespoke `requests.Session()` construction is duplicated in
`collectors/csgotrader_aggregator.py`, `collectors/supply_scraper.py`,
`scripts/backfill_ssr_history.py`, `scripts/backfill_steam_listing_history.py`,
`scripts/build_market_catalog.py`, `scripts/merge_17mafo_gap.py`,
`scripts/repair_catalog_gaps.py`. A fix in one does not propagate — the `count=100`
over-fetch bug below is the concrete precedent.

---

## 🟢 LIVE — simplification / test hygiene

### 11. `tests/test_price_history.py` is not a test — ✅ RESOLVED (deleted 2026-08-10)

The file was deleted, along with `scripts/test_social_signal.py`, which was what made a
bare `pytest -q` abort during collection. See `AGENTS.md` → Commands.

### 12. Dead `Souvenir` strip in `steam_types.py`

**File:** `backend/models/steam_types.py` — note the path; this is under `models/`, not
`collectors/`. Lines `141-143` strip a leading `"Souvenir "`; lines `149-151` strip it
again after the `StatTrak™` block (`:145`). The second block can only fire on
`"StatTrak™ Souvenir …"`, which is not a name CS2 produces — unreachable in practice.
~3 lines.

### 13. `QualityVariantOut` / `GroupedMarketItemOut` defined three times

`backend/api/schemas.py:175` and `:187`, redefined in
`backend/api/routes/market.py:31` and `:40`, and again as
`backend/api/routes/items.py:293`. ~40 lines removable by importing from `schemas.py`.

### 14. Remaining untested surface

`/price-history` (see #4), `api/routes/auth.py`, and `api/routes/portfolio.py` have no
tests at all — which is exactly where the pagination bug, the auth drift, and the
token-in-URL leak live.

---

## 🟠 DORMANT — real defects on dead paths

Do not triage these. Each is still in the file; none of them can run.

### D1. Supply scraper 10× Steam over-fetch

`backend/collectors/supply_scraper.py:48` requests `"count": 100` per page, but the
offset advances by 10 — `current_offset += 10` at `:158` (failure path) and `:173`
(success path). Pages are `[0-99]`, `[10-109]`, `[20-119]`: 90% overlap, ~1,000 requests
where ~100 would do. **Dormant:** the supply-scraper workflow no longer exists in
`.github/workflows/` (present as of 2026-08-21: `ab-harness-batch`, `aggregator-update`,
`backtest-accuracy`, `discover-new-items`, `event-correlation-analysis`,
`forecast-freshness-check`, `model-diagnostics`, `price-forecast`, `schema-drift-check`). Revive the
workflow and this ships a ban-risk bug on day one.

### D2. Silent transaction death in the social-sentiment collector

`backend/collectors/social_sentiment.py:288-292` catches the insert exception and only
`logger.warning`s it — **no `db.rollback()`** — then the loop continues issuing
statements on a transaction Postgres has already aborted, and `:294` calls `db.commit()`.
The Parquet write at `:297-299` succeeds regardless, so the run reports success while the
DB gained nothing. **Dormant:** no workflow invokes this collector, and Reddit sentiment
was refuted and removed as a feature.

### D3. 429 skips backoff while 5xx backs off

`backend/collectors/csmarketapi_backfill.py:205-207` logs the 429 and `return None`
immediately — no sleep — burning a key rotation, while the 5xx branch at `:212-214`
correctly does `time.sleep(2 ** attempt)`. **Dormant:** the CSMarketAPI free-tier quota is
permanently exhausted across all keys; nothing calls this successfully.

---

## ✅ RESOLVED since 2026-07-21

| Was | Fix evidence |
|---|---|
| Forecast fallback hardcoded `current_price = 0.0`, collapsing null Parquet bounds to zero | `api/routes/items.py:675` now `current_price = r.current_price or 0.0`; the low/high fallbacks immediately below derive from it correctly |
| Race on a shared lazy `lgb.Dataset` across ensemble threads | `models/forecaster.py:5720-5721` calls `dtrain.construct()` / `dval.construct()` eagerly and single-threaded before any submit; the contract is documented in the `_train_ensemble_member` docstring at `:4853-4861`, and the same pattern is applied at `:5984-5985` |
| CPU oversubscription (~40 threads on 10 cores) from nested process + thread pools | Both pools removed — no `ProcessPoolExecutor` or `ThreadPoolExecutor` **call sites** remain anywhere in `backend/models/` (only explanatory comments at `forecaster.py:4861`, `:5718`, `:5781`, `:5983`). Rationale: `docs/changelog/2026-07-21-remove-training-parallelism.md` |
| Unsafe `joblib.load` of Ridge residual models | Residual stacking deleted; no `joblib` reference anywhere in `models/forecaster.py` |
| `scripts/merge_hf_dataset.py` non-atomic writes | Was routed through the atomic library; the script has since been deleted entirely |
| "All 9 API route files untested" | Partially closed: `tests/test_opportunity_selection.py`, `tests/test_serving_policy.py`, `tests/test_trending_ranking.py`, `tests/test_trend_explanation_copy.py` now import and exercise API code. Gaps that remain are itemized in #14 |
