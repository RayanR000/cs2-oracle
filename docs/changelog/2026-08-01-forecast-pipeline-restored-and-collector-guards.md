# Forecast Pipeline Restored; Collectors Made to Fail Loudly (2026-08-01)

Execution log for the four commits that fixed the incident diagnosed in
`2026-07-31-accuracy-work-closed.md`, which found the problems but changed no code:

| Commit | Change |
|---|---|
| `ae17f13` | Persist forecast models via the Actions cache instead of `git add` |
| `324cfff` | Make collectors fail on zero rows; delete `reddit-sentiment` |
| `db5bddb` | Cap historical-fallback staleness; save boosters even when predict dies |
| `e11be1c` | Chunk predict-phase feature engineering to bound peak memory |

## `price-forecast` had two independent causes, not one

The 17-day outage (2026-07-14 → 07-31, 34+ failed runs) was diagnosed in stages, and
each stage corrected the previous one's story. Recording the corrections, because the
single-cause version is the intuitive one and it is wrong:

**Cause 1 — gitignored boosters (`ae17f13`).** The workflow persisted models with
`git add backend/models/saved_models/`, but `*.txt` / `*.pkl` are gitignored
(`.gitignore:68-71`), so the add was a no-op and only `meta.json` was tracked. Every
predict-only run checked out a `meta.json` with no boosters and aborted on
`No models available for prediction`.

Committing the boosters is **not** the fix — that is what produced the 1.8G
unpushable-`main` blob purged in `47d3195`. They now ride the Actions cache:
restore before predict-only on a rolling `forecast-models-<run_id>` key with a
`forecast-models-` prefix restore-key, saved after both `full` **and** `train-only`
(the old commit step missed `train-only`). `engineered_data.parquet` is excluded from
the cached path, which keeps the entry at ~16 MiB instead of ~2 GB. A `Verify models
are present` step fails with an actionable message when nothing restores, rather than
the opaque abort 60s later that hid this for 17 days.

**Cause 2 — the predict phase was OOM-killed (`db5bddb` + `e11be1c`).** `ae17f13`
assumed Monday's full retrain "predicted fine in its own workspace." It did not —
Monday runs never finished. Training completed in ~12 min and wrote valid boosters,
then the predict phase engineered all 1460 days for all 8,691 items (6.1M rows) and
was SIGKILLed (exit 143) on the 16 GB runner, always at `Adding item identity
features`: runs `30226424193`, `29793571744`, `30666903525`, `30668690592` — the last
a *fresh* predict-only process, which rules out training-frame residue as the cause.

`db5bddb` decoupled the cache save from the forecast step's exit status
(`if: always()` + a booster-count gate). Conditioning the save on success is what
discarded those boosters, so the cache stayed empty and predict-only runs aborted on
missing models *before the OOM was ever reached* — which is why Cause 2 stayed
invisible while Cause 1 was in play. It also corrected `ae17f13`'s claim that
`engineered_data.parquet` is a training intermediate prediction never reads: `predict()`
is its only writer (`forecaster.py:3361`) **and** its only reader.

`e11be1c` fixed the OOM itself. `predict()` only ever consumes `tail(3)` per item
(smoothed current price) and `last()` (the feature vector), yet engineered the whole
history for every item just to slice that tail off. It is now chunked —
`PREDICT_CHUNK_ITEMS`, default 1000 (~0.7M rows/chunk), `0` disables — so peak memory
scales with chunk size rather than catalogue size, at ~2x feature-engineering CPU.

**Why chunking is exact rather than approximate:** every cross-sectional feature is a
per-date **mean** across items, and a mean is recoverable from `(sum, count)` partials.
`_add_cross_sectional_features` split into `_accumulate_market_partials` /
`_market_from_partials` (build the per-date table) and `_apply_market_aggregates`
(attach features given one). The whole-frame and chunked paths call the same apply
function; only the table's provenance differs. Pass A accumulates partials across all
items, pass B applies the table and keeps each item's tail.

Two fidelity details worth knowing before touching this code: the old volume branch
used `df.merge()` (fresh `RangeIndex`) where `.map()` preserves the index, so the reset
is now explicit; and the volume block is gated on volume observed **globally**, not
per chunk, so a volume-less chunk still receives the market columns.

Known residual difference, accepted: `_detect_current_regime` reads the last non-NaN
`market_return_30d`, and the chunked frame holds 3 rows per item. It can only diverge
if an item's latest 3 dates all have a NaN market return — i.e. no item in the
catalogue had a 30d return on those dates.

## Verification — prod rows, not a green badge

Per `collectors-fail-silently`, green CI is not evidence. Both a dispatch and a
*scheduled* run were checked against the output table:

| Run | Trigger | Result |
|---|---|---|
| `30670215790` (07-31 22:48) | `workflow_dispatch`, full | 34,764 forecasts / 8,691 items → `item_forecasts` |
| `30674511100` (08-01 00:02) | scheduled, cache-restore + predict-only | 34,764 forecasts / 8,691 items, `lgbm-v3-regime` |

The scheduled run restored boosters from the cache, ran 9 chunk passes without OOM
(`Chunked engineering complete: 26,073 tail rows retained`), and its counts match the
last known-good manual run. The daily path is healthy end to end.

## Collectors: the zero-row guard actually guards now

`324cfff` audited the remaining workflows for the supply-scraper shape (log an ERROR,
store nothing, exit 0) and found two structural holes plus one more live instance:

- **The guard covered almost nothing.** It keyed on `items_collected`, a field only
  `collectors/pipeline.py` sets, so every other task could return `success` with zero
  rows and exit 0 — precisely how both dead collectors hid. `run_task.py` now checks
  every count field any task returns (`ROW_COUNT_FIELDS`) and fails when a task
  reports counts and all of them are zero. It also catches `status: "skipped"`, which
  slipped past both guards: not a failure status, and no counts to inspect.
- **`backtest-accuracy` could not fail.** It was the only workflow missing
  `set -o pipefail` before `| tee`, so the step status was `tee`'s — always 0 — making
  its own `sys.exit(main())` dead as a CI signal.
- **`reddit-sentiment` deleted.** `old.reddit.com` returns `403 Client Error: Blocked`
  for all three subreddits from runner IPs (run `30654776364`), then reports
  `status: success, 0 inserted`. Prod `social_mentions` held 0 rows all-time, there is
  no `social_mentions.parquet`, and `collection_runs` has zero reddit entries. Every
  run pulled a 519 MB FinBERT model to score nothing, 4x/day. Not repairable from
  hosted CI, and it served a feature group already refuted (per
  `2026-07-22`: social features rank outside the top 20 of 122 at every horizon *even
  when data flows*, and the projected +1–3pp sits at or below the 1.15pp noise floor).
  `social_sentiment.py` is kept for local/authenticated runs.

## The historical fallback was laundering stale prices

`db5bddb`'s other half. The fallback re-emits an old price stamped `timestamp=now` and
had no age cap, while prod `price_history` has been frozen since 2026-07-11 (16,487
rows, 5,503 of 35,058 items — daily data now goes only to CSV → Parquet). A total
upstream outage would therefore have written ~5.5K items' worth of 20-day-old prices as
if current, kept `items_collected` non-zero, and hidden the outage from the very guard
added above. Stale items now count as errors (`FALLBACK_MAX_AGE_DAYS=7`,
env-overridable), so an outage drives `items_collected` toward 0 where the guard catches
it. Both loaders share a new pure `_split_fresh_and_stale`; the newest row per item
decides freshness, since an older row cannot rescue an item whose latest price is
already stale.

`historical_fallback_stale_declined` was deliberately **not** added to
`source_breakdown`: `collection_runs.parquet` types that column as a fixed DuckDB
STRUCT, so an unknown key fails the cast, and that failure is swallowed by a
`logger.warning` that would have silently stopped Parquet updates.

## Tests

235 tests pass. The load-bearing ones for this entry:

- `test_chunked_engineering_matches_whole_frame` — chunked vs whole-frame tails via
  `assert_frame_equal` at `rtol=atol=1e-9`
- `test_market_partials_combine_to_whole_frame_means` — 3 uneven chunks reconstruct the
  global means
- two fallback-staleness tests, mutation-checked: disabling the cap makes the
  integration test fail with `historical_fallback: 3`, so it genuinely observes the
  laundering

## Still open

- **`event-correlation-analysis`** — the one unresolved item from the 07-31 audit.
  Green weekly (07-19, 07-26) but `event_correlations` has been stale since 2026-07-12.
  Its count fields (`impacts_written`, `patterns_written`, `correlations_written`) are
  now in `ROW_COUNT_FIELDS`, so the next scheduled run (Sun 04:00 UTC) will either
  write rows or fail loudly and file an issue. Left to that run rather than
  pre-debugged.
- **The weekly full retrain is unexercised on a schedule.** Cache *save* with freshly
  trained boosters has only run under `workflow_dispatch`. Relatedly, the restored
  bundle contains no regime models — predict logs
  `Regime model usage: 0/12 (0.0%) regime, 12/12 (100.0%) global` while writing
  `model_version: lgbm-v3-regime`. Worth confirming on the Monday run whether that is
  expected or a silent degradation to global models.
- **Latent staleness check that never fires.** `_save_engineered_cache` stores
  `_cache_date` in `df.attrs`, which `to_parquet` does not persist.
- **`discover-new-items` is not broken** — the 07-31 audit flagged it as "not firing."
  Its schedule was intentionally removed 2026-07-08 (the catalog is curated via the
  monthly CSMarketAPI backfill, not Steam discovery). Manual dispatch only. Its failure
  notification step is still conditioned on `github.event_name == 'schedule'` and so can
  never fire — harmless, noted in `operations.md`.

## Files changed

Code and workflows landed in the four commits above. Docs in this change:

- `docs/changelog/2026-08-01-forecast-pipeline-restored-and-collector-guards.md` — this entry
- `docs/operations.md` — removed the deleted `supply-scraper` / `reddit-sentiment` rows,
  data-flow lines, healthy-state expectations, and manual-test commands; added the
  model-cache mechanics and a data-freshness check that does not trust a green badge
- `docs/README.md` — changelog entry count

## Related

- `docs/changelog/2026-07-31-accuracy-work-closed.md` — the audit that found all of this
- `docs/changelog/2026-07-16-drop-supply-depth.md` — the original supply drop
