# A Deterministic Forecast Backtest (2026-08-01 → 08-02)

The live backtest reported a different number every time it ran over the same
forecasts. The same 5,512-forecast cohort scored **61.76% / 33.74% / 61.54% /
57.91%** directional accuracy on four evaluation dates without a single new
observation entering the cohort. This entry covers the 23 commits that made the
metric a function of the data instead of a function of the run date.

Scope note: `docs/changelog/2026-07-31-accuracy-work-closed.md` closed the
feature/architecture roadmap on the grounds of the A/B harness's 1.15pp noise
floor. That closure does not cover this work. `scripts/ab_test_*.py` and
`scripts/backtest_accuracy.py` are different artifacts, and every defect below is
>10x that floor.

## Root cause: the two legs of `actual_ret` were different estimators

`actual_ret = (actual - base) / base` was differencing incompatible quantities:

- **base leg** — `item_forecasts.current_price`, a 3-observation *median* written
  at serving time by `ItemForecaster.predict()` (`forecaster.py:3513`)
- **actual leg** — a raw *single-day* voted price, re-read from the archive on
  every run by the since-deleted `_load_actual_prices`

Against `FLAT_TOLERANCE = 0.005`, the mismatch between a smoothed and an
unsmoothed price is on the same order as the band itself, so the up/flat/down
label flipped based on which day the scorer happened to run. Re-reading the
actual leg every run made the whole history mutable: any archive revision, or
simply a later run date, produced new numbers for old forecasts.

Both legs now go through one function — `backtest.price_resolution.resolve_anchors`
— with the same window, the same voting, and the same source on both sides. That
is the entire determinism guarantee.

### The estimator and its staleness cap

`SMOOTH_WINDOW = 3` mirrors production's `tail(3)` median. Every selected
observation must lie within `MAX_WINDOW_SPAN_DAYS` of the **anchor**, not merely
within that span of each other — measuring the span between the selected
observations alone leaves the gap from the newest observation to the anchor
unbounded, so an anchor far past the end of the archive would resolve to a
carried-forward price stamped as its own value. That is the exact
price-laundering shape commit `db5bddb` removed from the collector's historical
fallback, and `MAX_WINDOW_SPAN_DAYS` is derived from
`collectors.pipeline.FALLBACK_MAX_AGE_DAYS` so the codebase keeps one staleness
convention. Measured cost: ~0.74% of item-days.

### Disjoint leg windows, not just one post-forecast observation

Requiring the actual leg's *newest* supporting observation to post-date the
forecast is insufficient, because two 3-observation windows can overlap in 2 of
3 slots and a median decided by the shared observations scores an exact 0.0%
return no matter what the unshared observation did. With prices on
07-01/02/03 at 1.0 and 07-06 at 1.5, a forecast dated 07-05 targeting 07-08 has
base `median(07-01,02,03) = 1.0` and actual `median(07-02,03,06) = 1.0` — a 50%
move scoring as flat. The rule is therefore on the **oldest** supporting
observation, which makes the windows disjoint. In the dry run, dropping these
manufactured flats moved the actual-flat share from 31.8% to 48.1%.

## Frozen actuals

Resolution is now insert-only. Once an outcome has a `base_price`,
`actual_price`, and `resolved_at`, those three columns are final — the daily run
is structurally incapable of moving them. Two deliberate escape hatches:

- `--rescore` recomputes verdicts from the frozen actuals and never reads the
  archive, so a *scoring* fix lands without re-resolution.
- `--reresolve` re-reads the archive for the whole cohort and overwrites. It is
  the only path that may move a frozen actual, and the only one that deletes:
  rows whose forecast no longer resolves are removed from both the DB and the
  Parquet mirror.

The asymmetry is the point. A green daily run cannot silently restate history.

## Tiered reporting

72% of the evaluated universe is sub-$0.50, where one cent is a 20% move and the
direction label is dominated by tick quantisation. `HEADLINE_MIN_TIER` aggregates
tier ≥1 (≥$1) into the headline; tier 0 is reported **separately rather than
filtered out**, because "the model is worse on penny items" is a real question
the tier rows keep answerable. Tier 0 rows carry a `(tick-dominated)` marker in
the log line.

## Maturity is bounded by archive coverage, not the calendar

Found by running this backfill against prod, and the most consequential fix of
the set. Maturity was `target_date <= today` while resolvability is bounded by
the archive's last day. The archive always lags the calendar, so every forecast
in the lag window was mature-by-calendar and unresolvable-by-data: 30,859 of
80,737 prod forecasts entered the cohort as guaranteed misses and tripped
`MAX_UNRESOLVABLE_PCT` at **38.5%**, so the run reported nothing at all.

The gate was right; the maturity definition was wrong. `archive_max_day` reads
the horizon from Parquet column statistics (metadata only, no row scan) and the
cutoff is `min(today, coverage_end)` — clamped to `today` so a backfilled archive
holding future days cannot pull unmatured forecasts into a historical cohort.

**This was not merely a backfill inconvenience.** Because the archive lag is
permanent, the daily CI backtest would have failed this gate every single day
after the branch merged.

## Backfill of the historical outcomes

Two things had to be true before the gate would pass, and the second was a data
problem rather than a code one:

1. The maturity fix above, which took the unresolvable rate from 38.5% to 11.6%.
2. **A local archive gap.** `2026-07-22` was absent from
   `price-archive/prices-2026.parquet`, and the mature `3d / lgbm-v3-regime`
   cohort is dated 07-19 targeting 07-22. With no observation on the target day,
   the actual leg's window fell back to 07-19/20/21, its oldest observation
   equalled the forecast date, and the disjoint-window rule correctly rejected
   all 5,542 — 96% of the remaining unresolvable rows. The `cs2-oracle-data`
   repo's monthly `prices-2026-*.parquet` files are a strict superset of the
   single-file layout (every day present, plus 07-10/22/26/28/29/31), so the
   local archive was switched to the monthly layout, extending coverage from
   07-25 to 07-31.

The successful run: cohort 60,962, unresolvable **225 (0.4%)**, 60,737 outcomes
rewritten, 48 accuracy records stored, 2m28s.

### Before / after, prod

The historical series moved, and it moved *down*. These are not numbers that were
always there — the old ones were inflated by the two-estimator bug, and this is
the fix landing:

| horizon | model | n before → after | dir_acc before → after |
|---|---|---|---|
| 3d | lgbm-v3 | 11,054 → 11,009 | 54.99% → **48.72%** |
| 3d | lgbm-v3-global-only | 5,542 → 5,542 | 31.70% → **23.64%** |
| 3d | lgbm-v3-regime | 14,233 → 14,233 | 41.71% → **42.70%** |
| 7d | lgbm-v3 | 10,973 → 11,059 | 51.27% → **47.89%** |
| 7d | lgbm-v3-global-only | 5,542 → 5,542 | 31.67% → **25.32%** |
| 7d | lgbm-v3-regime | 5,542 → 5,542 | 36.32% → **36.12%** |
| 14d | lgbm-v3 | 10,976 → 11,040 | 46.09% → **40.15%** |
| 14d | lgbm-v3-global-only | 5,542 → 5,542 | 36.32% → **36.32%** |
| 30d | lgbm-v3 | 5,360 → 5,461 | 54.24% → **42.37%** |

With three labels, chance is ~33%. The ≥$1 headline for `lgbm-v3` lands at
48.4% / 49.4% / 50.8% / 46.7% across 3/7/14/30d — above chance, but nowhere near
the ~55–62% the old scorer reported. Anyone reasoning about model quality from
the pre-08-02 series was reading an artifact.

## Deliberately not fixed

- **`pct_error` is divided by the base leg, not the actual.** Explicit human
  ruling. It makes the column a percentage *of the starting price*, which is not
  what "percent error" usually means, but it is what every stored row already
  used and changing it would move the number for reasons unrelated to this fix.
  `_derive_verdict` carries the note.
- **43,905 orphan rows in the Parquet mirror.** Their `forecast_id`s do not exist
  in `forecast_outcomes` at all and every one has a NULL `base_price`, i.e. they
  are old-estimator vintage. `--reresolve`'s orphan deletion only sweeps rows
  within the considered cohort, so nothing removes these. The API reads Parquet
  first with DB fallback, which makes them *serving* data — deleting 43,905 rows
  of it is a call for a human, not a cleanup to slip into this entry.
- **14,233 stale DB outcomes.** Target date 2026-08-01, written by the last
  old-maturity daily run, NULL `base_price`. These are permanently contaminating:
  the freeze makes a normal daily run skip any forecast that already has an
  outcome, so their old-estimator values would never be corrected on their own.
  They need deleting; the attempt was blocked by a permission classifier and is
  left as a follow-up. `delete from forecast_outcomes where base_price is null`.

## Tests

313 pass. The load-bearing ones:

- `test_backtest_forecasts_reports_identical_metrics_across_an_archive_revision`
  — the determinism centrepiece: two full runs across an archive revision must
  report identical metrics
- `test_maturity_is_bounded_by_archive_coverage_not_the_calendar` —
  mutation-checked: reverting the cutoff to `today` fails it with the real 50%
  gate error
- `test_archive_coverage_does_not_extend_maturity_past_today` — the clamp
- `test_backtest_scores_the_base_leg_from_the_archive_not_current_price` — pins
  the root cause: `current_price` at 99.0 against an archive at 3.0→3.3 must
  score +10% up, not −97% down

## Operational notes

`alembic upgrade` cannot replay on SQLite — revisions `0001`→`0018` contain
Postgres-only `ALTER COLUMN ... TYPE` DDL. Rehearsing against a SQLite snapshot
requires `alembic stamp 0018` first. `0019`/`0020` themselves are dialect-aware
and apply cleanly to both.

## Related

- `docs/changelog/2026-07-31-accuracy-work-closed.md` — the roadmap closure this
  work sits outside of
- `docs/changelog/2026-08-01-forecast-pipeline-restored-and-collector-guards.md`
  — the CI incident fixed immediately before this
