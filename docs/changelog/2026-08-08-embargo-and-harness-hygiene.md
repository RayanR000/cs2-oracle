# The embargo was 13 days too narrow, the published gate never applied it, and ten A/B harnesses had no significance test

**Date:** 2026-08-08
**Plan:** `docs/research/2026-08-07-next-steps.md` step 5
**Change:** `backend/models/forecaster.py` (new `embargo_days`),
`backend/scripts/merge_price_primitives_ab.py` (pairs sharded arms), `backend/models/item_parser.py`
(`BID_SOURCES` moved here, new `bid_sources_sql_filter`, `archive_universe_sql_filter`),
`backend/db/archive.py` (`ingested_at`, `_COLUMN_TYPES` → `COLUMN_TYPES`),
`backend/backtest/paired_mde.py` (new `paired_metric_difference`, `verdict`,
`paired_arm_contrasts`, `format_paired`), `backend/backtest/walkforward_records.py` (new
`paired_records`, `fold_level_records`, `without_records`),
`backend/scripts/walkforward_backtest.py` (embargo default-on, `build_parser`),
`backend/scripts/append_to_parquet.py`, `backend/scripts/normalize_price_schema.py`,
`backend/scripts/compact_price_archive.py`, all 13 `backend/scripts/ab_test_*.py`,
`backend/tests/test_ab_harness_universe.py` (new), `backend/tests/test_ab_harness_verdicts.py`
(new), `backend/tests/test_walkforward_embargo.py`, `backend/tests/test_purged_production_split.py`,
`backend/tests/test_append_to_parquet.py`, `backend/tests/test_normalize_price_schema.py`.
**Suite:** 1,477 pass (`venv/bin/python -m pytest tests/ -q` from `backend/`), 0 failures.

**Nothing here was measured on data.** No harness was run, the published gate was not re-run
under its new default, and the archive migration has not been executed. Every A/B result stored
in this repo predates all five changes, and none of the new intervals has been observed once.
This entry records five corrections to the *instrument*, plus six defects a `/code-review`
pass found in the corrections themselves (§6). The numbers any of it will change are unknown
and are not estimated below.

## 1. The label is not a point observation, so the embargo is `H + 13`

`_purge_overlapping_train_rows` purged exactly `horizon` days before each validation window.
That is the right band only if the label at `d + horizon` is a single observation on that
date. It is not. Both legs resolve through `backtest/price_resolution.py::resolve_anchors`,
which medians the last `SMOOTH_WINDOW` observations and will admit one from up to
`MAX_WINDOW_SPAN_DAYS` before the anchor, and on the feature side `LAG_TOLERANCE_DAYS` lets a
lag lookup reach back past its exact date when the archive dropped that calendar day. The
label's support therefore runs 13 days past its nominal date, and a bare-`H` purge left that
carry inside the validation window.

New `models/forecaster.py::embargo_days(horizon)`. The 13 is **derived at call time, not
typed**:

```
carry = ItemForecaster.LAG_TOLERANCE_DAYS (3)
      + price_resolution.SMOOTH_WINDOW    (3)
      + price_resolution.MAX_WINDOW_SPAN_DAYS (7)
```

| horizon | old band | new band |
|---:|---:|---:|
| 3 | 3 | **16** |
| 7 | 7 | **20** |
| 14 | 14 | **27** |
| 30 | 30 | **43** |

`_purge_overlapping_train_rows` now applies `embargo_days(horizon)`; both `_compute_cv_splits`
callers in `forecaster.py` and all four harness callers (`interval_sampling`, `q50_sampling`,
`recency_weights`, `direction_labels`) pass it too.

**The h=30 consequence was not softened.** The embargo (43d) exceeds `VALIDATION_WINDOW_DAYS`
(30). That is the cost the plan called for: a 30d fold cannot be built out of 30 days of
history, which was always true and was previously hidden by a band too narrow to expose it.
`test_at_30d_the_embargo_exceeds_the_validation_window` asserts it so it is not later
mistaken for a bug.

**One caveat worth stating.** `MAX_WINDOW_SPAN_DAYS` is derived from
`collectors.pipeline.FALLBACK_MAX_AGE_DAYS`, which reads `os.environ["FALLBACK_MAX_AGE_DAYS"]`
with a default of 7. So the embargo width **follows that override** — setting it to 14 widens
every purge by 7 days. This is deliberate (one staleness convention across collection,
resolution and embargo, which is why `price_resolution.py` derived it that way in the first
place), but it means the embargo is configuration-dependent and the table above holds only at
the default.

## 2. The published gate was unpurged; `--purge` is now `--no-purge`

`walkforward_backtest.py` shipped the embargo opt-in on 2026-08-07 so that flipping it would
be a dated act rather than a silent one. It is now on by default: `run_walkforward(purge=True)`,
and the flag is `--no-purge`. `main()` was split so `build_parser()` is testable — a
`store_false` default on the published gate is not something to leave un-asserted.

**This is a deliberate discontinuity**, accepted on the grounds that an un-purged published
number is not a number worth continuity — so the series is **versioned across it**. A purged
run writes `model_version = "lgbm-v4-embargoed"`; `--no-purge` writes `lgbm-v3-clustered` and
reproduces the old split for a like-for-like read against a pre-flip run. This follows the
file's own precedent: `lgbm-v3-tuned → lgbm-v3-clustered` was bumped for exactly this reason
when the metric definition changed.

Caught in review, and worth stating as the failure it would have been: the first version of
this change left `model_version` alone and put `"purge"` only in the in-memory report. Purged
rows would have appended to the un-purged series, and the step would have surfaced on the
dashboard trend and to `backtest-triage` as a **model regression**, with nothing in the stored
data to say otherwise. The persisted `metrics` now carry `purge` and `embargo_days` as well,
so a row is self-describing without anyone having to know what the version strings mean.

**The size of the discontinuity is unknown.** No run was executed. The
`+12.1pp unpurged → +6.1pp purged` figure quoted by step 5 comes from the research review,
describes an event-calendar arm at h=30, **was never replicated in this repo**, and is **not
evidence about this gate**. Nothing here licenses a claim about how far the published Backtest
Accuracy number will move. The six new harness docstrings and
`tests/test_walkforward_embargo.py` say so explicitly where they cite it; the three docstrings
written on 2026-08-07 still present it as a local measurement and were left alone.

These are offline gate numbers in any case, not production DA, and are not comparable to it.

## 3. `ingested_at`: a column that can only fill forward

`CANONICAL_PRICE_COLUMNS` in `db/archive.py` gains `ingested_at`, typed `TIMESTAMP`. The
archive had no arrival timestamp anywhere while carrying backfilled series under ordinary
`day` values for 13 years — so a purge computed from `day` assumes a row dated `d` was knowable
on `d`, which a backfill writer violates by construction.

The column is **NULL for every row written before today and cannot be reconstructed backwards**.
A NULL means "arrival unknown" and must never be read as "arrived on `day`". Until it
accumulates, an embargo derived from `day` remains a lower bound on the true one.

- `append_to_parquet.py` stamps **the run's wall clock, not `--date`** — backdating `--date` to
  re-export an old day writes an old `day` with a present-day arrival, which is the truth.
- `_append_parquet` preserves **first** arrival across a re-append: a
  `groupby(dedup_keys).transform("min")` runs before the `keep="last"` dedup, because
  `keep="last"` is right for a corrected price and wrong for the timestamp — the corrected value
  still became knowable on the day it first landed. `min` skips `NaT`, so a pre-column row takes
  the new stamp, dating it *later* than the truth, which is the conservative direction for a
  leakage filter.
- `normalize_price_schema.py` now NULLs a materialised column to `COLUMN_TYPES[c]` rather than
  `VARCHAR` unconditionally. That was only ever correct while `source` was the sole column that
  could be missing; a `VARCHAR` NULL under a `TIMESTAMP` column makes the migrated file disagree
  with `prices_relation`'s own `CAST`. Its fingerprint gains `count(ingested_at)` and
  `max(ingested_at)`, for the same reason it already counted `source`: a rewrite must not invent
  an arrival time nor drop one.
- `compact_price_archive.py`'s orphan projection got typed NULLs too, so the UNION cannot
  resolve a column to `INTEGER` off an untyped NULL.
- `_COLUMN_TYPES` became public `COLUMN_TYPES`, since two scripts now need the same type map.

**The migration has not been run.** Only CI writes the canonical archive
(`aggregator-update.yml`, `normalize_schema = true`), so that is an operational step still
outstanding, not something this change performed.

## 4. The A/B harnesses were measuring a different market than the model they advise

All 13 `ab_test_*.py` harnesses globbed the Parquet privately, and **none of them applied
`BID_SOURCES` or `PHASE_COLLAPSED_SLUG_PATTERNS`**. An A/B therefore measured a different item
universe and a different price consensus than production trains on — while being the instrument
used to decide what production should do.

`BID_SOURCES` moved from `forecaster.py` to `models/item_parser.py`, joining
`PHASE_COLLAPSED_SLUG_PATTERNS`, with two new predicates: `bid_sources_sql_filter()` and
`archive_universe_sql_filter()` (both rules at once). `forecaster.py` re-exports it, so
`from models.forecaster import BID_SOURCES` is unaffected — the same re-export pattern the
phase-collapsed filter established yesterday.

**Twelve of the thirteen now apply it.** `ab_test_recency_weights` is the exception and applies
nothing: it reads a pre-built frame and makes no archive read at all, which
`test_the_no_archive_harness_really_reads_no_archive` pins so the exemption cannot rot into an
oversight.

**Three were the worst case: `regime`, `ensemble`, `supply_side`.** Each carried a raw
`read_parquet('prices-*.parquet')` glob, which DuckDB narrows to the *first file's* schema —
where `source` is not even a column — with no universe rule of any kind. The BUFF bid voted
straight into their price series. All three now read through `db/archive.py::prices_relation`,
which projects `source` correctly whether or not the archive has been migrated.

**Frame-cache fingerprints in five harnesses now include the universe predicate**
(`volume_features`, `price_primitives`, `csfloat_basis`, `item_metadata`, `training_breadth`).
The rule moved into a module their fingerprints do not hash, so without this a stale cached
frame would carry the old universe with nothing in the logs to say so.

**Still outstanding on those same three:** they read *every* ask source, so an item-day arrives
once per source and `engineer_features` collapses the copies with a plain mean where production
uses an outlier-voted median. The only citable size for that is the **1.37× duplicate
item-days** `backend/AGENTS.md` records; nothing in this pass measured a per-source count, and
the harness comments point at the 1.37× rather than inventing one. Narrowing the cohort would
change what these harnesses measure and was out of scope.

## 5. Nine harnesses roll their own folds; six had no embargo at all

Nine of the thirteen build folds by hand rather than through `_compute_cv_splits`. Three were
fixed on 2026-08-07 (`csfloat_basis`, `item_metadata`, `training_breadth`). The other six —
`volume_features`, `price_primitives`, `feature_contribution`, `supply_side`, `regime`,
`ensemble` — **had no purge gap of any kind**: not a narrow one, none. Their split was
`train = every date ≤ window_end - 1`. All six now call
`ItemForecaster._purge_overlapping_train_rows` on the train side only; purging the validation
side would shrink the harnesses' 21-day window and empty it at h=30.

**And ten of the thirteen had no significance test.** What they had instead:

| harnesses | old rule |
|---|---|
| `interval_sampling`, `q50_sampling`, `recency_weights` | "wins on at least half the folds" |
| `volume_features`, `price_primitives`, `feature_contribution` | pooled delta vs a ±0.5pp emoji threshold |
| `supply_side` | sign of a mean of per-horizon deltas → `IMPROVEMENT`/`DEGRADATION` |
| `direction_labels` | raw mean per cell, take the highest of ten |
| `regime`, `ensemble` | a bare `a > b` boolean |

None of those is a test. A fold win-count fires 50% of the time on two arms that differ only by
seed. A ±0.5pp threshold sits inside the measured fold-clustered item-level MDE of
**2.21–3.69pp** (`2026-08-07-training-item-universe.md`) by a factor of five. Picking the best
of ten cells guarantees a winner whether or not any cell differs from control.

All ten now report a paired, fold-clustered interval, and a `verdict()` that says `positive`
/`negative` only when the interval excludes zero, `null` when it does not, and `unresolved`
when the design could not produce an interval at all — which is a different statement from
"no effect" and must not print as one. Win counts are still printed, labelled **"context, not
a gate"**.

Supporting work, so the ten do not each grow their own copy:

- `paired_metric_difference` — generic over any per-row scalar, factored out of
  `paired_da_difference`. The DA wrapper's keys (`mean_diff_pp`, `mde_pp`, …) and arithmetic
  are unchanged; `test_the_da_wrapper_is_arithmetically_unchanged` pins that.
- `verdict()`, `paired_arm_contrasts()`, `format_paired()` in `paired_mde.py`.
- `paired_records()`, `fold_level_records()`, `without_records()` in `walkforward_records.py`.

**Two pairing grains, and the reason for the weaker one.** The DA harnesses pair at **row**
grain on the ≥$1 non-flat cohort (`MIN_SERVED_PRICE_USD = 1.0`, the population production
serves; an exactly-zero forward return is not a directional call). The three pinball harnesses
pair at **fold** grain via `fold_level_records`, because they shard folds across processes and
merge per-fold CSVs — the rows are gone by the time the arms meet. At ~8 folds that is ~8
clusters and the interval is wide. **That width is the honest cost of the design, not a defect
in the estimator**, and it is still strictly better than the win-count it replaces.

The `fold_id` passed everywhere is the loop's own `window_end`, not a running counter: a counter
drifts the moment one arm skips a fold the other kept, and `paired_metric_difference` would
then silently compare whatever intersection survived.

## 6. Five defects the review found in this change

A `/code-review` pass over the diff found five, all fixed here. Recording them because four
are the same failure shape the change is about — a wrong answer that looks like a confident
one.

- **The purge flip had no durable marker.** §2 above; the fix is the version bump.
- **`paired_arm_contrasts` caught bare `ValueError`.** `paired_metric_difference` raises it
  for two different things, and the module docstring is explicit that the missing-`cluster_key`
  guard must stay loud — swallowing it restores the 2026-08-07 under-dispersion bug. A harness
  that forgot to thread `fold_id` would have printed the fixed string
  `"no_shared_rows (arms were not measured on the same folds)"`: a confident, wrong diagnosis,
  with the real message stashed in a `detail` key `format_paired` discards. There is now a
  `NoPairedRows(ValueError)` and only that is caught.
- **`regime` and `ensemble` still headlined the boolean they refuted.** Both kept
  `Winner: REGIME` / `Winner: ENS3` as the *last* line of the final report — printed after
  the paired intervals — and persisted only `regime_wins` / `ens3_wins` to the DB. The paired
  verdict is now the headline and is stored flat (`paired_verdict`, `paired_mean_diff_pp`,
  `paired_ci_*`, `paired_n_clusters`); the boolean is printed beside it, labelled "a bare a>b,
  not a test".
- **A NaN bound read as `null`.** `verdict` separated `unresolved` from `null` on
  `lower is None` alone, and a NaN fails both `> 0` and `< 0`. One NaN row propagates through
  `np.percentile` to both bounds, so a gate would have printed `null` — "no effect" — for a
  comparison that produced no number at all. `np.isfinite` now guards it, and `format_paired`
  no longer prints `+nan` beside a confident-looking interval.
- **`paired_records` did not length-check `keep`.** Every caller derives the mask from the
  prediction arrays, which are *different objects* from `item_ids` (that comes off `val_df`).
  A short mask would not have raised; it would have scored a subset while `ids[i]` labelled
  rows the mask never described — the silent mispairing the function exists to prevent.

A sixth, half-fixed: under `--arm` sharding `volume_features` and `price_primitives` have one
arm per process and so cannot contrast in-process, and `without_records` strips the rows from
`--out`. They now say why the verdict is missing rather than omitting it silently, and
`merge_price_primitives_ab.py` pairs the shards at **fold** grain off the `per_fold` series
already in the file, keyed on `val_start` (a positional index would not identify the same fold
across arms). Its per-fold win count is retained, relabelled "context, not a gate". Row-grain
pairing across shards would need the records in `--out`, which is hundreds of thousands of
rows per arm; that was not worth it for a merge step.

## Verification

`venv/bin/python -m pytest tests/ -q` from `backend/`: **1,477 pass, 0 failures** — against
1,317 recorded by `2026-08-08-phase-collapsed-names-dropped.md`. The root `AGENTS.md` had gone
stale at 1,258 and now states 1,466.

- **`tests/test_ab_harness_universe.py`** (new, 44 cases): the predicates themselves; that no
  harness globs the archive unfiltered; that the one harness with no archive read really has
  none; that all 13 still parse.
- **`tests/test_ab_harness_verdicts.py`** (new, 61 cases): `paired_records` shape and JSON
  round-trip, a known shift recovered, the missing-cluster-key and no-shared-rows refusals
  preserved, the DA wrapper unchanged, `verdict` under both `higher_is_better` directions,
  `fold_level_records` pairing and resampling at fold grain, `without_records` at every depth,
  and two family-wide guards: every harness computes a paired interval, and none gates on a
  fold win-count. `TestTheGuardsFoundInReview` and `TestMergedShardsStillGetAVerdict` pin the
  six §6 fixes.
- **`tests/test_walkforward_embargo.py`** broadened from 3 harnesses to all 13, split into the
  nine that roll folds (must call the purge, unconditionally, against `val_dates[0]`) and the
  four that pass `purge_days` (must not pass the bare horizon). New
  `TestTheEmbargoWidthIsDerived` asserts the carry equals the sum of the three constants rather
  than the literal 13 — if any of them moves, the embargo must move with it.
- `tests/test_purged_production_split.py` now asserts `dropped == embargo_days(horizon)` and
  `== horizon + 13`; `tests/test_append_to_parquet.py` and
  `tests/test_normalize_price_schema.py` cover the first-arrival rule and the typed NULLs.

**No script was executed.** Per `backend/AGENTS.md`, anything run from `backend/` points at
production Supabase, and none of this needed a run to land.

## What was deliberately not done

- **No harness was run.** Every A/B result in this repo predates all five changes; none of the
  new intervals has been observed on real data. In particular the intervals' *widths* are
  design predictions, not measurements — the "~8 clusters is wide" claim above follows from the
  fold count, not from an observed CI.
- **The published gate was not re-run** under its new default, so the discontinuity's size is
  unmeasured. It will appear in the stored `lgbm-v3-clustered` series as a step of unknown
  magnitude at the first run after this lands.
- **The archive migration was not run.** `ingested_at` exists in the schema and in the writers;
  no canonical file has it yet.
- **The three raw-glob harnesses still read every ask source** (§4). Fixing the consensus would
  change what they measure, which is a separate decision.
- **No backwards reconstruction of `ingested_at` was attempted.** The information does not
  exist; a plausible-looking backfill would be worse than the NULL.
- **The `--no-purge` escape hatch was kept on `walkforward_backtest.py` only.** The nine
  research harnesses got no such flag — `test_the_purge_is_unconditional` forbids one. The gate
  keeps it because it has a stored series to read back against; a research tool does not.

## Still open

- **Everything in the section above that ends "not run."** Until a harness runs, this change has
  improved what the instrument *would* report, and nothing else.
- **The three published refutations still awaiting re-derivation under fold clustering**
  (CSFloat, ByMykel, breadth) now also predate the wider embargo and the universe filter, so
  they are further from citable than `2026-08-07-paired-mde-fold-clustering.md` left them.
- **The `paired_mde` run-to-run divergence is still unexplained** (~0.155pp between identical
  commands, ~19× the interval one of them reported). It applies to every interval added here.
- **The embargo follows `FALLBACK_MAX_AGE_DAYS`.** Nothing warns when that override is set, and
  a widened embargo silently costs folds.
- **Whether folds are independent.** Fold clustering is an improvement, not a proof; adjacent
  expanding-window folds share most of their training data.

## Related

- `docs/research/2026-08-07-next-steps.md` step 5 — the plan this implements, and the source of
  the unreplicated +12.1pp → +6.1pp pair
- `docs/changelog/2026-08-07-paired-mde-fold-clustering.md` — the fold-clustered bootstrap this
  extends to a generic metric, and the divergence caveat above
- `docs/changelog/2026-08-07-training-item-universe.md` — the 2.21–3.69pp fold-clustered MDE
  floor every "no test" claim in §5 is measured against
- `docs/changelog/2026-08-08-phase-collapsed-names-dropped.md` — its "Still open" item about the
  unfiltered `ab_test_*` loaders is what §4 closes
- `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md` — `BID_SOURCES` and the
  NULL-safety trap both filters guard
- `docs/changelog/2026-08-07-archive-schema-and-keys.md` — `prices_relation` and the canonical
  column set `ingested_at` joins

## Docs touched

None besides this entry. `docs/research/2026-08-07-next-steps.md`, `docs/README.md`,
`AGENTS.md` and `backend/AGENTS.md` were updated outside this pass; `docs/architecture/` was
deliberately left alone here.
