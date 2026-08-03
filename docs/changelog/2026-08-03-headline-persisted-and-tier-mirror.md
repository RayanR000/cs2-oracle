# The Headline Is Now Stored, and the Mirror Can Tell Tiers Apart (2026-08-03)

The ≥$1 headline quoted in `2026-08-01-deterministic-backtest.md` was wrong by
up to 8pp. Correcting it is the small part of this entry. The reason nobody
could catch it, and the serving bug found underneath, are the rest.

## The correction

| horizon | as published | actual | overstated by |
|---|---|---|---|
| 3d | 48.4% | **45.13%** | 3.27pp |
| 7d | 49.4% | **49.22%** | 0.18pp |
| 14d | 50.8% | **45.25%** | 5.55pp |
| 30d | 46.7% | **38.69%** | 8.01pp |

Two independent derivations agree to 0.01pp: the sample-weighted average of the
stored per-tier rows in `prediction_accuracy.parquet` (tiers 1–4, verified to
sum exactly to the all-tiers `sample_count`), and a direct recount of
`direction_correct` over the frozen `forecast_outcomes`. The **full-universe**
figures in that entry reproduce exactly — 48.72 / 47.89 / 40.15 / 42.37 — so the
data and the scoring were fine. Only the headline line was wrong.

Where the published numbers came from is unknown and now unknowable, which is
the actual problem.

## Root cause: the headline was the one row that wasn't stored

`_score_groups` computed the headline via `score_cohort(headline_records(...))`
and passed it **only to `logger.info`**. Every other cohort — each price band,
the all-tiers aggregate — was appended to `results` and persisted. The headline
existed solely as console output from a single run.

So the number quoted as "the model's accuracy" was the only number in the system
that could not be recomputed, audited, or diffed against a later run. A
mistranscription into a changelog was undetectable by construction.

`score_by_tier` now emits it as a real row under `HEADLINE_TIER = -1`, and
`_score_groups` logs *that row* rather than deriving a second copy — the logged
figure and the stored figure are the same object, so they cannot drift.

`HEADLINE_TIER` is negative by construction. Real bands are `0..4` and the
all-tiers aggregate is `NULL`, so the headline needed a third thing to be; a
negative sentinel can never collide with anything `price_tier()` returns.

**The tier rows do not sum to the headline, and must not.** `HEADLINE_TIER`
deliberately overlaps bands 1–4. `test_tier_rows_partition_the_all_row` excludes
it explicitly, and `test_headline_row_covers_exactly_the_tiers_at_or_above_the_minimum`
pins that it equals the sum of bands ≥ `HEADLINE_MIN_TIER`.

## The serving bug underneath

`prediction_accuracy.parquet` had **no `price_tier` column at all** — migration
`0019` added it to the table and `0020` put it in the unique constraint, but the
mirror predated both.

`_append_parquet` computed `common_cols` as the **intersection** of the incoming
frame's columns and the file's. `price_tier` was in the former and not the
latter, so every write silently dropped it. Worse, line 90 then filtered
`dedup_keys` to `common_cols` too, so the dedup anti-join lost the discriminator
as well: six rows per `(horizon, model)` that differ *only* in `price_tier`
became indistinguishable to both the reader and the dedup.

This is the same defect class `50a9ce0` fixed for `replace_rows` ("widen rather
than intersect on a mirror rewrite"). That fix was applied to the rewrite path
only; the append path kept intersecting. `_append_parquet` now takes the union,
NULLing whichever side lacks a column, and preserves file column order so a
served file's layout is not reshuffled by caller dict ordering.

### Why it was a serving bug and not just a reporting one

`api/routes/accuracy.py` read `SELECT * FROM prediction_accuracy` with no tier
predicate — and none was possible, the column being absent. All three endpoints
therefore served six overlapping cohorts as peers: four price bands, the
all-tiers aggregate, and the tick-dominated tier 0. `/latest` picked whichever
sorted first, i.e. an arbitrary tier.

The endpoints now take an optional `price_tier` (0–4, `-1` for the headline,
omitted for the all-tiers aggregate) and always resolve to exactly one cohort.
Defaulting to the aggregate preserves what callers got before `price_tier`
existed. When the mirror lacks the column, a request for a *specific* cohort
returns `None` and falls back to the DB rather than silently answering from an
unlabelled file.

## Required one-time step: relabel the existing mirror

**The code fix is not sufficient on its own.** The 84 rows already in
`prediction_accuracy.parquet` have no tier, so after widening they read as
`price_tier = NULL` — i.e. they masquerade as all-tiers aggregates, when five of
every six are really band rows. The default endpoint filter will over-select
them.

They cannot be relabelled from the mirror, but the **DB is authoritative and
correct** (0019/0020 applied in prod). `scripts/migrate_to_parquet.py` now
selects `price_tier` and includes it in the dedup keys, so re-running it for
this table rebuilds the mirror with the right labels.

Until that runs, treat `price_tier IS NULL` rows dated on or before 2026-08-02
as unlabelled, not as aggregates. Note that a plain `--rescore` will **not** fix
them: the new rows carry a non-NULL `price_tier` and so will not dedup against
the old NULL-tier rows, leaving both.

## What the corrected numbers imply

Against trivial constant baselines on the same ≥$1 cohort:

| horizon | model | best constant | model trails by |
|---|---|---|---|
| 3d | 45.1% | 56.0% (always "down") | 10.9pp |
| 7d | 49.2% | 54.2% (always "down") | 5.0pp |
| 14d | 45.3% | 55.4% (always "down") | 10.1pp |
| 30d | 38.7% | 64.4% (always "up") | 25.8pp |

A one-line constant rule beats the model at every horizon. At 30d the model
predicts "down" 82% of the time (816 of 990) into a cohort that rose 64% of the
time — that is an inverted model, not a weak one.

Caveat: part of the constant's edge is that this cohort spans a downtrending
stretch, and a constant that wins in-sample need not generalise. That explains
why the *baseline* is high; it does not explain the model sitting below it.

**Not addressed here.** Investigating the 30d inversion is the follow-up, and it
needs this entry's fixes first — the headline had to become auditable before any
bias correction could be shown to have helped.

## A refuted idea, recorded so it is not re-scoped

Widening the *predicted* flat band was tried first, on the theory that the model
almost never emits "flat" (0 flat predictions for `lgbm-v3` tier ≥1) while the
backtest reported a 48.1% actual-flat share. Sweeping the band from 0.5% to 20%
made accuracy **monotonically worse at all four horizons** (3d: 51.5% → 20.2%).

The premise was wrong. The 48.1% flat share is a **full-universe** figure driven
by penny items, where one tick is a 20% move. On the ≥$1 headline cohort the
actual flat share is **2.8%–8.5%**. There was no unclaimed mass to win, and
predicting flat more often only discards correct up/down calls.

## Tests

`tests/test_accuracy_tier_mirror.py`, 7 new:

- `test_append_adds_a_column_the_existing_file_lacks` — the regression, against
  a file written without `price_tier`
- `test_pre_existing_rows_survive_with_null_in_the_new_column`
- `test_a_column_the_new_rows_lack_is_not_blanked_on_survivors`
- `test_dedup_distinguishes_tiers` — rewriting tier 1 must leave tiers 0 and 2
  untouched, the property the intersection destroyed
- `test_score_by_tier_emits_a_headline_row` — headline covers exactly the ≥$1
  records and is scored independently of the all-tiers row
- `test_headline_sentinel_cannot_collide_with_a_real_tier`
- `test_headline_absent_when_no_records_reach_the_min_tier`

Three existing tests in `test_backtest_scoring.py` were updated rather than
relaxed: the partition invariant still holds with `HEADLINE_TIER` excluded, and
the `--rescore` shape test now expects it in both paths.

Full suite: 321 pass (313 before this work, plus 7 new and 1 added to
`test_backtest_scoring.py`).

## Files changed

- `backend/backtest/scoring.py` — `HEADLINE_TIER`; `score_by_tier` emits it
- `backend/scripts/backtest_accuracy.py` — log the stored headline row
- `backend/db/parquet.py` — `_append_parquet` widens instead of intersecting
- `backend/api/routes/accuracy.py` — `price_tier` param on all three endpoints
- `backend/scripts/migrate_to_parquet.py` — `price_tier` in select + dedup keys
- `backend/tests/test_accuracy_tier_mirror.py` — new
- `backend/tests/test_backtest_scoring.py` — updated for the sentinel
- `docs/changelog/2026-08-01-deterministic-backtest.md` — headline corrected

## Related

- `docs/changelog/2026-08-01-deterministic-backtest.md` — the work this corrects
- `50a9ce0` — the same widen-don't-intersect fix, on the rewrite path
