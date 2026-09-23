# The $1 training floor ships on determinism, not accuracy — and it took the row budget and the horizon cap with it

**Date:** 2026-08-08
**Step:** 7 of `docs/research/2026-08-07-next-steps.md` — now **DONE**
**Change:** `backend/scripts/forecast_prices.py` (`DEFAULT_TRAIN_FEATURE_ROWS`,
`TRAIN_HORIZON_MAX_ROWS`, new `DEFAULT_TRAIN_MIN_MEDIAN_PRICE`, `_train_min_median_price`);
`backend/models/forecaster.py::train` signature defaults;
`backend/tests/test_training_item_coverage.py`; `.claude/rules/training-budget.md`
**Suite:** 1,578 pass, 0 failures

Production trained on **99 items of 5,542** and served forecasts for 8,691. It now trains on
the **926-item ≥ $1 cohort with no subsample at all**.

| knob | was | now |
|---|---:|---:|
| `DEFAULT_TRAIN_MIN_MEDIAN_PRICE` | *(absent — env default `None`)* | **1.0** |
| `DEFAULT_TRAIN_FEATURE_ROWS` | 100_000 | **1_200_000** |
| `TRAIN_HORIZON_MAX_ROWS` | 700_000 | **1_200_000** |
| `ItemForecaster.train(max_feature_rows=…)` | 100_000 | **1_200_000** |
| `ItemForecaster.train(min_median_price=…)` | `None` | **1.0** |

## Why, and what the reason is *not*

**It is not an accuracy change.** The +3.50pp at 30d that originally motivated this step
re-derives to **+1.642pp [−0.809, +4.505], null**, inside a ±3–4pp item-draw noise floor
(`2026-08-08-per-fold-price-filter-rederived.md`). That number is retired and must not be
cited. Nothing here restores it.

The reason is **measurability, and it is a determinism argument**. `_stratified_item_subsample`
has a hardcoded `seed=42`; changing *only* the seed moves `mean_classifier_acc_ge1` by **sd
1.5–3.1pp** across 8 retrains, with the draws near-disjoint (Jaccard 0.010–0.026 — 2 to 5
shared items of 99). That is larger than every effect this project has tried to measure. At
the $1 floor the served cohort is **926 items / 993,464 item-days**, so a 1.2M budget covers it
whole and **the subsample never runs**. The variance is removed by construction rather than
shrunk.

The cost objection this default carried — "700_000 costs 4.5x the wall-clock, more than the
pre-rewrite 40-model grid" — was priced as a daily cost. `price-forecast.yml` sets `mode=full`
only when `date +%u` is `1`. It is a weekly cost.

## Three things the next-steps entry got wrong, found in shipping it

**1. The title said `≥ 1.0M`; 1.0M is the wrong number.** The cohort is 993,464 rows, so a 1.0M
budget leaves **0.7% headroom** before `_stratified_item_subsample` re-engages — and it would
then re-engage *over the filtered universe*, i.e. a smaller draw rather than no draw, which is
strictly worse than either knob alone. 1.2M is the value the 2026-08-07 instrumented run
actually used and the one that reproduced 926 items / 993,464 rows.

**2. `TRAIN_HORIZON_MAX_ROWS` had to move, and the entry never mentions it.** It is a
*different* knob — rows per horizon *after* feature engineering — and `forecast_prices.py`
deliberately held it above the feature budget so "changing coverage does not silently also
change each horizon's slice". On the ≥ $1 universe the per-horizon frame measures **958,289
rows** (`2026-08-07-per-item-row-sampling.md`), so the old 700K would have started binding the
instant the floor landed. Shipping the floor alone would have silently added a uniform
700K-row cap as a second, unmeasured treatment. Raised to 1.2M to keep it non-binding.

**3. R19 never gated this.** The entry says "Still gated by R19 (multiplicity deflation)" while
R19's own entry, updated the same day, says the positive it deflates no longer exists and it is
"bookkeeping over an all-null set rather than step 7's entry criterion". Both cannot hold. The
withdrawal resolves it: **a determinism argument has no test statistic to deflate.** R19 is
still worth doing; it is not a precondition for this. `next-steps.md` step 7 is corrected.

## The knobs are one setting, and the code now says so

The failure mode is setting one and not the other:

- **budget without floor** → 12x the wall-clock buying the pool's tier mix, 44% stickers and
  graffiti at a $0.03 median.
- **floor without budget** → a smaller item draw, not no draw.

So `ItemForecaster.train`'s own signature defaults moved too: a bare `train()` is now
production's configuration rather than a half-configured one. `build_training_data` keeps
`min_median_price=None` and `max_feature_rows=100_000` — every direct caller of it is a test or
a research harness picking its own universe deliberately, and defaulting a floor there would
change what those arms measure without touching their code.

`TRAIN_MIN_MEDIAN_PRICE`'s fallback semantics inverted with the default, deliberately:

| value | before | now |
|---|---|---|
| unset | no floor | **$1** |
| `"dollars"`, `""` | no floor | **$1**, with a warning |
| `0`, `-1` | no floor | no floor — the explicit escape hatch, with a warning |

While the default was `None`, an unparseable value could only fall back to "no filter" and was
indistinguishable from the default. Now it is the difference between the served cohort and the
pool, and falling back to the pool is the dangerous direction. A typo must not silently widen
the training universe.

## Verification

- **1,578 pass, 0 failures** on the full `tests/` suite (1,576 before; two tests added).
- Tests updated rather than deleted, and two added:
  `test_price_floor_default_matches_the_served_cohort` (a bare `train()` carries the floor, not
  just the budget) and `test_the_production_default_matches_the_served_floor`
  (`DEFAULT_TRAIN_MIN_MEDIAN_PRICE == api/serving_policy.py::MIN_SERVED_PRICE_USD`, so the
  training cohort tracks the served one by assertion rather than by coincidence).
- `test_the_per_horizon_cap_does_not_leak_into_coverage` was kept discriminating on purpose:
  with both budgets now at 1.2M a leak would be invisible, so it passes a deliberately
  different `max_rows=700_000`.
- **`engineered_data.parquet` is not at risk.** Checked because it is a ~2GB artifact excluded
  from the Actions cache: it is written on the **predict** path from `PREDICT_FETCH_DAYS`, not
  from the training budget, so 12x the feature rows does not grow it.

## Not done — read this before citing the change as verified

- **No retrain was run.** Not locally, not in CI. The floor first bites at the next Monday
  `mode=full`, and that run is the first real read on wall-clock and on
  `mean_classifier_acc_ge1`. The review's **"+7 min Monday-only"** is an estimate that has not
  been checked against this configuration; the job's timeout is 180 min, so there is headroom
  either way.
- **The fresh-model gate cannot detect any of this**, so a green run is not evidence the change
  did what it says. Read the training log's item/row counts — they should print 926 items and
  ~993K rows, with no `_stratified_item_subsample` line at all.
- **Stage 2 (the anchored per-fold filter) was not shipped**, deliberately. Stage 1 measured the
  look-ahead at **−0.004pp [−0.664, +0.851] at 30d** — real but not load-bearing — so it is a
  cleanliness change and gets its own reviewable diff.
  `docs/plans/2026-08-08-per-fold-price-filter.md` stage 2 stays open.
- **`scripts/paired_retrain_bymykel.py` now inherits all three defaults**, including the floor
  it never passed. That is a change to what a re-run of that harness measures. It tracks
  production by design and the ByMykel result is already refuted, so it was left alone — but a
  re-run is not comparable to the stored one.
- **The interaction with step 6 is not additive.** Step 6 voids labels on frozen price runs at
  ~39% of the whole universe against 13.8–19.2% on the ≥ $1 subset; removing the sub-$1 items
  outright cuts step 6's effect to roughly a third. Do not sum a before/after across both.

## Related

- `docs/changelog/2026-08-08-per-fold-price-filter-rederived.md` — the re-derivation that
  withdrew the accuracy claim and left only this justification
- `docs/changelog/2026-08-07-training-item-universe.md` — the 8-seed item-draw measurement, and
  the retired +3.50pp
- `docs/changelog/2026-08-07-per-item-row-sampling.md` — the 958,289-row per-horizon frame that
  forced the `TRAIN_HORIZON_MAX_ROWS` move
- `docs/changelog/2026-08-04-minimal-model-results.md` — the wall-clock budget this spends
- `.claude/rules/training-budget.md` — updated
