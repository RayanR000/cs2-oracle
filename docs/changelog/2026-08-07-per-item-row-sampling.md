# Per-item row sampling is built and costs nothing — because 78% of a retrain is conformal CV, which never sees it

**Date:** 2026-08-07
**Change:** `backend/models/forecaster.py` gains `_per_item_row_sample`; `_build_production_split`,
`_train_horizon_inline` and `train` gain `per_item_row_sampling: bool = False`;
`backend/scripts/forecast_prices.py` gains `_train_per_item_rows()` reading
**`TRAIN_PER_ITEM_ROWS`**, defaulted off. Production is byte-identical until the variable is
set.
**Bears on:** `2026-08-07-training-item-universe.md`, whose paired harness produced the
+5.72pp this sampler exists to express, and whose `ge1_budgeted` arm it was written to make
shippable.

The training-item-universe entry closed with `ge1_budgeted` — the arm that measured best —
recorded as something that **cannot be expressed in production today**. This builds the
sampler that expresses it, and then measures that doing so buys no wall clock and, worse,
cannot be read by the instrument every decision in this project is read off.

## The arm this was built for

From the paired harness (25–26 folds, 150 held-out ≥$1 items, `cluster_key="fold_id"`):

| arm | rows/fold | 30d paired |
|---|---|---|
| `ge1_budgeted` | 71K | **+5.72 [+2.91, +9.02]** |
| `ge1_full` | 728K | +3.50 [+1.56, +5.98] |

Ten times the rows losing at both horizons, which replicates the breadth harness's
`wide_unbudgeted` finding: what pays is item diversity per row, not row count. The blocker
was that every sampler in the training path selects **whole item histories**, so a 110K
budget under the price floor buys ~93 items at full depth, never 728 items at ~98 rows each.

## The change

`_per_item_row_sample(train_set, max_rows, seed=42)` gives every item an equal quota,
`max_rows // n_items`, drawn **uniformly at random within each item** rather than from its
tail — the property a `tail()` cap silently destroyed once already
(`2026-07-16-training-window-audit.md`). Items shorter than the quota keep everything. When
items outnumber the budget the one-row floor cannot be honoured for all of them, so
`max_rows` wins; it is a memory guard upstream.

It mirrors `scripts/ab_test_training_breadth.py::_stratified_sample`, deliberately: that is
the implementation the +5.72pp was measured on, so a divergence here would mean the
production knob and the measurement are different things.

`_build_production_split` routes its cap through it when asked and leaves the uniform
`train_set.sample(n=max_rows)` in place otherwise. **Only `train_set` is thinned.** Thinning
`val_set` would move the evaluation cohort, which is the artifact pairing exists to remove.

**Placement is a correctness constraint, not a preference.** Per-item sampling is safe only
*after* feature engineering. Thinning before `engineer_features` computes lags over a
punctured series; thinning before `prepare_targets` voids the label of any row whose
`date + horizon` partner was dropped, because the target is a date-keyed self-merge.

`TRAIN_PER_ITEM_ROWS` falls back **off** for unrecognised values, not on — a typo must not
silently change the training set.

## The instrumented run

One retrain: `TRAIN_MIN_MEDIAN_PRICE=1.0`, `TRAIN_FEATURE_ROWS=1_200_000`,
`max_rows=110_000`, sampling on, into an isolated `model_dir`. Scratchpad-only; nothing
written to `price-archive/`, `models/saved_models/` or the database.

The universe reproduced the reference run exactly — **926/5,542 items, 993,464 rows** — and
the sampler hit its target shape at every horizon:

```
Per-item row sample: 925 items x 118 rows -> 109,150/958,289 rows (budget 110,000)
```

That is `ge1_budgeted`'s geometry (728 items × ~98 rows) on the production path.

**Total: 540.9s (9.0 min), against the reference run's 538s for the same universe without
the sampler.** It costs and saves nothing.

| stage | time | share |
|---|---|---|
| conformal CV (222.9 + 92.9 + 38.1 + 66.1) | **420.0s** | **78%** |
| regime models, permutation, calibration, save | ~41s | 8% |
| `build_training_data` (`engineer_features` 14s, `fetch_price_history` 2s, cross-sectional 3s) | 44s | 8% |
| **production fits — the only thing the sampler thins** | **35.6s** | **6.6%** |

`fetch_price_history` was 2s on a warm voted cache; in CI that leg is a cold DuckDB read plus
multi-source voting over 6.1M rows, so the total understates a fresh Monday retrain.

## Two findings, the second the serious one

**1. No wall-clock benefit.** `_cv_evaluate_horizon` builds each fold as
`train_df = tdf[tdf["date"].isin(train_dates)]` — the full 992K-row frame, uncapped and
unsampled. The sampler thins the 36-second part of a 541-second run.

**2. The knob is close to unmeasurable by the model's own instrument.**
`mean_classifier_acc_ge1` comes out of CV, so with the knob on, CV still trains at full depth
and the reported accuracy does not describe the shipped configuration.

| h | reference (no sampler) | this run (sampler on) |
|---|---|---|
| 3d | 50.30 | 50.3 |
| 7d | 49.70 | 49.7 |
| 14d | 51.30 | 50.2 |
| 30d | 58.60 | 56.8 |

**3d and 7d are identical to the decimal.** 3d skips HP search entirely
(`SKIP_HP_HORIZONS`), so the only path from the knob into CV is Optuna's chosen
hyperparameters. Where 14d and 30d move they move *down*, but that is HP noise, not a
measurement of the sampler. This is the same trap the ByMykel bundle fell into
(`2026-08-06-bymykel-metadata-refuted.md`): a change the instrument cannot see reads as a
null no matter what it does.

## Tests

`backend/tests/test_per_item_row_sampling.py` — 31 cases: the equal quota on a skewed frame,
no item dropped, short items kept whole, the budget never breached, determinism and seed
reach, the calendar window surviving, `val_set` untouched, the flag defaulted off at all
three levels, and `TRAIN_PER_ITEM_ROWS` falling back off on a typo. **394 pass** across the
seven affected files.

## Still open

* **Extending the sampler into `_cv_evaluate_horizon`'s fold training.** The one change that
  would both make CV measure the configuration being shipped and attack the 420s that is 78%
  of the run. Not built, not approved.
* **Whether the +5.72pp survives the real sampler.** It was measured on the harness's
  simulated shallow draw, not this one. Until CV can see the knob, the production path cannot
  answer that.
* The knob is **off**, so nothing in the daily or Monday path has changed.

## Related

* `2026-08-07-training-item-universe.md` — the source of the paired arms, the fold schedule
  and the `ge1_budgeted` gap this closes half of.
* `2026-08-06-bymykel-metadata-refuted.md` — the prior case of a change the in-model
  instrument could not resolve.
