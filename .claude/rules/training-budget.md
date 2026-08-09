---
paths:
  - "backend/models/forecaster.py"
  - "backend/scripts/{forecast_prices,evaluate_forecaster}.py"
---

# What a retrain actually sees

- **The budget and the price floor are one setting, and since 2026-08-08 production runs
  both.** `DEFAULT_TRAIN_FEATURE_ROWS = 1_200_000` and `DEFAULT_TRAIN_MIN_MEDIAN_PRICE = 1.0`
  (`scripts/forecast_prices.py`), mirrored on `ItemForecaster.train`'s own defaults so a bare
  `train()` is production's config. The floor restricts the universe by median price *before*
  `_stratified_item_subsample`, and at that budget the **926-item / 993,464-row ≥ $1 cohort**
  fits whole — **so the subsample never runs**. That is the entire justification: the
  subsample's hardcoded `seed=42` moves `mean_classifier_acc_ge1` by **sd 1.5–3.1pp** (8
  seeds, 2026-08-07), and no draw is the only setting that removes that variance rather than
  shrinking it. **It is not an accuracy claim** — the +3.50pp at 30d re-derives to +1.642pp
  [−0.809, +4.505], null (`docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`).
  Changing one knob alone is the error: the budget without the floor spends 12x the
  wall-clock on the pool's tier mix, the floor without the budget leaves a smaller draw.
  `TRAIN_HORIZON_MAX_ROWS` was raised to 1.2M with them to stay non-binding — the per-horizon
  frame is 958,289 rows on this universe, so the old 700K would have bound. Escape hatch:
  `TRAIN_MIN_MEDIAN_PRICE=0`. An unparseable value keeps the floor, deliberately. The
  fresh-model gate still cannot detect any of this. See
  `docs/changelog/2026-08-08-training-price-floor-shipped.md` and
  `docs/changelog/2026-08-07-training-item-universe.md`.
- **`TRAIN_PER_ITEM_ROWS` changes what the per-horizon cap spends the budget on.** Default
  off, where `_build_production_split` caps an oversized slice with a uniform
  `train_set.sample(n=max_rows)` and an item's share of the sample is its share of the
  rows. Set it and the cap draws an **equal quota per item** instead
  (`_per_item_row_sample`), which is the axis the paired harness measured: 71K rows/fold
  spread over 728 items beat 728K rows/fold at **+5.72pp vs +3.50pp at 30d**. Only
  `train_set` is thinned — never `val_set`. It is safe **only after feature engineering**:
  thinning earlier computes lags over a punctured series, and thinning before
  `prepare_targets` voids the label of any row whose `date + horizon` partner was dropped.
  It does **not** reduce feature-engineering time, which is what a wide
  `TRAIN_FEATURE_ROWS` actually buys.
- **Training is fully sequential.** Horizons, quantiles, and ensemble members train one at
  a time; LightGBM's OpenMP threads supply the CPU parallelism. Ensemble members get
  `n_jobs = max(1, cpu_count // 2)`; the Optuna search params still use `n_jobs: -1`.
- **Size/speed levers are already documented.** See `docs/architecture/model-optimization.md`
  for the options that retain ≥90% quality — don't re-derive them.
