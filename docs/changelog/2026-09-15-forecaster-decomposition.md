# Forecaster decomposition (2026-09-15)

`models/forecaster.py` is 11,805 lines / ~220 methods on one class. Ruff cannot
see cross-function invariant violations, tests cannot isolate clusters, and any
refactor is high-risk. The split strategy is **extract pure functions behind
`staticmethod` aliases — never a big-bang move, never call-site churn**.

## The recipe (proven by cut 1)

1. Pick methods with no `self`/`cls`/DB/env reads (verify by reading, not by
   decorator — `_blend_returns_with_prior` was an undecorated instance method
   that never touched self).
2. Move bodies verbatim to `models/<cluster>.py` under public names; module
   constants they read move with them (re-exported from forecaster with the
   redundant-alias form so ruff never strips them).
3. Replace each method with one alias line:
   `_x = staticmethod(direction.x)`. Every `self._x`, `ItemForecaster._x`,
   and `from models.forecaster import X` keeps working — zero call-site diff.
4. Prove equivalence BEFORE wiring: randomized input/output comparison of old
   vs new (cut 1: all 15 identical), then the direction-heavy test files,
   then the fast gate.

## Cut 1 (shipped): models/direction.py — 384 lines, 15 functions + 1 constant

Direction bucketing, recentring, rank-IC, and band-centre math:
`directional_accuracy`, `direction_classes`, `demean_returns`,
`has_date_coverage`, `direction_threshold`, `served_cohort_multiplier`,
`recenter_on_momentum`, `recenter_on_direction`, `fix_quantile_crossing`,
`blend_returns_with_prior`, `direction_records`,
`direction_records_from_classes`, `summarise_rank_ic`, `within_date_rank_ic`,
`within_date_rank_ic_detail`, plus `DIRECTION_FLAT_TOLERANCE_PCT`.
New code imports `models.direction` directly.

## Sequenced next cuts (each independently shippable, in dependency order)

1. **Env-gate flags** (~40 `*_enabled`/`*_served` one-liners): move to
   `models/flags.py` as plain functions; alias the same way. Mechanical,
   near-zero risk. Unblocks reading the class without the flag Mesa.
2. **Conformal record builders** (`_conformal_records`,
   `_holdout_conformal_records`, `_calibration_returns`, `_sigma_for_rows`):
   pure frame→records transforms; the calibration core stays until the band
   geometry settles (it moves every few weeks — do not extract a moving
   target).
3. **CV mechanics** (`_compute_cv_splits`, `_purge_overlapping_train_rows`,
   `_choose_validation_split`, `_build_production_split`, `_cv_can_run`):
   date logic with no model state; directly unit-testable once free.
4. **Feature selection** (`_select_feature_cols`, `_apply_feature_allowlist`,
   `_prune_features`, `_reduce_feature_cols`, `_skipped_feature_groups`):
   frame→column-list transforms. Large test payoff (the RSI prune incident).
5. **Artifact persistence** (`save_models`/`load_models`/version guard):
   last, because every cut above must keep the persisted field contract —
   see `VOTED_CACHE_VERSION` and the artifact-version comment discipline.

Explicitly NOT sequenced: `train`, `_train_horizon_inline`, `predict`,
`_cv_evaluate_horizon` — stateful orchestrators that stay on the class until
the clusters they call have moved. Splitting the orchestrator first inverts
the dependency and doubles the seam count.
