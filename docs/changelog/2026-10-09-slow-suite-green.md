# 2026-10-09 — The `slow` suite is green again

CI runs only `-m "not slow"`, so 14 `slow` tests had been failing on main unnoticed. There
were three causes. All three were stale tests, and none was a product bug.

- **12 tests: `EXCEEDANCE_HEAD` now defaults to on.** `ad2d43e` (2026-09-19, "promote ML
  heads") changed the default to `"1"`. With the head on, the CV path raises on a frame that
  has no `target_exceed_{h}d` column, and that is the minimal frame these tests build. The
  failing tests were in `test_minimal_model_shape.py` (7), `test_cv_cohort_parity.py` (3) and
  `test_naive_init_score.py` (2). Each module now has an autouse `_exceedance_head_off`
  fixture, the same pattern `test_cv_row_cap.py` and `test_tied_anchor_cohort.py` already use.
  Setting `EXCEEDANCE_HEAD=0` alone was enough; `ANOMALY_GBM` made no difference. The
  docstring of `exceedance_head_enabled` still said "Off by default" and has been corrected.
- **1 test: the invariant-4 source guard was reading the wrong method.**
  `test_cv_results_publish_both_invariant_4_signals` inspected `_train_horizon_inline`, but
  `21eaf4e` had moved the aggregation block into `_aggregate_cv_metrics`. Every guarded
  fragment is still present in the new method. The test now checks that the caller delegates
  to it, and then checks the block itself.
- **1 test: there are now 14 harnesses.** #93 added `ab_test_break_aware_lookbacks`, and it
  passes every family check. Only the count guard (13) was out of date.

Full suite after the fix: 3,070 passed, 3 skipped.
