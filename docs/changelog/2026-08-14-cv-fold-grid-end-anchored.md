# CV fold grid end-anchored — item 2

**Date:** 2026-08-14
**Item:** 2 of `docs/research/2026-08-13-next-steps.md`, closing the confound in
`2026-08-13-cv-folds-are-not-time-aligned-with-serving.md`.
**Prereq read:** item 1 (`2026-08-13-c1s-cv-edge-is-not-a-pre-2026-artefact.md`) — done, so the
before/after is interpretable.

## Change

`ItemForecaster._compute_cv_splits` strode forward from `CV_MIN_TRAIN_DAYS`, so the last
validation window landed up to `CV_STEP_DAYS − 1 = 149` dates short of the frame end. For a
1.2M-row retrain that put the 7d/14d/30d final fold's validation window ending ~2026-02-09 while
the served arms are read on anchors 2026-04-18 … 2026-06-08 — the two legs shared no dates.

The grid is now **end-anchored**: the most recent fold validates on the final
`VALIDATION_WINDOW_DAYS` (30) dates, and earlier folds step back by `CV_STEP_DAYS`. Walking
backward from `len(sorted_dates) − val_window` down to `min_train`.

## What it costs and what it moves

- **Fold count is unchanged** (not +1): the backward walk yields the same number of `step`-spaced
  folds the forward walk did, just shifted to end on the frame. So the ~1% training-cost estimate
  in the next-steps doc is an over-estimate — this is ≈ free.
- **It moves two served quantities, neither of them accuracy:** `q_hat` is calibrated on the
  out-of-fold residuals these windows produce, so the served band width changes; and the PT
  sample changes, so `invariant_4_signal` must be **re-read** on the next retrain, not carried.
  Both are the point — a `q_hat` fitted on a window ending 2026-02-09 and served in 2026-08
  carried the same misalignment.
- **Takes effect on the next retrain only.** No cache-version bump is involved (the CV grid is
  not part of `VOTED_CACHE`).

## Tests

`tests/test_forecaster.py::test_cv_last_fold_validates_on_frame_end` (new, was RED: the old grid
ended 2026-01-15 on a 500-date frame ending 2026-05-15). Existing CV, purged-split and
walk-forward-embargo suites stay green (`test_forecaster -k cv`,
`test_purged_production_split`, `test_walkforward_embargo`: 5 + 92 pass).

## Not done here

The band-width and PT re-read against the new grid needs a retrain to produce; this change only
makes the grid correct. Reading `q_hat`/`fold_q_hat` and `invariant_4_signal` off the next
`meta.json` is the follow-up.
