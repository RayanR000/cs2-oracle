# 2026-09-15 — q_hat dispersion harness built (Item 3, Phase A)

**Orders:** `docs/specs/2026-09-14-wait-window-workplan.md` Item 3.
**Prereg:** `docs/research/2026-09-14-qhat-dispersion-fold-design-preregistration.md`
(unchanged — the bars below are its, not fitted to any number).

## Built, not run

- `backend/scripts/measure_qhat_dispersion.py` (+
  `tests/test_measure_qhat_dispersion.py`, 15 tests): one shared training
  frame, then production CV per arm per horizon reusing
  `confirm_mondrian_oof.train_oof` verbatim. Arms: control (150/42),
  placebo (150/seed-b), dense (75/42), sparse (300/300-seed-42). Scores
  bars 1-2 (per-horizon q_hat CV, max/min, placebo stability); bar 3
  (matched-width replay) is Phase B and runs only for a 1-2 passer —
  the script prints the exact commands then.
- `CV_ROW_SEED` override in `forecaster.py` (`_cv_row_seed()`,
  `_sample_cv_train_rows()`): the placebo varies ONLY the 300K-cap row
  draw on an identical grid. Default 42 keeps every existing artifact
  byte-identical; booster seed stays 42 on all arms. Pinned by 4 new
  tests in `tests/test_cv_row_cap.py` (incl. seed-moves-rows-never-cohort).

## Two deliberate deltas from the prereg's letter

1. No Optuna / full production fits: folds train with the SKIP_HP
   defaults, common to all arms. Re-tuning per arm would add HP noise to
   a grid read; only the treatment-vs-placebo contrast transfers, and
   absolute q_hat levels are NOT comparable to production (same caveat
   as the Mondrian confirm's DELTA 1).
2. Bar 3 phased: no served artifacts exist until Phase A names a
   candidate, so no replay runs yet.

## Status

Phase A unrun. Run one arm at a time from `backend/` (read-only fenced
DB session, same footing as the confirm):
`venv/bin/python -m scripts.measure_qhat_dispersion --arms control`.
