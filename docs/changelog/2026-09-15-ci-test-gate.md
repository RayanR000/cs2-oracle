# CI test gate (2026-09-15)

`lint.yml` ran ruff + import smoke tests only. `main` was red in ways neither
could see, all found while wiring the gate:

1. **Broken collection.** `test_phase_collapsed_universe.py` imported
   `is_phase_collapsed` from `models.forecaster`; a ruff `--fix` (76e462a)
   stripped it as an unused import. Fixed by re-adding it as a redundant-alias
   re-export (`is_phase_collapsed as is_phase_collapsed`), which ruff treats as
   intentional and never strips. `test_script_imports_have_no_side_effects.py`
   pins the related class below.
2. **Import-time env leaks.** `scripts/confirm_mondrian_oof.py` set
   `EXCEEDANCE_HEAD=1` (+ `FEATURE_NATIVE_NAN=1`) at import, and three
   `anomaly_*_ab` scripts set `ANOMALY_GBM=1` at import. Test modules import
   these scripts for pure helpers, so collection alone flipped CV gates for
   every later file: 25 failures (`KeyError: target_exceed_3d`) that passed
   per-file. Fixed by moving all such sets into `main()`. Rule: **scripts set
   env in `main()`, never at import.**
3. **All-zero sample weights.** `_compute_sample_weights` on a flat frame
   normalised to all-zero weights → constant model → `rank_ic None`. Fixed
   with a uniform fallback (production never reaches it; dead-item filtering
   drops flat items upstream).
4. **Stale structural pins (3).** Substring-count on module source broke when a
   comment mentioned the counted call (fixed with AST call counting);
   `pt_clf = (...)` refactored to a conditional expression (assertion now
   matches fragments); the q50-centre quiet assertion predates the
   CLIMATOLOGY_SCALE fallback warning (now scoped to recentre warnings).
   `test_build_training_data_restricts_to_price_group` asserted RSI survives
   the prune on a random walk, where return_14d↔rsi_14 correlate 0.98 — now
   asserts RSI is engineered, not selected.
5. **Single-item groupby.apply.** `_tmp.groupby(...).apply(...)` returns a
   (1, n) DataFrame for one group vs a Series for many → `ValueError` on any
   single-item frame. Fixed with an explicit per-group loop.

## The gate

Full suite is ~12 min (2,827 tests) — too slow for the 10-min lint job, so the
PR gate runs the fast subset:

- `pytest backend/tests -m "not slow" -q` (collection errors fail the run —
  that alone catches breakage class 1).
- Files marked `slow` (module-level `pytestmark`) train real boosters or build
  full frames: test_forecaster, test_minimal_model_shape, test_naive_init_score,
  test_predict_tail_truncation, test_cv_cohort_parity, test_clean_anchor_gate,
  test_shelved_compute_skipped_in_prod, test_scale_free_features (~8 min
  combined; measured 2026-09-15, re-measure after trainer changes).
- New tests that train or build full frames must carry the mark; the fast
  subset must stay under ~5 min.

## Known gaps (not gating yet)

- **Order dependence.** Several CV files fail in full-suite runs but pass
  per-file (env-leak class fixed; remaining instances un-firewalled). A
  `--randomly`-style shuffle seed in CI would surface these; until then the
  fast subset passing per-file is necessary but not sufficient.
- **Slow files are un-gated on PRs.** A PR touching only trainer code gets no
  test signal until the nightly full suite runs. Mitigation: the nightly
  `backtest-accuracy.yml` chain runs the full suite; slow-file failures page
  there, not here.
- **No DB-backed tests in the gate.** `backend/.env` points at production;
  tests must never import a live engine (see root AGENTS.md gotcha).
