# 2026-10-09 — CI test gate runs in parallel, and tests never read the real archive

Performance review §3 (`research/2026-10-08-performance-review.md`).

## Measured (M4, the gate's own command and env guard)

| Run | Before | After |
|---|---|---|
| Serial, local | 86.9 s | 48.1 s |
| `-n 4`, `OMP_NUM_THREADS=1`, local | — | 20.5 s |

Both gates report 2,762 passed and 3 skipped.

## Changes

- **`_hermetic_archive` (autouse, `tests/conftest.py`).** Every `ItemForecaster` gets an
  empty temp `archive_dir`. The default is the real, gitignored `price-archive/`. CI has no
  copy, but locally `_attach_sidecars` read the real sidecars on every `engineer_features`
  call, so local and CI were running different code. That one read was 19 of the 21 s in
  `test_chunked_engineering_matches_whole_frame`, which now takes 0.45 s. Tests that need an
  archive still build one and assign `archive_dir` after construction.
- **`_fetch_supply_metadata` reads `self.archive_dir`.** It had hard-coded the same default
  path, so the fixture could not reach it. In production nothing changes, because
  `self.archive_dir` is that path.
- **`test_tied_anchor_cohort.py` fits each distinct CV frame once.** It used to run 8 fits,
  6 of them on identical inputs; now it runs 3. The fit is deterministic and the tests only
  read its output. Each test gets a deep copy.
- **pytest-xdist in the `dev` extra; the gate runs `-n 4` with `OMP_NUM_THREADS=1`.** One
  BLAS/LightGBM thread per worker, so 4 workers don't oversubscribe the runner. The gate's
  sqlite file (`/tmp/ci-gate.db`) is still 0 bytes after a run, so the workers can safely
  share it.
- **`lint.yml` also triggers on `.github/workflows/**`.** At least 6 test files read workflow
  YAML, and editing a workflow didn't run them.

## Not changed

- The `slow` suite has **14 failures on main**, before this change and after it, with the same
  test IDs, e.g. `test_naive_init_score.py` and `test_minimal_model_shape.py`. CI never runs
  `slow`, so nothing reported them. They are out of scope here.
