# Climatology band scale implemented, gated off (`CLIMATOLOGY_SCALE=1`)

Builds the scale the null test found beats the GBM band by 43-47% at matched coverage
(`docs/research/2026-08-19-climatology-vs-gbm-band.md`,
`docs/superpowers/specs/2026-08-19-climatology-band-scale.md`). Off by default; the served
headline stays gated by `MIN_FORECAST_DATES=20`.

## What (`models/forecaster.py`)

- `climatology_scale_enabled()` / `_climatology_scale_served()` — the flag and its artifact-follow
  serving switch, mirroring `EXCEEDANCE_SCALE`.
- `_build_climatology_table` (pure) — per-item std of h-day returns, shrunk toward the price
  tier's pool by `n/(n+K)` (`CLIMATOLOGY_SHRINK_K=20`); returns `(item table, tier pool, global)`.
- `_fit_climatology_scale` — builds the table from the training frame and returns the per-record
  scale, calibrated against via conformal's existing `learned_scale` path (no `conformal.py`
  change). `_climatology_lookup` / `_climatology_scale_for_rows` serve it, tier-pool fallback for
  unseen items, global fallback for unseen tiers.
- `_calibrate_conformal` — climatology takes precedence and is **mutually exclusive** with
  `SIGMA_EXPONENT` / `LEARNED_SCALE` / `EXCEEDANCE_SCALE` (raises if combined).
- `meta.json` persists `climatology_scale` (bool) + `climatology_scale_tables`; the load path
  rebuilds them with int horizon/tier keys. Same matched-pair rule as `conformal_beta`: never
  difference a q_hat across this flag — compare COVERAGE and WIDTH.

## Not done (intentional)

- No `MODEL_ARTIFACT_VERSION` bump: flag off ⇒ artifact byte-identical to before.
- The per-item table is static as-of the train cutoff (updates on retrain). The validated
  prototype was static and still won; a predict-time archive read (daily update) is a later
  refinement.
- **The A/B is the decision:** run the `climatology_scale` arm and read width-at-matched-coverage
  on the SERVED panel (`scripts/replay_serving.py`), not CV — that is what settles the production
  number. Model-diagnostics arm wiring is the remaining step.

## Tests

`tests/test_climatology_scale.py` — table math + shrinkage, lookup fallbacks, flag gating,
mutual-exclusion raise, artifact-follow serving, meta round-trip. 12 passed; conformal /
signed-band / serving suites green (the one `test_forecaster` RSI failure is pre-existing, from
the 2026-08-18 dead-weight shelving, and reproduces without this change).
