# 2026-09-18 — Multi-head champion–challenger backend: built, collecting shadows

**Status:** SHIPPED as a framework. No promotion, no serving change.
Production output is byte-identical with shadow mode enabled.
**Spec:** `docs/superpowers/specs/2026-09-18-multi-head-champion-challenger-design.md`.
**Plan:** `docs/superpowers/plans/2026-09-18-multi-head-champion-challenger.md` (Tasks 1–12).

## Production / shadow state

| Component | Production | Shadow | Routine training |
|---|---|---|---|
| Centre | `gbm_q50` (all horizons) | `last_price` | GBM required |
| Interval | signed conformal + climatology | same offsets on both centres | existing path |
| Anomaly | `anomaly_gbm_v1` | none | enabled |
| Exceedance | `exceedance_gbm_v1` | none | enabled |
| Ranking | none | `lambdarank_v1` | enabled for shadow collection |
| Direction | API-neutral | offline benchmark only | disabled |

Champion selection is explicit per-horizon code configuration
(`models/centre_policy.py::CENTRE_CHAMPIONS`, initially all `gbm_q50`).
Evaluation produces evidence; a reviewed configuration change promotes a
candidate. Promotion is manual-only: no report, scorer, workflow step, or
database row can change the champion mapping.

## What landed

- **Date gates split** (`backtest/scoring.py`): `MIN_HEADLINE_DATES = 20`
  for publishable claims, PT headlines, promotion gates and transfer
  conclusions; `MIN_FEEDBACK_DATES = 8` for operational served-width
  feedback only. Served recalibration defaults to the feedback floor.
- **Contracts, policy, equal geometry** (`models/prediction_contracts.py`,
  `models/centre_policy.py`, `models/forecast_assembly.py`): frozen typed
  outputs; applying one signed offset set to two centres yields identical
  percentage widths.
- **Shadow tables** (migration `0027_add_forecast_candidates`):
  `forecast_candidates` (stable identity + config fingerprint, whole-batch
  atomicity) and immutable `forecast_candidate_outcomes`.
- **Serving**: `predict()` captures exact challenger triples after final
  sanitization onto transient `pending_candidates`; `forecast_prices.py`
  persists them after the production write (production-first: integrity
  conflicts fail the run after production is stored, transient DB errors
  leave batches absent for retry). Public schemas unchanged; `rank_score`
  appears in neither `ItemForecast` nor `api/schemas.py`.
- **Resolution** (`backtest/candidate_resolution.py`): reuses the canonical
  resolver and all production rules; prefers frozen production legs so both
  arms share byte-identical outcomes; mismatch raises; `--reresolve` is the
  only path that moves frozen legs; `--rescore` refreshes derived metrics
  without touching the archive.
- **Evidence** (`backtest/candidate_scoring.py`, `backtest/promotion.py`):
  equal-weight-per-date paired bootstrap (seed 42, 90% CI); promotion needs
  20 shared dates, ≥95% row overlap, single version/fingerprint, ≥80%
  batch completeness, no resolution mismatch, MAE CI entirely below zero
  with point < 0, coverage regression ≥ −2pp, width delta ≤ 0.001pp.
- **Reports** (read-only, JSON + Markdown):
  `venv/bin/python -m scripts.centre_promotion_report --horizon {3,7,14,30,all}
  --json-out PATH --markdown-out PATH` (exit 0 pass, 1 unresolved/
  insufficient, 2 integrity/execution, 3 rejected) and the same shape for
  `scripts.ranking_transfer_report` (SUPPORTED leaves the score shadow-only).
  Published as `candidate-reports` workflow artifacts after backtest
  resolution; below-gate horizons are successful artifacts reading
  INSUFFICIENT_EVIDENCE.
- **Direction offline**: routine training fits, persists, restores and
  consults no direction booster; legacy `clf_*.txt` files are ignored on
  load; the API stays neutral. Reproducibility lives in
  `venv/bin/python -m scripts.direction_benchmark --horizon 7
  --archive-dir ../price-archive` (production universe/features, fixed
  rounds, horizon+13 embargo; reports PT with DA, realised down rate and
  constant-call accuracy together).
- **Ops**: per-horizon shadow logs (champion, challenger, rows, fingerprint
  prefix); `forecasts_written` joins the zero-row guard (production zero
  stays fatal); shadow gaps warn; freshness reports collection readiness.

## Deliberate deviations from the plan

1. **Load-time centre check warns instead of raising.** A strict raise
   broke legitimate partial-artifact round-trips
   (`test_single_member_ensemble_round_trips_through_disk`,
   `test_load_ignores_stale_extra_ensemble_members_on_disk`).
   `_require_centre_artifacts()` returns the missing horizons and
   `load_models()` logs them loudly; an empty dir stays on the retrain path.
2. **No `price-forecast.yml` change.** `RANKING_HEAD=1` is kept; the
   `CV_DIAGNOSTIC_CLASSIFIER` variable stays because the retained CV
   diagnostic path still reads it; promotion has deliberately no workflow
   step. The report/upload steps went to `backtest-accuracy.yml`, after
   backtest resolution.
3. **Two plan-named regression files do not exist**
   (`test_price_resolution.py`, `test_backtest_deterministic.py`);
   `test_backtest_resolution.py` is the equivalent and is green.

## Evidence clock

Both reports are expected to read INSUFFICIENT_EVIDENCE until 20 shared
dates. Collection starts at the first complete shadow date after this
merge — not before. Do not change `CENTRE_CHAMPIONS` in the implementation
branch; the first promotion is a separate reviewed change after one
horizon returns `PASS_FOR_MANUAL_PROMOTION`.

## Verification state

Full backend suite at merge: 2929 passed; 5 failures in
`test_event_correlation_prices.py`, all one root cause — the fixtures seed
a 2026-05-20 event and analyse a 120-day window, which no longer reaches
it from the September calendar (`no_events_in_window` instead of the
expected status). Pre-existing calendar drift, untouched by this change.
