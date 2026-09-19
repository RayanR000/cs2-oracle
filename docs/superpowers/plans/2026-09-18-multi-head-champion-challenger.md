# Multi-head Champion–Challenger Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]` syntax) for tracking.

**Goal:** Add a production-safe shadow framework that compares the served GBM centre with last-price, evaluates LambdaRank on served outcomes, retains anomaly/exceedance as production ML outputs, and removes the failed direction classifier from routine training.

**Architecture:** Keep `item_forecasts` and public APIs unchanged. Derive centre candidates from the final served GBM triple, persist exact shadow predictions in dedicated tables, resolve them with the canonical price resolver, and generate read-only promotion/transfer reports. Champion selection is explicit per horizon and changes only through reviewed code.

**Tech Stack:** Python 3.11/3.13, pandas, NumPy, LightGBM, SQLAlchemy, Alembic, PostgreSQL/SQLite, DuckDB, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-multi-head-champion-challenger-design.md`

## Global Constraints

- Run backend commands from `backend/` through `venv/bin/python`.
- `backend/.env` points at production; tests must inject SQLite or pure frames and must never instantiate the production engine accidentally.
- Read archive prices through `db/archive.py::prices_relation`; candidate resolution must reuse `backtest/price_resolution.py`.
- Preserve the `horizon + 13` embargo in the offline direction benchmark.
- Directional accuracy is never reported without Pesaran–Timmermann, realised down rate, and hindsight constant-call accuracy.
- The independent statistical unit is `forecast_date`, not item row.
- `item_forecasts` and public API schemas do not gain shadow fields.
- No report, scorer, or database row may mutate centre champion configuration.
- Candidate intervals reuse the final production interval's percentage offsets.
- Do not remove an environment feature flag without a retrain in the same PR.
- Existing anomaly and exceedance semantics remain unchanged.
- Centre promotion requires 20 shared resolved dates and manual review.
- Ranking remains shadow-only even after a positive transfer report.
- Use TDD for every task and commit after each reviewable deliverable.

---

### Task 1: Separate headline sufficiency from feedback activation

**Files:**
- Modify: `backend/backtest/scoring.py:145-159`
- Modify: `backend/models/served_recalibration.py`
- Modify: `backend/models/direction.py`
- Modify: `backend/models/forecaster.py`
- Modify: `backend/api/routes/accuracy.py`
- Modify: `backend/scripts/backtest_accuracy.py`
- Modify: `backend/scripts/check_outcome_dates.py`
- Modify: executable scripts returned by `rg -l 'MIN_FORECAST_DATES' backend/scripts`
- Modify: tests importing `MIN_FORECAST_DATES`
- Test: `backend/tests/test_date_threshold_policy.py`

**Interfaces:**
- Produces: `MIN_HEADLINE_DATES = 20`
- Produces: `MIN_FEEDBACK_DATES = 8`
- Removes: `MIN_FORECAST_DATES`
- Headline/PT/bias/reporting use the headline floor; only served-width feedback uses the feedback floor.

- [ ] **Step 1: Write routing tests**

Create `tests/test_date_threshold_policy.py`:

```python
import inspect

from backtest import scoring
from models import served_recalibration


def test_threshold_values_are_separate():
    assert scoring.MIN_HEADLINE_DATES == 20
    assert scoring.MIN_FEEDBACK_DATES == 8
    assert not hasattr(scoring, "MIN_FORECAST_DATES")


def test_served_feedback_defaults_to_feedback_floor():
    sig = inspect.signature(served_recalibration.served_coverage_factors)
    assert sig.parameters["min_dates"].default == scoring.MIN_FEEDBACK_DATES


def test_scoring_uses_only_the_headline_floor():
    source = inspect.getsource(scoring.score_cohort)
    assert "MIN_HEADLINE_DATES" in source
    assert "MIN_FEEDBACK_DATES" not in source
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
venv/bin/python -m pytest tests/test_date_threshold_policy.py -q
```

Expected: FAIL because the two constants do not exist.

- [ ] **Step 3: Define and route the constants**

Replace the old constant in `backtest/scoring.py`:

```python
# Publishable claims, PT headlines, promotion gates, and transfer conclusions.
MIN_HEADLINE_DATES = 20

# Operational served-width feedback only.
MIN_FEEDBACK_DATES = 8
```

Route imports as follows:

- `served_recalibration.py` defaults to `MIN_FEEDBACK_DATES`.
- `scoring.py`, `direction.py`, `forecaster.py`, `backtest_accuracy.py`, accuracy API, centre/ranking reports, and publication scripts use `MIN_HEADLINE_DATES`.
- `check_outcome_dates.py` imports `MIN_HEADLINE_DATES` rather than restating a number.
- Served-recalibration fixtures use `MIN_FEEDBACK_DATES`.
- Preserve the existing API key `min_forecast_dates` if compatibility requires it, but return `MIN_HEADLINE_DATES`.

- [ ] **Step 4: Prove the ambiguous constant is gone**

```bash
rg -n "MIN_FORECAST_DATES" backend
```

Expected: no executable import/comparison remains; historical prose in archived research scripts may be rewritten for clarity.

- [ ] **Step 5: Run threshold regressions**

```bash
venv/bin/python -m pytest \
  tests/test_date_threshold_policy.py \
  tests/test_served_recalibration.py \
  tests/test_served_recalibration_wiring.py \
  tests/test_accuracy_date_clustering.py \
  tests/test_accuracy_headline_route.py \
  tests/test_directional_test.py \
  tests/test_bias_fit_date_guard.py -q
```

- [ ] **Step 6: Commit**

```bash
git add backend/backtest/scoring.py backend/models/served_recalibration.py \
  backend/models/direction.py backend/models/forecaster.py backend/api/routes/accuracy.py \
  backend/scripts backend/tests
git commit -m "refactor(metrics): separate headline and feedback date gates"
```

---

### Task 2: Add component contracts, centre policy, and equal-geometry assembly

**Files:**
- Create: `backend/models/prediction_contracts.py`
- Create: `backend/models/centre_policy.py`
- Create: `backend/models/forecast_assembly.py`
- Test: `backend/tests/test_prediction_contracts.py`
- Test: `backend/tests/test_centre_policy.py`
- Test: `backend/tests/test_forecast_assembly.py`

**Interfaces:**
- Produces: `CentrePrediction`, `SignedIntervalOffsets`, `RankingPrediction`
- Produces: `CENTRE_CHAMPIONS`, `centre_champion(horizon)`, `centre_challengers(horizon)`
- Produces: `percentage_offsets(low, mid, high)` and `assemble_interval(centre, offsets)`

- [ ] **Step 1: Write contract tests**

```python
import numpy as np
import pytest

from models.prediction_contracts import CentrePrediction, RankingPrediction, SignedIntervalOffsets


def test_centre_requires_aligned_finite_positive_arrays():
    with pytest.raises(ValueError, match="same length"):
        CentrePrediction("x", "v1", np.array([1.0]), np.array([10.0, 11.0]))
    with pytest.raises(ValueError, match="positive"):
        CentrePrediction("x", "v1", np.array([0.0]), np.array([0.0]))


def test_offsets_require_ordered_finite_legs():
    with pytest.raises(ValueError, match="lower_pct"):
        SignedIntervalOffsets(np.array([5.0]), np.array([-5.0]))


def test_ranking_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        RankingPrediction("rank", "v1", np.array([np.nan]))
```

- [ ] **Step 2: Write policy and assembly tests**

```python
def test_initial_champion_is_gbm_at_every_horizon():
    assert CENTRE_CHAMPIONS == {3: "gbm_q50", 7: "gbm_q50", 14: "gbm_q50", 30: "gbm_q50"}
    assert all(centre_challengers(h) == ("last_price",) for h in HORIZONS)


def test_two_centres_receive_identical_percentage_geometry():
    offsets = percentage_offsets(
        low=np.array([90.0]), mid=np.array([100.0]), high=np.array([125.0])
    )
    low_a, mid_a, high_a = assemble_interval(np.array([100.0]), offsets)
    low_b, mid_b, high_b = assemble_interval(np.array([40.0]), offsets)
    np.testing.assert_allclose(low_a / mid_a, low_b / mid_b)
    np.testing.assert_allclose(high_a / mid_a, high_b / mid_b)
```

Also test unknown horizons, incomplete champion mappings, empty names, unequal arrays, non-finite offsets, and non-positive centres.

- [ ] **Step 3: Run tests and verify missing-module failures**

```bash
venv/bin/python -m pytest tests/test_prediction_contracts.py tests/test_centre_policy.py tests/test_forecast_assembly.py -q
```

- [ ] **Step 4: Implement immutable contracts**

Use frozen dataclasses with `__post_init__` validation. Convert inputs with `np.asarray(..., dtype=float)` for validation; reject empty names/versions, unequal lengths, non-finite values, non-positive centre prices, and any `lower_pct > upper_pct`.

- [ ] **Step 5: Implement centre policy**

```python
HORIZONS = (3, 7, 14, 30)
REGISTERED_CENTRES = frozenset({"gbm_q50", "last_price"})
CENTRE_CHAMPIONS = {3: "gbm_q50", 7: "gbm_q50", 14: "gbm_q50", 30: "gbm_q50"}


def _validate_policy() -> None:
    if set(CENTRE_CHAMPIONS) != set(HORIZONS):
        raise RuntimeError("centre policy must name every supported horizon exactly once")
    unknown = set(CENTRE_CHAMPIONS.values()) - REGISTERED_CENTRES
    if unknown:
        raise RuntimeError(f"unknown centre champion(s): {sorted(unknown)}")


def centre_champion(horizon: int) -> str:
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}")
    return CENTRE_CHAMPIONS[horizon]


def centre_challengers(horizon: int) -> tuple[str, ...]:
    champion = centre_champion(horizon)
    return tuple(name for name in ("gbm_q50", "last_price") if name != champion)


_validate_policy()
```

- [ ] **Step 6: Implement assembly**

Calculate offsets as `(leg / mid - 1) * 100` and assemble as `centre * (1 + offset/100)`. Reject invalid mids/order. Return NumPy arrays without rounding.

- [ ] **Step 7: Run and commit**

```bash
venv/bin/python -m pytest tests/test_prediction_contracts.py tests/test_centre_policy.py tests/test_forecast_assembly.py -q
git add backend/models/prediction_contracts.py backend/models/centre_policy.py \
  backend/models/forecast_assembly.py backend/tests/test_prediction_contracts.py \
  backend/tests/test_centre_policy.py backend/tests/test_forecast_assembly.py
git commit -m "feat(models): add component contracts and centre policy"
```

---

### Task 3: Add candidate and frozen-outcome schema

**Files:**
- Modify: `backend/database.py`
- Create: `backend/migrations/versions/0027_add_forecast_candidates.py`
- Test: `backend/tests/test_forecast_candidate_schema.py`

**Interfaces:**
- Produces ORM models `ForecastCandidate` and `ForecastCandidateOutcome`.
- Produces Alembic head `0027_add_forecast_candidates`.

- [ ] **Step 1: Write isolated SQLite schema tests**

Use `create_engine("sqlite://")` and `Base.metadata.create_all`. Test a valid centre row, valid ranking row, invalid horizon, missing component payload, invalid centre ordering, duplicate identity, and one-to-one outcome.

Example invalid payload:

```python
row = ForecastCandidate(
    item_id=1,
    forecast_date=date(2026, 9, 18),
    horizon_days=7,
    component="centre",
    candidate_name="last_price",
    candidate_version="v1",
    centre_price=None,
    predicted_price_low=None,
    predicted_price_high=None,
    score=None,
    anchor_price=10.0,
    feature_cutoff_at=datetime(2026, 9, 18),
    config_fingerprint="a" * 64,
)
session.add(row)
with pytest.raises(IntegrityError):
    session.commit()
```

- [ ] **Step 2: Run and verify import failure**

```bash
venv/bin/python -m pytest tests/test_forecast_candidate_schema.py -q
```

- [ ] **Step 3: Add ORM models**

Use the spec columns. Define exact constraints:

```python
UniqueConstraint(
    "item_id", "forecast_date", "horizon_days", "component",
    "candidate_name", "candidate_version",
    name="uq_forecast_candidate_identity",
)
CheckConstraint("horizon_days IN (3, 7, 14, 30)", name="ck_forecast_candidate_horizon")
CheckConstraint("component IN ('centre', 'ranking')", name="ck_forecast_candidate_component")
CheckConstraint(
    "(component = 'centre' AND centre_price IS NOT NULL AND predicted_price_low IS NOT NULL "
    "AND predicted_price_high IS NOT NULL AND score IS NULL) OR "
    "(component = 'ranking' AND score IS NOT NULL AND centre_price IS NULL "
    "AND predicted_price_low IS NULL AND predicted_price_high IS NULL)",
    name="ck_forecast_candidate_payload",
)
CheckConstraint(
    "component <> 'centre' OR (predicted_price_low > 0 AND "
    "predicted_price_low <= centre_price AND centre_price <= predicted_price_high)",
    name="ck_forecast_candidate_ordering",
)
```

Index candidates on `(forecast_date, horizon_days, component)` and `(item_id, forecast_date)`. Candidate outcome `candidate_id` is both PK and FK with cascade delete.

- [ ] **Step 4: Add Alembic revision**

Set `revision = "0027_add_forecast_candidates"` and `down_revision = "0026_remove_duplicate_indexes"`. Create outcomes after candidates; downgrade in reverse order.

- [ ] **Step 5: Verify the migration using a temporary DB**

```bash
task_db=$(mktemp /tmp/cs2-candidate-schema.XXXXXX.db)
DATABASE_URL="sqlite:///$task_db" venv/bin/python -m alembic upgrade head
DATABASE_URL="sqlite:///$task_db" venv/bin/python -m alembic current
```

Expected: `0027_add_forecast_candidates`.

- [ ] **Step 6: Run and commit**

```bash
venv/bin/python -m pytest tests/test_forecast_candidate_schema.py -q
git add backend/database.py backend/migrations/versions/0027_add_forecast_candidates.py \
  backend/tests/test_forecast_candidate_schema.py
git commit -m "feat(db): add forecast candidate shadow tables"
```

---

### Task 4: Build exact candidate records and fingerprints

**Files:**
- Create: `backend/models/candidate_predictions.py`
- Test: `backend/tests/test_candidate_predictions.py`

**Interfaces:**
- Produces: `CandidateRecord`
- Produces: `config_fingerprint(payload) -> str`
- Produces: `candidate_records(results, ...) -> list[CandidateRecord]`

- [ ] **Step 1: Write final-geometry tests**

Use one item with `current_price=40`, GBM `low=90, mid=100, high=125`, and `rank_score=0.7`. Assert last-price centre/legs are `40/36/50`, the ranking record score is `0.7`, input is unchanged, and `anchor_date` becomes forecast date.

Test stable fingerprints under dictionary key reordering and changed fingerprints for changes in champion policy, artifact version, signed offsets, feedback factor, or relevant feature flags.

- [ ] **Step 2: Run and verify missing module**

```bash
venv/bin/python -m pytest tests/test_candidate_predictions.py -q
```

- [ ] **Step 3: Implement canonical SHA-256 fingerprints**

```python
def config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Implement conversion**

For every final forecast:

1. Treat final `low/mid/high` as the captured GBM triple.
2. Derive percentage offsets.
3. Build the last-price triple around exact `current_price`.
4. Round candidate prices to two decimals at the record edge.
5. Emit the non-champion centre named by `centre_challengers`.
6. Emit `lambdarank_v1` only for finite `rank_score`.
7. When last-price is later champion, emit the captured GBM triple as shadow.

Include component/name/version, artifact version, cutoff timestamp, anchor, and fingerprint on every record.

- [ ] **Step 5: Run and commit**

```bash
venv/bin/python -m pytest tests/test_candidate_predictions.py tests/test_forecast_assembly.py -q
git add backend/models/candidate_predictions.py backend/tests/test_candidate_predictions.py
git commit -m "feat(models): derive exact shadow candidate records"
```

---

### Task 5: Persist shadow batches idempotently

**Files:**
- Create: `backend/db/candidate_store.py`
- Test: `backend/tests/test_candidate_store.py`

**Interfaces:**
- Produces: `CandidateWriteResult`
- Produces: `write_candidate_batches(db, records)`
- Raises: `CandidateFingerprintConflict`

- [ ] **Step 1: Write persistence tests**

Using in-memory SQLite, test new writes, identical retries, conflicting fingerprints, atomic rollback of a mixed valid/invalid group, and separate centre/ranking batch counts.

- [ ] **Step 2: Run and verify failure**

```bash
venv/bin/python -m pytest tests/test_candidate_store.py -q
```

- [ ] **Step 3: Implement grouped transactions**

Group by `(forecast_date, horizon_days, component)`. Query existing identities, compare all fingerprints before inserts, raise on mismatch, insert missing rows inside `db.begin_nested()`, flush, and commit after all groups.

Catch only connection-invalidated `DBAPIError` as transient. Roll back and report that entire group unwritten. Integrity/fingerprint failures propagate.

- [ ] **Step 4: Run and commit**

```bash
venv/bin/python -m pytest tests/test_candidate_store.py tests/test_forecast_candidate_schema.py -q
git add backend/db/candidate_store.py backend/tests/test_candidate_store.py
git commit -m "feat(db): persist idempotent shadow prediction batches"
```

---

### Task 6: Wire centre selection and shadow writes into forecasting

**Files:**
- Modify: `backend/models/forecaster.py:9800-10020`
- Modify: `backend/scripts/forecast_prices.py:289-430,560-670`
- Modify: `backend/models/forecaster.py:11000-11630`
- Test: `backend/tests/test_shadow_forecast_wiring.py`
- Test: `backend/tests/test_model_component_manifest.py`
- Test: `backend/tests/test_model_version_identity.py`

**Interfaces:**
- `predict()` keeps its existing public DataFrame shape.
- Production triples follow `centre_champion(horizon)`.
- Exact shadows live on transient `self.pending_candidates` and are not placed in public mirrors.
- Metadata gains descriptive `components` and `centre_champions`.

- [ ] **Step 1: Write wiring tests**

```python
def test_initial_policy_keeps_gbm_public_and_last_price_shadow():
    public, shadow = apply_centre_policy(results)
    assert public.iloc[0].forecasts[7]["mid"] == 100.0
    assert shadow[0].candidate_name == "last_price"


def test_promoted_policy_swaps_public_and_shadow(monkeypatch):
    monkeypatch.setitem(CENTRE_CHAMPIONS, 7, "last_price")
    public, shadow = apply_centre_policy(results)
    assert public.iloc[0].forecasts[7]["mid"] == 40.0
    assert shadow[0].candidate_name == "gbm_q50"
    assert shadow[0].centre_price == 100.0
```

Assert `_write_forecasts_to_db` receives no ranking field and API schemas remain unchanged.

- [ ] **Step 2: Run and verify failure**

```bash
venv/bin/python -m pytest tests/test_shadow_forecast_wiring.py tests/test_model_component_manifest.py -q
```

- [ ] **Step 3: Apply policy after final sanitization**

At the end of `predict()`, after `_sanitize_forecasts`, capture candidate records and replace the public triple only if policy names last-price. Clear `self.pending_candidates` at the start of each call. This ordering captures blend, bias, conformal feedback, rounding, and ordering repairs.

- [ ] **Step 4: Persist candidates after production**

After `_write_forecasts_to_db` succeeds, call `write_candidate_batches`. Return/log:

```python
{
    "forecasts_written": n_production,
    "centre_candidates_written": result.centre_written,
    "ranking_candidates_written": result.ranking_written,
    "candidate_batches_expected": result.batches_expected,
    "candidate_batches_written": result.batches_written,
}
```

Integrity failures propagate; transient candidate DB failures leave production intact and return missing batch counts.

- [ ] **Step 5: Add the component manifest**

Write:

```python
"components": {
    "centre": {"gbm_q50": MODEL_ARTIFACT_VERSION},
    "interval": {"signed_conformal_climatology": MODEL_ARTIFACT_VERSION},
    "anomaly": {"anomaly_gbm_v1": MODEL_ARTIFACT_VERSION} if self.anomaly_models else {},
    "exceedance": {"exceedance_gbm_v1": MODEL_ARTIFACT_VERSION} if self.exceedance_models else {},
    "ranking": {"lambdarank_v1": MODEL_ARTIFACT_VERSION} if self.ranking_models else {},
    "direction": {},
},
"centre_champions": {str(h): centre_champion(h) for h in self.HORIZONS},
```

On load, a GBM champion requires a q50 artifact; last-price requires none. Missing ranking artifacts disable ranking shadow only.

- [ ] **Step 6: Run focused tests**

```bash
venv/bin/python -m pytest \
  tests/test_shadow_forecast_wiring.py \
  tests/test_model_component_manifest.py \
  tests/test_candidate_predictions.py \
  tests/test_candidate_store.py \
  tests/test_model_version_identity.py \
  tests/test_forecast_exceed_p_persistence.py \
  tests/test_forecast_anomaly_p_persistence.py -q
```

- [ ] **Step 7: Commit**

```bash
git add backend/models/forecaster.py backend/models/forecast_assembly.py \
  backend/scripts/forecast_prices.py backend/tests/test_shadow_forecast_wiring.py \
  backend/tests/test_model_component_manifest.py
git commit -m "feat(forecast): collect centre and ranking shadows"
```

---

### Task 7: Move direction training offline

**Files:**
- Modify: `backend/models/forecaster.py:6415-6450,9860-9910,11040-11050,11575-11595`
- Modify: `.github/workflows/price-forecast.yml`
- Create: `backend/scripts/direction_benchmark.py`
- Test: `backend/tests/test_direction_training_disposition.py`
- Test: `backend/tests/test_direction_benchmark.py`
- Modify: `backend/tests/test_minimal_model_shape.py`

**Interfaces:**
- Routine training creates/loads/saves no direction boosters.
- API direction remains neutral.
- Offline benchmark reports PT, DA, realised down rate, and constant-call accuracy without production writes.

- [ ] **Step 1: Write disposition tests**

Inspect routine train/save/load sources and assert no call/save/load of direction boosters. Load a legacy directory containing `clf_7d.txt` and assert `direction_models == {}`. Assert API routes still pass raw values through `served_direction` and `DIRECTION_DISCLOSED is False`.

- [ ] **Step 2: Write report-format test**

Feed `format_direction_report` a fixed metric dictionary and assert one block contains `Pesaran-Timmermann`, `directional_accuracy`, `realised_down_rate`, and `constant_call_accuracy`.

- [ ] **Step 3: Run and verify failures**

```bash
venv/bin/python -m pytest tests/test_direction_training_disposition.py tests/test_direction_benchmark.py -q
```

- [ ] **Step 4: Remove routine orchestration only**

Remove the direction fit/save/load blocks. Retain the fitting and metric helpers for offline research. Production prediction takes the existing no-classifier fallback path; public APIs remain neutral.

Remove stale workflow `CV_DIAGNOSTIC_CLASSIFIER` comments/variable if no routine path reads it. Do not remove unrelated flags.

- [ ] **Step 5: Implement offline benchmark**

Parse `--horizon` and `--archive-dir`, use production universe/features, use `embargo_days(horizon)`, fit only in a temporary model directory, call `pesaran_timmermann(records, MIN_HEADLINE_DATES)`, format the metric quartet, and never call production save/write/config mutation.

- [ ] **Step 6: Run and commit**

```bash
venv/bin/python -m pytest \
  tests/test_direction_training_disposition.py \
  tests/test_direction_benchmark.py \
  tests/test_minimal_model_shape.py \
  tests/test_directional_test.py -q
git add backend/models/forecaster.py backend/scripts/direction_benchmark.py \
  backend/tests/test_direction_training_disposition.py backend/tests/test_direction_benchmark.py \
  backend/tests/test_minimal_model_shape.py .github/workflows/price-forecast.yml
git commit -m "refactor(models): move direction classifier offline"
```

---

### Task 8: Resolve and freeze candidate outcomes

**Files:**
- Create: `backend/backtest/candidate_resolution.py`
- Modify: `backend/scripts/backtest_accuracy.py`
- Test: `backend/tests/test_candidate_resolution.py`
- Test: `backend/tests/test_candidate_outcome_freeze.py`

**Interfaces:**
- Produces: `resolve_candidate_outcomes(db, *, today, archive_dir, reresolve=False)`
- Reuses canonical archive maturity, gap, voting, and anchor-resolution rules.
- Persists immutable `ForecastCandidateOutcome` rows.

- [ ] **Step 1: Write shared-leg tests**

Create production and candidate rows in isolated SQLite. Stub the resolver and assert candidate `base_price`, `actual_price`, and `resolved_at` exactly match production.

- [ ] **Step 2: Write freeze tests**

Resolve once, alter the fixture, resolve normally, and assert legs do not move. Resolve with `reresolve=True` and assert they move. Rescore derived metrics and assert no archive call.

- [ ] **Step 3: Run and verify missing module**

```bash
venv/bin/python -m pytest tests/test_candidate_resolution.py tests/test_candidate_outcome_freeze.py -q
```

- [ ] **Step 4: Implement resolution**

Resolve unique `(item, date, horizon)` once. Prefer an existing frozen `ForecastOutcome`; use `load_voted_prices` and `resolve_anchors` only when no production outcome exists. Apply the same maturity, gap, stale, excluded-date, and invalid-leg rules.

If a production outcome exists and either resolved leg differs, raise `RuntimeError("candidate/production resolution mismatch")`.

For centre candidates:

```python
absolute_error = abs(candidate.centre_price - actual)
percentage_error = absolute_error / base * 100.0
in_interval = candidate.predicted_price_low <= actual <= candidate.predicted_price_high
```

Ranking rows store legs with null error/coverage.

- [ ] **Step 5: Integrate with backtesting**

Call candidate resolution after production outcome refresh. Under `--rescore`, refresh candidate derived metrics only; do not open the archive.

- [ ] **Step 6: Run regressions and commit**

```bash
venv/bin/python -m pytest \
  tests/test_candidate_resolution.py \
  tests/test_candidate_outcome_freeze.py \
  tests/test_price_resolution.py \
  tests/test_backtest_deterministic.py \
  tests/test_backtest_scoring.py -q
git add backend/backtest/candidate_resolution.py backend/scripts/backtest_accuracy.py \
  backend/tests/test_candidate_resolution.py backend/tests/test_candidate_outcome_freeze.py
git commit -m "feat(backtest): freeze shadow candidate outcomes"
```

---

### Task 9: Add paired daily scoring and centre promotion policy

**Files:**
- Create: `backend/backtest/candidate_scoring.py`
- Create: `backend/backtest/promotion.py`
- Test: `backend/tests/test_candidate_scoring.py`
- Test: `backend/tests/test_centre_promotion.py`

**Interfaces:**
- Produces: `paired_daily_interval(...)`
- Produces: `evaluate_centre_promotion(...)`
- Verdicts: `PASS_FOR_MANUAL_PROMOTION`, `REJECTED`, `UNRESOLVED`, `INSUFFICIENT_EVIDENCE`, `DATA_INTEGRITY_FAILURE`

- [ ] **Step 1: Write equal-date bootstrap tests**

Use 20 dates where one has 10,000 rows and the rest 100. Assert the point estimate is the mean of 20 daily paired means, not a row-pooled mean. Assert deterministic seed-42 bounds.

- [ ] **Step 2: Write verdict matrix tests**

Cover:

- 19 dates → insufficient.
- 20 dates + MAE upper bound ≤0 + coverage lower bound ≥−0.02 + width delta ≤0.001 → PASS.
- MAE win crossing zero → unresolved.
- Coverage lower bound <−0.02 → unresolved.
- MAE lower bound >0 → rejected.
- <95% overlap, <80% batch completeness, mixed fingerprints, or outcome mismatch → integrity/ineligible.

- [ ] **Step 3: Run and verify failures**

```bash
venv/bin/python -m pytest tests/test_candidate_scoring.py tests/test_centre_promotion.py -q
```

- [ ] **Step 4: Implement equal-weight daily bootstrap**

Pair item rows, average differences inside each date, resample daily means with replacement using seed 42, 1,000 resamples, and a 90% interval. Do not reuse `paired_metric_difference` because it weights dates by row count.

- [ ] **Step 5: Implement explicit gates**

```python
MIN_SHARED_DATES = MIN_HEADLINE_DATES
MIN_ROW_OVERLAP = 0.95
MIN_BATCH_COMPLETENESS = 0.80
MAX_COVERAGE_REGRESSION = 0.02
MAX_WIDTH_DELTA_PP = 0.001
PROMOTION_CI = 90
```

Return frozen result dataclasses. `promotion.py` has no database imports.

- [ ] **Step 6: Run and commit**

```bash
venv/bin/python -m pytest tests/test_candidate_scoring.py tests/test_centre_promotion.py -q
git add backend/backtest/candidate_scoring.py backend/backtest/promotion.py \
  backend/tests/test_candidate_scoring.py backend/tests/test_centre_promotion.py
git commit -m "feat(backtest): add manual centre promotion gate"
```

---

### Task 10: Add read-only centre and ranking reports

**Files:**
- Create: `backend/scripts/centre_promotion_report.py`
- Create: `backend/scripts/ranking_transfer_report.py`
- Test: `backend/tests/test_centre_promotion_report.py`
- Test: `backend/tests/test_ranking_transfer_report.py`

**Interfaces:**
- Centre CLI: `python -m scripts.centre_promotion_report --horizon {3,7,14,30,all} --json-out PATH --markdown-out PATH`
- Ranking CLI: `python -m scripts.ranking_transfer_report --horizon {3,7,14,30,all} --json-out PATH --markdown-out PATH`
- Centre exit codes: pass 0, unresolved/insufficient 1, integrity/execution 2, rejected 3.
- Ranking remains non-authoritative and shadow-only.

- [ ] **Step 1: Write centre report tests**

Patch the DB loader with PASS, unresolved, rejected, and integrity frames. Assert exit codes, JSON keys, Markdown headings, and no update/insert/config writes.

- [ ] **Step 2: Write within-date ranking tests**

Build a frame with positive pooled correlation but negative correlation on every date. Assert the report uses the negative within-date result. Test 99/100 items per date, 19/20 dates, mixed fingerprints, and supported/unresolved/rejected intervals.

- [ ] **Step 3: Run and verify missing scripts**

```bash
venv/bin/python -m pytest tests/test_centre_promotion_report.py tests/test_ranking_transfer_report.py -q
```

- [ ] **Step 4: Implement centre report**

Join candidates/outcomes/items/item_forecasts/forecast_outcomes on exact item/date/horizon identity. Apply >=$1 cohort and excluded dates. JSON includes eligibility failures, verdict, MAE/coverage/width intervals, rows/dates, per-date deltas, completeness, versions, and fingerprints. Markdown renders the same fields.

- [ ] **Step 5: Implement ranking report**

Per eligible date, compute Spearman IC for `lambdarank_v1.score` and production q50 implied return against canonical realised return. Pair daily ICs through Task 9. Require 20 dates, ≥100 shared items on every included date, 80% completeness, and a single version/fingerprint. Lower bound >0 is `SUPPORTED`; upper bound <0 is `REJECTED`.

- [ ] **Step 6: Prove shadow-only behavior**

Assert `rank_score` appears in neither `ItemForecast` nor `backend/api/schemas.py`. It may appear only in internal forecasts, `ForecastCandidate.score`, and reports.

- [ ] **Step 7: Run and commit**

```bash
venv/bin/python -m pytest \
  tests/test_centre_promotion_report.py \
  tests/test_ranking_transfer_report.py \
  tests/test_centre_promotion.py -q
git add backend/scripts/centre_promotion_report.py backend/scripts/ranking_transfer_report.py \
  backend/tests/test_centre_promotion_report.py backend/tests/test_ranking_transfer_report.py
git commit -m "feat(reports): add candidate evidence reports"
```

---

### Task 11: Add observability and completeness checks

**Files:**
- Modify: `backend/scripts/forecast_prices.py`
- Modify: `backend/scripts/run_task.py`
- Modify: `backend/scripts/check_forecast_freshness.py`
- Modify: `.github/workflows/price-forecast.yml`
- Test: `backend/tests/test_run_task.py`
- Test: `backend/tests/test_shadow_forecast_freshness.py`

**Interfaces:**
- Forecast counts: `forecasts_written`, `centre_candidates_written`, `ranking_candidates_written`, `candidate_batches_expected`, `candidate_batches_written`.
- Production zero rows remain fatal.
- Shadow incompleteness is loud but non-fatal unless caused by integrity conflict.

- [ ] **Step 1: Write count-guard tests**

Add `forecasts_written` to `ROW_COUNT_FIELDS`. Assert zero production fails. Assert production >0 with zero shadow does not trip the generic zero-row guard but yields a dedicated warning.

- [ ] **Step 2: Write readiness tests**

Three consecutive dates with expected == written yield `shadow_collection_ready=True`. A missing date/horizon/component resets readiness and names the gap.

- [ ] **Step 3: Implement logs and counters**

Per horizon log champion, challenger, versions, production rows, candidate rows, fingerprint prefix, and duration. Return all counts from `run_forecast`.

- [ ] **Step 4: Update workflows**

Keep `RANKING_HEAD=1`, remove routine direction variables/comments, add no auto-promotion step, and upload centre/ranking JSON+Markdown as `candidate-reports` after backtest resolution. Reports below the gate remain successful artifacts with insufficient evidence.

- [ ] **Step 5: Run tests and YAML parse**

```bash
venv/bin/python -m pytest tests/test_run_task.py tests/test_shadow_forecast_freshness.py -q
venv/bin/python -c "import yaml; yaml.safe_load(open('../.github/workflows/price-forecast.yml')); print('yaml ok')"
```

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/forecast_prices.py backend/scripts/run_task.py \
  backend/scripts/check_forecast_freshness.py backend/tests/test_run_task.py \
  backend/tests/test_shadow_forecast_freshness.py .github/workflows/price-forecast.yml
git commit -m "feat(ops): expose shadow collection health and reports"
```

---

### Task 12: Document and verify the shipped backend

**Files:**
- Create: `docs/changelog/2026-09-18-multi-head-champion-challenger-built.md`
- Modify: `docs/architecture/model.md`
- Modify: `docs/README.md`
- Modify: `docs/experiment_log.csv`
- Modify: `backend/AGENTS.md` if new invariants need enforcement

**Interfaces:**
- Documents exact production/shadow state.
- Adds a shipped framework row without claiming candidate victory.

- [ ] **Step 1: Write the changelog**

Record GBM as initial champion; last-price/LambdaRank shadow-only; anomaly/exceedance production; direction offline-only/API neutral; both date thresholds; report commands/exit codes; and no promotion.

- [ ] **Step 2: Update architecture and experiment log**

Add:

```csv
2026-09-18,multi-head-champion-challenger,Exact served shadow predictions can safely arbitrate component promotion,framework contract,,,,,shipped,docs/changelog/2026-09-18-multi-head-champion-challenger-built.md
```

- [ ] **Step 3: Run documentation guards**

```bash
venv/bin/python -m pytest tests/test_experiment_log.py -q
rg -n "automatically promote|auto-promot" \
  ../docs/changelog/2026-09-18-multi-head-champion-challenger-built.md \
  ../docs/architecture/model.md
```

Expected: no automatic-promotion claim.

- [ ] **Step 4: Run the focused feature suite**

```bash
venv/bin/python -m pytest \
  tests/test_date_threshold_policy.py \
  tests/test_prediction_contracts.py \
  tests/test_centre_policy.py \
  tests/test_forecast_assembly.py \
  tests/test_forecast_candidate_schema.py \
  tests/test_candidate_predictions.py \
  tests/test_candidate_store.py \
  tests/test_shadow_forecast_wiring.py \
  tests/test_model_component_manifest.py \
  tests/test_direction_training_disposition.py \
  tests/test_direction_benchmark.py \
  tests/test_candidate_resolution.py \
  tests/test_candidate_outcome_freeze.py \
  tests/test_candidate_scoring.py \
  tests/test_centre_promotion.py \
  tests/test_centre_promotion_report.py \
  tests/test_ranking_transfer_report.py \
  tests/test_shadow_forecast_freshness.py -q
```

- [ ] **Step 5: Run regressions and full verification**

```bash
venv/bin/python -m pytest \
  tests/test_forecaster.py \
  tests/test_minimal_model_shape.py \
  tests/test_backtest_scoring.py \
  tests/test_price_resolution.py \
  tests/test_forecast_exceed_p_persistence.py \
  tests/test_forecast_anomaly_p_persistence.py \
  tests/test_accuracy_headline_route.py -q
venv/bin/python -m pytest tests/ -q
git diff --check
venv/bin/python -m alembic heads
```

Expected: all tests pass and Alembic reports one head, `0027_add_forecast_candidates`.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/changelog/2026-09-18-multi-head-champion-challenger-built.md \
  docs/architecture/model.md docs/README.md docs/experiment_log.csv backend/AGENTS.md
git commit -m "docs: record multi-head shadow forecasting backend"
```

---

## Operational verification after merge

Do not run a forecast locally from `backend/` against its default `.env`.

1. Apply migration `0027`.
2. Dispatch one full/train-only workflow so the artifact contains the component manifest and ranking head.
3. Confirm production rows and every expected shadow batch are written; confirm new artifacts contain no direction boosters.
4. Require three consecutive complete shadow dates before starting the evidence clock.
5. Let normal backtests resolve candidates as they mature.
6. Expect both reports to say `INSUFFICIENT_EVIDENCE` until 20 shared dates.
7. Do not change `CENTRE_CHAMPIONS` in the implementation branch.

The first promotion is a separate reviewed change after one horizon returns `PASS_FOR_MANUAL_PROMOTION`.
