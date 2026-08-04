# Remove the Accidental Retrain Work — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the daily forecast run from performing an unrequested full retrain, and stop the predict path from engineering 1460 days of history to keep 3 rows per item.

**Architecture:** Two independent changes to existing code, no new modules. (1) Drift detection keeps writing alerts but no longer triggers retraining, and gains a forecast-date-coverage guard that reuses machinery already present in `backend/backtest/scoring.py`. (2) `predict()` truncates its price frame to the last N rows per item after multi-source voting, which is exactly safe because voting collapses the frame to one row per item-day.

**Tech Stack:** Python 3.11, pandas, LightGBM, DuckDB, pytest, SQLAlchemy.

**Spec:** `docs/superpowers/specs/2026-08-04-remove-accidental-retrain-work-design.md`

## Global Constraints

- **Test invocation, verified 2026-08-04 — use exactly this:** `cd backend && ./venv/bin/python -m pytest tests/ -q`.
  - `python` is **not on PATH**. Bare `python -m pytest` fails with "command not found".
  - System `python3` (3.13) has the dependencies but **segfaults inside LightGBM** partway through the suite. Do not use it for pytest.
  - Scope to `tests/`. Collecting from `backend/` root picks up `scripts/test_social_signal.py`, which fails at import and aborts the whole run — pre-existing and unrelated to this plan.
  - `python3 -m py_compile <file>` is fine (no LightGBM import), per `AGENTS.md`.
- **Baseline is `417 passed` in ~3m08s** (measured on `363b659`, before any task). Any other number of failures is yours. No frontend change is involved, so `npm run lint` / `npm run build` do not apply.
- **No model-class, feature-set, or hyperparameter changes.** If a step would alter which features the model trains on or how it fits, stop and flag it.
- **Never let tests use the default `model_dir`.** Always pass `model_dir=str(tmp_path_factory.mktemp("saved_models"))` or `tmp_path`. The real `backend/models/saved_models/` holds gitignored, unrecoverable production artifacts.
- Reuse `MIN_FORECAST_DATES` from `backend/backtest/scoring.py` (currently 20). Do **not** introduce a second *date-coverage* threshold. Note this does not conflict with Task 1's `DRIFT_DA_THRESHOLD = 60.0`, which is an *accuracy* floor for alerting — a different quantity that is currently hardcoded at two call sites.
- The date-coverage guard **fails closed**: missing or absent coverage data reads as "we cannot tell", not as "fine".
- Existing behaviour to preserve: `check_concept_drift` must keep writing `AccuracyAlert` rows and keep resolving open alerts.

## Context an implementer needs

**What is broken.** `scripts/forecast_prices.py:194` calls `check_concept_drift(..., threshold=60.0)`. The model's reproducible directional accuracy is 46.7–50.8%, so drift is detected on every run and the `--predict-only` branch retrains before predicting. Measured cost in CI run `30864690456`: **465s of an 835s step**.

**Why the threshold cannot just be retuned.** The accuracy it reads spans 1–2 distinct forecast dates while carrying five-figure row counts. Every item sharing a forecast date is exposed to the same market-wide move, so the number tracks market direction, not model decay. See `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md`.

**What already exists and must be reused, not rebuilt:**

| Thing | Location | What it gives you |
|---|---|---|
| `MIN_FORECAST_DATES = 20` | `backend/backtest/scoring.py:38` | The repo's single definition of "enough forecast dates" |
| `"date_coverage_sufficient"` in metrics | `backend/backtest/scoring.py:202` | Per-accuracy-row boolean, already stored in `prediction_accuracy.metrics` |
| `ItemForecaster._has_date_coverage(dates)` | `backend/models/forecaster.py:3275` | Static helper for raw date lists. **Not needed in this plan** — the drift path reads the precomputed flag instead, because `prediction_accuracy` rows are already aggregates and do not carry per-row dates. |

**The invariant Change 2 rests on.** `_apply_multi_source_voting` collapses the frame to one row per item-day (8,103,167 raw → 6,155,446 voted rows). Rows are therefore never denser than daily, so the last N rows always span at least N calendar days. That single fact satisfies both feature requirements at once:

- Calendar-date lag joins (`forecaster.py:937-946`, `LAGS` up to 180) need ≥183 calendar days.
- Positional row-count rollings (`forecaster.py:962` windows to 60, `:1118` windows 100 and 200) need ≥203 rows.

`PREDICT_TAIL_ITEM_DAYS = 240` clears both with margin. **Tailing is a no-op for any item holding ≤240 rows**, so no item's features change — items are eligible with as little as `PREDICT_MIN_HISTORY_DAYS = 14` days and many hold far fewer than 240 rows.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `backend/models/forecaster.py` | Model, features, drift check, predict | Modify: `check_concept_drift` (`:4327-4393`), new class constants, tail in `predict()` (`:3603-3622`), engineered-cache version |
| `backend/scripts/forecast_prices.py` | CLI orchestration | Modify: delete `_drift_detected` (`:59-70`), predict-only branch (`:192-204`), full-mode condition (`:180-186`) |
| `backend/tests/test_drift_retrain_guard.py` | **Create.** All drift-guard and no-retrain assertions | New file — Tasks 1–3 |
| `backend/tests/test_predict_tail_truncation.py` | **Create.** Voting invariant + feature-equality assertions | New file — Task 4 |
| `AGENTS.md` | Backend gotchas | Modify: record that predict-only no longer auto-retrains |
| `docs/architecture/model-optimization.md` | Timing baseline | Modify: correct the inference-timing rows |

Two new test files rather than appending to `test_forecaster.py` (4,718-line source, already-large test file): each maps to one defect and can be read whole.

---

### Task 1: Guard `check_concept_drift` on forecast-date coverage

**Files:**
- Modify: `backend/models/forecaster.py:4327-4362` (signature + accuracy accumulation)
- Test: `backend/tests/test_drift_retrain_guard.py` (create)

**Interfaces:**
- Consumes: `MIN_FORECAST_DATES` from `backtest.scoring`.
- Produces: `ItemForecaster.DRIFT_DA_THRESHOLD: float` (class constant, 60.0). `check_concept_drift(horizon: int = 7, sliding_window: int = 7, threshold: Optional[float] = None) -> Optional[Dict]` — returns `None` when evidence is insufficient, else `{"drifted": bool, "accuracy": float, "threshold": float}`. Tasks 2 and 3 rely on the `None` return.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_drift_retrain_guard.py`:

```python
"""Drift detection must not trigger retraining, and must not read a
degenerate sample as evidence.

check_concept_drift averages stored prediction_accuracy rows. Those rows span
1-2 distinct forecast dates while carrying five-figure row counts, so the
average tracks which way the market moved rather than model decay. See
docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backtest.scoring import MIN_FORECAST_DATES
from models.forecaster import ItemForecaster


class _Row:
    """Stand-in for a prediction_accuracy result row."""

    def __init__(self, metrics, evaluation_date="2026-08-01"):
        self.metrics = metrics
        self.evaluation_date = evaluation_date


def _metrics(da, *, covered=True, include_flag=True):
    m = {"directional_accuracy": da}
    if include_flag:
        m["date_coverage_sufficient"] = covered
        m["distinct_forecast_dates"] = MIN_FORECAST_DATES if covered else 2
    return m


def _forecaster(rows, tmp_path):
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = rows
    return ItemForecaster(db_session=db, model_dir=str(tmp_path))


def test_threshold_is_a_named_constant():
    # 60.0 was hardcoded at two call sites above a model whose measured DA is
    # 46.7-50.8%. Naming it stops the next reader rediscovering that.
    assert ItemForecaster.DRIFT_DA_THRESHOLD == 60.0


def test_rows_without_date_coverage_are_not_evidence(tmp_path):
    rows = [_Row(_metrics(28.0, covered=False)) for _ in range(7)]
    assert _forecaster(rows, tmp_path).check_concept_drift(horizon=7) is None


def test_rows_missing_the_coverage_flag_fail_closed(tmp_path):
    # Rows written before scoring.py reported coverage carry no date
    # attribution. Absent must read as "we cannot tell", not as "fine".
    rows = [_Row(_metrics(28.0, include_flag=False)) for _ in range(7)]
    assert _forecaster(rows, tmp_path).check_concept_drift(horizon=7) is None


def test_covered_rows_below_threshold_still_report_drift(tmp_path):
    rows = [_Row(_metrics(28.0, covered=True)) for _ in range(7)]
    result = _forecaster(rows, tmp_path).check_concept_drift(horizon=7)
    assert result is not None
    assert result["drifted"] is True
    assert result["accuracy"] == pytest.approx(28.0)


def test_covered_rows_above_threshold_report_no_drift(tmp_path):
    rows = [_Row(_metrics(72.0, covered=True)) for _ in range(7)]
    result = _forecaster(rows, tmp_path).check_concept_drift(horizon=7)
    assert result is not None
    assert result["drifted"] is False


def test_uncovered_rows_are_excluded_from_the_average(tmp_path):
    # Three covered rows at 70% plus four uncovered at 10% must average to 70%,
    # not to 35.7%.
    rows = [_Row(_metrics(70.0, covered=True)) for _ in range(3)]
    rows += [_Row(_metrics(10.0, covered=False)) for _ in range(4)]
    result = _forecaster(rows, tmp_path).check_concept_drift(horizon=7)
    assert result is not None
    assert result["accuracy"] == pytest.approx(70.0)


def test_too_few_covered_rows_returns_none(tmp_path):
    # Two covered rows is below the existing len(accuracies) < 3 floor.
    rows = [_Row(_metrics(70.0, covered=True)) for _ in range(2)]
    rows += [_Row(_metrics(10.0, covered=False)) for _ in range(5)]
    assert _forecaster(rows, tmp_path).check_concept_drift(horizon=7) is None


def test_drift_alert_is_still_written(tmp_path):
    # Alerting is worth keeping even though it no longer triggers a retrain.
    rows = [_Row(_metrics(28.0, covered=True)) for _ in range(7)]
    f = _forecaster(rows, tmp_path)
    f.check_concept_drift(horizon=7)
    assert f.db.add.called
    assert f.db.commit.called
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_drift_retrain_guard.py -v
```

Expected: `test_threshold_is_a_named_constant` fails with `AttributeError: type object 'ItemForecaster' has no attribute 'DRIFT_DA_THRESHOLD'`. The coverage tests fail because drift is currently reported regardless of coverage.

- [ ] **Step 3: Add the class constant**

In `backend/models/forecaster.py`, near the other drift/threshold class constants (alongside `PREDICT_MIN_HISTORY_DAYS` at `:153`):

```python
    # Directional-accuracy floor for the drift *alert*. This no longer gates a
    # retrain: the model's measured production DA is 46.7-50.8%
    # (docs/architecture/model-optimization.md), so a 60% floor was never
    # attainable and fired on every run. Kept as an alert threshold only.
    DRIFT_DA_THRESHOLD = 60.0
```

- [ ] **Step 4: Make the threshold parameter fall back to the constant**

Replace the signature at `:4327-4328`:

```python
    def check_concept_drift(self, horizon: int = 7, sliding_window: int = 7,
                             threshold: Optional[float] = None) -> Optional[Dict]:
```

and insert as the first statement of the body, before the `from database import ...` line:

```python
        threshold = self.DRIFT_DA_THRESHOLD if threshold is None else threshold
```

- [ ] **Step 5: Add the coverage guard to the accuracy accumulation**

Replace `:4352-4359` (the `accuracies` loop and the `len(accuracies) < 3` check):

```python
        accuracies = []
        uncovered = 0
        for r in records:
            m = r.metrics if isinstance(r.metrics, dict) else json.loads(r.metrics)
            if "directional_accuracy" not in m:
                continue
            # Fail closed. Rows written before scoring.py began reporting
            # coverage carry no date attribution, and rows spanning 1-2 dates
            # describe those dates rather than the model. Either way this is
            # "we cannot tell", not "the model is fine".
            if not m.get("date_coverage_sufficient", False):
                uncovered += 1
                continue
            accuracies.append(m["directional_accuracy"])

        if uncovered:
            logger.info(
                f"  Drift check ({horizon}d): ignored {uncovered} accuracy row(s) "
                f"lacking {MIN_FORECAST_DATES}-date coverage"
            )

        if len(accuracies) < 3:
            return None
```

No import is needed: `MIN_FORECAST_DATES` is already imported at `forecaster.py:21` (`from backtest.scoring import MIN_FORECAST_DATES`).

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_drift_retrain_guard.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 7: Run the full backend suite and compile check**

```bash
cd backend && ./venv/bin/python -m pytest tests/ -q && python3 -m py_compile models/forecaster.py
```

Expected: no new failures. Note the pre-existing pass/fail count before you start so you can tell new breakage from old.

- [ ] **Step 8: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_drift_retrain_guard.py
git commit -m "fix: guard the drift check on forecast-date coverage

check_concept_drift averaged prediction_accuracy rows spanning 1-2 distinct
forecast dates, so it measured market direction rather than model decay.
Rows are now excluded unless they report date_coverage_sufficient, failing
closed when the field is absent. Hoists the 60.0 threshold to a named
constant documenting that it was never attainable."
```

---

### Task 2: Stop the predict-only path from retraining

**Files:**
- Modify: `backend/scripts/forecast_prices.py:192-204`
- Modify: `AGENTS.md` (backend gotchas list)
- Test: `backend/tests/test_drift_retrain_guard.py` (append)

**Interfaces:**
- Consumes: `check_concept_drift` returning `Optional[Dict]` from Task 1.
- Produces: no new symbols. Behavioural contract: with `--predict-only`, `forecaster.train` is never called unless `ALLOW_DRIFT_RETRAIN=1`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_drift_retrain_guard.py`:

```python
# ---------------------------------------------------------------------------
# The predict-only path must not retrain.
# ---------------------------------------------------------------------------

FORECAST_PRICES_SRC = (
    Path(__file__).resolve().parent.parent / "scripts" / "forecast_prices.py"
)


def test_predict_only_branch_has_no_retrain_trigger():
    """The --predict-only branch must not set do_train.

    Asserted on source rather than by running the pipeline: main() needs a live
    DB, a price archive, and saved boosters. The defect was one assignment, so
    pinning that assignment out is the honest unit-level check.
    """
    src = FORECAST_PRICES_SRC.read_text()
    start = src.index("elif predict_only and has_models:")
    branch = src[start:src.index("if do_train:", start)]

    # The retrain must exist only behind the explicit opt-in.
    assert "if drifted_horizons and allow_retrain:" in branch, (
        "Retraining must be gated on the ALLOW_DRIFT_RETRAIN opt-in."
    )
    assert branch.count("do_train = True") == 1, (
        "Exactly one gated retrain assignment expected in this branch; found "
        f"{branch.count('do_train = True')}."
    )
    assert "ALLOW_DRIFT_RETRAIN" in branch, (
        "The opt-in escape hatch must be present in this branch."
    )
```

Add `from pathlib import Path` to the file's imports.

This fails today for the right reason: the current branch has no `allow_retrain` gate, so the first assertion fails while `do_train = True` is present unconditionally.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_drift_retrain_guard.py::test_predict_only_branch_has_no_retrain_trigger -v
```

Expected: FAIL on the first assertion — `do_train = True` is still present in the branch.

- [ ] **Step 3: Rewrite the predict-only branch**

Replace `backend/scripts/forecast_prices.py:192-204` in full:

```python
        elif predict_only and has_models:
            # Drift is reported but does NOT trigger a retrain. The signal it
            # reads spans 1-2 distinct forecast dates, so it tracks market
            # direction rather than model decay, and the retrain it used to
            # trigger cost a measured 465s of every 835s daily run while
            # making the workflow's Monday-only retrain design fiction.
            # See docs/superpowers/specs/2026-08-04-remove-accidental-retrain-work-design.md
            allow_retrain = os.environ.get("ALLOW_DRIFT_RETRAIN") == "1"
            drifted_horizons = []
            for h in ItemForecaster.HORIZONS:
                drift_result = forecaster.check_concept_drift(horizon=h, sliding_window=7)
                if drift_result and drift_result.get("drifted"):
                    drifted_horizons.append(h)
            if drifted_horizons and allow_retrain:
                logger.warning(
                    f"Drift reported for horizons {drifted_horizons} and "
                    f"ALLOW_DRIFT_RETRAIN=1 — retraining before prediction."
                )
                do_train = True
            elif drifted_horizons:
                logger.warning(
                    f"Drift reported for horizons {drifted_horizons}. Not "
                    f"retraining: predict-only serves the scheduled model. Set "
                    f"ALLOW_DRIFT_RETRAIN=1 to retrain, or dispatch the "
                    f"workflow with mode=full."
                )
```

`os` is already imported at `forecast_prices.py:16`.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_drift_retrain_guard.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Record the gotcha in AGENTS.md**

Add to the `## Gotchas` list in the repo-root `AGENTS.md`:

```markdown
- **`--predict-only` does not retrain.** It used to: `check_concept_drift` ran against a hardcoded 60% floor above the model's measured 46.7–50.8% DA, so drift fired every run and cost a measured 465s of the 835s daily step. Drift is now reported only. Set `ALLOW_DRIFT_RETRAIN=1` or dispatch `price-forecast.yml` with `mode=full` to retrain.
```

Note this replaces the claim in `forecast_prices.py:9`'s docstring — update that line too:

```python
    python scripts/forecast_prices.py --predict-only  # use saved models (no auto-retrain)
```

- [ ] **Step 6: Run the full suite and compile check**

```bash
cd backend && ./venv/bin/python -m pytest tests/ -q && python3 -m py_compile scripts/forecast_prices.py
```

- [ ] **Step 7: Commit**

```bash
git add backend/scripts/forecast_prices.py backend/tests/test_drift_retrain_guard.py AGENTS.md
git commit -m "fix: stop --predict-only from auto-retraining on drift

Drift was detected on every run because the 60% threshold sits above the
model's 46.7-50.8% measured DA, so every daily run did a full four-horizon
retrain first: 465s of an 835s step. Drift is now reported only.
ALLOW_DRIFT_RETRAIN=1 restores the old behaviour."
```

---

### Task 3: Drop drift from the full-mode retrain condition and delete `_drift_detected`

**Files:**
- Modify: `backend/scripts/forecast_prices.py:59-70` (delete `_drift_detected`), `:180-186` (retrain condition)
- Test: `backend/tests/test_drift_retrain_guard.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: no new symbols. `_drift_detected` is removed; nothing may reference it afterwards.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_drift_retrain_guard.py`:

```python
# ---------------------------------------------------------------------------
# Full mode retrains on model age, not on drift.
# ---------------------------------------------------------------------------

def test_drift_detected_helper_is_gone():
    import scripts.forecast_prices as m
    assert not hasattr(m, "_drift_detected"), (
        "_drift_detected's only caller was the full-mode retrain condition. "
        "Leaving it behind invites the trigger being reinstated."
    )


def test_full_mode_retrains_on_age_not_drift():
    src = FORECAST_PRICES_SRC.read_text()
    start = src.index("if age is None or age >= retrain_interval")
    condition = src[start:src.index(":", start)]

    assert "drifted" not in condition, (
        "Full mode must retrain on model age alone. Drift reads a 1-2-date "
        "sample and cannot support the decision."
    )
    assert "retrain_interval" in condition
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_drift_retrain_guard.py -k "drift_detected_helper or full_mode" -v
```

Expected: both FAIL — the helper still exists and `drifted` is still in the condition.

- [ ] **Step 3: Delete the `_drift_detected` helper**

Delete `backend/scripts/forecast_prices.py:59-70` entirely:

```python
def _drift_detected(forecaster) -> bool:
    """Return True if any horizon currently shows concept drift."""
    for h in ItemForecaster.HORIZONS:
        try:
            drift_result = forecaster.check_concept_drift(
                horizon=h, sliding_window=7, threshold=60.0
            )
        except Exception:
            continue
        if drift_result and drift_result.get("drifted"):
            return True
    return False
```

- [ ] **Step 4: Rewrite the full-mode retrain condition**

At `backend/scripts/forecast_prices.py:180-186`, replace the `drifted` lookup and condition. Before:

```python
                drifted = _drift_detected(forecaster)
                if age is None or age >= retrain_interval or drifted:
                    reason = ("model stale" if (age is not None and age >= retrain_interval)
                              else "drift" if drifted else "unknown age")
```

After:

```python
                # Retrain on model age only. Drift is not a valid trigger: it
                # reads a 1-2-forecast-date sample, so it tracks market
                # direction rather than model decay.
                if age is None or age >= retrain_interval:
                    reason = ("model stale" if (age is not None and age >= retrain_interval)
                              else "unknown age")
```

Read the surrounding lines before editing — the exact indentation and the `else` branch that follows must be preserved.

- [ ] **Step 5: Confirm no references remain**

```bash
cd backend && grep -rn "_drift_detected" . --include=*.py
```

Expected: no output.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_drift_retrain_guard.py -v && ./venv/bin/python -m pytest tests/ -q
```

Expected: all drift-guard tests PASS, no new failures elsewhere.

- [ ] **Step 7: Commit**

```bash
git add backend/scripts/forecast_prices.py backend/tests/test_drift_retrain_guard.py
git commit -m "refactor: retrain on model age only, drop the drift trigger

Removes _drift_detected and the 'or drifted' clause from the full-mode
retrain condition. Monday's scheduled run already retrains on age; drift
reads a 1-2-forecast-date sample and cannot support the decision."
```

---

### Task 4: Truncate the predict frame by rows per item

**Files:**
- Modify: `backend/models/forecaster.py` — new constant beside `PREDICT_TAIL_ROWS` (`:3532`), tail insertion in `predict()` (after `:3611`), engineered-cache version (`:2558-2577`)
- Test: `backend/tests/test_predict_tail_truncation.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks — independent of Tasks 1–3.
- Produces: `ItemForecaster.PREDICT_TAIL_ITEM_DAYS: int = 240`. `ItemForecaster.ENGINEERED_CACHE_VERSION: int = 2`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_predict_tail_truncation.py`:

```python
"""Prediction must not engineer 1460 days of history to keep 3 rows per item.

Truncation has to be row-based, not calendar-based. The feature set mixes
calendar-date lag joins (LAGS up to 180 days, forecaster.py:937) with
positional row-count rollings (windows to 200, forecaster.py:1118) over an
archive that is only ~48% dense, so a 240-*day* cutoff can yield ~115 rows and
silently change every rolling feature at the serving edge.

Multi-source voting collapses the frame to one row per item-day, so the last N
rows always span >= N calendar days. That invariant is what makes a row-based
tail satisfy both requirements at once, and test_voting_yields_one_row_per_item_day
is what keeps it true.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(),
                          model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _price_frame(n_items=3, n_days=1460, start=date(2022, 1, 1)):
    """Dense daily series, one row per item-day, with a mild upward drift."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_items):
        price = 10.0 + i
        for d in range(n_days):
            price *= 1.0 + rng.normal(0.0005, 0.02)
            rows.append({
                "item_id": f"item-{i}",
                "date": start + timedelta(days=d),
                "price": round(max(price, 0.05), 4),
                "volume": float(rng.integers(1, 500)),
            })
    return pd.DataFrame(rows)


def test_tail_constant_covers_both_window_requirements():
    # 200-row positional rollings + PREDICT_TAIL_ROWS, and 180-day calendar
    # lags + PREDICT_TAIL_ROWS. The row-based tail must clear both.
    assert ItemForecaster.PREDICT_TAIL_ITEM_DAYS >= 200 + ItemForecaster.PREDICT_TAIL_ROWS
    assert ItemForecaster.PREDICT_TAIL_ITEM_DAYS >= 180 + ItemForecaster.PREDICT_TAIL_ROWS


def test_voting_yields_one_row_per_item_day(forecaster):
    """The invariant the row-based tail depends on.

    If voting ever emits two rows for one item-day, N rows would no longer span
    N calendar days and the 180-day calendar lags could silently fall short.
    """
    raw = _price_frame(n_items=2, n_days=30)
    # Same item-days observed by three sources, plus one outlier per day.
    multi = pd.concat([
        raw.assign(source="STEAMCOMMUNITY"),
        raw.assign(source="BUFF163", price=raw["price"] * 1.01),
        raw.assign(source="CSFLOAT", price=raw["price"] * 20),
    ], ignore_index=True)

    voted = forecaster._apply_multi_source_voting(multi)

    dupes = voted.groupby(["item_id", "date"]).size()
    assert (dupes == 1).all(), (
        f"Voting emitted duplicate item-days: "
        f"{dupes[dupes > 1].head().to_dict()}"
    )


def test_tail_is_a_noop_for_short_history_items(forecaster):
    """Items with <= PREDICT_TAIL_ITEM_DAYS rows must be untouched.

    PREDICT_MIN_HISTORY_DAYS is 14, so most eligible items hold far fewer than
    240 rows. Their features must be bit-identical after this change.
    """
    short = _price_frame(n_items=2, n_days=100)
    tailed = forecaster._tail_predict_frame(short)
    pd.testing.assert_frame_equal(
        short.sort_values(["item_id", "date"]).reset_index(drop=True),
        tailed.sort_values(["item_id", "date"]).reset_index(drop=True),
    )


def test_tail_keeps_exactly_the_last_n_rows_per_item(forecaster):
    long = _price_frame(n_items=3, n_days=1460)
    tailed = forecaster._tail_predict_frame(long)

    counts = tailed.groupby("item_id").size()
    assert (counts == ItemForecaster.PREDICT_TAIL_ITEM_DAYS).all()

    # It must be the *last* rows, not the first.
    for item, group in tailed.groupby("item_id"):
        expected_max = long[long["item_id"] == item]["date"].max()
        assert group["date"].max() == expected_max


def test_served_features_survive_truncation(forecaster):
    """THE LOAD-BEARING TEST.

    The served feature vector is the last row per item. It must be identical
    whether engineered from 1460 days or from the truncated tail. Anything that
    differs here is train/serve skew shipped to production.
    """
    full = _price_frame(n_items=3, n_days=1460)
    events = pd.DataFrame(columns=["date", "event_type", "name"])

    full_feats = forecaster.engineer_features(full, events)
    tail_feats = forecaster.engineer_features(forecaster._tail_predict_frame(full), events)

    def _last_rows(df):
        return (df.sort_values(["item_id", "date"])
                  .groupby("item_id").last()
                  .sort_index())

    a, b = _last_rows(full_feats), _last_rows(tail_feats)

    shared = [c for c in a.columns if c in b.columns
              and pd.api.types.is_numeric_dtype(a[c])]
    assert shared, "No numeric feature columns to compare"

    mismatched = [
        c for c in shared
        if not np.allclose(a[c].to_numpy(dtype=float),
                           b[c].to_numpy(dtype=float),
                           rtol=1e-9, atol=1e-9, equal_nan=True)
    ]
    assert not mismatched, (
        f"Truncation changed {len(mismatched)} served feature(s): "
        f"{sorted(mismatched)[:12]}"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_predict_tail_truncation.py -v
```

Expected: FAIL with `AttributeError` on `PREDICT_TAIL_ITEM_DAYS` and `_tail_predict_frame`.

`_apply_multi_source_voting` is a `@staticmethod` taking a single `df` (`forecaster.py:807`); calling it through the instance as the test does is fine. It expects a `source` column, which the fixture supplies.

Do **not** weaken `test_voting_yields_one_row_per_item_day` to make it pass. It documents the property the design reasons from. Note that `_tail_predict_frame` selects on distinct dates rather than row position, so it stays correct even if duplicates appear — but if voting genuinely emits duplicate item-days, stop and report anyway, because the spec's "N rows span ≥N calendar days" argument would need rewriting.

- [ ] **Step 3: Add the constant and the tail helper**

In `backend/models/forecaster.py`, extend the existing comment block at `:3527-3532`:

```python
    # Prediction consumes only the last few rows per item (tail(3) for the
    # smoothed current price, last() for the feature vector), yet the whole-frame
    # path engineers all 1460 days for every item just to slice that tail off.
    # At 8,691 items / 6.1M rows that peaks well past a 16GB CI runner and gets
    # SIGKILLed — see docs and runs 30226424193 / 30666903525 / 30668690592.
    PREDICT_TAIL_ROWS = 3

    # Rows of history per item retained for prediction. Row-based, not
    # calendar-based: features mix 180-day calendar lag joins (:937) with
    # 200-row positional rollings (:1118) over a ~48%-dense archive, so a
    # 240-*day* cutoff could yield ~115 rows and silently change every rolling
    # feature. Voting collapses to one row per item-day, so the last N rows
    # always span >= N calendar days — one parameter satisfies both.
    # Tailing is a no-op for items holding fewer rows than this.
    PREDICT_TAIL_ITEM_DAYS = 240

    def _tail_predict_frame(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the last PREDICT_TAIL_ITEM_DAYS item-days per item.

        Selects on distinct dates rather than on row position, so a frame that
        still carries intraday duplicates cannot yield fewer calendar days than
        the window promises. engineer_features resamples to one row per item-day
        itself (:2163), so keeping every row on a retained date is safe.
        """
        before = len(price_df)
        keep = (price_df[["item_id", "date"]]
                .drop_duplicates()
                .sort_values(["item_id", "date"])
                .groupby("item_id", sort=False, group_keys=False)
                .tail(self.PREDICT_TAIL_ITEM_DAYS))
        out = price_df.merge(keep, on=["item_id", "date"], how="inner")
        logger.info(
            f"  Predict tail: {before:,} -> {len(out):,} rows "
            f"(<= {self.PREDICT_TAIL_ITEM_DAYS} item-days per item)"
        )
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_predict_tail_truncation.py -v
```

Expected: all 6 tests PASS. If `test_served_features_survive_truncation` fails, **do not raise the tolerance.** Read the mismatched feature names it prints and find which window exceeds 240 rows — then raise `PREDICT_TAIL_ITEM_DAYS` to cover it and note the real requirement in the comment.

- [ ] **Step 5: Wire the tail into `predict()`**

In `backend/models/forecaster.py`, immediately after the eligibility filter at `:3611-3615` (the `logger.info` reporting eligible items) and before `events_df = self.fetch_events()`:

```python
            # Engineer only the history the features need. Must come after the
            # eligibility filter, which counts distinct days over full history.
            price_df = self._tail_predict_frame(price_df)
```

Leave `days_back=1460` at `:3603` alone — narrowing the fetch is Task 5, gated on its own measurement.

- [ ] **Step 6: Version the engineered cache**

The whole-frame path writes `engineered_data.parquet` (`:3633`). After truncation that file holds a tail, not full history, so a cache written by the old code must not be reused. Add beside `ENGINEERED_CACHE_NAME` at `:273`:

```python
    # Bump when the *shape* of the cached frame changes, not just its contents.
    # v2: the predict path now truncates to PREDICT_TAIL_ITEM_DAYS rows per
    # item, so a v1 cache holds full history the loader would misread as a tail.
    ENGINEERED_CACHE_VERSION = 2
```

In `_save_engineered_cache` (`:2558`), add the version as a real column — `DataFrame.attrs` is not reliably persisted by `to_parquet`, which is why the existing `_cache_date` attr already falls through to the DuckDB freshness path:

```python
        df = df.copy()
        df["_cache_version"] = self.ENGINEERED_CACHE_VERSION
```

immediately before `df.to_parquet(path, index=False)`.

In `_load_engineered_cache` (`:2565`), directly after the `if df.empty:` check:

```python
            version = df["_cache_version"].iloc[0] if "_cache_version" in df.columns else 1
            if int(version) != self.ENGINEERED_CACHE_VERSION:
                logger.info(
                    f"  Cache at {path} is v{version}, expected "
                    f"v{self.ENGINEERED_CACHE_VERSION} — will refresh"
                )
                return None
            df = df.drop(columns=["_cache_version"])
```

- [ ] **Step 7: Add a cache-version test**

Append to `backend/tests/test_predict_tail_truncation.py`:

```python
def test_v1_engineered_cache_is_rejected(forecaster):
    """A pre-truncation cache holds full history, not a tail."""
    df = _price_frame(n_items=2, n_days=20)
    path = forecaster._engineered_cache_path
    df.to_parquet(path, index=False)          # no _cache_version column -> v1
    assert forecaster._load_engineered_cache() is None


def test_roundtripped_cache_is_accepted_without_the_version_column(forecaster):
    df = _price_frame(n_items=2, n_days=20)
    forecaster._save_engineered_cache(df)
    loaded = forecaster._load_engineered_cache()
    assert loaded is not None
    assert "_cache_version" not in loaded.columns
    assert len(loaded) == len(df)
```

- [ ] **Step 8: Run everything**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_predict_tail_truncation.py -v && ./venv/bin/python -m pytest tests/ -q && python3 -m py_compile models/forecaster.py
```

Expected: all 8 tests in the new file PASS, no new failures elsewhere.

If `test_roundtripped_cache_is_accepted_without_the_version_column` fails on the staleness branch, the cache's own 3-day freshness logic is intervening. Read `_load_engineered_cache` fully and place the version check where it runs before staleness handling.

- [ ] **Step 9: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_predict_tail_truncation.py
git commit -m "perf: engineer only the history prediction needs

predict() fetched 1460 days and kept PREDICT_TAIL_ROWS=3 per item, engineering
6.1M rows across two chunked passes to produce 8,691 feature vectors. Now
truncates to PREDICT_TAIL_ITEM_DAYS=240 rows per item after voting.

Row-based, not calendar-based: features mix 180-day calendar lag joins with
200-row positional rollings over a ~48%-dense archive. Voting collapses to one
row per item-day, so N rows span >= N calendar days and one parameter covers
both requirements. A no-op for items holding fewer than 240 rows.

test_served_features_survive_truncation asserts the served feature vector is
identical to the 1460-day path."
```

---

### Task 5: Narrow the DuckDB fetch window — **measure first, may be skipped**

**Files:**
- Modify: `backend/models/forecaster.py:3603` (only if the measurement clears the bar)

**Interfaces:**
- Consumes: `_tail_predict_frame` from Task 4.
- Produces: nothing new.

> **This task is conditional.** The spec gates it on a measurement. Task 4's tail is the larger win (6.1M → ~2M engineered rows across two passes); this only shortens the 137s fetch. **If the bar is not met, skip to Task 6 and leave `days_back=1460` in place.** That is a successful outcome, not a failure.

- [ ] **Step 1: Measure per-item row coverage**

Write a throwaway script (do not commit it) that loads the voted frame and reports, for candidate windows, the share of eligible items retaining `min(full_row_count, 240)` rows:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from unittest.mock import MagicMock
from models.forecaster import ItemForecaster

f = ItemForecaster(db_session=MagicMock(), model_dir="/tmp/measure-only")
full = f.fetch_price_history(days_back=1460, backfilled_only=True)

day_counts = full.groupby("item_id")["date"].nunique()
eligible = day_counts[day_counts >= f.PREDICT_MIN_HISTORY_DAYS].index
full = full[full["item_id"].isin(eligible)]

target = (full.groupby("item_id").size()
              .clip(upper=f.PREDICT_TAIL_ITEM_DAYS))
cutoff_max = full["date"].max()

for window in (365, 548, 730, 913, 1095):
    lo = cutoff_max - __import__("pandas").Timedelta(days=window)
    kept = full[full["date"] >= lo].groupby("item_id").size()
    kept = kept.reindex(target.index, fill_value=0)
    share = (kept >= target).mean()
    print(f"{window:>5}d: {share:.5%} of {len(target):,} items retain their target rows")
```

Run it:

```bash
cd backend && ./venv/bin/python /tmp/measure_predict_window.py
```

- [ ] **Step 2: Apply the decision rule**

Adopt the **smallest** window at which the share is **≥ 99.9%**. If no candidate reaches 99.9%, **skip this task entirely** — mark it skipped in the plan, leave `days_back=1460`, and record the measured table in the commit message for Task 6.

- [ ] **Step 3: Add the constant and use it (only if Step 2 selected a window)**

Beside `PREDICT_TAIL_ITEM_DAYS`:

```python
    # Calendar prefilter for the predict fetch, purely to shrink the DuckDB
    # scan and the voting cost. Measured on <DATE>: at this window
    # <SHARE>% of eligible items still retain min(full_rows,
    # PREDICT_TAIL_ITEM_DAYS) rows, so the served feature vectors are unchanged.
    # Re-measure before lowering it.
    PREDICT_FETCH_DAYS = <SELECTED_WINDOW>
```

Replace `:3603`:

```python
            price_df = self.fetch_price_history(
                days_back=self.PREDICT_FETCH_DAYS, backfilled_only=True)
```

Fill `<DATE>`, `<SHARE>`, and `<SELECTED_WINDOW>` with the real measured values. Do not commit the placeholders.

- [ ] **Step 4: Re-run the feature-equality test**

```bash
cd backend && ./venv/bin/python -m pytest tests/test_predict_tail_truncation.py -v && ./venv/bin/python -m pytest tests/ -q
```

The synthetic fixtures are dense, so they will not catch a real-archive shortfall. Step 1's measurement is the actual evidence for this task; the suite only confirms nothing regressed.

- [ ] **Step 5: Commit (only if a window was adopted)**

```bash
git add backend/models/forecaster.py
git commit -m "perf: narrow the predict fetch window to <SELECTED_WINDOW>d

Shrinks the DuckDB scan and the voting cost on the predict path. Measured:
<SHARE>% of eligible items still retain min(full_rows, 240) rows at this
window, so served feature vectors are unchanged."
```

---

### Task 6: Verify in CI and correct the timing docs

**Files:**
- Modify: `docs/architecture/model-optimization.md` (the Inference and Cold-retrain rows of the Current Baseline table)

**Interfaces:** none.

- [ ] **Step 1: Confirm the suite is green**

```bash
cd backend && ./venv/bin/python -m pytest tests/ -q && python3 -m py_compile models/forecaster.py scripts/forecast_prices.py
```

- [ ] **Step 2: Push the branch and dispatch a predict-only run**

```bash
git push -u origin HEAD
gh workflow run price-forecast.yml -f mode=predict-only --ref "$(git rev-parse --abbrev-ref HEAD)"
```

- [ ] **Step 3: Wait for it, then pull the step timings**

```bash
gh run list --workflow=price-forecast.yml --limit 1
# substitute the run id below
gh api repos/:owner/:repo/actions/runs/<RUN_ID>/jobs \
  --jq '.jobs[].steps[] | select(.conclusion!="skipped") | "\(.name): \(.started_at) -> \(.completed_at)"'
```

- [ ] **Step 4: Check the four acceptance criteria in the log**

```bash
gh run download <RUN_ID> -n forecast-logs-<RUN_ID> -D /tmp/verify && \
  grep -cE "TRAINING LIGHTGBM FORECASTER" /tmp/verify/forecast.log; \
  grep -E "Predict tail|Not retraining|RESULT:" /tmp/verify/forecast.log
```

| Criterion | Expected |
|---|---|
| No retrain occurred | `TRAINING LIGHTGBM FORECASTER` count is **0** |
| Drift still reported | a `Drift reported for horizons` line is present |
| Tail applied | a `Predict tail: 6,155,446 -> ...` line is present |
| Forecasts still written | `RESULT: {'status': 'success', 'items': ~8691, 'forecasts': ~34764, ...}` and the `Verify forecasts were persisted` step passes |

The item and forecast counts must be **within ~1%** of the 8,691 / 34,764 baseline. A large drop means the tail or the prefilter dropped items that should have been eligible — investigate before merging.

- [ ] **Step 5: Correct the timing rows in `model-optimization.md`**

Update the **Inference** row of the Current Baseline table with the measured new figure, and add a note to the table that the daily path no longer retrains. Replace the existing Inference row:

```markdown
| **Inference** | **Measured <NEW>s in CI (2026-08-04, run <RUN_ID>)**, down from 835s. The old "~1–2 min warm / ~5 min cold" figures described the predict phase alone and omitted that every non-Monday run auto-retrained first: drift fired unconditionally against a 60% threshold above the model's 46.7–50.8% DA, adding 465s. Fixed — see `docs/superpowers/specs/2026-08-04-remove-accidental-retrain-work-design.md` |
```

Fill `<NEW>` and `<RUN_ID>` from Step 3. Also strike the now-false claim in the **Verification Protocol** section that "`--predict-only` can trigger an unwanted retrain" — replace with a pointer to `ALLOW_DRIFT_RETRAIN=1`.

- [ ] **Step 6: Commit**

```bash
git add docs/architecture/model-optimization.md
git commit -m "docs: correct the inference timing baseline

The daily forecast step measured 835s, not the '~1-2 min' the table claimed:
465s of it was an unrequested retrain triggered by an unreachable drift
threshold. Now <NEW>s, verified on run <RUN_ID>."
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `check_concept_drift` keeps the `AccuracyAlert` write | 1 (`test_drift_alert_is_still_written`) |
| Date-coverage guard, reusing `date_coverage_sufficient`, fail-closed | 1 |
| 60.0 hoisted to a named constant citing measured DA | 1 |
| Predict-only branch stops triggering a retrain | 2 |
| `ALLOW_DRIFT_RETRAIN=1` escape hatch | 2 |
| `drifted` dropped from the full-mode condition | 3 |
| `_drift_detected` deleted | 3 |
| `PREDICT_TAIL_ITEM_DAYS = 240`, row-based tail after voting | 4 |
| Feature-equality test (load-bearing) | 4 (`test_served_features_survive_truncation`) |
| One-row-per-item-day invariant test | 4 (`test_voting_yields_one_row_per_item_day`) |
| Calendar prefilter, gated on a ≥99.9% coverage measurement | 5 |
| Verification: no training block, forecasts still written, timing compared | 6 |
| Accuracy deliberately not re-measured | Honoured — no task runs `walkforward_backtest.py` |

Spec items intentionally carrying **no** task, per the spec's own Out-of-Scope section: skipping computation of non-allowlisted features, caching the voted frame in CI, the 133-item subsample, and the `walkforward_backtest.py` aggregation flaw.

One addition beyond the spec: the **engineered-cache version** (Task 4, Steps 6–7). The spec did not name it, but truncating the frame changes what `engineered_data.parquet` holds, and the loader has no version check — a stale v1 cache would be read as a tail. It is a direct correctness consequence of Change 2, so it belongs here rather than in a later spec.

**Placeholder scan:** The only intentional placeholders are `<SELECTED_WINDOW>`, `<SHARE>`, `<DATE>`, `<RUN_ID>`, and `<NEW>` — all values that can only come from a measurement or a CI run, each with an explicit step producing it and an instruction not to commit the placeholder.

**Type consistency:** `_tail_predict_frame(price_df: pd.DataFrame) -> pd.DataFrame` is defined in Task 4 Step 3 and used in Task 4 Step 5 and Task 5 Step 1. `PREDICT_TAIL_ITEM_DAYS`, `ENGINEERED_CACHE_VERSION`, `DRIFT_DA_THRESHOLD`, and `PREDICT_FETCH_DAYS` are each defined once and referenced by those exact names. `check_concept_drift`'s `threshold` parameter is `Optional[float] = None` in Task 1 and called without it in Tasks 2 and 3.

## Notes for the implementer

- **Tasks 1–3 and Task 4 are independent.** Either can go first. Tasks 1→2→3 are ordered; 5 depends on 4; 6 is last.
- **The two load-bearing tests are `test_served_features_survive_truncation` and `test_voting_yields_one_row_per_item_day`.** If either fails, do not loosen it — stop and report. They are the only things standing between this change and train/serve skew.
- **The baseline is 417 passed** (measured on `363b659`). See Global Constraints for the exact test invocation — three plausible-looking variants all fail for different reasons.
- `DataFrame.attrs` does not survive `to_parquet`. The existing `_cache_date` attr at `:2561` is very likely already a no-op, which is why the loader has a DuckDB fallback. Do not follow that pattern for the version — use a real column, as Task 4 Step 6 does.
