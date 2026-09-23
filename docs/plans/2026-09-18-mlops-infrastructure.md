# MLOps Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experiment tracking (MLflow), walk-forward backtesting, data drift detection (Evidently), and a model registry to cs2-oracle so every experiment is reproducible, temporally rigorous, and comparable.

**Architecture:** MLflow wraps the existing `ItemForecaster.train()` loop — parent run per training session, child runs per horizon. Walk-forward backtesting is a new module that repeatedly calls `build_training_data` + `_train_horizon_inline` with sliding date windows and logs each fold to MLflow. Evidently drift detection runs as a separate daily step comparing current features to a reference snapshot. A new `/monitoring/` API router exposes drift reports and model registry info.

**Tech Stack:** MLflow (local file store), Evidently AI, FastAPI, pytest

**Spec:** `docs/specs/2026-09-18-ml-upgrade-design.md` (Phase 2 section)

## Global Constraints

- Training time must stay under 30 minutes total (currently ~14 min)
- CPU-only compute, free tooling only
- All env flags use `os.environ.get("FLAG_NAME") == "1"` pattern (not pydantic Settings)
- Tests use `pytest` with `MagicMock`, `monkeypatch.setenv`, `tmp_path`
- Run tests from `backend/` via `venv/bin/python -m pytest tests/test_<name>.py -q`
- Models live in `backend/models/`, API routes in `backend/api/routes/`, scripts in `backend/scripts/`
- The `.env` in `backend/` points at production — never run scripts from that directory without `ENVIRONMENT=development`
- `mlruns/` directory must be gitignored

---

### Task 1: MLflow Integration into Training Loop

**Files:**
- Modify: `backend/models/forecaster.py` (lines ~6048-6152, the `train()` method; lines ~6154+, `_train_horizon_inline`; lines ~11004+, `save_models`)
- Modify: `backend/scripts/forecast_prices.py` (the `run_forecast` entry point)
- Create: `backend/tests/test_mlflow_tracking.py`
- Modify: `.gitignore` (add `mlruns/`, `mlartifacts/`)

**Interfaces:**
- Consumes: `ItemForecaster.train()`, `ItemForecaster.save_models()`, `ItemForecaster.cv_results`, `ItemForecaster.conformal_calibration`
- Produces: MLflow runs with logged params/metrics/artifacts accessible via `mlflow.get_run()`, `mlflow.search_runs()`

- [x] **Step 1: Add mlflow dependency**

```bash
cd backend && venv/bin/pip install mlflow
```

- [x] **Step 2: Add mlruns to gitignore**

Append to the repo root `.gitignore`:
```
# MLflow
mlruns/
mlartifacts/
```

- [x] **Step 3: Write the failing test**

Create `backend/tests/test_mlflow_tracking.py`:

```python
"""Tests for MLflow tracking integration in the training loop."""

import json
import os

import mlflow
import pytest


class TestMLflowTraining:
    """Verify that a training run logs params, metrics, and artifacts to MLflow."""

    def test_train_creates_mlflow_run(self, tmp_path, monkeypatch):
        """A train() call should create an MLflow run with expected params."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test")

        from models.forecaster import ItemForecaster

        forecaster = _make_test_forecaster(tmp_path, monkeypatch)
        forecaster.train(max_rows=500, max_feature_rows=2000, min_median_price=None)

        runs = mlflow.search_runs(experiment_names=["test"])
        assert len(runs) >= 1, "Expected at least one MLflow run from train()"

        run = runs.iloc[0]
        assert "params.min_median_price" in runs.columns
        assert "params.max_rows" in runs.columns

    def test_train_logs_per_horizon_metrics(self, tmp_path, monkeypatch):
        """Each horizon should log MAE and q_hat as metrics."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-hz")

        from models.forecaster import ItemForecaster

        forecaster = _make_test_forecaster(tmp_path, monkeypatch)
        forecaster.train(max_rows=500, max_feature_rows=2000, min_median_price=None)

        runs = mlflow.search_runs(experiment_names=["test-hz"])
        parent_runs = runs[runs["tags.mlflow.parentRunId"].isna()]
        assert len(parent_runs) >= 1

        run = parent_runs.iloc[0]
        for h in [3, 7, 14, 30]:
            assert f"metrics.mae_{h}d" in runs.columns or f"metrics.q_hat_{h}d" in runs.columns

    def test_train_logs_model_artifact(self, tmp_path, monkeypatch):
        """The meta.json artifact should be logged."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-artifact")

        from models.forecaster import ItemForecaster

        forecaster = _make_test_forecaster(tmp_path, monkeypatch)
        forecaster.train(max_rows=500, max_feature_rows=2000, min_median_price=None)

        runs = mlflow.search_runs(experiment_names=["test-artifact"])
        parent_runs = runs[runs["tags.mlflow.parentRunId"].isna()]
        run_id = parent_runs.iloc[0]["run_id"]

        client = mlflow.tracking.MlflowClient()
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert "meta.json" in artifacts, f"Expected meta.json in artifacts, got {artifacts}"

    def test_mlflow_disabled_by_default(self, tmp_path, monkeypatch):
        """Without MLFLOW_ENABLED=1, no MLflow calls should be made."""
        monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

        from models.forecaster import ItemForecaster

        forecaster = _make_test_forecaster(tmp_path, monkeypatch)
        forecaster.train(max_rows=500, max_feature_rows=2000, min_median_price=None)

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        runs = mlflow.search_runs(search_all_experiments=True)
        assert len(runs) == 0, "No runs should be logged when MLFLOW_ENABLED is unset"


def _make_test_forecaster(tmp_path, monkeypatch):
    """Build a minimal ItemForecaster pointing at the test fixture DB."""
    from unittest.mock import MagicMock

    monkeypatch.setenv("MLFLOW_ENABLED", "1")
    monkeypatch.chdir(tmp_path)

    from models.forecaster import ItemForecaster

    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    forecaster = ItemForecaster(db_session=MagicMock(), model_dir=model_dir)
    return forecaster
```

- [x] **Step 4: Run test to verify it fails**

```bash
cd backend && venv/bin/python -m pytest tests/test_mlflow_tracking.py -q
```

Expected: FAIL — `mlflow` not imported or no MLflow logging in `train()`.

- [x] **Step 5: Implement MLflow tracking wrapper**

Add a helper module `backend/models/mlflow_utils.py` to keep the tracking logic separate from the forecaster:

```python
"""MLflow tracking utilities for the training pipeline.

Gated by MLFLOW_ENABLED=1. When off, every function is a no-op.
"""

import json
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_ENABLED = None


def mlflow_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("MLFLOW_ENABLED") == "1"
    return _ENABLED


def _reset():
    """Test helper: re-read the env flag on next call."""
    global _ENABLED
    _ENABLED = None


@contextmanager
def training_run(experiment_name: str = "cs2-oracle", params: dict | None = None):
    """Context manager that wraps a training session in an MLflow run.

    Yields the run object (or None when disabled). Caller logs metrics
    and artifacts through the module-level helpers below.
    """
    if not mlflow_enabled():
        yield None
        return

    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield run


def log_horizon_metrics(horizon: int, metrics: dict):
    """Log per-horizon metrics (MAE, q_hat, coverage, etc.)."""
    if not mlflow_enabled():
        return
    import mlflow

    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            flat[f"{k}_{horizon}d"] = v
    mlflow.log_metrics(flat)


def log_artifact_file(path: str):
    """Log a file as an MLflow artifact."""
    if not mlflow_enabled():
        return
    import mlflow

    mlflow.log_artifact(path)


def log_meta_json(meta: dict, tmp_dir: str):
    """Write meta.json to tmp_dir and log it as an artifact."""
    if not mlflow_enabled():
        return
    import mlflow

    path = os.path.join(tmp_dir, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, default=str)
    mlflow.log_artifact(path)
```

- [x] **Step 6: Wire MLflow into `forecaster.py::train()`**

In `backend/models/forecaster.py`, at the top of `train()` (after `_train_start`), wrap the body in the tracking context:

```python
from models import mlflow_utils

# Inside train(), after _train_start = datetime.now()
with mlflow_utils.training_run(
    params={
        "max_rows": max_rows,
        "max_feature_rows": max_feature_rows,
        "min_median_price": min_median_price,
        "per_item_row_sampling": per_item_row_sampling,
        "train_days_back": train_days_back,
        "climatology_scale": self.climatology_scale_enabled(),
        "exceedance_head": self.exceedance_head_enabled(),
        "anomaly_gbm": self.anomaly_gbm_enabled(),
        "feature_native_nan": self.feature_native_nan_enabled(),
    }
):
    # ... existing train body ...
```

At the end of each `_train_horizon_inline()`, after computing cv_results, add:

```python
mlflow_utils.log_horizon_metrics(horizon, {
    "mae": cv_data.get("mae", 0),
    "q_hat": self.conformal_calibration.get(horizon, {}).get("q_hat", 0),
    "q_lo": float(self.conformal_q_lo.get(horizon, 0)),
    "q_hi": float(self.conformal_q_hi.get(horizon, 0)),
})
```

In `save_models()`, after writing `meta.json` to disk, add:

```python
mlflow_utils.log_artifact_file(os.path.join(self.model_dir, "meta.json"))
```

- [x] **Step 7: Run tests to verify they pass**

```bash
cd backend && venv/bin/python -m pytest tests/test_mlflow_tracking.py -q
```

Expected: tests that can run with the test fixture should pass. Tests requiring full training data will need the `price-archive/` — mark those with `@pytest.mark.skipif` if the archive isn't present.

- [x] **Step 8: Commit**

```bash
git add backend/models/mlflow_utils.py backend/tests/test_mlflow_tracking.py backend/models/forecaster.py .gitignore
git commit -m "feat: add MLflow experiment tracking to training loop

Gated by MLFLOW_ENABLED=1. Logs params, per-horizon metrics
(MAE, q_hat, q_lo, q_hi), and meta.json artifact per run."
```

---

### Task 2: Walk-Forward Temporal Backtesting

**Files:**
- Create: `backend/models/walk_forward_eval.py`
- Create: `backend/tests/test_walk_forward_eval.py`

**Interfaces:**
- Consumes: `ItemForecaster.build_training_data()`, `ItemForecaster._train_horizon_inline()`, `ItemForecaster.predict()`, `mlflow_utils.training_run()`
- Produces: `WalkForwardResult` dataclass with per-fold coverage, width, MAE, IC; overall aggregates; MLflow child runs per fold

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_walk_forward_eval.py`:

```python
"""Tests for the walk-forward temporal backtesting module."""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


class TestWalkForwardSplits:
    """Verify that the walk-forward splitter produces correct temporal folds."""

    def test_splits_are_non_overlapping(self):
        from models.walk_forward_eval import generate_wf_splits

        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        splits = generate_wf_splits(
            dates,
            train_window_days=365,
            test_window_days=30,
            stride_days=30,
            embargo_days=43,
        )
        assert len(splits) >= 5, f"Expected >=5 folds, got {len(splits)}"

        for i, split in enumerate(splits):
            assert split.train_end < split.test_start, (
                f"Fold {i}: train_end {split.train_end} >= test_start {split.test_start}"
            )
            gap = (split.test_start - split.train_end).days
            assert gap >= 43, (
                f"Fold {i}: embargo gap {gap} < 43 days"
            )

    def test_splits_advance_monotonically(self):
        from models.walk_forward_eval import generate_wf_splits

        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        splits = generate_wf_splits(
            dates,
            train_window_days=365,
            test_window_days=30,
            stride_days=30,
            embargo_days=43,
        )
        for i in range(1, len(splits)):
            assert splits[i].test_start > splits[i - 1].test_start, (
                f"Fold {i} test_start does not advance"
            )

    def test_embargo_respects_horizon(self):
        """Embargo = horizon + 13 per project invariant #3."""
        from models.walk_forward_eval import generate_wf_splits

        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        for horizon in [3, 7, 14, 30]:
            embargo = horizon + 13
            splits = generate_wf_splits(
                dates,
                train_window_days=365,
                test_window_days=30,
                stride_days=30,
                embargo_days=embargo,
            )
            for fold in splits:
                gap = (fold.test_start - fold.train_end).days
                assert gap >= embargo


class TestIntervalScoring:
    """Verify interval scoring metrics are correct."""

    def test_perfect_coverage(self):
        from models.walk_forward_eval import score_intervals

        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        hi = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        result = score_intervals(actual, lo, hi)
        assert result.coverage == pytest.approx(1.0)
        assert result.mean_width == pytest.approx(1.0)

    def test_zero_coverage(self):
        from models.walk_forward_eval import score_intervals

        actual = np.array([10.0, 20.0, 30.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 1.0, 1.0])

        result = score_intervals(actual, lo, hi)
        assert result.coverage == pytest.approx(0.0)

    def test_partial_coverage(self):
        from models.walk_forward_eval import score_intervals

        actual = np.array([1.0, 5.0])
        lo = np.array([0.5, 0.5])
        hi = np.array([1.5, 1.5])

        result = score_intervals(actual, lo, hi)
        assert result.coverage == pytest.approx(0.5)
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && venv/bin/python -m pytest tests/test_walk_forward_eval.py -q
```

Expected: FAIL — module `models.walk_forward_eval` does not exist.

- [x] **Step 3: Implement walk-forward module**

Create `backend/models/walk_forward_eval.py`:

```python
"""Walk-forward temporal backtesting for interval forecasts.

Produces temporally honest evaluation: train on [T-W, T], predict [T+embargo, T+embargo+test],
slide T forward by stride, repeat. Each fold is logged to MLflow as a child run.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """One temporal fold's date boundaries."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    fold_idx: int


@dataclass
class IntervalScore:
    """Metrics for a set of prediction intervals."""

    coverage: float
    mean_width: float
    median_width: float = 0.0
    n: int = 0


@dataclass
class FoldResult:
    """Results from one walk-forward fold."""

    fold_idx: int
    split: WalkForwardSplit
    interval_score: IntervalScore
    mae: float = 0.0
    ic: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward evaluation results."""

    horizon: int
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_coverage(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.interval_score.coverage for f in self.folds]))

    @property
    def mean_width(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.interval_score.mean_width for f in self.folds]))

    @property
    def mean_mae(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.mae for f in self.folds]))

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def generate_wf_splits(
    dates: pd.DatetimeIndex,
    train_window_days: int = 365,
    test_window_days: int = 30,
    stride_days: int = 30,
    embargo_days: int = 43,
) -> list[WalkForwardSplit]:
    """Generate walk-forward date splits with embargo.

    The embargo enforces invariant #3: never pass a bare horizon to a purge.
    For horizon h, the caller should pass embargo_days = h + 13.
    """
    min_date = dates.min().date() if hasattr(dates.min(), "date") else dates.min()
    max_date = dates.max().date() if hasattr(dates.max(), "date") else dates.max()

    splits = []
    fold_idx = 0
    train_start = min_date

    while True:
        train_end = train_start + timedelta(days=train_window_days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_window_days)

        if test_end > max_date:
            break

        splits.append(
            WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_idx=fold_idx,
            )
        )
        fold_idx += 1
        train_start += timedelta(days=stride_days)

    return splits


def score_intervals(
    actual: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> IntervalScore:
    """Score prediction intervals: coverage, width, and adaptivity."""
    mask = np.isfinite(actual) & np.isfinite(lo) & np.isfinite(hi)
    actual, lo, hi = actual[mask], lo[mask], hi[mask]
    n = len(actual)
    if n == 0:
        return IntervalScore(coverage=0.0, mean_width=0.0, n=0)

    covered = (actual >= lo) & (actual <= hi)
    widths = hi - lo

    return IntervalScore(
        coverage=float(covered.mean()),
        mean_width=float(widths.mean()),
        median_width=float(np.median(widths)),
        n=n,
    )
```

- [x] **Step 4: Run tests to verify they pass**

```bash
cd backend && venv/bin/python -m pytest tests/test_walk_forward_eval.py -q
```

Expected: PASS

- [x] **Step 5: Commit**

```bash
git add backend/models/walk_forward_eval.py backend/tests/test_walk_forward_eval.py
git commit -m "feat: add walk-forward temporal backtesting module

Generates embargoed temporal splits (invariant #3: embargo = horizon + 13)
and scores prediction intervals on coverage, width, and adaptivity."
```

---

### Task 3: Data Drift Detection with Evidently

**Files:**
- Create: `backend/monitoring/__init__.py`
- Create: `backend/monitoring/drift.py`
- Create: `backend/tests/test_drift_detection.py`

**Interfaces:**
- Consumes: Feature DataFrames from `ItemForecaster.build_training_data()` / `engineer_features()`
- Produces: `DriftReport` dataclass with per-feature drift scores, overall drift detected flag, and JSON/HTML report paths

- [x] **Step 1: Install evidently**

```bash
cd backend && venv/bin/pip install evidently
```

- [x] **Step 2: Write the failing test**

Create `backend/tests/test_drift_detection.py`:

```python
"""Tests for Evidently-based data drift detection."""

import numpy as np
import pandas as pd
import pytest


class TestDriftDetection:
    """Verify drift detection on synthetic data."""

    def test_no_drift_on_identical_data(self, tmp_path):
        from monitoring.drift import detect_drift

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "price": rng.normal(100, 10, 1000),
            "return_1d": rng.normal(0, 0.02, 1000),
            "volume": rng.poisson(50, 1000).astype(float),
        })

        report = detect_drift(
            reference=df,
            current=df,
            output_dir=str(tmp_path),
        )
        assert not report.drift_detected
        assert report.n_drifted_features == 0

    def test_detects_mean_shift(self, tmp_path):
        from monitoring.drift import detect_drift

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "price": rng.normal(100, 10, 1000),
            "return_1d": rng.normal(0, 0.02, 1000),
        })
        cur = pd.DataFrame({
            "price": rng.normal(200, 10, 1000),
            "return_1d": rng.normal(0.1, 0.02, 1000),
        })

        report = detect_drift(
            reference=ref,
            current=cur,
            output_dir=str(tmp_path),
        )
        assert report.drift_detected
        assert report.n_drifted_features >= 1

    def test_report_files_written(self, tmp_path):
        from monitoring.drift import detect_drift
        import os

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"price": rng.normal(100, 10, 500)})

        report = detect_drift(
            reference=df,
            current=df,
            output_dir=str(tmp_path),
        )
        assert os.path.exists(report.json_path)
        assert os.path.exists(report.html_path)

    def test_per_feature_scores(self, tmp_path):
        from monitoring.drift import detect_drift

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "stable": rng.normal(0, 1, 1000),
            "drifted": rng.normal(0, 1, 1000),
        })
        cur = pd.DataFrame({
            "stable": rng.normal(0, 1, 1000),
            "drifted": rng.normal(10, 1, 1000),
        })

        report = detect_drift(reference=ref, current=cur, output_dir=str(tmp_path))
        assert "drifted" in report.drifted_features
```

- [x] **Step 3: Run test to verify it fails**

```bash
cd backend && venv/bin/python -m pytest tests/test_drift_detection.py -q
```

Expected: FAIL — `monitoring.drift` does not exist.

- [x] **Step 4: Implement drift detection module**

Create `backend/monitoring/__init__.py` (empty file).

Create `backend/monitoring/drift.py`:

```python
"""Data drift detection using Evidently AI.

Compares a reference feature distribution to a current one and produces
a drift report (JSON + HTML). Catches frozen quotes, source changes,
and label seams.
"""

import json
import logging
import os
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Results of a drift detection run."""

    drift_detected: bool
    n_drifted_features: int
    n_total_features: int
    drifted_features: list[str] = field(default_factory=list)
    feature_scores: dict[str, float] = field(default_factory=dict)
    json_path: str = ""
    html_path: str = ""


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str = "data/drift_reports",
    feature_cols: list[str] | None = None,
) -> DriftReport:
    """Run Evidently drift detection on numeric columns.

    Args:
        reference: The reference (training) feature DataFrame.
        current: The current (serving) feature DataFrame.
        output_dir: Where to write JSON and HTML reports.
        feature_cols: Subset of columns to check. None = all numeric columns.

    Returns:
        DriftReport with per-feature drift scores and overall flag.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    os.makedirs(output_dir, exist_ok=True)

    if feature_cols:
        ref = reference[feature_cols].select_dtypes(include="number")
        cur = current[feature_cols].select_dtypes(include="number")
    else:
        common = list(set(reference.columns) & set(current.columns))
        ref = reference[common].select_dtypes(include="number")
        cur = current[common].select_dtypes(include="number")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    html_path = os.path.join(output_dir, "drift_report.html")
    json_path = os.path.join(output_dir, "drift_report.json")

    report.save_html(html_path)
    report.save_json(json_path)

    with open(json_path) as f:
        result = json.load(f)

    drifted_features = []
    feature_scores = {}
    n_drifted = 0

    metrics = result.get("metrics", [])
    for metric in metrics:
        mr = metric.get("result", {})
        if "drift_by_columns" in mr:
            for col_name, col_data in mr["drift_by_columns"].items():
                score = col_data.get("drift_score", 0.0)
                is_drifted = col_data.get("drift_detected", False)
                feature_scores[col_name] = score
                if is_drifted:
                    drifted_features.append(col_name)
                    n_drifted += 1

    overall_drift = n_drifted > 0

    return DriftReport(
        drift_detected=overall_drift,
        n_drifted_features=n_drifted,
        n_total_features=len(ref.columns),
        drifted_features=drifted_features,
        feature_scores=feature_scores,
        json_path=json_path,
        html_path=html_path,
    )
```

- [x] **Step 5: Run tests to verify they pass**

```bash
cd backend && venv/bin/python -m pytest tests/test_drift_detection.py -q
```

Expected: PASS

- [x] **Step 6: Commit**

```bash
git add backend/monitoring/__init__.py backend/monitoring/drift.py backend/tests/test_drift_detection.py
git commit -m "feat: add Evidently-based data drift detection

Compares reference vs current feature distributions, writes
JSON + HTML reports, and flags drifted features."
```

---

### Task 4: Monitoring API Routes

**Files:**
- Create: `backend/api/routes/monitoring.py`
- Modify: `backend/main.py` (register new router)
- Create: `backend/tests/test_monitoring_routes.py`

**Interfaces:**
- Consumes: `monitoring.drift.DriftReport`, `mlflow` (for model registry info), drift report JSON files on disk
- Produces: `/monitoring/drift` (latest drift report), `/monitoring/model-info` (current model metadata), `/monitoring/health` (pipeline health summary)

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_monitoring_routes.py`:

```python
"""Tests for the monitoring API routes."""

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///")
    from main import app
    return TestClient(app)


class TestDriftEndpoint:
    def test_drift_returns_empty_when_no_report(self, client):
        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_report"

    def test_drift_returns_report_when_present(self, client, tmp_path, monkeypatch):
        report_dir = tmp_path / "drift_reports"
        report_dir.mkdir()
        report_data = {
            "drift_detected": True,
            "n_drifted_features": 2,
            "n_total_features": 10,
            "drifted_features": ["price", "return_1d"],
        }
        (report_dir / "drift_summary.json").write_text(json.dumps(report_data))
        monkeypatch.setenv("DRIFT_REPORT_DIR", str(report_dir))

        response = client.get("/monitoring/drift")
        assert response.status_code == 200


class TestModelInfoEndpoint:
    def test_model_info_returns_metadata(self, client, tmp_path, monkeypatch):
        model_dir = tmp_path / "saved_models"
        model_dir.mkdir()
        meta = {
            "trained_at": "2026-09-18T00:00:00",
            "model_artifact_version": 6,
            "climatology_scale": True,
            "conformal_calibration": {"3": {"q_hat": 95.5}},
        }
        (model_dir / "meta.json").write_text(json.dumps(meta))
        monkeypatch.setenv("MODEL_DIR", str(model_dir))

        response = client.get("/monitoring/model-info")
        assert response.status_code == 200


class TestHealthEndpoint:
    def test_monitoring_health(self, client):
        response = client.get("/monitoring/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && venv/bin/python -m pytest tests/test_monitoring_routes.py -q
```

Expected: FAIL — monitoring router not registered.

- [x] **Step 3: Implement monitoring routes**

Create `backend/api/routes/monitoring.py`:

```python
"""Monitoring API routes: drift reports, model info, pipeline health."""

import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_DEFAULT_DRIFT_DIR = os.path.join("data", "drift_reports")
_DEFAULT_MODEL_DIR = os.path.join("models", "saved_models")


def _drift_report_dir() -> str:
    return os.environ.get("DRIFT_REPORT_DIR", _DEFAULT_DRIFT_DIR)


def _model_dir() -> str:
    return os.environ.get("MODEL_DIR", _DEFAULT_MODEL_DIR)


@router.get("/drift")
def get_drift_report():
    """Return the latest drift detection report."""
    drift_dir = _drift_report_dir()
    summary_path = os.path.join(drift_dir, "drift_summary.json")

    if not os.path.exists(summary_path):
        return {"status": "no_report", "message": "No drift report found. Run drift detection first."}

    with open(summary_path) as f:
        report = json.load(f)

    return {
        "status": "ok",
        "report": report,
        "generated_at": datetime.fromtimestamp(os.path.getmtime(summary_path)).isoformat(),
    }


@router.get("/model-info")
def get_model_info():
    """Return current model metadata from meta.json."""
    model_dir = _model_dir()
    meta_path = os.path.join(model_dir, "meta.json")

    if not os.path.exists(meta_path):
        return {"status": "no_model", "message": "No model artifact found."}

    with open(meta_path) as f:
        meta = json.load(f)

    safe_keys = [
        "model_artifact_version", "trained_at", "climatology_scale",
        "exceedance_scale", "anomaly_gbm", "feature_native_nan",
        "naive_init_score", "conformal_calibration", "conformal_beta",
        "conformal_q_lo", "conformal_q_hi", "feature_importance",
        "cv_results", "training_window_days", "n_ensembles",
        "sigma_clip", "served_coverage_factor",
    ]
    filtered = {k: meta[k] for k in safe_keys if k in meta}

    return {"status": "ok", "model": filtered}


@router.get("/health")
def monitoring_health():
    """Pipeline health summary: model age, drift status, data freshness."""
    model_dir = _model_dir()
    meta_path = os.path.join(model_dir, "meta.json")

    health = {"status": "ok", "checks": {}}

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        trained_at = meta.get("trained_at", "unknown")
        health["checks"]["model_age"] = {"trained_at": trained_at}
    else:
        health["checks"]["model_age"] = {"trained_at": None, "warning": "No model found"}

    drift_dir = _drift_report_dir()
    summary_path = os.path.join(drift_dir, "drift_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            drift = json.load(f)
        health["checks"]["drift"] = {
            "drift_detected": drift.get("drift_detected", False),
            "n_drifted": drift.get("n_drifted_features", 0),
        }
    else:
        health["checks"]["drift"] = {"status": "no_report"}

    return health
```

- [x] **Step 4: Register the router in main.py**

In `backend/main.py`, add the import and include:

```python
from api.routes import ab_test, accuracy, auth, events, items, market, monitoring, opportunities

# ... existing includes ...
app.include_router(monitoring.router)
```

- [x] **Step 5: Run tests to verify they pass**

```bash
cd backend && venv/bin/python -m pytest tests/test_monitoring_routes.py -q
```

Expected: PASS

- [x] **Step 6: Commit**

```bash
git add backend/api/routes/monitoring.py backend/main.py backend/tests/test_monitoring_routes.py
git commit -m "feat: add monitoring API routes

Endpoints: /monitoring/drift (latest drift report),
/monitoring/model-info (model metadata),
/monitoring/health (pipeline health summary)."
```

---

### Task 5: Drift Detection Script for CI

**Files:**
- Create: `backend/scripts/run_drift_check.py`
- Create: `backend/tests/test_run_drift_check.py`

**Interfaces:**
- Consumes: `monitoring.drift.detect_drift()`, `ItemForecaster.build_training_data()`, `ItemForecaster.engineer_features()`
- Produces: Drift report files in `data/drift_reports/`, exit code 0 (no drift) or 1 (drift detected)

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_run_drift_check.py`:

```python
"""Tests for the drift check CI script."""

import pytest


class TestDriftCheckScript:
    def test_module_imports(self):
        """The drift check script should be importable."""
        from scripts import run_drift_check
        assert hasattr(run_drift_check, "main")

    def test_reference_snapshot_roundtrip(self, tmp_path):
        """Saving and loading a reference snapshot should be lossless."""
        import pandas as pd
        import numpy as np
        from scripts.run_drift_check import save_reference, load_reference

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "price": rng.normal(100, 10, 100),
            "return_1d": rng.normal(0, 0.02, 100),
        })
        path = str(tmp_path / "ref.parquet")

        save_reference(ref, path)
        loaded = load_reference(path)

        pd.testing.assert_frame_equal(ref, loaded)
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && venv/bin/python -m pytest tests/test_run_drift_check.py -q
```

Expected: FAIL — module `scripts.run_drift_check` does not exist.

- [x] **Step 3: Implement drift check script**

Create `backend/scripts/run_drift_check.py`:

```python
"""Run data drift detection and save reports.

Usage:
    venv/bin/python -m scripts.run_drift_check [--save-reference] [--output-dir DIR]

When --save-reference is passed, the current feature frame is saved as
the reference snapshot for future comparisons. Without it, the current
frame is compared against the saved reference.
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.path.join("data", "drift_reports")
DEFAULT_REF_PATH = os.path.join("data", "drift_reference.parquet")


def save_reference(df: pd.DataFrame, path: str = DEFAULT_REF_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Reference snapshot saved: {path} ({len(df)} rows, {len(df.columns)} cols)")


def load_reference(path: str = DEFAULT_REF_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(description="Run data drift detection")
    parser.add_argument("--save-reference", action="store_true", help="Save current features as reference")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ref-path", default=DEFAULT_REF_PATH)
    args = parser.parse_args()

    from unittest.mock import MagicMock

    from models.forecaster import ItemForecaster

    logger.info("Building current feature frame...")
    forecaster = ItemForecaster(db_session=MagicMock())
    df = forecaster.build_training_data(
        days_back=90,
        backfilled_only=True,
        max_feature_rows=100_000,
        min_median_price=1.0,
        universe="train",
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_frame = df[numeric_cols]

    if args.save_reference:
        save_reference(feature_frame, args.ref_path)
        return

    if not os.path.exists(args.ref_path):
        logger.warning(f"No reference snapshot at {args.ref_path}. Run with --save-reference first.")
        save_reference(feature_frame, args.ref_path)
        logger.info("Saved current frame as reference. No comparison to make on first run.")
        return

    reference = load_reference(args.ref_path)
    logger.info(f"Reference: {len(reference)} rows, {len(reference.columns)} cols")
    logger.info(f"Current: {len(feature_frame)} rows, {len(feature_frame.columns)} cols")

    from monitoring.drift import detect_drift

    report = detect_drift(
        reference=reference,
        current=feature_frame,
        output_dir=args.output_dir,
    )

    summary = {
        "drift_detected": report.drift_detected,
        "n_drifted_features": report.n_drifted_features,
        "n_total_features": report.n_total_features,
        "drifted_features": report.drifted_features,
        "feature_scores": report.feature_scores,
    }
    summary_path = os.path.join(args.output_dir, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Drift detected: {report.drift_detected}")
    logger.info(f"Drifted features ({report.n_drifted_features}/{report.n_total_features}): {report.drifted_features}")

    if report.drift_detected:
        logger.warning("DRIFT DETECTED — review drift_report.html")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run tests to verify they pass**

```bash
cd backend && venv/bin/python -m pytest tests/test_run_drift_check.py -q
```

Expected: PASS

- [x] **Step 5: Add drift check to GH Actions workflow**

In `.github/workflows/price-forecast.yml`, add a step after the forecast step that runs drift detection. This runs only on the daily `predict-only` mode:

```yaml
    - name: Run drift detection
      if: steps.mode.outputs.mode == 'predict-only'
      working-directory: backend
      run: |
        python -m scripts.run_drift_check --output-dir data/drift_reports
      env:
        DRIFT_REPORT_DIR: data/drift_reports
      continue-on-error: true  # drift is advisory, not blocking
```

- [x] **Step 6: Commit**

```bash
git add backend/scripts/run_drift_check.py backend/tests/test_run_drift_check.py .github/workflows/price-forecast.yml
git commit -m "feat: add drift detection CI script

Compares current feature distributions against a saved reference
snapshot. Runs in GH Actions after forecast (advisory, non-blocking)."
```

---

### Task 6: MLflow Model Registry Integration

**Files:**
- Modify: `backend/models/mlflow_utils.py` (add registry helpers)
- Create: `backend/tests/test_model_registry.py`

**Interfaces:**
- Consumes: MLflow runs from Task 1
- Produces: `register_model()` and `promote_model()` functions that move models between staging/production in MLflow's local registry

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_model_registry.py`:

```python
"""Tests for MLflow model registry helpers."""

import os
import pytest
import mlflow


class TestModelRegistry:
    def test_register_model(self, tmp_path, monkeypatch):
        from models.mlflow_utils import register_model, _reset

        monkeypatch.setenv("MLFLOW_ENABLED", "1")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
        _reset()

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment("test-registry")

        with mlflow.start_run() as run:
            mlflow.log_param("test", "true")
            run_id = run.info.run_id

        mv = register_model(run_id, "cs2-oracle-forecaster")
        assert mv is not None
        assert mv.name == "cs2-oracle-forecaster"

    def test_register_noop_when_disabled(self, tmp_path, monkeypatch):
        from models.mlflow_utils import register_model, _reset

        monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
        _reset()

        result = register_model("fake-run-id", "cs2-oracle-forecaster")
        assert result is None
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && venv/bin/python -m pytest tests/test_model_registry.py -q
```

Expected: FAIL — `register_model` not defined.

- [x] **Step 3: Implement registry helpers**

Add to `backend/models/mlflow_utils.py`:

```python
def register_model(run_id: str, model_name: str = "cs2-oracle-forecaster"):
    """Register a training run's artifacts as a model version."""
    if not mlflow_enabled():
        return None
    import mlflow

    try:
        uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(uri, model_name)
        logger.info(f"Registered model {model_name} version {mv.version}")
        return mv
    except Exception as e:
        logger.warning(f"Model registration failed: {e}")
        return None


def promote_model(model_name: str, version: int, stage: str = "Production"):
    """Transition a model version to a new stage (Staging/Production/Archived)."""
    if not mlflow_enabled():
        return None
    import mlflow

    client = mlflow.tracking.MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name, version=str(version), stage=stage
        )
        logger.info(f"Promoted {model_name} v{version} to {stage}")
        return True
    except Exception as e:
        logger.warning(f"Model promotion failed: {e}")
        return None
```

- [x] **Step 4: Run tests to verify they pass**

```bash
cd backend && venv/bin/python -m pytest tests/test_model_registry.py -q
```

Expected: PASS

- [x] **Step 5: Commit**

```bash
git add backend/models/mlflow_utils.py backend/tests/test_model_registry.py
git commit -m "feat: add MLflow model registry helpers

register_model() and promote_model() for staging/production
lifecycle tracking, gated by MLFLOW_ENABLED=1."
```
