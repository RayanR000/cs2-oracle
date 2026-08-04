"""Drift detection must not trigger retraining, and must not read a
degenerate sample as evidence.

check_concept_drift averages stored prediction_accuracy rows. Those rows span
1-2 distinct forecast dates while carrying five-figure row counts, so the
average tracks which way the market moved rather than model decay. See
docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
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


# ---------------------------------------------------------------------------
# The predict-only path must not retrain.
# ---------------------------------------------------------------------------

FORECAST_PRICES_SRC = (
    Path(__file__).resolve().parent.parent / "scripts" / "forecast_prices.py"
)


def test_predict_only_branch_has_no_retrain_trigger():
    """The --predict-only branch must not set do_train unconditionally.

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


# ---------------------------------------------------------------------------
# The same contract, asserted behaviourally.
#
# The source-text tests above pin the shape of the code; they cannot see what it
# evaluates to. Flipping the opt-in from `== "1"` to `!= "0"` reinstates
# unconditional retraining and still satisfies them. These drive run_forecast()
# with a fake forecaster and assert on whether train() is actually called.
# ---------------------------------------------------------------------------

class _TrainCalled(Exception):
    """Raised by the fake's train() so the run stops before any real DB work."""


def _fake_forecast_env(monkeypatch, tmp_path, *, drifted, trained_days_ago=0):
    """Patch run_forecast's collaborators. Returns the fake forecaster class."""
    import json

    import scripts.forecast_prices as fp

    (tmp_path / "meta.json").write_text(json.dumps({
        "trained_at": (datetime.now(timezone.utc)
                       - timedelta(days=trained_days_ago)).isoformat(),
    }))

    class FakeForecaster:
        HORIZONS = [3, 7, 14, 30]
        train_called = False

        def __init__(self, *a, **kw):
            self.model_dir = str(tmp_path)
            self.db = None
            type(self).instance = self

        def load_models(self):
            return True

        def check_concept_drift(self, horizon=7, sliding_window=7, threshold=None):
            return {"drifted": drifted, "accuracy": 28.0, "threshold": 60.0}

        def train(self, *a, **kw):
            type(self).train_called = True
            raise _TrainCalled

        def predict(self):
            return pd.DataFrame()

    fake_db = MagicMock()
    fake_db.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr(fp, "ItemForecaster", FakeForecaster)
    monkeypatch.setattr(fp, "SessionLocal", lambda: fake_db)
    monkeypatch.delenv("ALLOW_DRIFT_RETRAIN", raising=False)
    monkeypatch.delenv("FORCE_RETRAIN", raising=False)
    return FakeForecaster


def test_predict_only_does_not_train_though_drift_is_reported(monkeypatch, tmp_path):
    import scripts.forecast_prices as fp

    fake = _fake_forecast_env(monkeypatch, tmp_path, drifted=True)
    fp.run_forecast(predict_only=True)
    assert fake.train_called is False, (
        "Drift is reported on every real run. Retraining on it costs a measured "
        "465s of an 835s daily step."
    )


def test_predict_only_trains_when_the_opt_in_is_set(monkeypatch, tmp_path):
    import scripts.forecast_prices as fp

    fake = _fake_forecast_env(monkeypatch, tmp_path, drifted=True)
    monkeypatch.setenv("ALLOW_DRIFT_RETRAIN", "1")
    fp.run_forecast(predict_only=True)
    assert fake.train_called is True


def test_predict_only_opt_in_requires_exactly_one(monkeypatch, tmp_path):
    """A truthy-ish value must not enable it.

    Guards the gate against being loosened to `!= "0"` or `bool(...)`, which
    would restore the old behaviour by default.
    """
    import scripts.forecast_prices as fp

    fake = _fake_forecast_env(monkeypatch, tmp_path, drifted=True)
    monkeypatch.setenv("ALLOW_DRIFT_RETRAIN", "true")
    fp.run_forecast(predict_only=True)
    assert fake.train_called is False


def test_full_mode_does_not_train_on_drift_alone(monkeypatch, tmp_path):
    """A fresh model plus reported drift must not retrain."""
    import scripts.forecast_prices as fp

    fake = _fake_forecast_env(monkeypatch, tmp_path, drifted=True,
                              trained_days_ago=0)
    fp.run_forecast()
    assert fake.train_called is False


def test_full_mode_still_trains_a_stale_model(monkeypatch, tmp_path):
    import scripts.forecast_prices as fp

    fake = _fake_forecast_env(monkeypatch, tmp_path, drifted=False,
                              trained_days_ago=99)
    fp.run_forecast()
    assert fake.train_called is True
