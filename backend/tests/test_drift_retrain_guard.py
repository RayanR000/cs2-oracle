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
