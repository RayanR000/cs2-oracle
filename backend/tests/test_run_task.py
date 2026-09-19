"""Forecast count guards: production zero rows stay fatal, shadow gaps warn."""

import logging

import pytest
from scripts.run_task import ROW_COUNT_FIELDS, check_results


def _forecast_result(**overrides):
    base = {
        "status": "success",
        "forecasts_written": 100,
        "centre_candidates_written": 50,
        "ranking_candidates_written": 50,
        "candidate_batches_expected": 8,
        "candidate_batches_written": 8,
    }
    base.update(overrides)
    return base


def test_forecasts_written_is_guarded():
    assert "forecasts_written" in ROW_COUNT_FIELDS


def test_zero_production_fails():
    with pytest.raises(SystemExit):
        check_results(
            "forecast",
            [_forecast_result(forecasts_written=0, centre_candidates_written=0, ranking_candidates_written=0,
                              candidate_batches_expected=0, candidate_batches_written=0)],
        )


def test_production_without_shadow_warns_but_passes(caplog):
    with caplog.at_level(logging.WARNING):
        check_results(
            "forecast",
            [_forecast_result(centre_candidates_written=0, ranking_candidates_written=0,
                              candidate_batches_written=0)],
        )
    assert any("shadow" in r.message.lower() for r in caplog.records)


def test_complete_shadow_is_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        check_results("forecast", [_forecast_result()])
    assert not [r for r in caplog.records if "shadow" in r.message.lower()]
