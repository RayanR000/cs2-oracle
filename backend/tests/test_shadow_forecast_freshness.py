"""Shadow collection readiness: three consecutive complete dates."""

from datetime import date, timedelta

from scripts.check_forecast_freshness import shadow_collection_readiness

HORIZONS = (3, 7, 14, 30)


def _batches(dates, drop=()):
    return {(d, h, "centre") for d in dates for h in HORIZONS} - set(drop)


def test_ready_when_three_consecutive_complete():
    dates = [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18)]
    result = shadow_collection_readiness(dates, _batches(dates))
    assert result["shadow_collection_ready"] is True
    assert result["gap"] is None


def test_missing_batch_resets_and_names_the_gap():
    dates = [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18)]
    missing = (date(2026, 9, 17), 14, "centre")
    result = shadow_collection_readiness(dates, _batches(dates, drop=[missing]))
    assert result["shadow_collection_ready"] is False
    assert "2026-09-17" in result["gap"] and "14" in result["gap"]


def test_short_history_is_not_ready():
    dates = [date(2026, 9, 17), date(2026, 9, 18)]
    result = shadow_collection_readiness(dates, _batches(dates))
    assert result["shadow_collection_ready"] is False


def test_older_history_does_not_excuse_a_recent_gap():
    dates = [date(2026, 9, d) for d in range(1, 19)]
    result = shadow_collection_readiness(dates, _batches(dates, drop=[(date(2026, 9, 18), 3, "centre")]))
    assert result["shadow_collection_ready"] is False
    assert "2026-09-18" in result["gap"]
