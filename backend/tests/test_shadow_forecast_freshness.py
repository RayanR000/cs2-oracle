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


def test_shadow_inputs_come_from_the_partitions():
    import pandas as pd
    import scripts.check_forecast_freshness as mod
    from db import ops_partitions

    days = [date(2026, 10, 20) + timedelta(days=i) for i in range(3)]
    ops_partitions.upsert(
        "item_forecasts",
        pd.DataFrame([{"id": i, "item_id": 1, "forecast_date": d, "horizon_days": 3} for i, d in enumerate(days)]),
        allow_closed=True,
    )
    ops_partitions.upsert(
        "forecast_candidates",
        pd.DataFrame(
            [
                {
                    "id": k,
                    "item_id": 1,
                    "forecast_date": d,
                    "horizon_days": h,
                    "component": "centre",
                    "candidate_name": "c",
                    "candidate_version": "v",
                }
                for k, (d, h) in enumerate((d, h) for d in days for h in (3, 7, 14, 30))
            ]
        ),
        allow_closed=True,
    )
    dates, batches = mod._shadow_inputs()
    assert dates == sorted(days, reverse=True)
    assert mod.shadow_collection_readiness(dates, batches)["shadow_collection_ready"] is True
