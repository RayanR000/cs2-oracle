from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backtest.price_resolution as price_resolution
from backtest.price_resolution import smoothed_prices
from database import Base, ForecastOutcome, Item, ItemForecast, PredictionAccuracy


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)()
    engine.dispose()


def test_forecast_outcome_has_freeze_columns(session):
    row = ForecastOutcome(
        forecast_id=1,
        item_id=1,
        forecast_date=date(2026, 7, 1),
        horizon_days=3,
        target_date=date(2026, 7, 4),
        current_price=1.0,
        base_price=1.05,
        predicted_price_mid=1.1,
        actual_price=1.2,
        direction_correct=1,
        abs_error=0.1,
        resolved_at=datetime(2026, 7, 4, 9, 0, 0),
    )
    session.add(row)
    session.commit()

    stored = session.query(ForecastOutcome).one()
    assert stored.base_price == 1.05
    assert stored.resolved_at == datetime(2026, 7, 4, 9, 0, 0)


def test_prediction_accuracy_has_price_tier(session):
    row = PredictionAccuracy(
        prediction_type="forecast",
        evaluation_date=date(2026, 8, 1),
        horizon_days=3,
        model_version="lgbm-v3-regime",
        price_tier=1,
        sample_count=10,
        metrics={"directional_accuracy": 55.0},
        created_at=datetime(2026, 8, 1, 9, 0, 0),
    )
    session.add(row)
    session.commit()

    assert session.query(PredictionAccuracy).one().price_tier == 1


from backtest.scoring import (
    FLAT_TOLERANCE,
    direction_from_return,
    price_tier,
    score_cohort,
)


def _record(**overrides):
    base = {
        "abs_error": 0.10,
        "pct_error": 10.0,
        "sq_error": 0.01,
        "direction_correct": 1,
        "predicted_direction": "up",
        "actual_direction": "up",
        "in_interval": 1,
        "confidence": "high",
        "base_price": 1.00,
        "actual_price": 1.10,
        "price_tier": 1,
        "item_id": 1,
    }
    base.update(overrides)
    return base


def test_direction_from_return_respects_flat_tolerance():
    assert direction_from_return(FLAT_TOLERANCE * 2) == "up"
    assert direction_from_return(-FLAT_TOLERANCE * 2) == "down"
    assert direction_from_return(0.0) == "flat"
    assert direction_from_return(FLAT_TOLERANCE) == "flat"  # boundary is inclusive-flat


def test_price_tier_boundaries():
    assert price_tier(0.99) == 0
    assert price_tier(1.0) == 1
    assert price_tier(5.0) == 2
    assert price_tier(20.0) == 3
    assert price_tier(100.0) == 4


def test_score_cohort_is_pure_and_repeatable():
    records = [_record(item_id=i) for i in range(20)]
    first, n_first = score_cohort(records)
    second, n_second = score_cohort(records)
    assert first == second
    assert n_first == n_second == 20


def test_score_cohort_does_not_mutate_its_input():
    records = [_record(item_id=i) for i in range(20)]
    snapshot = [dict(r) for r in records]
    score_cohort(records)
    assert records == snapshot


def test_score_cohort_computes_directional_accuracy():
    records = [_record(direction_correct=1) for _ in range(6)]
    records += [_record(direction_correct=0, predicted_direction="down") for _ in range(4)]
    metrics, n = score_cohort(records)
    assert n == 10
    assert metrics["directional_accuracy"] == 60.0


def test_score_cohort_uses_base_price_for_the_persistence_baseline():
    """baseline_mae is |base - actual|, the error a predict-no-change model
    would make. It must read base_price, not the retired current_price."""
    records = [_record(base_price=1.00, actual_price=1.50) for _ in range(10)]
    metrics, _ = score_cohort(records)
    assert metrics["baseline_mae"] == 0.5


def test_both_legs_use_the_same_estimator_so_a_flat_market_scores_flat():
    """The end-to-end symmetry property. Under the old code the base leg was a
    3-observation median and the actual leg a single-day price, so a perfectly
    flat market could still produce non-flat direction labels."""
    import pandas as pd

    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(12)]
    voted = pd.DataFrame(
        {"item_id": ["ak"] * 12, "date": days, "price": [3.0] * 12}
    )

    forecast_date, target_date = date(2026, 7, 5), date(2026, 7, 8)
    prices = smoothed_prices(voted, {("ak", forecast_date), ("ak", target_date)})

    base = prices[("ak", forecast_date)]
    actual = prices[("ak", target_date)]
    assert direction_from_return((actual - base) / base) == "flat"


def test_resolution_drops_rather_than_falling_back_when_a_leg_is_unresolvable():
    """Substituting a fallback for a missing leg would reintroduce exactly the
    asymmetry this change removes."""
    import pandas as pd

    voted = pd.DataFrame(
        {
            "item_id": ["ak", "ak"],
            "date": [date(2026, 5, 1), date(2026, 7, 8)],
            "price": [3.0, 3.0],
        }
    )
    prices = smoothed_prices(voted, {("ak", date(2026, 7, 8))})
    # Only 2 observations, spanning 68 days — beyond the cap, so unresolvable.
    assert ("ak", date(2026, 7, 8)) not in prices


# ---------------------------------------------------------------------------
# End-to-end coverage of backtest_forecasts itself.
#
# The tests above exercise smoothed_prices in isolation. These drive the
# function Task 5 actually changed, through a real in-memory SQLite session and
# a real (tiny) Parquet archive under tmp_path. They never touch the git-tracked
# price-archive/ at the repo root, and db.parquet.append_table is stubbed so
# nothing is written to price-archive/ops/.
# ---------------------------------------------------------------------------

FORECAST_DATE = date(2026, 7, 5)
TARGET_DATE = date(2026, 7, 8)
HORIZON = 3
EVAL_DATE = date(2026, 7, 20)


def _write_archive(tmp_path, rows):
    """rows: list of (slug, date, price). Returns the archive directory."""
    archive = tmp_path / "price-archive"
    archive.mkdir()
    pd.DataFrame(
        {
            "item_slug": [r[0] for r in rows],
            "day": pd.to_datetime([r[1] for r in rows]),
            "mean_price": [float(r[2]) for r in rows],
            "volume": [10] * len(rows),
            "source": ["a"] * len(rows),
        }
    ).to_parquet(archive / "prices-2026.parquet")
    return archive


def _seed(session, pk, slug, *, current_price, price_mid, direction="flat",
          price_low=None, price_high=None):
    session.add(Item(id=pk, item_id=slug, name=slug, type="skin"))
    session.add(
        ItemForecast(
            id=pk,
            item_id=pk,
            forecast_date=FORECAST_DATE,
            horizon_days=HORIZON,
            price_low=price_low,
            price_mid=price_mid,
            price_high=price_high,
            current_price=current_price,
            direction=direction,
            confidence="high",
            model_version="lgbm-test",
        )
    )


def _run_backtest(session, archive, monkeypatch, today=EVAL_DATE):
    """Run backtest_forecasts with the archive redirected to tmp_path.

    backtest_forecasts hardcodes archive_dir to the repo-root price-archive/,
    so the loader is wrapped to substitute the test archive. The real
    load_voted_prices still runs — only the directory it reads is swapped.
    """
    from scripts import backtest_accuracy

    real_loader = price_resolution.load_voted_prices

    def loader(archive_dir, slugs, min_date, max_date, **kwargs):
        assert Path(archive_dir).name == "price-archive"
        return real_loader(archive, slugs, min_date, max_date, **kwargs)

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", loader)

    import db.parquet as parquet_mod
    monkeypatch.setattr(parquet_mod, "append_table", lambda *a, **k: None)

    return backtest_accuracy.backtest_forecasts(session, today=today)


def test_backtest_scores_the_base_leg_from_the_archive_not_current_price(
    session, tmp_path, monkeypatch
):
    """The base leg of actual_ret must be the archive-resolved smoothed price,
    not item_forecasts.current_price.

    current_price is stored at 99.0 while the archive sits at 3.0 rising to
    3.3. Resolved correctly the return is (3.3-3.0)/3.0 = +10% -> "up". If the
    base leg came from current_price instead it would be (3.3-99)/99 = -97%
    -> "down", the tier would be 3 rather than 1, and MAPE would be 0.3%
    rather than 10%. This is the pre-Task-5 behaviour.
    """
    archive = _write_archive(
        tmp_path,
        [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        + [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)],
    )
    _seed(
        session, 1, "ak",
        current_price=99.0,       # deliberately nothing like the archive
        price_mid=3.6, price_low=3.0, price_high=4.0,
        direction="up",
    )
    session.commit()

    results = _run_backtest(session, archive, monkeypatch)

    outcome = session.query(ForecastOutcome).one()
    assert outcome.base_price == pytest.approx(3.0)
    assert outcome.actual_price == pytest.approx(3.3)
    # Written straight through for reference, never synthesized from base.
    assert outcome.current_price == pytest.approx(99.0)
    assert outcome.direction_actual == "up"
    assert outcome.direction_correct == 1
    # |3.6 - 3.3| / 3.0 * 100 — divided by the base leg.
    assert outcome.pct_error == pytest.approx(10.0)

    metrics = results[0]["metrics"]
    assert results[0]["sample_count"] == 1
    assert metrics["mape"] == pytest.approx(10.0)
    # price_tier(3.0) == 1; price_tier(99.0) would be 3.
    assert metrics["mape_by_tier"] == {"tier_1": 10.0}


def test_missing_current_price_is_stored_as_null_not_synthesized_from_base(
    session, tmp_path, monkeypatch
):
    """A forecast with no serving-time current_price must store NULL, not the
    backtest-resolved base price. update_bias_corrections_from_outcomes reads
    this column to compute approx_mid_ret, which feeds production predict()
    thresholds — injecting base there would feed a backtest artefact into
    serving. A genuine NULL is distinguishable; a stand-in is not."""
    archive = _write_archive(
        tmp_path, [("ak", date(2026, 7, d), 3.0) for d in range(3, 9)]
    )
    _seed(session, 1, "ak", current_price=None, price_mid=3.0)
    session.commit()

    _run_backtest(session, archive, monkeypatch)

    outcome = session.query(ForecastOutcome).one()
    assert outcome.current_price is None
    assert outcome.base_price == pytest.approx(3.0)


def test_backtest_drops_a_forecast_whose_base_leg_is_unresolvable(
    session, tmp_path, monkeypatch
):
    """An unresolvable base leg must drop the forecast, not fall back to
    current_price. The dropped item carries current_price=50.0, so a fallback
    would silently produce an outcome row for it."""
    rows = []
    for i in range(10):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0)
    # "gap" has no observation at or before the 07-05 forecast date, so its
    # base leg is unresolvable even though its 07-08 target leg resolves.
    rows += [("gap", date(2026, 7, d), 3.0) for d in (6, 7, 8)]
    _seed(session, 11, "gap", current_price=50.0, price_mid=3.0)
    session.commit()

    archive = _write_archive(tmp_path, rows)

    # 1 unresolvable of 11 considered = 9.1%, under MAX_UNRESOLVABLE_PCT.
    results = _run_backtest(session, archive, monkeypatch)

    stored = session.query(ForecastOutcome).all()
    assert len(stored) == 10
    assert 11 not in {o.forecast_id for o in stored}
    assert results[0]["sample_count"] == 10


def test_unresolvable_forecasts_count_toward_the_gate_denominator(
    session, tmp_path, monkeypatch
):
    """Dropped forecasts must land in both the numerator and the denominator of
    the unresolvable gate. 1 of 9 is 11.1%, over the 10% cap, so the run must
    refuse to report a number rather than scoring the surviving 8."""
    from scripts import backtest_accuracy

    rows = []
    for i in range(8):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0)
    rows += [("gap", date(2026, 7, d), 3.0) for d in (6, 7, 8)]
    _seed(session, 9, "gap", current_price=50.0, price_mid=3.0)
    session.commit()

    archive = _write_archive(tmp_path, rows)

    with pytest.raises(RuntimeError, match="could not be resolved"):
        _run_backtest(session, archive, monkeypatch)

    # The gate fires before anything is persisted.
    assert session.query(ForecastOutcome).count() == 0
    assert backtest_accuracy.MAX_UNRESOLVABLE_PCT == 10.0


def test_resolution_is_insert_only(session, monkeypatch):
    import scripts.backtest_accuracy as bt
    import db.parquet as parquet_mod

    # Human ruling: append_table writes to git-tracked production data
    # (price-archive/ops/forecast_outcomes.parquet). Never let a test touch it.
    monkeypatch.setattr(parquet_mod, "append_table", lambda *a, **k: None)

    outcome = {
        "forecast_id": 7,
        "item_id": 1,
        "forecast_date": date(2026, 7, 1),
        "horizon_days": 3,
        "target_date": date(2026, 7, 4),
        "current_price": 1.0,
        "base_price": 1.0,
        "predicted_price_low": 0.9,
        "predicted_price_mid": 1.1,
        "predicted_price_high": 1.3,
        "actual_price": 1.2,
        "direction_predicted": "up",
        "direction_actual": "up",
        "direction_correct": 1,
        "in_interval": 1,
        "abs_error": 0.1,
        "pct_error": 10.0,
        "model_version": "lgbm-v3-regime",
    }
    assert bt._store_forecast_outcomes(session, [dict(outcome)]) == 1

    # The archive is revised: the same forecast now resolves to a different
    # actual. Freezing means the stored row does not move.
    revised = dict(outcome, actual_price=99.0, direction_actual="down", direction_correct=0)
    assert bt._store_forecast_outcomes(session, [revised]) == 0

    stored = session.query(ForecastOutcome).filter_by(forecast_id=7).one()
    assert stored.actual_price == 1.2
    assert stored.direction_correct == 1
    assert stored.resolved_at is not None


def test_reresolve_overrides_the_freeze(session, monkeypatch):
    import scripts.backtest_accuracy as bt
    import db.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod, "append_table", lambda *a, **k: None)

    outcome = {
        "forecast_id": 8,
        "item_id": 1,
        "forecast_date": date(2026, 7, 1),
        "horizon_days": 3,
        "target_date": date(2026, 7, 4),
        "current_price": 1.0,
        "base_price": 1.0,
        "predicted_price_mid": 1.1,
        "actual_price": 1.2,
        "direction_predicted": "up",
        "direction_actual": "up",
        "direction_correct": 1,
        "in_interval": 1,
        "abs_error": 0.1,
        "pct_error": 10.0,
        "model_version": "lgbm-v3-regime",
    }
    bt._store_forecast_outcomes(session, [dict(outcome)])
    revised = dict(outcome, actual_price=99.0)
    assert bt._store_forecast_outcomes(session, [revised], reresolve=True) == 1

    assert session.query(ForecastOutcome).filter_by(forecast_id=8).one().actual_price == 99.0


def test_forecasts_with_no_slug_mapping_count_toward_the_gate(
    session, tmp_path, monkeypatch
):
    """A whole group with no slug mappings yields no anchors and `continue`s
    before the per-forecast loop. Those forecasts are still mature and still
    unscored, so they must be counted — otherwise the gate divides by zero
    considered and reports green over an empty cohort."""
    session.add(
        ItemForecast(
            id=1,
            item_id=4242,  # no matching row in items -> no slug
            forecast_date=FORECAST_DATE,
            horizon_days=HORIZON,
            price_mid=3.0,
            current_price=3.0,
            direction="flat",
            confidence="high",
            model_version="lgbm-test",
        )
    )
    session.commit()

    archive = _write_archive(tmp_path, [("ak", date(2026, 7, 5), 3.0)])

    with pytest.raises(RuntimeError, match="could not be resolved"):
        _run_backtest(session, archive, monkeypatch)
