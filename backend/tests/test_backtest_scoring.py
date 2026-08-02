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
          price_low=None, price_high=None, horizon=HORIZON):
    session.add(Item(id=pk, item_id=slug, name=slug, type="skin"))
    session.add(
        ItemForecast(
            id=pk,
            item_id=pk,
            forecast_date=FORECAST_DATE,
            horizon_days=horizon,
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


def test_backtest_drops_a_target_beyond_archive_coverage_rather_than_resolving_it_backwards(
    session, tmp_path, monkeypatch
):
    """Task 8d, test 4 of the brief. A forecast whose TARGET date lies past the
    end of the archive must be dropped, never scored.

    "future" has observations only through 07-05 — the forecast date itself.
    Both anchors then select the SAME 07-03/04/05 window: the base leg because
    07-05 is its own date, the actual leg because 07-08 is only 5 days past the
    oldest of them and so passes the anchor-staleness rule on its own terms.
    Two identical windows give actual_ret == 0.0 exactly, and the forecast
    scores "flat" — a manufactured flat, not a measurement, produced for every
    forecast whose target is beyond coverage. It is asserted below that the
    estimator really does return identical legs here, so this test fails for the
    right reason if the drop is removed.

    A second beyond-coverage shape is driven alongside it, because the two
    halves of this change guard different cases and a test that only covers one
    would let the other regress. "stale10" has a 10-day horizon (target 07-15)
    and observations through 07-06: its actual leg IS supported by an
    observation after the forecast date, so the leg-pair drop does not fire —
    but 07-04 is 11 days before the 07-15 anchor, so the anchor-staleness rule
    does. Under the old between-observations rule its window spans 2 days and it
    resolves to a price nine days stale.

    Twenty fully-covered forecasts keep 2/22 = 9.1% under MAX_UNRESOLVABLE_PCT,
    so the drops are observable in the outcome rows rather than masked by the
    gate.
    """
    rows = []
    for i in range(20):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0)
    rows += [("future", date(2026, 7, d), 7.0) for d in (3, 4, 5)]
    _seed(session, 21, "future", current_price=7.0, price_mid=7.0, direction="flat")
    rows += [("stale10", date(2026, 7, d), 7.0) for d in (3, 4, 5, 6)]
    _seed(session, 22, "stale10", current_price=7.0, price_mid=7.0,
          direction="flat", horizon=10)
    session.commit()

    archive = _write_archive(tmp_path, rows)

    # The premise: leg-agnostic resolution resolves BOTH anchors, identically.
    voted = pd.DataFrame(
        {
            "item_id": ["future"] * 3,
            "date": [date(2026, 7, d) for d in (3, 4, 5)],
            "price": [7.0] * 3,
        }
    )
    both = price_resolution.resolve_anchors(
        voted, {("future", FORECAST_DATE), ("future", TARGET_DATE)}
    )
    assert both[("future", FORECAST_DATE)].price == both[("future", TARGET_DATE)].price
    # ...off the same window, whose newest observation is not after the forecast
    # date. That is what makes the equality an artefact rather than a flat market.
    assert both[("future", TARGET_DATE)].newest_observation == FORECAST_DATE

    results = _run_backtest(session, archive, monkeypatch)

    stored = session.query(ForecastOutcome).all()
    assert len(stored) == 20
    assert {21, 22}.isdisjoint({o.forecast_id for o in stored})
    # No outcome anywhere carries the 7.0 price the stale carry-forward would
    # have produced, on either leg, and none was scored as a manufactured flat.
    assert all(o.base_price == pytest.approx(3.0) for o in stored)
    assert all(o.actual_price == pytest.approx(3.0) for o in stored)
    assert sum(
        r["sample_count"] for r in results
        if r["price_tier"] is None
    ) == 20


def test_an_actual_leg_supported_only_by_pre_forecast_observations_is_unresolvable(
    session, tmp_path, monkeypatch
):
    """The drop above is a resolution failure, so it must reach the gate rather
    than quietly shrinking the cohort — the exact invisibility this plan exists
    to remove. Two beyond-coverage forecasts against seven covered ones is
    2/9 = 22.2%, over the 10% cap, so the run must refuse to report a number.

    The complement is asserted too: a single observation strictly AFTER the
    forecast date is enough to make the actual leg a real measurement."""
    rows = []
    for i in range(7):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0)
    for i in range(2):
        rows += [(f"end{i}", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        _seed(session, 100 + i, f"end{i}", current_price=3.0, price_mid=3.0)
    session.commit()

    archive = _write_archive(tmp_path, rows)

    with pytest.raises(RuntimeError, match="could not be resolved"):
        _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 0

    # One observation past the forecast date is enough: 07-06 supports the
    # 07-08 actual leg, so both forecasts now resolve and the gate clears.
    for i in range(2):
        rows += [(f"end{i}", date(2026, 7, 6), 3.6)]
    _revise_archive(archive, rows)

    results = _run_backtest(session, archive, monkeypatch)
    assert next(r for r in results if r["price_tier"] is None)["sample_count"] == 9


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


from backtest.scoring import HEADLINE_MIN_TIER, score_by_tier


def test_tier_rows_partition_the_all_row():
    records = [_record(price_tier=0, item_id=i) for i in range(30)]
    records += [_record(price_tier=1, item_id=100 + i) for i in range(20)]

    scored = score_by_tier(records)
    per_tier = {tier: n for tier, _, n in scored if tier is not None}
    all_rows = [(m, n) for tier, m, n in scored if tier is None]

    assert per_tier == {0: 30, 1: 20}
    assert len(all_rows) == 1
    assert all_rows[0][1] == 50
    assert sum(per_tier.values()) == all_rows[0][1]


def test_empty_tiers_are_omitted_not_zero_filled():
    records = [_record(price_tier=4, item_id=i) for i in range(12)]
    tiers = {tier for tier, _, _ in score_by_tier(records) if tier is not None}
    assert tiers == {4}


def test_headline_tier_is_one_dollar_and_up():
    assert HEADLINE_MIN_TIER == 1
    assert price_tier(0.99) < HEADLINE_MIN_TIER
    assert price_tier(1.00) >= HEADLINE_MIN_TIER


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


def test_rescore_path_emits_the_same_tier_rows_as_the_normal_path(
    session, tmp_path, monkeypatch
):
    """Task 6 introduced --rescore ahead of score_by_tier existing, so it
    scored one blended cohort per (horizon, model_version) as a placeholder.
    Now that score_by_tier exists, both paths must emit the same set of
    (horizon_days, model_version, price_tier) rows, with matching sample
    counts, for the same underlying data — otherwise --rescore silently
    reports a different shape than a normal run."""
    from scripts import backtest_accuracy

    archive = _write_archive(
        tmp_path,
        [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        + [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
        + [("awp", date(2026, 7, d), 10.0) for d in (3, 4, 5)]
        + [("awp", date(2026, 7, d), 11.0) for d in (6, 7, 8)],
    )
    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6,
          price_low=3.0, price_high=4.0, direction="up")
    _seed(session, 2, "awp", current_price=10.0, price_mid=11.5,
          price_low=10.0, price_high=13.0, direction="up")
    session.commit()

    normal_results = _run_backtest(session, archive, monkeypatch)

    import db.parquet as parquet_mod
    monkeypatch.setattr(parquet_mod, "append_table", lambda *a, **k: None)
    rescore_results = backtest_accuracy.backtest_forecasts(
        session, today=EVAL_DATE, rescore=True
    )

    def shape(results):
        return {
            (r["horizon_days"], r["model_version"], r["price_tier"]): r["sample_count"]
            for r in results
        }

    normal_shape = shape(normal_results)
    rescore_shape = shape(rescore_results)

    # Both a per-tier row (tier 1 for "ak", tier 2 for "awp") and the
    # all-tiers aggregate (price_tier=None) must be present in both paths.
    assert {None, 1, 2} == {t for (_, _, t) in normal_shape}
    assert normal_shape == rescore_shape


# ---------------------------------------------------------------------------
# Task 8: the determinism regression test.
#
# The bug: the base leg of actual_ret was item_forecasts.current_price (a
# 3-observation median written once at serving time) and the actual leg was a
# raw single-day voted price re-read from the archive on every run. Two
# different estimators, differenced against a 0.5% flat band. When the
# archive gained or revised source rows for an already-scored target date,
# every item's return shifted together and direction labels flipped in bulk —
# the same 5,512-forecast cohort scored 61.76%, 33.74%, 61.54%, 57.91% across
# four dates with no change to the model or the forecasts themselves.
#
# Layer (a) below exercises the shared estimator directly: an already-
# resolved anchor's smoothed price must not move when the archive gains an
# outlier source row for that date. Layer (b) drives the full path —
# backtest_forecasts, twice, across an archive revision — which is the
# actual claim this plan makes ("an archive revision must not move the
# reported metric") and exercises both the estimator and the Task 6 freeze.
# ---------------------------------------------------------------------------


def test_estimator_price_is_unchanged_when_archive_gains_an_outlier_source_row(tmp_path):
    """Layer (a). The voted+smoothed price for an already-resolved anchor
    (07-08) must not move when a new outlier source row appears for that
    date. Multi-source voting rejects the outlier and the window median
    absorbs anything that gets through — this is what makes re-resolving a
    growing mature cohort every day safe in the first place.
    """
    archive = tmp_path / "price-archive"
    archive.mkdir()
    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(12)]

    def write(extra_rows):
        frame = pd.DataFrame(
            {
                "item_slug": ["ak"] * 12,
                "day": pd.to_datetime(days),
                "mean_price": [3.0] * 12,
                "volume": [5] * 12,
                "source": ["a"] * 12,
            }
        )
        if extra_rows is not None:
            frame = pd.concat([frame, extra_rows], ignore_index=True)
        frame.to_parquet(archive / "prices-2026.parquet")

    anchors = {("ak", date(2026, 7, 5)), ("ak", date(2026, 7, 8))}

    write(None)
    before = smoothed_prices(
        price_resolution.load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 12)),
        anchors,
    )

    # A second source appears for an already-resolved day, well off consensus.
    write(
        pd.DataFrame(
            {
                "item_slug": ["ak"],
                "day": pd.to_datetime([date(2026, 7, 8)]),
                "mean_price": [75.0],
                "volume": [5],
                "source": ["b"],
            }
        )
    )
    after = smoothed_prices(
        price_resolution.load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 12)),
        anchors,
    )

    assert direction_from_return(
        (before[("ak", date(2026, 7, 8))] - before[("ak", date(2026, 7, 5))])
        / before[("ak", date(2026, 7, 5))]
    ) == "flat"

    # Voting rejects the outlier source; even unfrozen, the estimator holds.
    assert after[("ak", date(2026, 7, 8))] == before[("ak", date(2026, 7, 8))]


def test_backtest_forecasts_reports_identical_metrics_across_an_archive_revision(
    session, tmp_path, monkeypatch
):
    """Layer (b), the centrepiece: backtest_forecasts run twice across an
    archive revision, same forecast cohort both times, must report identical
    metrics and must not duplicate the frozen per-forecast outcome rows.

    backtest_forecasts re-derives its scoring records fresh from the archive
    on every call (the mature cohort is re-scored daily, growing over time) —
    it does not read them back from the frozen ForecastOutcome rows. So the
    metrics-stability half of this assertion is carried by the shared
    estimator, exercised through the real pipeline rather than in isolation.
    The row-count half is carried by the Task 6 freeze: a freeze that
    silently duplicated rather than skipped rows would leave the *values*
    unchanged but the table would grow every run, which the plain
    before/after dict comparison used in Task 6's own test cannot see.
    """
    rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    rows += [("awp", date(2026, 7, d), 10.0) for d in (3, 4, 5, 6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(
        session, 1, "ak", current_price=3.0, price_mid=3.6,
        price_low=3.0, price_high=4.0, direction="up",
    )
    _seed(
        session, 2, "awp", current_price=10.0, price_mid=10.1,
        price_low=9.0, price_high=11.0, direction="flat",
    )
    session.commit()

    results_before = _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 2

    # The archive is revised: an outlier source row appears for the
    # already-resolved 07-08 target date. Voting rejects it, so the resolved
    # "ak" price at 07-08 is unchanged — but this is driven through the real
    # backtest_accuracy.load_voted_prices / smoothed_prices call path, not a
    # direct call, so it also proves the wiring, not just the estimator.
    base_frame = pd.DataFrame(
        {
            "item_slug": [r[0] for r in rows],
            "day": pd.to_datetime([r[1] for r in rows]),
            "mean_price": [float(r[2]) for r in rows],
            "volume": [10] * len(rows),
            "source": ["a"] * len(rows),
        }
    )
    extra = pd.DataFrame(
        {
            "item_slug": ["ak"],
            "day": pd.to_datetime([date(2026, 7, 8)]),
            "mean_price": [75.0],
            "volume": [5],
            "source": ["b"],
        }
    )
    pd.concat([base_frame, extra], ignore_index=True).to_parquet(
        archive / "prices-2026.parquet"
    )

    results_after = _run_backtest(session, archive, monkeypatch)

    def shape(results):
        return {
            (r["horizon_days"], r["model_version"], r["price_tier"]): (
                r["sample_count"],
                r["metrics"],
            )
            for r in results
        }

    assert shape(results_before) == shape(results_after)

    # Not just unchanged values — exactly as many rows as forecasts, both
    # before and after the revision.
    assert session.query(ForecastOutcome).count() == 2
    outcomes = {
        o.forecast_id: (o.base_price, o.actual_price)
        for o in session.query(ForecastOutcome).all()
    }
    assert outcomes[1] == (pytest.approx(3.0), pytest.approx(3.3))
    assert outcomes[2] == (pytest.approx(10.0), pytest.approx(10.0))


# ---------------------------------------------------------------------------
# Task 8b: the normal (non---rescore) path must score the FROZEN outcomes, not
# a fresh re-resolution of the whole mature cohort.
#
# Task 8's mutation check exposed the gap: the freeze gated what was written to
# forecast_outcomes but never what was scored, so the daily headline number's
# stability rested entirely on the estimator's window-median robustness. A
# large enough archive revision could still move an already-reported number.
# ---------------------------------------------------------------------------


def _revise_archive(archive, rows):
    """Overwrite the whole test archive with `rows` — a genuine revision, not
    an added outlier source that voting would reject."""
    pd.DataFrame(
        {
            "item_slug": [r[0] for r in rows],
            "day": pd.to_datetime([r[1] for r in rows]),
            "mean_price": [float(r[2]) for r in rows],
            "volume": [10] * len(rows),
            "source": ["a"] * len(rows),
        }
    ).to_parquet(archive / "prices-2026.parquet")


def test_frozen_values_drive_the_metric_not_a_revised_archive(
    session, tmp_path, monkeypatch
):
    """Run 1 freezes "ak" at base 3.0 -> actual 3.3 (MAPE 10%). The archive is
    then rewritten so "ak" would now resolve to 30.0 at the target date — a
    2-tier move that no amount of window-median robustness absorbs. A second,
    unfrozen forecast ("awp", tier 2) is added so the archive genuinely IS
    read on run 2; this is not passing merely because nothing was resolved.

    The tier-1 row must still report the frozen 10%, not 880%."""
    ak_rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    ak_rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, ak_rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6,
          price_low=3.0, price_high=4.0, direction="up")
    session.commit()

    before = _run_backtest(session, archive, monkeypatch)
    tier1_before = next(r for r in before if r["price_tier"] == 1)
    assert tier1_before["metrics"]["mape"] == pytest.approx(10.0)

    # The archive is revised outright: "ak" now sits at 30.0 from 07-06 on.
    revised = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    revised += [("ak", date(2026, 7, d), 30.0) for d in (6, 7, 8)]
    revised += [("awp", date(2026, 7, d), 10.0) for d in (3, 4, 5, 6, 7, 8)]
    _revise_archive(archive, revised)

    _seed(session, 2, "awp", current_price=10.0, price_mid=10.1,
          price_low=9.0, price_high=11.0, direction="flat")
    session.commit()

    after = _run_backtest(session, archive, monkeypatch)

    tier1_after = next(r for r in after if r["price_tier"] == 1)
    # Re-resolving "ak" would give |3.6-30|/3*100 = 880% and actual "up".
    assert tier1_after["sample_count"] == 1
    assert tier1_after["metrics"] == tier1_before["metrics"]

    # The frozen row itself is untouched.
    frozen = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert frozen.actual_price == pytest.approx(3.3)

    # ...and the new forecast was resolved and scored on its own tier.
    tier2_after = next(r for r in after if r["price_tier"] == 2)
    assert tier2_after["sample_count"] == 1
    assert next(r for r in after if r["price_tier"] is None)["sample_count"] == 2
    assert session.query(ForecastOutcome).count() == 2


def test_archive_is_not_read_when_every_mature_forecast_is_frozen(
    session, tmp_path, monkeypatch
):
    """Nothing new to resolve => no archive access at all, and no divide-by-zero
    in the unresolvable gate over an empty denominator."""
    from scripts import backtest_accuracy

    rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6,
          price_low=3.0, price_high=4.0, direction="up")
    session.commit()

    before = _run_backtest(session, archive, monkeypatch)

    def explode(*a, **k):
        raise AssertionError("load_voted_prices was called with nothing to resolve")

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", explode)
    import db.parquet as parquet_mod
    monkeypatch.setattr(parquet_mod, "append_table", lambda *a, **k: None)

    after = backtest_accuracy.backtest_forecasts(session, today=EVAL_DATE)

    def shape(results):
        return {
            (r["horizon_days"], r["model_version"], r["price_tier"]): (
                r["sample_count"], r["metrics"],
            )
            for r in results
        }

    assert shape(before) == shape(after)
    assert session.query(ForecastOutcome).count() == 1


def test_a_scoring_fix_lands_on_frozen_rows_without_touching_the_archive(
    session, tmp_path, monkeypatch
):
    """The reason records are re-derived rather than read back column-for-column
    from forecast_outcomes: changing the flat tolerance must change the reported
    directional accuracy of already-frozen rows, with no archive read.

    "ak" moves 3.00 -> 3.01 (+0.33%), inside the 0.5% flat band, so a "flat"
    prediction scores correct. Widen nothing and shrink FLAT_TOLERANCE to 0.1%
    and the same frozen row is now "up" — and must score wrong."""
    from scripts import backtest_accuracy
    import backtest.scoring as scoring

    rows = [("ak", date(2026, 7, d), 3.00) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.01) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.0,
          price_low=2.0, price_high=4.0, direction="flat")
    session.commit()

    before = _run_backtest(session, archive, monkeypatch)
    all_before = next(r for r in before if r["price_tier"] is None)
    assert all_before["metrics"]["directional_accuracy"] == 100.0

    def explode(*a, **k):
        raise AssertionError("a scoring fix must not need the archive")

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", explode)
    import db.parquet as parquet_mod
    monkeypatch.setattr(parquet_mod, "append_table", lambda *a, **k: None)
    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)

    after = backtest_accuracy.backtest_forecasts(session, today=EVAL_DATE)
    all_after = next(r for r in after if r["price_tier"] is None)
    assert all_after["metrics"]["directional_accuracy"] == 0.0

    # The frozen ACTUALS are untouched — only the derivation moved...
    frozen = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert frozen.base_price == pytest.approx(3.00)
    assert frozen.actual_price == pytest.approx(3.01)
    # ...and as of Task 8c the stored VERDICT columns follow the derivation
    # rather than staying at their write-time values. See
    # test_a_scoring_change_refreshes_the_stored_verdict_columns below.
    assert frozen.direction_actual == "up"
    assert frozen.direction_correct == 0


def test_gate_denominator_counts_only_forecasts_requiring_resolution(
    session, tmp_path, monkeypatch
):
    """20 frozen forecasts plus 10 new ones of which 2 are unresolvable is a 20%
    resolution failure rate and must trip the gate. Counting the frozen majority
    in the denominator would read 2/30 = 6.7% and wave it through — a frozen
    forecast was not resolved this run and belongs on neither side of the ratio.
    """
    rows = []
    for i in range(20):
        rows += [(f"old{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, i + 1, f"old{i}", current_price=3.0, price_mid=3.0)
    session.commit()
    archive = _write_archive(tmp_path, rows)

    _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 20

    # 8 resolvable newcomers and 2 with no observation at or before 07-05.
    for i in range(8):
        rows += [(f"new{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, 100 + i, f"new{i}", current_price=3.0, price_mid=3.0)
    for i in range(2):
        rows += [(f"gap{i}", date(2026, 7, d), 3.0) for d in (6, 7, 8)]
        _seed(session, 200 + i, f"gap{i}", current_price=3.0, price_mid=3.0)
    session.commit()
    _revise_archive(archive, rows)

    with pytest.raises(RuntimeError, match="could not be resolved"):
        _run_backtest(session, archive, monkeypatch)

    # The gate fired before anything new was persisted.
    assert session.query(ForecastOutcome).count() == 20


def test_unusable_frozen_rows_are_counted_and_logged_not_swallowed(
    session, tmp_path, monkeypatch, caplog
):
    """A frozen row with no usable base_price is dropped from scoring — and
    because it is frozen it is never re-resolved and never reaches the
    unresolvable gate either. That combination is invisible cohort shrinkage,
    the exact failure mode this plan exists to eliminate, so the drop must be
    counted and logged loudly rather than swallowed.

    Deliberately a log and not a raise: legacy rows predate the freeze and must
    not block Task 9's backfill."""
    import logging

    from scripts import backtest_accuracy

    rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    rows += [("awp", date(2026, 7, d), 10.0) for d in (3, 4, 5, 6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6,
          price_low=3.0, price_high=4.0, direction="up")
    _seed(session, 2, "awp", current_price=10.0, price_mid=10.1,
          price_low=9.0, price_high=11.0, direction="flat")
    session.commit()

    _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 2

    # A legacy-shaped row: resolved once, but with no usable base leg.
    stale = session.query(ForecastOutcome).filter_by(forecast_id=2).one()
    stale.base_price = None
    session.commit()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backtest_accuracy"):
        after = _run_backtest(session, archive, monkeypatch)

    # Dropped from the metric...
    assert next(r for r in after if r["price_tier"] is None)["sample_count"] == 1
    # ...but not from the run's output.
    assert "1 scored of 2 considered" in caplog.text
    unusable = [
        r for r in caplog.records
        if "UNUSABLE" in r.message and r.levelno >= logging.WARNING
    ]
    assert len(unusable) == 1
    assert "1 frozen outcome(s) UNUSABLE" in unusable[0].message

    # Visibility, not enforcement — the run still reports a number.
    assert backtest_accuracy.MAX_UNRESOLVABLE_PCT == 10.0


def test_frozen_outcome_query_is_restricted_in_sql_not_in_python(
    session, tmp_path, monkeypatch
):
    """The forecast_ids restriction must be a chunked SQL IN list, not a full
    table scan filtered afterwards — forecast_outcomes is the largest table in
    the daily path."""
    from scripts import backtest_accuracy

    rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)
    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6,
          price_low=3.0, price_high=4.0, direction="up")
    session.commit()
    _run_backtest(session, archive, monkeypatch)

    groups = backtest_accuracy._records_from_frozen_outcomes(session, forecast_ids=[])
    assert groups == {}

    groups = backtest_accuracy._records_from_frozen_outcomes(session, forecast_ids=[1])
    assert sum(len(v) for v in groups.values()) == 1

    # 2,000 ids is past the SQLite 999-parameter cap; the chunking must hold.
    groups = backtest_accuracy._records_from_frozen_outcomes(
        session, forecast_ids=list(range(1, 2001))
    )
    assert sum(len(v) for v in groups.values()) == 1


def test_headline_log_line_handles_fewer_than_ten_samples_without_raising(
    session, tmp_path, monkeypatch
):
    """bootstrap_ci returns (None, None) under 10 values. The >=$1 headline
    log line in backtest_forecasts guards ci_lower is not None before
    multiplying by 100 — this drives that branch explicitly with a 3-forecast
    cohort rather than relying on it firing incidentally in other tests."""
    rows = []
    for i in range(3):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        rows += [(f"ak{i}", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
        _seed(
            session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.6,
            price_low=3.0, price_high=4.0, direction="up",
        )
    session.commit()

    archive = _write_archive(tmp_path, rows)

    results = _run_backtest(session, archive, monkeypatch)

    all_row = next(r for r in results if r["price_tier"] is None)
    assert all_row["sample_count"] == 3
    assert all_row["metrics"]["directional_accuracy_ci_lower"] is None
    assert all_row["metrics"]["directional_accuracy_ci_upper"] is None


# ---------------------------------------------------------------------------
# Task 8c: the stored VERDICT columns are refreshed to match current scoring.
#
# ForecastOutcome holds two kinds of column. The ACTUALS (base_price,
# actual_price) are frozen observations. The VERDICTS (direction_actual,
# direction_correct, in_interval, abs_error, pct_error) are metrics derived
# from them. Task 8b made the reported metric re-derive the verdicts every run
# but left the stored copies write-once, so the two consumers that read them as
# authoritative — tiered_breakdown.py and, more seriously,
# update_bias_corrections_from_outcomes, which fits PRODUCTION predict()
# thresholds — would silently diverge from the headline the moment scoring
# changed. The verdicts now follow the derivation; the actuals do not move.
# ---------------------------------------------------------------------------


def _capture_append(monkeypatch):
    """Stub db.parquet.append_table and record its calls.

    Human ruling: append_table writes to git-tracked production data under
    price-archive/ops/. No test may reach the real one.
    """
    import db.parquet as parquet_mod

    calls = []
    monkeypatch.setattr(
        parquet_mod,
        "append_table",
        lambda table, rows, dedup_keys: calls.append((table, rows, dedup_keys)),
    )
    return calls


def _freeze_one_flat_forecast(session, tmp_path, monkeypatch):
    """Run 1: "ak" moves 3.00 -> 3.01 (+0.33%), inside the 0.5% flat band, so
    a "flat" prediction is frozen as correct. Returns the archive dir."""
    rows = [("ak", date(2026, 7, d), 3.00) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.01) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.0,
          price_low=2.0, price_high=4.0, direction="flat")
    session.commit()

    _run_backtest(session, archive, monkeypatch)

    frozen = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert frozen.direction_actual == "flat"
    assert frozen.direction_correct == 1
    return archive


def _actuals_snapshot(session):
    """The three columns the refresh must never write, plus their exact
    values, for every stored outcome."""
    return {
        o.forecast_id: (o.base_price, o.actual_price, o.resolved_at)
        for o in session.query(ForecastOutcome).all()
    }


def test_a_scoring_change_refreshes_the_stored_verdict_columns(
    session, tmp_path, monkeypatch
):
    """Test 1 of the brief. Shrink FLAT_TOLERANCE so the frozen row reclassifies
    from "flat" to "up", run the normal path, and the STORED verdict columns
    must now match the new derivation — while base_price, actual_price and
    resolved_at are identical to before, compared value-for-value."""
    from scripts import backtest_accuracy
    import backtest.scoring as scoring

    archive = _freeze_one_flat_forecast(session, tmp_path, monkeypatch)

    before_row = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    before_verdict = (
        before_row.direction_actual, before_row.direction_correct,
        before_row.abs_error, before_row.pct_error,
    )
    actuals_before = _actuals_snapshot(session)

    def explode(*a, **k):
        raise AssertionError("refreshing a verdict must not need the archive")

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", explode)
    _capture_append(monkeypatch)
    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)

    backtest_accuracy.backtest_forecasts(session, today=EVAL_DATE)

    session.expire_all()
    after = session.query(ForecastOutcome).filter_by(forecast_id=1).one()

    # +0.33% is now outside the 0.1% band, so the verdict flips.
    assert after.direction_actual == "up"
    assert after.direction_correct == 0
    assert (after.direction_actual, after.direction_correct) != before_verdict[:2]

    # The frozen actuals did not move. Asserted by comparing values, not by
    # reading the UPDATE statement.
    assert _actuals_snapshot(session) == actuals_before

    # ...and the derived-from-actuals magnitudes are unchanged too, because the
    # actuals they derive from are unchanged.
    assert after.abs_error == pytest.approx(before_verdict[2])
    assert after.pct_error == pytest.approx(before_verdict[3])

    # The archive was never consulted (explode above would have fired).
    assert archive.exists()


def test_the_verdict_refresh_is_idempotent(session, tmp_path, monkeypatch):
    """Test 2 of the brief. With no scoring change, a second pass refreshes
    zero rows and writes nothing — including nothing to the Parquet mirror."""
    from scripts import backtest_accuracy
    import backtest.scoring as scoring

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)
    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)

    calls = _capture_append(monkeypatch)

    # First pass under the new scoring: one row reclassifies.
    assert backtest_accuracy._refresh_verdict_columns(session) == 1
    assert [c[0] for c in calls] == ["forecast_outcomes"]

    # Second pass, same scoring: nothing to do, and no write at all.
    calls.clear()
    assert backtest_accuracy._refresh_verdict_columns(session) == 0
    assert calls == []

    # A third pass restricted to the same ids agrees.
    assert backtest_accuracy._refresh_verdict_columns(session, forecast_ids=[1]) == 0


def test_the_refresh_cannot_move_the_frozen_actuals(session, tmp_path, monkeypatch):
    """Test 3 of the brief. Corrupt every verdict column on a frozen row and
    refresh: all five must be rewritten from the actuals, while base_price,
    actual_price and resolved_at come back byte-identical.

    Asserted on values before and after, not by inspecting the SQL — a
    whole-row write that happened to carry the same actuals would pass a code
    reading and fail here the moment it stopped happening to."""
    from scripts import backtest_accuracy

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)
    _capture_append(monkeypatch)

    row = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    actuals_before = (row.base_price, row.actual_price, row.resolved_at)
    assert actuals_before[2] is not None

    row.direction_actual = "down"
    row.direction_correct = 0
    row.in_interval = 0
    row.abs_error = 999.0
    row.pct_error = 999.0
    session.commit()

    assert backtest_accuracy._refresh_verdict_columns(session) == 1

    session.expire_all()
    after = session.query(ForecastOutcome).filter_by(forecast_id=1).one()

    # Every verdict column is back on the derivation...
    assert after.direction_actual == "flat"
    assert after.direction_correct == 1
    assert after.in_interval == 1
    assert after.abs_error == pytest.approx(0.01, abs=1e-4)
    assert after.pct_error == pytest.approx(abs(3.0 - 3.01) / 3.0 * 100, rel=1e-6)

    # ...and the frozen actuals are exactly what they were.
    assert (after.base_price, after.actual_price, after.resolved_at) == actuals_before


def test_the_refresh_updates_the_parquet_mirror_with_the_frozen_actuals_intact(
    session, tmp_path, monkeypatch
):
    """Test 4 of the brief. forecast_outcomes lives in the DB *and* in
    price-archive/ops/forecast_outcomes.parquet, and the API reads Parquet
    first with a DB fallback (backend/AGENTS.md). A DB-only refresh would leave
    the served copy disagreeing with the headline.

    append_table dedups on forecast_id — the refreshed row replaces the stale
    one — so the whole row is supplied, with the frozen columns carried through
    from the DB row verbatim."""
    from scripts import backtest_accuracy
    import backtest.scoring as scoring

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)
    frozen = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    base_before, actual_before = frozen.base_price, frozen.actual_price
    resolved_before = frozen.resolved_at

    calls = _capture_append(monkeypatch)
    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)
    assert backtest_accuracy._refresh_verdict_columns(session) == 1

    assert len(calls) == 1
    table, rows, dedup_keys = calls[0]
    assert table == "forecast_outcomes"
    assert dedup_keys == ["forecast_id"]
    assert len(rows) == 1

    mirrored = rows[0]
    assert mirrored["forecast_id"] == 1
    # The refreshed verdict reaches the served copy...
    assert mirrored["direction_actual"] == "up"
    assert mirrored["direction_correct"] == 0
    # ...carrying the frozen actuals unchanged, so the replace cannot lose or
    # move them, and resolved_at is preserved rather than reset.
    assert mirrored["base_price"] == pytest.approx(base_before)
    assert mirrored["actual_price"] == pytest.approx(actual_before)
    assert mirrored["resolved_at"] == resolved_before
    # evaluated_at means "when this verdict was last computed", so it moves.
    assert mirrored["evaluated_at"] >= resolved_before


def test_the_rescore_path_also_refreshes_the_stored_verdicts(
    session, tmp_path, monkeypatch
):
    """--rescore is the other path that derives records from frozen rows. It
    must refresh too, otherwise `--rescore` reports one number while the
    columns feeding production bias correction keep another."""
    from scripts import backtest_accuracy
    import backtest.scoring as scoring

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)

    _capture_append(monkeypatch)
    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)

    def explode(*a, **k):
        raise AssertionError("--rescore must not read the archive")

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", explode)

    results = backtest_accuracy.backtest_forecasts(
        session, today=EVAL_DATE, rescore=True
    )
    assert next(r for r in results if r["price_tier"] is None)[
        "metrics"]["directional_accuracy"] == 0.0

    session.expire_all()
    after = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert after.direction_actual == "up"
    assert after.direction_correct == 0


def test_reresolve_writes_current_verdicts_so_the_refresh_is_a_no_op(
    session, tmp_path, monkeypatch
):
    """--reresolve rewrites the rows wholesale from the same derivation the
    refresh uses, so the refresh must find nothing to do rather than
    double-writing every row it just wrote."""
    from scripts import backtest_accuracy

    archive = _freeze_one_flat_forecast(session, tmp_path, monkeypatch)

    real_loader = price_resolution.load_voted_prices

    def loader(archive_dir, slugs, min_date, max_date, **kwargs):
        return real_loader(archive, slugs, min_date, max_date, **kwargs)

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", loader)
    _capture_append(monkeypatch)

    calls = []
    real_refresh = backtest_accuracy._refresh_verdict_columns

    def spy(db, forecast_ids=None):
        n = real_refresh(db, forecast_ids=forecast_ids)
        calls.append(n)
        return n

    monkeypatch.setattr(backtest_accuracy, "_refresh_verdict_columns", spy)

    backtest_accuracy.backtest_forecasts(
        session, today=EVAL_DATE, reresolve=True
    )

    assert calls == [0]
    assert session.query(ForecastOutcome).count() == 1


def test_the_refresh_logs_a_non_zero_count_legibly_and_is_quiet_at_zero(
    session, tmp_path, monkeypatch, caplog
):
    """Zero refreshed is the normal daily case and must not add noise; a
    non-zero count follows a scoring change and must say so."""
    import logging

    from scripts import backtest_accuracy
    import backtest.scoring as scoring

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)
    _capture_append(monkeypatch)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backtest_accuracy"):
        assert backtest_accuracy._refresh_verdict_columns(session) == 0
    assert "Refreshed verdict columns" not in caplog.text

    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backtest_accuracy"):
        assert backtest_accuracy._refresh_verdict_columns(session) == 1
    assert "Refreshed verdict columns on 1 frozen outcome(s)" in caplog.text
    assert "were NOT touched" in caplog.text


def test_the_refresh_skips_rows_it_cannot_derive_a_verdict_for(
    session, tmp_path, monkeypatch
):
    """A legacy row with no usable base leg has no derivable verdict. It must
    be left alone rather than crashing the refresh or having its verdict
    columns nulled out — _records_from_frozen_outcomes already logs it."""
    from scripts import backtest_accuracy

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)
    _capture_append(monkeypatch)

    row = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    row.base_price = None
    row.direction_actual = "down"
    session.commit()

    assert backtest_accuracy._refresh_verdict_columns(session) == 0

    session.expire_all()
    after = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert after.direction_actual == "down"
    assert after.base_price is None


def test_a_failed_mirror_write_leaves_the_refresh_able_to_re_converge(
    session, tmp_path, monkeypatch
):
    """Fix round 1, Finding 1. Idempotence keys off a DB-vs-derived diff, so if
    the DB were committed before the Parquet mirror was written, an
    append_table that raised — or a process killed between the two — would
    leave the served copy stale AND the next run computing zero differences.
    Permanent, silent divergence, reached by a crash rather than by a scoring
    change. It matters most on Task 9's one-off ~65k-row run, the largest and
    most interruptible write this code will ever do.

    Parquet is therefore written first. This test kills the mirror write and
    asserts the DB was NOT advanced past it, so the retry still sees the
    difference and converges both copies. Under the reverse order the retry
    would return 0 and write nothing."""
    import db.parquet as parquet_mod
    from scripts import backtest_accuracy
    import backtest.scoring as scoring

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)
    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)

    def boom(*a, **k):
        raise RuntimeError("parquet mirror exploded")

    monkeypatch.setattr(parquet_mod, "append_table", boom)

    with pytest.raises(RuntimeError, match="parquet mirror exploded"):
        backtest_accuracy._refresh_verdict_columns(session)

    # The DB did not run ahead of the mirror, so the difference still exists.
    session.rollback()
    session.expire_all()
    assert session.query(ForecastOutcome).filter_by(
        forecast_id=1).one().direction_actual == "flat"

    # The retry converges both copies.
    calls = _capture_append(monkeypatch)
    assert backtest_accuracy._refresh_verdict_columns(session) == 1
    assert len(calls) == 1
    assert calls[0][1][0]["direction_actual"] == "up"

    session.expire_all()
    assert session.query(ForecastOutcome).filter_by(
        forecast_id=1).one().direction_actual == "up"


def test_the_refresh_streams_in_bounded_flushes_and_never_re_reads_a_row(
    session, tmp_path, monkeypatch
):
    """Fix round 1, Finding 2 + the memory note. The one-off historical refresh
    is ~65k rows: it must not materialize the whole table, must not hold every
    update and mirror row at once, and must not issue a SELECT per row on the
    write side (the mirror rows are snapshotted during the diff, before any
    commit, so an expire-on-commit cannot force a re-read).

    Driven with 12 forecasts, REFRESH_FLUSH lowered to 5 and the read page to
    3: the refresh must flush in several bounded writes rather than one big
    one, and every row must still converge. A SELECT counter caps total queries
    well below one per row."""
    from scripts import backtest_accuracy
    import backtest.scoring as scoring
    from sqlalchemy import event

    rows = []
    for i in range(12):
        rows += [(f"ak{i}", date(2026, 7, d), 3.00) for d in (3, 4, 5)]
        rows += [(f"ak{i}", date(2026, 7, d), 3.01) for d in (6, 7, 8)]
        _seed(session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0,
              price_low=2.0, price_high=4.0, direction="flat")
    session.commit()
    archive = _write_archive(tmp_path, rows)
    _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 12

    calls = _capture_append(monkeypatch)
    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)
    monkeypatch.setattr(backtest_accuracy, "REFRESH_FLUSH", 5)
    # Read pages smaller than the flush size, to prove flushes land on batch
    # boundaries and the keyset walk survives the commits in between.
    monkeypatch.setattr(backtest_accuracy, "CHUNK", 3)

    selects = []
    conn = session.connection().engine

    @event.listens_for(conn, "before_cursor_execute")
    def count(c, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    try:
        assert backtest_accuracy._refresh_verdict_columns(session) == 12
    finally:
        event.remove(conn, "before_cursor_execute", count)

    # Flushes land on the first read-page boundary at or past REFRESH_FLUSH, so
    # with pages of 3 and a flush size of 5 that is 6 + 6: several bounded
    # writes, never one write of the whole table and never one write per row.
    flushed = [len(rows_) for _, rows_, _ in calls]
    assert flushed == [6, 6]
    assert sum(flushed) == 12
    assert all(n <= backtest_accuracy.REFRESH_FLUSH + backtest_accuracy.CHUNK
               for n in flushed)

    # The write side re-reads nothing. Only the keyset walk selects: 12 rows at
    # 3 per page is 4 pages plus the terminating empty page.
    assert len(selects) <= 6, selects

    session.expire_all()
    stored = session.query(ForecastOutcome).all()
    assert len(stored) == 12
    assert all(o.direction_actual == "up" for o in stored)
    assert all(o.direction_correct == 0 for o in stored)
    # ...and every frozen actual survived the streamed rewrite.
    assert all(o.base_price == pytest.approx(3.00) for o in stored)
    assert all(o.actual_price == pytest.approx(3.01) for o in stored)


def test_the_rescore_walk_does_not_materialize_the_whole_table(
    session, tmp_path, monkeypatch
):
    """The unrestricted (--rescore) read is a keyset walk over the primary key,
    not `SELECT *` into memory. Paging at 3 rows over 12 outcomes must still
    visit every row exactly once — the keyset is stable because the refresh
    never writes `id`."""
    from scripts import backtest_accuracy

    rows = []
    for i in range(12):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0)
    session.commit()
    archive = _write_archive(tmp_path, rows)
    _run_backtest(session, archive, monkeypatch)

    seen = []
    for batch in backtest_accuracy._iter_outcome_rows(session, batch=3):
        assert len(batch) <= 3
        seen.extend(r.forecast_id for r in batch)

    assert sorted(seen) == list(range(1, 13))
    assert len(seen) == len(set(seen))


def test_the_refreshed_mirror_row_replaces_the_stale_one_in_a_real_parquet_file(
    session, tmp_path, monkeypatch
):
    """A round trip through the REAL append_table, with the ops directory
    redirected into tmp_path — the git-tracked price-archive/ops/ is never
    touched.

    Two things this catches that a stubbed append_table cannot. First, the
    refreshed row must REPLACE the stale one on the forecast_id dedup key, not
    append a second row — Task 9 rewrites ~65k rows and a duplicate-per-row
    mirror would be discovered in production. Second, the mirror row's column
    types must match what the insert path wrote: append_table intersects the
    new frame's columns with the file's, so a column with a drifted type (or a
    missing column) corrupts or silently drops data for the whole file."""
    import db.parquet as parquet_mod
    from scripts import backtest_accuracy
    import backtest.scoring as scoring

    real_append = parquet_mod.append_table

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)

    ops = tmp_path / "ops"
    monkeypatch.setattr(parquet_mod, "OPS_DIR", ops)
    monkeypatch.setattr(parquet_mod, "append_table", real_append)

    # Seed the mirror exactly as the insert path does: the frozen row, with the
    # write-time verdict on it.
    frozen = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    seed_row = backtest_accuracy._outcome_to_mapping(frozen)
    seed_row["evaluated_at"] = frozen.evaluated_at
    seed_row["resolved_at"] = frozen.resolved_at
    real_append("forecast_outcomes", [seed_row], ["forecast_id"])

    stored = parquet_mod.read_table("forecast_outcomes")
    assert len(stored) == 1
    assert stored.iloc[0]["direction_actual"] == "flat"

    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)
    assert backtest_accuracy._refresh_verdict_columns(session) == 1

    stored = parquet_mod.read_table("forecast_outcomes")
    # Replaced, not appended.
    assert len(stored) == 1
    row = stored.iloc[0]
    assert row["direction_actual"] == "up"
    assert row["direction_correct"] == 0
    # The frozen actuals came through the round trip intact...
    assert row["base_price"] == pytest.approx(3.00)
    assert row["actual_price"] == pytest.approx(3.01)
    # ...and the date/datetime columns did not drift to strings, which is what
    # would quietly poison the schema of the real 65k-row file.
    assert set(stored.columns) == set(seed_row)
    assert pd.api.types.is_datetime64_any_dtype(stored["resolved_at"])
    assert pd.api.types.is_datetime64_any_dtype(stored["evaluated_at"])
    assert pd.api.types.is_datetime64_any_dtype(stored["target_date"])
    assert not (tmp_path.parent / "price-archive").exists()
