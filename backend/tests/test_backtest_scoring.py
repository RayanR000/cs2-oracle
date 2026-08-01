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

    # The stored row is untouched — only the derivation moved.
    frozen = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert frozen.direction_actual == "flat"
    assert frozen.direction_correct == 1


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
