from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import backtest.price_resolution as price_resolution
import pandas as pd
import pytest
from backtest.price_resolution import resolve_anchors
from backtest.scoring import (
    FLAT_TOLERANCE,
    FLOOR_SWEEP,
    HEADLINE_MIN_TIER,
    HEADLINE_TIER,
    direction_from_return,
    floor_records,
    price_tier,
    score_by_tier,
    score_cohort,
)
from database import Base, ForecastOutcome, Item, ItemForecast, PredictionAccuracy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def smoothed_prices(voted, anchors, **kwargs):
    """resolve_anchors projected to prices. See the note in
    tests/test_backtest_resolution.py — test-local by design."""
    return {k: r.price for k, r in resolve_anchors(voted, anchors, **kwargs).items()}


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
        # The friction-conditioned metric's two inputs. A +10% predicted move at
        # tier 1 is deliberately BELOW the 23.1% actionable threshold, so the
        # default record is in scope and not actionable — the ordinary case.
        "predicted_mid": 1.10,
        "horizon_days": 14,
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
    assert price_tier(999.99) == 4
    assert price_tier(1000.0) == 5
    assert price_tier(29_685.0) == 5  # the priciest name in the archive


def test_tier_4_no_longer_merges_the_two_most_liquid_cohorts():
    """The reason the cut exists: tier 4 used to hold both the 10.8%-spread
    ($50-500) and the 5.2%-spread ($1000+) populations, which are the two most
    DIFFERENT liquidity cohorts in the market. Pooling them made every tier-4
    number uninterpretable.

    Note for anyone reading the stored series: rows written before 2026-08-07
    with price_tier == 4 mean >= $100, not $100-1000.
    """
    assert price_tier(200.0) == 4
    assert price_tier(2000.0) == 5
    assert price_tier(200.0) != price_tier(2000.0)


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


class TestUnchangedPriceSplit:
    """A price the archive carried forward is not a prediction the model got right.

    Measured 2026-08-05: 30-36% of scored outcomes have actual_price bit-identical
    to base_price, at a rate that barely decays from 3d (32.4%) to 30d (31.3%).
    Genuine no-trade would decay with horizon, so that population is dominated by
    archive carry-forward. Those rows label "flat" by construction, and pooling
    them into one headline means the number partly measures archive staleness.
    The split is reported rather than filtered so "how much of our accuracy is
    unchanged prices" stays answerable.
    """

    def test_counts_rows_whose_price_never_moved(self):
        records = [_record(base_price=1.0, actual_price=1.0) for _ in range(3)]
        records += [_record(base_price=1.0, actual_price=1.1) for _ in range(7)]
        metrics, _ = score_cohort(records)
        assert metrics["n_unchanged"] == 3
        assert metrics["unchanged_pct"] == 30.0

    def test_splits_directional_accuracy_by_whether_the_price_moved(self):
        # All 4 unchanged rows correct, 2 of 6 moved rows correct.
        records = [_record(base_price=1.0, actual_price=1.0, direction_correct=1) for _ in range(4)]
        records += [_record(base_price=1.0, actual_price=1.1, direction_correct=1) for _ in range(2)]
        records += [_record(base_price=1.0, actual_price=1.1, direction_correct=0) for _ in range(4)]
        metrics, _ = score_cohort(records)
        assert metrics["directional_accuracy"] == 60.0
        assert metrics["directional_accuracy_unchanged"] == 100.0
        assert metrics["directional_accuracy_moved"] == pytest.approx(33.33, abs=0.01)

    def test_moved_accuracy_is_none_when_every_price_was_carried_forward(self):
        """None, not 0.0 — an empty partition has no accuracy, and a zero here
        would be averaged into reports as though the model scored nothing."""
        records = [_record(base_price=1.0, actual_price=1.0) for _ in range(10)]
        metrics, _ = score_cohort(records)
        assert metrics["directional_accuracy_moved"] is None
        assert metrics["directional_accuracy_unchanged"] == 100.0

    def test_unchanged_accuracy_is_none_when_every_price_moved(self):
        records = [_record(base_price=1.0, actual_price=1.1) for _ in range(10)]
        metrics, _ = score_cohort(records)
        assert metrics["directional_accuracy_unchanged"] is None
        assert metrics["n_unchanged"] == 0
        assert metrics["unchanged_pct"] == 0.0

    def test_split_partitions_the_cohort(self):
        """The two partitions must reconstruct the pooled figure, or the split is
        measuring something other than the headline it sits beside.

        Tolerance is 0.01, not exact: every figure in this module is rounded to
        two decimals, so each partition carries up to 0.005 of rounding error and
        the weighted recombination inherits that bound.
        """
        records = [_record(base_price=1.0, actual_price=1.0, direction_correct=i % 2) for i in range(6)]
        records += [_record(base_price=1.0, actual_price=1.2, direction_correct=i % 3 == 0) for i in range(9)]
        metrics, n = score_cohort(records)
        n_unchanged = metrics["n_unchanged"]
        n_moved = n - n_unchanged
        pooled = (
            metrics["directional_accuracy_unchanged"] * n_unchanged + metrics["directional_accuracy_moved"] * n_moved
        ) / n
        assert pooled == pytest.approx(metrics["directional_accuracy"], abs=0.01)


def test_both_legs_use_the_same_estimator_so_a_flat_market_scores_flat():
    """The end-to-end symmetry property. Under the old code the base leg was a
    3-observation median and the actual leg a single-day price, so a perfectly
    flat market could still produce non-flat direction labels."""
    import pandas as pd

    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(12)]
    voted = pd.DataFrame({"item_id": ["ak"] * 12, "date": days, "price": [3.0] * 12})

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


def _seed(
    session,
    pk,
    slug,
    *,
    current_price,
    price_mid,
    direction="flat",
    price_low=None,
    price_high=None,
    horizon=HORIZON,
    model_version="lgbm-test",
    forecast_date=None,
):
    session.add(Item(id=pk, item_id=slug, name=slug, type="skin"))
    session.add(
        ItemForecast(
            id=pk,
            item_id=pk,
            forecast_date=forecast_date or FORECAST_DATE,
            horizon_days=horizon,
            price_low=price_low,
            price_mid=price_mid,
            price_high=price_high,
            current_price=current_price,
            direction=direction,
            confidence="high",
            model_version=model_version,
        )
    )


def _stub_parquet_writes(monkeypatch):
    """Neutralise EVERY mirror write path.

    Human ruling: these write git-tracked production data under
    price-archive/ops/. append_table is not the only one — replace_rows
    (Task 8e's delete-and-insert) writes there too, and a test that stubbed
    only the former would rewrite the real 65k-row file.
    """
    import db.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod, "append_table", lambda *a, **k: None)
    monkeypatch.setattr(parquet_mod, "replace_rows", lambda *a, **k: None)


def _run_backtest(session, archive, monkeypatch, today=EVAL_DATE, **kwargs):
    """Run backtest_forecasts with the archive redirected to tmp_path.

    backtest_forecasts hardcodes archive_dir to the repo-root price-archive/,
    so the loader is wrapped to substitute the test archive. The real
    load_voted_prices still runs — only the directory it reads is swapped.
    """
    from scripts import backtest_accuracy

    real_loader = price_resolution.load_voted_prices

    def loader(archive_dir, slugs, min_date, max_date, **kw):
        assert Path(archive_dir).name == "price-archive"
        return real_loader(archive, slugs, min_date, max_date, **kw)

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", loader)

    # Maturity is bounded by archive coverage, so this must read the test
    # archive too — otherwise the cutoff comes from the repo's real archive and
    # every test's cohort depends on when the collector last ran.
    real_max_day = price_resolution.archive_max_day

    def max_day(archive_dir):
        assert Path(archive_dir).name == "price-archive"
        return real_max_day(archive)

    monkeypatch.setattr(backtest_accuracy, "archive_max_day", max_day)

    # Same reasoning as archive_max_day: gap classification reads the archive's
    # interior day coverage, so it must see the test archive. Left pointing at
    # the repo's real archive it would decide gaps from whenever the collector
    # last ran.
    real_covered_days = price_resolution.archive_covered_days

    def covered_days(archive_dir):
        assert Path(archive_dir).name == "price-archive"
        return real_covered_days(archive)

    monkeypatch.setattr(backtest_accuracy, "archive_covered_days", covered_days)
    _stub_parquet_writes(monkeypatch)

    return backtest_accuracy.backtest_forecasts(session, today=today, **kwargs)


def test_maturity_is_bounded_by_archive_coverage_not_the_calendar(session, tmp_path, monkeypatch):
    """A forecast is evaluable only once the archive covers its target date.

    Found running Task 9's backfill against prod: maturity was
    `target_date <= today` while resolvability is bounded by the archive's last
    day, which always lags the calendar. 30,859 of 80,737 prod forecasts were
    mature-by-calendar and unresolvable-by-data, tripping the 10% gate at 38.5%
    and refusing to report any number at all. Because the archive lags every
    day, the daily CI run would have failed the same way every day.

    Archive covers through 07-08. `today` is 07-20. The 3d forecast maturing
    07-08 is evaluable; the 3d forecast maturing 07-15 is not, and must be
    excluded from the cohort rather than counted as an unresolvable member of
    it. Pre-fix, both are admitted, 1 of 2 fails to resolve, and the 50%
    unresolvable rate raises.
    """
    archive = _write_archive(
        tmp_path,
        [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        + [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
        + [("awp", date(2026, 7, d), 5.0) for d in range(3, 9)],
    )
    # Matures 2026-07-08 — inside archive coverage.
    _seed(session, 1, "ak", current_price=3.0, price_mid=3.3, direction="up")
    # Matures 2026-07-15 — past archive coverage, before `today`.
    session.add(Item(id=2, item_id="awp", name="awp", type="skin"))
    session.add(
        ItemForecast(
            id=2,
            item_id=2,
            forecast_date=date(2026, 7, 12),
            horizon_days=HORIZON,
            price_low=4.0,
            price_mid=5.0,
            price_high=6.0,
            current_price=5.0,
            direction="flat",
            confidence="high",
            model_version="lgbm-test",
        )
    )
    session.commit()

    results = _run_backtest(session, archive, monkeypatch, today=date(2026, 7, 20))

    outcomes = session.query(ForecastOutcome).all()
    assert [o.forecast_id for o in outcomes] == [1]
    assert results[0]["sample_count"] == 1


def test_archive_coverage_does_not_extend_maturity_past_today(session, tmp_path, monkeypatch):
    """The cutoff is min(today, archive_max), not the archive alone.

    A backfilled archive can hold days beyond `today` (the collector writes a
    range, and callers pass an explicit `today` to score a historical cohort).
    Clamping to the archive only would score forecasts the caller deliberately
    placed in the future.
    """
    archive = _write_archive(tmp_path, [("ak", date(2026, 7, d), 3.0) for d in range(3, 31)])
    _seed(session, 1, "ak", current_price=3.0, price_mid=3.0)
    session.add(Item(id=2, item_id="awp", name="awp", type="skin"))
    session.add(
        ItemForecast(
            id=2,
            item_id=2,
            forecast_date=date(2026, 7, 20),
            horizon_days=HORIZON,  # matures 07-23, after `today`
            price_mid=3.0,
            current_price=3.0,
            direction="flat",
            confidence="high",
            model_version="lgbm-test",
        )
    )
    session.commit()

    _run_backtest(session, archive, monkeypatch, today=date(2026, 7, 10))

    assert [o.forecast_id for o in session.query(ForecastOutcome).all()] == [1]


def test_backtest_scores_the_base_leg_from_the_archive_not_current_price(session, tmp_path, monkeypatch):
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
        [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)] + [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)],
    )
    _seed(
        session,
        1,
        "ak",
        current_price=99.0,  # deliberately nothing like the archive
        price_mid=3.6,
        price_low=3.0,
        price_high=4.0,
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


def test_missing_current_price_is_stored_as_null_not_synthesized_from_base(session, tmp_path, monkeypatch):
    """A forecast with no serving-time current_price must store NULL, not the
    backtest-resolved base price. update_bias_corrections_from_outcomes reads
    this column to compute approx_mid_ret, which feeds production predict()
    thresholds — injecting base there would feed a backtest artefact into
    serving. A genuine NULL is distinguishable; a stand-in is not."""
    archive = _write_archive(tmp_path, [("ak", date(2026, 7, d), 3.0) for d in range(3, 9)])
    _seed(session, 1, "ak", current_price=None, price_mid=3.0)
    session.commit()

    _run_backtest(session, archive, monkeypatch)

    outcome = session.query(ForecastOutcome).one()
    assert outcome.current_price is None
    assert outcome.base_price == pytest.approx(3.0)


def test_backtest_drops_a_forecast_whose_base_leg_is_unresolvable(session, tmp_path, monkeypatch):
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
    guards in this change are disjoint and a test that only covers one would let
    the other regress. "stale30" has a 30-day horizon (target 08-04) and
    observations on 07-03/04/05 then 07-10/11/12. Its actual window is
    07-10/11/12 — wholly AFTER the 07-05 forecast date, so the leg-pair drop does
    not fire — but 07-10 is 25 days before the 08-04 anchor, so the
    anchor-staleness rule does. Under the old between-observations rule that
    window spans 2 days and resolves to a price 23 days stale.

    Twenty fully-covered forecasts keep 2/22 = 9.1% under MAX_UNRESOLVABLE_PCT,
    so the drops are observable in the outcome rows rather than masked by the
    gate. `today` is moved out to 08-10 so the 30-day forecast is mature.
    """
    rows = []
    for i in range(20):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0)
    rows += [("future", date(2026, 7, d), 7.0) for d in (3, 4, 5)]
    _seed(session, 21, "future", current_price=7.0, price_mid=7.0, direction="flat")
    rows += [("stale30", date(2026, 7, d), 7.0) for d in (3, 4, 5, 10, 11, 12)]
    _seed(session, 22, "stale30", current_price=7.0, price_mid=7.0, direction="flat", horizon=30)
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
    both = price_resolution.resolve_anchors(voted, {("future", FORECAST_DATE), ("future", TARGET_DATE)})
    assert both[("future", FORECAST_DATE)].price == both[("future", TARGET_DATE)].price
    # ...off the same window, whose newest observation is not after the forecast
    # date. That is what makes the equality an artefact rather than a flat market.
    assert both[("future", TARGET_DATE)].newest_observation == FORECAST_DATE

    # ...and the complementary premise for "stale30": its actual window IS
    # disjoint from the base leg, so only the anchor-staleness rule can drop it.
    voted30 = pd.DataFrame(
        {
            "item_id": ["stale30"] * 6,
            "date": [date(2026, 7, d) for d in (3, 4, 5, 10, 11, 12)],
            "price": [7.0] * 6,
        }
    )
    base30 = resolve_anchors(voted30, {("stale30", FORECAST_DATE)})
    assert ("stale30", FORECAST_DATE) in base30  # base leg resolves
    assert resolve_anchors(voted30, {("stale30", date(2026, 8, 4))}) == {}

    results = _run_backtest(session, archive, monkeypatch, today=date(2026, 8, 10))

    stored = session.query(ForecastOutcome).all()
    assert len(stored) == 20
    assert {21, 22}.isdisjoint({o.forecast_id for o in stored})
    # No outcome anywhere carries the 7.0 price the stale carry-forward would
    # have produced, on either leg, and none was scored as a manufactured flat.
    assert all(o.base_price == pytest.approx(3.0) for o in stored)
    assert all(o.actual_price == pytest.approx(3.0) for o in stored)
    assert sum(r["sample_count"] for r in results if r["price_tier"] is None) == 20


def test_overlapping_leg_windows_are_dropped_even_with_a_post_forecast_observation(session, tmp_path, monkeypatch):
    """Fix round 1, Important finding. A single post-forecast observation is NOT
    enough to make the actual leg a measurement.

    "overlap" is observed at 1.0 on 07-01/02/03 and then at 1.5 on 07-06 — a 50%
    move, and 07-06 is the ONLY observation after the 07-05 forecast date. The
    two 3-observation windows still overlap in 2 of 3 slots:

        base   = median(07-01, 07-02, 07-03) = median(1.0, 1.0, 1.0) = 1.0
        actual = median(07-02, 07-03, 07-06) = median(1.0, 1.0, 1.5) = 1.0

    actual_ret is exactly 0.0 and the forecast scores "flat" — a fabricated zero
    over a 50% move. Both anchors pass the staleness rule (gaps of 4 and 6 days,
    inside the 7-day cap) and a guard testing only newest_observation passes too,
    because 07-06 > 07-05. Requiring the OLDEST supporting observation to
    post-date the forecast makes the windows disjoint and drops it.

    Note the mid-archive gap: the dry run's mass shape was a truncated tail,
    which is why this variant did not show up there.

    Ten covered forecasts keep 1/11 = 9.1% under MAX_UNRESOLVABLE_PCT so the drop
    is visible in the outcome rows rather than masked by the gate."""
    rows = []
    for i in range(10):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in range(3, 9)]
        _seed(session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0)
    rows += [("overlap", date(2026, 7, d), 1.0) for d in (1, 2, 3)]
    rows += [("overlap", date(2026, 7, 6), 1.5)]
    _seed(session, 11, "overlap", current_price=1.0, price_mid=1.0, direction="flat")
    session.commit()

    archive = _write_archive(tmp_path, rows)

    # The premise: both legs resolve, to the same price, off overlapping windows
    # whose NEWEST observation does post-date the forecast.
    voted = pd.DataFrame(
        {
            "item_id": ["overlap"] * 4,
            "date": [date(2026, 7, d) for d in (1, 2, 3, 6)],
            "price": [1.0, 1.0, 1.0, 1.5],
        }
    )
    both = resolve_anchors(voted, {("overlap", FORECAST_DATE), ("overlap", TARGET_DATE)})
    base_res, actual_res = both[("overlap", FORECAST_DATE)], both[("overlap", TARGET_DATE)]
    assert base_res.price == actual_res.price == 1.0  # the fabricated zero
    assert actual_res.newest_observation > FORECAST_DATE  # newest-guard passes
    assert actual_res.oldest_observation <= FORECAST_DATE  # oldest-guard fires

    results = _run_backtest(session, archive, monkeypatch)

    stored = session.query(ForecastOutcome).all()
    assert len(stored) == 10
    assert 11 not in {o.forecast_id for o in stored}
    assert next(r for r in results if r["price_tier"] is None)["sample_count"] == 10


def test_a_fully_covered_forecast_has_disjoint_leg_windows_at_every_horizon():
    """The stricter rule must cost nothing in production. ItemForecaster.HORIZONS
    is [3, 7, 14, 30] and SMOOTH_WINDOW is 3, so a daily-observed forecast's
    actual window is {f+h-2 ... f+h} — entirely after f whenever h >= 3. Asserted
    against the real HORIZONS list so a new horizon below the smoothing window
    cannot be added without this failing."""
    from backtest.price_resolution import SMOOTH_WINDOW
    from models.forecaster import ItemForecaster

    f = date(2026, 7, 15)
    voted = pd.DataFrame(
        {
            "item_id": ["ak"] * 120,
            "date": [date(2026, 6, 1) + timedelta(days=i) for i in range(120)],
            "price": [3.0] * 120,
        }
    )

    assert min(ItemForecaster.HORIZONS) >= SMOOTH_WINDOW
    for h in ItemForecaster.HORIZONS:
        target = f + timedelta(days=h)
        res = resolve_anchors(voted, {("ak", f), ("ak", target)})
        assert res[("ak", target)].oldest_observation > f, h
        assert res[("ak", target)].oldest_observation == target - timedelta(days=SMOOTH_WINDOW - 1)


def test_an_actual_leg_supported_only_by_pre_forecast_observations_is_unresolvable(session, tmp_path, monkeypatch):
    """The drop above is a resolution failure, so it must reach the gate rather
    than quietly shrinking the cohort — the exact invisibility this plan exists
    to remove. Two beyond-coverage forecasts against seven covered ones is
    2/9 = 22.2%, over the 10% cap, so the run must refuse to report a number.

    The complement is asserted too: once the actual leg's window is wholly after
    the forecast date, it is a real measurement and the gate clears."""
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

    # A DISJOINT actual window clears it: 07-06/07/08 backs the 07-08 leg with
    # nothing the 07-03/04/05 base leg already saw. Note one observation on 07-06
    # alone would NOT be enough — the 07-08 window would still reach back to
    # 07-04 and overlap the base leg. That is the finding this rule closes.
    for i in range(2):
        rows += [(f"end{i}", date(2026, 7, d), 3.6) for d in (6, 7, 8)]
    _revise_archive(archive, rows)

    results = _run_backtest(session, archive, monkeypatch)
    assert next(r for r in results if r["price_tier"] is None)["sample_count"] == 9


def test_unresolvable_forecasts_count_toward_the_gate_denominator(session, tmp_path, monkeypatch):
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


def test_a_forecast_spanning_a_missing_archive_day_is_a_gap_not_a_fresh_failure(session, tmp_path, monkeypatch):
    """A collection outage must not read as cohort shrinkage.

    This is the 2026-08-02/03 shape reproduced small. Two horizon-3 forecasts
    have NO archive day at all inside their actual-leg window, so
    `resolve_anchors` reaches back past the forecast date and the disjoint-leg
    guard drops them — permanently, because backfill is unavailable and a
    forecast that never resolves never earns a frozen outcome.

    Before the GAP category these landed in the fatal coverage ratio: 2 of 10 is
    20%, over the 10% cap, so the run refused to report anything and would have
    done so every day forever. They must now be classified as unscoreable,
    reported, and stepped over, leaving the 8 genuinely scoreable forecasts to
    produce a number.
    """
    rows = []
    # 8 scoreable forecasts at horizon 14. Their actual leg (target 07-19) has
    # 07-17/18/19 behind it — three observations strictly after the forecast.
    for i in range(8):
        rows += [(f"ok{i}", date(2026, 7, d), 3.0) for d in (3, 4, 5, 17, 18, 19)]
        _seed(session, i + 1, f"ok{i}", current_price=3.0, price_mid=3.0, horizon=14)
    # 2 forecasts at horizon 3 (target 07-08). Nothing exists between 07-05 and
    # 07-08 anywhere in the archive, so the 07-08 anchor resolves from
    # 07-03/04/05 — the base leg's own window.
    for i in range(2):
        rows += [(f"hole{i}", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        _seed(session, 100 + i, f"hole{i}", current_price=3.0, price_mid=3.0)
    session.commit()

    archive = _write_archive(tmp_path, rows)

    results = _run_backtest(session, archive, monkeypatch)

    headline = next(r for r in results if r["price_tier"] is None)
    assert headline["sample_count"] == 8
    # The two gap forecasts are stepped over, not scored and not frozen.
    assert session.query(ForecastOutcome).count() == 8


def test_a_gap_population_does_not_excuse_a_real_resolution_failure(session, tmp_path, monkeypatch):
    """The loophole check, end to end.

    Gap rows leave the fresh denominator, so this has to prove the FRESH RATE
    leg specifically — not just that some ratio caught the failure. The numbers
    are chosen so coverage cannot be the one that fires: 2 genuine failures out
    of 20 mature is exactly 10.0%, which is not *over* the 10% cap. Against the
    8 attempts that carried information it is 25%, and that is what must fail.

    Left in the denominator, the 12 gap rows would have diluted those 2 failures
    to 10% and waved a broken resolver through. That is the dilution the module
    docstring warns about, now reachable through the new category.
    """
    rows = []
    # 6 scoreable.
    for i in range(6):
        rows += [(f"ok{i}", date(2026, 7, d), 3.0) for d in (3, 4, 5, 17, 18, 19)]
        _seed(session, i + 1, f"ok{i}", current_price=3.0, price_mid=3.0, horizon=14)
    # 2 genuine failures. Days 07-17/18/19 DO exist in the archive, so these
    # items' silence is about the items, not the calendar — their 07-19 anchor
    # has nothing within MAX_WINDOW_SPAN_DAYS. Not a gap.
    for i in range(2):
        rows += [(f"dead{i}", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        _seed(session, 50 + i, f"dead{i}", current_price=3.0, price_mid=3.0, horizon=14)
    # 12 gap rows — nothing exists in the archive inside their actual-leg window.
    for i in range(12):
        rows += [(f"hole{i}", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        _seed(session, 100 + i, f"hole{i}", current_price=3.0, price_mid=3.0)
    session.commit()

    archive = _write_archive(tmp_path, rows)

    with pytest.raises(RuntimeError, match="resolution rate"):
        _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 0


def test_resolution_is_insert_only(session, monkeypatch):
    import scripts.backtest_accuracy as bt

    # Human ruling: the mirror writes touch git-tracked production data
    # (price-archive/ops/forecast_outcomes.parquet). Never let a test reach them.
    _stub_parquet_writes(monkeypatch)

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

    _stub_parquet_writes(monkeypatch)

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
    assert bt._store_forecast_outcomes(session, [revised], reresolve=True, considered_ids={8}) == 1

    assert session.query(ForecastOutcome).filter_by(forecast_id=8).one().actual_price == 99.0



def test_tier_rows_partition_the_all_row():
    """The price bands partition the all-tiers row.

    Every FLOOR_SWEEP sentinel is excluded on purpose: they are floor
    aggregates, not bands, and they deliberately overlap bands 1..5. Summing
    them with the bands would double-count.
    """
    records = [_record(price_tier=0, item_id=i) for i in range(30)]
    records += [_record(price_tier=1, item_id=100 + i) for i in range(20)]

    scored = score_by_tier(records)
    per_tier = {tier: n for tier, _, n in scored if tier is not None and tier not in FLOOR_SWEEP}
    all_rows = [(m, n) for tier, m, n in scored if tier is None]

    assert per_tier == {0: 30, 1: 20}
    assert len(all_rows) == 1
    assert all_rows[0][1] == 50
    assert sum(per_tier.values()) == all_rows[0][1]


def test_headline_row_covers_exactly_the_tiers_at_or_above_the_minimum():
    records = [_record(price_tier=0, item_id=i) for i in range(30)]
    records += [_record(price_tier=1, item_id=100 + i) for i in range(20)]
    records += [_record(price_tier=3, item_id=200 + i) for i in range(5)]

    scored = score_by_tier(records)
    headline_n = next(n for tier, _, n in scored if tier == HEADLINE_TIER)
    above_min = sum(
        n for tier, _, n in scored if tier is not None and tier != HEADLINE_TIER and tier >= HEADLINE_MIN_TIER
    )
    assert headline_n == above_min == 25


def test_empty_tiers_are_omitted_not_zero_filled():
    records = [_record(price_tier=4, item_id=i) for i in range(12)]
    tiers = {tier for tier, _, _ in score_by_tier(records) if tier is not None and tier not in FLOOR_SWEEP}
    assert tiers == {4}


def test_headline_tier_is_one_dollar_and_up():
    assert HEADLINE_MIN_TIER == 1
    assert price_tier(0.99) < HEADLINE_MIN_TIER
    assert price_tier(1.00) >= HEADLINE_MIN_TIER


# ---------------------------------------------------------------------------
# The friction-conditioned metric and the headline floor sweep.
# ---------------------------------------------------------------------------


def test_score_cohort_publishes_the_actionable_metric():
    records = [
        _record(item_id=i, base_price=2000.0, actual_price=3000.0, predicted_mid=3000.0, price_tier=5, horizon_days=14)
        for i in range(20)
    ]
    metrics, _ = score_cohort(records)
    assert metrics["actionable_scope"] == "in_scope"
    assert metrics["actionable_n"] == 20
    assert metrics["actionable_da"] == 100.0


def test_score_cohort_marks_short_horizons_out_of_scope():
    """A cohort at h=3 must say why the metric is absent, not report zeros."""
    records = [_record(item_id=i, horizon_days=3) for i in range(20)]
    metrics, _ = score_cohort(records)
    assert metrics["actionable_scope"] == "out_of_scope"
    assert metrics["actionable_n"] is None


def test_the_actionable_keys_are_present_on_every_cohort():
    """Fixed shape. A key set that varies by cohort is the defect that forced
    every nested metrics value in this store to JSON text."""
    in_scope, _ = score_cohort([_record(item_id=i, horizon_days=14) for i in range(20)])
    out_of, _ = score_cohort([_record(item_id=i, horizon_days=3) for i in range(20)])
    actionable = {k for k in in_scope if k.startswith("actionable_")}
    assert actionable
    assert actionable == {k for k in out_of if k.startswith("actionable_")}


def test_every_sweep_floor_is_a_band_edge():
    """floor_records filters in tier space, which is only equivalent to a dollar
    floor when the floor is a price_tier cut. Pinned so a later $50 floor cannot
    be added silently and quietly round down to the $20 band."""
    for floor in FLOOR_SWEEP.values():
        assert price_tier(floor - 0.01) < price_tier(floor), f"${floor} is not a band edge"


def test_floor_sweep_emits_one_stored_row_per_floor():
    """The sweep answers 'where does the headline stabilise'. Stored, not just
    logged, for the reason HEADLINE_TIER is stored: a headline that exists only
    in console output cannot be audited or recomputed."""
    records = [_record(item_id=i, base_price=2.0, price_tier=price_tier(2.0)) for i in range(10)] + [
        _record(item_id=20 + i, base_price=50.0, price_tier=price_tier(50.0)) for i in range(10)
    ]
    by_tier = {t: n for t, _, n in score_by_tier(records)}
    assert FLOOR_SWEEP == {-1: 1.0, -2: 5.0, -3: 20.0}
    assert by_tier[-1] == 20  # >= $1
    assert by_tier[-2] == 10  # >= $5
    assert by_tier[-3] == 10  # >= $20


def test_the_floors_nest():
    records = [
        _record(item_id=i, base_price=base, price_tier=price_tier(base))
        for i, base in enumerate([0.5, 2.0, 8.0, 50.0, 500.0, 5000.0] * 4)
    ]
    by_tier = {t: n for t, _, n in score_by_tier(records)}
    assert by_tier[-3] <= by_tier[-2] <= by_tier[-1]


def test_headline_tier_is_still_the_dollar_floor():
    """/accuracy/headline, the homepage placard and the stored series all key on
    HEADLINE_TIER. The sweep must not renumber it."""
    assert HEADLINE_TIER == -1
    assert FLOOR_SWEEP[HEADLINE_TIER] == 1.0


def test_the_dollar_floor_is_exactly_the_old_headline_cohort():
    """headline_records changed from a price_tier comparison to a FLOOR_SWEEP
    lookup. The population it selects must not have moved."""
    records = [
        _record(item_id=i, base_price=base, price_tier=price_tier(base))
        for i, base in enumerate([0.2, 0.99, 1.0, 7.0, 300.0, 9000.0])
    ]
    assert floor_records(records, FLOOR_SWEEP[HEADLINE_TIER]) == [
        r for r in records if r["price_tier"] >= HEADLINE_MIN_TIER
    ]


def test_a_floor_no_record_reaches_is_omitted_not_emitted_as_zero():
    records = [_record(item_id=i, base_price=2.0, price_tier=price_tier(2.0)) for i in range(10)]
    tiers = {t for t, _, _ in score_by_tier(records)}
    assert -1 in tiers
    assert -2 not in tiers and -3 not in tiers


def test_forecasts_with_no_slug_mapping_count_toward_the_gate(session, tmp_path, monkeypatch):
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

    # Must reach TARGET_DATE: maturity is bounded by archive coverage, so an
    # archive stopping at FORECAST_DATE would exclude this forecast from the
    # cohort entirely and the gate would never engage. The missing slug has to
    # be the only reason it fails to resolve.
    archive = _write_archive(tmp_path, [("ak", date(2026, 7, d), 3.0) for d in range(3, 9)])

    with pytest.raises(RuntimeError, match="could not be resolved"):
        _run_backtest(session, archive, monkeypatch)


def test_rescore_path_emits_the_same_tier_rows_as_the_normal_path(session, tmp_path, monkeypatch):
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
    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6, price_low=3.0, price_high=4.0, direction="up")
    _seed(session, 2, "awp", current_price=10.0, price_mid=11.5, price_low=10.0, price_high=13.0, direction="up")
    session.commit()

    normal_results = _run_backtest(session, archive, monkeypatch)

    import db.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod, "append_table", lambda *a, **k: None)
    rescore_results = backtest_accuracy.backtest_forecasts(session, today=EVAL_DATE, rescore=True)

    def shape(results):
        return {(r["horizon_days"], r["model_version"], r["price_tier"]): r["sample_count"] for r in results}

    normal_shape = shape(normal_results)
    rescore_shape = shape(rescore_results)

    # A per-tier row (tier 1 for "ak", tier 2 for "awp"), the >=$1 and >=$5
    # floor sentinels the fixture's prices reach, and the all-tiers aggregate
    # (price_tier=None) must all be present in both paths. There is no >=$20
    # row: nothing in the fixture is that expensive, and an unreached floor is
    # omitted rather than zero-filled.
    assert {None, 1, 2, HEADLINE_TIER, -2} == {t for (_, _, t) in normal_shape}
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

    assert (
        direction_from_return(
            (before[("ak", date(2026, 7, 8))] - before[("ak", date(2026, 7, 5))]) / before[("ak", date(2026, 7, 5))]
        )
        == "flat"
    )

    # Voting rejects the outlier source; even unfrozen, the estimator holds.
    assert after[("ak", date(2026, 7, 8))] == before[("ak", date(2026, 7, 8))]


def test_backtest_forecasts_reports_identical_metrics_across_an_archive_revision(session, tmp_path, monkeypatch):
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
        session,
        1,
        "ak",
        current_price=3.0,
        price_mid=3.6,
        price_low=3.0,
        price_high=4.0,
        direction="up",
    )
    _seed(
        session,
        2,
        "awp",
        current_price=10.0,
        price_mid=10.1,
        price_low=9.0,
        price_high=11.0,
        direction="flat",
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
    pd.concat([base_frame, extra], ignore_index=True).to_parquet(archive / "prices-2026.parquet")

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
    outcomes = {o.forecast_id: (o.base_price, o.actual_price) for o in session.query(ForecastOutcome).all()}
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


def test_frozen_values_drive_the_metric_not_a_revised_archive(session, tmp_path, monkeypatch):
    """Run 1 freezes "ak" at base 3.0 -> actual 3.3 (MAPE 10%). The archive is
    then rewritten so "ak" would now resolve to 30.0 at the target date — a
    2-tier move that no amount of window-median robustness absorbs. A second,
    unfrozen forecast ("awp", tier 2) is added so the archive genuinely IS
    read on run 2; this is not passing merely because nothing was resolved.

    The tier-1 row must still report the frozen 10%, not 880%."""
    ak_rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    ak_rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, ak_rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6, price_low=3.0, price_high=4.0, direction="up")
    session.commit()

    before = _run_backtest(session, archive, monkeypatch)
    tier1_before = next(r for r in before if r["price_tier"] == 1)
    assert tier1_before["metrics"]["mape"] == pytest.approx(10.0)

    # The archive is revised outright: "ak" now sits at 30.0 from 07-06 on.
    revised = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    revised += [("ak", date(2026, 7, d), 30.0) for d in (6, 7, 8)]
    revised += [("awp", date(2026, 7, d), 10.0) for d in (3, 4, 5, 6, 7, 8)]
    _revise_archive(archive, revised)

    _seed(session, 2, "awp", current_price=10.0, price_mid=10.1, price_low=9.0, price_high=11.0, direction="flat")
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


def test_archive_is_not_read_when_every_mature_forecast_is_frozen(session, tmp_path, monkeypatch):
    """Nothing new to resolve => no archive access at all, and no divide-by-zero
    in the unresolvable gate over an empty denominator."""
    from scripts import backtest_accuracy

    rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6, price_low=3.0, price_high=4.0, direction="up")
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
                r["sample_count"],
                r["metrics"],
            )
            for r in results
        }

    assert shape(before) == shape(after)
    assert session.query(ForecastOutcome).count() == 1


def test_a_scoring_fix_lands_on_frozen_rows_without_touching_the_archive(session, tmp_path, monkeypatch):
    """The reason records are re-derived rather than read back column-for-column
    from forecast_outcomes: changing the flat tolerance must change the reported
    directional accuracy of already-frozen rows, with no archive read.

    "ak" moves 3.00 -> 3.01 (+0.33%), inside the 0.5% flat band, so a "flat"
    prediction scores correct. Widen nothing and shrink FLAT_TOLERANCE to 0.1%
    and the same frozen row is now "up" — and must score wrong."""
    import backtest.scoring as scoring
    from scripts import backtest_accuracy

    rows = [("ak", date(2026, 7, d), 3.00) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.01) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.0, price_low=2.0, price_high=4.0, direction="flat")
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


def test_gate_is_not_diluted_by_the_frozen_majority(session, tmp_path, monkeypatch):
    """20 frozen forecasts plus 10 new ones of which 2 are unresolvable is a 20%
    resolution failure rate and must trip the gate.

    This is the DILUTION half of the gate's contract. Cohort coverage here is
    only 2/30 = 6.7% and passes, so it is the fresh-rate ratio — 2 of 10
    newly-resolvable forecasts — that has to catch this. A gate measuring
    coverage alone would wave it through.

    The 2 failures are classified fresh, not chronic: their target date is 3
    days behind the archive's coverage end, inside the MAX_WINDOW_SPAN_DAYS
    grace window, so they still look like data that could arrive. See
    test_unresolvable_gate_denominator.py for the HYPERSENSITIVITY half.
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


def test_unusable_frozen_rows_are_counted_and_logged_not_swallowed(session, tmp_path, monkeypatch, caplog):
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

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6, price_low=3.0, price_high=4.0, direction="up")
    _seed(session, 2, "awp", current_price=10.0, price_mid=10.1, price_low=9.0, price_high=11.0, direction="flat")
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
    unusable = [r for r in caplog.records if "UNUSABLE" in r.message and r.levelno >= logging.WARNING]
    assert len(unusable) == 1
    assert "1 frozen outcome(s) UNUSABLE" in unusable[0].message

    # Visibility, not enforcement — the run still reports a number.
    assert backtest_accuracy.MAX_UNRESOLVABLE_PCT == 10.0


def test_the_unusable_hint_does_not_promise_reresolve_will_recover_them(session, caplog):
    """The warning used to end "Re-resolve them with --reresolve to bring them
    back into the metric." For part of the population that actually triggered it
    the truth was the opposite: --reresolve DELETES the row.

    Run 31057993603 (2026-08-05) dropped 14,233 frozen outcomes. They were
    pre-freeze vintage with a NULL base_price — the column arrived in migration
    0019 on 2026-08-01, and predicted_price_mid / actual_price are
    nullable=False, so a NULL base_price was the only reachable trigger — all
    with target_date 2026-08-01. At h=3 (forecast 07-29) the archive holds only
    07-31 and 08-01 in the actual-leg window because 2026-07-30 is missing, so
    classify_archive_gap calls it unresolvable, --reresolve puts it in
    considered_ids and its frozen row is deleted instead of repaired. At h=7/14/30
    the same target date re-resolves normally.

    So the hint must name the precondition — the archive can still supply a
    clean actual leg for their target_date — and the consequence when it does
    not hold, rather than promising recovery for all of them.
    """
    import logging

    from scripts import backtest_accuracy

    session.add(
        ForecastOutcome(
            forecast_id=1,
            item_id=1,
            forecast_date=date(2026, 7, 29),
            horizon_days=3,
            target_date=date(2026, 8, 1),
            base_price=None,
            predicted_price_mid=1.1,
            actual_price=1.2,
            abs_error=0.1,
        )
    )
    session.commit()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="backtest_accuracy"):
        backtest_accuracy._records_from_frozen_outcomes(session)

    msg = next(r.message for r in caplog.records if "UNUSABLE" in r.message)
    assert "1 frozen outcome(s) UNUSABLE" in msg
    # The refuted claim must be gone.
    assert "bring them back into the metric" not in msg
    # The consequence when the archive cannot cover their actual leg.
    assert "delete" in msg.lower()
    # And the check that tells the two cases apart.
    assert "target_date" in msg


def test_frozen_outcome_query_is_restricted_in_sql_not_in_python(session, tmp_path, monkeypatch):
    """The forecast_ids restriction must be a chunked SQL IN list, not a full
    table scan filtered afterwards — forecast_outcomes is the largest table in
    the daily path."""
    from scripts import backtest_accuracy

    rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)
    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6, price_low=3.0, price_high=4.0, direction="up")
    session.commit()
    _run_backtest(session, archive, monkeypatch)

    groups = backtest_accuracy._records_from_frozen_outcomes(session, forecast_ids=[])
    assert groups == {}

    groups = backtest_accuracy._records_from_frozen_outcomes(session, forecast_ids=[1])
    assert sum(len(v) for v in groups.values()) == 1

    # 2,000 ids is past the SQLite 999-parameter cap; the chunking must hold.
    groups = backtest_accuracy._records_from_frozen_outcomes(session, forecast_ids=list(range(1, 2001)))
    assert sum(len(v) for v in groups.values()) == 1


def test_frozen_outcome_records_carry_the_prediction_leg_and_the_horizon(session, tmp_path, monkeypatch):
    """ActionableDA needs r_hat, so the record needs predicted_mid; and it is
    scoped by horizon, so the record needs the horizon.

    Both ride ON the record rather than as score_cohort parameters: records are
    grouped by (horizon, model_version) so every record in a cohort shares the
    horizon, and eight test modules call score_cohort(records) positionally.
    """
    from scripts import backtest_accuracy

    rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)
    _seed(session, 1, "ak", current_price=3.0, price_mid=3.6, price_low=3.0, price_high=4.0, direction="up")
    session.commit()
    _run_backtest(session, archive, monkeypatch)

    groups = backtest_accuracy._records_from_frozen_outcomes(session)
    records = [r for rs in groups.values() for r in rs]
    assert records, "fixture produced no records"
    for r in records:
        assert r["predicted_mid"] == 3.6
        assert r["horizon_days"] in (3, 7, 14, 30)


def test_headline_log_line_handles_fewer_than_ten_samples_without_raising(session, tmp_path, monkeypatch):
    """bootstrap_ci returns (None, None) under 10 values. The >=$1 headline
    log line in backtest_forecasts guards ci_lower is not None before
    multiplying by 100 — this drives that branch explicitly with a 3-forecast
    cohort rather than relying on it firing incidentally in other tests."""
    rows = []
    for i in range(3):
        rows += [(f"ak{i}", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
        rows += [(f"ak{i}", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
        _seed(
            session,
            i + 1,
            f"ak{i}",
            current_price=3.0,
            price_mid=3.6,
            price_low=3.0,
            price_high=4.0,
            direction="up",
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
    # replace_rows reaches the same production file; --reresolve goes through it.
    monkeypatch.setattr(parquet_mod, "replace_rows", lambda *a, **k: None)
    return calls


def _freeze_one_flat_forecast(session, tmp_path, monkeypatch):
    """Run 1: "ak" moves 3.00 -> 3.01 (+0.33%), inside the 0.5% flat band, so
    a "flat" prediction is frozen as correct. Returns the archive dir."""
    rows = [("ak", date(2026, 7, d), 3.00) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.01) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(session, 1, "ak", current_price=3.0, price_mid=3.0, price_low=2.0, price_high=4.0, direction="flat")
    session.commit()

    _run_backtest(session, archive, monkeypatch)

    frozen = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert frozen.direction_actual == "flat"
    assert frozen.direction_correct == 1
    return archive


def _actuals_snapshot(session):
    """The three columns the refresh must never write, plus their exact
    values, for every stored outcome."""
    return {o.forecast_id: (o.base_price, o.actual_price, o.resolved_at) for o in session.query(ForecastOutcome).all()}


def test_a_scoring_change_refreshes_the_stored_verdict_columns(session, tmp_path, monkeypatch):
    """Test 1 of the brief. Shrink FLAT_TOLERANCE so the frozen row reclassifies
    from "flat" to "up", run the normal path, and the STORED verdict columns
    must now match the new derivation — while base_price, actual_price and
    resolved_at are identical to before, compared value-for-value."""
    import backtest.scoring as scoring
    from scripts import backtest_accuracy

    archive = _freeze_one_flat_forecast(session, tmp_path, monkeypatch)

    before_row = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    before_verdict = (
        before_row.direction_actual,
        before_row.direction_correct,
        before_row.abs_error,
        before_row.pct_error,
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
    import backtest.scoring as scoring
    from scripts import backtest_accuracy

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


def test_the_refresh_updates_the_parquet_mirror_with_the_frozen_actuals_intact(session, tmp_path, monkeypatch):
    """Test 4 of the brief. forecast_outcomes lives in the DB *and* in
    price-archive/ops/forecast_outcomes.parquet, and the API reads Parquet
    first with a DB fallback (backend/AGENTS.md). A DB-only refresh would leave
    the served copy disagreeing with the headline.

    append_table dedups on forecast_id — the refreshed row replaces the stale
    one — so the whole row is supplied, with the frozen columns carried through
    from the DB row verbatim."""
    import backtest.scoring as scoring
    from scripts import backtest_accuracy

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


def test_the_rescore_path_also_refreshes_the_stored_verdicts(session, tmp_path, monkeypatch):
    """--rescore is the other path that derives records from frozen rows. It
    must refresh too, otherwise `--rescore` reports one number while the
    columns feeding production bias correction keep another."""
    import backtest.scoring as scoring
    from scripts import backtest_accuracy

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)

    _capture_append(monkeypatch)
    monkeypatch.setattr(scoring, "FLAT_TOLERANCE", 0.001)

    def explode(*a, **k):
        raise AssertionError("--rescore must not read the archive")

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", explode)

    results = backtest_accuracy.backtest_forecasts(session, today=EVAL_DATE, rescore=True)
    assert next(r for r in results if r["price_tier"] is None)["metrics"]["directional_accuracy"] == 0.0

    session.expire_all()
    after = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert after.direction_actual == "up"
    assert after.direction_correct == 0


def test_reresolve_writes_current_verdicts_so_the_refresh_is_a_no_op(session, tmp_path, monkeypatch):
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

    backtest_accuracy.backtest_forecasts(session, today=EVAL_DATE, reresolve=True)

    assert calls == [0]
    assert session.query(ForecastOutcome).count() == 1


def test_the_refresh_logs_a_non_zero_count_legibly_and_is_quiet_at_zero(session, tmp_path, monkeypatch, caplog):
    """Zero refreshed is the normal daily case and must not add noise; a
    non-zero count follows a scoring change and must say so."""
    import logging

    import backtest.scoring as scoring
    from scripts import backtest_accuracy

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


def test_the_refresh_skips_rows_it_cannot_derive_a_verdict_for(session, tmp_path, monkeypatch):
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


def test_a_failed_mirror_write_leaves_the_refresh_able_to_re_converge(session, tmp_path, monkeypatch):
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
    import backtest.scoring as scoring
    import db.parquet as parquet_mod
    from scripts import backtest_accuracy

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
    assert session.query(ForecastOutcome).filter_by(forecast_id=1).one().direction_actual == "flat"

    # The retry converges both copies.
    calls = _capture_append(monkeypatch)
    assert backtest_accuracy._refresh_verdict_columns(session) == 1
    assert len(calls) == 1
    assert calls[0][1][0]["direction_actual"] == "up"

    session.expire_all()
    assert session.query(ForecastOutcome).filter_by(forecast_id=1).one().direction_actual == "up"


def test_the_refresh_streams_in_bounded_flushes_and_never_re_reads_a_row(session, tmp_path, monkeypatch):
    """Fix round 1, Finding 2 + the memory note. The one-off historical refresh
    is ~65k rows: it must not materialize the whole table, must not hold every
    update and mirror row at once, and must not issue a SELECT per row on the
    write side (the mirror rows are snapshotted during the diff, before any
    commit, so an expire-on-commit cannot force a re-read).

    Driven with 12 forecasts, REFRESH_FLUSH lowered to 5 and the read page to
    3: the refresh must flush in several bounded writes rather than one big
    one, and every row must still converge. A SELECT counter caps total queries
    well below one per row."""
    import backtest.scoring as scoring
    from scripts import backtest_accuracy
    from sqlalchemy import event

    rows = []
    for i in range(12):
        rows += [(f"ak{i}", date(2026, 7, d), 3.00) for d in (3, 4, 5)]
        rows += [(f"ak{i}", date(2026, 7, d), 3.01) for d in (6, 7, 8)]
        _seed(
            session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0, price_low=2.0, price_high=4.0, direction="flat"
        )
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
    assert all(n <= backtest_accuracy.REFRESH_FLUSH + backtest_accuracy.CHUNK for n in flushed)

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


def test_the_rescore_walk_does_not_materialize_the_whole_table(session, tmp_path, monkeypatch):
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


def test_the_refreshed_mirror_row_replaces_the_stale_one_in_a_real_parquet_file(session, tmp_path, monkeypatch):
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
    import backtest.scoring as scoring
    import db.parquet as parquet_mod
    from scripts import backtest_accuracy

    real_append = parquet_mod.append_table

    _freeze_one_flat_forecast(session, tmp_path, monkeypatch)

    ops = tmp_path / "ops"
    monkeypatch.setattr(parquet_mod, "OPS_DIR", ops)
    monkeypatch.setattr(parquet_mod, "append_table", real_append)

    # Seed the mirror exactly as the insert path does: the frozen row, with the
    # write-time verdict on it and the denormalised item_slug the mirror
    # carries so it can be joined to prices-*.parquet without the DB.
    frozen = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    seed_row = backtest_accuracy._outcome_to_mapping(frozen)
    seed_row["evaluated_at"] = frozen.evaluated_at
    seed_row["resolved_at"] = frozen.resolved_at
    (seed_row,) = backtest_accuracy._with_item_slug([seed_row], backtest_accuracy._id_to_slug(session))
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


# ---------------------------------------------------------------------------
# Task 8e: --reresolve must not leave orphaned stale rows behind.
#
# The delete set used to be derived from the forecasts that RESOLVED on the run.
# A forecast that previously had a stored row and is now unresolvable — dropped
# by the anchor-staleness rule (8d) or the leg-window disjointness rule (8d) —
# never appears there, so its row survived. On the Task 9b dry run that was
# 8,784 rows still holding values from the old broken estimator: scored into the
# published metric, fed into update_bias_corrections_from_outcomes, and
# uncorrectable, because --reresolve is the only path that can move a frozen
# value at all.
#
# The delete set is now the CONSIDERED set. Only --reresolve deletes.
# ---------------------------------------------------------------------------


def _cohort_rows(n, broken=()):
    """Archive rows for n items: 3.00 before the forecast date, 3.01 after.

    A slug in *broken* keeps only its pre-forecast observations, so its target
    anchor resolves off the SAME window as its base anchor and the leg-pair
    guard drops it — the exact shape that produced the orphans.
    """
    rows = []
    for i in range(n):
        slug = f"ak{i}"
        rows += [(slug, date(2026, 7, d), 3.00) for d in (3, 4, 5)]
        if slug not in broken:
            rows += [(slug, date(2026, 7, d), 3.01) for d in (6, 7, 8)]
    return rows


def _seed_cohort(session, tmp_path, n=12):
    """n mature flat forecasts, all resolvable. Returns the archive dir.

    n = 12 keeps one unresolvable forecast at 8.3%, under the 10%
    MAX_UNRESOLVABLE_PCT gate, which is not weakened anywhere here.
    """
    for i in range(n):
        _seed(
            session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0, price_low=2.0, price_high=4.0, direction="flat"
        )
    session.commit()
    return _write_archive(tmp_path, _cohort_rows(n))


def _break_ak0(archive, n=12):
    """Revise the archive so ak0 (forecast_id 1) can no longer be resolved."""
    _revise_archive(archive, _cohort_rows(n, broken={"ak0"}))


def _run_with_real_mirror(session, archive, monkeypatch, today=EVAL_DATE, **kwargs):
    """_run_backtest, but with the REAL Parquet writers left in place.

    Callers must have redirected db.parquet.OPS_DIR into tmp_path first; the
    assert below is the guard that they did, so no test can reach the
    git-tracked price-archive/ops/.
    """
    import db.parquet as parquet_mod
    from scripts import backtest_accuracy

    assert "price-archive" not in str(parquet_mod.OPS_DIR), parquet_mod.OPS_DIR
    # ...and that a stub left over from an earlier _run_backtest in the same
    # test is not quietly swallowing the writes this test means to inspect.
    for name in ("append_table", "replace_rows"):
        assert getattr(parquet_mod, name).__module__ == "db.parquet", name

    real_loader = price_resolution.load_voted_prices

    def loader(archive_dir, slugs, min_date, max_date, **kw):
        return real_loader(archive, slugs, min_date, max_date, **kw)

    monkeypatch.setattr(backtest_accuracy, "load_voted_prices", loader)
    return backtest_accuracy.backtest_forecasts(session, today=today, **kwargs)


def test_reresolve_deletes_the_row_of_a_forecast_that_no_longer_resolves(session, tmp_path, monkeypatch):
    """Test 1 of the brief. The orphan is deleted from the DB.

    ak0 resolves on run 1 and is frozen. The archive is then revised so its
    actual leg can only be supported by pre-forecast observations, which the
    leg-pair guard rejects — so ak0 produces no outcome on the --reresolve run
    and never appears in the write set. Its stale row must still go."""
    archive = _seed_cohort(session, tmp_path)
    _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 12

    # Mark the row so a survivor is unmistakably the OLD, broken-estimator one.
    stale = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    stale.actual_price = 999.0
    session.commit()

    _break_ak0(archive)
    _run_backtest(session, archive, monkeypatch, reresolve=True)

    session.expire_all()
    assert session.query(ForecastOutcome).filter_by(forecast_id=1).first() is None
    # ...and only that one went: the other eleven were rewritten, not dropped.
    assert session.query(ForecastOutcome).count() == 11
    assert all(o.actual_price == pytest.approx(3.01) for o in session.query(ForecastOutcome).all())


def test_reresolve_deletes_the_orphan_from_the_parquet_mirror_too(session, tmp_path, monkeypatch):
    """Test 2 of the brief, against a REAL Parquet file with OPS_DIR redirected.

    The mirror is what the API serves (backend/AGENTS.md: routes read Parquet
    first, DB fallback), so a DB-only delete would just move the bug to the copy
    users actually see. append_table cannot express a delete at all — this is
    the half that needs db.parquet.replace_rows."""
    import db.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod, "OPS_DIR", tmp_path / "ops")

    archive = _seed_cohort(session, tmp_path)
    _run_with_real_mirror(session, archive, monkeypatch)

    mirror = parquet_mod.read_table("forecast_outcomes")
    assert len(mirror) == 12
    assert 1 in set(mirror["forecast_id"])

    _break_ak0(archive)
    _run_with_real_mirror(session, archive, monkeypatch, reresolve=True)

    mirror = parquet_mod.read_table("forecast_outcomes")
    assert len(mirror) == 11
    assert 1 not in set(mirror["forecast_id"])
    # The eleven survivors were replaced in place, not duplicated, and their
    # frozen actuals and column types survived the delete-and-insert rewrite.
    assert sorted(mirror["forecast_id"]) == list(range(2, 13))
    assert mirror["actual_price"].round(4).eq(3.01).all()
    assert pd.api.types.is_datetime64_any_dtype(mirror["resolved_at"])
    assert pd.api.types.is_datetime64_any_dtype(mirror["target_date"])

    # The DB agrees with the served copy.
    session.expire_all()
    assert session.query(ForecastOutcome).count() == 11


def test_the_default_path_never_deletes_a_frozen_row(session, tmp_path, monkeypatch):
    """Test 3 of the brief — the guard against overreach.

    Same scenario without --reresolve. A forecast that stops resolving on the
    daily path keeps its frozen row: the likely cause is a transient archive
    problem, and reacting to a resolution failure by destroying good history
    would be silent and unrecoverable. Deletion is an operator action only.

    THIS TEST DOES NOT CARRY THE GUARANTEE — the unit test below does. It
    cannot: on the daily path a forecast that already has a row is frozen and
    never enters `to_resolve`, so it never reaches the considered set, and a
    mutation that made the default branch delete would leave this test green
    (measured). What this test pins is the end-to-end story — the row is still
    there, with its actuals, after the archive regressed. Keep both; if one has
    to go, keep
    test_the_default_path_does_not_delete_even_when_handed_a_considered_set."""
    archive = _seed_cohort(session, tmp_path)
    _run_backtest(session, archive, monkeypatch)
    before = {
        o.forecast_id: (o.base_price, o.actual_price, o.resolved_at) for o in session.query(ForecastOutcome).all()
    }
    assert len(before) == 12

    _break_ak0(archive)
    _run_backtest(session, archive, monkeypatch)

    session.expire_all()
    after = {o.forecast_id: (o.base_price, o.actual_price, o.resolved_at) for o in session.query(ForecastOutcome).all()}
    assert after == before


def test_the_default_path_does_not_delete_even_when_handed_a_considered_set(session, monkeypatch):
    """The asymmetry asserted directly on _store_forecast_outcomes, so it holds
    whatever the caller passes: with reresolve=False, a considered id whose
    forecast produced no outcome keeps its row.

    THIS IS THE TEST THAT CARRIES THE INSERT-ONLY GUARANTEE. The end-to-end
    version above cannot reach the case — frozen forecasts never re-enter the
    considered set on the daily path — so this is the only test that fails when
    the default branch is mutated to delete. Do not delete it as a duplicate."""
    import scripts.backtest_accuracy as bt

    _stub_parquet_writes(monkeypatch)

    stored = {
        "forecast_id": 7,
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
    assert bt._store_forecast_outcomes(session, [dict(stored)]) == 1

    other = dict(stored, forecast_id=8)
    # 7 was considered and did not resolve; 8 did. Insert-only means 7 survives.
    assert bt._store_forecast_outcomes(session, [other], reresolve=False, considered_ids={7, 8}) == 1

    assert session.query(ForecastOutcome).filter_by(forecast_id=7).one().actual_price == 1.2
    assert session.query(ForecastOutcome).count() == 2


def test_reresolve_replaces_a_still_resolvable_row_without_duplicating_it(session, tmp_path, monkeypatch):
    """Test 4 of the brief. Nothing becomes unresolvable, so a --reresolve —
    and a second one — must leave the row count exactly where it was, in the DB
    and in a real Parquet mirror. Delete-then-insert must not lose or double."""
    import db.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod, "OPS_DIR", tmp_path / "ops")

    archive = _seed_cohort(session, tmp_path)
    _run_with_real_mirror(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 12

    for _ in range(2):
        _run_with_real_mirror(session, archive, monkeypatch, reresolve=True)
        session.expire_all()
        assert session.query(ForecastOutcome).count() == 12
        mirror = parquet_mod.read_table("forecast_outcomes")
        assert len(mirror) == 12
        assert sorted(mirror["forecast_id"]) == list(range(1, 13))


def test_a_considered_forecast_that_never_had_a_row_is_harmless(session, tmp_path, monkeypatch):
    """Test 5 of the brief. A forecast considered for the first time and found
    unresolvable has nothing to delete; the delete must be a no-op, not an
    error, and must not disturb the rest of the cohort."""
    import scripts.backtest_accuracy as bt

    _stub_parquet_writes(monkeypatch)

    # Direct: an id with no stored row at all.
    assert bt._store_forecast_outcomes(session, [], reresolve=True, considered_ids={999}) == 0
    assert session.query(ForecastOutcome).count() == 0

    # End to end: ak0 is broken from the very first run, so --reresolve
    # considers it, resolves nothing for it, and finds no row to remove.
    for i in range(12):
        _seed(
            session, i + 1, f"ak{i}", current_price=3.0, price_mid=3.0, price_low=2.0, price_high=4.0, direction="flat"
        )
    session.commit()
    archive = _write_archive(tmp_path, _cohort_rows(12, broken={"ak0"}))

    _run_backtest(session, archive, monkeypatch, reresolve=True)
    assert session.query(ForecastOutcome).count() == 11
    assert session.query(ForecastOutcome).filter_by(forecast_id=1).first() is None


def test_reresolve_refuses_to_run_without_the_considered_set(session, monkeypatch):
    """The delete set must never be re-derivable from `outcomes` alone. Making
    considered_ids required rather than defaulted is what stops a future caller
    from silently reintroducing the orphan bug."""
    import scripts.backtest_accuracy as bt

    _stub_parquet_writes(monkeypatch)

    with pytest.raises(ValueError, match="considered_ids"):
        bt._store_forecast_outcomes(session, [], reresolve=True)


def test_reresolve_leaves_min_price_filtered_rows_alone(session, tmp_path, monkeypatch):
    """A forecast excluded by --min-price is not resolved-and-failed: the run
    writes no replacement for it, so it is not in the considered set and its
    stored row must survive. Deleting it would make `--reresolve --min-price`
    a silent history-truncation tool."""
    archive = _seed_cohort(session, tmp_path)
    _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 12

    _run_backtest(session, archive, monkeypatch, reresolve=True, min_price=100.0)

    session.expire_all()
    assert session.query(ForecastOutcome).count() == 12


# ---------------------------------------------------------------------------
# Fix round 1, Finding 1: replace_rows must widen the schema and never narrow
# it. It rewrites the WHOLE file, and the backfill is the next thing to run.
#
# The production forecast_outcomes mirror has 18 columns and no base_price; the
# re-resolve write carries 20. Under _append_parquet's column INTERSECTION the
# two extra columns were discarded, so a --reresolve would have rewritten the
# whole 65k-row served file without the frozen actuals — the values the backfill
# exists to produce. Same schema-drift class as the SQLite text()/VARCHAR trap
# earlier in this plan: same file, same blast radius.
# ---------------------------------------------------------------------------


def test_replace_rows_widens_the_schema_instead_of_dropping_new_columns(tmp_path, monkeypatch):
    """A column in the incoming rows that the file lacks must be ADDED, NULL on
    the rows that survive — not silently discarded. This is the case that
    decides whether the production mirror gets base_price at all."""
    import db.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod, "OPS_DIR", tmp_path / "ops")

    parquet_mod.append_table("t", [{"k": 1, "a": 10.0}, {"k": 2, "a": 20.0}], ["k"])
    assert list(parquet_mod.read_table("t").columns) == ["k", "a"]

    parquet_mod.replace_rows("t", "k", [], [{"k": 3, "a": 30.0, "extra": 1.5}])

    out = parquet_mod.read_table("t").set_index("k").sort_index()
    assert "extra" in out.columns
    assert out.loc[3, "extra"] == pytest.approx(1.5)
    # The rows the write did not carry survive, keep their own values, and take
    # NULL for the added column rather than vanishing.
    assert out.loc[1, "a"] == pytest.approx(10.0)
    assert out.loc[2, "a"] == pytest.approx(20.0)
    assert pd.isna(out.loc[1, "extra"]) and pd.isna(out.loc[2, "extra"])
    # File column order is preserved, with the new column appended.
    assert list(parquet_mod.read_table("t").columns) == ["k", "a", "extra"]


def test_replace_rows_refuses_to_narrow_the_file_schema(tmp_path, monkeypatch):
    """A column the file has and the incoming rows lack cannot be preserved on
    the rows being rewritten. Blanking them quietly is how a whole-file rewrite
    loses a column for good, so it raises — and the file on disk is untouched,
    because the rewrite only ever lands via os.replace."""
    import db.parquet as parquet_mod

    monkeypatch.setattr(parquet_mod, "OPS_DIR", tmp_path / "ops")

    parquet_mod.append_table(
        "t",
        [{"k": 1, "a": 10.0, "extra": "keep"}, {"k": 2, "a": 20.0, "extra": "y"}],
        ["k"],
    )

    with pytest.raises(ValueError, match="extra"):
        parquet_mod.replace_rows("t", "k", [2], [{"k": 3, "a": 30.0}])

    out = parquet_mod.read_table("t").set_index("k").sort_index()
    assert list(out.columns) == ["a", "extra"]
    assert sorted(out.index) == [1, 2]
    assert out.loc[1, "extra"] == "keep"
    # No temp file was left behind by the failed rewrite.
    assert list((tmp_path / "ops").glob("*.tmp")) == []

    # Supplying the whole row is accepted, and the untouched row keeps its
    # column and its value.
    parquet_mod.replace_rows("t", "k", [2], [{"k": 3, "a": 30.0, "extra": "new"}])
    out = parquet_mod.read_table("t").set_index("k").sort_index()
    assert sorted(out.index) == [1, 3]
    assert out.loc[1, "extra"] == "keep"
    assert out.loc[3, "extra"] == "new"


def test_reresolve_adds_base_price_to_a_mirror_that_predates_the_column(session, tmp_path, monkeypatch):
    """The production shape, end to end: the served mirror has the pre-freeze
    18 columns and no base_price, while --reresolve writes 20. The rewritten
    file must GAIN base_price with real values, not silently drop it — the
    frozen actuals are the whole point of the backfill and the API serves this
    copy first."""
    import db.parquet as parquet_mod

    real_append = parquet_mod.append_table
    real_replace = parquet_mod.replace_rows

    archive = _seed_cohort(session, tmp_path)
    _run_backtest(session, archive, monkeypatch)  # DB only, writes stubbed

    monkeypatch.setattr(parquet_mod, "OPS_DIR", tmp_path / "ops")
    monkeypatch.setattr(parquet_mod, "append_table", real_append)
    monkeypatch.setattr(parquet_mod, "replace_rows", real_replace)

    # A legacy mirror: every column the freeze added is absent.
    legacy = []
    for o in session.query(ForecastOutcome).all():
        row = _outcome_to_mapping_for_test(o)
        row.pop("base_price")
        row["evaluated_at"] = o.evaluated_at
        legacy.append(row)
    parquet_mod.append_table("forecast_outcomes", legacy, ["forecast_id"])

    before = parquet_mod.read_table("forecast_outcomes")
    assert "base_price" not in before.columns
    assert "resolved_at" not in before.columns
    assert len(before) == 12

    _break_ak0(archive)
    _run_with_real_mirror(session, archive, monkeypatch, reresolve=True)

    after = parquet_mod.read_table("forecast_outcomes")
    assert "base_price" in after.columns
    assert "resolved_at" in after.columns
    assert len(after) == 11
    assert 1 not in set(after["forecast_id"])
    # Real frozen values, not NULLs.
    assert after["base_price"].notna().all()
    assert after["base_price"].round(4).eq(3.00).all()
    assert after["actual_price"].round(4).eq(3.01).all()
    # ...and no pre-existing column was lost in the widening rewrite.
    assert set(before.columns) <= set(after.columns)


def _outcome_to_mapping_for_test(o):
    from scripts import backtest_accuracy

    return backtest_accuracy._outcome_to_mapping(o)


def test_reresolve_min_price_deletes_only_the_unresolvable(session, tmp_path, monkeypatch):
    """Fix round 1, Minor 1 — the real boundary of considered_ids under
    --min-price, in one MIXED cohort.

    Every forecast here resolves below the threshold, but ak0 does not resolve
    at all. A forecast leaves considered_ids only when the run positively
    established it is out of scope, which needs a resolved base price. ak0 has
    none, so it stays in the set and its row goes; the eleven that resolved and
    were then filtered keep theirs, because the run wrote no replacement for
    them. Deleting those would make `--reresolve --min-price` a silent
    history-truncation tool."""
    archive = _seed_cohort(session, tmp_path)
    _run_backtest(session, archive, monkeypatch)
    assert session.query(ForecastOutcome).count() == 12

    _break_ak0(archive)
    # Threshold far above every resolved base price (3.00).
    _run_backtest(session, archive, monkeypatch, reresolve=True, min_price=100.0)

    session.expire_all()
    assert session.query(ForecastOutcome).filter_by(forecast_id=1).first() is None
    assert session.query(ForecastOutcome).count() == 11
    # The filtered-but-resolvable rows were neither deleted nor rewritten.
    assert all(o.actual_price == pytest.approx(3.01) for o in session.query(ForecastOutcome).all())


def test_the_actionable_prediction_leg_reaches_the_scorer(session, tmp_path, monkeypatch):
    """End-to-end, because the fix is inert unless `current_price` is SELECTed
    onto the record.

    The archive resolves "ak" to a base of 2.0 while serving quoted 2.8 — a 40%
    wedge, which is the shape 85% of production rows have. The mid is 2.828, a
    +1% predicted move on the price it was actually quoted from, and 1% cannot
    clear tier 1's 23.1% bar at any venue. Scored against the resolved base the
    same row reads +41% and lands in the subset, which is how 1,120 of 1,141
    production rows got there.
    """
    # h=14 from FORECAST_DATE (07-05) targets 07-19, so the archive has to
    # cover the anchor window and the target window both.
    rows = [("ak", date(2026, 7, d), 2.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 2.0) for d in (17, 18, 19)]
    archive = _write_archive(tmp_path, rows)

    _seed(
        session, 1, "ak", current_price=2.8, price_mid=2.828, price_low=2.5, price_high=3.1, direction="up", horizon=14
    )
    session.commit()

    out = _run_backtest(session, archive, monkeypatch)
    scored = [r for r in out if r["horizon_days"] == 14 and r["sample_count"] == 1]
    assert scored, "the forecast was not scored at all"
    m = scored[0]["metrics"]
    assert m["actionable_scope"] == "in_scope"
    assert m["actionable_n"] == 0, (
        "a +1% forecast was reported as clearing a 23.1% friction bar; r_hat is still dividing by the resolved base"
    )
    assert m["actionable_n_served_basis"] == 1
    assert m["actionable_n_fallback_basis"] == 0


# ---------------------------------------------------------------------------
# `in_interval` is measured on the basis the band was QUOTED from.
#
# `predict()` publishes the band as `current_price x (1 + low_ret/high_ret)`,
# but the actual it is compared against is archive-resolved off `base_price`.
# The two bases disagree on 85% of production rows by a median 5.70% / p90
# 37.82% against half-widths of 10-31%, so the stored predicate was answering
# "did the served dollar band contain the resolved price" while every reader
# took it for coverage of the calibrated band. Same defect the actionable
# prediction leg had, and the same fix: rebase the PREDICTION, leave both legs
# of the outcome on resolve_anchors.
# ---------------------------------------------------------------------------


def test_in_interval_rebases_the_band_onto_the_resolved_base():
    """The wedge alone must not put a covered row outside its own band.

    Serving quoted 12.0 and published a [-10%, +15%] band as [10.8, 13.8]. The
    archive resolves the base to 10.0 and the actual to 10.5 — a +5% move, which
    is inside the band the model actually stated. Compared in dollars it reads as
    a miss, and the entire miss is the 20% wedge between the two bases.
    """
    from scripts.backtest_accuracy import _derive_verdict

    v = _derive_verdict(10.0, 10.5, 12.6, 10.8, 13.8, "up", quote=12.0)
    assert v["in_interval"] == 1


def test_in_interval_rebasing_rejects_as_well_as_admits():
    """The rebase is not a coverage-inflating transform: the same wedge that
    admits a +5% move excludes a +30% one, which the dollar predicate calls a
    hit. A fix that could only ever raise coverage would be indistinguishable
    from widening the band."""
    from scripts.backtest_accuracy import _derive_verdict

    v = _derive_verdict(10.0, 13.0, 12.6, 10.8, 13.8, "up", quote=12.0)
    assert v["in_interval"] == 0
    assert v["in_interval_dollar"] == 1


def test_in_interval_falls_back_to_the_resolved_base_without_a_quote():
    """Legacy outcomes predate `current_price`, and the walkforward harness
    builds its band as `base * (1 + ret)` so the resolved base IS its quote.
    Both must keep scoring, on a predicate that reduces to the old one."""
    from scripts.backtest_accuracy import _derive_verdict

    for quote in (None, 0.0, -1.0):
        v = _derive_verdict(10.0, 10.5, 10.6, 9.0, 11.0, "up", quote=quote)
        assert v["in_interval"] == 1
        assert v["in_interval"] == v["in_interval_dollar"]


def test_a_missing_band_is_still_unknown_not_a_miss():
    """None, not 0. A row with no band was never a coverage observation, and
    counting it as a miss reports under-coverage for a reason that is not the
    calibration."""
    from scripts.backtest_accuracy import _derive_verdict

    v = _derive_verdict(10.0, 10.5, 10.6, None, None, "up", quote=12.0)
    assert v["in_interval"] is None
    assert v["in_interval_dollar"] is None


def test_the_dollar_predicate_is_not_a_stored_column():
    """`in_interval_dollar` is a reporting split, not a verdict: adding it to
    the stored set would need a migration, and `_REFRESH_VERDICTS_SQL` binds
    exactly the columns `_verdict_for_storage` hands it."""
    from scripts.backtest_accuracy import (
        VERDICT_COLUMNS,
        _derive_verdict,
        _verdict_for_storage,
    )

    stored = _verdict_for_storage(_derive_verdict(10.0, 13.0, 12.6, 10.8, 13.8, "up", quote=12.0))
    assert "in_interval_dollar" not in stored
    assert set(stored) == set(VERDICT_COLUMNS)


def test_score_cohort_reports_both_bases_and_which_rows_used_which():
    """Two coverage figures that differ by the wedge, plus the counts that say
    which convention formed each row's band. A payload that cannot say whether
    its rows were scored on the served quote or the resolved fallback is not
    self-describing — the same reasoning `actionable_n_served_basis` follows."""
    from backtest.scoring import score_cohort

    def rec(in_interval, in_interval_dollar, served):
        return {
            "abs_error": 0.1,
            "sq_error": 0.01,
            "pct_error": 1.0,
            "direction_correct": 1,
            "predicted_direction": "up",
            "actual_direction": "up",
            "in_interval": in_interval,
            "in_interval_dollar": in_interval_dollar,
            "interval_basis_served": served,
            "confidence": "low",
            "base_price": 10.0,
            "actual_price": 10.5,
            "price_tier": 1,
            "item_id": 1,
            "forecast_date": date(2026, 7, 5),
        }

    metrics, _ = score_cohort([rec(1, 0, True), rec(1, 1, True), rec(0, 0, False)])
    assert metrics["interval_coverage"] == pytest.approx(66.67, abs=0.01)
    assert metrics["interval_coverage_dollar_basis"] == pytest.approx(33.33, abs=0.01)
    assert metrics["interval_n_served_basis"] == 2
    assert metrics["interval_n_fallback_basis"] == 1


def test_score_cohort_defaults_the_dollar_split_to_the_rebased_predicate():
    """`walkforward_records` builds mid/low/high off the resolved base, so its
    rows have one basis and carry neither key. They must still score, and the
    two figures must agree rather than the dollar one reading 0%."""
    from backtest.scoring import score_cohort

    rec = {
        "abs_error": 0.1,
        "sq_error": 0.01,
        "pct_error": 1.0,
        "direction_correct": 1,
        "predicted_direction": "up",
        "actual_direction": "up",
        "in_interval": 1,
        "confidence": "low",
        "base_price": 10.0,
        "actual_price": 10.5,
        "price_tier": 1,
        "item_id": 1,
        "forecast_date": date(2026, 7, 5),
    }
    metrics, _ = score_cohort([rec])
    assert metrics["interval_coverage"] == 100.0
    assert metrics["interval_coverage_dollar_basis"] == 100.0
    assert metrics["interval_n_served_basis"] == 0
    assert metrics["interval_n_fallback_basis"] == 1


def test_a_wedged_forecast_is_scored_against_the_band_it_was_quoted_from(session, tmp_path, monkeypatch):
    """End to end, through the real resolver: the wedge must not decide coverage.

    The archive resolves "ak" to a base of 2.0 and an actual of 2.1 (+5%).
    Serving quoted 2.8 — a 40% wedge — and published [-10%, +15%] as
    [2.52, 3.22]. The +5% move is inside the band the model stated and outside
    the dollars it printed, and both numbers now reach the payload.
    """
    rows = [("ak", date(2026, 7, d), 2.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 2.1) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)

    _seed(session, 1, "ak", current_price=2.8, price_mid=2.828, price_low=2.52, price_high=3.22, direction="up")
    session.commit()

    out = _run_backtest(session, archive, monkeypatch)
    scored = [r for r in out if r["horizon_days"] == HORIZON and r["sample_count"] == 1]
    assert scored, "the forecast was not scored at all"
    m = scored[0]["metrics"]
    assert m["interval_coverage"] == 100.0, (
        "a +5% move inside a [-10%, +15%] band was scored as a miss; the band "
        "is still being compared in the dollars it was quoted in"
    )
    assert m["interval_coverage_dollar_basis"] == 0.0
    assert m["interval_n_served_basis"] == 1
    assert m["interval_n_fallback_basis"] == 0

    stored = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert stored.in_interval == 1


# ---------------------------------------------------------------------------
# F3: the served identity is not its configuration
# ---------------------------------------------------------------------------


def test_served_identity_collapses_the_serving_config_suffixes():
    """`-regime` and `-global-only` are the same artifact with SKIP_REGIMES
    flipped. Keying the scoring cohort on them forked the date panel three ways
    and no cohort could ever reach MIN_HEADLINE_DATES."""
    from backtest.scoring import served_identity

    assert served_identity("lgbm-v3-regime") == "lgbm-v3"
    assert served_identity("lgbm-v3-global-only") == "lgbm-v3"
    assert served_identity("lgbm-v3") == "lgbm-v3"


def test_served_identity_does_not_merge_genuinely_different_artifacts():
    """The suffix list is an allowlist, not a prefix strip. lgbm-v1 and
    lgbm-catboost-v2 are different models, and the `-ens3`/`-ens6` labels are
    an A/B harness's own series in prediction_accuracy — merging any of them
    would pool cohorts that never shared an artifact."""
    from backtest.scoring import served_identity

    for label in ("lgbm-v1", "lgbm-catboost-v2", "lgbm-v3-ens3", "lgbm-v3-ens6", "lgbm-v4", "lgbm-v3-clustered"):
        assert served_identity(label) == label


def test_served_identity_names_the_missing_label_rather_than_dropping_it():
    """Both grouping sites used `r.model_version or "unknown"`. That mapping
    belongs with the rest of it, so there is one function to read."""
    from backtest.scoring import served_identity

    assert served_identity(None) == "unknown"
    assert served_identity("") == "unknown"


def test_a_merged_cohort_reports_which_configs_it_pooled():
    """Merging is only honest if the payload says what was merged. Distinct
    forecast dates per stored label, not row counts: dates are the unit
    MIN_HEADLINE_DATES counts, and a config that contributed one date to a
    20-date panel is a different claim from one that contributed ten."""
    records = [_record(model_version_raw="lgbm-v3-regime", forecast_date=date(2026, 8, d)) for d in (1, 2, 3)]
    records += [_record(model_version_raw="lgbm-v3-global-only", forecast_date=date(2026, 8, 4))]
    # Two rows, one date -- must count once.
    records += [
        _record(model_version_raw="lgbm-v3", forecast_date=date(2026, 7, 17)),
        _record(model_version_raw="lgbm-v3", forecast_date=date(2026, 7, 17)),
    ]

    metrics, _ = score_cohort(records)

    assert metrics["config_dates"] == {
        "lgbm-v3": 1,
        "lgbm-v3-global-only": 1,
        "lgbm-v3-regime": 3,
    }
    assert metrics["distinct_forecast_dates"] == 5


def test_config_dates_is_absent_rather_than_empty_on_unlabelled_records():
    """Every record predating the field. An empty dict reads as "we pooled
    nothing"; absent reads as "this payload does not say", which is true."""
    metrics, _ = score_cohort([_record(forecast_date=date(2026, 8, 1))])
    assert "config_dates" not in metrics


def test_scoring_merges_two_configs_into_one_cohort(session, tmp_path, monkeypatch):
    """End to end: two forecasts on two dates under the two suffixed labels
    must score as ONE cohort spanning two forecast dates, not two cohorts of
    one date each."""
    from scripts import backtest_accuracy

    rows = [(slug, date(2026, 7, d), 3.0) for slug in ("ak", "ak2") for d in range(1, 9)]
    archive = _write_archive(tmp_path, rows)

    _seed(
        session,
        1,
        "ak",
        current_price=3.0,
        price_mid=3.0,
        price_low=2.0,
        price_high=4.0,
        direction="flat",
        model_version="lgbm-v3-regime",
    )
    _seed(
        session,
        2,
        "ak2",
        current_price=3.0,
        price_mid=3.0,
        price_low=2.0,
        price_high=4.0,
        direction="flat",
        model_version="lgbm-v3-global-only",
        forecast_date=FORECAST_DATE - timedelta(days=1),
    )
    session.commit()
    _run_backtest(session, archive, monkeypatch)

    groups = backtest_accuracy._records_from_frozen_outcomes(session)
    keys = [k for k in groups if k[0] == HORIZON]
    assert keys == [(HORIZON, "lgbm-v3")], f"cohort still forked: {sorted(groups)}"

    metrics, n = score_cohort(groups[keys[0]])
    assert n == 2
    assert metrics["distinct_forecast_dates"] == 2
    assert metrics["config_dates"] == {
        "lgbm-v3-global-only": 1,
        "lgbm-v3-regime": 1,
    }


def test_the_frozen_outcome_inherits_the_canonical_label(session, tmp_path, monkeypatch):
    """The stored outcome must carry the identity, not the config: it is what
    the next run groups on, and a suffixed row would re-fork the panel from
    the outcome side even after the forecast side was fixed."""
    rows = [("ak", date(2026, 7, d), 3.0) for d in (3, 4, 5)]
    rows += [("ak", date(2026, 7, d), 3.3) for d in (6, 7, 8)]
    archive = _write_archive(tmp_path, rows)
    _seed(
        session,
        1,
        "ak",
        current_price=3.0,
        price_mid=3.6,
        price_low=3.0,
        price_high=4.0,
        direction="up",
        model_version="lgbm-v3-regime",
    )
    session.commit()
    _run_backtest(session, archive, monkeypatch)

    stored = session.query(ForecastOutcome).filter_by(forecast_id=1).one()
    assert stored.model_version == "lgbm-v3"


def test_the_headline_line_names_a_pooled_cohort():
    """The disclosure has to appear where the number is read, not only in the
    stored payload."""
    from scripts import backtest_accuracy

    records = [
        _record(model_version_raw="lgbm-v3-regime", forecast_date=date(2026, 8, 1)),
        _record(model_version_raw="lgbm-v3-global-only", forecast_date=date(2026, 8, 2)),
    ]
    metrics, n = score_cohort(records)
    _, msg = backtest_accuracy._headline_line(3, "lgbm-v3", metrics, n)
    assert "pooling lgbm-v3-global-only:1d, lgbm-v3-regime:1d" in msg


def test_the_headline_line_stays_quiet_on_a_single_config():
    """One label is the ordinary case; a "pooling" note there would be noise."""
    from scripts import backtest_accuracy

    records = [_record(model_version_raw="lgbm-v3", forecast_date=date(2026, 8, 1))]
    metrics, n = score_cohort(records)
    _, msg = backtest_accuracy._headline_line(3, "lgbm-v3", metrics, n)
    assert "pooling" not in msg


def _seed_two_dates(session, tmp_path):
    """One forecast on the excluded 2026-07-19, one on the kept 2026-07-17.

    Both are h=3 and both resolve: the excluded date's outcome is still frozen
    into the table, which is the point — the exclusion is a SCORING rule, so the
    row must survive resolution and then not reach a metric.
    """
    _seed(
        session,
        1,
        "kept",
        current_price=3.0,
        price_mid=3.6,
        price_low=3.0,
        price_high=4.0,
        direction="up",
        forecast_date=date(2026, 7, 17),
    )
    _seed(
        session,
        2,
        "dropped",
        current_price=3.0,
        price_mid=3.6,
        price_low=3.0,
        price_high=4.0,
        direction="up",
        forecast_date=date(2026, 7, 19),
    )
    session.commit()
    rows = [(slug, date(2026, 7, d), 3.0) for slug in ("kept", "dropped") for d in range(15, 23)]
    return _write_archive(tmp_path, rows)


def test_the_dead_band_date_is_resolved_but_never_scored(session, tmp_path, monkeypatch):
    """2026-07-19's directions came from a rule no other date shares.

    `served_identity` cannot drop it — those rows are `lgbm-v3-regime`, the
    production daily path at that commit — so the exclusion is a dated one, and
    it belongs on the scoring side only.
    """
    from scripts import backtest_accuracy

    archive = _seed_two_dates(session, tmp_path)
    _run_backtest(session, archive, monkeypatch, today=date(2026, 7, 26))

    # Resolution is untouched: both rows are frozen in the outcome table.
    frozen = session.query(ForecastOutcome).all()
    assert {o.forecast_id for o in frozen} == {1, 2}

    groups = backtest_accuracy._records_from_frozen_outcomes(session)
    # str(), because the driver decides the type: SQLite yields an ISO string
    # here and psycopg2 a date. The exclusion normalises both; the assertion
    # must not depend on which one arrived.
    dates = {str(r["forecast_date"]) for rs in groups.values() for r in rs}
    assert "2026-07-17" in dates
    assert "2026-07-19" not in dates


def test_score_all_dates_restores_the_excluded_row(session, tmp_path, monkeypatch):
    """The escape hatch has to work, or a pre-exclusion figure is unreproducible."""
    from scripts import backtest_accuracy

    archive = _seed_two_dates(session, tmp_path)
    _run_backtest(session, archive, monkeypatch, today=date(2026, 7, 26))

    monkeypatch.setenv("SCORE_ALL_DATES", "1")
    groups = backtest_accuracy._records_from_frozen_outcomes(session)
    dates = {str(r["forecast_date"]) for rs in groups.values() for r in rs}
    assert {"2026-07-17", "2026-07-19"} <= dates
