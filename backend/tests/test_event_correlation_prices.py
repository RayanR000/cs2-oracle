"""Event correlation reads its price windows from Parquet, not Postgres.

`event_correlation_analysis.py` used to read `price_history`, a table that is
empty by design: migration 0008 deleted every `is_backfilled = 1` item's rows,
and `run_analysis` selects precisely those items. `collectors/pipeline.py`
stopped writing the table at the 2026-07-11 CSV->Parquet cutover. So
`_compute_impacts` returned `[]` for every event, `run_analysis` logged "No
price data found ... skipping", and the task wrote zero rows on every run from
2026-07-19 onward while exiting green — it only turned red on 2026-08-02, when
324cfff added the impact counts to `run_task.py`'s `ROW_COUNT_FIELDS`.

The AGENTS.md invariant these tests pin: training and analysis data come from
`price-archive/*.parquet`; the DB supplies only the `is_backfilled` flag and the
events metadata.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pandas as pd
import pytest
import scripts.event_correlation_analysis as eca
from database import (
    Base,
    Event,
    EventCorrelation,
    EventImpact,
    Item,
    PriceHistory,
)
from scripts.event_correlation_analysis import (
    NO_EVENTS_STATUS,
    ItemRef,
    PriceStore,
    _control_change_distribution,
    _post_event_price,
    _pre_event_price,
    _price_on_date,
    run_analysis,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The event fixtures seed events at 2026-05-20. The lookback window must be
# wide enough that this date stays inside it regardless of when the suite runs.
# 365 days is safe through at least 2027-05.
_EVENT_DAYS_BACK = 365


def _store(rows: list[tuple[int, date, float]]) -> PriceStore:
    """Build a PriceStore from (db_item_id, day, price) triples."""
    universe = {item_id: ItemRef(f"slug-{item_id}", "skin") for item_id, _, _ in rows}
    voted = pd.DataFrame(
        [{"item_id": f"slug-{i}", "date": d, "price": p} for i, d, p in rows],
        columns=["item_id", "date", "price"],
    )
    return PriceStore.from_voted(voted, universe)


def _day_numbered_store(item_id: int = 1, month: int = 5) -> PriceStore:
    """One row per day of 2026-05 whose price IS the day of month.

    A distinguishable price per day makes any off-by-one in a window bound
    visible in the mean: shifting a 7-day window by one day moves it by 1.0.
    """
    return _store([(item_id, date(2026, month, d), float(d)) for d in range(1, 32)])


def _write_archive(tmp_path, rows):
    """Write a minimal `prices-YYYY-MM.parquet` set from (slug, day, price)."""
    archive = tmp_path / "price-archive"
    archive.mkdir(exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "item_slug": slug,
                "day": pd.Timestamp(day),
                "mean_price": price,
                "volume": 10,
                "source": "csgotrader",
            }
            for slug, day, price in rows
        ]
    )
    for ym, group in df.groupby(df["day"].dt.strftime("%Y-%m")):
        group.to_parquet(archive / f"prices-{ym}.parquet", index=False)
    return archive


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


@pytest.fixture()
def captured_mirror(monkeypatch):
    """Intercept the event_impacts_denorm append so tests never touch the real
    price-archive/ops tree, and can assert on the rows that would be written."""
    calls: list[dict] = []

    def _fake_append(table, rows, dedup_keys):
        calls.append({"table": table, "rows": rows, "dedup_keys": dedup_keys})

    monkeypatch.setattr(eca, "append_table", _fake_append)
    return calls


# ---------------------------------------------------------------------------
# 1. The price store's window mean
# ---------------------------------------------------------------------------


class TestWindowMean:
    def test_bounds_are_inclusive_on_both_ends(self):
        store = _day_numbered_store()
        # [05-10, 05-12] -> days 10, 11, 12. Excluding either end changes it.
        assert store.window_mean(1, date(2026, 5, 10), date(2026, 5, 12)) == 11.0

    def test_single_day_window_returns_that_day(self):
        store = _day_numbered_store()
        assert store.window_mean(1, date(2026, 5, 7), date(2026, 5, 7)) == 7.0

    def test_empty_window_is_none_not_zero(self):
        """None, never 0.0: a missing window must not read as a free price."""
        store = _store([(1, date(2026, 5, 10), 5.0)])
        assert store.window_mean(1, date(2026, 5, 1), date(2026, 5, 5)) is None

    def test_unknown_item_is_none(self):
        store = _store([(1, date(2026, 5, 10), 5.0)])
        assert store.window_mean(999, date(2026, 5, 1), date(2026, 5, 31)) is None

    def test_inverted_window_is_none(self):
        store = _day_numbered_store()
        assert store.window_mean(1, date(2026, 5, 12), date(2026, 5, 10)) is None

    def test_averages_only_the_days_present(self):
        """One row per item-day, and gaps are gaps — not carried forward."""
        store = _store(
            [
                (1, date(2026, 5, 1), 10.0),
                (1, date(2026, 5, 5), 20.0),
            ]
        )
        assert store.window_mean(1, date(2026, 5, 1), date(2026, 5, 31)) == 15.0

    def test_items_with_prices_counts_covered_items(self):
        store = _store([(1, date(2026, 5, 1), 1.0), (2, date(2026, 5, 1), 2.0)])
        assert store.items_with_prices == 2


# ---------------------------------------------------------------------------
# 2. The documented day ranges
# ---------------------------------------------------------------------------


class TestWindowSemantics:
    def test_pre_event_window_is_the_seven_days_before(self):
        """Docstring says "the 7 days before the event": [d-7, d-1]."""
        store = _day_numbered_store()
        event_day = date(2026, 5, 15)
        # days 8..14 -> mean 11.0. An 8-day window (the old span) gives 10.5,
        # and including the event day itself gives 11.5.
        assert _pre_event_price(store, 1, event_day) == 11.0

    def test_centred_window_is_target_minus_one_to_target_plus_one(self):
        store = _day_numbered_store()
        assert _price_on_date(store, 1, date(2026, 5, 20)) == 20.0
        # The mean equalling the target day is not enough to catch a shift:
        # check the bound days explicitly via a store missing one side.
        one_sided = _store(
            [
                (1, date(2026, 5, 19), 1.0),
                (1, date(2026, 5, 20), 2.0),
                (1, date(2026, 5, 21), 6.0),
                (1, date(2026, 5, 22), 100.0),  # outside the window
            ]
        )
        assert _price_on_date(one_sided, 1, date(2026, 5, 20)) == 3.0

    def test_post_event_offsets_land_on_the_documented_days(self):
        store = _day_numbered_store()
        event_day = date(2026, 5, 10)
        # offset 1 -> target 05-11, window [10, 12] -> 11.0
        assert _post_event_price(store, 1, event_day, 1) == 11.0
        # offset 3 -> target 05-13, window [12, 14] -> 13.0
        assert _post_event_price(store, 1, event_day, 3) == 13.0
        # offset 7 -> target 05-17, window [16, 18] -> 17.0
        assert _post_event_price(store, 1, event_day, 7) == 17.0

    def test_pre_and_post_windows_do_not_overlap_the_event_day(self):
        """The event day itself belongs to neither leg."""
        store = _store([(1, date(2026, 5, 10), 999.0)])  # ONLY the event day
        assert _pre_event_price(store, 1, date(2026, 5, 10)) is None


# ---------------------------------------------------------------------------
# 3. Leave-one-out control statistics
# ---------------------------------------------------------------------------


def _naive_stats(changes: dict[int, float], exclude: int) -> tuple[float, float]:
    """Recompute the control stats from scratch without *exclude*.

    This is a transcription of the pre-Parquet `_control_group_prices` tail:
    population variance (denominator n), a 0.001 std floor when the variance is
    not > 0, and (0.0, 0.0) when there is nothing left.
    """
    vals = [v for k, v in changes.items() if k != exclude]
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(variance) if variance > 0 else 0.001
    return mean, std


class TestLeaveOneOutControls:
    """The closed form must equal a from-scratch recompute for every item.

    `_control_group_prices` used to be called once per item with
    `exclude_item_ids={item_id}` — O(items^2) once the prices are in memory. The
    distribution is now computed once per (event, item_type, offset) and the
    excluding-one mean/std derived from n, sum(x), sum(x^2).
    """

    def _distribution(self, offsets: dict[int, tuple[float, float]], offset_days: int = 7):
        """Build a distribution where item i moves from before -> after."""
        event_day = date(2026, 5, 10)
        rows: list[tuple[int, date, float]] = []
        for item_id, (before, after) in offsets.items():
            for d in range(3, 10):  # inside [05-03, 05-09] = the pre window
                rows.append((item_id, date(2026, 5, d), before))
            target = event_day + timedelta(days=offset_days)
            for delta in (-1, 0, 1):
                rows.append((item_id, target + timedelta(days=delta), after))
        store = _store(rows)
        dist = _control_change_distribution(store, list(offsets), event_day, offset_days)
        return dist

    def test_changes_are_percentage_moves(self):
        dist = self._distribution({1: (10.0, 11.0), 2: (10.0, 8.0)})
        assert dist.changes[1] == pytest.approx(10.0)
        assert dist.changes[2] == pytest.approx(-20.0)

    def test_closed_form_matches_naive_recompute_for_every_item(self):
        dist = self._distribution(
            {
                1: (10.0, 11.0),
                2: (10.0, 8.0),
                3: (4.0, 4.6),
                4: (50.0, 49.0),
                5: (2.0, 2.0),
            }
        )
        for item_id in dist.changes:
            got = dist.excluding(item_id)
            want = _naive_stats(dist.changes, item_id)
            assert got[0] == pytest.approx(want[0], abs=1e-9)
            assert got[1] == pytest.approx(want[1], abs=1e-9)

    def test_excluding_an_item_outside_the_distribution_subtracts_nothing(self):
        """An item lacking either leg was never in the old SQL's set either, so
        there is nothing to subtract for it."""
        dist = self._distribution({1: (10.0, 11.0), 2: (10.0, 8.0)})
        assert dist.excluding(4242) == pytest.approx(_naive_stats(dist.changes, 4242))

    def test_two_items_leaves_one_and_the_std_floor_applies(self):
        """n' = 1: variance is exactly 0, so the std floor of 0.001 kicks in."""
        dist = self._distribution({1: (10.0, 11.0), 2: (10.0, 8.0)})
        mean, std = dist.excluding(1)
        assert mean == pytest.approx(-20.0)
        assert std == 0.001
        assert (mean, std) == pytest.approx(_naive_stats(dist.changes, 1))

    def test_single_item_excluded_yields_zeros(self):
        """n' = 0: the old code returned (0.0, 0.0) with no changes at all."""
        dist = self._distribution({1: (10.0, 11.0)})
        assert dist.excluding(1) == (0.0, 0.0)

    def test_identical_movers_hit_the_std_floor_not_zero(self):
        dist = self._distribution({1: (10.0, 11.0), 2: (20.0, 22.0), 3: (5.0, 5.5)})
        _, std = dist.excluding(1)
        assert std == 0.001

    def test_item_missing_a_leg_is_not_in_the_distribution(self):
        event_day = date(2026, 5, 10)
        rows = [(1, date(2026, 5, d), 10.0) for d in range(3, 10)]
        rows += [(2, date(2026, 5, d), 10.0) for d in range(3, 10)]
        # Only item 1 has a target-window observation.
        rows += [(1, date(2026, 5, 17), 11.0)]
        dist = _control_change_distribution(_store(rows), [1, 2], event_day, 7)
        assert set(dist.changes) == {1}

    def test_no_changes_at_all_yields_zeros(self):
        dist = _control_change_distribution(_store([]), [], date(2026, 5, 10), 7)
        assert dist.excluding(1) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 4. Prices come from the archive, not the DB  (the regression test)
# ---------------------------------------------------------------------------


def _seed_items(session, specs: list[tuple[int, str, str]]):
    for db_id, slug, type_ in specs:
        session.add(Item(id=db_id, item_id=slug, name=slug, type=type_, is_backfilled=1))
    session.commit()


def _seed_event(session, event_id: int, when: datetime, type_: str = "operation"):
    session.add(
        Event(
            id=event_id,
            type=type_,
            timestamp=when,
            description=f"test event {event_id}",
        )
    )
    session.commit()


def _archive_rows(slugs, start: date, days: int, price_of):
    return [(slug, start + timedelta(days=i), price_of(slug, i)) for slug in slugs for i in range(days)]


class TestPricesComeFromTheArchive:
    def test_impacts_are_computed_with_an_empty_price_history_table(self, session, tmp_path, captured_mirror):
        """THE regression test. Postgres `price_history` holds nothing; every
        price window must resolve out of `prices-2026-05.parquet`."""
        slugs = [f"item-{i}" for i in range(1, 6)]
        _seed_items(session, [(i, f"item-{i}", "skin") for i in range(1, 6)])
        _seed_event(session, 1, datetime(2026, 5, 20, 13, 45))

        archive = _write_archive(
            tmp_path,
            _archive_rows(
                slugs,
                date(2026, 5, 1),
                31,
                # Each slug drifts at its own rate, so impacts are non-zero and
                # the control distribution has spread.
                lambda slug, i: 10.0 + i * (0.1 * (int(slug.split("-")[1]))),
            ),
        )

        assert session.query(PriceHistory).count() == 0

        result = run_analysis(days_back=_EVENT_DAYS_BACK, db=session, archive_dir=archive)

        assert result["status"] == "success", result
        assert result["events_analyzed"] == 1
        assert result["impacts_written"] == 5
        assert session.query(EventImpact).count() == 5

        impact = session.query(EventImpact).filter_by(item_id=1).one()
        assert impact.price_day_before is not None
        assert impact.price_day_7 is not None
        assert impact.impact_pct_7day > 0  # every slug drifts upward

    def test_a_missing_archive_raises_rather_than_reading_as_no_prices(self, session, tmp_path):
        """A missing archive checkout is a broken run, not an empty market."""
        _seed_items(session, [(1, "item-1", "skin")])
        _seed_event(session, 1, datetime(2026, 5, 20, 13, 45))

        result = run_analysis(days_back=_EVENT_DAYS_BACK, db=session, archive_dir=tmp_path / "does-not-exist")
        assert result["status"] == "error"
        assert "price archive not found" in result["error"]

    def test_archive_without_the_universe_slugs_is_a_loud_error(self, session, tmp_path):
        """Universe non-empty, events present, zero items priced: a collection
        gap, not "no impacts". The message has to carry enough to tell them
        apart without a second run."""
        _seed_items(session, [(1, "item-1", "skin")])
        _seed_event(session, 1, datetime(2026, 5, 20, 13, 45))
        archive = _write_archive(tmp_path, _archive_rows(["someone-else"], date(2026, 5, 1), 31, lambda slug, i: 10.0))

        result = run_analysis(days_back=_EVENT_DAYS_BACK, db=session, archive_dir=archive)

        assert result["status"] == "error"
        assert "1" in result["error"]  # slugs requested
        assert "2026-05-31" in result["error"]  # archive coverage edge


# ---------------------------------------------------------------------------
# 5. No events in window is an explicit, non-failing outcome
# ---------------------------------------------------------------------------


class TestNoEventsInWindow:
    def test_zero_events_returns_the_distinct_status(self, session, tmp_path):
        """`data/cs2_events.json`'s newest event is 2026-05-10, so at
        days_back=90 the window empties around 2026-08-08. That is a calendar
        fact, not a fault."""
        _seed_items(session, [(1, "item-1", "skin")])
        result = run_analysis(days_back=90, db=session, archive_dir=tmp_path)
        assert result["status"] == NO_EVENTS_STATUS
        assert result["events_analyzed"] == 0

    def test_zero_events_does_not_need_an_archive(self, session, tmp_path):
        """It returns before loading prices, so an absent archive is not
        reported as the reason."""
        _seed_items(session, [(1, "item-1", "skin")])
        result = run_analysis(days_back=90, db=session, archive_dir=tmp_path / "nope")
        assert result["status"] == NO_EVENTS_STATUS

    def test_run_task_treats_the_status_as_exit_zero_and_says_so(self, caplog):
        """Exit 0, but never silently: an unrecognised status also happens to
        fall through every guard, and "the run said nothing and exited 0" is the
        shape that hid this task's zero rows for two weeks."""
        from scripts.run_task import check_results

        with caplog.at_level("WARNING"):
            check_results(
                "event_correlation",
                (
                    {
                        "status": NO_EVENTS_STATUS,
                        "events_analyzed": 0,
                        "impacts_written": 0,
                        "patterns_written": 0,
                        "correlations_written": 0,
                    },
                ),
            )  # must not raise

        assert any(NO_EVENTS_STATUS in record.getMessage() for record in caplog.records), caplog.text
        assert "nothing to do" in caplog.text

    def test_run_task_still_fails_on_events_with_zero_impacts(self):
        """The bug being fixed: events exist, nothing written. Stays fatal."""
        from scripts.run_task import check_results

        with pytest.raises(SystemExit) as exc:
            check_results(
                "event_correlation",
                (
                    {
                        "status": "success",
                        "events_analyzed": 4,
                        "impacts_written": 0,
                        "patterns_written": 0,
                        "correlations_written": 0,
                    },
                ),
            )
        assert exc.value.code == 1

    def test_run_task_still_fails_on_error_status(self):
        from scripts.run_task import check_results

        with pytest.raises(SystemExit):
            check_results("event_correlation", ({"status": "error", "error": "x"},))

    def test_run_task_passes_a_normal_success(self):
        from scripts.run_task import check_results

        check_results(
            "event_correlation",
            (
                {
                    "status": "success",
                    "events_analyzed": 2,
                    "impacts_written": 10,
                    "patterns_written": 10,
                    "correlations_written": 10,
                },
            ),
        )


# ---------------------------------------------------------------------------
# 6. The denorm mirror carries confidence_score for EVERY item
# ---------------------------------------------------------------------------


class TestDenormMirror:
    def test_every_row_carries_its_own_confidence_score(self, session, tmp_path, captured_mirror):
        """The old code appended the rows with `confidence_score: None`, then
        read the whole file back and patched it from
        `{... for r in [data]}` — `data` being whatever the correlations loop
        variable happened to hold last. One item's score, every other row left
        NULL, and the whole file rewritten by hand."""
        slugs = [f"item-{i}" for i in range(1, 4)]
        _seed_items(session, [(i, f"item-{i}", "skin") for i in range(1, 4)])
        _seed_event(session, 1, datetime(2026, 5, 20, 13, 45))
        archive = _write_archive(
            tmp_path,
            _archive_rows(slugs, date(2026, 5, 1), 31, lambda slug, i: 10.0 + i * 0.2 * int(slug.split("-")[1])),
        )

        result = run_analysis(days_back=_EVENT_DAYS_BACK, db=session, archive_dir=archive)
        assert result["status"] == "success", result

        appends = [c for c in captured_mirror if c["table"] == "event_impacts_denorm"]
        assert len(appends) == 1, "one append per event, not one per pass"
        rows = appends[0]["rows"]
        assert appends[0]["dedup_keys"] == ["event_id", "item_id"]
        assert len(rows) == 3

        from_db = {c.item_id: c.confidence_score for c in session.query(EventCorrelation).all()}
        assert len(from_db) == 3
        for row in rows:
            assert row["confidence_score"] is not None
            assert row["confidence_score"] == from_db[row["item_id"]]

    def test_correlations_see_the_patterns_written_in_the_same_run(self, session, tmp_path, captured_mirror):
        """The pattern rows are now preloaded in ONE query instead of one per
        item, which only works because `_compute_and_upsert_patterns` commits
        before the correlations pass. A `pattern_consistency_score` of None on
        every row is what a preload done too early would look like."""
        _seed_items(session, [(1, "item-1", "skin"), (2, "item-2", "skin")])
        _seed_event(session, 1, datetime(2026, 5, 20, 13, 45))
        archive = _write_archive(
            tmp_path,
            _archive_rows(
                ["item-1", "item-2"], date(2026, 5, 1), 31, lambda slug, i: 10.0 + i * 0.3 * int(slug.split("-")[1])
            ),
        )

        run_analysis(days_back=_EVENT_DAYS_BACK, db=session, archive_dir=archive)

        rows = session.query(EventCorrelation).all()
        assert len(rows) == 2
        for row in rows:
            assert row.pattern_consistency_score is not None
            assert row.pattern_passed == 1
            # 0.10 control + 0.20 pattern + 0.10 confounding + 0.15 lag; the
            # 0.25 significance term and the 0.20 holdout term are data/sample
            # dependent (a single event gives no holdout split).
            assert row.confidence_score >= 0.55

    def test_no_nested_values_reach_append_table(self, session, tmp_path, captured_mirror):
        """AGENTS.md: never hand raw dicts to `append_table`. These rows are all
        scalars; keep them that way."""
        _seed_items(session, [(1, "item-1", "skin"), (2, "item-2", "skin")])
        _seed_event(session, 1, datetime(2026, 5, 20, 13, 45))
        archive = _write_archive(
            tmp_path,
            _archive_rows(["item-1", "item-2"], date(2026, 5, 1), 31, lambda slug, i: 10.0 + i * 0.3),
        )

        run_analysis(days_back=_EVENT_DAYS_BACK, db=session, archive_dir=archive)

        for call in captured_mirror:
            for row in call["rows"]:
                for key, value in row.items():
                    assert not isinstance(value, (dict, list, set)), key
