"""Tests for the historical price-source import."""
from datetime import date, datetime

import duckdb
import pandas as pd
import pytest

from collectors.price_history_sources import cs2_prices_tracker as tracker
from collectors.price_history_import import (
    StalledSourceError,
    detect_stalled_days,
)


def test_source_label_is_exact():
    assert tracker.SOURCE == "tracker_steam_24h"


def test_day_url_uses_the_dated_price_file():
    url = tracker.day_url(date(2025, 6, 15))
    assert url == (
        "https://raw.githubusercontent.com/LukeX404/cs2-prices-tracker/"
        "main/static/prices/date/2025-06-15.json"
    )


def test_parse_day_reads_last_24h_only():
    payload = {
        "AK-47 | Redline (Field-Tested)": {
            "steam": {"last_24h": 12.5, "last_7d": 99.0, "last_30d": 98.0}
        }
    }
    assert tracker.parse_day(payload, date(2025, 6, 15)) == [
        ("AK-47 | Redline (Field-Tested)", date(2025, 6, 15), 12.5)
    ]


def test_parse_day_treats_null_last_24h_as_an_absent_row():
    payload = {
        "Sticker | Sherry": {"steam": {"last_24h": None, "last_7d": 4.0}},
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]


def test_parse_day_drops_non_positive_prices_rather_than_writing_zero():
    payload = {
        "Zero Item": {"steam": {"last_24h": 0.0}},
        "Negative Item": {"steam": {"last_24h": -1.0}},
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]


def test_parse_day_tolerates_a_missing_or_null_steam_object():
    payload = {
        "No Steam Key": {"buff": {"price": 3.0}},
        "Null Steam": {"steam": None},
        "Null Item": None,
        "AK-47 | Redline (Field-Tested)": {"steam": {"last_24h": 12.5}},
    }
    records = tracker.parse_day(payload, date(2025, 6, 15))
    assert [r[0] for r in records] == ["AK-47 | Redline (Field-Tested)"]


def test_detect_stalled_days_finds_consecutive_identical_files():
    digests = {
        date(2026, 7, 26): "aaa",
        date(2026, 7, 27): "bbb",
        date(2026, 7, 28): "bbb",
        date(2026, 7, 29): "bbb",
        date(2026, 7, 30): "ccc",
    }
    assert detect_stalled_days(digests) == [
        [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]
    ]


def test_detect_stalled_days_ignores_identical_files_that_are_not_adjacent():
    digests = {
        date(2025, 6, 1): "aaa",
        date(2025, 6, 2): "bbb",
        date(2025, 6, 3): "aaa",
    }
    assert detect_stalled_days(digests) == []


def test_detect_stalled_days_is_empty_for_all_distinct_files():
    digests = {date(2025, 6, d): f"h{d}" for d in range(1, 6)}
    assert detect_stalled_days(digests) == []


def test_stalled_source_error_is_raisable_with_the_groups():
    with pytest.raises(StalledSourceError):
        raise StalledSourceError([[date(2026, 7, 27), date(2026, 7, 28)]])


from datetime import timedelta

from collectors.price_history_import import (
    MAX_GAP_DAYS,
    MIN_DISTINCT_DAYS,
    apply_gap_gate,
)


def _series(name, days, price=1.0):
    """Records for *name* on the given day offsets from 2025-06-01."""
    return [(name, date(2025, 6, 1) + timedelta(days=d), price) for d in days]


# The gap-specific tests set min_distinct_days=1 so they isolate the gap
# condition; the sparse floor gets its own tests below.
def _gaps_only(records):
    return apply_gap_gate(records, min_distinct_days=1)


def test_max_gap_matches_the_archive_window_constant():
    from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS
    assert MAX_GAP_DAYS == MAX_WINDOW_SPAN_DAYS == 7


def test_min_distinct_days_floor_is_180():
    assert MIN_DISTINCT_DAYS == 180


def test_a_seven_day_gap_is_kept():
    kept, report = _gaps_only(_series("Item A", [0, 7, 14]))
    assert {r[0] for r in kept} == {"Item A"}
    assert report.rejected_gap_items == 0


def test_an_eight_day_gap_is_rejected():
    kept, report = _gaps_only(_series("Item A", [0, 8, 16]))
    assert kept == []
    assert report.rejected_gap_items == 1
    assert report.rejected_rows == 3
    assert report.worst_gap_days == 8


def test_the_gate_is_per_item_not_global():
    records = _series("Dense", [0, 1, 2]) + _series("Sparse", [0, 30])
    kept, report = _gaps_only(records)
    assert {r[0] for r in kept} == {"Dense"}
    assert report.kept_items == 1
    assert report.rejected_gap_items == 1


def test_edges_are_not_penalised_only_interior_gaps_count():
    # Starts late and ends early inside the range; interior spacing is daily.
    kept, _ = _gaps_only(_series("Late Starter", [40, 41, 42, 43]))
    assert {r[0] for r in kept} == {"Late Starter"}


def test_a_dense_item_spanning_180_days_passes_the_default_floor():
    kept, report = apply_gap_gate(_series("Dense", range(180)))
    assert {r[0] for r in kept} == {"Dense"}
    assert report.rejected_sparse_items == 0


def test_179_distinct_days_is_one_short_and_is_rejected():
    kept, report = apply_gap_gate(_series("Nearly", range(179)))
    assert kept == []
    assert report.rejected_sparse_items == 1


def test_two_adjacent_observations_do_not_pass_trivially():
    """Max-gap alone is necessary and NOT sufficient.

    Two consecutive days have a max interior gap of 1, so without the distinct-
    day floor this item sails through and flips is_backfilled on two rows —
    measured at 904 such items (611 at >=$1) on the 90-day block.
    """
    kept, report = apply_gap_gate(_series("Two Days", [0, 1]))
    assert kept == []
    assert report.rejected_sparse_items == 1
    assert report.rejected_gap_items == 0


def test_a_single_observation_item_is_rejected_by_the_sparse_floor():
    kept, report = apply_gap_gate(_series("Lonely", [5]))
    assert kept == []
    assert report.rejected_sparse_items == 1


def test_the_gap_condition_takes_precedence_in_the_report():
    """An item failing BOTH conditions counts once, as a gap rejection."""
    kept, report = apply_gap_gate(_series("Both", [0, 30]))
    assert kept == []
    assert report.rejected_gap_items == 1
    assert report.rejected_sparse_items == 0


def test_records_are_deduplicated_on_item_and_day_keeping_the_first():
    records = [
        ("Item A", date(2025, 6, 1), 10.0),
        ("Item A", date(2025, 6, 1), 99.0),
        ("Item A", date(2025, 6, 2), 11.0),
    ]
    kept, report = _gaps_only(records)
    assert sorted(kept) == [
        ("Item A", date(2025, 6, 1), 10.0),
        ("Item A", date(2025, 6, 2), 11.0),
    ]
    assert report.kept_rows == 2


from db.archive import CANONICAL_PRICE_COLUMNS, prices_relation
from collectors.price_history_import import to_archive_frame, write_archive_frame

_INGESTED = datetime(2026, 8, 8, 12, 0, 0)


def test_archive_frame_has_exactly_the_canonical_columns():
    frame = to_archive_frame(
        [("AK-47 | Redline (Field-Tested)", date(2025, 6, 1), 12.5)],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    assert list(frame.columns) == list(CANONICAL_PRICE_COLUMNS)


def test_archive_frame_leaves_volume_null_never_zero():
    frame = to_archive_frame(
        [("AK-47 | Redline (Field-Tested)", date(2025, 6, 1), 12.5)],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    assert frame["volume"].isna().all()
    assert not (frame["volume"].fillna(-1) == 0).any()


def test_archive_frame_stamps_arrival_not_the_day_it_describes():
    frame = to_archive_frame(
        [("AK-47 | Redline (Field-Tested)", date(2025, 6, 1), 12.5)],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    assert frame["ingested_at"].iloc[0] == pd.Timestamp(_INGESTED)
    assert frame["day"].iloc[0] != frame["ingested_at"].iloc[0]


def test_written_rows_read_back_through_the_typed_reader(tmp_path):
    frame = to_archive_frame(
        [
            ("AK-47 | Redline (Field-Tested)", date(2025, 6, 1), 12.5),
            ("AK-47 | Redline (Field-Tested)", date(2026, 3, 1), 13.5),
        ],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    out_dir = tmp_path / "price-archive"
    assert write_archive_frame(frame, out_dir) == 2

    con = duckdb.connect()
    try:
        rel = prices_relation(
            con,
            archive_dir=out_dir,
            columns=["item_slug", "day", "source", "mean_price", "volume"],
        )
        rows = con.sql(
            f"SELECT item_slug, source, mean_price, volume FROM {rel} ORDER BY day"
        ).fetchall()
    finally:
        con.close()

    assert rows == [
        ("AK-47 | Redline (Field-Tested)", "tracker_steam_24h", 12.5, None),
        ("AK-47 | Redline (Field-Tested)", "tracker_steam_24h", 13.5, None),
    ]


def test_rows_fan_out_to_one_file_per_month(tmp_path):
    frame = to_archive_frame(
        [
            ("Item A", date(2025, 6, 1), 1.0),
            ("Item A", date(2025, 7, 1), 1.0),
        ],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    out_dir = tmp_path / "price-archive"
    write_archive_frame(frame, out_dir)
    names = sorted(p.name for p in out_dir.glob("prices-*.parquet"))
    assert names == ["prices-2025-06.parquet", "prices-2025-07.parquet"]


def test_a_reappend_keeps_the_first_arrival_not_the_latest(tmp_path):
    """append_monthly replaces a colliding row wholesale, ingested_at included.

    Without _preserve_first_arrival a re-run dates every touched row forward,
    which is exactly what the embargo reads. The daily CI writer solves the
    same problem at scripts/append_to_parquet.py:267-270.
    """
    out_dir = tmp_path / "price-archive"
    first = datetime(2026, 8, 8, 12, 0, 0)
    later = datetime(2026, 9, 1, 9, 30, 0)
    records = [("Item A", date(2025, 6, 1), 10.0)]

    write_archive_frame(to_archive_frame(records, "tracker_steam_24h", first), out_dir)
    write_archive_frame(to_archive_frame(records, "tracker_steam_24h", later), out_dir)

    con = duckdb.connect()
    try:
        rel = prices_relation(
            con, archive_dir=out_dir, columns=["item_slug", "ingested_at"]
        )
        stored = con.sql(f"SELECT ingested_at FROM {rel}").fetchall()
    finally:
        con.close()

    assert len(stored) == 1
    assert stored[0][0] == first


def test_a_row_with_no_prior_arrival_takes_the_new_timestamp(tmp_path):
    """min skips NaT: a row predating the column must not stay unknown."""
    out_dir = tmp_path / "price-archive"
    stamped = to_archive_frame(
        [("Item A", date(2025, 6, 1), 10.0)], "tracker_steam_24h", _INGESTED
    )
    unstamped = stamped.copy()
    unstamped["ingested_at"] = pd.NaT
    write_archive_frame(unstamped, out_dir)
    write_archive_frame(stamped, out_dir)

    con = duckdb.connect()
    try:
        rel = prices_relation(
            con, archive_dir=out_dir, columns=["item_slug", "ingested_at"]
        )
        stored = con.sql(f"SELECT ingested_at FROM {rel}").fetchall()
    finally:
        con.close()

    assert stored[0][0] == pd.Timestamp(_INGESTED)


def test_a_reappend_does_not_duplicate_the_same_item_day_source(tmp_path):
    frame = to_archive_frame(
        [("Item A", date(2025, 6, 1), 1.0)],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    out_dir = tmp_path / "price-archive"
    write_archive_frame(frame, out_dir)
    write_archive_frame(frame, out_dir)

    con = duckdb.connect()
    try:
        rel = prices_relation(con, archive_dir=out_dir, columns=["item_slug"])
        count = con.sql(f"SELECT count(*) FROM {rel}").fetchone()[0]
    finally:
        con.close()
    assert count == 1


import json

from scripts.import_price_history_source import (
    cached_path,
    daterange,
    load_cached_day,
)


def test_daterange_is_inclusive_of_both_ends():
    days = daterange(date(2025, 6, 1), date(2025, 6, 4))
    assert days == [
        date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3), date(2025, 6, 4)
    ]


def test_daterange_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        daterange(date(2025, 6, 4), date(2025, 6, 1))


def test_cached_path_is_namespaced_by_source(tmp_path):
    path = cached_path(tmp_path, "cs2_prices_tracker", date(2025, 6, 1))
    assert path == tmp_path / "cs2_prices_tracker" / "2025-06-01.json"


def test_load_cached_day_returns_none_for_a_missing_file(tmp_path):
    assert load_cached_day(tmp_path / "nope.json") is None


def test_load_cached_day_returns_none_for_an_empty_or_corrupt_file(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("")
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert load_cached_day(empty) is None
    assert load_cached_day(corrupt) is None


def test_load_cached_day_parses_a_good_file(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"Item A": {"steam": {"last_24h": 1.0}}}))
    assert load_cached_day(good) == {"Item A": {"steam": {"last_24h": 1.0}}}


from scripts.import_price_history_source import report_promotion_gate


def test_report_promotion_gate_counts_rows_items_and_finds_no_zero_volume(tmp_path):
    staging = tmp_path / "staging" / "price-archive"
    frame = to_archive_frame(
        [
            ("Item A", date(2025, 6, 1), 10.0),
            ("Item A", date(2026, 3, 1), 11.0),
            ("Item B", date(2026, 3, 1), 20.0),
        ],
        source="tracker_steam_24h",
        ingested_at=_INGESTED,
    )
    write_archive_frame(frame, staging)

    archive = tmp_path / "archive" / "price-archive"
    existing = to_archive_frame(
        [("Item A", date(2026, 3, 1), 10.0)],
        source="aggregator_sync",
        ingested_at=_INGESTED,
    )
    write_archive_frame(existing, archive)

    result = report_promotion_gate(staging, archive, "tracker_steam_24h")
    assert result["rows"] == 3
    assert result["items"] == 2
    assert result["pre_2026_items"] == 1
    # Item A has a pre-2026 staged row and the real archive has none for it,
    # so it flips. Item B is 2026-only and flips nothing.
    assert result["new_gate_items"] == 1
    assert result["zero_volume_rows"] == 0
    assert result["duplicate_keys"] == 0
    assert result["overlap_items"] == 1
    assert result["overlap_ratio_median"] == pytest.approx(1.1)


def test_report_promotion_gate_excludes_items_already_inside_the_gate(tmp_path):
    staging = tmp_path / "staging" / "price-archive"
    write_archive_frame(
        to_archive_frame(
            [("Already Gated", date(2025, 6, 1), 10.0)],
            source="tracker_steam_24h",
            ingested_at=_INGESTED,
        ),
        staging,
    )
    archive = tmp_path / "archive" / "price-archive"
    write_archive_frame(
        to_archive_frame(
            [("Already Gated", date(2024, 5, 1), 9.0)],
            source="aggregator_sync",
            ingested_at=_INGESTED,
        ),
        archive,
    )

    result = report_promotion_gate(staging, archive, "tracker_steam_24h")
    assert result["pre_2026_items"] == 1
    assert result["new_gate_items"] == 0
