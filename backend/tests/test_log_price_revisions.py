"""log_price_revisions: the archive keeps no git history, so revisions are logged as rows.

The log must name changed, removed and late-added rows, stay silent about a routine new day,
treat a NULL source as a real key, be idempotent per run date, and collapse to aggregates on
a whole-archive rewrite.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.log_price_revisions import log_revisions

RUN = dt.date(2026, 9, 30)


def _write(directory, rows, name="prices-2026-09.parquet"):
    directory.mkdir(parents=True, exist_ok=True)
    cols = list(zip(*rows, strict=True)) if rows else [(), (), (), (), ()]
    table = pa.table(
        {
            "item_slug": pa.array(cols[0], type=pa.string()),
            "day": pa.array(cols[1], type=pa.date32()),
            "source": pa.array(cols[2], type=pa.string()),
            "mean_price": pa.array(cols[3], type=pa.float64()),
            "volume": pa.array(cols[4], type=pa.int64()),
        }
    )
    pq.write_table(table, directory / name)


def _log(archive):
    return pq.read_table(archive / "ops" / "price_revisions.parquet").to_pylist()


D1, D2, D3 = dt.date(2026, 9, 28), dt.date(2026, 9, 29), dt.date(2026, 9, 30)


def test_changed_removed_and_late_added(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    _write(before, [("a", D1, "s", 1.0, 5), ("b", D1, "s", 2.0, 5), ("c", D2, "s", 3.0, 5)])
    _write(after, [("a", D1, "s", 1.5, 5), ("c", D2, "s", 3.0, 5), ("d", D1, "s", 4.0, 1), ("e", D3, "s", 9.0, 1)])

    report = log_revisions(before, after, RUN)

    rows = {(r["item_slug"], r["kind"]): r for r in _log(after)}
    assert set(rows) == {("a", "changed"), ("b", "removed"), ("d", "late_added")}  # e is a routine new day
    assert (rows["a", "changed"]["prev_mean_price"], rows["a", "changed"]["new_mean_price"]) == (1.0, 1.5)
    assert rows["b", "removed"]["prev_mean_price"] == 2.0
    assert rows["d", "late_added"]["new_mean_price"] == 4.0
    assert report["revisions"] == 3


def test_unchanged_archive_writes_nothing(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    same = [("a", D1, "s", 1.0, 5), ("b", D2, None, 2.0, 5)]
    _write(before, same)
    _write(after, [*same, ("a", D3, "s", 1.0, 5)])

    assert log_revisions(before, after, RUN)["wrote"] is False
    assert not (after / "ops" / "price_revisions.parquet").exists()


def test_null_source_is_a_key_and_a_null_price_is_a_revision(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    _write(before, [("a", D1, None, 1.0, 5), ("a", D1, "s", None, 5)])
    _write(after, [("a", D1, None, 1.0, 5), ("a", D1, "s", 2.0, 5)])

    log_revisions(before, after, RUN)

    assert [(r["source"], r["kind"]) for r in _log(after)] == [("s", "changed")]


def test_volume_only_change_is_logged(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    _write(before, [("a", D1, "s", 1.0, 5)])
    _write(after, [("a", D1, "s", 1.0, 7)])

    log_revisions(before, after, RUN)

    (row,) = _log(after)
    assert (row["prev_volume"], row["new_volume"]) == (5, 7)


def test_window_excludes_older_days(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    old = dt.date(2026, 9, 1)
    _write(before, [("a", old, "s", 1.0, 5), ("b", D1, "s", 1.0, 5)])
    _write(after, [("a", old, "s", 9.0, 5), ("b", D1, "s", 9.0, 5)])

    log_revisions(before, after, RUN, window_days=14)
    assert [r["item_slug"] for r in _log(after)] == ["b"]

    log_revisions(before, after, RUN, window_days=None)  # 'all' replaces this run date's rows
    assert sorted(r["item_slug"] for r in _log(after)) == ["a", "b"]


def test_rerun_same_date_is_idempotent_and_other_dates_accumulate(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    _write(before, [("a", D1, "s", 1.0, 5)])
    _write(after, [("a", D1, "s", 2.0, 5)])

    log_revisions(before, after, RUN)
    log_revisions(before, after, RUN)
    assert len(_log(after)) == 1

    log_revisions(before, after, dt.date(2026, 10, 1))
    assert sorted(r["run_date"] for r in _log(after)) == [RUN, dt.date(2026, 10, 1)]


def test_window_spanning_a_month_boundary_reads_both_files(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    aug = dt.date(2026, 8, 28)
    _write(before, [("a", aug, "s", 1.0, 5)], "prices-2026-08.parquet")
    _write(after, [("a", aug, "s", 2.0, 5)], "prices-2026-08.parquet")
    _write(before, [("b", dt.date(2026, 9, 2), "s", 1.0, 5)])
    _write(after, [("b", dt.date(2026, 9, 2), "s", 1.0, 5)])

    log_revisions(before, after, dt.date(2026, 9, 5), window_days=14)

    assert [r["item_slug"] for r in _log(after)] == ["a"]


def test_whole_archive_rewrite_collapses_to_aggregates(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    _write(before, [(f"i{n}", D1, "s", 1.0, 5) for n in range(10)])
    _write(after, [(f"i{n}", D1, "s", 2.0, 5) for n in range(10)])

    report = log_revisions(before, after, RUN, max_detail_rows=5)

    (row,) = _log(after)
    assert (row["item_slug"], row["n_items"], row["kind"], row["day"]) == (None, 10, "changed", D1)
    assert report["aggregated"] is True


def test_missing_before_files_logs_nothing_and_does_not_crash(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    _write(after, [("a", D1, "s", 1.0, 5)])

    assert log_revisions(before, after, RUN)["wrote"] is False


def test_workflow_snapshots_before_the_writers_and_logs_before_the_publish():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / ".github/workflows/aggregator-update.yml").read_text()
    snapshot = text.index("- name: Snapshot prices before writes")
    append = text.index("- name: Append to Parquet archive")
    log = text.index("- name: Log price revisions")
    publish = text.index("- name: Publish updated archive")
    assert snapshot < append < log < publish
    assert "continue-on-error: true" in text[log:publish], "the audit log must never fail the daily chain"
