"""Closed ops months are released once and then act as an immutability check.

The release step must create a release only for a month that can no longer change,
never replace an asset, and fail the run when a released month's files differ.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import ops_releases as rel


def test_month_closes_43_days_after_calendar_end():
    assert rel.closes_on("2026-09") == date(2026, 11, 12)  # the spec's worked example
    assert rel.closes_on("2026-08") == date(2026, 10, 13)
    assert rel.closes_on("2026-02") == date(2026, 4, 12)


def test_sums_round_trip_in_sha256sum_format():
    sums = {"b.parquet": "22", "a.parquet": "11"}
    text = rel.format_sums(sums)
    assert text == "11  a.parquet\n22  b.parquet\n"
    assert rel.parse_sums(text) == sums


@pytest.fixture
def ops(tmp_path):
    for table, month, body in [
        ("item_forecasts", "2026-07", b"fc7"),
        ("forecast_outcomes", "2026-07", b"fo7"),
        ("item_forecasts", "2026-09", b"fc9"),
    ]:
        (tmp_path / table).mkdir(exist_ok=True)
        (tmp_path / table / f"{month}.parquet").write_bytes(body)
    (tmp_path / "item_forecasts.parquet").write_bytes(b"single-file mirror, not a partition")
    return tmp_path


@pytest.fixture
def gh(monkeypatch):
    """A fake release store: {tag: {name: sha}}."""
    store: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(rel, "release_tags", lambda: set(store))
    monkeypatch.setattr(rel, "released_sums", lambda tag: dict(store[tag]))

    def create(tag, files, sums_text):
        assert tag not in store, "an existing release must never be re-created"
        store[tag] = rel.parse_sums(sums_text)
        assert set(store[tag]) == set(files)

    monkeypatch.setattr(rel, "create_release", create)
    return store


def test_releases_only_closed_months(ops, gh):
    assert rel.publish(ops, date(2026, 10, 12), dry_run=False) == 0
    assert set(gh) == {"ops-2026-07"}
    assert set(gh["ops-2026-07"]) == {"item_forecasts-2026-07.parquet", "forecast_outcomes-2026-07.parquet"}
    assert gh["ops-2026-07"]["item_forecasts-2026-07.parquet"] == rel.sha256(ops / "item_forecasts" / "2026-07.parquet")


def test_second_run_verifies_and_creates_nothing(ops, gh):
    rel.publish(ops, date(2026, 10, 12), dry_run=False)
    before = {t: dict(s) for t, s in gh.items()}
    assert rel.publish(ops, date(2026, 10, 12), dry_run=False) == 0
    assert gh == before


@pytest.mark.parametrize(
    "tamper",
    [
        lambda ops: (ops / "item_forecasts" / "2026-07.parquet").write_bytes(b"rewritten"),
        lambda ops: (ops / "forecast_outcomes" / "2026-07.parquet").unlink(),
        lambda ops: (
            (ops / "forecast_candidates").mkdir() or (ops / "forecast_candidates" / "2026-07.parquet").write_bytes(b"x")
        ),
    ],
    ids=["rewritten", "deleted", "added"],
)
def test_a_changed_closed_month_fails_the_run(ops, gh, tamper):
    rel.publish(ops, date(2026, 10, 12), dry_run=False)
    tamper(ops)
    assert rel.publish(ops, date(2026, 10, 12), dry_run=False) == 1


def test_dry_run_creates_nothing(ops, gh):
    assert rel.publish(ops, date(2026, 12, 1), dry_run=True) == 0
    assert gh == {}


def test_pull_verifies_what_it_holds(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    good = dest / "ops-2026-07"
    good.mkdir(parents=True)
    (good / "a.parquet").write_bytes(b"a")
    (good / rel.SUMS).write_text(rel.format_sums({"a.parquet": rel.sha256(good / "a.parquet")}))
    monkeypatch.setattr(rel, "release_tags", lambda: {"ops-2026-07"})
    assert rel.pull(dest) == 0
    (good / "a.parquet").write_bytes(b"bit rot")
    assert rel.pull(dest) == 1
