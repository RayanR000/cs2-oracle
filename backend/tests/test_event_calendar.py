"""Tests for the two date-level exogenous ingests.

Both are pure once the payload is in hand — no network here. The failure mode
these guard against is not a crash but a *plausible wrong number*: a date-level
feature that quietly reads forward, a trailing window off by a day, or a dead
source forward-filled into a constant that a model would happily split on. The
RUB case in `ingest_fx_history` is the second of those bugs found in this repo's
data layer, after the supply scraper's 16 green days storing nothing, so the
lookahead and fill-cap invariants are asserted directly rather than inferred.
"""

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from scripts.ingest_fx_history import (
    MAX_FILL_DAYS,
    build_frame,
)
from scripts.ingest_fx_history import (
    OUTPUT_COLUMNS as FX_COLUMNS,
)
from scripts.ingest_steam_news import (
    OUTPUT_COLUMNS as CAL_COLUMNS,
)
from scripts.ingest_steam_news import (
    build_calendar,
    crate_events,
    news_events,
    normalise_date,
)


def _ts(y, m, d) -> int:
    return int(datetime(y, m, d, 12, 0, tzinfo=UTC).timestamp())


def _news(y, m, d, title="Counter-Strike 2 Update", feed_type=1, gid=None):
    return {
        "gid": gid or f"{y}{m}{d}{title}{feed_type}",
        "date": _ts(y, m, d),
        "title": title,
        "feed_type": feed_type,
        "contents": "",
    }


# ---------------------------------------------------------------- news events


def test_valve_and_press_are_counted_separately():
    df = news_events(
        [
            _news(2024, 1, 10, feed_type=1),
            _news(2024, 1, 10, title="CS2 news roundup", feed_type=0),
            _news(2024, 1, 10, title="Another outlet writes it up", feed_type=0),
        ]
    )
    row = df[df["day"] == date(2024, 1, 10)].iloc[0]
    assert row["valve_announcements"] == 1
    assert row["press_articles"] == 2


def test_cross_posted_announcements_are_deduped_on_day_and_title():
    """Steam serves one announcement under several gids.

    Counting those separately would inflate the announcement count on exactly
    the days an announcement happened, which is every day the column matters.
    """
    df = news_events(
        [
            _news(2024, 1, 10, gid="a"),
            _news(2024, 1, 10, gid="b"),
            _news(2024, 1, 10, gid="c"),
        ]
    )
    assert df.iloc[0]["valve_announcements"] == 1


def test_items_without_a_usable_timestamp_are_dropped_not_zero_dated():
    df = news_events([_news(2024, 1, 10), {"gid": "x", "title": "t", "feed_type": 1}])
    assert list(df["day"]) == [date(2024, 1, 10)]


# --------------------------------------------------------------- crate events


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-01-16", date(2024, 1, 16)),
        ("2013/12/17", date(2013, 12, 17)),
        ("2014/5/2", date(2014, 5, 2)),
        ("  2014-12-05  ", date(2014, 12, 5)),
    ],
)
def test_normalise_date_handles_every_shape_in_crates_json(raw, expected):
    assert normalise_date(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "not a date", "2024-13-45", 20240116])
def test_normalise_date_returns_none_rather_than_guessing(raw):
    assert normalise_date(raw) is None


def test_crate_events_splits_cases_from_capsules_and_ignores_undated():
    df = crate_events(
        {
            "crates": [
                {"type": "Case", "first_sale_date": "2024-01-16"},
                {"type": "Sticker Capsule", "first_sale_date": "2024-01-16"},
                {"type": "Autograph Capsule", "first_sale_date": "2024-01-16"},
                {"type": "Souvenir", "first_sale_date": None},
                {"type": "Case", "first_sale_date": "garbage"},
            ],
            "collections": [{"release_date": "2024-01-16"}],
        }
    ).set_index("day")
    row = df.loc[date(2024, 1, 16)]
    assert row["crate_case_first_sales"] == 1
    assert row["crate_capsule_first_sales"] == 2
    assert row["collection_releases"] == 1


# ------------------------------------------------------------ calendar shape


def _calendar(news_items, crates=None, start=date(2024, 1, 1), end=date(2024, 3, 1)):
    crates = crates or {"crates": [], "collections": []}
    return build_calendar(news_events(news_items), crate_events(crates), start=start, end=end)


def test_calendar_is_dense_and_holds_the_declared_schema():
    cal = _calendar([_news(2024, 1, 10)])
    assert list(cal.columns) == list(CAL_COLUMNS)
    assert len(cal) == 61
    assert cal["day"].is_monotonic_increasing
    assert cal["day"].is_unique


def test_event_free_days_are_zero_not_null():
    cal = _calendar([_news(2024, 1, 10)]).set_index("day")
    assert cal.loc[date(2024, 1, 11), "valve_announcements"] == 0


# ------------------------------------------------------- causality invariants


def test_trailing_window_includes_the_event_day_itself():
    cal = _calendar([_news(2024, 2, 1)]).set_index("day")
    assert cal.loc[date(2024, 2, 1), "valve_announcements_7d"] == 1


def test_trailing_window_expires_after_its_span():
    cal = _calendar([_news(2024, 2, 1)]).set_index("day")
    assert cal.loc[date(2024, 2, 7), "valve_announcements_7d"] == 1
    assert cal.loc[date(2024, 2, 8), "valve_announcements_7d"] == 0
    assert cal.loc[date(2024, 2, 8), "valve_announcements_30d"] == 1


def test_days_since_is_zero_on_the_event_day_and_counts_up_after():
    cal = _calendar([_news(2024, 2, 1)]).set_index("day")
    assert cal.loc[date(2024, 2, 1), "days_since_valve_announcement"] == 0
    assert cal.loc[date(2024, 2, 5), "days_since_valve_announcement"] == 4


def test_days_since_is_null_before_the_first_event_not_zero():
    """Zero-filling here would assert an event on the calendar's first day."""
    cal = _calendar([_news(2024, 2, 1)]).set_index("day")
    assert pd.isna(cal.loc[date(2024, 1, 15), "days_since_valve_announcement"])


def test_truncating_the_future_changes_nothing():
    """The no-lookahead invariant, asserted on every column at once.

    A date-level table is where lookahead is easiest to introduce and hardest to
    see: a centred window or a back-fill would still produce a plausible series.
    Building the calendar over a short span must give byte-identical rows to
    building it over a long span and slicing, or something reads forward.
    """
    news = [_news(2024, 1, 10), _news(2024, 2, 1), _news(2024, 2, 20), _news(2024, 2, 25, title="Press", feed_type=0)]
    crates = {
        "crates": [
            {"type": "Case", "first_sale_date": "2024-01-20"},
            {"type": "Case", "first_sale_date": "2024-02-22"},
            {"type": "Sticker Capsule", "first_sale_date": "2024-02-05"},
        ],
        "collections": [{"release_date": "2024-02-10"}],
    }
    cutoff = date(2024, 2, 15)
    short = _calendar(news, crates, end=cutoff)
    long = _calendar(news, crates, end=date(2024, 3, 1))
    long_head = long[long["day"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(short, long_head)


def test_no_column_looks_forward_by_name():
    """A `days_until_*` column would be lookahead by construction."""
    assert not [c for c in CAL_COLUMNS if "until" in c or "next" in c]


# ------------------------------------------------------------------------ FX


def _rates(days: dict[str, dict]) -> dict:
    return days


def test_fx_frame_holds_the_declared_schema_and_is_long_format():
    df = build_frame({"2024-01-01": {"CNY": 7.1, "EUR": 0.9}}, date(2024, 1, 1), date(2024, 1, 1))
    assert list(df.columns) == list(FX_COLUMNS)
    assert set(df["currency"]) == {"CNY", "EUR"}


def test_weekend_gaps_are_forward_filled_and_flagged():
    df = build_frame(
        {"2024-01-05": {"CNY": 7.0}, "2024-01-08": {"CNY": 7.2}}, date(2024, 1, 5), date(2024, 1, 8)
    ).set_index("day")
    assert df.loc[date(2024, 1, 6), "rate"] == 7.0
    assert df.loc[date(2024, 1, 6), "is_filled"] == 1
    assert df.loc[date(2024, 1, 5), "is_filled"] == 0
    assert df.loc[date(2024, 1, 8), "rate"] == 7.2


def test_fill_stops_after_the_cap_so_a_dead_series_does_not_become_a_constant():
    """The RUB trap: the ECB stopped quoting it on 2022-03-01.

    Uncapped, one frozen number was carried across the next four years — 2,621 of
    RUB's 4,966 days — which reads to a model as a perfectly stable feature rather
    than as missing data.
    """
    df = build_frame({"2024-01-01": {"RUB": 90.0}}, date(2024, 1, 1), date(2024, 3, 1))
    assert len(df) == MAX_FILL_DAYS + 1
    assert df["day"].max() == date(2024, 1, 1 + MAX_FILL_DAYS)


def test_days_before_the_first_quote_are_dropped_not_back_filled():
    """Back-filling would put a future rate on a past day."""
    df = build_frame({"2024-01-10": {"CNY": 7.0}}, date(2024, 1, 1), date(2024, 1, 10))
    assert df["day"].min() == date(2024, 1, 10)


def test_currencies_with_different_end_dates_are_independent():
    df = build_frame(
        {"2024-01-01": {"CNY": 7.0, "RUB": 90.0}, "2024-01-02": {"CNY": 7.1}}, date(2024, 1, 1), date(2024, 1, 20)
    )
    cny = df[df["currency"] == "CNY"]
    rub = df[df["currency"] == "RUB"]
    assert cny["day"].max() == date(2024, 1, 2 + MAX_FILL_DAYS)
    assert rub["day"].max() == date(2024, 1, 1 + MAX_FILL_DAYS)


def test_empty_payload_yields_an_empty_frame_with_the_right_columns():
    df = build_frame({}, date(2024, 1, 1), date(2024, 1, 5))
    assert list(df.columns) == list(FX_COLUMNS)
    assert df.empty
