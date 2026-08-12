"""2026-07-19 is out of the scored panel, because it is a different estimator.

That date's directions came from a global ±0.5% dead band on `mid_ret`, which
`f71ffb4` made live 42 minutes before the run and which no other date in the
panel shares — it called `flat` on 64.0% of the ≥$1 cross-section against a
23.1% realised flat rate. `served_identity` deliberately merges `-regime` and
`-global-only`, so it cannot separate a direction-rule change; only an explicit
dated exclusion can.

See `docs/changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.
"""

from datetime import date, datetime

import pytest

from backtest.scoring import (
    EXCLUDED_FORECAST_DATES,
    excluded_forecast_date,
    served_identity,
)


def test_the_dead_band_date_is_excluded_with_a_reason():
    reason = excluded_forecast_date(date(2026, 7, 19))
    assert reason is not None
    # The reason is published beside the count, so it has to name the mechanism
    # rather than just asserting the exclusion.
    assert "dead-band" in reason


@pytest.mark.parametrize("d", [
    date(2026, 7, 17),
    date(2026, 7, 18),
    date(2026, 7, 29),
    date(2025, 12, 1),
    date(2026, 8, 7),
])
def test_every_other_panel_date_is_kept(d):
    assert excluded_forecast_date(d) is None


def test_a_datetime_is_normalised_to_its_date():
    """Outcome rows arrive as date or datetime depending on the driver."""
    assert excluded_forecast_date(datetime(2026, 7, 19, 23, 39, 12)) is not None
    assert excluded_forecast_date(datetime(2026, 7, 20, 0, 0, 0)) is None


@pytest.mark.parametrize("s, excluded", [
    ("2026-07-19", True),
    ("2026-07-19 23:39:12", True),
    ("2026-07-19T23:39:12", True),
    ("2026-07-18", False),
])
def test_an_iso_string_is_normalised(s, excluded):
    """SQLite hands the column back as a string, psycopg2 as a date.

    Matching only `date` made the exclusion silently never fire under SQLite —
    it dropped nothing and reported nothing, which is the worst of both.
    """
    assert (excluded_forecast_date(s) is not None) is excluded


def test_a_missing_or_unparseable_date_is_not_excluded():
    """An unknown date is unknown, not excluded — it must not vanish silently."""
    assert excluded_forecast_date(None) is None
    assert excluded_forecast_date("not a date") is None
    assert excluded_forecast_date(object()) is None


def test_the_exclusion_is_not_reachable_through_served_identity():
    """The two mechanisms are independent, and this is why the exclusion exists.

    07-19's rows are `lgbm-v3-regime`, which is the PRODUCTION daily path at that
    commit — run A writes `-regime`, and `-global-only` is `--compare-regime`'s
    run B overwriting it. So the cohort key cannot be what drops this date.
    """
    assert served_identity("lgbm-v3-regime") == "lgbm-v3"
    assert served_identity("lgbm-v3") == "lgbm-v3"


def test_the_exclusion_list_stays_small_and_deliberate():
    """A growing list means dates are being dropped to move a number."""
    assert len(EXCLUDED_FORECAST_DATES) == 1
    assert all(isinstance(d, date) for d in EXCLUDED_FORECAST_DATES)
    assert all(r and isinstance(r, str) for r in EXCLUDED_FORECAST_DATES.values())
