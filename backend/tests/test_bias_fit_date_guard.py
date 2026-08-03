"""The outcome-fitted bias loop must see date clustering, not just row counts.

update_bias_corrections_from_outcomes matches the predicted up/down/flat split
to the observed base rate. Every item sharing a forecast_date is exposed to the
same market-wide move, so 11,000 rows on two opposite-direction days describe
those two days, not the market. Row-count guards cannot detect this: the real
cohort is 11,000 rows and 2 dates.
"""
from __future__ import annotations

import inspect
import json
import re
from datetime import date
from unittest.mock import MagicMock

import pytest

from backtest.scoring import MIN_FORECAST_DATES
from models.forecaster import ItemForecaster


def _dates(n_distinct, rows_each=500):
    """rows_each rows on each of n_distinct consecutive days."""
    out = []
    for i in range(n_distinct):
        out.extend([date(2026, 1, 1 + i)] * rows_each)
    return out


def test_two_dates_lack_coverage_regardless_of_row_count():
    # The real cohort shape: five-figure row count, two market days.
    assert ItemForecaster._has_date_coverage(_dates(2, rows_each=5500)) is False


def test_coverage_is_met_at_the_minimum():
    assert ItemForecaster._has_date_coverage(_dates(MIN_FORECAST_DATES, 1)) is True


def test_one_below_the_minimum_lacks_coverage():
    assert ItemForecaster._has_date_coverage(_dates(MIN_FORECAST_DATES - 1, 1)) is False


def test_null_dates_never_count_toward_coverage():
    # Rows predating the forecast_date backfill must not manufacture coverage.
    dates = _dates(MIN_FORECAST_DATES, 1) + [None] * 5000
    assert ItemForecaster._has_date_coverage(dates) is True
    assert ItemForecaster._has_date_coverage([None] * 5000) is False


def test_empty_input_lacks_coverage():
    assert ItemForecaster._has_date_coverage([]) is False


def test_forecaster_does_not_define_its_own_min_forecast_dates():
    """A second, drifting threshold is the failure this guards against.

    scoring.MIN_FORECAST_DATES gates what gets *reported*; the guard gates
    what gets *fitted*. If forecaster.py ever grew its own
    `MIN_FORECAST_DATES = ...` instead of importing scoring's, the two could
    drift independently of each other — production could fit thresholds on a
    cohort the same codebase refuses to quote.

    An `is`/`==` check on `forecaster.MIN_FORECAST_DATES` can't catch that:
    CPython caches small integers, so a module that independently defines
    `MIN_FORECAST_DATES = 20` would still pass an identity or value check
    against `scoring.MIN_FORECAST_DATES`. Instead, inspect forecaster.py's own
    source for a local assignment to the name -- there must be none; the only
    way the name can exist in that module is via the `from backtest.scoring
    import MIN_FORECAST_DATES` at the top of the file.
    """
    from models import forecaster as fc

    source = inspect.getsource(fc)
    own_assignments = [
        line for line in source.splitlines()
        if re.match(r"^MIN_FORECAST_DATES\s*=", line.strip())
    ]
    assert own_assignments == [], (
        "forecaster.py must not define its own MIN_FORECAST_DATES -- found: "
        f"{own_assignments!r}"
    )


DEFAULT_T = 0.5  # DIRECTION_FLAT_TOLERANCE_PCT


def _write_corrections(model_dir, payload):
    (model_dir / "bias_corrections.json").write_text(json.dumps(payload))


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "saved_models"
    d.mkdir()
    return d


def _load(model_dir):
    # ItemForecaster only hydrates persisted state (models, bias thresholds,
    # ...) via load_models(); __init__ leaves it empty. load_models() itself
    # requires a full meta.json + model files, which these tests don't
    # provide, so exercise the bias-corrections load directly -- this is the
    # same call load_models() makes internally.
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(model_dir))
    f._load_bias_corrections()
    return f


def test_unversioned_thresholds_are_discarded(model_dir):
    """The real production file: rail-clamped values, no provenance."""
    _write_corrections(model_dir, {
        "corrections": {},
        "thresholds": {"30": {"$1-5": {"t_down": -3.0, "t_up": -2.92}}},
        "ewma_state": {"30": {"$1-5": 2}},
    })
    f = _load(model_dir)
    assert f.bias_thresholds[30]["$1-5"] == {"t_down": -DEFAULT_T, "t_up": DEFAULT_T}
    assert f.bias_ewma_state.get(30, {}).get("$1-5", 0) == 0


def test_versioned_thresholds_are_kept(model_dir):
    _write_corrections(model_dir, {
        "schema_version": ItemForecaster.BIAS_FIT_SCHEMA_VERSION,
        "corrections": {},
        "thresholds": {"30": {"$1-5": {"t_down": -0.4, "t_up": 0.6}}},
        "ewma_state": {"30": {"$1-5": 2}},
    })
    f = _load(model_dir)
    assert f.bias_thresholds[30]["$1-5"] == {"t_down": -0.4, "t_up": 0.6}
    assert f.bias_ewma_state[30]["$1-5"] == 2


def test_save_stamps_the_schema_version(model_dir):
    f = _load(model_dir)
    f._save_bias_corrections()
    data = json.loads((model_dir / "bias_corrections.json").read_text())
    assert data["schema_version"] == ItemForecaster.BIAS_FIT_SCHEMA_VERSION


def test_a_discarded_load_survives_a_save_round_trip(model_dir):
    """Discard then save must not write the rails back out."""
    _write_corrections(model_dir, {
        "corrections": {},
        "thresholds": {"30": {"$1-5": {"t_down": -3.0, "t_up": -2.92}}},
        "ewma_state": {"30": {"$1-5": 2}},
    })
    f = _load(model_dir)
    f._save_bias_corrections()
    reloaded = _load(model_dir)
    assert reloaded.bias_thresholds[30]["$1-5"] == {"t_down": -DEFAULT_T, "t_up": DEFAULT_T}


@pytest.mark.parametrize("bad_version", [None, "abc", [1, 2], {"nested": True}], ids=[
    "null", "non_numeric_string", "list", "dict",
])
def test_malformed_schema_version_discards_thresholds_without_crashing(model_dir, bad_version):
    """A malformed schema_version (not an int and not int-able) must not raise.

    It is treated the same as an unversioned (v0) file -- stored thresholds
    are discarded to defaults -- rather than falling into the corrupt-file
    except clause, which would also wipe the still-valid `corrections` dict
    parsed just above the version check. Assert on `corrections` surviving to
    prove which branch was taken, not merely that nothing raised.
    """
    _write_corrections(model_dir, {
        "schema_version": bad_version,
        "corrections": {"7": {"$1-5": 1.5}},
        "thresholds": {"30": {"$1-5": {"t_down": -3.0, "t_up": -2.92}}},
        "ewma_state": {"30": {"$1-5": 2}},
    })
    f = _load(model_dir)  # must not raise
    # Discard-to-defaults branch: thresholds reset to the flat-tolerance default.
    assert f.bias_thresholds[30]["$1-5"] == {"t_down": -DEFAULT_T, "t_up": DEFAULT_T}
    # Not the corrupt-file branch: _set_default_bias_corrections() would have
    # reset `corrections` to `{}` for every horizon; instead the parsed value
    # from the file survives untouched.
    assert f.bias_corrections[7]["$1-5"] == 1.5
