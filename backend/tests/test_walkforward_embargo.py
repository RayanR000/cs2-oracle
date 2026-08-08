"""The walkforward A/B harnesses must embargo the train side, like production.

The `ab_test_*` walkforward family built its folds as `train = every date <
val_start` with no purge gap. A row dated `d` is labelled with the price at
`d + horizon`, so every training row in the last `horizon` days before the
boundary carried a target resolved from *inside* the validation window.
Production never had this bug: `_compute_cv_splits(..., purge_days=horizon)`
and `_purge_overlapping_train_rows` both purge, and
`ab_test_direction_labels.py` / `ab_test_recency_weights.py` call the former.
The walkforward family had simply diverged.

The overlap is symmetric across arms, which is why it went unnoticed -- but only
an arm that can *locate* the overlapping rows exploits it, and only a date-level
or date-proxy column can. Measured 2026-08-07 on an event-calendar arm at h=30
(25 folds, held-out >=$1 cohort, fold-clustered): +12.1pp unpurged -> +6.1pp
purged.

Two invariants are locked down here:
  1. the embargo removes exactly the `horizon`-day band before `val_start`, so
     every surviving train row's label lands strictly before the window opens;
  2. the VALIDATION window is untouched. Purging the val side would shrink the
     21-day window and, at horizon=30, empty it outright.
"""
from __future__ import annotations

import inspect
import re
from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

import scripts.ab_test_csfloat_basis as cf
import scripts.ab_test_item_metadata as meta
import scripts.ab_test_training_breadth as breadth
import scripts.walkforward_backtest as wf
from models.forecaster import ItemForecaster

AB_HARNESSES = pytest.mark.parametrize(
    "module", [cf, meta, breadth],
    ids=["csfloat_basis", "item_metadata", "training_breadth"],
)

HORIZONS = [3, 7, 14, 30]


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(
        db_session=MagicMock(),
        model_dir=str(tmp_path_factory.mktemp("saved_models")),
    )


def _daily_frame(start="2024-01-01", days=400, items=(1, 2)):
    """One row per item per calendar day -- the geometry the harnesses assume."""
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame([
        {"item_id": i, "date": d.date(), "price": 10.0}
        for i in items for d in dates
    ])


class TestTheEmbargoRemovesTheRightRows:
    """Semantics, at the harnesses' own fold geometry (21-day val window)."""

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_train_stops_exactly_a_horizon_band_before_the_window(
            self, forecaster, horizon):
        df = _daily_frame()
        dates = sorted(df["date"].unique())
        window_end = len(dates) * 2 // 3 + 1
        val_dates = dates[window_end:window_end + cf.VAL_WINDOW_DAYS]
        val_start = val_dates[0]

        unpurged = df[df["date"] <= dates[window_end - 1]]
        purged = forecaster._purge_overlapping_train_rows(
            unpurged, val_start, horizon)

        # Exactly the `horizon` calendar days before val_start are gone.
        dropped_days = (set(pd.to_datetime(unpurged["date"]).dt.date)
                        - set(pd.to_datetime(purged["date"]).dt.date))
        assert len(dropped_days) == horizon
        assert max(dropped_days) == val_start - timedelta(days=1)
        assert min(dropped_days) == val_start - timedelta(days=horizon)

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_every_surviving_label_lands_before_the_window_opens(
            self, forecaster, horizon):
        df = _daily_frame()
        dates = sorted(df["date"].unique())
        window_end = len(dates) * 2 // 3 + 1
        val_start = dates[window_end]

        unpurged = df[df["date"] <= dates[window_end - 1]]
        leaking = [d for d in unpurged["date"]
                   if d + timedelta(days=horizon) >= val_start]
        assert len(leaking) > 0, "precondition: the un-purged split leaks"

        purged = forecaster._purge_overlapping_train_rows(
            unpurged, val_start, horizon)
        assert not purged.empty
        assert all(d + timedelta(days=horizon) < val_start
                   for d in purged["date"])

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_the_validation_window_width_is_unchanged(self, forecaster, horizon):
        """The purge is train-side only. At h=30 a val-side purge would empty a
        21-day window entirely; the width must not move at any horizon."""
        df = _daily_frame()
        dates = sorted(df["date"].unique())
        window_end = len(dates) * 2 // 3 + 1
        val_dates = dates[window_end:window_end + cf.VAL_WINDOW_DAYS]
        val_df = df[df["date"].isin(val_dates)]

        forecaster._purge_overlapping_train_rows(
            df[df["date"] <= dates[window_end - 1]], val_dates[0], horizon)

        assert len(val_dates) == cf.VAL_WINDOW_DAYS == 21
        assert val_df["date"].nunique() == 21
        assert len(df[df["date"].isin(val_dates)]) == len(val_df)

    def test_the_purge_is_a_no_op_on_a_gap_wider_than_the_horizon(
            self, forecaster):
        """Nothing is dropped when the train side already stops early enough."""
        df = _daily_frame(days=60)
        val_start = df["date"].max() + timedelta(days=45)
        purged = forecaster._purge_overlapping_train_rows(df, val_start, 30)
        assert len(purged) == len(df)


@AB_HARNESSES
class TestTheAbHarnessesAreWired:

    def test_the_harness_calls_productions_purge(self, module):
        src = inspect.getsource(module)
        assert "_purge_overlapping_train_rows(" in src, (
            f"{module.__name__} must reuse "
            "ItemForecaster._purge_overlapping_train_rows rather than "
            "re-deriving the embargo -- a second implementation of the same "
            "rule is how this drifted from production in the first place"
        )

    def test_no_unpurged_train_frame_survives(self, module):
        """The exact defect: `train_df = sub[<train mask> & <item mask>]`."""
        src = inspect.getsource(module)
        offenders = re.findall(
            r"^\s*train_df\s*=\s*sub\[[^\]]*\]\s*$", src, flags=re.M)
        assert not offenders, (
            f"{module.__name__} still builds an un-embargoed training frame: "
            f"{offenders}"
        )

    def test_the_purge_is_unconditional(self, module):
        """These are research tools: there is no un-purged mode to fall back to."""
        src = inspect.getsource(module)
        for token in ("--no-purge", "--purge-days", "purge=False", "if purge"):
            assert token not in src, (
                f"{module.__name__} exposes {token!r}; the embargo must be the "
                "only behaviour here"
            )

    def test_the_val_side_is_never_purged(self, module):
        src = inspect.getsource(module)
        for line in src.splitlines():
            if "_purge_overlapping_train_rows" in line:
                assert "val_df" not in line, (
                    f"{module.__name__} purges the validation frame; at h=30 "
                    "that empties the 21-day window"
                )
        # The val mask is still built straight off the window bounds.
        assert re.search(r"sub_days\s*>=\s*dates_dt\[window_end\]", src)


class TestTheProductionGateKeepsItsDefault:
    """`walkforward_backtest` is the published Backtest Accuracy gate. The
    embargo is available but must not move the reported series by default."""

    def test_purge_is_a_parameter(self):
        assert "purge" in inspect.signature(wf.run_walkforward).parameters

    def test_purge_defaults_to_off(self):
        assert (inspect.signature(wf.run_walkforward)
                .parameters["purge"].default is False)

    def test_the_cli_exposes_an_opt_in_flag(self):
        src = inspect.getsource(wf.main)
        assert '"--purge"' in src and 'action="store_true"' in src

    def test_the_gate_uses_productions_purge_helper(self):
        src = inspect.getsource(wf.run_walkforward)
        assert "_purge_overlapping_train_rows(" in src
        # Train side only -- val_df is built before the purge and not touched.
        purge_lines = [l for l in src.splitlines()
                       if "_purge_overlapping_train_rows" in l]
        assert purge_lines and all("val_df" not in l for l in purge_lines)

    def test_the_report_records_which_way_it_ran(self):
        """Two runs of the gate differ only by this flag; the report has to say."""
        assert '"purge": bool(purge)' in inspect.getsource(wf.run_walkforward)
