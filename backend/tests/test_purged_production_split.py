"""The production train/val split must purge label-overlapping training rows.

CV has always purged (`_compute_cv_splits(..., purge_days=horizon)`), but the
production split at `_train_horizon_inline` did not. A row dated `d` carries the
price at `d + horizon` as its label, so every row in
`[split_date - horizon, split_date)` is labelled from inside the validation
window. At horizon=30 with VALIDATION_WINDOW_DAYS=30 that is the *entire*
window's worth of future prices sitting in the training frame.

That frame is not a bystander: it is the `dval` early stopping stops on, the set
Optuna scores every trial against, and the classifier's stopping set. So the
shipped tree count, the shipped hyperparameters and the classifier's stopping
point were all selected against partly-seen labels.
"""

import pandas as pd
import pytest
from datetime import timedelta
from unittest.mock import MagicMock

from models.forecaster import ItemForecaster


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(
        db_session=MagicMock(),
        model_dir=str(tmp_path_factory.mktemp("saved_models")),
    )


def _frame(start="2026-01-01", days=120):
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({
        "item_id": 1,
        "date": [d.date() for d in dates],
        "price": 10.0,
    })


class TestPurgeOverlappingTrainRows:

    def test_rows_labelled_from_the_validation_window_are_dropped(self, forecaster):
        df = _frame()
        split = pd.Timestamp("2026-04-01")
        train = df[pd.to_datetime(df["date"]) < split]

        purged = forecaster._purge_overlapping_train_rows(train, split, horizon=30)

        assert not purged.empty
        latest = pd.to_datetime(purged["date"]).max()
        # Every surviving row's label date must land strictly before the
        # validation window opens.
        assert latest + timedelta(days=30) < split

    @pytest.mark.parametrize("horizon", [3, 7, 14, 30])
    def test_purge_band_is_exactly_horizon_days(self, forecaster, horizon):
        df = _frame()
        split = pd.Timestamp("2026-04-01")
        train = df[pd.to_datetime(df["date"]) < split]

        purged = forecaster._purge_overlapping_train_rows(train, split, horizon)

        dropped = len(train) - len(purged)
        assert dropped == horizon

    def test_at_30d_the_whole_validation_window_would_otherwise_leak(self, forecaster):
        """The specific defect: VALIDATION_WINDOW_DAYS == 30 == horizon."""
        df = _frame()
        max_date = pd.to_datetime(df["date"].max())
        split = max_date - timedelta(days=ItemForecaster.VALIDATION_WINDOW_DAYS)
        train = df[pd.to_datetime(df["date"]) < split]

        leaking = [
            d for d in pd.to_datetime(train["date"])
            if d + timedelta(days=30) >= split
        ]
        assert len(leaking) == 30, "precondition: unpurged split leaks 30 rows"

        purged = forecaster._purge_overlapping_train_rows(train, split, horizon=30)
        still_leaking = [
            d for d in pd.to_datetime(purged["date"])
            if d + timedelta(days=30) >= split
        ]
        assert still_leaking == []

    def test_empty_frame_is_returned_unchanged(self, forecaster):
        empty = pd.DataFrame(columns=["item_id", "date", "price"])
        out = forecaster._purge_overlapping_train_rows(
            empty, pd.Timestamp("2026-04-01"), horizon=7)
        assert out.empty

    def test_frame_without_a_date_column_is_returned_unchanged(self, forecaster):
        no_date = pd.DataFrame({"item_id": [1, 2], "price": [1.0, 2.0]})
        out = forecaster._purge_overlapping_train_rows(
            no_date, pd.Timestamp("2026-04-01"), horizon=7)
        assert len(out) == 2

    def test_a_horizon_wider_than_the_frame_purges_everything(self, forecaster):
        df = _frame(days=10)
        split = pd.Timestamp("2026-01-11")
        train = df[pd.to_datetime(df["date"]) < split]
        out = forecaster._purge_overlapping_train_rows(train, split, horizon=365)
        assert out.empty
