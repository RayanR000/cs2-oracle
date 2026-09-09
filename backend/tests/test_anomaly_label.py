"""`target_anomaly_{h}d` in production's `prepare_targets`.

The anomaly head has shipped in the API schema as `anomaly_p` since before it
was ever measured, and until 2026-09-09 nothing tested the label it trains on.
The property that matters is that the 2σ threshold is knowable at the row's own
date — the original definition normalised by trailing `target_return_{h}d`,
each of which resolves h days later. See
`docs/changelog/2026-09-09-anomaly-head-beats-its-null.md`.
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.forecaster import ItemForecaster  # noqa: E402

COL = "target_anomaly_7d"


@pytest.fixture
def forecaster():
    return ItemForecaster(db_session=None)


def _frame(n=120, seed=0, item="A"):
    """A random-walk price with its own backward 7-day return.

    `return_7d` has to be the real backward return of `price`, not independent
    noise: the threshold is 2x its rolling std and the label compares it against
    the FORWARD return of the same series, so an unrelated `return_7d` sets a
    threshold on a different scale and the label degenerates to all-zero.
    """
    rng = np.random.default_rng(seed)
    price = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    ret7 = np.full(n, np.nan)
    ret7[7:] = (price[7:] - price[:-7]) / price[:-7] * 100
    return pd.DataFrame({
        "item_id": [item] * n,
        "date": [d.date() for d in pd.date_range("2025-01-01", periods=n, freq="D")],
        "price": price,
        "return_7d": ret7,
    })


class TestGate:
    def test_no_label_without_the_flag(self, forecaster, monkeypatch):
        monkeypatch.delenv("ANOMALY_GBM", raising=False)
        out = forecaster.prepare_targets(_frame(), 7)
        assert COL not in out.columns

    def test_label_appears_with_the_flag(self, forecaster, monkeypatch):
        monkeypatch.setenv("ANOMALY_GBM", "1")
        out = forecaster.prepare_targets(_frame(), 7)
        assert COL in out.columns
        assert set(out[COL].dropna().unique()) <= {0.0, 1.0}


class TestThresholdIsKnowableAtTheRowDate:
    def test_a_later_row_cannot_change_an_earlier_label(self, forecaster,
                                                        monkeypatch):
        # The regression this guards: the threshold used to be built from
        # trailing `target_return_{h}d`, which resolves h days AFTER its row,
        # so information from the prediction window reached the label.
        monkeypatch.setenv("ANOMALY_GBM", "1")
        base = _frame()
        tampered = base.copy()
        tampered.loc[tampered.index[-1], "return_7d"] = 400.0
        a = forecaster.prepare_targets(base, 7)[COL].reset_index(drop=True)
        b = forecaster.prepare_targets(tampered, 7)[COL].reset_index(drop=True)
        pd.testing.assert_series_equal(a.iloc[:-1], b.iloc[:-1])

    def test_the_threshold_ignores_the_row_s_own_return(self, forecaster,
                                                        monkeypatch):
        monkeypatch.setenv("ANOMALY_GBM", "1")
        base = _frame()
        tampered = base.copy()
        i = 50
        tampered.loc[tampered.index[i], "return_7d"] = 400.0
        a = forecaster.prepare_targets(base, 7)[COL].reset_index(drop=True)
        b = forecaster.prepare_targets(tampered, 7)[COL].reset_index(drop=True)
        assert a.iloc[i] == b.iloc[i] or (np.isnan(a.iloc[i])
                                          and np.isnan(b.iloc[i]))


class TestDegenerateInputs:
    def test_a_missing_backward_return_yields_no_label_rather_than_a_leak(
            self, forecaster, monkeypatch):
        # Falling back to the old forward-overlapping threshold here would be
        # worse than emitting nothing: it would be silent.
        monkeypatch.setenv("ANOMALY_GBM", "1")
        out = forecaster.prepare_targets(_frame().drop(columns=["return_7d"]), 7)
        assert COL not in out.columns

    def test_early_rows_without_enough_history_are_nan(self, forecaster,
                                                       monkeypatch):
        monkeypatch.setenv("ANOMALY_GBM", "1")
        out = forecaster.prepare_targets(_frame(), 7).sort_values("date")
        assert out[COL].iloc[:10].isna().all()

    def test_labels_are_not_all_one_class(self, forecaster, monkeypatch):
        # A threshold accidentally built from a near-zero std would fire on
        # every row; one built from an enormous one would never fire.
        monkeypatch.setenv("ANOMALY_GBM", "1")
        out = forecaster.prepare_targets(_frame(n=400, seed=7), 7)
        rate = out[COL].mean()
        assert 0.0 < rate < 0.5
