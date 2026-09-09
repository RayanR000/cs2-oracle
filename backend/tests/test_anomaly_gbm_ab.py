"""The anomaly A/B's featureless null.

`item_rate_predictions` is the arm the GBM has to beat, so a bug that makes it
artificially weak turns a decorative head into an apparent win. Two ways that
can happen silently — reading val labels, and letting a three-row item post a
rate of 0.0 or 1.0 — each get a test.
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.anomaly_gbm_ab import item_rate_predictions  # noqa: E402

TARGET = "target_anomaly_7d"


def _df(pairs):
    return pd.DataFrame({"item_id": [i for i, _ in pairs],
                         TARGET: [v for _, v in pairs]}, dtype=object).astype(
        {TARGET: float})


class TestItemRatePredictions:
    def test_a_well_observed_item_gets_its_own_train_rate(self):
        train = _df([("A", 1.0)] * 5 + [("A", 0.0)] * 5 + [("B", 0.0)] * 20)
        val = pd.DataFrame({"item_id": ["A", "B"]})
        p, pooled = item_rate_predictions(train, val, TARGET, min_obs=10)
        assert p[0] == pytest.approx(0.5)
        assert p[1] == pytest.approx(0.0)
        assert pooled == pytest.approx(5 / 30)

    def test_a_thinly_observed_item_falls_back_to_pooled(self):
        # Three rows all positive would otherwise post a rate of 1.0 and
        # dominate the log loss on nothing but noise.
        train = _df([("A", 1.0)] * 3 + [("B", 0.0)] * 20 + [("B", 1.0)] * 5)
        val = pd.DataFrame({"item_id": ["A"]})
        p, pooled = item_rate_predictions(train, val, TARGET, min_obs=10)
        assert p[0] == pytest.approx(pooled)
        assert p[0] != pytest.approx(1.0)

    def test_an_item_absent_from_train_falls_back_to_pooled(self):
        train = _df([("A", 1.0)] * 10 + [("A", 0.0)] * 10)
        val = pd.DataFrame({"item_id": ["ZZZ"]})
        p, pooled = item_rate_predictions(train, val, TARGET, min_obs=10)
        assert p[0] == pytest.approx(pooled) == pytest.approx(0.5)

    def test_val_labels_are_never_read(self):
        # The null must be computable at serve time. If val labels leaked in,
        # flipping every one of them would move the predictions.
        train = _df([("A", 1.0)] * 6 + [("A", 0.0)] * 6)
        val_a = pd.DataFrame({"item_id": ["A"] * 4, TARGET: [1.0, 1.0, 1.0, 1.0]})
        val_b = pd.DataFrame({"item_id": ["A"] * 4, TARGET: [0.0, 0.0, 0.0, 0.0]})
        pa, _ = item_rate_predictions(train, val_a, TARGET, min_obs=5)
        pb, _ = item_rate_predictions(train, val_b, TARGET, min_obs=5)
        assert np.array_equal(pa, pb)

    def test_output_length_and_order_match_val_rows(self):
        train = _df([("A", 1.0)] * 10 + [("B", 0.0)] * 10)
        val = pd.DataFrame({"item_id": ["B", "A", "B"]})
        p, _ = item_rate_predictions(train, val, TARGET, min_obs=10)
        assert len(p) == 3
        assert p[0] == pytest.approx(0.0)
        assert p[1] == pytest.approx(1.0)

    def test_predictions_are_finite_probabilities(self):
        train = _df([("A", 1.0)] * 12 + [("B", 0.0)] * 12)
        val = pd.DataFrame({"item_id": ["A", "B", "C"]})
        p, _ = item_rate_predictions(train, val, TARGET, min_obs=10)
        assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))


class TestCleanAnomalyLabel:
    """The strictly-prior threshold must not see the future.

    The whole point of the rebuild is that production's threshold does. If this
    one leaks too, the re-run confirms nothing.
    """

    @staticmethod
    def _frame(n=40, item="A"):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "item_id": [item] * n,
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "return_7d": rng.normal(0, 5, n),
            "target_return_7d": rng.normal(0, 5, n),
        })

    def test_a_later_row_cannot_change_an_earlier_label(self):
        from scripts.anomaly_gbm_ab import clean_anomaly_label
        base = self._frame()
        bumped = base.copy()
        bumped.loc[bumped.index[-1], "return_7d"] = 400.0
        a = clean_anomaly_label(base, 7)
        b = clean_anomaly_label(bumped, 7)
        # Every label before the tampered row must be untouched.
        pd.testing.assert_series_equal(a.iloc[:-1], b.iloc[:-1])

    def test_the_row_s_own_return_is_excluded_from_its_threshold(self):
        from scripts.anomaly_gbm_ab import clean_anomaly_label
        base = self._frame()
        bumped = base.copy()
        i = 30
        bumped.loc[bumped.index[i], "return_7d"] = 400.0
        assert clean_anomaly_label(base, 7).iloc[i] == \
            clean_anomaly_label(bumped, 7).iloc[i]

    def test_rows_without_enough_history_are_nan_not_false(self):
        from scripts.anomaly_gbm_ab import clean_anomaly_label
        out = clean_anomaly_label(self._frame(), 7, min_periods=10)
        # shift(1) + min_periods=10 means the first 10 rows cannot form one.
        assert out.iloc[:10].isna().all()
        assert out.iloc[15:].notna().any()

    def test_a_void_target_stays_void(self):
        from scripts.anomaly_gbm_ab import clean_anomaly_label
        df = self._frame()
        df.loc[df.index[20], "target_return_7d"] = np.nan
        assert np.isnan(clean_anomaly_label(df, 7).iloc[20])

    def test_thresholds_do_not_bleed_across_items(self):
        from scripts.anomaly_gbm_ab import clean_anomaly_label
        a = self._frame(item="A")
        b = self._frame(item="B")
        b["return_7d"] = b["return_7d"] * 50
        both = pd.concat([a, b], ignore_index=True)
        alone = clean_anomaly_label(a, 7)
        joint = clean_anomaly_label(both, 7).iloc[:len(a)]
        pd.testing.assert_series_equal(alone, joint, check_names=False)

    def test_a_missing_return_column_is_a_hard_error(self):
        from scripts.anomaly_gbm_ab import clean_anomaly_label
        df = self._frame().drop(columns=["return_7d"])
        with pytest.raises(SystemExit, match="return_7d"):
            clean_anomaly_label(df, 7)
