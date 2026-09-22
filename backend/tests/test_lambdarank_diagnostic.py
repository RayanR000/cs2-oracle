"""C2 lambdarank diagnostic — the pure label/group builder and the flag.

The trap-prone unit is `_lambdarank_labels`: LightGBM ranking needs graded
integer relevance plus a `group` vector of contiguous per-query row counts, and
the relevance must be a WITHIN-date rank bucket (a global bucketing would
re-import the market factor as relevance — see the design spec). These tests pin
that construction; the trainer and CV wiring are exercised separately.
"""

import numpy as np
import pandas as pd
from models.forecaster import ItemForecaster

K = 8


class TestLambdarankLabels:
    def test_group_sizes_match_date_counts_and_sum_to_len(self):
        # Rows already sorted so equal dates are contiguous (the trainer sorts).
        dates = np.array(["2026-01-01"] * 3 + ["2026-01-02"] * 5)
        y = np.arange(8, dtype=float)
        rel, group = ItemForecaster._lambdarank_labels(dates, y, K)
        assert group == [3, 5]
        assert sum(group) == len(y) == len(rel)

    def test_relevance_is_monotone_in_return_within_date(self):
        dates = np.array(["2026-01-01"] * 4 + ["2026-01-02"] * 4)
        # Within each date the higher forward return must get >= relevance.
        y = np.array([0.1, 0.2, 0.3, 0.4, 9.0, 1.0, 5.0, 3.0])
        rel, _ = ItemForecaster._lambdarank_labels(dates, y, K)
        g1, g2 = rel[:4], rel[4:]
        assert list(g1) == sorted(g1)  # y already ascending in group 1
        # group 2: order by y is 1.0<3.0<5.0<9.0 → indices 1,3,2,0
        assert g2[0] >= g2[1] and g2[2] >= g2[3] and g2[0] >= g2[2]

    def test_relevance_bounded_by_k(self):
        dates = np.array(["2026-01-01"] * 50)
        y = np.random.default_rng(0).normal(size=50)
        rel, _ = ItemForecaster._lambdarank_labels(dates, y, K)
        assert rel.min() >= 0
        assert rel.max() <= K - 1
        assert rel.dtype.kind in "iu"

    def test_ties_do_not_crash_and_stay_valid(self):
        dates = np.array(["2026-01-01"] * 6)
        y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        rel, group = ItemForecaster._lambdarank_labels(dates, y, K)
        assert group == [6]
        assert rel.min() >= 0 and rel.max() <= K - 1
        # Equal returns get equal relevance.
        assert len(set(rel[:3])) == 1 and len(set(rel[3:])) == 1
        assert rel[3] >= rel[0]

    def test_degenerate_all_equal_date_collapses_to_one_level(self):
        dates = np.array(["2026-01-01"] * 4)
        y = np.array([2.5, 2.5, 2.5, 2.5])
        rel, group = ItemForecaster._lambdarank_labels(dates, y, K)
        assert group == [4]
        assert len(set(rel)) == 1  # nothing to rank → one relevance level

    def test_single_row_date(self):
        dates = np.array(["2026-01-01", "2026-01-02", "2026-01-02"])
        y = np.array([5.0, 1.0, 2.0])
        rel, group = ItemForecaster._lambdarank_labels(dates, y, K)
        assert group == [1, 2]
        assert len(rel) == 3


class TestLambdarankTrainer:
    """The non-pure seam: does the ranker train and return val-aligned scores?"""

    def _forecaster(self, feature_cols):
        # Bypass __init__ (DB/session); the trainer only needs feature_cols and
        # the class-level constants/classmethods.
        fc = ItemForecaster.__new__(ItemForecaster)
        fc.feature_cols = feature_cols
        return fc

    def _frame(self, dates, feats, rng):
        n = len(dates)
        data = {"date": dates, "target_return_7d": rng.normal(size=n)}
        for f in feats:
            data[f] = rng.normal(size=n)
        return pd.DataFrame(data)

    def test_fold_scores_align_to_val_rows(self):
        rng = np.random.default_rng(1)
        feats = ["f0", "f1", "f2"]
        train_dates = np.repeat(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]), 40)
        val_dates = np.repeat(pd.to_datetime(["2026-01-10", "2026-01-11"]), 30)
        train_df = self._frame(train_dates, feats, rng)
        val_df = self._frame(val_dates, feats, rng)
        fc = self._forecaster(feats)
        scores = fc._lambdarank_fold_scores(train_df, val_df, 7, {0.5: {}})
        assert len(scores) == len(val_df)
        assert np.isfinite(np.asarray(scores)).all()

    def test_trainer_tolerates_unsorted_train_dates(self):
        rng = np.random.default_rng(2)
        feats = ["f0", "f1"]
        train_dates = pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-02", "2026-01-01"] * 15)
        val_dates = np.repeat(pd.to_datetime(["2026-01-10"]), 25)
        train_df = self._frame(train_dates, feats, rng)
        val_df = self._frame(val_dates, feats, rng)
        fc = self._forecaster(feats)
        scores = fc._lambdarank_fold_scores(train_df, val_df, 7, {0.5: {}})
        assert len(scores) == len(val_df)


class TestLambdarankFlag:
    def test_flag_off_by_default(self, monkeypatch):
        monkeypatch.delenv("LAMBDARANK", raising=False)
        assert ItemForecaster._lambdarank_enabled() is False

    def test_flag_on_when_set_to_1(self, monkeypatch):
        monkeypatch.setenv("LAMBDARANK", "1")
        assert ItemForecaster._lambdarank_enabled() is True

    def test_flag_off_for_other_values(self, monkeypatch):
        monkeypatch.setenv("LAMBDARANK", "true")
        assert ItemForecaster._lambdarank_enabled() is False
