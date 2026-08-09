"""Boost rounds are fixed, not early-stopped, and the CV carries invariant #4.

Early stopping was scored against `_build_production_split`'s trailing window,
whose effective sample size is ~a dozen dates because items move together
within a date. Measured 2026-08-08 it left 9 of 33 CV folds fitting a single
tree and cost 23-88% of out-of-fold rank IC at 7/14/30d. See
docs/research/2026-08-08-model-review.md.

These tests pin the mechanism, not the calibrated numbers: the counts in
FIXED_BOOST_ROUNDS are expected to move when the sweep is re-run.
"""
import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


class TestBoostRoundSelection:
    def test_fixed_rounds_are_the_default(self, monkeypatch):
        monkeypatch.delenv("EARLY_STOPPING", raising=False)
        assert ItemForecaster._early_stopping_enabled() is False

    def test_every_horizon_has_a_production_and_a_cv_count(self):
        for h in ItemForecaster.HORIZONS:
            assert h in ItemForecaster.FIXED_BOOST_ROUNDS, f"{h}d has no fixed count"
            assert h in ItemForecaster.CV_FIXED_BOOST_ROUNDS

    def test_counts_are_positive(self):
        for table in (ItemForecaster.FIXED_BOOST_ROUNDS,
                      ItemForecaster.CV_FIXED_BOOST_ROUNDS):
            assert all(v > 0 for v in table.values())

    def test_unknown_horizon_falls_back_rather_than_training_zero_rounds(self):
        """A HORIZONS change must not silently produce a 0-round model."""
        assert ItemForecaster._boost_rounds(999) > 0
        assert ItemForecaster._boost_rounds(999, cv=True) > 0

    def test_env_restores_early_stopping_and_the_legacy_caps(self, monkeypatch):
        monkeypatch.setenv("EARLY_STOPPING", "1")
        assert ItemForecaster._early_stopping_enabled() is True
        assert ItemForecaster._boost_rounds(3) == 1000
        assert ItemForecaster._boost_rounds(3, cv=True) == 200

    def test_only_the_exact_value_1_enables_early_stopping(self, monkeypatch):
        """A typo must not quietly restore the defect this change removes."""
        for raw in ("0", "", "true", "yes", "TRUE"):
            monkeypatch.setenv("EARLY_STOPPING", raw)
            assert ItemForecaster._early_stopping_enabled() is False


class TestEnsembleMemberHonoursTheFlag:
    """The regression: with early stopping off, `dval` must not be attached.

    Nothing reads the per-round metric once stopping is gone, and evaluating it
    every round is pure cost.
    """

    @staticmethod
    def _fit(monkeypatch, early_stopping):
        import lightgbm as lgb
        rng = np.random.default_rng(0)
        X = rng.standard_normal((400, 3))
        y = X[:, 0] + rng.standard_normal(400) * 0.1
        params = {"objective": "regression", "verbosity": -1, "num_leaves": 4,
                  "learning_rate": 0.2, "min_data_in_leaf": 5}
        ds = {"feature_pre_filter": False}
        dtrain = lgb.Dataset(X, y, params=ds)
        dval = lgb.Dataset(X[:100], y[:100], reference=dtrain, params=ds)
        dtrain.construct()
        dval.construct()
        return ItemForecaster._train_ensemble_member(
            params, dtrain, dval, num_boost_round=30,
            early_stopping=early_stopping)

    def test_fixed_rounds_trains_the_full_count(self, monkeypatch):
        booster = self._fit(monkeypatch, early_stopping=False)
        assert booster.num_trees() == 30, (
            "early stopping is off, so every requested round must be trained")

    def test_early_stopping_path_still_works(self, monkeypatch):
        booster = self._fit(monkeypatch, early_stopping=True)
        assert booster.num_trees() <= 30


class TestRankIC:
    def test_perfect_ordering_scores_one(self):
        dates = pd.Series(["2026-01-01"] * 30)
        actual = np.arange(30, dtype=float)
        ic = ItemForecaster._within_date_rank_ic(actual, actual, dates)
        assert ic == pytest.approx(1.0)

    def test_reversed_ordering_scores_minus_one(self):
        dates = pd.Series(["2026-01-01"] * 30)
        actual = np.arange(30, dtype=float)
        ic = ItemForecaster._within_date_rank_ic(-actual, actual, dates)
        assert ic == pytest.approx(-1.0)

    def test_it_is_within_date_not_pooled(self):
        """Two dates each perfectly ordered, but with opposite LEVELS.

        Pooled, the level difference dominates and the correlation collapses.
        Within-date, both days are perfect, so the mean must be 1.0. This is
        the whole reason the metric exists: it removes the market factor.
        """
        dates = pd.Series(["2026-01-01"] * 30 + ["2026-01-02"] * 30)
        within = np.arange(30, dtype=float)
        pred = np.concatenate([within, within])
        actual = np.concatenate([within, within - 1000.0])
        assert ItemForecaster._within_date_rank_ic(pred, actual, dates) == pytest.approx(1.0)

    def test_dates_below_the_row_floor_are_skipped_not_scored_zero(self):
        dates = pd.Series(["2026-01-01"] * 5)
        a = np.arange(5, dtype=float)
        assert ItemForecaster._within_date_rank_ic(a, a, dates) is None

    def test_a_constant_prediction_contributes_nothing(self):
        """No variation means no ordering, which is absent, not zero."""
        dates = pd.Series(["2026-01-01"] * 30)
        assert ItemForecaster._within_date_rank_ic(
            np.ones(30), np.arange(30, dtype=float), dates) is None

    def test_mask_restricts_to_the_served_cohort(self):
        dates = pd.Series(["2026-01-01"] * 60)
        a = np.arange(60, dtype=float)
        pred = np.concatenate([a[:30], -a[30:]])       # ordered, then reversed
        served = np.array([True] * 30 + [False] * 30)
        assert ItemForecaster._within_date_rank_ic(pred, a, dates, served) == pytest.approx(1.0)


class TestDirectionRecords:
    def test_shape_matches_what_the_pt_test_consumes(self):
        from backtest.directional_test import pesaran_timmermann
        recs = ItemForecaster._direction_records(
            [5.0, -5.0, 0.0], [5.0, -5.0, 0.0],
            pd.Series(["2026-01-01"] * 3))
        assert {"predicted_direction", "actual_direction",
                "direction_correct", "forecast_date"} <= set(recs[0])
        pesaran_timmermann(recs, 20)          # must not raise

    def test_flat_band_matches_the_scorer(self):
        from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT as tol
        recs = ItemForecaster._direction_records(
            [tol * 2, -tol * 2, 0.0], [0.0, 0.0, 0.0],
            pd.Series(["2026-01-01"] * 3))
        assert [r["predicted_direction"] for r in recs] == ["up", "down", "flat"]
        assert [r["actual_direction"] for r in recs] == ["flat"] * 3

    def test_constant_call_baseline_reads_these_records(self):
        from backtest.directional_test import constant_call_baseline
        recs = ItemForecaster._direction_records(
            np.zeros(10), np.array([-5.0] * 7 + [5.0] * 3),
            pd.Series(["2026-01-01"] * 10))
        direction, acc = constant_call_baseline(recs)
        assert direction == "down"
        assert acc == pytest.approx(70.0)


class TestDiagnosticClassifierGate:
    def test_off_by_default(self, monkeypatch):
        """Was on by default until 2026-08-09. See
        test_cv_diagnostic_classifier_defaults_off below for the cost."""
        monkeypatch.delenv("CV_DIAGNOSTIC_CLASSIFIER", raising=False)
        assert ItemForecaster._cv_diagnostic_classifier_enabled() is False

    def test_zero_turns_it_off(self, monkeypatch):
        monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "0")
        assert ItemForecaster._cv_diagnostic_classifier_enabled() is False

    def test_anything_else_turns_it_on(self, monkeypatch):
        """Only a literal "0" is off, so a typo cannot silently skip a
        diagnostic the caller asked for."""
        for raw in ("1", "yes", "", "true"):
            monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", raw)
            assert ItemForecaster._cv_diagnostic_classifier_enabled() is True


def test_ci_skips_the_diagnostic_classifier_but_never_cv():
    """The CI workflow buys minutes from the diagnostic, not from calibration."""
    from pathlib import Path
    wf = (Path(__file__).resolve().parents[2]
          / ".github" / "workflows" / "price-forecast.yml").read_text()
    assert 'CV_DIAGNOSTIC_CLASSIFIER: "0"' in wf
    assert "SKIP_CV: " not in wf, "SKIP_CV must never be set in CI"


def test_cv_diagnostic_classifier_defaults_off(monkeypatch):
    """932s / 52% of a classifier-on retrain, to populate a meta.json field no
    served artifact reads. Measured 2026-08-09: 872s off vs 1804s on. On by
    default put a local retrain at 30.1 min, over the project's own 30-minute
    run cap."""
    monkeypatch.delenv("CV_DIAGNOSTIC_CLASSIFIER", raising=False)
    assert ItemForecaster._cv_diagnostic_classifier_enabled() is False


def test_cv_diagnostic_classifier_can_be_re_enabled(monkeypatch):
    monkeypatch.setenv("CV_DIAGNOSTIC_CLASSIFIER", "1")
    assert ItemForecaster._cv_diagnostic_classifier_enabled() is True
