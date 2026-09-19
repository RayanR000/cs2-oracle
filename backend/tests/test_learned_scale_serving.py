"""The learned band scale, where it meets the artifact and the serving path.

`test_scale_model.py` covers the estimator. This covers the two ways it can be
wrong in production while looking fine offline:

- `q_hat` and the scale come apart, so a q_hat in the learned scale's units is
  served against `sigma`. That is not a degraded band, it is an unrelated one.
- the model is served against a different feature matrix from the one it was
  fitted on, which returns a plausible number computed from the wrong columns.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models import conformal
from models.forecaster import ItemForecaster


def _forecaster(tmp_path):
    """Never the default model_dir — save paths there clobber production
    artifacts, which are gitignored and unrecoverable."""
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _rows(n=6_000, seed=0):
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.5, 4.0, n)
    return pd.DataFrame(
        {
            "price_std_60d": driver,
            "return_1d": rng.normal(size=n),
            "price": np.full(n, 100.0),
        }
    ), driver


class TestTheDefaultPathIsUntouched:
    def test_no_scale_model_means_no_learned_scale(self, tmp_path):
        """Every artifact written before 2026-08-12 is this case, and it must
        serve exactly the band it was calibrated for."""
        fc = _forecaster(tmp_path)
        rows, driver = _rows(50)
        assert fc.band_scale(7, rows, driver) is None

    def test_the_flag_off_leaves_calibration_on_sigma(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LEARNED_SCALE", raising=False)
        fc = _forecaster(tmp_path)
        rng = np.random.default_rng(1)
        n = 8_000
        records = pd.DataFrame(
            {
                "residual_pct": rng.normal(scale=2.0, size=n),
                "sigma": rng.uniform(0.05, 0.5, size=n),
                "mid_ret": np.zeros(n),
                "fold": np.arange(n) % 4,
                "row_index": np.arange(n),
            }
        )
        frame = pd.DataFrame({"price_std_60d": rng.normal(size=n), "price": np.full(n, 100.0)})
        fc._calibrate_conformal(7, records, frame)
        assert fc.scale_models == {}
        assert fc.band_scale(7, frame, records["sigma"].to_numpy()) is None


class TestTheTwoFlagsAreAlternatives:
    def test_setting_both_raises_at_the_first_calibration(self, tmp_path, monkeypatch):
        """Not a preference. Applying an exponent on top of a fitted scale
        re-tilts the band the other way, which is the failure measured in
        2026-08-12-served-sigma-profile.md."""
        monkeypatch.setenv("LEARNED_SCALE", "1")
        monkeypatch.setenv("SIGMA_EXPONENT", "1")
        fc = _forecaster(tmp_path)
        records = pd.DataFrame({"residual_pct": [1.0], "sigma": [0.1], "mid_ret": [0.0]})
        with pytest.raises(RuntimeError, match="alternative band denominators"):
            fc._calibrate_conformal(7, records, None)


class TestTheMatchedPairSurvivesTheArtifact:
    def _train_a_scale(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LEARNED_SCALE", "1")
        monkeypatch.delenv("SIGMA_EXPONENT", raising=False)
        # CLIMATOLOGY_SCALE went default-ON on 2026-08-19 and is a fourth
        # alternative band denominator, so leaving it set makes
        # `_calibrate_conformal` raise the mutual-exclusion guard before this
        # helper can fit anything. Neutralised the same way
        # tests/test_exceedance_scale.py neutralises it to test its own arm.
        monkeypatch.setenv("CLIMATOLOGY_SCALE", "0")
        fc = _forecaster(tmp_path)
        fc.feature_cols = ["price_std_60d", "return_1d"]
        fc.horizon_feature_cols = {7: ["price_std_60d", "return_1d"]}

        n = 24_000
        rng = np.random.default_rng(2)
        frame, driver = _rows(n, seed=2)
        resid = rng.normal(scale=driver)
        records = pd.DataFrame(
            {
                "residual_pct": resid,
                # sigma over-reacts, exactly as it does on the real archive
                "sigma": driver**2.0,
                "mid_ret": np.zeros(n),
                "fold": np.arange(n) % 6,
                "row_index": frame.index.to_numpy(),
            }
        )
        q_hat = fc._calibrate_conformal(7, records, frame)
        return fc, frame, records, q_hat

    def test_the_scale_is_fitted_persisted_and_restored(self, tmp_path, monkeypatch):
        fc, frame, records, _q_hat = self._train_a_scale(tmp_path, monkeypatch)
        assert 7 in fc.scale_models
        assert fc.scale_features[7][-1] == "sigma"

        sigma = records["sigma"].to_numpy()
        before = fc.band_scale(7, frame, sigma)
        assert before is not None

        fc.save_models()
        restored = _forecaster(tmp_path)
        restored.feature_cols = fc.feature_cols
        restored.horizon_feature_cols = fc.horizon_feature_cols
        restored.scale_models = {}
        # Restore just the scale half the way load_models does, without
        # requiring a full artifact on disk.
        import lightgbm as lgb

        restored.scale_models[7] = lgb.Booster(model_file=str(tmp_path / "scale_7d.txt"))
        restored.scale_norm[7] = fc.scale_norm[7]
        restored.scale_clip[7] = fc.scale_clip[7]
        restored.scale_features[7] = fc.scale_features[7]

        after = restored.band_scale(7, frame, sigma)
        np.testing.assert_allclose(before, after)

    def test_the_band_it_serves_covers_at_nominal_across_the_scale(self, tmp_path, monkeypatch):
        """The point of the exercise: coverage that holds across the range,
        which is what `sigma` fails at (62->95% across its deciles)."""
        fc, frame, records, q_hat = self._train_a_scale(tmp_path, monkeypatch)
        sigma = records["sigma"].to_numpy()
        resid = records["residual_pct"].to_numpy()

        learned = fc.band_scale(7, frame, sigma)
        lo, hi = conformal.band(np.zeros(len(frame)), sigma, q_hat, learned_scale=learned)
        covered = (resid >= lo) & (resid <= hi)

        order = np.argsort(np.argsort(learned))
        strata = (order * 5) // len(learned)
        prof = np.array([covered[strata == s].mean() for s in range(5)])
        tilt = np.mean(np.abs(prof - conformal.NOMINAL_COVERAGE))
        assert tilt < 0.05, f"served profile is tilted: {prof.round(3)}"

    def test_a_stale_scale_file_is_removed_rather_than_left_to_be_loaded(self, tmp_path, monkeypatch):
        """A scale_*.txt from a previous run would be loaded beside a q_hat
        calibrated without it — the matched pair coming apart across runs."""
        fc, _frame, _records, _q_hat = self._train_a_scale(tmp_path, monkeypatch)
        fc.save_models()
        assert (tmp_path / "scale_7d.txt").exists()

        fc.scale_models = {}  # a later run with the flag off
        fc.save_models()
        assert not (tmp_path / "scale_7d.txt").exists()


class TestServingRefusesTheWrongFeatures:
    def test_a_column_mismatch_falls_back_instead_of_scoring_junk(self, tmp_path, monkeypatch):
        """LightGBM will score a frame whose columns mean something else and
        return a plausible number. The band built from it looks entirely
        normal, which is why this refuses rather than reindexes."""
        fc, frame, records, _q_hat = TestTheMatchedPairSurvivesTheArtifact()._train_a_scale(tmp_path, monkeypatch)
        fc.scale_features[7] = ["something_else", "sigma"]
        assert fc.band_scale(7, frame, records["sigma"].to_numpy()) is None
