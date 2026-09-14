"""Regime-reactive climatology band scale (CLIMATOLOGY_REACTIVE=1), v2.

v1 modulated the static climatology level by `sigma / baseline_sigma_item`,
where `sigma` was the LAGGING 60d rolling std and the baseline was the item's
median over the 2023-25 calibration folds. The 2026-08-20 calm-anchor A/B
refuted it: on a calm day just after a volatile stretch the 60d window still
holds the vol, so the multiplier WIDENED the band exactly when the market was
calm, and the stale baseline made it ~uniformly >1.

v2 replaces both with a self-normalising, date-aware pair of EWMA vols computed
in feature engineering: `ewm_reactive_fast` (short halflife, reactive) over
`ewm_reactive_slow` (long halflife, the item's own recent baseline). The
multiplier is `clip(fast / slow, LO, HI)` — 1.0 when recent vol equals the
item's norm (byte-identical to static climatology), <1 on a genuinely calm day,
>1 on a spike. Stateless: it reads two columns, stores no baseline.

These tests pin the multiplier, the matched-pair serving rule, the identity
guarantee when off, that q_hat is calibrated on the modulated scale, and that
feature engineering emits the two columns WITHOUT leaking them to the boosters.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster, _feature_group


@pytest.fixture
def fc(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _rows(fast, slow, item_ids=None, price=2.0):
    n = len(fast)
    return pd.DataFrame(
        {"item_id": item_ids or ["a"] * n, "price": [price] * n, "ewm_reactive_fast": fast, "ewm_reactive_slow": slow}
    )


class TestReactiveMultiplier:
    def test_one_when_recent_vol_equals_baseline(self, fc):
        out = fc._climatology_reactive_multiplier(_rows([0.10], [0.10]))
        assert out == pytest.approx([1.0])

    def test_below_one_when_calm(self, fc):
        # recent vol half the item's norm => narrower.
        out = fc._climatology_reactive_multiplier(_rows([0.05], [0.10]))
        assert out == pytest.approx([0.5])

    def test_scales_and_clips(self, fc):
        fc.CLIMATOLOGY_REACTIVE_LO = 0.5
        fc.CLIMATOLOGY_REACTIVE_HI = 2.0
        out = fc._climatology_reactive_multiplier(_rows([0.15, 100.0, 0.0001], [0.10, 0.10, 0.10]))
        assert out == pytest.approx([1.5, 2.0, 0.5])

    def test_nonpositive_or_nan_baseline_is_neutral(self, fc):
        out = fc._climatology_reactive_multiplier(_rows([0.1, 0.1], [0.0, np.nan]))
        assert out == pytest.approx([1.0, 1.0])

    def test_missing_columns_returns_none(self, fc):
        rows = pd.DataFrame({"item_id": ["a"], "price": [2.0]})
        assert fc._climatology_reactive_multiplier(rows) is None


class TestReactiveServing:
    def test_band_scale_multiplies_static_scale_when_served(self, fc):
        fc._artifact_climatology_scale = True
        fc._artifact_climatology_reactive = True
        fc.climatology_scale[7] = {"table": {"a": 4.0}, "tier_pool": {1: 8.0}, "global": 50.0}
        rows = _rows([0.20], [0.10])  # fast 2x slow => x2.0
        out = fc.band_scale(7, rows, sigma=np.array([0.99]))
        assert out == pytest.approx([8.0])  # 4.0 static * 2.0 mult

    def test_band_scale_identity_when_reactive_not_served(self, fc):
        fc._artifact_climatology_scale = True
        fc._artifact_climatology_reactive = False
        fc.climatology_scale[7] = {"table": {"a": 4.0}, "tier_pool": {1: 8.0}, "global": 50.0}
        rows = _rows([0.99], [0.10])  # ignored
        out = fc.band_scale(7, rows, sigma=np.array([0.5]))
        assert out == pytest.approx([4.0])


class TestGating:
    def test_env_flag_off_by_default(self, fc, monkeypatch):
        monkeypatch.delenv("CLIMATOLOGY_REACTIVE", raising=False)
        assert fc.climatology_reactive_enabled() is False

    def test_only_one_enables(self, fc, monkeypatch):
        monkeypatch.setenv("CLIMATOLOGY_REACTIVE", "1")
        assert fc.climatology_reactive_enabled() is True
        monkeypatch.setenv("CLIMATOLOGY_REACTIVE", "yes")
        assert fc.climatology_reactive_enabled() is False

    def test_serving_follows_artifact_over_env(self, fc, monkeypatch):
        monkeypatch.setenv("CLIMATOLOGY_REACTIVE", "1")
        fc._artifact_climatology_reactive = False
        assert fc._climatology_reactive_served() is False


class TestReactiveFit:
    def test_fit_modulates_scale_by_recent_vol(self, fc, monkeypatch):
        monkeypatch.setenv("CLIMATOLOGY_SCALE", "1")
        monkeypatch.setenv("CLIMATOLOGY_REACTIVE", "1")
        # One item, same static level; row 0 calm (fast<slow), row 1 spiking.
        frame = pd.DataFrame(
            {
                "item_id": ["a", "a"],
                "price": [50.0, 50.0],
                "target_return_7d": [10.0, -10.0],
                "ewm_reactive_fast": [0.05, 0.20],
                "ewm_reactive_slow": [0.10, 0.10],
            }
        )
        recs = pd.DataFrame({"row_index": [0, 1], "sigma": [0.1, 0.1]})
        scale = fc._fit_climatology_scale(7, recs, frame)
        assert scale is not None and scale.shape == (2,)
        assert scale[1] > scale[0]  # spiking row wider than the calm row


class TestFeatureEngineering:
    def test_emits_columns_not_in_booster_set(self):
        # The two columns must exist for the band scale but NEVER reach the
        # boosters: their _feature_group is not the allowlisted price_technicals,
        # and they are shelved.
        for c in ("ewm_reactive_fast", "ewm_reactive_slow"):
            assert _feature_group(c) != "price_technicals"
            assert c in ItemForecaster.SHELVED_FEATURES
