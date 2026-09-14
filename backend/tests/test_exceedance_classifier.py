"""`_fit_exceedance_classifier`: a binary P(move clears round-trip cost) head.

Mirrors the direction classifier but trains on the precomputed one-sided
`target_exceed_{h}d` labels. Rows with a NaN label are dropped (so it trains on the
range model's rows); it returns None when fewer than two classes survive.
Scope: docs/superpowers/plans/2026-08-16-exceedance-band-scale-phase2-plan.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from models.forecaster import ItemForecaster


def _forecaster(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _xy(seed=0):
    rng = np.random.default_rng(seed)
    n = 2000
    x = rng.normal(size=n)
    y = (x > 0.5).astype(float)  # feature `f` separates the classes
    y[:100] = np.nan  # NaN-labelled rows must be dropped, not trained on
    X = pd.DataFrame({"f": x, "g": rng.normal(size=n)})
    return X, y, x


def test_exceedance_head_learns_signal_and_drops_nan_labels(tmp_path):
    f = _forecaster(tmp_path)
    X, y, x = _xy()
    booster = f._fit_exceedance_classifier(
        X, y, boosting_type="gbdt", tree_params={}, horizon=7, tier_train=np.full(len(y), 2), num_boost_round=100
    )
    assert booster is not None
    p = booster.predict(X)
    assert p.min() >= 0.0 and p.max() <= 1.0  # probabilities
    assert p[x > 0.5].mean() > p[x <= 0.5].mean() + 0.2  # learned the separating feature


def test_exceedance_head_returns_none_when_single_class(tmp_path):
    f = _forecaster(tmp_path)
    X, _, _ = _xy()
    y = np.zeros(len(X))  # only one class survives
    assert f._fit_exceedance_classifier(X, y, boosting_type="gbdt", tree_params={}, horizon=7) is None
