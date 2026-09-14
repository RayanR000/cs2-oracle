"""The exceedance head is stored on the forecaster and persists across a warm reload.

Task 1 of docs/superpowers/plans/2026-08-16-exceedance-band-scale-phase2-plan.md wires
`_fit_exceedance_classifier` into the training loop beside the direction head and saves the
boosters beside `clf_{h}d.txt`. A warm retrain restores the artifact, so the head Phase 2's
band scale reads must survive save/load — a served scale calibrated against a head the reload
dropped would mis-size the band. These guard the storage seam, not the offline A/B.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from models.forecaster import ItemForecaster


def _forecaster(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _trained_head(f, horizon=7):
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = (x > 0.5).astype(float)
    X = pd.DataFrame({"f": x, "g": rng.normal(size=n)})
    return X, f._fit_exceedance_classifier(
        X, y, boosting_type="gbdt", tree_params={}, horizon=horizon, tier_train=np.full(n, 2), num_boost_round=100
    )


def test_fresh_forecaster_exposes_an_empty_exceedance_head_store(tmp_path):
    assert _forecaster(tmp_path).exceedance_models == {}


def test_exceedance_head_round_trips_through_save_and_load(tmp_path):
    """Saved with a head, reloaded in a fresh forecaster, it predicts identically."""
    f = _forecaster(tmp_path)
    X, head = _trained_head(f, horizon=7)
    assert head is not None
    f.exceedance_models = {7: head}
    f.feature_cols = ["f", "g"]
    f.feature_medians = pd.Series({"f": 0.0, "g": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()

    g = _forecaster(tmp_path)
    g.load_models()
    assert 7 in g.exceedance_models
    np.testing.assert_allclose(g.exceedance_models[7].predict(X), head.predict(X), rtol=0, atol=0)


def test_training_loop_fits_and_stores_the_exceedance_head(tmp_path):
    """The head is trained beside the direction head and stored per horizon.

    A full train is the ~5-min path (the controller walk-forward in Task 3 is the real
    integration proof); this guards the wiring at the seam so it cannot silently drop
    out of the loop, mirroring the source-anchored guards elsewhere in the suite.
    """
    src = inspect.getsource(ItemForecaster._train_horizon_inline)
    assert "_fit_exceedance_classifier(" in src
    assert "self.exceedance_models[horizon]" in src
