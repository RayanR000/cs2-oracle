"""Volume-derived features are withheld from training.

Volume is identically 0 in the price archive from 2026-05 onward (verified:
0 non-zero of 725,056 August rows, 0 of 6,172,930 July rows; Jan-Mar were ~100%
populated). It is stored as 0, never NULL, which defeats every guard the feature
code has:

* ``has_volume`` tests ``notna().any()``, and 0 is not NaN, so it stays True.
* ``volume_missing`` is therefore 0 — the flag built to say "no volume" reports
  that volume is present.
* The raw level features are handed a real-looking 0 against training medians of
  98.0 (``volume_lag_1d``), 99.0, 115.7, 43.9 and 124.2. They are not
  median-filled, because they are not NaN.

So the eleven volume features carry real signal in the training rows that
predate 2026-05 and are dead-or-misleading on every served row. Measured effect:
the served direction mix skews to "down" 48.2% at 7d against 34.8% on interior
rows. See docs/changelog for the write-up.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from models.forecaster import ItemForecaster, _feature_group

# Every volume-derived column engineer_features produces. The first eleven are
# the ones that appeared in the 47-column feature_cols of the 2026-08-05
# artifact; volume_mean_7d and volume_std_60d did not, because the >0.95
# correlation prune dropped them in favour of their 30d partners. Shelving the
# partners removes what they correlated against, so they survive the prune and
# re-enter production unless they are shelved too. They are equally dead: served
# as 0 against a real training level.
VOLUME_FEATURES = frozenset({
    "volume_missing",
    "volume_lag_1d",
    "volume_lag_7d",
    "volume_mean_30d",
    "volume_std_30d",
    "volume_mean_60d",
    "volume_log_change_1d",
    "volume_log_change_7d",
    "volume_zscore_30d",
    "volume_price_conf_7d",
    "volume_price_conf_1d",
    "volume_mean_7d",
    "volume_std_60d",
})


def test_every_volume_feature_is_shelved():
    """The knob that withholds them from training."""
    missing = sorted(VOLUME_FEATURES - ItemForecaster.SHELVED_FEATURES)
    assert not missing, f"not shelved: {missing}"


def test_volume_features_are_price_technicals_so_the_allowlist_cannot_drop_them():
    """Why shelving is required rather than relying on FEATURE_GROUP_ALLOWLIST.

    Every volume_* name resolves to the one group production allows, so the
    allowlist passes them straight through. If this ever stops being true the
    shelving above is redundant and can be reconsidered.
    """
    assert ItemForecaster.FEATURE_GROUP_ALLOWLIST == ["price_technicals"]
    for name in sorted(VOLUME_FEATURES):
        assert _feature_group(name) == "price_technicals", name


def test_feature_selection_excludes_shelved_volume_columns():
    """The shelving must reach the list the trainer actually fits on."""
    df = pd.DataFrame({
        "item_id": ["a", "b"],
        "date": pd.to_datetime(["2026-08-01", "2026-08-02"]),
        "price": [1.0, 2.0],
        "volume": [0.0, 0.0],
        "return_7d": [0.1, -0.2],
        # Witness that the volume shelving does not over-reach into the price
        # technicals. Was price_std_60d until 2026-08-06, when that column was
        # shelved by _DOLLAR_SCALE_FEATURES for an unrelated reason (dollars vs
        # a percentage target); price_cv_60d is its scale-free replacement and
        # keeps this assertion testing what it was written to test.
        "price_cv_60d": [0.3, 0.4],
        "target_7d": [1.1, 2.1],
        "target_return_7d": [0.1, 0.05],
        **{name: [1.0, 2.0] for name in sorted(VOLUME_FEATURES)},
    })

    cols = ItemForecaster._select_feature_cols(
        df, ItemForecaster.HORIZONS, ItemForecaster.SHELVED_FEATURES)

    assert set(cols) & VOLUME_FEATURES == set()
    # The price technicals it should keep are still there.
    assert "return_7d" in cols
    assert "price_cv_60d" in cols
    # And the metadata/target columns are still excluded.
    for excluded in ("item_id", "date", "price", "volume",
                     "target_7d", "target_return_7d"):
        assert excluded not in cols


def test_volume_columns_are_still_computed_for_downstream_features():
    """Shelved, not deleted — two downstream features still read these columns.

    ``supply_to_volume_ratio`` reads ``volume_mean_30d`` and
    ``item_volume_vs_market_30d`` reads ``volume``, so deleting the computation
    would break them. Shelved-from-training is a different thing from
    not-computed.
    """
    df = pd.DataFrame({
        "item_id": ["a"] * 40,
        "volume": np.arange(40, dtype=float),
        "return_1d": np.full(40, 0.01),
        "return_7d": np.full(40, 0.02),
    })

    out = ItemForecaster._compute_volume_features(df.copy())

    for name in sorted(VOLUME_FEATURES):
        assert name in out.columns, name
    # Real volume must still produce real values, not the all-NaN fallback.
    assert out["volume_mean_30d"].notna().any()


def test_all_zero_volume_still_yields_the_columns_downstream_needs():
    """The production case today: volume present as 0, never NULL."""
    df = pd.DataFrame({
        "item_id": ["a"] * 40,
        "volume": np.zeros(40),
        "return_1d": np.full(40, 0.01),
        "return_7d": np.full(40, 0.02),
    })

    out = ItemForecaster._compute_volume_features(df.copy())

    for name in sorted(VOLUME_FEATURES):
        assert name in out.columns, name


def test_no_volume_feature_survives_the_real_selection_and_prune(tmp_path):
    """End-to-end guard, and the only test that catches the prune interaction.

    Asserting against a hand-written name list cannot see that shelving
    ``volume_mean_30d`` lets the previously-pruned ``volume_mean_7d`` through.
    This runs the actual build_training_data path and asserts on the result.
    """
    from unittest.mock import MagicMock, patch

    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))

    def mock_fetch(*a, **k):
        np.random.seed(42)
        rows = []
        for item in range(5):
            price = 50.0
            for offset in range(200):
                price *= 1 + np.random.randn() * 0.01
                rows.append({
                    "item_id": f"item_{item}",
                    "date": date(2026, 1, 1) + timedelta(days=offset),
                    "price": round(max(price, 0.01), 2),
                    # Real volume, so the has_volume=True branch is exercised.
                    "volume": int(max(np.random.poisson(200), 0)),
                })
        return pd.DataFrame(rows)

    def mock_events(*a, **k):
        return pd.DataFrame([{
            "id": 1, "type": "major", "timestamp": pd.Timestamp("2026-06-01"),
            "description": "Major", "date": date(2026, 6, 1),
        }])

    with patch.object(f, "fetch_price_history", mock_fetch), \
         patch.object(f, "fetch_events", mock_events):
        df = f.build_training_data(days_back=200, backfilled_only=False)

    survivors = [c for c in f.feature_cols if "volume" in c]
    assert survivors == [], f"volume features reached training: {survivors}"
    # The columns must still be engineered for the downstream readers.
    assert "volume_mean_30d" in df.columns


def test_artifact_version_rejects_a_model_fitted_on_the_volume_features():
    """Changing feature_cols must force a retrain, not a silent mismatch.

    Without a version bump the 2026-08-05 artifact keeps its own 47-column
    feature_cols in meta.json, the columns are still engineered, so predict()
    would happily keep serving the eleven dead features until the 14-day age
    trigger fired.
    """
    assert ItemForecaster.MODEL_ARTIFACT_VERSION >= 3
