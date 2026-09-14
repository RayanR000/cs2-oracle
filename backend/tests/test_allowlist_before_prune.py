"""Prune 33 columns, not 123.

`df[self.feature_cols].corr()` is O(rows x p^2) single-threaded pandas: 25.2s
over the 123 selected candidates, 1.65s over the 33 the allowlist keeps
(measured 2026-08-09 on the production frame). On that frame the prune drops
zero price_technicals features, so the reorder is output-identical -- but
_prune_features keeps the LOWER-INDEXED member of each >0.95 pair and index
order does not follow group, so that is a property of the frame, not a theorem.
Hence the flag and this test.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_flag_defaults_on():
    assert ItemForecaster.ALLOWLIST_BEFORE_PRUNE is True


def test_prune_sees_only_allowlisted_columns(tmp_path):
    """The expensive call must run on the post-allowlist set."""
    f = _f(tmp_path)
    f.feature_cols = ["return_1d", "rsi_14", "day_of_week", "rarity_ordinal"]
    seen = {}
    original = ItemForecaster._prune_features

    def spy(self, df):
        seen["n"] = len(self.feature_cols)
        return original(self, df)

    ItemForecaster._prune_features = spy
    try:
        rng = np.random.default_rng(3)
        df = pd.DataFrame({c: rng.normal(size=200) for c in f.feature_cols})
        f._reduce_feature_cols(df)
    finally:
        ItemForecaster._prune_features = original
    assert seen["n"] == 2, "prune ran on the pre-allowlist column set"


def test_reorder_is_output_identical_on_uncorrelated_features(tmp_path):
    """Both orderings agree when no >0.95 pair crosses the allowlist boundary."""
    rng = np.random.default_rng(11)
    cols = ["return_1d", "rsi_14", "day_of_week", "rarity_ordinal"]
    df = pd.DataFrame({c: rng.normal(size=500) for c in cols})

    a = _f(tmp_path / "a")
    a.feature_cols = list(cols)
    a.ALLOWLIST_BEFORE_PRUNE = True
    a._reduce_feature_cols(df)

    b = _f(tmp_path / "b")
    b.feature_cols = list(cols)
    b.ALLOWLIST_BEFORE_PRUNE = False
    b._reduce_feature_cols(df)

    assert a.feature_cols == b.feature_cols
