import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")


def _make_price_df(n_items=5, n_days=250, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    base_date = pd.Timestamp("2024-06-01")
    for i in range(n_items):
        price = 8.0 + rng.randn() * 2
        for d in range(n_days):
            price *= np.exp(rng.randn() * 0.015)
            rows.append({
                "item_id": f"item_{i}",
                "date": (base_date + pd.Timedelta(days=d)).date(),
                "price": round(max(price, 1.0), 2),
                "volume": max(1, int(rng.poisson(15))),
            })
    return pd.DataFrame(rows)


class TestTFTTrainAndPredict:
    """End-to-end: build dataset from voted-style df, train TFT, get predictions."""

    def test_cv_produces_oof_for_conformal(self, tmp_path):
        from models.tft.trainer import TFTTrainer
        from models.tft.model import TFTConfig

        df = _make_price_df(n_items=5, n_days=250)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))

        dates = sorted(df["date"].unique())
        folds = [
            (dates[:120], dates[120:150]),
            (dates[:150], dates[150:180]),
        ]
        oof = trainer.train_cv(df, folds, max_epochs=3, patience=2)

        # OOF must have the columns conformal needs
        assert len(oof) > 0
        for h in [3, 7, 14, 30]:
            residuals = oof[f"actual_{h}d"] - oof[f"pred_{h}d"]
            assert not residuals.isna().all(), f"All NaN residuals at h={h}"

    def test_predict_returns_per_item_latest(self, tmp_path):
        from models.tft.trainer import TFTTrainer
        from models.tft.model import TFTConfig

        df = _make_price_df(n_items=5, n_days=200)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))

        dates = sorted(df["date"].unique())
        trainer.train_fold(df, dates[:130], dates[130:160],
                           max_epochs=2, patience=2)
        preds = trainer.predict(df, dates[130:160])

        # Should have predictions for each item
        assert preds["item_id"].nunique() > 0
        assert all(f"pred_{h}d" in preds.columns for h in [3, 7, 14, 30])
