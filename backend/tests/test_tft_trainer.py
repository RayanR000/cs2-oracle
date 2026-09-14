import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _make_price_df(n_items=10, n_days=300, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(n_items):
        price = 10.0 + rng.randn() * 3
        for d in range(n_days):
            price *= np.exp(rng.randn() * 0.015)
            rows.append(
                {
                    "item_id": f"item_{i}",
                    "date": (base_date + pd.Timedelta(days=d)).date(),
                    "price": round(max(price, 0.5), 2),
                    "volume": max(1, int(rng.poisson(15))),
                }
            )
    return pd.DataFrame(rows)


class TestQuantileLoss:
    def test_symmetric_at_half(self):
        from models.tft.trainer import quantile_loss

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 2.0])
        loss = quantile_loss(pred, target, quantile=0.5)
        # At q=0.5, quantile loss = 0.5 * MAE
        expected = 0.5 * torch.tensor([1.0, 0.0, 1.0]).mean()
        assert torch.allclose(loss, expected, atol=1e-5)


class TestTFTTrainer:
    def test_train_single_fold(self, tmp_path):
        from models.tft.model import TFTConfig
        from models.tft.trainer import TFTTrainer

        df = _make_price_df(n_items=5, n_days=200)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))
        dates = sorted(df["date"].unique())
        split = len(dates) * 2 // 3
        train_dates = dates[:split]
        val_dates = dates[split:]
        val_loss = trainer.train_fold(df, train_dates, val_dates, max_epochs=3, patience=2)
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_predict_after_train(self, tmp_path):
        from models.tft.model import TFTConfig
        from models.tft.trainer import TFTTrainer

        df = _make_price_df(n_items=5, n_days=200)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))
        dates = sorted(df["date"].unique())
        split = len(dates) * 2 // 3
        trainer.train_fold(df, dates[:split], dates[split:], max_epochs=2, patience=2)
        preds = trainer.predict(df, dates[split:])
        assert "item_id" in preds.columns
        assert "date" in preds.columns
        for h in [3, 7, 14, 30]:
            assert f"pred_{h}d" in preds.columns
        assert len(preds) > 0
        assert not preds["pred_3d"].isna().all()

    def test_save_and_load(self, tmp_path):
        from models.tft.model import TFTConfig
        from models.tft.trainer import TFTTrainer

        df = _make_price_df(n_items=3, n_days=150)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))
        dates = sorted(df["date"].unique())
        split = len(dates) * 2 // 3
        trainer.train_fold(df, dates[:split], dates[split:], max_epochs=2, patience=2)
        trainer.save()
        loaded = TFTTrainer.load(str(tmp_path))
        preds_orig = trainer.predict(df, dates[split:])
        preds_loaded = loaded.predict(df, dates[split:])
        pd.testing.assert_frame_equal(preds_orig, preds_loaded)

    def test_oof_predictions(self, tmp_path):
        from models.tft.model import TFTConfig
        from models.tft.trainer import TFTTrainer

        df = _make_price_df(n_items=5, n_days=300)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))
        dates = sorted(df["date"].unique())
        # 2 folds: 0..150 train / 150..180 val, 0..180 train / 180..210 val
        folds = [
            (dates[:150], dates[150:180]),
            (dates[:180], dates[180:210]),
        ]
        oof = trainer.train_cv(df, folds, max_epochs=2, patience=2)
        assert "item_id" in oof.columns
        assert "date" in oof.columns
        assert "pred_3d" in oof.columns
        assert "actual_3d" in oof.columns
        assert len(oof) > 0
