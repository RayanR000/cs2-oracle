import pytest
import numpy as np
import pandas as pd

torch = pytest.importorskip("torch")


def _make_price_df(n_items=5, n_days=120, seed=42):
    """Synthetic daily price data matching voted frame schema."""
    rng = np.random.RandomState(seed)
    rows = []
    base_date = pd.Timestamp("2025-01-01")
    for i in range(n_items):
        price = 10.0 + rng.randn() * 2
        for d in range(n_days):
            price *= np.exp(rng.randn() * 0.02)
            rows.append({
                "item_id": f"item_{i}",
                "date": (base_date + pd.Timedelta(days=d)).date(),
                "price": round(max(price, 0.5), 2),
                "volume": max(1, int(rng.poisson(20))),
            })
    return pd.DataFrame(rows)


class TestSequenceDataset:

    def test_dataset_length(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=3, n_days=100)
        ds = SequenceDataset(df, lookback=60, horizons=[3, 7, 14, 30])
        # Each item has 100 days. Valid anchors: day 60..69 (day+30 <= 99).
        # 3 items × 10 valid anchors = 30
        assert len(ds) == 30

    def test_sample_shapes(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=2, n_days=100)
        ds = SequenceDataset(df, lookback=60, horizons=[3, 7, 14, 30])
        past, static, future, targets, meta = ds[0]
        assert past.shape == (60, 5)       # lookback × n_past_features
        assert static.shape == (1,)        # price_tier
        assert future.shape == (4, 1)      # n_horizons × n_future_features
        assert targets.shape == (4,)       # n_horizons
        assert "item_id" in meta
        assert "date" in meta

    def test_targets_are_percentage_returns(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=1, n_days=100)
        ds = SequenceDataset(df, lookback=60, horizons=[3])
        _, _, _, targets, meta = ds[0]
        # Manually compute expected return
        item_df = df[df["item_id"] == meta["item_id"]].sort_values("date")
        anchor_idx = item_df[item_df["date"] == meta["date"]].index[0]
        anchor_pos = list(item_df.index).index(anchor_idx)
        p_anchor = item_df.iloc[anchor_pos]["price"]
        p_target = item_df.iloc[anchor_pos + 3]["price"]
        expected = (p_target - p_anchor) / p_anchor * 100
        assert abs(targets[0].item() - expected) < 0.01

    def test_no_nan_in_past_features(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=2, n_days=100)
        ds = SequenceDataset(df, lookback=60, horizons=[3, 7, 14, 30])
        for i in range(min(10, len(ds))):
            past, _, _, _, _ = ds[i]
            assert not torch.isnan(past).any(), f"NaN in sample {i}"

    def test_nan_targets_when_future_missing(self):
        from models.tft.dataset import SequenceDataset
        # NOTE: the plan specified n_days=75 here, but with lookback=60 and
        # max_horizon=30 a 75-day history holds no anchor where ALL horizons
        # are valid (75 < 60 + 30), so the dataset is empty and both asserts
        # below fail. n_days=95 keeps the test's intent: a short history that
        # still yields samples, each with all horizons valid.
        df = _make_price_df(n_items=1, n_days=95)
        # lookback=60, max horizon=30 → valid anchors: day 60..64
        ds = SequenceDataset(df, lookback=60, horizons=[3, 7, 14, 30])
        # Should still create samples where at least h=3 is valid
        assert len(ds) > 0
        # (dataset only creates samples where ALL horizons are valid)
        _, _, _, targets, _ = ds[0]
        assert not torch.isnan(targets).any()


class TestBuildDataloaders:

    def test_train_val_split(self):
        from models.tft.dataset import SequenceDataset, build_dataloaders
        df = _make_price_df(n_items=5, n_days=200)
        dates = sorted(df["date"].unique())
        split = len(dates) // 2
        train_dates = dates[:split]
        val_dates = dates[split:]
        train_dl, val_dl = build_dataloaders(
            df, train_dates, val_dates,
            lookback=60, horizons=[3, 7, 14, 30], batch_size=16,
        )
        assert len(train_dl) > 0
        assert len(val_dl) > 0
        batch = next(iter(train_dl))
        past, static, future, targets, meta = batch
        assert past.shape[0] == 16  # batch size
