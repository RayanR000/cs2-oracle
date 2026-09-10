from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PAST_FEATURE_NAMES = [
    "log_price", "log_volume", "return_1d", "volatility_20d", "momentum_14d",
]
N_PAST_FEATURES = len(PAST_FEATURE_NAMES)
N_STATIC_FEATURES = 1   # price_tier
N_FUTURE_FEATURES = 1   # normalized horizon index
PRICE_TIER_BINS = [0, 1, 3, 7, 15, 30, 70, 150, 400, 1000, float("inf")]


def _price_tier(price: float) -> int:
    for i, edge in enumerate(PRICE_TIER_BINS[1:]):
        if price < edge:
            return i
    return len(PRICE_TIER_BINS) - 2


def _preprocess_item(item_df: pd.DataFrame) -> dict:
    """Compute per-item time series arrays from a sorted item DataFrame."""
    prices = item_df["price"].values.astype(np.float64)
    volumes = item_df["volume"].values.astype(np.float64)
    dates = item_df["date"].values

    log_prices = np.log(np.maximum(prices, 1e-6))
    log_volumes = np.log(np.maximum(volumes, 1e-6))

    # return_1d: percentage return
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / np.maximum(prices[:-1], 1e-6) * 100
    returns[0] = 0.0

    # volatility_20d: rolling std(return) / price (CV proxy)
    vol = np.zeros_like(prices)
    for t in range(len(prices)):
        start = max(0, t - 19)
        window = returns[start:t + 1]
        if len(window) >= 2:
            vol[t] = np.std(window) / max(np.mean(np.abs(prices[start:t + 1])), 1e-6) * 100
        else:
            vol[t] = 0.0

    # momentum_14d: (price - price_14d_ago) / price_14d_ago * 100, clamped
    mom = np.zeros_like(prices)
    for t in range(14, len(prices)):
        p_old = prices[t - 14]
        if p_old > 1e-6:
            mom[t] = (prices[t] - p_old) / p_old * 100

    # Stack: [T, 5]
    features = np.stack([log_prices, log_volumes, returns, vol, mom], axis=-1)

    median_price = float(np.median(prices))
    tier = _price_tier(median_price)

    return {
        "features": features.astype(np.float32),
        "prices": prices.astype(np.float64),
        "dates": dates,
        "tier": tier,
        "item_id": item_df["item_id"].iloc[0],
    }


class SequenceDataset(Dataset):
    """Sliding-window dataset over item price histories."""

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
        horizons: list[int] | None = None,
        date_filter: list | None = None,
    ):
        self.lookback = lookback
        self.horizons = horizons or [3, 7, 14, 30]
        self.max_horizon = max(self.horizons)
        norm_max = float(self.max_horizon)
        self.horizon_indices = np.array(
            [h / norm_max for h in self.horizons], dtype=np.float32
        ).reshape(-1, 1)

        # date_filter restricts ANCHOR dates, not rows: windows draw their
        # 60-day lookback (and 30-day targets) from the full history. Filtering
        # rows first would leave any <90-day validation window with zero
        # samples and make predict-on-latest-date always empty.
        date_set = set(date_filter) if date_filter is not None else None

        self.samples: list[tuple[dict, int]] = []
        for item_id, group in df.groupby("item_id"):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) < lookback + self.max_horizon:
                continue
            item_data = _preprocess_item(group)
            n_days = len(group)
            for t in range(lookback, n_days - self.max_horizon):
                if date_set is not None and item_data["dates"][t] not in date_set:
                    continue
                self.samples.append((item_data, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item_data, t = self.samples[idx]
        lb = self.lookback

        past = torch.from_numpy(item_data["features"][t - lb:t].copy())
        static = torch.tensor([item_data["tier"]], dtype=torch.long)
        future = torch.from_numpy(self.horizon_indices.copy())

        # Targets: percentage return at each horizon
        anchor_price = item_data["prices"][t]
        targets = np.array([
            (item_data["prices"][t + h] - anchor_price) / anchor_price * 100
            for h in self.horizons
        ], dtype=np.float32)
        targets = torch.from_numpy(targets)

        meta = {
            "item_id": item_data["item_id"],
            "date": item_data["dates"][t],
        }
        return past, static, future, targets, meta


def _collate_fn(batch):
    """Custom collate that handles the meta dict."""
    past = torch.stack([b[0] for b in batch])
    static = torch.stack([b[1] for b in batch])
    future = torch.stack([b[2] for b in batch])
    targets = torch.stack([b[3] for b in batch])
    meta = {
        "item_id": [b[4]["item_id"] for b in batch],
        "date": [b[4]["date"] for b in batch],
    }
    return past, static, future, targets, meta


def build_dataloaders(
    df: pd.DataFrame,
    train_dates: list,
    val_dates: list,
    lookback: int = 60,
    horizons: list[int] | None = None,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from date-based splits."""
    train_ds = SequenceDataset(df, lookback=lookback, horizons=horizons,
                               date_filter=train_dates)
    val_ds = SequenceDataset(df, lookback=lookback, horizons=horizons,
                             date_filter=val_dates)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=_collate_fn, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=_collate_fn, drop_last=False)
    return train_dl, val_dl
