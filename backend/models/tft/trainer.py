from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from models.tft.dataset import (
    N_FUTURE_FEATURES,
    N_PAST_FEATURES,
    N_STATIC_FEATURES,
    SequenceDataset,
    build_dataloaders,
)
from models.tft.model import TemporalFusionTransformer, TFTConfig

logger = logging.getLogger(__name__)

HORIZONS = [3, 7, 14, 30]
CHECKPOINT_NAME = "tft_model.pt"
CONFIG_NAME = "tft_config.json"


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.5,
) -> torch.Tensor:
    error = target - pred
    return torch.mean(torch.max(quantile * error, (quantile - 1) * error))


class TFTTrainer:
    def __init__(
        self,
        config: TFTConfig | None = None,
        model_dir: str = "models/saved_models",
        horizons: list[int] | None = None,
        lookback: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        self.config = config or TFTConfig()
        self.model_dir = model_dir
        self.horizons = horizons or HORIZONS
        self.lookback = lookback
        self.batch_size = batch_size
        self.lr = lr
        self.model: TemporalFusionTransformer | None = None

    def _build_model(self) -> TemporalFusionTransformer:
        cfg = TFTConfig(
            n_past_features=N_PAST_FEATURES,
            n_static_features=N_STATIC_FEATURES,
            n_future_features=N_FUTURE_FEATURES,
            n_horizons=len(self.horizons),
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            lookback=self.lookback,
            lstm_layers=self.config.lstm_layers,
            n_price_tiers=self.config.n_price_tiers,
        )
        return TemporalFusionTransformer(cfg)

    def train_fold(
        self,
        df: pd.DataFrame,
        train_dates: list,
        val_dates: list,
        max_epochs: int = 50,
        patience: int = 5,
    ) -> float:
        """Train on one fold. Returns best validation loss."""
        self.model = self._build_model()
        train_dl, val_dl = build_dataloaders(
            df,
            train_dates,
            val_dates,
            lookback=self.lookback,
            horizons=self.horizons,
            batch_size=self.batch_size,
        )
        if len(train_dl) == 0:
            logger.warning("Empty training set — skipping fold")
            return float("inf")

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=len(train_dl),
            epochs=max_epochs,
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(max_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for past, static, future, targets, _ in train_dl:
                optimizer.zero_grad()
                preds = self.model(past, static, future)
                loss = quantile_loss(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(n_batches, 1)

            # Validate
            val_loss = self._evaluate(val_dl)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or no_improve == 0:
                logger.info(
                    f"  Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} best={best_val_loss:.4f}"
                )

            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_val_loss

    @torch.no_grad()
    def _evaluate(self, dl) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for past, static, future, targets, _ in dl:
            preds = self.model(past, static, future)
            loss = quantile_loss(preds, targets)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, dates: list | None = None) -> pd.DataFrame:
        """Run inference. Returns DataFrame with item_id, date, pred_{h}d columns."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        self.model.eval()

        ds = SequenceDataset(
            df,
            lookback=self.lookback,
            horizons=self.horizons,
            date_filter=dates,
        )
        if len(ds) == 0:
            cols = ["item_id", "date"] + [f"pred_{h}d" for h in self.horizons]
            return pd.DataFrame(columns=cols)

        from torch.utils.data import DataLoader

        from models.tft.dataset import _collate_fn

        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=_collate_fn)

        all_preds = []
        all_items = []
        all_dates = []

        for past, static, future, _, meta in dl:
            preds = self.model(past, static, future)  # [B, H]
            all_preds.append(preds.numpy())
            all_items.extend(meta["item_id"])
            all_dates.extend(meta["date"])

        preds_arr = np.concatenate(all_preds, axis=0)
        result = pd.DataFrame(
            {
                "item_id": all_items,
                "date": all_dates,
            }
        )
        for i, h in enumerate(self.horizons):
            result[f"pred_{h}d"] = preds_arr[:, i]
        return result

    def train_cv(
        self,
        df: pd.DataFrame,
        folds: list[tuple[list, list]],
        max_epochs: int = 50,
        patience: int = 5,
    ) -> pd.DataFrame:
        """Expanding-window CV. Returns OOF predictions with actuals."""
        all_oof = []

        for fi, (train_dates, val_dates) in enumerate(folds):
            logger.info(f"TFT CV fold {fi + 1}/{len(folds)}: train={len(train_dates)}d val={len(val_dates)}d")
            self.train_fold(df, train_dates, val_dates, max_epochs=max_epochs, patience=patience)
            fold_preds = self.predict(df, val_dates)

            # Attach actuals
            ds = SequenceDataset(df, lookback=self.lookback, horizons=self.horizons, date_filter=val_dates)
            actuals = {}
            for idx in range(len(ds)):
                _, _, _, targets, meta = ds[idx]
                key = (meta["item_id"], str(meta["date"]))
                actuals[key] = targets.numpy()

            for i, h in enumerate(self.horizons):
                fold_preds[f"actual_{h}d"] = fold_preds.apply(
                    lambda r: actuals.get((r["item_id"], str(r["date"])), np.full(len(self.horizons), np.nan))[i],
                    axis=1,
                )
            fold_preds["fold"] = fi
            all_oof.append(fold_preds)

        return pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()

    def save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, CHECKPOINT_NAME))
        config_dict = {
            "hidden_dim": self.config.hidden_dim,
            "num_heads": self.config.num_heads,
            "dropout": self.config.dropout,
            "lookback": self.lookback,
            "lstm_layers": self.config.lstm_layers,
            "n_price_tiers": self.config.n_price_tiers,
            "horizons": self.horizons,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }
        with open(os.path.join(self.model_dir, CONFIG_NAME), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> TFTTrainer:
        with open(os.path.join(model_dir, CONFIG_NAME)) as f:
            cfg = json.load(f)
        config = TFTConfig(
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
            lstm_layers=cfg.get("lstm_layers", 1),
            n_price_tiers=cfg.get("n_price_tiers", 10),
        )
        trainer = cls(
            config=config,
            model_dir=model_dir,
            horizons=cfg.get("horizons", HORIZONS),
            lookback=cfg.get("lookback", 60),
            batch_size=cfg.get("batch_size", 256),
            lr=cfg.get("lr", 1e-3),
        )
        trainer.model = trainer._build_model()
        state = torch.load(os.path.join(model_dir, CHECKPOINT_NAME), weights_only=True)
        trainer.model.load_state_dict(state)
        return trainer
