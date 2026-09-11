#!/usr/bin/env python
"""Compare TFT vs LightGBM centre predictions on OOF folds.

Usage (from backend/):
    TFT_CENTRE=1 venv/bin/python scripts/eval_tft.py

Reads the voted price history, runs expanding-window CV for TFT,
and compares against LightGBM OOF predictions (from meta.json cv_results).
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tft import TFTTrainer, TFTConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [3, 7, 14, 30]


def main():
    from models.forecaster import ItemForecaster
    from unittest.mock import MagicMock

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
    forecaster = ItemForecaster(db_session=MagicMock(), model_dir=model_dir)

    logger.info("Loading price history...")
    # universe="train": archive-derived cohort, no DB read. This is also the
    # universe train() (and ItemForecaster._train_tft) uses, so the eval
    # scores the same population TFT trains on. The "serve" universe needs
    # is_backfilled from Postgres, which a local MagicMock-db run cannot
    # provide (and backend/.env points at production — see AGENTS.md).
    price_df = forecaster.fetch_price_history(days_back=1460, backfilled_only=True,
                                              universe="train")
    # NOTE: the plan drafted this filter via transform("median").loc[...].index,
    # but that compares item_ids against row indices and selects nothing.
    # Filter on the per-item median directly (same statistic as
    # ItemForecaster._filter_by_median_price).
    item_median = price_df.groupby("item_id")["price"].median()
    keep = set(item_median[item_median >= 1.0].index)
    price_df = price_df[price_df["item_id"].isin(keep)]
    logger.info(f"  {price_df['item_id'].nunique()} items, {len(price_df):,} rows")

    sorted_dates = sorted(price_df["date"].unique())
    folds = forecaster._compute_cv_splits(sorted_dates, purge_days=0)
    logger.info(f"  {len(folds)} CV folds")

    # Train TFT
    config = TFTConfig(hidden_dim=32, num_heads=4, dropout=0.1)
    trainer = TFTTrainer(config, model_dir=os.path.join(model_dir, "tft_eval"))
    oof = trainer.train_cv(price_df, folds, max_epochs=50, patience=5)

    # Score
    logger.info("\n" + "=" * 70)
    logger.info("TFT OOF RESULTS")
    logger.info("=" * 70)

    rows = []
    for h in HORIZONS:
        pred_col = f"pred_{h}d"
        actual_col = f"actual_{h}d"
        valid = oof[[pred_col, actual_col, "fold"]].dropna()
        if valid.empty:
            continue

        residuals = valid[actual_col] - valid[pred_col]
        mae = residuals.abs().mean()
        ic, _ = spearmanr(valid[pred_col], valid[actual_col])

        # Naive baseline: predict 0 (no change)
        naive_mae = valid[actual_col].abs().mean()

        logger.info(
            f"  {h:2d}d: rank_ic={ic:+.4f}  MAE={mae:.2f}%  "
            f"naive_MAE={naive_mae:.2f}%  edge={naive_mae - mae:+.2f}%  "
            f"n={len(valid):,}"
        )
        rows.append({
            "horizon": h, "rank_ic": ic, "mae": mae,
            "naive_mae": naive_mae, "n": len(valid),
        })

    if rows:
        # train_cv() never persists a checkpoint, so the eval dir may not exist
        # (only TFTTrainer.save() calls makedirs). Create it before writing.
        os.makedirs(os.path.join(model_dir, "tft_eval"), exist_ok=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(model_dir, "tft_eval", "tft_eval.csv"), index=False)
        logger.info(f"\nResults saved to {model_dir}/tft_eval/tft_eval.csv")


if __name__ == "__main__":
    main()
