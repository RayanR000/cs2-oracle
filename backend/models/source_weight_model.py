"""Learned source quality weights for multi-source price voting.

Trains a small LightGBM predicting per-source absolute error vs.
next-day multi-source consensus, conditioned on source identity,
price tier, and source count. The predicted inverse-error serves
as vote weight in place of equal weighting.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "source_encoded",  # label-encoded source name
    "log_price",  # log of item price (proxy for tier)
    "n_sources",  # number of active sources for this item-day
]

# Known ask sources that participate in voting (excludes bid, trailing-window,
# and spot sources which are dropped before voting).
KNOWN_SOURCES = [
    "aggregator_buff163",
    "aggregator_csfloat",
    "aggregator_csmoney",
    "aggregator_skinport",
    "aggregator_youpin",
    "aggregator_sync",
    "aggregator_steam_17mafo",
]

# Static fallback weights derived from measured reliability.
# aggregator_sync degrades the vote by ~17% (split-half reliability),
# so it gets lower weight; the five third-party asks share equal weight;
# aggregator_steam_17mafo sits between (last_24h for >=1 items only).
DEFAULT_WEIGHTS: dict[str, float] = {
    "aggregator_buff163": 0.16,
    "aggregator_csfloat": 0.16,
    "aggregator_csmoney": 0.16,
    "aggregator_skinport": 0.16,
    "aggregator_youpin": 0.16,
    "aggregator_sync": 0.08,  # ~17% degradation penalty
    "aggregator_steam_17mafo": 0.12,
}


def build_training_data(archive_df: pd.DataFrame) -> pd.DataFrame:
    """Build (source, item, day) -> absolute_error training set.

    For each item-day with >=2 sources, compute the consensus (median
    of all sources) and each source's absolute deviation from it.

    Parameters
    ----------
    archive_df : DataFrame
        Must have columns: item_slug, day, source, price.

    Returns
    -------
    DataFrame with columns:
        item_slug, day, source, price, consensus, abs_error, n_sources, log_price
    """
    required = {"item_slug", "day", "source", "price"}
    missing = required - set(archive_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = archive_df.dropna(subset=["source", "price"]).copy()

    # Count sources per item-day
    group_sizes = df.groupby(["item_slug", "day"])["source"].transform("nunique")
    df["n_sources"] = group_sizes

    # Keep only multi-source item-days
    df = df[df["n_sources"] >= 2].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "item_slug",
                "day",
                "source",
                "price",
                "consensus",
                "abs_error",
                "n_sources",
                "log_price",
            ]
        )

    # Consensus = median price across sources for each item-day
    consensus = df.groupby(["item_slug", "day"])["price"].transform("median")
    df["consensus"] = consensus
    df["abs_error"] = np.abs(df["price"] - df["consensus"])
    df["log_price"] = np.log1p(df["price"].clip(lower=0))

    return df[
        [
            "item_slug",
            "day",
            "source",
            "price",
            "consensus",
            "abs_error",
            "n_sources",
            "log_price",
        ]
    ]


def train_source_model(training_data: pd.DataFrame):
    """Train a small LightGBM regressor predicting abs_error.

    Parameters
    ----------
    training_data : DataFrame
        Output of ``build_training_data``.

    Returns
    -------
    tuple of (model, source_encoder)
        model: fitted LightGBM regressor
        source_encoder: dict mapping source name -> int
    """
    import lightgbm as lgb

    if training_data.empty:
        raise ValueError("Cannot train on empty data")

    # Label-encode sources
    unique_sources = sorted(training_data["source"].unique())
    source_encoder = {s: i for i, s in enumerate(unique_sources)}

    X = training_data[["source", "log_price", "n_sources"]].copy()
    X["source_encoded"] = X["source"].map(source_encoder)
    X = X[["source_encoded", "log_price", "n_sources"]]

    y = training_data["abs_error"].values

    model = lgb.LGBMRegressor(
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    return model, source_encoder


def predict_weights(
    model,
    source_encoder: dict[str, int],
    sources: list[str],
    prices: np.ndarray,
    n_sources: int,
) -> np.ndarray:
    """Predict per-source vote weight as 1/(1+predicted_error).

    Parameters
    ----------
    model : fitted LightGBM regressor
    source_encoder : dict mapping source name -> int
    sources : list of source names for one item-day group
    prices : array of prices corresponding to each source
    n_sources : number of sources in this group

    Returns
    -------
    array of weights summing to 1
    """
    prices = np.asarray(prices, dtype=float)

    # Build feature matrix
    encoded = np.array(
        [source_encoder.get(s, -1) for s in sources], dtype=float
    )
    log_prices = np.log1p(np.clip(prices, 0, None))
    n_src = np.full(len(sources), n_sources, dtype=float)

    X = pd.DataFrame(
        {"source_encoded": encoded, "log_price": log_prices, "n_sources": n_src}
    )

    predicted_error = model.predict(X)
    # Ensure non-negative predicted errors
    predicted_error = np.clip(predicted_error, 0, None)

    weights = 1.0 / (1.0 + predicted_error)

    # Normalize to sum to 1
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones(len(sources)) / len(sources)

    return weights
