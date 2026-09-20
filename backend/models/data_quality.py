"""Cross-sectional data quality scoring.

Uses an Isolation Forest on per-date features to detect dates where
the price feed is anomalous -- frozen quotes, feed outages, or
abnormally correlated returns.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def compute_date_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-date cross-sectional features.

    Parameters
    ----------
    prices_df : DataFrame
        Must have columns: item_slug, day, price, source.

    Returns
    -------
    DataFrame indexed by day with features:
      - pct_unchanged: fraction of items with identical price to prior day
      - mean_abs_return: mean |daily return| across items
      - source_count: number of distinct sources active
      - return_dispersion: std of daily returns across items
      - n_items: number of items with prices
    """
    required = {"item_slug", "day", "price"}
    missing = required - set(prices_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = prices_df.copy()
    df["day"] = pd.to_datetime(df["day"])

    # Collapse to one price per item per day (median across sources)
    daily = (
        df.groupby(["item_slug", "day"])["price"]
        .median()
        .reset_index()
        .sort_values(["item_slug", "day"])
    )

    # Compute daily returns per item
    daily["prev_price"] = daily.groupby("item_slug")["price"].shift(1)
    daily["return_1d"] = (daily["price"] - daily["prev_price"]) / daily[
        "prev_price"
    ].replace(0, np.nan)
    daily["unchanged"] = (daily["price"] == daily["prev_price"]).astype(int)

    # Per-date aggregation
    date_stats = daily.dropna(subset=["prev_price"]).groupby("day").agg(
        pct_unchanged=("unchanged", "mean"),
        mean_abs_return=("return_1d", lambda x: np.abs(x).mean()),
        return_dispersion=("return_1d", "std"),
        n_items=("item_slug", "nunique"),
    )

    # Source count (requires source column)
    if "source" in df.columns:
        source_counts = (
            df.dropna(subset=["source"])
            .groupby("day")["source"]
            .nunique()
            .rename("source_count")
        )
        date_stats = date_stats.join(source_counts, how="left")
        date_stats["source_count"] = date_stats["source_count"].fillna(0).astype(int)
    else:
        date_stats["source_count"] = 1

    return date_stats


def fit_quality_model(
    date_features: pd.DataFrame, contamination: float = 0.05
) -> IsolationForest:
    """Fit Isolation Forest on date-level features.

    Train on all available data; contamination controls the anomaly threshold.

    Parameters
    ----------
    date_features : DataFrame
        Output of ``compute_date_features``.
    contamination : float
        Expected proportion of anomalies.

    Returns
    -------
    Fitted IsolationForest.
    """
    feature_cols = [
        "pct_unchanged",
        "mean_abs_return",
        "source_count",
        "return_dispersion",
        "n_items",
    ]
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(date_features[feature_cols].fillna(0))
    return model


def score_dates(model: IsolationForest, date_features: pd.DataFrame) -> pd.Series:
    """Score each date: 0=anomalous, 1=normal, with continuous scores.

    Parameters
    ----------
    model : fitted IsolationForest
    date_features : DataFrame with the same feature columns.

    Returns
    -------
    Series indexed by day with quality_score in [0, 1].
    """
    feature_cols = [
        "pct_unchanged",
        "mean_abs_return",
        "source_count",
        "return_dispersion",
        "n_items",
    ]
    raw_scores = model.decision_function(date_features[feature_cols].fillna(0))
    # Normalize to [0, 1] where 1 = normal, 0 = most anomalous
    score_range = raw_scores.max() - raw_scores.min()
    normalized = (raw_scores - raw_scores.min()) / (score_range + 1e-10)
    return pd.Series(
        normalized, index=date_features.index, name="quality_score"
    )
