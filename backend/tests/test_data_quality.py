"""Tests for backend.models.data_quality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.data_quality import compute_date_features, fit_quality_model, score_dates


def _make_normal_prices(n_items: int = 50, n_days: int = 30) -> pd.DataFrame:
    """Build a synthetic price panel with normal market behaviour."""
    rng = np.random.RandomState(42)
    rows = []
    for item_idx in range(n_items):
        slug = f"item_{item_idx}"
        price = 10.0 + item_idx * 2
        for day_idx in range(n_days):
            day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day_idx)
            # Random walk with ~2% daily vol
            price = price * (1 + rng.normal(0, 0.02))
            rows.append(
                {
                    "item_slug": slug,
                    "day": day,
                    "price": max(price, 0.01),
                    "source": "aggregator_buff163",
                }
            )
    return pd.DataFrame(rows)


def _make_frozen_prices(n_items: int = 50, n_days: int = 10) -> pd.DataFrame:
    """Build a price panel where all items have frozen (identical) prices."""
    rows = []
    for item_idx in range(n_items):
        slug = f"item_{item_idx}"
        price = 10.0 + item_idx
        for day_idx in range(n_days):
            day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_idx)
            # Price never changes
            rows.append(
                {
                    "item_slug": slug,
                    "day": day,
                    "price": price,
                    "source": "aggregator_buff163",
                }
            )
    return pd.DataFrame(rows)


class TestComputeDateFeatures:
    def test_shape(self):
        df = _make_normal_prices(n_items=20, n_days=15)
        features = compute_date_features(df)
        # First day has no prior, so n_days - 1 dates with returns
        assert len(features) <= 15
        assert len(features) >= 13  # at least most days present
        expected_cols = {
            "pct_unchanged",
            "mean_abs_return",
            "source_count",
            "return_dispersion",
            "n_items",
        }
        assert expected_cols.issubset(set(features.columns))

    def test_pct_unchanged_range(self):
        df = _make_normal_prices()
        features = compute_date_features(df)
        assert (features["pct_unchanged"] >= 0).all()
        assert (features["pct_unchanged"] <= 1).all()

    def test_n_items_positive(self):
        df = _make_normal_prices()
        features = compute_date_features(df)
        assert (features["n_items"] > 0).all()


class TestFrozenDayDetection:
    def test_frozen_day_has_high_pct_unchanged(self):
        """Days with all-frozen prices should have pct_unchanged near 1."""
        frozen = _make_frozen_prices(n_items=30, n_days=10)
        features = compute_date_features(frozen)
        # After the first day, every day should be nearly 100% unchanged
        if len(features) > 0:
            assert features["pct_unchanged"].mean() > 0.9


class TestScoring:
    def test_normal_day_scores_higher_than_frozen(self):
        """Normal-market dates should score higher than frozen-feed dates."""
        normal = _make_normal_prices(n_items=50, n_days=60)
        frozen = _make_frozen_prices(n_items=50, n_days=20)

        # Combine: normal days then frozen days
        combined = pd.concat([normal, frozen], ignore_index=True)
        features = compute_date_features(combined)

        if len(features) < 10:
            pytest.skip("Not enough dates for meaningful test")

        model = fit_quality_model(features, contamination=0.1)
        scores = score_dates(model, features)

        # Split scores by era
        normal_dates = pd.date_range("2026-01-02", periods=58, freq="D")
        frozen_dates = pd.date_range("2026-06-02", periods=8, freq="D")

        normal_scores = scores.reindex(normal_dates).dropna()
        frozen_scores = scores.reindex(frozen_dates).dropna()

        if len(normal_scores) > 0 and len(frozen_scores) > 0:
            assert normal_scores.mean() > frozen_scores.mean(), (
                f"Normal mean {normal_scores.mean():.3f} should exceed "
                f"frozen mean {frozen_scores.mean():.3f}"
            )

    def test_score_range_is_zero_to_one(self):
        df = _make_normal_prices(n_items=30, n_days=40)
        features = compute_date_features(df)
        model = fit_quality_model(features, contamination=0.05)
        scores = score_dates(model, features)

        assert scores.min() >= -1e-9, f"Min score {scores.min()} below 0"
        assert scores.max() <= 1.0 + 1e-9, f"Max score {scores.max()} above 1"

    def test_fit_and_score_roundtrip(self):
        """fit_quality_model + score_dates should work end-to-end."""
        df = _make_normal_prices(n_items=20, n_days=30)
        features = compute_date_features(df)

        assert len(features) > 0, "Should produce date features"

        model = fit_quality_model(features, contamination=0.05)
        scores = score_dates(model, features)

        assert len(scores) == len(features)
        assert scores.name == "quality_score"
        assert not scores.isna().any(), "No NaN scores expected"
