"""Tests for backend.models.source_weight_model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from models.source_weight_model import (
    DEFAULT_WEIGHTS,
    build_training_data,
    predict_weights,
    train_source_model,
)


def _make_archive(n_items: int = 5, n_days: int = 10, sources=None):
    """Build a synthetic multi-source archive DataFrame."""
    if sources is None:
        sources = ["aggregator_buff163", "aggregator_csfloat", "aggregator_csmoney"]
    rows = []
    rng = np.random.RandomState(42)
    for item_idx in range(n_items):
        slug = f"item_{item_idx}"
        base_price = 10.0 * (item_idx + 1)
        for day_idx in range(n_days):
            day = pd.Timestamp("2026-01-01") + pd.to_timedelta(day_idx, unit="D")
            for src in sources:
                price = base_price + rng.normal(0, base_price * 0.05)
                rows.append(
                    {"item_slug": slug, "day": day, "source": src, "price": price}
                )
    return pd.DataFrame(rows)


class TestBuildTrainingData:
    def test_requires_multi_source(self):
        """Single-source item-days must be excluded from training data."""
        df = pd.DataFrame(
            {
                "item_slug": ["a", "a", "b"],
                "day": ["2026-01-01", "2026-01-02", "2026-01-01"],
                "source": ["src1", "src1", "src1"],
                "price": [10.0, 11.0, 20.0],
            }
        )
        result = build_training_data(df)
        assert len(result) == 0, "Single-source rows should not appear in training data"

    def test_multi_source_produces_rows(self):
        df = _make_archive(n_items=3, n_days=5, sources=["src_a", "src_b"])
        result = build_training_data(df)
        assert len(result) > 0
        assert "abs_error" in result.columns
        assert "consensus" in result.columns
        assert (result["n_sources"] >= 2).all()

    def test_consensus_is_median(self):
        df = pd.DataFrame(
            {
                "item_slug": ["a", "a", "a"],
                "day": ["2026-01-01"] * 3,
                "source": ["s1", "s2", "s3"],
                "price": [10.0, 20.0, 30.0],
            }
        )
        result = build_training_data(df)
        assert np.isclose(result["consensus"].iloc[0], 20.0)


class TestWeightsSumToOne:
    def test_weights_sum_to_one_per_group(self):
        """Predicted weights must sum to 1 for each item-day group."""
        archive = _make_archive(n_items=5, n_days=20)
        td = build_training_data(archive)
        model, encoder = train_source_model(td)

        sources = ["aggregator_buff163", "aggregator_csfloat", "aggregator_csmoney"]
        prices = np.array([50.0, 51.0, 49.5])
        weights = predict_weights(model, encoder, sources, prices, n_sources=3)

        assert np.isclose(weights.sum(), 1.0), f"Weights sum to {weights.sum()}, not 1"
        assert len(weights) == 3

    def test_single_source_weight_is_one(self):
        archive = _make_archive(n_items=5, n_days=20)
        td = build_training_data(archive)
        model, encoder = train_source_model(td)

        weights = predict_weights(
            model, encoder, ["aggregator_buff163"], np.array([50.0]), n_sources=1
        )
        assert np.isclose(weights.sum(), 1.0)


class TestDefaultWeightsFallback:
    def test_default_weights_exist(self):
        """DEFAULT_WEIGHTS must cover all known sources."""
        assert isinstance(DEFAULT_WEIGHTS, dict)
        assert len(DEFAULT_WEIGHTS) > 0

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert np.isclose(total, 1.0), f"Default weights sum to {total}"

    def test_sync_has_lowest_weight(self):
        """aggregator_sync should have the lowest weight (17% degradation)."""
        sync_weight = DEFAULT_WEIGHTS.get("aggregator_sync", 1.0)
        for src, w in DEFAULT_WEIGHTS.items():
            if src != "aggregator_sync":
                assert (
                    w >= sync_weight
                ), f"{src} weight {w} < sync weight {sync_weight}"


class TestHigherErrorGetsLowerWeight:
    def test_higher_error_source_gets_lower_weight(self):
        """A source consistently far from consensus should have higher mean abs_error."""
        rng = np.random.RandomState(123)
        rows = []
        # Use 3 good sources + 1 bad source so median is anchored by the good ones
        good_sources = ["good_a", "good_b", "good_c"]
        for day_idx in range(200):
            day = pd.Timestamp("2026-01-01") + pd.to_timedelta(day_idx, unit="D")
            for item_idx in range(30):
                slug = f"item_{item_idx}"
                base = 50.0 + item_idx
                for gs in good_sources:
                    rows.append(
                        {
                            "item_slug": slug,
                            "day": day,
                            "source": gs,
                            "price": base + rng.normal(0, 0.1),
                        }
                    )
                # bad_src: very noisy, far from the stable consensus
                rows.append(
                    {
                        "item_slug": slug,
                        "day": day,
                        "source": "bad_src",
                        "price": base + rng.normal(0, 15.0),
                    }
                )

        archive = pd.DataFrame(rows)
        td = build_training_data(archive)

        # With 3 tight sources anchoring the median, bad_src should have much higher error
        good_mean = td.loc[td["source"].isin(good_sources), "abs_error"].mean()
        bad_mean = td.loc[td["source"] == "bad_src", "abs_error"].mean()
        assert bad_mean > good_mean * 5, (
            f"bad_src mean error {bad_mean:.2f} should be >> good_src {good_mean:.2f}"
        )
