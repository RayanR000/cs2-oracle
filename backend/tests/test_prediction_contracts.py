import numpy as np
import pytest

from models.prediction_contracts import CentrePrediction, RankingPrediction, SignedIntervalOffsets


def test_centre_requires_aligned_finite_positive_arrays():
    with pytest.raises(ValueError, match="same length"):
        CentrePrediction("x", "v1", np.array([1.0]), np.array([10.0, 11.0]))
    with pytest.raises(ValueError, match="positive"):
        CentrePrediction("x", "v1", np.array([0.0]), np.array([0.0]))


def test_offsets_require_ordered_finite_legs():
    with pytest.raises(ValueError, match="lower_pct"):
        SignedIntervalOffsets(np.array([5.0]), np.array([-5.0]))


def test_ranking_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        RankingPrediction("rank", "v1", np.array([np.nan]))


def test_centre_rejects_empty_names():
    with pytest.raises(ValueError, match="name"):
        CentrePrediction("", "v1", np.array([1.0]), np.array([10.0]))
    with pytest.raises(ValueError, match="version"):
        CentrePrediction("x", "", np.array([1.0]), np.array([10.0]))


def test_centre_rejects_nonfinite_prices():
    with pytest.raises(ValueError, match="finite"):
        CentrePrediction("x", "v1", np.array([1.0]), np.array([np.inf]))


def test_offsets_reject_unequal_and_nonfinite():
    with pytest.raises(ValueError, match="same length"):
        SignedIntervalOffsets(np.array([1.0, 2.0]), np.array([3.0]))
    with pytest.raises(ValueError, match="finite"):
        SignedIntervalOffsets(np.array([np.nan]), np.array([1.0]))


def test_ranking_rejects_empty_names():
    with pytest.raises(ValueError, match="name"):
        RankingPrediction("", "v1", np.array([0.5]))
