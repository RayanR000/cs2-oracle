import numpy as np
import pytest
from models.forecast_assembly import assemble_interval, percentage_offsets


def test_percentage_offsets_reproduce_legs():
    offsets = percentage_offsets(low=np.array([90.0]), mid=np.array([100.0]), high=np.array([125.0]))
    np.testing.assert_allclose(offsets.lower_pct, np.array([-10.0]))
    np.testing.assert_allclose(offsets.upper_pct, np.array([25.0]))


def test_assemble_rejects_invalid_mid():
    from models.prediction_contracts import SignedIntervalOffsets

    offsets = SignedIntervalOffsets(lower_pct=np.array([-10.0]), upper_pct=np.array([10.0]))
    with pytest.raises(ValueError, match="positive"):
        assemble_interval(np.array([0.0]), offsets)


def test_assemble_rejects_unordered_offsets():
    import numpy as np
    from models.forecast_assembly import assemble_interval
    from models.prediction_contracts import SignedIntervalOffsets

    # Bypass constructor validation to test assembly-time ordering check
    offsets = SignedIntervalOffsets.__new__(SignedIntervalOffsets)
    object.__setattr__(offsets, "lower_pct", np.array([5.0]))
    object.__setattr__(offsets, "upper_pct", np.array([-5.0]))
    with pytest.raises(ValueError, match="lower_pct"):
        assemble_interval(np.array([100.0]), offsets)
