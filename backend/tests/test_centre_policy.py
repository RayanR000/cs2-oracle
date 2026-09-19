import numpy as np
import pytest
from models.centre_policy import CENTRE_CHAMPIONS, centre_challengers, centre_champion
from models.forecast_assembly import assemble_interval, percentage_offsets


def test_initial_champion_is_gbm_at_every_horizon():
    from models.centre_policy import HORIZONS

    assert CENTRE_CHAMPIONS == {3: "gbm_q50", 7: "gbm_q50", 14: "gbm_q50", 30: "gbm_q50"}
    assert all(centre_challengers(h) == ("last_price",) for h in HORIZONS)


def test_two_centres_receive_identical_percentage_geometry():
    offsets = percentage_offsets(low=np.array([90.0]), mid=np.array([100.0]), high=np.array([125.0]))
    low_a, mid_a, high_a = assemble_interval(np.array([100.0]), offsets)
    low_b, mid_b, high_b = assemble_interval(np.array([40.0]), offsets)
    np.testing.assert_allclose(low_a / mid_a, low_b / mid_b)
    np.testing.assert_allclose(high_a / mid_a, high_b / mid_b)


def test_unknown_horizon_rejected():
    with pytest.raises(ValueError, match="unsupported horizon"):
        centre_champion(99)
    with pytest.raises(ValueError, match="unsupported horizon"):
        centre_challengers(99)


def test_nonpositive_centre_rejected():
    offsets = percentage_offsets(low=np.array([90.0]), mid=np.array([100.0]), high=np.array([110.0]))
    with pytest.raises(ValueError, match="positive"):
        assemble_interval(np.array([0.0]), offsets)
