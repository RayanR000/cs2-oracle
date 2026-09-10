"""The served direction call is withheld: every horizon reports "neutral".

Withdrawn 2026-09-10 on the first backtest panel after the leg-window resolver
fix (run 34427386566). Pesaran-Timmermann on the >=$1 `lgbm-v3%` cohort is
NEGATIVE at every horizon -- -1.62pp (p=0.032, 17 dates), -0.66pp (p=0.193),
-0.54pp (p=0.617), -1.31pp (p=0.118) at 3/7/14/30d -- directional accuracy is
below a coin flip everywhere (43.8/48.2/48.9/45.3%), and a constant "down" call
beats the model at every horizon (55.0/58.1/65.2%). Two cells are
significant-negative: h=3 at >=$1, and h=7 at the >=$20 floor (-1.89pp,
p=0.037). No horizon and no price floor shows positive directional skill, so
the call is anti-informative rather than merely uninformative.

This mirrors the 2026-08-12 `confidence` withdrawal: the `direction` column is
still written by the forecaster and still scored as `pt_pp`/`da_pct`, it is
simply no longer disclosed. Republish only once a panel shows positive PT.
"""
from __future__ import annotations

import pytest

from api.serving_policy import SERVED_HORIZONS, served_direction


class TestServedDirection:
    @pytest.mark.parametrize("raw", ["up", "down", "flat", None, "bullish", ""])
    def test_every_raw_direction_is_neutralised(self, raw):
        assert served_direction(raw) == "neutral"

    @pytest.mark.parametrize("horizon", SERVED_HORIZONS)
    def test_no_horizon_discloses_a_call(self, horizon):
        # The gate is horizon-independent by construction: no horizon in the
        # panel showed positive PT, so there is no served-horizon tuple here
        # to drift out of sync with the evidence.
        assert served_direction("up", horizon) == "neutral"

    def test_routes_do_not_map_direction_themselves(self):
        """No route may rebuild the label from the stored column.

        ``_build_trend_explanation`` keeps its bullish/bearish branches on
        purpose (they are pinned by test_trend_explanation_copy and become
        live again the moment DIRECTION_DISCLOSED flips), so this guards the
        route-level mapping only.
        """
        from pathlib import Path
        import api.routes.items as items

        src = Path(items.__file__).read_text()
        assert "direction_map" not in src, (
            "a route is reconstructing the withheld direction label; "
            "route it through served_direction() instead"
        )
        assert 'direction or "neutral"' not in src
