"""Wiring tests: the volatility tag fields on PredictionOut and the /volatility route.

The substantive logic (swing, tertile label, ranking) is behaviour-tested in
test_volatility_tags.py. Here we pin only the wiring: the schema carries the
fields, and the ranking endpoint exists and routes through the tested helper.
"""
from __future__ import annotations

import inspect

from api.schemas import PredictionOut
import api.routes.items as items_mod


class TestPredictionOutCarriesTags:
    def _minimal(self, **kw):
        base = dict(item_id=1, item_name="X", current_price=100.0,
                    forecast_low=90.0, forecast_mid=100.0, forecast_high=110.0,
                    forecast_period="7_days", trend_direction="neutral")
        base.update(kw)
        return PredictionOut(**base)

    def test_tag_fields_default_none(self):
        p = self._minimal()
        assert p.expected_swing_pct is None
        assert p.move_odds is None
        assert p.stability_label is None

    def test_tag_fields_round_trip(self):
        p = self._minimal(expected_swing_pct=0.12, move_odds=0.30,
                          stability_label="Moderate")
        assert p.expected_swing_pct == 0.12
        assert p.move_odds == 0.30
        assert p.stability_label == "Moderate"


class TestVolatilityRoute:
    def test_endpoint_is_registered(self):
        paths = {r.path for r in items_mod.router.routes}
        assert "/items/volatility" in paths

    def test_route_uses_the_tested_ranking_helper(self):
        source = inspect.getsource(items_mod)
        assert "build_ranking" in source

    def test_route_applies_the_price_floor(self):
        # a volatility ranking over the served universe must respect the >=$1
        # serving floor, like the other universe-spanning routes
        rank_fn = items_mod.get_volatility_ranking
        assert "min_price" in inspect.signature(rank_fn).parameters


class TestPerItemEnrichment:
    def test_threshold_provider_exists(self):
        assert hasattr(items_mod, "_horizon_swing_thresholds")

    def test_both_prediction_paths_attach_tags(self):
        # both the parquet path and the DB fallback must enrich with tag_fields;
        # the no-forecast placeholder path deliberately does not (no real band)
        src = inspect.getsource(items_mod)
        assert "tag_fields" in src
        assert src.count("tag_fields(") >= 2
