"""Wiring tests: the volatility tag fields on PredictionOut and the /volatility route.

The substantive logic (swing, tertile label, ranking) is behaviour-tested in
test_volatility_tags.py. Here we pin only the wiring: the schema carries the
fields, and the ranking endpoint exists and routes through the tested helper.
"""

from __future__ import annotations

import inspect

import api.routes.items as items_mod
import pytest
from api.schemas import PredictionOut
from fastapi import HTTPException


class TestPredictionOutCarriesTags:
    def _minimal(self, **kw):
        base = dict(
            item_id=1,
            item_name="X",
            current_price=100.0,
            forecast_low=90.0,
            forecast_mid=100.0,
            forecast_high=110.0,
            forecast_period="7_days",
            trend_direction="neutral",
        )
        base.update(kw)
        return PredictionOut(**base)

    def test_tag_fields_default_none(self):
        p = self._minimal()
        assert p.expected_swing_pct is None
        assert p.move_odds is None
        assert p.stability_label is None

    def test_tag_fields_round_trip(self):
        p = self._minimal(expected_swing_pct=0.12, move_odds=0.30, stability_label="Moderate")
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

    def test_label_is_an_optional_query_param(self):
        # the discovery filter: request only one stability class
        param = inspect.signature(items_mod.get_volatility_ranking).parameters.get("label")
        assert param is not None
        assert param.default.default is None  # optional, defaults to no filter

    def test_move_odds_sort_rejected_at_uncalibrated_horizon(self):
        # sorting by move_odds at h30 asks the API to rank on a probability that
        # is not calibrated there -- reject rather than serve a misleading order.
        # The guard runs before any DB access, so db=None is safe here.
        with pytest.raises(HTTPException) as exc:
            items_mod.get_volatility_ranking(
                horizon=30, sort="move_odds", order="desc", min_price=1.0, limit=10, db=None
            )
        assert exc.value.status_code == 400

    def test_unserved_horizon_rejected(self):
        # only 3/7/14/30 are trained+served; any other horizon returns no rows,
        # so reject it explicitly instead of a silently-empty ranking.
        # The guard runs before any DB access, so db=None is safe here.
        with pytest.raises(HTTPException) as exc:
            items_mod.get_volatility_ranking(horizon=5, sort="swing", order="desc", min_price=1.0, limit=10, db=None)
        assert exc.value.status_code == 400

    def test_served_horizon_is_accepted_past_the_guard(self):
        # a served horizon must not be rejected by the horizon guard; it fails
        # later on db=None (an AttributeError/TypeError), NOT an HTTPException 400.
        with pytest.raises(Exception) as exc:
            items_mod.get_volatility_ranking(horizon=14, sort="swing", order="desc", min_price=1.0, limit=10, db=None)
        assert not (isinstance(exc.value, HTTPException) and exc.value.status_code == 400)

    def test_route_gates_move_odds_by_calibration(self):
        # the ranking path must thread the calibration flag through, not publish
        # move_odds unconditionally
        src = inspect.getsource(items_mod)
        assert "move_odds_calibrated" in src


class TestPerItemEnrichment:
    def test_threshold_provider_exists(self):
        assert hasattr(items_mod, "_horizon_swing_thresholds")

    def test_both_prediction_paths_attach_tags(self):
        # both the parquet path and the DB fallback must enrich with tag_fields;
        # the no-forecast placeholder path deliberately does not (no real band)
        src = inspect.getsource(items_mod)
        assert "tag_fields" in src
        assert src.count("tag_fields(") >= 2

    def test_both_prediction_paths_gate_move_odds_by_horizon(self):
        # both tag_fields calls must pass the per-horizon calibration flag so a
        # h14/h30 lookup does not publish an uncalibrated move_odds
        src = inspect.getsource(items_mod)
        assert src.count("calibrated_move_odds=") >= 2
