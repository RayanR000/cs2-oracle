"""The climatological band scale (CLIMATOLOGY_SCALE=1).

A per-item h-day return dispersion, shrunk toward the price tier, served as a
conformal `learned_scale`. Measured 43-47% narrower than sigma at matched
coverage (docs/research/2026-08-19-climatology-vs-gbm-band.md). These tests pin
the scale's construction, its fallbacks, the mutual-exclusion guard, and the
artifact round-trip so the served band matches the calibrated one.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


@pytest.fixture
def fc(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(),
                          model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _frame(tcol="target_return_7d"):
    """A volatile item, a calm item, and a thin (1-obs) item, all tier 3."""
    rng = np.random.default_rng(0)
    rows = []
    for iid, scale, n in [("vol", 30.0, 200), ("calm", 2.0, 200), ("thin", 0.0, 1)]:
        for _ in range(n):
            rows.append({"item_id": iid, "price": 50.0,
                         tcol: float(rng.normal(0, scale)) if scale else 0.0})
    return pd.DataFrame(rows)


class TestTable:
    def test_dispersion_orders_items(self, fc):
        table, _, _ = fc._build_climatology_table(_frame(), "target_return_7d")
        assert table["vol"] > table["calm"], "a volatile item gets a wider scale"

    def test_thin_item_is_shrunk_to_the_tier_pool(self, fc):
        table, tier_pool, _ = fc._build_climatology_table(
            _frame(), "target_return_7d")
        # 1 obs => weight 1/(1+K) ~ 0.05, so it sits essentially at the pool.
        assert table["thin"] == pytest.approx(tier_pool[3], rel=0.1)

    def test_empty_frame_returns_nothing(self, fc):
        df = pd.DataFrame({"item_id": [], "price": [], "target_return_7d": []})
        table, tier_pool, g = fc._build_climatology_table(df, "target_return_7d")
        assert table == {} and tier_pool == {} and np.isnan(g)


class TestLookup:
    def test_unseen_item_falls_back_to_tier_then_global(self, fc):
        fc.climatology_scale[7] = {
            "table": {"known": 5.0}, "tier_pool": {3: 9.0}, "global": 99.0}
        out = fc._climatology_lookup(
            7, np.array(["known", "unseen_tier3", "unseen_tier0"]),
            np.array([50.0, 50.0, 0.5]))  # tier 3, tier 3, tier 0 (no pool)
        assert list(out) == [5.0, 9.0, 99.0]

    def test_no_table_returns_none(self, fc):
        assert fc._climatology_lookup(7, np.array(["x"]), np.array([1.0])) is None


class TestGating:
    def test_disabled_returns_none(self, fc, monkeypatch):
        # On by default since 2026-08-19; only an explicit "0" disables.
        monkeypatch.setenv("CLIMATOLOGY_SCALE", "0")
        recs = pd.DataFrame({"row_index": [0], "residual_pct": [1.0],
                             "sigma": [0.1], "mid_ret": [0.0]})
        assert fc._fit_climatology_scale(7, recs, _frame()) is None

    def test_enabled_builds_and_returns_per_record_scale(self, fc, monkeypatch):
        monkeypatch.setenv("CLIMATOLOGY_SCALE", "1")
        frame = _frame().reset_index(drop=True)
        recs = pd.DataFrame({"row_index": [0, len(frame) - 1]})
        scale = fc._fit_climatology_scale(7, recs, frame)
        assert scale is not None and scale.shape == (2,)
        assert 7 in fc.climatology_scale
        # row 0 is the volatile item, last row is the thin item.
        assert scale[0] > scale[1]

    def test_missing_columns_falls_back(self, fc, monkeypatch):
        monkeypatch.setenv("CLIMATOLOGY_SCALE", "1")
        frame = _frame().drop(columns=["price"])
        recs = pd.DataFrame({"row_index": [0]})
        assert fc._fit_climatology_scale(7, recs, frame) is None


class TestMutualExclusion:
    def test_raises_beside_sigma_exponent(self, fc, monkeypatch):
        monkeypatch.setenv("CLIMATOLOGY_SCALE", "1")
        monkeypatch.setenv("SIGMA_EXPONENT", "1")
        recs = pd.DataFrame({"residual_pct": [1.0, 2.0], "sigma": [0.1, 0.2],
                             "mid_ret": [0.0, 0.0], "row_index": [0, 1]})
        with pytest.raises(RuntimeError, match="CLIMATOLOGY_SCALE"):
            fc._calibrate_conformal(7, recs, _frame())


class TestServing:
    def test_band_scale_serves_the_table_when_the_artifact_says_so(self, fc):
        fc._artifact_climatology_scale = True
        fc.climatology_scale[7] = {
            "table": {"a": 4.0}, "tier_pool": {1: 8.0}, "global": 50.0}
        rows = pd.DataFrame({"item_id": ["a", "b"], "price": [2.0, 2.0]})
        out = fc.band_scale(7, rows, sigma=np.array([0.1, 0.1]))
        assert list(out) == [4.0, 8.0]  # known item, then tier-1 pool

    def test_artifact_flag_overrides_env(self, fc, monkeypatch):
        # Artifact says the band was NOT calibrated on climatology, so serving
        # must ignore CLIMATOLOGY_SCALE=1 in the env (matched-pair rule).
        monkeypatch.setenv("CLIMATOLOGY_SCALE", "1")
        fc._artifact_climatology_scale = False
        assert fc._climatology_scale_served() is False


def test_meta_round_trip_rebuilds_int_keyed_tables(fc, monkeypatch, tmp_path):
    monkeypatch.setenv("CLIMATOLOGY_SCALE", "1")
    fc.climatology_scale[7] = {
        "table": {"a": 4.0}, "tier_pool": {3: 9.0}, "global": 50.0}
    import json
    meta = {"climatology_scale": True,
            "climatology_scale_tables": {
                "7": fc.climatology_scale[7]}}
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    reloaded = json.loads(path.read_text())
    # Simulate the load path's reconstruction.
    fc._artifact_climatology_scale = reloaded.get("climatology_scale")
    fc.climatology_scale = {
        int(h): {"table": dict(cfg["table"]),
                 "tier_pool": {int(t): float(v) for t, v in cfg["tier_pool"].items()},
                 "global": float(cfg["global"])}
        for h, cfg in reloaded["climatology_scale_tables"].items()}
    assert 7 in fc.climatology_scale
    assert fc.climatology_scale[7]["tier_pool"] == {3: 9.0}
    assert fc._climatology_scale_served() is True
