"""BREAK_AWARE_LOOKBACKS: a feature whose lookback spans a source break is NaN.

docs/specs/2026-09-30-composition-break-calendar-design.md, Part B. Off by
default and unmeasured; these tests pin the rule, the window table's coverage,
and train/serve parity.
"""

import datetime as dt
import json
import tempfile

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster, _feature_group
from models.lookback_windows import FEATURE_LOOKBACK_DAYS, mask_break_lookbacks

from tests._source import method_closure_source

B = dt.date(2026, 7, 9)


def _rows(days):
    return pd.DataFrame(
        {
            "date": [B + dt.timedelta(days=k) for k in days],
            "return_1d": 1.0,
            "return_7d": 1.0,
            "price_dist_ma200": 1.0,
            "price_tier": 2,
        }
    )


def test_window_spanning_a_break_is_nan_and_others_are_untouched():
    df = _rows([-1, 0, 1, 6, 7, 199, 200])
    n = mask_break_lookbacks(df, ["return_1d", "return_7d", "price_dist_ma200", "price_tier"], frozenset({B}))
    by_day = df.set_index(df["date"].map(lambda d: (d - B).days))
    assert by_day.loc[[-1, 1], "return_1d"].notna().all() and np.isnan(by_day.loc[0, "return_1d"])
    assert by_day.loc[[0, 1, 6], "return_7d"].isna().all() and by_day.loc[[-1, 7], "return_7d"].notna().all()
    assert by_day.loc[[0, 199], "price_dist_ma200"].isna().all()
    assert by_day.loc[[-1, 200], "price_dist_ma200"].notna().all()
    assert by_day["price_tier"].notna().all()  # window 0: a level, not a lookback
    assert n == 1 + 3 + 5


def test_no_breaks_is_a_no_op():
    df = _rows([0, 1])
    assert mask_break_lookbacks(df, ["return_7d"], frozenset()) == 0
    assert df["return_7d"].notna().all()


def test_unknown_column_raises():
    with pytest.raises(KeyError, match="FEATURE_LOOKBACK_DAYS"):
        mask_break_lookbacks(_rows([0]).assign(new_feat=1.0), ["new_feat"], frozenset({B}))


def test_table_covers_every_price_technical_engineer_features_emits():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        p = 10.0
        for k in range(400):
            p *= 1 + rng.normal(0, 0.02)
            ts = pd.Timestamp("2025-01-01") + pd.Timedelta(days=k)
            rows.append({"item_id": f"i{i}", "date": ts.date(), "timestamp": ts, "price": p, "volume": 5.0})
    fc = ItemForecaster(db_session=None, model_dir=tempfile.mkdtemp())
    events = pd.DataFrame(columns=["id", "type", "timestamp", "description"])
    out = fc.engineer_features(pd.DataFrame(rows), events, skip_unused_groups=True)
    emitted = {c for c in out.columns if _feature_group(c) == "price_technicals"}
    assert emitted, "engineer_features emitted no price_technicals columns"
    assert emitted <= set(FEATURE_LOOKBACK_DAYS), sorted(emitted - set(FEATURE_LOOKBACK_DAYS))


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("BREAK_AWARE_LOOKBACKS", raising=False)
    assert ItemForecaster.break_aware_lookbacks_enabled() is False


def test_serving_follows_the_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("BREAK_AWARE_LOOKBACKS", "1")
    fc = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    fc._artifact_break_aware_lookbacks = False
    assert fc._break_aware_lookbacks_served() is False


def test_flag_round_trips_through_meta_json(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("BREAK_AWARE_LOOKBACKS", "1")
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    assert json.loads((tmp_path / "meta.json").read_text())["break_aware_lookbacks"] is True
    monkeypatch.delenv("BREAK_AWARE_LOOKBACKS")
    g = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    g.load_models()
    assert g._break_aware_lookbacks_served() is True


def test_train_and_predict_both_apply_the_mask():
    assert "_apply_break_mask" in method_closure_source(ItemForecaster, "build_training_data")
    assert "_apply_break_mask" in method_closure_source(ItemForecaster, "_prepare_predict_features")


def test_predict_on_a_cache_hit_uses_the_artifact_calendar(tmp_path):
    fc = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    fc._artifact_break_aware_lookbacks = True
    fc._artifact_break_dates = frozenset({B})
    fc.break_dates = None  # cache hit: no fetch ran
    df = _rows([0, 1])
    fc._apply_break_mask(df, ["return_7d"], served=True)
    assert df["return_7d"].isna().all()


def test_mask_off_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("BREAK_AWARE_LOOKBACKS", raising=False)
    fc = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    fc.break_dates = frozenset({B})
    df = _rows([0, 1])
    fc._apply_break_mask(df, ["return_7d"], served=False)
    assert df["return_7d"].notna().all()
