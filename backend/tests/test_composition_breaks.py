"""_composition_break_dates: a market-wide switch in which sources vote.

The item-count rule (_collection_shift_dates) cannot see 2026-04-16: every
train-universe item switched from buff163/csfloat/youpin to steam_17mafo with
the item count flat (+0.02%). Nor, on the train universe, 2026-01-01, where the
NULL-sourced legacy series ends. See
docs/specs/2026-09-30-composition-break-calendar-design.md.
"""

import json
import logging
from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
from models.forecaster import ItemForecaster

D0 = date(2026, 4, 14)


def _frame(n_items: int, sets_by_day: list[list[str]]) -> pd.DataFrame:
    rows = []
    for k, sets in enumerate(sets_by_day):
        for i in range(n_items):
            rows.append({"item_id": f"i{i}", "date": D0 + timedelta(days=k), "price": 1.0, "source_set": sets[i % len(sets)]})
    df = pd.DataFrame(rows)
    df["source_set"] = df["source_set"].astype("category")
    return df


def test_full_switch_fires_on_the_switch_day_only():
    df = _frame(100, [["a|b"], ["a|b"], ["c"], ["c"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset({D0 + timedelta(days=2)})


def test_ordinary_churn_does_not_fire():
    # ~56% of items change set: the highest ordinary day measured on the train universe.
    day1 = ["a|b"] * 100
    day2 = ["a"] * 56 + ["a|b"] * 44
    assert ItemForecaster._composition_break_dates(_frame(100, [day1, day2])) == frozenset()


def test_ninety_percent_is_the_boundary():
    day1 = ["a"] * 100
    at = ["b"] * 90 + ["a"] * 10
    below = ["b"] * 89 + ["a"] * 11
    assert ItemForecaster._composition_break_dates(_frame(100, [day1, at])) == frozenset({D0 + timedelta(days=1)})
    assert ItemForecaster._composition_break_dates(_frame(100, [day1, below])) == frozenset()


def test_small_cross_section_never_fires():
    assert ItemForecaster._composition_break_dates(_frame(24, [["a"], ["b"]])) == frozenset()


def test_null_sourced_history_is_one_composition():
    df = _frame(100, [["<null>"], ["<null>"], ["<null>"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_the_null_to_labelled_year_boundary_fires():
    df = _frame(100, [["<null>"], ["<null>"], ["aggregator_sync"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset({D0 + timedelta(days=2)})


def test_new_items_do_not_count_as_changed():
    # Day 2 adds 1,000 items that did not exist on day 1; only the 100 paired items are judged.
    paired = _frame(100, [["a"], ["a"]])
    newcomers = _frame(1000, [["a"], ["b"]])
    newcomers = newcomers[newcomers["date"] == D0 + timedelta(days=1)].assign(item_id=lambda d: "new" + d["item_id"])
    df = pd.concat([paired, newcomers], ignore_index=True)
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_missing_column_is_empty_not_an_error():
    df = _frame(100, [["a"], ["b"]]).drop(columns=["source_set"])
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_a_price_crash_with_a_stable_source_set_does_not_fire():
    df = _frame(100, [["a|b"], ["a|b"]])
    df.loc[df["date"] == D0 + timedelta(days=1), "price"] = 0.5
    assert ItemForecaster._composition_break_dates(df) == frozenset()


# -- the calendar: fetch -> prepare_targets -> meta.json ----------------------


def _priced(n_items: int, days: int, switch_day: int | None) -> pd.DataFrame:
    rows = []
    for k in range(days):
        s = "c" if switch_day is not None and k >= switch_day else "a|b"
        for i in range(n_items):
            rows.append(
                {"item_id": f"i{i}", "date": D0 + timedelta(days=k), "price": 10.0 + i + 0.01 * k, "volume": 1.0, "source_set": s}
            )
    df = pd.DataFrame(rows)
    df["source_set"] = df["source_set"].astype("category")
    return df


def _fc(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _saveable(f):
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    return f


def test_record_break_dates_is_the_union(tmp_path):
    fc = _fc(tmp_path)
    fc._record_break_dates(_priced(100, 6, switch_day=3))
    assert fc.break_dates == frozenset({D0 + timedelta(days=3)})
    assert fc.composition_break_dates == frozenset({D0 + timedelta(days=3)})
    assert fc.collection_shift_dates == frozenset()


def test_label_spanning_a_composition_break_is_voided(tmp_path):
    fc = _fc(tmp_path)
    df = _priced(100, 10, switch_day=5)
    fc._record_break_dates(df)
    out = fc.prepare_targets(df.drop(columns=["source_set"]), horizon=3)
    anchor = pd.to_datetime(out["date"])
    b = pd.Timestamp(D0 + timedelta(days=5))
    spans = (anchor < b) & (b <= anchor + pd.to_timedelta(3, unit="D"))
    assert spans.any()
    assert out.loc[spans, "target_return_3d"].isna().all()
    assert out.loc[~spans & out["target_3d"].notna(), "target_return_3d"].notna().all()
    assert fc.label_voiding["composition_break_dates"] == [str(D0 + timedelta(days=5))]
    assert fc.label_voiding["break_dates"] == [str(D0 + timedelta(days=5))]
    assert fc.label_voiding["collection_shift_dates"] == []


def test_without_a_fetch_prepare_targets_falls_back_and_warns(tmp_path, caplog):
    fc = _fc(tmp_path)
    assert fc.break_dates is None
    with caplog.at_level(logging.WARNING, logger="models.forecaster"):
        out = fc.prepare_targets(_priced(100, 10, switch_day=5).drop(columns=["source_set"]), horizon=3)
    assert "item-count rule only" in caplog.text
    assert out["target_return_3d"].notna().any()  # the composition switch is NOT voided on this path


def test_break_dates_round_trip_through_meta_json(tmp_path):
    f = _saveable(_fc(tmp_path))
    f.break_dates = frozenset({date(2026, 4, 16)})
    f.composition_break_dates = frozenset({date(2026, 4, 16)})
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["break_dates"] == ["2026-04-16"]
    assert meta["composition_break_dates"] == ["2026-04-16"]
    g = _fc(tmp_path)
    g.load_models()
    assert g._artifact_break_dates == frozenset({date(2026, 4, 16)})


def test_artifact_without_break_dates_loads_empty(tmp_path):
    f = _saveable(_fc(tmp_path))
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta.pop("break_dates", None)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    g = _fc(tmp_path)
    g.load_models()
    assert g._artifact_break_dates == frozenset()


def test_an_artifact_with_the_retired_shrink_k_and_vol_rank_keys_still_loads(tmp_path):
    f = _fc(tmp_path)
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta.update({"shrink_k_gbm": True, "vol_rank_gbm": True, "vol_rank_norm": {"3": 1.2}})
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "shrink_k_3d.txt").write_text("not a booster")
    (tmp_path / "vol_rank_3d_e0.txt").write_text("not a booster")
    g = _fc(tmp_path)
    g.load_models()  # must not raise, must not try to parse the stale files
    assert not hasattr(g, "shrink_k_models") and not hasattr(g, "vol_rank_models")


def test_save_removes_stale_retired_booster_files(tmp_path):
    (tmp_path / "shrink_k_7d.txt").write_text("x")
    (tmp_path / "vol_rank_7d_e1.txt").write_text("x")
    f = _fc(tmp_path)
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    assert not list(tmp_path.glob("shrink_k_*.txt")) and not list(tmp_path.glob("vol_rank_*.txt"))
