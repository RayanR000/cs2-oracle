"""`_upsert_accuracy` writes `ops/prediction_accuracy.parquet` and nothing else (Phase 4).

It used to upsert Postgres too, after the Parquet copy; the order tests that guarded that went
with Postgres. What still matters: the NULL tier and the headline land side by side, a re-run of
the same evaluation date replaces rather than grows, and the caller's row dicts survive.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd


def _row(tier, mae):
    return {
        "prediction_type": "forecast",
        "evaluation_date": date(2026, 8, 5),
        "horizon_days": 7,
        "model_version": "lgbm-v3",
        "price_tier": tier,
        "evaluation_window_days": None,
        "sample_count": 100,
        "metrics": {"mae": mae, "skill_vs_baseline": None},
        "created_at": datetime(2026, 8, 5, 12, 0, 0),
    }


def _mirror(tmp_path):
    return pd.read_parquet(tmp_path / "prediction_accuracy.parquet")


def test_the_null_tier_and_the_headline_both_land(tmp_path, monkeypatch):
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    import scripts.backtest_accuracy as ba

    # The all-tiers aggregate (NULL tier) and the >=$1 headline together, since
    # the NULL key is what the dedup used to mishandle.
    ba._upsert_accuracy([_row(None, 1.0), _row(-1, 2.0)])

    mirror = _mirror(tmp_path)
    assert len(mirror) == 2
    by_tier = {
        (None if pd.isna(r.price_tier) else int(r.price_tier)): json.loads(r.metrics) for r in mirror.itertuples()
    }
    assert by_tier[None]["mae"] == 1.0
    assert by_tier[-1]["mae"] == 2.0
    assert by_tier[None]["skill_vs_baseline"] is None


def test_a_rerun_replaces_rather_than_grows(tmp_path, monkeypatch):
    """The daily run rescores the same evaluation_date; the file may not grow."""
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    import scripts.backtest_accuracy as ba

    ba._upsert_accuracy([_row(None, 1.0), _row(-1, 2.0)])
    ba._upsert_accuracy([_row(None, 9.0), _row(-1, 8.0)])

    mirror = _mirror(tmp_path)
    assert len(mirror) == 2, f"the file duplicated instead of replacing:\n{mirror}"
    assert sorted(json.loads(m)["mae"] for m in mirror["metrics"]) == [8.0, 9.0]


def test_no_rows_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    import scripts.backtest_accuracy as ba

    ba._upsert_accuracy([])
    assert not (tmp_path / "prediction_accuracy.parquet").exists()


def test_the_caller_s_row_dicts_are_not_mutated(tmp_path, monkeypatch):
    """The write serialises metrics to JSON; the scorer's dicts must stay dicts."""
    monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
    import scripts.backtest_accuracy as ba

    rows = [_row(None, 1.0)]
    ba._upsert_accuracy(rows)
    assert isinstance(rows[0]["metrics"], dict), "the caller's dict was stringified"
