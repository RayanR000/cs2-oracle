import copy
from datetime import date, datetime

import pytest


def _rows():
    return [
        {
            "item_id": 7,
            "current_price": 40.0,
            "gbm_low": 90.0,
            "gbm_mid": 100.0,
            "gbm_high": 125.0,
            "rank_score": 0.7,
        }
    ]


def _kwargs(**overrides):
    base = dict(
        forecast_date=date(2026, 9, 18),
        horizon_days=7,
        champion="gbm_q50",
        candidate_version="v1",
        artifact_version="a123",
        feature_cutoff_at=datetime(2026, 9, 18, 12, 0, 0),
        fingerprint_payload={
            "champion": "gbm_q50",
            "artifact_version": "a123",
            "offsets": [-10.0, 25.0],
            "feedback_factor": 1.0,
            "flags": {"RANKING_HEAD": "1"},
        },
    )
    base.update(overrides)
    return base


def test_last_price_shadow_uses_equal_geometry():
    from models.candidate_predictions import candidate_records

    records = candidate_records(_rows(), **_kwargs())
    centres = [r for r in records if r.component == "centre"]
    assert len(centres) == 1
    shadow = centres[0]
    assert shadow.candidate_name == "last_price"
    assert shadow.centre_price == pytest.approx(40.0)
    assert shadow.predicted_price_low == pytest.approx(36.0)
    assert shadow.predicted_price_high == pytest.approx(50.0)
    assert shadow.anchor_price == pytest.approx(40.0)
    assert shadow.forecast_date == date(2026, 9, 18)


def test_ranking_record_emitted_with_score():
    from models.candidate_predictions import candidate_records

    records = candidate_records(_rows(), **_kwargs())
    ranks = [r for r in records if r.component == "ranking"]
    assert len(ranks) == 1
    assert ranks[0].candidate_name == "lambdarank_v1"
    assert ranks[0].score == pytest.approx(0.7)
    assert ranks[0].centre_price is None


def test_input_rows_are_not_mutated():
    from models.candidate_predictions import candidate_records

    rows = _rows()
    snapshot = copy.deepcopy(rows)
    candidate_records(rows, **_kwargs())
    assert rows == snapshot


def test_no_ranking_record_without_finite_score():
    from models.candidate_predictions import candidate_records

    for bad in (None, float("nan")):
        rows = _rows()
        rows[0]["rank_score"] = bad
        records = candidate_records(rows, **_kwargs())
        assert [r for r in records if r.component == "ranking"] == []


def test_promoted_last_price_emits_gbm_shadow():
    from models.candidate_predictions import candidate_records

    records = candidate_records(_rows(), **_kwargs(champion="last_price"))
    centres = [r for r in records if r.component == "centre"]
    assert len(centres) == 1
    assert centres[0].candidate_name == "gbm_q50"
    assert centres[0].centre_price == pytest.approx(100.0)
    assert centres[0].predicted_price_low == pytest.approx(90.0)
    assert centres[0].predicted_price_high == pytest.approx(125.0)


def test_fingerprint_stable_under_key_reordering():
    from models.candidate_predictions import config_fingerprint

    payload = {"b": 1, "a": {"y": 2, "x": 1}}
    reordered = {"a": {"x": 1, "y": 2}, "b": 1}
    assert config_fingerprint(payload) == config_fingerprint(reordered)


def test_fingerprint_moves_with_policy_geometry_and_flags():
    from models.candidate_predictions import config_fingerprint

    base = {"champion": "gbm_q50", "artifact": "a", "offsets": [1.0], "feedback": 1.0, "flags": {"X": "1"}}
    variants = [
        {"champion": "last_price", "artifact": "a", "offsets": [1.0], "feedback": 1.0, "flags": {"X": "1"}},
        {"champion": "gbm_q50", "artifact": "b", "offsets": [1.0], "feedback": 1.0, "flags": {"X": "1"}},
        {"champion": "gbm_q50", "artifact": "a", "offsets": [2.0], "feedback": 1.0, "flags": {"X": "1"}},
        {"champion": "gbm_q50", "artifact": "a", "offsets": [1.0], "feedback": 0.9, "flags": {"X": "1"}},
        {"champion": "gbm_q50", "artifact": "a", "offsets": [1.0], "feedback": 1.0, "flags": {"X": "0"}},
    ]
    base_fp = config_fingerprint(base)
    assert len(base_fp) == 64
    for v in variants:
        assert config_fingerprint(v) != base_fp
