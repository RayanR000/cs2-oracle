import copy
from datetime import date

import pandas as pd


def _results():
    return pd.DataFrame(
        [
            {
                "item_id": "slug-a",
                "current_price": 40.0,
                "anchor_date": date(2026, 9, 18),
                "forecasts": {
                    7: {
                        "low": 90.0,
                        "mid": 100.0,
                        "high": 125.0,
                        "direction": "flat",
                        "confidence": "low",
                        "exceed_p": None,
                        "anomaly_p": None,
                        "rank_score": 0.7,
                    }
                },
            }
        ]
    )


def test_initial_policy_keeps_gbm_public_and_last_price_shadow():
    from models.candidate_predictions import apply_centre_policy

    public, shadow = apply_centre_policy(_results())
    assert public.iloc[0].forecasts[7]["mid"] == 100.0
    assert shadow[0].candidate_name == "last_price"


def test_promoted_policy_swaps_public_and_shadow(monkeypatch):
    from models.candidate_predictions import apply_centre_policy
    from models.centre_policy import CENTRE_CHAMPIONS

    monkeypatch.setitem(CENTRE_CHAMPIONS, 7, "last_price")
    public, shadow = apply_centre_policy(_results())
    assert public.iloc[0].forecasts[7]["mid"] == 40.0
    assert shadow[0].candidate_name == "gbm_q50"
    assert shadow[0].centre_price == 100.0


def test_policy_does_not_mutate_its_input():
    from models.candidate_predictions import apply_centre_policy

    results = _results()
    snapshot = copy.deepcopy(results.iloc[0].forecasts)
    apply_centre_policy(results)
    assert results.iloc[0].forecasts == snapshot


def test_writer_receives_no_ranking_field():
    import inspect

    import scripts.forecast_prices as fp

    assert "rank_score" not in inspect.getsource(fp._write_forecasts_to_db)


def test_public_schemas_carry_no_shadow():
    from pathlib import Path

    import database

    assert "rank_score" not in set(database.ItemForecast.__table__.columns.keys())
    schemas = (Path(__file__).resolve().parent.parent / "api" / "schemas.py").read_text()
    assert "rank_score" not in schemas
