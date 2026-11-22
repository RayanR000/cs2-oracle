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
    from models.candidate_predictions import CENTRE_CHAMPIONS, apply_centre_policy

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

    assert "rank_score" not in inspect.getsource(fp._write_forecasts)


def test_public_schemas_carry_no_shadow():

    import database

    assert "rank_score" not in set(database.ItemForecast.__table__.columns.keys())
    # No public schemas since the API was retired (Phase 4 PR 5); the served store is the
    # item_forecasts partitions, whose schema must not carry it either.
    from db.ops_partitions import SPECS

    assert "rank_score" not in SPECS["item_forecasts"].columns
