"""The ranked surfaces show only items whose anchor quote is its own median.

Measured, not assumed. On a fresh artifact trained in CI and replayed at four
non-overlapping anchors, served rank IC on the **tied** cohort -- items where
the latest quote equals its local median -- is +0.1321 / +0.1562 / +0.1747 at
3/7/14d, positive at 4 anchors of 4. On the **deviating** cohort it is
-0.2014 at 0 of 4 for h=3 and indistinguishable from zero at the rest
(`docs/changelog/2026-08-11-clean-anchor-confirmed-in-ci.md`).

`/opportunities` ranks by predicted return, which is precisely the ordering
rank IC measures, so it is the surface the evidence covers. Per-item lookups
still serve their forecast and carry the flag: the finding is about ordering,
and suppressing a forecast someone asked for by name goes further than what was
measured.

Two things this is NOT:

- **Not an accuracy improvement.** Gating changes nothing about the model. It
  stops publishing a ranking on a cohort whose measured served rank IC is zero
  or negative.
- **Not the >10% outlier test.** `predict` already computes a deviation mask at
  `ANCHOR_OUTLIER_TOLERANCE`, and that is a different, far smaller population
  (~5% of items against ~2/3). Nothing was measured on it. The gate uses exact
  equality, matching `replay_serving._tied_mask`, because that is the split
  every published figure came from.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from api.serving_policy import anchor_clean_clause, meets_anchor_gate
from models.forecaster import ItemForecaster

# --------------------------------------------------------------- the predicate


def test_a_clean_anchor_passes_and_a_deviating_one_does_not():
    assert meets_anchor_gate(True) is True
    assert meets_anchor_gate(False) is False


def test_an_unknown_anchor_passes():
    """NULL is 'not recorded', which is every row written before the column
    existed. Excluding those would empty the ranked surfaces the moment the
    migration lands and refill them only after the next forecast run -- an
    outage caused by a disclosure change."""
    assert meets_anchor_gate(None) is True


def test_the_sql_clause_and_the_python_predicate_agree():
    """Two filters, one rule. `_latest_forecasts` filters in SQL and
    `select_opportunities` re-filters in Python, exactly as the price floor
    does; a disagreement between them would silently change the cohort with the
    query plan."""
    from database import ItemForecast

    clause = str(anchor_clean_clause(ItemForecast.anchor_clean).compile(compile_kwargs={"literal_binds": True}))
    assert "IS NULL" in clause.upper(), f"the SQL leg drops unknown rows where the Python leg keeps them: {clause}"


# ------------------------------------------------------- what predict computes


def test_the_wedge_is_measured_against_the_raw_quote(forecaster_with_models):
    """The load-bearing test, and the one thing easy to get wrong.

    `predict` REASSIGNS `latest_rows["price"]` to the served base, and under the
    shipped arm that base IS the smoothed median -- so a mask computed after
    that line would find `p == S` for every item and call the whole catalogue
    clean. The disclosure has to read the raw quote.
    """
    f, price_df = forecaster_with_models
    # item_spike's last quote is 200 against a ~115 local median; item_flat's
    # last quote IS its median.
    result = _predict(f, price_df)

    clean = dict(zip(result["item_id"], result["anchor_clean"]))
    assert clean["item_flat"] is True
    assert clean["item_spike"] is False, (
        "every item read as clean — the mask was computed after the served base replaced the raw quote"
    )


def test_the_wedge_percentage_is_signed_and_relative_to_the_median(forecaster_with_models):
    """The value, not just the flag. The split at exact equality is what was
    measured; whether the effect is really a cliff there or monotone in
    |p/S - 1| is unmeasured, and publishing the size is what makes that
    readable later without another serving change."""
    f, price_df = forecaster_with_models
    result = _predict(f, price_df)
    wedge = dict(zip(result["item_id"], result["anchor_wedge_pct"]))

    assert wedge["item_flat"] == pytest.approx(0.0, abs=1e-9)
    assert wedge["item_spike"] > 50.0


def test_the_flag_does_not_move_with_the_freshness_arm(forecaster_with_models, monkeypatch):
    """`SERVE_OUTLIER_GATED_ANCHOR` changes which price is served, not which
    items the model can order. If the flag followed it, the two arms would be
    scored on different cohorts -- the failure the tied cohort exists to
    remove."""
    f, price_df = forecaster_with_models

    monkeypatch.delenv("SERVE_OUTLIER_GATED_ANCHOR", raising=False)
    off = _predict(f, price_df).set_index("item_id")["anchor_clean"]
    monkeypatch.setenv("SERVE_OUTLIER_GATED_ANCHOR", "1")
    on = _predict(f, price_df).set_index("item_id")["anchor_clean"]

    assert off.to_dict() == on.to_dict()


def test_the_gate_is_not_the_ten_percent_outlier_test(forecaster_with_models):
    """An item 3% off its median is deviating. The outlier mask would call it
    clean, and nothing measured supports that split."""
    f, price_df = forecaster_with_models
    result = _predict(f, price_df).set_index("item_id")

    assert bool(result.loc["item_drift", "anchor_clean"]) is False
    assert abs(float(result.loc["item_drift", "anchor_wedge_pct"])) < 10.0


# ---------------------------------------------------------------- the ranking


def _forecast(item_id, mid, current=10.0, clean=True, direction="up"):
    from database import ItemForecast

    return ItemForecast(
        item_id=item_id,
        forecast_date=date(2026, 8, 11),
        horizon_days=7,
        price_mid=mid,
        current_price=current,
        direction=direction,
        anchor_clean=clean,
    )


def _items(ids):
    from database import Item

    return {i: Item(id=i, name=f"AK-47 | Skin {i}", item_id=f"ak_{i}") for i in ids}


def test_a_deviating_item_is_not_ranked():
    from api.routes.opportunities import select_opportunities

    forecasts = [_forecast(1, 20.0, clean=False), _forecast(2, 11.0, clean=True)]
    out = select_opportunities(forecasts, _items([1, 2]), None, 10)

    assert [o.item_id for o in out] == [2], "the deviating item scored the larger move and would top the list"


def test_an_unknown_anchor_is_still_ranked():
    """Every row written before the column existed carries NULL. The gate must
    degrade to today's behaviour, not to an empty list."""
    from api.routes.opportunities import select_opportunities

    forecasts = [_forecast(1, 20.0, clean=None), _forecast(2, 11.0, clean=True)]
    out = select_opportunities(forecasts, _items([1, 2]), None, 10)

    assert [o.item_id for o in out] == [1, 2]


def test_the_gate_applies_before_the_limit():
    """`limit=2` must return two CLEAN items, not two minus the ones dropped.
    Filtering after the truncation would make the list shorter the more
    deviating items happened to rank highest."""
    from api.routes.opportunities import select_opportunities

    forecasts = [_forecast(i, 30.0 - i, clean=(i >= 3)) for i in range(1, 6)]
    out = select_opportunities(forecasts, _items(range(1, 6)), None, 2)

    assert len(out) == 2
    assert all(o.item_id >= 3 for o in out)


def test_the_query_filters_on_the_column_too():
    """The Python predicate alone would pull every forecast out of Postgres and
    discard two thirds in the API process."""
    import inspect

    from api.routes import opportunities

    src = inspect.getsource(opportunities._latest_forecasts)
    assert "anchor_clean_clause" in src


# ------------------------------------------------------------ the item lookup


def test_a_missing_flag_reads_as_unknown_not_as_clean():
    """The trap: a Parquet NULL arrives from DuckDB as `nan`, and `bool(nan)`
    is True. Coerced naively, every row the mirror predates would publish
    itself as clean -- the one wrong answer, since those rows are exactly the
    ones nothing is known about."""
    from api.routes.items import _optional_bool, _optional_float

    assert _optional_bool(None) is None
    assert _optional_bool(float("nan")) is None
    assert _optional_bool(False) is False
    assert _optional_bool(True) is True
    assert _optional_float(float("nan")) is None
    assert _optional_float(-13.5) == pytest.approx(-13.5)


def test_the_item_lookup_serves_a_deviating_forecast_and_says_so():
    """Gated on ranking, disclosed on lookup. Suppressing a forecast someone
    asked for by name goes past what was measured -- rank IC is an ordering
    statistic."""
    from api.routes import items as items_route

    item = type("I", (), {"id": 7, "name": "AK-47 | Redline"})()
    row = items_route._DictObj(
        {
            "current_price": 10.0,
            "price_low": 9.0,
            "price_mid": 11.0,
            "price_high": 13.0,
            "direction": "up",
            "confidence": "low",
            "anchor_clean": False,
            "anchor_wedge_pct": -13.5,
        }
    )
    with patch.object(items_route, "_forecast_parquet", return_value=row):
        out = items_route._prediction_parquet(item, "7_days", 7)

    assert out.forecast_mid == pytest.approx(11.0)
    assert out.anchor_clean is False
    assert out.anchor_wedge_pct == pytest.approx(-13.5)


# --------------------------------------------------------------- the write path


def test_the_write_path_carries_both_columns():

    results = pd.DataFrame(
        [
            {
                "item_id": "ak_1",
                "current_price": 10.0,
                "anchor_clean": False,
                "anchor_wedge_pct": -13.5,
                "forecasts": {7: {"low": 9.0, "mid": 10.5, "high": 12.0, "direction": "up", "confidence": "low"}},
            }
        ]
    )

    rows = _write_and_capture(
        results,
        db_columns={
            "item_id",
            "forecast_date",
            "horizon_days",
            "price_low",
            "price_mid",
            "price_high",
            "current_price",
            "direction",
            "confidence",
            "model_version",
            "created_at",
            "anchor_clean",
            "anchor_wedge_pct",
        },
    )

    assert rows[0]["anchor_clean"] is False
    assert rows[0]["anchor_wedge_pct"] == pytest.approx(-13.5)


def test_an_unmigrated_table_does_not_take_the_forecast_run_down(caplog):
    """This repo's prod schema runs behind its migrations — `daily_analysis`
    was dropped by 0015 and is still in prod. An INSERT naming a column the
    table lacks fails the whole batch, so a disclosure field would have taken
    out the daily forecast run.

    It degrades, and it says so: a silent skip is how this project has ended up
    with a green run and no data more than once.
    """
    results = pd.DataFrame(
        [
            {
                "item_id": "ak_1",
                "current_price": 10.0,
                "anchor_clean": False,
                "anchor_wedge_pct": -13.5,
                "forecasts": {7: {"low": 9.0, "mid": 10.5, "high": 12.0, "direction": "up", "confidence": "low"}},
            }
        ]
    )

    with caplog.at_level("WARNING"):
        rows = _write_and_capture(
            results,
            db_columns={
                "item_id",
                "forecast_date",
                "horizon_days",
                "price_low",
                "price_mid",
                "price_high",
                "current_price",
                "direction",
                "confidence",
                "model_version",
                "created_at",
            },
        )

    # The mirror keeps them; only the DB payload is narrowed.
    assert rows[0]["anchor_clean"] is False
    assert "anchor_clean" in caplog.text and "alembic upgrade head" in caplog.text


def _write_and_capture(results, db_columns):
    """Drive `_write_forecasts_to_db` against a table with `db_columns`,
    returning the rows it handed to the Parquet mirror."""
    from scripts.forecast_prices import _write_forecasts_to_db

    captured = {}
    inserted = []

    class _DB:
        def execute(self, stmt, *a, **k):
            inserted.append(stmt)

        def commit(self):
            return None

        def get_bind(self):
            return type("_Bind", (), {"dialect": type("_D", (), {"name": "postgresql"})()})()

    class _Inspector:
        def get_columns(self, name):
            return [{"name": c} for c in db_columns]

    with patch("db.parquet.append_table", side_effect=lambda name, rows, keys: captured.update(rows=rows)):
        with patch("sqlalchemy.inspect", return_value=_Inspector()):
            with patch("sqlalchemy.dialects.postgresql.insert") as ins:
                (ins.return_value.values.return_value.on_conflict_do_update.return_value) = "stmt"
                _write_forecasts_to_db(_DB(), results, "lgbm-v3", {"ak_1": 1}, date(2026, 8, 11))
                values_call = ins.return_value.values.call_args
    batch = values_call[0][0]
    for column in ("anchor_clean", "anchor_wedge_pct"):
        assert (column in batch[0]) == (column in db_columns), f"{column} in the DB payload but not in the table"
    return captured["rows"]


# ------------------------------------------------------------------- fixtures


def _price_frame():
    """Four items whose anchor day sits at a known distance from its median.

    - `item_flat`   quote == median            -> clean
    - `item_spike`  quote ~74% above median    -> deviating, and an outlier
    - `item_drift`  quote ~3% above median     -> deviating, NOT an outlier
    - `item_trend`  monotone, so the newest quote is the window's maximum
    """
    rows = []
    for d in range(40):
        day = date(2026, 6, 1) + timedelta(days=d)
        rows.append({"item_id": "item_flat", "date": day, "price": 100.0 + (d % 3) - 1, "volume": 100})
        rows.append({"item_id": "item_spike", "date": day, "price": 100.0 + d * 0.5, "volume": 100})
        rows.append({"item_id": "item_drift", "date": day, "price": 50.0 + (d % 3) - 1, "volume": 100})
        rows.append({"item_id": "item_trend", "date": day, "price": 30.0 + d * 0.5, "volume": 100})
    df = pd.DataFrame(rows)
    last = df["date"].max()
    # `item_flat` must land exactly on its own 3-day median on the last day.
    df.loc[(df["item_id"] == "item_flat") & (df["date"] >= last - timedelta(days=2)), "price"] = [99.0, 101.0, 100.0]
    df.loc[(df["item_id"] == "item_spike") & (df["date"] == last), "price"] = 200.0
    df.loc[(df["item_id"] == "item_drift") & (df["date"] >= last - timedelta(days=2)), "price"] = [49.0, 51.0, 51.5]
    return df


@pytest.fixture
def forecaster_with_models(tmp_path):
    """A forecaster whose `predict` returns rows: real boosters over two
    features, which is all the anchor disclosure needs to travel through."""
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.HORIZONS = [7]
    f.QUANTILES = [0.5]
    cols = ["price_log", "price_lag_1d"]
    f.feature_cols = list(cols)
    f.horizon_feature_cols = {7: list(cols)}
    f.feature_medians = pd.Series(dtype=float)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)).astype(np.float32)
    y = rng.normal(size=200)
    booster = lgb.train(
        {
            "objective": "regression",
            "verbosity": -1,
            "max_bin": 63,
            "min_data_in_leaf": 1,
            "num_leaves": 3,
            "learning_rate": 0.1,
        },
        lgb.Dataset(X, y),
        num_boost_round=5,
    )
    f.models[(7, 0.5)] = [booster]
    f.conformal_calibration = {7: 1.5}
    return f, _price_frame()


def _predict(f, price_df):
    empty_events = pd.DataFrame(columns=["id", "type", "timestamp", "description"])
    with patch.object(f, "fetch_price_history", return_value=price_df):
        with patch.object(f, "fetch_events", return_value=empty_events):
            result = f.predict()
    assert not result.empty, "fixture produced no forecasts"
    return result
