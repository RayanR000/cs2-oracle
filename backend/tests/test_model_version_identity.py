"""`model_version` is the served artifact's identity, never its configuration.

F3. The two were one string: `scripts/forecast_prices.py` wrote
`lgbm-v3-regime` on the daily path and `lgbm-v3-global-only` on the comparison
run, and `backtest/scoring.py::score_cohort` keys the cohort on that string. So
a config flag forked the date panel — the stored mirror splits 8 forecast dates
of `-regime`, 2 of `lgbm-v3` and 1 of `-global-only` — and no cohort could reach
`MIN_FORECAST_DATES = 20`. Run 31409508960 scored 110,615 frozen outcomes and
returned `NO HEADLINE (insufficient_dates)` at every horizon.

The suffix bought nothing in exchange. `item_forecasts` is unique on
(item_id, forecast_date, horizon_days) with no `model_version` in the key, so
two configs writing the same day never coexisted as rows: the second overwrote
the first and only relabelled it.

These pin the split — identity in the DB, configuration in the Parquet mirror —
and the observation that produces the configuration label.
"""

import inspect
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Base, Item, ItemForecast  # noqa: E402
from scripts.forecast_prices import (  # noqa: E402
    MODEL_VERSION,
    _write_forecasts_to_db,
    run_forecast,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(Item(id=1, item_id="ak", name="ak", type="skin"))
    db.commit()
    yield db
    engine.dispose()


@pytest.fixture
def ops(tmp_path, monkeypatch):
    """Redirect the ops store into tmp_path. The real price-archive/ops/ is
    production data and no test may write it."""
    import db.parquet as parquet_mod
    monkeypatch.setattr(parquet_mod, "OPS_DIR", tmp_path / "ops")
    return tmp_path / "ops"


def _results():
    return pd.DataFrame([{
        "item_id": "ak",
        "current_price": 3.0,
        "forecasts": {3: {"low": 2.0, "mid": 3.0, "high": 4.0,
                          "direction": "flat", "confidence": "high"}},
    }])


def _write(session, config):
    return _write_forecasts_to_db(
        session, _results(), MODEL_VERSION, {"ak": 1}, date(2026, 8, 11),
        model_config=config,
    )


def test_the_db_row_carries_the_identity_and_not_the_config(session, ops):
    """The scoring key. A suffix here is what forked the panel."""
    assert _write(session, "regime") == 1

    stored = session.query(ItemForecast).one()
    assert stored.model_version == "lgbm-v3"
    assert "regime" not in stored.model_version
    assert "global-only" not in stored.model_version


def test_two_configs_write_one_identity(session, ops):
    """Both arms of --compare-regime, in order. The row is overwritten on the
    unique key either way — which is the pre-existing behaviour this change does
    not alter — and what is left must not name a configuration."""
    _write(session, "regime")
    _write(session, "global-only")

    rows = session.query(ItemForecast).all()
    assert len(rows) == 1, "the unique key stopped collapsing the two writes"
    assert rows[0].model_version == "lgbm-v3"


def test_the_config_reaches_the_mirror(session, ops):
    """Mirror-only, the same split `item_slug` already follows: the ops store is
    where the archive is analysed, and the DB carries only what is served."""
    _write(session, "global-only")

    df = duckdb.sql(
        f"SELECT * FROM read_parquet('{ops / 'item_forecasts.parquet'}')"
    ).df()
    assert df["model_version"].tolist() == ["lgbm-v3"]
    assert df["model_config"].tolist() == ["global-only"]


def test_the_config_column_is_deliberately_not_on_the_db_model(session):
    """Prod's schema runs behind the migrations, so a new column is a live
    hazard for a field nothing served reads. If this ever moves, it needs a
    migration AND a catalog check, not just a model attribute."""
    assert "model_config" not in {c.name for c in sa_inspect(ItemForecast).columns}


def test_the_config_label_is_observed_not_assumed():
    """SKIP_REGIMES=1 has been production since 2026-08-10 and leaves
    `regime_models` empty, so the old unconditional "-regime" label described a
    configuration that did not run. `predict` prefers a regime model over the
    global one where it has one, so this must be read off the forecaster."""
    src = inspect.getsource(run_forecast)

    assert 'config_a = "regime" if forecaster.regime_models else "global-only"' in src


def test_no_writer_appends_a_config_suffix_to_the_version():
    """Source-level, across the whole module: the f-string that built
    `f"{MODEL_VERSION}-regime"` is the defect, and it is one character from
    coming back."""
    import scripts.forecast_prices as fp

    src = Path(fp.__file__).read_text()
    assert 'MODEL_VERSION}-' not in src
    assert 'MODEL_VERSION + "-' not in src
