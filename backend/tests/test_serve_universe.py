"""The served universe: established items at every horizon, young releases at h=3.

`models/serve_universe.py` is the one definition the predict universe, the prior-row
blend and the forecast writer read, and the panels that size or judge the
established band (feedback refit, PID prereg) must never see a young row.
"""

import datetime as dt

import pandas as pd
import pytest
from database import Base, ForecastOutcome, Item, ItemForecast
from models import serve_universe
from models.forecaster import ItemForecaster
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ESTABLISHED = "AK-47 | Redline (Field-Tested)"
YOUNG_OLD_ENOUGH = "AK-47 | AUTOEXEC (Field-Tested)"
YOUNG_TOO_NEW = "AK-47 | AUTOEXEC (Factory New)"
CATALOG_ONLY = "Sticker | Old Snapshot Item"


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add_all(
        [
            Item(id=1, item_id=ESTABLISHED, name=ESTABLISHED, type="skin", is_backfilled=1),
            Item(
                id=2,
                item_id=YOUNG_OLD_ENOUGH,
                name=YOUNG_OLD_ENOUGH,
                type="skin",
                is_backfilled=0,
                release_date=dt.datetime(2026, 7, 17),
            ),
            Item(
                id=3,
                item_id=YOUNG_TOO_NEW,
                name=YOUNG_TOO_NEW,
                type="skin",
                is_backfilled=0,
                release_date=dt.datetime(2026, 9, 1),
            ),
            # A catalog row with no release_date is neither established nor young.
            Item(id=4, item_id=CATALOG_ONLY, name=CATALOG_ONLY, type="sticker", is_backfilled=0),
        ]
    )
    s.commit()
    yield s
    engine.dispose()


def test_served_items_is_established_plus_young_releases_past_the_history_gate(session):
    served = serve_universe.served_items(session, dt.date(2026, 10, 1))

    assert served == {ESTABLISHED: (1, False), YOUNG_OLD_ENOUGH: (2, True)}


def test_the_history_gate_is_inclusive_at_exactly_sixty_days(session):
    on_the_day = dt.date(2026, 7, 17) + dt.timedelta(days=serve_universe.YOUNG_MIN_HISTORY_DAYS)

    assert YOUNG_OLD_ENOUGH in serve_universe.served_items(session, on_the_day)
    assert YOUNG_OLD_ENOUGH not in serve_universe.served_items(session, on_the_day - dt.timedelta(days=1))


def test_a_release_date_on_an_established_item_does_not_make_it_young(session):
    session.get(Item, 1).release_date = dt.datetime(2026, 9, 29)
    session.commit()

    assert serve_universe.served_items(session, dt.date(2026, 10, 1))[ESTABLISHED] == (1, False)


def test_slug_maps_split_established_from_young(session):
    served = serve_universe.served_items(session, dt.date(2026, 10, 1))

    assert serve_universe.slug_to_id(served) == {ESTABLISHED: 1, YOUNG_OLD_ENOUGH: 2}
    assert serve_universe.slug_to_id(served, established_only=True) == {ESTABLISHED: 1}
    assert serve_universe.young_slugs(served) == {YOUNG_OLD_ENOUGH}


def _result(item, horizons):
    return {"item_id": item, "forecasts": {h: {"mid": 10.0, "low": 9.0, "high": 11.0} for h in horizons}}


def test_young_items_keep_only_their_h3_forecast():
    df = pd.DataFrame([_result(ESTABLISHED, [3, 7, 14, 30]), _result(YOUNG_OLD_ENOUGH, [3, 7, 14, 30])])

    out = ItemForecaster._restrict_young_horizons(df, {YOUNG_OLD_ENOUGH})

    by_item = dict(zip(out["item_id"], out["forecasts"]))
    assert set(by_item[ESTABLISHED]) == {3, 7, 14, 30}
    assert set(by_item[YOUNG_OLD_ENOUGH]) == {3}


def test_a_young_item_with_no_h3_forecast_is_dropped_not_served_empty():
    df = pd.DataFrame([_result(ESTABLISHED, [3, 7]), _result(YOUNG_OLD_ENOUGH, [7, 14])])

    out = ItemForecaster._restrict_young_horizons(df, {YOUNG_OLD_ENOUGH})

    assert list(out["item_id"]) == [ESTABLISHED]


def test_no_young_set_leaves_the_frame_untouched():
    df = pd.DataFrame([_result(ESTABLISHED, [3, 7])])

    assert ItemForecaster._restrict_young_horizons(df, set()) is df


def test_the_pid_prereg_panel_excludes_young_releases(session):
    from scripts.measure_conformal_pid import load_prod_panel

    fd = dt.date(2026, 10, 2)
    for fid, item_id in ((1, 1), (2, 2)):
        session.add(
            ItemForecast(
                id=fid,
                item_id=item_id,
                forecast_date=fd,
                horizon_days=3,
                price_low=9.0,
                price_mid=10.0,
                price_high=11.0,
                current_price=10.0,
                band_multiplier=0.5385,
            )
        )
        session.add(
            ForecastOutcome(
                forecast_id=fid,
                item_id=item_id,
                forecast_date=fd,
                horizon_days=3,
                target_date=fd + dt.timedelta(days=3),
                base_price=10.0,
                current_price=10.0,
                predicted_price_low=9.0,
                predicted_price_mid=10.0,
                predicted_price_high=11.0,
                actual_price=10.5,
                direction_correct=0,
                abs_error=0.5,
            )
        )
    session.commit()

    panel = load_prod_panel(session)

    assert set(panel["item_id"]) == {1}
