"""Read-path changes from the 2026-10-08 performance review, API section.

The opportunities freshness floor, the momentum horizon check, gzip, and the
`/accuracy/` Cache-Control prefix.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient


def _forecast(item_id, forecast_date, horizon=7):
    from database import ItemForecast

    return ItemForecast(
        item_id=item_id,
        forecast_date=forecast_date,
        horizon_days=horizon,
        price_mid=12.0,
        current_price=10.0,
        direction="up",
        anchor_clean=True,
    )


def test_latest_forecasts_drops_items_whose_newest_band_is_stale(db_session):
    """An item that left the forecast universe must not keep ranking on its
    last band. Same window as `/items/trending`."""
    from api.routes.opportunities import _latest_forecasts
    from api.serving_policy import MAX_ARCHIVE_LAG_DAYS

    today = date.today()
    db_session.add_all(
        [
            _forecast(1, today - timedelta(days=1)),
            _forecast(2, today - timedelta(days=MAX_ARCHIVE_LAG_DAYS + 1)),
        ]
    )
    db_session.commit()

    assert [f.item_id for f in _latest_forecasts(db_session, 7)] == [1]


def test_latest_forecasts_keeps_a_band_at_the_lag_limit(db_session):
    from api.routes.opportunities import _latest_forecasts
    from api.serving_policy import MAX_ARCHIVE_LAG_DAYS

    db_session.add(_forecast(1, date.today() - timedelta(days=MAX_ARCHIVE_LAG_DAYS)))
    db_session.commit()

    assert [f.item_id for f in _latest_forecasts(db_session, 7)] == [1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    from main import app

    # A large public GET under /accuracy/, without a database behind it.
    app.add_api_route("/accuracy/_test_large", lambda: {"rows": ["x" * 40] * 200})
    try:
        yield TestClient(app)
    finally:
        app.router.routes.pop()


def test_large_responses_are_gzipped(client):
    response = client.get("/accuracy/_test_large", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"


def test_gzip_runs_at_level_5():
    from fastapi.middleware.gzip import GZipMiddleware
    from main import app

    (gzip,) = [m for m in app.user_middleware if m.cls is GZipMiddleware]
    assert gzip.kwargs["compresslevel"] == 5


def test_accuracy_routes_get_cache_control(client):
    response = client.get("/accuracy/_test_large")
    assert response.headers["cache-control"].startswith("public, max-age=300")


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_momentum_rejects_unserved_horizons(client, horizon):
    response = client.get("/opportunities/momentum", params={"horizon_days": horizon})
    assert response.status_code == 400
    assert "must be one of 3, 7, 14, 30" in response.json()["detail"]
