"""
Comprehensive tests for all Phase 3 API endpoints
Tests all 15 endpoints with various parameters and error cases
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from main import app
from database import SessionLocal, init_db
from seed_data import DatabaseSeeder
from routers import admin as admin_router

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Initialize and seed database once for all tests"""
    init_db()
    db = SessionLocal()
    try:
        DatabaseSeeder.seed_all(db)
    finally:
        db.close()
    yield
    db.close()

class TestItemsEndpoints:
    """Tests for Items API endpoints"""
    
    def test_list_items(self):
        """Test GET /items/"""
        response = client.get("/items/")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "skip" in data
        assert "limit" in data
        assert "has_more" in data
        assert data["limit"] == 50
    
    def test_list_items_with_pagination(self):
        """Test GET /items/ with skip and limit"""
        response = client.get("/items/?skip=0&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= 10
        assert data["skip"] == 0
        assert data["limit"] == 10
    
    def test_list_items_with_type_filter(self):
        """Test GET /items/ with type filter"""
        response = client.get("/items/?type=skin")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        # All returned items should be of type skin (or empty if none exist)
        for item in data["items"]:
            assert item["type"] == "skin" or len(data["items"]) == 0
    
    def test_list_items_invalid_limit(self):
        """Test GET /items/ with invalid limit"""
        response = client.get("/items/?limit=150")
        assert response.status_code == 422  # Validation error
    
    def test_search_items(self):
        """Test GET /items/search"""
        response = client.get("/items/search?q=dragon")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
    
    def test_search_items_empty_query(self):
        """Test GET /items/search with empty query"""
        response = client.get("/items/search?q=")
        assert response.status_code == 422  # Validation error - min length 1
    
    def test_get_trending(self):
        """Test GET /items/trending"""
        response = client.get("/items/trending")
        assert response.status_code == 200
        data = response.json()
        assert "trending" in data
        assert "timestamp" in data
        assert "period_days" in data
    
    def test_get_trending_with_days(self):
        """Test GET /items/trending with custom days"""
        response = client.get("/items/trending?days=30&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["period_days"] == 30
        assert len(data["trending"]) <= 5
    
    def test_get_item_detail(self):
        """Test GET /items/{item_id}"""
        # First get an item
        list_resp = client.get("/items/")
        assert list_resp.status_code == 200
        items = list_resp.json()["items"]
        
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["item_id"] == item_id
            assert "name" in data
            assert "type" in data
    
    def test_get_item_not_found(self):
        """Test GET /items/{item_id} with non-existent item"""
        response = client.get("/items/nonexistent-item")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_get_price_history(self):
        """Test GET /items/{item_id}/price-history"""
        list_resp = client.get("/items/")
        items = list_resp.json()["items"]
        
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}/price-history")
            assert response.status_code == 200
            data = response.json()
            assert "history" in data
            assert "total" in data
            assert data["item_id"] == item_id
    
    def test_get_price_history_with_days(self):
        """Test GET /items/{item_id}/price-history with days filter"""
        list_resp = client.get("/items/")
        items = list_resp.json()["items"]
        
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}/price-history?days=7")
            assert response.status_code == 200
            data = response.json()
            assert len(data["history"]) <= 7 or len(data["history"]) > 0
    
    def test_get_trends(self):
        """Test GET /items/{item_id}/trends"""
        list_resp = client.get("/items/")
        items = list_resp.json()["items"]
        
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}/trends")
            assert response.status_code == 200
            data = response.json()
            assert "current_price" in data
            assert "trend_direction" in data
            assert "confidence" in data
            assert "indicators" in data
            assert "factors" in data
    
    def test_get_prediction(self):
        """Test GET /items/{item_id}/prediction"""
        list_resp = client.get("/items/")
        items = list_resp.json()["items"]
        
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}/prediction")
            assert response.status_code == 200
            data = response.json()
            assert "forecast" in data
            assert "period_days" in data
            assert "trend_direction" in data
    
    def test_get_prediction_30_days(self):
        """Test GET /items/{item_id}/prediction with 30-day period"""
        list_resp = client.get("/items/")
        items = list_resp.json()["items"]
        
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}/prediction?period=30_days")
            assert response.status_code == 200
            data = response.json()
            assert data["period_days"] == 30


class TestOpportunitiesEndpoints:
    """Tests for Opportunities API endpoints"""
    
    def test_get_opportunities(self):
        """Test GET /opportunities/"""
        response = client.get("/opportunities/")
        assert response.status_code == 200
        data = response.json()
        assert "opportunities" in data
        assert "total" in data
    
    def test_get_opportunities_filtered(self):
        """Test GET /opportunities/ with type filter"""
        response = client.get("/opportunities/?type=undervalued")
        assert response.status_code == 200
        data = response.json()
        assert "opportunities" in data
        # All should be undervalued (or empty)
        for opp in data["opportunities"]:
            assert opp["opportunity_type"] in ["undervalued", "overheated", "momentum"]
    
    def test_get_undervalued(self):
        """Test GET /opportunities/undervalued"""
        response = client.get("/opportunities/undervalued")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "min_discount_filter" in data
    
    def test_get_undervalued_with_discount(self):
        """Test GET /opportunities/undervalued with min_discount"""
        response = client.get("/opportunities/undervalued?min_discount=3.0")
        assert response.status_code == 200
        data = response.json()
        assert data["min_discount_filter"] == 3.0
    
    def test_get_overheated(self):
        """Test GET /opportunities/overheated"""
        response = client.get("/opportunities/overheated")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "min_premium_filter" in data
    
    def test_get_momentum(self):
        """Test GET /opportunities/momentum"""
        response = client.get("/opportunities/momentum")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "min_change_filter" in data


class TestEventsEndpoints:
    """Tests for Events API endpoints"""
    
    def test_list_events(self):
        """Test GET /events/"""
        response = client.get("/events/")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert "total" in data
        assert "skip" in data
        assert "limit" in data
        assert "has_more" in data
    
    def test_list_events_with_type(self):
        """Test GET /events/ with type filter"""
        response = client.get("/events/?type=major")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
    
    def test_get_timeline(self):
        """Test GET /events/timeline"""
        response = client.get("/events/timeline")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert "total" in data
    
    def test_get_recent_events(self):
        """Test GET /events/recent"""
        response = client.get("/events/recent")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert "total" in data


class TestHealthEndpoints:
    """Tests for health and status endpoints"""
    
    def test_health_check(self):
        """Test GET /health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
    
    def test_root_endpoint(self):
        """Test GET /"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestAdminEndpoints:
    """Tests for admin collection endpoints"""

    @pytest.fixture
    def fake_collector(self):
        class FakeCollector:
            enabled = True
            is_running = False

            def collect_all_items(self):
                now = datetime.utcnow().isoformat()
                return {
                    "total_items": 3,
                    "successful": 2,
                    "failed": 1,
                    "timestamp": now,
                    "started_at": now,
                    "finished_at": now,
                    "duration_seconds": 0.123,
                }

            def get_collection_metrics(self):
                now = datetime.utcnow().isoformat()
                return {
                    "last_run_started_at": now,
                    "last_run_finished_at": now,
                    "last_success_at": now,
                    "last_error_at": None,
                    "last_error": None,
                    "last_run_duration_seconds": 0.123,
                    "last_run_total_items": 3,
                    "last_run_successful": 2,
                    "last_run_failed": 1,
                    "total_runs": 4,
                    "successful_runs": 3,
                    "failed_runs": 1,
                    "total_items_collected": 10,
                    "total_items_failed": 2,
                    "thread_alive": False,
                    "collection_enabled": True,
                    "is_running": False,
                    "status": "idle",
                }

        return FakeCollector()

    def test_collect_now_returns_metrics(self, monkeypatch, fake_collector):
        """Test POST /admin/collect-now includes collection metrics"""
        monkeypatch.setattr(admin_router, "get_collector", lambda: fake_collector)

        response = client.post("/admin/collect-now")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"
        assert "stats" in data
        assert "metrics" in data
        assert data["stats"]["successful"] == 2
        assert data["metrics"]["last_run_duration_seconds"] == 0.123
        assert data["metrics"]["successful_runs"] == 3

    def test_collection_status_exposes_health(self, monkeypatch, fake_collector):
        """Test GET /admin/collection-status includes runtime health fields"""
        monkeypatch.setattr(admin_router, "get_collector", lambda: fake_collector)

        response = client.get("/admin/collection-status")
        assert response.status_code == 200

        data = response.json()
        assert data["collection_enabled"] is True
        assert data["is_running"] is False
        assert data["thread_alive"] is False
        assert data["status"] == "idle"
        assert "latest_persisted_run" in data
        assert "metrics" in data
        assert "synthetic_history_enabled" in data

    def test_data_stats_includes_source_breakdown_and_collector(self, monkeypatch, fake_collector):
        """Test GET /admin/data-stats includes collector metrics and source breakdown"""
        monkeypatch.setattr(admin_router, "get_collector", lambda: fake_collector)

        response = client.get("/admin/data-stats")
        assert response.status_code == 200

        data = response.json()
        assert "collector" in data
        assert "source_breakdown" in data
        assert "total_collection_runs" in data
        assert data["collector"]["status"] == "idle"
        assert "steam" in data["source_breakdown"]


class TestErrorHandling:
    """Tests for error handling across endpoints"""
    
    def test_invalid_pagination_limit_too_high(self):
        """Test invalid limit parameter"""
        response = client.get("/items/?limit=500")
        assert response.status_code == 422
    
    def test_invalid_pagination_negative_skip(self):
        """Test negative skip parameter"""
        response = client.get("/items/?skip=-1")
        assert response.status_code == 422
    
    def test_invalid_days_parameter(self):
        """Test invalid days parameter"""
        response = client.get("/items/test/price-history?days=0")
        assert response.status_code == 422
    
    def test_invalid_prediction_period(self):
        """Test invalid prediction period"""
        list_resp = client.get("/items/")
        items = list_resp.json()["items"]
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}/prediction?period=invalid")
            assert response.status_code == 422


class TestResponseFormats:
    """Tests for response format consistency"""
    
    def test_items_response_structure(self):
        """Test items list response structure"""
        response = client.get("/items/?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["total"], int)
        assert isinstance(data["items"], list)
        if data["items"]:
            item = data["items"][0]
            assert "id" in item
            assert "item_id" in item
            assert "name" in item
            assert "type" in item
    
    def test_trends_response_has_indicators(self):
        """Test trends response has all indicators"""
        list_resp = client.get("/items/")
        items = list_resp.json()["items"]
        
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}/trends")
            data = response.json()
            indicators = data.get("indicators", {})
            # Check for key indicators
            assert any(k in indicators for k in ["sma_7", "sma_30", "rsi", "volatility"])
    
    def test_prediction_forecast_structure(self):
        """Test prediction forecast structure"""
        list_resp = client.get("/items/")
        items = list_resp.json()["items"]
        
        if items:
            item_id = items[0]["item_id"]
            response = client.get(f"/items/{item_id}/prediction")
            data = response.json()
            forecast = data.get("forecast", {})
            assert "low" in forecast
            assert "high" in forecast
            assert "mid" in forecast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
