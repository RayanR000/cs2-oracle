"""
Admin endpoints for data collection management
"""

from fastapi import APIRouter
from sqlalchemy import func
from typing import Optional
from collectors.real_data_collector import get_collector
from config import settings
from database import SessionLocal, Item, PriceHistory, CollectionRun

router = APIRouter(prefix="/admin", tags=["admin"])


def _serialize_collection_run(run: Optional[CollectionRun]) -> Optional[dict]:
    """Convert a persisted collection run into a JSON-friendly payload."""
    if run is None:
        return None

    return {
        "id": run.id,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "status": run.status,
        "total_items": run.total_items,
        "successful": run.successful,
        "failed": run.failed,
        "duration_seconds": run.duration_seconds,
        "error_message": run.error_message,
        "source_breakdown": run.source_breakdown or {},
    }

@router.post("/collect-now")
async def trigger_collection():
    """Manually trigger data collection immediately"""
    collector = get_collector()
    stats = collector.collect_all_items()
    
    return {
        "status": "completed",
        "stats": stats,
        "metrics": collector.get_collection_metrics()
    }

@router.get("/collection-status")
async def get_collection_status():
    """Get current data collection status"""
    collector = get_collector()
    
    # Get latest price collection timestamp
    db = SessionLocal()
    try:
        latest_price = db.query(PriceHistory).order_by(
            PriceHistory.timestamp.desc()
        ).first()
        latest_run = db.query(CollectionRun).order_by(
            CollectionRun.started_at.desc()
        ).first()
        
        latest_timestamp = latest_price.timestamp if latest_price else None
        total_records = db.query(PriceHistory).count()
        metrics = collector.get_collection_metrics()
        
        return {
            "collection_enabled": collector.enabled,
            "is_running": collector.is_running,
            "thread_alive": metrics.get("thread_alive", False),
            "latest_collection": latest_timestamp,
            "latest_persisted_run": _serialize_collection_run(latest_run),
            "total_price_records": total_records,
            "environment": settings.environment,
            "synthetic_history_enabled": settings.demo_bootstrap_enabled(),
            "status": metrics.get("status", "inactive"),
            "metrics": metrics
        }
    finally:
        db.close()

@router.get("/data-stats")
async def get_data_statistics():
    """Get database data statistics"""
    db = SessionLocal()
    try:
        total_items = db.query(Item).count()
        total_prices = db.query(PriceHistory).count()
        total_runs = db.query(CollectionRun).count()
        collector = get_collector()
        metrics = collector.get_collection_metrics()
        source_rows = db.query(
            PriceHistory.source,
            func.count(PriceHistory.id)
        ).group_by(PriceHistory.source).all()
        
        # Get price range
        all_prices = db.query(PriceHistory).all()
        if all_prices:
            prices = [p.price for p in all_prices]
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
        else:
            min_price = max_price = avg_price = 0
        
        return {
            "total_items": total_items,
            "total_price_records": total_prices,
            "total_collection_runs": total_runs,
            "collector": metrics,
            "source_breakdown": {
                source: count for source, count in source_rows
            },
            "price_statistics": {
                "min": round(min_price, 2),
                "max": round(max_price, 2),
                "average": round(avg_price, 2),
                "count": total_prices
            }
        }
    finally:
        db.close()
