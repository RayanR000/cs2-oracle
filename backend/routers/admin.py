"""
Admin endpoints for data collection management
"""

from fastapi import APIRouter, Query, HTTPException
from datetime import datetime
from collectors.real_data_collector import get_collector
from config import settings
from database import SessionLocal, Item, PriceHistory

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/collect-now")
async def trigger_collection():
    """Manually trigger data collection immediately"""
    collector = get_collector()
    stats = collector.collect_all_items()
    
    return {
        "status": "completed",
        "stats": stats
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
        
        latest_timestamp = latest_price.timestamp if latest_price else None
        total_records = db.query(PriceHistory).count()
        
        return {
            "collection_enabled": collector.enabled,
            "is_running": collector.is_running,
            "latest_collection": latest_timestamp,
            "total_price_records": total_records,
            "environment": settings.environment,
            "synthetic_history_enabled": settings.demo_bootstrap_enabled(),
            "status": "active" if collector.is_running else "inactive"
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
            "price_statistics": {
                "min": round(min_price, 2),
                "max": round(max_price, 2),
                "average": round(avg_price, 2),
                "count": total_prices
            }
        }
    finally:
        db.close()
