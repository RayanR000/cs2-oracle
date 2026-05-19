"""
Item endpoints
"""

from fastapi import APIRouter, Query, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from database import get_db, Item
from repositories import ItemRepository, PriceHistoryRepository
from schemas import ItemResponse, PriceHistoryResponse

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/", response_model=dict)
async def list_items(
    type: str = Query(None, description="Filter by type: skin, case, sticker"),
    skip: int = Query(0, ge=0, description="Number of items to skip (pagination)"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    db: Session = Depends(get_db)
):
    """List all items with optional filtering and pagination"""
    if type:
        items = ItemRepository.get_items_by_type(db, type, skip + limit)
        items = items[skip:skip + limit]
    else:
        items = ItemRepository.get_all_items(db, skip, limit)
    
    total = db.query(Item).count()
    
    return {
        "items": [
            {
                "id": item.id,
                "item_id": item.item_id,
                "name": item.name,
                "type": item.type,
                "release_date": item.release_date
            }
            for item in items
        ],
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": (skip + limit) < total
    }

@router.get("/search", response_model=dict)
async def search_items(
    q: str = Query(..., min_length=1),
    db: Session = Depends(get_db)
):
    """Search items by name"""
    results = ItemRepository.search_items(db, q)
    return {
        "results": [
            {
                "id": item.id,
                "item_id": item.item_id,
                "name": item.name,
                "type": item.type
            }
            for item in results
        ],
        "total": len(results)
    }

@router.get("/trending", response_model=dict)
async def get_trending(
    limit: int = Query(10, ge=1, le=50),
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get trending items"""
    trending = ItemRepository.get_trending_items(db, days, limit)
    return {
        "trending": trending,
        "timestamp": None,
        "period_days": days
    }

@router.get("/{item_id}", response_model=dict)
async def get_item(item_id: str, db: Session = Depends(get_db)):
    """Get item details"""
    item = ItemRepository.get_item_by_id(db, item_id)
    
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    return {
        "id": item.id,
        "item_id": item.item_id,
        "name": item.name,
        "type": item.type,
        "release_date": item.release_date
    }

@router.get("/{item_id}/price-history", response_model=dict)
async def get_price_history(
    item_id: str,
    days: int = Query(30, ge=1, le=365),
    skip: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
    db: Session = Depends(get_db)
):
    """Get price history for an item"""
    item = ItemRepository.get_item_by_id(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    history = PriceHistoryRepository.get_price_history(db, item.id, days, skip, limit)
    
    return {
        "item_id": item_id,
        "history": [
            {
                "id": h.id,
                "timestamp": h.timestamp,
                "price": h.price,
                "volume": h.volume,
                "median_price": h.median_price
            }
            for h in history
        ],
        "total": len(history)
    }

@router.get("/{item_id}/trends", response_model=dict)
async def get_trends(item_id: str, db: Session = Depends(get_db)):
    """Get trend analysis for an item"""
    from analytics.trend_analyzer import TrendAnalyzer, OpportunityDetector
    
    item = ItemRepository.get_item_by_id(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    # Get price history
    price_history = sorted(item.price_histories[-90:], key=lambda h: h.timestamp)
    prices = [h.price for h in price_history]
    
    if len(prices) < 7:
        return {
            "item_id": item_id,
            "item_name": item.name,
            "current_price": prices[-1] if prices else None,
            "trend_direction": "insufficient_data",
            "confidence": "low",
            "message": "Insufficient price history"
        }
    
    # Compute indicators
    sma_7 = TrendAnalyzer.compute_sma(prices, 7)
    sma_30 = TrendAnalyzer.compute_sma(prices, 30)
    volatility = TrendAnalyzer.compute_volatility(prices)
    rsi = TrendAnalyzer.compute_rsi(prices)
    bollinger = TrendAnalyzer.compute_bollinger_bands(prices)
    macd = TrendAnalyzer.compute_macd(prices)
    support_resist = TrendAnalyzer.compute_support_resistance(prices)
    
    trend_score = TrendAnalyzer.compute_trend_score(prices)
    direction, confidence = TrendAnalyzer.classify_trend(trend_score)
    
    current_price = prices[-1]
    baseline = OpportunityDetector.compute_baseline_trend(prices)
    
    # Build factors explanation
    factors = []
    if sma_7 and sma_30:
        if sma_7 > sma_30:
            factors.append("7-day MA above 30-day MA (bullish)")
        elif sma_7 < sma_30:
            factors.append("7-day MA below 30-day MA (bearish)")
    
    if rsi:
        if rsi > 70:
            factors.append("RSI > 70 (overbought)")
        elif rsi < 30:
            factors.append("RSI < 30 (oversold)")
    
    if bollinger:
        if current_price > bollinger['upper']:
            factors.append("Price above upper Bollinger Band")
        elif current_price < bollinger['lower']:
            factors.append("Price below lower Bollinger Band")
    
    return {
        "item_id": item_id,
        "item_name": item.name,
        "current_price": round(current_price, 2),
        "trend_direction": direction,
        "confidence": confidence,
        "trend_score": round(trend_score, 3) if trend_score else None,
        "indicators": {
            "sma_7": round(sma_7, 2) if sma_7 else None,
            "sma_30": round(sma_30, 2) if sma_30 else None,
            "volatility": round(volatility, 4) if volatility else None,
            "rsi": round(rsi, 2) if rsi else None,
            "bollinger_upper": round(bollinger['upper'], 2) if bollinger else None,
            "bollinger_middle": round(bollinger['middle'], 2) if bollinger else None,
            "bollinger_lower": round(bollinger['lower'], 2) if bollinger else None,
            "macd": round(macd['macd'], 4) if macd else None,
            "macd_signal": round(macd['signal'], 4) if macd else None,
            "support": round(support_resist['support'], 2) if support_resist else None,
            "resistance": round(support_resist['resistance'], 2) if support_resist else None,
        },
        "factors": factors,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/{item_id}/prediction", response_model=dict)
async def get_prediction(
    item_id: str,
    period: str = Query("7_days", regex="^(7_days|30_days)$"),
    db: Session = Depends(get_db)
):
    """Get price prediction for an item"""
    from analytics.trend_analyzer import TrendAnalyzer
    
    item = ItemRepository.get_item_by_id(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    # Get price history
    price_history = sorted(item.price_histories[-90:], key=lambda h: h.timestamp)
    prices = [h.price for h in price_history]
    
    if len(prices) < 7:
        return {
            "item_id": item_id,
            "item_name": item.name,
            "current_price": prices[-1] if prices else None,
            "message": "Insufficient data for prediction",
            "forecast_low": None,
            "forecast_high": None,
            "confidence": "low"
        }
    
    current_price = prices[-1]
    volatility = TrendAnalyzer.compute_volatility(prices)
    trend_score = TrendAnalyzer.compute_trend_score(prices)
    direction, confidence = TrendAnalyzer.classify_trend(trend_score)
    
    # Compute price range
    forecast_low, forecast_high = TrendAnalyzer.compute_price_range(prices, volatility)
    
    # Determine period (days for prediction)
    days = 7 if period == "7_days" else 30
    
    return {
        "item_id": item_id,
        "item_name": item.name,
        "current_price": round(current_price, 2),
        "forecast": {
            "low": round(forecast_low, 2),
            "mid": round((forecast_low + forecast_high) / 2, 2),
            "high": round(forecast_high, 2)
        },
        "period_days": days,
        "period_label": period,
        "trend_direction": direction,
        "confidence": confidence,
        "volatility": round(volatility, 4) if volatility else None,
        "methodology": "Linear regression with volatility-adjusted bands",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/{item_id}/events", response_model=dict)
async def get_item_events(
    item_id: str,
    limit: int = Query(20),
    db: Session = Depends(get_db)
):
    """Get market events related to an item"""
    item = ItemRepository.get_item_by_id(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    return {
        "item_id": item_id,
        "events": []
    }
