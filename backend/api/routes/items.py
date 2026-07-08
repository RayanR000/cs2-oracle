from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
import re
import math

from database import (
    get_db, Item, PriceHistory, ChartPoint, DailyAnalysis, ItemForecast,
    Event, EventImpact, backfilled_item_clause,
)
from api.cache import get_or_build
from api.schemas import (
    ItemOut, PricePointOut, TrendAnalysisOut, PredictionOut,
    SourcePriceOut, MultiSourcePricesOut, EventOut, TrendingItemOut
)

router = APIRouter(prefix="/items", tags=["items"])


def _resolve_item(item_id: str, db: Session) -> Item:
    item = db.query(Item).filter(Item.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.get("/", response_model=list[ItemOut])
def list_items(
    type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    def build():
        q = db.query(Item).filter(backfilled_item_clause())
        if type:
            q = q.filter(Item.type == type)
        return q.order_by(Item.name).offset(skip).limit(limit).all()

    return get_or_build(f"items_list:{type or ''}:{skip}:{limit}", 300, build)


@router.get("/search", response_model=list[ItemOut])
def search_items(
    q: str = Query(min_length=1),
    db: Session = Depends(get_db),
):
    return (
        db.query(Item)
        .filter(Item.name.ilike(f"%{q}%"), backfilled_item_clause())
        .order_by(Item.name)
        .limit(50)
        .all()
    )


@router.get("/trending", response_model=list[TrendingItemOut])
def trending_items(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return get_or_build(
        f"items_trending:{limit}", 600, lambda: _build_trending(db, limit)
    )


def _latest_prices(db: Session, item_ids: list[int]) -> dict[int, float]:
    """Latest price per item from chart_points, falling back to price_history for snapshot items."""
    prices = {}

    cp_rows = (
        db.query(ChartPoint.item_id, ChartPoint.close)
        .filter(ChartPoint.item_id.in_(item_ids))
        .distinct(ChartPoint.item_id)
        .order_by(ChartPoint.item_id, desc(ChartPoint.day))
        .all()
    )
    prices.update({r.item_id: r.close for r in cp_rows})

    remaining = [iid for iid in item_ids if iid not in prices]
    if remaining:
        ph_rows = (
            db.query(PriceHistory.item_id, PriceHistory.price)
            .filter(PriceHistory.item_id.in_(remaining))
            .distinct(PriceHistory.item_id)
            .order_by(PriceHistory.item_id, desc(PriceHistory.timestamp))
            .all()
        )
        prices.update({r.item_id: r.price for r in ph_rows})

    return prices


def _build_trending(db: Session, limit: int):
    items = (
        db.query(Item)
        .filter(Item.icon_url.isnot(None), backfilled_item_clause())
        .order_by(desc(Item.updated_at))
        .limit(max(limit * 10, 100))
        .all()
    )
    item_ids = [i.id for i in items]
    latest_prices = _latest_prices(db, item_ids) if item_ids else {}

    result = [
        TrendingItemOut(
            id=item.id,
            item_id=item.item_id,
            name=item.name,
            type=item.type,
            icon_url=item.icon_url,
            latest_price=latest_prices.get(item.id, 0.0),
        )
        for item in items
        if latest_prices.get(item.id, 0.0) > 0
    ]
    return result[:limit]


def _parse_item_name(name: str):
    match = re.match(r'^(.+?)\s*\(([^)]+)\)\s*$', name)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return name, None


class QualityVariantOut(BaseModel):
    item_id: str
    name: str
    quality: str
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h: Optional[int] = None


@router.get("/{item_id}/variants", response_model=List[QualityVariantOut])
def get_item_variants(
    item_id: str,
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    base_name, _ = _parse_item_name(item.name)

    all_items = db.query(Item).filter(
        Item.name.ilike(f"%{base_name}%"),
        Item.type == item.type,
    ).all()

    matching = [i for i in all_items if _parse_item_name(i.name)[0] == base_name]
    if not matching:
        matching = [item]

    item_ids = [i.id for i in matching]

    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    price_rows = (
        db.query(PriceHistory)
        .filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= cutoff,
        )
        .order_by(PriceHistory.item_id, PriceHistory.timestamp)
        .all()
    )
    prices_by_item: dict[int, list] = {}
    for pr in price_rows:
        prices_by_item.setdefault(pr.item_id, []).append(pr)

    latest_sub = (
        db.query(DailyAnalysis.item_id, DailyAnalysis.analysis_date)
        .distinct(DailyAnalysis.item_id)
        .order_by(DailyAnalysis.item_id, desc(DailyAnalysis.analysis_date))
        .subquery()
    )
    daily_rows = (
        db.query(DailyAnalysis)
        .join(
            latest_sub,
            (DailyAnalysis.item_id == latest_sub.c.item_id)
            & (DailyAnalysis.analysis_date == latest_sub.c.analysis_date),
        )
        .filter(DailyAnalysis.item_id.in_(item_ids))
        .all()
    )
    daily_map = {d.item_id: d for d in daily_rows}

    by_quality: dict[str, dict] = {}
    for i in matching:
        da = daily_map.get(i.id)
        ph_list = prices_by_item.get(i.id, [])

        current_price = None
        price_change_24h = None
        volume_24h = None

        if da and da.current_price:
            current_price = da.current_price
        elif ph_list:
            current_price = ph_list[-1].price

        if len(ph_list) >= 2:
            first = ph_list[0]
            last = ph_list[-1]
            if first.price > 0:
                price_change_24h = round(((last.price - first.price) / first.price) * 100, 2)
            volume_24h = sum((p.volume or 0) for p in ph_list)

        _, quality = _parse_item_name(i.name)
        quality = quality or "Standard"

        if quality not in by_quality or (current_price is not None and by_quality[quality].get("current_price") is None):
            by_quality[quality] = {
                "item_id": i.item_id,
                "name": i.name,
                "quality": quality,
                "current_price": current_price,
                "price_change_24h": price_change_24h,
                "volume_24h": volume_24h,
            }

    result = [QualityVariantOut(**v) for v in by_quality.values()]
    result.sort(key=lambda x: x.quality)
    return result


@router.get("/{item_id}", response_model=ItemOut)
def get_item(item_id: str, db: Session = Depends(get_db)):
    return _resolve_item(item_id, db)


@router.get("/{item_id}/price-history", response_model=list[PricePointOut])
def get_price_history(
    item_id: str,
    days: int = Query(30, ge=1, le=5000),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)

    # Use chart_points for deep history (>= 365 days) or all-time range
    if days >= 365:
        cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=days)
        records = (
            db.query(ChartPoint)
            .filter(
                ChartPoint.item_id == item.id,
                ChartPoint.day >= cutoff_date,
            )
            .order_by(ChartPoint.day)
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [
            PricePointOut(
                timestamp=datetime.combine(r.day, datetime.min.time()),
                price=r.close,
                volume=None,
                median_price=None,
            )
            for r in records
        ]

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    records = (
        db.query(PriceHistory)
        .filter(
            PriceHistory.item_id == item.id,
            PriceHistory.timestamp >= cutoff,
        )
        .order_by(PriceHistory.timestamp)
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        PricePointOut(
            timestamp=r.timestamp,
            price=r.price,
            volume=r.volume,
            median_price=r.median_price,
        )
        for r in records
    ]


@router.get("/{item_id}/trends", response_model=TrendAnalysisOut)
def get_item_trends(item_id: str, db: Session = Depends(get_db)):
    item = _resolve_item(item_id, db)
    latest_analysis = (
        db.query(DailyAnalysis)
        .filter(DailyAnalysis.item_id == item.id)
        .order_by(desc(DailyAnalysis.analysis_date))
        .first()
    )
    latest_price = (
        db.query(PriceHistory)
        .filter(PriceHistory.item_id == item.id)
        .order_by(desc(PriceHistory.timestamp))
        .first()
    )
    if latest_price is None:
        cp = (
            db.query(ChartPoint.close)
            .filter(ChartPoint.item_id == item.id)
            .order_by(desc(ChartPoint.day))
            .first()
        )
        current_price = cp.close if cp else 0.0
    else:
        current_price = latest_price.price
    trend_dir = "neutral"
    sma_7 = None
    sma_30 = None
    volatility = None
    trend_score = None

    if latest_analysis:
        trend_dir = latest_analysis.trend_direction or "neutral"
        sma_7 = latest_analysis.ma_7day
        sma_30 = latest_analysis.ma_30day
        volatility = latest_analysis.volatility
        trend_score = latest_analysis.momentum_score

    confidence = "low"
    if trend_score is not None:
        if abs(trend_score) > 50:
            confidence = "high"
        elif abs(trend_score) > 20:
            confidence = "medium"

    explanation = _build_trend_explanation(trend_dir, confidence, sma_7, current_price)
    return TrendAnalysisOut(
        item_id=item.id,
        item_name=item.name,
        current_price=current_price,
        trend_direction=trend_dir,
        confidence=confidence,
        sma_7=sma_7,
        sma_30=sma_30,
        volatility=volatility,
        trend_score=trend_score,
        explanation=explanation,
    )


def _build_trend_explanation(direction: str, confidence: str, sma_7, current_price) -> str:
    if direction == "bullish":
        return f"Price momentum is strong. Confidence is {confidence}."
    elif direction == "bearish":
        return f"Price showing downward momentum. Confidence is {confidence}."
    return f"Price is relatively stable. Confidence is {confidence}."


@router.get("/{item_id}/prediction", response_model=PredictionOut)
def get_item_prediction(
    item_id: str,
    period: str = Query("7_days", pattern="^(7_days|30_days)$"),
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    horizon = 7 if period == "7_days" else 30

    forecast = (
        db.query(ItemForecast)
        .filter(
            ItemForecast.item_id == item.id,
            ItemForecast.horizon_days == horizon,
        )
        .order_by(desc(ItemForecast.forecast_date))
        .first()
    )

    latest_price = (
        db.query(PriceHistory)
        .filter(PriceHistory.item_id == item.id)
        .order_by(desc(PriceHistory.timestamp))
        .first()
    )
    if latest_price is None:
        cp = (
            db.query(ChartPoint.close)
            .filter(ChartPoint.item_id == item.id)
            .order_by(desc(ChartPoint.day))
            .first()
        )
        current_price = cp.close if cp else 0.0
    else:
        current_price = latest_price.price

    if forecast:
        fl = forecast.price_low or current_price * 0.9
        fh = forecast.price_high or current_price * 1.1
        fm = forecast.price_mid or (fl + fh) / 2
        return PredictionOut(
            item_id=item.id,
            item_name=item.name,
            current_price=forecast.current_price or current_price,
            forecast_low=fl,
            forecast_mid=fm,
            forecast_high=fh,
            forecast_period=period,
            trend_direction=forecast.direction or "neutral",
            confidence=forecast.confidence or "low",
        )

    fl = current_price * 0.9
    fh = current_price * 1.1
    return PredictionOut(
        item_id=item.id,
        item_name=item.name,
        current_price=current_price,
        forecast_low=fl,
        forecast_mid=(fl + fh) / 2,
        forecast_high=fh,
        forecast_period=period,
        trend_direction="neutral",
        confidence="low",
    )


@router.get("/{item_id}/events", response_model=list[EventOut])
def get_item_events(
    item_id: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    event_ids = (
        db.query(EventImpact.event_id)
        .filter(EventImpact.item_id == item.id)
        .subquery()
    )
    events = (
        db.query(Event)
        .filter(Event.id.in_(event_ids))
        .order_by(desc(Event.timestamp))
        .limit(limit)
        .all()
    )
    return events


@router.get("/{item_id}/prices", response_model=MultiSourcePricesOut)
def get_multi_source_prices(
    item_id: str,
    source: str = Query("all", description="Comma-separated sources, or 'all' for every real source"),
    # Historical series reach back to 2013; the chart's "ALL" range needs
    # the full depth, not a one-year window.
    days: int = Query(30, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    requested = [s.strip() for s in source.split(",") if s.strip()]

    # Use chart_points for deep history (>= 365 days) or historical source
    if days >= 365 or "historical" in requested:
        cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=days)
        records = (
            db.query(ChartPoint)
            .filter(
                ChartPoint.item_id == item.id,
                ChartPoint.day >= cutoff_date,
            )
            .order_by(ChartPoint.day)
            .all()
        )
        data: dict[str, list[SourcePriceOut]] = {"historical": [
            SourcePriceOut(
                timestamp=datetime.combine(r.day, datetime.min.time()),
                price=r.close,
                volume=None,
                median_price=None,
            )
            for r in records
        ]}
        return MultiSourcePricesOut(
            item_id=item.item_id,
            name=item.name,
            sources=["historical"],
            data=data,
        )

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Single query for all requested sources instead of one per source.
    # Synthetic and fallback rows are copies/placeholders, never charted.
    query = (
        db.query(PriceHistory)
        .filter(
            PriceHistory.item_id == item.id,
            PriceHistory.timestamp >= cutoff,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        )
    )
    if requested and "all" not in requested:
        query = query.filter(PriceHistory.source.in_(requested))
    records = query.order_by(PriceHistory.source, PriceHistory.timestamp).all()

    data = {src: [] for src in requested if src != "all"}
    for r in records:
        data.setdefault(r.source, []).append(
            SourcePriceOut(
                timestamp=r.timestamp,
                price=r.price,
                volume=r.volume,
                median_price=r.median_price,
            )
        )

    return MultiSourcePricesOut(
        item_id=item.item_id,
        name=item.name,
        sources=[src for src, points in data.items() if points],
        data=data,
    )
