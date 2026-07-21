from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, text
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
import re
import math
import os
import json

from database import (
    get_db, Item, PriceHistory, ItemForecast,
    Event, EventImpact, EventCorrelation, backfilled_item_clause,
)
from api.cache import get_or_build
from api.schemas import (
    ItemOut, PricePointOut, TrendAnalysisOut, PredictionOut,
    SourcePriceOut, MultiSourcePricesOut, EventOut, TrendingItemOut,
    EventImpactOut, FeatureImportanceOut, FeatureImportanceItem,
    SocialMentionOut, SocialSentimentSummaryOut,
)

router = APIRouter(prefix="/items", tags=["items"])


def _resolve_item(item_id: str, db: Session) -> Item:
    item = db.query(Item).filter(Item.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.get("/count")
def items_count(db: Session = Depends(get_db)):
    return db.query(Item).filter(backfilled_item_clause()).count()


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
    """Latest price per item from price_history."""
    prices = {}
    for iid in item_ids:
        ph = (
            db.query(PriceHistory.price)
            .filter(PriceHistory.item_id == iid)
            .order_by(desc(PriceHistory.timestamp))
            .first()
        )
        if ph:
            prices[iid] = ph.price
    return prices


def _build_trending(db: Session, limit: int):
    from sqlalchemy import case
    from datetime import date

    today = date.today()
    subq = (
        db.query(
            ItemForecast.item_id,
            ItemForecast.forecast_date,
            ItemForecast.direction,
            ItemForecast.confidence,
            ItemForecast.price_mid,
            ItemForecast.current_price,
        )
        .filter(ItemForecast.forecast_date == today, ItemForecast.horizon_days == 7)
        .distinct(ItemForecast.item_id)
        .order_by(ItemForecast.item_id, desc(ItemForecast.forecast_date))
        .subquery()
    )

    confidence_order = case(
        (subq.c.confidence == "high", 3),
        (subq.c.confidence == "medium", 2),
        else_=1,
    )

    items = (
        db.query(Item, subq.c.direction, subq.c.confidence, subq.c.price_mid, subq.c.current_price)
        .outerjoin(subq, Item.id == subq.c.item_id)
        .filter(Item.icon_url.isnot(None), backfilled_item_clause())
        .order_by(desc(confidence_order), desc(subq.c.price_mid / func.nullif(subq.c.current_price, 0)))
        .limit(limit)
        .all()
    )
    item_ids = [i.Item.id for i in items]
    latest_prices = _latest_prices(db, item_ids) if item_ids else {}

    result = [
        TrendingItemOut(
            id=row.Item.id,
            item_id=row.Item.item_id,
            name=row.Item.name,
            type=row.Item.type,
            icon_url=row.Item.icon_url,
            latest_price=latest_prices.get(row.Item.id, 0.0),
        )
        for row in items
        if latest_prices.get(row.Item.id, 0.0) > 0
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

    by_quality: dict[str, dict] = {}
    for i in matching:
        ph_list = prices_by_item.get(i.id, [])

        current_price = None
        price_change_24h = None
        volume_24h = None

        if ph_list:
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

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    all_records = (
        db.query(PriceHistory)
        .filter(
            PriceHistory.item_id == item.id,
            PriceHistory.timestamp >= cutoff,
        )
        .order_by(PriceHistory.timestamp)
        .all()
    )
    all_prices = [r.price for r in all_records]
    if all_records:
        records = all_records[skip:skip + limit]
    else:
        records = []

    sma_7 = None
    sma_30 = None
    if len(all_prices) >= 7:
        sma_7 = sum(all_prices[-7:]) / 7
    if len(all_prices) >= 30:
        sma_30 = sum(all_prices[-30:]) / 30

    records_slice = records[skip:skip + limit]

    return [
        PricePointOut(
            timestamp=r.timestamp,
            price=r.price,
            volume=r.volume,
            median_price=r.median_price,
            sma_7=sma_7,
            sma_30=sma_30,
        )
        for r in records_slice
    ]


def _compute_bollinger_bands(prices, window=20, num_std=2):
    if len(prices) < window:
        return None, None, None
    recent = prices[-window:]
    sma = sum(recent) / window
    variance = sum((p - sma) ** 2 for p in recent) / window
    std = math.sqrt(variance)
    return sma + num_std * std, sma, sma - num_std * std

def _compute_rsi(prices, window=14):
    if len(prices) < window + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-window, 0):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / window
    avg_loss = losses / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _compute_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return None, None
    def ema(data, period):
        k = 2.0 / (period + 1)
        result = [data[0]]
        for v in data[1:]:
            result.append(v * k + result[-1] * (1 - k))
        return result
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema(macd_line, signal)
    return macd_line[-1], signal_line[-1]

def _compute_support_resistance(prices, window=20):
    if len(prices) < window:
        return None, None
    recent = prices[-window:]
    return min(recent), max(recent)


class _DictObj:
    def __init__(self, d):
        self.__dict__["_d"] = d
    def __getattr__(self, k):
        return self._d.get(k)


def _forecast_parquet(item_id: int, horizon_days: int = 7):
    from db.parquet import ParquetQuery
    with ParquetQuery("item_forecasts") as q:
        df = q.query(f"""
            SELECT * FROM item_forecasts
            WHERE item_id = {item_id} AND horizon_days = {horizon_days}
            ORDER BY forecast_date DESC
            LIMIT 1
        """)
        if df.empty:
            return None
        return _DictObj(df.iloc[0].to_dict())


def _trends_parquet(item, item_id: str, db: Session):
    r = _forecast_parquet(item.id, 7)
    if r is None:
        return None
    direction_map = {"up": "bullish", "down": "bearish", "flat": "neutral", None: "neutral"}
    trend_dir = direction_map.get(r.direction, "neutral")
    confidence = r.confidence if r.confidence else "low"
    latest_price = (
        db.query(PriceHistory)
        .filter(PriceHistory.item_id == item.id)
        .order_by(desc(PriceHistory.timestamp))
        .first()
    )
    current_price = latest_price.price if latest_price else 0.0
    explanation = _build_trend_explanation(trend_dir, confidence, current_price)
    price_points = [
        p.price for p in (
            db.query(PriceHistory)
            .filter(PriceHistory.item_id == item.id)
            .order_by(PriceHistory.timestamp)
            .all()
        )
    ]
    sma_7 = sum(price_points[-7:]) / 7 if len(price_points) >= 7 else None
    sma_30 = sum(price_points[-30:]) / 30 if len(price_points) >= 30 else None
    bollinger_upper, bollinger_middle, bollinger_lower = _compute_bollinger_bands(price_points)
    rsi = _compute_rsi(price_points)
    macd, macd_signal = _compute_macd(price_points) if price_points else (None, None)
    support, resistance = _compute_support_resistance(price_points)
    factors = []
    if rsi is not None:
        if rsi > 70:
            factors.append("RSI overbought (>70)")
        elif rsi < 30:
            factors.append("RSI oversold (<30)")
    if trend_dir == "bullish":
        factors.append("Forecast predicts upward movement")
    elif trend_dir == "bearish":
        factors.append("Forecast predicts downward movement")
    if support is not None and resistance is not None:
        band_width = ((resistance - support) / support) * 100
        factors.append(f"Trading range: {band_width:.1f}%")
    return TrendAnalysisOut(
        item_id=item.id,
        item_name=item.name,
        current_price=current_price,
        trend_direction=trend_dir,
        confidence=confidence,
        explanation=explanation,
        rsi=rsi,
        bollinger_upper=bollinger_upper,
        bollinger_middle=bollinger_middle,
        bollinger_lower=bollinger_lower,
        macd=macd,
        macd_signal=macd_signal,
        support=support,
        resistance=resistance,
        factors=factors,
        sma_7=sma_7,
        sma_30=sma_30,
    )


@router.get("/{item_id}/trends", response_model=TrendAnalysisOut)
def get_item_trends(item_id: str, db: Session = Depends(get_db)):
    item = _resolve_item(item_id, db)

    try:
        result = _trends_parquet(item, item_id, db)
        if result is not None:
            return result
    except Exception:
        pass

    latest_forecast = (
        db.query(ItemForecast)
        .filter(
            ItemForecast.item_id == item.id,
            ItemForecast.horizon_days == 7,
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
    current_price = latest_price.price if latest_price else 0.0

    direction_map = {"up": "bullish", "down": "bearish", "flat": "neutral", None: "neutral"}
    trend_dir = direction_map.get(latest_forecast.direction if latest_forecast else None, "neutral")
    confidence = latest_forecast.confidence if latest_forecast and latest_forecast.confidence else "low"

    explanation = _build_trend_explanation(trend_dir, confidence, current_price)

    price_points = [
        r.price for r in (
            db.query(PriceHistory)
            .filter(PriceHistory.item_id == item.id)
            .order_by(PriceHistory.timestamp)
            .all()
        )
    ]

    # Compute SMAs from raw price history
    sma_7 = None
    sma_30 = None
    if len(price_points) >= 7:
        sma_7 = sum(price_points[-7:]) / 7
    if len(price_points) >= 30:
        sma_30 = sum(price_points[-30:]) / 30

    bollinger_upper, bollinger_middle, bollinger_lower = _compute_bollinger_bands(price_points)
    rsi = _compute_rsi(price_points)
    macd, macd_signal = _compute_macd(price_points) if price_points else (None, None)
    support, resistance = _compute_support_resistance(price_points)

    factors = []
    if rsi is not None:
        if rsi > 70:
            factors.append("RSI overbought (>70)")
        elif rsi < 30:
            factors.append("RSI oversold (<30)")
    if trend_dir == "bullish":
        factors.append("Forecast predicts upward movement")
    elif trend_dir == "bearish":
        factors.append("Forecast predicts downward movement")
    if support is not None and resistance is not None:
        band_width = ((resistance - support) / support) * 100
        factors.append(f"Trading range: {band_width:.1f}%")

    return TrendAnalysisOut(
        item_id=item.id,
        item_name=item.name,
        current_price=current_price,
        trend_direction=trend_dir,
        confidence=confidence,
        explanation=explanation,
        rsi=rsi,
        bollinger_upper=bollinger_upper,
        bollinger_middle=bollinger_middle,
        bollinger_lower=bollinger_lower,
        macd=macd,
        macd_signal=macd_signal,
        support=support,
        resistance=resistance,
        factors=factors,
        sma_7=sma_7,
        sma_30=sma_30,
    )


def _build_trend_explanation(direction: str, confidence: str, current_price) -> str:
    if direction == "bullish":
        return f"ML forecast predicts upward movement. Confidence is {confidence}."
    elif direction == "bearish":
        return f"ML forecast predicts downward movement. Confidence is {confidence}."
    return f"ML forecast predicts stable price. Confidence is {confidence}."


def _prediction_parquet(item, period: str, horizon: int):
    r = _forecast_parquet(item.id, horizon)
    if r is None:
        return None
    current_price = r.current_price or 0.0
    fl = r.price_low or current_price * 0.9
    fh = r.price_high or current_price * 1.1
    fm = r.price_mid or (fl + fh) / 2
    return PredictionOut(
        item_id=item.id,
        item_name=item.name,
        current_price=current_price,
        forecast_low=fl,
        forecast_mid=fm,
        forecast_high=fh,
        forecast_period=period,
        trend_direction=r.direction or "neutral",
        confidence=r.confidence or "low",
    )


@router.get("/{item_id}/prediction", response_model=PredictionOut)
def get_item_prediction(
    item_id: str,
    period: str = Query("7_days", pattern="^(3_days|7_days|14_days|30_days)$"),
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    horizon = {"3_days": 3, "7_days": 7, "14_days": 14, "30_days": 30}[period]

    try:
        result = _prediction_parquet(item, period, horizon)
        if result is not None:
            return result
    except Exception:
        pass

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
    current_price = latest_price.price if latest_price else 0.0

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


def _item_events_parquet(item_id: int, limit: int):
    from db.parquet import ParquetQuery
    with ParquetQuery("event_impacts_denorm") as q:
        df = q.query(f"""
            SELECT DISTINCT event_id, event_type, event_description, event_timestamp
            FROM event_impacts_denorm
            WHERE item_id = {item_id}
            ORDER BY event_timestamp DESC
            LIMIT {limit}
        """)
        if df.empty:
            return []
        return [
            EventOut(
                id=int(r.event_id),
                type=str(r.event_type),
                timestamp=r.event_timestamp,
                description=str(r.event_description),
                created_at=r.event_timestamp,
            )
            for r in df.itertuples()
        ]


def _event_impacts_parquet(item_id: int, limit: int):
    from db.parquet import ParquetQuery
    with ParquetQuery("event_impacts_denorm") as q:
        df = q.query(f"""
            SELECT event_id, event_type, event_description, event_timestamp,
                   price_day_before, price_day_1, price_day_3, price_day_7,
                   impact_pct_1day, impact_pct_3day, impact_pct_7day,
                   peak_impact_pct, peak_impact_day, duration_days, z_score,
                   confidence_score
            FROM event_impacts_denorm
            WHERE item_id = {item_id}
            ORDER BY event_timestamp DESC
            LIMIT {limit}
        """)
        if df.empty:
            return []
        result = []
        for r in df.itertuples():
            result.append(EventImpactOut(
                event_id=int(r.event_id),
                event_type=str(r.event_type),
                event_description=str(r.event_description),
                event_timestamp=r.event_timestamp,
                price_day_before=r.price_day_before,
                price_day_1=r.price_day_1,
                price_day_3=r.price_day_3,
                price_day_7=r.price_day_7,
                impact_pct_1day=r.impact_pct_1day,
                impact_pct_3day=r.impact_pct_3day,
                impact_pct_7day=r.impact_pct_7day,
                peak_impact_pct=r.peak_impact_pct,
                peak_impact_day=int(r.peak_impact_day) if r.peak_impact_day is not None and not (isinstance(r.peak_impact_day, float) and r.peak_impact_day != r.peak_impact_day) else None,
                duration_days=int(r.duration_days) if r.duration_days is not None and not (isinstance(r.duration_days, float) and r.duration_days != r.duration_days) else None,
                z_score=r.z_score,
                confidence_score=r.confidence_score,
            ))
        return result


@router.get("/{item_id}/events", response_model=list[EventOut])
def get_item_events(
    item_id: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    try:
        return _item_events_parquet(item.id, limit)
    except Exception:
        pass
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


@router.get("/{item_id}/event-impacts", response_model=list[EventImpactOut])
def get_item_event_impacts(
    item_id: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    try:
        return _event_impacts_parquet(item.id, limit)
    except Exception:
        pass
    rows = (
        db.query(EventImpact, Event, EventCorrelation.confidence_score)
        .join(Event, Event.id == EventImpact.event_id)
        .outerjoin(
            EventCorrelation,
            (EventCorrelation.event_id == EventImpact.event_id) &
            (EventCorrelation.item_id == EventImpact.item_id),
        )
        .filter(EventImpact.item_id == item.id)
        .order_by(desc(Event.timestamp))
        .limit(limit)
        .all()
    )
    result = []
    for impact, event, confidence in rows:
        result.append(EventImpactOut(
            event_id=event.id,
            event_type=event.type,
            event_description=event.description,
            event_timestamp=event.timestamp,
            price_day_before=impact.price_day_before,
            price_day_1=impact.price_day_1,
            price_day_3=impact.price_day_3,
            price_day_7=impact.price_day_7,
            impact_pct_1day=impact.impact_pct_1day,
            impact_pct_3day=impact.impact_pct_3day,
            impact_pct_7day=impact.impact_pct_7day,
            peak_impact_pct=impact.peak_impact_pct,
            peak_impact_day=impact.peak_impact_day,
            duration_days=impact.duration_days,
            z_score=impact.z_score,
            confidence_score=confidence,
        ))
    return result


@router.get("/{item_id}/feature-importance", response_model=FeatureImportanceOut)
def get_item_feature_importance(
    item_id: str,
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    meta_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved_models", "meta.json")
    import json
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="No trained model found")

    with open(meta_path) as f:
        meta = json.load(f)

    fi_raw = meta.get("feature_importance", {})
    horizons = {}
    for h_str, items in fi_raw.items():
        horizons[h_str] = [FeatureImportanceItem(**i) for i in items]

    return FeatureImportanceOut(
        item_id=item.item_id,
        item_name=item.name,
        horizons=horizons,
    )


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

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    data: dict[str, list[SourcePriceOut]] = {}

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

    for r in records:
        data.setdefault(r.source, []).append(
            SourcePriceOut(
                timestamp=r.timestamp,
                price=r.price,
                volume=r.volume,
                median_price=r.median_price,
            )
        )

    sources = [s for s in data if data[s]]

    return MultiSourcePricesOut(
        item_id=item.item_id,
        name=item.name,
        sources=sources,
        data=data,
    )


def _social_sentiment_parquet(item_id: int, item_slug: str, item_name: str):
    from db.parquet import ParquetQuery
    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(days=1)
    cutoff_7d = now - timedelta(days=7)

    with ParquetQuery("social_mentions") as q:
        all_df = q.query("SELECT * FROM social_mentions")
        if all_df.empty:
            return SocialSentimentSummaryOut(
                item_id=item_slug, item_name=item_name,
                mentions_24h=0, mentions_7d=0, mention_velocity=0,
                avg_sentiment_7d=0, avg_score_7d=0, recent_mentions=[],
            )
        item_df = all_df[all_df["item_id"] == item_id]
        if item_df.empty:
            return SocialSentimentSummaryOut(
                item_id=item_slug, item_name=item_name,
                mentions_24h=0, mentions_7d=0, mention_velocity=0,
                avg_sentiment_7d=0, avg_score_7d=0, recent_mentions=[],
            )
        item_df = item_df[item_df["source"] == "reddit"]
        if item_df.empty:
            return SocialSentimentSummaryOut(
                item_id=item_slug, item_name=item_name,
                mentions_24h=0, mentions_7d=0, mention_velocity=0,
                avg_sentiment_7d=0, avg_score_7d=0, recent_mentions=[],
            )
        mentions_24h = int(len(item_df[item_df["mentioned_at"] >= cutoff_24h]))
        mentions_7d = int(len(item_df[item_df["mentioned_at"] >= cutoff_7d]))
        avg_sent = float(item_df[item_df["mentioned_at"] >= cutoff_7d]["sentiment_score"].mean()) if mentions_7d > 0 else 0.0
        avg_score = float(item_df[item_df["mentioned_at"] >= cutoff_7d]["post_score"].mean()) if mentions_7d > 0 else 0.0
        mention_velocity = mentions_24h / max(mentions_7d, 1)
        recent = item_df.sort_values("mentioned_at", ascending=False).head(20)
        recent_mentions = [
            SocialMentionOut(
                post_id=str(r.post_id),
                subreddit=str(r.subreddit) if r.subreddit else None,
                post_title=str(r.post_title) if r.post_title else None,
                post_score=int(r.post_score) if r.post_score else None,
                sentiment_score=float(r.sentiment_score),
                mentioned_at=r.mentioned_at,
            )
            for r in recent.itertuples()
        ]
        return SocialSentimentSummaryOut(
            item_id=item_slug,
            item_name=item_name,
            mentions_24h=mentions_24h,
            mentions_7d=mentions_7d,
            mention_velocity=mention_velocity,
            avg_sentiment_7d=avg_sent,
            avg_score_7d=avg_score,
            recent_mentions=recent_mentions,
        )


@router.get("/{item_id}/social-sentiment", response_model=SocialSentimentSummaryOut)
def item_social_sentiment(
    item_id: str,
    db: Session = Depends(get_db),
):
    item = _resolve_item(item_id, db)
    try:
        return _social_sentiment_parquet(item.id, item.item_id, item.name)
    except Exception:
        pass
    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(days=1)
    cutoff_7d = now - timedelta(days=7)

    mentions_24h = db.execute(text("""
        SELECT COUNT(*) FROM social_mentions
        WHERE item_id = :iid AND source = 'reddit' AND mentioned_at >= :cutoff
    """), {"iid": item.id, "cutoff": cutoff_24h}).scalar() or 0

    mentions_7d = db.execute(text("""
        SELECT COUNT(*) FROM social_mentions
        WHERE item_id = :iid AND source = 'reddit' AND mentioned_at >= :cutoff
    """), {"iid": item.id, "cutoff": cutoff_7d}).scalar() or 0

    sentiment_row = db.execute(text("""
        SELECT AVG(sentiment_score) AS avg_sent, AVG(post_score) AS avg_score
        FROM social_mentions
        WHERE item_id = :iid AND source = 'reddit' AND mentioned_at >= :cutoff
    """), {"iid": item.id, "cutoff": cutoff_7d}).first()
    avg_sent = float(sentiment_row.avg_sent) if sentiment_row and sentiment_row.avg_sent else 0.0
    avg_score = float(sentiment_row.avg_score) if sentiment_row and sentiment_row.avg_score else 0.0

    mention_velocity = mentions_24h / max(mentions_7d, 1)

    recent_rows = db.execute(text("""
        SELECT post_id, subreddit, post_title, post_score,
               sentiment_score, mentioned_at
        FROM social_mentions
        WHERE item_id = :iid AND source = 'reddit'
        ORDER BY mentioned_at DESC
        LIMIT 20
    """), {"iid": item.id}).fetchall()

    recent_mentions = [
        SocialMentionOut(
            post_id=r.post_id,
            subreddit=r.subreddit,
            post_title=r.post_title,
            post_score=r.post_score,
            sentiment_score=r.sentiment_score,
            mentioned_at=r.mentioned_at,
        )
        for r in recent_rows
    ]

    return SocialSentimentSummaryOut(
        item_id=item.item_id,
        item_name=item.name,
        mentions_24h=mentions_24h,
        mentions_7d=mentions_7d,
        mention_velocity=mention_velocity,
        avg_sentiment_7d=avg_sent,
        avg_score_7d=avg_score,
        recent_mentions=recent_mentions,
    )
