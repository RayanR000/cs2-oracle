import json
import logging
import math
import os
from datetime import UTC, datetime, timedelta

from database import (
    Event,
    EventCorrelation,
    EventImpact,
    Item,
    ItemForecast,
    PriceHistory,
    backfilled_item_clause,
    get_db,
)
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from api.cache import get_or_build
from api.schemas import (
    EventImpactOut,
    EventOut,
    FeatureImportanceItem,
    FeatureImportanceOut,
    ItemOut,
    MultiSourcePricesOut,
    PredictionOut,
    PricePointOut,
    QualityVariantOut,
    SourcePriceOut,
    TrendAnalysisOut,
    TrendingItemOut,
    VolatilityRankOut,
    parse_item_name,
)
from api.serving_policy import (
    MAX_ARCHIVE_LAG_DAYS,
    MIN_SERVED_PRICE_USD,
    SERVED_HORIZONS,
    price_floor_clause,
    served_direction,
)
from api.volatility_tags import (
    CALIBRATED_MOVE_ODDS_HORIZONS,
    build_ranking,
    compute_thresholds,
    move_odds_calibrated,
    swing_pct,
    tag_fields,
)

router = APIRouter(prefix="/items", tags=["items"])

_DB_DEP = Depends(get_db)

logger = logging.getLogger(__name__)


def _resolve_item(item_id: str, db: Session) -> Item:
    item = db.query(Item).filter(Item.item_id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.get("/count")
def items_count(db: Session = _DB_DEP):
    return db.query(Item).filter(backfilled_item_clause()).count()


@router.get("/", response_model=list[ItemOut])
def list_items(
    type: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = _DB_DEP,
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
    db: Session = _DB_DEP,
):
    return (
        db.query(Item).filter(Item.name.ilike(f"%{q}%"), backfilled_item_clause()).order_by(Item.name).limit(50).all()
    )


@router.get("/trending", response_model=list[TrendingItemOut])
def trending_items(
    limit: int = Query(10, ge=1, le=100),
    db: Session = _DB_DEP,
):
    return get_or_build(f"items_trending:{limit}", 600, lambda: _build_trending(db, limit))


def _latest_prices(db: Session, item_ids: list[int]) -> dict[int, float]:
    """Latest price per item from price_history (single batched query)."""
    if not item_ids:
        return {}
    subq = (
        db.query(
            PriceHistory.item_id,
            PriceHistory.price,
            func.row_number()
            .over(
                partition_by=PriceHistory.item_id,
                order_by=PriceHistory.timestamp.desc(),
            )
            .label("rn"),
        )
        .filter(PriceHistory.item_id.in_(item_ids))
        .subquery()
    )
    rows = db.query(subq.c.item_id, subq.c.price).filter(subq.c.rn == 1).all()
    return {r.item_id: float(r.price) for r in rows}


def _build_trending(db: Session, limit: int):
    from datetime import date, timedelta

    today = date.today()
    # `forecast_date` is the day the band was anchored on -- the newest archived
    # day, which lags the calendar (the dump lands ~22:00 UTC), so it is usually
    # today-1. A `== today` pin empties this list after every normal run; the
    # window keeps it a freshness guard while tolerating archive lag. The
    # distinct-on below still selects each item's newest forecast within it.
    freshness_floor = today - timedelta(days=MAX_ARCHIVE_LAG_DAYS)
    subq = (
        db.query(
            ItemForecast.item_id,
            ItemForecast.forecast_date,
            ItemForecast.direction,
            ItemForecast.price_mid,
            ItemForecast.current_price,
        )
        .filter(
            ItemForecast.forecast_date >= freshness_floor,
            ItemForecast.horizon_days == 7,
            price_floor_clause(ItemForecast.current_price),
        )
        .distinct(ItemForecast.item_id)
        .order_by(ItemForecast.item_id, desc(ItemForecast.forecast_date))
        .subquery()
    )

    # An inner join: an item with no fresh h=7 forecast has no ratio and is not
    # trending. The outer join it replaced sorted those NULL ratios FIRST on
    # Postgres (DESC defaults to NULLS FIRST; SQLite puts them last, so the
    # tests never saw it) and on 2026-10-09 the whole prod list was items with
    # no forecast at all, 1,805 of 2,376 listed. nulls_last() still covers a
    # NULL price_mid on a forecast row.
    ratio = subq.c.price_mid / func.nullif(subq.c.current_price, 0)
    items = (
        db.query(Item, subq.c.current_price)
        .join(subq, Item.id == subq.c.item_id)
        .filter(Item.icon_url.isnot(None), backfilled_item_clause())
        .order_by(desc(ratio).nulls_last(), Item.id)
        .limit(limit)
        .all()
    )

    # The price is the forecast's own quote, which the floor already holds at
    # >= $1. price_history is not written by the aggregator (it stopped at
    # 2026-07-11, 7,139 rows), so reading it here returned a months-old price
    # or none, and a missing one dropped the row AFTER the LIMIT.
    return [
        TrendingItemOut(
            id=row.Item.id,
            item_id=row.Item.item_id,
            name=row.Item.name,
            type=row.Item.type,
            icon_url=row.Item.icon_url,
            latest_price=float(row.current_price),
        )
        for row in items
    ]


@router.get("/volatility", response_model=list[VolatilityRankOut])
def get_volatility_ranking(
    horizon: int = Query(7, description="Forecast horizon in days"),
    sort: str = Query("swing", pattern="^(swing|move_odds)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    min_price: float = Query(MIN_SERVED_PRICE_USD, ge=0.0),
    limit: int = Query(100, ge=1, le=1000),
    label: str | None = Query(
        None, pattern="^(Stable|Moderate|Volatile)$", description="Keep only items with this stability class"
    ),
    db: Session = _DB_DEP,
):
    """Rank the served universe by volatility for a horizon.

    Each item carries its expected swing (half-band / mid), its move odds
    (`exceed_p`, the magnitude signal, published only at calibrated horizons
    h3/h7), and a Stable/Moderate/Volatile label from the within-horizon
    universe tertiles. The label is relative-to-peers. `label`, when given,
    returns only that stability class (computed over the full universe first,
    then filtered).
    """
    if horizon not in SERVED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=(f"horizon must be one of {', '.join(map(str, SERVED_HORIZONS))}; got {horizon}"),
        )
    if sort == "move_odds" and not move_odds_calibrated(horizon):
        raise HTTPException(
            status_code=400,
            detail=(
                f"move_odds is calibrated only at horizons "
                f"{', '.join(map(str, CALIBRATED_MOVE_ODDS_HORIZONS))}; "
                f"cannot sort by it at horizon {horizon}"
            ),
        )
    return get_or_build(
        f"items_volatility:{horizon}:{sort}:{order}:{min_price}:{limit}:{label}",
        600,
        lambda: _volatility_ranking(db, horizon, sort, order, min_price, limit, label),
    )


def _volatility_ranking(
    db: Session, horizon: int, sort: str, order: str, min_price: float, limit: int, label: str | None = None
):
    from datetime import date, timedelta

    freshness_floor = date.today() - timedelta(days=MAX_ARCHIVE_LAG_DAYS)
    subq = (
        db.query(
            ItemForecast.item_id,
            ItemForecast.forecast_date,
            ItemForecast.price_low,
            ItemForecast.price_mid,
            ItemForecast.price_high,
            ItemForecast.current_price,
            ItemForecast.exceed_p,
        )
        .filter(
            ItemForecast.forecast_date >= freshness_floor,
            ItemForecast.horizon_days == horizon,
            ItemForecast.current_price >= min_price,
        )
        .distinct(ItemForecast.item_id)
        .order_by(ItemForecast.item_id, desc(ItemForecast.forecast_date))
        .subquery()
    )
    rows = (
        db.query(
            Item.item_id,
            Item.name,
            subq.c.price_low,
            subq.c.price_mid,
            subq.c.price_high,
            subq.c.current_price,
            subq.c.exceed_p,
        )
        .join(subq, Item.id == subq.c.item_id)
        .filter(backfilled_item_clause())
        .all()
    )
    universe = [
        dict(
            item_id=r.item_id,
            name=r.name,
            current_price=r.current_price,
            low=r.price_low,
            high=r.price_high,
            mid=r.price_mid,
            exceed_p=r.exceed_p,
        )
        for r in rows
    ]
    ranked = build_ranking(
        universe, sort=sort, order=order, limit=limit, calibrated_move_odds=move_odds_calibrated(horizon), label=label
    )
    return [VolatilityRankOut(**t) for t in ranked]


# The stability label on a single item is relative to that horizon's whole
# universe, so a per-item lookup needs the universe's swing tertiles. Computing
# them per request would query every served item; memoise per (horizon, day)
# since the served bands do not change intra-day.
_SWING_THRESHOLD_CACHE: dict = {}


def _horizon_swing_thresholds(db: Session, horizon: int):
    from datetime import date, timedelta

    key = (horizon, date.today())
    if key in _SWING_THRESHOLD_CACHE:
        return _SWING_THRESHOLD_CACHE[key]

    freshness_floor = date.today() - timedelta(days=MAX_ARCHIVE_LAG_DAYS)
    subq = (
        db.query(
            ItemForecast.item_id,
            ItemForecast.forecast_date,
            ItemForecast.price_low,
            ItemForecast.price_mid,
            ItemForecast.price_high,
        )
        .filter(
            ItemForecast.forecast_date >= freshness_floor,
            ItemForecast.horizon_days == horizon,
            price_floor_clause(ItemForecast.current_price),
        )
        .distinct(ItemForecast.item_id)
        .order_by(ItemForecast.item_id, desc(ItemForecast.forecast_date))
        .subquery()
    )
    rows = db.query(subq.c.price_low, subq.c.price_mid, subq.c.price_high).all()
    swings = [s for s in (swing_pct(r.price_low, r.price_high, r.price_mid) for r in rows) if s is not None]
    thresholds = compute_thresholds(swings) if swings else None
    _SWING_THRESHOLD_CACHE[key] = thresholds
    return thresholds


@router.get("/{item_id}/variants", response_model=list[QualityVariantOut])
def get_item_variants(
    item_id: str,
    db: Session = _DB_DEP,
):
    item = _resolve_item(item_id, db)
    base_name, _ = parse_item_name(item.name)

    all_items = (
        db.query(Item)
        .filter(
            Item.name.ilike(f"%{base_name}%"),
            Item.type == item.type,
        )
        .all()
    )

    matching = [i for i in all_items if parse_item_name(i.name)[0] == base_name]
    if not matching:
        matching = [item]

    item_ids = [i.id for i in matching]

    cutoff = datetime.now(UTC) - timedelta(days=2)
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

        _, quality = parse_item_name(i.name)
        quality = quality or "Standard"

        if quality not in by_quality or (
            current_price is not None and by_quality[quality].get("current_price") is None
        ):
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
def get_item(item_id: str, db: Session = _DB_DEP):
    return _resolve_item(item_id, db)


@router.get("/{item_id}/price-history", response_model=list[PricePointOut])
def get_price_history(
    item_id: str,
    days: int = Query(30, ge=1, le=5000),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = _DB_DEP,
):
    item = _resolve_item(item_id, db)

    cutoff = datetime.now(UTC) - timedelta(days=days)
    # SMA computation — only needs the last 30 prices (lightweight query)
    sma_prices = (
        db.query(PriceHistory.price)
        .filter(
            PriceHistory.item_id == item.id,
            PriceHistory.timestamp >= cutoff,
        )
        .order_by(desc(PriceHistory.timestamp))
        .limit(30)
        .all()
    )
    sma_values = [r.price for r in reversed(sma_prices)]

    # Paginated main query — page slice is applied in SQL, not memory
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

    sma_7 = None
    sma_30 = None
    if len(sma_values) >= 7:
        sma_7 = sum(sma_values[-7:]) / 7
    if len(sma_values) >= 30:
        sma_30 = sum(sma_values[-30:]) / 30

    records_slice = records

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


def _trend_indicators(price_points: list) -> dict:
    """SMA/Bollinger/RSI/MACD/support-resistance/factors from a price list.

    Shared by the parquet and DB-fallback trend paths, which differ only in
    how they resolve the trend direction and current price.
    """
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
    if support is not None and resistance is not None:
        band_width = ((resistance - support) / support) * 100
        factors.append(f"Trading range: {band_width:.1f}%")
    return {
        "sma_7": sma_7,
        "sma_30": sma_30,
        "bollinger_upper": bollinger_upper,
        "bollinger_middle": bollinger_middle,
        "bollinger_lower": bollinger_lower,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "support": support,
        "resistance": resistance,
        "factors": factors,
    }


def _recent_price_points(db: Session, item_db_id: int, n: int = 35) -> list[float]:
    """Last `n` prices in chronological order — the full window every trend
    indicator needs (MACD slow 26 + signal 9 = 35, SMA-30, Bollinger 20)."""
    rows = (
        db.query(PriceHistory.price)
        .filter(PriceHistory.item_id == item_db_id)
        .order_by(desc(PriceHistory.timestamp))
        .limit(n)
        .all()
    )
    return [r.price for r in reversed(rows)]


def _build_trend_response(item, trend_dir: str, price_points: list) -> TrendAnalysisOut:
    current_price = price_points[-1] if price_points else 0.0
    ind = _trend_indicators(price_points)
    return TrendAnalysisOut(
        item_id=item.id,
        item_name=item.name,
        current_price=current_price,
        trend_direction=trend_dir,
        explanation=_build_trend_explanation(),
        rsi=ind["rsi"],
        bollinger_upper=ind["bollinger_upper"],
        bollinger_middle=ind["bollinger_middle"],
        bollinger_lower=ind["bollinger_lower"],
        macd=ind["macd"],
        macd_signal=ind["macd_signal"],
        support=ind["support"],
        resistance=ind["resistance"],
        factors=ind["factors"],
        sma_7=ind["sma_7"],
        sma_30=ind["sma_30"],
    )


def _trends_parquet(item, item_id: str, db: Session):
    r = _forecast_parquet(item.id, 7)
    if r is None:
        return None
    trend_dir = served_direction(r.direction, 7)
    price_points = _recent_price_points(db, item.id)
    return _build_trend_response(item, trend_dir, price_points)


@router.get("/{item_id}/trends", response_model=TrendAnalysisOut)
def get_item_trends(item_id: str, db: Session = _DB_DEP):
    item = _resolve_item(item_id, db)

    try:
        result = _trends_parquet(item, item_id, db)
        if result is not None:
            return result
    except Exception as e:
        logger.debug("parquet trends read failed, falling back to DB: %s", e)

    latest_forecast = (
        db.query(ItemForecast)
        .filter(
            ItemForecast.item_id == item.id,
            ItemForecast.horizon_days == 7,
        )
        .order_by(desc(ItemForecast.forecast_date))
        .first()
    )

    price_points = _recent_price_points(db, item.id)
    trend_dir = served_direction(latest_forecast.direction if latest_forecast else None, 7)
    return _build_trend_response(item, trend_dir, price_points)


def _build_trend_explanation() -> str:
    return "Forecast range covers expected price movement over the next 7 days."


def _optional_bool(v):
    """NULL / NaN out of DuckDB is 'not recorded', never False.

    A column the mirror predates comes back as None from `_DictObj`, and a
    Parquet NULL in a float or object column arrives as `nan` -- and `bool(nan)`
    is True, which would publish an unrecorded anchor as clean.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return bool(v)


def _optional_float(v):
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) else f


def _prediction_parquet(item, period: str, horizon: int, thresholds=None):
    r = _forecast_parquet(item.id, horizon)
    if r is None:
        return None
    current_price = r.current_price or 0.0
    fl = r.price_low or current_price * 0.9
    fh = r.price_high or current_price * 1.1
    fm = r.price_mid or (fl + fh) / 2
    tags = tag_fields(
        fl, fh, fm, _optional_float(r.exceed_p), thresholds, calibrated_move_odds=move_odds_calibrated(horizon)
    )
    return PredictionOut(
        item_id=item.id,
        item_name=item.name,
        current_price=current_price,
        forecast_low=fl,
        forecast_mid=fm,
        forecast_high=fh,
        forecast_period=period,
        trend_direction=served_direction(r.direction, horizon),
        # Disclosed, not gated. `/opportunities` drops the deviating cohort
        # because ranking is what the clean-anchor evidence covers; a lookup by
        # name still answers, and says which cohort the answer comes from.
        # `_DictObj` returns None for a column the mirror predates.
        anchor_clean=_optional_bool(r.anchor_clean),
        anchor_wedge_pct=_optional_float(r.anchor_wedge_pct),
        anomaly_p=_optional_float(r.anomaly_p),
        **tags,
    )


@router.get("/{item_id}/prediction", response_model=PredictionOut)
def get_item_prediction(
    item_id: str,
    period: str = Query("7_days", pattern="^(3_days|7_days|14_days|30_days)$"),
    db: Session = _DB_DEP,
):
    item = _resolve_item(item_id, db)
    horizon = {"3_days": 3, "7_days": 7, "14_days": 14, "30_days": 30}[period]
    thresholds = _horizon_swing_thresholds(db, horizon)

    try:
        result = _prediction_parquet(item, period, horizon, thresholds)
        if result is not None:
            return result
    except Exception as e:
        logger.debug("parquet prediction read failed, falling back to DB: %s", e)

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
        db.query(PriceHistory).filter(PriceHistory.item_id == item.id).order_by(desc(PriceHistory.timestamp)).first()
    )
    current_price = latest_price.price if latest_price else 0.0

    if forecast:
        fl = forecast.price_low or current_price * 0.9
        fh = forecast.price_high or current_price * 1.1
        fm = forecast.price_mid or (fl + fh) / 2
        tags = tag_fields(fl, fh, fm, forecast.exceed_p, thresholds, calibrated_move_odds=move_odds_calibrated(horizon))
        return PredictionOut(
            item_id=item.id,
            item_name=item.name,
            current_price=forecast.current_price or current_price,
            forecast_low=fl,
            forecast_mid=fm,
            forecast_high=fh,
            forecast_period=period,
            trend_direction=served_direction(forecast.direction, horizon),
            anomaly_p=forecast.anomaly_p,
            **tags,
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
        trend_direction=served_direction(None, horizon),
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
            result.append(
                EventImpactOut(
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
                    peak_impact_day=int(r.peak_impact_day)
                    if r.peak_impact_day is not None
                    and not (isinstance(r.peak_impact_day, float) and r.peak_impact_day != r.peak_impact_day)
                    else None,
                    duration_days=int(r.duration_days)
                    if r.duration_days is not None
                    and not (isinstance(r.duration_days, float) and r.duration_days != r.duration_days)
                    else None,
                    z_score=r.z_score,
                    confidence_score=r.confidence_score,
                )
            )
        return result


@router.get("/{item_id}/events", response_model=list[EventOut])
def get_item_events(
    item_id: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = _DB_DEP,
):
    item = _resolve_item(item_id, db)
    try:
        return _item_events_parquet(item.id, limit)
    except Exception as e:
        logger.debug("parquet events read failed, falling back to DB: %s", e)
    event_ids = db.query(EventImpact.event_id).filter(EventImpact.item_id == item.id).subquery()
    events = db.query(Event).filter(Event.id.in_(event_ids)).order_by(desc(Event.timestamp)).limit(limit).all()
    return events


@router.get("/{item_id}/event-impacts", response_model=list[EventImpactOut])
def get_item_event_impacts(
    item_id: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = _DB_DEP,
):
    item = _resolve_item(item_id, db)
    try:
        return _event_impacts_parquet(item.id, limit)
    except Exception as e:
        logger.debug("parquet event-impacts read failed, falling back to DB: %s", e)
    rows = (
        db.query(EventImpact, Event, EventCorrelation.confidence_score)
        .join(Event, Event.id == EventImpact.event_id)
        .outerjoin(
            EventCorrelation,
            (EventCorrelation.event_id == EventImpact.event_id) & (EventCorrelation.item_id == EventImpact.item_id),
        )
        .filter(EventImpact.item_id == item.id)
        .order_by(desc(Event.timestamp))
        .limit(limit)
        .all()
    )
    result = []
    for impact, event, confidence in rows:
        result.append(
            EventImpactOut(
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
            )
        )
    return result


@router.get("/{item_id}/feature-importance", response_model=FeatureImportanceOut)
def get_item_feature_importance(
    item_id: str,
    db: Session = _DB_DEP,
):
    item = _resolve_item(item_id, db)
    meta_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved_models", "meta.json")
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
    db: Session = _DB_DEP,
):
    item = _resolve_item(item_id, db)
    requested = [s.strip() for s in source.split(",") if s.strip()]

    cutoff = datetime.now(UTC) - timedelta(days=days)
    data: dict[str, list[SourcePriceOut]] = {}

    query = db.query(PriceHistory).filter(
        PriceHistory.item_id == item.id,
        PriceHistory.timestamp >= cutoff,
        ~PriceHistory.source.like("synthetic_demo"),
        ~PriceHistory.source.like("historical_fallback:%"),
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
