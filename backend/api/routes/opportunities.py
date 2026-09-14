from database import Item, ItemForecast, get_db
from fastapi import APIRouter, Depends, Query
from models.item_parser import is_phantom_slug, is_phase_collapsed
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from api.cache import get_or_build
from api.schemas import OpportunityOut
from api.serving_policy import (
    anchor_clean_clause,
    meets_anchor_gate,
    meets_price_floor,
    price_floor_clause,
)

router = APIRouter(prefix="/opportunities", tags=["opportunities"])


def _build_opportunity(item: Item, forecast: ItemForecast, opp_type: str) -> OpportunityOut:
    current_price = forecast.current_price or 0.0
    predicted_return = (
        ((forecast.price_mid or current_price) - current_price) / current_price * 100 if current_price > 0 else 0
    )
    return OpportunityOut(
        item_id=item.id,
        item_name=item.name,
        current_price=current_price,
        opportunity_type=opp_type,
        opportunity_score=round(predicted_return, 2),
        reason=_reason_for_type(opp_type),
        current_trend=forecast.direction or "neutral",
        volatility=None,
    )


def _reason_for_type(opp_type: str) -> str:
    if opp_type == "undervalued":
        return "ML forecast predicts upward movement over the next 7 days."
    if opp_type == "overheated":
        return "ML forecast predicts downward movement over the next 7 days."
    return "ML forecast shows strong predicted price movement."


def _load_items(item_ids: list[int], db: Session) -> dict[int, Item]:
    """Items backing a set of forecasts. Every caller here skips a forecast
    whose item is missing from the map, which makes this the one place that has
    to drop the names outside the universe.

    Both rules matter here because these queries take each item's *latest*
    forecast with no date bound, so a row that stops being forecast sits on the
    ranked surfaces permanently.

    - Phase-collapsed names left the forecast universe on 2026-08-08, and they
      are $200-500 knives, which rank.
    - Phantom keys are a second copy of an item already listed — 3,149 of the
      8,691 forecast items as of 2026-08-08. Checked on `item_id`, not `name`:
      the phantom row carries the **real** market_hash_name in `name`, which is
      what makes it repairable and also what makes a name check miss it.
    """
    if not item_ids:
        return {}
    items = db.query(Item).filter(Item.id.in_(item_ids)).all()
    return {i.id: i for i in items if not is_phase_collapsed(i.name) and not is_phantom_slug(i.item_id)}


def _latest_forecasts(db: Session, horizon_days: int = 7):
    """Get the latest forecast per item for a given horizon."""
    subq = (
        db.query(
            ItemForecast.item_id,
            ItemForecast.forecast_date,
        )
        .filter(ItemForecast.horizon_days == horizon_days)
        .distinct(ItemForecast.item_id)
        .order_by(
            ItemForecast.item_id,
            desc(ItemForecast.forecast_date),
        )
        .subquery()
    )
    return (
        db.query(ItemForecast)
        .join(subq, (ItemForecast.item_id == subq.c.item_id) & (ItemForecast.forecast_date == subq.c.forecast_date))
        .filter(
            ItemForecast.horizon_days == horizon_days,
            price_floor_clause(ItemForecast.current_price),
            # Ranking is the surface the clean-anchor evidence covers, so the
            # gate belongs here as well as in select_opportunities -- filtering
            # only in Python would pull every forecast out of Postgres and
            # discard two thirds of them in the API process.
            anchor_clean_clause(ItemForecast.anchor_clean),
        )
        .all()
    )


@router.get("/", response_model=list[OpportunityOut])
def get_opportunities(
    type: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return get_or_build(
        f"opportunities:{type or ''}:{limit}",
        300,
        lambda: _build_opportunities(db, type, limit),
    )


def _build_opportunities(db: Session, type: str | None, limit: int):
    forecasts = _latest_forecasts(db)
    item_ids = [f.item_id for f in forecasts if f.direction is not None]
    items_map = _load_items(item_ids, db)
    return select_opportunities(forecasts, items_map, type, limit)


def opportunity_type_for(direction: str | None) -> str:
    """Direction alone decides the label.

    This previously required ``confidence == "high"`` for the directional
    labels, which meant the two headline buckets were populated exclusively by
    the anti-predictive subset. Confidence no longer participates.
    """
    if direction == "up":
        return "undervalued"
    if direction == "down":
        return "overheated"
    return "momentum"


def select_opportunities(forecasts, items_map, type_filter, limit):
    """Pure selection over already-fetched rows, ranked by |predicted return|.

    The clean-anchor gate is applied HERE, before the sort and the truncation.
    Filtering after `[:limit]` would make the list shorter the more deviating
    items happened to rank highest -- and they rank high by construction, since
    a deviating anchor is a quote far from its own median and the predicted
    return is measured against that median.
    """
    results = []
    for f in forecasts:
        if f.direction is None:
            continue
        if not meets_price_floor(f.current_price):
            continue
        if not meets_anchor_gate(getattr(f, "anchor_clean", None)):
            continue
        item = items_map.get(f.item_id)
        if not item:
            continue
        opp_type = opportunity_type_for(f.direction)
        if type_filter and opp_type != type_filter:
            continue
        results.append(_build_opportunity(item, f, opp_type))

    results.sort(key=lambda x: abs(x.opportunity_score), reverse=True)
    return results[:limit]


@router.get("/undervalued", response_model=list[OpportunityOut])
def get_undervalued(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    subq = (
        db.query(
            ItemForecast.item_id,
            ItemForecast.forecast_date,
        )
        .filter(
            ItemForecast.horizon_days == 7,
            ItemForecast.direction == "up",
        )
        .distinct(ItemForecast.item_id)
        .order_by(ItemForecast.item_id, desc(ItemForecast.forecast_date))
        .subquery()
    )
    forecasts = (
        db.query(ItemForecast)
        .join(subq, (ItemForecast.item_id == subq.c.item_id) & (ItemForecast.forecast_date == subq.c.forecast_date))
        .filter(
            ItemForecast.horizon_days == 7,
            ItemForecast.direction == "up",
            ItemForecast.current_price.isnot(None),
            price_floor_clause(ItemForecast.current_price),
            ItemForecast.price_mid.isnot(None),
        )
        .order_by(desc((ItemForecast.price_mid - ItemForecast.current_price) / ItemForecast.current_price * 100))
        .limit(limit)
        .all()
    )
    items_map = _load_items([f.item_id for f in forecasts], db)
    results = []
    for f in forecasts:
        item = items_map.get(f.item_id)
        if not item:
            continue
        results.append(_build_opportunity(item, f, "undervalued"))
    return results


@router.get("/overheated", response_model=list[OpportunityOut])
def get_overheated(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    subq = (
        db.query(
            ItemForecast.item_id,
            ItemForecast.forecast_date,
        )
        .filter(
            ItemForecast.horizon_days == 7,
            ItemForecast.direction == "down",
        )
        .distinct(ItemForecast.item_id)
        .order_by(ItemForecast.item_id, desc(ItemForecast.forecast_date))
        .subquery()
    )
    forecasts = (
        db.query(ItemForecast)
        .join(subq, (ItemForecast.item_id == subq.c.item_id) & (ItemForecast.forecast_date == subq.c.forecast_date))
        .filter(
            ItemForecast.horizon_days == 7,
            ItemForecast.direction == "down",
            ItemForecast.current_price.isnot(None),
            price_floor_clause(ItemForecast.current_price),
            ItemForecast.price_mid.isnot(None),
        )
        .order_by(desc((ItemForecast.current_price - ItemForecast.price_mid) / ItemForecast.current_price * 100))
        .limit(limit)
        .all()
    )
    items_map = _load_items([f.item_id for f in forecasts], db)
    results = []
    for f in forecasts:
        item = items_map.get(f.item_id)
        if not item:
            continue
        results.append(_build_opportunity(item, f, "overheated"))
    return results


@router.get("/momentum", response_model=list[OpportunityOut])
def get_momentum(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    subq = (
        db.query(
            ItemForecast.item_id,
            ItemForecast.forecast_date,
        )
        .filter(ItemForecast.horizon_days == 7)
        .distinct(ItemForecast.item_id)
        .order_by(ItemForecast.item_id, desc(ItemForecast.forecast_date))
        .subquery()
    )
    forecasts = (
        db.query(ItemForecast)
        .join(subq, (ItemForecast.item_id == subq.c.item_id) & (ItemForecast.forecast_date == subq.c.forecast_date))
        .filter(
            ItemForecast.horizon_days == 7,
            ItemForecast.current_price.isnot(None),
            price_floor_clause(ItemForecast.current_price),
            ItemForecast.price_mid.isnot(None),
        )
        .order_by(
            desc(func.abs((ItemForecast.price_mid - ItemForecast.current_price) / ItemForecast.current_price * 100))
        )
        .limit(limit)
        .all()
    )
    items_map = _load_items([f.item_id for f in forecasts], db)
    results = []
    for f in forecasts:
        item = items_map.get(f.item_id)
        if not item:
            continue
        results.append(_build_opportunity(item, f, "momentum"))
    return results
