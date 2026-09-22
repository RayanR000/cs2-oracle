import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from database import Item, PriceHistory, backfilled_item_clause, get_db
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import case, func, or_
from sqlalchemy.orm import Session

from api.cache import get_or_build
from api.schemas import GroupedMarketItemOut, QualityVariantOut, parse_item_name

router = APIRouter(prefix="/market", tags=["market"])

_DB_DEP = Depends(get_db)


class MarketItemOut(BaseModel):
    id: int
    item_id: str
    name: str
    type: str
    icon_url: str | None = None
    current_price: float | None = None
    price_change_24h: float | None = None
    volume_24h: int | None = None

    class Config:
        from_attributes = True


def _normalize(s: str) -> str:
    """Strip non-alphanumeric characters and lowercase for fuzzy matching."""
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


@router.get("/summary", response_model=list[GroupedMarketItemOut])
def market_summary(
    type: str | None = Query(None),
    q: str | None = Query(None, description="Search query for item name"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = _DB_DEP,
):
    # Building the grouped summary scans every item plus its recent prices
    # (~3.5s); the result only changes when the daily pipelines run, so it
    # is cached whole and paginated from memory.
    groups = get_or_build(
        f"market_summary:{type or ''}:{q or ''}",
        ttl_seconds=120 if q else 600,
        builder=lambda: _build_market_summary(db, type, q),
    )
    return groups[skip : skip + limit]


def _build_market_summary(db: Session, type: str | None, q: str | None):
    query = db.query(Item).filter(backfilled_item_clause())
    if type:
        query = query.filter(Item.type == type)
    if q:
        normalized = _normalize(q)
        name_norm = func.regexp_replace(func.lower(Item.name), "[^a-zA-Z0-9]", "", "g")

        direct = Item.name.ilike(f"%{q}%")
        norm_match = name_norm.ilike(f"%{normalized}%")

        conditions = [direct, norm_match]
        if len(q) >= 3:
            fuzzy = func.similarity(Item.name, q) > 0.12
            conditions.append(fuzzy)

        query = query.filter(or_(*conditions))
        query = query.order_by(
            case((direct, 0), else_=1),
            case((norm_match, 0), else_=1),
            func.similarity(Item.name, q).desc(),
            Item.name,
        )
    else:
        query = query.order_by(Item.name)

    _MAX_ITEMS = 2000
    items = query.limit(_MAX_ITEMS).all()
    if not items:
        return []

    item_ids = [i.id for i in items]

    cutoff = datetime.now(UTC) - timedelta(days=2)
    price_query = db.query(PriceHistory).filter(PriceHistory.timestamp >= cutoff)
    price_query = price_query.filter(PriceHistory.item_id.in_(item_ids))
    price_rows = price_query.order_by(PriceHistory.item_id, PriceHistory.timestamp).all()
    prices_by_item: dict[int, list] = {}
    for pr in price_rows:
        prices_by_item.setdefault(pr.item_id, []).append(pr)

    per_item = {}
    for item in items:
        ph_list = prices_by_item.get(item.id, [])

        current_price = ph_list[-1].price if ph_list else None
        price_change_24h = None
        volume_24h = None

        if len(ph_list) >= 2:
            first = ph_list[0]
            last = ph_list[-1]
            if first.price > 0:
                price_change_24h = round(((last.price - first.price) / first.price) * 100, 2)
            volume_24h = sum((p.volume or 0) for p in ph_list)

        base_name, quality = parse_item_name(item.name)

        per_item[item.item_id] = {
            "base_name": base_name,
            "quality": quality,
            "item": item,
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "volume_24h": volume_24h,
        }

    groups: dict[str, list] = defaultdict(list)
    for data in per_item.values():
        groups[data["base_name"]].append(data)

    result = []
    for base_name, variants in groups.items():
        deduped: dict[str, dict] = {}
        for v in variants:
            q = v["quality"] or "Standard"
            if q not in deduped or (v["current_price"] is not None and deduped[q].get("current_price") is None):
                deduped[q] = v
        variants = list(deduped.values())

        prices = [v["current_price"] for v in variants if v["current_price"] is not None]
        volumes = [v["volume_24h"] for v in variants if v["volume_24h"] is not None]
        changes = [v["price_change_24h"] for v in variants if v["price_change_24h"] is not None]

        first_variant = variants[0]
        item = first_variant["item"]

        price_avg = round(sum(prices) / len(prices), 2) if prices else None
        price_min = round(min(prices), 2) if prices else None
        price_max = round(max(prices), 2) if prices else None
        avg_change = round(sum(changes) / len(changes), 2) if changes else None
        total_volume = sum(volumes) if volumes else None

        quality_list = []
        for v in variants:
            quality_list.append(
                QualityVariantOut(
                    item_id=v["item"].item_id,
                    name=v["item"].name,
                    quality=v["quality"] or "Standard",
                    current_price=v["current_price"],
                    price_change_24h=v["price_change_24h"],
                    volume_24h=v["volume_24h"],
                )
            )

        quality_list.sort(key=lambda x: x.quality)

        result.append(
            GroupedMarketItemOut(
                base_name=base_name,
                type=item.type,
                icon_url=item.icon_url,
                price_avg=price_avg,
                price_min=price_min,
                price_max=price_max,
                price_change_24h=avg_change,
                volume_24h=total_volume,
                quality_count=len(variants),
                qualities=quality_list,
            )
        )

    return result
