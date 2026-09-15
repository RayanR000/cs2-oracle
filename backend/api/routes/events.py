import logging

from database import Event, get_db
from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from api.schemas import EventOut

router = APIRouter(prefix="/events", tags=["events"])

logger = logging.getLogger(__name__)


def _read_events_parquet(
    type_filter: str | None = None, skip: int = 0, limit: int = 50, recent_only: bool = False, recent_limit: int = 20
):
    from db.parquet import ParquetQuery

    with ParquetQuery("events") as q:
        where = []
        if type_filter:
            where.append(f"type = '{type_filter.replace(chr(39), chr(39) + chr(39))}'")
        if recent_only:
            where.append("1=1")
        where_clause = " AND ".join(where) if where else "1=1"
        sql = f"SELECT * FROM events WHERE {where_clause} ORDER BY timestamp DESC"
        if recent_only:
            sql += f" LIMIT {recent_limit}"
        else:
            sql += f" OFFSET {skip} LIMIT {limit}"
        df = q.query(sql)
        if df.empty:
            return []
        return [
            EventOut(
                id=int(r.id),
                type=str(r.type),
                timestamp=r.timestamp,
                description=str(r.description),
                created_at=r.created_at,
            )
            for r in df.itertuples()
        ]


@router.get("/", response_model=list[EventOut])
def list_events(
    type: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    try:
        return _read_events_parquet(type_filter=type, skip=skip, limit=limit)
    except Exception as e:
        logger.debug("parquet events read failed, falling back to DB: %s", e)
        q = db.query(Event)
        if type:
            q = q.filter(Event.type == type)
        return q.order_by(desc(Event.timestamp)).offset(skip).limit(limit).all()


@router.get("/recent", response_model=list[EventOut])
def recent_events(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    try:
        return _read_events_parquet(recent_only=True, recent_limit=limit)
    except Exception as e:
        logger.debug("parquet recent-events read failed, falling back to DB: %s", e)
        return db.query(Event).order_by(desc(Event.timestamp)).limit(limit).all()
