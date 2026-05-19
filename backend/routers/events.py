"""
Event endpoints
"""

from fastapi import APIRouter, Query, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from database import get_db
from repositories import EventRepository

router = APIRouter(prefix="/events", tags=["events"])

@router.get("/", response_model=dict)
async def list_events(
    type: str = Query(None, description="Filter by type: major, update, case_drop, operation"),
    skip: int = Query(0, ge=0, description="Number of events to skip (pagination)"),
    limit: int = Query(50, ge=1, le=100, description="Number of events to return"),
    db: Session = Depends(get_db)
):
    """List market events with pagination"""
    if type:
        events = EventRepository.get_events_by_type(db, type, limit)
        events = events[skip:skip + limit]
    else:
        events = EventRepository.get_all_events(db, skip, limit)
    
    return {
        "events": events,
        "total": len(events),
        "skip": skip,
        "limit": limit,
        "has_more": len(events) >= limit
    }

@router.get("/timeline")
async def get_timeline(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get events in chronological order (timeline view)"""
    events = EventRepository.get_all_events(db, skip, limit)
    return {
        "events": events,
        "total": len(events)
    }

@router.get("/recent")
async def get_recent_events(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get most recent market events"""
    events = EventRepository.get_recent_events(db, days, limit)
    return {
        "events": events,
        "total": len(events)
    }
