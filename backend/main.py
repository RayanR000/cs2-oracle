"""
FastAPI server for CS2 Market Intelligence Platform
"""

import threading
from contextlib import asynccontextmanager

from api.routes import accuracy, auth, events, items, market, monitoring, opportunities
from config import settings
from database import init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.check_secret_key()
    init_db()
    threading.Thread(target=_warm_cache, daemon=True).start()
    yield


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(items.router)
app.include_router(opportunities.router)
app.include_router(events.router)
app.include_router(auth.router)
app.include_router(market.router)
app.include_router(accuracy.router)
app.include_router(monitoring.router)


@app.middleware("http")
async def cache_control_middleware(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if request.method != "GET" or response.status_code != 200:
        return response
    # Data changes once daily (collection 23:00 UTC, analysis ~03:00 UTC);
    # let browsers reuse responses across navigation instead of refetching.
    if path.startswith(("/items/", "/market/", "/events/", "/opportunities/")):
        response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=600"
    elif path == "/health":
        response.headers["Cache-Control"] = "public, max-age=5"
    return response


def _warm_cache():
    """Pre-build the expensive market summary so the first page load is fast."""
    from api.cache import get_or_build
    from api.routes.market import _build_market_summary
    from database import SessionLocal

    db = SessionLocal()
    try:
        get_or_build("market_summary::", 600, lambda: _build_market_summary(db, None, None))
    except Exception:  # warming is best-effort; requests build on miss anyway
        pass
    finally:
        db.close()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": settings.api_version,
        "environment": settings.environment,
    }
