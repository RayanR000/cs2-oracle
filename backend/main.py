"""
CS2 Market Intelligence Platform - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from routers import items, opportunities, events, auth, portfolio
from database import init_db
from config import settings
from collectors.comprehensive_loader import load_all_cs2_data
from collectors.real_data_collector import start_real_data_collection, stop_real_data_collection
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CS2 Market Intelligence API",
    description="Backend API for CS2 market tracking and analysis",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Session Middleware
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize database and load data on startup"""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")

    demo_bootstrap = settings.demo_bootstrap_enabled()
    logger.info(
        "Bootstrapping catalog data in %s mode (%s history)",
        settings.environment,
        "synthetic demo" if demo_bootstrap else "no synthetic"
    )
    try:
        stats = load_all_cs2_data(generate_history=demo_bootstrap)
        logger.info(f"Data load complete: {stats}")
        logger.info(f"  Items: {stats.get('items_added', 0)} added, {stats.get('items_skipped', 0)} skipped")
        logger.info(f"  Price records: {stats.get('price_records_added', 0)}")
        logger.info(f"  Events: {stats.get('events_added', 0)}")
    except Exception as e:
        # Allow app to start even if initial data load fails.
        # Live collection can still populate data over time.
        logger.error(f"Error loading data: {e}", exc_info=True)

    logger.info("Starting real-time market data collection...")
    try:
        start_real_data_collection()
        logger.info("Real-time data collection started")
    except Exception as e:
        logger.error(f"Error starting data collection pipeline: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down...")
    stop_real_data_collection()
    logger.info("Real-time data collection stopped")

# Include routers
app.include_router(items.router)
app.include_router(opportunities.router)
app.include_router(events.router)
app.include_router(auth.router)
app.include_router(portfolio.router)

# Include admin router (for data collection management)
from routers import admin
app.include_router(admin.router)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "cs2-market-api"}

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "CS2 Market Intelligence API",
        "version": "0.1.0",
        "docs": "/api/docs",
        "data_source": "Real-time Steam API",
        "bootstrap_mode": "demo" if settings.demo_bootstrap_enabled() else "production",
        "synthetic_history": settings.demo_bootstrap_enabled()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
