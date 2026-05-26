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

    # Production/Optimized startup: Skip heavy bootstrapping if not explicitly requested
    if settings.environment.lower() == "production" or not settings.debug:
        logger.info("Skipping heavy bootstrapping (Production/Non-Debug mode)")
        return

    demo_bootstrap = settings.demo_bootstrap_enabled()
    logger.info(
        "Bootstrapping catalog data in %s mode (%s history)",
        settings.environment,
        "synthetic demo" if demo_bootstrap else "no synthetic"
    )
    try:
        # Only run this if the database is truly empty
        from database import SessionLocal, Item
        db = SessionLocal()
        item_count = db.query(Item).count()
        db.close()
        
        if item_count == 0:
            logger.info("Database empty, running initial catalog load...")
            stats = load_all_cs2_data(generate_history=demo_bootstrap)
            logger.info(f"Initial load complete: {stats}")
        else:
            logger.info(f"Database already has {item_count} items. Skipping initial load.")
            
    except Exception as e:
        logger.error(f"Error checking/loading initial data: {e}")

    # Real-time collection is now recommended to be run as a separate process: 
    # python scripts/background_collect.py
    # We will only start it here if specifically configured, to keep API startup fast.
    if settings.debug:
        logger.info("Running in debug mode: Real-time collection thread will NOT start automatically.")
        logger.info("Run 'python scripts/background_collect.py' in a separate terminal for live updates.")

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
