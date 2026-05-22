"""
Migration script: SQLite (local) → PostgreSQL (Supabase)
Reads from cs2_market.db, writes to Supabase, with progress tracking.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from database import Base, Item, PriceHistory, CollectionRun, Event, TrendIndicator, User
from config import settings
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

def migrate():
    """Migrate data from SQLite to Supabase PostgreSQL"""

    # Source: SQLite (local)
    sqlite_url = "sqlite:///cs2_market.db"
    source_engine = create_engine(sqlite_url, echo=False)
    SourceSession = sessionmaker(bind=source_engine)
    source_session = SourceSession()

    # Target: Supabase PostgreSQL
    target_url = os.getenv("DATABASE_URL")
    if not target_url:
        logger.error("DATABASE_URL not found in environment")
        sys.exit(1)

    try:
        target_engine = create_engine(target_url, echo=False)
        TargetSession = sessionmaker(bind=target_engine)
        target_session = TargetSession()

        # Test connection
        logger.info("Testing Supabase connection...")
        with target_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.commit()
        logger.info("✓ Supabase connection successful")

        # Create tables
        logger.info("Creating tables in Supabase...")
        Base.metadata.create_all(target_engine)
        logger.info("✓ Tables created")

        # Migrate data
        tables = [
            (Item, "Items"),
            (PriceHistory, "Price History"),
            (CollectionRun, "Collection Runs"),
            (Event, "Events"),
            (TrendIndicator, "Trend Indicators"),
            (User, "Users"),
        ]

        for model_class, label in tables:
            try:
                count = source_session.query(model_class).count()
                if count > 0:
                    logger.info(f"Migrating {count} {label}...")

                    # Batch insert (more efficient for large datasets)
                    batch_size = 1000
                    rows = source_session.query(model_class).all()

                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i+batch_size]
                        for row in batch:
                            # Detach from source session and add to target
                            target_session.merge(row)
                        target_session.commit()
                        logger.info(f"  → {min(i+batch_size, len(rows))}/{len(rows)}")

                    logger.info(f"✓ {label} migrated")
                else:
                    logger.info(f"⊘ {label}: 0 records (skipped)")
            except Exception as e:
                logger.warning(f"⚠ {label}: {e}")
                target_session.rollback()

        target_session.close()
        logger.info("\n✓ Migration complete!")

    except Exception as e:
        logger.error(f"Connection failed: {e}")
        sys.exit(1)
    finally:
        source_session.close()

if __name__ == "__main__":
    migrate()
