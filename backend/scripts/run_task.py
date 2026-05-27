#!/usr/bin/env python3
"""
Task runner for automated maintenance and collection.
Used by GitHub Actions to trigger specific pipeline tasks.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal
from collectors.pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("task_runner")

def run_task(task_name):
    db = SessionLocal()
    pipeline = DataPipeline(db_session=db)
    
    try:
        if task_name == "aggregate":
            logger.info("Task: Full Aggregator Scrape (All 17k items)")
            result = pipeline.run_full_aggregator_collection()
            print(f"RESULT: {result}")
            
        elif task_name == "priority":
            logger.info("Task: Priority Aggregator Scrape (Top 2000)")
            result = pipeline.run_priority_collection()
            print(f"RESULT: {result}")
            
        elif task_name == "daily":
            logger.info("Task: Full Steam Sweep (The Reddit Method - Price + Volume)")
            result = pipeline.run_daily_collection()
            print(f"RESULT: {result}")
            
        elif task_name == "prune":
            logger.info("Task: Database Pruning & Downsampling")
            result = pipeline.run_database_pruning()
            print(f"RESULT: {result}")
            
        elif task_name == "trends":
            logger.info("Task: Trend Analysis & Opportunity Detection")
            result = pipeline.run_feature_computation()
            result2 = pipeline.run_trend_analysis()
            print(f"RESULT: {result}, {result2}")
            
        else:
            logger.error(f"Unknown task: {task_name}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_task.py <task_name>")
        print("Tasks: aggregate, priority, prune, trends")
        sys.exit(1)
        
    run_task(sys.argv[1])
