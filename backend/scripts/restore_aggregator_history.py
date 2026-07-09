#!/usr/bin/env python3
import csv, gzip, logging, re, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from sqlalchemy import text as sql
from database import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("restore_aggregator_history")

CSV_GIT_REF = "6bf989e:price-archive/snapshot-tier-history-through-2026-07-08.csv.gz"

INSERT_SQL = """
    INSERT INTO price_history (item_id, timestamp, price, volume, median_price, source)
    VALUES (:item_id, :timestamp, :price, :volume, :median_price, :source)
    ON CONFLICT (item_id, timestamp, source) DO NOTHING
"""

def normalize(name):
    n = name.strip()
    n = re.sub(r"^★\s*", "", n)
    n = n.replace("™", "").replace("®", "")
    n = re.sub(r"\s+", " ", n)
    return n.lower().strip()

logger.info("Reading CSV...")
result = subprocess.run(
    ["git", "show", CSV_GIT_REF],
    capture_output=True,
    cwd=str(Path(__file__).parent.parent.parent),
)
raw = gzip.decompress(result.stdout)
lines = [l.decode("utf-8") for l in raw.split(b"\n") if l.strip()]
reader = csv.DictReader(lines)

db = SessionLocal()
all_rows = db.execute(sql("SELECT id, name FROM items")).fetchall()
db_norm = {}
for id_, name in all_rows:
    db_norm[normalize(name)] = id_
logger.info(f"DB items: {len(all_rows)}, unique normalized: {len(db_norm)}")

rows_to_insert = []
matched = 0
skipped_no_match = 0
skipped_source = 0
unmatched_samples = set()

for row in reader:
    source = row.get("source", "")
    if source not in ("aggregator_sync",):
        skipped_source += 1
        continue
    norm = normalize(row["name"])
    item_id = db_norm.get(norm)
    if item_id is None:
        if len(unmatched_samples) < 20:
            unmatched_samples.add(row["name"])
        skipped_no_match += 1
        continue
    matched += 1
    rows_to_insert.append({
        "item_id": item_id,
        "timestamp": row["timestamp"],
        "price": float(row["price"]),
        "volume": int(row["volume"]) if row.get("volume") else None,
        "median_price": float(row["median_price"]) if row.get("median_price") else None,
        "source": source,
    })

logger.info(f"Matched: {matched}, skipped (no match): {skipped_no_match}, skipped (source): {skipped_source}")
if unmatched_samples:
    logger.info(f"Sample unmatched ({len(unmatched_samples)} shown):")
    for s in sorted(unmatched_samples):
        logger.info(f"  {s}")

if not rows_to_insert:
    logger.warning("No rows to import.")
    db.close()
    sys.exit(0)

batch_size = 5000
total = 0
for i in range(0, len(rows_to_insert), batch_size):
    batch = rows_to_insert[i:i + batch_size]
    db.execute(sql(INSERT_SQL), batch)
    db.commit()
    total += len(batch)
    logger.info(f"Inserted {total}/{len(rows_to_insert)}")

count = db.execute(sql("SELECT COUNT(*) FROM price_history WHERE source = 'aggregator_sync'")).scalar()
rows = db.execute(sql("SELECT MIN(timestamp), MAX(timestamp) FROM price_history WHERE source = 'aggregator_sync'")).fetchone()
logger.info(f"Total aggregator_sync rows: {count}")
logger.info(f"Date range: {rows[0]} to {rows[1]}")

daily = db.execute(sql("SELECT DATE(timestamp) as d, COUNT(*) FROM price_history WHERE source = 'aggregator_sync' GROUP BY d ORDER BY d")).fetchall()
logger.info("Per-day:")
for d, c in daily:
    logger.info(f"  {d}: {c}")

db.close()
logger.info("Done.")