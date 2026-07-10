#!/usr/bin/env python3
"""
Daily trend analysis for CS2 market items.
Computes moving averages, momentum, volatility, and opportunity scores.
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import sqlalchemy as sa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, Item, PriceHistory, DailyAnalysis, utcnow_naive
from sqlalchemy import func, case
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trend_analyzer")

DAILY_ANALYSIS_WRITE_COLUMNS = (
    "item_id",
    "analysis_date",
    "current_price",
    "ma_7day",
    "ma_30day",
    "ma_90day",
    "momentum_7day",
    "momentum_30day",
    "volatility",
    "trend_direction",
    "momentum_score",
    "opportunity_score",
    "trading_volume_trend",
    "price_stability",
    "created_at",
)


def _filter_daily_analysis_row(row):
    """Drop non-portable fields before writing daily analysis rows."""
    return {key: value for key, value in row.items() if key in DAILY_ANALYSIS_WRITE_COLUMNS}

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"


def _load_from_parquet(item_ids, days=90):
    """Load daily price history from Parquet archive via DuckDB."""
    import duckdb

    if not item_ids or not ARCHIVE_DIR.exists():
        return {}

    from database import Item, SessionLocal
    db = SessionLocal()
    try:
        slug_rows = db.query(Item.id, Item.item_id).filter(
            Item.id.in_(item_ids)
        ).all()
        slug_set = {row.item_id for row in slug_rows}
        slug_to_int = {row.item_id: row.id for row in slug_rows}
    finally:
        db.close()

    cutoff = (datetime.utcnow() - timedelta(days=days)).date()

    con = duckdb.connect()
    try:
        slug_list = list(slug_set)
        placeholders = ','.join('?' for _ in slug_list)
        rows = con.sql(f"""
            SELECT item_slug, day, mean_price AS price
            FROM read_parquet('{ARCHIVE_DIR}/*.parquet')
            WHERE item_slug IN ({placeholders})
              AND day >= ?
            ORDER BY item_slug, day
        """, params=[*slug_list, cutoff.isoformat()]).fetchall()

        raw = defaultdict(list)
        for slug, day, price in rows:
            raw[slug_to_int[slug]].append((day, price))
        return dict(raw)
    finally:
        con.close()


class TrendAnalyzer:
    MIN_REQUIRED_HISTORY_POINTS = 7

    def __init__(self, db_session):
        self.db = db_session
        self.now = utcnow_naive()
        self.analysis_date = self.now.date()

    def _daily_analysis_upsert(self, rows):
        """Upsert daily analysis rows in one database round trip."""
        if not rows:
            return

        bind = self.db.get_bind()
        dialect_name = bind.dialect.name if bind is not None else "sqlite"
        insert_stmt = sqlite_insert if dialect_name == "sqlite" else pg_insert
        table = DailyAnalysis.__table__
        target_table = table
        if bind is not None:
            try:
                target_table = sa.Table(table.name, sa.MetaData(), autoload_with=bind)
            except Exception:
                # Fall back to the ORM table if reflection is unavailable.
                target_table = table
        actual_columns = set(DAILY_ANALYSIS_WRITE_COLUMNS)

        filtered_rows = [_filter_daily_analysis_row(row) for row in rows]

        CHUNK_SIZE = 500
        for i in range(0, len(filtered_rows), CHUNK_SIZE):
            chunk = filtered_rows[i:i + CHUNK_SIZE]
            stmt = insert_stmt(target_table).values(chunk)
            excluded = stmt.excluded
            update_columns = {
                column.name: getattr(excluded, column.name)
                for column in target_table.columns
                if column.name in actual_columns
                and column.name not in {"id", "item_id", "analysis_date", "created_at"}
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["item_id", "analysis_date"],
                set_=update_columns,
            )
            self.db.execute(stmt)

    @staticmethod
    def _to_daily(rows):
        """Collapse raw (timestamp, price) rows to one (date, mean price) per day.

        Prices are collected several times a day, so windowing over raw rows
        makes "7-day" metrics actually span ~2 days. All MA/momentum math
        operates on this daily series instead.
        """
        by_day = defaultdict(list)
        for timestamp, price in rows:
            by_day[timestamp.date()].append(price)
        return [(day, sum(prices) / len(prices)) for day, prices in sorted(by_day.items())]

    def get_item_price_history(self, item_id, days=90):
        """Fetch daily price history for an item."""
        if days > 14 and ARCHIVE_DIR.exists():
            data = _load_from_parquet([item_id], days)
            if data and item_id in data:
                return data[item_id]

        cutoff_date = self.now - timedelta(days=days)

        rows = self.db.query(
            PriceHistory.timestamp,
            PriceHistory.price
        ).filter(
            PriceHistory.item_id == item_id,
            PriceHistory.timestamp >= cutoff_date,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        ).order_by(PriceHistory.timestamp).all()

        return self._to_daily(rows)

    def get_recent_price_history_bulk(self, item_ids, days=90):
        """Fetch recent daily price history for many items in one query."""
        if not item_ids:
            return {}

        # Use Parquet archive for longer windows
        if days > 14 and ARCHIVE_DIR.exists():
            parquet_data = _load_from_parquet(item_ids, days)
            if parquet_data:
                return parquet_data

        cutoff_date = self.now - timedelta(days=days)
        rows = self.db.query(
            PriceHistory.item_id,
            PriceHistory.timestamp,
            PriceHistory.price
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= cutoff_date,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        ).order_by(PriceHistory.item_id, PriceHistory.timestamp).all()

        raw_by_item = defaultdict(list)
        for item_id, timestamp, price in rows:
            raw_by_item[item_id].append((timestamp, price))

        return {item_id: self._to_daily(item_rows) for item_id, item_rows in raw_by_item.items()}

    def get_update_counts_bulk(self, item_ids, start_dt, end_dt=None):
        """Fetch per-item update counts for a date window in one query."""
        if not item_ids:
            return {}

        query = self.db.query(
            PriceHistory.item_id,
            func.count(PriceHistory.item_id)
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= start_dt,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        )

        if end_dt is not None:
            query = query.filter(PriceHistory.timestamp < end_dt)

        rows = query.group_by(PriceHistory.item_id).all()
        return {item_id: count for item_id, count in rows}

    def get_volume_window_counts_bulk(self, item_ids, mid_dt, start_dt):
        """Fetch recent (>= mid) and older (start..mid) update counts in one query."""
        if not item_ids:
            return {}, {}

        rows = self.db.query(
            PriceHistory.item_id,
            func.sum(case((PriceHistory.timestamp >= mid_dt, 1), else_=0)),
            func.sum(case((PriceHistory.timestamp < mid_dt, 1), else_=0)),
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= start_dt,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        ).group_by(PriceHistory.item_id).all()

        recent = {item_id: int(r or 0) for item_id, r, _ in rows}
        older = {item_id: int(o or 0) for item_id, _, o in rows}
        return recent, older

    def calculate_moving_average(self, prices, days):
        """Calculate moving average for last N days."""
        if len(prices) < days:
            return None

        recent_prices = [p[1] for p in prices[-days:]]
        return float(np.mean(recent_prices))

    def calculate_momentum(self, prices, days):
        """Calculate % change over N days."""
        if len(prices) < days:
            return None

        old_price = prices[-days][1]
        new_price = prices[-1][1]

        if old_price == 0:
            return None

        return float(((new_price - old_price) / old_price) * 100)

    def calculate_volatility(self, prices):
        """Calculate price volatility (standard deviation)."""
        if len(prices) < 2:
            return 0.0

        price_list = [p[1] for p in prices[-30:]]  # Last 30 days
        if len(price_list) < 2:
            return 0.0

        mean_price = np.mean(price_list)
        if mean_price == 0:
            return 0.0

        std_dev = float(np.std(price_list))
        volatility_pct = (std_dev / mean_price) * 100

        return min(volatility_pct, 100.0)  # Cap at 100%

    def determine_trend(self, ma_7, ma_30):
        """Determine trend direction."""
        if ma_7 is None or ma_30 is None:
            return 'flat'

        if ma_7 > ma_30 * 1.02:  # 2% threshold
            return 'up'
        elif ma_7 < ma_30 * 0.98:
            return 'down'
        else:
            return 'flat'

    def calculate_momentum_score(self, momentum_7, momentum_30):
        """Score momentum strength (-100 to +100)."""
        if momentum_7 is None or momentum_30 is None:
            return 0.0

        # Average of 7-day and 30-day momentum, clamped
        avg_momentum = (momentum_7 + momentum_30) / 2
        score = float(np.clip(avg_momentum, -100, 100))

        return score

    def calculate_opportunity_score(self, current_price, ma_30, momentum_score, volatility):
        """Score investment opportunity (-100 to +100)."""
        if current_price is None or ma_30 is None:
            return 0.0

        # How far from 30-day average
        price_deviation = ((current_price - ma_30) / ma_30) * 100

        # Undervalued (below average) + positive momentum = opportunity
        # Overheated (above average) + negative momentum = warning

        if price_deviation < -10:  # >10% below average
            opportunity = 50 + (momentum_score * 0.5)
        elif price_deviation > 15:  # >15% above average
            opportunity = -50 + (momentum_score * 0.3)
        else:
            opportunity = momentum_score * 0.7

        # Adjust for volatility (high volatility = higher risk)
        if volatility > 10:
            opportunity *= 0.8

        return float(np.clip(opportunity, -100, 100))

    def analyze_item(self, item_id, prices=None, recent_updates=None, older_updates=None):
        """Analyze a single item."""
        try:
            if prices is None:
                prices = self.get_item_price_history(item_id, days=90)

            if not prices or len(prices) < self.MIN_REQUIRED_HISTORY_POINTS:
                return None  # Not enough data

            current_price = prices[-1][1]
            ma_7 = self.calculate_moving_average(prices, 7)
            ma_30 = self.calculate_moving_average(prices, 30)
            ma_90 = self.calculate_moving_average(prices, 90)

            momentum_7 = self.calculate_momentum(prices, 7)
            momentum_30 = self.calculate_momentum(prices, 30)
            volatility = self.calculate_volatility(prices)

            trend = self.determine_trend(ma_7, ma_30)
            momentum_score = self.calculate_momentum_score(momentum_7, momentum_30)
            opportunity_score = self.calculate_opportunity_score(
                current_price, ma_30, momentum_score, volatility
            )

            # Simple volume trend (more price updates = more trading)
            if recent_updates is None or older_updates is None:
                seven_days_ago = self.now - timedelta(days=7)
                fourteen_days_ago = self.now - timedelta(days=14)

                if recent_updates is None:
                    recent_updates = self.db.query(PriceHistory).filter(
                        PriceHistory.item_id == item_id,
                        PriceHistory.timestamp >= seven_days_ago
                    ).count()

                if older_updates is None:
                    older_updates = self.db.query(PriceHistory).filter(
                        PriceHistory.item_id == item_id,
                        PriceHistory.timestamp >= fourteen_days_ago,
                        PriceHistory.timestamp < seven_days_ago
                    ).count()

            volume_trend = 0.0
            if older_updates > 0:
                volume_trend = float(((recent_updates - older_updates) / older_updates) * 100)

            price_stability = max(0, 100 - volatility)  # Inverse of volatility

            return {
                'item_id': item_id,
                'analysis_date': self.analysis_date,
                'current_price': current_price,
                'ma_7day': ma_7,
                'ma_30day': ma_30,
                'ma_90day': ma_90,
                'momentum_7day': momentum_7,
                'momentum_30day': momentum_30,
                'volatility': volatility,
                'trend_direction': trend,
                'momentum_score': momentum_score,
                'opportunity_score': opportunity_score,
                'trading_volume_trend': volume_trend,
                'price_stability': price_stability
            }

        except Exception as e:
            logger.warning(f"Error analyzing item {item_id}: {e}")
            return None

    def run_analysis(self):
        """Run analysis for all items."""
        logger.info("="*60)
        logger.info("TREND ANALYSIS")
        logger.info(f"Date: {self.analysis_date}")
        logger.info("="*60)

        # Get all items
        items = self.db.query(Item.id).all()
        item_ids = [item_id for (item_id,) in items]
        total_items = len(item_ids)

        logger.info(f"Analyzing {total_items} items...")

        analyzed = 0
        skipped = 0
        results = []

        # Only pull full history for items that already have enough recent data.
        ninety_day_cutoff = self.now - timedelta(days=90)
        ninety_day_counts = self.db.query(
            PriceHistory.item_id,
            func.count(PriceHistory.item_id)
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= ninety_day_cutoff,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        ).group_by(PriceHistory.item_id).all()
        ninety_day_counts = {item_id: count for item_id, count in ninety_day_counts}

        eligible_item_ids = [
            item_id for item_id in item_ids
            if ninety_day_counts.get(item_id, 0) >= self.MIN_REQUIRED_HISTORY_POINTS
        ]

        logger.info(
            "Eligible items with at least %s recent data points: %s",
            self.MIN_REQUIRED_HISTORY_POINTS,
            len(eligible_item_ids),
        )

        prices_by_item = self.get_recent_price_history_bulk(eligible_item_ids, days=90)

        seven_days_ago = self.now - timedelta(days=7)
        fourteen_days_ago = self.now - timedelta(days=14)
        recent_update_counts, older_update_counts = self.get_volume_window_counts_bulk(
            eligible_item_ids, seven_days_ago, fourteen_days_ago
        )

        skipped = total_items - len(eligible_item_ids)
        processed = 0

        for item_id in eligible_item_ids:
            result = self.analyze_item(
                item_id,
                prices=prices_by_item.get(item_id, []),
                recent_updates=recent_update_counts.get(item_id, 0),
                older_updates=older_update_counts.get(item_id, 0)
            )

            if result:
                results.append(result)
                analyzed += 1
            else:
                skipped += 1

            # Log progress every 1000 items
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"Progress: {processed}/{len(eligible_item_ids)} eligible items")

        # Bulk insert results
        if results:
            logger.info(f"Inserting {len(results)} analysis results...")

            self._daily_analysis_upsert(results)
            self.db.commit()

        logger.info(f"✅ Analysis complete: {analyzed} analyzed, {skipped} skipped")
        logger.info(f"Total records inserted/updated: {len(results)}")

        return {
            'status': 'success',
            'analyzed': analyzed,
            'skipped': skipped,
            'total': len(results)
        }

def main():
    db = SessionLocal()

    try:
        analyzer = TrendAnalyzer(db)
        result = analyzer.run_analysis()

        logger.info(f"\nRESULT: {result}")
        print(f"RESULT: {result}")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
        return 1

    finally:
        db.close()

if __name__ == "__main__":
    sys.exit(main())
