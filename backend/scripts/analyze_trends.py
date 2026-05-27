#!/usr/bin/env python3
"""
Daily trend analysis for CS2 market items.
Computes moving averages, momentum, volatility, and opportunity scores.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, Item, PriceHistory
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trend_analyzer")

class TrendAnalyzer:
    def __init__(self, db_session):
        self.db = db_session
        self.analysis_date = datetime.utcnow().date()

    def get_item_price_history(self, item_id, days=90):
        """Fetch price history for an item."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        prices = self.db.query(PriceHistory).filter(
            PriceHistory.item_id == item_id,
            PriceHistory.timestamp >= cutoff_date
        ).order_by(PriceHistory.timestamp).all()

        return [(p.timestamp, p.price) for p in prices]

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

    def analyze_item(self, item_id):
        """Analyze a single item."""
        try:
            prices = self.get_item_price_history(item_id, days=90)

            if not prices or len(prices) < 7:
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
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            fourteen_days_ago = datetime.utcnow() - timedelta(days=14)

            recent_updates = self.db.query(PriceHistory).filter(
                PriceHistory.item_id == item_id,
                PriceHistory.timestamp >= seven_days_ago
            ).count()

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
        total_items = len(items)

        logger.info(f"Analyzing {total_items} items...")

        analyzed = 0
        skipped = 0
        results = []

        for (item_id,) in items:
            result = self.analyze_item(item_id)

            if result:
                results.append(result)
                analyzed += 1
            else:
                skipped += 1

            # Log progress every 1000 items
            if (analyzed + skipped) % 1000 == 0:
                logger.info(f"Progress: {analyzed + skipped}/{total_items}")

        # Bulk insert results
        if results:
            logger.info(f"Inserting {len(results)} analysis results...")

            insert_sql = """
                INSERT INTO daily_analysis
                (item_id, analysis_date, current_price, ma_7day, ma_30day, ma_90day,
                 momentum_7day, momentum_30day, volatility, trend_direction,
                 momentum_score, opportunity_score, trading_volume_trend, price_stability)
                VALUES (:item_id, :analysis_date, :current_price, :ma_7day, :ma_30day, :ma_90day,
                        :momentum_7day, :momentum_30day, :volatility, :trend_direction,
                        :momentum_score, :opportunity_score, :trading_volume_trend, :price_stability)
                ON CONFLICT (item_id, analysis_date) DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    ma_7day = EXCLUDED.ma_7day,
                    ma_30day = EXCLUDED.ma_30day,
                    ma_90day = EXCLUDED.ma_90day,
                    momentum_7day = EXCLUDED.momentum_7day,
                    momentum_30day = EXCLUDED.momentum_30day,
                    volatility = EXCLUDED.volatility,
                    trend_direction = EXCLUDED.trend_direction,
                    momentum_score = EXCLUDED.momentum_score,
                    opportunity_score = EXCLUDED.opportunity_score,
                    trading_volume_trend = EXCLUDED.trading_volume_trend,
                    price_stability = EXCLUDED.price_stability
            """

            for result in results:
                self.db.execute(text(insert_sql), result)

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
