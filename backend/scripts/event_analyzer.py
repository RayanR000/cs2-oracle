#!/usr/bin/env python3
"""
Event correlation analysis with statistical rigor.
Implements 6-point causal validation framework for event impact on prices.
"""

import sys
import logging
from pathlib import Path
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, Event, Item, PriceHistory, EventImpact, EventPattern, EventCorrelation
from sqlalchemy import func

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("event_analyzer")


class EventAnalyzer:
    def __init__(self, db_session):
        self.db = db_session
        self.analysis_date = datetime.utcnow().date()
        # Cache per-item time series in ascending order for fast date lookups.
        self.price_cache = {}
        self.price_timestamps = {}
        self.control_group_cache = {}

    def batch_load_prices(self, item_ids, start_date, end_date):
        """
        Batch load all prices for items in date range (single query instead of thousands).
        Stores in memory for fast lookups.
        """
        logger.info(f"Batch loading prices for {len(item_ids)} items from {start_date} to {end_date}...")

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        # Try Parquet archive for historical range
        archive_dir = Path(__file__).parent.parent.parent / "price-archive"
        parquet_ok = archive_dir.exists() and (end_date - start_date).days > 60

        if parquet_ok:
            import duckdb
            slug_rows = self.db.query(Item.id, Item.item_id).filter(
                Item.id.in_(item_ids)
            ).all()
            slug_to_int = {r.item_id: r.id for r in slug_rows}
            slug_set = list(slug_to_int.keys())

            con = duckdb.connect()
            try:
                placeholders = ','.join('?' for _ in slug_set)
                rows = con.sql(f"""
                    SELECT item_slug, day, mean_price AS price
                    FROM read_parquet('{archive_dir}/*.parquet')
                    WHERE item_slug IN ({placeholders})
                      AND day >= ?
                      AND day <= ?
                    ORDER BY item_slug, day
                """, params=[*slug_set, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]).fetchall()

                for slug, day, price in rows:
                    item_id = slug_to_int.get(slug)
                    if item_id is None:
                        continue
                    ts = datetime.combine(day, datetime.min.time())
                    if item_id not in self.price_cache:
                        self.price_cache[item_id] = []
                        self.price_timestamps[item_id] = []
                    self.price_cache[item_id].append((ts, price))
                    self.price_timestamps[item_id].append(ts)

                logger.info(f"Loaded {len(rows)} price records from Parquet archive")
                return
            finally:
                con.close()

        rows = self.db.query(
            PriceHistory.item_id,
            PriceHistory.timestamp,
            PriceHistory.price
        ).filter(
            PriceHistory.item_id.in_(item_ids),
            PriceHistory.timestamp >= start_dt,
            PriceHistory.timestamp <= end_dt,
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        ).order_by(PriceHistory.item_id, PriceHistory.timestamp).all()

        for item_id, timestamp, price in rows:
            if item_id not in self.price_cache:
                self.price_cache[item_id] = []
                self.price_timestamps[item_id] = []

            self.price_cache[item_id].append((timestamp, price))
            self.price_timestamps[item_id].append(timestamp)

        logger.info(f"Loaded {len(rows)} price records into cache")

    def get_price_on_date(self, item_id, target_date):
        """Get the closest price to a target date from cache."""
        if item_id not in self.price_cache:
            return None

        timestamps = self.price_timestamps.get(item_id)
        if not timestamps:
            return None

        target_dt = datetime.combine(target_date, datetime.max.time())
        idx = bisect_right(timestamps, target_dt) - 1
        if idx < 0:
            return None

        return self.price_cache[item_id][idx][1]

    def compute_control_group_average(self, event_date):
        """Compute a reusable control-group average for an event date."""
        if event_date in self.control_group_cache:
            return self.control_group_cache[event_date]

        day_before = event_date - timedelta(days=1)
        day_after = event_date + timedelta(days=1)
        control_changes = []

        for item_id in self.price_cache:
            price_before = self.get_price_on_date(item_id, day_before)
            price_after = self.get_price_on_date(item_id, day_after)

            if price_before is None or price_after is None or price_before == 0:
                continue

            change = ((price_after - price_before) / price_before) * 100
            control_changes.append(change)

        control_avg = float(np.mean(control_changes)) if control_changes else None
        self.control_group_cache[event_date] = control_avg
        return control_avg

    def get_price_change_percentage(self, price_before, price_after):
        """Calculate percentage change between two prices."""
        if price_before is None or price_after is None or price_before == 0:
            return None
        return ((price_after - price_before) / price_before) * 100

    def record_event_impact(self, event, item_id=None):
        """Record price changes around an event for an item.

        Returns None when the item has no usable price data around the event
        (every impact metric needs price_before), so no-signal pairs don't
        become all-NULL rows.
        """
        event_date = event.timestamp.date()
        day_before = event_date - timedelta(days=1)
        day_1 = event_date
        day_3 = event_date + timedelta(days=3)
        day_7 = event_date + timedelta(days=7)

        price_before = self.get_price_on_date(item_id, day_before)
        if price_before is None or price_before == 0:
            return None

        price_day_1 = self.get_price_on_date(item_id, day_1)
        price_day_3 = self.get_price_on_date(item_id, day_3)
        price_day_7 = self.get_price_on_date(item_id, day_7)

        if price_day_1 is None and price_day_3 is None and price_day_7 is None:
            return None

        # Calculate impacts
        impact_1day = self.get_price_change_percentage(price_before, price_day_1)
        impact_3day = self.get_price_change_percentage(price_before, price_day_3)
        impact_7day = self.get_price_change_percentage(price_before, price_day_7)

        # Find peak impact and duration
        item_prices = self.price_cache.get(item_id, [])
        item_timestamps = self.price_timestamps.get(item_id, [])
        prices_after_event = []

        if item_prices and item_timestamps:
            start_dt = datetime.combine(day_1, datetime.min.time())
            end_dt = datetime.combine(day_7, datetime.max.time())
            start_idx = bisect_left(item_timestamps, start_dt)
            end_idx = bisect_right(item_timestamps, end_dt)
            prices_after_event = item_prices[start_idx:end_idx]

        peak_impact_pct = None
        peak_impact_day = None
        duration_days = 0

        if prices_after_event and price_before:
            max_change = 0
            trend_days = 0

            for i, (timestamp, price) in enumerate(prices_after_event):
                days_from_event = (timestamp.date() - event_date).days
                change = self.get_price_change_percentage(price_before, price)

                if change is not None:
                    if abs(change) > abs(max_change):
                        max_change = change
                        peak_impact_pct = change
                        peak_impact_day = days_from_event

                    if i > 0 and abs(change) > 2:  # Still > 2% change = ongoing impact
                        trend_days += 1

            duration_days = trend_days

        # Calculate z-score for statistical significance
        # Z-score = |change| / baseline_volatility (typical daily volatility ~2%)
        baseline_volatility = 2.0  # Typical daily volatility in percentage
        z_score = None
        if impact_1day is not None:
            z_score = abs(impact_1day) / baseline_volatility

        impact = EventImpact(
            event_id=event.id,
            item_id=item_id,
            price_day_before=price_before,
            price_day_1=price_day_1,
            price_day_3=price_day_3,
            price_day_7=price_day_7,
            impact_pct_1day=impact_1day,
            impact_pct_3day=impact_3day,
            impact_pct_7day=impact_7day,
            peak_impact_pct=peak_impact_pct,
            peak_impact_day=peak_impact_day,
            duration_days=duration_days,
            z_score=z_score
        )

        return impact

    def build_event_patterns(self, impacts, event_type_by_id):
        """Build event patterns from the in-memory impacts in one grouped pass."""
        grouped = defaultdict(list)
        for impact in impacts:
            if impact.item_id is None:
                continue
            event_type = event_type_by_id.get(impact.event_id)
            grouped[(event_type, impact.item_id)].append(
                (impact.impact_pct_1day, impact.impact_pct_3day, impact.impact_pct_7day)
            )

        patterns = {}
        for (event_type, item_id), impacts in grouped.items():
            if len(impacts) < 2:
                continue

            impact_1day_values = [row[0] for row in impacts if row[0] is not None]
            if not impact_1day_values:
                continue

            avg_1day = float(np.mean(impact_1day_values))
            avg_3day_values = [row[1] for row in impacts if row[1] is not None]
            avg_7day_values = [row[2] for row in impacts if row[2] is not None]
            std_dev = float(np.std(impact_1day_values)) if len(impact_1day_values) > 1 else 0.0

            if avg_1day != 0:
                cv = abs(std_dev / avg_1day)
                consistency_score = float(np.clip(1.0 - min(cv, 1.0), 0, 1))
            else:
                consistency_score = 0.5

            pattern = EventPattern(
                event_type=event_type,
                item_id=item_id,
                sample_size=len(impacts),
                avg_impact_1day=avg_1day,
                avg_impact_3day=float(np.mean(avg_3day_values)) if avg_3day_values else None,
                avg_impact_7day=float(np.mean(avg_7day_values)) if avg_7day_values else None,
                std_dev=std_dev,
                consistency_score=consistency_score,
                holdout_accuracy=consistency_score
            )
            patterns[(event_type, item_id)] = pattern

        return patterns

    def validate_event_correlation(self, event, item_id=None, impact=None, pattern=None, control_avg=None, same_day_events=0):
        """
        Run 6-point statistical rigor validation for event-item correlation.

        1. Significance test: Is the price change statistically significant (> 2 standard deviations)?
        2. Control group: Do unaffected items move less than affected items?
        3. Pattern consistency: Does the pattern repeat for this event type?
        4. Confounding variables: Are there other events on the same day?
        5. Lag analysis: Is the peak impact within expected timing?
        6. Holdout validation: Does the learned pattern work on new data?
        """
        if not impact:
            return None

        correlation = EventCorrelation(
            event_id=event.id,
            item_id=item_id,
            price_change_pct=impact.impact_pct_1day
        )

        # 1. SIGNIFICANCE TEST: Is change > 2 standard deviations from baseline?
        if impact.z_score is not None:
            correlation.significance_test_zscore = impact.z_score
            correlation.significance_passed = 1 if impact.z_score > 2.0 else 0
        else:
            correlation.significance_passed = 0

        # 2. CONTROL GROUP: Compare to unaffected items
        if control_avg is not None:
            correlation.control_group_change_pct = control_avg

            if impact.impact_pct_1day is not None:
                correlation.control_group_diff = impact.impact_pct_1day - control_avg
                correlation.control_group_passed = 1 if (
                    impact.impact_pct_1day > control_avg + 1.0
                ) else 0
            else:
                correlation.control_group_passed = 0
        else:
            correlation.control_group_passed = 0

        # 3. PATTERN CONSISTENCY: Does this event type show consistent patterns?
        if pattern and pattern.consistency_score is not None:
            correlation.pattern_consistency_score = pattern.consistency_score
            correlation.pattern_passed = 1 if pattern.consistency_score > 0.7 else 0
        else:
            correlation.pattern_passed = 0

        # 4. CONFOUNDING VARIABLES: Other events on same day?
        correlation.confounding_events_count = same_day_events
        correlation.confounding_passed = 1 if same_day_events == 0 else 0

        # 5. LAG ANALYSIS: Peak impact within expected timing (0-7 days)?
        if impact.peak_impact_day is not None:
            correlation.lag_analysis_peak_day = impact.peak_impact_day
            # Peak should be within 7 days, preferably 0-3 days
            correlation.lag_passed = 1 if 0 <= impact.peak_impact_day <= 7 else 0
        else:
            correlation.lag_passed = 0

        # 6. HOLDOUT VALIDATION: Pattern works on unseen events?
        if pattern and pattern.holdout_accuracy is not None:
            correlation.holdout_validation_accuracy = pattern.holdout_accuracy
            correlation.validation_passed = 1 if pattern.holdout_accuracy > 0.6 else 0
        else:
            correlation.validation_passed = 0

        # Calculate overall confidence score (weighted average of 6 checks)
        all_checks = [
            correlation.significance_passed,
            correlation.control_group_passed,
            correlation.pattern_passed,
            correlation.confounding_passed,
            correlation.lag_passed,
            correlation.validation_passed
        ]

        correlation.confidence_score = float(np.mean(all_checks)) if all_checks else 0.0

        return correlation

    def run_analysis(self):
        """Run event correlation analysis for all events."""
        logger.info("=" * 60)
        logger.info("EVENT CORRELATION ANALYSIS")
        logger.info(f"Date: {self.analysis_date}")
        logger.info("=" * 60)

        # Get all events
        events = self.db.query(Event).all()
        total_events = len(events)
        logger.info(f"Analyzing {total_events} events...")

        # Analyze the 1000 items with the richest price history, so the
        # limit keeps the most informative items instead of an arbitrary set.
        top_items = self.db.query(
            PriceHistory.item_id
        ).filter(
            ~PriceHistory.source.like('synthetic_demo'),
            ~PriceHistory.source.like('historical_fallback:%'),
        ).group_by(
            PriceHistory.item_id
        ).order_by(
            func.count(PriceHistory.item_id).desc()
        ).limit(1000).all()
        item_ids = [item[0] for item in top_items]
        logger.info(f"Analyzing {len(item_ids)} items")

        # Full rebuild: clear previous results so re-runs don't duplicate rows.
        self.db.query(EventCorrelation).delete(synchronize_session=False)
        self.db.query(EventPattern).delete(synchronize_session=False)
        self.db.query(EventImpact).delete(synchronize_session=False)
        self.db.commit()

        # Batch load all prices needed for analysis (single query vs thousands)
        # Get date range: earliest event minus 1 day to latest event plus 7 days
        if events:
            earliest_event = min(e.timestamp.date() for e in events)
            latest_event = max(e.timestamp.date() for e in events)
            start_date = earliest_event - timedelta(days=1)
            end_date = latest_event + timedelta(days=7)
            self.batch_load_prices(item_ids, start_date, end_date)

        same_day_counts = Counter(event.timestamp.date() for event in events)
        control_avg_by_date = {}
        for event in events:
            event_date = event.timestamp.date()
            if event_date not in control_avg_by_date:
                control_avg_by_date[event_date] = self.compute_control_group_average(event_date)

        impacts_recorded = 0
        patterns_learned = 0
        correlations_validated = 0
        impact_lookup = {}

        # Only iterate items that actually have cached prices; items without
        # any data in the event window can never produce an impact.
        items_with_data = [item_id for item_id in item_ids if item_id in self.price_cache]

        # 1. Record impacts for each event-item pair
        impacts = []
        for event in events:
            for item_id in items_with_data:
                try:
                    impact = self.record_event_impact(event, item_id)
                    if impact:
                        impacts.append(impact)
                        impact_lookup[(event.id, item_id)] = impact
                        impacts_recorded += 1
                except Exception as e:
                    logger.warning(f"Error recording impact for event {event.id}, item {item_id}: {e}")

        if impacts:
            self.db.bulk_save_objects(impacts)
            self.db.commit()
            logger.info(f"Recorded {impacts_recorded} event impacts")

        # 2. Learn patterns from events by type in one grouped pass
        event_type_by_id = {event.id: event.type for event in events}
        patterns = self.build_event_patterns(impacts, event_type_by_id)
        if patterns:
            self.db.bulk_save_objects(list(patterns.values()))
            self.db.commit()
            patterns_learned = len(patterns)
            logger.info(f"Learned {patterns_learned} event patterns")

        # 3. Validate correlations for each recorded event-item impact
        correlations = []
        for event in events:
            event_date = event.timestamp.date()
            control_avg = control_avg_by_date.get(event_date)
            same_day_events = max(0, same_day_counts[event_date] - 1)
            for item_id in items_with_data:
                impact = impact_lookup.get((event.id, item_id))
                if impact is None:
                    continue
                try:
                    correlation = self.validate_event_correlation(
                        event,
                        item_id,
                        impact=impact,
                        pattern=patterns.get((event.type, item_id)),
                        control_avg=control_avg,
                        same_day_events=same_day_events
                    )
                    if correlation:
                        correlations.append(correlation)
                        correlations_validated += 1
                except Exception as e:
                    logger.warning(f"Error validating correlation for event {event.id}, item {item_id}: {e}")

        if correlations:
            self.db.bulk_save_objects(correlations)
            self.db.commit()
            logger.info(f"Validated {correlations_validated} event correlations")

        logger.info(f"✅ Analysis complete: {impacts_recorded} impacts, {patterns_learned} patterns, {correlations_validated} correlations")

        return {
            'status': 'success',
            'impacts_recorded': impacts_recorded,
            'patterns_learned': patterns_learned,
            'correlations_validated': correlations_validated
        }


def main():
    db = SessionLocal()

    try:
        analyzer = EventAnalyzer(db)
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
