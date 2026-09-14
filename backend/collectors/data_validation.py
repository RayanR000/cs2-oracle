"""
Data validation and cleaning pipeline
"""

import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates and cleans market data"""

    @staticmethod
    def validate_price(price: float, min_price: float = 0.01, max_price: float = 100000.0) -> bool:
        """
        Validate price is within reasonable range

        Args:
            price: Price to validate
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(price, (int, float)):
            return False
        if price < min_price or price > max_price:
            logger.warning(f"Price {price} outside acceptable range [{min_price}, {max_price}]")
            return False
        return True

    @staticmethod
    def validate_volume(volume: int | None) -> bool:
        """
        Validate volume is reasonable

        Args:
            volume: Volume to validate

        Returns:
            True if valid or None, False if invalid
        """
        if volume is None:
            return True
        if not isinstance(volume, int):
            return False
        if volume < 0 or volume > 1000000:
            logger.warning(f"Volume {volume} outside acceptable range [0, 1000000]")
            return False
        return True

    @staticmethod
    def validate_item_name(name: str) -> bool:
        """
        Validate item name format

        Args:
            name: Item name to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(name, str):
            return False
        if len(name) < 1 or len(name) > 500:
            return False
        return True

    @staticmethod
    def validate_item_type(item_type: str) -> bool:
        """
        Validate item type

        Args:
            item_type: Type to validate

        Returns:
            True if valid, False otherwise
        """
        valid_types = {"skin", "case", "sticker", "agent", "glove", "graffiti", "patch", "other"}
        return item_type.lower() in valid_types

    @staticmethod
    def validate_price_record(price_record: dict) -> tuple[bool, str | None]:
        """Validate a complete price record."""
        return DataCleaner.validate_price_record(price_record)

    @staticmethod
    def compute_anomaly_score(current_price: float, historical_prices: list[float], window_size: int = 30) -> float:
        """Compatibility wrapper for anomaly scoring."""
        return DataCleaner.compute_anomaly_score(current_price, historical_prices, window_size)

    @staticmethod
    def detect_market_manipulation(
        prices: list[float], volumes: list[int], lookback: int = 7
    ) -> tuple[bool, str | None]:
        """Compatibility wrapper for market manipulation detection."""
        return DataCleaner.detect_market_manipulation(prices, volumes, lookback)

    @staticmethod
    def filter_outliers(prices: list[float], threshold: float = 3.0) -> list[float]:
        """Compatibility wrapper for outlier filtering."""
        return DataCleaner.filter_outliers(prices, threshold)

    @staticmethod
    def parse_item_name(market_hash_name: str) -> tuple[str, str | None]:
        """
        Parse market hash name into item name and rarity/condition

        Example: 'AK-47 | Aquamarine Revenge (Field-Tested)'
        -> ('AK-47 | Aquamarine Revenge', 'Field-Tested')

        Args:
            market_hash_name: Full market hash name

        Returns:
            Tuple of (item_name, condition) or (market_hash_name, None) if unparseable
        """
        try:
            # Try to extract condition in parentheses
            match = re.match(r"(.+?)\s*\(([^)]+)\)$", market_hash_name)
            if match:
                return (match.group(1).strip(), match.group(2).strip())
            return (market_hash_name, None)
        except Exception as e:
            logger.error(f"Error parsing item name {market_hash_name}: {e}")
            return (market_hash_name, None)

    @staticmethod
    def detect_item_type(name: str) -> str:
        """
        Attempt to detect item type from name

        Args:
            name: Item name

        Returns:
            Detected type or 'other'
        """
        name_lower = name.lower()

        if "case" in name_lower and "case hardened" not in name_lower:
            return "case"
        elif "agent" in name_lower:
            return "agent"
        elif "glove" in name_lower:
            return "glove"
        elif "graffiti" in name_lower:
            return "graffiti"
        elif "patch" in name_lower:
            return "patch"
        else:
            # Default to skin if contains condition keywords
            if any(
                cond in name_lower
                for cond in ["factory new", "minimal wear", "field-tested", "well-worn", "battle-scarred"]
            ):
                return "skin"

        return "other"


class DataCleaner:
    """Cleans and normalizes market data"""

    @staticmethod
    def clean_price_data(price: float) -> float:
        """Clean and normalize price data"""
        return round(float(price), 2)

    @staticmethod
    def clean_volume_data(volume: int | None) -> int | None:
        """Clean and normalize volume data"""
        if volume is None:
            return None
        return max(0, int(volume))

    @staticmethod
    def clean_item_name(name: str) -> str:
        """Clean item name (strip whitespace, normalize unicode)"""
        if not name:
            return ""
        # Normalize unicode and strip whitespace
        import unicodedata

        name = unicodedata.normalize("NFKD", name)
        return name.strip()

    @staticmethod
    def sanitize_price(price: float) -> float:
        """Compatibility wrapper for older collector code."""
        return DataCleaner.clean_price_data(price)

    @staticmethod
    def sanitize_volume(volume: int | None) -> int | None:
        """Compatibility wrapper for older collector code."""
        return DataCleaner.clean_volume_data(volume)

    @staticmethod
    def sanitize_item_name(name: str) -> str:
        """Compatibility wrapper for older collector code."""
        return DataCleaner.clean_item_name(name)

    @staticmethod
    def deduplicate_items(items: list[dict]) -> list[dict]:
        """
        Remove duplicate items by name

        Args:
            items: List of item dictionaries

        Returns:
            Deduplicated list (keeps first occurrence)
        """
        seen = set()
        result = []
        for item in items:
            name = item.get("name", "")
            if name not in seen:
                seen.add(name)
                result.append(item)
        return result

    @staticmethod
    def filter_outliers(prices: list[float], threshold: float = 3.0) -> list[float]:
        """
        Filter price outliers using z-score

        Args:
            prices: List of prices
            threshold: Z-score threshold (default: 3.0 = 99.7% of data)

        Returns:
            Filtered list without outliers
        """
        if len(prices) < 2:
            return prices

        import statistics

        try:
            mean = statistics.mean(prices)
            stdev = statistics.stdev(prices)

            if stdev == 0:
                return prices

            return [p for p in prices if abs((p - mean) / stdev) <= threshold]
        except Exception as e:
            logger.error(f"Error filtering outliers: {e}")
            return prices

    @staticmethod
    def validate_price_record(price_record: dict) -> tuple[bool, str | None]:
        """
        Validate a complete price record

        Args:
            price_record: Dictionary with price, volume, timestamp, item_name

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["price", "volume", "timestamp", "item_name"]

        # Check required fields
        for field in required_fields:
            if field not in price_record:
                return (False, f"Missing required field: {field}")

        # Validate each field
        price = price_record.get("price")
        if not DataValidator.validate_price(price):
            return (False, f"Invalid price: {price}")

        volume = price_record.get("volume")
        if not DataValidator.validate_volume(volume):
            return (False, f"Invalid volume: {volume}")

        item_name = price_record.get("item_name")
        if not DataValidator.validate_item_name(item_name):
            return (False, f"Invalid item name: {item_name}")

        timestamp = price_record.get("timestamp")
        if not isinstance(timestamp, datetime):
            return (False, f"Invalid timestamp: {timestamp}")

        return (True, None)

    @staticmethod
    def compute_anomaly_score(current_price: float, historical_prices: list[float], window_size: int = 30) -> float:
        """
        Compute anomaly score for a price (0.0 = normal, 1.0 = very anomalous)

        Args:
            current_price: Current price to evaluate
            historical_prices: List of historical prices for context
            window_size: Number of recent prices to consider

        Returns:
            Anomaly score between 0.0 and 1.0
        """
        if len(historical_prices) < 2:
            return 0.0

        import statistics

        try:
            # Use recent prices for comparison
            recent_prices = (
                historical_prices[-window_size:] if len(historical_prices) > window_size else historical_prices
            )

            mean = statistics.mean(recent_prices)
            stdev = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0

            if stdev == 0:
                return 0.0

            # Calculate z-score
            z_score = abs((current_price - mean) / stdev)

            # Convert z-score to anomaly score (0-1 scale)
            # 0 std dev away = 0.0, 3 std dev away = 1.0
            anomaly_score = min(z_score / 3.0, 1.0)

            return round(anomaly_score, 3)
        except Exception as e:
            logger.error(f"Error computing anomaly score: {e}")
            return 0.0

    @staticmethod
    def detect_market_manipulation(
        prices: list[float], volumes: list[int], lookback: int = 7
    ) -> tuple[bool, str | None]:
        """
        Detect potential market manipulation patterns

        Args:
            prices: List of recent prices
            volumes: List of corresponding volumes
            lookback: Number of days to analyze

        Returns:
            Tuple of (is_suspicious, description)
        """
        if len(prices) < lookback or len(volumes) < lookback:
            return (False, None)

        try:
            import statistics

            recent_prices = prices[-lookback:]
            recent_volumes = volumes[-lookback:]

            # Check 1: Sudden price spike with low volume (unlikely)
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
            avg_volume = statistics.mean(recent_volumes)

            if abs(price_change) > 0.50 and recent_volumes[-1] < avg_volume * 0.5:
                return (True, "Sudden price change with low volume - potential manipulation")

            # Check 2: Extreme volume spike
            if recent_volumes[-1] > statistics.mean(recent_volumes) * 5:
                return (True, "Extreme volume spike detected - may indicate pump/dump")

            # Check 3: Repeated identical prices (suspicious)
            unique_prices = len(set(recent_prices))
            if unique_prices < len(recent_prices) * 0.3:
                return (True, "Repeated identical prices - potential manipulation")

            return (False, None)
        except Exception as e:
            logger.error(f"Error detecting market manipulation: {e}")
            return (False, None)
