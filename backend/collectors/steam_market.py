"""
Steam Community Market data collector
Handles scraping and API calls to gather CS2 market data
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class SteamMarketCollector:
    """Collects price data from Steam Community Market"""
    
    BASE_URL = "https://steamcommunity.com/market"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Extremely conservative rate limiting
    REQUEST_DELAY = 15.0  # seconds between requests
    RETRY_ATTEMPTS = 5
    RETRY_DELAY = 10.0
    
    # Rotate User-Agents to avoid bot detection
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    ]
    
    def __init__(self, rate_limit_delay: float = 15.0):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self._rotate_user_agent()
        self.last_request_time = 0
        self.hash_name_cache = {}

    def _rotate_user_agent(self):
        import random
        ua = random.choice(self.USER_AGENTS)
        self.session.headers.update({'User-Agent': ua})


    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None, timeout: int = 20) -> Optional[Dict]:
        """
        Make HTTP request with robust retry logic (including backoff for 429s)
        """
        self._rate_limit()
        
        current_delay = self.RETRY_DELAY
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                
                # Handle 429 (Too Many Requests) with explicit backoff
                if response.status_code == 429:
                    logger.warning(f"Rate limited (429) on {url}, backing off for {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= 2 # Exponential backoff
                    continue
                    
                response.raise_for_status()
                
                if response.text:
                    return response.json()
                return None
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.RETRY_ATTEMPTS}): {e}")
                if attempt < self.RETRY_ATTEMPTS - 1:
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    return None
        return None

    def resolve_hash_name(self, item_name: str) -> Optional[str]:
        """
        Resolve item name to Steam market hash name.
        Uses Steam's market search endpoint to find the exact hash.
        Results are cached to avoid repeated lookups.
        """
        # Check cache first
        if item_name in self.hash_name_cache:
            return self.hash_name_cache[item_name]

        # Sanitize query: Steam search often fails with exact pipes and parens
        search_query = item_name.replace('|', '').replace('(', '').replace(')', '').replace('  ', ' ')

        # Query Steam market search endpoint
        url = "https://steamcommunity.com/market/search/render/"
        params = {
            'query': search_query,
            'start': 0,
            'count': 10,
            'search_descriptions': 0,
            'sort_column': 'name',
            'sort_dir': 'asc',
            'norender': 1
        }

        data = self._make_request(url, params)
        if not data or 'results' not in data or not data['results']:
            logger.warning(f"No market hash found for: {item_name} (queried: {search_query})")
            return None

        # Try to find an exact match or a very close match in the results
        hash_name = None
        item_name_lower = item_name.lower()
        
        # Split item_name into weapon and skin part if it's a skin
        name_parts = item_name_lower.replace(' | ', ' ').split(' ')
        
        for result in data['results']:
            res_hash = result.get('hash_name')
            if not res_hash:
                continue
                
            res_hash_lower = res_hash.lower()
            
            # Exact match is best
            if res_hash == item_name:
                hash_name = res_hash
                break
            
            # Close match: must contain at least the first two significant parts of the name
            # (e.g., "M4A4" and "Poseidon")
            matches_all = True
            for part in name_parts[:2]:
                if part not in res_hash_lower:
                    matches_all = False
                    break
            
            if matches_all:
                hash_name = res_hash
                break
        
        if hash_name:
            # Cache it
            self.hash_name_cache[item_name] = hash_name
            logger.debug(f"Resolved {item_name} -> {hash_name}")
            return hash_name

        logger.warning(f"Could not extract hash_name from result for: {item_name}")
        return None

    def get_item_price_history(self, item_name_or_hash: str) -> Optional[Tuple[float, int, datetime]]:
        """
        Get current price and volume for an item using the Price Overview API.
        Accepts either item name or market hash name.
        """
        # Resolve hash name if necessary
        if '%' not in item_name_or_hash and ' | ' not in item_name_or_hash:
            hash_name = self.resolve_hash_name(item_name_or_hash)
            if not hash_name:
                return None
        else:
            hash_name = item_name_or_hash

        # Use the Price Overview API, which is reliable for snapshots
        trend = self.get_price_trend(hash_name)
        if trend and trend.get('lowest_price'):
            price = trend['lowest_price']
            # Volume can be a string or integer
            volume_raw = trend.get('volume', '0')
            if isinstance(volume_raw, str):
                volume = int(volume_raw.replace(',', ''))
            else:
                volume = int(volume_raw or 0)
            return (price, volume, datetime.utcnow())

        logger.warning(f"No price data available for: {hash_name}")
        return None
    
    def _parse_price(self, price_str: str) -> float:
        """Parse currency string like '$1,234.56' to float 1234.56"""
        if not price_str:
            return 0.0
        try:
            # Remove currency symbols and thousands separators
            clean_str = ''.join(c for c in price_str if c.isdigit() or c == '.')
            return float(clean_str)
        except ValueError:
            return 0.0

    def get_market_listings(self, start: int = 0, count: int = 100) -> Optional[Dict]:
        """
        Get market listings for CS2 (AppID 730)
        
        Args:
            start: Starting index
            count: Number of items to fetch (max 100)
            
        Returns:
            Dictionary containing processed results and total count
        """
        url = f"{self.BASE_URL}/search/render/"
        params = {
            'query': '',
            'appid': 730,
            'search_descriptions': 0,
            'sort_column': 'name',
            'sort_dir': 'asc',
            'start': start,
            'count': min(count, 100),
            'norender': 1,
            'currency': 1  # USD
        }
        
        data = self._make_request(url, params)
        if not data or not data.get('success') or 'results' not in data:
            return None
            
        processed_results = []
        for res in data['results']:
            processed_results.append({
                'hash_name': res.get('hash_name'),
                'price': self._parse_price(res.get('sell_price_text')),
                'volume': int(res.get('sell_listings', 0)),
                'median_price': self._parse_price(res.get('sell_price_text')) # Render doesn't give median easily
            })
            
        return {
            'total_count': data.get('total_count', 0),
            'results': processed_results
        }
    
    def get_item_name_id(self, item_name: str) -> Optional[int]:
        """
        Get the nameid for an item (used for historical data)
        
        Args:
            item_name: Item name to search for
            
        Returns:
            Item nameid or None if not found
        """
        try:
            data = self.get_market_listings(count=1)
            if data and 'results' in data:
                for result in data['results']:
                    if result.get('hash_name', '').lower() == item_name.lower():
                        return result.get('name_id')
            return None
        except Exception as e:
            logger.error(f"Error getting nameid for {item_name}: {e}")
            return None
    
    def collect_batch_items(self, item_names: List[str]) -> Dict[str, Optional[Tuple[float, int, datetime]]]:
        """
        Collect price data for multiple items
        
        Args:
            item_names: List of item market hash names
            
        Returns:
            Dictionary mapping item names to (price, volume, timestamp) tuples
        """
        results = {}
        successful_items = 0
        failed_items = 0
        
        for item_name in item_names:
            try:
                result = self.get_item_price_history(item_name)
                if result:
                    results[item_name] = result
                    successful_items += 1
                    logger.info(f"Successfully collected price for: {item_name}")
                else:
                    results[item_name] = None
                    failed_items += 1
                    logger.warning(f"Failed to collect price for: {item_name}")
            except Exception as e:
                results[item_name] = None
                failed_items += 1
                logger.error(f"Error collecting {item_name}: {e}")
        
        logger.info(f"Batch collection completed: {successful_items} successful, {failed_items} failed out of {len(item_names)}")
        return results
    
    def get_price_trend(self, item_name: str) -> Optional[Dict]:
        """
        Get price trend data (low/high) for an item
        
        Args:
            item_name: Item market hash name
            
        Returns:
            Dictionary with low, high, volume trend or None if failed
        """
        url = f"{self.BASE_URL}/priceoverview/"
        params = {
            'appid': 730,
            'market_hash_name': item_name,
            'currency': 1
        }
        
        try:
            data = self._make_request(url, params)
            if data and data.get('success'):
                return {
                    'lowest_price': float(data.get('lowest_price', '0').replace('$', '').replace(',', '')) or None,
                    'highest_price': float(data.get('median_price', '0').replace('$', '').replace(',', '')) or None,
                    'volume': data.get('volume'),
                    'timestamp': datetime.utcnow()
                }
            return None
        except Exception as e:
            logger.error(f"Error getting price trend for {item_name}: {e}")
            return None


class MockSteamMarketCollector(SteamMarketCollector):
    """Mock collector for testing without hitting Steam API"""
    
    def get_item_price_history(self, hash_name: str) -> Optional[Tuple[float, int, datetime]]:
        """Return mock data for testing"""
        import random
        price = random.uniform(10, 500)
        volume = random.randint(100, 10000)
        return (price, volume, datetime.utcnow())
    
    def get_market_listings(self, start: int = 0, count: int = 100) -> Optional[Dict]:
        """Return mock listings"""
        return {
            'success': True,
            'results_html': '<div>mock</div>',
            'results': []
        }
