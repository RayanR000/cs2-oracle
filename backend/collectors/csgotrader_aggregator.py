"""
CSGOTrader market aggregator.
Fetches comprehensive price data from public JSON endpoints.
"""

import logging
import requests
from datetime import datetime
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CSGOTraderAggregator:
    """Aggregator that fetches prices from public CSGOTrader endpoints."""
    
    # Standard public endpoints
    STEAM_URL = "https://prices.csgotrader.app/latest/steam.json"
    SKINPORT_URL = "https://prices.csgotrader.app/latest/skinport.json"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CS2Analyzer/1.0)"})
        self._price_cache = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize item names for resilient matching."""
        normalized = unicodedata.normalize("NFKD", name or "")
        normalized = normalized.casefold()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _is_sticker_name(name: str) -> bool:
        return name.lower().startswith("sticker | ")

    @staticmethod
    def _sticker_variants(name: str) -> List[str]:
        """
        Generate progressively shorter sticker variants.

        Example:
            Sticker | noway (Holo) | Shanghai 2024
            -> Sticker | noway (Holo) | Shanghai 2024
            -> Sticker | noway (Holo)
        """
        parts = [part.strip() for part in name.split(" | ")]
        if len(parts) < 3 or parts[0].lower() != "sticker":
            return []

        variants = []
        current = parts[:]
        while len(current) > 2:
            current = current[:-1]
            candidate = " | ".join(current).strip()
            if candidate:
                variants.append(candidate)
        return variants

    @staticmethod
    def _diagnostic_terms(name: str) -> List[str]:
        """
        Extract informative tokens for approximate source-key diagnostics.
        """
        normalized = CSGOTraderAggregator._normalize_name(name)
        stop_terms = {
            "sticker",
            "holo",
            "foil",
            "glitter",
            "gold",
            "paper",
            "team",
            "capsule",
            "legends",
            "challengers",
            "contenders",
            "2021",
            "2022",
            "2023",
            "2024",
            "2025",
            "rio",
            "stockholm",
            "antwerp",
            "paris",
            "copenhagen",
            "shanghai",
        }
        tokens = []
        for token in re.split(r"[^a-z0-9]+", normalized):
            if len(token) < 4 or token in stop_terms:
                continue
            tokens.append(token)
        return tokens

    def find_source_key_candidates(self, name: str, limit: int = 10) -> List[str]:
        """
        Find probable source keys that may correspond to a missing item name.

        This is a diagnostic helper used to understand whether a miss is due to
        naming drift or a true source-coverage gap.
        """
        if not self._price_cache:
            self._price_cache = self.fetch_all_prices()

        terms = self._diagnostic_terms(name)
        if not terms:
            return []

        candidates: List[str] = []
        for source_key in self._price_cache.keys():
            normalized_key = self._normalize_name(source_key)
            if any(term in normalized_key for term in terms):
                candidates.append(source_key)
                if len(candidates) >= limit:
                    break
        return candidates

    def fetch_all_prices(self) -> Dict[str, float]:
        """Fetch latest prices from public endpoints and merge them."""
        prices = {}
        for url in [self.STEAM_URL, self.SKINPORT_URL]:
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # CSGOTrader price data is nested under a "data" key in some versions
                if "data" in data:
                    data = data["data"]
                    
                # Iterate and extract
                for item_name, info in data.items():
                    if isinstance(info, dict):
                        # CSGOTrader price data is often under 'last_24h'
                        price = info.get('last_24h') or info.get('price')
                        if price is not None:
                            prices[item_name] = float(price)
                    elif isinstance(info, (int, float)):
                        prices[item_name] = float(info)
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
        return prices

    def collect_batch_items(self, item_names: List[str]) -> Dict[str, Optional[Tuple[float, int, datetime]]]:
        """Fetch prices for a list of items using fuzzy matching."""
        if not self._price_cache:
            self._price_cache = self.fetch_all_prices()
            
        results = {}
        # Pre-process cache keys to lower-case
        cache_keys = {k.lower(): k for k in self._price_cache.keys()}
        normalized_cache_keys = {self._normalize_name(k): k for k in self._price_cache.keys()}
        
        qualities = ["(Factory New)", "(Minimal Wear)", "(Field-Tested)", "(Well-Worn)", "(Battle-Scarred)"]
            
        for name in item_names:
            name_lower = name.lower()
            normalized_name = self._normalize_name(name)
            found_key = None
            
            # 1. Direct/Exact match
            if name_lower in cache_keys:
                found_key = cache_keys[name_lower]
            elif normalized_name in normalized_cache_keys:
                found_key = normalized_cache_keys[normalized_name]
            elif self._is_sticker_name(name):
                # Stickers often include event/capsule suffixes in the DB that
                # may not exist in the external feed. Try shorter sticker keys
                # before falling back to a miss.
                for candidate in self._sticker_variants(name):
                    candidate_lower = candidate.lower()
                    normalized_candidate = self._normalize_name(candidate)
                    if candidate_lower in cache_keys:
                        found_key = cache_keys[candidate_lower]
                        break
                    if normalized_candidate in normalized_cache_keys:
                        found_key = normalized_cache_keys[normalized_candidate]
                        break
            else:
                # 2. Try adding quality suffix
                for q in qualities:
                    candidate = f"{name_lower} {q.lower()}".replace("  ", " ")
                    if candidate in cache_keys:
                        found_key = cache_keys[candidate]
                        break
                    normalized_candidate = self._normalize_name(f"{name} {q}")
                    if normalized_candidate in normalized_cache_keys:
                        found_key = normalized_cache_keys[normalized_candidate]
                        break
            
            if found_key:
                price = self._price_cache[found_key]
                results[name] = (float(price), 0, datetime.utcnow())
        
        return results
