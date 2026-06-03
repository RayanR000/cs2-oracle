"""
CSGOTrader market aggregator.
Fetches comprehensive price data from public JSON endpoints.
"""

import logging
import requests
from datetime import datetime, timezone
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
        normalized = (name or "").replace("™", "").replace("®", "")
        normalized = unicodedata.normalize("NFKD", normalized)
        normalized = normalized.casefold()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _is_sticker_name(name: str) -> bool:
        return name.lower().startswith("sticker | ")

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

    @staticmethod
    def _sticker_match_candidates(name: str) -> List[str]:
        """Generate conservative sticker aliases that preserve the event suffix."""
        if not CSGOTraderAggregator._is_sticker_name(name):
            return [name]

        candidates = [name]

        # Many source keys omit the finish variant but keep the sticker name and event suffix.
        stripped_variant = re.sub(
            r"\s*\((Holo|Glitter|Gold|Foil|Paper)\)(?=\s*\|)",
            "",
            name,
            flags=re.IGNORECASE,
        )
        if stripped_variant != name:
            candidates.append(stripped_variant)

        # Keep ordering stable while removing duplicates.
        return list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))

    @staticmethod
    def _general_match_candidates(name: str) -> List[str]:
        """Generate conservative non-sticker aliases for known source formatting drift."""
        candidates = [name]

        if name.startswith("★ "):
            candidates.append(name[2:].strip())

        return list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))

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
            elif not self._is_sticker_name(name):
                # 2. Try conservative aliases and quality suffixes.
                for candidate_name in self._general_match_candidates(name):
                    candidate_lower = candidate_name.lower()
                    if candidate_lower in cache_keys:
                        found_key = cache_keys[candidate_lower]
                        break
                    normalized_candidate = self._normalize_name(candidate_name)
                    if normalized_candidate in normalized_cache_keys:
                        found_key = normalized_cache_keys[normalized_candidate]
                        break

                    for q in qualities:
                        candidate = f"{candidate_lower} {q.lower()}".replace("  ", " ")
                        if candidate in cache_keys:
                            found_key = cache_keys[candidate]
                            break
                        normalized_candidate = self._normalize_name(f"{candidate_name} {q}")
                        if normalized_candidate in normalized_cache_keys:
                            found_key = normalized_cache_keys[normalized_candidate]
                            break
                    if found_key:
                        break
            else:
                # 2. Sticker-specific fallback: strip finish variants while keeping event suffix.
                for candidate_name in self._sticker_match_candidates(name):
                    candidate_lower = candidate_name.lower()
                    if candidate_lower in cache_keys:
                        found_key = cache_keys[candidate_lower]
                        break
                    normalized_candidate = self._normalize_name(candidate_name)
                    if normalized_candidate in normalized_cache_keys:
                        found_key = normalized_cache_keys[normalized_candidate]
                        break

            if found_key:
                price = self._price_cache[found_key]
                results[name] = (float(price), 0, datetime.now(timezone.utc).replace(tzinfo=None))
        
        return results
