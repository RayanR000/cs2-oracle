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

SourceData = Dict[str, Tuple[float, Optional[int], datetime]]


def _get_safe(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class CSGOTraderAggregator:
    """Aggregator that fetches prices from public CSGOTrader endpoints."""

    STEAM_URL = "https://prices.csgotrader.app/latest/steam.json"
    SKINPORT_URL = "https://prices.csgotrader.app/latest/skinport.json"
    BUFF163_URL = "https://prices.csgotrader.app/latest/buff163.json"
    CSFLOAT_URL = "https://prices.csgotrader.app/latest/csfloat.json"
    CSMONEY_URL = "https://prices.csgotrader.app/latest/csmoney.json"
    CSGOTRADER_URL = "https://prices.csgotrader.app/latest/csgotrader.json"
    YOUPIN_URL = "https://prices.csgotrader.app/latest/youpin.json"
    EXCHANGE_RATES_URL = "https://prices.csgotrader.app/latest/exchange_rates.json"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CS2Analyzer/1.0)"})
        self._raw_sources: Dict[str, Dict[str, dict]] = {}
        self._price_cache: Dict[str, float] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
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
        normalized = CSGOTraderAggregator._normalize_name(name)
        stop_terms = {
            "sticker", "holo", "foil", "glitter", "gold", "paper",
            "team", "capsule", "legends", "challengers", "contenders",
            "2021", "2022", "2023", "2024", "2025",
            "rio", "stockholm", "antwerp", "paris", "copenhagen", "shanghai",
        }
        tokens = []
        for token in re.split(r"[^a-z0-9]+", normalized):
            if len(token) < 4 or token in stop_terms:
                continue
            tokens.append(token)
        return tokens

    def find_source_key_candidates(self, name: str, limit: int = 10) -> List[str]:
        if not self._price_cache:
            self.fetch_all_market_data()
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
        if not CSGOTraderAggregator._is_sticker_name(name):
            return [name]
        candidates = [name]
        stripped_variant = re.sub(
            r"\s*\((Holo|Glitter|Gold|Foil|Paper)\)(?=\s*\|)",
            "", name, flags=re.IGNORECASE,
        )
        if stripped_variant != name:
            candidates.append(stripped_variant)
        return list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))

    @staticmethod
    def _general_match_candidates(name: str) -> List[str]:
        candidates = [name]
        if name.startswith("★ "):
            candidates.append(name[2:].strip())
        if name.lower().startswith("souvenir charm | "):
            candidates.append("Souvenir | " + name.split("| ", 1)[1])

        stripped_stattrak = re.sub(
            r"^\s*(★\s*)?StatTrak™\s*", "", name, flags=re.IGNORECASE
        ).strip()
        if stripped_stattrak != name:
            candidates.append(stripped_stattrak)

        stripped_souvenir = re.sub(
            r"^\s*Souvenir\s+", "", name, flags=re.IGNORECASE
        ).strip()
        if stripped_souvenir != name and stripped_souvenir not in candidates:
            candidates.append(stripped_souvenir)

        return list(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))

    def fetch_all_market_data(self) -> Dict[str, Dict[str, dict]]:
        """Fetch raw data from all configured endpoints.

        Returns:
            {source_name: {item_name: raw_dict, ...}, ...}
        """
        endpoints = {
            "steam": self.STEAM_URL,
            "skinport": self.SKINPORT_URL,
            "buff163": self.BUFF163_URL,
            "csfloat": self.CSFLOAT_URL,
            "csmoney": self.CSMONEY_URL,
            "csgotrader": self.CSGOTRADER_URL,
            "youpin": self.YOUPIN_URL,
        }
        self._raw_sources = {}
        self._price_cache = {}

        failed = 0
        for source_name, url in endpoints.items():
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                if "data" in data:
                    data = data["data"]
                if isinstance(data, dict):
                    self._raw_sources[source_name] = data
                    for item_name, info in data.items():
                        price = self._extract_primary_price(source_name, info)
                        if price is not None:
                            self._price_cache[item_name] = price
                    logger.info("Fetched %s: %s items", url, len(data))
                else:
                    failed += 1
                    logger.warning("Unexpected response format from %s", url)
            except Exception as e:
                failed += 1
                logger.warning("Failed to fetch %s: %s", url, e)

        ok = len(endpoints) - failed
        logger.info("Market data fetch complete: %s/%s endpoints succeeded", ok, len(endpoints))
        if failed == len(endpoints):
            logger.critical("ALL %s market data endpoints failed — no data available for this run", len(endpoints))
        elif not self._raw_sources:
            logger.warning("All market data sources failed — no data available")

        return self._raw_sources

    @staticmethod
    def _extract_primary_price(source: str, info) -> Optional[float]:
        """Extract the primary/representative price from a source data dict."""
        if not isinstance(info, dict):
            return _get_safe(info)
        if source == "steam":
            return _get_safe(info.get("last_24h"))
        if source == "skinport":
            return _get_safe(info.get("starting_at"))
        if source == "buff163":
            starting = info.get("starting_at")
            if isinstance(starting, dict):
                return _get_safe(starting.get("price"))
            return _get_safe(starting)
        if source == "csfloat":
            return _get_safe(info.get("price"))
        if source in ("csmoney", "csgotrader"):
            return _get_safe(info.get("price"))
        if source == "youpin":
            return _get_safe(info.get("price")) or _get_safe(info)
        return None

    @staticmethod
    def _build_source_lookup(source_data: Dict[str, dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Pre-build lowercase + normalized lookup dicts for a source."""
        cache_keys = {k.lower(): k for k in source_data.keys()}
        normalized_cache_keys = {}
        for k in source_data.keys():
            normalized_cache_keys[CSGOTraderAggregator._normalize_name(k)] = k
        return cache_keys, normalized_cache_keys

    def _match_item(self, name: str,
                    cache_keys: Dict[str, str],
                    normalized_cache_keys: Dict[str, str]) -> Optional[str]:
        """Fuzzy-match an item name against pre-built lookup dicts. Returns the matched key or None."""
        name_lower = name.lower()
        normalized_name = self._normalize_name(name)

        if name_lower in cache_keys:
            return cache_keys[name_lower]
        if normalized_name in normalized_cache_keys:
            return normalized_cache_keys[normalized_name]

        if not self._is_sticker_name(name):
            qualities = ["(Factory New)", "(Minimal Wear)", "(Field-Tested)", "(Well-Worn)", "(Battle-Scarred)"]
            for candidate_name in self._general_match_candidates(name):
                candidate_lower = candidate_name.lower()
                if candidate_lower in cache_keys:
                    return cache_keys[candidate_lower]
                normalized_candidate = self._normalize_name(candidate_name)
                if normalized_candidate in normalized_cache_keys:
                    return normalized_cache_keys[normalized_candidate]
                for q in qualities:
                    candidate = f"{candidate_lower} {q.lower()}".replace("  ", " ")
                    if candidate in cache_keys:
                        return cache_keys[candidate]
                    normalized_q = self._normalize_name(f"{candidate_name} {q}")
                    if normalized_q in normalized_cache_keys:
                        return normalized_cache_keys[normalized_q]
        else:
            for candidate_name in self._sticker_match_candidates(name):
                candidate_lower = candidate_name.lower()
                if candidate_lower in cache_keys:
                    return cache_keys[candidate_lower]
                normalized_candidate = self._normalize_name(candidate_name)
                if normalized_candidate in normalized_cache_keys:
                    return normalized_cache_keys[normalized_candidate]
        return None

    def collect_batch_items(self, item_names: List[str]) -> Dict[str, Optional[SourceData]]:
        """Fetch prices for a list of items from all available sources.

        Returns:
            {item_name: {"steam": (price, 0, ts), "steam_7d": ..., "skinport": ..., ...} | None}
        """
        if not self._raw_sources:
            self.fetch_all_market_data()

        if not self._raw_sources:
            logger.error("No market data available — returning empty results")
            return {}

        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Pre-build lookup dicts once per source instead of per item per source
        source_lookups: Dict[str, Tuple[Dict[str, str], Dict[str, str]]] = {}
        for src_name, src_data in self._raw_sources.items():
            source_lookups[src_name] = self._build_source_lookup(src_data)

        results: Dict[str, Optional[SourceData]] = {}

        matched_count = 0
        for name in item_names:
            sources: SourceData = {}

            for src_name, src_data in self._raw_sources.items():
                cache_keys, normalized_cache_keys = source_lookups[src_name]
                matched_key = self._match_item(name, cache_keys, normalized_cache_keys)
                if matched_key is None:
                    continue

                info = src_data[matched_key]
                if not isinstance(info, dict):
                    price = _get_safe(info)
                    if price is not None:
                        sources[src_name] = (price, None, now)
                    continue

                if src_name == "steam":
                    p24 = _get_safe(info.get("last_24h"))
                    p7 = _get_safe(info.get("last_7d"))
                    p30 = _get_safe(info.get("last_30d"))
                    p90 = _get_safe(info.get("last_90d"))

                    if p24 is not None:
                        sources["steam"] = (p24, 0, now)
                    elif p7 is not None:
                        sources["steam"] = (p7, 0, now)
                    elif p30 is not None:
                        sources["steam"] = (p30, 0, now)
                    elif p90 is not None:
                        sources["steam"] = (p90, 0, now)

                    if p7 is not None:
                        sources["steam_7d"] = (p7, None, now)
                    if p30 is not None:
                        sources["steam_30d"] = (p30, None, now)
                    if p90 is not None:
                        sources["steam_90d"] = (p90, None, now)

                elif src_name == "skinport":
                    p = _get_safe(info.get("starting_at"))
                    if p is not None:
                        sources["skinport"] = (p, None, now)

                elif src_name == "buff163":
                    starting = info.get("starting_at")
                    if isinstance(starting, dict):
                        p = _get_safe(starting.get("price"))
                        if p is not None:
                            sources["buff163"] = (p, None, now)
                    else:
                        p = _get_safe(starting)
                        if p is not None:
                            sources["buff163"] = (p, None, now)

                    highest = info.get("highest_order")
                    if isinstance(highest, dict):
                        p = _get_safe(highest.get("price"))
                        if p is not None:
                            sources["buff163_buy"] = (p, None, now)

                elif src_name == "csfloat":
                    p = _get_safe(info.get("price"))
                    if p is not None:
                        sources["csfloat"] = (p, None, now)

                elif src_name == "csmoney":
                    p = _get_safe(info.get("price"))
                    if p is not None:
                        sources["csmoney"] = (p, None, now)

                elif src_name == "csgotrader":
                    p = _get_safe(info.get("price"))
                    if p is not None:
                        sources["csgotrader"] = (p, None, now)

                elif src_name == "youpin":
                    p = _get_safe(info.get("price")) or _get_safe(info)
                    if p is not None:
                        sources["youpin"] = (p, None, now)

            if sources:
                results[name] = sources
                matched_count += 1

        logger.info("Aggregator match results: %s/%s items matched across sources",
                     matched_count, len(item_names))
        if matched_count == 0:
            logger.warning("No items matched any source — check upstream data format")
        elif matched_count < len(item_names) * 0.5:
            logger.warning("Low match rate: %s/%s (%.0f%%) — sources may have changed format",
                           matched_count, len(item_names), 100 * matched_count / len(item_names))

        return results

    def fetch_exchange_rates(self) -> Optional[Dict[str, float]]:
        """Fetch currency exchange rates from CSGOTrader."""
        try:
            response = self.session.get(self.EXCHANGE_RATES_URL, timeout=15)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                logger.info("Fetched exchange rates: %s currencies", len(data))
                return data
            logger.warning("Unexpected exchange_rates format")
        except Exception as e:
            logger.warning("Failed to fetch exchange rates: %s", e)
        return None
