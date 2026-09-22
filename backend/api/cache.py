"""
Tiny in-process TTL cache for read-heavy endpoints.

The underlying data changes once a day (collection at 23:00 UTC, analysis
at ~03:00 UTC), so short TTLs are purely a freshness guard for manual
backfills — cache hits serve from memory instead of re-running multi-second
aggregation queries against Supabase.

Single-process only: with multiple uvicorn workers each holds its own cache,
which is fine (worst case one cold build per worker per TTL).
"""

import threading
import time

_MAX_ENTRIES = 256


class TTLCache:
    def __init__(self):
        self._data: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() > expires_at:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key, value, ttl_seconds: float):
        with self._lock:
            if len(self._data) >= _MAX_ENTRIES:
                # Drop the entry closest to expiry rather than tracking LRU.
                oldest = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest, None)
            self._data[key] = (time.time() + ttl_seconds, value)

    def clear(self):
        with self._lock:
            self._data.clear()


cache = TTLCache()


def get_or_build(key: str, ttl_seconds: float, builder):
    """Return the cached value for key, building and storing it on a miss."""
    value = cache.get(key)
    if value is None:
        value = builder()
        cache.set(key, value, ttl_seconds)
    return value
