"""Adapters for one-shot historical price-source imports.

An adapter supplies two things and nothing else: how to fetch one day, and how
to parse that day into ``(item_slug, day, price)`` records. Stall detection, the
gap quality gate, schema normalisation and the write are shared across sources
and live in ``collectors/price_history_import.py``.

A second backfill source is expected; add a module here and register it below.
"""

from . import cs2_prices_tracker

ADAPTERS = {
    "cs2_prices_tracker": cs2_prices_tracker,
}
