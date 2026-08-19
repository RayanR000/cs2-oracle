"""
Parse CS2 item names into structured fields, and hold the archive's universe
rules.

The universe rules — which sources may vote and which names price a single
asset — live here rather than in `models/forecaster.py` because every archive
reader needs them and not every archive reader can afford to import LightGBM.
`forecaster.py` re-exports them so its own callers are unaffected. Both are
spelled once, as SQL predicates, because each of them has already cost a
separate fix per loader: the bid exclusion needed three, and the phase-collapsed
names needed four.

Examples:
    AK-47 | Redline (Field-Tested)
    StatTrak™ M4A4 | Desolate (Factory New)
    ★ Bayonet | Doppler (Factory New)
    Sticker | s1mple (Holo) | Paris 2024
    Souvenir AWP | Safari Mesh (Battle-Scarred)
    Glove Case
    Special Agent Ava | FBI
    Music Kit | PVRIS, Evergreen
    Sealed Graffiti | X-Axes (Blood Red)
    Charm | Gritty
"""

import re

# Sources that quote a BID, not an ask. Excluded from consensus voting: a bid is
# a different quantity, so median-voting it against asks is not noise reduction
# but a basis change. `aggregator_buff163_buy` is BUFF's `highest_order`, live
# since 2026-07-11 at 0.579x Steam against asks at 0.717-0.809x.
# Not a quality filter — these rows are good data, just not asks. Anything that
# wants the bid should read the source row from the archive directly.
BID_SOURCES = frozenset({"aggregator_buff163_buy"})

# Steam's trailing-window MEAN SALE price -- MA(7)/MA(30)/MA(90) -- which must
# not vote on equal terms against point-in-time asks
# (collectors/csgotrader_aggregator.py:294-312). Kept separate from
# BID_SOURCES because it is a different error: a bid is the wrong side of the
# book, an MA is the wrong TIME basis. Measured against genuine third-party
# asks (buff163/csfloat/csmoney/skinport/youpin; the Steam-derived
# aggregator_sync and aggregator_steam_17mafo excluded from the comparison
# group), the trailing window sits ~32% ABOVE them (median ratio 1.316, a
# premium on 76.6% of item-days) -- the same cash-out fee wedge every
# Steam-derived feed carries. So unlike the bid, excluding these pulls the
# consensus DOWN. The defect is not the level, it is the time basis: a
# trailing 90-day mean barely moves when live asks move, so it damps the
# consensus and mechanically manufactures mean-reversion in the resulting
# returns -- which matters acutely here because a reversal effect is the
# signal this project is currently trying to validate.
#
# These three sources exist only 2026-07-11 to 2026-08-08 (not "since
# 2026-03"), so it is labels and stored A/B verdicts from 2026-07-11 onward
# that sit downstream of this exclusion.
#
# Measured over 2026, >=$1, universe-filtered: excluding them costs 650
# item-days of 2,589,787 that have no other ask source, but moves the voted
# median on 16.89% of item-days (median -8.47%) and flips 6.17% of
# consecutive-day return directions.
#
# NOT a staleness fix: aggregator_sync and aggregator_steam_17mafo are
# `last_24h` FALLING BACK to these same windows on exactly the illiquid
# items. There is no point-in-time Steam price in this archive at all.
#
# Do NOT also drop aggregator_sync: that deletes 2026-01 and 2026-02 in full
# for the >=$1 cohort (52,048 item-days) to buy a further 1.4pp.
TRAILING_WINDOW_SOURCES = frozenset({
    "aggregator_steam_7d", "aggregator_steam_30d", "aggregator_steam_90d",
})

# Steam's point-in-time `last_24h` price, stored WITHOUT the trailing-window
# fallback that contaminates `aggregator_sync` (which is `last_24h` falling back
# to 7d/30d/90d means on exactly the illiquid items). Emitted by
# collectors/pipeline.py's steam branch from 2026-08-17 onward as the clean spot
# leg the cross-venue basis needs — see
# docs/research/2026-08-16-cross-venue-basis-steam-buff.md. It is NOT a new ask:
# Steam already votes through `aggregator_sync`, so this must be excluded from
# the consensus exactly like TRAILING_WINDOW_SOURCES, or it double-counts Steam.
# A diagnostic/feature source only; readers that want the clean spot select this
# source row from the archive directly. No historical backfill exists — past
# `last_24h` was folded into `aggregator_sync` and cannot be separated — so the
# series is empty before 2026-08-17 and NULL-safe membership keeps the pre-2026
# `source IS NULL` series voting.
STEAM_SPOT_SOURCES = frozenset({"aggregator_steam_spot"})


def bid_sources_sql_filter(column: str = "source") -> str:
    """SQL predicate dropping the bid sources from an archive read.

    NULL-safe: `source` is NULL for the whole pre-2026 series and a bare
    `NOT IN` over a NULL evaluates to NULL, which silently drops 13 years of
    prices. That is the same trap `phase_collapsed_sql_filter` guards.
    """
    quoted = ", ".join(f"'{s}'" for s in sorted(BID_SOURCES))
    return f"({column} IS NULL OR {column} NOT IN ({quoted}))"


# `historical_fallback:<source>` rows are a re-stamped stale price: when a day's
# collection misses an item, `collectors/pipeline.py:236-248` re-writes a quote up
# to 7 days old under *today's* `day`, prefixing the original source. Production
# already drops them (`forecaster.py`'s DB and DuckDB voted reads both spell the
# exclusion), but every archive-globbing loader that bypasses the voted path —
# `walkforward_backtest.py`, the `ab_test_*` harnesses — kept them, so a stale
# print entered features and labels under a fresh date. 12,655 rows over exactly
# six days (2026-07-11..16), on the cohort that failed to match that day. See
# docs/research/2026-08-19-deep-model-review.md 1d.
HISTORICAL_FALLBACK_PREFIX = "historical_fallback:"


def historical_fallback_sql_filter(column: str = "source") -> str:
    """SQL predicate dropping re-stamped stale-fallback rows from an archive read.

    NULL-safe like `bid_sources_sql_filter`: `source` is NULL for the whole
    pre-2026 series and a bare `NOT LIKE` over a NULL evaluates to NULL, which
    silently drops 13 years of prices.
    """
    return f"({column} IS NULL OR {column} NOT LIKE '{HISTORICAL_FALLBACK_PREFIX}%')"


# Keys that are a second copy of an item already in the universe. The archive's
# `item_slug` is `items.item_id` verbatim, and two writers keyed rows on
# something other than the `market_hash_name` every other inserter uses:
# `migrate_historical_data.py:345` wrote `slugify(name)` (3,145 keys) and a
# since-deleted `real_data_collector.py` wrote `f"steam_{...}"` (4 keys). The
# aggregator matches its price lookup on `name`, so both copies collect every
# day's price.
#
# Measured over the whole archive 2026-08-08: 3,149 keys / 786,408 rows (3.6%),
# and every one pairs 1:1 onto a correctly-keyed row — 99.93% of same-day
# same-source pairs agree to the cent. So this is a de-duplication, not a
# universe reduction: nothing is dropped that is not also present under its real
# name. The cost of leaving them is validation, not storage — a split that
# partitions by item can seat the same price series on both sides of a fold.
#
# The two forms need separate arms: the `steam_` keys hold '_' and '|', so the
# slug regex does not match them. See
# docs/changelog/2026-08-06-steam-listing-backfill-and-phantom-items.md.
PHANTOM_SLUG_PATTERN = r"[a-z0-9][a-z0-9\-]*"
PHANTOM_SLUG_PREFIX = "steam_"


def is_phantom_slug(item_slug: str) -> bool:
    """True when *item_slug* is a duplicate key rather than a market_hash_name.

    A real `market_hash_name` always carries a capital, a space or a delimiter
    ('|', '(', '™', '★'), so it cannot match the all-lowercase slug form.
    """
    if not isinstance(item_slug, str):
        return False
    return (re.fullmatch(PHANTOM_SLUG_PATTERN, item_slug) is not None
            or item_slug.startswith(PHANTOM_SLUG_PREFIX))


def phantom_slug_sql_filter(column: str = "item_slug") -> str:
    """SQL predicate keeping only the correctly-keyed copy of each item.

    NULL-safe like the other two: a bare `NOT regexp_full_match` over a NULL
    slug evaluates to NULL and silently drops the row.
    """
    return (f"({column} IS NULL OR NOT ("
            f"regexp_full_match({column}, '{PHANTOM_SLUG_PATTERN}')"
            f" OR starts_with({column}, '{PHANTOM_SLUG_PREFIX}')))")


def archive_universe_sql_filter(slug_column: str = "item_slug",
                                source_column: str = "source") -> str:
    """Every universe rule at once, for a loader that reads the archive direct.

    Production applies these inside `_fetch_voted_price_history`; anything that
    globs the Parquet itself — `walkforward_backtest.py`, every `ab_test_*.py`
    harness — bypasses that and has to spell them out. Before 2026-08-08 they
    all did bypass it, so an A/B measured a different item universe and a
    different price consensus than the model it was advising.

    Pass ``source_column=None`` for a read whose relation genuinely has no
    `source` column. That is safe only because the bid sources and the
    ``historical_fallback:`` re-stamps are all 2026 feeds: a file old enough to
    lack the column is old enough to contain none of them. The slug rules carry
    no such caveat and always apply.
    """
    parts = [phase_collapsed_sql_filter(slug_column),
             phantom_slug_sql_filter(slug_column)]
    if source_column:
        parts.append(bid_sources_sql_filter(source_column))
        parts.append(historical_fallback_sql_filter(source_column))
    return " AND ".join(parts)


# Names that are not one asset. A `market_hash_name` encodes weapon + finish +
# wear + StatTrak/Souvenir and nothing else, so a Doppler or Gamma Doppler name
# collapses every phase (Ruby / Sapphire / Black Pearl / Emerald / P1-P4) into a
# single series: 29 base names cover 181 distinct `paint_index` assets, the
# median max/min phase ratio *within one name* is 3.25x (p90 6.08x, max 23.5x),
# and 87.3% carry >2x internal dispersion. The quoted headline is the *cheapest*
# phase 95.5% of the time, so the series steps whenever which phase is cheapest
# changes — a level shift with no asset repricing, i.e. a fabricated return.
# `★ StatTrak™ M9 Bayonet | Doppler (Minimal Wear)` spans $1,261-$29,685 under
# one name.
#
# Matched on the name rather than against the BUFF dump's `doppler` sub-object:
# the sub-object is the detection rule but it lives in a live feed, and the
# archive read has to work offline and 13 years back. Measured over the whole
# archive (41,725 slugs), the word matches 129 slugs / 47,081 rows, of which 127
# are genuine Doppler or Gamma Doppler finishes and **two are not**:
# `Sticker | Doppler Poison Frog (Foil)` and its Sticker Slab twin, which are
# ordinary single assets that merely borrow the name. Hence the exemption.
# See docs/research/2026-08-07-cs2-forecasting-research.md 25.
PHASE_COLLAPSED_SLUG_PATTERNS = ("doppler",)
PHASE_COLLAPSED_EXEMPT_PATTERNS = ("sticker",)


def phase_collapsed_sql_filter(column: str = "item_slug") -> str:
    """SQL predicate keeping only names that price a single asset.

    Written as a helper so every archive reader spells the exclusion the same
    way — the bid exclusion needed three separate fixes because each loader
    carried its own glob.
    """
    hits = " OR ".join(f"lower({column}) LIKE '%{p}%'"
                       for p in PHASE_COLLAPSED_SLUG_PATTERNS)
    exempt = " OR ".join(f"lower({column}) LIKE '%{p}%'"
                         for p in PHASE_COLLAPSED_EXEMPT_PATTERNS)
    # NULL-safe like the BID_SOURCES filter: a bare NOT LIKE over a NULL slug
    # evaluates to NULL and silently drops the row.
    return f"({column} IS NULL OR NOT ({hits}) OR ({exempt}))"


def is_phase_collapsed(item_name: str) -> bool:
    """True when *item_name* (a market_hash_name or its slug) names several
    assets at once."""
    if not isinstance(item_name, str):
        return False
    lowered = item_name.lower()
    if any(p in lowered for p in PHASE_COLLAPSED_EXEMPT_PATTERNS):
        return False
    return any(p in lowered for p in PHASE_COLLAPSED_SLUG_PATTERNS)


QUALITY_RANK = {
    "Factory New": 5,
    "Minimal Wear": 4,
    "Field-Tested": 3,
    "Well-Worn": 2,
    "Battle-Scarred": 1,
    "FN": 5,
    "MW": 4,
    "FT": 3,
    "WW": 2,
    "BS": 1,
}

QUALITY_CANONICAL = {
    "Factory New": "Factory New",
    "FN": "Factory New",
    "Minimal Wear": "Minimal Wear",
    "MW": "Minimal Wear",
    "Field-Tested": "Field-Tested",
    "FT": "Field-Tested",
    "Well-Worn": "Well-Worn",
    "WW": "Well-Worn",
    "Battle-Scarred": "Battle-Scarred",
    "BS": "Battle-Scarred",
}


def parse_item_name(name: str) -> dict:
    """Parse a CS2 item name into structured identity fields.

    Returns a dict with keys: weapon, skin_name, quality, quality_rank,
    is_stattrak, is_souvenir, is_knife, is_glove, is_sticker, is_case,
    is_capsule, is_agent, is_music_kit, is_graffiti, is_charm, is_patch.
    """
    result = {
        "weapon": None,
        "skin_name": None,
        "quality": None,
        "quality_rank": 0,
        "is_stattrak": False,
        "is_souvenir": False,
        "is_knife": False,
        "is_glove": False,
        "is_sticker": False,
        "is_case": False,
        "is_capsule": False,
        "is_agent": False,
        "is_music_kit": False,
        "is_graffiti": False,
        "is_charm": False,
        "is_patch": False,
    }

    if not name or not isinstance(name, str):
        return result

    # StatTrak / Souvenir prefix detection
    if "StatTrak" in name or name.startswith("StatTrak\u2122"):
        result["is_stattrak"] = True
    if name.startswith("Souvenir"):
        result["is_souvenir"] = True

    # ★ prefix → knife or glove
    if name.startswith("\u2605"):
        is_glove = "Glove" in name or "Gloves" in name
        result["is_knife"] = not is_glove
        result["is_glove"] = is_glove

    # Category prefixes — use "in name" rather than startswith to handle
    # cases like "StatTrak™ Music Kit | ...", "Sticker Slab | ..." etc.
    if name.startswith("Sticker |") or name.startswith("Sticker Slab |"):
        result["is_sticker"] = True
    elif "|" in name and "Music Kit" in name and name.index("Music Kit") < name.index("|"):
        result["is_music_kit"] = True
    elif name.startswith("Sealed Graffiti |"):
        result["is_graffiti"] = True
    elif name.startswith("Agent |") or name.startswith("Special Agent"):
        result["is_agent"] = True
    elif name.startswith("Charm |"):
        result["is_charm"] = True
    elif "Patch" in name and ("Patch Pack" in name or name.startswith("Patch |")):
        result["is_patch"] = True

    # Case / Capsule detection (must check after category prefixes)
    if name.endswith("Case") and not result["is_patch"]:
        result["is_case"] = True
    if name.endswith("Capsule"):
        result["is_capsule"] = True

    # Quality extraction from parenthetical suffix
    quality_match = re.search(r"\(([^)]+)\)\s*$", name)
    if quality_match:
        raw_quality = quality_match.group(1).strip()
        result["quality"] = QUALITY_CANONICAL.get(raw_quality, raw_quality)
        result["quality_rank"] = QUALITY_RANK.get(raw_quality, 0)

    # Weapon & skin name extraction (only for skin-type items)
    if not any([
        result["is_sticker"], result["is_music_kit"],
        result["is_graffiti"], result["is_agent"], result["is_charm"],
    ]):
        clean = name
        if result["is_souvenir"]:
            clean = re.sub(r"^Souvenir\s+", "", clean)
        clean = re.sub(r"^StatTrak\u2122\s*", "", clean)
        clean = re.sub(r"^\u2605\s*", "", clean)

        if "|" in clean:
            parts = [p.strip() for p in clean.split("|")]
            if len(parts) >= 2:
                result["weapon"] = parts[0]
                skin_part = parts[1]
                skin_match = re.match(r"^(.+?)\s*\([^)]+\)\s*$", skin_part)
                if skin_match:
                    result["skin_name"] = skin_match.group(1).strip()
                else:
                    result["skin_name"] = skin_part

    return result
