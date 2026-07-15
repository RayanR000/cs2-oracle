#!/usr/bin/env python3
"""
Backfill supply-side metadata (rarity, weapon_type) from the Steam Market catalog.

Reads market_catalog.db, parses each item's Steam `type` field into rarity
and weapon_type, then writes the results to:
  1. price-archive/item-metadata.parquet  (single static Parquet file)
  2. The main DB items table              (if --write-db is passed)

Usage:
    python scripts/backfill_supply_metadata.py                          # Parquet only
    python scripts/backfill_supply_metadata.py --write-db               # Parquet + DB
    python scripts/backfill_supply_metadata.py --parquet-only           # Parquet only (default)
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import sqlite3
import duckdb
from models.steam_types import parse_steam_type
from models.item_parser import parse_item_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill_supply_metadata")

CATALOG_DB = Path(__file__).parent.parent / "runtime" / "market_catalog.db"
PRICE_ARCHIVE = Path(__file__).parent.parent.parent / "price-archive"
OUTPUT_PARQUET = PRICE_ARCHIVE / "item-metadata.parquet"


# Weapon type inference from item name (fallback for items without Steam type)
WEAPON_TYPE_FROM_PARSER = {
    "is_sticker": "sticker",
    "is_case": "case",
    "is_capsule": "case",
    "is_knife": "knife",
    "is_glove": "glove",
    "is_agent": "agent",
    "is_music_kit": "musickit",
    "is_graffiti": "graffiti",
    "is_charm": "charm",
    "is_patch": "patch",
}

# Weapon → weapon_type mapping (display names)
WEAPON_TO_TYPE = {
    "AK-47": "rifle", "M4A4": "rifle", "M4A1-S": "rifle",
    "AUG": "rifle", "SG 553": "rifle", "FAMAS": "rifle",
    "Galil AR": "rifle", "AWP": "sniper", "SSG 08": "sniper",
    "SCAR-20": "sniper", "G3SG1": "sniper",
    "MAC-10": "smg", "MP9": "smg", "MP7": "smg", "MP5-SD": "smg",
    "UMP-45": "smg", "P90": "smg", "PP-Bizon": "smg",
    "Nova": "shotgun", "XM1014": "shotgun", "MAG-7": "shotgun",
    "Sawed-Off": "shotgun", "M249": "machinegun", "Negev": "machinegun",
    "Desert Eagle": "pistol", "R8 Revolver": "pistol",
    "USP-S": "pistol", "P2000": "pistol", "Glock-18": "pistol",
    "P250": "pistol", "Five-SeveN": "pistol", "CZ75-Auto": "pistol",
    "Tec-9": "pistol", "Dual Berettas": "pistol",
    "Bayonet": "knife", "Flip Knife": "knife", "Gut Knife": "knife",
    "Karambit": "knife", "M9 Bayonet": "knife", "Huntsman Knife": "knife",
    "Falchion Knife": "knife", "Bowie Knife": "knife",
    "Butterfly Knife": "knife", "Shadow Daggers": "knife",
    "Navaja Knife": "knife", "Stiletto Knife": "knife",
    "Talon Knife": "knife", "Ursus Knife": "knife",
    "Classic Knife": "knife", "Paracord Knife": "knife",
    "Survival Knife": "knife", "Nomad Knife": "knife",
    "Skeleton Knife": "knife", "Kukri Knife": "knife",
    "Zeus x27": "pistol", "Zeus X27": "pistol",
    "Bloodhound Gloves": "glove", "Driver Gloves": "glove",
    "Hand Wraps": "glove", "Moto Gloves": "glove",
    "Specialist Gloves": "glove", "Sport Gloves": "glove",
    "Broken Fang Gloves": "glove",
}

# Slug-form weapon map (lowercase, hyphenated)
WEAPON_SLUG_TO_TYPE = {
    "ak-47": "rifle", "m4a4": "rifle", "m4a1-s": "rifle", "m4a1s": "rifle",
    "aug": "rifle", "sg-553": "rifle", "sg 553": "rifle",
    "famas": "rifle", "galil-ar": "rifle", "galil ar": "rifle",
    "awp": "sniper", "ssg-08": "sniper", "ssg 08": "sniper",
    "scar-20": "sniper", "g3sg1": "sniper",
    "mac-10": "smg", "mp9": "smg", "mp7": "smg", "mp5-sd": "smg",
    "ump-45": "smg", "p90": "smg", "pp-bizon": "smg",
    "nova": "shotgun", "xm1014": "shotgun", "mag-7": "shotgun",
    "sawed-off": "shotgun",
    "m249": "machinegun", "negev": "machinegun",
    "desert-eagle": "pistol", "r8-revolver": "pistol",
    "usp-s": "pistol", "p2000": "pistol", "glock-18": "pistol",
    "p250": "pistol", "five-seven": "pistol", "cz75-auto": "pistol",
    "tec-9": "pistol", "dual-berettas": "pistol",
    "bayonet": "knife", "flip-knife": "knife", "gut-knife": "knife",
    "karambit": "knife", "m9-bayonet": "knife", "huntsman-knife": "knife",
    "falchion-knife": "knife", "bowie-knife": "knife",
    "butterfly-knife": "knife", "shadow-daggers": "knife",
    "navaja-knife": "knife", "stiletto-knife": "knife",
    "talon-knife": "knife", "ursus-knife": "knife",
    "classic-knife": "knife", "paracord-knife": "knife",
    "survival-knife": "knife", "nomad-knife": "knife",
    "skeleton-knife": "knife", "kukri-knife": "knife",
    "bloodhound-gloves": "glove", "driver-gloves": "glove",
    "hand-wraps": "glove", "moto-gloves": "glove",
    "specialist-gloves": "glove", "sport-gloves": "glove",
    "broken-fang-gloves": "glove",
    "zeus-x27": "pistol", "zeus x27": "pistol", "zeus-x27": "pistol",
}

# Agent NPC names
AGENT_SLUGS = {
    "sir-bloody", "the-elite", "special-agent", "bio-haz-specialist",
    "dragomir", "1st-lieutenant", "doctor-romanov", "two-times",
    "medium-rare", "professor", "colonel", "commander", "operator",
    "soldier", "slingshot", "arno",
}

# Sticker/tournament prefixes (items named by event, not by "sticker-")
STICKER_EVENT_PREFIXES = {
    "berlin-2019", "stockholm-2021", "antwerp-2022", "rio-2022",
    "paris-2023", "copenhagen-2024", "shanghai-2024", "austin-2025",
    "budapest-2025", "colonge-2026", "london-2018", "boston-2018",
    "katowice-2019", "dreamhack-",
}

# Quality suffixes that indicate a skin item
WEAR_SUFFIXES = [
    "factory-new", "minimal-wear", "field-tested", "well-worn", "battle-scarred",
    "factory new", "minimal wear", "field tested", "well worn", "battle scarred",
]


def infer_weapon_type_from_name(name: str) -> str | None:
    """Infer weapon_type from item name — handles both display names and slugs."""

    # Try parser first (works for display names with | separator)
    parsed = parse_item_name(name)
    for flag, wtype in WEAPON_TYPE_FROM_PARSER.items():
        if parsed.get(flag):
            return wtype
    weapon = parsed.get("weapon")
    if weapon:
        result = WEAPON_TO_TYPE.get(weapon)
        if result:
            return result

    # Slug-based inference
    slug = name.lower().strip()
    slug = slug.replace(" | ", "-").replace(" | ", "-").replace("|", "-")

    # Remove Stattrak/Souvenir prefix
    orig_slug = slug
    if slug.startswith("stattrak-") or slug.startswith("stattrak™-"):
        slug = slug[len("stattrak-"):] if slug.startswith("stattrak-") else slug
        slug = slug[len("stattrak™-"):] if slug.startswith("stattrak™-") else slug
    if slug.startswith("souvenir-"):
        slug = slug[len("souvenir-"):]

    # Remove ★- prefix
    if slug.startswith("★-") or slug.startswith("star-"):
        slug = slug[2:] if slug.startswith("★-") else slug[5:]

    # Pattern: sealed-graffiti-* → graffiti
    if slug.startswith("sealed-graffiti-"):
        return "graffiti"

    # Pattern: sticker-* → sticker
    if slug.startswith("sticker-"):
        return "sticker"

    # Pattern: music-kit-* → musickit
    if slug.startswith("music-kit-"):
        return "musickit"

    # Pattern: charm-* → charm
    if slug.startswith("charm-"):
        return "charm"

    # Pattern: patch-* → patch
    if slug.startswith("patch-"):
        return "patch"

    # Pattern: agent-* → agent
    if slug.startswith("agent-"):
        return "agent"

    # Pattern: sir-bloody-*, the-elite-*, special-agent-*, etc. → agent
    for agent_prefix in AGENT_SLUGS:
        if slug.startswith(agent_prefix):
            return "agent"

    # Pattern: berlin-2019-*, stockholm-2021-*, etc. → sticker
    for event_prefix in STICKER_EVENT_PREFIXES:
        if slug.startswith(event_prefix):
            # Souvenir packages are still "case" type but stickers are "sticker"
            if "souvenir-package" in slug or "souvenir package" in slug:
                return "case"
            return "sticker"

    # Pattern: steam_sealed_graffiti_|_* → graffiti (underscore-hyphen hybrid)
    if "sealed_graffiti" in slug or "sealed-graffiti" in slug:
        return "graffiti"

    # Pattern: steam_sticker_|_* → sticker
    if slug.startswith("steam_sticker"):
        return "sticker"

    # Pattern: *-souvenir-package, *-patch-pack → case
    if slug.endswith("-souvenir-package") or slug.endswith("-patch-pack") or slug.endswith("souvenir package"):
        return "case"

    # Pattern: *-pin → collectible
    if slug.endswith("-pin") or slug.endswith("pin"):
        return "collectible"

    # Pattern: *-package → case
    if slug.endswith("-package") or slug.endswith("package"):
        return "case"

    # Pattern: items with | (pipe) that look like agents (Specialist | SWAT)
    if "specialist" in slug or "soldier" in slug.split("-"):
        return "agent"

    # Pattern: weapon-case-2, weapon-case, etc (display name style items)
    if "weapon case" in slug.lower():
        return "case"

    # Pattern: contains "gloves" or "glove" → glove
    if "glove" in slug:
        return "glove"

    # Pattern: *capsule* → case
    if "capsule" in slug:
        return "case"

    # Pattern: *-sticker-capsule, *sticker capsule* → case
    if "sticker capsule" in slug:
        return "case"

    # Pattern: *-case or *-capsule → case
    if slug.endswith("-case") or slug.endswith("-capsule") or slug.endswith(" case") or slug.endswith(" capsule"):
        return "case"

    # Pattern: *-key → key
    if slug.endswith("-key"):
        return "key"

    # Pattern: *-pass → pass
    if slug.endswith("-pass"):
        return "pass"

    # Pattern: *-tool → tool
    if slug.endswith("-tool"):
        return "tool"

    # Pattern: *-tag → tag
    if slug.endswith("-tag"):
        return "tag"

    # Pattern: *-gift → gift
    if slug.endswith("-gift"):
        return "gift"

    # Skin items: extract potential weapon slug (first part before quality suffix or last part)
    # Common patterns:
    #   weapon-skin-quality → e.g., "ak-47-asiimov-minimal-wear"
    #   weapon-skin → e.g., "ak-47-asiimov"
    # Remove wear suffix first
    cleaned = slug
    for suffix in WEAR_SUFFIXES:
        if cleaned.endswith("-" + suffix):
            cleaned = cleaned[:-(len(suffix) + 1)]
            break
        if cleaned.endswith(" " + suffix):
            cleaned = cleaned[:-(len(suffix) + 1)]
            break

    # Try weapon slug match on the remaining text
    # The weapon slug is usually at the start
    for weapon_slug, wtype in WEAPON_SLUG_TO_TYPE.items():
        if cleaned.startswith(weapon_slug + "-") or cleaned == weapon_slug:
            return wtype

    # Fallback: try on the original slug too (before wear stripping)
    for weapon_slug, wtype in WEAPON_SLUG_TO_TYPE.items():
        if slug.startswith(weapon_slug + "-") or slug == weapon_slug:
            return wtype

    return None


def build_metadata() -> pd.DataFrame:
    """Build metadata DataFrame from market_catalog.db + name-based fallback."""

    # ── 1. Load catalog data ──
    if not CATALOG_DB.exists():
        logger.warning(f"Catalog DB not found at {CATALOG_DB}; building from name inference only")
        cat_rows = []
    else:
        cat = sqlite3.connect(str(CATALOG_DB))
        try:
            cat_rows = cat.execute(
                "SELECT hash_name, name, type FROM market_items"
            ).fetchall()
            logger.info(f"Loaded {len(cat_rows):,} items from market_catalog.db")
        finally:
            cat.close()

    # ── 2. Build name → metadata mapping ──
    name_to_meta: dict[str, dict] = {}  # name -> {rarity, weapon_type, rarity_rank}

    for hash_name, display_name, steam_type in cat_rows:
        parsed = parse_steam_type(steam_type)
        name_to_meta[display_name] = {
            "rarity": parsed["rarity"],
            "rarity_rank": parsed["rarity_rank"],
            "weapon_type": parsed["weapon_type"],
        }

        # Also store by hash_name
        if hash_name != display_name:
            name_to_meta[hash_name] = name_to_meta[display_name]

    logger.info(f"Built metadata for {len(name_to_meta):,} catalog entries")

    # ── 3. Get all unique item_slugs from price Parquet files ──
    con = duckdb.connect()
    try:
        all_slugs = con.sql("""
            SELECT DISTINCT item_slug
            FROM read_parquet('{}')
            ORDER BY item_slug
        """.format(PRICE_ARCHIVE / "prices-*.parquet")).fetchall()
        all_slugs = [r[0] for r in all_slugs]
        logger.info(f"Found {len(all_slugs):,} unique items in price Parquet archive")
    finally:
        con.close()

    # ── 4. Build the metadata DF ──
    # We need to map item_slug -> name -> metadata
    # The DB items table maps item_id (= item_slug) to name
    # For slugs not in the DB, we try to reconstruct the name

    # Try to get the name mapping from the local DB
    db_name_map: dict[str, str] = {}
    local_db = Path(__file__).parent.parent / "cs2_market.db"
    if local_db.exists():
        conn = sqlite3.connect(str(local_db))
        try:
            rows = conn.execute("SELECT item_id, name FROM items").fetchall()
            for item_id, name in rows:
                db_name_map[item_id] = name
            logger.info(f"Loaded {len(db_name_map):,} name mappings from local DB")
        finally:
            conn.close()

    records = []
    from_source = {"catalog": 0, "inferred": 0, "missing": 0}

    for slug in all_slugs:
        rarity = None
        rarity_rank = 0
        weapon_type = None

        # Try direct slug match in catalog metadata
        if slug in name_to_meta:
            meta = name_to_meta[slug]
            rarity = meta["rarity"]
            rarity_rank = meta["rarity_rank"]
            weapon_type = meta["weapon_type"]

        # Augment with inference if still missing weapon_type
        if weapon_type is None:
            inferred = infer_weapon_type_from_name(slug)
            if inferred:
                weapon_type = inferred
                from_source["inferred"] += 1
            else:
                from_source["missing"] += 1
        else:
            from_source["catalog"] += 1

        records.append({
            "item_slug": slug,
            "rarity": rarity,
            "rarity_rank": rarity_rank,
            "weapon_type": weapon_type,
        })

    df = pd.DataFrame(records)
    logger.info(f"Metadata built: {from_source}")
    return df


def write_parquet(df: pd.DataFrame):
    PRICE_ARCHIVE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info(f"Wrote {len(df):,} rows to {OUTPUT_PARQUET}")


def write_db(df: pd.DataFrame):
    """Update rarity, weapon_type in the main DB items table."""
    try:
        from database import SessionLocal, Item, engine
    except ImportError:
        logger.error("Could not import database module; DB write skipped")
        return

    # Ensure new columns exist (idempotent — skips if present)
    try:
        from sqlalchemy import text as sa_text
        for col, col_type in [("rarity", "VARCHAR(50)"), ("rarity_rank", "INTEGER"), ("weapon_type", "VARCHAR(50)")]:
            try:
                with engine.connect() as conn:
                    conn.execute(sa_text(f"ALTER TABLE items ADD COLUMN {col} {col_type}"))
                    conn.commit()
            except Exception:
                pass  # Column already exists
    except Exception:
        pass

    db = SessionLocal()
    try:
        updated = 0
        for _, row in df.iterrows():
            item = db.query(Item).filter(Item.item_id == row["item_slug"]).first()
            if item is None:
                continue
            changed = False
            if row["rarity"] is not None and (item.rarity is None or item.rarity != row["rarity"]):
                item.rarity = row["rarity"]
                changed = True
            if row["rarity_rank"] and (item.rarity_rank is None or item.rarity_rank != row["rarity_rank"]):
                item.rarity_rank = int(row["rarity_rank"])
                changed = True
            if row["weapon_type"] is not None and (item.weapon_type is None or item.weapon_type != row["weapon_type"]):
                item.weapon_type = row["weapon_type"]
                changed = True
            if changed:
                updated += 1

        db.commit()
        logger.info(f"Updated {updated} items in DB")
    except Exception as e:
        db.rollback()
        logger.error(f"DB update failed: {e}")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill supply-side metadata")
    parser.add_argument("--write-db", action="store_true", help="Also update the main DB")
    parser.add_argument("--parquet-only", action="store_true", help="Write to Parquet only (default)")
    args = parser.parse_args()

    df = build_metadata()
    write_parquet(df)

    if args.write_db:
        write_db(df)
    else:
        logger.info("Skipping DB update (pass --write-db to enable)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
