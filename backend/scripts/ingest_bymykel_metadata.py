#!/usr/bin/env python3
"""Ingest ByMykel/CSGO-API item metadata into price-archive/item-metadata-bymykel.parquet.

`docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md` measured this
metadata bundle as the one acquisition candidate with a placebo-controlled positive
effect (+0.75 / +0.99 / +1.85pp at 7d/14d/30d, held-out >=$1 CV). The table it was
measured against was built in a scratchpad and never committed; this script is that
builder, made reproducible.

It writes a NEW file rather than extending `item-metadata.parquet`, which
`scripts/backfill_supply_metadata.py` owns and would clobber on its next run.

Nine columns, keyed on `item_slug` (== `market_hash_name`):

    item_age_first_sale_date  earliest crate first_sale_date, else collection
                              release_date. NOT an age -- see below.
    item_age_ambiguous        1 when the item sits in crates carrying more than
                              one distinct first_sale_date (knives and gloves,
                              which appear in every case that ever held them).
    rarity_meta_rank          models.steam_types.RARITY_RANK, so the scale
                              matches the `rarity_ordinal` already in the model.
    is_meta_stattrak          this slug is the StatTrak variant.
    is_meta_souvenir          this slug is the Souvenir variant.
    float_meta_min            skins.json min_float.
    float_meta_max            skins.json max_float.
    type_meta_crate_id        stable code for the item's earliest crate.
    type_meta_collection_id   stable code for the item's earliest collection.

`item_age_days` is deliberately NOT stored. It is (observation date - first sale
date), so it is a property of a row and not of an item; storing a "days since sale
as of today" column would freeze the calendar into a per-item constant. The
consumer derives it -- see `ab_test_item_metadata.py::_join_metadata`.

Ten of the twelve dumps carry `market_hash_name` directly and join with no parsing.
Only `skins.json` does not: it holds a base name plus a `wears` list, so its entries
are expanded forward into market hash names here. That direction is deliberate --
parsing the other way round means stripping wear and StatTrak off archive names, and
`models.item_parser.parse_item_name` gets that wrong for knives ("* StatTrak(tm)
Karambit | Fade" returns weapon="StatTrak(tm) Karambit").

Usage:
    python scripts/ingest_bymykel_metadata.py                  # fetch + build
    python scripts/ingest_bymykel_metadata.py --offline        # use cached dumps
    python scripts/ingest_bymykel_metadata.py --coverage-only  # report, no write
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
from models.steam_types import RARITY_KEYWORDS, RARITY_RANK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_bymykel_metadata")

BASE_URL = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en"
CACHE_DIR = Path(__file__).parent.parent / "runtime" / "bymykel"
PRICE_ARCHIVE = Path(__file__).parent.parent.parent / "price-archive"
OUTPUT_PARQUET = PRICE_ARCHIVE / "item-metadata-bymykel.parquet"
CODE_BOOK = PRICE_ARCHIVE / "item-metadata-bymykel-codes.json"

# skins.json first: it is the only dump needing name expansion, and the only one
# carrying float caps. The rest are keyed by market_hash_name and merely add rows.
DUMPS = (
    "skins",
    "crates",
    "collections",
    "stickers",
    "sticker_slabs",
    "graffiti",
    "collectibles",
    "highlights",
    "music_kits",
    "patches",
    "keychains",
    "agents",
)
# Dumps that are a straight market_hash_name -> attributes read.
NAME_KEYED_DUMPS = tuple(d for d in DUMPS if d not in ("skins", "collections"))

OUTPUT_COLUMNS = (
    "item_slug",
    "item_age_first_sale_date",
    "item_age_ambiguous",
    "rarity_meta_rank",
    # The categorical twin of the rank. Rank 3 collapses milspec/high_grade/
    # distinguished, so the rank alone cannot rebuild the rarity string that
    # `item-metadata.parquet` and the training subsample both key on.
    "rarity_meta",
    "is_meta_stattrak",
    "is_meta_souvenir",
    "float_meta_min",
    "float_meta_max",
    "type_meta_crate_id",
    "type_meta_collection_id",
)

# Ranks absent from RARITY_RANK, which was written against Steam's `type` strings
# and never saw these two. Contraband sits above Covert; Default is the unranked
# placeholder Valve puts on tools and gift packages.
EXTRA_RARITY_RANK = {"contraband": 7, "default": 0}

STATTRAK_PREFIX = "StatTrak™ "
SOUVENIR_PREFIX = "Souvenir "
STAR_PREFIX = "★ "


# --------------------------------------------------------------------------
# Parsing primitives
# --------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s*$")


def normalise_date(raw) -> date | None:
    """Parse ByMykel's date strings, which come in five shapes in one file.

    `crates.json` alone carries `2024-01-16`, `2013/12/17`, `2014/5/2`,
    `2014/12/5` and `2014/5/2` -- both separators, and neither month nor day
    reliably zero-padded. Anything that does not match is returned as None
    rather than guessed at.
    """
    if not raw or not isinstance(raw, str):
        return None
    m = _DATE_RE.match(raw)
    if not m:
        return None
    year, month, day = (int(g) for g in m.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def rarity_rank(rarity) -> int | None:
    """Map a ByMykel rarity object onto the repo's existing rank scale.

    Reuses `models.steam_types.RARITY_RANK` deliberately: the model already
    carries `rarity_ordinal` on that scale, and a second, differently-numbered
    rarity column would be silently incomparable to it.
    """
    if not isinstance(rarity, dict):
        return None
    name = (rarity.get("name") or "").lower()
    if not name:
        return None
    for keyword, key in RARITY_KEYWORDS:
        if keyword in name:
            return RARITY_RANK.get(key, 0)
    for keyword, rank in EXTRA_RARITY_RANK.items():
        if keyword in name:
            return rank
    return None


def rarity_token(rarity) -> str | None:
    """The ByMykel rarity name as one of the repo's rarity tokens.

    The rank alone cannot restore this: rank 3 collapses `milspec`,
    `high_grade` and `distinguished`, so a consumer holding only the rank
    cannot tell a Mil-Spec rifle from a High Grade sticker. `rarity_meta_rank`
    stays the numeric axis; this is the categorical one, and the two are kept
    in lockstep — both resolve, or neither does.
    """
    if not isinstance(rarity, dict):
        return None
    name = (rarity.get("name") or "").lower()
    if not name:
        return None
    for keyword, key in RARITY_KEYWORDS:
        if keyword in name:
            return key
    for keyword in EXTRA_RARITY_RANK:
        if keyword in name:
            return keyword
    return None


class CodeBook:
    """Dense, append-only integer codes for crate and collection ids.

    Both constraints matter and they pull against each other:

    * **Dense, from zero.** The A/B harness declares these columns categorical
      (`META_CATEGORICAL`), and LightGBM allocates over a categorical's value
      RANGE, not its cardinality. A sparse encoding is not merely wasteful --
      a 31-bit hash was the first thing tried here and it OOM-killed the
      treatment arm outright, which is what LightGBM's "consider renumbering to
      consecutive integers started from zero" warning is about.
    * **Stable across runs.** Enumerating a sorted id list is dense but
      renumbers every id after an insertion point the next time a crate ships,
      silently changing what an already-trained booster's splits mean.

    Persisting the assignment satisfies both: existing ids keep their code
    forever, and new ids append at the end.
    """

    def __init__(self, path: Path):
        self.path = path
        self.codes: dict[str, dict[str, int]] = {}
        if path.exists():
            self.codes = json.loads(path.read_text())

    def assign(self, namespace: str, identifiers) -> None:
        """Give every unseen id in `identifiers` the next free code."""
        known = self.codes.setdefault(namespace, {})
        next_code = max(known.values(), default=-1) + 1
        for identifier in sorted(set(identifiers) - set(known)):
            known[identifier] = next_code
            next_code += 1

    def get(self, namespace: str, identifier: str | None) -> int | None:
        if not identifier:
            return None
        return self.codes.get(namespace, {}).get(identifier)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.codes, indent=1, sort_keys=True))
        sizes = ", ".join(f"{ns} {len(v)}" for ns, v in sorted(self.codes.items()))
        logger.info(f"  code book:                {sizes} -> {self.path.name}")


def expand_skin_names(entry: dict) -> list[tuple[str, bool, bool]]:
    """Expand one skins.json entry into (market_hash_name, stattrak, souvenir).

    ByMykel gives a base name (`AK-47 | Redline`, or `* Bayonet | Doppler` with
    the star already attached for knives and gloves) plus the wears the skin
    actually ships in. Steam's market hash name is that base with the wear
    appended and the StatTrak / Souvenir prefix inserted AFTER the star:
    `* StatTrak(tm) Karambit | Fade (Factory New)`.

    A skin with no wears is a vanilla knife and has no wear suffix at all.
    """
    name = entry.get("name")
    if not name:
        return []

    star = ""
    base = name
    if base.startswith(STAR_PREFIX):
        star, base = STAR_PREFIX, base[len(STAR_PREFIX) :]

    variants = [("", False, False)]
    if entry.get("stattrak"):
        variants.append((STATTRAK_PREFIX, True, False))
    if entry.get("souvenir"):
        variants.append((SOUVENIR_PREFIX, False, True))

    wears = [w.get("name") for w in (entry.get("wears") or []) if isinstance(w, dict) and w.get("name")]
    suffixes = [f" ({w})" for w in wears] or [""]

    out = []
    for prefix, is_st, is_sv in variants:
        for suffix in suffixes:
            out.append((f"{star}{prefix}{base}{suffix}", is_st, is_sv))
    return out


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def fetch_dump(name: str, cache_dir: Path, offline: bool = False):
    """Read one dump, from cache when offline or when already downloaded."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.json"

    if offline:
        if not path.exists():
            raise SystemExit(f"--offline but {path} is missing; run once online")
        return json.loads(path.read_text())

    resp = requests.get(f"{BASE_URL}/{name}.json", timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    path.write_text(json.dumps(payload))
    logger.info(f"  {name:14s} {len(payload):>6,} entries ({len(resp.content) / 1e6:.1f} MB)")
    return payload


def fetch_all(cache_dir: Path, offline: bool = False) -> dict:
    logger.info(f"Fetching {len(DUMPS)} dumps from ByMykel/CSGO-API{' (offline, from cache)' if offline else ''}...")
    return {name: fetch_dump(name, cache_dir, offline) for name in DUMPS}


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_date_maps(dumps: dict) -> tuple[dict, dict]:
    """crate id -> first sale date, collection id -> release date.

    Only 261 of 481 crates carry `first_sale_date`. A further 163 sit inside a
    collection that carries a `release_date`, and a crate's contents went on
    sale when the crate did -- so the collection date is inherited one level
    down rather than left null. The item-level fallback cannot reach these:
    it fires only when the ITEM itself lists the collection, and items inside a
    crate frequently do not.
    """
    collection_dates = {}
    for entry in dumps["collections"]:
        d = normalise_date(entry.get("release_date"))
        if d is not None:
            collection_dates[entry["id"]] = d

    crate_dates = {}
    for entry in dumps["crates"]:
        d = normalise_date(entry.get("first_sale_date"))
        if d is not None:
            crate_dates[entry["id"]] = d
    n_direct = len(crate_dates)

    for entry in dumps["collections"]:
        released = collection_dates.get(entry["id"])
        if released is None:
            continue
        for crate in entry.get("crates") or []:
            crate_id = crate.get("id") if isinstance(crate, dict) else None
            if crate_id and crate_id not in crate_dates:
                crate_dates[crate_id] = released

    logger.info(
        f"  crate first_sale_date:    {n_direct:>5,}/"
        f"{len(dumps['crates']):,} direct, "
        f"{len(crate_dates) - n_direct:,} inherited from a collection"
    )
    logger.info(f"  collection release_date:  {len(collection_dates):>5,}/{len(dumps['collections']):,}")
    return crate_dates, collection_dates


def build_crate_index(dumps: dict) -> dict[str, list[str]]:
    """ByMykel entity id -> the crates that contain it, from crates.json.

    `crates.json` lists its contents in `contains` and `contains_rare`, and that
    reverse index is more complete than the forward `crates` field on each item:
    478 of 481 crates carry it. Both directions are unioned in `build_records`,
    because neither alone covers everything.
    """
    index: dict[str, list[str]] = {}
    for crate in dumps["crates"]:
        crate_id = crate.get("id")
        if not crate_id:
            continue
        for key in ("contains", "contains_rare"):
            for item in crate.get(key) or []:
                item_id = item.get("id") if isinstance(item, dict) else None
                if item_id:
                    index.setdefault(item_id, []).append(crate_id)
    logger.info(f"  reverse crate index:      {len(index):>5,} entities")
    return index


def _earliest(ids: list[str], date_map: dict) -> tuple[str | None, date | None, int]:
    """Pick the earliest-dated member and report how many distinct dates existed.

    Undated members still win the id slot if nothing is dated, so an item in a
    crate with no recorded sale date keeps its crate identity. Ties and the
    all-undated case break on the id, which keeps the choice deterministic
    across runs.
    """
    if not ids:
        return None, None, 0
    dated = [(date_map[i], i) for i in ids if i in date_map]
    n_distinct = len({d for d, _ in dated})
    if not dated:
        return min(ids), None, 0
    best_date, best_id = min(dated)
    return best_id, best_date, n_distinct


def build_records(dumps: dict, codebook: CodeBook) -> dict[str, dict]:
    """One record per market hash name, merged across all twelve dumps."""
    crate_dates, collection_dates = build_date_maps(dumps)
    crate_index = build_crate_index(dumps)
    codebook.assign("crate", (c["id"] for c in dumps["crates"] if c.get("id")))
    codebook.assign("collection", (c["id"] for c in dumps["collections"] if c.get("id")))
    records: dict[str, dict] = {}

    def upsert(
        slug,
        rarity,
        crates,
        collections,
        *,
        entity_id=None,
        own_date=None,
        stattrak=False,
        souvenir=False,
        float_min=None,
        float_max=None,
    ):
        crate_ids = [c["id"] for c in (crates or []) if isinstance(c, dict) and c.get("id")]
        crate_ids = sorted(set(crate_ids) | set(crate_index.get(entity_id, [])))
        coll_ids = [c["id"] for c in (collections or []) if isinstance(c, dict) and c.get("id")]

        crate_id, crate_date, n_crate_dates = _earliest(crate_ids, crate_dates)
        coll_id, coll_date, _ = _earliest(coll_ids, collection_dates)

        # A crate is itself a tradable item, and its own `first_sale_date` is
        # its age directly. That beats any date inferred from what it contains.
        first_sale = own_date
        if first_sale is None:
            first_sale = crate_date if crate_date is not None else coll_date

        rec = {
            "item_slug": slug,
            "item_age_first_sale_date": first_sale,
            # Ambiguity is a property of the crate tier only. Knives and gloves
            # sit in every case that ever contained them, so their candidate
            # dates span years; a single-crate item has exactly one and is not
            # ambiguous even when a collection date also exists.
            "item_age_ambiguous": 1 if n_crate_dates > 1 else 0,
            "rarity_meta_rank": rarity_rank(rarity),
            "rarity_meta": rarity_token(rarity),
            "is_meta_stattrak": int(bool(stattrak)),
            "is_meta_souvenir": int(bool(souvenir)),
            "float_meta_min": float_min,
            "float_meta_max": float_max,
            "type_meta_crate_id": codebook.get("crate", crate_id),
            "type_meta_collection_id": codebook.get("collection", coll_id),
        }
        # skins.json is read first and is the only source of float caps, so a
        # later name-keyed dump must not overwrite a richer skins record.
        existing = records.get(slug)
        if existing is None:
            records[slug] = rec
        else:
            for key, value in rec.items():
                if existing.get(key) is None and value is not None:
                    existing[key] = value

    for entry in dumps["skins"]:
        for slug, is_st, is_sv in expand_skin_names(entry):
            upsert(
                slug,
                entry.get("rarity"),
                entry.get("crates"),
                entry.get("collections"),
                entity_id=entry.get("id"),
                stattrak=is_st,
                souvenir=is_sv,
                float_min=entry.get("min_float"),
                float_max=entry.get("max_float"),
            )

    for dump_name in NAME_KEYED_DUMPS:
        for entry in dumps[dump_name]:
            slug = entry.get("market_hash_name")
            if not slug:
                continue
            upsert(
                slug,
                entry.get("rarity"),
                entry.get("crates"),
                entry.get("collections"),
                entity_id=entry.get("id"),
                own_date=normalise_date(entry.get("first_sale_date")),
            )

    return records


def build_frame(dumps: dict, codebook: CodeBook) -> pd.DataFrame:
    records = build_records(dumps, codebook)
    df = pd.DataFrame.from_records(list(records.values()), columns=list(OUTPUT_COLUMNS))
    df["item_age_first_sale_date"] = pd.to_datetime(df["item_age_first_sale_date"], errors="coerce")
    for col in ("item_age_ambiguous", "is_meta_stattrak", "is_meta_souvenir"):
        df[col] = df[col].astype("int8")
    for col in ("rarity_meta_rank", "type_meta_crate_id", "type_meta_collection_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("float_meta_min", "float_meta_max"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("item_slug").reset_index(drop=True)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def archive_slugs(min_price: float | None = None, min_days: int | None = None, before: str | None = None) -> set[str]:
    """Distinct item_slug in the local price archive, optionally cohort-filtered.

    The cohorts mirror `ab_test_item_metadata.py`: the deep >=$1 universe is
    median mean_price >= $1 with >=180 distinct days and first seen before 2026,
    on the `aggregator_sync` series; the served cohort is the plain >=$1 median.
    """
    import duckdb

    con = duckdb.connect()
    files = sorted(str(p) for p in PRICE_ARCHIVE.glob("prices-*.parquet"))
    if not files:
        raise SystemExit(f"No prices-*.parquet under {PRICE_ARCHIVE}")

    parts = []
    for path in files:
        cols = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
        where = " WHERE source = 'aggregator_sync'" if "source" in cols else ""
        parts.append(f"SELECT item_slug, day, mean_price FROM read_parquet('{path}'){where}")
    union = " UNION ALL BY NAME ".join(parts)

    having = []
    if min_price is not None:
        having.append(f"median(mean_price) >= {min_price}")
    if min_days is not None:
        having.append(f"count(DISTINCT day) >= {min_days}")
    if before is not None:
        having.append(f"min(day) < DATE '{before}'")
    clause = f" HAVING {' AND '.join(having)}" if having else ""

    rows = con.sql(f"SELECT item_slug FROM ({union}) GROUP BY item_slug{clause}").fetchall()
    return {r[0] for r in rows}


def report_coverage(df: pd.DataFrame, label: str, slugs: set[str]) -> None:
    """Print per-column coverage of `df` over `slugs`.

    The published figures this reproduces, on the 870-item deep >=$1 universe:
    rarity 99.5%, item age 87.1%, crate 69.1%, collection 81.0%, float caps and
    the StatTrak/Souvenir flags 73.7%
    (`docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md`).
    """
    matched = df[df["item_slug"].isin(slugs)]
    n = len(slugs)
    logger.info(f"{label}: {n:,} slugs, {len(matched):,} matched ({100.0 * len(matched) / n:.1f}%)")
    for col in OUTPUT_COLUMNS[1:]:
        if col in ("is_meta_stattrak", "is_meta_souvenir", "item_age_ambiguous"):
            # Zero is a real value on these, so "populated" means the row exists
            # at all, not that the flag is set. Report the set rate separately.
            populated = len(matched)
            extra = f"  (set on {int(matched[col].sum()):,})"
        else:
            populated = int(matched[col].notna().sum())
            extra = ""
        logger.info(f"    {col:26s} {populated:>7,}/{n:,} ({100.0 * populated / n:5.1f}%){extra}")


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest ByMykel/CSGO-API item metadata")
    parser.add_argument("--offline", action="store_true", help="Use cached dumps instead of refetching")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--out", default=str(OUTPUT_PARQUET))
    parser.add_argument(
        "--code-book",
        default=str(CODE_BOOK),
        help="Append-only crate/collection code assignment. "
        "Delete it only alongside a retrain: existing "
        "codes are what a trained booster's splits mean.",
    )
    parser.add_argument("--coverage-only", action="store_true", help="Report coverage and exit without writing")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip the archive join (no DuckDB read)")
    args = parser.parse_args()

    dumps = fetch_all(Path(args.cache_dir), offline=args.offline)
    logger.info("Building metadata table...")
    codebook = CodeBook(Path(args.code_book))
    df = build_frame(dumps, codebook)
    logger.info(f"  {len(df):,} market hash names, {len(df.columns)} columns")

    if not args.skip_coverage:
        logger.info("Coverage against the local price archive:")
        report_coverage(df, "  deep >=$1 universe", archive_slugs(min_price=1.0, min_days=180, before="2026-01-01"))
        report_coverage(df, "  served >=$1 cohort", archive_slugs(min_price=1.0))
        report_coverage(df, "  full archive", archive_slugs())

    if args.coverage_only:
        logger.info("--coverage-only: nothing written")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    codebook.save()
    df.to_parquet(out, index=False)
    logger.info(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
