"""One way to read `price-archive/prices-*.parquet`.

The price files do not share a schema. `prices-2013..2025` predate the `source`
column, `prices-2026-03/04` carry real `min_price`/`max_price` intraday range,
`prices-2026-08` orders its columns differently from the months before it, and
`day` is `TIMESTAMP` in the yearly files against `TIMESTAMP_NS` in the monthly
ones.

A plain glob read hides all of that instead of failing on it::

    DESCRIBE SELECT * FROM read_parquet('prices-*.parquet')
    -- item_slug, day, mean_price, volume        <- no source, no error

DuckDB narrows the whole glob to the schema of the first file it reads, which
alphabetically is `prices-2013.parquet`. Values stay correct; `source`,
`min_price` and `max_price` simply disappear, so a query that groups or filters
by source returns a plausible answer computed over a column that was never
there. Three call sites independently grew the same per-file DESCRIBE + `NULL AS
source` + `UNION ALL BY NAME` workaround to get around it. This module is that
workaround, written once.

:func:`prices_relation` projects an explicit column list and NULLs whatever the
archive lacks, so it is correct both before and after
`scripts/normalize_price_schema.py` has run — which matters because the
canonical archive is the `cs2-oracle-data` repo and only CI can migrate it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent.parent / "price-archive"

PRICE_GLOB = "prices-*.parquet"

#: Column order every prices file should be written in. `append_to_parquet.py`
#: emits exactly this for a new month, and `normalize_price_schema.py` rewrites
#: the older files into it.
CANONICAL_PRICE_COLUMNS: tuple[str, ...] = (
    "item_slug", "day", "source", "mean_price", "volume", "ingested_at",
)

#: `ingested_at` is when the row ARRIVED, as distinct from `day`, which is what
#: it describes. Added 2026-08-08, and **NULL for every row written before that
#: date** — the archive had no arrival timestamp anywhere and one cannot be
#: reconstructed backwards, so the column starts empty and fills forward.
#:
#: It exists for the embargo. A purge computed from `day` assumes a row dated
#: `d` was knowable on `d`; a backfill writer violates that by definition, and
#: the archive has carried backfilled series under ordinary `day` values for
#: 13 years. Until enough of this column accumulates, an embargo can only be
#: derived from `day` and is a lower bound on the true one.
#:
#: First arrival wins on a re-append: a row rewritten with a corrected price
#: keeps the timestamp of the day it first became knowable. A row that predates
#: the column reads NULL and is stamped by whichever append touches it next,
#: which dates it LATER than the truth — conservative in the safe direction for
#: a leakage filter, and the reason a NULL must never be read as "arrived at
#: `day`".

#: Kept only where they hold real intraday range (see `compact_price_archive.py`),
#: so they trail the canonical columns rather than sitting among them.
RANGE_PRICE_COLUMNS: tuple[str, ...] = ("min_price", "max_price")

#: The type a missing column is NULLed to, so both sides of a union agree.
#: Public because `normalize_price_schema.py` materialises the same columns
#: into the files themselves and has to NULL them to the same types.
COLUMN_TYPES: dict[str, str] = {
    "item_slug": "VARCHAR",
    "day": "DATE",
    "source": "VARCHAR",
    "mean_price": "DOUBLE",
    "volume": "BIGINT",
    "ingested_at": "TIMESTAMP",
    "min_price": "DOUBLE",
    "max_price": "DOUBLE",
}


def canonical_order(present: Iterable[str]) -> list[str]:
    """Order the columns a frame already has: canonical, then range, then rest.

    Only reorders — it never invents a column. `append_to_parquet.py` uses this
    so a freshly created month lands in the same order the migration rewrote
    the older files into, and `normalize_price_schema.py` builds its target
    list on top of it.
    """
    present = list(present)
    head = [c for c in CANONICAL_PRICE_COLUMNS if c in present]
    rng = [c for c in RANGE_PRICE_COLUMNS if c in present]
    placed = set(head) | set(rng)
    return head + rng + [c for c in present if c not in placed]


def resolve_archive_dir(archive_dir: Optional[Path] = None) -> Path:
    """The archive directory to read, defaulting to the repo-root copy.

    Every caller used to spell this as its own
    ``Path(__file__).parent.parent.parent / "price-archive"``, and two scripts
    defaulted to a CWD-relative ``../price-archive`` that resolves somewhere
    different depending on where they were invoked from.
    """
    return Path(archive_dir) if archive_dir is not None else ARCHIVE_ROOT


def price_files(archive_dir: Optional[Path] = None) -> list[Path]:
    """Every `prices-*.parquet` in the archive, sorted.

    Raises FileNotFoundError when the directory is missing or holds none.
    An absent archive must never read as "no price data", which is the shape
    of a green run that scored nothing.
    """
    directory = resolve_archive_dir(archive_dir)
    if not directory.exists():
        raise FileNotFoundError(f"price archive not found at {directory}")
    files = sorted(directory.glob(PRICE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"price archive at {directory} contains no {PRICE_GLOB}")
    return files


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _file_list_sql(files: Sequence[Path]) -> str:
    return "[" + ", ".join(_quote(p) for p in files) + "]"


def present_columns(con, archive_dir: Optional[Path] = None) -> set[str]:
    """The union of column names across every prices file."""
    files = price_files(archive_dir)
    rows = con.sql(
        f"DESCRIBE SELECT * FROM read_parquet({_file_list_sql(files)}, "
        f"union_by_name = true)"
    ).fetchall()
    return {r[0] for r in rows}


def prices_relation(
    con,
    archive_dir: Optional[Path] = None,
    columns: Optional[Iterable[str]] = None,
    where: Optional[str] = None,
) -> str:
    """A SQL table expression over the whole price archive.

    *columns* defaults to :data:`CANONICAL_PRICE_COLUMNS`. Any requested column
    no file carries comes back as a typed NULL rather than raising, so a caller
    can ask for `source` against a pre-2026-only archive (which the test
    fixtures and a freshly bootstrapped checkout both are).

    *where* is applied **on top of** the projection, so it may reference only
    columns named in *columns* — and it sees a NULLed-in column as NULL rather
    than failing on a name no file carries. `source IS NULL` therefore means
    "the pre-2026 series" whether or not the archive has been migrated yet.

    Interpolate the result straight into a query::

        rel = prices_relation(con, columns=["item_slug", "day", "mean_price"])
        con.sql(f"SELECT count(*) FROM {rel} WHERE day >= DATE '2026-01-01'")
    """
    files = price_files(archive_dir)
    wanted = list(columns) if columns is not None else list(CANONICAL_PRICE_COLUMNS)

    unknown = [c for c in wanted if c not in COLUMN_TYPES]
    if unknown:
        raise ValueError(
            f"unknown price column(s) {unknown}; known: {sorted(COLUMN_TYPES)}")

    present = present_columns(con, archive_dir)
    projection = ", ".join(
        # CAST rather than a bare reference: `day` is TIMESTAMP in the yearly
        # files and TIMESTAMP_NS in the monthly ones, and callers compare it
        # against dates. Normalising here means they all see one type whether
        # or not the archive has been migrated yet.
        (f"CAST({c} AS {COLUMN_TYPES[c]}) AS {c}" if c in present
         else f"NULL::{COLUMN_TYPES[c]} AS {c}")
        for c in wanted
    )

    sql = (f"SELECT {projection} FROM read_parquet({_file_list_sql(files)}, "
           f"union_by_name = true)")
    if where:
        sql = f"SELECT * FROM ({sql}) WHERE {where}"
    return f"({sql})"
