"""Detect a read-only Postgres and report it loudly.

Prod Supabase is on the free tier and measured 986 MB against its 500 MB cap on
2026-10-11. Past the cap Supabase can put the database in read-only mode, where
every INSERT/UPDATE fails with SQLSTATE 25006. The daily chain must still land its
Parquet output when that happens: a forecast that is never written anywhere
cannot be recreated. Callers catch the error with `is_read_only_error`, skip the
Postgres write, and call `report_read_only` — never a silent skip.

docs/specs/2026-10-11-postgres-to-parquet-history-design.md.
"""

from __future__ import annotations

import logging

READ_ONLY_SQLSTATE = "25006"  # read_only_sql_transaction

logger = logging.getLogger(__name__)


def is_read_only_error(exc: BaseException | None) -> bool:
    """True when `exc`, or anything in its cause chain, is a read-only rejection."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "pgcode", None) == READ_ONLY_SQLSTATE:
            return True
        orig = getattr(exc, "orig", None)
        if orig is not None and getattr(orig, "pgcode", None) == READ_ONLY_SQLSTATE:
            return True
        if "read-only transaction" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def report_read_only(what: str) -> None:
    """Log the skipped write and annotate the GitHub Actions run."""
    msg = f"Postgres is read-only; skipped {what}. The Parquet output still lands."
    logger.error(msg)
    # A workflow annotation, so a green run still shows the skip on its summary page.
    print(f"::warning title=Postgres read-only::{msg}", flush=True)
