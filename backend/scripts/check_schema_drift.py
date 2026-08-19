"""Fail if the ORM (`Base.metadata`) has drifted from what the migrations produce.

WHY THIS EXISTS: `Item.is_trainable` was added to the ORM (`database.py`) and merged
with no Alembic migration. Prod `items` never got the column, the aggregator crashed on
its first query, and the whole nightly chain skipped silently for days — review, tests,
and merge all passed it through. See docs/research/2026-08-19-deep-model-review.md §2.

HOW IT CATCHES THAT: point `DATABASE_URL` at a scratch DB, run `alembic upgrade head`,
then run this. Alembic's `compare_metadata` diffs the live (migrated) schema against
`Base.metadata` exactly as `--autogenerate` would. A model column with no migration shows
up as `add_column` -> this exits non-zero.

FAIL vs WARN: only *structural* presence drift fails the build (a table/column the ORM
declares that the migrations do not create, or a modeled column the migrations drop).
Type / default / nullable / index diffs are reported as warnings — Postgres reflection
compares those noisily (server defaults, varchar length, index naming), and gating CI on
them would flake. Set SCHEMA_DRIFT_STRICT=1 to escalate every diff to a failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from config import settings
from database import Base

# Ops that mean "the ORM and the migrations disagree about what exists". These are the
# missing-migration signature and carry no reflection noise.
STRUCTURAL_OPS = {"add_table", "add_column", "remove_column"}


def _flatten(diffs):
    """compare_metadata yields tuples, plus lists-of-tuples for column modifications."""
    for entry in diffs:
        if isinstance(entry, list):
            yield from entry
        else:
            yield entry


def _describe(diff) -> str:
    op = diff[0]
    if op in ("add_table", "remove_table"):
        return f"{op}: {diff[1].name}"
    if op in ("add_column", "remove_column"):
        # (op, schema, table_name, Column)
        return f"{op}: {diff[2]}.{diff[3].name}"
    if op in ("add_index", "remove_index", "add_constraint", "remove_constraint", "add_fk", "remove_fk"):
        # (op, SchemaItem) — name is often None for unnamed constraints.
        obj = diff[1]
        return f"{op}: {getattr(obj, 'name', None) or obj!r}"
    # modify_* ops: (op, schema, table_name, column_name, ...) plus scalars.
    return f"{op}: " + ", ".join(str(x) for x in diff[1:] if isinstance(x, (str, bool)))


def main() -> int:
    strict = os.environ.get("SCHEMA_DRIFT_STRICT") == "1"
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        diffs = list(_flatten(compare_metadata(ctx, Base.metadata)))

    if not diffs:
        print("No schema drift: Base.metadata matches the migrated schema.")
        return 0

    failing = [d for d in diffs if strict or d[0] in STRUCTURAL_OPS]
    warning = [d for d in diffs if d not in failing]

    if warning:
        print("Cosmetic schema diffs (warning only):")
        for d in warning:
            print(f"  - {_describe(d)}")

    if failing:
        print("\nSchema drift detected — the ORM and the migrations disagree:")
        for d in failing:
            print(f"  - {_describe(d)}")
        print(
            "\nA model was changed without a matching Alembic migration (or vice versa). "
            "Generate one with `alembic revision --autogenerate` and commit it."
        )
        return 1

    print("Only cosmetic diffs; no structural drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
