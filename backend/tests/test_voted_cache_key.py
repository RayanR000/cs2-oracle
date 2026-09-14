"""The voted cache key must not depend on mtime.

CI checks the archive out fresh every run, so st_mtime_ns is new on every run
and the key changed unconditionally -- the cache could never hit there. Cost of
the miss is ~48s per run (21.1s DuckDB read + 27.2s voting), paid daily by the
predict path too, not just by the Monday retrain.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from unittest.mock import MagicMock

import pandas as pd
import pytest
from models.forecaster import ItemForecaster


def _code(func) -> str:
    """Source of `func` with comments and docstrings stripped.

    The method documents the mtime scheme it replaced, so a raw getsource grep
    matches the prose and fails on the fix. (Same helper as
    test_optuna_objective.py; kept local rather than adding a conftest for two
    call sites.)
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)  # ast.unparse never emits comments


def _prices(rows: int, extra_cols: bool = False) -> pd.DataFrame:
    """The archive's real schema. The date column is `day`, not `date` -- the
    first content-based fingerprint keyed on MAX(date) and passed every test
    here while failing the retrain with a binder error, because these fixtures
    had invented a `date` column."""
    df = pd.DataFrame(
        {
            "item_slug": [f"item-{i}" for i in range(rows)],
            "day": pd.to_datetime(["2026-01-01"] * rows),
            "source": ["aggregator_sync"] * rows,
            "mean_price": [float(i) for i in range(rows)],
            "volume": list(range(rows)),
            "ingested_at": pd.to_datetime(["2026-01-02"] * rows),
        }
    )
    if extra_cols:
        # prices-2026-03 and -04 really do carry these; 19 other files do not.
        df["min_price"] = df["mean_price"]
        df["max_price"] = df["mean_price"]
    return df


@pytest.fixture
def archive(tmp_path):
    d = tmp_path / "price-archive"
    d.mkdir()
    _prices(2).to_parquet(d / "prices-2026-01.parquet")
    return d


def test_fingerprint_survives_a_non_uniform_schema(tmp_path, archive):
    """The archive is not schema-uniform, so the fingerprint may not name a
    column. Regression: MAX(date) died on the real files."""
    _prices(3, extra_cols=True).to_parquet(archive / "prices-2026-03.parquet")
    f = _f(tmp_path, archive)
    assert f._archive_fingerprint()  # does not raise


def _f(tmp_path, archive):
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))
    f.archive_dir = archive
    return f


def test_fingerprint_is_stable_across_a_touch(tmp_path, archive):
    f = _f(tmp_path, archive)
    before = f._archive_fingerprint()
    path = archive / "prices-2026-01.parquet"
    os.utime(path, (1_600_000_000, 1_600_000_000))
    assert f._archive_fingerprint() == before


def test_fingerprint_changes_when_content_changes(tmp_path, archive):
    f = _f(tmp_path, archive)
    before = f._archive_fingerprint()
    _prices(3).to_parquet(archive / "prices-2026-01.parquet")
    assert f._archive_fingerprint() != before


def test_fingerprint_does_not_read_mtime(tmp_path, archive):
    assert "st_mtime" not in _code(ItemForecaster._archive_fingerprint)
