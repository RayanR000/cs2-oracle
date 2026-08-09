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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)          # ast.unparse never emits comments


@pytest.fixture
def archive(tmp_path):
    d = tmp_path / "price-archive"
    d.mkdir()
    pd.DataFrame({
        "item_slug": ["a", "b"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "mean_price": [1.0, 2.0],
    }).to_parquet(d / "prices-2026-01.parquet")
    return d


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
    pd.DataFrame({
        "item_slug": ["a", "b", "c"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "mean_price": [1.0, 2.0, 3.0],
    }).to_parquet(archive / "prices-2026-01.parquet")
    assert f._archive_fingerprint() != before


def test_fingerprint_does_not_read_mtime(tmp_path, archive):
    assert "st_mtime" not in _code(ItemForecaster._archive_fingerprint)
