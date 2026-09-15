"""Crash safety of the ops Parquet store (`db/parquet.py`).

Both writers must be atomic: `_append_parquet` COPYs to a sibling temp file
and `os.replace`s it into place, and `delete_table` rewrites via
`_atomic_write` (temp + `os.replace`). A process killed mid-write — or a
failed COPY — must leave either the whole pre-state or the whole post-state,
never a half-written file and never a stray temp file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import db.parquet as parquet
from db.parquet import _append_parquet, _atomic_write, _tmp_path, delete_table


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})


def test_tmp_paths_are_writer_unique():
    base = Path("/tmp/ops/tbl.parquet")
    assert _tmp_path(base) != _tmp_path(base)
    assert _tmp_path(base).parent == base.parent


def test_delete_table_removes_only_matching_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(parquet, "OPS_DIR", tmp_path)
    parquet.append_table("t", _frame(), ["id"])
    delete_table("t", {"id": 2})
    out = parquet.read_table("t").sort_values("id").reset_index(drop=True)
    assert out["id"].tolist() == [1, 3]
    assert out["val"].tolist() == ["a", "c"]


def test_delete_table_missing_table_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(parquet, "OPS_DIR", tmp_path)
    delete_table("nope", {"id": 1})  # must not raise


def test_delete_table_uses_atomic_write(tmp_path, monkeypatch):
    """The rewrite goes through _atomic_write — no direct overwrite."""
    monkeypatch.setattr(parquet, "OPS_DIR", tmp_path)
    parquet.append_table("t", _frame(), ["id"])
    before = (tmp_path / "t.parquet").read_bytes()
    with patch("db.parquet._atomic_write") as mock_atomic:
        delete_table("t", {"id": 1})
        assert mock_atomic.call_count == 1
        path, filtered = mock_atomic.call_args[0]
        assert path == tmp_path / "t.parquet"
        assert sorted(filtered["id"].tolist()) == [2, 3]
    # The spy wrote nothing: the live file is untouched.
    assert (tmp_path / "t.parquet").read_bytes() == before


def test_append_parquet_leaves_no_temp_files(tmp_path):
    path = tmp_path / "t.parquet"
    _frame().to_parquet(path, index=False)
    _append_parquet(path, pd.DataFrame({"id": [4], "val": ["d"]}), ["id"])
    assert list(tmp_path.glob("*.tmp")) == []
    out = pd.read_parquet(path).sort_values("id").reset_index(drop=True)
    assert out["id"].tolist() == [1, 2, 3, 4]


def test_append_parquet_cleans_temp_and_keeps_original_on_copy_failure(tmp_path, monkeypatch):
    """A failed COPY must unlink the temp and preserve the pre-state."""
    monkeypatch.setattr(parquet, "_tmp_path", lambda p: tmp_path / "no-such-dir" / "t.tmp")
    path = tmp_path / "t.parquet"
    _frame().to_parquet(path, index=False)
    before = path.read_bytes()
    with pytest.raises(Exception):
        _append_parquet(path, pd.DataFrame({"id": [9], "val": ["z"]}), ["id"])
    assert list(tmp_path.glob("*.tmp")) == []
    assert not (tmp_path / "no-such-dir").exists()
    assert path.read_bytes() == before


def test_atomic_write_never_leaves_partial_file(tmp_path):
    path = tmp_path / "t.parquet"
    _frame().to_parquet(path, index=False)
    before = path.read_bytes()
    with patch.object(pd.DataFrame, "to_parquet", side_effect=RuntimeError("disk full")):
        try:
            _atomic_write(path, _frame())
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected the write failure to propagate")
    assert list(tmp_path.glob("*.tmp")) == []
    assert path.read_bytes() == before
