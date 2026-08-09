"""How many ask sources voted on each item-day.

The composition partition in docs/research/2026-08-08-model-review.md §5 -- which
takes the reversal signal from rank IC 0.168 to 0.101 and to ~0 on the cleanest
subset -- is currently recomputed ad hoc in a scratchpad. It needs to be a column
so the partition is auditable, and so a future basis-change detector has
something to key on that is not a price move.

It is also the best candidate cause of the harness's undiagnosed run-to-run
non-reproducibility (mean_diff_pp -0.1581 -> -0.0026 on identical commands).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _rows(*specs):
    return pd.DataFrame([
        {"item_id": i, "date": pd.Timestamp(d), "price": p, "source": s}
        for i, d, p, s in specs
    ])


def test_single_source_day_counts_one(tmp_path):
    out = _f(tmp_path)._apply_multi_source_voting(
        _rows(("a", "2026-07-01", 10.0, "aggregator_sync")))
    assert out["n_ask_sources"].iloc[0] == 1


def test_three_sources_count_three(tmp_path):
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-07-11", 10.0, "aggregator_buff163"),
        ("a", "2026-07-11", 10.2, "aggregator_youpin"),
        ("a", "2026-07-11", 9.9, "aggregator_csfloat"),
    ))
    assert out["n_ask_sources"].iloc[0] == 3


def test_bid_does_not_count_as_an_ask_source(tmp_path):
    """aggregator_buff163_buy is the wrong side of the book and was removed
    from voting 2026-08-07. It must not inflate the composition count either."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-07-11", 10.0, "aggregator_buff163"),
        ("a", "2026-07-11", 5.8, "aggregator_buff163_buy"),
    ))
    assert out["n_ask_sources"].iloc[0] == 1


def test_null_source_counts_one_not_zero(tmp_path):
    """Every pre-2026 row has source IS NULL -- 9.4M rows, 4,523 days. A NULL
    source is one backfill source, not the absence of one."""
    out = _f(tmp_path)._apply_multi_source_voting(
        _rows(("a", "2020-01-01", 10.0, None)))
    assert out["n_ask_sources"].iloc[0] == 1


def test_column_is_never_null(tmp_path):
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2020-01-01", 10.0, None),
        ("b", "2026-07-11", 3.0, "aggregator_skinport"),
    ))
    assert out["n_ask_sources"].notna().all()
    assert out["n_ask_sources"].dtype.kind in "iu"


def test_cache_version_bumped():
    assert ItemForecaster.VOTED_CACHE_VERSION >= 5, (
        "the voted frame gained a column; a stale cache would train the next "
        "model on a frame without it")
