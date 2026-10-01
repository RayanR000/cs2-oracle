"""The archive read votes in DuckDB; `_apply_multi_source_voting` is the reference.

The retrain's train-universe read (12.4M rows, 2026-09-28) no longer fit the CI
runner once pandas materialised every raw row and built one Series per
multi-source group, so `_fetch_voted_price_history` now votes in SQL and pandas
only ever sees the voted frame. The DB path and every voting test still use the
pandas implementation, so the two must agree on every branch: the source drops,
the conditional sync stand-down, the <3-row median, the 2-sigma mask, NULL-source
bucketing, duplicate same-source rows and NaN volume.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster
from models.item_parser import (
    BID_SOURCES,
    CONDITIONAL_STEAM_SOURCES,
    STEAM_SPOT_SOURCES,
    TRAILING_WINDOW_SOURCES,
)

ASKS = ["aggregator_buff163", "aggregator_skinport", "aggregator_csfloat", "aggregator_steam_17mafo"]
SOURCES = (
    [None] * 3
    + ASKS
    + sorted(BID_SOURCES | TRAILING_WINDOW_SOURCES | STEAM_SPOT_SOURCES)
    + sorted(CONDITIONAL_STEAM_SOURCES) * 3
)


def _random_rows(seed: int, n_groups: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        item, day = f"item-{g % 37}", pd.Timestamp("2026-01-01") + pd.Timedelta(days=g // 37)
        base = float(rng.uniform(0.5, 500))
        for _ in range(int(rng.choice([1, 1, 2, 3, 4, 6, 9]))):
            price = base * float(rng.normal(1, 0.05))
            if rng.random() < 0.1:
                price *= float(rng.choice([0.2, 5.0]))  # an outlier for the 2-sigma mask
            if rng.random() < 0.05:
                price = base  # exact ties, including all-equal groups (std == 0)
            volume = np.nan if rng.random() < 0.2 else float(rng.integers(0, 50))
            rows.append((item, day.date(), price, volume, SOURCES[int(rng.integers(len(SOURCES)))]))
    return pd.DataFrame(rows, columns=["item_id", "date", "price", "volume", "source"])


def _sql_vote(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("_rows", df)
    out = con.sql(ItemForecaster._multi_source_voting_sql("SELECT * FROM _rows")).df()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["item_id", "date", "price", "volume", "n_ask_sources"]]
    return df.sort_values(["item_id", "date"]).reset_index(drop=True)


@pytest.mark.parametrize("seed", range(8))
def test_sql_vote_matches_pandas_reference(seed):
    df = _random_rows(seed)
    expected = _canon(ItemForecaster._apply_multi_source_voting(df.copy()))
    actual = _canon(_sql_vote(df))

    assert len(actual) == len(expected)
    pd.testing.assert_frame_equal(
        actual[["item_id", "date"]], expected[["item_id", "date"]], check_dtype=False
    )
    np.testing.assert_array_equal(actual["n_ask_sources"].to_numpy(), expected["n_ask_sources"].to_numpy())
    assert actual["n_ask_sources"].dtype == np.int64
    np.testing.assert_allclose(actual["price"], expected["price"], rtol=1e-12)
    np.testing.assert_allclose(actual["volume"], expected["volume"], rtol=1e-12)


def test_sql_vote_all_excluded_returns_no_rows():
    df = pd.DataFrame(
        [("a", pd.Timestamp("2026-07-01").date(), 5.0, 1.0, s) for s in sorted(BID_SOURCES | STEAM_SPOT_SOURCES)],
        columns=["item_id", "date", "price", "volume", "source"],
    )
    assert len(_sql_vote(df)) == 0
    assert len(ItemForecaster._apply_multi_source_voting(df)) == 0


def test_exact_two_sigma_tie_is_kept_by_both():
    """Souvenir MP9 | Sand Dashed (FT), 2026-08-10: 0.66 sits exactly 0.28 from
    the median and 2 sigma is 0.28, but only up to the std's last ulp, which
    depends on summation order. Both implementations must keep it."""
    day = pd.Timestamp("2026-08-10").date()
    prices = {"aggregator_buff163": 0.34, "aggregator_csfloat": 0.35, "aggregator_csgotrader": 0.66,
              "aggregator_skinport": 0.62, "aggregator_youpin": 0.38}
    df = pd.DataFrame(
        [("a", day, p, 0.0, s) for s, p in prices.items()],
        columns=["item_id", "date", "price", "volume", "source"],
    )
    assert _sql_vote(df)["price"].tolist() == [0.38]
    assert ItemForecaster._apply_multi_source_voting(df)["price"].tolist() == [0.38]


def _canon_sets(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["item_id", "date", "source_set"]].copy()
    out["source_set"] = out["source_set"].astype(str)
    return out.sort_values(["item_id", "date"]).reset_index(drop=True)


@pytest.mark.parametrize("seed", range(8))
def test_source_set_matches_between_sql_and_pandas(seed):
    df = _random_rows(seed)
    expected = _canon_sets(ItemForecaster._apply_multi_source_voting(df.copy()))
    actual = _canon_sets(_sql_vote(df))
    pd.testing.assert_frame_equal(actual, expected)


def test_source_set_is_the_kept_sources_sorted_and_deduplicated():
    day = pd.Timestamp("2026-04-16").date()
    df = pd.DataFrame(
        [
            ("a", day, 10.0, 1.0, "aggregator_youpin"),
            ("a", day, 10.1, 1.0, "aggregator_buff163"),
            ("a", day, 10.2, 1.0, "aggregator_buff163"),  # duplicate source
            ("a", day, 10.3, 1.0, sorted(CONDITIONAL_STEAM_SOURCES)[0]),  # stands down: 2 other asks
            ("a", day, 9.0, 1.0, sorted(BID_SOURCES)[0]),  # bids never vote
            ("b", day, 5.0, 1.0, None),
        ],
        columns=["item_id", "date", "price", "volume", "source"],
    )
    for out in (ItemForecaster._apply_multi_source_voting(df.copy()), _sql_vote(df)):
        sets = dict(zip(out["item_id"], out["source_set"].astype(str)))
        assert sets == {"a": "aggregator_buff163|aggregator_youpin", "b": "<null>"}
        assert (out["source_set"].astype(str).str.count(r"\|") + 1 == out["n_ask_sources"]).all()


def test_pandas_vote_emits_category_dtype():
    out = ItemForecaster._apply_multi_source_voting(_random_rows(0))
    assert isinstance(out["source_set"].dtype, pd.CategoricalDtype)


def test_aligned_categories_survive_concat():
    a = pd.DataFrame({"source_set": pd.Categorical(["x"])})
    b = pd.DataFrame({"source_set": pd.Categorical(["y"])})
    cats = sorted(set().union(*(p["source_set"].cat.categories for p in (a, b))))
    for p in (a, b):
        p["source_set"] = p["source_set"].cat.set_categories(cats)
    assert isinstance(pd.concat([a, b], ignore_index=True)["source_set"].dtype, pd.CategoricalDtype)


def test_chunked_archive_read_keeps_source_set_categorical(tmp_path, monkeypatch):
    """Each item-hash chunk sees different source sets, so its categories differ.
    Unaligned, pd.concat falls back to `object` -- ~1 GB on the train read."""
    from datetime import UTC, date, datetime, timedelta
    from unittest.mock import MagicMock

    archive = tmp_path / "price-archive"
    archive.mkdir()
    rows = []
    for i in range(40):
        for k in range(3):
            rows.append((f"Item {i}", date(2026, 7, 10) + timedelta(days=k), 10.0 + k, 1, ASKS[i % len(ASKS)]))
    pd.DataFrame(rows, columns=["item_slug", "day", "mean_price", "volume", "source"]).assign(
        day=lambda d: pd.to_datetime(d["day"])
    ).to_parquet(archive / "prices-2026.parquet")
    monkeypatch.setenv("VOTED_CHUNKS", "4")
    f = ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path / "m"))
    f.archive_dir = archive
    f._now = lambda: datetime(2026, 7, 20, tzinfo=UTC)

    out = f._fetch_voted_price_history(days_back=30, backfilled_only=False)

    assert isinstance(out["source_set"].dtype, pd.CategoricalDtype)
    assert set(out["source_set"].astype(str)) == set(ASKS)
