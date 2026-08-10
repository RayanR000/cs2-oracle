"""A bid must never vote in the consensus median.

`aggregator_buff163_buy` is BUFF's `highest_order` — a *bid*. It entered the
archive on 2026-07-11 and voted unfiltered against asks sitting 0.717-0.809x of
Steam while it sat at 0.579x, displacing the consensus by a median -10.8% on
95.4% of item-days in the >=$1 served cohort. See
`docs/changelog/2026-08-07-bid-source-excluded-from-voting.md`.

The measurement also refuted the mechanism these tests were first sketched
against: the 2-sigma guard was *eligible* on 99.3% of bid item-days and simply
failed to reject the bid 80.5% of the time, because an 11-source panel spanning
0.58-1.04x of Steam has a sigma too wide for 2 sigma to catch anything. So the
fix cannot be a tighter guard, and these tests pin the two channels that
actually did the damage: median parity, and the bid counting toward the
three-source gate.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest.price_resolution import load_voted_prices
from models.forecaster import BID_SOURCES, ItemForecaster


def _frame(rows):
    """(item_id, day, price, source) tuples -> a pre-voting frame."""
    df = pd.DataFrame(rows, columns=["item_id", "date", "price", "source"])
    df["volume"] = 1
    return df


def vote(rows):
    return ItemForecaster._apply_multi_source_voting(_frame(rows))


# The measured source ladder as a multiple of Steam (n=461,540 item-days,
# 2026-07-11..2026-08-04), which is the panel 55% of bid item-days actually see.
# A narrower toy panel is not a substitute: at 11 sources spanning 0.58-1.04x,
# sigma is wide enough that the bid survives the 2-sigma mask 83.3% of the time,
# and a test built on three tight asks passes without the fix.
LADDER = [
    ("aggregator_buff163_buy", 0.579),
    ("aggregator_buff163", 0.717),
    ("aggregator_youpin", 0.724),
    ("aggregator_csfloat", 0.732),
    ("aggregator_csmoney", 0.753),
    ("aggregator_skinport", 0.809),
    ("aggregator_steam_7d", 1.000),
    ("aggregator_steam_30d", 1.000),
    ("aggregator_sync", 1.000),
    ("aggregator_csgotrader", 1.000),
    ("aggregator_steam_90d", 1.043),
]

# Consensus over the ten non-bid legs (2026-08-07 baseline, before Task 4):
# median 9.045, an 8.09-consensus drag of -10.6% once the bid is added back in.
#
# 2026-08-09: three of those ten legs (aggregator_steam_7d/30d/90d) are
# themselves excluded from voting now (docs/changelog/
# 2026-08-09-trailing-window-sources-excluded.md), so `vote()` also drops them
# here -- this constant is `_apply_multi_source_voting`'s actual output over
# the seven remaining real asks (buff163/youpin/csfloat/csmoney/skinport plus
# aggregator_sync and aggregator_csgotrader), not just the bid's removal.
LADDER_ASK_CONSENSUS = 7.32


# A real `market_hash_name`. The archive readers drop all-lowercase keys as
# phantom duplicates (`item_parser.is_phantom_slug`), so a toy slug like "ak"
# would leave `_load_all_prices` empty and pass the bid assertion vacuously.
LADDER_ITEM = "AK-47 | Redline (Field-Tested)"


def _ladder_rows(item=LADDER_ITEM, d=date(2026, 7, 15), steam=10.0):
    return [(item, d, round(steam * mult, 4), src) for src, mult in LADDER]


def test_bid_source_is_excluded_from_the_consensus_median():
    """Median parity, the largest single channel: adding one low value to an
    even ask panel steps the median down a rung of the source ladder."""
    out = vote(_ladder_rows())

    assert len(out) == 1
    assert out.iloc[0]["price"] == pytest.approx(LADDER_ASK_CONSENSUS)


def test_bid_survives_the_outlier_mask_so_the_mask_cannot_be_the_fix():
    """Pins the refuted alternative. On the real panel the bid is within 2
    sigma, so it is *kept* by the guard — tightening or relying on the guard
    would not have removed it. Asserted directly on `vote`'s own arithmetic so
    the claim is checked, not just asserted in a comment."""
    import numpy as np

    prices = np.array([p for _, _, p, _ in _ladder_rows()])
    median = np.median(prices)
    assert abs(5.79 - median) <= 2.0 * np.std(prices, ddof=0)


def test_bid_does_not_count_toward_the_three_source_outlier_gate():
    """`vote()` only runs the 2-sigma rejection at >= 3 sources. A bid must not
    be what lifts an item-day over that gate, or it changes the estimator
    itself and not merely the inputs.

    Uses two genuine ask sources (neither a bid nor a trailing window) so this
    stays a test of the bid/gate interaction and does not also exercise
    Task 4's separate exclusion."""
    d = date(2026, 7, 15)
    out = vote([
        ("ak", d, 10.0, "aggregator_csfloat"),
        ("ak", d, 30.0, "aggregator_youpin"),
        ("ak", d, 2.0, "aggregator_buff163_buy"),
    ])

    # Two asks is the bare-median path: median(10.0, 30.0) = 20.0. Counting the
    # bid makes it three "sources", and the 2-sigma mask over (2, 10, 30) keeps
    # everything, so the broken consensus is median(2, 10, 30) = 10.0.
    assert out.iloc[0]["price"] == pytest.approx(20.0)


def test_item_day_carrying_only_a_bid_yields_no_row():
    """2,338 item-days across 217 items have no price at all without the bid.
    They must drop out rather than fall back to the bid: a series whose basis
    alternates between bid and ask fabricates the wedge as a return, which is
    the whole defect."""
    d = date(2026, 7, 15)
    out = vote([
        ("ak", d, 6.0, "aggregator_buff163_buy"),
        ("awp", d, 50.0, "aggregator_csfloat"),
    ])

    assert set(out["item_id"]) == {"awp"}


def test_all_bid_input_returns_an_empty_frame_with_the_voted_columns():
    out = vote([
        ("ak", date(2026, 7, 15), 6.0, "aggregator_buff163_buy"),
    ])

    assert out.empty
    assert {"item_id", "date", "price", "volume"} <= set(out.columns)


def test_untagged_pre_2026_rows_still_vote():
    """`source IS NULL` selects the pre-2026 backfilled series, which is 13
    years of the archive. A NULL-safe filter is the difference between
    excluding one feed and deleting the history."""
    d = date(2019, 3, 2)
    out = vote([
        ("ak", d, 10.0, None),
        ("ak", d, 12.0, None),
    ])

    assert out.iloc[0]["price"] == pytest.approx(11.0)


def test_bid_sources_is_not_empty():
    """Guards against the constant being emptied and every test above passing
    for the wrong reason."""
    assert "aggregator_buff163_buy" in BID_SOURCES


def test_label_resolution_path_excludes_the_bid(tmp_path):
    """Both legs of every scored label resolve through `load_voted_prices`, so
    the exclusion has to hold there and not only in the training path."""
    archive = tmp_path / "price-archive"
    archive.mkdir()
    rows = _ladder_rows()
    pd.DataFrame({
        "item_slug": [r[0] for r in rows],
        "day": pd.to_datetime([r[1] for r in rows]),
        "mean_price": [r[2] for r in rows],
        "volume": 1,
        "source": [r[3] for r in rows],
    }).to_parquet(archive / "prices-2026.parquet")

    out = load_voted_prices(archive, [LADDER_ITEM],
                            date(2026, 7, 15), date(2026, 7, 15))

    assert len(out) == 1
    assert out.iloc[0]["price"] == pytest.approx(LADDER_ASK_CONSENSUS)


def test_published_gate_loader_excludes_the_bid(tmp_path, monkeypatch):
    """`walkforward_backtest._load_all_prices` does not call
    `_apply_multi_source_voting` at all — it hands duplicate item-days to
    `engineer_features`, which collapses them with a plain *mean*. A mean has no
    outlier rejection whatever, so the bid lands in the published Backtest
    Accuracy number undiluted. The filter has to be in the query.
    """
    import duckdb

    from scripts import walkforward_backtest as wf

    archive = tmp_path / "price-archive"
    archive.mkdir()
    rows = _ladder_rows()
    pd.DataFrame({
        "item_slug": [r[0] for r in rows],
        "day": pd.to_datetime([r[1] for r in rows]),
        "mean_price": [r[2] for r in rows],
        "volume": 1,
        "source": [r[3] for r in rows],
    }).to_parquet(archive / "prices-2026.parquet")
    monkeypatch.setattr(wf, "ARCHIVE_DIR", archive)

    con = duckdb.connect()
    try:
        out = wf._load_all_prices(con, [(LADDER_ITEM,)])
    finally:
        con.close()

    assert len(out) == len(LADDER) - 1
    assert out["price"].min() == pytest.approx(7.17)
    # The mean the gate actually scores on, with and without the bid.
    assert out["price"].mean() == pytest.approx(8.778, abs=1e-3)
