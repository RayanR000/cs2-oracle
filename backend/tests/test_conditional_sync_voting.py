"""`aggregator_sync` must not vote when genuine asks are available.

MEASURED 2026-08-29, split-half reliability on the 2026 archive (sync added to
ONE half only, so the two halves never share a source -- putting it in BOTH
inflates the correlation by +0.11..+0.23, which is a pure shared-noise
artifact and must not be used as evidence):

    h    corr(A,B)   corr(A+sync,B)   delta    corr(sync alone, B)
    1     0.1265        0.1154       -0.0111        0.0028
    3     0.1793        0.1666       -0.0127        0.0068
    7     0.2284        0.2083       -0.0201        0.0050
    14    0.2834        0.2530       -0.0304        0.0108
    30    0.3510        0.2921       -0.0590        0.0095

sync correlates 0.003-0.011 with the non-Steam composite's returns -- it
carries essentially NO information about the common latent price -- and
including it COSTS reliability, 17% relative at h=30.

Cause: `aggregator_sync` is Steam `last_24h` FALLING BACK to the MA(7)/MA(30)/
MA(90) chain (collectors/csgotrader_aggregator.py:296-305) on exactly the
illiquid items, so where it matters it is a trailing mean wearing a spot
label. Steam-vs-venue RETURN correlation is ~0.05 even though LEVEL
correlation is 0.96.

Why CONDITIONAL and not a flat drop: item_parser.py already records that
dropping sync outright "deletes 2026-01 and 2026-02 in full for the >=$1
cohort (52,048 item-days)", where sync is the only source there is. So sync
stays as the sole-source fallback and only stands down once >=2 genuine asks
are present -- exactly the item-days where the measurement above says it hurts.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from models.forecaster import ItemForecaster
from models.item_parser import CONDITIONAL_STEAM_SOURCES, MIN_ASKS_TO_DROP_SYNC


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _rows(*specs):
    return pd.DataFrame([
        {"item_id": i, "date": pd.Timestamp(d), "price": p, "source": s,
         "volume": 0.0}
        for i, d, p, s in specs
    ])


def test_the_conditional_source_is_named():
    assert CONDITIONAL_STEAM_SOURCES == frozenset({"aggregator_sync"})
    assert MIN_ASKS_TO_DROP_SYNC == 2
    assert ItemForecaster.CONDITIONAL_STEAM_SOURCES == CONDITIONAL_STEAM_SOURCES


def test_it_is_separate_from_the_unconditional_drops():
    """A conditional stand-down is a different rule from an outright exclusion;
    folding it into TRAILING_WINDOW_SOURCES would delete 2026-01 and 2026-02."""
    assert not (CONDITIONAL_STEAM_SOURCES & set(ItemForecaster.BID_SOURCES))
    assert not (CONDITIONAL_STEAM_SOURCES
                & set(ItemForecaster.TRAILING_WINDOW_SOURCES))
    assert not (CONDITIONAL_STEAM_SOURCES
                & set(ItemForecaster.STEAM_SPOT_SOURCES))


def test_sync_stands_down_when_two_genuine_asks_are_present(tmp_path):
    """sync is the outlier here: if it voted, the median of 10/10/100 is 10;
    with it dropped the median of the two real asks is 10 as well, so assert on
    n_ask_sources, which is what actually records the stand-down."""
    df = _rows(
        (1, "2026-08-01", 10.0, "aggregator_buff163"),
        (1, "2026-08-01", 12.0, "aggregator_csfloat"),
        (1, "2026-08-01", 100.0, "aggregator_sync"),
    )
    out = _f(tmp_path)._apply_multi_source_voting(df)
    assert len(out) == 1
    assert out.iloc[0]["n_ask_sources"] == 2
    assert out.iloc[0]["price"] == 11.0          # median(10, 12), sync excluded


def test_sync_keeps_voting_when_it_is_the_only_source(tmp_path):
    """The 2026-01/02 guarantee: 52,048 item-days must not vanish."""
    df = _rows((1, "2026-01-15", 42.0, "aggregator_sync"))
    out = _f(tmp_path)._apply_multi_source_voting(df)
    assert len(out) == 1
    assert out.iloc[0]["price"] == 42.0
    assert out.iloc[0]["n_ask_sources"] == 1


def test_sync_keeps_voting_against_a_single_other_ask(tmp_path):
    """Below the threshold there is no consensus to defer to, so a second
    opinion still beats one. Dropping here would leave a single unchecked ask."""
    df = _rows(
        (1, "2026-08-01", 10.0, "aggregator_buff163"),
        (1, "2026-08-01", 20.0, "aggregator_sync"),
    )
    out = _f(tmp_path)._apply_multi_source_voting(df)
    assert out.iloc[0]["n_ask_sources"] == 2
    assert out.iloc[0]["price"] == 15.0


def test_null_source_rows_still_vote(tmp_path):
    """`source` IS NULL for the whole pre-2026 series; `isin` is False for NaN
    and that is what keeps 13 years of archive voting."""
    df = _rows(
        (1, "2025-06-01", 10.0, None),
        (1, "2025-06-01", 12.0, None),
        (1, "2025-06-01", 14.0, None),
    )
    out = _f(tmp_path)._apply_multi_source_voting(df)
    assert len(out) == 1
    assert out.iloc[0]["price"] == 12.0


def test_a_null_series_does_not_trigger_the_stand_down(tmp_path):
    """NULL sources collapse to one bucket, so they can never reach the
    >=2-genuine-asks threshold on their own and strand a sync row."""
    df = _rows(
        (1, "2026-08-01", 10.0, None),
        (1, "2026-08-01", 12.0, None),
        (1, "2026-08-01", 30.0, "aggregator_sync"),
    )
    out = _f(tmp_path)._apply_multi_source_voting(df)
    assert out.iloc[0]["n_ask_sources"] == 2      # {NULL, sync}: sync stays
