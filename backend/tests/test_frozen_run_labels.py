"""Labels must not be measured to or from a day inside a frozen price run.

The per-item companion to `test_degenerate_label_dates.py`. That file covers the
two CROSS-SECTIONAL defects — a re-published snapshot day and a collector
cutover. This one covers the defect that lives in a single item's series: a run
of bit-identical prices, which is the Getmansky-Lo-Makarov MA(k) mechanism and
which the +/-500% winsorization cannot see, because a fabricated 0% return is
well inside the clip.

Both legs are tested, matching the snapshot rule's endpoint semantics. Measured
2026-08-08 on the >=$1 cohort, the anchor leg alone catches 65-72% of the
exact-zero return mass at h=3-14 and 31.2% at h=30; anchor-or-target reaches
86-91% and 52.0%.

The rule is NOT neutral: 13.7-15.9% of the >=$1 labels it voids carry a
non-zero return. See `docs/superpowers/specs/2026-08-08-frozen-price-runs-design.md`.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from models import forecaster as fc
from models.forecaster import ItemForecaster
from models.staleness import STALE_RUN_GAP_BREAK_DAYS


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path_factory.mktemp("saved_models")))


def _panel(n_items=40, n_days=40, start=date(2026, 1, 1), seed=7):
    """A clean panel: every item moves every day, so nothing is stale."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        price = 10.0 + i
        for d in range(n_days):
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"item_id": f"i{i}", "date": start + timedelta(days=d), "price": price})
    return pd.DataFrame(rows)


def _freeze(df, item, days):
    """Carry `item`'s price forward, unchanged, across `days`."""
    df = df.copy()
    mask = (df["item_id"] == item) & (df["date"].isin(days))
    held = df.loc[(df["item_id"] == item) & (df["date"] == min(days) - timedelta(days=1)), "price"]
    df.loc[mask, "price"] = float(held.iloc[0])
    return df


def _republish(df, day):
    """Overwrite `day` with the previous day's prices, across the whole panel.

    Fires `_snapshot_dates`, so the voiding log line has something to report
    that is not the frozen-run rule.
    """
    df = df.copy()
    prev = df[df["date"] == day - timedelta(days=1)].set_index("item_id")["price"]
    mask = df["date"] == day
    df.loc[mask, "price"] = df.loc[mask, "item_id"].map(prev).to_numpy()
    return df


def _label(out, item, day, horizon):
    row = out[(out["item_id"] == item) & (out["date"] == day)]
    assert len(row) == 1
    return row[f"target_return_{horizon}d"].iloc[0]


def test_clean_panel_keeps_its_labels(forecaster):
    """Guards the rest of the file: a moving series must void nothing."""
    df = _panel()
    out = forecaster.prepare_targets(df, horizon=3)
    kept = out["target_return_3d"].notna().sum()
    # Every row except the last 3 days per item has a partner.
    assert kept == 40 * (40 - 3)


def test_label_anchored_on_a_frozen_day_is_voided(forecaster):
    horizon = 3
    frozen = [date(2026, 1, 11), date(2026, 1, 12)]
    df = _freeze(_panel(), "i0", frozen)
    out = forecaster.prepare_targets(df, horizon=horizon)

    # 01-11 is the first repeat (stale_run_days == 1) -> voided as the anchor.
    assert pd.isna(_label(out, "i0", frozen[0], horizon))
    assert pd.isna(_label(out, "i0", frozen[1], horizon))
    # 01-10 set the level, so it is a fresh price and survives as an anchor.
    # Its TARGET is 01-13, which is not frozen.
    assert not pd.isna(_label(out, "i0", date(2026, 1, 10), horizon))


def test_label_whose_target_lands_on_a_frozen_day_is_voided(forecaster):
    """The endpoint rule's second half. Anchor is clean, target is not."""
    horizon = 3
    frozen = [date(2026, 1, 11), date(2026, 1, 12)]
    df = _freeze(_panel(), "i0", frozen)
    out = forecaster.prepare_targets(df, horizon=horizon)

    # 01-08 + 3d = 01-11, a frozen day. The anchor itself is a fresh level.
    assert pd.isna(_label(out, "i0", date(2026, 1, 8), horizon))


def test_a_frozen_run_voids_only_the_frozen_item(forecaster):
    """The bug this test exists for: keying on the day alone, not (item, day).

    A date-only stale set voids every item's label on any day some OTHER item
    happened to be frozen, which at a 20% archive-wide stale rate would empty
    the label set.
    """
    horizon = 3
    frozen = [date(2026, 1, 11), date(2026, 1, 12)]
    df = _freeze(_panel(), "i0", frozen)
    out = forecaster.prepare_targets(df, horizon=horizon)

    for day in frozen:
        assert pd.isna(_label(out, "i0", day, horizon))
        for other in ("i1", "i2", "i17"):
            assert not pd.isna(_label(out, other, day, horizon))


def test_a_gap_wider_than_the_break_breaks_the_run_rather_than_voiding(forecaster):
    """Identical prices either side of a collection outage are not a frozen feed."""
    horizon = 3
    start = date(2026, 1, 1)
    gap = STALE_RUN_GAP_BREAK_DAYS + 1
    rows = []
    for d in range(6):
        rows.append({"item_id": "i0", "date": start + timedelta(days=d), "price": 10.0 + d})
    # One observation `gap` days after the last, at an identical price.
    resumed = start + timedelta(days=5 + gap)
    rows.append({"item_id": "i0", "date": resumed, "price": 15.0})
    rows.append({"item_id": "i0", "date": resumed + timedelta(days=horizon), "price": 20.0})
    out = ItemForecaster.prepare_targets(
        ItemForecaster(db_session=MagicMock(), model_dir="/tmp/x"), pd.DataFrame(rows), horizon=horizon
    )

    # `resumed` repeats 15.0 but is 8 days after it, so it is a fresh level.
    assert not pd.isna(_label(out, "i0", resumed, horizon))


def test_threshold_is_read_from_the_constant(forecaster, monkeypatch):
    """Raising LABEL_MAX_STALE_RUN_DAYS must admit shorter runs.

    This is what lets the harness contrast 0/1/2 without a code edit.
    """
    horizon = 3
    frozen = [date(2026, 1, 11), date(2026, 1, 12)]
    df = _freeze(_panel(), "i0", frozen)

    monkeypatch.setattr(fc, "LABEL_MAX_STALE_RUN_DAYS", 1)
    out = forecaster.prepare_targets(df, horizon=horizon)

    # 01-11 is run 1, now allowed. 01-12 is run 2, still voided.
    assert not pd.isna(_label(out, "i0", frozen[0], horizon))
    assert pd.isna(_label(out, "i0", frozen[1], horizon))


def test_voiding_is_reported_in_the_log_line(forecaster, caplog):
    frozen = [date(2026, 1, 11), date(2026, 1, 12)]
    df = _freeze(_panel(), "i0", frozen)
    with caplog.at_level("INFO"):
        forecaster.prepare_targets(df, horizon=3)
    assert any("frozen price run" in r.message for r in caplog.records)


def test_the_log_says_disabled_rather_than_a_count_when_the_rule_is_off(forecaster, caplog, monkeypatch):
    """`None` formatted into the count read as '(> Noned)', which looks like a
    threshold rather than an off switch — and the off arm is exactly the one a
    reader is checking when they scan this line."""
    monkeypatch.setattr(fc, "LABEL_MAX_STALE_RUN_DAYS", None)
    # A republished day, so the snapshot rule still voids something and the
    # line fires at all. With the frozen-run rule off and a clean panel there
    # is nothing to void and nothing to log.
    df = _republish(_panel(), date(2026, 1, 20))
    with caplog.at_level("INFO"):
        forecaster.prepare_targets(df, horizon=3)
    messages = [r.message for r in caplog.records if "Voided" in r.message]
    assert messages, "the snapshot voiding should still report"
    assert any("frozen-run rule DISABLED" in m for m in messages)
    assert not any("None" in m for m in messages)


def test_frame_without_a_price_column_does_not_raise(forecaster):
    """prepare_targets is called on frames the caller assembled; degrade, don't crash."""
    df = _panel()[["item_id", "date"]].copy()
    df["price"] = 1.0
    out = forecaster.prepare_targets(df.drop(columns=["price"]).assign(price=1.0), horizon=3)
    assert "target_return_3d" in out.columns
