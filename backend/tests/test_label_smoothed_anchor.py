"""The label's denominator, and whether it is the quote `predict` serves from.

`prepare_targets` divides by the RAW price observed on the anchor day. That same
quote drives `return_1d` and every level feature built from it, so a noisy-high
anchor deflates the label and inflates the feature at once, and a model that
reads the noise scores as if it had read the market. The serving path divides by
`_smoothed_anchor_prices` -- a span-bounded median -- so it has no such quote to
share.

The 2026-08-11 replay attributed 95% of the CV/serving rank IC gap to exactly
this axis: swapping only the denominator moved +0.1398 of a +0.1464 gap, and the
gap is zero on items whose anchor quote already equals its local median
(`docs/changelog/2026-08-11-the-gap-is-the-anchor-denominator.md`,
`2026-08-11-clean-anchor-confirmed-in-ci.md`, 16 cells of 16 in CI).

`LABEL_SMOOTHED_ANCHOR=1` puts the label on the served denominator. It is a
LABEL change, so a rank IC measured under it is NOT comparable to one measured
without it -- the two arms are scored against different targets. The referee is
`scripts/replay_serving.py`, which scores both on one basis.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from backtest.price_resolution import MAX_WINDOW_SPAN_DAYS, SMOOTH_WINDOW
from models.forecaster import ItemForecaster


@pytest.fixture
def forecaster(tmp_path_factory):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path_factory.mktemp("saved_models")))


@pytest.fixture
def smoothed(monkeypatch):
    monkeypatch.setenv("LABEL_SMOOTHED_ANCHOR", "1")


def _panel(n_items=8, n_days=40, start=date(2026, 1, 1), seed=7):
    """A clean panel: every item moves every day, so nothing is voided."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        price = 10.0 + i
        for d in range(n_days):
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"item_id": f"i{i}", "date": start + timedelta(days=d), "price": price})
    return pd.DataFrame(rows)


def _row(out, item, day):
    r = out[(out["item_id"] == item) & (out["date"] == day)]
    assert len(r) == 1, f"{item} {day}: {len(r)} rows"
    return r.iloc[0]


def _series(df, item, day):
    """This item's prices up to and including `day`, oldest first."""
    s = df[(df["item_id"] == item) & (df["date"] <= day)].sort_values("date")
    return s["price"].to_numpy()


# ----------------------------------------------------------------- default off


def test_default_is_the_raw_anchor(forecaster, monkeypatch):
    """Off by default, and off means byte-identical to the shipped label."""
    monkeypatch.delenv("LABEL_SMOOTHED_ANCHOR", raising=False)
    assert ItemForecaster.label_smoothed_anchor_enabled() is False

    df = _panel()
    out = forecaster.prepare_targets(df, horizon=3)
    row = _row(out, "i0", date(2026, 1, 20))
    expected = (row["target_3d"] - row["price"]) / row["price"] * 100
    assert row["target_return_3d"] == pytest.approx(expected, abs=1e-12)


def test_the_two_arms_disagree(forecaster, monkeypatch):
    """Guards every test below: if the flag changed nothing, they would all
    pass against the raw label and prove nothing."""
    df = _panel()
    monkeypatch.delenv("LABEL_SMOOTHED_ANCHOR", raising=False)
    raw = forecaster.prepare_targets(df, horizon=3)["target_return_3d"]
    monkeypatch.setenv("LABEL_SMOOTHED_ANCHOR", "1")
    smooth = forecaster.prepare_targets(df, horizon=3)["target_return_3d"]

    both = raw.notna() & smooth.notna()
    assert both.sum() > 100
    assert not np.allclose(raw[both], smooth[both])


# ------------------------------------------------------------------- the value


def test_denominator_is_the_span_bounded_median(forecaster, smoothed):
    """The median of the last SMOOTH_WINDOW observations, and BOTH legs move.

    Subtracting the raw anchor from a smoothed denominator would leave
    `(S - p_raw)/S` in the label -- the contamination term itself, rescaled.
    """
    df = _panel()
    day = date(2026, 1, 20)
    out = forecaster.prepare_targets(df, horizon=3)
    row = _row(out, "i0", day)

    den = float(np.median(_series(df, "i0", day)[-SMOOTH_WINDOW:]))
    assert row["target_return_3d"] == pytest.approx((row["target_3d"] - den) / den * 100, abs=1e-9)


def test_it_matches_the_price_predict_quotes_from(forecaster, smoothed):
    """The point of the change: one denominator, shared with the serving path.

    `_smoothed_anchor_prices` is what `predict` converts a return-space forecast
    to dollars with. A label divided by anything else is quoted against a base
    production never uses.
    """
    df = _panel()
    day = date(2026, 1, 20)
    out = forecaster.prepare_targets(df, horizon=3)

    served = ItemForecaster._smoothed_anchor_prices(df[df["date"] <= day], pd.Timestamp(day))
    for item in df["item_id"].unique():
        row = _row(out, item, day)
        implied = row["target_3d"] / (1 + row["target_return_3d"] / 100)
        assert implied == pytest.approx(served[item], rel=1e-9)


def test_a_spike_at_the_anchor_barely_reaches_the_label(forecaster, monkeypatch):
    """The mechanism, in one row.

    A single quote well above the local level is what `return_1d` reads as a
    move. Under the raw denominator it also deflates that day's label,
    manufacturing the reversal the feature predicts.

    The median does not ignore the spike outright -- if the anchor quote WAS the
    median, removing it upward promotes its neighbour. What it does is bound the
    damage by the local dispersion instead of letting it scale with the spike.
    """
    day = date(2026, 1, 20)
    df = _panel()
    level = float(_row(df, "i0", day)["price"])

    def _spike(mult):
        s = df.copy()
        s.loc[(s["item_id"] == "i0") & (s["date"] == day), "price"] = level * mult
        return s

    monkeypatch.delenv("LABEL_SMOOTHED_ANCHOR", raising=False)
    raw = [_row(forecaster.prepare_targets(_spike(m), 3), "i0", day)["target_return_3d"] for m in (1.0, 1.4, 10.0)]
    monkeypatch.setenv("LABEL_SMOOTHED_ANCHOR", "1")
    sm = [_row(forecaster.prepare_targets(_spike(m), 3), "i0", day)["target_return_3d"] for m in (1.0, 1.4, 10.0)]

    # Raw: unbounded in the size of the spike, and tens of points at 1.4x alone.
    assert abs(raw[1] - raw[0]) > 20
    assert abs(raw[2] - raw[1]) > 50
    # Smoothed: once the spiked quote clears the window's maximum, making it
    # larger cannot move the median again. 1.4x and 10x give the same label.
    assert sm[2] == pytest.approx(sm[1], abs=1e-9)
    assert abs(sm[1] - sm[0]) < abs(raw[1] - raw[0]) / 10


# ------------------------------------------------------------------ the window


def test_an_observation_outside_the_span_does_not_vote(forecaster, smoothed):
    """MAX_WINDOW_SPAN_DAYS bounds the window in CALENDAR days, not rows.

    The unbounded `tail(3)` this rule replaced would anchor a sparsely observed
    item on prices months apart and serve their median as today's value.
    """
    day = date(2026, 2, 1)
    old = day - timedelta(days=MAX_WINDOW_SPAN_DAYS + 5)
    df = pd.DataFrame(
        [
            {"item_id": "a", "date": old, "price": 1000.0},
            {"item_id": "a", "date": day - timedelta(days=1), "price": 10.0},
            {"item_id": "a", "date": day, "price": 20.0},
            {"item_id": "a", "date": day + timedelta(days=3), "price": 30.0},
        ]
    )
    row = _row(forecaster.prepare_targets(df, horizon=3), "a", day)
    # In span: {10, 20} -> 15. An unbounded tail(3) would take {1000, 10, 20}
    # -> 20, which is also the raw anchor, so all three readings differ.
    assert row["target_return_3d"] == pytest.approx(100.0, abs=1e-9)


def test_only_the_last_smooth_window_observations_vote(forecaster, smoothed):
    """Four in-span observations, and the oldest must not count."""
    day = date(2026, 2, 8)
    rows = [
        {"item_id": "a", "date": day - timedelta(days=k), "price": p}
        for k, p in zip(range(SMOOTH_WINDOW, -1, -1), [1000.0, 10.0, 12.0, 20.0])
    ]
    rows.append({"item_id": "a", "date": day + timedelta(days=3), "price": 24.0})
    df = pd.DataFrame(rows)

    # {10, 12, 20} -> 12, against a raw anchor of 20 and an all-four median of 16.
    row = _row(forecaster.prepare_targets(df, horizon=3), "a", day)
    assert row["target_return_3d"] == pytest.approx(100.0, abs=1e-9)


def test_the_denominator_reads_nothing_after_the_anchor(forecaster, smoothed):
    """No lookahead. A trailing window that reached forward would put the
    outcome in its own denominator -- the defect `_resolve`'s `after=` guard
    exists for on the outcome leg.
    """
    df = _panel()
    day = date(2026, 1, 20)
    moved = df.copy()
    after = moved["date"] > day
    moved.loc[after, "price"] = moved.loc[after, "price"] * 3.0

    base = _row(forecaster.prepare_targets(df, 3), "i0", day)
    shifted = _row(forecaster.prepare_targets(moved, 3), "i0", day)
    # The target moved, so the return must. The denominator may not.
    den_base = base["target_3d"] / (1 + base["target_return_3d"] / 100)
    den_shift = shifted["target_3d"] / (1 + shifted["target_return_3d"] / 100)
    assert den_shift == pytest.approx(den_base, rel=1e-9)


# ------------------------------------------------------- what must not change


def test_the_target_column_stays_the_raw_future_price(forecaster, smoothed):
    """Only the denominator moves. `target_{h}d` is the observed price at
    `date + h`, and `_resolve`'s smoothed OUTCOME is a separate axis worth
    +0.0251 of the gap -- not this change.
    """
    df = _panel()
    day = date(2026, 1, 20)
    out = forecaster.prepare_targets(df, horizon=3)
    future = _row(df, "i0", day + timedelta(days=3))["price"]
    assert _row(out, "i0", day)["target_3d"] == pytest.approx(future, abs=1e-12)


def test_voiding_rules_still_fire(forecaster, smoothed):
    """A smoothed denominator does not launder a frozen run: a median of three
    identical prices is still that price, and the label is still fabricated.
    """
    df = _panel()
    frozen = [date(2026, 1, 11), date(2026, 1, 12)]
    held = float(_row(df, "i0", date(2026, 1, 10))["price"])
    mask = (df["item_id"] == "i0") & (df["date"].isin(frozen))
    df.loc[mask, "price"] = held

    out = forecaster.prepare_targets(df, horizon=3)
    assert pd.isna(_row(out, "i0", frozen[0])["target_return_3d"])
    assert pd.isna(_row(out, "i0", frozen[1])["target_return_3d"])
    assert forecaster.label_voiding["frozen_run_labels"][3] > 0


def test_zero_denominator_yields_no_label(forecaster, smoothed):
    """A zero median must not divide. The raw path guards this with
    `.replace(0, np.nan)`; the smoothed one needs its own guard.
    """
    day = date(2026, 2, 1)
    df = pd.DataFrame(
        [
            {"item_id": "a", "date": day, "price": 0.0},
            {"item_id": "a", "date": day + timedelta(days=3), "price": 5.0},
        ]
    )
    out = forecaster.prepare_targets(df, horizon=3)
    assert pd.isna(_row(out, "a", day)["target_return_3d"])


def test_row_order_is_preserved(forecaster, smoothed):
    """The denominator is computed on a sorted copy. If the sort leaked back
    into the returned frame, every caller's feature/label alignment would
    silently rotate.
    """
    df = _panel().sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = forecaster.prepare_targets(df, horizon=3)
    # prepare_targets sorts by (item, date) itself; what must hold is that a
    # row's label belongs to that row's item and date.
    row = _row(out, "i5", date(2026, 1, 20))
    den = float(np.median(_series(df, "i5", date(2026, 1, 20))[-SMOOTH_WINDOW:]))
    assert row["target_return_3d"] == pytest.approx((row["target_3d"] - den) / den * 100, abs=1e-9)
