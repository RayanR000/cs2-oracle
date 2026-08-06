# Market-Relative Direction Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a default-off training path that trains the directional classifier on market-demeaned returns, plus a leak-free market forecast that rebuilds the absolute call, so the hypothesis can be measured on paired cold retrains without touching production.

**Architecture:** A new pure-function module (`backend/models/market_factor.py`) builds a chain-linked daily price index over the `>= $1` cohort and derives both the realized market factor `m[d, h]` and a history-only forecast `m̂[d, h]`. `ItemForecaster` gains a `market_relative_labels` flag; when set, it joins `m` onto the training frame by date, demeans the classifier's label, matches the flat-class share to the control, and reports three extra CV metrics. A driver script runs both arms via `FORECAST_MODEL_DIR`.

**Tech Stack:** Python 3.13 (3.11 in CI), pandas 2.3.3, numpy, LightGBM, pytest.

**Spec:** `docs/superpowers/specs/2026-08-06-market-relative-labels-design.md`

## Global Constraints

- **Production behaviour must stay byte-identical with the flag off.** `DEFAULT_MARKET_RELATIVE_LABELS = False`. Every new code path is guarded.
- Run all commands from `backend/` through the venv: `venv/bin/python`.
- `MIN_INDEX_ITEMS = 30`, `INDEX_TOLERANCE_DAYS = 3`, `MARKET_MIN_PRICE_USD = 1.0`, `TRAILING_DRIFT_DAYS = 180`, `TRAILING_K_WINDOWS = 4`.
- `DIRECTION_FLAT_TOLERANCE_PCT = 0.5` — the *actual*-class yardstick never changes, in either arm. Only the *training label's* band is rematched.
- `HEADLINE_MIN_TIER = 1` marks the `>= $1` cohort. Already imported in `forecaster.py`.
- The market factor is **future information**. It must never enter `feature_cols`. Task 3 Step 1 is the test that pins this.
- A row with no valid `m` is demeaned by **zero** (`e = r`), never dropped. Row counts must be identical across arms.
- Horizons are `[3, 7, 14, 30]` (`ItemForecaster.HORIZONS`).
- Non-trivial decisions get a dated note in `docs/changelog/` (AGENTS.md rule 4). That is Task 6.

---

### Task 1: The chain-linked market index

**Files:**
- Create: `backend/models/market_factor.py`
- Test: `backend/tests/test_market_factor.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build_market_index(price_df, min_items=30, min_price=1.0) -> pd.DataFrame`. Input frame needs columns `item_id`, `date`, `price`. Returns a frame indexed by `date` (sorted, unique) with columns `log_return: float`, `n_items: int`, `valid: bool`, `level: float`, `invalid_cum: int`. Module constants `MIN_INDEX_ITEMS`, `INDEX_TOLERANCE_DAYS`, `MARKET_MIN_PRICE_USD`, `TRAILING_DRIFT_DAYS`, `TRAILING_K_WINDOWS`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_market_factor.py`:

```python
"""Tests for the market factor used by the market-relative label experiment.

See docs/superpowers/specs/2026-08-06-market-relative-labels-design.md.
"""
import numpy as np
import pandas as pd
import pytest

from models.market_factor import (
    MIN_INDEX_ITEMS,
    build_market_index,
)


def _frame(rows):
    """rows: list of (item_id, 'YYYY-MM-DD', price)."""
    return pd.DataFrame(rows, columns=["item_id", "date", "price"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def _flat_panel(n_items, dates, price=10.0, start_id=0):
    return _frame([
        (start_id + i, d, price)
        for i in range(n_items)
        for d in dates
    ])


def test_index_is_flat_when_no_prices_move():
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    idx = build_market_index(_flat_panel(40, dates), min_items=5)
    assert list(idx.index) == list(pd.to_datetime(dates))
    # First date has no predecessor, so no return.
    assert not idx["valid"].iloc[0]
    assert idx["log_return"].iloc[1:].eq(0.0).all()
    assert np.allclose(idx["level"].iloc[1:], 1.0)


def test_index_tracks_a_uniform_ten_percent_move():
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 11.0))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["log_return"].iloc[1] == pytest.approx(np.log(1.1))
    assert idx["level"].iloc[1] == pytest.approx(1.1)


def test_median_ignores_a_single_outlier_item():
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 10.0 if i else 1000.0))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["log_return"].iloc[1] == pytest.approx(0.0)


def test_sub_dollar_items_do_not_move_the_index():
    """A penny item swinging 50% must not register. Its prior price is < $1."""
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 10.0))
    for j in range(100, 200):  # 100 penny items, all up 50%
        rows.append((j, "2026-01-01", 0.03))
        rows.append((j, "2026-01-02", 0.045))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["log_return"].iloc[1] == pytest.approx(0.0)
    assert idx["n_items"].iloc[1] == 40


def test_thin_days_are_marked_invalid_not_dropped():
    rows = []
    for i in range(3):  # only 3 pairs, below min_items
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 12.0))
    idx = build_market_index(_frame(rows), min_items=MIN_INDEX_ITEMS)
    assert pd.to_datetime("2026-01-02") in idx.index
    assert not idx.loc[pd.to_datetime("2026-01-02"), "valid"]
    assert np.isnan(idx.loc[pd.to_datetime("2026-01-02"), "log_return"])


def test_an_invalid_day_does_not_poison_later_levels():
    """One thin day must not NaN the whole tail of the index."""
    rows = []
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-03", 10.0))
    for i in range(2):  # 2026-01-02 is thin
        rows.append((i, "2026-01-02", 10.0))
    idx = build_market_index(_frame(rows), min_items=MIN_INDEX_ITEMS)
    assert np.isfinite(idx["level"].iloc[-1])
    # invalid_cum lets a consumer detect that a window spanned the bad day.
    assert idx["invalid_cum"].iloc[-1] > idx["invalid_cum"].iloc[0]


def test_items_entering_midway_do_not_create_a_jump():
    """Chain-linking: a new item joining at a different price level must not
    move the index, because only paired day-over-day returns contribute."""
    rows = []
    for i in range(40):
        for d in ["2026-01-01", "2026-01-02", "2026-01-03"]:
            rows.append((i, d, 10.0))
    for j in range(100, 140):  # join on day 2 at a much higher level
        rows.append((j, "2026-01-02", 500.0))
        rows.append((j, "2026-01-03", 500.0))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["log_return"].iloc[1] == pytest.approx(0.0)
    assert idx["log_return"].iloc[2] == pytest.approx(0.0)
    assert idx["n_items"].iloc[1] == 40   # entrant has no prior day
    assert idx["n_items"].iloc[2] == 80


def test_non_positive_prices_are_excluded():
    rows = []
    for i in range(40):
        rows.append((i, "2026-01-01", 10.0))
        rows.append((i, "2026-01-02", 10.0))
    rows.append((999, "2026-01-01", 0.0))
    rows.append((999, "2026-01-02", 5.0))
    idx = build_market_index(_frame(rows), min_items=5)
    assert idx["n_items"].iloc[1] == 40


def test_empty_frame_returns_empty_index():
    idx = build_market_index(_frame([]), min_items=5)
    assert idx.empty
    assert list(idx.columns) == [
        "log_return", "n_items", "valid", "level", "invalid_cum"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_market_factor.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'models.market_factor'`.

- [ ] **Step 3: Write the implementation**

Create `backend/models/market_factor.py`:

```python
"""Market factor for the market-relative direction-label experiment.

An item's forward return decomposes as ``r = m + e``: a market-wide move plus
an idiosyncratic residual. The directional classifier is trained on ``r``,
whose variance is dominated by ``m`` -- a term the 36 price technicals carry
no information about, so the model settles on a near-constant tilt (it calls
"down" on 57-87% of rows regardless of date, see
docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md).

This module supplies ``m`` (for demeaning the label) and ``m_hat`` (a
history-only forecast, for rebuilding the absolute call). Pure functions over
DataFrames: no DB, no I/O, no ItemForecaster internals, so the factor can be
reasoned about and tested without loading a 5,600-line module.

Design: docs/superpowers/specs/2026-08-06-market-relative-labels-design.md
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fewest paired items a day needs before its index return is trusted. Below
# this the cross-sectional median is a handful of draws.
MIN_INDEX_ITEMS = 30

# The archive is missing whole days (docs: aggregator-archive-day-gaps), so a
# horizon window's end date may not exist. Resolve to the next available date
# within this many days, matching the LAG_TOLERANCE_DAYS convention already
# used for lag features.
INDEX_TOLERANCE_DAYS = 3

# Only items priced at or above this contribute. This is the served cohort
# (MIN_SERVED_PRICE_USD in api/serving_policy.py, equal to HEADLINE_MIN_TIER).
# Below it a one-cent tick on a $0.03 item is a 33% move, so the penny tier's
# median is rounding noise rather than a market signal.
MARKET_MIN_PRICE_USD = 1.0

# Trailing window for the pre-registered drift estimator.
TRAILING_DRIFT_DAYS = 180

# Number of non-overlapping resolved windows for the trailing-median
# diagnostic estimator.
TRAILING_K_WINDOWS = 4

_INDEX_COLUMNS = ["log_return", "n_items", "valid", "level", "invalid_cum"]


def build_market_index(price_df: pd.DataFrame,
                       min_items: int = MIN_INDEX_ITEMS,
                       min_price: float = MARKET_MIN_PRICE_USD) -> pd.DataFrame:
    """Chain-linked equal-weighted daily price index over the >= $1 cohort.

    Chain-linked (median of paired day-over-day log returns, cumulated) rather
    than a median of h-day returns, for three reasons: items enter and leave
    the archive and only day-level pairing handles that without biasing toward
    long-history cheap items; one groupby.shift yields every horizon; and the
    daily return series is exactly what ``forecast_market_factor`` needs.

    ``min_price`` is applied to the *prior* day's price, so an item's inclusion
    is decided by information available before the return it contributes.

    Returns a frame indexed by date (sorted, unique) with:
      log_return  median paired daily log return; NaN when < ``min_items``
      n_items     contributing pairs
      valid       log_return is finite
      level       exp(cumsum(log_return treating invalid days as 0)). Invalid
                  days are bridged rather than propagated so one thin day does
                  not NaN the entire tail; ``invalid_cum`` is how a consumer
                  detects that a window spanned one.
      invalid_cum running count of invalid days
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=_INDEX_COLUMNS,
                            index=pd.DatetimeIndex([], name="date"))

    df = price_df[["item_id", "date", "price"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0].dropna(subset=["price"])
    if df.empty:
        return pd.DataFrame(columns=_INDEX_COLUMNS,
                            index=pd.DatetimeIndex([], name="date"))

    # One row per (item, date). Duplicate sources are already voted upstream,
    # but a stray duplicate would double-count an item in the median.
    df = df.groupby(["item_id", "date"], as_index=False)["price"].mean()
    df = df.sort_values(["item_id", "date"])

    grp = df.groupby("item_id", sort=False)
    df["prev_price"] = grp["price"].shift(1)
    df["prev_date"] = grp["date"].shift(1)

    # Consecutive calendar days only. A gap means the "daily" return would
    # silently span multiple days and overstate that day's move.
    pairs = df[
        df["prev_price"].notna()
        & (df["prev_price"] >= min_price)
        & ((df["date"] - df["prev_date"]).dt.days == 1)
    ].copy()
    pairs["log_return"] = np.log(pairs["price"] / pairs["prev_price"])

    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()), name="date")
    agg = pairs.groupby("date")["log_return"].agg(["median", "size"])
    out = pd.DataFrame(index=all_dates)
    out["log_return"] = agg["median"].reindex(all_dates)
    out["n_items"] = agg["size"].reindex(all_dates).fillna(0).astype(int)
    out.loc[out["n_items"] < min_items, "log_return"] = np.nan
    out["valid"] = out["log_return"].notna()
    out["level"] = np.exp(out["log_return"].fillna(0.0).cumsum())
    out["invalid_cum"] = (~out["valid"]).cumsum().astype(int)
    return out[_INDEX_COLUMNS]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_market_factor.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/models/market_factor.py backend/tests/test_market_factor.py
git commit -m "feat: chain-linked market index over the >=\$1 cohort"
```

---

### Task 2: Realized factor, leak-free forecast, and diagnostics

**Files:**
- Modify: `backend/models/market_factor.py`
- Test: `backend/tests/test_market_factor.py`

**Interfaces:**
- Consumes: `build_market_index` from Task 1, and the module constants.
- Produces:
  - `market_factor_for_horizon(index, horizon, tolerance_days=3) -> pd.Series` — percent, indexed by date, NaN where unresolvable.
  - `forecast_market_factor(index, as_of, horizon, trailing_days=180) -> float` — percent; the pre-registered estimator. Returns `0.0` when there is no usable history.
  - `forecast_market_factor_diagnostics(index, as_of, horizon) -> dict` with keys `trailing_drift`, `trailing_k_median`, `past_h_momentum`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_market_factor.py`:

```python
from models.market_factor import (
    INDEX_TOLERANCE_DAYS,
    forecast_market_factor,
    forecast_market_factor_diagnostics,
    market_factor_for_horizon,
)


def _index_from_daily(returns, start="2026-01-01"):
    """Build an index directly from a list of daily log returns."""
    dates = pd.date_range(start, periods=len(returns), freq="D")
    out = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    out["log_return"] = returns
    out["n_items"] = 100
    out["valid"] = out["log_return"].notna()
    out["level"] = np.exp(out["log_return"].fillna(0.0).cumsum())
    out["invalid_cum"] = (~out["valid"]).cumsum().astype(int)
    return out


def test_factor_is_the_percent_move_over_the_window():
    idx = _index_from_daily([np.nan] + [np.log(1.01)] * 10)
    m = market_factor_for_horizon(idx, horizon=3)
    d0 = pd.Timestamp("2026-01-01")
    # 3 days of +1% compounding = 1.01^3 - 1
    assert m.loc[d0] == pytest.approx((1.01 ** 3 - 1) * 100)


def test_factor_is_nan_past_the_end_of_the_index():
    idx = _index_from_daily([np.nan] + [0.0] * 4)
    m = market_factor_for_horizon(idx, horizon=3)
    assert np.isnan(m.iloc[-1])
    assert np.isnan(m.iloc[-2])


def test_factor_resolves_a_short_calendar_gap_within_tolerance():
    """The window end date is missing but a date 2 days later exists."""
    dates = pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06"])
    idx = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    idx["log_return"] = [np.nan, 0.0, 0.0, np.log(1.05)]
    idx["n_items"] = 100
    idx["valid"] = idx["log_return"].notna()
    idx["level"] = np.exp(idx["log_return"].fillna(0.0).cumsum())
    idx["invalid_cum"] = (~idx["valid"]).cumsum().astype(int)
    # From 01-02, +3d lands on 01-05 which is absent; 01-06 is 1 day later.
    m = market_factor_for_horizon(idx, horizon=3, tolerance_days=3)
    assert m.loc[pd.Timestamp("2026-01-02")] == pytest.approx(5.0)


def test_factor_is_nan_when_the_gap_exceeds_tolerance():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-20"])
    idx = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    idx["log_return"] = [np.nan, 0.0, 0.0]
    idx["n_items"] = 100
    idx["valid"] = idx["log_return"].notna()
    idx["level"] = np.exp(idx["log_return"].fillna(0.0).cumsum())
    idx["invalid_cum"] = (~idx["valid"]).cumsum().astype(int)
    m = market_factor_for_horizon(idx, horizon=3, tolerance_days=INDEX_TOLERANCE_DAYS)
    assert np.isnan(m.loc[pd.Timestamp("2026-01-02")])


def test_factor_is_nan_when_the_window_spans_an_invalid_day():
    rets = [np.nan] + [0.0] * 3 + [np.nan] + [0.0] * 5   # index 4 is thin
    idx = _index_from_daily(rets)
    m = market_factor_for_horizon(idx, horizon=3)
    # 2026-01-02 (i=1) -> 2026-01-05 (i=4) spans the invalid day.
    assert np.isnan(m.loc[pd.Timestamp("2026-01-02")])
    # 2026-01-06 (i=5) -> 2026-01-09 (i=8) does not.
    assert m.loc[pd.Timestamp("2026-01-06")] == pytest.approx(0.0)


def test_degenerate_all_flat_market_gives_zero_not_nan():
    idx = _index_from_daily([np.nan] + [0.0] * 10)
    m = market_factor_for_horizon(idx, horizon=3)
    assert m.loc[pd.Timestamp("2026-01-02")] == pytest.approx(0.0)


# --- the leakage tests: the single most important thing in this file ---

def test_forecast_ignores_everything_after_as_of():
    """Truncating the index at `as_of` must not change the forecast. If it
    does, the estimator is reading the future and any measured gain is fake."""
    rng = np.random.RandomState(0)
    rets = [np.nan] + list(rng.normal(0, 0.01, 400))
    idx = _index_from_daily(rets)
    as_of = idx.index[300]
    full = forecast_market_factor(idx, as_of, horizon=7)
    truncated = forecast_market_factor(idx.loc[:as_of], as_of, horizon=7)
    assert full == pytest.approx(truncated)


def test_forecast_is_unchanged_when_the_future_is_nulled():
    rng = np.random.RandomState(1)
    rets = [np.nan] + list(rng.normal(0, 0.01, 400))
    idx = _index_from_daily(rets)
    as_of = idx.index[300]
    before = forecast_market_factor(idx, as_of, horizon=14)
    poisoned = idx.copy()
    poisoned.loc[poisoned.index > as_of, "log_return"] = 99.0
    poisoned["level"] = np.exp(poisoned["log_return"].fillna(0.0).cumsum())
    after = forecast_market_factor(poisoned, as_of, horizon=14)
    assert before == pytest.approx(after)


def test_all_diagnostic_estimators_ignore_the_future():
    rng = np.random.RandomState(2)
    rets = [np.nan] + list(rng.normal(0, 0.01, 400))
    idx = _index_from_daily(rets)
    as_of = idx.index[300]
    before = forecast_market_factor_diagnostics(idx, as_of, horizon=7)
    poisoned = idx.copy()
    poisoned.loc[poisoned.index > as_of, "log_return"] = 99.0
    poisoned["level"] = np.exp(poisoned["log_return"].fillna(0.0).cumsum())
    after = forecast_market_factor_diagnostics(poisoned, as_of, horizon=7)
    assert set(before) == {
        "trailing_drift", "trailing_k_median", "past_h_momentum"}
    for key in before:
        assert before[key] == pytest.approx(after[key]), key


def test_forecast_scales_a_constant_drift_to_the_horizon():
    daily = np.log(1.001)
    idx = _index_from_daily([np.nan] + [daily] * 300)
    as_of = idx.index[-1]
    got = forecast_market_factor(idx, as_of, horizon=7)
    assert got == pytest.approx((np.exp(7 * daily) - 1) * 100)


def test_forecast_returns_zero_with_no_usable_history():
    idx = _index_from_daily([np.nan, np.nan])
    assert forecast_market_factor(idx, idx.index[-1], horizon=7) == 0.0


def test_forecast_before_the_index_starts_returns_zero():
    idx = _index_from_daily([np.nan] + [0.0] * 5)
    assert forecast_market_factor(
        idx, pd.Timestamp("2020-01-01"), horizon=7) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_market_factor.py -q`
Expected: `ImportError: cannot import name 'market_factor_for_horizon'`.

- [ ] **Step 3: Write the implementation**

Append to `backend/models/market_factor.py`:

```python
def market_factor_for_horizon(index: pd.DataFrame, horizon: int,
                              tolerance_days: int = INDEX_TOLERANCE_DAYS
                              ) -> pd.Series:
    """Realized market move over ``d -> d + horizon``, in percent.

    NaN when the window's end date is absent beyond ``tolerance_days``, or when
    the window spans a day whose cross-section was too thin to trust. NaN is
    *not* a reason to drop the row -- callers demean such rows by zero, because
    changing row counts between arms would break the paired comparison this
    exists to serve.

    This reads future index levels by construction: it is a label, not a
    feature. ``forecast_market_factor`` is the serve-time counterpart.
    """
    if index is None or index.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))

    dates = index.index
    targets = dates + pd.Timedelta(days=horizon)
    # searchsorted "left" gives the first available date >= target.
    pos = np.searchsorted(dates.values, targets.values, side="left")

    n = len(dates)
    in_range = pos < n
    end_pos = np.where(in_range, np.minimum(pos, n - 1), n - 1)
    end_dates = dates[end_pos]
    within = in_range & (
        (end_dates - targets).days <= tolerance_days)

    level = index["level"].to_numpy(dtype=float)
    invalid_cum = index["invalid_cum"].to_numpy()
    start_level = level
    end_level = level[end_pos]
    # A window that steps over a thin day carries an index move that was never
    # measured; only the count of invalid days can detect it after bridging.
    spans_invalid = invalid_cum[end_pos] != invalid_cum

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (end_level / start_level - 1.0) * 100.0
    pct = np.where(within & ~spans_invalid, pct, np.nan)
    return pd.Series(pct, index=dates, name=f"market_factor_{horizon}d")


def _returns_as_of(index: pd.DataFrame, as_of) -> pd.Series:
    """Valid daily log returns at or before ``as_of``. The single choke point
    through which every estimator sees history, so none of them can read the
    future by accident."""
    if index is None or index.empty:
        return pd.Series(dtype=float)
    as_of = pd.Timestamp(as_of)
    window = index.loc[index.index <= as_of]
    if window.empty:
        return pd.Series(dtype=float)
    return window.loc[window["valid"], "log_return"].astype(float)


def forecast_market_factor(index: pd.DataFrame, as_of, horizon: int,
                           trailing_days: int = TRAILING_DRIFT_DAYS) -> float:
    """History-only forecast of ``m[as_of, horizon]``, in percent.

    Pre-registered estimator: the trailing ``trailing_days`` mean daily index
    log-return, compounded to the horizon. This is the honest analogue of the
    always-down constant that currently beats the model -- a slow-moving
    unconditional drift, estimated only from data available at ``as_of``.

    Returns 0.0 (a flat market) when there is no usable history, which is the
    neutral choice: it reduces the reconstruction to the residual call alone.
    """
    rets = _returns_as_of(index, as_of)
    if rets.empty:
        return 0.0
    recent = rets.iloc[-trailing_days:]
    if recent.empty:
        return 0.0
    return float((np.exp(horizon * recent.mean()) - 1.0) * 100.0)


def forecast_market_factor_diagnostics(index: pd.DataFrame, as_of,
                                       horizon: int) -> dict:
    """Two alternate estimators alongside the pre-registered one.

    Reported so the estimator choice cannot be blamed for a negative result,
    but explicitly excluded from the decision rule: picking the best of three
    after seeing the outcome is how a null result becomes a false positive.
    """
    rets = _returns_as_of(index, as_of)
    out = {
        "trailing_drift": forecast_market_factor(index, as_of, horizon),
        "trailing_k_median": 0.0,
        "past_h_momentum": 0.0,
    }
    if rets.empty:
        return out

    # K non-overlapping resolved windows immediately before `as_of`.
    needed = TRAILING_K_WINDOWS * horizon
    tail = rets.iloc[-needed:]
    if len(tail) >= horizon:
        usable = len(tail) - (len(tail) % horizon)
        blocks = tail.iloc[-usable:].to_numpy().reshape(-1, horizon).sum(axis=1)
        out["trailing_k_median"] = float(
            (np.exp(np.median(blocks)) - 1.0) * 100.0)

    window = rets.iloc[-horizon:]
    if not window.empty:
        out["past_h_momentum"] = float(
            (np.exp(window.sum()) - 1.0) * 100.0)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_market_factor.py -q`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/models/market_factor.py backend/tests/test_market_factor.py
git commit -m "feat: realized market factor and leak-free market forecast"
```

---

### Task 3: Join the factor onto the training frame, default off

**Files:**
- Modify: `backend/models/forecaster.py` (`__init__` ~line 418; `build_training_data` ~line 2846; `_select_feature_cols` ~line 3776)
- Test: `backend/tests/test_market_relative_labels.py`

**Interfaces:**
- Consumes: `build_market_index`, `market_factor_for_horizon` from Tasks 1-2.
- Produces: `ItemForecaster(..., market_relative_labels: bool = False)`; attribute `self.market_relative_labels: bool`; attribute `self.market_index: Optional[pd.DataFrame]` (set by `build_training_data`, `None` when the flag is off); columns `market_factor_{h}d` on the training frame for `h in HORIZONS`, present **only** when the flag is on, and never in `feature_cols`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_market_relative_labels.py`:

```python
"""Tests for the default-off market-relative label path.

Design: docs/superpowers/specs/2026-08-06-market-relative-labels-design.md
"""
import numpy as np
import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def test_flag_defaults_to_off():
    f = ItemForecaster(db_session=None)
    assert f.market_relative_labels is False
    assert f.market_index is None


def test_flag_can_be_set():
    f = ItemForecaster(db_session=None, market_relative_labels=True)
    assert f.market_relative_labels is True


def test_market_factor_columns_are_never_features():
    """The factor is built from future prices. If it reaches feature_cols the
    model trains on the answer and every metric downstream is meaningless."""
    f = ItemForecaster(db_session=None)
    df = pd.DataFrame({
        "item_id": [1, 2],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "price": [10.0, 11.0],
        "price_lag_1d": [9.0, 10.0],
        "target_3d": [11.0, 12.0],
        "target_return_3d": [10.0, 9.0],
        "market_factor_3d": [1.0, 2.0],
        "market_factor_7d": [1.5, 2.5],
        "market_factor_14d": [2.0, 3.0],
        "market_factor_30d": [2.5, 3.5],
    })
    cols = f._select_feature_cols(df, ItemForecaster.HORIZONS,
                                  ItemForecaster.SHELVED_FEATURES)
    for h in ItemForecaster.HORIZONS:
        assert f"market_factor_{h}d" not in cols
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_market_relative_labels.py -q`
Expected: `TypeError: __init__() got an unexpected keyword argument 'market_relative_labels'` and an `AttributeError` on `market_relative_labels`.

- [ ] **Step 3: Write the implementation**

In `backend/models/forecaster.py`, add the import near the other `models.` imports at the top of the file:

```python
from models.market_factor import (
    build_market_index,
    forecast_market_factor,
    market_factor_for_horizon,
)
```

Change the constructor signature (currently ends `served_cohort_share: Optional[float] = None`):

```python
    def __init__(self, db_session, model_dir: str = None, prune_failed_groups: bool = True,
                 served_cohort_share: Optional[float] = None,
                 market_relative_labels: bool = False):
```

and add, immediately after the `self.served_cohort_share = served_cohort_share` line:

```python
        # Train the directional classifier on market-demeaned returns
        # (r - m) instead of raw returns, and rebuild the absolute call from a
        # history-only market forecast. False = byte-identical to the
        # pre-2026-08-06 model. Set from TRAIN_MARKET_RELATIVE_LABELS by
        # scripts/forecast_prices.py. See
        # docs/superpowers/specs/2026-08-06-market-relative-labels-design.md.
        self.market_relative_labels = bool(market_relative_labels)
        # Daily chain-linked market index, built by build_training_data when
        # the flag is on. Needed at fold time for the leak-free m_hat, so it
        # cannot be a local.
        self.market_index: Optional[pd.DataFrame] = None
```

In `_select_feature_cols`, next to the existing target exclusions (currently
`exclude |= {f"target_return_{h}d" for h in horizons}`), add:

```python
        # The market factor is computed from other items' FUTURE prices. It is
        # a label input, never a feature -- if it reaches feature_cols the
        # model trains on the answer.
        exclude |= {f"market_factor_{h}d" for h in horizons}
```

In `build_training_data`, insert the index build **before** the subsample
(immediately after the `corrupt_items = self._flag_corrupt_items(price_df)`
line):

```python
        # The market index must be measured on the FULL frame, before
        # subsampling. _stratified_item_subsample cuts to ~99 items to hit the
        # row budget, which leaves roughly 20 priced >= $1 per date -- far too
        # thin for a cross-sectional median. The index is a per-date aggregate
        # (~1,460 rows), so carrying it past the subsample costs nothing.
        if self.market_relative_labels:
            _tm = datetime.now()
            self.market_index = build_market_index(
                price_df[~price_df["item_id"].isin(corrupt_items)])
            n_valid = int(self.market_index["valid"].sum()) if not self.market_index.empty else 0
            logger.info(
                f"  market index: {len(self.market_index):,} dates, "
                f"{n_valid:,} valid, "
                f"took {(datetime.now() - _tm).total_seconds():.0f}s")
```

Then, immediately after the `df = self._add_supply_depth_features(df)` line
(so the columns exist before `_select_feature_cols` runs and excludes them):

```python
        # Join the realized market factor on by date, one column per horizon.
        # Excluded from feature_cols by _select_feature_cols; consumed only by
        # the directional classifier's label.
        if self.market_relative_labels and self.market_index is not None \
                and not self.market_index.empty:
            join_dates = pd.to_datetime(df["date"])
            for h in self.HORIZONS:
                factor = market_factor_for_horizon(self.market_index, h)
                df[f"market_factor_{h}d"] = join_dates.map(factor).astype(float)
                cov = float(df[f"market_factor_{h}d"].notna().mean()) * 100
                logger.info(f"  market factor {h}d coverage: {cov:.1f}% of rows")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_market_relative_labels.py -q`
Expected: 3 passed.

- [ ] **Step 5: Verify the control arm is untouched**

Run: `venv/bin/python -m pytest tests/test_forecaster.py tests/test_served_cohort_weighting.py -q`
Expected: all pass. These cover the default construction path.

- [ ] **Step 6: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_market_relative_labels.py
git commit -m "feat: join the market factor onto the training frame, default off"
```

---

### Task 4: Demean the classifier label and report the new metrics

**Files:**
- Modify: `backend/models/forecaster.py` (new static helpers near `_direction_classes` ~line 3814; `_train_horizon_inline` classifier call ~line 3413; `_cv_evaluate_horizon` metric block ~line 5057-5108)
- Test: `backend/tests/test_market_relative_labels.py`

**Interfaces:**
- Consumes: `self.market_relative_labels`, `self.market_index`, `market_factor_{h}d` columns from Task 3; `forecast_market_factor` from Task 2.
- Produces:
  - `ItemForecaster._matched_flat_band(residuals, flat_share) -> float`
  - `ItemForecaster._demean_returns(returns, factor) -> np.ndarray`
  - `ItemForecaster._residual_point_estimate(pred_cls, band) -> np.ndarray`
  - Three new keys in each `fold_metrics` dict: `relative_accuracy_ge1`, `market_factor_coverage`, `market_relative_classifier_accuracy_ge1`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_market_relative_labels.py`:

```python
def test_demean_subtracts_the_factor():
    got = ItemForecaster._demean_returns(
        np.array([5.0, -2.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert got == pytest.approx([4.0, -3.0, -1.0])


def test_demean_treats_a_missing_factor_as_zero():
    """A NaN factor must fall back to the control's own label, never drop the
    row -- changing row counts between arms breaks the paired comparison."""
    got = ItemForecaster._demean_returns(
        np.array([5.0, -2.0]), np.array([1.0, np.nan]))
    assert got == pytest.approx([4.0, -2.0])


def test_demean_handles_a_none_factor():
    got = ItemForecaster._demean_returns(np.array([5.0, -2.0]), None)
    assert got == pytest.approx([5.0, -2.0])


def test_matched_flat_band_reproduces_the_target_share():
    rng = np.random.RandomState(0)
    resid = rng.normal(0, 3.0, 10_000)
    band = ItemForecaster._matched_flat_band(resid, 0.30)
    got = float((np.abs(resid) <= band).mean())
    assert got == pytest.approx(0.30, abs=0.01)


def test_matched_flat_band_ignores_nans():
    resid = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    band = ItemForecaster._matched_flat_band(resid, 0.5)
    assert np.isfinite(band)


def test_matched_flat_band_falls_back_on_an_empty_input():
    from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT
    band = ItemForecaster._matched_flat_band(np.array([]), 0.3)
    assert band == DIRECTION_FLAT_TOLERANCE_PCT


def test_residual_point_estimate_maps_classes_to_signed_band_edges():
    got = ItemForecaster._residual_point_estimate(
        np.array([0, 1, 2]), band=2.5)
    assert got == pytest.approx([-2.5, 0.0, 2.5])


def test_reconstruction_with_a_flat_market_reduces_to_the_residual_call():
    """m_hat = 0 must give back exactly the residual arm's own direction."""
    from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT
    pred_cls = np.array([0, 1, 2])
    band = 3.0
    e_hat = ItemForecaster._residual_point_estimate(pred_cls, band)
    r_hat = 0.0 + e_hat
    rebuilt = ItemForecaster._direction_classes(
        r_hat, DIRECTION_FLAT_TOLERANCE_PCT)
    assert list(rebuilt) == [0, 1, 2]


def test_reconstruction_shifts_the_call_when_the_market_forecast_is_large():
    from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT
    e_hat = ItemForecaster._residual_point_estimate(np.array([0]), band=1.0)
    # Residual says down by 1%, but the market is forecast up 5%.
    r_hat = 5.0 + e_hat
    rebuilt = ItemForecaster._direction_classes(
        r_hat, DIRECTION_FLAT_TOLERANCE_PCT)
    assert list(rebuilt) == [2]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_market_relative_labels.py -q`
Expected: `AttributeError: type object 'ItemForecaster' has no attribute '_demean_returns'`.

- [ ] **Step 3: Write the helpers**

In `backend/models/forecaster.py`, add immediately after `_direction_classes`:

```python
    @staticmethod
    def _demean_returns(returns, factor) -> np.ndarray:
        """Subtract the market factor from % returns, giving the idiosyncratic
        residual ``e = r - m``.

        A missing factor demeans by zero rather than producing NaN. Dropping
        those rows would change n_train/n_val in the treatment arm only, and
        the paired comparison this whole path exists to serve depends on the
        two arms seeing identical rows.
        """
        r = np.asarray(returns, dtype=float)
        if factor is None:
            return r.copy()
        m = np.asarray(factor, dtype=float)
        return r - np.nan_to_num(m, nan=0.0)

    @staticmethod
    def _matched_flat_band(residuals, flat_share: float) -> float:
        """Flat-band edge placing ``flat_share`` of ``residuals`` in the flat
        class.

        Residuals are less dispersed than raw returns, so reusing the fixed
        ±DIRECTION_FLAT_TOLERANCE_PCT band would inflate the flat class, push
        the classifier toward always predicting flat, and read as "relabelling
        hurt" when the real cause is a class-balance artifact. Matching the
        control arm's flat share holds that confounder fixed.

        This is a controlled confounder, not a tuned hyperparameter: the target
        share is read off the control arm and never optimized against the
        outcome.
        """
        e = np.asarray(residuals, dtype=float)
        e = e[np.isfinite(e)]
        if e.size == 0:
            return DIRECTION_FLAT_TOLERANCE_PCT
        share = float(np.clip(flat_share, 0.0, 1.0))
        return float(np.quantile(np.abs(e), share))

    @staticmethod
    def _residual_point_estimate(pred_cls, band: float) -> np.ndarray:
        """Signed magnitude implied by a predicted residual class.

        down(0) -> -band, flat(1) -> 0, up(2) -> +band. The classifier emits a
        class, not a magnitude, so the band edge is the only defensible
        representative. Coarse by construction: it lets the market forecast
        shift the decision boundary but not rank within a class. Accepted for
        this measurement rather than building a residual quantile model to test
        a hypothesis stage 1 may kill outright.
        """
        c = np.asarray(pred_cls, dtype=int)
        return np.select([c == 0, c == 2], [-float(band), float(band)], 0.0)
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_market_relative_labels.py -q`
Expected: 12 passed.

- [ ] **Step 5: Commit the helpers**

```bash
git add backend/models/forecaster.py backend/tests/test_market_relative_labels.py
git commit -m "feat: demeaning, flat-band matching and residual point estimate"
```

- [ ] **Step 6: Wire the demeaned label into the production trainer**

In `_train_horizon_inline`, replace the `self.direction_models[horizon] = self._fit_direction_classifier(...)` call with:

```python
            _dir_y_train, _dir_y_val = y_train, y_val
            if self.market_relative_labels:
                _mf_col = f"market_factor_{horizon}d"
                _dir_y_train = self._demean_returns(
                    y_train, train_set[_mf_col] if _mf_col in train_set else None)
                _dir_y_val = self._demean_returns(
                    y_val, val_set[_mf_col] if _mf_col in val_set else None)
                _ctl_flat = float(
                    (self._direction_classes(y_train) == 1).mean())
                _band = self._matched_flat_band(_dir_y_train, _ctl_flat)
                self.direction_bands[horizon] = _band
                logger.info(
                    f"  {horizon}d market-relative labels: control flat share "
                    f"{_ctl_flat * 100:.1f}%, matched residual band ±{_band:.3f}%")
            self.direction_models[horizon] = self._fit_direction_classifier(
                X_train, _dir_y_train, X_val, _dir_y_val, boosting_type,
                self._direction_tree_params(per_quantile_params),
                horizon=horizon,
                sigma_train=None,
                sigma_val=None,
                flat_band=self.direction_bands.get(horizon),
                tier_train=(train_set["price_tier"].to_numpy()
                            if "price_tier" in train_set.columns else None),
            )
```

Add `self.direction_bands: Dict[int, float] = {}` in `__init__`, immediately
after the `self.direction_models` line.

Give `_fit_direction_classifier` a `flat_band: Optional[float] = None`
parameter, and change its `_thr` helper so an explicit band wins:

```python
        def _thr(sigma):
            if flat_band is not None:
                return float(flat_band)
            if sigma is None:
                return DIRECTION_FLAT_TOLERANCE_PCT
            return self._direction_threshold(np.asarray(sigma, dtype=float),
                                              horizon, k, floor, cap)
```

- [ ] **Step 7: Wire the metrics into CV**

In `_cv_evaluate_horizon`, replace the block from the `clf = self._fit_direction_classifier(...)` call through the construction of `classifier_acc_ge1` with:

```python
            # Market-relative arm: train the classifier on r - m, then rebuild
            # the absolute call from a history-only market forecast. Off by
            # default, in which case every line below reduces to the control.
            _mf_col = f"market_factor_{horizon}d"
            _dir_y_train, _dir_y_val = y_train, y_val
            _band = None
            _mf_val = None
            mf_coverage = None
            if self.market_relative_labels:
                _mf_train = train_df[_mf_col] if _mf_col in train_df else None
                _mf_val = (val_df[_mf_col].to_numpy(dtype=float)
                           if _mf_col in val_df else None)
                _dir_y_train = self._demean_returns(y_train, _mf_train)
                _dir_y_val = self._demean_returns(y_val, _mf_val)
                _ctl_flat = float((self._direction_classes(y_train) == 1).mean())
                _band = self._matched_flat_band(_dir_y_train, _ctl_flat)
                mf_coverage = (round(float(np.isfinite(_mf_val).mean()) * 100, 1)
                               if _mf_val is not None else 0.0)

            clf = self._fit_direction_classifier(
                X_train, _dir_y_train, X_val, _dir_y_val, self.BOOSTING_TYPE,
                self._direction_tree_params(per_quantile_params),
                horizon=horizon,
                sigma_train=None,
                sigma_val=None,
                flat_band=_band,
                tier_train=(train_df["price_tier"].to_numpy()
                            if "price_tier" in train_df.columns else None))
            pred_cls = clf.predict(X_val).argmax(axis=1)
            actual_cls = self._direction_classes(actual_returns)  # FIXED ±0.5% yardstick

            relative_acc_ge1 = None
            if self.market_relative_labels:
                # Stage 1: can the model call the idiosyncratic direction at
                # all? If this is ~50% everywhere, no market model rescues it.
                actual_rel_cls = self._direction_classes(_dir_y_val, _band)
                # Stage 2: rebuild the absolute call as m_hat + e_hat, scored
                # against the same fixed yardstick the control uses.
                e_hat = self._residual_point_estimate(pred_cls, _band)
                m_hat = np.array([
                    forecast_market_factor(self.market_index, d, horizon)
                    for d in pd.to_datetime(val_df["date"]).to_numpy()
                ]) if self.market_index is not None else np.zeros(len(val_df))
                pred_cls = self._direction_classes(m_hat + e_hat)

            classifier_acc = round(float((pred_cls == actual_cls).mean()) * 100, 1)

            # The same accuracy again over the production cohort only. The
            # figure above pools every price tier and the training frame is
            # ~83% tier-0, so it is approximately the penny-item score, while
            # the production headline is >=$1 (HEADLINE_MIN_TIER).
            #
            # None, not 0.0, when a fold holds no >=$1 rows -- an empty
            # partition has no accuracy, and a zero reads as "scored nothing
            # right". Same rule score_cohort follows for its own partitions.
            classifier_acc_ge1 = None
            if "price_tier" in val_df.columns:
                ge1 = val_df["price_tier"].to_numpy() >= HEADLINE_MIN_TIER
                if ge1.any():
                    classifier_acc_ge1 = round(
                        float((pred_cls[ge1] == actual_cls[ge1]).mean()) * 100, 1)
                    if self.market_relative_labels:
                        relative_acc_ge1 = round(float(
                            (clf.predict(X_val).argmax(axis=1)[ge1]
                             == actual_rel_cls[ge1]).mean()) * 100, 1)
```

Add the three keys to the `fold_metrics.append({...})` dict, after
`"classifier_accuracy_ge1": classifier_acc_ge1,`:

```python
                "relative_accuracy_ge1": relative_acc_ge1,
                "market_factor_coverage": mf_coverage,
```

- [ ] **Step 8: Aggregate the new metrics into `cv_results`**

In `train()`, next to the existing `mean_clf_ge1` aggregation, add:

```python
            # Reported, never gated. The stage-1 kill switch: if the model
            # cannot call the idiosyncratic direction, no market forecast
            # rescues the absolute call.
            rel_ge1 = [m["relative_accuracy_ge1"] for m in cv_metrics
                       if m.get("relative_accuracy_ge1") is not None]
            mean_rel_ge1 = round(float(np.mean(rel_ge1)), 1) if rel_ge1 else None
            covs = [m["market_factor_coverage"] for m in cv_metrics
                    if m.get("market_factor_coverage") is not None]
            mean_cov = round(float(np.mean(covs)), 1) if covs else None
```

and add both to the same dict that already carries `"mean_classifier_acc_ge1"`:

```python
                "mean_relative_acc_ge1": mean_rel_ge1,
                "mean_market_factor_coverage": mean_cov,
```

- [ ] **Step 9: Run the full backend suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass. Record the count for the changelog. If `tests/test_forecaster.py` or `tests/test_served_cohort_weighting.py` fail, the control path was not preserved — fix before continuing.

- [ ] **Step 10: Commit**

```bash
git add backend/models/forecaster.py
git commit -m "feat: market-relative label arm and its CV metrics, default off"
```

---

### Task 5: The env knob and the two-arm driver

**Files:**
- Modify: `backend/scripts/forecast_prices.py` (constants ~line 60; `run_forecast` ctor ~line 249)
- Create: `backend/scripts/ab_test_market_relative_labels.py`
- Test: `backend/tests/test_market_relative_labels.py`

**Interfaces:**
- Consumes: `ItemForecaster(market_relative_labels=...)` from Task 3.
- Produces: `_market_relative_labels() -> bool` in `forecast_prices.py`, reading `TRAIN_MARKET_RELATIVE_LABELS`; `DEFAULT_MARKET_RELATIVE_LABELS = False`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_market_relative_labels.py`:

```python
def test_env_knob_defaults_to_false(monkeypatch):
    import scripts.forecast_prices as fp
    monkeypatch.delenv("TRAIN_MARKET_RELATIVE_LABELS", raising=False)
    assert fp._market_relative_labels() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_env_knob_accepts_truthy_spellings(monkeypatch, raw):
    import scripts.forecast_prices as fp
    monkeypatch.setenv("TRAIN_MARKET_RELATIVE_LABELS", raw)
    assert fp._market_relative_labels() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "banana"])
def test_env_knob_rejects_everything_else(monkeypatch, raw):
    import scripts.forecast_prices as fp
    monkeypatch.setenv("TRAIN_MARKET_RELATIVE_LABELS", raw)
    assert fp._market_relative_labels() is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_market_relative_labels.py -q`
Expected: `AttributeError: module 'scripts.forecast_prices' has no attribute '_market_relative_labels'`.

- [ ] **Step 3: Add the knob**

In `backend/scripts/forecast_prices.py`, after `DEFAULT_SERVED_COHORT_SHARE = None`:

```python
DEFAULT_MARKET_RELATIVE_LABELS = False
_TRUTHY = {"1", "true", "yes", "on"}


def _market_relative_labels() -> bool:
    """Train the directional classifier on market-demeaned returns.

    Env-configured for the same reason as TRAIN_FEATURE_ROWS: this script
    parses argv as a plain set of flags. Default False keeps production
    byte-identical. See
    docs/superpowers/specs/2026-08-06-market-relative-labels-design.md.
    """
    raw = os.environ.get("TRAIN_MARKET_RELATIVE_LABELS")
    if not raw:
        return DEFAULT_MARKET_RELATIVE_LABELS
    if raw.strip().lower() not in _TRUTHY:
        return DEFAULT_MARKET_RELATIVE_LABELS
    logger.info(
        "TRAIN_MARKET_RELATIVE_LABELS=1: the directional classifier trains on "
        "r - m and the absolute call is rebuilt from a history-only market "
        "forecast. Watch mean_relative_acc_ge1 (the stage-1 kill switch) and "
        "mean_market_factor_coverage."
    )
    return True
```

and pass it at the constructor:

```python
        forecaster = ItemForecaster(db_session=db, prune_failed_groups=False,
                                    served_cohort_share=_served_cohort_share(),
                                    market_relative_labels=_market_relative_labels(),
                                    model_dir=_model_dir())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_market_relative_labels.py -q`
Expected: 23 passed.

- [ ] **Step 5: Write the driver script**

Create `backend/scripts/ab_test_market_relative_labels.py`:

```python
#!/usr/bin/env python3
"""Paired A/B: does training the directional classifier on market-demeaned
returns beat training it on raw returns?

This is a DRIVER, not a harness. It sets env vars and invokes two real cold
retrains via scripts/forecast_prices.py --train-only, then diffs their
meta.json. Both arms therefore run through the live _cv_evaluate_horizon.

That distinction is the whole point. scripts/ab_test_*.py mostly build their
own walk-forward loops, and that pattern produced
docs/changelog/2026-08-06-volume-ab-and-harness-defects.md -- that harness scored
a cohort 92% of which production never serves, so ~31pp of its reported
directional accuracy was free hits.

Both arms write into scratch directories via FORECAST_MODEL_DIR, so the
deployed artifact in backend/models/saved_models/ is never touched.

PRE-REGISTERED DECISION RULE (spec, set before either arm ran):
  1. KILL if relative_accuracy_ge1 <= 51% at EVERY horizon. No market model
     rescues an absent idiosyncratic signal. Do not tune m_hat to save it.
  2. RECOMMEND ADOPTION only on a paired mean diff in classifier_accuracy_ge1
     of > +2pp at two or more horizons AND not worse than -1pp at any.
  3. WITHIN +/-1pp: market-date domination is not addressable by relabelling.
     Record and close.
Adoption is a recommendation for a follow-up change. Nothing ships from here.

Usage:
    python scripts/ab_test_market_relative_labels.py --out /tmp/mrl
    python scripts/ab_test_market_relative_labels.py --out /tmp/mrl --report-only
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [3, 7, 14, 30]
ARMS = {"control": "0", "treatment": "1"}

# Minimum market-factor coverage before a run is readable. Below this the
# treatment arm is diluted toward the control by the zero-fill fallback and
# the contrast measures nothing.
MIN_COVERAGE_PCT = 95.0


def run_arm(name: str, flag: str, out_root: Path) -> Path:
    """Run one cold --train-only retrain into its own scratch model dir."""
    model_dir = out_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["FORECAST_MODEL_DIR"] = str(model_dir)
    env["TRAIN_MARKET_RELATIVE_LABELS"] = flag
    logger.info(f"\n=== arm: {name} (TRAIN_MARKET_RELATIVE_LABELS={flag}) ===")
    logger.info(f"    artifacts -> {model_dir}")
    proc = subprocess.run(
        [sys.executable, "scripts/forecast_prices.py", "--train-only"],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"arm {name!r} failed with exit code {proc.returncode}")
    return model_dir / "meta.json"


def _folds(meta: dict, horizon: int) -> list:
    return meta["cv_results"][str(horizon)]["per_fold"]


def verify_pairing(control: dict, treatment: dict) -> None:
    """Confirm the arms are comparable BEFORE any metric is read.

    Identical fold ids, row counts, date bounds and tuned params. If these
    differ, the two arms scored different data and the contrast is void --
    reading the accuracy numbers first is how a broken run gets believed.
    """
    problems = []
    for h in HORIZONS:
        c, t = _folds(control, h), _folds(treatment, h)
        if len(c) != len(t):
            problems.append(f"{h}d: fold count {len(c)} vs {len(t)}")
            continue
        for cf, tf in zip(c, t):
            for key in ("fold", "n_train", "n_val",
                        "train_start", "train_end", "val_start", "val_end"):
                if cf.get(key) != tf.get(key):
                    problems.append(
                        f"{h}d fold {cf.get('fold')}: {key} "
                        f"{cf.get(key)!r} vs {tf.get(key)!r}")
        if control.get("tuned_params", {}).get(str(h)) != \
                treatment.get("tuned_params", {}).get(str(h)):
            problems.append(f"{h}d: tuned_params differ")
    if problems:
        for p in problems:
            logger.error(f"  PAIRING BROKEN: {p}")
        raise SystemExit("arms are not paired; the contrast is void")
    logger.info("  pairing verified: identical folds, row counts, dates, params")


def check_coverage(treatment: dict) -> None:
    for h in HORIZONS:
        covs = [f.get("market_factor_coverage") for f in _folds(treatment, h)]
        covs = [c for c in covs if c is not None]
        if covs and min(covs) < MIN_COVERAGE_PCT:
            logger.error(
                f"  COVERAGE TOO LOW at {h}d: min fold coverage {min(covs):.1f}% "
                f"< {MIN_COVERAGE_PCT}%. The treatment arm is diluted toward "
                f"the control; this run is void.")
            raise SystemExit("insufficient market-factor coverage")
    logger.info("  market-factor coverage OK on every fold")


def paired_table(control: dict, treatment: dict, metric: str) -> list:
    rows = []
    for h in HORIZONS:
        c = [f.get(metric) for f in _folds(control, h)]
        t = [f.get(metric) for f in _folds(treatment, h)]
        pairs = [(a, b) for a, b in zip(c, t) if a is not None and b is not None]
        if not pairs:
            rows.append((h, 0, None, None, None, None, None))
            continue
        ca = np.array([p[0] for p in pairs], dtype=float)
        ta = np.array([p[1] for p in pairs], dtype=float)
        diff = ta - ca
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        tstat = (float(diff.mean()) / (sd / np.sqrt(len(diff)))
                 if sd > 0 else None)
        rows.append((h, len(pairs), float(ca.mean()), float(ta.mean()),
                     float(diff.mean()), sd, tstat))
    return rows


def _fmt(rows, title):
    logger.info(f"\n{title}")
    logger.info("| horizon | folds | control | treatment | paired diff | sd | t |")
    logger.info("|---|---|---|---|---|---|---|")
    for h, n, c, t, d, sd, ts in rows:
        if c is None:
            logger.info(f"| {h}d | 0 | - | - | - | - | - |")
            continue
        logger.info(f"| {h}d | {n} | {c:.2f} | {t:.2f} | **{d:+.2f}** | "
                    f"{sd:.2f} | {'-' if ts is None else f'{ts:.2f}'} |")


def verdict(ge1_rows, treatment: dict) -> None:
    rel = {}
    for h in HORIZONS:
        vals = [f.get("relative_accuracy_ge1") for f in _folds(treatment, h)]
        vals = [v for v in vals if v is not None]
        rel[h] = float(np.mean(vals)) if vals else None

    logger.info("\nStage 1 — relative_accuracy_ge1 (the kill switch):")
    for h in HORIZONS:
        v = rel[h]
        logger.info(f"  {h}d: {'n/a' if v is None else f'{v:.2f}%'}")

    live = [v for v in rel.values() if v is not None]
    if live and all(v <= 51.0 for v in live):
        logger.info(
            "\nVERDICT: KILL (rule 1). The classifier cannot call the "
            "idiosyncratic direction at any horizon, so there is no signal for "
            "a market forecast to recover. Do not interpret stage 2 and do not "
            "tune the m_hat estimator to rescue it.")
        return

    diffs = {h: d for h, _n, _c, _t, d, _sd, _ts in ge1_rows if d is not None}
    if not diffs:
        logger.info("\nVERDICT: no comparable folds.")
        return
    wins = [h for h, d in diffs.items() if d > 2.0]
    worst = min(diffs.values())
    if len(wins) >= 2 and worst >= -1.0:
        logger.info(
            f"\nVERDICT: RECOMMEND ADOPTION (rule 2). >+2pp at {wins}, worst "
            f"horizon {worst:+.2f}pp. This is a recommendation for a follow-up "
            f"change — nothing ships from this script.")
    elif all(abs(d) <= 1.0 for d in diffs.values()):
        logger.info(
            "\nVERDICT: NO EFFECT (rule 3). Every horizon within ±1pp. "
            "Market-date domination is not addressable by relabelling. Leave "
            "the default off and record the result.")
    else:
        logger.info(
            f"\nVERDICT: DOES NOT CLEAR THE BAR (rule 2 not met). "
            f"{len(wins)} horizon(s) above +2pp, worst {worst:+.2f}pp. "
            f"Leave the default off.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="scratch root; each arm gets a subdirectory")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the retrains and read existing meta.json files")
    args = ap.parse_args()

    out_root = Path(args.out).expanduser()
    metas = {}
    for name, flag in ARMS.items():
        path = out_root / name / "meta.json"
        if not args.report_only:
            path = run_arm(name, flag, out_root)
        if not path.exists():
            raise SystemExit(f"missing {path}")
        metas[name] = json.loads(path.read_text())

    logger.info("\n=== validity checks (before any metric is read) ===")
    verify_pairing(metas["control"], metas["treatment"])
    check_coverage(metas["treatment"])

    ge1 = paired_table(metas["control"], metas["treatment"],
                       "classifier_accuracy_ge1")
    _fmt(ge1, "classifier_accuracy_ge1 — the headline (>= $1 served cohort)")
    _fmt(paired_table(metas["control"], metas["treatment"],
                      "classifier_accuracy"),
         "classifier_accuracy — pooled, all tiers (~83% penny rows)")
    verdict(ge1, metas["treatment"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the script parses and its pure functions work**

Run: `venv/bin/python -c "import ast,pathlib; ast.parse(pathlib.Path('scripts/ab_test_market_relative_labels.py').read_text()); print('ok')"`
Expected: `ok`

Run: `venv/bin/python scripts/ab_test_market_relative_labels.py --help`
Expected: usage text, exit 0.

- [ ] **Step 7: Run the full backend suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add backend/scripts/forecast_prices.py backend/scripts/ab_test_market_relative_labels.py backend/tests/test_market_relative_labels.py
git commit -m "feat: TRAIN_MARKET_RELATIVE_LABELS knob and the paired A/B driver"
```

---

### Task 6: Changelog

**Files:**
- Create: `docs/changelog/2026-08-06-market-relative-labels-instrument.md`

- [ ] **Step 1: Write the entry**

Record: the hypothesis and the evidence behind it; that this is an instrument, not a result; the pre-registered rule copied verbatim from the spec so it cannot drift; where the code lives and that the default is off; the test count from Task 5 Step 7; and how to run the arms (`python scripts/ab_test_market_relative_labels.py --out /tmp/mrl`). Link the spec, `2026-08-03-accuracy-is-clustered-by-forecast-date.md`, `2026-08-06-served-cohort-weighting-refuted.md` and `2026-08-06-volume-ab-and-harness-defects.md`.

State explicitly that no measurement has been taken yet, so no conclusion about the hypothesis is recorded here.

- [ ] **Step 2: Commit**

```bash
git add docs/changelog/2026-08-06-market-relative-labels-instrument.md
git commit -m "docs: record the market-relative label instrument"
```

---

## Self-Review

**Spec coverage.** Chain-linked index → Task 1. `>= $1` restriction, `MIN_INDEX_ITEMS`, composition → Task 1. Calendar-gap tolerance, realized factor, leak-free `m̂`, diagnostics → Task 2. Full-frame-before-subsample, never-a-feature → Task 3. NaN-`m` zero-fill, flat-share matching, demeaning, reconstruction, the three new metrics → Task 4. Env knob, driver, pairing verification, coverage gate, pre-registered rule → Task 5. Changelog → Task 6. Every "Explicitly not in scope" item is respected: no serving change, no residual quantile model, no estimator tuning, no allowlist change.

**Type consistency.** `build_market_index` returns the five columns every later task reads. `market_factor_for_horizon` returns a `pd.Series` named `market_factor_{h}d`, matching the column Task 3 creates and Task 4 reads. `forecast_market_factor` returns a `float` percent, summed with `_residual_point_estimate`'s float array in Task 4. `_matched_flat_band` returns the float passed as `flat_band` to `_fit_direction_classifier`. Metric keys `relative_accuracy_ge1` / `market_factor_coverage` are written in Task 4 Step 7 and read in Task 5's driver under the same names.

**Known limitation carried from the spec, not a plan gap.** `ê` is a class band-edge, not a magnitude, so `m̂` can shift the decision boundary but cannot rank within a class. The spec accepts this explicitly; a residual quantile model is out of scope until stage 1 reports.
