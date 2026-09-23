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

Design: docs/specs/2026-08-06-market-relative-labels-design.md
"""

import logging

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

_INDEX_COLUMNS = ["log_return", "n_items", "valid", "level", "invalid_cum"]


def build_market_index(
    price_df: pd.DataFrame, min_items: int = MIN_INDEX_ITEMS, min_price: float = MARKET_MIN_PRICE_USD
) -> pd.DataFrame:
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
        return pd.DataFrame(columns=_INDEX_COLUMNS, index=pd.DatetimeIndex([], name="date"))

    df = price_df[["item_id", "date", "price"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0].dropna(subset=["price"])
    if df.empty:
        return pd.DataFrame(columns=_INDEX_COLUMNS, index=pd.DatetimeIndex([], name="date"))

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
        df["prev_price"].notna() & (df["prev_price"] >= min_price) & ((df["date"] - df["prev_date"]).dt.days == 1)
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


def market_factor_for_horizon(
    index: pd.DataFrame, horizon: int, tolerance_days: int = INDEX_TOLERANCE_DAYS
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
    # `unit="D"` rather than `days=`: the keyword form goes through NumPy's
    # deprecated generic timedelta unit, which is slated to become an error.
    targets = dates + pd.Timedelta(horizon, unit="D")
    # searchsorted "left" gives the first available date >= target.
    pos = np.searchsorted(dates.values, targets.values, side="left")

    n = len(dates)
    in_range = pos < n
    end_pos = np.where(in_range, np.minimum(pos, n - 1), n - 1)
    end_dates = dates[end_pos]
    within = in_range & ((end_dates - targets).days <= tolerance_days)

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


def forecast_market_factor(index: pd.DataFrame, as_of, horizon: int, trailing_days: int = TRAILING_DRIFT_DAYS) -> float:
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
