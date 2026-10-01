"""Calendar lookback, in days, of every price-technical feature.

A feature on row date d reads prices from (d - W, d]. If a source-composition
break b falls in that span, the feature compares two price bases and measures
the switch rather than the market; `mask_break_lookbacks` sets it to NaN.
W = 0 marks a level, or a flag that depends only on how many rows exist.

EWM features have infinite memory; they get an effective window of 3 x span
(about 95% of the weight). MACD: line 3 x 26 = 78; the signal line adds
3 x 9 = 27 on top. Windows assume one row per calendar day, which the 2026
archive (the only era with breaks) satisfies.

`tests/test_break_aware_lookbacks.py` fails if engineer_features emits a
price-technical column that is not listed here.
"""

from __future__ import annotations

import pandas as pd

_N = (1, 3, 7, 14, 30, 60, 90, 120, 180)

FEATURE_LOOKBACK_DAYS: dict[str, int] = {
    **{f"price_lag_{n}d": n for n in _N},
    **{f"return_{n}d": n for n in _N},
    **{f"price_{s}_{n}d": n for s in ("mean", "std", "min", "max") for n in (7, 14, 20, 30, 60)},
    **{f"price_cv_{n}d": n for n in (7, 14, 20, 30, 60)},
    "price_zscore_30d": 30,
    "vol_regime_60_30": 60,
    "trend_divergence_30_60": 60,
    "price_accel_7d": 14,
    "price_log": 0,
    "price_tier": 0,
    "log_return_1d": 1,
    "log_return_7d": 7,
    "autocorr_1d": 2,
    "autocorr_7d": 14,
    **{f"bb_{s}": 20 for s in ("upper", "lower", "pct_b", "width")},
    "rsi_14": 14,
    "rsi_missing": 14,
    "macd_line": 78,
    "macd_line_rel": 78,
    "macd_signal": 105,
    "macd_histogram": 105,
    "macd_histogram_rel": 105,
    "macd_missing": 0,
    "price_mean_100d": 100,
    "price_dist_ma100": 100,
    "price_mean_200d": 200,
    "price_dist_ma200": 200,
    "trend_up_fraction_30d": 30,
}


def mask_break_lookbacks(df: pd.DataFrame, cols: list[str], breaks: frozenset) -> int:
    """NaN each `cols` cell whose lookback (d - W, d] contains a break. Returns the count."""
    if not breaks:
        return 0
    unknown = [c for c in cols if c not in FEATURE_LOOKBACK_DAYS]
    if unknown:
        raise KeyError(f"no FEATURE_LOOKBACK_DAYS entry for {unknown}; declare its window")
    day = pd.to_datetime(df["date"])
    bks = pd.to_datetime(sorted(breaks))
    n = 0
    for c in cols:
        w = FEATURE_LOOKBACK_DAYS[c]
        if w == 0 or c not in df.columns:
            continue
        # b in (d - w, d]  <=>  b <= d < b + w
        hit = pd.Series(False, index=df.index)
        for b in bks:
            hit |= (day >= b) & (day < b + pd.to_timedelta(w, unit="D"))
        n += int((hit & df[c].notna()).sum())
        df.loc[hit, c] = float("nan")
    return n
