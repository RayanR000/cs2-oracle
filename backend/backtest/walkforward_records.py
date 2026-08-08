"""Fold arrays -> score_cohort records, for the walkforward gate.

Pure: no I/O, no LightGBM, no clock. Exists as its own module because it owns
two unit conversions that were previously wrong or absent in
scripts/walkforward_backtest.py:

1. `target_return_{h}d` is in PERCENT; `direction_from_return` takes a
   FRACTION. The old `_compute_metrics` never called it at all — it compared
   floats exactly, so "flat" never fired and the gate was a 2-label problem at
   50% chance while production is 3-label at ~33%.
2. The predicted direction must come from the classifier's argmax, because that
   is what production serves (forecaster.py:2981-2982). The old code took the
   sign of the p50 regression.
"""
from __future__ import annotations

import numpy as np

from backtest.scoring import direction_from_return, price_tier

# forecaster._direction_classes buckets returns as 0=down, 1=flat, 2=up.
CLASS_TO_DIRECTION = {0: "down", 1: "flat", 2: "up"}


def fold_records(
    *,
    item_ids,
    forecast_dates,
    base_prices,
    actual_returns_pct,
    mid_returns_pct,
    low_returns_pct,
    high_returns_pct,
    predicted_classes,
    fold_id=None,
    horizon_days=None,
) -> list[dict]:
    """Build score_cohort records for one fold.

    All arrays must be the same length and aligned row-for-row. Returns are in
    percent (the units of `target_return_{h}d`); prices are reconstructed as
    `base * (1 + ret/100)`.

    `predicted_classes` are the directional classifier's argmax values. Pass
    None to score the median's sign instead — used to score both estimators
    side by side (spec Task 1b).

    `fold_id` identifies the walkforward fold these rows came from. It exists
    because `forecast_date` is *not* an independent resampling unit: every date
    inside one fold's validation window is scored by the same trained model, and
    at short horizons adjacent dates' forward-return windows overlap. Clustering
    on dates therefore under-disperses the bootstrap — measured 2026-08-07, a
    seed-only placebo at h=3 returned a 95% CI of [-0.310, -0.014]pp that
    excluded zero on what is by construction pure noise. See
    `backtest/paired_mde.py`.

    `horizon_days` is the forecast horizon these rows were built for. It is
    optional and defaults to None, which scores as out_of_scope for the
    friction-conditioned metric — correct for a caller that has not said which
    horizon it is measuring, and wrong to guess at.
    """
    arrays = {
        "item_ids": item_ids,
        "forecast_dates": forecast_dates,
        "base_prices": base_prices,
        "actual_returns_pct": actual_returns_pct,
        "mid_returns_pct": mid_returns_pct,
        "low_returns_pct": low_returns_pct,
        "high_returns_pct": high_returns_pct,
    }
    n = len(base_prices)
    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(
                f"all inputs must be of equal length; {name} has {len(arr)}, expected {n}"
            )
    if predicted_classes is not None and len(predicted_classes) != n:
        raise ValueError(
            f"all inputs must be of equal length; predicted_classes has "
            f"{len(predicted_classes)}, expected {n}"
        )

    base = np.asarray(base_prices, dtype=float)
    actual_ret = np.asarray(actual_returns_pct, dtype=float)
    mid_ret = np.asarray(mid_returns_pct, dtype=float)
    low_ret = np.asarray(low_returns_pct, dtype=float)
    high_ret = np.asarray(high_returns_pct, dtype=float)

    actual = base * (1.0 + actual_ret / 100.0)
    mid = base * (1.0 + mid_ret / 100.0)
    low = base * (1.0 + low_ret / 100.0)
    high = base * (1.0 + high_ret / 100.0)

    records = []
    for i in range(n):
        if base[i] <= 0:
            continue
        # /100: percent -> fraction, which is what direction_from_return's
        # FLAT_TOLERANCE = 0.005 is expressed in.
        actual_direction = direction_from_return(actual_ret[i] / 100.0)
        if predicted_classes is not None:
            predicted_direction = CLASS_TO_DIRECTION[int(predicted_classes[i])]
        else:
            predicted_direction = direction_from_return(mid_ret[i] / 100.0)

        abs_error = abs(mid[i] - actual[i])
        records.append({
            "abs_error": abs_error,
            "sq_error": (mid[i] - actual[i]) ** 2,
            # Divided by the BASE leg, matching backtest_accuracy._derive_verdict.
            "pct_error": abs(abs_error / base[i]) * 100.0,
            "direction_correct": 1 if predicted_direction == actual_direction else 0,
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "in_interval": 1 if low[i] <= actual[i] <= high[i] else 0,
            # The harness has no confidence estimator, so score_cohort's
            # conf_* fields are structurally degenerate for these arms.
            "confidence": "low",
            "base_price": float(base[i]),
            "actual_price": float(actual[i]),
            # The prediction leg and the horizon, for the friction-conditioned
            # metric. See backtest/actionable.py.
            "predicted_mid": float(mid[i]),
            "horizon_days": horizon_days,
            "price_tier": price_tier(float(base[i])),
            "item_id": item_ids[i],
            # Rows sharing a date share one market move. This is the PAIRING key
            # (with item_id), and it is the right grain for that. It is NOT the
            # resampling unit — see `fold_id` and the docstring above.
            "forecast_date": forecast_dates[i],
            # The clustering unit: rows sharing a fold share one trained model.
            "fold_id": fold_id,
        })
    return records
