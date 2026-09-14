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


def fold_level_records(fold_ids, values, *, metric: str):
    """Paired records at FOLD grain, for an arm scored one number per fold.

    The row-grain form above is strictly better — differencing within
    `(item, date)` removes the common market term — and every harness that can
    reach its per-row predictions should use it. The three pinball harnesses
    cannot without restructuring: they shard folds across processes and merge
    per-fold CSVs, so the rows are gone by the time the arms meet.

    Fold grain still gives the thing that was missing: an interval resampled on
    the fold, which is the independent unit. It is wide, because ~8 folds is ~8
    clusters. That width is the honest cost of the design and is the point —
    the rule it replaces ("wins on at least half the folds") fires 50% of the
    time on two identical arms.
    """
    ids = list(fold_ids)
    vals = list(values)
    if len(ids) != len(vals):
        raise ValueError(f"fold_ids has {len(ids)} entries, values has {len(vals)}")
    return [
        # Pairing key and cluster key are the same here, by construction: the
        # fold IS the observation. `paired_metric_difference` pairs on
        # (item_id, forecast_date) and resamples on fold_id, and at this grain
        # all three are the fold.
        {"item_id": f, "forecast_date": str(f), "fold_id": f, metric: v}
        for f, v in zip(ids, vals)
    ]


def without_records(obj):
    """`obj` with every ``"records"`` list dropped, for printing or writing.

    A harness's records are its pairing input, not its result: one arm at one
    horizon carries tens of thousands of them, so dumping a results dict to
    stdout or to `--out` without this buries the summary under them. The paired
    interval is computed before this is called and survives it.
    """
    if isinstance(obj, dict):
        return {k: without_records(v) for k, v in obj.items() if k != "records"}
    if isinstance(obj, list):
        return [without_records(v) for v in obj]
    return obj


def paired_records(*, item_ids, forecast_dates, fold_id, keep=None, **metrics):
    """Minimal per-row records for pairing two A/B arms on the same folds.

    `fold_records` above is the full `score_cohort` shape and is what the
    walkforward gate needs. An A/B harness needs three things only: the pairing
    key `(item_id, forecast_date)`, the resampling key `fold_id`, and whichever
    scalars it is comparing — so this exists to stop each of the ten harnesses
    fixed on 2026-08-08 from growing its own copy of the same six lines.

    `keep` is an optional boolean mask of rows to score, applied to every array.
    Pass the arm-independent one: a mask that differs between arms would break
    the pairing rather than narrow it, and `paired_metric_difference` would
    silently compare whatever intersection survived.

    `fold_id` must identify the fold the same way in every arm. Use the loop's
    own window bound rather than a running counter — a counter drifts the
    moment one arm skips a fold the other kept.

    Each keyword in `metrics` becomes a column: `direction_correct=<bool array>`
    for a hit rate, `pinball=<float array>` for a loss. Values are cast per row
    by `paired_metric_difference`, so bools and floats both work.
    """
    ids = np.asarray(item_ids)
    dates = np.asarray(forecast_dates)
    # tolist(): a np.bool_ does not support `-`, which is exactly how the
    # difference is taken, and a np.float32 does not survive a json round trip.
    cols = {k: np.asarray(v).tolist() for k, v in metrics.items()}
    if not cols:
        raise ValueError("paired_records needs at least one metric to compare")

    n = len(ids)
    checked = [("forecast_dates", dates), *cols.items()]
    # `keep` is checked too. Every caller derives it from the prediction/actual
    # arrays, which are different objects from `item_ids` (that comes off
    # `val_df`) — so a short mask would not raise, it would score a subset while
    # `ids[i]` labelled rows the mask never described. A silent mispairing is
    # the one failure this function exists to prevent.
    if keep is not None:
        checked.append(("keep", np.asarray(keep)))
    for name, arr in checked:
        if len(arr) != n:
            raise ValueError(f"all inputs must be of equal length; {name} has {len(arr)}, expected {n}")

    idx = np.flatnonzero(np.asarray(keep)) if keep is not None else range(n)
    return [
        {
            "item_id": ids[i],
            # str(): the pairing key has to compare equal across arms, and a
            # datetime64 on one side against a datetime.date on the other
            # stringifies differently and pairs zero rows.
            "forecast_date": str(dates[i]),
            "fold_id": fold_id,
            **{k: v[i] for k, v in cols.items()},
        }
        for i in idx
    ]


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
            raise ValueError(f"all inputs must be of equal length; {name} has {len(arr)}, expected {n}")
    if predicted_classes is not None and len(predicted_classes) != n:
        raise ValueError(
            f"all inputs must be of equal length; predicted_classes has {len(predicted_classes)}, expected {n}"
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
        records.append(
            {
                "abs_error": abs_error,
                "sq_error": (mid[i] - actual[i]) ** 2,
                # Divided by the BASE leg, matching backtest_accuracy._derive_verdict.
                "pct_error": abs(abs_error / base[i]) * 100.0,
                "direction_correct": 1 if predicted_direction == actual_direction else 0,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                # No rebase, unlike backtest_accuracy._derive_verdict: the band here
                # is built as `base * (1 + ret)` a few lines up, so the resolved base
                # IS the quote and the two predicates coincide. score_cohort defaults
                # its dollar-basis split to this value for the same reason.
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
            }
        )
    return records
