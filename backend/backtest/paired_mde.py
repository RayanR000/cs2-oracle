"""Paired, date-clustered confidence interval on the DA difference between two
arms measured on the SAME folds.

Why paired: the arms share folds, so most of the uncertainty in either arm's
absolute DA is between-date variance that affects both identically (the
2025-12-01 rising / 2026-07-17 falling asymmetry documented in
docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md). An
unpaired comparison of two wide intervals reports an MDE no design could pass.
Differencing within (item, date) removes the common term.

Still clustered: what remains is between-date variation in the DIFFERENCE, so
dates are the resampling unit, not rows.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtest.scoring import BOOTSTRAP_CI, BOOTSTRAP_RNG_SEED, N_BOOTSTRAP


def paired_da_difference(
    records_a: list[dict],
    records_b: list[dict],
    n_resamples: int = N_BOOTSTRAP,
    ci: int = BOOTSTRAP_CI,
) -> dict:
    """Bootstrap the mean of (b - a) direction_correct over shared (item, date).

    Returns percentage-point figures. `mde_pp` is the half-width of the
    interval: the smallest difference this gate can resolve. None when fewer
    than 2 dates are shared.
    """
    index_a = {(r["item_id"], r["forecast_date"]): r["direction_correct"]
               for r in records_a}
    by_date: dict = defaultdict(list)
    n_paired = 0
    for r in records_b:
        key = (r["item_id"], r["forecast_date"])
        if key in index_a:
            by_date[r["forecast_date"]].append(
                r["direction_correct"] - index_a[key]
            )
            n_paired += 1

    if n_paired == 0:
        raise ValueError(
            "no paired records: the two arms share no (item_id, forecast_date) "
            "pairs, so they were not measured on the same folds"
        )

    dates = sorted(by_date)
    groups = [np.array(by_date[d], dtype=float) for d in dates]
    all_diffs = np.concatenate(groups)
    mean_diff_pp = float(all_diffs.mean()) * 100.0

    out = {
        "mean_diff_pp": round(mean_diff_pp, 4),
        "n_paired": n_paired,
        "n_dates": len(dates),
        "ci_lower_pp": None,
        "ci_upper_pp": None,
        "mde_pp": None,
    }
    if len(groups) < 2:
        return out

    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    sums = np.array([g.sum() for g in groups], dtype=float)
    counts = np.array([g.size for g in groups], dtype=float)
    n_groups = len(groups)

    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_groups, size=n_groups)
        stats[i] = sums[idx].sum() / counts[idx].sum()
    stats *= 100.0

    alpha = (100 - ci) / 2
    lower = float(np.percentile(stats, alpha))
    upper = float(np.percentile(stats, 100 - alpha))
    out["ci_lower_pp"] = round(lower, 4)
    out["ci_upper_pp"] = round(upper, 4)
    out["mde_pp"] = round((upper - lower) / 2, 4)
    return out
