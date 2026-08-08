"""Paired, fold-clustered confidence interval on the DA difference between two
arms measured on the SAME folds.

Why paired: the arms share folds, so most of the uncertainty in either arm's
absolute DA is between-date variance that affects both identically (the
2025-12-01 rising / 2026-07-17 falling asymmetry documented in
docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md). An
unpaired comparison of two wide intervals reports an MDE no design could pass.
Differencing within (item, date) removes the common term.

Still clustered: what remains is between-*fold* variation in the DIFFERENCE, so
folds are the resampling unit — not rows, and not dates.

## Why the unit is the fold and not the date (changed 2026-08-07)

This module clustered on `forecast_date` until 2026-08-07. That was wrong, and
it was wrong in the dangerous direction: it made intervals too NARROW, so it
overstated significance rather than understating it.

Dates are not independent. Every date inside one fold's 21-day validation
window is scored by the same trained model, and at short horizons adjacent
dates' forward-return windows overlap. Resampling 1,569 dates as if they were
1,569 independent draws, when they came from ~70 fitted models, understates the
variance by roughly the within-fold correlation.

The failure was demonstrated, not theorised. `scripts/compute_mde.py` at h=3,
300 items, `--step-days 21` compares two arms that differ **only in the
LightGBM seed** — pure noise by construction, true difference zero:

    mean_diff_pp -0.1581   n_paired 468,759   n_dates 1,569
    ci [-0.3099, -0.0140]  mde_pp 0.1479

The 95% interval excludes zero. A correctly-sized interval rejects a true null
5% of the time; this one rejected it on the first try, on the one comparison
where the answer is known a priori. Any "CI excludes zero" produced by this
module before 2026-08-07 should be re-derived before it is believed — that
includes the CSFloat basis, ByMykel metadata and training-breadth A/Bs.

`cluster_key` is explicit and defaults to `"fold_id"`. It does **not** silently
fall back to dates when the key is missing: silent fallback would restore
exactly the bug above on any caller that forgot to thread the fold through.
Pass `cluster_key="forecast_date"` deliberately if you want the old behaviour.
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
    cluster_key: str = "fold_id",
) -> dict:
    """Bootstrap the mean of (b - a) direction_correct over shared (item, date).

    Pairing is always on `(item_id, forecast_date)` — that is the row identity.
    *Resampling* is on `cluster_key`, which is a different and coarser grain;
    see the module docstring for why conflating the two under-disperses the
    interval.

    Returns percentage-point figures. `mde_pp` is the half-width of the
    interval: the smallest difference this gate can resolve. None when fewer
    than 2 clusters are shared.
    """
    index_a = {(r["item_id"], r["forecast_date"]): r["direction_correct"]
               for r in records_a}
    by_cluster: dict = defaultdict(list)
    dates_seen: set = set()
    n_paired = 0
    for r in records_b:
        key = (r["item_id"], r["forecast_date"])
        if key not in index_a:
            continue
        if cluster_key not in r or r[cluster_key] is None:
            raise ValueError(
                f"record is missing cluster key {cluster_key!r}. Records from "
                f"`walkforward_records.fold_records` carry 'fold_id'; if these "
                f"records genuinely have no folds, pass "
                f"cluster_key='forecast_date' explicitly. Falling back to dates "
                f"silently is the 2026-08-07 under-dispersion bug — see the "
                f"module docstring."
            )
        by_cluster[r[cluster_key]].append(
            r["direction_correct"] - index_a[key]
        )
        dates_seen.add(r["forecast_date"])
        n_paired += 1

    if n_paired == 0:
        raise ValueError(
            "no paired records: the two arms share no (item_id, forecast_date) "
            "pairs, so they were not measured on the same folds"
        )

    clusters = sorted(by_cluster)
    groups = [np.array(by_cluster[c], dtype=float) for c in clusters]
    all_diffs = np.concatenate(groups)
    mean_diff_pp = float(all_diffs.mean()) * 100.0

    out = {
        "mean_diff_pp": round(mean_diff_pp, 4),
        "n_paired": n_paired,
        # Both are reported so a result is self-describing about which grain it
        # was resampled at. n_dates >> n_clusters is the signature of the old
        # too-narrow interval.
        "n_dates": len(dates_seen),
        "n_clusters": len(clusters),
        "cluster_key": cluster_key,
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
