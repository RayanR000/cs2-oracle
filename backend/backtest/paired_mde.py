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

The failure was demonstrated, not theorised. `scripts/archive/compute_mde.py` at h=3,
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


class NoPairedRows(ValueError):
    """Two arms share no `(item_id, forecast_date)` rows.

    Its own type because `paired_arm_contrasts` has to report this one and
    re-raise everything else. The other `ValueError` this module raises is the
    missing-`cluster_key` guard, and that one must stay loud: swallowing it
    would restore the 2026-08-07 under-dispersion bug on any caller that forgot
    to thread the fold through, while printing a confident and wrong
    "not measured on the same folds".
    """


def paired_metric_difference(
    records_a: list[dict],
    records_b: list[dict],
    *,
    value_key: str,
    n_resamples: int = N_BOOTSTRAP,
    ci: int = BOOTSTRAP_CI,
    cluster_key: str = "fold_id",
    scale: float = 1.0,
) -> dict:
    """Bootstrap the mean of (b - a) `value_key` over shared (item, date).

    Pairing is always on `(item_id, forecast_date)` — that is the row identity.
    *Resampling* is on `cluster_key`, which is a different and coarser grain;
    see the module docstring for why conflating the two under-disperses the
    interval.

    `value_key` is any per-row scalar both arms carry. It is `direction_correct`
    for a hit rate (see :func:`paired_da_difference`) and `pinball` for a
    quantile loss — the harnesses that compare pinball had no interval at all
    before 2026-08-08, only a "wins on more than half the folds" rule, which
    answers a different and much weaker question: a coin lands that way 50% of
    the time and the rule fires at 50%.

    `scale` multiplies the reported difference. Pass 100 for a rate you want in
    percentage points; leave it at 1 for a loss, which has its own units.

    `mde` is the half-width of the interval: the smallest difference this
    design can resolve. None when fewer than 2 clusters are shared, because a
    single cluster carries no between-cluster variance to resample.
    """
    index_a = {(r["item_id"], r["forecast_date"]): r[value_key] for r in records_a}
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
        # float() on both sides: `direction_correct` reaches here as a Python
        # bool from one harness and a np.bool_ from another, and np.bool_ has
        # no `-`.
        by_cluster[r[cluster_key]].append(float(r[value_key]) - float(index_a[key]))
        dates_seen.add(r["forecast_date"])
        n_paired += 1

    if n_paired == 0:
        raise NoPairedRows(
            "no paired records: the two arms share no (item_id, forecast_date) "
            "pairs, so they were not measured on the same folds"
        )

    clusters = sorted(by_cluster)
    groups = [np.array(by_cluster[c], dtype=float) for c in clusters]
    all_diffs = np.concatenate(groups)

    out = {
        "mean_diff": round(float(all_diffs.mean()) * scale, 4),
        "n_paired": n_paired,
        # Both are reported so a result is self-describing about which grain it
        # was resampled at. n_dates >> n_clusters is the signature of the old
        # too-narrow interval.
        "n_dates": len(dates_seen),
        "n_clusters": len(clusters),
        "cluster_key": cluster_key,
        "value_key": value_key,
        "ci_lower": None,
        "ci_upper": None,
        "mde": None,
    }
    if len(groups) < 2:
        return out

    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    sums = np.array([g.sum() for g in groups], dtype=float)
    counts = np.array([g.size for g in groups], dtype=float)
    n_groups = len(groups)

    # Single vectorized draw: bit-identical to the per-iteration
    # rng.integers(0, n_groups, size=n_groups) loop, in one
    # (n_resamples, n_groups) operation.
    all_idx = rng.integers(0, n_groups, size=(n_resamples, n_groups))
    stats = sums[all_idx].sum(axis=1) / counts[all_idx].sum(axis=1)
    stats *= scale

    alpha = (100 - ci) / 2
    lower = float(np.percentile(stats, alpha))
    upper = float(np.percentile(stats, 100 - alpha))
    out["ci_lower"] = round(lower, 4)
    out["ci_upper"] = round(upper, 4)
    out["mde"] = round((upper - lower) / 2, 4)
    return out


def paired_da_difference(
    records_a: list[dict],
    records_b: list[dict],
    n_resamples: int = N_BOOTSTRAP,
    ci: int = BOOTSTRAP_CI,
    cluster_key: str = "fold_id",
) -> dict:
    """The directional-accuracy case of :func:`paired_metric_difference`.

    Returns percentage-point figures under the `_pp` names the stored A/B
    results and every existing caller use. Identical arithmetic — the generic
    function was factored out of this one on 2026-08-08 so the pinball
    harnesses could stop verdicting on fold win-counts.
    """
    out = paired_metric_difference(
        records_a,
        records_b,
        value_key="direction_correct",
        n_resamples=n_resamples,
        ci=ci,
        cluster_key=cluster_key,
        scale=100.0,
    )
    return {
        "mean_diff_pp": out["mean_diff"],
        "n_paired": out["n_paired"],
        "n_dates": out["n_dates"],
        "n_clusters": out["n_clusters"],
        "cluster_key": out["cluster_key"],
        "ci_lower_pp": out["ci_lower"],
        "ci_upper_pp": out["ci_upper"],
        "mde_pp": out["mde"],
    }


def paired_arm_contrasts(
    arm_records: dict,
    base: str,
    *,
    value_key: str = "direction_correct",
    scale: float = 100.0,
    higher_is_better: bool = True,
    cluster_key: str = "fold_id",
) -> dict:
    """Every arm against *base*, paired and fold-clustered, with a verdict.

    `arm_records` maps an arm name to its record list. The base arm is skipped
    (its contrast with itself is zero by construction) and so is any arm that
    shares no rows with it — which is a result about the design, not the arm,
    and is reported as `no_shared_rows` rather than swallowed.

    Exists so the ten harnesses fixed on 2026-08-08 share one contrast step.
    Each of them previously had its own, and none of the ten was a test: three
    counted fold wins, five averaged a pooled delta against a ±0.5pp threshold,
    and two printed `a > b`.
    """
    out: dict = {}
    for arm, records in arm_records.items():
        if arm == base or arm.startswith("_"):
            continue
        try:
            paired = paired_metric_difference(
                arm_records[base], records, value_key=value_key, scale=scale, cluster_key=cluster_key
            )
        except NoPairedRows as exc:
            # Only this one is reportable. A missing `cluster_key` is a wiring
            # bug in the caller and propagates.
            out[arm] = {"verdict": "no_shared_rows", "detail": str(exc)}
            continue
        paired["verdict"] = verdict(paired, higher_is_better=higher_is_better)
        out[arm] = paired
    return out


def format_paired(paired: dict, unit: str = "pp") -> str:
    """One line for a paired result, verdict first.

    The verdict leads because the number does not speak for itself: the whole
    point of the 2026-08-08 change is that a +0.4pp delta with an interval
    spanning zero and a +0.4pp delta with an interval clear of it are different
    findings, and the harnesses used to print them identically.
    """
    if paired.get("verdict") == "no_shared_rows":
        return "no_shared_rows (arms were not measured on the same folds)"
    mean = paired.get("mean_diff")
    if mean is None:
        # A caller that could not get far enough to estimate anything. Print
        # the verdict and whatever counts it does have; a `+nan` alongside a
        # confident-looking interval reads worse than saying nothing.
        return f"{paired.get('verdict', '?'):<8} (no estimate — {paired.get('n_clusters', 0)} shared cluster(s))"
    lo, hi = paired.get("ci_lower"), paired.get("ci_upper")
    interval = "interval unresolved" if lo is None or not np.isfinite(lo) else f"[{lo:+.3f}, {hi:+.3f}]{unit}"
    return (
        f"{paired.get('verdict', '?'):<8} "
        f"{mean:+.3f}{unit} "
        f"{interval} "
        f"n={paired.get('n_paired', 0):,} "
        f"folds={paired.get('n_clusters', 0)}"
    )


def verdict(paired: dict, higher_is_better: bool = True) -> str:
    """One word for what a paired interval says, so callers stop inventing one.

    The ten harnesses fixed on 2026-08-08 each had their own rule — a fold
    win-count, a ±0.5pp emoji threshold, a bare `a > b` boolean — and none of
    them was a test. This is: an effect is `positive`/`negative` only when the
    interval excludes zero, and `null` when it does not. `unresolved` means the
    design could not produce an interval at all (fewer than 2 shared clusters),
    which is a different statement from "no effect" and must not be printed as
    one.
    """
    lower, upper = paired.get("ci_lower"), paired.get("ci_upper")
    # `np.isfinite` as well as the None check: a NaN bound fails both `> 0` and
    # `< 0` and would fall through to `null`, printing "no effect" for a
    # comparison that produced no number at all. One NaN row is enough — it
    # propagates through `np.percentile` to both bounds.
    if lower is None or upper is None or not np.isfinite(lower) or not np.isfinite(upper):
        return "unresolved"
    if lower > 0:
        return "positive" if higher_is_better else "negative"
    if upper < 0:
        return "negative" if higher_is_better else "positive"
    return "null"
