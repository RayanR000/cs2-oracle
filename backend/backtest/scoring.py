"""Pure scoring for the forecast backtest.

No I/O, no archive access, no clock. Given frozen outcome records, produces
metrics. Keeping this pure is what makes the reported accuracy reproducible:
after resolution has run, the number is a function of stored data only.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

FLAT_TOLERANCE = 0.005
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95
BOOTSTRAP_RNG_SEED = 42
CONFIDENCE_TARGET_ACCURACY = 80.0

# Below this many distinct forecast dates, a cohort cannot separate model skill
# from the market's direction on the days it happens to cover, whatever its
# sample_count says.
#
# Directional outcomes are CLUSTERED BY forecast_date: every item forecast on
# the same day is exposed to the same market-wide move, so N forecasts on one
# date are closer to one observation than to N. The 2026-08-02 cohorts made
# this concrete — 11,009 forecasts at 3d spanning two dates, and 5,461 at 30d
# spanning one. The two dates ran opposite (2025-12-01 rising, 2026-07-17
# falling) and a model that predicts "down" 57-87% of the time regardless
# scored 33% on the first and 64% on the second. The horizon-to-horizon
# "differences" in that report are mostly which of the two dates each cohort
# happened to contain.
#
# 20 is a judgement call, not a derivation: enough dates to span more than one
# market swing without demanding a quarter of history before any number is
# quoted. It is deliberately well above the 2 currently stored.
MIN_FORECAST_DATES = 20


def direction_from_return(ret: float) -> str:
    if ret > FLAT_TOLERANCE:
        return "up"
    if ret < -FLAT_TOLERANCE:
        return "down"
    return "flat"


def price_tier(price: float) -> int:
    if price >= 100:
        return 4
    if price >= 20:
        return 3
    if price >= 5:
        return 2
    if price >= 1:
        return 1
    return 0


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, ci=BOOTSTRAP_CI):
    if len(values) < 10:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    stats = np.empty(n_resamples)
    n = len(values)
    arr = np.array(values)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        stats[i] = np.mean(sample)
    alpha = (100 - ci) / 2
    return (
        round(float(np.percentile(stats, alpha)), 4),
        round(float(np.percentile(stats, 100 - alpha)), 4),
    )


def block_bootstrap_ci(values, clusters, n_resamples=N_BOOTSTRAP, ci=BOOTSTRAP_CI):
    """Bootstrap the mean of *values* by resampling whole *clusters*.

    ``bootstrap_ci`` resamples individual records, which assumes they are
    independent draws. Directional outcomes are not: they are clustered by
    forecast date (see MIN_FORECAST_DATES). Resampling items therefore measures
    only the within-day spread and reports a tight interval around a quantity
    whose real uncertainty is between-day.

    This resamples dates with replacement and recomputes the pooled mean over
    the drawn dates, so the interval reflects the variation that actually
    matters. With few dates it is very wide — that is the honest answer, not a
    defect.

    Returns (None, None) for fewer than 2 clusters: a single date carries no
    information about between-date variation, and any interval derived from one
    would be a fabrication.
    """
    by_cluster: dict = defaultdict(list)
    for value, cluster in zip(values, clusters):
        by_cluster[cluster].append(value)
    # Sorted so the resample draw does not depend on dict insertion order.
    groups = [np.array(by_cluster[k]) for k in sorted(by_cluster)]
    if len(groups) < 2:
        return None, None

    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    n_groups = len(groups)
    sums = np.array([g.sum() for g in groups], dtype=float)
    counts = np.array([g.size for g in groups], dtype=float)

    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_groups, size=n_groups)
        stats[i] = sums[idx].sum() / counts[idx].sum()

    alpha = (100 - ci) / 2
    return (
        round(float(np.percentile(stats, alpha)), 4),
        round(float(np.percentile(stats, 100 - alpha)), 4),
    )


def _as_percent(bounds: tuple) -> tuple:
    """Rescale a (lower, upper) pair of proportions to percent, preserving None.

    None means "not enough data to bootstrap" and must stay None: a 0.0 bound
    would read as a real interval reaching zero accuracy.
    """
    lower, upper = bounds
    return (
        None if lower is None else round(lower * 100, 2),
        None if upper is None else round(upper * 100, 2),
    )


def score_cohort(records: list[dict]) -> tuple[dict, int]:
    """Metrics for one (horizon, model_version, tier) cohort.

    Returns (metrics, sample_count). Does not mutate ``records``.
    """
    n = len(records)
    if n == 0:
        return {}, 0

    mae = sum(r["abs_error"] for r in records) / n
    rmse = math.sqrt(sum(r["sq_error"] for r in records) / n)
    mape = sum(r["pct_error"] for r in records) / n

    directional_accuracy = sum(r["direction_correct"] for r in records) / n * 100

    interval_records = [r for r in records if r["in_interval"] is not None]
    interval_total = len(interval_records)
    interval_hits = sum(r["in_interval"] for r in interval_records)
    interval_coverage = (interval_hits / interval_total * 100) if interval_total else 0

    total_actual = sum(r["actual_price"] for r in records)
    wmape = (sum(r["abs_error"] for r in records) / total_actual * 100) if total_actual > 0 else 0

    tier_errors = defaultdict(list)
    for r in records:
        tier_errors[r["price_tier"]].append(r["pct_error"])
    mape_by_tier = {
        f"tier_{t}": round(sum(errs) / len(errs), 2)
        for t, errs in sorted(tier_errors.items())
    }

    baseline_hits = sum(1 for r in records if r["actual_direction"] == "flat")
    baseline_directional_accuracy = baseline_hits / n * 100
    baseline_mae = sum(abs(r["base_price"] - r["actual_price"]) for r in records) / n

    # A price the archive carried forward is not a prediction the model got
    # right. Measured 2026-08-05: 30-36% of scored outcomes have actual_price
    # BIT-IDENTICAL to base_price, and the rate barely decays from 3d (32.4%) to
    # 30d (31.3%) — genuine no-trade would decay with horizon, so that population
    # is dominated by archive carry-forward, not market behaviour. Those rows
    # label "flat" by construction, so pooling them into one headline makes the
    # number partly a measure of archive staleness.
    #
    # Reported as a split rather than filtered out: "how much of our accuracy is
    # unchanged prices" is a question the partition keeps answerable, and the
    # same reasoning the tier rows follow (see HEADLINE_MIN_TIER). Each partition
    # is None when empty rather than 0.0 — an empty partition has no accuracy,
    # and a zero would be read as the model scoring nothing.
    unchanged = [r for r in records if r["actual_price"] == r["base_price"]]
    moved = [r for r in records if r["actual_price"] != r["base_price"]]
    n_unchanged = len(unchanged)

    def _dir_acc(rows):
        if not rows:
            return None
        return round(sum(r["direction_correct"] for r in rows) / len(rows) * 100, 2)

    high_conf = [r for r in records if r["confidence"] == "high"]
    low_conf = [r for r in records if r["confidence"] == "low"]
    high_dir_acc = sum(r["direction_correct"] for r in high_conf) / len(high_conf) * 100 if high_conf else 0
    low_dir_acc = sum(r["direction_correct"] for r in low_conf) / len(low_conf) * 100 if low_conf else 0

    high_interval = [r for r in high_conf if r["in_interval"] is not None]
    high_int_cov = (
        round(sum(r["in_interval"] for r in high_interval) / len(high_interval) * 100, 2)
        if high_interval else 0
    )

    # The bootstraps average the raw 0/1 direction_correct indicators, so their
    # bounds come back as FRACTIONS while directional_accuracy is a PERCENT.
    # Reported side by side under names differing only by suffix, that was a
    # reading trap: the 2026-08-05 dump carried
    # 'directional_accuracy': 49.57 next to 'directional_accuracy_ci_lower':
    # 0.4854, which reads as an interval excluding its own point estimate by 49
    # points. Rescaled here, at the one place the fractions are produced, rather
    # than at each display site — _score_groups compensated with a * 100 in the
    # log line only, so the console was right and the STORED row, the one that
    # gets audited, was not.
    #
    # mae_ci_* is deliberately NOT rescaled: it is in dollars, the same units as
    # mae, and always was.
    dir_ci_lower, dir_ci_upper = _as_percent(
        bootstrap_ci([r["direction_correct"] for r in records])
    )
    mae_ci_lower, mae_ci_upper = bootstrap_ci([r["abs_error"] for r in records])

    # Records predating this field score with no date attributed rather than
    # crashing; they then report 0 distinct dates and fail the sufficiency
    # check, which is the correct reading of "we cannot tell".
    forecast_dates = [r.get("forecast_date") for r in records]
    distinct_dates = len({d for d in forecast_dates if d is not None})
    dir_ci_cl_lower, dir_ci_cl_upper = _as_percent(
        block_bootstrap_ci(
            [r["direction_correct"] for r in records], forecast_dates
        )
    )

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "wmape": round(wmape, 2),
        "mape_by_tier": mape_by_tier,
        "directional_accuracy": round(directional_accuracy, 2),
        # Carry-forward split. See the comment above the partition.
        "directional_accuracy_moved": _dir_acc(moved),
        "directional_accuracy_unchanged": _dir_acc(unchanged),
        "n_unchanged": n_unchanged,
        "unchanged_pct": round(n_unchanged / n * 100, 2),
        "interval_coverage": round(interval_coverage, 2),
        "baseline_directional_accuracy": round(baseline_directional_accuracy, 2),
        "improvement_over_baseline_pp": round(directional_accuracy - baseline_directional_accuracy, 2),
        "baseline_mae": round(baseline_mae, 4),
        "skill_vs_baseline": round(mae / baseline_mae, 4) if baseline_mae > 0 else None,
        "conf_gap_pp": round(high_dir_acc - low_dir_acc, 2),
        "conf_high_interval_cov": high_int_cov,
        "conf_calibration_error": round(abs(high_dir_acc - CONFIDENCE_TARGET_ACCURACY), 2),
        # All four bounds are PERCENT, matching directional_accuracy. Rows
        # written before 2026-08-05 hold the same numbers as fractions.
        #
        # Item-resampled. Retained for continuity with the stored series, but
        # it understates the uncertainty — prefer the clustered pair below.
        "directional_accuracy_ci_lower": dir_ci_lower,
        "directional_accuracy_ci_upper": dir_ci_upper,
        # Forecast-date-resampled: the interval that respects the clustering.
        "directional_accuracy_ci_clustered_lower": dir_ci_cl_lower,
        "directional_accuracy_ci_clustered_upper": dir_ci_cl_upper,
        "distinct_forecast_dates": distinct_dates,
        "date_coverage_sufficient": distinct_dates >= MIN_FORECAST_DATES,
        "mae_ci_lower": mae_ci_lower,
        "mae_ci_upper": mae_ci_upper,
    }
    return metrics, n


# Tiers at or above this are aggregated into the headline figure. Tier 0
# (<$1) is 72% of the evaluated universe and one cent there is a 20% move,
# so its up/flat/down label is dominated by tick quantisation. It is
# reported separately rather than filtered out — "the model is worse on
# penny items" is a real question the tier rows keep answerable.
HEADLINE_MIN_TIER = 1

# The >=$1 aggregate is stored under this sentinel tier. It is NOT a price
# band: real tiers are 0..4 and the all-tiers aggregate is NULL, so the
# headline needed a third thing to be. Negative by construction so it can
# never collide with a band price_tier() returns.
#
# It is stored rather than only logged because the headline is the one number
# quoted as "the model's accuracy", and a figure that exists only in a run's
# console output cannot be audited or recomputed. The 2026-08-01 changelog
# quoted a headline up to 8pp off the stored tier rows and nothing could catch
# it. Every other row in this function was already persisted; this one wasn't.
HEADLINE_TIER = -1


def score_by_tier(records: list[dict]) -> list[tuple[int | None, dict, int]]:
    """Score per price tier, the >=$1 headline, plus an all-tiers aggregate.

    Returns [(tier, metrics, n), ..., (HEADLINE_TIER, ...), (None, metrics, n)].
    Cohorts with no records are omitted rather than emitted as zeros, so the
    headline is absent when nothing reaches HEADLINE_MIN_TIER.
    """
    by_tier: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_tier[r["price_tier"]].append(r)

    out: list[tuple[int | None, dict, int]] = []
    for tier in sorted(by_tier):
        metrics, n = score_cohort(by_tier[tier])
        if n:
            out.append((tier, metrics, n))

    metrics, n = score_cohort(headline_records(records))
    if n:
        out.append((HEADLINE_TIER, metrics, n))

    metrics, n = score_cohort(records)
    if n:
        out.append((None, metrics, n))
    return out


def headline_records(records: list[dict]) -> list[dict]:
    """The >=$1 subset used for the headline log line."""
    return [r for r in records if r["price_tier"] >= HEADLINE_MIN_TIER]
