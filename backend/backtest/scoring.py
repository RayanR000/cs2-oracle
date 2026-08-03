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

    high_conf = [r for r in records if r["confidence"] == "high"]
    low_conf = [r for r in records if r["confidence"] == "low"]
    high_dir_acc = sum(r["direction_correct"] for r in high_conf) / len(high_conf) * 100 if high_conf else 0
    low_dir_acc = sum(r["direction_correct"] for r in low_conf) / len(low_conf) * 100 if low_conf else 0

    high_interval = [r for r in high_conf if r["in_interval"] is not None]
    high_int_cov = (
        round(sum(r["in_interval"] for r in high_interval) / len(high_interval) * 100, 2)
        if high_interval else 0
    )

    dir_ci_lower, dir_ci_upper = bootstrap_ci([r["direction_correct"] for r in records])
    mae_ci_lower, mae_ci_upper = bootstrap_ci([r["abs_error"] for r in records])

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "wmape": round(wmape, 2),
        "mape_by_tier": mape_by_tier,
        "directional_accuracy": round(directional_accuracy, 2),
        "interval_coverage": round(interval_coverage, 2),
        "baseline_directional_accuracy": round(baseline_directional_accuracy, 2),
        "improvement_over_baseline_pp": round(directional_accuracy - baseline_directional_accuracy, 2),
        "baseline_mae": round(baseline_mae, 4),
        "skill_vs_baseline": round(mae / baseline_mae, 4) if baseline_mae > 0 else None,
        "conf_gap_pp": round(high_dir_acc - low_dir_acc, 2),
        "conf_high_interval_cov": high_int_cov,
        "conf_calibration_error": round(abs(high_dir_acc - CONFIDENCE_TARGET_ACCURACY), 2),
        "directional_accuracy_ci_lower": dir_ci_lower,
        "directional_accuracy_ci_upper": dir_ci_upper,
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
