"""Pure scoring for the forecast backtest.

No I/O, no archive access, no clock. Given frozen outcome records, produces
metrics. Keeping this pure is what makes the reported accuracy reproducible:
after resolution has run, the number is a function of stored data only.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from backtest.actionable import actionable_metrics
from backtest.directional_test import (
    constant_call_baseline,
    pesaran_timmermann,
    realised_down_rate,
)

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


# Staleness bands over `base_stale_run_days`, as (label, lower, upper) with
# upper exclusive and None meaning open.
#
# FIXED BANDS, NOT QUARTILES. The friction-conditioned tier-scoring spec asked
# for a 4-quartile staleness axis, and quartiles are the wrong estimator here:
# the distribution is ~80% zeros on the >=$1 cohort (measured 2026-08-08), so
# q1 = q2 = q3 = 0 and a data-driven cut collapses to a single populated bucket
# that would then silently be reported as four. These bands are chosen to
# straddle the mechanism instead — a fresh level, a single repeat, a short run,
# and a run long enough that the series has plainly stopped reporting.
#
# `unknown` is a band, not a default. Every outcome resolved before 2026-08-08
# carries a NULL base_stale_run_days, and folding those into the fresh bucket
# would report 13 years of unmeasured rows as measured-fresh.
STALENESS_BANDS = (
    ("fresh", 0, 1),
    ("repeat_1", 1, 2),
    ("run_2_6", 2, 7),
    ("run_7_plus", 7, None),
)
STALENESS_UNKNOWN = "unknown"


def staleness_band(run_days) -> str:
    """Which STALENESS_BANDS label a `base_stale_run_days` value falls in."""
    if run_days is None:
        return STALENESS_UNKNOWN
    try:
        v = int(run_days)
    except (TypeError, ValueError):
        return STALENESS_UNKNOWN
    if v < 0:
        return STALENESS_UNKNOWN
    for label, lo, hi in STALENESS_BANDS:
        if v >= lo and (hi is None or v < hi):
            return label
    return STALENESS_UNKNOWN


def score_by_staleness(records: list[dict]) -> dict:
    """Directional accuracy per staleness band. Never pooled.

    The axis `docs/superpowers/specs/2026-08-07-friction-conditioned-tier-scoring-design.md`
    deferred to step 6. Reported as a split rather than used as a filter, for
    the same reason the carry-forward partition is: "how much of our accuracy
    is frozen prices" has to stay answerable.

    Read this against the anchor, not the raw series. `base_stale_run_days` is
    measured on a SMOOTHED anchor (median of the last SMOOTH_WINDOW
    observations), where the >=$1 cohort reads 0-1.8% stale, against 12-27% on
    the unsmoothed voted series the training label path sees. The two are
    different quantities and neither sizes the other.
    """
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        buckets[staleness_band(r.get("base_stale_run_days"))].append(r)

    out = {}
    n = len(records)
    for label in [b[0] for b in STALENESS_BANDS] + [STALENESS_UNKNOWN]:
        rows = buckets.get(label, [])
        out[label] = {
            "n": len(rows),
            "share_pct": round(len(rows) / n * 100, 2) if n else 0.0,
            # None rather than 0.0 on an empty band: an empty band has no
            # accuracy, and a zero reads as the model scoring nothing.
            "directional_accuracy": (
                round(sum(r["direction_correct"] for r in rows) / len(rows) * 100, 2)
                if rows else None
            ),
        }
    return out


def direction_from_return(ret: float) -> str:
    if ret > FLAT_TOLERANCE:
        return "up"
    if ret < -FLAT_TOLERANCE:
        return "down"
    return "flat"


def price_tier(price: float) -> int:
    """Liquidity band, not a display bucket.

    The $1000 cut exists because the bid-ask spread is 10.8% at $50-500 and
    5.2% at $1000+ (n = 22,449) — the two most different liquidity populations
    in the market, which tier 4 used to merge. See backtest/friction.py.

    Rows stored before 2026-08-07 with price_tier == 4 mean >= $100. Tiers 0-3
    are unchanged, so HEADLINE_MIN_TIER and MIN_SERVED_PRICE_USD are unaffected.
    """
    if price >= 1000:
        return 5
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

    # NOTE: this is the always-FLAT call specifically, not the best constant
    # call. It is kept under its original name because the stored series goes
    # back months under this definition; the number DA actually has to beat is
    # `constant_call_accuracy` below.
    baseline_hits = sum(1 for r in records if r["actual_direction"] == "flat")
    baseline_directional_accuracy = baseline_hits / n * 100
    baseline_mae = sum(abs(r["base_price"] - r["actual_price"]) for r in records) / n

    # The headline triple. `directional_accuracy` on its own says nothing: the
    # realised down-rate swings 29.4% -> 76.9% between stored forecast dates
    # while the model's call distribution barely moves, so the same DA is skill
    # on one date and incompetence on another. These three always travel
    # together, and the significance test below is what is actually quoted.
    constant_call_direction, constant_call_accuracy = constant_call_baseline(records)
    down_rate = realised_down_rate(records)

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

    # Serial-correlation-robust Pesaran-Timmermann, computed per forecast date
    # with a t-stat over dates. THIS is the headline; DA is context for it. See
    # backtest/directional_test.py for why the plain version does not apply.
    pt = pesaran_timmermann(records, MIN_FORECAST_DATES)

    # Friction-conditioned accuracy, on the subset whose predicted move clears
    # the round trip plus the tier's spread. Scoped to h in {14, 30}; outside
    # that it reports out_of_scope rather than zeros. See backtest/actionable.py
    # for why a low error and zero utility are compatible.
    #
    # The horizon is read off the first record because records are grouped by
    # (horizon, model_version) before they get here, so a cohort cannot mix
    # horizons. A record predating the field yields None, which is out_of_scope.
    actionable = actionable_metrics(
        records, records[0].get("horizon_days"), MIN_FORECAST_DATES
    )

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "wmape": round(wmape, 2),
        "mape_by_tier": mape_by_tier,
        "directional_accuracy": round(directional_accuracy, 2),
        # The two figures DA must never be quoted without. See the triple
        # comment above.
        "constant_call_accuracy": (
            None if constant_call_accuracy is None else round(constant_call_accuracy, 2)
        ),
        "constant_call_direction": constant_call_direction,
        "realised_down_rate": None if down_rate is None else round(down_rate, 2),
        # Carry-forward split. See the comment above the partition.
        "directional_accuracy_moved": _dir_acc(moved),
        "directional_accuracy_unchanged": _dir_acc(unchanged),
        "n_unchanged": n_unchanged,
        "unchanged_pct": round(n_unchanged / n * 100, 2),
        # The finer staleness axis the 2-bucket carry-forward split above
        # approximates. `unchanged` asks whether the two legs came out equal;
        # this asks how long the anchor had already been frozen before the
        # forecast was made, which is the direction the MA(k) mechanism runs.
        "staleness_bands": score_by_staleness(records),
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
    metrics.update(pt)
    metrics.update(actionable)
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

# price_tier sentinels for the headline floor sweep, mapping sentinel -> floor in
# USD. HEADLINE_TIER is -1 and keeps meaning >= $1: /accuracy/headline, the
# homepage placard and months of stored series all key on it, so the sweep
# extends downward from it rather than renumbering.
#
# The sweep exists because $1 is a CONVENTION, not a derivation — the spread
# evidence (35.5% sub-$1 against 5.2% at $1000+) argues the honest floor is
# above it. These rows are what answers "where does the headline stabilise",
# and they are stored for the same reason HEADLINE_TIER is.
#
# Negative by construction so they can never collide with a band price_tier()
# returns; the API's price_tier query bound is widened to -3 to match. Every
# floor must be a price_tier cut — see floor_records.
FLOOR_SWEEP = {
    -1: 1.0,
    -2: 5.0,
    -3: 20.0,
}


def score_by_tier(records: list[dict]) -> list[tuple[int | None, dict, int]]:
    """Score per price band, once per headline floor, plus an all-tiers aggregate.

    Returns [(tier, metrics, n), ..., (floor sentinels), (None, metrics, n)].
    Cohorts with no records are omitted rather than emitted as zeros, so a floor
    nothing reaches is absent instead of reporting a fabricated 0%.

    The all-tiers (None) row is retained for API defaults and for continuity of a
    series months deep. It is POOLED ACROSS LIQUIDITY POPULATIONS whose spreads
    run 35.5% to 5.2%, so nothing quotes it — the logged headline reads
    HEADLINE_TIER.
    """
    by_tier: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_tier[r["price_tier"]].append(r)

    out: list[tuple[int | None, dict, int]] = []
    for tier in sorted(by_tier):
        metrics, n = score_cohort(by_tier[tier])
        if n:
            out.append((tier, metrics, n))

    # Descending sentinel order (-1, -2, -3) so the widest cohort is emitted
    # first and the log reads as a sweep upward through the floors.
    for sentinel in sorted(FLOOR_SWEEP, reverse=True):
        metrics, n = score_cohort(floor_records(records, FLOOR_SWEEP[sentinel]))
        if n:
            out.append((sentinel, metrics, n))

    metrics, n = score_cohort(records)
    if n:
        out.append((None, metrics, n))
    return out


def floor_records(records: list[dict], floor: float) -> list[dict]:
    """The subset at or above a price floor, in USD.

    Filters in TIER space rather than on base_price directly. Every floor in
    FLOOR_SWEEP is a price_tier cut, so the two are equivalent — but filtering on
    tiers keeps each floor cohort exactly a UNION OF BANDS, which is the property
    the partition tests rely on to catch double-counting, and it keeps
    headline_records selecting the identical population it always has.

    A floor that is not a band edge would silently round down to the band below
    it. `test_every_sweep_floor_is_a_band_edge` is what stops one being added.
    """
    min_tier = price_tier(floor)
    return [r for r in records if r["price_tier"] >= min_tier]


def headline_records(records: list[dict]) -> list[dict]:
    """The >=$1 subset used for the headline log line."""
    return floor_records(records, FLOOR_SWEEP[HEADLINE_TIER])
