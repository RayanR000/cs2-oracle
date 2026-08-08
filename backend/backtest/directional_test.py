"""Significance testing for directional forecasts.

Pure, like ``backtest.scoring``: given frozen outcome records this produces a
test statistic and nothing else. No I/O, no archive access, no clock.

WHY THIS EXISTS
---------------
Raw directional accuracy is not interpretable on this data, and quoting it as
the headline was measuring the market rather than the model.

``2026-08-03-accuracy-is-clustered-by-forecast-date.md`` measured an
always-down constant call scoring **29.4% on 2025-12-01 and 76.9% on
2026-07-17** at 7d, against a model that says "down" 57-87% of the time no
matter what the date is. A DA of 60% is therefore excellent on the first date
and terrible on the second, and a pooled DA over a handful of dates is mostly a
report of which dates the cohort happened to contain.

That observation *is* the null hypothesis of Pesaran & Timmermann (1992, JBES
10(4), 461-465): under independence between the forecast and the realisation,
the expected hit rate is not 50%, it is

    P* = sum_k P(predicted = k) * P(actual = k)

which is exactly what "always-down wins when the market falls" describes. The
statistic below measures the hit rate in *excess* of that, so a date on which
everything fell raises the null in step with the model's score and contributes
nothing.

WHAT IS COMPUTED
----------------
1. **Per forecast date**, the excess hit rate ``e_d = hit_d - P*_d``, with
   ``P*_d`` estimated from that date's own predicted and realised label
   distributions. Estimating the null per date is what removes the market
   effect; a pooled ``P*`` would still let a one-sided market period masquerade
   as skill.
2. **Over dates**, a t-statistic on the mean of ``e_d`` using a Newey-West
   (Bartlett-kernel) HAC long-run variance. This is the Blaskowitz & Herwartz
   (2014, IJF 30(1)) serial-correlation-robust treatment: the plain PT variance
   assumes independent draws, and this archive's carry-forward prices guarantee
   they are not — a price the archive repeated for six days produces six
   outcomes that are nearly the same observation.

Both departures from textbook PT push the same way: the per-date null is
higher than a pooled one on trending markets, and the HAC variance is larger
than the i.i.d. one under positive autocorrelation. The statistic is therefore
conservative by construction, which is the direction an honest headline should
err in.

CAVEATS, STATED RATHER THAN HIDDEN
----------------------------------
- ``hit_d`` and ``P*_d`` come from the same sample, so ``e_d`` carries the
  within-date estimation noise of both. That noise lands in the between-date
  variance and inflates the standard error; it does not bias the mean.
- The HAC lag is in *observations*, not calendar days. Forecast dates are
  near-daily but not evenly spaced, so lag j means "j forecast dates back", not
  "j days back". Standard practice, and it only matters if the gaps are large
  relative to the autocorrelation length.
- A significantly NEGATIVE statistic is a finding, not a failure to report. It
  means the model's calls are anti-correlated with outcomes after the market
  effect is removed, and ``pt_verdict`` names it ``perverse`` for that reason.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from scipy import stats

# Harvey, Liu & Zhu (2016, RFS 29(1), 5-68) argue that a t-stat of ~2.0 is far
# too weak a bar once you account for how many hypotheses a research programme
# actually tests, and propose ~3.0 for a newly claimed effect. This repo has run
# well over a dozen feature A/Bs against the same outcome series, so 3.0 is the
# applicable bar, not the conventional 1.96.
PT_T_HURDLE = 3.0

# A date contributes nothing usable below this many records. The null P*_d is
# estimated from that date's own label distributions, so at tiny n_d it is
# degenerate — at n_d = 1 the excess is identically 0 whatever happened, which
# would drag the mean toward zero AND shrink the estimated variance, i.e. look
# like precision that is not there.
#
# 30 is a judgement call in the same spirit as MIN_FORECAST_DATES: large enough
# that the two label distributions are estimable, small enough that no real
# forecast date is excluded (production dates carry thousands of rows).
PT_MIN_ROWS_PER_DATE = 30


def _sort_key(value):
    """Order forecast dates without assuming they are all the same type.

    ``date``, ``datetime`` and ISO strings all sort correctly under ``str``, and
    a mixed column — which the frozen-outcome and rescore paths could in
    principle produce — sorts rather than raising. Order matters here: the HAC
    autocovariances are computed over adjacent observations.
    """
    return str(value)


def newey_west_lag(n_obs: int) -> int:
    """Newey & West (1994) automatic bandwidth, ``floor(4 * (T/100)^(2/9))``.

    At the 20-date floor this returns 2; it reaches 3 at 61 dates and 4 at 145.
    Capped at ``T - 1`` because there is no lag-T autocovariance to estimate.
    """
    if n_obs < 2:
        return 0
    lag = int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    return max(0, min(lag, n_obs - 1))


def hac_long_run_variance(values: list[float], lag: int) -> float:
    """Bartlett-kernel long-run variance of a series.

    ``gamma_0 + 2 * sum_j (1 - j/(lag+1)) * gamma_j``. The Bartlett weights make
    the estimator positive semi-definite, so the clamp at zero can only ever fire
    on floating-point dust — it is there so a caller never takes the square root
    of a negative number.
    """
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    dev = [v - mean for v in values]
    omega = sum(d * d for d in dev) / n
    for j in range(1, lag + 1):
        gamma_j = sum(dev[i] * dev[i - j] for i in range(j, n)) / n
        omega += 2.0 * (1.0 - j / (lag + 1.0)) * gamma_j
    return max(omega, 0.0)


def _excess_hit_rate(rows: list[dict]) -> float:
    """One date's hit rate minus the Pesaran-Timmermann independence null.

    The hit leg reads the stored ``direction_correct`` verdict rather than
    re-comparing the labels, so this can never disagree with the
    ``directional_accuracy`` reported beside it — ``_derive_verdict`` in
    ``scripts/backtest_accuracy.py`` is the single producer of both, and both
    legs of a row are refreshed together.

    The null leg needs the label distributions, which is what makes it a null:
    it is the hit rate the same two marginal distributions would produce if the
    prediction carried no information about the outcome.
    """
    n = len(rows)
    hit_rate = sum(r["direction_correct"] for r in rows) / n
    predicted = Counter(r["predicted_direction"] for r in rows)
    actual = Counter(r["actual_direction"] for r in rows)
    independent = sum(predicted[k] * actual[k] for k in predicted) / (n * n)
    return hit_rate - independent


def constant_call_baseline(records: list[dict]) -> tuple[str | None, float | None]:
    """The best single fixed call and the accuracy it would have scored, in percent.

    This is the number DA has to beat to mean anything, and it is NOT
    ``baseline_directional_accuracy`` in ``score_cohort`` — that one is the
    always-*flat* call specifically, kept for continuity of the stored series.
    Ties break on the label so the answer is deterministic.
    """
    counts = Counter(r["actual_direction"] for r in records)
    if not counts:
        return None, None
    direction, hits = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return direction, hits / len(records) * 100


def realised_down_rate(records: list[dict]) -> float | None:
    """Fraction of outcomes that actually fell, in percent.

    Published beside DA because it is the single number that says whether a
    given DA is impressive: the same 60% is skill against a 50% down-rate and
    incompetence against a 77% one.
    """
    if not records:
        return None
    n_down = sum(1 for r in records if r["actual_direction"] == "down")
    return n_down / len(records) * 100


def pesaran_timmermann(records: list[dict], min_dates: int) -> dict:
    """Serial-correlation-robust PT statistic for one cohort.

    Returns a dict that is always the same shape — every field present, None
    where it could not be computed — so a consumer never has to branch on key
    existence, and an absent statistic is visibly absent rather than silently
    missing from the stored metrics.

    ``min_dates`` is the caller's date-coverage floor (``MIN_FORECAST_DATES``);
    it affects only ``pt_verdict``. The statistic itself is computed and
    reported whenever it is computable, because "we ran it and it was
    inconclusive" and "we could not run it" are different states and the second
    one must not be reported as the first.
    """
    empty = {
        "pt_excess_pp": None,
        "pt_t_stat": None,
        "pt_p_value": None,
        "pt_n_dates": 0,
        "pt_n_dates_dropped": 0,
        "pt_nw_lag": None,
        "pt_verdict": "insufficient_dates",
        "pt_hurdle_t": PT_T_HURDLE,
    }

    by_date: dict = defaultdict(list)
    for r in records:
        forecast_date = r.get("forecast_date")
        # A record with no date cannot be assigned to a cluster, and pooling
        # such rows into one pseudo-date would invent a market day. Dropped, and
        # counted, rather than guessed at.
        if forecast_date is not None:
            by_date[forecast_date].append(r)

    ordered = sorted(by_date, key=_sort_key)
    usable = [d for d in ordered if len(by_date[d]) >= PT_MIN_ROWS_PER_DATE]
    dropped = len(ordered) - len(usable)
    empty["pt_n_dates_dropped"] = dropped

    n_dates = len(usable)
    empty["pt_n_dates"] = n_dates
    if n_dates < 2:
        # One date carries no between-date variation, so there is no standard
        # error to form. Reporting a point estimate with no interval here is
        # what produced the numbers this whole step exists to retire.
        return empty

    excess = [_excess_hit_rate(by_date[d]) for d in usable]
    mean_excess = sum(excess) / n_dates

    lag = newey_west_lag(n_dates)
    omega = hac_long_run_variance(excess, lag)
    se = math.sqrt(omega / n_dates)

    out = dict(empty)
    out["pt_excess_pp"] = round(mean_excess * 100, 3)
    out["pt_nw_lag"] = lag

    if se <= 0.0:
        # Every date produced an identical excess, so there is no between-date
        # variation to form a standard error from. Degenerate rather than
        # infinitely significant; refuse to divide.
        #
        # The important instance of this is not an edge case: a CONSTANT call
        # sits at excess exactly 0 on every date, because when the model says
        # "down" every time, P*_d equals the realised down-rate and cancels the
        # hit rate term for term. "Always-down beats the model" is therefore the
        # null holding, at t undefined rather than t large — and the verdict
        # here must not read as skill.
        out["pt_verdict"] = "degenerate"
        return out

    t_stat = mean_excess / se
    out["pt_t_stat"] = round(t_stat, 4)
    # Kept to significant figures rather than decimal places: a decisive result
    # has a p-value many orders below 1e-6, and round(p, 6) stores it as
    # exactly 0.0 — a number that is both false and unfalsifiable.
    p_value = float(2 * stats.t.sf(abs(t_stat), n_dates - 1))
    out["pt_p_value"] = float(f"{p_value:.4g}")

    if n_dates < min_dates:
        out["pt_verdict"] = "insufficient_dates"
    elif t_stat > PT_T_HURDLE:
        out["pt_verdict"] = "skill"
    elif t_stat < -PT_T_HURDLE:
        out["pt_verdict"] = "perverse"
    else:
        out["pt_verdict"] = "no_skill"
    return out
