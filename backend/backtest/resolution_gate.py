"""Whether a backtest run's resolution coverage is good enough to report.

Pure: no DB, no archive, no clock. Split out of backtest_accuracy so the
decision can be tested on numbers instead of fixtures.

The gate exists because silently scoring a shrunken cohort is how the pre-fix
metric moved unnoticed. Getting its denominator right turned out to be subtler
than a single ratio, because two failure modes pull opposite ways:

HYPERSENSITIVITY. Dividing only by the forecasts that required resolution this
run works while the cohort is fresh and breaks completely once it is frozen.
After the 2026-08-01 backfill froze 75,195 mature forecasts, the only rows
still requiring resolution were the 225 chronically-unresolvable stragglers
that same backfill had logged as "225 (0.4%)". They are by definition 100% of
themselves, so the daily run failed with "100.0% of mature forecasts could not
be resolved" over 0.3% of the real cohort — and would have failed that way
every day.

DILUTION. Dividing by the whole mature cohort instead lets a genuinely broken
resolver drown: 500 consecutive fresh failures against 75,000 frozen rows is
0.7%, under any sane cap. This is why the original denominator was chosen, and
the concern is legitimate.

The discriminator is NOT how many rows failed — 225 chronic and 500 fresh
failures are indistinguishable by count, and any attempt floor tuned between
them is fragile against a drifting straggler population. It is whether the
archive has already moved past the forecast's target date:

* CHRONIC — the archive covers well beyond target_date and the forecast still
  will not resolve, so the data is never arriving. These re-enter `to_resolve`
  every single run, because a forecast that cannot resolve never earns a frozen
  outcome. They are a fixed tax, not a signal.
* FRESH — target_date is close to the archive's coverage edge. A cluster of
  these failing is what a broken resolver or a stalled collector looks like.

So the gate checks two ratios, neither of which needs an arbitrary floor:

* COVERAGE — all unresolvable / mature. Catches mass shrinkage, the failure the
  gate was built for. Insensitive to how much of the cohort is frozen.
* FRESH RATE — fresh unresolvable / fresh attempts. Catches a resolver
  regression at full sensitivity, undiluted by either the frozen majority or
  the chronic tail.

Chronic rows are reported and warned on, never fatal. They still count toward
coverage, so a chronic population that grows large enough to matter fails the
run on that ratio rather than being excused indefinitely.
"""

from __future__ import annotations

from dataclasses import dataclass

# Above this rate, refuse to report a number rather than silently score a
# shrunken cohort. Applied to both ratios below.
MAX_UNRESOLVABLE_PCT = 10.0


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str
    coverage_pct: float | None
    fresh_rate_pct: float | None
    warn: bool = False


def evaluate_gate(
    n_mature: int,
    n_attempted: int,
    n_unresolvable_fresh: int,
    n_unresolvable_chronic: int = 0,
    max_pct: float = MAX_UNRESOLVABLE_PCT,
) -> GateResult:
    """Decide whether to report metrics for this run.

    n_mature: the whole mature cohort, frozen rows included. Frozen rows are
        covered — they have usable actuals — so they belong in this denominator
        even though this run did not resolve them.
    n_attempted: forecasts this run tried to resolve, chronic ones included.
    n_unresolvable_fresh: failures whose target date sits near the archive's
        coverage edge.
    n_unresolvable_chronic: failures the archive has already moved past. See
        `classify_chronic`.
    """
    n_unresolvable = n_unresolvable_fresh + n_unresolvable_chronic
    # A chronic row always fails, so chronic attempts and chronic failures are
    # the same set; removing them leaves the attempts that carried information.
    n_attempted_fresh = max(0, n_attempted - n_unresolvable_chronic)

    coverage_pct = (n_unresolvable / n_mature * 100) if n_mature else None
    fresh_rate_pct = (
        (n_unresolvable_fresh / n_attempted_fresh * 100) if n_attempted_fresh else None
    )

    if coverage_pct is not None and coverage_pct > max_pct:
        return GateResult(
            ok=False,
            reason=(
                f"{n_unresolvable:,} of {n_mature:,} mature forecasts "
                f"({coverage_pct:.1f}%) could not be resolved and have no "
                f"usable outcome, over the {max_pct}% cap. Silent cohort "
                f"shrinkage is how this metric moved unnoticed before — "
                f"refusing to report a number."
            ),
            coverage_pct=coverage_pct,
            fresh_rate_pct=fresh_rate_pct,
        )

    if fresh_rate_pct is not None and fresh_rate_pct > max_pct:
        return GateResult(
            ok=False,
            reason=(
                f"resolution rate: {n_unresolvable_fresh:,} of "
                f"{n_attempted_fresh:,} newly-resolvable forecasts "
                f"({fresh_rate_pct:.1f}%) could not be resolved, over the "
                f"{max_pct}% cap. Cohort coverage is still "
                f"{coverage_pct:.1f}%, so this is a resolver or archive "
                f"regression rather than cohort shrinkage — refusing to report "
                f"a number."
            ),
            coverage_pct=coverage_pct,
            fresh_rate_pct=fresh_rate_pct,
        )

    if n_unresolvable_chronic:
        return GateResult(
            ok=True,
            warn=True,
            reason=(
                f"{n_unresolvable_chronic:,} chronically unresolvable forecast(s) "
                f"re-attempted and skipped — the archive covers past their target "
                f"date, so the data is not coming. They are "
                f"{coverage_pct:.1f}% of the {n_mature:,}-forecast cohort and are "
                f"re-attempted every run because a forecast that cannot resolve "
                f"never earns a frozen outcome. Reporting anyway."
            ),
            coverage_pct=coverage_pct,
            fresh_rate_pct=fresh_rate_pct,
        )

    parts = []
    if coverage_pct is not None:
        parts.append(f"coverage {n_unresolvable:,}/{n_mature:,} ({coverage_pct:.1f}%)")
    if fresh_rate_pct is not None:
        parts.append(
            f"fresh rate {n_unresolvable_fresh:,}/{n_attempted_fresh:,} "
            f"({fresh_rate_pct:.1f}%)"
        )
    return GateResult(
        ok=True,
        reason="; ".join(parts) or "nothing mature to resolve",
        coverage_pct=coverage_pct,
        fresh_rate_pct=fresh_rate_pct,
    )


def classify_chronic(target_date, coverage_end, grace_days: int) -> bool:
    """True when the archive has moved far enough past *target_date* that a
    failure to resolve cannot be waiting on data.

    *grace_days* comes from the resolver's own staleness bound
    (MAX_WINDOW_SPAN_DAYS), so "the archive has moved past it" means the same
    span here as everywhere else: once coverage extends a full window beyond the
    target and the anchors still do not resolve, no future collection changes
    that.
    """
    if target_date is None or coverage_end is None:
        return False
    return (coverage_end - target_date).days > grace_days
