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

GAP — a third category, added 2026-08-04. The 08-02/03 collection outage left
the archive holding 2026-08-01 and 2026-08-04 with nothing between (three of
those missing days came from cron drift, not failures — see
collectors.snapshot_date). A horizon-3 forecast dated 08-01 targeting 08-04 then
has fewer than SMOOTH_WINDOW observations available inside `(f, target]`, so
`resolve_anchors` reaches back past the forecast date and the disjoint-leg guard
drops it. 17,382 forecasts died that way in run 30901398468.

Neither existing category fits. Not FRESH: nothing is regressing now, and the
run's own anchor resolution succeeded completely (34,764 of 34,764). Not
CHRONIC: `classify_chronic` asks whether coverage has moved a full window past
target_date, and it had not. So they fell into the fatal coverage ratio and
pinned the gate at 17.9% permanently — a forecast that never resolves never
earns a frozen outcome, so it re-enters `to_resolve` every run forever. Nor could
it be waited out: backfill is unavailable (CSGOTrader serves only `/latest/`, and
CSMarketAPI's per-item history costs 1 request/item against 4,000/month for
41,294 items).

Gap rows are therefore excluded from the fatal ratio — but the exclusion is
deliberately narrow, because "excuse the rows we cannot score" is exactly the
loophole this module exists to prevent:

* It requires positive evidence of missing days from the archive itself, not the
  mere fact that a forecast failed. `classify_archive_gap` returns False when it
  has no coverage information at all.
* Gap rows leave the fresh denominator too, so a real resolver regression behind
  a large gap population still fails at full sensitivity.
* The count, the percentage and the surviving cohort are reported and warned on
  every run, so the shrinkage is attributable rather than invisible. The
  survivor count is quoted as an upper bound only: further rows are dropped
  below this module and the scored denominator is not the gate's to state. See
  the comment on the gap branch.
* If the gap swallows the entire cohort there is no metric left and the run fails
  regardless.

LEG WINDOW, added 2026-09-09. The check above counts covered days over the whole
``(f_date, target_date]`` horizon, which coincides with the leg's effective range
only while the horizon fits inside the resolver's staleness bound. Past it the
count is blind: the 2026-08-28..09-05 collection holes left exactly 2 covered
days in the 7-day window the h=30 actual leg may draw from while the 30-day
horizon still held 23-24, so 16,626 forecasts read as FRESH and failed the run
at 14.3% over a hole no item could resolve through — at most 2 observations
  existed globally where 3 are required. The caller therefore passes the
  staleness bound and only ``f < day <= target`` with ``day >= target -
  staleness`` is counted: the days ``resolve_anchors`` can actually select,
  since anything older fails the anchor-staleness rule and anything at or
  before the forecast fails the disjoint-leg guard. Identical below the
  bound; strictly the leg above it.

BASE LEG, added 2026-09-12. The leg-window check above is blind in the other
direction: the base anchor resolves from up to ``staleness`` days BEFORE the
forecast, and a hole there starves it symmetrically. The same 08-28..09-05
holes left exactly 2 covered days in ``[09-06 - 7, 09-06]`` ({09-02, 09-06})
where ``resolve_anchors`` requires 3, so the whole 5,536-forecast h=3 cohort
dated 09-06 — the first forecast date after the hole — dropped on ``base_none``
while its actual leg (09-07/08/09, all present) classified clean. It counted
FRESH and pinned three runs at 20-25% (34691910482) over a hole no item could
resolve through — permanently, since history is fixed: 08-29 will always sit 8
days before 09-06, so the cell re-enters ``to_resolve`` forever and taxes every
future fresh rate until dilution. ``classify_base_gap`` counts the base window
on the same positive-evidence standard; behaviour elsewhere is unchanged.
  """

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

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
    gap_pct: float | None = None


def evaluate_gate(
    n_mature: int,
    n_attempted: int,
    n_unresolvable_fresh: int,
    n_unresolvable_chronic: int = 0,
    n_unresolvable_gap: int = 0,
    max_pct: float = MAX_UNRESOLVABLE_PCT,
) -> GateResult:
    """Decide whether to report metrics for this run.

    n_mature: the whole mature cohort, frozen rows included. Frozen rows are
        covered — they have usable actuals — so they belong in this denominator
        even though this run did not resolve them.
    n_attempted: forecasts this run tried to resolve, chronic and gap ones
        included.
    n_unresolvable_fresh: failures whose target date sits near the archive's
        coverage edge.
    n_unresolvable_chronic: failures the archive has already moved past. See
        `classify_chronic`.
    n_unresolvable_gap: failures caused by days genuinely absent from the
        archive inside the actual leg's window. See `classify_archive_gap`.
    """
    n_unresolvable = n_unresolvable_fresh + n_unresolvable_chronic
    # Gap rows can never be scored, so what is left is the cohort a metric can
    # honestly be computed over.
    n_scoreable = n_mature - n_unresolvable_gap
    # A chronic row always fails, so chronic attempts and chronic failures are
    # the same set; removing them leaves the attempts that carried information.
    # Gap attempts are removed for the same reason — the archive decided their
    # outcome, not the resolver.
    n_attempted_fresh = max(
        0, n_attempted - n_unresolvable_chronic - n_unresolvable_gap
    )

    coverage_pct = (n_unresolvable / n_mature * 100) if n_mature else None
    gap_pct = (n_unresolvable_gap / n_mature * 100) if n_mature else None
    fresh_rate_pct = (
        (n_unresolvable_fresh / n_attempted_fresh * 100) if n_attempted_fresh else None
    )

    # Checked before the ratios: if nothing is scoreable there is no number to
    # report and no ratio worth quoting, however clean the survivors look.
    if n_mature and n_scoreable <= 0:
        return GateResult(
            ok=False,
            reason=(
                f"nothing scoreable left: all {n_mature:,} mature forecasts fall "
                f"in an archive gap. No metric can be computed — refusing to "
                f"report a number."
            ),
            coverage_pct=coverage_pct,
            fresh_rate_pct=fresh_rate_pct,
            gap_pct=gap_pct,
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
            gap_pct=gap_pct,
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
            gap_pct=gap_pct,
        )

    # Reported before chronic because it is the larger and more surprising
    # population when present, and because it names a fixable upstream cause.
    #
    # It reports n_scoreable as an UPPER BOUND, never as the metric's
    # denominator. Run 31057993603 (2026-08-05) said "Reporting on the 80,737
    # scoreable forecasts" and the metric was then computed over 66,279 — the
    # gate was overstating the cohort by 14,458, an over-report as large as the
    # 17.7% shrinkage it had just warned about. Two drops happen below it and
    # neither is visible from here:
    #
    #   * 14,233 frozen rows with a NULL base_price, dropped by
    #     backtest_accuracy._records_from_frozen_outcomes.
    #   * the 225 chronic rows, which never earn an outcome row at all and so are
    #     not in the scored table either, yet are still inside n_scoreable —
    #     only gap rows are subtracted from it.
    #
    # The fix is for the gate to stop claiming a number it does not own, rather
    # than for the scored count to be threaded back into it. This module is pure
    # by contract (no DB, no archive, no clock) and it runs BEFORE the freeze and
    # the scoring step, so the scored count does not exist yet at this point —
    # reconciling here would mean either reaching into the DB or accepting a
    # value the caller cannot have computed. And nothing is lost by dropping the
    # claim: `_records_from_frozen_outcomes` logs "N scored of M considered" a
    # line later in the same run, which is the honest denominator.
    #
    # n_scoreable itself is deliberately left alone. It still decides the
    # "nothing scoreable left" failure above, and changing its definition would
    # change what the run fails on — this is a reporting fix only.
    if n_unresolvable_gap:
        chronic_note = (
            f" A further {n_unresolvable_chronic:,} are chronically unresolvable."
            if n_unresolvable_chronic
            else ""
        )
        return GateResult(
            ok=True,
            warn=True,
            reason=(
                f"{n_unresolvable_gap:,} of {n_mature:,} mature forecasts "
                f"({gap_pct:.1f}%) are unscoreable: days are missing from the "
                f"archive inside the window one of their legs resolves from — "
                f"the actual leg's (f, target] range or the base leg's "
                f"[f - staleness, f] range — so the leg cannot be established "
                f"within the resolver's window. This is "
                f"a collection gap, not a resolver regression — the fresh "
                f"resolution rate is measured over the {n_attempted_fresh:,} "
                f"attempts that carried information.{chronic_note} That leaves "
                f"{n_scoreable:,} forecasts, which is an UPPER BOUND on the "
                f"scoreable cohort and not the metric's denominator — the count "
                f"actually scored is reported by the frozen-outcome scoring step "
                f"below, which drops rows this gate cannot see."
            ),
            coverage_pct=coverage_pct,
            fresh_rate_pct=fresh_rate_pct,
            gap_pct=gap_pct,
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
            gap_pct=gap_pct,
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
        gap_pct=gap_pct,
    )


def classify_archive_gap(
    f_date,
    target_date,
    covered_days,
    window: int = 3,
    staleness_days: int | None = None,
) -> bool:
    """True when the archive cannot supply a clean actual leg for this forecast.

    The actual leg needs *window* observations drawn from ``(f_date,
    target_date]`` — strictly after the forecast — or ``resolve_anchors`` reaches
    back past ``f_date`` and the disjoint-leg guard drops the forecast. So the
    question is not "is any day missing somewhere in the horizon" but "are there
    even *window* days present in the range that matters". That distinction keeps
    the category tight: a 30-day horizon missing one day 25 days out still has 29
    usable days and is NOT excused, while a 3-day horizon spanning the
    2026-08-02/03 hole has one and is.

    *window* is the resolver's ``SMOOTH_WINDOW``. It is a parameter rather than
    an import so this module stays free of pandas and the collectors package.

    *staleness_days* is the resolver's ``MAX_WINDOW_SPAN_DAYS``. The anchor is
    unresolvable unless *window* observations sit within *staleness_days* of it,
    and the disjoint-leg guard additionally excludes every day at or before the
    forecast — so the only days that can ever back the actual leg satisfy
    ``f_date < day <= target_date`` with ``day >= target_date - staleness_days``. Days outside
    that range cannot be selected by ``resolve_anchors`` under any data, which
    makes counting them a statement about the calendar rather than about the
    leg. Pass the bound and only that range is counted; leave it None for the
    legacy whole-horizon count, which coincides exactly whenever
    ``target_date - f_date <= staleness_days`` (every production horizon at or
    below the bound, including the h=3 shape this category was built for).

    The legacy count is blind past the bound: the 2026-08-28..09-05 collection
    holes left exactly 2 covered days in the 7-day leg window before the
    09-04/05/06 targets while the 30-day horizon still held 23-24, so 16,626
    h=30 forecasts read as FRESH and pinned the gate at 14.3% over a hole no
    item could have resolved through — at most 2 observations existed globally
    where 3 are required.

    An empty *covered_days* means we have no coverage information, which is not
    evidence of a gap — returns False so an unknown never becomes an excuse.
    """
    if not covered_days or f_date is None or target_date is None:
        return False
    # The staleness floor is INCLUSIVE: resolve_anchors drops an observation
    # only when (anchor - day).days > max_span_days, so target - staleness
    # itself still backs the leg. The forecast side stays exclusive — the
    # disjoint-leg guard requires the oldest supporting observation to
    # strictly post-date it.
    if staleness_days is None:
        present = sum(
            1 for day in covered_days if f_date < day <= target_date
        )
    else:
        floor = target_date - timedelta(days=staleness_days)
        present = sum(
            1 for day in covered_days
            if f_date < day <= target_date and day >= floor
        )
    return present < window


def classify_base_gap(
    f_date,
    covered_days,
    window: int = 3,
    *,
    staleness_days: int,
) -> bool:
    """True when the archive cannot supply a clean BASE leg for this forecast.

    The base anchor is the median of the last *window* observations at or
    before ``f_date`` lying within *staleness_days* of it: ``resolve_anchors``
    keeps ``d <= anchor`` and drops the anchor unless the oldest selected
    observation satisfies ``(anchor - day).days <= max_span_days``. So the
    question is whether the range ``[f_date - staleness_days, f_date]`` — both
    ends inclusive, matching the resolver's strict ``>`` comparison — holds
    even *window* covered days globally. Fewer, and no item resolves its base
    through the hole: the h=3 cohort dated 2026-09-06 holds exactly {09-02,
    09-06} in that range against the 08-28..09-05 collection holes.

    The twin of ``classify_archive_gap`` for the other leg, on the same
    positive-evidence standard: an empty *covered_days* is no coverage
    information, not evidence of a gap, and returns False — and item-level
    sparsity (enough days globally, this item still failing) still counts
    FRESH. Only a globally missing range is excused.

    *staleness_days* is required and has no legacy default: unlike the actual
    leg there is no pre-bound caller to stay compatible with, and an unbounded
    lookback would span the whole history and silently never fire.
    """
    if not covered_days or f_date is None:
        return False
    floor = f_date - timedelta(days=staleness_days)
    present = sum(1 for day in covered_days if floor <= day <= f_date)
    return present < window


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
