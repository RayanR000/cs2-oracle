"""Resolve and freeze shadow candidate outcomes.

Candidate resolution reuses the canonical production price resolver and all
existing rules for maturity, archive coverage, stale runs, excluded forecast
dates, and universe membership. For an item/date/horizon shared by production
and a candidate, the outcome legs must be identical; the resolver reports
mismatched shared outcomes as an invariant violation rather than silently
scoring the two arms on different estimators.

Outcome values are immutable after first successful resolution. Only an
explicit `--reresolve` maintenance path may replace them. Ordinary rescoring
recomputes derived metrics from the frozen prices and never reopens the
archive.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backtest.price_resolution import (
    MAX_WINDOW_SPAN_DAYS,
    SMOOTH_WINDOW,
    archive_covered_days,
    archive_max_day,
    load_voted_prices,
    resolve_anchors,
)
from backtest.resolution_gate import classify_archive_gap, classify_base_gap, classify_chronic

logger = logging.getLogger(__name__)

RESOLUTION_VERSION = "v1"

# SQLite's bind-parameter cap is 999; every chunked statement here uses the
# same conservative batch size as scripts/backtest_accuracy.py.
CHUNK = 900


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _derived_metrics(candidate, base: float, actual: float) -> dict:
    """Derived metrics for one candidate against frozen legs.

    Centre candidates carry absolute/percentage error and interval
    membership against their stored absolute low/high. Ranking rows store
    the legs with null error/coverage.
    """
    if candidate.component == "centre":
        absolute_error = abs(float(candidate.centre_price) - actual)
        percentage_error = absolute_error / base * 100.0 if base else None
        low = float(candidate.predicted_price_low)
        high = float(candidate.predicted_price_high)
        in_interval = low <= actual <= high
        return {
            "absolute_error": absolute_error,
            "percentage_error": percentage_error,
            "in_interval": in_interval,
        }
    return {"absolute_error": None, "percentage_error": None, "in_interval": None}


def _rescore_candidate_outcomes(db) -> dict:
    """Recompute derived metrics from frozen legs. Never reads the archive."""
    from database import ForecastCandidate, ForecastCandidateOutcome

    outcomes = db.query(ForecastCandidateOutcome).all()
    if not outcomes:
        return {"rescored": 0}
    candidates = {
        c.id: c
        for c in db.query(ForecastCandidate)
        .filter(ForecastCandidate.id.in_([o.candidate_id for o in outcomes]))
        .all()
    }
    rescored = 0
    for outcome in outcomes:
        candidate = candidates.get(outcome.candidate_id)
        if candidate is None:
            continue
        derived = _derived_metrics(candidate, outcome.base_price, outcome.actual_price)
        changed = False
        for key, value in derived.items():
            if getattr(outcome, key) != value:
                setattr(outcome, key, value)
                changed = True
        if changed:
            rescored += 1
    db.commit()
    logger.info(f"  Candidate rescore: refreshed derived metrics on {rescored} outcome(s)")
    return {"rescored": rescored}


def resolve_candidate_outcomes(
    db, *, today, archive_dir=None, reresolve: bool = False, rescore: bool = False
) -> dict:
    """Resolve mature shadow candidates against frozen outcome legs.

    Prefers an existing frozen production outcome for each shared
    (item, date, horizon); the archive is read only where production has no
    row. A production outcome whose legs differ from a fresh archive
    resolution raises RuntimeError. Returns count stats.
    """
    from database import ForecastCandidate, ForecastCandidateOutcome, ForecastOutcome

    if rescore:
        return _rescore_candidate_outcomes(db)
    if archive_dir is None:
        raise ValueError("archive_dir is required unless rescore=True")
    archive_dir = Path(archive_dir)
    today = _as_date(today)

    coverage_end = archive_max_day(archive_dir)
    cutoff = min(today, coverage_end)

    candidates = db.query(ForecastCandidate).all()
    stats = {"resolved": 0, "reresolved": 0, "skipped_frozen": 0, "skipped_immature": 0, "unresolvable": 0}
    if not candidates:
        return stats

    existing = {o.candidate_id: o for o in db.query(ForecastCandidateOutcome).all()}

    item_ids = list({c.item_id for c in candidates})
    prod_rows = []
    for i in range(0, len(item_ids), CHUNK):
        batch = item_ids[i : i + CHUNK]
        prod_rows.extend(db.query(ForecastOutcome).filter(ForecastOutcome.item_id.in_(batch)).all())
    prod_map: dict[tuple, Any] = {}
    for row in prod_rows:
        key = (row.item_id, _as_date(row.forecast_date), row.horizon_days)
        prod_map.setdefault(key, row)

    slug_rows = db.execute(text("SELECT id, item_id FROM items")).fetchall()
    id_to_slug = {r.id: r.item_id for r in slug_rows}

    # Unique (item, date, horizon) keys still needing archive legs.
    need_archive: dict[tuple, list] = {}
    for candidate in candidates:
        fdate = _as_date(candidate.forecast_date)
        if fdate is None or fdate + timedelta(days=candidate.horizon_days) > cutoff:
            stats["skipped_immature"] += 1
            continue
        outcome = existing.get(candidate.id)
        if outcome is not None and not reresolve:
            stats["skipped_frozen"] += 1
            continue
        prod = prod_map.get((candidate.item_id, fdate, candidate.horizon_days))
        if prod is not None and not reresolve:
            _create_or_update(db, existing, candidate, prod.base_price, prod.actual_price, prod.resolved_at, stats)
            continue
        need_archive.setdefault((candidate.item_id, fdate, candidate.horizon_days), []).append(candidate)

    if need_archive:
        covered_days = archive_covered_days(archive_dir)
        slugs: set[str] = set()
        anchors: set[tuple[str, date]] = set()
        key_slugs: dict[tuple, str | None] = {}
        for (item_id, fdate, horizon) in need_archive:
            slug = id_to_slug.get(item_id)
            key_slugs[(item_id, fdate, horizon)] = slug
            if slug is None:
                continue
            slugs.add(slug)
            anchors.add((slug, fdate))
            anchors.add((slug, fdate + timedelta(days=horizon)))
        prices: dict = {}
        if anchors:
            anchor_dates = [a[1] for a in anchors]
            voted = load_voted_prices(archive_dir, sorted(slugs), min(anchor_dates), max(anchor_dates))
            prices = resolve_anchors(voted, anchors)

        for key, group in need_archive.items():
            item_id, fdate, horizon = key
            assert fdate is not None  # narrowed at insertion site above
            slug = key_slugs[key]
            prod = prod_map.get(key)
            if slug is None:
                if reresolve and prod is not None:
                    # No archive identity to verify against; track production's
                    # current legs rather than leaving the batch unresolved.
                    for candidate in group:
                        _create_or_update(
                            db, existing, candidate, prod.base_price, prod.actual_price, prod.resolved_at, stats
                        )
                else:
                    stats["unresolvable"] += len(group)
                continue
            target_date = fdate + timedelta(days=horizon)
            chronic = classify_chronic(target_date, coverage_end, MAX_WINDOW_SPAN_DAYS)
            gap = classify_archive_gap(
                fdate, target_date, covered_days, SMOOTH_WINDOW, staleness_days=MAX_WINDOW_SPAN_DAYS
            ) or classify_base_gap(fdate, covered_days, SMOOTH_WINDOW, staleness_days=MAX_WINDOW_SPAN_DAYS)
            base_res = prices.get((slug, fdate))
            actual_res = prices.get((slug, target_date))
            legs = None
            if (
                gap is None
                and not chronic
                and base_res is not None
                and actual_res is not None
                and base_res.price > 0
                and actual_res.price > 0
                and actual_res.oldest_observation > fdate
            ):
                legs = (base_res.price, actual_res.price)
            if legs is None:
                stats["unresolvable"] += len(group)
                continue
            base, actual = legs
            if prod is not None and (prod.base_price != base or prod.actual_price != actual):
                raise RuntimeError(
                    "candidate/production resolution mismatch at "
                    f"(item_id={item_id}, forecast_date={fdate}, horizon={horizon}): "
                    f"archive ({base}, {actual}) != production "
                    f"({prod.base_price}, {prod.actual_price})"
                )
            for candidate in group:
                _create_or_update(db, existing, candidate, base, actual, None, stats)

    db.commit()
    logger.info(
        "  Candidate outcomes: "
        f"{stats['resolved']} resolved, {stats['reresolved']} re-resolved, "
        f"{stats['skipped_frozen']} frozen, {stats['skipped_immature']} immature, "
        f"{stats['unresolvable']} unresolvable"
    )
    return stats


def _create_or_update(db, existing, candidate, base, actual, resolved_at, stats):
    """Create or refresh one candidate outcome row with derived metrics."""
    from database import ForecastCandidateOutcome, utcnow_naive

    derived = _derived_metrics(candidate, base, actual)
    outcome = existing.get(candidate.id)
    if outcome is None:
        db.add(
            ForecastCandidateOutcome(
                candidate_id=candidate.id,
                base_price=base,
                actual_price=actual,
                resolved_at=resolved_at if resolved_at is not None else utcnow_naive(),
                resolution_version=RESOLUTION_VERSION,
                **derived,
            )
        )
        stats["resolved"] += 1
    else:
        outcome.base_price = base
        outcome.actual_price = actual
        outcome.resolved_at = resolved_at if resolved_at is not None else utcnow_naive()
        for key, value in derived.items():
            setattr(outcome, key, value)
        stats["reresolved"] += 1
