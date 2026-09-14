#!/usr/bin/env python3
"""
Event correlation analysis for CS2 market events.

Computes historical price impacts around market events (operations, cases,
updates) and writes results to event_impacts, event_patterns, and
event_correlations tables.

PRICES COME FROM THE PARQUET ARCHIVE, NOT POSTGRES. Until 2026-08-05 every
price window here was a `price_history` query, and that table is empty for
exactly the items this script analyses: migration 0008 deleted every
`is_backfilled = 1` item's rows, and `run_analysis` selects precisely
`is_backfilled == 1`; `collectors/pipeline.py` then stopped writing the table
at all at the 2026-07-11 CSV->Parquet cutover. Prod holds 16,487 rows spanning
05-27 -> 07-11 and nothing in any recent event window, while
`price-archive/prices-2026-05.parquet` has ~25k item-days for every day of it.
So `_compute_impacts` returned `[]` for every event, `run_analysis` logged "No
price data found ... skipping", and the task wrote ZERO rows on every run from
2026-07-19 onward while exiting green. It only turned red on 2026-08-02, when
324cfff added the impact counts to `run_task.py`'s `ROW_COUNT_FIELDS`.

The AGENTS.md invariant: "Training data comes from Parquet, not the DB. The DB
supplies only the `is_backfilled` flag and events metadata."

Usage:
    python scripts/event_correlation_analysis.py
    python scripts/event_correlation_analysis.py --days-back 90
"""

import argparse
import bisect
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.price_resolution import archive_max_day, load_voted_prices
from database import (
    Event,
    EventCorrelation,
    EventImpact,
    EventPattern,
    Item,
    SessionLocal,
)
from db.parquet import append_table
from sqlalchemy import desc, func

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("event_correlation")

# Repo-root `price-archive/`, the same resolution `scripts/evaluate_forecaster.py`
# and `db/parquet.py::ARCHIVE_ROOT` use. CI has to check the archive repo out to
# this path (see .github/workflows/event-correlation-analysis.yml) or the run
# dies on FileNotFoundError — which is the intended outcome, not a fallback.
ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "price-archive"

# WINDOW SEMANTICS, NORMALISED TO WHOLE DAYS. The old SQL compared
# `PriceHistory.timestamp` against `event.timestamp`, which carries a time of
# day, so the effective window silently depended on an event's clock time: a
# `>= event - 8 days` bound over midnight-stamped rows admits 8 days for an
# event at 00:00 and 7 for one at 13:45. The archive is one row per DAY, so
# every bound below is a date and the windows are exactly what the docstrings
# advertise. This is a deliberate documented change: the pre-event window is now
# always 7 days, where before it was 7 or 8 depending on the event's timestamp.
PRE_EVENT_WINDOW_DAYS = 7  # [event_day - 7, event_day - 1] inclusive
CENTRED_WINDOW_DAYS = 3  # [target - 1, target + 1] inclusive
IMPACT_OFFSETS = (1, 3, 7)  # days after the event that get an impact column
CONTROL_OFFSET_DAYS = 7  # the offset the z-score's control group uses

# `run_analysis`'s status when the lookback window holds no events at all.
# `data/cs2_events.json`'s newest event is 2026-05-10, so at days_back=90 the
# window empties around 2026-08-08 — a calendar fact, not a fault. It gets its
# own status so `run_task.py`'s zero-row guard can let it through loudly instead
# of reporting a failure nobody can fix, while "events exist but zero impacts
# were written" stays fatal, because that is the bug documented above.
NO_EVENTS_STATUS = "no_events_in_window"


class ItemRef(NamedTuple):
    """The two item identities this script has to keep apart.

    `Item.id` (the int PK, used by `event_impacts.item_id`) is what the DB and
    the impact rows key on; `Item.item_id` is the VARCHAR slug the archive keys
    on. Mixing them up reads as "no price data". `type` rides along because the
    control group needs it and querying it per item inside the impact loop was
    one DB round trip per item.
    """

    slug: str
    type: str


@dataclass(frozen=True)
class _DailySeries:
    """One item's prices, one entry per observed day, ascending by day."""

    days: list[date]
    prices: list[float]


class PriceStore:
    """Voted daily prices for the analysis universe, keyed by DB item id.

    Loaded once per run and queried by window, because the analysis asks for
    O(items x events x windows) means and each one used to be a SQL round trip.

    The prices are whatever `backtest/price_resolution.py::load_voted_prices`
    returns — i.e. after `ItemForecaster._apply_multi_source_voting`, so the
    archive's duplicate item-days are collapsed the way production collapses
    them, not with a plain mean. Reusing that loader rather than writing raw
    DuckDB here is also what keeps the "missing archive raises" behaviour: an
    absent checkout must fail loudly, never read as "no price data".
    """

    def __init__(self, series: dict[int, _DailySeries]):
        self._series = series

    @property
    def items_with_prices(self) -> int:
        return len(self._series)

    @classmethod
    def from_voted(cls, voted, universe: dict[int, ItemRef]) -> "PriceStore":
        """Index a voted frame (`item_id` = slug, `date`, `price`) by DB id."""
        series: dict[int, _DailySeries] = {}
        if voted is None or len(voted) == 0:
            return cls(series)

        db_id_by_slug = {ref.slug: db_id for db_id, ref in universe.items()}
        for slug, group in voted.groupby("item_id", sort=False):
            db_id = db_id_by_slug.get(slug)
            if db_id is None:
                continue
            ordered = group.sort_values("date")
            series[db_id] = _DailySeries(
                # pd.Timestamp is a datetime subclass, so this normalises both
                # what DuckDB hands back and what a test fixture passes in.
                days=[d.date() if isinstance(d, datetime) else d for d in ordered["date"]],
                prices=[float(p) for p in ordered["price"]],
            )
        return cls(series)

    @classmethod
    def load(cls, archive_dir: Path, universe: dict[int, ItemRef], min_date: date, max_date: date) -> "PriceStore":
        voted = load_voted_prices(
            Path(archive_dir),
            [ref.slug for ref in universe.values()],
            min_date,
            max_date,
        )
        return cls.from_voted(voted, universe)

    def window_mean(self, item_id: int, start: date, end: date) -> float | None:
        """Mean price over the INCLUSIVE day window [start, end], or None.

        None — never 0.0 — when the window holds no observation: a missing
        window must not read as a free item. Bisection rather than a pandas
        `.loc` slice because this is called O(items x events x windows) times
        and the per-call overhead dominates at 5.5k items.
        """
        series = self._series.get(item_id)
        if series is None or end < start:
            return None
        lo = bisect.bisect_left(series.days, start)
        hi = bisect.bisect_right(series.days, end)
        if hi <= lo:
            return None
        return sum(series.prices[lo:hi]) / (hi - lo)


@dataclass
class _ControlChanges:
    """The per-item pct-change distribution for one (event, type, offset).

    `_control_group_prices` used to be called once PER ITEM with
    `exclude_item_ids={item_id}`, which was O(items^2) window means once the
    prices are in memory. The distribution is computed ONCE and each item's
    leave-one-out mean/std comes from n, sum(x) and sum(x^2) in closed form.
    That is exactly equivalent to recomputing without the item: the old SQL
    excluded it before aggregating, and an item lacking either leg never
    entered the set at all — so for such an item there is nothing to subtract.
    """

    changes: dict[int, float] = field(default_factory=dict)
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, item_id: int, change: float) -> None:
        self.changes[item_id] = change
        self.total += change
        self.total_sq += change * change

    def excluding(self, item_id: int) -> tuple[float, float]:
        """(mean, std) of the distribution with *item_id* removed.

        Edge cases are the pre-Parquet ones, preserved exactly: POPULATION
        variance (denominator n, not n-1), a std floor of 0.001 when the
        variance is not > 0 (so the z-score divides by something), and
        (0.0, 0.0) when nothing is left — which the caller reads as "no control
        group", yielding z = 0.0.
        """
        n = len(self.changes)
        total, total_sq = self.total, self.total_sq

        own = self.changes.get(item_id)
        if own is not None:
            n -= 1
            total -= own
            total_sq -= own * own

        if n <= 0:
            return 0.0, 0.0

        mean = total / n
        # Clamped at 0 before the sqrt: sum(x^2)/n - mean^2 is algebraically
        # non-negative but can land at -1e-17 in floating point when every
        # change is identical, and math.sqrt would raise on it.
        variance = max(total_sq / n - mean * mean, 0.0)
        std = math.sqrt(variance) if variance > 0 else 0.001
        return mean, std


def _event_day(event: Event) -> date:
    """The event's calendar day. See the window-semantics note above: the time
    of day is dropped rather than being allowed to shift every window."""
    ts = event.timestamp
    return ts.date() if isinstance(ts, datetime) else ts


def _price_on_date(
    store: PriceStore, item_id: int, target: date, window_days: int = CENTRED_WINDOW_DAYS
) -> float | None:
    """Average price around a target date (centered window)."""
    half = window_days // 2
    return store.window_mean(item_id, target - timedelta(days=half), target + timedelta(days=half))


def _pre_event_price(store: PriceStore, item_id: int, event_day: date) -> float | None:
    """Average price in the 7 days before the event: [day - 7, day - 1]."""
    return store.window_mean(
        item_id,
        event_day - timedelta(days=PRE_EVENT_WINDOW_DAYS),
        event_day - timedelta(days=1),
    )


def _post_event_price(store: PriceStore, item_id: int, event_day: date, offset_days: int) -> float | None:
    """Average price at offset_days after the event (3-day centered window)."""
    return _price_on_date(store, item_id, event_day + timedelta(days=offset_days))


def _control_change_distribution(
    store: PriceStore, item_ids: list[int], event_day: date, offset_days: int
) -> _ControlChanges:
    """Pct change from the pre-event window to the offset window, per item.

    *item_ids* is the control cohort — every item of the same type. Items
    missing either leg are omitted, as the old two-query version omitted them
    (it inner-joined the before and target aggregates).

    Note the cohort narrowed with the move to Parquet: it is now the
    `is_backfilled` universe this script analyses, where the SQL version drew on
    any item with `price_history` rows. In prod that set is empty, so the old
    cohort was the empty set on every run.
    """
    dist = _ControlChanges()
    target = event_day + timedelta(days=offset_days)
    for item_id in item_ids:
        before = _pre_event_price(store, item_id, event_day)
        if before is None or before <= 0:
            continue
        after = _price_on_date(store, item_id, target)
        if after is None:
            continue
        dist.add(item_id, (after - before) / before * 100)
    return dist


def _compute_impacts(event: Event, item_ids: list[int], store: PriceStore, universe: dict[int, ItemRef]):
    """Compute impact metrics for all items around an event."""
    event_day = _event_day(event)
    pre_prices: dict[int, float] = {}
    for item_id in item_ids:
        p = _pre_event_price(store, item_id, event_day)
        if p is not None:
            pre_prices[item_id] = p

    if not pre_prices:
        return []

    ids_by_type: dict[str, list[int]] = defaultdict(list)
    for db_id, ref in universe.items():
        ids_by_type[ref.type or "skin"].append(db_id)

    # One distribution per item type, reused by every item of that type.
    control_cache: dict[str, _ControlChanges] = {}

    impacts: list[dict] = []
    for item_id, price_before in pre_prices.items():
        p1 = _post_event_price(store, item_id, event_day, 1)
        p3 = _post_event_price(store, item_id, event_day, 3)
        p7 = _post_event_price(store, item_id, event_day, 7)

        if p1 is None and p3 is None and p7 is None:
            continue

        impact_1d = ((p1 - price_before) / price_before * 100) if p1 else None
        impact_3d = ((p3 - price_before) / price_before * 100) if p3 else None
        impact_7d = ((p7 - price_before) / price_before * 100) if p7 else None

        impacts_vals = [v for v in [impact_1d, impact_3d, impact_7d] if v is not None]
        peak = max(impacts_vals, key=abs) if impacts_vals else 0.0
        peak_day = None
        if peak == impact_1d:
            peak_day = 1
        elif peak == impact_3d:
            peak_day = 3
        elif peak == impact_7d:
            peak_day = 7

        duration = None
        for d, imp in [(1, impact_1d), (3, impact_3d), (7, impact_7d)]:
            if imp is not None and abs(imp) > 0.5:
                duration = d

        item_type = (universe[item_id].type if item_id in universe else None) or "skin"
        if item_type not in control_cache:
            control_cache[item_type] = _control_change_distribution(
                store,
                ids_by_type.get(item_type, []),
                event_day,
                CONTROL_OFFSET_DAYS,
            )
        control_mean, control_std = control_cache[item_type].excluding(item_id)

        item_impact = impact_7d or impact_3d or impact_1d or 0.0
        z_score = (item_impact - control_mean) / control_std if control_std > 0 else 0.0

        impacts.append(
            {
                "event_id": event.id,
                "item_id": item_id,
                "price_day_before": price_before,
                "price_day_1": p1,
                "price_day_3": p3,
                "price_day_7": p7,
                "impact_pct_1day": impact_1d,
                "impact_pct_3day": impact_3d,
                "impact_pct_7day": impact_7d,
                "peak_impact_pct": peak,
                "peak_impact_day": peak_day,
                "duration_days": duration,
                "z_score": round(z_score, 4),
            }
        )

    return impacts


def _upsert_event_impacts(
    db, impacts: list[dict], event_type: str = "", event_description: str = "", event_timestamp=None
):
    """Write event_impacts rows to the DB and return the denormalised mirror.

    The mirror rows are RETURNED, not written: `confidence_score` is only known
    after `_compute_and_upsert_correlations` has run, and the previous version
    appended them with `confidence_score: None` and then tried to patch the
    value in by reading the whole Parquet file back — from
    `{... for r in [data]}`, where `data` was whatever the correlations loop
    variable happened to hold last. That wrote one item's confidence, left every
    other row NULL, and rewrote the entire table by hand outside `append_table`.
    """
    written = 0
    for row in impacts:
        existing = (
            db.query(EventImpact)
            .filter(
                EventImpact.event_id == row["event_id"],
                EventImpact.item_id == row["item_id"],
            )
            .first()
        )
        if existing:
            for key, val in row.items():
                setattr(existing, key, val)
        else:
            db.add(EventImpact(**row))
        written += 1
    db.commit()

    denorm_rows = [
        {
            "event_id": r["event_id"],
            "item_id": r["item_id"],
            "event_type": event_type,
            "event_description": event_description,
            "event_timestamp": event_timestamp,
            "price_day_before": r.get("price_day_before"),
            "price_day_1": r.get("price_day_1"),
            "price_day_3": r.get("price_day_3"),
            "price_day_7": r.get("price_day_7"),
            "impact_pct_1day": r.get("impact_pct_1day"),
            "impact_pct_3day": r.get("impact_pct_3day"),
            "impact_pct_7day": r.get("impact_pct_7day"),
            "peak_impact_pct": r.get("peak_impact_pct"),
            "peak_impact_day": r.get("peak_impact_day"),
            "duration_days": r.get("duration_days"),
            "z_score": r.get("z_score"),
            "confidence_score": None,
        }
        for r in impacts
    ]
    return written, denorm_rows


def _write_impacts_mirror(denorm_rows: list[dict], confidence_by_item: dict[int, float]) -> int:
    """Append the event_impacts_denorm mirror once, confidence already filled.

    Every value here is a scalar. Per AGENTS.md, nested values reaching
    `append_table` are the `prediction_accuracy.metrics` STRUCT failure mode —
    don't introduce one.
    """
    if not denorm_rows:
        return 0
    for row in denorm_rows:
        row["confidence_score"] = confidence_by_item.get(row["item_id"])
    append_table("event_impacts_denorm", denorm_rows, ["event_id", "item_id"])
    return len(denorm_rows)


def _compute_and_upsert_patterns(db, event_type: str, impacts: list[dict]):
    """Aggregate impacts into event_patterns for this event type."""
    by_item: dict[int, list[dict]] = defaultdict(list)
    for imp in impacts:
        by_item[imp["item_id"]].append(imp)

    for item_id, item_impacts in by_item.items():
        impacts_1d = [i["impact_pct_1day"] for i in item_impacts if i["impact_pct_1day"] is not None]
        impacts_3d = [i["impact_pct_3day"] for i in item_impacts if i["impact_pct_3day"] is not None]
        impacts_7d = [i["impact_pct_7day"] for i in item_impacts if i["impact_pct_7day"] is not None]
        z_scores = [i["z_score"] for i in item_impacts if i["z_score"] is not None]

        n = len(item_impacts)

        def _avg(vals):
            return sum(vals) / len(vals) if vals else None

        def _std(vals):
            if len(vals) < 2:
                return None
            m = sum(vals) / len(vals)
            return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

        avg_1d = _avg(impacts_1d)
        avg_3d = _avg(impacts_3d)
        avg_7d = _avg(impacts_7d)
        std_dev = _std(impacts_1d + impacts_3d + impacts_7d)

        # consistency: fraction of impacts with same sign
        all_signs: list[int] = []
        for imp in impacts_1d + impacts_3d + impacts_7d:
            if imp is not None:
                all_signs.append(1 if imp > 0 else -1)
        if all_signs:
            majority = max(set(all_signs), key=all_signs.count)
            consistency = sum(1 for s in all_signs if s == majority) / len(all_signs)
        else:
            consistency = 0.0

        # holdout accuracy: if we split impacts into train/test halves
        holdout_acc = None
        if len(z_scores) >= 4:
            mid = len(z_scores) // 2
            train_z = z_scores[:mid]
            test_z = z_scores[mid:]
            if train_z and test_z:
                train_sign = 1 if sum(train_z) / len(train_z) > 0 else -1
                correct = sum(1 for z in test_z if (z > 0) == (train_sign > 0))
                holdout_acc = correct / len(test_z)

        existing = (
            db.query(EventPattern)
            .filter(
                EventPattern.event_type == event_type,
                EventPattern.item_id == item_id,
            )
            .first()
        )
        data = {
            "sample_size": n,
            "avg_impact_1day": avg_1d,
            "avg_impact_3day": avg_3d,
            "avg_impact_7day": avg_7d,
            "std_dev": std_dev,
            "consistency_score": round(consistency, 4),
            "holdout_accuracy": round(holdout_acc, 4) if holdout_acc is not None else None,
        }
        if existing:
            for key, val in data.items():
                setattr(existing, key, val)
        else:
            db.add(EventPattern(event_type=event_type, item_id=item_id, **data))
    db.commit()
    return len(by_item)


def _compute_and_upsert_correlations(db, event: Event, db_item_ids: list[int], impacts: list[dict]):
    """Write event_correlations with statistical rigor checks.

    Returns (rows_written, {item_id: confidence_score}). The scores go into the
    denormalised Parquet mirror; see `_write_impacts_mirror` for why they are
    returned rather than patched into the file afterwards.
    """
    impact_by_item = {i["item_id"]: i for i in impacts}
    event_date = event.timestamp

    # Both of these used to be queried once PER ITEM inside the loop below. The
    # confounding count is a property of the EVENT — identical for all ~5.5k
    # items — and the patterns are one row per (event_type, item), so a single
    # query fetches every one of them. The cost never showed up because
    # `_compute_impacts` returned [] and this loop never ran; the first run that
    # actually had impacts would have issued ~11k round trips per event against
    # Supabase.
    day_start = event_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    confounding = (
        db.query(func.count(Event.id))
        .filter(
            Event.id != event.id,
            Event.timestamp >= day_start,
            Event.timestamp < day_end,
        )
        .scalar()
    ) or 0
    confounding_passed = 1 if confounding == 0 else 0

    patterns_by_item = {
        p.item_id: p for p in db.query(EventPattern).filter(EventPattern.event_type == event.type).all()
    }

    written = 0
    confidence_by_item: dict[int, float] = {}
    for item_id in db_item_ids:
        imp = impact_by_item.get(item_id)
        if imp is None:
            continue

        impact_7d = imp.get("impact_pct_7day") or imp.get("impact_pct_3day") or 0.0
        z_score = imp.get("z_score") or 0.0

        # Significance check: |z| > 2 => 95% confidence
        significance_passed = 1 if abs(z_score) >= 2.0 else 0

        # Control group diff (same as z-score based)
        control_diff = impact_7d
        control_passed = 1 if abs(control_diff) > 0.0 else 0

        # Pattern consistency from event_patterns table
        pattern = patterns_by_item.get(item_id)
        pattern_consistency = pattern.consistency_score if pattern else None
        pattern_passed = 1 if (pattern_consistency is not None and pattern_consistency >= 0.7) else 0

        # Lag analysis
        peak_day = imp.get("peak_impact_day")
        lag_passed = 1 if (peak_day is not None and peak_day <= 7) else 0

        # Holdout validation from pattern
        holdout_acc = pattern.holdout_accuracy if pattern else None
        validation_passed = 1 if (holdout_acc is not None and holdout_acc >= 0.6) else 0

        # Overall confidence score (weighted average of 6 checks)
        checks = [
            significance_passed,
            control_passed,
            pattern_passed,
            confounding_passed,
            lag_passed,
            validation_passed,
        ]
        weights = [0.25, 0.10, 0.20, 0.10, 0.15, 0.20]
        confidence = sum(c * w for c, w in zip(checks, weights))

        existing = (
            db.query(EventCorrelation)
            .filter(
                EventCorrelation.event_id == event.id,
                EventCorrelation.item_id == item_id,
            )
            .first()
        )
        data = {
            "event_id": event.id,
            "item_id": item_id,
            "price_change_pct": round(impact_7d, 4) if impact_7d else None,
            "control_group_change_pct": 0.0,
            "significance_test_zscore": round(z_score, 4),
            "significance_passed": significance_passed,
            "control_group_diff": round(control_diff, 4) if control_diff else None,
            "control_group_passed": control_passed,
            "pattern_consistency_score": pattern_consistency,
            "pattern_passed": pattern_passed,
            "confounding_events_count": confounding,
            "confounding_passed": confounding_passed,
            "lag_analysis_peak_day": peak_day,
            "lag_passed": lag_passed,
            "holdout_validation_accuracy": holdout_acc,
            "validation_passed": validation_passed,
            "confidence_score": round(confidence, 4),
        }
        if existing:
            for key, val in data.items():
                setattr(existing, key, val)
        else:
            db.add(EventCorrelation(**data))
        confidence_by_item[item_id] = data["confidence_score"]
        written += 1
    db.commit()

    return written, confidence_by_item


def run_analysis(days_back: int = 90, db=None, archive_dir: Path | None = None):
    """Main entry point: analyze events and write results to DB.

    *db* and *archive_dir* are injectable so tests can drive this against an
    in-memory SQLite session and a fixture archive. `backend/.env` points at
    PRODUCTION Supabase and the engine binds at import, so a test that lets
    this open its own session would be querying prod.
    """
    archive_dir = Path(archive_dir) if archive_dir is not None else ARCHIVE_DIR
    owns_session = db is None
    if owns_session:
        db = SessionLocal()
    try:
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days_back)

        events = db.query(Event).filter(Event.timestamp >= cutoff).order_by(desc(Event.timestamp)).all()
        logger.info(f"Found {len(events)} events in the last {days_back} days")

        if not events:
            # An empty window is a calendar outcome, not a fault, and it must
            # not look like the zero-row bug this file was rewritten to fix.
            #
            # Since 2026-08-08 the table is fed by `sync_events_from_news.py`
            # from Steam's own announcement feed, which runs immediately before
            # this step in `event-correlation-analysis.yml`. So an empty window
            # now means one of two things, and they are worth telling apart:
            # Valve genuinely posted nothing in the period, or that sync failed
            # (it is `continue-on-error`, so a Steam blip does not fail the
            # workflow -- check the step above). Before that date the table came
            # from `data/cs2_events.json`, frozen at 2026-05-10.
            logger.warning(
                "No events within %d days of %s — nothing to correlate. The "
                "events table is fed by scripts/sync_events_from_news.py; check "
                "the 'Refresh the CS2 event calendar' step, which is "
                "continue-on-error and may have failed silently. Returning %r "
                "(not a failure).",
                days_back,
                cutoff.date(),
                NO_EVENTS_STATUS,
            )
            return {
                "status": NO_EVENTS_STATUS,
                "events_analyzed": 0,
                "impacts_written": 0,
                "patterns_written": 0,
                "correlations_written": 0,
            }

        # id -> (slug, type), built once. The slug is the archive's key and the
        # id is what event_impacts stores; `type` used to cost one query per
        # item inside the impact loop.
        universe = {
            r.id: ItemRef(r.item_id, r.type or "skin")
            for r in db.query(Item.id, Item.item_id, Item.type).filter(Item.is_backfilled == 1).all()
        }
        item_ids = list(universe)
        logger.info(f"Found {len(item_ids)} backfilled items for analysis")

        if not item_ids:
            return {
                "status": "error",
                "error": "no items have is_backfilled = 1 — the archive-derived "
                "backfill gate is empty, so there is nothing to analyse",
            }

        event_days = [_event_day(e) for e in events]
        min_date = min(event_days) - timedelta(days=PRE_EVENT_WINDOW_DAYS)
        max_date = max(event_days) + timedelta(days=max(IMPACT_OFFSETS) + 1)
        store = PriceStore.load(archive_dir, universe, min_date, max_date)
        logger.info(
            "Loaded archive prices for %d/%d items over %s..%s",
            store.items_with_prices,
            len(item_ids),
            min_date,
            max_date,
        )

        if store.items_with_prices == 0:
            # The archive exists (load_voted_prices raises otherwise) but holds
            # nothing for this universe in this window. That is a collection gap
            # or a slug mismatch, not "no impacts" — and reporting it as zero
            # rows is the silent-success shape this area keeps regressing into.
            return {
                "status": "error",
                "error": (
                    f"archive at {archive_dir} priced 0 of {len(item_ids)} "
                    f"requested slugs over {min_date}..{max_date}; archive "
                    f"coverage ends {archive_max_day(archive_dir)}"
                ),
            }

        total_impacts = 0
        total_patterns = 0
        total_correlations = 0

        for event in events:
            logger.info(f"Analyzing event #{event.id}: {event.type} - {event.description[:60]}")

            impacts = _compute_impacts(event, item_ids, store, universe)
            if not impacts:
                logger.warning(
                    "  No archive prices in event #%s's windows (%s), skipping",
                    event.id,
                    _event_day(event),
                )
                continue

            n_impacts, denorm_rows = _upsert_event_impacts(
                db,
                impacts,
                event_type=event.type,
                event_description=event.description,
                event_timestamp=event.timestamp,
            )
            total_impacts += n_impacts
            logger.info(f"  Wrote {n_impacts} event_impacts")

            n_patterns = _compute_and_upsert_patterns(db, event.type, impacts)
            total_patterns += n_patterns
            logger.info(f"  Wrote {n_patterns} event_patterns")

            n_correlations, confidence_by_item = _compute_and_upsert_correlations(
                db,
                event,
                item_ids,
                impacts,
            )
            total_correlations += n_correlations
            logger.info(f"  Wrote {n_correlations} event_correlations")

            n_mirrored = _write_impacts_mirror(denorm_rows, confidence_by_item)
            logger.info(f"  Mirrored {n_mirrored} event_impacts_denorm rows")

        logger.info(f"Done: {total_impacts} impacts, {total_patterns} patterns, {total_correlations} correlations")
        return {
            "status": "success",
            "events_analyzed": len(events),
            "impacts_written": total_impacts,
            "patterns_written": total_patterns,
            "correlations_written": total_correlations,
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "error": str(e)}
    finally:
        if owns_session:
            db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event correlation analysis")
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Analyze events within this many days (default: 90)",
    )
    args = parser.parse_args()
    result = run_analysis(days_back=args.days_back)
    print(f"RESULT: {result}")
    if result.get("status") == "error":
        sys.exit(1)
