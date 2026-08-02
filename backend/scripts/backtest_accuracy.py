#!/usr/bin/env python3
"""
Backtesting system for ML forecast accuracy.
Computes accuracy metrics by comparing past predictions against actual outcomes.

Analysis type:
  - forecast:  ML price forecasts (7d / 30d horizons) from live DB

Usage:
    python scripts/backtest_accuracy.py                  # resolve new + score
    python scripts/backtest_accuracy.py --rescore        # score frozen only
    python scripts/backtest_accuracy.py --reresolve      # re-read the archive
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, PredictionAccuracy
from sqlalchemy import bindparam, select, text
from backtest.price_resolution import load_voted_prices, resolve_anchors
from backtest.scoring import (
    FLAT_TOLERANCE,
    HEADLINE_MIN_TIER,
    bootstrap_ci,
    direction_from_return,
    headline_records,
    price_tier,
    score_by_tier,
    score_cohort,
)

# Above this rate of mature forecasts that can't be resolved by the shared
# estimator, refuse to report a number rather than silently score a shrunken
# cohort — that silent shrinkage is how the pre-fix metric moved unnoticed.
MAX_UNRESOLVABLE_PCT = 10.0

# SQLite's bind-parameter cap is 999; every chunked statement in this file uses
# the same conservative batch size.
CHUNK = 900

# The verdict columns: derived from the frozen actuals, never observations in
# their own right. Task 8c refreshes exactly these (plus evaluated_at) whenever
# the current scoring logic disagrees with what is stored.
VERDICT_COLUMNS = (
    "direction_actual",
    "direction_correct",
    "in_interval",
    "abs_error",
    "pct_error",
)

# Float comparison tolerance for deciding whether a stored verdict still agrees
# with the derived one. The derivation is deterministic over the same stored
# doubles, so a genuine no-change run compares bit-identical; the epsilon only
# absorbs round-tripping through the DB driver.
_VERDICT_EPS = 1e-9

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backtest_accuracy")


def _upsert_accuracy(db, rows):
    """Replace existing accuracy rows for the same key then insert new ones."""
    for row in rows:
        filters = {
            "prediction_type": row["prediction_type"],
            "evaluation_date": row["evaluation_date"],
        }
        if row.get("horizon_days") is not None:
            filters["horizon_days"] = row["horizon_days"]
        else:
            filters["horizon_days"] = None
        if row.get("model_version") is not None:
            filters["model_version"] = row["model_version"]
        else:
            filters["model_version"] = None
        filters["price_tier"] = row.get("price_tier")

        existing = db.query(PredictionAccuracy).filter_by(**filters).first()
        if existing:
            existing.sample_count = row["sample_count"]
            existing.metrics = row["metrics"]
            existing.evaluation_window_days = row.get("evaluation_window_days")
            existing.created_at = row["created_at"]
        else:
            db.add(PredictionAccuracy(**row))
    db.commit()

    if rows:
        from db.parquet import append_table
        append_table("prediction_accuracy", rows,
                     ["prediction_type", "evaluation_date", "horizon_days",
                      "model_version", "price_tier"])


# ---------------------------------------------------------------------------
# 1. Forecast backtesting
# ---------------------------------------------------------------------------

def _derive_verdict(base, actual, mid, low, high, direction_predicted):
    """The single derivation of the verdict columns from the frozen actuals.

    Every path that produces a verdict — the first resolution, the re-derived
    scoring records, and the Task 8c refresh of the stored columns — goes
    through this one function, so a scoring change cannot land on some of them
    and not others. Pure: no DB, no clock, no archive.
    """
    abs_error = abs(mid - actual)
    predicted_direction = direction_predicted or "flat"
    actual_direction = direction_from_return((actual - base) / base)
    return {
        "direction_actual": actual_direction,
        "direction_correct": 1 if predicted_direction == actual_direction else 0,
        "in_interval": (
            None if (low is None or high is None)
            else (1 if low <= actual <= high else 0)
        ),
        "abs_error": abs_error,
        # Divided by the BASE leg, not the actual. Explicit human ruling.
        "pct_error": abs(abs_error / base) * 100,
    }


def _verdict_for_storage(verdict):
    """The verdict as it is written to the forecast_outcomes columns."""
    stored = dict(verdict)
    stored["abs_error"] = round(stored["abs_error"], 4)
    return stored


def _verdicts_differ(stored_row, derived):
    """True if any stored verdict column disagrees with *derived*."""
    for col in VERDICT_COLUMNS:
        old = getattr(stored_row, col)
        new = derived[col]
        if old is None or new is None:
            if old is not new:
                return True
            continue
        if isinstance(new, float) or isinstance(old, float):
            if abs(float(old) - float(new)) > _VERDICT_EPS:
                return True
        elif old != new:
            return True
    return False


# The refresh statement. The SET clause names the five verdict columns and
# evaluated_at and NOTHING else: base_price, actual_price and resolved_at are
# the frozen actuals plus the freeze timestamp, and this operation is
# structurally incapable of moving them. Only --reresolve may.
_REFRESH_VERDICTS_SQL = text("""
    UPDATE forecast_outcomes
       SET direction_actual = :direction_actual,
           direction_correct = :direction_correct,
           in_interval = :in_interval,
           abs_error = :abs_error,
           pct_error = :pct_error,
           evaluated_at = :evaluated_at
     WHERE id = :id
""")


# How many refreshed rows accumulate before they are flushed to Parquet and the
# DB. Bounds peak memory on the one-off historical refresh (~65k rows) without
# rewriting the whole Parquet file once per read page — append_table rewrites
# the file wholesale, so the flush wants to be much coarser than the read and
# UPDATE batching.
REFRESH_FLUSH = 10_000


def _iter_outcome_rows(db, forecast_ids=None, batch=None):
    """Yield pages of at most *batch* forecast_outcomes rows.

    Core selects over the mapped table, NOT ORM entities. Deliberate on both
    counts: plain Rows have no identity map to grow across a 65k-row refresh,
    and — unlike ORM instances under SessionLocal's default
    expire_on_commit=True — they are not expired by the commits this walk
    interleaves with, so nothing read from them afterwards can fire a per-row
    SELECT. Core rather than raw text() because the table's column types come
    with it: on SQLite a raw text() SELECT hands back dates and datetimes as
    plain strings, which would then be written into the Parquet mirror as
    VARCHAR against the existing TIMESTAMP columns.

    With no id restriction the table is walked by keyset pagination on the
    primary key rather than materialized whole — --rescore covers every row
    ever resolved. The refresh never writes `id`, so the keyset is stable
    across the commits that happen mid-walk.
    """
    from database import ForecastOutcome

    tbl = ForecastOutcome.__table__
    batch = batch or CHUNK

    if forecast_ids is None:
        last_id = -1
        while True:
            rows = db.execute(
                select(tbl).where(tbl.c.id > last_id)
                .order_by(tbl.c.id).limit(batch)
            ).fetchall()
            if not rows:
                return
            last_id = rows[-1].id
            yield rows
            if len(rows) < batch:
                return
    else:
        ids = list(forecast_ids)
        for i in range(0, len(ids), batch):
            rows = db.execute(
                select(tbl).where(tbl.c.forecast_id.in_(ids[i:i + batch]))
            ).fetchall()
            if rows:
                yield rows


def _flush_verdict_refresh(db, updates, mirror):
    """Write one batch of refreshed verdicts to the Parquet mirror, then the DB.

    ORDER IS LOAD-BEARING: Parquet first, DB commit second.

    Idempotence keys off a DB-vs-derived diff. Under the reverse order, an
    append_table that raised — or a process killed between the commit and the
    write — would leave the DB refreshed and the served Parquet copy stale, and
    the *next* run would compute zero differences, issue no write, and never
    re-converge. That is the same "served copy disagrees with the headline"
    failure this function exists to close, reached through a crash instead of a
    scoring change, and it would be silent and permanent.

    Written first, the worst case is a mirror that is briefly AHEAD of the DB:
    the DB still holds the old verdict, so the next run recomputes the same
    difference and rewrites both. Being ahead is benign in content, too — the
    mirror would hold exactly the verdict the headline already reports, since
    the reported metric is derived from the frozen actuals and never read from
    these columns.

    append_table dedups on forecast_id, so the refreshed row REPLACES the stale
    one and the whole row must be supplied; re-writing the same row on a retry
    is a replace, not a duplicate.
    """
    from db.parquet import append_table
    append_table("forecast_outcomes", mirror, ["forecast_id"])

    for i in range(0, len(updates), CHUNK):
        db.execute(_REFRESH_VERDICTS_SQL, updates[i:i + CHUNK])
    db.commit()


def _refresh_verdict_columns(db, forecast_ids=None) -> int:
    """Bring the stored verdict columns back in line with current scoring.

    The actuals (base_price, actual_price) are frozen observations; the verdict
    columns (direction_actual, direction_correct, in_interval, abs_error,
    pct_error) are *metrics* derived from them. Task 8b made the reported metric
    re-derive them every run; this makes the stored copies follow, so the two
    remaining consumers that read them as authoritative —
    models/forecaster.py::update_bias_corrections_from_outcomes, which fits
    production predict() thresholds, and scripts/tiered_breakdown.py — cannot
    silently disagree with the headline after a scoring change.

    Only rows whose stored verdict actually differs are written, so a run with
    no scoring change issues no UPDATE at all and the operation is idempotent.
    Rows that cannot be derived (no usable base/actual/mid) are left alone;
    _records_from_frozen_outcomes already logs them loudly.

    evaluated_at is bumped on refreshed rows only. It means "when this row's
    verdict was last computed", which is exactly what a refresh changes, and
    leaving untouched rows alone keeps tiered_breakdown.py's
    `evaluated_at >= NOW() - INTERVAL '2 days'` window meaning what it meant
    before — a blanket bump would drag the whole 65k-row history into it.

    Work is streamed in batches: rows are read by keyset pages, and the
    refreshed ones accumulate only until REFRESH_FLUSH before being written and
    committed. Nothing proportional to the whole table is held at once, and the
    mirror row for each refreshed row is snapshotted into a plain dict at the
    moment it is diffed — before any commit — so the flush cannot come back to
    re-read it.

    Returns the number of rows refreshed.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    updates, mirror = [], []
    total = 0

    for batch in _iter_outcome_rows(db, forecast_ids):
        for r in batch:
            base, actual, mid = r.base_price, r.actual_price, r.predicted_price_mid
            if base is None or base <= 0 or actual is None or actual <= 0 or mid is None:
                continue
            derived = _verdict_for_storage(_derive_verdict(
                base, actual, mid,
                r.predicted_price_low, r.predicted_price_high, r.direction_predicted,
            ))
            if not _verdicts_differ(r, derived):
                continue

            updates.append(dict(derived, id=r.id, evaluated_at=now))
            # The mirror row is built HERE, from the row already in hand. The
            # frozen columns are carried through verbatim — this function does
            # not touch them — and building it now rather than after the commit
            # is what keeps the write side free of a per-row re-read.
            mirror_row = _outcome_to_mapping(r)
            mirror_row.update(derived)
            mirror_row["evaluated_at"] = now
            mirror_row["resolved_at"] = r.resolved_at
            mirror.append(mirror_row)

        # Flush only on a batch boundary, never mid-batch.
        if len(updates) >= REFRESH_FLUSH:
            _flush_verdict_refresh(db, updates, mirror)
            total += len(updates)
            updates, mirror = [], []

    if updates:
        _flush_verdict_refresh(db, updates, mirror)
        total += len(updates)

    if not total:
        # The normal case. Quiet on purpose — a daily run says nothing here.
        logger.debug("  Verdict columns already match current scoring")
        return 0

    logger.info(
        f"  Refreshed verdict columns on {total:,} frozen outcome(s) to "
        f"match current scoring (the frozen actuals were NOT touched). A "
        f"non-zero count here means the scoring logic moved since those rows "
        f"were last evaluated — expected right after a scoring change, and a "
        f"bug otherwise."
    )
    return total


def _store_forecast_outcomes(db, outcomes, reresolve: bool = False) -> int:
    """Persist per-forecast outcomes. Insert-only unless *reresolve*.

    Resolved actuals are frozen: a forecast_id that already has a row keeps
    its base_price and actual_price forever. Re-running the backtest after the
    archive gains source rows for an old target date must not rewrite history
    — that silent rewriting is why the same 5,512 forecasts scored 61.76% on
    07-18 and 33.74% on 07-19. Metrics stay derived and are recomputed every
    run, so a *scoring* fix still lands without --reresolve.
    """
    if not outcomes:
        return 0
    from database import ForecastOutcome

    all_fids = [o["forecast_id"] for o in outcomes]
    existing_ids = set()
    for i in range(0, len(all_fids), 900):
        batch = all_fids[i:i + 900]
        rows = db.query(ForecastOutcome.forecast_id).filter(
            ForecastOutcome.forecast_id.in_(batch)
        ).all()
        existing_ids.update(r[0] for r in rows)

    if reresolve:
        stale = list(existing_ids)
        for i in range(0, len(stale), 900):
            batch = stale[i:i + 900]
            db.query(ForecastOutcome).filter(
                ForecastOutcome.forecast_id.in_(batch)
            ).delete(synchronize_session=False)
        to_write = outcomes
    else:
        to_write = [o for o in outcomes if o["forecast_id"] not in existing_ids]

    if not to_write:
        db.commit()
        logger.info(f"  All {len(outcomes):,} outcomes already resolved (frozen)")
        return 0

    resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
    for o in to_write:
        o["evaluated_at"] = resolved_at
        o["resolved_at"] = resolved_at

    db.bulk_insert_mappings(ForecastOutcome, to_write)
    db.commit()

    from db.parquet import append_table
    append_table("forecast_outcomes", to_write, ["forecast_id"])

    logger.info(
        f"  Resolved {len(to_write):,} new outcomes "
        f"({len(outcomes) - len(to_write):,} already frozen)"
    )
    return len(to_write)


def _records_from_frozen_outcomes(db, min_price=0, forecast_ids=None):
    """Rebuild scoring records from stored outcomes, without the archive.

    Every metric input is *derived* here from the frozen base_price /
    actual_price plus the frozen prediction legs. The stored direction_actual,
    direction_correct and in_interval columns are deliberately NOT read back —
    freezing the actuals rather than the metrics is what lets a later scoring
    fix (a different flat tolerance, say) land on historical rows with no
    archive access at all. Those stored columns are kept in step by
    _refresh_verdict_columns rather than being authoritative here.

    This is the single derivation used by every scoring path: the daily run,
    --rescore and --reresolve all score through it, so they cannot drift apart.

    `confidence` is not on ForecastOutcome, so it is joined back from
    item_forecasts.

    forecast_ids: optional restriction to a set of forecast ids, applied as a
        chunked SQL IN list (900 per batch, under the SQLite bind-parameter cap).

    Rows that cannot be scored — a missing or non-positive base_price or
    actual_price, or a missing predicted_price_mid — are dropped, but never
    silently: they are counted and logged. A frozen row is no longer re-resolved
    from the archive each day, so an unusable one would otherwise vanish from
    both the metric and the unresolvable gate permanently, which is precisely
    the silent cohort shrinkage this plan exists to eliminate. Deliberately a
    log and not a raise — legacy rows predate the freeze and must not block the
    backfill.
    """
    select_sql = """
        SELECT o.forecast_id, o.item_id, o.horizon_days, o.model_version,
               o.base_price, o.actual_price, o.predicted_price_low,
               o.predicted_price_mid, o.predicted_price_high,
               o.direction_predicted, f.confidence
        FROM forecast_outcomes o
        LEFT JOIN item_forecasts f ON f.id = o.forecast_id
    """
    if forecast_ids is None:
        rows = db.execute(text(select_sql)).fetchall()
    else:
        stmt = text(select_sql + " WHERE o.forecast_id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        ids = list(forecast_ids)
        rows = []
        for i in range(0, len(ids), 900):
            rows.extend(db.execute(stmt, {"ids": ids[i:i + 900]}).fetchall())

    n_unusable = 0
    n_below_min_price = 0

    groups = defaultdict(list)
    for r in rows:
        base, actual, mid = r.base_price, r.actual_price, r.predicted_price_mid
        if base is None or base <= 0 or actual is None or actual <= 0 or mid is None:
            n_unusable += 1
            continue
        if min_price > 0 and base < min_price:
            n_below_min_price += 1
            continue

        verdict = _derive_verdict(
            base, actual, mid,
            r.predicted_price_low, r.predicted_price_high, r.direction_predicted,
        )

        groups[(r.horizon_days, r.model_version or "unknown")].append({
            "abs_error": verdict["abs_error"],
            "pct_error": verdict["pct_error"],
            "sq_error": (mid - actual) ** 2,
            "direction_correct": verdict["direction_correct"],
            "predicted_direction": r.direction_predicted or "flat",
            "actual_direction": verdict["direction_actual"],
            "in_interval": verdict["in_interval"],
            "confidence": r.confidence or "low",
            "base_price": base,
            "actual_price": actual,
            "price_tier": price_tier(base),
            "item_id": r.item_id,
        })

    n_scored = sum(len(v) for v in groups.values())
    logger.info(
        f"  Frozen outcomes: {n_scored:,} scored of {len(rows):,} considered"
    )
    if n_below_min_price:
        logger.info(
            f"  {n_below_min_price:,} frozen outcome(s) below --min-price ${min_price:.2f}"
        )
    if n_unusable:
        # Loud on purpose. These rows are re-resolved by nothing and counted by
        # no gate — without this line a shrinking scored cohort is invisible.
        logger.warning(
            f"  {n_unusable:,} frozen outcome(s) UNUSABLE and dropped from scoring "
            f"(base_price/actual_price missing or non-positive, or no "
            f"predicted_price_mid). They are frozen, so they are never "
            f"re-resolved and never counted by the unresolvable gate. "
            f"Re-resolve them with --reresolve to bring them back into the metric."
        )
    return groups


def _score_groups(groups, today):
    """Turn {(horizon, model_version): records} into prediction_accuracy rows."""
    results = []
    for (horizon, model_version), records in sorted(groups.items()):
        tiered = score_by_tier(records)
        if not tiered:
            logger.info(f"  [{horizon}d / {model_version}] No valid comparisons")
            continue

        for tier, metrics, n in tiered:
            results.append({
                "prediction_type": "forecast",
                "evaluation_date": today,
                "horizon_days": horizon,
                "model_version": model_version,
                "price_tier": tier,
                "evaluation_window_days": None,
                "sample_count": n,
                "metrics": metrics,
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            })

        head_metrics, head_n = score_cohort(headline_records(records))
        penny_metrics, penny_n = score_cohort(
            [r for r in records if r["price_tier"] < HEADLINE_MIN_TIER]
        )

        if head_n:
            head_ci_lower = head_metrics["directional_accuracy_ci_lower"]
            head_ci_upper = head_metrics["directional_accuracy_ci_upper"]
            ci_str = ""
            if head_ci_lower is not None:
                ci_str = f" [CI: {head_ci_lower * 100:.1f}–{head_ci_upper * 100:.1f}]"
            logger.info(
                f"  [{horizon}d / {model_version}] >=$1: {head_n:,} samples — "
                f"MAE=${head_metrics['mae']:.2f} MAPE={head_metrics['mape']:.1f}% "
                f"DirAcc={head_metrics['directional_accuracy']:.1f}%{ci_str} "
                f"IntCov={head_metrics['interval_coverage']:.1f}% "
                f"ConfGap={head_metrics['conf_gap_pp']:.1f}pp "
                f"Skill={head_metrics['skill_vs_baseline']}"
            )
        if penny_n:
            logger.info(
                f"  [{horizon}d / {model_version}] <$1: {penny_n:,} samples — "
                f"DirAcc={penny_metrics['directional_accuracy']:.1f}% (tick-dominated)"
            )
    return results


def _outcome_to_mapping(row):
    """An already-frozen ForecastOutcome ORM row as an outcome dict.

    The whole mature cohort — frozen rows included — is handed to
    _store_forecast_outcomes each run so that the freeze is enforced in exactly
    one place and its "already frozen" count is the truth. Only the rows the
    freeze lets through are written.
    """
    return {
        "forecast_id": row.forecast_id,
        "item_id": row.item_id,
        "forecast_date": row.forecast_date,
        "horizon_days": row.horizon_days,
        "target_date": row.target_date,
        "current_price": row.current_price,
        "base_price": row.base_price,
        "predicted_price_low": row.predicted_price_low,
        "predicted_price_mid": row.predicted_price_mid,
        "predicted_price_high": row.predicted_price_high,
        "actual_price": row.actual_price,
        "direction_predicted": row.direction_predicted,
        "direction_actual": row.direction_actual,
        "direction_correct": row.direction_correct,
        "in_interval": row.in_interval,
        "abs_error": row.abs_error,
        "pct_error": row.pct_error,
        "model_version": row.model_version,
    }


def backtest_forecasts(db, today=None, min_price=0, reresolve=False, rescore=False):
    """Compare mature ML forecasts against actual prices.

    Stores both aggregate accuracy metrics (prediction_accuracy) and
    per-forecast outcome records (forecast_outcomes).

    Metrics breakdown:
      - Point error:      MAE, RMSE, MAPE, wMAPE (dollar-weighted), MAPE by price tier
      - Direction:        directional_accuracy, baseline comp, bootstrap CI
      - Probabilistic:    interval_coverage, conf_gap_pp, conf_calibration_error
      - Skill:            skill_vs_baseline (Theil's U analog, <1 beats persistence)

    Parameters:
        min_price: Minimum current_price to include (filter out cheap items).
        reresolve: Overwrite already-frozen actuals with freshly resolved ones.
        rescore: Recompute metrics from stored (frozen) outcomes only — does
            not touch the archive at all.
    """
    today = today or date.today()
    logger.info("=" * 60)
    logger.info("BACKTEST: ML Forecasts")
    logger.info("=" * 60)

    if rescore:
        logger.info("  --rescore: scoring from frozen outcomes, archive not read")
        _refresh_verdict_columns(db)
        groups = _records_from_frozen_outcomes(db, min_price=min_price)
        results = _score_groups(groups, today)
        if results:
            _upsert_accuracy(db, results)
            logger.info(f"  Stored {len(results)} forecast accuracy records")
        return results

    # Fetch all forecasts with a midpoint price, filter for maturity in Python
    rows = db.execute(text("""
        SELECT f.id, f.item_id, f.forecast_date, f.horizon_days,
               f.price_low, f.price_mid, f.price_high,
               f.current_price, f.direction, f.confidence, f.model_version
        FROM item_forecasts f
        WHERE f.price_mid IS NOT NULL
    """)).fetchall()

    # Filter for mature forecasts (forecast_date + horizon <= today)
    mature = []
    for r in rows:
        forecast_date = r.forecast_date if isinstance(r.forecast_date, date) else date.fromisoformat(r.forecast_date)
        maturity_date = forecast_date + timedelta(days=r.horizon_days)
        if maturity_date <= today:
            mature.append(r)

    if not mature:
        logger.info("  No mature forecasts to evaluate.")
        return []

    logger.info(f"  Found {len(mature)} mature forecasts to evaluate")

    from database import ForecastOutcome

    mature_ids = [r.id for r in mature]

    # Forecasts that already have a frozen outcome are NOT re-resolved. Their
    # base_price/actual_price are final and the metric is derived from them, so
    # the archive is only read for what is genuinely new. --reresolve is the
    # deliberate exception: it re-reads the archive for the whole cohort and
    # overwrites, which is what Task 9's backfill needs.
    frozen_rows = []
    if not reresolve:
        for i in range(0, len(mature_ids), 900):
            batch = mature_ids[i:i + 900]
            frozen_rows.extend(
                db.query(ForecastOutcome)
                .filter(ForecastOutcome.forecast_id.in_(batch))
                .all()
            )
    frozen_ids = {o.forecast_id for o in frozen_rows}
    to_resolve = [r for r in mature if r.id not in frozen_ids]
    logger.info(
        f"  {len(frozen_ids):,} already frozen, {len(to_resolve):,} to resolve"
    )

    # Group by horizon + model_version. Only the unfrozen forecasts are grouped:
    # a group with nothing new performs no archive read at all.
    groups = defaultdict(list)
    for r in to_resolve:
        key = (r.horizon_days, r.model_version or "unknown")
        groups[key].append(r)

    archive_dir = Path(__file__).parent.parent.parent / "price-archive"

    id_to_slug = {}
    if to_resolve:
        slug_rows = db.execute(text("SELECT id, item_id FROM items")).fetchall()
        id_to_slug = {r.id: r.item_id for r in slug_rows}

    new_outcomes = []
    # The gate measures resolution quality, so its denominator is the forecasts
    # that actually required resolution this run. A frozen forecast was not
    # resolved and belongs on neither side of the ratio; counting the frozen
    # majority would dilute a genuinely broken resolution rate to nothing.
    n_unresolvable = 0
    n_considered = 0

    for (horizon, model_version), forecasts in sorted(groups.items()):
        anchors = set()
        slugs = set()
        for f in forecasts:
            slug = id_to_slug.get(f.item_id)
            if slug is None:
                continue
            f_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            slugs.add(slug)
            anchors.add((slug, f_date))
            anchors.add((slug, f_date + timedelta(days=horizon)))

        if not anchors:
            logger.info(f"  [{horizon}d / {model_version}] No slug mappings")
            n_considered += len(forecasts)
            n_unresolvable += len(forecasts)
            continue

        anchor_dates = [a[1] for a in anchors]
        voted = load_voted_prices(archive_dir, sorted(slugs), min(anchor_dates), max(anchor_dates))
        prices = resolve_anchors(voted, anchors)
        logger.info(f"  Resolved {len(prices):,} of {len(anchors):,} anchors")

        for f in forecasts:
            n_considered += 1
            slug = id_to_slug.get(f.item_id)
            f_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            target_date = f_date + timedelta(days=horizon)

            base_res = prices.get((slug, f_date))
            actual_res = prices.get((slug, target_date))
            if base_res is None or actual_res is None:
                n_unresolvable += 1
                continue

            base, actual = base_res.price, actual_res.price
            if base <= 0 or actual <= 0:
                n_unresolvable += 1
                continue

            # EVERY observation backing the actual leg must post-date the
            # forecast. When the archive ends before the target date, both legs
            # can resolve from the SAME window — the anchor-staleness rule passes
            # each anchor individually — and actual_ret comes out exactly 0.0.
            # That is not a flat market, it is a manufactured flat, produced for
            # every forecast whose target date is beyond coverage. In the Task 9a
            # dry run it moved the actual-flat share from 31.8% to 48.1%.
            #
            # Testing the NEWEST observation is not enough: a single post-forecast
            # observation admits the pair while the actual leg's median is still
            # decided by pre-forecast observations, because two 3-observation
            # windows can overlap in 2 of 3 slots. With observations on
            # 07-01/02/03 at 1.0 and 07-06 at 1.5, a forecast dated 07-05
            # targeting 07-08 has base median(07-01,02,03) = 1.0 and actual
            # median(07-02,03,06) = 1.0 — a 50% move on the only post-forecast
            # observation scoring as exactly 0.0%. Requiring the OLDEST supporting
            # observation to post-date the forecast makes the two windows
            # disjoint, so the actual leg carries no information the base leg
            # already had. It costs nothing at any production horizon:
            # ItemForecaster.HORIZONS is [3, 7, 14, 30] and SMOOTH_WINDOW is 3, so
            # a fully-covered actual window is {f+h-2 ... f+h}, entirely after f
            # whenever h >= 3.
            #
            # This lives here, not in resolve_anchors: only the caller knows two
            # anchors form a leg pair. resolve_anchors stays symmetric and
            # leg-agnostic, and the forecast is DROPPED (and counted
            # unresolvable), never given a fallback.
            if actual_res.oldest_observation <= f_date:
                n_unresolvable += 1
                continue

            mid, low, high = f.price_mid, f.price_low, f.price_high
            if min_price > 0 and base < min_price:
                continue

            # Same derivation the refresh and the scoring records use, so a
            # freshly written row is by construction already "current" and the
            # refresh below finds nothing to do for it.
            verdict = _verdict_for_storage(
                _derive_verdict(base, actual, mid, low, high, f.direction)
            )

            new_outcomes.append({
                "forecast_id": f.id,
                "item_id": f.item_id,
                "forecast_date": f_date,
                "horizon_days": horizon,
                "target_date": target_date,
                # current_price is retained for reference only; nothing reads
                # it for scoring, and it is never synthesized. Written through
                # as-is (nullable) so it stays distinguishable from base_price,
                # which is always archive-resolved. Downstream consumers
                # (update_bias_corrections_from_outcomes, retro_bias_check,
                # tiered_breakdown) still read this column and must see the
                # real serving-time value or a genuine NULL, not a stand-in.
                "current_price": f.current_price,
                "base_price": base,
                "predicted_price_low": low,
                "predicted_price_mid": mid,
                "predicted_price_high": high,
                "actual_price": actual,
                "direction_predicted": f.direction or "flat",
                "direction_actual": verdict["direction_actual"],
                "direction_correct": verdict["direction_correct"],
                "in_interval": verdict["in_interval"],
                "abs_error": verdict["abs_error"],
                "pct_error": verdict["pct_error"],
                "model_version": model_version,
            })

    # A run with nothing new to resolve has an empty denominator; there is no
    # resolution rate to gate on, and dividing here would raise.
    if n_considered:
        unresolvable_pct = n_unresolvable / n_considered * 100
        logger.info(
            f"  Unresolvable: {n_unresolvable:,}/{n_considered:,} "
            f"({unresolvable_pct:.1f}% of forecasts requiring resolution)"
        )
        if unresolvable_pct > MAX_UNRESOLVABLE_PCT:
            raise RuntimeError(
                f"{unresolvable_pct:.1f}% of mature forecasts could not be resolved "
                f"(cap {MAX_UNRESOLVABLE_PCT}%). Silent cohort shrinkage is how this "
                f"metric moved unnoticed before — refusing to report a number."
            )

    # The whole mature cohort goes through the freeze, which writes only what is
    # new. Passing the frozen rows too keeps the freeze the single gate on the
    # table, and the table is what gets scored below.
    all_outcomes = [_outcome_to_mapping(o) for o in frozen_rows] + new_outcomes
    if all_outcomes:
        _store_forecast_outcomes(db, all_outcomes, reresolve=reresolve)

    # The actuals are frozen; the verdict columns are not, because they are
    # metrics. Bring any that the current scoring logic disagrees with back in
    # line before scoring. Under --reresolve the rows were just rewritten from
    # this same derivation, so this is a no-op there rather than a double write.
    _refresh_verdict_columns(db, forecast_ids=mature_ids)

    # Score the persisted cohort, not the freshly resolved values. Metrics are
    # re-derived every run from the frozen base/actual, so a scoring fix still
    # lands without --reresolve — but an archive revision cannot move a number
    # that has already been reported. That separation is the point of the plan:
    # before it, the same 5,512-forecast cohort scored 61.76%, 33.74%, 61.54%
    # and 57.91% on four consecutive evaluation dates.
    groups = _records_from_frozen_outcomes(
        db, min_price=min_price, forecast_ids=mature_ids
    )
    results = _score_groups(groups, today)

    if results:
        _upsert_accuracy(db, results)
        logger.info(f"  Stored {len(results)} forecast accuracy records")

    return results



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(types=None, min_price=0, update_bias=False, reresolve=False, rescore=False):
    db = SessionLocal()
    today = date.today()
    allowed = ["forecast"]
    types = types or allowed

    try:
        results = {}
        for t in types:
            if t == "forecast":
                results["forecast"] = backtest_forecasts(
                    db, today, min_price=min_price, reresolve=reresolve, rescore=rescore
                )
            else:
                logger.warning(f"Unknown backtest type: {t}")

        total = sum(len(v or []) for v in results.values())
        logger.info(f"\nBacktest complete: {total} accuracy records stored")

        if update_bias:
            logger.info("=" * 60)
            logger.info("Updating per-tier bias corrections from outcomes...")
            logger.info("=" * 60)
            try:
                from models.forecaster import ItemForecaster
                forecaster = ItemForecaster(db_session=db)
                forecaster.load_models()
                forecaster.update_bias_corrections_from_outcomes()
                logger.info("Bias corrections updated successfully")
            except Exception as e:
                logger.error(f"Bias update failed: {e}", exc_info=True)

        return {"status": "success", "total_records": total, "types": list(results.keys())}

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "message": str(e)}

    finally:
        db.close()


def main():
    args = list(sys.argv[1:])
    types = None
    min_price = 0.0
    update_bias = True
    reresolve = "--reresolve" in args
    rescore = "--rescore" in args
    i = 0
    while i < len(args):
        if args[i] == "--type" and i + 1 < len(args):
            types = [args[i + 1]]
            i += 2
        elif args[i] == "--min-price" and i + 1 < len(args):
            min_price = float(args[i + 1])
            i += 2
        elif args[i] == "--update-bias":
            update_bias = True
            i += 1
        else:
            i += 1
    if min_price:
        logger.info(f"Filtering items with current_price >= ${min_price:.2f}")
    if update_bias:
        logger.info("Bias correction update enabled")
    result = run_backtest(
        types, min_price=min_price, update_bias=update_bias,
        reresolve=reresolve, rescore=rescore,
    )
    print(f"RESULT: {json.dumps(result, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
