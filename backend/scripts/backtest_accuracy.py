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

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.directional_test import PT_T_HURDLE
from backtest.price_resolution import (
    MAX_WINDOW_SPAN_DAYS,
    SMOOTH_WINDOW,
    archive_covered_days,
    archive_max_day,
    load_voted_prices,
    resolve_anchors,
)
from backtest.resolution_gate import (
    MAX_UNRESOLVABLE_PCT as _MAX_UNRESOLVABLE_PCT,
)
from backtest.resolution_gate import (
    classify_archive_gap,
    classify_base_gap,
    classify_chronic,
    evaluate_gate,
)
from backtest.scoring import (
    FLOOR_SWEEP,
    HEADLINE_MIN_TIER,
    HEADLINE_TIER,
    MIN_HEADLINE_DATES,
    direction_from_return,
    excluded_forecast_date,
    price_tier,
    score_by_tier,
    score_cohort,
    served_identity,
)
from database import PredictionAccuracy, SessionLocal
from models.staleness import stale_run_lookup
from sqlalchemy import bindparam, select, text

# Re-exported so the gate's threshold has one definition. The gate itself —
# and the reason it needs two ratios rather than one — lives in
# backtest.resolution_gate.
MAX_UNRESOLVABLE_PCT = _MAX_UNRESOLVABLE_PCT

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_accuracy")


def _upsert_accuracy(db, rows):
    """Replace existing accuracy rows for the same key then insert new ones.

    ORDER IS LOAD-BEARING: Parquet first, DB commit second — the same discipline
    _flush_verdict_refresh and the --reresolve path already follow, for the same
    reason.

    Under the reverse order every failing append left the DB rows committed and
    the mirror untouched, and the run then exited 1. The API reads Parquet first
    and falls back to the DB only on None or an exception, so a
    present-but-stale mirror WINS over a current DB: the divergence is served,
    not detected, and each failed run widened it. That is exactly what the
    2026-08-05 append failures did, twice.

    Written first, the worst case is a mirror briefly AHEAD of the DB. The next
    run recomputes the same rows from the same frozen outcomes and rewrites
    both, so it converges; and the append dedups on the full key, so rewriting a
    row is a replace rather than a duplicate.
    """
    if rows:
        from db.parquet import append_table

        append_table(
            "prediction_accuracy",
            rows,
            ["prediction_type", "evaluation_date", "horizon_days", "model_version", "price_tier"],
        )

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


# ---------------------------------------------------------------------------
# 1. Forecast backtesting
# ---------------------------------------------------------------------------


def _quote_basis(current_price, base):
    """The price the forecast was QUOTED FROM, falling back to the resolved base.

    Scalar twin of ``backtest.actionable._prediction_base`` — one rule, two
    record shapes (SQLAlchemy Rows here, plain dicts there). Keep them in step.

    ``predict()`` builds the whole triple as ``current_price x (1 + ret)``, so
    every published dollar figure is anchored on the served quote while the
    outcome is anchored on ``resolve_anchors``. The fallback covers the two
    populations with no separate quote: legacy outcomes predating the column,
    and ``walkforward_records``, which builds ``mid = base * (1 + mid_ret)`` so
    the resolved base *is* its quote. Dropping either would shrink the cohort
    silently.
    """
    if current_price is not None and current_price > 0:
        return float(current_price)
    return float(base)


def _derive_verdict(base, actual, mid, low, high, direction_predicted, *, quote):
    """The single derivation of the verdict columns from the frozen actuals.

    Every path that produces a verdict — the first resolution, the re-derived
    scoring records, and the Task 8c refresh of the stored columns — goes
    through this one function, so a scoring change cannot land on some of them
    and not others. Pure: no DB, no clock, no archive.

    ``quote`` is keyword-only and REQUIRED. It has no default on purpose: a
    default would let a new caller silently reintroduce the basis wedge below,
    which is precisely how this defect survived being fixed on the actionable
    leg (`2026-08-11-actionable-selection-is-the-base-wedge.md`) and not here.

    **`in_interval` is measured in return space, off `quote`.** The band was
    published as ``quote x (1 + low_ret/high_ret)`` but ``actual`` is resolved
    off ``base``, and comparing them in dollars asks whether the realised price
    landed inside a band anchored somewhere else. The two anchors disagree on
    85% of production rows by a median 5.70% / p90 37.82% against half-widths of
    10-31%, so the wedge — not the calibration — was deciding coverage. Rebasing
    the band by ``base / quote`` is the same predicate `q_hat` was fitted with
    and the same one `scripts/replay_serving.py` reports, so the two coverage
    figures are finally the same measurement.

    **`abs_error` and `pct_error` do NOT move**, and that is not an oversight.
    A dollar error is basis-free: "how far was the published price from the
    realised one" is answerable without asking what the prediction was quoted
    from, and a forecast quoted off a stale anchor really is that wrong. Only a
    predicate about a CALIBRATED WIDTH needs the width's own basis.

    **Neither leg of the outcome moves.** ``base`` and ``actual`` both stay on
    ``resolve_anchors``; only the prediction is rebased. Differencing two
    estimators across the outcome legs is the bug that let one cohort score
    61.76% and 33.74% on consecutive days, and this does not go near it.
    """
    abs_error = abs(mid - actual)
    predicted_direction = direction_predicted or "flat"
    actual_direction = direction_from_return((actual - base) / base)
    # low/quote - 1 <= actual/base - 1 <= high/quote - 1, cleared of divisions.
    # Both bases are positive, so the inequality cannot flip.
    rebase = base / _quote_basis(quote, base)
    no_band = low is None or high is None
    return {
        "direction_actual": actual_direction,
        "direction_correct": 1 if predicted_direction == actual_direction else 0,
        "in_interval": (None if no_band else (1 if low * rebase <= actual <= high * rebase else 0)),
        # The published-dollar question, kept because it is a real one: the API
        # serves a dollar band and a consumer reads it in dollars. Reported
        # beside the calibrated figure rather than in place of it — the gap
        # between the two IS the anchor wedge, which makes it attributable.
        # Not a stored column; `_verdict_for_storage` drops it.
        "in_interval_dollar": (None if no_band else (1 if low <= actual <= high else 0)),
        "abs_error": abs_error,
        # Divided by the BASE leg, not the actual. Explicit human ruling.
        "pct_error": abs(abs_error / base) * 100,
    }


def _verdict_for_storage(verdict):
    """The verdict as it is written to the forecast_outcomes columns.

    Drops the reporting-only keys, so ``_REFRESH_VERDICTS_SQL`` binds exactly
    the columns it names and adding another split needs no migration.
    """
    stored = {k: v for k, v in verdict.items() if k in VERDICT_COLUMNS}
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
            rows = db.execute(select(tbl).where(tbl.c.id > last_id).order_by(tbl.c.id).limit(batch)).fetchall()
            if not rows:
                return
            last_id = rows[-1].id
            yield rows
            if len(rows) < batch:
                return
    else:
        ids = list(forecast_ids)
        for i in range(0, len(ids), batch):
            rows = db.execute(select(tbl).where(tbl.c.forecast_id.in_(ids[i : i + batch]))).fetchall()
            if rows:
                yield rows


def _id_to_slug(db) -> dict:
    """Postgres surrogate `items.id` -> `items.item_id`, the market_hash_name slug.

    The price archive keys on `item_slug`; the ops tables key on `item_id`. With
    no bridge between them, an outcome in `ops/forecast_outcomes.parquet` cannot
    be joined to its own price history without a round-trip to Supabase, which
    is the network hop the Parquet store exists to avoid.
    """
    return {r.id: r.item_id for r in db.execute(text("SELECT id, item_id FROM items")).fetchall()}


def _with_item_slug(rows, id_to_slug):
    """The Parquet projection of outcome rows: the DB payload plus `item_slug`.

    Applied to the mirror only. The column is deliberately NOT added to the
    Postgres table — that side already has `items` to join against, and a
    denormalised copy there would be a second thing to keep true. Every writer
    of `ops/forecast_outcomes.parquet` must go through this, because
    `append_table` dedups on `forecast_id` and REPLACES the whole row: a writer
    that omitted the column would blank it on every row it refreshed.
    """
    return [{**r, "item_slug": id_to_slug.get(r["item_id"])} for r in rows]


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
        db.execute(_REFRESH_VERDICTS_SQL, updates[i : i + CHUNK])
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
    now = datetime.now(UTC).replace(tzinfo=None)
    updates, mirror = [], []
    total = 0
    # Read once for the whole walk. The mirror row must carry item_slug or the
    # refresh would blank it on every row it touches — append_table replaces
    # the whole row, it does not merge columns.
    id_to_slug = _id_to_slug(db)

    for batch in _iter_outcome_rows(db, forecast_ids):
        for r in batch:
            base, actual, mid = r.base_price, r.actual_price, r.predicted_price_mid
            if base is None or base <= 0 or actual is None or actual <= 0 or mid is None:
                continue
            derived = _verdict_for_storage(
                _derive_verdict(
                    base,
                    actual,
                    mid,
                    r.predicted_price_low,
                    r.predicted_price_high,
                    r.direction_predicted,
                    quote=r.current_price,
                )
            )
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
            mirror_row["item_slug"] = id_to_slug.get(r.item_id)
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


def _store_forecast_outcomes(db, outcomes, reresolve: bool = False, considered_ids=None, id_to_slug=None) -> int:
    """Persist per-forecast outcomes. Insert-only unless *reresolve*.

    Resolved actuals are frozen: a forecast_id that already has a row keeps
    its base_price and actual_price forever. Re-running the backtest after the
    archive gains source rows for an old target date must not rewrite history
    — that silent rewriting is why the same 5,512 forecasts scored 61.76% on
    07-18 and 33.74% on 07-19. Metrics stay derived and are recomputed every
    run, so a *scoring* fix still lands without --reresolve.

    considered_ids: every forecast this run ATTEMPTED to resolve, whether or not
        it produced an outcome. Required under --reresolve and ignored
        otherwise. The delete set is derived from it and NOT from *outcomes*,
        which is the whole point of Task 8e: a forecast that had a stored row
        and is now unresolvable — dropped by the anchor-staleness rule or the
        leg-window disjointness rule — never appears in *outcomes*, so a delete
        set built from *outcomes* left its row behind. Measured on the Task 9b
        dry run that was 8,784 rows still carrying values from the old broken
        estimator, scored into the published metric and into
        update_bias_corrections_from_outcomes, with --reresolve the only path
        that could ever have corrected them. It is a required argument rather
        than an optional one so that a future caller cannot silently reproduce
        the defect by omitting it.

    THE ASYMMETRY IS DELIBERATE. Only --reresolve deletes. On the default daily
    path a forecast that stops resolving keeps its frozen row, because there the
    likely cause is a transient archive problem, and reacting to a resolution
    failure by destroying good history would be unrecoverable and silent.
    Deletion is an explicit operator action, never an automatic reaction.
    """
    if reresolve and considered_ids is None:
        raise ValueError(
            "_store_forecast_outcomes(reresolve=True) requires considered_ids: "
            "the delete set must cover every forecast considered this run, not "
            "just the ones that resolved."
        )
    from database import ForecastOutcome

    if reresolve:
        # Every considered forecast loses its stored row; the ones that
        # resolved get a freshly written one back. Includes ids with no
        # existing row, which delete over harmlessly.
        delete_ids = set(considered_ids) | {o["forecast_id"] for o in outcomes}
        to_write = list(outcomes)
        if not delete_ids:
            return 0
    else:
        # The default path NEVER deletes. considered_ids is accepted and
        # ignored here on purpose: the caller passes it unconditionally, so
        # this empty set is the only thing standing between the daily path and
        # a delete, and a mutation of this line is visible end to end.
        delete_ids = set()
        if not outcomes:
            return 0
        all_fids = [o["forecast_id"] for o in outcomes]
        existing_ids = set()
        for i in range(0, len(all_fids), CHUNK):
            batch = all_fids[i : i + CHUNK]
            rows = db.query(ForecastOutcome.forecast_id).filter(ForecastOutcome.forecast_id.in_(batch)).all()
            existing_ids.update(r[0] for r in rows)
        to_write = [o for o in outcomes if o["forecast_id"] not in existing_ids]

        if not to_write:
            db.commit()
            logger.info(f"  All {len(outcomes):,} outcomes already resolved (frozen)")
            return 0

    resolved_at = datetime.now(UTC).replace(tzinfo=None)
    for o in to_write:
        o["evaluated_at"] = resolved_at
        o["resolved_at"] = resolved_at

    # The DB payload stays exactly as built; only the Parquet mirror gains the
    # slug. `to_write` is handed to bulk_insert_mappings below and an unmapped
    # key there is an error, so the two payloads have to diverge here.
    if id_to_slug is None:
        id_to_slug = _id_to_slug(db)
    mirror = _with_item_slug(to_write, id_to_slug)

    if delete_ids:
        from db.parquet import replace_rows

        # ORDER IS LOAD-BEARING, same discipline as _flush_verdict_refresh:
        # the Parquet mirror — the copy the API serves — is written first, in a
        # single atomic delete-and-insert, and the DB follows. A crash between
        # them leaves the mirror already corrected while the DB still holds the
        # pre-run rows, i.e. the run simply has not landed yet; re-running
        # --reresolve recomputes the same considered set and converges both.
        # Under the reverse order the same crash would leave the DB clean and
        # the SERVED copy holding orphans that no daily run can ever remove,
        # because the daily path is insert-only by design. Nothing detects that
        # state, and only a second --reresolve would fix it.
        replace_rows("forecast_outcomes", "forecast_id", delete_ids, mirror)

        deleted = 0
        ids = sorted(delete_ids)
        for i in range(0, len(ids), CHUNK):
            deleted += (
                db.query(ForecastOutcome)
                .filter(ForecastOutcome.forecast_id.in_(ids[i : i + CHUNK]))
                .delete(synchronize_session=False)
            )
        if to_write:
            db.bulk_insert_mappings(ForecastOutcome, to_write)
        db.commit()

        orphaned = deleted - len(to_write)
        logger.info(
            f"  --reresolve: rewrote {len(to_write):,} outcome(s) over "
            f"{len(delete_ids):,} considered forecast(s); {max(orphaned, 0):,} "
            f"stored row(s) whose forecast no longer resolves were REMOVED "
            f"from the DB and the Parquet mirror."
        )
        return len(to_write)

    db.bulk_insert_mappings(ForecastOutcome, to_write)
    db.commit()

    from db.parquet import append_table

    append_table("forecast_outcomes", mirror, ["forecast_id"])

    logger.info(f"  Resolved {len(to_write):,} new outcomes ({len(outcomes) - len(to_write):,} already frozen)")
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
               o.forecast_date, o.base_price, o.actual_price, o.current_price,
               o.predicted_price_low, o.predicted_price_mid,
               o.predicted_price_high, o.direction_predicted, f.confidence,
               f.model_version AS forecast_model_version
        FROM forecast_outcomes o
        LEFT JOIN item_forecasts f ON f.id = o.forecast_id
    """
    if forecast_ids is None:
        rows = db.execute(text(select_sql)).fetchall()
    else:
        stmt = text(select_sql + " WHERE o.forecast_id IN :ids").bindparams(bindparam("ids", expanding=True))
        ids = list(forecast_ids)
        rows = []
        for i in range(0, len(ids), 900):
            rows.extend(db.execute(stmt, {"ids": ids[i : i + 900]}).fetchall())

    n_unusable = 0
    n_below_min_price = 0
    # Counted per reason and logged below, never dropped silently — a scored
    # panel that shrinks without saying why is how the 07-19 rule change stayed
    # invisible for three weeks in the first place.
    n_excluded_date = defaultdict(int)
    # Escape hatch for a like-for-like read against a figure published before the
    # exclusion landed. It restores rows served by a superseded direction rule, so
    # nothing it produces is quotable as a current number.
    keep_excluded_dates = os.environ.get("SCORE_ALL_DATES") == "1"
    if keep_excluded_dates:
        logger.warning(
            "  SCORE_ALL_DATES=1: scoring dates served by superseded direction "
            "rules. NOT a publishable figure — for comparison against pre-"
            "exclusion numbers only."
        )

    groups = defaultdict(list)
    for r in rows:
        base, actual, mid = r.base_price, r.actual_price, r.predicted_price_mid
        if base is None or base <= 0 or actual is None or actual <= 0 or mid is None:
            n_unusable += 1
            continue
        if min_price > 0 and base < min_price:
            n_below_min_price += 1
            continue
        if not keep_excluded_dates:
            reason = excluded_forecast_date(r.forecast_date)
            if reason:
                n_excluded_date[reason] += 1
                continue

        verdict = _derive_verdict(
            base,
            actual,
            mid,
            r.predicted_price_low,
            r.predicted_price_high,
            r.direction_predicted,
            quote=r.current_price,
        )

        # served_identity, not the raw label: rows differing only by serving
        # configuration are one cohort, or no cohort ever reaches
        # MIN_HEADLINE_DATES. The raw label rides on the record so the merge is
        # disclosed in `config_dates` rather than silent.
        groups[(r.horizon_days, served_identity(r.model_version))].append(
            {
                # The forecast's OWN stored label, which is where the configuration
                # survives: the outcome carries the identity (it is what the next run
                # groups on), and item_forecasts is not migrated, so the legacy
                # `-regime` / `-global-only` fork stays readable from this side.
                # Falls back to the outcome's label when the join finds no forecast
                # row.
                "model_version_raw": (getattr(r, "forecast_model_version", None) or r.model_version),
                "abs_error": verdict["abs_error"],
                "pct_error": verdict["pct_error"],
                "sq_error": (mid - actual) ** 2,
                "direction_correct": verdict["direction_correct"],
                "predicted_direction": r.direction_predicted or "flat",
                "actual_direction": verdict["direction_actual"],
                "in_interval": verdict["in_interval"],
                # The two coverage figures and which basis formed this row's band.
                # `interval_coverage` alone cannot say whether it describes the
                # calibrated width or the published dollars, and the stored series
                # breaks at 2026-08-11 — a payload that cannot name its own
                # convention is not self-describing.
                "in_interval_dollar": verdict["in_interval_dollar"],
                "interval_basis_served": _quote_basis(r.current_price, base) != base,
                "confidence": r.confidence or "low",
                "base_price": base,
                "actual_price": actual,
                "price_tier": price_tier(base),
                # Frozen at resolution time, so the staleness axis stays available
                # to an archive-free --rescore. None on every row resolved before
                # 2026-08-08; score_by_staleness buckets those as `unknown`.
                "base_stale_run_days": getattr(r, "base_stale_run_days", None),
                "item_id": r.item_id,
                # The clustering unit. Outcomes sharing a forecast_date share a
                # market-wide move, so the CI must resample these, not items.
                "forecast_date": r.forecast_date,
                # The prediction leg, for the friction-conditioned metric. Both
                # columns are frozen, so this stays archive-free and --rescore keeps
                # working.
                #
                # `current_price` is here because `r_hat` divides by the price the
                # forecast was QUOTED FROM -- `predicted_mid` was built as
                # `current_price x (1 + r_hat)`, and dividing by the resolved base
                # instead recovers the wedge between the two bases, which is a median
                # 13.74% on this cohort against a 7.2-37.5% bar. It does NOT breach
                # "current_price is never scored on": the legs of `actual_ret` are
                # untouched and both stay on resolve_anchors.
                # docs/changelog/2026-08-11-actionable-selection-is-the-base-wedge.md
                "predicted_mid": mid,
                "current_price": r.current_price,
                # ActionableDA is scoped to h in {14, 30}. Carried on the record
                # rather than passed into score_cohort: the grouping key already
                # fixes it per cohort, and eight test modules call score_cohort
                # positionally.
                "horizon_days": r.horizon_days,
            }
        )

    n_scored = sum(len(v) for v in groups.values())
    logger.info(f"  Frozen outcomes: {n_scored:,} scored of {len(rows):,} considered")
    if n_below_min_price:
        logger.info(f"  {n_below_min_price:,} frozen outcome(s) below --min-price ${min_price:.2f}")
    for reason, n in sorted(n_excluded_date.items()):
        # WARNING, not info: this shrinks the panel MIN_HEADLINE_DATES counts,
        # and the reason has to travel with the number every run.
        logger.warning(
            f"  {n:,} frozen outcome(s) excluded — {reason}. SCORE_ALL_DATES=1 restores them for comparison only."
        )
    if n_unusable:
        # Loud on purpose. These rows are re-resolved by nothing and counted by
        # no gate — without this line a shrinking scored cohort is invisible.
        #
        # The hint used to read "Re-resolve them with --reresolve to bring them
        # back into the metric." That is not a promise this code can make, and
        # for part of the population that actually triggered it the truth was the
        # opposite. Run 31057993603 (2026-08-05) dropped 14,233 rows of
        # pre-freeze vintage carrying a NULL base_price — the column arrived in
        # migration 0019 on 2026-08-01, and predicted_price_mid / actual_price
        # are nullable=False, so a NULL base_price was the only reachable
        # trigger — all with target_date 2026-08-01.
        #
        # Checked against the archive rather than assumed: with the local
        # coverage set, a target of 08-01 has only {07-31, 08-01} inside the
        # actual-leg window at h=3, because 2026-07-30 is missing. That is fewer
        # than SMOOTH_WINDOW, so classify_archive_gap calls it unresolvable,
        # --reresolve puts it in considered_ids and DELETES the frozen row.
        # At h=7/14/30 the same target has 5/12/28 covered days and re-resolves
        # normally. So the old hint was correct for the long horizons and
        # destructive for the short ones, and which branch a row takes is decided
        # by archive coverage over its actual leg — hence the wording below.
        #
        # That population is gone: the cleanup documented in
        # docs/changelog/2026-08-01-deterministic-backtest.md was executed on
        # 2026-08-06, forecast_outcomes went 89,203 -> 74,970 rows and
        # `base_price IS NULL` now returns 0. This warning should therefore be
        # dormant, which makes a future occurrence a NEW population with a cause
        # nobody has diagnosed — worth investigating before it is routinely
        # deleted the way this one was.
        logger.warning(
            f"  {n_unusable:,} frozen outcome(s) UNUSABLE and dropped from scoring "
            f"(base_price/actual_price missing or non-positive, or no "
            f"predicted_price_mid). They are frozen, so they are never "
            f"re-resolved and never counted by the unresolvable gate. "
            f"--reresolve only recovers the ones whose target_date window the "
            f"archive can still supply a clean actual leg for; for the rest it "
            f"DELETES the frozen row instead of repairing it. Check their "
            f"target_date against archive coverage, and what wrote them, before "
            f"acting: the known 2026-08-01 population was cleared by deletion on "
            f"2026-08-06, so this is a new one."
        )
    return groups


def _resolve_candidates(db, today, archive_dir, reresolve=False, rescore=False):
    """Resolve shadow candidate outcomes after the production refresh.

    Candidate failures never fail production scoring — except a
    production/candidate resolution mismatch, which is an invariant
    violation and propagates. A database without migration 0027 logs and
    continues until the migration lands.
    """
    from backtest.candidate_resolution import resolve_candidate_outcomes

    try:
        return resolve_candidate_outcomes(db, today=today, archive_dir=archive_dir, reresolve=reresolve, rescore=rescore)
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"  Candidate resolution skipped ({e}); production scoring continues.")
        return {}


def _pct(value):
    return "n/a" if value is None else f"{value:.1f}%"


def _pt_str(metrics) -> str:
    """The Pesaran-Timmermann verdict as one field.

    Excess is in percentage points of hit rate ABOVE the per-date independence
    null, so 0.0 means "indistinguishable from calling the market's own
    direction distribution at random" — not 50%.
    """
    excess = metrics.get("pt_excess_pp")
    if excess is None:
        return f"n/a ({metrics.get('pt_verdict', 'unknown')})"
    t_stat = metrics.get("pt_t_stat")
    if t_stat is None:
        return f"{excess:+.2f}pp t=n/a ({metrics.get('pt_verdict')})"
    p_value = metrics.get("pt_p_value")
    p_str = "n/a" if p_value is None else f"{p_value:.3g}"
    return (
        f"{excess:+.2f}pp t={t_stat:+.2f} p={p_str} "
        f"[NW lag {metrics.get('pt_nw_lag')}, {metrics.get('pt_n_dates')} dates]"
    )


def _actionable_str(metrics) -> str:
    """The friction-conditioned numbers as one field.

    Four numbers or none: a share, a hit rate, a net return and a PT verdict. A
    hit rate on its own is what this metric exists to stop being quoted.
    """
    scope = metrics.get("actionable_scope")
    if scope != "in_scope":
        return f"n/a ({scope})"
    n = metrics["actionable_n"]
    if not n:
        return "0 rows — NO call cleared its round trip + tier spread"
    return (
        f"n={n:,} ({metrics['actionable_share_pct']:.1f}% of rows) "
        f"DA={metrics['actionable_da']:.1f}% "
        f"E[net]={metrics['actionable_e_net_pct']:+.2f}% "
        f"PT={metrics.get('actionable_pt_verdict')}"
    )


def _headline_line(horizon, model_version, metrics, n) -> tuple[int, str]:
    """The >=$1 headline as (log level, message).

    The headline is the SIGNIFICANCE TEST, not the accuracy. Raw DA appears
    only inside the triple — DA, the best constant call, and the realised
    down-rate — because on this data the first is uninterpretable without the
    other two: an always-down call scored 29.4% on one stored forecast date and
    76.9% on another, so a fixed DA is skill on one and incompetence on the
    other. See backtest/directional_test.py.

    A significantly negative statistic is REPORTED, at warning level, rather
    than folded into "no skill". It means the calls are anti-correlated with
    outcomes once the market effect is removed, which is a finding about the
    model and not an absence of one.
    """
    verdict = metrics["pt_verdict"]
    n_dates = metrics["distinct_forecast_dates"]
    lo = metrics["directional_accuracy_ci_clustered_lower"]
    hi = metrics["directional_accuracy_ci_clustered_upper"]
    # Already percent, same units as directional_accuracy — the * 100 that used
    # to live here was compensating for a scoring bug that is now fixed at the
    # source. See the units note in score_cohort.
    ci_str = f" [CI: {lo:.1f}-{hi:.1f}]" if lo is not None else " [CI: n/a, <2 forecast dates]"
    triple = (
        f"DA={metrics['directional_accuracy']:.1f}%{ci_str} "
        f"vs constant-call {_pct(metrics['constant_call_accuracy'])} "
        f"('{metrics['constant_call_direction']}') "
        f"down-rate={_pct(metrics['realised_down_rate'])}"
    )
    # The carry-forward split travels with every branch, so DA can never be read
    # without it. ~30-36% of scored outcomes have actual_price bit-identical to
    # base_price and label "flat" for free, so a pooled DA partly measures
    # archive staleness — see the partition comment in backtest.scoring.
    common = (
        f"Unchanged={metrics['unchanged_pct']:.1f}% of rows "
        f"(DA there {_pct(metrics['directional_accuracy_unchanged'])}) "
        f"DAMoved={_pct(metrics['directional_accuracy_moved'])} "
        f"MAE=${metrics['mae']:.2f} MAPE={metrics['mape']:.1f}% "
        # Both bases, always together. The calibrated figure alone invites the
        # reading that the served band covers; the dollar figure alone is what
        # was mistaken for a calibration defect. Their gap is the anchor wedge.
        f"IntCov={metrics['interval_coverage']:.1f}% "
        f"($-basis {metrics['interval_coverage_dollar_basis']:.1f}%) "
        f"ConfGap={metrics['conf_gap_pp']:.1f}pp "
        f"Skill={metrics['skill_vs_baseline']} "
        f"Actionable={_actionable_str(metrics)}"
    )
    # Named in the prefix when the cohort pooled more than one stored label, so
    # the merge is visible where the number is read. One label is the ordinary
    # case and stays quiet.
    configs = metrics.get("config_dates") or {}
    config_str = ""
    if len(configs) > 1:
        config_str = " pooling " + ", ".join(f"{k}:{v}d" for k, v in sorted(configs.items()))
    prefix = f"  [{horizon}d / {model_version}] >=$1: {n:,} samples over {n_dates} forecast dates{config_str}"

    if verdict == "skill":
        return logging.INFO, (
            f"{prefix} — DIRECTIONAL SKILL (PT t > {PT_T_HURDLE}): PT={_pt_str(metrics)} {triple} {common}"
        )
    if verdict == "perverse":
        return logging.WARNING, (
            f"{prefix} — PERVERSE (PT t < -{PT_T_HURDLE}): the model's calls are "
            f"significantly ANTI-correlated with outcomes once the per-date market "
            f"direction is removed. This is a finding, not a null. "
            f"PT={_pt_str(metrics)} {triple} {common}"
        )
    if verdict == "no_skill":
        return logging.INFO, (
            f"{prefix} — NO DIRECTIONAL SKILL: PT is inside +/-{PT_T_HURDLE}, so the "
            f"hit rate is not distinguishable from the per-date independence null. "
            f"PT={_pt_str(metrics)} {triple} {common}"
        )
    # insufficient_dates / degenerate. Refuse to quote a headline the cohort
    # cannot support: the sample_count is not the evidence here, the date count
    # is, and a DA over <MIN_HEADLINE_DATES dates mostly measures which way the
    # market moved on those days.
    return logging.WARNING, (
        f"{prefix} — NO HEADLINE ({verdict}): {metrics['pt_n_dates']} usable date(s) "
        f"below the {MIN_HEADLINE_DATES} required "
        f"({metrics['pt_n_dates_dropped']} date(s) too thin to test). Unquotable "
        f"PT={_pt_str(metrics)} {triple} {common}"
    )


def _score_groups(groups, today):
    """Turn {(horizon, model_version): records} into prediction_accuracy rows."""
    results = []
    for (horizon, model_version), records in sorted(groups.items()):
        tiered = score_by_tier(records)
        if not tiered:
            logger.info(f"  [{horizon}d / {model_version}] No valid comparisons")
            continue

        for tier, metrics, n in tiered:
            results.append(
                {
                    "prediction_type": "forecast",
                    "evaluation_date": today,
                    "horizon_days": horizon,
                    "model_version": model_version,
                    "price_tier": tier,
                    "evaluation_window_days": None,
                    "sample_count": n,
                    "metrics": metrics,
                    "created_at": datetime.now(UTC).replace(tzinfo=None),
                }
            )

        # The logged headline is the row that was just stored, not a second
        # derivation of it. A headline that is computed only for the log is a
        # number nothing can audit — see HEADLINE_TIER in backtest.scoring.
        head_metrics, head_n = next(((m, n) for t, m, n in tiered if t == HEADLINE_TIER), ({}, 0))
        penny_metrics, penny_n = score_cohort([r for r in records if r["price_tier"] < HEADLINE_MIN_TIER])

        if head_n:
            logger.log(*_headline_line(horizon, model_version, head_metrics, head_n))

        # Where does the headline stabilise? $1 is a convention and the spread
        # evidence argues the honest floor is above it, so the floors are printed
        # side by side. Reading the stored rows, not re-deriving them.
        sweep = [(FLOOR_SWEEP[t], m, n) for t, m, n in tiered if t in FLOOR_SWEEP]
        if len(sweep) > 1:
            parts = " | ".join(
                f">=${floor:g}: n={n:,} DA={m['directional_accuracy']:.1f}% PT={_pt_str(m)}" for floor, m, n in sweep
            )
            logger.info(f"  [{horizon}d / {model_version}] floor sweep — {parts}")
        if penny_n:
            # Diagnostic only, and never the headline: one cent is a 20% move
            # down here, so the up/flat/down label is tick quantisation. Carries
            # its PT verdict so it is comparable to the line above.
            logger.info(
                f"  [{horizon}d / {model_version}] <$1: {penny_n:,} samples — "
                f"PT={_pt_str(penny_metrics)} "
                f"DA={penny_metrics['directional_accuracy']:.1f}% (tick-dominated)"
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
        # Carried through even though nothing here recomputes it. append_table
        # dedups on forecast_id and REPLACES the whole row, so a projection
        # that omitted the column would blank it on every row it touched.
        "base_stale_run_days": row.base_stale_run_days,
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
      - Direction:        pt_* — the Pesaran-Timmermann headline (per forecast
                          date, HAC t-stat over dates), with directional_accuracy,
                          constant_call_accuracy and realised_down_rate as the
                          triple that makes it readable, plus the clustered CI
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
        _resolve_candidates(db, today, archive_dir=None, reresolve=False, rescore=True)
        groups = _records_from_frozen_outcomes(db, min_price=min_price)
        results = _score_groups(groups, today)
        if results:
            _upsert_accuracy(db, results)
            logger.info(f"  Stored {len(results)} forecast accuracy records")
        return results

    archive_dir = Path(__file__).parent.parent.parent / "price-archive"

    # A forecast is evaluable only when the archive covers its target date, so
    # the cutoff is the archive's last day, not the calendar's. The archive
    # always lags `today` — the collector writes yesterday's prices at best —
    # and every forecast in that lag window is mature-by-calendar and
    # unresolvable-by-data. Admitting them put 30,859 of 80,737 prod forecasts
    # into the cohort as guaranteed misses, tripping MAX_UNRESOLVABLE_PCT at
    # 38.5% so the run reported nothing at all. Because the lag is permanent,
    # the daily run would have failed identically every day.
    #
    # Clamped to `today` rather than taken from the archive alone: a backfilled
    # archive can hold days past `today`, and a caller passing an explicit
    # `today` to score a historical cohort must not have future forecasts pulled
    # in behind its back.
    coverage_end = archive_max_day(archive_dir)
    # The coverage EDGE is not enough. It says how far the archive reaches, not
    # whether the range is solid: the 2026-08-02/03 outage left a max day of
    # 08-04 that looked perfectly healthy with two days missing in the middle.
    # The interior is what separates "the collector missed a day, and no future
    # collection will fill it" from a live resolver regression. See
    # backtest.resolution_gate's GAP category.
    covered_days = archive_covered_days(archive_dir)
    cutoff = min(today, coverage_end)
    if coverage_end < today:
        logger.info(
            f"  Archive covers through {coverage_end}; evaluating forecasts "
            f"matured on or before that date rather than {today} "
            f"({(today - coverage_end).days}d of lag)"
        )

    # Fetch all forecasts with a midpoint price, filter for maturity in Python
    rows = db.execute(
        text("""
        SELECT f.id, f.item_id, f.forecast_date, f.horizon_days,
               f.price_low, f.price_mid, f.price_high,
               f.current_price, f.direction, f.confidence, f.model_version
        FROM item_forecasts f
        WHERE f.price_mid IS NOT NULL
    """)
    ).fetchall()

    # Filter for mature forecasts (forecast_date + horizon <= cutoff)
    mature = []
    for r in rows:
        forecast_date = r.forecast_date if isinstance(r.forecast_date, date) else date.fromisoformat(r.forecast_date)
        maturity_date = forecast_date + timedelta(days=r.horizon_days)
        if maturity_date <= cutoff:
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
            batch = mature_ids[i : i + 900]
            frozen_rows.extend(db.query(ForecastOutcome).filter(ForecastOutcome.forecast_id.in_(batch)).all())
    frozen_ids = {o.forecast_id for o in frozen_rows}
    to_resolve = [r for r in mature if r.id not in frozen_ids]
    logger.info(f"  {len(frozen_ids):,} already frozen, {len(to_resolve):,} to resolve")

    # Group by horizon + served identity. Only the unfrozen forecasts are
    # grouped: a group with nothing new performs no archive read at all.
    #
    # The identity, not the stored label, because this key is what
    # `resolve_outcomes` writes to `forecast_outcomes.model_version` — a
    # suffixed row here would re-fork the panel from the outcome side on every
    # later run, even with the forecast writer fixed.
    groups = defaultdict(list)
    for r in to_resolve:
        key = (r.horizon_days, served_identity(r.model_version))
        groups[key].append(r)

    id_to_slug = {}
    if to_resolve:
        slug_rows = db.execute(text("SELECT id, item_id FROM items")).fetchall()
        id_to_slug = {r.id: r.item_id for r in slug_rows}

    new_outcomes = []
    # Unresolvable failures are split by whether the archive has already moved
    # past the target date, because the two mean opposite things and no single
    # ratio separates them. See backtest.resolution_gate for the full argument;
    # in short, a chronic failure is a fixed tax that re-enters `to_resolve`
    # every run forever, while a cluster of fresh ones is a regression.
    n_unresolvable_fresh = 0
    n_unresolvable_chronic = 0
    n_unresolvable_gap = 0
    n_considered = 0
    # The forecasts this run attempted to resolve, MINUS the ones it went on to
    # exclude by --min-price. Under --reresolve this is the delete set.
    #
    # The exact invariant, because it is NOT quite "this run owns this row":
    # a forecast leaves the set only when the run positively established it is
    # out of scope, which requires a resolved base price to compare against
    # --min-price. A forecast that fails to resolve therefore stays in the set
    # even if it would have been below the threshold — there is no price to
    # test. So under `--reresolve --min-price` an unresolvable cheap forecast's
    # row IS removed while a resolvable cheap one's row is kept.
    #
    # That asymmetry is the safe direction and is deliberate. The row that is
    # kept belongs to a forecast the run declined to replace; the row that is
    # removed belongs to one that no longer resolves at all, which is precisely
    # what --reresolve exists to clear out. The reverse rule would let
    # `--reresolve --min-price` delete rows it never established anything about.
    # Pinned by test_reresolve_min_price_deletes_only_the_unresolvable.
    considered_ids: set[int] = set()

    for (horizon, model_version), forecasts in sorted(groups.items()):
        anchors = set()
        slugs = set()
        for f in forecasts:
            slug = id_to_slug.get(f.item_id)
            if slug is None:
                continue
            f_date = (
                f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            )
            slugs.add(slug)
            anchors.add((slug, f_date))
            anchors.add((slug, f_date + timedelta(days=horizon)))

        if not anchors:
            logger.info(f"  [{horizon}d / {model_version}] No slug mappings")
            n_considered += len(forecasts)
            # Counted FRESH regardless of target date. A missing slug mapping is
            # a referential-integrity break between item_forecasts and items,
            # not the archive lagging — no amount of future collection fixes it
            # and it must never be excused as a chronic archive gap.
            n_unresolvable_fresh += len(forecasts)
            considered_ids.update(f.id for f in forecasts)
            continue

        anchor_dates = [a[1] for a in anchors]
        voted = load_voted_prices(archive_dir, sorted(slugs), min(anchor_dates), max(anchor_dates))
        prices = resolve_anchors(voted, anchors)
        # Run length on the UNSMOOTHED voted series, keyed (slug, day). Computed
        # here rather than in scoring because `voted` exists only on this path:
        # backtest/scoring.py is a pure function of the frozen row and
        # --rescore never opens the archive. See migration 0021.
        stale_runs = stale_run_lookup(voted)
        logger.info(f"  Resolved {len(prices):,} of {len(anchors):,} anchors")

        for f in forecasts:
            n_considered += 1
            slug = id_to_slug.get(f.item_id)
            f_date = (
                f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            )
            target_date = f_date + timedelta(days=horizon)

            chronic = classify_chronic(target_date, coverage_end, MAX_WINDOW_SPAN_DAYS)
            # Checked ahead of chronic: a missing collection day is the more
            # specific explanation and the one with a fixable upstream cause, and
            # a row can satisfy both predicates. Both are non-fatal, so the
            # precedence only decides which reason gets reported — name the one
            # an operator can act on.
            # Counted over the leg-effective window, not the whole horizon: the
            # actual leg can only draw from (max(f, target - MAX_WINDOW_SPAN_DAYS),
            # target] — older days fail the anchor-staleness rule and days at or
            # before f fail the disjoint guard — so a hole at the target end of
            # a long horizon is a gap even when the horizon as a whole still
            # holds SMOOTH_WINDOW days. Whole-horizon counting pinned the
            # 2026-09-09 run at 14.3% fresh over 16,626 h=30 rows whose leg
            # window held exactly 2 days globally where 3 are required.
            #
            # And the actual leg is only half the pair. The base anchor resolves
            # from [f - MAX_WINDOW_SPAN_DAYS, f], so a hole BEFORE the forecast
            # starves it symmetrically: the 08-28..09-05 holes left exactly
            # {09-02, 09-06} in 09-06's base window where 3 are required,
            # dropping the whole 5,536-forecast h=3 cell on base_none while its
            # actual leg classified clean — a permanent fresh-tax that pinned
            # three runs at 20-25% (34691910482), since history is fixed and the
            # cell re-enters to_resolve forever. Checked on the same
            # positive-evidence standard as the actual leg.
            gap = classify_archive_gap(
                f_date,
                target_date,
                covered_days,
                SMOOTH_WINDOW,
                staleness_days=MAX_WINDOW_SPAN_DAYS,
            ) or classify_base_gap(
                f_date,
                covered_days,
                SMOOTH_WINDOW,
                staleness_days=MAX_WINDOW_SPAN_DAYS,
            )

            def _count_unresolvable(gap=gap, chronic=chronic):
                nonlocal n_unresolvable_fresh, n_unresolvable_chronic
                nonlocal n_unresolvable_gap
                if gap:
                    n_unresolvable_gap += 1
                elif chronic:
                    n_unresolvable_chronic += 1
                else:
                    n_unresolvable_fresh += 1

            base_res = prices.get((slug, f_date))
            actual_res = prices.get((slug, target_date))
            if base_res is None or actual_res is None:
                _count_unresolvable()
                considered_ids.add(f.id)
                continue

            base, actual = base_res.price, actual_res.price
            if base <= 0 or actual <= 0:
                _count_unresolvable()
                considered_ids.add(f.id)
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
                _count_unresolvable()
                considered_ids.add(f.id)
                continue

            mid, low, high = f.price_mid, f.price_low, f.price_high
            if min_price > 0 and base < min_price:
                # Resolved, then positively established to be out of scope. The
                # run writes no replacement for it, so it stays out of the
                # delete set and --reresolve leaves any existing row alone.
                # This is the ONLY exit from considered_ids — see the invariant
                # note where the set is declared; a forecast that never
                # resolved cannot reach here and is not excluded.
                continue

            considered_ids.add(f.id)

            # Same derivation the refresh and the scoring records use, so a
            # freshly written row is by construction already "current" and the
            # refresh below finds nothing to do for it.
            verdict = _verdict_for_storage(
                _derive_verdict(base, actual, mid, low, high, f.direction, quote=f.current_price)
            )

            new_outcomes.append(
                {
                    "forecast_id": f.id,
                    "item_id": f.item_id,
                    "forecast_date": f_date,
                    "horizon_days": horizon,
                    "target_date": target_date,
                    # current_price is the PREDICTION's basis and is now read for
                    # scoring twice — `r_hat` in backtest/actionable.py and the band
                    # rebase in `_derive_verdict` — because `predict()` quoted the
                    # whole triple from it. It is never synthesized. Written through
                    # as-is (nullable) so it stays distinguishable from base_price,
                    # which is always archive-resolved. Downstream consumers
                    # (update_bias_corrections_from_outcomes, retro_bias_check,
                    # tiered_breakdown) still read this column and must see the
                    # real serving-time value or a genuine NULL, not a stand-in.
                    "current_price": f.current_price,
                    "base_price": base,
                    # None, not 0, when the slug-day is absent from `voted` — an
                    # unknown run length must never read as "the price was fresh".
                    "base_stale_run_days": stale_runs.get((slug, f_date)),
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
                }
            )

    # Both ratios are evaluated against the whole mature cohort and against
    # this run's informative attempts respectively — see
    # backtest.resolution_gate. Divisions by zero are the gate's problem, not
    # this call site's.
    gate = evaluate_gate(
        n_mature=len(mature_ids),
        n_attempted=n_considered,
        n_unresolvable_fresh=n_unresolvable_fresh,
        n_unresolvable_chronic=n_unresolvable_chronic,
        n_unresolvable_gap=n_unresolvable_gap,
    )
    if gate.warn:
        logger.warning(f"  Resolution gate: {gate.reason}")
    else:
        logger.info(f"  Resolution gate: {gate.reason}")
    if not gate.ok:
        raise RuntimeError(gate.reason)

    # The whole mature cohort goes through the freeze, which writes only what is
    # new. Passing the frozen rows too keeps the freeze the single gate on the
    # table, and the table is what gets scored below.
    all_outcomes = [_outcome_to_mapping(o) for o in frozen_rows] + new_outcomes
    # Under --reresolve there is work to do even with nothing resolved: the
    # considered forecasts still have stale stored rows to remove.
    if all_outcomes or (reresolve and considered_ids):
        # Passed unconditionally, not only under --reresolve, so the
        # insert-only guarantee lives in _store_forecast_outcomes' own branch
        # rather than in a None the caller happens to send.
        #
        # Note what that does NOT buy: on the daily path a forecast that
        # already has a stored row is frozen and never enters `to_resolve`, so
        # it is never in considered_ids either. The daily path is therefore
        # structurally incapable of deleting a frozen row whatever this set
        # contains — which is also why only the unit test
        # test_the_default_path_does_not_delete_even_when_handed_a_considered_set
        # can discriminate a "default path deletes" mutation. Verified: that
        # mutation fails that test alone.
        _store_forecast_outcomes(
            db,
            all_outcomes,
            reresolve=reresolve,
            considered_ids=considered_ids,
            # Already read above for anchor resolution; passed so the mirror's
            # item_slug costs no second scan of `items`.
            id_to_slug=id_to_slug,
        )

    # Shadow candidates resolve against the same frozen legs production just
    # wrote, preferring those rows so both arms share byte-identical outcomes.
    _resolve_candidates(db, today, archive_dir, reresolve=reresolve)

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
    groups = _records_from_frozen_outcomes(db, min_price=min_price, forecast_ids=mature_ids)
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
        types,
        min_price=min_price,
        update_bias=update_bias,
        reresolve=reresolve,
        rescore=rescore,
    )
    print(f"RESULT: {json.dumps(result, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
