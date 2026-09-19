"""Idempotent persistence for shadow candidate batches.

Writes are grouped by (forecast_date, horizon_days, component) and applied
atomically per group: a partial candidate batch is rolled back rather than
leaving a selected subset of items. A retry treats an existing candidate row
as identical only when its `config_fingerprint` matches; the same identity
with a different fingerprint is a hard error — the system must never
overwrite one experiment arm with another under the same identity.

Only connection-invalidated `DBAPIError` is transient (that group is left
absent for retry). Integrity and fingerprint failures propagate.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy.exc import DBAPIError

logger = logging.getLogger(__name__)


class CandidateFingerprintConflict(Exception):
    """The same candidate identity was written with a different fingerprint."""


@dataclass
class CandidateWriteResult:
    centre_written: int = 0
    ranking_written: int = 0
    batches_expected: int = 0
    batches_written: int = 0


def _identity(record) -> tuple:
    return (
        record.item_id,
        record.forecast_date,
        record.horizon_days,
        record.component,
        record.candidate_name,
        record.candidate_version,
    )


def write_candidate_batches(db, records: list) -> CandidateWriteResult:
    """Persist candidate records idempotently, grouped per date/horizon/component."""
    from database import ForecastCandidate

    result = CandidateWriteResult()
    if not records:
        return result

    groups: dict[tuple, list] = defaultdict(list)
    for record in records:
        groups[(record.forecast_date, record.horizon_days, record.component)].append(record)
    result.batches_expected = len(groups)

    for (forecast_date, horizon_days, component), group in sorted(
        groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])
    ):
        [_identity(r) for r in group]
        item_ids = list({r.item_id for r in group})
        names = list({r.candidate_name for r in group})
        versions = list({r.candidate_version for r in group})
        existing = (
            db.query(ForecastCandidate)
            .filter(
                ForecastCandidate.forecast_date == forecast_date,
                ForecastCandidate.horizon_days == horizon_days,
                ForecastCandidate.component == component,
                ForecastCandidate.item_id.in_(item_ids),
                ForecastCandidate.candidate_name.in_(names),
                ForecastCandidate.candidate_version.in_(versions),
            )
            .all()
        )
        known = {
            (
                row.item_id,
                row.forecast_date,
                row.horizon_days,
                row.component,
                row.candidate_name,
                row.candidate_version,
            ): row.config_fingerprint
            for row in existing
        }
        for record in group:
            if _identity(record) in known and known[_identity(record)] != record.config_fingerprint:
                raise CandidateFingerprintConflict(
                    f"candidate {_identity(record)} already stored with a different "
                    "config_fingerprint; refusing to overwrite one experiment arm "
                    "with another"
                )
        missing = [r for r in group if _identity(r) not in known]
        if not missing:
            result.batches_written += 1
            continue
        try:
            with db.begin_nested():
                for record in missing:
                    db.add(
                        ForecastCandidate(
                            item_id=record.item_id,
                            forecast_date=record.forecast_date,
                            horizon_days=record.horizon_days,
                            component=record.component,
                            candidate_name=record.candidate_name,
                            candidate_version=record.candidate_version,
                            centre_price=record.centre_price,
                            predicted_price_low=record.predicted_price_low,
                            predicted_price_high=record.predicted_price_high,
                            score=record.score,
                            anchor_price=record.anchor_price,
                            feature_cutoff_at=record.feature_cutoff_at,
                            artifact_version=record.artifact_version,
                            config_fingerprint=record.config_fingerprint,
                        )
                    )
                db.flush()
        except DBAPIError as e:
            if not e.connection_invalidated:
                raise
            logger.warning(
                "  candidate batch %s/%s/%s unwritten (transient DB error); "
                "left absent for retry",
                forecast_date,
                horizon_days,
                component,
            )
            continue
        for record in missing:
            if record.component == "centre":
                result.centre_written += 1
            elif record.component == "ranking":
                result.ranking_written += 1
        result.batches_written += 1

    db.commit()
    return result
