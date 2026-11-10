"""Idempotent persistence for shadow candidate batches.

Writes are grouped by (forecast_date, horizon_days, component) and applied
one group at a time into the monthly ops partitions. A retry treats an existing candidate row
as identical only when its `config_fingerprint` matches; the same identity
with a different fingerprint is a hard error — the system must never
overwrite one experiment arm with another under the same identity.

Fingerprint conflicts and write failures propagate.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

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


def _known_fingerprints(dates) -> dict[tuple, str]:
    """Stored {identity: config_fingerprint} for the given forecast dates; empty before the first write."""
    from db import ops_partitions, ops_store

    if not ops_partitions.month_files("forecast_candidates"):
        return {}
    days = ", ".join(f"'{d}'" for d in sorted({str(d) for d in dates}))
    rows = ops_store.reader(("forecast_candidates",))(
        "SELECT item_id, forecast_date, horizon_days, component, candidate_name, candidate_version, "
        f"config_fingerprint FROM forecast_candidates WHERE forecast_date IN ({days})"
    )
    return {
        (
            r.item_id,
            r.forecast_date,
            r.horizon_days,
            r.component,
            r.candidate_name,
            r.candidate_version,
        ): r.config_fingerprint
        for r in rows
    }


def write_candidate_batches(records: list) -> CandidateWriteResult:
    """Persist candidate records idempotently into the ops partitions, per date/horizon/component.

    Phase 4: the partitions are the only store. Ids come from `assign_ids` (an identity already
    stored keeps its id). Failures raise: with no Postgres a swallowed error loses the batch.
    """
    from dataclasses import asdict

    import pandas as pd
    from database import utcnow_naive

    from db import ops_partitions

    result = CandidateWriteResult()
    if not records:
        return result

    groups: dict[tuple, list] = defaultdict(list)
    for record in records:
        groups[(record.forecast_date, record.horizon_days, record.component)].append(record)
    result.batches_expected = len(groups)
    known = _known_fingerprints({r.forecast_date for r in records})
    cols = ops_partitions.SPECS["forecast_candidates"].columns

    for (_, _, _), group in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
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
        frame = pd.DataFrame([{k: v for k, v in asdict(r).items() if k in cols} for r in missing])
        frame = ops_partitions.assign_ids("forecast_candidates", frame, base=ops_partitions.READ_ONLY_ID_BASE)
        frame["created_at"] = utcnow_naive()
        ops_partitions.upsert("forecast_candidates", frame)
        for record in missing:
            if record.component == "centre":
                result.centre_written += 1
            elif record.component == "ranking":
                result.ranking_written += 1
        result.batches_written += 1
    return result
