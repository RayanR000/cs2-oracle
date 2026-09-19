"""Exact shadow candidate records and config fingerprints.

Converts final served forecast triples into persistence-ready candidate
records without database imports. The production triple is treated as the
captured champion triple; the shadow is derived from it with identical
percentage geometry (see models/forecast_assembly.py).

A retry may treat an existing candidate row as identical only when its
`config_fingerprint` matches. The fingerprint covers candidate name and
version, champion mapping, interval geometry and calibration identifiers,
artifact version, relevant feature flags, and feature cutoff semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from models.forecast_assembly import assemble_interval, percentage_offsets

RANKING_CANDIDATE_NAME = "lambdarank_v1"


def config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateRecord:
    item_id: int
    forecast_date: date
    horizon_days: int
    component: str  # centre | ranking
    candidate_name: str
    candidate_version: str
    centre_price: float | None
    predicted_price_low: float | None
    predicted_price_high: float | None
    score: float | None
    anchor_price: float
    feature_cutoff_at: datetime
    artifact_version: str | None
    config_fingerprint: str


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def candidate_records(
    rows: list[dict[str, Any]],
    *,
    forecast_date: date,
    horizon_days: int,
    champion: str,
    candidate_version: str = "v1",
    artifact_version: str | None = None,
    feature_cutoff_at: datetime,
    fingerprint_payload: dict[str, Any],
) -> list[CandidateRecord]:
    """Derive shadow records from final served forecast rows.

    Each row carries the final (post-sanitization) GBM triple
    (`gbm_low`, `gbm_mid`, `gbm_high`), the exact serving anchor
    (`current_price`), and optionally `rank_score`. Input rows are never
    mutated. Candidate prices are rounded to two decimals at the record
    edge, matching production rounding.
    """
    fingerprint = config_fingerprint(fingerprint_payload)
    records: list[CandidateRecord] = []
    for row in rows:
        item_id = row["item_id"]
        anchor = float(row["current_price"])
        gbm_low = float(row["gbm_low"])
        gbm_mid = float(row["gbm_mid"])
        gbm_high = float(row["gbm_high"])

        offsets = percentage_offsets(
            low=np.array([gbm_low]), mid=np.array([gbm_mid]), high=np.array([gbm_high])
        )

        if champion == "last_price":
            shadow_name = "gbm_q50"
            shadow_centre = np.array([gbm_mid])
        else:
            shadow_name = "last_price"
            shadow_centre = np.array([anchor])
        low, mid, high = assemble_interval(shadow_centre, offsets)
        records.append(
            CandidateRecord(
                item_id=item_id,
                forecast_date=forecast_date,
                horizon_days=horizon_days,
                component="centre",
                candidate_name=shadow_name,
                candidate_version=candidate_version,
                centre_price=round(float(mid[0]), 2),
                predicted_price_low=round(float(low[0]), 2),
                predicted_price_high=round(float(high[0]), 2),
                score=None,
                anchor_price=anchor,
                feature_cutoff_at=feature_cutoff_at,
                artifact_version=artifact_version,
                config_fingerprint=fingerprint,
            )
        )

        rank_score = row.get("rank_score")
        if _finite(rank_score):
            records.append(
                CandidateRecord(
                    item_id=item_id,
                    forecast_date=forecast_date,
                    horizon_days=horizon_days,
                    component="ranking",
                    candidate_name=RANKING_CANDIDATE_NAME,
                    candidate_version=candidate_version,
                    centre_price=None,
                    predicted_price_low=None,
                    predicted_price_high=None,
                    score=float(rank_score),
                    anchor_price=anchor,
                    feature_cutoff_at=feature_cutoff_at,
                    artifact_version=artifact_version,
                    config_fingerprint=fingerprint,
                )
            )
    return records
