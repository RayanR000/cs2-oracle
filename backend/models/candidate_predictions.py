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
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from models.centre_policy import HORIZONS, centre_champion
from models.forecast_assembly import assemble_interval, percentage_offsets

logger = logging.getLogger(__name__)

RANKING_CANDIDATE_NAME = "lambdarank_v1"

# Environment flags that change what a candidate means. Read at call time so
# the fingerprint describes the run that produced the rows.
CANDIDATE_FINGERPRINT_FLAGS = (
    "RANKING_HEAD",
    "EXCEEDANCE_HEAD",
    "ANOMALY_GBM",
    "CLIMATOLOGY_SCALE",
    "CLIMATOLOGY_REACTIVE",
    "EXCEEDANCE_SCALE",
    "SIGMA_EXPONENT",
    "LEARNED_SCALE",
    "FEATURE_NATIVE_NAN",
    "NAIVE_INIT_SCORE",
    "LABEL_SMOOTHED_ANCHOR",
)

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
            assert rank_score is not None  # narrowing; _finite rejects None
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


def _first_present(values) -> Any | None:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return None


def apply_centre_policy(
    result_df: pd.DataFrame,
    *,
    forecast_date: date | None = None,
    horizons: Iterable[int] | None = None,
    candidate_version: str = "v1",
    artifact_version: str | None = None,
    feature_cutoff_at: datetime | None = None,
    feedback_factors: dict[int, float] | None = None,
    interval_offsets: dict[int, Any] | None = None,
    feature_flags: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[CandidateRecord]]:
    """Split final forecasts into champion public triples and shadow records.

    The public frame keeps its shape; only horizons whose configured
    champion is not `gbm_q50` get their low/mid/high replaced (by the
    challenger triple at identical percentage geometry). Shadows are exact
    per-horizon candidate records. A horizon that cannot be built logs a
    structured warning and keeps its production triple with no shadows —
    shadow loss never fails production.
    """
    records = result_df.to_dict("records") if len(result_df) else []
    if forecast_date is None:
        forecast_date = _first_present([r.get("anchor_date") for r in records])
        if forecast_date is None:
            forecast_date = date.today()
        elif not isinstance(forecast_date, date):
            forecast_date = pd.to_datetime(forecast_date).date()
    if horizons is None:
        seen: set[int] = set()
        for rec in records:
            for key in (rec.get("forecasts") or {}):
                try:
                    seen.add(int(key))
                except (TypeError, ValueError):
                    continue
        horizons = sorted(seen)
    if feature_cutoff_at is None:
        feature_cutoff_at = _first_present([r.get("generated_at") for r in records])
        if feature_cutoff_at is None:
            feature_cutoff_at = datetime.now().replace(microsecond=0)
    feedback_factors = dict(feedback_factors or {})
    interval_offsets = dict(interval_offsets or {})
    if feature_flags is None:
        feature_flags = {flag: os.environ.get(flag, "0") for flag in CANDIDATE_FINGERPRINT_FLAGS}

    public_forecasts = [{h: dict(f) for h, f in (rec.get("forecasts") or {}).items()} for rec in records]
    shadows: list[CandidateRecord] = []
    for horizon in list(horizons):
        try:
            champion = centre_champion(int(horizon))
        except ValueError as e:
            logger.warning("  Shadow %sd: skipped (%s)", horizon, e)
            continue
        rows: list[dict[str, Any]] = []
        for rec in records:
            fcast = (rec.get("forecasts") or {}).get(horizon)
            if not fcast:
                continue
            rows.append(
                {
                    "item_id": rec.get("item_id"),
                    "current_price": rec.get("current_price"),
                    "gbm_low": fcast.get("low"),
                    "gbm_mid": fcast.get("mid"),
                    "gbm_high": fcast.get("high"),
                    "rank_score": fcast.get("rank_score"),
                }
            )
        if not rows:
            continue
        offsets = interval_offsets.get(int(horizon))
        payload = {
            "champion": champion,
            "candidate_version": candidate_version,
            "centre_champions": {str(h): centre_champion(h) for h in HORIZONS},
            "artifact_version": artifact_version,
            "offsets": (list(offsets) if offsets is not None else None),
            "feedback_factor": feedback_factors.get(int(horizon)),
            "flags": dict(feature_flags),
            "feature_cutoff_at": str(feature_cutoff_at),
        }
        try:
            horizon_shadows = candidate_records(
                rows,
                forecast_date=forecast_date,
                horizon_days=int(horizon),
                champion=champion,
                candidate_version=candidate_version,
                artifact_version=artifact_version,
                feature_cutoff_at=feature_cutoff_at,
                fingerprint_payload=payload,
            )
        except Exception as e:
            logger.warning(
                "  Shadow %sd: omitted %d-row batch (%s); production unaffected",
                horizon,
                len(rows),
                e,
            )
            continue
        shadows.extend(horizon_shadows)
        if champion != "gbm_q50":
            # Public follows the promoted champion: the last-price triple at
            # identical geometry, recomputed from the served GBM triple and
            # the exact anchor.
            for rec, pub in zip(records, public_forecasts):
                fcast = pub.get(horizon)
                if not fcast:
                    continue
                try:
                    off = percentage_offsets(
                        low=np.array([float(fcast["low"])]),
                        mid=np.array([float(fcast["mid"])]),
                        high=np.array([float(fcast["high"])]),
                    )
                    anchor = float(rec.get("current_price"))
                    low, mid, high = assemble_interval(np.array([anchor]), off)
                except Exception as e:
                    logger.warning(
                        "  Shadow %sd: public replacement skipped for one row (%s)",
                        horizon,
                        e,
                    )
                    continue
                fcast["low"] = round(float(low[0]), 2)
                fcast["mid"] = round(float(mid[0]), 2)
                fcast["high"] = round(float(high[0]), 2)

    public_df = result_df.copy()
    public_df["forecasts"] = public_forecasts
    return public_df, shadows
