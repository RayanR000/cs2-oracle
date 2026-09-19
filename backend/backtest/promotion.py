"""Manual centre-promotion gate.

Evaluation produces evidence; a reviewed configuration change promotes a
candidate. Verdicts never mutate `CENTRE_CHAMPIONS`. This module has no
database imports: reports load rows and call
`evaluate_centre_promotion` with plain structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtest.candidate_scoring import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PairedDailyInterval,
    paired_daily_interval,
)
from backtest.scoring import MIN_HEADLINE_DATES

MIN_SHARED_DATES = MIN_HEADLINE_DATES
MIN_ROW_OVERLAP = 0.95
MIN_BATCH_COMPLETENESS = 0.80
MAX_COVERAGE_REGRESSION = 0.02
MAX_WIDTH_DELTA_PP = 0.001
PROMOTION_CI = 90

PASS_FOR_MANUAL_PROMOTION = "PASS_FOR_MANUAL_PROMOTION"
REJECTED = "REJECTED"
UNRESOLVED = "UNRESOLVED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
DATA_INTEGRITY_FAILURE = "DATA_INTEGRITY_FAILURE"


@dataclass(frozen=True)
class PromotionResult:
    verdict: str
    horizon: int
    shared_dates: int
    delta_mae: PairedDailyInterval | None
    delta_coverage: PairedDailyInterval | None
    width_delta_pp: float | None
    median_width_last_pp: float | None
    median_width_gbm_pp: float | None
    completeness: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _overlap_by_date(rows_last, rows_gbm) -> dict:
    """Per-date (intersection, union) item counts of the two arms."""
    last_items: dict[str, set] = {}
    gbm_items: dict[str, set] = {}
    for row in rows_last:
        last_items.setdefault(str(row["forecast_date"]), set()).add(row["item_id"])
    for row in rows_gbm:
        gbm_items.setdefault(str(row["forecast_date"]), set()).add(row["item_id"])
    overlap = {}
    for key in set(last_items) | set(gbm_items):
        inter = last_items.get(key, set()) & gbm_items.get(key, set())
        union = last_items.get(key, set()) | gbm_items.get(key, set())
        overlap[key] = (len(inter), len(union))
    return overlap


def evaluate_centre_promotion(
    *,
    horizon: int,
    rows_last: list[dict],
    rows_gbm: list[dict],
    candidate_versions,
    candidate_fingerprints,
    expected_batches: int,
    observed_batches: int,
    resolution_ok: bool = True,
) -> PromotionResult:
    """Paired centre verdict for one horizon. Never mutates configuration."""
    shared = sorted(set(str(r["forecast_date"]) for r in rows_last) & set(str(r["forecast_date"]) for r in rows_gbm))
    completeness = (observed_batches / expected_batches) if expected_batches else 0.0

    if len(shared) < MIN_SHARED_DATES:
        return PromotionResult(
            verdict=INSUFFICIENT_EVIDENCE,
            horizon=horizon,
            shared_dates=len(shared),
            delta_mae=None,
            delta_coverage=None,
            width_delta_pp=None,
            median_width_last_pp=None,
            median_width_gbm_pp=None,
            completeness=completeness,
            reasons=(f"only {len(shared)} shared dates < {MIN_SHARED_DATES}",),
        )

    overlap = _overlap_by_date(rows_last, rows_gbm)
    bad_overlap = sorted(k for k, (inter, union) in overlap.items() if union and inter / union < MIN_ROW_OVERLAP)
    versions = set(candidate_versions or ())
    fingerprints = set(candidate_fingerprints or ())
    integrity_reasons = []
    if bad_overlap:
        integrity_reasons.append(f"row overlap below {MIN_ROW_OVERLAP} on {len(bad_overlap)} date(s)")
    if len(versions) != 1:
        integrity_reasons.append(f"mixed candidate versions: {sorted(versions)}")
    if len(fingerprints) != 1:
        integrity_reasons.append(f"mixed config fingerprints: {len(fingerprints)} distinct")
    if completeness < MIN_BATCH_COMPLETENESS:
        integrity_reasons.append(f"batch completeness {completeness:.2f} < {MIN_BATCH_COMPLETENESS}")
    if not resolution_ok:
        integrity_reasons.append("candidate/production resolution mismatch")
    if integrity_reasons:
        return PromotionResult(
            verdict=DATA_INTEGRITY_FAILURE,
            horizon=horizon,
            shared_dates=len(shared),
            delta_mae=None,
            delta_coverage=None,
            width_delta_pp=None,
            median_width_last_pp=None,
            median_width_gbm_pp=None,
            completeness=completeness,
            reasons=tuple(integrity_reasons),
        )

    # Paired items within each shared date.
    paired_last_err, paired_gbm_err, pair_dates = [], [], []
    paired_last_cov, paired_gbm_cov = [], []
    widths_last, widths_gbm = [], []
    shared_set = set(shared)
    gbm_by_key = {(str(r["forecast_date"]), r["item_id"]): r for r in rows_gbm}
    for row in rows_last:
        key = (str(row["forecast_date"]), row["item_id"])
        if key[0] not in shared_set or key not in gbm_by_key:
            continue
        mate = gbm_by_key[key]
        paired_last_err.append(float(row["abs_error"]))
        paired_gbm_err.append(float(mate["abs_error"]))
        pair_dates.append(key[0])
        paired_last_cov.append(1.0 if row["in_interval"] else 0.0)
        paired_gbm_cov.append(1.0 if mate["in_interval"] else 0.0)
        widths_last.append(float(row["width_pct"]))
        widths_gbm.append(float(mate["width_pct"]))

    delta_mae = paired_daily_interval(
        paired_last_err, paired_gbm_err, pair_dates,
        ci=PROMOTION_CI, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
    )
    delta_coverage = paired_daily_interval(
        paired_last_cov, paired_gbm_cov, pair_dates,
        ci=PROMOTION_CI, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
    )
    median_last = float(np.median(widths_last)) if widths_last else None
    median_gbm = float(np.median(widths_gbm)) if widths_gbm else None
    width_delta = (median_last - median_gbm) if (median_last is not None and median_gbm is not None) else None

    if (
        delta_mae.upper is not None
        and delta_mae.upper <= 0
        and delta_mae.point < 0
        and delta_coverage.lower is not None
        and delta_coverage.lower >= -MAX_COVERAGE_REGRESSION
        and width_delta is not None
        and abs(width_delta) <= MAX_WIDTH_DELTA_PP
    ):
        verdict = PASS_FOR_MANUAL_PROMOTION
        reasons: tuple[str, ...] = ()
    elif delta_mae.lower is not None and delta_mae.lower > 0:
        verdict = REJECTED
        reasons = ("last-price MAE is significantly worse",)
    else:
        verdict = UNRESOLVED
        reasons = ("evidence does not establish either direction",)

    return PromotionResult(
        verdict=verdict,
        horizon=horizon,
        shared_dates=len(shared),
        delta_mae=delta_mae,
        delta_coverage=delta_coverage,
        width_delta_pp=width_delta,
        median_width_last_pp=median_last,
        median_width_gbm_pp=median_gbm,
        completeness=completeness,
        reasons=reasons,
    )
