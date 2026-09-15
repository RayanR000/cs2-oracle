"""
A/B test results API — serves regime-switching vs global-only and ensemble-size comparison data.
"""

from database import PredictionAccuracy, get_db
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/ab-test", tags=["ab_test"])

#: Served with every A/B payload. Stored results predate the 2026-08-08
#: statistics fix and the 2026-08-13 trainer fix, so they are unpowered
#: historical reads, not decision evidence.
AB_RESULTS_DISCLAIMER = (
    "Historical A/B read-out: stored results predate the 2026-08-08 statistics "
    "fix and the 2026-08-13 trainer fix. Treat as unpowered; do not cite a "
    "verdict without re-running the harness."
)


def _row_to_dict(row: PredictionAccuracy) -> dict:
    return {
        "id": row.id,
        "prediction_type": row.prediction_type,
        "evaluation_date": row.evaluation_date.isoformat() if row.evaluation_date else None,
        "horizon_days": row.horizon_days,
        "model_version": row.model_version,
        "sample_count": row.sample_count,
        "metrics": row.metrics,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _ab_comparison(
    db,
    *,
    delta_type: str,
    arms: dict[str, tuple[str, str]],  # arm_name -> (prediction_type, model_version)
    no_data_message: str,
) -> dict:
    """Fetch latest A/B evaluation for given arms, merge by horizon.

    Shared pattern: MAX(evaluation_date) for the delta records → one
    SELECT per arm + one for the delta at that date → merge by horizon.
    Arm entries carry {"sample_count", "metrics"}; the delta entry is the
    raw metrics value — matching the pre-refactor endpoint shapes.
    """
    from sqlalchemy import text

    latest_date = db.execute(
        text("""
        SELECT MAX(evaluation_date)
        FROM prediction_accuracy
        WHERE prediction_type = :t
    """),
        {"t": delta_type},
    ).scalar()

    if not latest_date:
        return {
            "status": "no_data",
            "message": no_data_message,
        }

    by_arm: dict[str, dict] = {}
    for arm_name, (pred_type, model_ver) in arms.items():
        rows = db.execute(
            text("""
            SELECT horizon_days, sample_count, metrics
            FROM prediction_accuracy
            WHERE prediction_type = :pt
              AND model_version = :mv
              AND evaluation_date = :d
            ORDER BY horizon_days
        """),
            {"pt": pred_type, "mv": model_ver, "d": latest_date},
        ).fetchall()
        by_arm[arm_name] = {r.horizon_days: r for r in rows}

    delta_rows = db.execute(
        text("""
        SELECT horizon_days, metrics
        FROM prediction_accuracy
        WHERE prediction_type = :dt
          AND evaluation_date = :d
        ORDER BY horizon_days
    """),
        {"dt": delta_type, "d": latest_date},
    ).fetchall()
    delta_by_h = {r.horizon_days: r for r in delta_rows}

    horizons = []
    all_keys: set = set(delta_by_h.keys())
    for key_map in by_arm.values():
        all_keys.update(key_map.keys())
    for h in sorted(all_keys):
        entry: dict = {"horizon_days": h}
        for arm_name, key_map in by_arm.items():
            if h in key_map:
                r = key_map[h]
                entry[arm_name] = {"sample_count": r.sample_count, "metrics": r.metrics}
        if h in delta_by_h:
            entry["delta"] = delta_by_h[h].metrics
        horizons.append(entry)

    return {
        "test_date": latest_date.isoformat() if latest_date else None,
        "disclaimer": AB_RESULTS_DISCLAIMER,
        "horizons": horizons,
    }


@router.get("/regime")
def get_regime_ab_test(
    db=Depends(get_db),
):
    """Return the latest A/B test comparison between regime and global-only models."""
    return _ab_comparison(
        db,
        delta_type="ab_test_regime_delta",
        arms={
            "regime": ("ab_test_regime", "lgbm-v3-regime"),
            "global_only": ("ab_test_regime", "lgbm-v3-global-only"),
        },
        no_data_message="No A/B test results found. Run `python scripts/ab_test_regime.py` first.",
    )


@router.get("/ensemble")
def get_ensemble_ab_test(
    db=Depends(get_db),
):
    """Return the latest A/B test comparison between 3-member and 6-member ensembles."""
    return _ab_comparison(
        db,
        delta_type="ab_test_ensemble_delta",
        arms={
            "ens3": ("ab_test_ensemble", "lgbm-v3-ens3"),
            "ens6": ("ab_test_ensemble", "lgbm-v3-ens6"),
        },
        no_data_message="No ensemble A/B test results found. Run `python scripts/ab_test_ensemble.py` first.",
    )
