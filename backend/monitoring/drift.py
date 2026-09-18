"""Data drift detection using Evidently AI.

Compares a reference feature distribution to a current one and produces
a drift report (JSON + HTML).
"""

import json
import logging
import os
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Results of a drift detection run."""

    drift_detected: bool
    n_drifted_features: int
    n_total_features: int
    drifted_features: list[str] = field(default_factory=list)
    feature_scores: dict[str, float] = field(default_factory=dict)
    json_path: str = ""
    html_path: str = ""


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str = "data/drift_reports",
    feature_cols: list[str] | None = None,
) -> DriftReport:
    """Run Evidently drift detection on numeric columns."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    os.makedirs(output_dir, exist_ok=True)

    if feature_cols:
        ref = reference[feature_cols].select_dtypes(include="number")
        cur = current[feature_cols].select_dtypes(include="number")
    else:
        common = sorted(set(reference.columns) & set(current.columns))
        ref = reference[common].select_dtypes(include="number")
        cur = current[common].select_dtypes(include="number")

    # Evidently raises on all-NaN columns; drop them.
    non_empty = [c for c in ref.columns if ref[c].notna().any() and cur[c].notna().any()]
    dropped = len(ref.columns) - len(non_empty)
    if dropped:
        logger.info(f"Dropped {dropped} all-NaN columns before drift check")
    ref = ref[non_empty]
    cur = cur[non_empty]

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)

    html_path = os.path.join(output_dir, "drift_report.html")
    json_path = os.path.join(output_dir, "drift_report.json")

    snapshot.save_html(html_path)
    snapshot.save_json(json_path)

    with open(json_path) as f:
        result = json.load(f)

    drifted_features: list[str] = []
    feature_scores: dict[str, float] = {}

    metrics = result.get("metrics", [])
    for metric in metrics:
        config = metric.get("config", {})
        metric_type = config.get("type", "")

        # Per-column drift metric: evidently:metric_v2:ValueDrift
        if "ValueDrift" in metric_type:
            col_name = config.get("column", "")
            threshold = config.get("threshold", 0.05)
            score = metric.get("value", 1.0)
            feature_scores[col_name] = score
            # score is a p-value; drift detected when score < threshold
            if score < threshold:
                drifted_features.append(col_name)

    n_drifted = len(drifted_features)
    overall_drift = n_drifted > 0

    return DriftReport(
        drift_detected=overall_drift,
        n_drifted_features=n_drifted,
        n_total_features=len(ref.columns),
        drifted_features=drifted_features,
        feature_scores=feature_scores,
        json_path=json_path,
        html_path=html_path,
    )
