"""Log interval scoring metrics to MLflow.

Designed to be called after ``score_intervals`` returns its dict. Skips
non-numeric values (lists, nested dicts other than known sub-keys) and NaNs.
"""

from __future__ import annotations

import math


def log_interval_metrics(metrics: dict, prefix: str = "") -> None:
    """Log all numeric metrics from a score_intervals result dict to MLflow.

    Parameters
    ----------
    metrics : dict
        Output of ``evaluation.interval_scoring.score_intervals``.
    prefix : str
        Optional prefix for all metric keys (e.g. ``"h7_"``).
    """
    import mlflow

    for key, value in metrics.items():
        if isinstance(value, dict):
            # Known sub-dicts: adaptivity, coverage_by_tier
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float)) and not math.isnan(sub_val):
                    mlflow.log_metric(f"{prefix}{key}.{sub_key}", sub_val)
        elif isinstance(value, (int, float)) and not math.isnan(value):
            mlflow.log_metric(f"{prefix}{key}", value)
        # Skip lists (calibration_curve), strings, etc.
