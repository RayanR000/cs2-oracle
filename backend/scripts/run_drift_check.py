"""Run data drift detection and save reports.

Usage:
    venv/bin/python -m scripts.run_drift_check [--save-reference] [--output-dir DIR]
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.path.join("data", "drift_reports")
DEFAULT_REF_PATH = os.path.join("data", "drift_reference.parquet")


def save_reference(df: pd.DataFrame, path: str = DEFAULT_REF_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Reference snapshot saved: {path} ({len(df)} rows, {len(df.columns)} cols)")


def load_reference(path: str = DEFAULT_REF_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(description="Run data drift detection")
    parser.add_argument("--save-reference", action="store_true", help="Save current features as reference")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ref-path", default=DEFAULT_REF_PATH)
    args = parser.parse_args()

    from unittest.mock import MagicMock

    from models.forecaster import ItemForecaster

    logger.info("Building current feature frame...")
    forecaster = ItemForecaster(db_session=MagicMock())
    df = forecaster.build_training_data(
        days_back=90,
        backfilled_only=True,
        max_feature_rows=100_000,
        min_median_price=1.0,
        universe="train",
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_frame = df[numeric_cols]

    if args.save_reference:
        save_reference(feature_frame, args.ref_path)
        return

    if not os.path.exists(args.ref_path):
        logger.warning(f"No reference snapshot at {args.ref_path}. Run with --save-reference first.")
        save_reference(feature_frame, args.ref_path)
        logger.info("Saved current frame as reference. No comparison to make on first run.")
        return

    reference = load_reference(args.ref_path)
    logger.info(f"Reference: {len(reference)} rows, {len(reference.columns)} cols")
    logger.info(f"Current: {len(feature_frame)} rows, {len(feature_frame.columns)} cols")

    from monitoring.drift import detect_drift

    report = detect_drift(
        reference=reference,
        current=feature_frame,
        output_dir=args.output_dir,
    )

    summary = {
        "drift_detected": report.drift_detected,
        "n_drifted_features": report.n_drifted_features,
        "n_total_features": report.n_total_features,
        "drifted_features": report.drifted_features,
        "feature_scores": report.feature_scores,
    }
    summary_path = os.path.join(args.output_dir, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Drift detected: {report.drift_detected}")
    logger.info(f"Drifted features ({report.n_drifted_features}/{report.n_total_features}): {report.drifted_features}")

    if report.drift_detected:
        logger.warning("DRIFT DETECTED — review drift_report.html")
        sys.exit(1)


if __name__ == "__main__":
    main()
