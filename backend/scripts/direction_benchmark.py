#!/usr/bin/env python3
"""Offline-only direction benchmark.

Trains the three-class direction classifier with the production universe,
features, fixed boost rounds, and the `horizon + 13` embargo, then reports
the Pesaran-Timmermann statistic together with directional accuracy, the
realised down rate, and the hindsight constant-call accuracy.

Offline-only by construction: it trains into a temporary model directory
and never calls production save/write paths, never writes forecasts, and
never mutates champion configuration. The API stays neutral regardless.

Usage:
    venv/bin/python -m scripts.direction_benchmark --horizon 7 --archive-dir ../price-archive
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.directional_test import constant_call_baseline, pesaran_timmermann, realised_down_rate
from backtest.scoring import MIN_HEADLINE_DATES
from models import direction as direction_lib
from models.forecaster import ItemForecaster, embargo_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("direction_benchmark")


def format_direction_report(metrics: dict) -> str:
    """One block carrying the metric quartet. DA is never reported alone."""
    lines = [
        f"horizon: {metrics.get('horizon')}d over {metrics.get('n_dates')} dates ({metrics.get('n_rows'):,} rows)",
        f"Pesaran-Timmermann: excess={metrics.get('pt_excess_pp')}pp "
        f"t={metrics.get('pt_t_stat')} p={metrics.get('pt_p_value')} "
        f"verdict={metrics.get('pt_verdict')}",
        f"directional_accuracy: {metrics.get('directional_accuracy')}%",
        f"constant_call_accuracy: {metrics.get('constant_call_accuracy')}% "
        f"('{metrics.get('constant_call_direction')}')",
        f"realised_down_rate: {metrics.get('realised_down_rate')}%",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline direction-classifier benchmark")
    parser.add_argument("--horizon", type=int, default=7, choices=[3, 7, 14, 30])
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--days-back", type=int, default=730)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--boost-rounds", type=int, default=200)
    args = parser.parse_args()

    horizon = args.horizon
    embargo = embargo_days(horizon)
    logger.info(f"horizon={horizon}d embargo={embargo}d (horizon + 13)")

    tmpdir = tempfile.mkdtemp(prefix="direction-benchmark-")
    from database import SessionLocal

    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db, model_dir=tmpdir)
        if args.archive_dir:
            forecaster.archive_dir = Path(args.archive_dir)
        logger.info("Building training data (production universe and features)...")
        df = forecaster.build_training_data(
            days_back=args.days_back,
            backfilled_only=True,
            max_feature_rows=args.max_rows,
            min_median_price=1.0,
            universe="train",
        )
    finally:
        db.close()

    tdf = forecaster.prepare_targets(df, horizon).sort_values("date")
    dates = sorted(pd.to_datetime(tdf["date"].unique()))
    cut = dates[int(len(dates) * 0.7)]
    holdout_start = cut + pd.to_timedelta(int(embargo), unit="D")
    train_df = tdf[pd.to_datetime(tdf["date"]) < cut]
    hold_df = tdf[pd.to_datetime(tdf["date"]) >= holdout_start]
    logger.info(
        f"train: {len(train_df):,} rows to {cut.date()}; "
        f"holdout: {len(hold_df):,} rows from {holdout_start.date()} ({embargo}d embargo)"
    )
    if hold_df.empty or train_df.empty:
        sys.exit("Empty train or holdout split; widen --days-back.")

    feature_cols = list(forecaster.feature_cols)
    medians = train_df[feature_cols].median()
    X_train = forecaster._impute_features(train_df[feature_cols], medians)
    X_hold = forecaster._impute_features(hold_df[feature_cols], medians)
    y_train = train_df[f"target_return_{horizon}d"].to_numpy()
    tier_train = train_df["price_tier"].to_numpy() if "price_tier" in train_df.columns else None

    t0 = time.time()
    clf = forecaster._fit_direction_classifier(
        X_train,
        y_train,
        X_hold,
        hold_df[f"target_return_{horizon}d"].to_numpy(),
        forecaster.BOOSTING_TYPE,
        ItemForecaster._direction_tree_params({}),
        horizon=horizon,
        sigma_train=None,
        sigma_val=None,
        tier_train=tier_train,
        num_boost_round=args.boost_rounds,
        early_stopping=False,
    )
    logger.info(f"  direction fit took {time.time() - t0:.0f}s ({args.boost_rounds} fixed rounds)")

    pred_cls = clf.predict(X_hold).argmax(axis=1)
    actual_cls = direction_lib.direction_classes(
        hold_df[f"target_return_{horizon}d"].to_numpy(), direction_lib.DIRECTION_FLAT_TOLERANCE_PCT
    )
    hold_dates = pd.to_datetime(hold_df["date"]).dt.strftime("%Y-%m-%d").tolist()
    records = direction_lib.direction_records_from_classes(pred_cls, actual_cls, hold_dates)

    pt = pesaran_timmermann(records, MIN_HEADLINE_DATES)
    constant_call_direction, constant_call_accuracy = constant_call_baseline(records)
    down_rate = realised_down_rate(records)
    hits = sum(r["direction_correct"] for r in records)
    metrics = {
        "horizon": horizon,
        "n_dates": len(set(hold_dates)),
        "n_rows": len(records),
        "directional_accuracy": round(hits / len(records) * 100, 2),
        "constant_call_accuracy": None if constant_call_accuracy is None else round(constant_call_accuracy, 2),
        "constant_call_direction": constant_call_direction,
        "realised_down_rate": None if down_rate is None else round(down_rate, 2),
        **pt,
    }
    print(format_direction_report(metrics))
    # Temporary model dir and a closed read-only DB session: nothing
    # production is written, and champion configuration is untouched.


if __name__ == "__main__":
    main()
