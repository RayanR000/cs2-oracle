#!/usr/bin/env python3
"""Print per-tier, per-horizon accuracy breakdown from forecast_outcomes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    sql = """
        SELECT horizon_days, tier, n, correct,
               ROUND((dir_acc_raw * 100)::numeric, 1) AS dir_acc,
               ROUND(mae_raw::numeric, 2) AS mae,
               tp, fp, fn, tn
        FROM (
            SELECT 
                horizon_days,
                CASE 
                    WHEN current_price < 1.0 THEN '<$1'
                    WHEN current_price < 5.0 THEN '$1-5'
                    WHEN current_price < 20.0 THEN '$5-20'
                    WHEN current_price < 100.0 THEN '$20-100'
                    ELSE '>$100'
                END AS tier,
                COUNT(*) AS n,
                SUM(direction_correct) AS correct,
                AVG(direction_correct) AS dir_acc_raw,
                AVG(abs_error) AS mae_raw,
                SUM(CASE WHEN direction_predicted = 'up' AND direction_actual = 'up' THEN 1 ELSE 0 END) AS tp,
                SUM(CASE WHEN direction_predicted = 'up' AND direction_actual = 'down' THEN 1 ELSE 0 END) AS fp,
                SUM(CASE WHEN direction_predicted = 'down' AND direction_actual = 'up' THEN 1 ELSE 0 END) AS fn,
                SUM(CASE WHEN direction_predicted = 'down' AND direction_actual = 'down' THEN 1 ELSE 0 END) AS tn
            FROM forecast_outcomes
            WHERE model_version = 'lgbm-v3'
              AND evaluated_at >= NOW() - INTERVAL '2 days'
            GROUP BY horizon_days, 
                CASE 
                    WHEN current_price < 1.0 THEN '<$1'
                    WHEN current_price < 5.0 THEN '$1-5'
                    WHEN current_price < 20.0 THEN '$5-20'
                    WHEN current_price < 100.0 THEN '$20-100'
                    ELSE '>$100'
                END
        ) sub
        ORDER BY horizon_days, 
            CASE tier
                WHEN '<$1' THEN 1 WHEN '$1-5' THEN 2 
                WHEN '$5-20' THEN 3 WHEN '$20-100' THEN 4 
                WHEN '>$100' THEN 5
            END
    """
    rows = db.execute(text(sql)).fetchall()

    header = f"{'Horizon':>7} {'Tier':>9} {'n':>6} {'DirAcc':>7} {'MAE':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r.horizon_days:>4}d {r.tier:>9} {r.n:>6} {r.dir_acc:>6}% {r.mae:>6}$ {r.tp:>5} {r.fp:>5} {r.fn:>5} {r.tn:>5}")

    inv_sql = """
        SELECT horizon_days, n, dir_acc, mae
        FROM (
            SELECT 
                horizon_days,
                COUNT(*) AS n,
                ROUND((AVG(direction_correct) * 100)::numeric, 1) AS dir_acc,
                ROUND(AVG(abs_error)::numeric, 2) AS mae
            FROM forecast_outcomes
            WHERE model_version = 'lgbm-v3'
              AND evaluated_at >= NOW() - INTERVAL '2 days'
              AND current_price >= 5.0 AND current_price < 100.0
            GROUP BY horizon_days
        ) sub
        ORDER BY horizon_days
    """
    inv = db.execute(text(inv_sql)).fetchall()
    print()
    print("--- Investment Tier ($5-100) ---")
    for r in inv:
        print(f"  {r.horizon_days:>4}d  n={r.n:>5}  DirAcc={r.dir_acc:>5}%  MAE=${r.mae}")

    print()
    print("--- Change vs Pre-Correction Baseline ---")
    baseline = {
        (3,  '$5-20'): 40.3, (3,  '$20-100'): 38.3,
        (7,  '$5-20'): 38.2, (7,  '$20-100'): 36.5,
        (14, '$5-20'): 39.6, (14, '$20-100'): 40.2,
        (30, '$5-20'): 39.8, (30, '$20-100'): 39.0,
    }
    for r in rows:
        if r.tier in ('$5-20', '$20-100'):
            prev = baseline.get((r.horizon_days, r.tier), None)
            delta = f"{float(r.dir_acc) - prev:+.1f}pp" if prev is not None else ""
            print(f"  {r.horizon_days:>4}d {r.tier:>9}  {r.dir_acc:>5}%  (was {prev}%)  {delta}")

finally:
    db.close()
