#!/usr/bin/env python3
"""
A/B test: does giving the directional classifier its OWN Optuna search (against
multi_logloss) beat inheriting the q50 quantile-regression params?

The served up/flat/down signal (`direction_models`) is trained by
`_fit_direction_classifier`, which today borrows its tree params from the q50
quantile config via `_direction_tree_params` — params optimized for pinball
loss on a GOSS median model, not for 3-class classification. This harness checks
whether a classification-tuned search actually improves the served metric.

Two arms per horizon, sharing the same items / window / folds / seed:
  - inherited: classifier tree params from `_direction_tree_params(q50)` (the q50
               Optuna winner) — current production behavior.
  - tuned:     classifier tree params from `_optuna_search_classifier_params`
               (multi_logloss Optuna), searched once on the pre-fold window.

Gate metric: plain 3-class directional accuracy on each fold — the SAME quantity
production records as `classifier_accuracy` (forecaster.py:3486). Higher = better.
Because folds are identical across arms, the paired per-fold delta is attributable.

Pre-registered gate: SHIP the tuned search iff tuned beats inherited in >= half
the paired folds AND mean accuracy delta >= +0.2pp. Otherwise keep inheriting.

Usage:
    python scripts/ab_test_classifier_hp.py [--max-items 100] [--horizon 14]
                                            [--trials 15] [--step 120]
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from database import SessionLocal
from models.forecaster import ItemForecaster
from ab_test_hp_search import load_features, _build_folds, VAL_WINDOW_DAYS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ab_test_classifier_hp")

GATE_MIN_MEAN_DELTA_PP = 0.2  # tuned must beat inherited by at least this on average


def _accuracy(model, X_val, actual_ret, forecaster):
    """Plain 3-class accuracy — matches production classifier_accuracy."""
    pred_cls = model.predict(X_val.values).argmax(axis=1)
    actual_cls = forecaster._direction_classes(actual_ret)
    return float((pred_cls == actual_cls).mean()) * 100


def run(max_items, horizon_filter, trials, step):
    import duckdb
    con = duckdb.connect()
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        db.close()
        df, feat_cols = load_features(con, forecaster, events_df, max_items)
    finally:
        con.close()

    horizons = [h for h in ItemForecaster.HORIZONS
                if horizon_filter is None or h == horizon_filter]
    results = {}

    for horizon in horizons:
        logger.info(f"\n{'=' * 60}\n  {horizon}d horizon\n{'=' * 60}")
        tdf = forecaster.prepare_targets(df, horizon)
        tcol = f"target_return_{horizon}d"
        tdf = tdf.dropna(subset=[tcol]).sort_values(["item_id", "date"]).copy()
        if tdf.empty:
            logger.warning(f"  no targets for {horizon}d")
            continue
        available = [c for c in feat_cols if c in tdf.columns]
        dates = sorted(tdf["date"].unique())
        folds = _build_folds(dates, step)
        if not folds:
            logger.warning(f"  no folds for {horizon}d")
            continue
        boosting_type = ItemForecaster.BOOSTING_TYPE_MAP.get(horizon, "gbdt")

        # ── Search both arms' params ONCE on the pre-fold train window ──
        split_idx = len(dates) * 2 // 3
        pre = tdf[tdf["date"].isin(dates[:split_idx])]
        cut = dates[split_idx - VAL_WINDOW_DAYS]
        hp_tr = pre[pre["date"] < cut]
        hp_va = pre[pre["date"] >= cut]
        med = hp_tr[available].median()
        Xtr, Xva = hp_tr[available].fillna(med), hp_va[available].fillna(med)
        ytr, yva = hp_tr[tcol].to_numpy(), hp_va[tcol].to_numpy()

        # inherited: q50 quantile Optuna winner -> _direction_tree_params
        logger.info(f"  q50 quantile search {horizon}d ({trials} trials) for inherited baseline...")
        q50 = forecaster._optuna_search_params(
            Xtr, hp_tr[tcol], Xva, hp_va[tcol],
            quantile=0.5, boosting_type=boosting_type, n_trials=trials, horizon=horizon)
        inherited_params = forecaster._direction_tree_params({0.5: q50})

        # tuned: classifier's own multi_logloss search
        logger.info(f"  classifier search {horizon}d ({trials} trials, {boosting_type})...")
        c_tr = forecaster._direction_classes(ytr)
        w_tr = forecaster._direction_sample_weights(ytr, forecaster.DIRECTION_MOVER_WEIGHT)
        c_va = forecaster._direction_classes(yva)
        tuned_params = forecaster._optuna_search_classifier_params(
            Xtr, c_tr, w_tr, Xva, c_va,
            boosting_type=boosting_type, n_trials=trials, horizon=horizon)

        arms = {"inherited": inherited_params, "tuned": tuned_params}
        results[horizon] = {"folds": len(folds), "arms": {}, "per_fold_acc": {},
                            "params": {k: v for k, v in arms.items()}}

        for arm_name, tree_params in arms.items():
            per_fold = []
            for fi, (train_dates, val_dates) in enumerate(folds):
                train_df = tdf[tdf["date"].isin(train_dates)]
                val_df = tdf[tdf["date"].isin(val_dates)]
                if len(val_df) < 50:
                    continue
                if len(train_df) > 200000:
                    train_df = train_df.sort_values("date").tail(200000)
                med = train_df[available].median()
                X_train = train_df[available].fillna(med)
                X_val = val_df[available].fillna(med)
                model = forecaster._fit_direction_classifier(
                    X_train, train_df[tcol].to_numpy(),
                    X_val, val_df[tcol].to_numpy(),
                    boosting_type, tree_params)
                acc = _accuracy(model, X_val, val_df[tcol].to_numpy(), forecaster)
                per_fold.append({"fold": fi, "val_start": str(val_dates[0]),
                                 "clf_acc": round(acc, 2), "n": len(val_df)})

            accs = [f["clf_acc"] for f in per_fold]
            results[horizon]["arms"][arm_name] = {
                "mean_clf_acc": round(float(np.mean(accs)), 2) if accs else None,
                "n_folds": len(per_fold),
                "per_fold": per_fold,
            }
            results[horizon]["per_fold_acc"][arm_name] = accs

    return results


def print_report(results):
    print("\n" + "=" * 78)
    print("CLASSIFIER HP A/B — inherited q50 params vs Optuna-tuned (identical folds)")
    print("=" * 78)
    ship_votes = []
    for horizon in sorted(results):
        r = results[horizon]
        print(f"\n  ┌─ {horizon}d ({r['folds']} folds) {'─' * 42}┐")
        print(f"  │ {'Arm':<12} {'dir-acc (higher=better)':>24} {'folds':>7}")
        print(f"  │ {'─'*12} {'─'*24} {'─'*7}")
        for arm in ("inherited", "tuned"):
            a = r["arms"].get(arm)
            if not a:
                continue
            print(f"  │ {arm:<12} {a['mean_clf_acc']:>23}% {a['n_folds']:>7}")
        inh = r["per_fold_acc"].get("inherited", [])
        tun = r["per_fold_acc"].get("tuned", [])
        if inh and tun and len(inh) == len(tun):
            deltas = np.array(tun) - np.array(inh)
            better = int(np.sum(deltas > 0))
            mean_delta = float(deltas.mean())
            gate_ship = (better >= len(deltas) / 2) and (mean_delta >= GATE_MIN_MEAN_DELTA_PP)
            ship_votes.append(gate_ship)
            print(f"  │")
            print(f"  │ Paired acc delta (tuned − inherited): mean {mean_delta:+.2f}pp"
                  f"  ({better}/{len(deltas)} folds better)")
            gate = ("SHIP (tune classifier)" if gate_ship
                    else "KEEP INHERITING (tuning did not clear the bar)")
            print(f"  │ GATE [≥half folds better AND mean Δ ≥ +{GATE_MIN_MEAN_DELTA_PP}pp]: {gate}")
        print(f"  └{'─' * 62}┘")
    if ship_votes:
        overall = "SHIP" if all(ship_votes) else ("MIXED" if any(ship_votes) else "KEEP INHERITING")
        print(f"\n  OVERALL across horizons: {overall} "
              f"({sum(ship_votes)}/{len(ship_votes)} horizons pass the gate)")


def main():
    import argparse
    p = argparse.ArgumentParser(description="A/B: directional classifier HP search vs q50-inherited")
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--horizon", type=int, default=None, choices=[3, 7, 14, 30])
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--step", type=int, default=120,
                   help="Days between walk-forward fold starts (larger = fewer folds)")
    args = p.parse_args()

    logger.info("=" * 70)
    logger.info("A/B TEST: directional classifier HP search vs q50-inherited params")
    logger.info(f"  max_items={args.max_items} trials={args.trials} step={args.step}")
    logger.info("=" * 70)

    results = run(args.max_items, args.horizon, args.trials, args.step)
    print_report(results)
    print(f"\n  JSON: {json.dumps(results, indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
