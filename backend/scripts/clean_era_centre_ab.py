#!/usr/bin/env python3
"""Clean-era centre test: does the 2025 composite panel change the CONVERGED verdict?

Implements `docs/research/2026-09-13-clean-era-centre-preregistration.md` — a single
falsification test, not a reopened investigation. Walk-forward q50 GBM centre vs
last-price (level) and vs `-return_1d` (ranking) on the 2025 voted composite
(`source IS NULL` + `source='buff_iflow'`, 360 joint days), with a capacity-matched
shuffled-feature placebo.

Panel novelty (the only reason this run exists): every prior centre read ran where
the ceiling was low (2026: 18-33% frozen quotes) or the label was single-source by
construction (the 08-22 within-source clean label). Post iflow-promotion the 2025
voted composite is high-ceiling (max IC 0.93/0.96 at h=14/30) AND composite.

Fixed design (see prereg; do not tune after reading):
  frame:   voted 2025 composite -> engineer_features -> price_technicals
           allowlist only (the served set; no corr prune, no metadata).
  folds:   21-day val windows every 21 calendar days from 2025-04-15, train on
           pre-boundary rows purged via production's own
           `_purge_overlapping_train_rows` (horizon + 13 embargo, derived at
           call time — never a bare horizon).
  trainer: production's `_train_ensemble_member` at `_boost_rounds(h, cv=True)`,
           quantile alpha=0.5, same tree params as the ab_test_* family.
  metric:  within-date rank IC (Spearman), date-block bootstrap CI — never a
           row bootstrap (one date's items share the market factor).
  bars:    KILL at either h=14/30 if model IC CI spans zero or the paired
           model-minus-naive CI is not entirely positive. PASS needs both
           horizons: IC CI clear of zero AND paired CI positive AND pooled
           MAE skill vs last-price positive.

Usage:
    python -m scripts.clean_era_centre_ab --frame-cache /tmp/cec_frame.parquet \\
        --build-cache-only
    python -m scripts.clean_era_centre_ab --frame-cache /tmp/cec_frame.parquet \\
        --horizon 14 --out /tmp/cec_h14.json
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

from database import SessionLocal
from models.forecaster import ItemForecaster, ANCHOR_TIED_COL
from db.archive import prices_relation
from models.item_parser import archive_universe_sql_filter
from scripts.ab_test_item_metadata import assign_items, _stratified_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("clean_era_centre_ab")

PANEL_START = "2025-01-01"
PANEL_END = "2025-12-31"
MIN_MEDIAN_PRICE = 1.0
MIN_ITEM_DAYS = 180

N_TRAIN_ITEMS = 500  # via imported assign_items (150 held-out / 150 trained-eval)
ROW_BUDGET = 200_000

VAL_WINDOW_DAYS = 21
STEP_DAYS = 21
FOLD_START = "2025-04-15"  # train + feature warmup before the first boundary

MIN_ROWS_PER_DATE = 10
N_BOOTSTRAP = 1000
BOOT_SEED = 42
PLACEBO_SEED = 42

NAIVE_COL = "return_1d"

DS_PARAMS = {"max_bin": 63, "feature_pre_filter": False}

ARMS = ("model", "placebo")


def _frame_fingerprint():
    src = Path(__file__).parent.parent / "models" / "forecaster.py"
    h = hashlib.sha256(src.read_bytes())
    h.update(repr((PANEL_START, PANEL_END, MIN_MEDIAN_PRICE, MIN_ITEM_DAYS,
                   N_TRAIN_ITEMS, ROW_BUDGET, VAL_WINDOW_DAYS, STEP_DAYS,
                   FOLD_START)).encode())
    return h.hexdigest()[:16]


def load_2025_voted():
    """Voted 2025 composite: one price per (item, day).

    Universe-filtered NULL + buff_iflow rows, restricted to items with a 2025
    median >= $1 and >= MIN_ITEM_DAYS distinct days, voted through production's
    own `_apply_multi_source_voting` (median of two on joint days — the same
    2-source average the ceiling measurement characterises).
    """
    import duckdb
    con = duckdb.connect()
    try:
        rel = prices_relation(con)
        uni = archive_universe_sql_filter()
        rows = con.sql(f"""
            SELECT item_slug, CAST(day AS DATE) AS day, source,
                   mean_price AS price, volume
            FROM {rel}
            WHERE CAST(day AS DATE) BETWEEN DATE '{PANEL_START}'
                                        AND DATE '{PANEL_END}'
              AND (source IS NULL OR source = 'buff_iflow')
              AND ({uni}) AND mean_price IS NOT NULL AND mean_price > 0
        """).fetchdf()
    finally:
        con.close()
    rows["day"] = pd.to_datetime(rows["day"])
    med = rows.groupby("item_slug")["price"].median()
    days = rows.groupby("item_slug")["day"].nunique()
    keep = set(med[(med >= MIN_MEDIAN_PRICE) & (days >= MIN_ITEM_DAYS)].index)
    logger.info(f"  2025 rows: {len(rows):,} / {rows['item_slug'].nunique():,} items; "
                f"universe >=${MIN_MEDIAN_PRICE:.0f} + >={MIN_ITEM_DAYS}d: "
                f"{len(keep):,} items")
    if len(keep) < N_TRAIN_ITEMS + 150:
        raise SystemExit(
            f"universe {len(keep)} items cannot fill "
            f"{N_TRAIN_ITEMS} train + 150 held-out — panel UNDERPOWERED, void.")
    rows = rows[rows["item_slug"].isin(keep)].copy()
    rows = rows.rename(columns={"item_slug": "item_id", "day": "date",
                                "mean_price": "price"})
    voted = ItemForecaster._apply_multi_source_voting(
        rows[["item_id", "date", "source", "price", "volume"]])
    voted["date"] = pd.to_datetime(voted["date"])
    logger.info(f"  voted frame: {len(voted):,} rows, "
                f"{voted['item_id'].nunique():,} items, "
                f"{voted['date'].nunique():,} days")
    return voted


def build_frame(cache_path=None):
    if cache_path is not None:
        cache_path = Path(cache_path)
        meta_path = cache_path.with_suffix(".meta.json")
        if cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("fingerprint") != _frame_fingerprint():
                raise SystemExit(
                    f"Frame cache {cache_path} was built from different code "
                    f"or constants. Rebuild with --build-cache-only.")
            df = pd.read_parquet(cache_path)
            logger.info(f"  Loaded cached frame {cache_path} ({len(df):,} rows)")
            return df, meta["features"]
    db = SessionLocal()
    try:
        forecaster = ItemForecaster(db_session=db)
        events_df = forecaster.fetch_events()
        voted = load_2025_voted()
        all_prices = voted.rename(columns={"date": "timestamp"})
        all_prices["date"] = all_prices["timestamp"].dt.date
        all_prices = all_prices.sort_values(
            ["item_id", "timestamp"], kind="stable").reset_index(drop=True)
        logger.info(f"  Loaded {len(all_prices):,} voted price rows")
        df = forecaster.engineer_features(all_prices, events_df)
        df = forecaster._add_cross_sectional_features(df)
        exclude = {"item_id", "date", "timestamp", "price", "volume",
                   "name", "release_date"}
        numeric = (np.float64, np.float32, np.int64, int, float)
        all_cols = [c for c in df.columns
                    if c not in exclude and df[c].dtype in numeric]
        kept = [c for c in all_cols if c not in ItemForecaster.SHELVED_FEATURES]
        features = ItemForecaster._apply_feature_allowlist(
            kept, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
        logger.info(f"  Features: {len(all_cols)} -> {len(features)} "
                    f"after shelving + allowlist (no corr prune: served set)")
        if NAIVE_COL not in features:
            raise SystemExit(
                f"{NAIVE_COL} not in the allowlisted set — naive arm undefined.")
        df = df[["item_id", "date", "price"] + features].copy()
    finally:
        db.close()
    if cache_path is not None:
        tmp_frame = cache_path.with_suffix(f".{os.getpid()}.tmp.parquet")
        tmp_meta = cache_path.with_suffix(f".{os.getpid()}.tmp.json")
        df.to_parquet(tmp_frame, index=False)
        tmp_meta.write_text(json.dumps({
            "fingerprint": _frame_fingerprint(),
            "features": features, "rows": len(df),
        }))
        os.replace(tmp_frame, cache_path)
        os.replace(tmp_meta, cache_path.with_suffix(".meta.json"))
        logger.info(f"  Wrote frame cache {cache_path} ({len(df):,} rows)")
    return df, features


def fold_boundaries(dates):
    """Val-window start dates: every STEP_DAYS from FOLD_START while a full
    window fits inside the panel. Pure function of the date list (tested)."""
    dates = sorted(pd.to_datetime(dates))
    start = max(dates[0], pd.Timestamp(FOLD_START))
    last = dates[-1]
    out = []
    b = pd.Timestamp(start)
    while True:
        window = [d for d in dates if b <= d < b + pd.Timedelta(days=VAL_WINDOW_DAYS)]
        if len(window) < 7:
            if b + pd.Timedelta(days=VAL_WINDOW_DAYS) > last + pd.Timedelta(days=1):
                break
            b += pd.Timedelta(days=STEP_DAYS)
            continue
        out.append(b)
        b += pd.Timedelta(days=STEP_DAYS)
        if b > last:
            break
    return out


def _spearman(x, y):
    """Spearman rank correlation, or None when undefined (constant input)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < MIN_ROWS_PER_DATE:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def per_date_ic(dates, pred, actual):
    """date -> rank IC over rows scored that date (None when unscorable)."""
    out = {}
    for d in sorted(set(dates)):
        m = np.asarray(dates) == d
        out[str(d)] = _spearman(np.asarray(pred, dtype=float)[m],
                                np.asarray(actual, dtype=float)[m])
    return {d: v for d, v in out.items() if v is not None}


def bootstrap_ci(values, n=N_BOOTSTRAP, seed=BOOT_SEED):
    """Percentile 95% CI of the mean under date resampling. `values` maps
    date -> stat; resamples dates with replacement (never rows)."""
    dates = sorted(values)
    if len(dates) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array([values[d] for d in dates], dtype=float)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean()
                      for _ in range(n)])
    return {"n_dates": len(dates), "mean": round(float(arr.mean()), 5),
            "ci_low": round(float(np.percentile(means, 2.5)), 5),
            "ci_high": round(float(np.percentile(means, 97.5)), 5)}


def paired_delta_ci(a, b, n=N_BOOTSTRAP, seed=BOOT_SEED):
    """Bootstrap CI of mean(a - b) on shared dates."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    return bootstrap_ci({d: a[d] - b[d] for d in shared}, n=n, seed=seed)


def evaluate_bars(summaries):
    """Prereg bars over per-horizon summaries. Pure function (tested).

    summaries: {h: {"model": ci, "naive": ci, "paired": ci,
                    "placebo": ci, "mae_skill": float}}
    A ci "spans zero" when ci_low <= 0 <= ci_high; "positive" when ci_low > 0.
    Returns {h: verdict} with verdict in PASS / KILL / VOID / UNDERPOWERED.
    """
    verdicts = {}
    for h, s in summaries.items():
        m, nv, p, pb = s["model"], s["naive"], s["paired"], s["placebo"]
        if m is None or nv is None or p is None or pb is None:
            verdicts[h] = "UNDERPOWERED"
            continue
        if not (pb["ci_low"] <= 0 <= pb["ci_high"]) or abs(pb["mean"]) > 0.02:
            verdicts[h] = "VOID"
            continue
        if nv["ci_low"] <= 0 <= nv["ci_high"]:
            verdicts[h] = "UNDERPOWERED"
            continue
        model_sig = m["ci_low"] > 0
        beats_naive = p["ci_low"] > 0
        level = s.get("mae_skill") is not None and s["mae_skill"] > 0
        if model_sig and beats_naive and level:
            verdicts[h] = "PASS"
        else:
            verdicts[h] = "KILL"
    return verdicts


def run(df, features, horizon_filter=None, max_folds=None, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) // 2)
    if ItemForecaster.label_smoothed_anchor_enabled():
        raise SystemExit("LABEL_SMOOTHED_ANCHOR is on: label basis is not the "
                         "pre-registered raw composite. Refusing to run.")
    eval_items, train_items, trained_eval = assign_items(df)
    set_eval, set_train, set_trained = (set(eval_items), set(train_items),
                                        set(trained_eval))

    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    try:
        horizons = [h for h in ItemForecaster.HORIZONS
                    if horizon_filter is None or h == horizon_filter]
        results = {}
        for horizon in horizons:
            logger.info(f"\n  {'=' * 60}\n  Clean-era centre {horizon}d\n  {'=' * 60}")
            tdf = forecaster.prepare_targets(df, horizon)
            tcol = f"target_return_{horizon}d"
            if tcol not in tdf.columns:
                logger.warning(f"    {tcol} absent — skipping")
                continue
            sub = tdf.dropna(subset=[tcol]).copy()
            sub = sub[sub["price"] >= 1.0]
            sub = sub.sort_values(["item_id", "date"]).reset_index(drop=True)
            if sub.empty:
                continue
            keep = (["item_id", "date", "price", tcol, ANCHOR_TIED_COL]
                    + [c for c in features if c in sub.columns])
            sub = sub[keep].copy()

            dates = sorted(pd.to_datetime(sub["date"].unique()))
            bounds = fold_boundaries(dates)
            if max_folds is not None:
                bounds = bounds[-max_folds:]
            sub_days = pd.to_datetime(sub["date"]).to_numpy()
            is_held = sub["item_id"].isin(set_eval).to_numpy()
            is_trained_eval = sub["item_id"].isin(set_trained).to_numpy()
            is_train_item = sub["item_id"].isin(set_train).to_numpy()
            tied = sub[ANCHOR_TIED_COL].to_numpy(dtype=bool)

            ic = {arm: {"all": {}, "tied": {}} for arm in ARMS}
            ic["naive"] = {"all": {}, "tied": {}}
            fold_skills = []
            fold_gain = []
            n_folds = 0
            for fold_idx, b in enumerate(bounds):
                b_end = b + pd.Timedelta(days=VAL_WINDOW_DAYS)
                in_fit = sub_days < b.to_numpy()
                in_val = (sub_days >= b.to_numpy()) & (sub_days < b_end.to_numpy())
                fit = ItemForecaster._purge_overlapping_train_rows(
                    sub[in_fit & is_train_item], b, horizon)
                val = sub[in_val & (is_held | is_trained_eval)]
                if len(val) < 50 or fit.empty:
                    continue
                fit = _stratified_sample(fit, train_items, ROW_BUDGET, fold_idx)

                feats = [c for c in features if c in sub.columns]
                med = fit[feats].median()
                X_fit = fit[feats].fillna(med)
                X_val = val[feats].fillna(med)
                y_fit = fit[tcol].to_numpy(dtype=float)
                y_val = val[tcol].to_numpy(dtype=float)

                arm_pred = {}
                for arm in ARMS:
                    Xf, Xv = X_fit, X_val
                    if arm == "placebo":
                        rng = np.random.default_rng(PLACEBO_SEED)
                        Xf = Xf.copy()
                        Xv = Xv.copy()
                        for col in feats:
                            Xf[col] = rng.permutation(Xf[col].values)
                            Xv[col] = rng.permutation(Xv[col].values)
                    dtrain = lgb.Dataset(Xf, y_fit, params=DS_PARAMS,
                                         free_raw_data=False)
                    dval = lgb.Dataset(Xv, y_val, reference=dtrain,
                                       params=DS_PARAMS, free_raw_data=False)
                    params = {
                        "objective": "quantile", "alpha": 0.5,
                        "metric": "quantile", "boosting_type": "gbdt",
                        "num_leaves": 31, "max_depth": 5,
                        "min_data_in_leaf": 15, "min_gain_to_split": 0.1,
                        "learning_rate": 0.03, "feature_fraction": 0.7,
                        "bagging_fraction": 0.7, "bagging_freq": 5,
                        "lambda_l1": 0.5, "lambda_l2": 0.5,
                        "verbosity": -1, "random_state": 42, "n_jobs": n_jobs,
                        "force_row_wise": True, **DS_PARAMS,
                    }
                    booster = ItemForecaster._train_ensemble_member(
                        params, dtrain, dval,
                        num_boost_round=ItemForecaster._boost_rounds(
                            horizon, cv=True),
                        early_stopping=ItemForecaster._early_stopping_enabled(),
                    )
                    arm_pred[arm] = booster.predict(Xv)
                    if arm == "model":
                        gain = booster.feature_importance(
                            importance_type="gain")
                        order = np.argsort(gain)[::-1][:5]
                        fold_gain.append({
                            "fold": fold_idx, "val_start": str(b.date()),
                            "top": [(feats[i], round(float(gain[i]), 1))
                                    for i in order if gain[i] > 0]})

                v_dates = val["date"].to_numpy()
                v_tied = val[ANCHOR_TIED_COL].to_numpy(dtype=bool)
                v_naive = -val[NAIVE_COL].to_numpy(dtype=float)
                for arm in ARMS:
                    p = arm_pred[arm]
                    for d, d_ic in per_date_ic(v_dates, p, y_val).items():
                        ic[arm]["all"].setdefault(d, []).append(d_ic)
                    tm = v_tied
                    for d, d_ic in per_date_ic(v_dates[tm], p[tm],
                                               y_val[tm]).items():
                        ic[arm]["tied"].setdefault(d, []).append(d_ic)
                for d, d_ic in per_date_ic(v_dates, v_naive, y_val).items():
                    ic["naive"]["all"].setdefault(d, []).append(d_ic)
                for d, d_ic in per_date_ic(v_dates[v_tied], v_naive[v_tied],
                                           y_val[v_tied]).items():
                    ic["naive"]["tied"].setdefault(d, []).append(d_ic)

                mae_m = float(np.mean(np.abs(y_val - arm_pred["model"])))
                mae_n = float(np.mean(np.abs(y_val)))
                fold_skills.append(1 - mae_m / mae_n if mae_n > 0 else None)
                n_folds += 1
                logger.info(f"    fold {fold_idx} ({b.date()}): "
                            f"n_fit={len(fit):>6,} n_val={len(val):>5,} "
                            f"skill={fold_skills[-1]:+.3f}")

            # One IC per date: dates never repeat across folds (STEP == VAL),
            # but average defensively.
            pooled = {}
            for arm in list(ARMS) + ["naive"]:
                pooled[arm] = {}
                for cohort in ("all", "tied"):
                    pooled[arm][cohort] = {
                        d: float(np.mean(v)) for d, v in ic[arm][cohort].items()}
            entry = {"n_folds": n_folds, "fold_mae_skills": fold_skills,
                     "fold_gain": fold_gain,
                     "per_date": pooled, "summary": {}}
            for cohort in ("all", "tied"):
                summ = {
                    "model": bootstrap_ci(pooled["model"][cohort]),
                    "naive": bootstrap_ci(pooled["naive"][cohort]),
                    "placebo": bootstrap_ci(pooled["placebo"][cohort]),
                }
                summ["paired"] = paired_delta_ci(pooled["model"][cohort],
                                                 pooled["naive"][cohort])
                skills = [s for s in fold_skills if s is not None]
                summ["mae_skill"] = (round(float(np.mean(skills)), 5)
                                     if skills else None)
                entry["summary"][cohort] = summ
            # Prereg bars read the TIED cohort (neither basis carries p/S).
            tied_summ = {k: v for k, v in entry["summary"]["tied"].items()}
            entry["bars"] = evaluate_bars({horizon: tied_summ})
            results[horizon] = entry
        return results
    finally:
        db.close()


def print_summary(results):
    print("\n" + "=" * 78)
    print("CLEAN-ERA CENTRE — within-date rank IC vs composite (tied cohort primary)")
    print("=" * 78)
    for horizon, entry in sorted(results.items()):
        print(f"\nh={horizon}d   folds={entry['n_folds']}")
        for cohort in ("tied", "all"):
            s = entry["summary"][cohort]
            print(f"  [{cohort}]")
            for key in ("model", "naive", "placebo", "paired"):
                c = s.get(key)
                if c is None:
                    print(f"    {key:8s} n/a (<2 dates)")
                    continue
                flag = "*" if c["ci_low"] > 0 else ("-" if c["ci_high"] < 0 else " ")
                print(f"    {key:8s} mean={c['mean']:+.4f} "
                      f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]{flag} "
                      f"n={c['n_dates']}")
            print(f"    mae_skill(pooled)={s['mae_skill']:+.4f} "
                  f"(fold means: "
                  + ", ".join(f"{v:+.3f}" for v in entry["fold_mae_skills"]
                              if v is not None) + ")")
        print(f"  bars(h={horizon}, tied): {entry['bars'][horizon]}")
        from collections import Counter
        lead = Counter(n for g in entry.get("fold_gain", [])
                       for n, _ in g["top"])
        print(f"  gain top-5 counts: {lead.most_common(8)}")
        if entry.get("fold_gain"):
            print(f"  fold0 top: {entry['fold_gain'][0]['top']}")
    print("\n* = CI entirely positive. Bars: PASS needs model>0 AND paired>0 "
          "AND mae_skill>0 at BOTH h=14/30; KILL at either kills.")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--frame-cache", default=None)
    ap.add_argument("--build-cache-only", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df, features = build_frame(args.frame_cache)
    if args.build_cache_only:
        return
    results = run(df, features, horizon_filter=args.horizon,
                  max_folds=args.max_folds)
    print_summary(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        logger.info(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
