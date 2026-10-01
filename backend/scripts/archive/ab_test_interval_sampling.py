#!/usr/bin/env python3
"""A/B the q10/q90 row-sampling fix (`bagging_freq`) over purge-gap CV.

Motivation
----------
`_row_sampling_params` (`forecaster.py`) sets `bagging_freq=1` for **q50 only**
— that is the exact config the 2026-07-29 q50 A/B validated, so it shipped
narrow on purpose. LightGBM defaults `bagging_freq` to 0, which ignores
`bagging_fraction`/`subsample` entirely and trains on every row. Consequences
for the two interval quantiles:

  1. q10/q90 have **never** subsampled rows, on any horizon. The persisted
     `meta.json:tuned_params` confirms it: `subsample` is present for 0.1/0.9
     on all four horizons, `bagging_freq` is absent everywhere.
  2. The Optuna objective searches `subsample` for every quantile
     (`forecaster.py` ~1799), so for q10/q90 it has been spending its budget
     on a **dead dimension** — noisier search, wasted trials.
  3. Suspected knock-on: 3d q10 saved 11 / 1 / 5 trees across ensemble members
     in the 2026-07-27 artifacts — the same near-intercept collapse shape the
     q50 GOSS bug produced. Raw interval coverage is 39-48% against an 80%
     nominal target.

Arms (identical data, identical folds, identical tree params — only row
sampling / split gain differ):
  - **control**       : production. `subsample` set, `bagging_freq` absent
                        => LightGBM trains on all rows (the no-op).
  - **bagfreq**       : `bagging_freq=1`, so `subsample` finally takes effect.
  - **bagfreq_gain0** : bagfreq + `min_gain_to_split=0` (opt-in, --with-gain0).

The third arm is opt-in so the default run stays cheap. Attribution is clean
because the arms are nested: control->bagfreq isolates `bagging_freq`,
bagfreq->bagfreq_gain0 isolates `min_gain_to_split`. The gain0 arm is here
because the q50 root-cause repro measured bagging + `min_gain_to_split=0` at
523 iters / 5.6797 pinball vs bagging alone at 266 / 5.7281 on 3d (a further
~0.85% relative) — favorable but never A/B'd. See
`docs/changelog/2026-07-29-q50-row-sampling.md`.

Pre-registered gate (decided before running)
--------------------------------------------
SHIP an arm for a horizon iff ALL of:
  1. mean paired pinball improvement (q10 and q90 averaged) >= 0.5% relative
     vs control, AND
  2. the arm wins on mean pinball in >= half the paired folds, AND
  3. conformalized interval width does not *increase* by more than 2%
     relative (sharpness guard).
Otherwise keep the current config for that horizon.

**Why pinball is primary and not coverage.** Coverage is already owned
downstream: `forecaster.py` calibrates a conformal q_hat at alpha=0.10 from
OOF predictions and serves `[p10 - q_hat, p90 + q_hat]`. A change that widens
the raw quantiles buys raw coverage for free while making the served interval
*less* informative, so a coverage-primary gate would reward exactly the wrong
thing. Pinball is the loss each quantile is actually trained on, and the
served quality is sharpness at fixed coverage — captured by gate 3, which
re-derives q_hat per fold the same way production does and reports the width
needed to hit the target. Raw coverage/width are reported for diagnosis.

Tree params come from the persisted production `meta.json:tuned_params` for
each (horizon, quantile), so every arm uses the params production trains with.
Nothing here writes model constants or artifacts.

Usage:
    python -m scripts.archive.ab_test_interval_sampling [--max-items 200]
        [--horizon 7] [--max-folds 8] [--single-fold] [--with-gain0]
        [--out recs.csv] [--feature-cache /tmp/abfeat]
    # shard the slow long horizons across processes, then merge:
    python -m scripts.archive.ab_test_interval_sampling --horizon 14 \
        --fold-start 0 --fold-count 4 --out h14a.csv
    python -m scripts.archive.ab_test_interval_sampling --merge h14a.csv h14b.csv
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from backtest.paired_mde import format_paired, paired_arm_contrasts
from backtest.walkforward_records import fold_level_records
from database import SessionLocal
from db.archive import ARCHIVE_ROOT, prices_relation
from models.forecaster import (
    ItemForecaster,
    embargo_days,
    phase_collapsed_sql_filter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ab_test_interval_sampling")

ARCHIVE_DIR = ARCHIVE_ROOT

# Production's item universe, spelled into this harness's archive read. The bid
# sources need no clause: the `STEAMCOMMUNITY`/NULL cohort already excludes
# them. See `models/item_parser.py`.
_UNIVERSE = phase_collapsed_sql_filter()

META_PATH = Path(__file__).resolve().parents[2] / "models" / "saved_models" / "meta.json"

HORIZONS = [3, 7, 14, 30]
LOW_Q, HIGH_Q = 0.1, 0.9
QUANTILES = (LOW_Q, HIGH_Q)
CONTROL_ARM = "control"
ARMS = [CONTROL_ARM, "bagfreq"]
GAIN0_ARM = "bagfreq_gain0"

MIN_TRAIN_ROWS = 2000
# Higher than the q50 harness's 200: the val block is split in half here
# (calibration / test) to derive the conformal q_hat, so each half needs rows.
MIN_VAL_ROWS = 400
MAX_FOLDS = 8

# Gate thresholds (pre-registered)
GATE_MIN_REL_PINBALL_GAIN = 0.005  # 0.5% relative, mean over q10+q90
GATE_MAX_REL_WIDTH_INCREASE = 0.02  # 2% relative conformalized width

# Nominal coverage of the raw [p10, p90] band, and the conformal target
# production calibrates to (forecaster.py: alpha = 0.10).
TARGET_RAW_COVERAGE = 0.80
CONFORMAL_ALPHA = 0.10

# Production ensemble uses feature_fraction 0.6/0.7/0.8 across members; the
# A/B trains one member at the middle value so the arms differ only in the
# parameter under test.
SINGLE_MEMBER_FEATURE_FRACTION = 0.7

# Boosting rounds per model. Was `DART_NUM_BOOST_ROUND if dart else 1000`;
# DART is gone from the forecaster, so only the GBDT arm of that branch
# survives.
NUM_BOOST_ROUND = 1000


def pinball_loss(y_true, y_pred, alpha):
    """Mean pinball (quantile) loss — the objective each quantile trains on."""
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def conformal_q_hat(y_cal, low_cal, high_cal, alpha=CONFORMAL_ALPHA):
    """CQR nonconformity quantile, mirroring forecaster.py:2626-2634.

    Score is max(low - y, y - high): positive when the interval misses. The
    (1-alpha)(1+1/n) quantile is the finite-sample-corrected level.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    scores = np.maximum(np.asarray(low_cal, dtype=float) - y_cal, y_cal - np.asarray(high_cal, dtype=float))
    n = len(scores)
    if n == 0:
        return 0.0
    q_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return float(np.quantile(scores, q_level))


def interval_metrics(y, low, high):
    """Empirical coverage and mean width of a [low, high] band."""
    y = np.asarray(y, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    return (float(np.mean((y >= low) & (y <= high))), float(np.mean(high - low)))


def load_features(con, forecaster, events_df, max_items):
    """Mirrors ab_test_q50_sampling.load_features so this A/B trains on the
    same data path as the other harnesses."""
    # `source IS NULL` is the pre-2026 CSMarketAPI STEAMCOMMUNITY series — that
    # series predates the column, so a NULL here means the same cohort the
    # explicit label selects in the monthly files (`init_local_db.py` documents
    # the equivalence). This reproduces the per-file branch it replaced, which
    # took every row from a file with no source column.
    # `source = 'STEAMCOMMUNITY'` matches 0 rows post archive-rebuild, so this
    # degenerated to the NULL (pre-2026) branch; the dead disjunct is dropped,
    # keeping the intended pre-2026 cohort. See 2026-08-13 harness repin.
    union_sql = prices_relation(
        con,
        ARCHIVE_DIR,
        columns=["item_slug", "day", "mean_price", "volume", "source"],
        where=f"source IS NULL AND {_UNIVERSE}",
    )

    items = con.sql(f"""
        SELECT item_slug, COUNT(*) AS row_count
        FROM {union_sql}
        GROUP BY item_slug HAVING row_count >= 90
        ORDER BY row_count DESC, item_slug LIMIT {max_items}
    """).fetchall()
    if not items:
        raise RuntimeError(
            "interval_sampling universe query selected 0 items — the source pin "
            "matched no rows (see 2026-08-13 harness repin)."
        )
    logger.info(f"  {len(items)} items for evaluation")

    all_rows = []
    for item_slug, _ in items:
        rows = con.sql(
            f"""
            SELECT item_slug AS item_id, day AS timestamp, mean_price AS price, volume
            FROM {union_sql} WHERE item_slug = ? ORDER BY day
        """,
            params=[item_slug],
        ).fetchall()
        idf = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "volume"])
        idf["timestamp"] = pd.to_datetime(idf["timestamp"])
        idf["date"] = idf["timestamp"].dt.date
        all_rows.append(idf)

    all_prices = pd.concat(all_rows, ignore_index=True)
    df = forecaster.engineer_features(all_prices, events_df)
    df = forecaster._add_cross_sectional_features(df)

    EXCLUDE = {"item_id", "date", "timestamp", "price", "volume", "name", "release_date"}
    feat_cols = [
        c for c in df.columns if c not in EXCLUDE and df[c].dtype in (np.float64, np.float32, np.int64, int, float)
    ]
    if len(feat_cols) > 2:
        corr = df[feat_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            to_drop.update(upper[col][upper[col] > 0.95].index)
        feat_cols = [c for c in feat_cols if c not in to_drop]
    logger.info(f"  {len(feat_cols)} features after corr prune")
    return df, feat_cols


def load_production_params(horizon, quantile):
    """Production tuned params for one (horizon, quantile), from the persisted
    meta.json — the same dict the warm-retrain path feeds to lgb.train."""
    with open(META_PATH) as fh:
        meta = json.load(fh)
    tp = meta.get("tuned_params", {})
    src = tp.get(str(horizon)) or tp.get(horizon) or {}
    q = src.get(str(quantile)) or src.get(quantile)
    if not q:
        raise RuntimeError(f"no persisted q{quantile} params for horizon {horizon} in {META_PATH}")
    return dict(q)


def build_arm_params(base, arm, quantile, boosting_type, forecaster):
    """Production params with only the parameter under test swapped."""
    p = dict(base)
    p["objective"] = "quantile"
    p["alpha"] = quantile
    p["metric"] = "quantile"
    p["boosting_type"] = boosting_type
    p["max_bin"] = forecaster.MAX_BIN
    p["feature_pre_filter"] = False
    p["feature_fraction"] = SINGLE_MEMBER_FEATURE_FRACTION
    p["verbosity"] = -1
    p["device"] = "cpu"
    # Pin per-process thread count when sharding across processes, so N
    # concurrent shards don't each grab every core and thrash. Separate OS
    # processes (not a shared ThreadPoolExecutor) deliberately avoid the
    # nested-pool + libomp hang documented at forecaster.py:2384.
    if os.environ.get("AB_N_JOBS"):
        p["n_jobs"] = int(os.environ["AB_N_JOBS"])

    # Rewrite row sampling from scratch so nothing cached in meta.json leaks.
    for k in ("data_sample_strategy", "top_rate", "other_rate", "subsample", "bagging_fraction", "bagging_freq"):
        p.pop(k, None)
    p["data_sample_strategy"] = "bagging"
    p["subsample"] = float(base.get("subsample", 0.8))
    if arm == CONTROL_ARM:
        # Deliberately omit bagging_freq: this reproduces production, where
        # subsample is silently inert. Do not "fix" this in the control.
        pass
    else:
        p["bagging_freq"] = 1
    if arm == GAIN0_ARM:
        p["min_gain_to_split"] = 0.0
    return p


def _fit_quantile(params, X_tr, y_tr, w_tr, X_va, y_va, w_va, nbr, boosting_type, max_bin):
    """Train one quantile model and return (predictions, trees, best_iter)."""
    ds_params = {"max_bin": max_bin, "feature_pre_filter": False}
    dtrain = lgb.Dataset(X_tr, y_tr, params=ds_params, **({"weight": w_tr} if w_tr is not None else {}))
    dval = lgb.Dataset(X_va, y_va, reference=dtrain, params=ds_params, **({"weight": w_va} if w_va is not None else {}))

    model = ItemForecaster._train_ensemble_member(
        params, dtrain, dval, num_boost_round=nbr, early_stopping=ItemForecaster._early_stopping_enabled()
    )
    pred = model.predict(X_va, num_iteration=model.best_iteration or None)
    return pred, int(model.num_trees()), int(model.best_iteration or model.num_trees())


def _run_horizon(fc, tdf, feat_cols, horizon, max_folds, arms, fold_start=0, fold_count=None):
    """Train every arm on identical folds; yield one record per (arm, fold).

    `fold_start`/`fold_count` shard the fold list across processes. Fold
    indices stay GLOBAL (offset by fold_start) so per-shard CSVs merge without
    key collisions, and all arms of a given fold always run in the same shard
    so the paired comparison is never split across processes.
    """
    target_col = f"target_return_{horizon}d"
    dates = np.array(sorted(tdf["date"].unique()))
    folds = fc._compute_cv_splits(dates, purge_days=embargo_days(horizon))
    boosting_type = fc.BOOSTING_TYPE
    if not folds:
        logger.warning(f"  no folds for {horizon}d after purge gap")
        return
    folds = folds[-max_folds:]
    n_total = len(folds)
    end = n_total if fold_count is None else fold_start + fold_count
    sharded = list(enumerate(folds))[fold_start:end]
    logger.info(
        f"  {horizon}d: {n_total} folds total, running "
        f"{len(sharded)} (global idx {fold_start}..{end - 1}), "
        f"boosting={boosting_type}, arms={arms}"
    )

    base_by_q = {q: load_production_params(horizon, q) for q in QUANTILES}
    # Production's per-horizon table, not a 1000-round cap. `best_iter` in
    # the records below is therefore the fixed count unless EARLY_STOPPING=1,
    # so the 'q10 saved 11/1/5 trees' collapse in the docstring reproduces
    # only under that flag.
    nbr = ItemForecaster._boost_rounds(horizon, cv=True)

    for fold_idx, (train_dates, val_dates) in sharded:
        tr = tdf[tdf["date"].isin(train_dates)]
        va = tdf[tdf["date"].isin(val_dates)]
        if len(tr) < MIN_TRAIN_ROWS or len(va) < MIN_VAL_ROWS:
            logger.info(f"    fold {fold_idx}: skipped ({len(tr)} train, {len(va)} val)")
            continue

        X_tr = tr[feat_cols].replace([np.inf, -np.inf], np.nan)
        med = X_tr.median()
        X_tr = X_tr.fillna(med)
        X_va = va[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(med)
        y_tr = tr[target_col].to_numpy(dtype=float)
        y_va = va[target_col].to_numpy(dtype=float)

        # Production weighting: early stopping sees the weighted val metric.
        w_tr = fc._compute_sample_weights(tr, horizon)
        w_va = fc._compute_sample_weights(va, horizon)

        # Chronological calibration/test split of the val block, so q_hat is
        # never fit on the rows it is scored against.
        va_dates = np.array(sorted(va["date"].unique()))
        cut = va_dates[len(va_dates) // 2] if len(va_dates) > 1 else None
        cal_mask = (va["date"].to_numpy() < cut) if cut is not None else None

        for arm in arms:
            t0 = time.time()
            preds, trees, best_iters = {}, {}, {}
            for q in QUANTILES:
                params = build_arm_params(base_by_q[q], arm, q, boosting_type, fc)
                preds[q], trees[q], best_iters[q] = _fit_quantile(
                    params, X_tr, y_tr, w_tr, X_va, y_va, w_va, nbr, boosting_type, fc.MAX_BIN
                )
            fit_s = time.time() - t0

            low, high = preds[LOW_Q], preds[HIGH_Q]
            # Quantile crossing: production sorts the band, so do the same
            # before any interval metric.
            low, high = np.minimum(low, high), np.maximum(low, high)

            raw_cov, raw_width = interval_metrics(y_va, low, high)
            pin_lo = pinball_loss(y_va, preds[LOW_Q], LOW_Q)
            pin_hi = pinball_loss(y_va, preds[HIGH_Q], HIGH_Q)

            # Conformalized sharpness: derive q_hat on the calibration half,
            # score coverage/width on the held-out half.
            if cal_mask is not None and cal_mask.sum() >= 50 and (~cal_mask).sum() >= 50:
                q_hat = conformal_q_hat(y_va[cal_mask], low[cal_mask], high[cal_mask])
                te = ~cal_mask
                conf_cov, conf_width = interval_metrics(y_va[te], low[te] - q_hat, high[te] + q_hat)
            else:
                q_hat, conf_cov, conf_width = float("nan"), float("nan"), float("nan")

            rec = {
                "horizon": horizon,
                "arm": arm,
                "fold": fold_idx,
                "pinball_q10": pin_lo,
                "pinball_q90": pin_hi,
                "pinball_mean": 0.5 * (pin_lo + pin_hi),
                "raw_coverage": raw_cov,
                "raw_width": raw_width,
                "q_hat": q_hat,
                "conf_coverage": conf_cov,
                "conf_width": conf_width,
                "trees_q10": trees[LOW_Q],
                "trees_q90": trees[HIGH_Q],
                "best_iter_q10": best_iters[LOW_Q],
                "best_iter_q90": best_iters[HIGH_Q],
                "fit_s": round(fit_s, 1),
                "n_train": len(tr),
                "n_val": len(va),
            }
            logger.info(
                f"    fold {fold_idx} {arm:>13}: "
                f"pinball={rec['pinball_mean']:.5f} "
                f"(q10={pin_lo:.5f} q90={pin_hi:.5f}) "
                f"rawcov={100 * raw_cov:.1f}% w={raw_width:.2f} "
                f"confcov={100 * conf_cov:.1f}% confw={conf_width:.2f} "
                f"trees={trees[LOW_Q]}/{trees[HIGH_Q]} ({fit_s:.1f}s)"
            )
            yield rec


def summarize(records):
    """Paired per-fold comparison + pre-registered gate verdict per horizon."""
    df = pd.DataFrame(records)
    if df.empty:
        print("\nno results")
        return df, {}

    print("\n" + "=" * 104)
    print("PAIRED RESULTS (each arm vs control, same folds)")
    print(f"raw coverage target {100 * TARGET_RAW_COVERAGE:.0f}% | conformal target {100 * (1 - CONFORMAL_ALPHA):.0f}%")
    print("=" * 104)
    print(
        f"{'h':>3} {'arm':>14} {'pinball':>9} {'q10':>9} {'q90':>9} "
        f"{'rawcov%':>8} {'raww':>7} {'confcov%':>9} {'confw':>7} "
        f"{'trees10':>8} {'trees90':>8} {'folds':>6} {'fit_s':>7}"
    )

    verdicts = {}
    for horizon in sorted(df["horizon"].unique()):
        hd = df[df["horizon"] == horizon]
        arms_present = [a for a in hd["arm"].unique()]
        ordered = ([CONTROL_ARM] if CONTROL_ARM in arms_present else []) + [a for a in arms_present if a != CONTROL_ARM]
        for arm in ordered:
            a = hd[hd["arm"] == arm]
            print(
                f"{horizon:>3} {arm:>14} {a['pinball_mean'].mean():>9.5f} "
                f"{a['pinball_q10'].mean():>9.5f} {a['pinball_q90'].mean():>9.5f} "
                f"{100 * a['raw_coverage'].mean():>8.2f} {a['raw_width'].mean():>7.2f} "
                f"{100 * a['conf_coverage'].mean():>9.2f} {a['conf_width'].mean():>7.2f} "
                f"{a['trees_q10'].mean():>8.1f} {a['trees_q90'].mean():>8.1f} "
                f"{len(a):>6} {a['fit_s'].sum():>7.1f}"
            )

        ctl = hd[hd["arm"] == CONTROL_ARM].set_index("fold")
        if ctl.empty:
            print(f"  -> {horizon}d: no control arm, cannot gate")
            continue

        for arm in [a for a in ordered if a != CONTROL_ARM]:
            trt = hd[hd["arm"] == arm].set_index("fold")
            common = sorted(set(ctl.index) & set(trt.index))
            if not common:
                continue
            c, t = ctl.loc[common], trt.loc[common]

            rel_gain = ((c["pinball_mean"] - t["pinball_mean"]) / c["pinball_mean"]).mean()
            folds_won = int((t["pinball_mean"] < c["pinball_mean"]).sum())
            # Positive => the arm's conformalized interval got wider (worse).
            rel_width = float(((t["conf_width"] - c["conf_width"]) / c["conf_width"]).mean())

            # Fold-clustered paired interval on the pinball difference,
            # replacing the "wins on at least half the folds" condition this
            # gate used until 2026-08-08. A win count is not a test: two arms
            # differing only by seed clear it half the time. Lower pinball is
            # better, so a SHIP needs the interval strictly below zero.
            paired = paired_arm_contrasts(
                {
                    "control": fold_level_records(common, c["pinball_mean"], metric="pinball"),
                    arm: fold_level_records(common, t["pinball_mean"], metric="pinball"),
                },
                base="control",
                value_key="pinball",
                scale=1.0,
                higher_is_better=False,
            )[arm]

            c1 = rel_gain >= GATE_MIN_REL_PINBALL_GAIN
            c2 = paired["verdict"] == "positive"
            c3 = (not np.isfinite(rel_width)) or rel_width <= GATE_MAX_REL_WIDTH_INCREASE
            ship = bool(c1 and c2 and c3)
            verdicts[(horizon, arm)] = {
                "ship": ship,
                "rel_pinball_gain": rel_gain,
                "folds_won": folds_won,
                "n_folds": len(common),
                "rel_conf_width_change": rel_width,
                "paired_pinball": paired,
                "gate": {"pinball_gain": c1, "paired_interval": c2, "width_no_regress": c3},
            }
            print(
                f"  -> {horizon}d {arm} vs control: "
                f"pinball {rel_gain * 100:+.2f}% "
                f"(gate >= +{GATE_MIN_REL_PINBALL_GAIN * 100:.1f}%) "
                f"[{'PASS' if c1 else 'FAIL'}] | "
                f"paired {format_paired(paired, unit='')} "
                f"[{'PASS' if c2 else 'FAIL'}] | "
                f"conf width {rel_width * 100:+.2f}% "
                f"(gate <= +{GATE_MAX_REL_WIDTH_INCREASE * 100:.0f}%) "
                f"[{'PASS' if c3 else 'FAIL'}] | "
                f"folds won {folds_won}/{len(common)} (context, not a gate)"
            )
            print(f"     VERDICT {horizon}d {arm}: {'SHIP' if ship else 'KEEP control'}")

    return df, verdicts


def run(max_items, horizon_filter, max_folds, arms, fold_start=0, fold_count=None, feature_cache=None):
    db = SessionLocal()
    forecaster = ItemForecaster(db_session=db)
    events_df = forecaster.fetch_events()
    db.close()

    # The engineered matrix is a pure function of (archive contents, max_items),
    # so cache it: reruns and sibling shards skip the ~30s rebuild. Cache key
    # includes max_items; delete the file to force a rebuild. Shares the
    # ab_test_q50_sampling cache layout, so a prefix can be reused across both.
    cache_ok = False
    if feature_cache:
        cpath = Path(f"{feature_cache}.items{max_items}.parquet")
        ccols = Path(f"{feature_cache}.items{max_items}.cols.json")
        if cpath.exists() and ccols.exists():
            t0 = time.time()
            df = pd.read_parquet(cpath)
            feat_cols = json.loads(ccols.read_text())
            logger.info(f"  loaded cached features {cpath.name} ({time.time() - t0:.0f}s, {df.shape})")
            cache_ok = True

    if not cache_ok:
        import duckdb

        con = duckdb.connect()
        try:
            t0 = time.time()
            df, feat_cols = load_features(con, forecaster, events_df, max_items)
            logger.info(f"  feature build took {time.time() - t0:.0f}s")
        finally:
            con.close()
        if feature_cache:
            df.to_parquet(cpath)
            ccols.write_text(json.dumps(list(feat_cols)))
            logger.info(f"  wrote feature cache {cpath.name}")

    # Match production: price_technicals only.
    feat_cols = forecaster._apply_feature_allowlist(feat_cols, ItemForecaster.FEATURE_GROUP_ALLOWLIST)
    feat_cols = [c for c in feat_cols if c in df.columns]
    logger.info(f"  {len(feat_cols)} features after production allowlist")

    horizons = [h for h in HORIZONS if horizon_filter is None or h == horizon_filter]

    records = []
    for horizon in horizons:
        target_col = f"target_return_{horizon}d"
        tdf = forecaster.prepare_targets(df, horizon)
        tdf = tdf.dropna(subset=[target_col]).sort_values(["item_id", "date"]).copy()
        if tdf.empty:
            logger.warning(f"  no targets for {horizon}d")
            continue
        for rec in _run_horizon(
            forecaster, tdf, feat_cols, horizon, max_folds, arms, fold_start=fold_start, fold_count=fold_count
        ):
            records.append(rec)

    return summarize(records)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="A/B q10/q90 row sampling: production bagging_freq=0 no-op vs bagging_freq=1"
    )
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=None, choices=HORIZONS)
    ap.add_argument("--max-folds", type=int, default=MAX_FOLDS)
    ap.add_argument("--single-fold", action="store_true", help="1 fold only — for timing calibration")
    ap.add_argument("--with-gain0", action="store_true", help=f"add the {GAIN0_ARM} arm (min_gain_to_split=0)")
    ap.add_argument("--out", type=str, default=None, help="write per-fold records to this CSV")
    ap.add_argument("--fold-start", type=int, default=0, help="shard: first (global) fold index to run")
    ap.add_argument("--fold-count", type=int, default=None, help="shard: number of folds to run from --fold-start")
    ap.add_argument(
        "--feature-cache",
        type=str,
        default=None,
        help="path prefix for caching the engineered feature matrix across runs/shards (skips the ~30s rebuild)",
    )
    ap.add_argument(
        "--merge",
        nargs="+",
        default=None,
        help="merge per-shard CSVs and print the combined gate verdict; skips all training",
    )
    args = ap.parse_args()

    # Merge mode: recombine shard CSVs and apply the gate over all folds.
    if args.merge:
        frames = [pd.read_csv(p) for p in args.merge]
        allrecs = pd.concat(frames, ignore_index=True)
        dupes = allrecs.duplicated(subset=["horizon", "arm", "fold"]).sum()
        if dupes:
            logger.warning(f"{dupes} duplicate (horizon,arm,fold) rows — check shard fold ranges for overlap")
        logger.info(f"merged {len(allrecs)} records from {len(args.merge)} files")
        summarize(allrecs.to_dict("records"))
        return 0

    max_folds = 1 if args.single_fold else args.max_folds
    arms = list(ARMS) + ([GAIN0_ARM] if args.with_gain0 else [])

    logger.info("=" * 70)
    logger.info("A/B: q10/q90 row sampling (bagging_freq no-op vs =1)")
    logger.info(f"  max_items={args.max_items} horizon={args.horizon} max_folds={max_folds} arms={arms}")
    logger.info("=" * 70)

    t0 = time.time()
    df, _verdicts = run(
        args.max_items,
        args.horizon,
        max_folds,
        arms,
        fold_start=args.fold_start,
        fold_count=args.fold_count,
        feature_cache=args.feature_cache,
    )
    logger.info(f"total wall clock: {time.time() - t0:.0f}s")

    if args.out and not df.empty:
        df.to_csv(args.out, index=False)
        logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
