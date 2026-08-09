"""Re-derive ItemForecaster.FIXED_BOOST_ROUNDS / CV_FIXED_BOOST_ROUNDS.

Trains each CV fold ONCE to ROUND_CAP with no early stopping, then scores every
checkpoint via predict(num_iteration=k) — so the whole curve costs one training
pass per fold, not one per checkpoint. Reports pooled within-date rank IC beside
the mean validation pinball loss, which is what early stopping was optimising.

The two disagree, and that disagreement is why the constants exist: at 14d and
30d the val loss bottoms out at 25 rounds and rises monotonically while rank IC
keeps climbing to 500-750. See docs/research/2026-08-08-model-review.md.

Reads production's tuned params out of the model artifact, so run it against a
model directory that has a meta.json:

    FORECAST_MODEL_DIR=models/saved_models venv/bin/python \
        scripts/calibrate_boost_rounds.py

Set the constants from the KNEE (smallest count within 1% of peak IC), not the
peak — the curve is flat past it and rounds are the dominant training cost.
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
from database import SessionLocal
from models.forecaster import ItemForecaster, embargo_days, HEADLINE_MIN_TIER

OUT = os.environ.get("CALIBRATION_OUT_DIR", ".")
ROUND_CAP = 1500
CHECKPOINTS = [25, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500]
MODEL_DIR = os.environ.get("FORECAST_MODEL_DIR", "models/saved_models")
with open(os.path.join(MODEL_DIR, "meta.json")) as fh:
    TUNED = json.load(fh)["tuned_params"]

f = ItemForecaster(db_session=SessionLocal(), prune_failed_groups=False)
df = f.build_training_data(days_back=1460, backfilled_only=True,
                           max_feature_rows=1_200_000, min_median_price=1.0)

rows = []
for horizon in f.HORIZONS:
    params = dict(TUNED[str(horizon)]["0.5"])
    params.update(verbosity=-1, n_jobs=-1, random_state=42)
    params.pop("device", None)
    tdf = f.prepare_targets(df, horizon)
    tdf = tdf.dropna(subset=[f"target_return_{horizon}d"]).copy().sort_values("date")
    splits = f._compute_cv_splits(sorted(tdf["date"].unique()),
                                  purge_days=embargo_days(horizon))
    per_ckpt = {k: [] for k in CHECKPOINTS}
    val_loss = {k: [] for k in CHECKPOINTS}
    for fold_id, (train_dates, val_dates) in enumerate(splits):
        tr = tdf[tdf["date"].isin(train_dates)]
        va = tdf[tdf["date"].isin(val_dates)]
        if len(va) < 50:
            continue
        Xtr_pre = tr[f.feature_cols].replace([np.inf, -np.inf], np.nan)
        med = Xtr_pre.median()
        Xtr, ytr = Xtr_pre.fillna(med), tr[f"target_return_{horizon}d"]
        Xv = va[f.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(med)
        yv = va[f"target_return_{horizon}d"].to_numpy(dtype=float)

        d_tr = lgb.Dataset(Xtr, ytr, params={"max_bin": f.MAX_BIN,
                                             "feature_pre_filter": False})
        m = lgb.train(params, d_tr, num_boost_round=ROUND_CAP,
                      callbacks=[lgb.log_evaluation(0)])

        keep = va["price_tier"].to_numpy() >= HEADLINE_MIN_TIER
        dates = pd.to_datetime(va["date"]).to_numpy()
        for k in CHECKPOINTS:
            p = m.predict(Xv, num_iteration=k)
            # pinball loss at alpha=0.5 == 0.5 * MAE
            val_loss[k].append(float(np.mean(np.abs(yv - p)) * 0.5))
            g = pd.DataFrame({"d": dates[keep], "p": p[keep], "a": yv[keep]})
            ics = [spearmanr(x["p"], x["a"]).statistic
                   for _, x in g.groupby("d")
                   if len(x) >= 20 and x["p"].nunique() > 1 and x["a"].nunique() > 1]
            if ics:
                per_ckpt[k].append(float(np.mean(ics)))
        print(f"  {horizon}d fold {fold_id+1} done", flush=True)

    for k in CHECKPOINTS:
        if per_ckpt[k]:
            rows.append(dict(horizon=horizon, rounds=k,
                             rank_ic=round(float(np.mean(per_ckpt[k])), 4),
                             val_pinball=round(float(np.mean(val_loss[k])), 4)))
    sub = [r for r in rows if r["horizon"] == horizon]
    best_ic = max(sub, key=lambda r: r["rank_ic"])
    best_loss = min(sub, key=lambda r: r["val_pinball"])
    print(f"\n{horizon}d: best rank IC {best_ic['rank_ic']} @ {best_ic['rounds']} rounds | "
          f"best val loss @ {best_loss['rounds']} rounds\n", flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "round_calibration.csv"), index=False)
print(out.pivot(index="rounds", columns="horizon", values="rank_ic").to_string())
