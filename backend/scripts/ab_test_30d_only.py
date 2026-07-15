#!/usr/bin/env python3
"""Quick: evaluate 30d horizon WITH player counts only."""
import sys, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, lightgbm as lgb
from database import SessionLocal
from models.forecaster import ItemForecaster
logging.basicConfig(level=logging.INFO)

ARCHIVE_DIR = Path(__file__).parent.parent.parent / "price-archive"
import duckdb
con = duckdb.connect()
db = SessionLocal()
forecaster = ItemForecaster(db_session=db)
events_df = forecaster.fetch_events()
db.close()

where = """WHERE item_slug IN (SELECT DISTINCT item_slug FROM read_parquet('{}/prices-*.parquet') WHERE source = 'STEAMCOMMUNITY')""".format(ARCHIVE_DIR)
rows = con.sql(f"SELECT item_slug FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet') {where} GROUP BY item_slug HAVING COUNT(*) >= 90 ORDER BY COUNT(*) DESC LIMIT 100").fetchall()
all_rows = []
for (item_slug,) in rows:
    r = con.sql(f"SELECT item_slug AS item_id, day AS timestamp, mean_price AS price, volume FROM read_parquet('{ARCHIVE_DIR}/prices-*.parquet') WHERE item_slug = ? ORDER BY day", params=[item_slug]).fetchall()
    df = pd.DataFrame(r, columns=["item_id","timestamp","price","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]); df["date"] = df["timestamp"].dt.date
    all_rows.append(df)
all_prices = pd.concat(all_rows, ignore_index=True)

df = forecaster.engineer_features(all_prices, events_df)
df = forecaster._add_cross_sectional_features(df)
df = forecaster._add_player_count_features(df)
exclude = {"item_id","date","timestamp","price","volume","name","release_date"}
fc = [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64,np.float32,np.int64,int,float)]
if len(fc) > 2:
    corr = df[fc].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    to_drop = set()
    for c in upper.columns:
        hc = upper[c][upper[c] > 0.95].index; to_drop.update(hc)
    fc = [c for c in fc if c not in to_drop]

tdf = forecaster.prepare_targets(df, 30).dropna(subset=["target_return_30d"]).sort_values(["item_id","date"])
dates = sorted(tdf["date"].unique())
si = len(dates) * 2 // 3
dh, dt, ih, it, mt, mc = 0, 0, 0, 0, 0.0, 0
for we in range(si + 1, len(dates), 60):
    vd = dates[we:we + 21]
    if len(vd) < 7: continue
    tr = tdf[tdf["date"].isin(dates[:we])]
    vl = tdf[tdf["date"].isin(vd)]
    if len(vl) < 50: continue
    if len(tr) > 200000: tr = tr.sort_values("date").tail(200000)
    Xtr = tr[fc].fillna(tr[fc].median()); ytr = tr["target_return_30d"]
    Xvl = vl[fc].fillna(tr[fc].median()); yvl = vl["target_return_30d"]
    ms = {}
    for q in [0.1,0.5,0.9]:
        p = {"objective":"quantile","alpha":q,"metric":"quantile","boosting_type":"gbdt","num_leaves":31,"max_depth":5,"min_data_in_leaf":15,"min_gain_to_split":0.1,"learning_rate":0.03,"feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":5,"lambda_l1":0.5,"lambda_l2":0.5,"verbosity":-1,"random_state":42,"n_jobs":-1}
        m = lgb.train(p, lgb.Dataset(Xtr.values, ytr.values), num_boost_round=100, valid_sets=[lgb.Dataset(Xvl.values, yvl.values)], callbacks=[lgb.early_stopping(15,verbose=False),lgb.log_evaluation(0)])
        ms[q] = m.predict(Xvl.values)
    p10, p50, p90 = ms[0.1], ms[0.5], ms[0.9]
    cm = (p10 > p50) | (p50 > p90)
    nc = ~cm
    lo, hi = np.minimum(p10,p50), np.maximum(p50,p90)
    if nc.any():
        ah = np.mean([np.mean(p50[nc]-p10[nc]),np.mean(p90[nc]-p50[nc])])
        if ah > 0: lo[cm], hi[cm] = p50[cm]-ah, p50[cm]+ah
    lo, hi = np.minimum(lo,p50), np.maximum(hi,p50)
    cp = vl["price"].values; ar = yvl.values
    for i in range(len(vl)):
        ad = "up" if ar[i] > 0 else "down"
        pd_ = "up" if p50[i] > 0 else "down"
        dh += 1 if pd_ == ad else 0; dt += 1
        mt += abs(cp[i]*(1+p50[i]/100)-cp[i]*(1+ar[i]/100)); mc += 1
        af = cp[i]*(1+ar[i]/100)
        it += 1 if cp[i]*(1+lo[i]/100) <= af <= cp[i]*(1+hi[i]/100) else 0; ih += 1

da = dh/dt*100; mae = mt/mc; ic = it/ih*100 if ih > 0 else 0
print(f"30d treatment: DirAcc={da:.1f}% ({dt:,} samples, {da-50:.1f}pp above baseline) MAE=${mae:.2f} IntCov={ic:.1f}%")
