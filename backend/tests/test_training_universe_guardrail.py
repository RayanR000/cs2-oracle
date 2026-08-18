# backend/tests/test_training_universe_guardrail.py
import duckdb, pandas as pd

def test_trainable_excludes_iflow_only_items_at_scale(tmp_path):
    rows = []
    # 5 non-iflow items (trainable) + 5 iflow-only items (serve-only)
    for i in range(5):
        rows.append({"item_slug": f"Legit Item {i} (Factory New)", "day": "2024-01-01", "source": None, "mean_price": 10.0, "volume": None, "ingested_at": "2026-01-01"})
    for i in range(5):
        rows.append({"item_slug": f"Iflow Item {i} (Factory New)", "day": "2024-01-01", "source": "buff_iflow", "mean_price": 10.0, "volume": None, "ingested_at": "2026-01-01"})
    pd.DataFrame(rows).to_parquet(tmp_path / "prices-fixture-2024-01.parquet")
    con = duckdb.connect()
    trainable = {r[0] for r in con.sql(
        f"SELECT DISTINCT item_slug FROM read_parquet('{tmp_path}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01' AND source IS DISTINCT FROM 'buff_iflow'").fetchall()}
    backfilled = {r[0] for r in con.sql(
        f"SELECT DISTINCT item_slug FROM read_parquet('{tmp_path}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01'").fetchall()}
    con.close()
    assert len(trainable) == 5                 # training set does NOT grow with iflow
    assert len(backfilled) == 10               # serve set widens with iflow
    assert all("Iflow Item" not in s for s in trainable)
