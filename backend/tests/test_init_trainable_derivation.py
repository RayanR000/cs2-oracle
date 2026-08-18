import duckdb, pandas as pd
from pathlib import Path

def _write_archive(tmp_path):
    # non-iflow item: pre-2026 NULL-source row  -> backfilled AND trainable
    # iflow-only item: pre-2026 buff_iflow row  -> backfilled, NOT trainable
    df = pd.DataFrame([
        {"item_slug": "AK-47 | Redline (Field-Tested)", "day": "2024-06-01", "source": None,        "mean_price": 12.0, "volume": None, "ingested_at": "2026-01-01"},
        {"item_slug": "AWP | Acheron (Field-Tested)",   "day": "2024-06-01", "source": "buff_iflow", "mean_price": 3.0,  "volume": None, "ingested_at": "2026-01-01"},
    ])
    p = tmp_path / "prices-fixture-2024-06.parquet"
    df.to_parquet(p)
    return tmp_path

def test_iflow_only_item_is_backfilled_but_not_trainable(tmp_path):
    arc = _write_archive(tmp_path)
    con = duckdb.connect()
    backfilled = {r[0] for r in con.sql(
        f"SELECT DISTINCT item_slug FROM read_parquet('{arc}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01'").fetchall()}
    trainable = {r[0] for r in con.sql(
        f"SELECT DISTINCT item_slug FROM read_parquet('{arc}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01' AND source IS DISTINCT FROM 'buff_iflow'").fetchall()}
    con.close()
    assert "AWP | Acheron (Field-Tested)" in backfilled
    assert "AWP | Acheron (Field-Tested)" not in trainable   # iflow-only -> serve but not train
    assert "AK-47 | Redline (Field-Tested)" in trainable       # non-iflow -> train
