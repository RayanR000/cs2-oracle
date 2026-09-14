import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.init_local_db as init_local_db
from database import Base, Item
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _write_archive(tmp_path):
    # non-iflow item: pre-2026 NULL-source row  -> backfilled AND trainable
    # iflow-only item: pre-2026 buff_iflow row  -> backfilled, NOT trainable
    df = pd.DataFrame(
        [
            {
                "item_slug": "AK-47 | Redline (Field-Tested)",
                "day": "2024-06-01",
                "source": None,
                "mean_price": 12.0,
                "volume": None,
                "ingested_at": "2026-01-01",
            },
            {
                "item_slug": "AWP | Acheron (Field-Tested)",
                "day": "2024-06-01",
                "source": "buff_iflow",
                "mean_price": 3.0,
                "volume": None,
                "ingested_at": "2026-01-01",
            },
        ]
    )
    p = tmp_path / "prices-fixture-2024-06.parquet"
    df.to_parquet(p)
    return tmp_path


def test_iflow_only_item_is_backfilled_but_not_trainable(tmp_path):
    arc = _write_archive(tmp_path)
    con = duckdb.connect()
    backfilled = {
        r[0]
        for r in con.sql(
            f"SELECT DISTINCT item_slug FROM read_parquet('{arc}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01'"
        ).fetchall()
    }
    trainable = {
        r[0]
        for r in con.sql(
            f"SELECT DISTINCT item_slug FROM read_parquet('{arc}/prices-*.parquet', union_by_name=true) WHERE day < '2026-01-01' AND source IS DISTINCT FROM 'buff_iflow'"
        ).fetchall()
    }
    con.close()
    assert "AWP | Acheron (Field-Tested)" in backfilled
    assert "AWP | Acheron (Field-Tested)" not in trainable  # iflow-only -> serve but not train
    assert "AK-47 | Redline (Field-Tested)" in trainable  # non-iflow -> train


def _temp_session(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "t.db"))
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_populate_items_wires_is_trainable_and_is_backfilled(tmp_path, monkeypatch):
    """Behavioral test for the real insert path: exercises `populate_items`
    against an isolated temp SQLite DB + fixture archive, not just the raw
    SQL predicate."""
    arc_dir = tmp_path / "archive"
    arc_dir.mkdir()
    _write_archive(arc_dir)
    monkeypatch.setattr(init_local_db, "ARCHIVE_DIR", arc_dir)

    db = _temp_session(tmp_path)
    try:
        init_local_db.populate_items(db)

        non_iflow = db.query(Item).filter(Item.item_id == "AK-47 | Redline (Field-Tested)").one()
        iflow_only = db.query(Item).filter(Item.item_id == "AWP | Acheron (Field-Tested)").one()

        assert non_iflow.is_trainable == 1
        assert non_iflow.is_backfilled == 1
        assert iflow_only.is_trainable == 0
        assert iflow_only.is_backfilled == 1
    finally:
        db.close()


def test_populate_items_re_derive_corrects_stale_is_trainable(tmp_path, monkeypatch):
    """Pre-insert an iflow-only item with a wrong (stale) is_trainable=1 flag
    and confirm the re-derive loop corrects it back to 0 on the next run."""
    arc_dir = tmp_path / "archive"
    arc_dir.mkdir()
    _write_archive(arc_dir)
    monkeypatch.setattr(init_local_db, "ARCHIVE_DIR", arc_dir)

    db = _temp_session(tmp_path)
    try:
        now = datetime.now(UTC).replace(tzinfo=None)
        db.add(
            Item(
                item_id="AWP | Acheron (Field-Tested)",
                name="AWP | Acheron (Field-Tested)",
                type="skin",
                is_backfilled=1,
                is_trainable=1,  # wrong: this slug is iflow-only, should be 0
                created_at=now,
                updated_at=now,
            )
        )
        db.commit()

        init_local_db.populate_items(db)

        corrected = db.query(Item).filter(Item.item_id == "AWP | Acheron (Field-Tested)").one()
        assert corrected.is_trainable == 0
        assert corrected.is_backfilled == 1
    finally:
        db.close()
