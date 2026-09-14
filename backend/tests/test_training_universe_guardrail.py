# backend/tests/test_training_universe_guardrail.py
#
# Behavioral guardrail: exercises the SHIPPED `populate_items` derivation
# (not an inlined copy of its SQL predicate) so a regression in the real
# is_trainable/is_backfilled derivation actually fails this test.
#
# NOTE on scale: the ABSOLUTE ~926-item (is_trainable ∩ median >= $1) real
# cohort count is verified in the Task 5 prod-migration gate, not here — it
# depends on the real archive and is not hermetically unit-testable. This
# test only checks the derivation's correctness on a small fixture: iflow
# items must never leak into is_trainable while remaining in is_backfilled.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import scripts.init_local_db as init_local_db
from database import Base, Item
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _write_archive(tmp_path):
    rows = []
    # 5 non-iflow items (source=None, pre-2026) -> trainable AND backfilled
    for i in range(5):
        rows.append(
            {
                "item_slug": f"Legit Item {i} (Factory New)",
                "day": "2024-01-01",
                "source": None,
                "mean_price": 10.0,
                "volume": None,
                "ingested_at": "2026-01-01",
            }
        )
    # 5 iflow-only items (source="buff_iflow", pre-2026) -> backfilled only
    for i in range(5):
        rows.append(
            {
                "item_slug": f"Iflow Item {i} (Factory New)",
                "day": "2024-01-01",
                "source": "buff_iflow",
                "mean_price": 10.0,
                "volume": None,
                "ingested_at": "2026-01-01",
            }
        )
    p = tmp_path / "prices-fixture-2024-01.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return tmp_path


def _temp_session(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "t.db"))
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_trainable_excludes_iflow_only_items_at_scale(tmp_path, monkeypatch):
    """Behavioral test for the real insert path: exercises `populate_items`
    against an isolated temp SQLite DB + fixture archive (10 items), not an
    inlined copy of the SQL predicate. Must fail if the shipped predicate
    regressed to include iflow-sourced rows in is_trainable."""
    arc_dir = tmp_path / "archive"
    arc_dir.mkdir()
    _write_archive(arc_dir)
    monkeypatch.setattr(init_local_db, "ARCHIVE_DIR", arc_dir)

    db = _temp_session(tmp_path)
    try:
        init_local_db.populate_items(db)

        items = db.query(Item).all()
        assert len(items) == 10

        trainable = [it for it in items if it.is_trainable == 1]
        backfilled = [it for it in items if it.is_backfilled == 1]

        assert len(trainable) == 5  # training set does NOT grow with iflow
        assert len(backfilled) == 10  # serve set widens with iflow
        assert all("Iflow Item" not in it.item_id for it in trainable)
    finally:
        db.close()
